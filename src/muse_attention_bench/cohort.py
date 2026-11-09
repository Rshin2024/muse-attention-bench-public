"""Trusted local API broker; actor shells and H200 evaluation execute on GCP.

No provider credentials are sent to GCP or included in actor/evaluator mounts.
The CLI defaults to planning. Actual execution requires --execute and a budget.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import stat
import time

from .protocol import ROOT,DEFAULT_PROFILE,load_profile,canonical_sha256,verify_vendor
from .providers import load_configuration,local_context_environment
from .credentials import DEFAULT_CONFIG_ROOT
from .transports import generate,build_request,TransportError
from .costs import CostLedger
from .trials import prepare_trial,mark_trial_started,seal_trial
from .gpu import evaluator_fingerprint

ACTION_INSTRUCTION='''Return exactly one JSON object, with no Markdown fence:
{"action":"shell","command":"...","note":"..."}
or {"action":"gpu_test","case_ids":["scored-01","scored-02"],"note":"..."}
or {"action":"done","note":"..."}.
Only these tools exist. Shell commands run on GCP in a CPU-only, network-disabled
container, working in /workspace. Use cd kernel when needed. GPU access is only
through gpu_test: at most three requests and two public scored cases per request.
The 1800-second research cap includes API, shell, diagnostic and queue time.
Keep durable notes in NOTES.md: the common broker retains the latest eight
action/feedback rounds and resets native conversation state when pruning.
No provider built-in tools, network, reference oracle, other trials or credentials
are available. Leave final code in kernel/submission.py. Do not game timing,
cache outputs, import the evaluator, or lower precision outside the task rules.
'''


def _json_new(path:Path,data:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
    path.chmod(0o600)


def _json_replace(path:Path,data:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(path)


def credential_environment(context_path:Path)->dict:
    cfg=load_configuration()
    return local_context_environment(context_path,cfg)


def _key_loader(entry:dict,environment:dict)->str:
    value=environment.get(entry.get('api_key_env',''))
    if value:return value
    name=entry.get('api_key_file_env');reference=environment.get(name,'') if name else ''
    if not reference:raise ValueError('Provider credential is not configured')
    path=Path(reference).expanduser()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Credential symlinks are not supported')
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode&0o077:raise ValueError('Credential file must be private and regular')
    value=path.read_text().strip()
    if not value or any(c.isspace() for c in value):raise ValueError('Credential file is empty or malformed')
    return value


def missing_credentials(config:dict,environment:dict)->list[str]:
    missing=[]
    for entry in config['providers']:
        if entry['id']=='gemini':
            if not environment.get(entry['project_env']):missing.append('gemini: Google project missing')
            continue
        try:_key_loader(entry,environment)
        except (ValueError,OSError):missing.append(entry['id'])
    return missing


def cohort_identity(config,providers,catalog):
    actual=[(r['provider'],r['model']) for r in config['models']]
    expected={(p['id'],m) for p in providers['providers'] for m in p['configured_models']}
    if len(actual)!=7 or len(set(actual))!=7 or set(actual)!=expected:
        raise ValueError('Cohort must contain the seven distinct configured provider/model pairs')
    return {'roster':config['models'],'provider_config_sha256':canonical_sha256(providers),
            'pricing_sha256':canonical_sha256(catalog),
            'evaluation_protocol_sha256':canonical_sha256(load_profile(config['profile'])),
            'evaluator_sha256':evaluator_fingerprint(),
            'broker_prompt_sha256':hashlib.sha256(ACTION_INSTRUCTION.encode()).hexdigest()}


def routing_identity(providers,environment):
    google=next(p for p in providers['providers'] if p['id']=='gemini')
    location=environment.get(google['location_env'],google['location_default'])
    glm=next(p for p in providers['providers'] if p['id']=='glm')
    if location!='global' or glm['base_url']!='https://api.z.ai/api/paas/v4':
        raise ValueError('Frozen pricing requires Google global routing and GLM general metered endpoint')
    return {'google_project':environment.get(google['project_env']),'google_location':location,'glm_endpoint':glm['base_url']}


def validate_final_report(report,seal):
    if not isinstance(report,dict):raise ValueError('GPU report is not an object')
    if report.get('candidate_sha256')!=seal['candidate_sha256'] or report.get('protocol_sha256')!=seal['protocol_sha256']:
        raise ValueError('GPU report does not bind the frozen candidate/profile')
    identity=report.get('comparison_identity')
    required=['protocol_sha256','execution_backend','image_digest','gpu_name','gpu_uuid','torch_version','cuda_version','driver_version','evaluator_sha256']
    if not isinstance(identity,dict) or any(not identity.get(k) for k in required) or not (identity.get('sglang_commit') or identity.get('sglang_source_sha256')):
        raise ValueError('GPU runtime identity is incomplete')
    if identity['protocol_sha256']!=seal['protocol_sha256']:raise ValueError('GPU runtime/profile identity differs')
    if report.get('status')=='passed':
        if report.get('gpu_validated') is not True:raise ValueError('Passing GPU report lacks device validation')
        for field,count,prefix in [('correctness_cases',16,'test'),('scored_cases',12,'scored')]:
            rows=report.get(field,[])
            if not isinstance(rows,list) or len(rows)!=count or {r.get('id') for r in rows if isinstance(r,dict)}!={f'{prefix}-{i:02d}' for i in range(1,count+1)} or not all(isinstance(r,dict) and r.get('passed') is True for r in rows):
                raise ValueError('Passing GPU report has incomplete or duplicate case coverage')
        number=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and __import__('math').isfinite(x) and x>0
        value=report.get('speedup');score=report.get('score',{})
        bounds=score.get('confidence_interval_95') if isinstance(score,dict) else None
        if not number(value) or not isinstance(bounds,list) or len(bounds)!=2 or not all(number(b) for b in bounds) or not bounds[0]<=value<=bounds[1]:raise ValueError('Passing GPU timing result is malformed')
        if set(score.get('loads',{}))!={f'{mode} {b}' for mode in ['decode','verify'] for b in [1,16,32]}:raise ValueError('GPU load aggregation is incomplete')
    elif report.get('status')!='failed':raise ValueError('Unexpected final GPU evaluation status')
    return report


def _generation_worker(connection,payload):
    try:
        if hasattr(os,'setsid'):os.setsid()
        if evaluator_fingerprint()!=payload['expected_source_sha256']:
            connection.send({'ok':False,'error':{'kind':'source_drift','cost_unknown':False}});return
        result=generate(payload['entry'],payload['model'],payload['messages'],payload['remaining'],
            lambda e:_key_loader(e,payload['environment']),max_output_tokens=payload['output_limit'],
            native_state=payload.get('native_state'),environment=payload['environment'],json_mode=True,
            pricing=payload['pricing'],timeout_cap=600)
        connection.send({'ok':True,'result':result})
    except TransportError as exc:connection.send({'ok':False,'error':exc.as_dict()})
    except Exception as exc:connection.send({'ok':False,'error':{'kind':'controller_request_error','type':type(exc).__name__,'cost_unknown':True}})
    finally:connection.close()


def bounded_generate(payload:dict,seconds:float)->dict:
    """A process deadline also bounds slow/dribbling native HTTP responses."""
    if seconds<=0:return {'ok':False,'error':{'kind':'trial_deadline','cost_unknown':False}}
    context=mp.get_context('spawn');parent,child=context.Pipe(duplex=False)
    process=context.Process(target=_generation_worker,args=(child,payload))
    process.start();child.close()
    try:
        if parent.poll(seconds):
            try:return parent.recv()
            except EOFError:return {'ok':False,'error':{'kind':'request_process_failed','cost_unknown':True}}
        return {'ok':False,'error':{'kind':'request_process_deadline','cost_unknown':True}}
    finally:
        if process.is_alive():
            try:os.killpg(process.pid,signal.SIGTERM)
            except (ProcessLookupError,PermissionError):process.terminate()
        process.join(timeout=3)
        if process.is_alive():process.kill();process.join(timeout=2)
        parent.close()


def parse_action(text:str)->dict:
    value=text.strip()
    if value.startswith('```') and value.endswith('```'):
        value=value.split('\n',1)[1].rsplit('```',1)[0].strip()
    action=json.loads(value)
    if not isinstance(action,dict) or action.get('action') not in {'shell','gpu_test','done'}:raise ValueError('Return one supported JSON action')
    if action['action']=='shell' and (not isinstance(action.get('command'),str) or not action['command'] or len(action['command'])>200000):raise ValueError('Invalid shell command')
    if action['action']=='gpu_test' and (not isinstance(action.get('case_ids'),list) or not 1<=len(action['case_ids'])<=2 or any(not isinstance(x,str) for x in action['case_ids'])):raise ValueError('Select one or two public case IDs')
    return action


def trim_history(messages:list[dict],native_state:dict|None,entry:dict,model:str,environment:dict,maximum_bytes:int=120000):
    """Same last-eight-round and conservative serialized-byte policy for all."""
    pruned=False
    while len(messages)>18:
        del messages[2:4];pruned=True
    if pruned:native_state=None
    while True:
        plan=build_request(entry,model,messages,native_state=native_state,environment=environment)
        size=len(json.dumps(plan['body'],ensure_ascii=False).encode())
        if size<=maximum_bytes:return messages,native_state,size
        if native_state is not None:native_state=None;continue
        if len(messages)<=4:raise ValueError('Single action exceeds the common request context bound')
        del messages[2:4];pruned=True


def _request(entry,model,messages,state,remaining,environment,catalog,ledger,phase,output_limit=32768,expected_source=None):
    index=ledger.reserve(model,phase,100 if phase=='research' else 2)
    payload={'entry':entry,'model':model,'messages':messages,'native_state':state,
             'remaining':remaining,'output_limit':output_limit,'environment':environment,'pricing':catalog['models'][model],
             'expected_source_sha256':expected_source or evaluator_fingerprint()}
    reply=bounded_generate(payload,remaining)
    if reply['ok']:
        ledger.settle(index,reply['result'],catalog)
    else:ledger.fail(index,reply['error'],catalog)
    return reply


def preflight(config:dict,providers:dict,environment:dict,catalog:dict,ledger:CostLedger,path:Path)->dict:
    identity=cohort_identity(config,providers,catalog)
    identity['routing']=routing_identity(providers,environment)
    missing=missing_credentials(providers,environment)
    if missing:
        report={'status':'missing_credentials','missing':missing,'model_calls':0,'identity':identity}
        _json_replace(path,report);return report
    entries={p['id']:p for p in providers['providers']};rows=[]
    for row in config['models']:
        messages=[{'role':'system','content':'Respond only with JSON. No tools are available.'},
                  {'role':'user','content':'Return {"action":"done","note":"API readiness probe"}.'}]
        response=_request(entries[row['provider']],row['model'],messages,None,180,environment,catalog,ledger,'preflight',1024,identity['evaluator_sha256'])
        if response['ok']:
            r=response['result']
            try:parse_action(r['output_text']);action_valid=True
            except (ValueError,TypeError):action_valid=False
            ready=action_valid and r['model_reported']==row['model'] and r['finish_reason'] in {'completed','end_turn','STOP','stop'}
            rows.append({**row,'ready':ready,'action_format_validated':action_valid,'model_reported':r['model_reported'],'request_id':r['request_id'],'usage':r['usage'],'finish_reason':r['finish_reason']})
        else:rows.append({**row,'ready':False,'error':response['error']})
    report={'status':'ready' if len(rows)==7 and all(r['ready'] for r in rows) else 'access_or_transport_failed','identity':identity,'models':rows,'cost':ledger.summary()}
    _json_replace(path,report);return report


def run_trial(row,config,providers,environment,catalog,ledger,output,session_factory):
    model,provider=row['model'],row['provider'];entry=next(e for e in providers['providers'] if e['id']==provider)
    trial_id=f"{config['cohort']}-{model.replace('.','-')}"
    trial=prepare_trial(ROOT,model,trial_id,Path(config['profile']),provider=provider,effort='high',
        context_budget={'native_input_observed_limit':128000,'serialized_request_byte_bound':120000,'latest_rounds':8,'native_reset_on_prune':True},
        tool_policy={'runner':'gcp-iap-json-action-v1','broker_prompt_sha256':hashlib.sha256(ACTION_INSTRUCTION.encode()).hexdigest(),'network':'disabled','shell_cpu':8,'shell_memory_gb':32,'diagnostics':3,'diagnostic_cases':2},
        wall_clock_seconds=1800,diagnostic_budget=3,cohort=config['cohort'])
    session=session_factory(trial)
    result={'model':model,'provider':provider,'trial_id':trial_id,'status':'running','research_status':'running','actions':0,'diagnostics':0,'errors':[]}
    state=None;messages=[{'role':'system','content':ACTION_INSTRUCTION},{'role':'user','content':(trial/'actor/PROMPT.md').read_text()}]
    transcript=trial/'broker-actions.jsonl';started=None;eval_started=None;phase='initialization'
    try:
        mark_trial_started(trial);started=time.monotonic();deadline=started+1800
        result['remote_identity']=session.start()
        phase='research'
        while deadline-time.monotonic()>15:
            remaining=deadline-time.monotonic()-15
            _json_replace(output/'progress.json',{'model':model,'provider':provider,'phase':'model_request','actions':result['actions'],'diagnostics':result['diagnostics'],'elapsed_seconds':time.monotonic()-started,'remaining_seconds':remaining,'cost':ledger.summary()})
            try:messages,state,request_bytes=trim_history(messages,state,entry,model,environment)
            except ValueError:
                result['research_status']='failed_context_policy';result['research_stop_reason']='common_request_context_bound';break
            if config.get('validated_runtime_identity') and evaluator_fingerprint()!=config['validated_runtime_identity']['evaluator_sha256']:
                raise RuntimeError('Frozen cohort source changed; stop instead of mixing harness versions')
            response=_request(entry,model,messages,state,remaining,environment,catalog,ledger,'research',expected_source=config.get('validated_runtime_identity',{}).get('evaluator_sha256'))
            if not response['ok']:
                result['errors'].append(response['error']);result['research_stop_reason']='provider_request_failed';result['research_status']='failed_provider';break
            reply=response['result'];state=reply.pop('native_state')
            result['actions']+=1
            if reply.get('model_reported')!=model:
                result['research_status']='failed_model_identity';result['research_stop_reason']='provider_reported_different_model';break
            if reply.get('service_tier') not in (None,'default','standard') or reply.get('speed') not in (None,'standard'):
                result['research_status']='failed_service_policy';result['research_stop_reason']='nonstandard_provider_service_tier';break
            if reply.get('received_after_trial_deadline') or deadline-time.monotonic()<=15:
                result['research_stop_reason']='deadline';break
            if (reply.get('usage',{}).get('input_tokens') or 0)>128000:
                result['research_stop_reason']='observed_native_context_limit';result['research_status']='failed_context_policy';break
            text=reply['output_text'];messages.append({'role':'assistant','content':text})
            try:
                action=parse_action(text)
                if action['action']=='done':
                    result['research_stop_reason']='voluntary_completion';break
                tool_remaining=deadline-time.monotonic()-45
                if tool_remaining<=0:
                    result['research_stop_reason']='deadline_cleanup_reserve';break
                if action['action']=='shell':
                    feedback=session.execute(action['command'],timeout=min(120,tool_remaining))
                else:
                    if result['diagnostics']>=3:feedback={'error':'Diagnostic budget exhausted'}
                    else:
                        result['diagnostics']+=1
                        feedback=session.run_diagnostic(action['case_ids'],remaining_seconds=tool_remaining)
            except (ValueError,TypeError,json.JSONDecodeError) as exc:
                feedback={'error':'Invalid action: '+str(exc)[:300]}
            encoded=json.dumps(feedback,default=str)
            if len(encoded)>12000:encoded=encoded[:12000]+' [feedback truncated by common policy]'
            messages.append({'role':'user','content':encoded})
            with transcript.open('a') as stream:stream.write(json.dumps({'action':result['actions'],'model_output':text,'feedback':encoded,'usage':reply['usage'],'request_id':reply['request_id'],'request_bytes':request_bytes})+'\n')
            transcript.chmod(0o600)
        if result['research_status']=='running':
            result['research_stop_reason']=result.get('research_stop_reason','deadline')
            result['research_status']='completed_voluntarily' if result['research_stop_reason']=='voluntary_completion' else 'completed_budget'
        phase='source_finalization';final=session.finish()
        body=final['candidate_bytes']
        if hashlib.sha256(body).hexdigest()!=final['candidate_sha256']:raise ValueError('Remote frozen source hash differs')
        (trial/'actor/kernel/submission.py').write_bytes(body)
        result['seal']=seal_trial(trial)
        try:
            candidate_ast=ast.dump(ast.parse(body),include_attributes=False)
            result['unchanged_baseline_ast']=candidate_ast==ast.dump(ast.parse((ROOT/'third_party/muse_attention_decode/baseline.py').read_bytes()),include_attributes=False)
            result['unchanged_starter_ast']=candidate_ast==ast.dump(ast.parse((ROOT/'third_party/muse_attention_decode/submission.py').read_bytes()),include_attributes=False)
        except (SyntaxError,ValueError,RecursionError):result['candidate_ast_status']='not_parseable'
        result['recorded_within_budget']=result['seal']['recorded_within_budget']
        result['research_elapsed_seconds']=time.monotonic()-started
        result['remote_finish']={k:v for k,v in final.items() if k not in {'candidate_bytes','candidate_b64'}}
        _json_replace(output/'progress.json',{'model':model,'phase':'final_gpu_evaluation','research_status':result['research_status'],'elapsed_seconds':result['research_elapsed_seconds'],'cost':ledger.summary()})
        phase='final_evaluation';eval_started=time.monotonic()
        measured=session.evaluate()
        result['gpu_evaluation_wall_seconds']=time.monotonic()-eval_started
        if isinstance(measured,dict) and 'result' in measured:measured=measured['result']
        validate_final_report(measured,result['seal'])
        expected_runtime=config.get('validated_runtime_identity')
        if expected_runtime and canonical_sha256(measured['comparison_identity'])!=canonical_sha256(expected_runtime):
            raise ValueError('Final GPU runtime differs from the validated cohort control')
        _json_replace(trial/'final-gpu-result.json',measured)
        result.update(status=measured.get('status','failed'),correctness_passed=measured.get('gpu_validated') is True,
            mandatory_passed=sum(c.get('passed') is True for c in measured.get('correctness_cases',[])),
            scored_passed=sum(c.get('passed') is True for c in measured.get('scored_cases',[])),
            speedup=measured.get('speedup'),confidence_interval_95=measured.get('score',{}).get('confidence_interval_95'),
            gpu_evaluation_seconds=measured.get('elapsed_seconds'),comparison_identity=measured.get('comparison_identity'),source_review='pending independent review')
        result['evaluation_status']=measured.get('status','failed')
        if not result['research_status'].startswith('completed_'):result['status']='research_failed_candidate_measured'
        if not result['recorded_within_budget']:result['status']='measured_but_recorded_over_budget'
    except Exception as exc:
        if phase=='source_finalization' and getattr(exc,'remote_type',None)=='CandidateSourceError':
            result.update(status='failed_submission',research_status='failed_submission',research_stop_reason='invalid_final_source',evaluation_status='not_run',error_type='CandidateSourceError',error=str(exc)[:300])
        else:result.update(status='infrastructure_or_controller_failed',research_status='failed_infrastructure_or_budget',error_type=type(exc).__name__,error=str(exc)[:300])
    finally:
        if started is not None and 'research_elapsed_seconds' not in result:
            result['research_elapsed_seconds']=time.monotonic()-started
        if eval_started is not None and 'gpu_evaluation_wall_seconds' not in result:
            result['gpu_evaluation_wall_seconds']=time.monotonic()-eval_started
        try:session.stop()
        except Exception as exc:result['cleanup_error']=type(exc).__name__
        result['cost_requests']=[r for r in ledger.data['requests'] if r['model']==model]
        result['api_cost_estimate_usd']=sum(r.get('estimated_usd') or 0 for r in result['cost_requests'])
        result['research_api_cost_estimate_usd']=sum(r.get('estimated_usd') or 0 for r in result['cost_requests'] if r['phase']=='research')
        result['preflight_api_cost_estimate_usd']=sum(r.get('estimated_usd') or 0 for r in result['cost_requests'] if r['phase']!='research')
        result['api_cost_unknown_reserve_usd']=sum(r['reserved_usd'] for r in result['cost_requests'] if r.get('estimated_usd') is None)
        rate=float(config.get('gcp_node_reference_usd_per_hour',42.40))
        result['gcp_node_reference_usd_per_hour']=rate
        result['gcp_phase_cost_estimate_usd']=(result.get('research_elapsed_seconds',0)+result.get('gpu_evaluation_wall_seconds',0))*rate/3600
        result['phase_total_cost_estimate_usd']=result['api_cost_estimate_usd']+result['gcp_phase_cost_estimate_usd']
        result['cost_excludes_shared_setup_disk_network_tax_and_unresolved_requests']=True
        _json_replace(output/f'{trial_id}.json',result)
    return result


def render_report(rows:list[dict],path:Path,budget:dict):
    lines=['# Seven-model Muse attention comparison','',
           'One 1800-second research opportunity per model. Voluntary completion, failures, and budget stops are recorded. API costs are tariff estimates, not invoices. Source review is independent of numeric correctness.','',
           '| Model | Research outcome | GPU outcome | Mandatory | Scored | Measured speedup | API estimate USD | GCP phase estimate USD | Known phase subtotal USD | Unknown reserve USD |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        speed=f"{r['speedup']:.4f}x" if r.get('correctness_passed') and r.get('speedup') else '—'
        research=r.get('research_status',r['status'])+': '+r.get('research_stop_reason','')
        lines.append(f"| {r['model']} | {research} | {r.get('evaluation_status',r['status'])} | {r.get('mandatory_passed','—')} | {r.get('scored_passed','—')} | {speed} | {r.get('api_cost_estimate_usd',0):.4f} | {r.get('gcp_phase_cost_estimate_usd',0):.4f} | {r.get('phase_total_cost_estimate_usd',0):.4f} | {r.get('api_cost_unknown_reserve_usd',0):.2f} |")
    lines+=['','## Budget accounting','',json.dumps(budget,indent=2),'',
            'Speedups are raw correctness-gated observations, not a qualified ranking: independent source review remains required. Research failure is distinct from correctness of a partially produced artifact. API totals include recorded preflight calls; JSON separates preflight and research. GCP phase estimates use research plus final-evaluation seconds at the frozen reference node rate; they exclude shared build/baseline/idle overhead, disk/network/tax and unresolved API requests. The lifecycle receipt is needed to reconcile total compute cost. Published tariffs are estimates, not invoices. One trial per model is an observation, not a statistically established model ranking.']
    path.write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','preflight','run'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/cohort-live.json')
    p.add_argument('--credential-context',type=Path,default=DEFAULT_CONFIG_ROOT/'credential-context.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-usd',type=float,default=1000)
    p.add_argument('--gcp-reserve-usd',type=float,default=400)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    cfg=json.loads(args.config.read_text());providers=load_configuration()
    catalog=json.loads((ROOT/'configs/pricing.example.json').read_text())
    current_identity=cohort_identity(cfg,providers,catalog)
    if len(cfg['models'])!=7 or cfg.get('wall_clock_seconds')!=1800 or cfg.get('repetitions_per_model')!=1:raise ValueError('This cohort requires seven models, one 1800-second trial each')
    if not verify_vendor()['ok']:raise ValueError('Vendored source changed')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.command=='plan' or not args.execute:
        print(json.dumps({'models':cfg['models'],'research_hours':3.5,'final_evaluation_extra':True,'credential_context':str(args.credential_context),'keys_on_local_controller':True,'shell_and_gpu':'GCP','budget_usd':args.budget_usd,'gcp_reserve_usd':args.gcp_reserve_usd,'contingency_usd':100,'no_calls_made':True},indent=2));return 0
    if cfg.get('inference_enabled') is not True:raise ValueError('Actual calls require an explicitly enabled live cohort configuration')
    env=credential_environment(args.credential_context)
    current_identity['routing']=routing_identity(providers,env)
    ledger=CostLedger(output/'cost-ledger.json',args.budget_usd,args.gcp_reserve_usd,100)
    if args.command=='preflight':
        result=preflight(cfg,providers,env,catalog,ledger,output/'preflight.json');print(json.dumps(result,indent=2));return 0 if result['status']=='ready' else 2
    readiness=json.loads((output/'preflight.json').read_text())
    if readiness.get('status')!='ready' or readiness.get('identity')!=current_identity:raise ValueError('All seven live API preflights must bind the current frozen roster/config/pricing/source before GCP runs')
    cloud=cfg.get('cloud')
    if not cloud:raise ValueError('Stage the GCP actor/evaluator runtime and configure cloud fields before running')
    validated=cfg.get('validated_runtime_identity')
    if not isinstance(validated,dict) or validated.get('evaluator_sha256')!=evaluator_fingerprint() or validated.get('protocol_sha256')!=canonical_sha256(load_profile(cfg['profile'])):
        raise ValueError('A live full H200 control must bind this exact evaluator/profile before model clocks start')
    from .remote import RemoteSession
    order=list(cfg['models']);random.Random(cfg.get('order_seed',20261107)).shuffle(order)
    _json_new(output/'frozen-cohort.json',{**cfg,'actual_order':order,'provider_config_sha256':canonical_sha256(providers),'pricing_sha256':canonical_sha256(catalog),'executed_at':dt.datetime.now(dt.timezone.utc).isoformat()})
    rows=[];aborted=False
    for row in order:
        factory=lambda trial:RemoteSession(cloud,trial,Path(cfg['profile']),seconds=1800,gpu='0')
        rows.append(run_trial(row,cfg,providers,env,catalog,ledger,output,factory))
        render_report(rows,output/'REPORT.md',ledger.summary())
        _json_replace(output/'comparison.json',{'models':rows,'budget':ledger.summary()})
        identities=[r['comparison_identity'] for r in rows if r.get('comparison_identity')]
        if len({canonical_sha256(i) for i in identities})>1:
            aborted=True;_json_replace(output/'cohort-aborted.json',{'reason':'Physical GPU/runtime changed; results cannot be compared as identical conditions'});break
        if any(r.get('status')=='infrastructure_or_controller_failed' or r.get('cleanup_error') for r in rows):aborted=True;break
    print(json.dumps({'report':str(output/'REPORT.md'),'completed_models':len(rows),'budget':ledger.summary()},indent=2))
    if aborted:_json_replace(output/'cohort-aborted.json',{'reason':'Runtime, cleanup, infrastructure or unresolved cost condition; partial results are not a complete matched comparison','completed_rows':len(rows),'comparison_valid':False})
    return 0 if len(rows)==7 and not aborted else 2


if __name__=='__main__':raise SystemExit(main())
