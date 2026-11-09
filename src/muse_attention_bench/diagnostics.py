"""Limited public-case GPU feedback; never a final competition score."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .container import LOCAL_IMAGE, run_container
from .protocol import BASE_IMAGE, load_profile, canonical_sha256


def select_public_cases(protocol: dict, case_ids: list[str]) -> list[dict]:
    if not isinstance(case_ids, list) or not 1 <= len(case_ids) <= 2 or len(set(case_ids)) != len(case_ids):
        raise ValueError('Select one or two distinct public scored-case IDs')
    public = {c['id']: c for c in protocol['scored_cases']}
    if any(not isinstance(k, str) or k not in public for k in case_ids):
        raise ValueError('Diagnostics expose public scored cases only')
    return [copy.deepcopy(public[k]) for k in case_ids]


def evaluate_diagnostic(candidate: Path, output: Path, protocol: dict, case_ids: list[str], *, seed: int) -> dict:
    """Device entry point. Final evaluator's implementation stays unchanged."""
    from .gpu import evaluate
    cases = select_public_cases(protocol, case_ids)
    reduced = copy.deepcopy(protocol)
    reduced['test_cases'] = []
    reduced['scored_cases'] = cases
    reduced['seed'] = seed
    reduced['timing'].update(repeats=6, calibration_repeats=20, bootstrap_samples=100)
    result = evaluate(candidate, output, reduced)
    result.update(evaluation_kind='public-case-diagnostic', diagnostic_only=True,
                  full_suite_passed=False, parent_protocol_sha256=canonical_sha256(protocol),
                  selected_case_ids=case_ids, diagnostic_speedup=result.pop('speedup', None),
                  speedup=None)
    if 'score' in result:
        result['diagnostic_score'] = result.pop('score')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def diagnostic_command(candidate: Path, profile: Path, output: Path, case_ids: list[str], *, gpu='0', seed=0) -> list[str]:
    return ['docker', 'run', '--rm', '--gpus', f'device={gpu}', '--network', 'none',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', '512', '--shm-size', '2g', '--ulimit', 'memlock=-1',
            '--tmpfs', '/tmp:rw,exec,nosuid,size=8g', '--tmpfs', '/root/.cache:rw,exec,nosuid,size=8g',
            '--mount', f'type=bind,src={candidate.resolve()},dst=/candidate/submission.py,readonly',
            '--mount', f'type=bind,src={profile.resolve()},dst=/protocol.json,readonly',
            '--mount', f'type=bind,src={output.parent.resolve()},dst=/results',
            '--env', 'MUSE_ATTENTION_BENCH_CONTAINER=1',
            '--env', 'MUSE_ATTENTION_BENCH_BASE_IMAGE_DIGEST='+BASE_IMAGE.split('@',1)[1],
            '--env', 'MUSE_ATTENTION_BENCH_SXM_ATTESTED=1',
            '--env', 'TORCH_EXTENSIONS_DIR=/tmp/torch-extensions',
            '--env', 'TRITON_CACHE_DIR=/tmp/triton', '--env', 'CUDA_CACHE_PATH=/tmp/cuda-cache',
            '--entrypoint', 'python3', LOCAL_IMAGE, '-m', 'muse_attention_bench.diagnostics',
            'internal', '--candidate', '/candidate/submission.py', '--profile', '/protocol.json',
            '--output', '/results/'+output.name, '--seed', str(seed), '--cases', *case_ids]


class DiagnosticBridge:
    """Trusted per-trial counter and snapshotter; pause the actor before calling."""
    def __init__(self, trial_dir: Path, profile_path: Path, *, gpu='0', max_requests=3,
                 max_cases=2, timeout_seconds=300):
        self.trial_dir = Path(trial_dir).resolve()
        self.profile_path = Path(profile_path).resolve()
        self.protocol = load_profile(self.profile_path)
        if max_requests != 3 or max_cases != 2 or timeout_seconds <= 0:
            raise ValueError('Cohort diagnostics require three requests and at most two cases')
        self.gpu, self.max_requests, self.max_cases, self.timeout_seconds = gpu, max_requests, max_cases, timeout_seconds
        self.directory = self.trial_dir/'evaluator/diagnostics'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.used = len(list(self.directory.glob('request-*')))

    def run(self, case_ids: list[str] | None = None, *, remaining_seconds: float) -> dict:
        case_ids = case_ids or ['scored-01','scored-02']
        select_public_cases(self.protocol, case_ids)
        if self.used >= self.max_requests:
            return {'status':'budget_exhausted','diagnostic_requests_used':self.used,'inference_or_gpu_called':False}
        if not isinstance(remaining_seconds,(int,float)) or isinstance(remaining_seconds,bool) or not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            return {'status':'deadline_reached','diagnostic_requests_used':self.used,'inference_or_gpu_called':False}
        actor=self.trial_dir/'actor'
        source=actor/'kernel/submission.py'
        resolved=source.resolve(strict=True)
        if source.is_symlink() or not resolved.is_relative_to(actor.resolve()) or not resolved.is_file():
            raise ValueError('Diagnostic candidate must be a regular actor file')
        if resolved.stat().st_size > 2*1024*1024:
            raise ValueError('Diagnostic candidate exceeds the 2 MiB source limit')
        self.used += 1
        # The same request ordinal uses the same hidden seed bank across models;
        # different ordinals still change graph inputs. It is not actor feedback.
        diagnostic_seed=int.from_bytes(hashlib.sha256((canonical_sha256(self.protocol)+':diagnostic:'+str(self.used)).encode()).digest()[:4],'little') & 0x7fffffff
        request=self.directory/f'request-{self.used:02d}'
        request.mkdir(exist_ok=False)
        snapshot=request/'submission.py';body=resolved.read_bytes();snapshot.write_bytes(body);snapshot.chmod(0o444)
        output=request/'result.json'
        manifest={'selected_case_ids':case_ids,'candidate_sha256':hashlib.sha256(body).hexdigest(),
                  'parent_protocol_sha256':canonical_sha256(self.protocol),'diagnostic_only':True,
                  'diagnostic_seed':diagnostic_seed,
                  'diagnostic_requests_used':self.used,'timeout_seconds':min(self.timeout_seconds,float(remaining_seconds))}
        (request/'request.json').write_text(json.dumps(manifest,indent=2)+'\n')
        started=time.monotonic()
        try:
            code=run_container(diagnostic_command(snapshot,self.profile_path,output,case_ids,gpu=self.gpu,seed=diagnostic_seed),
                               timeout=min(self.timeout_seconds,float(remaining_seconds)))
            if output.is_symlink() or (output.exists() and not output.is_file()) or (output.is_file() and output.stat().st_size>8*1024*1024):
                raise ValueError('Diagnostic output is not an admissible regular report')
            if not output.is_file():
                feedback={'status':'infrastructure_failure','error':'Diagnostic produced no report','returncode':code}
            else:
                report=json.loads(output.read_text())
                if report.get('candidate_sha256')!=manifest['candidate_sha256'] or report.get('parent_protocol_sha256')!=manifest['parent_protocol_sha256']:
                    raise ValueError('Diagnostic report identity mismatch')
                feedback={'status':report.get('status'),'diagnostic_only':True,'full_suite_passed':False,
                          'selected_case_ids':case_ids,'diagnostic_speedup':report.get('diagnostic_speedup'),
                          'diagnostic_confidence_interval_95':report.get('diagnostic_score',{}).get('confidence_interval_95'),
                          'cases':[{k:r.get(k) for k in ['id','passed','speedup','baseline_us','candidate_us']} for r in report.get('scored_cases',[])],
                          'error':report.get('error'),'returncode':code}
        except (OSError,TimeoutError,ValueError) as exc:
            feedback={'status':'diagnostic_failed','error':{'type':type(exc).__name__,'message':str(exc)}}
        feedback.update(diagnostic_requests_used=self.used,diagnostic_request_budget=self.max_requests,
                        elapsed_seconds=time.monotonic()-started)
        (request/'feedback.json').write_text(json.dumps(feedback,indent=2,allow_nan=False)+'\n')
        return feedback


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['internal'])
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(argv)
    if os.environ.get('MUSE_ATTENTION_BENCH_CONTAINER')!='1':
        parser.error('Device diagnostics are container-only')
    result=evaluate_diagnostic(args.candidate,args.output,load_profile(args.profile),args.cases,seed=args.seed)
    return 0 if result.get('status')=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
