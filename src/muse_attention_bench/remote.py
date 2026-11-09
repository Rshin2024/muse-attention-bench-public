"""IAP SSH RPC: API credentials remain local; actor and GPU work run on GCP.

The remote server is trusted controller code. It never receives provider keys.
Only the actor directory is mounted in the model's CPU sandbox. A transport
failure must abort the trial until cleanup is established; do not score it zero.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import queue
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time

from .actor import DockerActor
from .diagnostics import DiagnosticBridge
from .container import inspect_image, run_command, run_container
from .protocol import ROOT, canonical_sha256, load_profile

MAX_FRAME=8*1024*1024
MAX_SOURCE=2*1024*1024
HEARTBEAT_INTERVAL=3.
HEARTBEAT_LEASE=15.
INITIAL_FILES={'PROMPT.md','kernel/submission.py','kernel/TASK.md','kernel/AGENTS.md','kernel/baseline.py'}
IDENTIFIER=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')


class RemoteError(RuntimeError):
    def __init__(self,message,*,remote_type=None):
        super().__init__(message)
        self.remote_type=remote_type


class CandidateSourceError(ValueError):
    """The actor did not leave an admissible final source, not a cloud fault."""


def _encoded(body: bytes) -> str:
    return base64.b64encode(body).decode('ascii')


def _decoded(value: str, limit=MAX_SOURCE) -> bytes:
    if not isinstance(value,str) or len(value)>limit*2:
        raise ValueError('Invalid or oversized encoded file')
    body=base64.b64decode(value,validate=True)
    if len(body)>limit:raise ValueError('Decoded file exceeds limit')
    return body


def _private_new(path: Path, body: bytes, mode=0o600):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('xb') as stream:stream.write(body)
    path.chmod(mode)


class RemoteTrial:
    """One model's remote lifecycle; watchdog freezes source on its deadline."""
    def __init__(self, run_root: Path):
        self.run_root=Path(run_root).resolve()
        self.run_root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.actor=None
        self.trial=None
        self.profile_path=None
        self.bridge=None
        self.started=None
        self.seconds=None
        self.final=None
        self.actor_restarts=0
        self.diagnostic_actions=0
        self.lock=threading.RLock()
        self.watchdog_done=threading.Event()

    def initialize(self, payload: dict) -> dict:
        if self.actor is not None:raise RemoteError('Session already initialized')
        trial_id=payload.get('trial_id','')
        if not isinstance(trial_id,str) or not IDENTIFIER.fullmatch(trial_id):
            raise ValueError('Invalid trial identifier')
        seconds=payload.get('seconds',1800)
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not 1<=seconds<=1800:
            raise ValueError('Research budget must be between one and 1800 seconds')
        files=payload.get('files')
        if not isinstance(files,dict) or not {'PROMPT.md','kernel/submission.py','kernel/TASK.md','kernel/AGENTS.md'}<=set(files) or not set(files)<=INITIAL_FILES:
            raise ValueError('Initial actor files differ from the allowed file set')
        trial=self.run_root/trial_id
        trial.mkdir(mode=0o700,exist_ok=False)
        for relative,value in files.items():
            body=_decoded(value)
            body.decode('utf-8')
            _private_new(trial/'actor'/relative,body)
        profile_raw=_decoded(payload['profile_b64'])
        if hashlib.sha256(profile_raw).hexdigest()!=payload['profile_file_sha256']:
            raise ValueError('Transported profile bytes differ')
        profile_path=trial/'evaluator/protocol.json'
        _private_new(profile_path,profile_raw,0o444)
        profile=load_profile(profile_path)
        if canonical_sha256(profile)!=payload['protocol_sha256']:
            raise ValueError('Transported canonical profile differs')
        actor=DockerActor(trial/'actor',seconds=seconds)
        # Own the container object before start: connection loss can interrupt
        # Docker after it creates a container but before start() returns.
        self.actor,self.trial,self.profile_path=actor,trial,profile_path
        try:identity=actor.start()
        except BaseException:
            actor.stop()
            raise
        self.bridge=DiagnosticBridge(trial,profile_path,gpu=str(payload.get('gpu','0')))
        self.gpu=str(payload.get('gpu','0'))
        self.started=time.monotonic();self.seconds=seconds
        actor.deadline=self.started+seconds
        record={'trial_id':trial_id,'seconds':seconds,'protocol_sha256':canonical_sha256(profile),
                'actor_identity':identity,'api_credentials_received':False,
                'initial_hashes':{k:hashlib.sha256(_decoded(v)).hexdigest() for k,v in files.items()}}
        _private_new(trial/'remote-manifest.json',json.dumps(record,indent=2).encode()+b'\n')
        threading.Thread(target=self._watchdog,daemon=True).start()
        return {**record,'research_elapsed_seconds':0.0}

    def remaining(self):
        return max(0.,self.seconds-(time.monotonic()-self.started)) if self.started is not None else 0.

    def _watchdog(self):
        if not self.watchdog_done.wait(self.remaining()):
            try:self.finish()
            except Exception:
                # The trusted client sees failure/missing final evidence; no pass
                # is manufactured by watchdog failure.
                pass

    def finish(self) -> dict:
        with self.lock:
            if self.final is not None:return self.final
            if self.actor is None:raise RemoteError('Session is not initialized')
            cleanup=self.actor.stop()
            elapsed=time.monotonic()-self.started
            source=self.trial/'actor/kernel/submission.py'
            try:resolved=source.resolve(strict=True)
            except OSError:raise CandidateSourceError('Final candidate source is missing') from None
            if source.is_symlink() or not resolved.is_relative_to((self.trial/'actor').resolve()) or not resolved.is_file():
                raise CandidateSourceError('Final source is not a regular actor file')
            if resolved.stat().st_size>MAX_SOURCE:raise CandidateSourceError('Final source exceeds 2 MiB')
            body=resolved.read_bytes()
            try:body.decode('utf-8')
            except UnicodeError:raise CandidateSourceError('Final source is not readable UTF-8') from None
            _private_new(self.trial/'final/submission.py',body,0o444)
            self.final={'candidate_b64':_encoded(body),'candidate_sha256':hashlib.sha256(body).hexdigest(),
                        'protocol_sha256':canonical_sha256(load_profile(self.profile_path)),
                        'research_elapsed_seconds':elapsed,'research_limit_seconds':self.seconds,
                        'deadline_reached':elapsed>=self.seconds,'actor_cleanup':cleanup,
                        'actor_restarts':self.actor_restarts,
                        'source_frozen_after_actor_stop':True}
            _private_new(self.trial/'final/seal.json',json.dumps({k:v for k,v in self.final.items() if k!='candidate_b64'},indent=2).encode()+b'\n')
            self.watchdog_done.set()
            return self.final

    def close(self):
        """Source failure does not make a confirmed container removal unknown."""
        if self.actor is None:return {'not_initialized':True}
        try:
            final=self.finish()
            return {'actor_cleanup':final['actor_cleanup'],'source_sealed':True,
                    'candidate_sha256':final['candidate_sha256']}
        except CandidateSourceError as exc:
            cleanup=self.actor.stop()
            self.watchdog_done.set()
            return {'actor_cleanup':{**cleanup,'container_removed':True,'all_actor_processes_stopped':True},
                    'source_sealed':False,'source_freeze_error':{'type':type(exc).__name__,'message':str(exc)}}

    def execute(self,payload):
        if self.final is not None or self.remaining()<=0:
            self.finish();raise RemoteError('Research deadline reached or source sealed')
        if self.actor.closed:
            # A per-action timeout killed every old process, but it must not
            # silently terminate the model's remaining research opportunity.
            # Recreate only the CPU sandbox, preserving files and global clock.
            replacement=DockerActor(self.trial/'actor',seconds=self.remaining(),
                                    deadline=self.started+self.seconds)
            replacement.start();self.actor=replacement;self.actor_restarts+=1
        result=self.actor.execute(payload.get('command'),timeout=min(float(payload.get('timeout',120)),self.remaining()))
        result['actor_reset_required']=bool(self.actor.closed and self.remaining()>0)
        result['research_elapsed_seconds']=time.monotonic()-self.started
        return result

    def diagnostic(self,payload):
        if self.final is not None or self.remaining()<=0:
            self.finish();raise RemoteError('Research deadline reached or source sealed')
        if self.diagnostic_actions>=3:
            return {'status':'budget_exhausted','diagnostic_requests_used':3,'gpu_called':False}
        self.diagnostic_actions+=1
        requested=payload.get('remaining_seconds',self.remaining())
        if isinstance(requested,bool) or not isinstance(requested,(int,float)) or not math.isfinite(requested) or requested<=0:
            return {'status':'deadline_reached','diagnostic_requests_used':self.diagnostic_actions,'gpu_called':False}
        self.actor.pause()
        try:
            result=self.bridge.run(payload.get('case_ids'),remaining_seconds=min(self.remaining(),requested))
        except (ValueError,TypeError,FileNotFoundError) as exc:
            result={'status':'invalid_diagnostic','error':{'type':type(exc).__name__,'message':str(exc)[:500]},'gpu_called':False}
        finally:
            if self.remaining()>0 and self.final is None:self.actor.unpause()
        result['diagnostic_requests_used']=self.diagnostic_actions
        result['research_elapsed_seconds']=time.monotonic()-self.started
        return result

    def evaluate(self):
        frozen=self.finish()
        directory=self.trial/'final-evaluation';directory.mkdir(exist_ok=False)
        output=directory/'result.json'
        protocol=load_profile(self.profile_path)
        from .gpu import evaluator_fingerprint
        expected_image=inspect_image()['image_id']
        expected_evaluator=evaluator_fingerprint()
        started=time.monotonic()
        try:
            code=run_container(run_command(self.trial/'final/submission.py',output,self.profile_path,
                                          gpu=self.gpu,attest_sxm=True),
                               timeout=protocol['timing']['run_timeout_seconds'])
        except TimeoutError:code=124
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise RemoteError('GPU output is not a regular owned report; abort the cohort')
        if output.is_file() and output.stat().st_size>MAX_FRAME:
            raise RemoteError('GPU output exceeds the report size limit; abort the cohort')
        if not output.is_file():
            if code==0:raise RemoteError('Successful GPU process produced no result; controller state is unknown')
            # A crashing generated kernel is a model outcome only after a fresh
            # trusted baseline proves that the GPU/runtime is healthy again.
            health_dir=directory/'crash-health';health_dir.mkdir(exist_ok=False)
            baseline=health_dir/'baseline.py'
            baseline.write_bytes((ROOT/'third_party/muse_attention_decode/baseline.py').read_bytes());baseline.chmod(0o444)
            health_path=health_dir/'result.json'
            try:
                health_code=run_container(run_command(baseline,health_path,self.profile_path,gpu=self.gpu,
                                                     attest_sxm=True,correctness_only=True),
                                          timeout=min(300,protocol['timing']['run_timeout_seconds']))
                if health_path.is_symlink() or not health_path.is_file() or health_path.stat().st_size>MAX_FRAME:
                    raise RemoteError('Trusted health output is not an admissible report')
                health=json.loads(health_path.read_text())
            except Exception as exc:
                raise RemoteError('Trusted GPU health recovery failed; abort the cohort') from exc
            identity=health.get('comparison_identity',{})
            required=['protocol_sha256','execution_backend','image_digest','gpu_name','gpu_uuid',
                      'torch_version','cuda_version','driver_version','evaluator_sha256']
            tests=health.get('correctness_cases',[]);scored=health.get('scored_cases',[])
            healthy=(health_code==0 and health.get('status')=='correctness_passed' and health.get('gpu_validated') is True
                     and isinstance(identity,dict) and all(isinstance(identity.get(k),str) and identity[k] for k in required)
                     and bool(identity.get('sglang_commit') or identity.get('sglang_source_sha256'))
                     and health.get('candidate_sha256')==hashlib.sha256(baseline.read_bytes()).hexdigest()
                     and health.get('protocol_sha256')==frozen['protocol_sha256']
                     and identity.get('protocol_sha256')==frozen['protocol_sha256']
                     and identity.get('image_digest')==expected_image
                     and identity.get('evaluator_sha256')==expected_evaluator
                     and isinstance(tests,list) and len(tests)==16
                     and all(isinstance(r,dict) and r.get('passed') is True for r in tests)
                     and {r.get('id') for r in tests}=={f'test-{i:02d}' for i in range(1,17)}
                     and isinstance(scored,list) and len(scored)==12
                     and all(isinstance(r,dict) and r.get('passed') is True for r in scored)
                     and {r.get('id') for r in scored}=={f'scored-{i:02d}' for i in range(1,13)})
            if not healthy:raise RemoteError('Trusted GPU health recovery was incomplete or failed; abort the cohort')
            report={'schema_version':1,'benchmark':'muse_attention_decode','status':'failed','gpu_validated':False,
                    'candidate_sha256':frozen['candidate_sha256'],'protocol_sha256':frozen['protocol_sha256'],
                    'comparison_identity':identity,'official_score':False,'speedup':None,
                    'correctness_cases':[],'scored_cases':[],'controller_returncode':code,
                    'error':{'type':'CandidateProcessCrash','message':'Candidate process exited without an admissible report','returncode':code},
                    'health_recovery':{'status':'passed','trusted_baseline_result':'crash-health/result.json',
                                       'candidate_was_not_retried':True},
                    'source_review':{'approved':False,'required_separately':True},
                    'elapsed_seconds':time.monotonic()-started}
            _private_new(output,json.dumps(report,indent=2,allow_nan=False).encode()+b'\n')
            return report
        report=json.loads(output.read_text())
        if report.get('candidate_sha256')!=frozen['candidate_sha256'] or report.get('protocol_sha256')!=frozen['protocol_sha256']:
            raise RemoteError('Final result is not bound to the frozen source/profile')
        report['controller_returncode']=code
        return report


def _serve(run_root: Path):
    """Protocol output has its own FD, so child compiler logs cannot corrupt RPC."""
    trial=RemoteTrial(run_root)
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    log_path=run_root/'rpc-server.log';log_path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    termination_requested=threading.Event()
    def terminate(signum,frame):
        if termination_requested.is_set():return
        termination_requested.set()
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,terminate)
    frames=queue.Queue(maxsize=2)
    connection_lock=threading.Lock()
    connection_closed=threading.Event()
    operation_active=threading.Event()
    reader_done=threading.Event()
    last_frame=[time.monotonic()]

    def disconnected():
        with connection_lock:
            connection_closed.set()
            interrupted=operation_active.is_set()
        try:frames.put_nowait(None)
        except queue.Full:pass
        if interrupted:
            os.kill(os.getpid(),signal.SIGTERM)

    def lease_watchdog():
        # IAP can leave the remote TCP/SSH channel open after its local process
        # dies. A trusted heartbeat lease bounds cleanup independently of EOF.
        while not reader_done.wait(min(.5,HEARTBEAT_LEASE/4)):
            if time.monotonic()-last_frame[0]>HEARTBEAT_LEASE:
                disconnected();return

    def read_frames():
        """Observe EOF even while the main thread waits on a GPU child."""
        selector=selectors.DefaultSelector()
        selector.register(sys.stdin.fileno(),selectors.EVENT_READ)
        pending=bytearray()
        try:
            while not reader_done.is_set():
                if not selector.select(.2):continue
                block=os.read(sys.stdin.fileno(),65536)
                if not block:break
                pending.extend(block)
                while b'\n' in pending:
                    line,_,rest=pending.partition(b'\n');pending=bytearray(rest)
                    if len(line)+1>MAX_FRAME:return
                    value=json.loads(line)
                    last_frame[0]=time.monotonic()
                    if isinstance(value,dict) and value.get('v')==1 and value.get('op')=='heartbeat':continue
                    try:frames.put(bytes(line),timeout=.5)
                    except queue.Full:return
                if len(pending)>=MAX_FRAME:break
        finally:
            selector.close()
            disconnected()

    reader=threading.Thread(target=read_frames,daemon=True)
    reader.start()
    lease=threading.Thread(target=lease_watchdog,daemon=True)
    lease.start()
    try:
        while True:
            line=frames.get()
            if line is None:break
            request=json.loads(line)
            with connection_lock:
                if connection_closed.is_set():break
                operation_active.set()
            ident=request.get('id')
            op=request.get('op')
            response={'v':1,'id':ident}
            # Protect RPC stdout from both Python and inherited native child FDs.
            saved_out,saved_err=os.dup(1),os.dup(2)
            with log_path.open('ab',buffering=0) as logfile:
                sys.stdout.flush();sys.stderr.flush()
                os.dup2(logfile.fileno(),1);os.dup2(logfile.fileno(),2)
                try:
                    if request.get('v')!=1:raise RemoteError('Unsupported RPC version')
                    payload=request.get('payload') or {}
                    if op=='init':result=trial.initialize(payload)
                    elif op=='shell':result=trial.execute(payload)
                    elif op=='gpu_test':result=trial.diagnostic(payload)
                    elif op=='finish':result=trial.finish()
                    elif op=='evaluate':result=trial.evaluate()
                    elif op=='stop':result=trial.close()
                    else:raise RemoteError('Unknown RPC operation')
                    response.update(ok=True,result=result)
                except Exception as exc:
                    response.update(ok=False,error={'type':type(exc).__name__,'message':str(exc)[:3000]})
                finally:
                    sys.stdout.flush();sys.stderr.flush()
                    os.dup2(saved_out,1);os.dup2(saved_err,2);os.close(saved_out);os.close(saved_err)
                    with connection_lock:operation_active.clear()
            protocol.write(json.dumps(response,separators=(',',':'),allow_nan=False).encode()+b'\n')
            if op=='stop':break
    finally:
        # Do not let a second SSH signal interrupt owned-container cleanup.
        termination_requested.set()
        reader_done.set()
        reader.join(timeout=1)
        lease.join(timeout=1)
        if trial.actor is not None:
            try:trial.close()
            except Exception:
                try:trial.actor.stop()
                except Exception:pass
        protocol.close()


class RemoteSession:
    """One persistent IAP SSH channel per model, carrying no provider keys."""
    def __init__(self, cloud: dict, local_trial: Path, profile_path: Path, *, seconds=1800,gpu='0'):
        self.cloud=dict(cloud);self.local_trial=Path(local_trial).resolve();self.profile_path=Path(profile_path).resolve()
        for key in ('project','zone','instance','remote_repo','remote_run_root'):
            if not isinstance(self.cloud.get(key),str) or not self.cloud[key] or '\n' in self.cloud[key]:
                raise ValueError('GCP remote configuration is incomplete')
        if not self.cloud['remote_repo'].startswith('/') or not self.cloud['remote_run_root'].startswith('/'):
            raise ValueError('Remote paths must be absolute')
        self.seconds,self.gpu=seconds,str(gpu)
        self.process=None;self.sequence=0;self.pending=bytearray();self.transport_noise=bytearray()
        self.stderr=None
        self.rpc_lock=threading.Lock()
        self.write_lock=threading.Lock()
        self.heartbeat_done=threading.Event()
        self.heartbeat_thread=None
        self.transport_failed=False

    def ssh_command(self):
        remote=shlex.join(['sudo','-n','env','PYTHONPATH='+self.cloud['remote_repo']+'/src',
                          'DOCKER_HOST=unix:///var/run/docker.sock','DOCKER_CONFIG='+self.cloud['remote_repo']+'/host-docker',
                          'python3','-u','-m','muse_attention_bench.remote','serve','--run-root',self.cloud['remote_run_root']])
        return ['gcloud','compute','ssh',self.cloud['instance'],'--project',self.cloud['project'],
                '--zone',self.cloud['zone'],'--tunnel-through-iap','--ssh-flag=-T',
                '--ssh-flag=-F/dev/null','--ssh-flag=-oForwardAgent=no','--ssh-flag=-oSendEnv=-*','--command',remote]

    def _rpc(self,op,payload=None,timeout=60):
        with self.rpc_lock:
            return self._exchange(op,payload,timeout)

    def _abort_transport(self):
        """A partial frame cannot be reused; no remote-cleanup claim is made."""
        self.transport_failed=True
        self.heartbeat_done.set()
        if self.process is None:return
        try:
            if self.process.stdin:self.process.stdin.close()
        except OSError:pass
        pid=getattr(self.process,'pid',None)
        if pid is not None and self.process.poll() is None:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.process.wait(timeout=5)
        if self.heartbeat_thread is not None and self.heartbeat_thread is not threading.current_thread():
            self.heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL+1)

    def _send_frame(self,encoded,deadline):
        # Pipe writes are included in the same deadline as the response. A
        # stalled SSH process must not block a large init/source frame forever.
        remaining=deadline-time.monotonic()
        if remaining<=0 or not self.write_lock.acquire(timeout=remaining):
            raise TimeoutError('GCP frame write lock exceeded its deadline')
        try:
            input_fd=self.process.stdin.fileno();os.set_blocking(input_fd,False)
            writer=selectors.DefaultSelector();writer.register(input_fd,selectors.EVENT_WRITE)
            pending=memoryview(encoded)
            try:
                while pending:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('GCP RPC request write exceeded its deadline')
                    if not writer.select(min(.2,remaining)):continue
                    try:count=os.write(input_fd,pending)
                    except BlockingIOError:continue
                    except OSError:raise RemoteError('GCP RPC request pipe failed') from None
                    if count<=0:raise RemoteError('GCP RPC request pipe closed')
                    pending=pending[count:]
            finally:writer.close()
        finally:self.write_lock.release()

    def _heartbeat(self):
        while not self.heartbeat_done.wait(HEARTBEAT_INTERVAL):
            try:self._send_frame(b'{"v":1,"op":"heartbeat"}\n',time.monotonic()+HEARTBEAT_INTERVAL)
            except Exception:
                if not self.transport_failed:self._abort_transport()
                return

    def _exchange(self,op,payload=None,timeout=60):
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('RPC timeout must be finite and positive')
        if self.process is None or self.process.poll() is not None:raise RemoteError('GCP SSH session is not running')
        self.sequence+=1
        request={'v':1,'id':self.sequence,'op':op,'payload':payload or {}}
        encoded=json.dumps(request,separators=(',',':')).encode()+b'\n'
        if len(encoded)>MAX_FRAME:raise ValueError('RPC request too large')
        deadline=time.monotonic()+timeout
        try:self._send_frame(encoded,deadline)
        except Exception:
            self._abort_transport();raise
        fd=self.process.stdout.fileno()
        select=selectors.DefaultSelector();select.register(fd,selectors.EVENT_READ)
        try:
            while time.monotonic()<deadline:
                while b'\n' in self.pending:
                    line,_,rest=self.pending.partition(b'\n');self.pending=bytearray(rest)
                    try:response=json.loads(line)
                    except (ValueError,UnicodeError):
                        self.transport_noise.extend(line[:max(0,262144-len(self.transport_noise))]);continue
                    if not isinstance(response,dict) or response.get('id')!=self.sequence:
                        self._abort_transport()
                        raise RemoteError('GCP RPC response identity mismatch')
                    if response.get('ok') is not True:
                        error=response.get('error',{});raise RemoteError(str(error.get('type'))+': '+str(error.get('message')),remote_type=error.get('type'))
                    return response['result']
                ready=select.select(min(.2,max(0.,deadline-time.monotonic())))
                if ready:
                    block=os.read(fd,65536)
                    if not block:
                        self._abort_transport();raise RemoteError('GCP RPC stream closed; inspect transport log')
                    self.pending.extend(block)
                    if len(self.pending)>MAX_FRAME:
                        self._abort_transport();raise RemoteError('GCP RPC response exceeds frame limit')
            self._abort_transport()
            raise TimeoutError('GCP RPC operation exceeded its deadline; abort until cleanup is established')
        finally:select.close()

    def start(self):
        if self.process is not None:raise RemoteError('Session cannot start twice')
        actor=self.local_trial/'actor'
        files={}
        for p in actor.rglob('*'):
            if p.is_symlink():raise ValueError('Initial actor files must not contain symlinks')
            if p.is_file():
                relative=str(p.relative_to(actor))
                if relative not in INITIAL_FILES:raise ValueError('Unexpected initial actor file')
                files[relative]=_encoded(p.read_bytes())
        raw=self.profile_path.read_bytes()
        profile=load_profile(self.profile_path)
        self.stderr=(self.local_trial/'gcp-transport.stderr.log').open('ab')
        self.process=subprocess.Popen(self.ssh_command(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=self.stderr,start_new_session=True)
        os.set_blocking(self.process.stdout.fileno(),False)
        self.heartbeat_thread=threading.Thread(target=self._heartbeat,daemon=True)
        self.heartbeat_thread.start()
        try:
            return self._rpc('init',{'trial_id':self.local_trial.name,'seconds':self.seconds,'gpu':self.gpu,
                                'files':files,'profile_b64':_encoded(raw),
                                'profile_file_sha256':hashlib.sha256(raw).hexdigest(),
                                'protocol_sha256':canonical_sha256(profile)},timeout=120)
        except Exception:
            try:self.stop()
            except Exception:pass
            raise

    def execute(self,command,timeout=120):return self._rpc('shell',{'command':command,'timeout':timeout},timeout=timeout+30)
    def run_diagnostic(self,case_ids=None,remaining_seconds=300):
        return self._rpc('gpu_test',{'case_ids':case_ids,'remaining_seconds':remaining_seconds},timeout=min(300,remaining_seconds)+30)
    def finish(self):
        result=self._rpc('finish',timeout=45)
        body=_decoded(result['candidate_b64'])
        if hashlib.sha256(body).hexdigest()!=result['candidate_sha256']:raise RemoteError('Recovered source hash differs')
        result['candidate_bytes']=body
        return result
    def evaluate(self):return self._rpc('evaluate',timeout=1260)
    def stop(self):
        result={}
        unverified=self.transport_failed
        try:
            if not unverified and self.process is not None and self.process.poll() is None:
                result=self._rpc('stop',timeout=45)
        finally:
            self.heartbeat_done.set()
            if self.heartbeat_thread is not None:self.heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL+1)
            if self.process is not None:
                if self.process.stdin:self.process.stdin.close()
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid,signal.SIGTERM)
                    self.process.wait(timeout=10)
                if self.process.stdout:self.process.stdout.close()
            if self.stderr:self.stderr.close()
        if unverified:raise RemoteError('SSH transport failed; remote cleanup is unverified and the cohort must not continue')
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['serve'])
    parser.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args(argv)
    if os.getuid()!=0:parser.error('Remote trusted controller must have Docker/actor ownership access (run through sudo)')
    _serve(args.run_root)
    return 0


if __name__=='__main__':raise SystemExit(main())
