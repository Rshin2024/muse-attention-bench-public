"""Credential-free, CPU-only shell actors for the shared model broker.

Docker isolation is a practical execution boundary, not a hostile-kernel
security certification. The trusted controller and Docker socket stay outside.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import selectors
import signal
import subprocess
import time
import uuid

from .protocol import BASE_IMAGE


class ActorError(RuntimeError):
    pass


def actor_command(actor_dir: Path, name: str, *, uid: int, gid: int,
                  cpu_limit: int = 8, memory: str = '32g') -> list[str]:
    """Only the official base and actor directory are exposed, never evaluator."""
    return ['docker', 'run', '--detach', '--name', name,
            '--label', 'muse-attention-bench.role=actor', '--runtime', 'runc',
            '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '256',
            '--cpus', str(cpu_limit), '--memory', memory, '--memory-swap', memory,
            '--user', f'{uid}:{gid}', '--workdir', '/workspace',
            '--tmpfs', '/tmp:rw,exec,nosuid,size=8g,mode=1777',
            '--mount', f'type=bind,src={actor_dir},dst=/workspace',
            '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'CUDA_VISIBLE_DEVICES=',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONHASHSEED=0',
            '--env', 'XDG_CACHE_HOME=/tmp/cache', '--env', 'TRITON_CACHE_DIR=/tmp/triton',
            '--env', 'TORCH_EXTENSIONS_DIR=/tmp/torch-extensions',
            '--env', 'CUDA_CACHE_PATH=/tmp/cuda-cache', '--env', 'MAX_JOBS=8',
            '--env', 'OMP_NUM_THREADS=8', '--env', 'MKL_NUM_THREADS=8',
            '--entrypoint', '/bin/bash', BASE_IMAGE, '--noprofile', '--norc', '-c',
            'exec sleep infinity']


class DockerActor:
    """Persistent CPU actor; timeout removal kills every background process."""
    def __init__(self, actor_dir: Path, *, image: str = BASE_IMAGE, seconds: float = 1800,
                 deadline: float | None = None, cpu_limit: int = 8, memory: str = '32g',
                 max_output_bytes: int = 64000):
        selected = Path(actor_dir)
        if selected.is_symlink() or not selected.is_dir():
            raise ValueError('Actor directory must be an existing non-symlink directory')
        if image != BASE_IMAGE:
            raise ValueError('Actors use the pinned base image, never the evaluator image')
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds <= 0 or cpu_limit <= 0 or max_output_bytes <= 0:
            raise ValueError('Actor limits must be positive')
        if deadline is not None and (not isinstance(deadline,(int,float)) or not math.isfinite(deadline)):
            raise ValueError('Actor deadline must be finite')
        self.actor_dir = selected.resolve()
        self.seconds, self.deadline = float(seconds), deadline
        self.cpu_limit, self.memory, self.max_output_bytes = cpu_limit, memory, max_output_bytes
        self.uid = os.getuid() or 65534
        self.gid = os.getgid() if os.getuid() else 65534
        self.name = 'muse-actor-' + uuid.uuid4().hex[:16]
        self.started = False
        self.closed = False
        self.paused = False

    def _remaining(self) -> float:
        return max(0., self.deadline - time.monotonic()) if self.deadline is not None else self.seconds

    def _writable(self):
        if os.getuid() != 0:
            return
        for path in [self.actor_dir, *self.actor_dir.rglob('*')]:
            if path.is_symlink():
                # A restarted actor may contain model-created build symlinks.
                # Never follow them into a host path while setting ownership.
                os.lchown(path,self.uid,self.gid)
                continue
            os.chown(path, self.uid, self.gid)
            path.chmod(0o750 if path.is_dir() else 0o640)

    def start(self) -> dict:
        if self.started or self.closed:
            raise ActorError('Actor cannot be started twice')
        self._writable()
        image = subprocess.run(['docker', 'image', 'inspect', BASE_IMAGE],
                               capture_output=True, text=True, timeout=30)
        if image.returncode:
            raise ActorError('Pinned actor image must be present before the research clock starts')
        command = actor_command(self.actor_dir, self.name, uid=self.uid, gid=self.gid,
                                cpu_limit=self.cpu_limit, memory=self.memory)
        # A Docker client timeout can occur after the daemon created the named
        # container. Mark it owned before launch so failure cleanup still runs.
        self.started = True
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise ActorError('Actor container failed to start: ' + result.stderr[-2000:])
        except BaseException:
            self.stop()
            raise
        if self.deadline is None:
            self.deadline = time.monotonic() + self.seconds
        return {'container': self.name, 'actor_image': BASE_IMAGE, 'network': 'none',
                'gpu_access': False, 'evaluator_mounted': False, 'credentials_mounted': False,
                'uid': self.uid, 'gid': self.gid, 'cpu_limit': self.cpu_limit, 'memory': self.memory}

    def pause(self):
        if self.started and not self.closed and not self.paused:
            subprocess.run(['docker', 'pause', self.name], check=True, capture_output=True, timeout=15)
            self.paused = True

    def unpause(self):
        if self.started and not self.closed and self.paused:
            if self._remaining() <= 0:
                self.stop()
                return
            subprocess.run(['docker', 'unpause', self.name], check=True, capture_output=True, timeout=15)
            self.paused = False

    def execute(self, command: str, timeout: float = 120) -> dict:
        if not self.started or self.closed or self.paused:
            raise ActorError('Actor must be running and unpaused')
        if not isinstance(command, str) or not command.strip() or len(command.encode()) > 262144:
            raise ValueError('Shell command must be nonempty and at most 256 KiB')
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('Action timeout must be finite and positive')
        allowance = min(float(timeout), self._remaining())
        if allowance <= 0:
            self.stop()
            return {'stdout': '', 'stderr': 'Research wall-clock deadline reached',
                    'exit_code': 124, 'duration_seconds': 0., 'timed_out': True, 'truncated': False}
        args = ['docker', 'exec', '--user', f'{self.uid}:{self.gid}', '--workdir', '/workspace',
                self.name, '/bin/bash', '--noprofile', '--norc', '-c', command]
        started = time.monotonic()
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        selector = selectors.DefaultSelector()
        buffers = {'stdout': bytearray(), 'stderr': bytearray()}
        totals = {'stdout': 0, 'stderr': 0}
        assert process.stdout is not None and process.stderr is not None
        for label, pipe in [('stdout', process.stdout), ('stderr', process.stderr)]:
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, label)
        timed_out = False
        try:
            while selector.get_map():
                remaining = allowance - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    self.stop()
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                for key, _ in selector.select(min(.2, remaining)):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    label = key.data
                    totals[label] += len(block)
                    room = self.max_output_bytes - len(buffers[label])
                    buffers[label].extend(block[:max(0, room)])
            try:
                code = process.wait(timeout=max(.01,allowance-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                timed_out=True
                self.stop()
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
                code=process.wait(timeout=5)
        finally:
            selector.close()
            for pipe in (process.stdout, process.stderr):
                pipe.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        return {'stdout': buffers['stdout'].decode(errors='replace'),
                'stderr': buffers['stderr'].decode(errors='replace'),
                'exit_code': 124 if timed_out else code,
                'duration_seconds': time.monotonic() - started, 'timed_out': timed_out,
                'truncated': any(totals[k] > len(buffers[k]) for k in totals),
                'output_bytes': totals}

    def stop(self) -> dict:
        if self.closed:
            return {'container': self.name, 'already_closed': True}
        if self.started:
            result = subprocess.run(['docker', 'rm', '--force', self.name],
                                    capture_output=True, text=True, timeout=20)
            if result.returncode and 'No such container' not in result.stderr:
                raise ActorError('Actor cleanup could not be established')
        self.closed, self.paused = True, False
        return {'container': self.name, 'container_removed': True, 'all_actor_processes_stopped': True}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
