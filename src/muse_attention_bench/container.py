"""Build/replay commands for a dedicated Linux NVIDIA host, without credentials."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from .protocol import ROOT, BASE_IMAGE, load_profile

LOCAL_IMAGE = "muse-attention-bench:0.1.0"


def host_doctor() -> dict:
    linux = platform.system() == "Linux"
    return {"host_system": platform.system(), "host_architecture": platform.machine(),
            "docker": shutil.which("docker"), "nvidia_smi": shutil.which("nvidia-smi"),
            "gpu_evaluation_ready": bool(linux and shutil.which("docker") and shutil.which("nvidia-smi")),
            "note": "Host prerequisites only; run container-doctor on an exclusive H200 SXM node before evaluation"}


def build_command(image: str = LOCAL_IMAGE) -> list[str]:
    return ["docker", "build", "--platform", "linux/amd64", "--build-arg", f"BASE_IMAGE={BASE_IMAGE}",
            "--tag", image, str(ROOT)]


def inspect_image(image: str = LOCAL_IMAGE) -> dict:
    """Verify the locally built image metadata before a GPU launch."""
    r=subprocess.run(['docker','image','inspect',image],check=True,capture_output=True,text=True,timeout=30)
    info=json.loads(r.stdout)[0]
    base=info.get('Config',{}).get('Labels',{}).get('org.opencontainers.image.base.name')
    if base!=BASE_IMAGE or info.get('Architecture')!='amd64' or info.get('Os')!='linux':
        raise ValueError('Evaluator image base/platform differs; rebuild with build-image')
    return {'image_id':info['Id'],'base_image':base,'platform':'linux/amd64'}


def run_command(candidate: Path, output: Path, profile: Path, *, gpu: str = "0",
                image: str = LOCAL_IMAGE, correctness_only: bool = False,
                attest_sxm: bool = False) -> list[str]:
    candidate, output, profile = candidate.resolve(), output.resolve(), profile.resolve()
    if not candidate.is_file() or not profile.is_file():
        raise ValueError("Candidate and profile must be existing files")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise ValueError("Create the output directory before starting Docker")
    if not attest_sxm:
        raise ValueError("Verify exclusive H200 SXM access and pass --attest-h200-sxm; form factor cannot be proved from GPU name alone")
    config = load_profile(profile)
    image_base = config["runtime"]["base_image"].split("@", 1)[1]
    args = ["docker", "run", "--rm", "--gpus", f"device={gpu}", "--network", "none",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "512", "--shm-size", "2g", "--ulimit", "memlock=-1",
            "--tmpfs", "/tmp:rw,exec,nosuid,size=8g", "--tmpfs", "/root/.cache:rw,exec,nosuid,size=8g",
            "--mount", f"type=bind,src={candidate},dst=/candidate/submission.py,readonly",
            "--mount", f"type=bind,src={profile},dst=/protocol.json,readonly",
            "--mount", f"type=bind,src={output.parent},dst=/results",
            "--env", "MUSE_ATTENTION_BENCH_CONTAINER=1",
            "--env", f"MUSE_ATTENTION_BENCH_BASE_IMAGE_DIGEST={image_base}",
            "--env", "MUSE_ATTENTION_BENCH_SXM_ATTESTED=1",
            "--env", "CUDA_MODULE_LOADING=LAZY", "--env", "PYTHONHASHSEED=0",
            "--env", "TORCH_EXTENSIONS_DIR=/tmp/torch-extensions",
            "--env", "TRITON_CACHE_DIR=/tmp/triton", "--env", "CUDA_CACHE_PATH=/tmp/cuda-cache",
            image, "internal-evaluate", "/candidate/submission.py", "--profile", "/protocol.json",
            "--output", f"/results/{output.name}"]
    if correctness_only:
        args.append("--correctness-only")
    return args


def doctor_command(*, gpu: str='0', attest_sxm: bool=False) -> list[str]:
    if not attest_sxm:
        raise ValueError('Verify H200 SXM hardware and pass --attest-h200-sxm')
    return ['docker','run','--rm','--gpus',f'device={gpu}','--network','none',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--tmpfs','/tmp:rw,exec,nosuid,size=2g',
            '--tmpfs','/root/.cache:rw,exec,nosuid,size=2g',
            '--env','MUSE_ATTENTION_BENCH_CONTAINER=1',
            '--env','MUSE_ATTENTION_BENCH_SXM_ATTESTED=1',
            '--env',f'MUSE_ATTENTION_BENCH_BASE_IMAGE_DIGEST={BASE_IMAGE.split("@",1)[1]}',
            LOCAL_IMAGE,'internal-doctor']


def run_container(args: list[str], timeout: int = 900) -> int:
    """Bound the Docker job; no model API or cloud provisioning is performed."""
    # A named container permits explicit cleanup if the host timeout fires.
    import uuid
    name = "muse-attention-" + uuid.uuid4().hex[:12]
    inspected=inspect_image()
    command = list(args)
    image_index=command.index(LOCAL_IMAGE)
    command[image_index]=inspected['image_id']
    command[image_index:image_index]=['--env',f'MUSE_ATTENTION_BENCH_IMAGE_DIGEST={inspected["image_id"]}']
    command[2:2] = ["--name", name]
    try:
        return subprocess.run(command, timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"GPU evaluation exceeded {timeout}s including compilation; container removed")
    finally:
        # Killing an SSH/RPC client does not itself stop a Docker container.
        # Remove only this launcher's UUID-owned name on success or any unwind,
        # including SystemExit/SIGHUP from the trusted remote server.
        cleanup=subprocess.run(['docker','rm','--force',name],check=False,capture_output=True,text=True,timeout=30)
        if cleanup.returncode and 'No such container' not in (cleanup.stderr or ''):
            raise RuntimeError('Owned GPU container cleanup could not be established; abort the cohort')
