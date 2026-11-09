"""Local GPU evaluator for the published Muse attention contract.

Only launch inside the repository's restricted evaluator container. The candidate
is ordinary executable Python: in-process correctness checks are not an adversarial
security boundary and do not reproduce the organizer's anti-cheat enforcement.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from .scoring import bootstrap_score, summarize_shape
from .timing import ColdL2, calibrate_graph_cost, capture, event_replay_us, replay_orders
from .runtime_proof import runtime_proof

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "muse_attention_decode"


class EvaluationError(RuntimeError):
    pass


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _observed_sglang_commit() -> tuple[str | None, str]:
    try:
        distribution = importlib.metadata.distribution("sglang")
        direct = distribution.read_text("direct_url.json")
        if direct:
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit, "installed distribution direct_url.json"
        for relative in ("sglang/__init__.py",):
            package = Path(distribution.locate_file(relative)).resolve()
            for ancestor in list(package.parents)[:5]:
                if (ancestor / ".git").exists():
                    result = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "HEAD"],
                                            capture_output=True, text=True, check=True, timeout=5)
                    return result.stdout.strip(), "installed source Git HEAD"
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError, subprocess.SubprocessError):
        pass
    return None, "unavailable; expected commit is not observed evidence"


def _source_fingerprint() -> tuple[str | None, list[dict]]:
    paths = ["srt/models/muse_spark.py", "srt/layers/attention/flashattention_backend.py",
             "kernels/ops/muse_spark_v12/fused_qk_norm.py",
             "kernels/ops/muse_spark_v12/norm_gate.py", "kernels/ops/kvcache/kvcache.py"]
    spec = importlib.util.find_spec("sglang")
    directories = list(spec.submodule_search_locations or []) if spec else []
    if not directories:
        return None, []
    package = Path(directories[0])
    entries = []
    combined = hashlib.sha256()
    for relative in paths:
        path = package / relative
        if not path.is_file():
            return None, entries
        contents = path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        entries.append({"path": relative, "sha256": digest})
        combined.update(relative.encode() + b"\0" + contents + b"\0")
    return combined.hexdigest(), entries


def evaluator_fingerprint() -> str:
    combined = hashlib.sha256()
    paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    paths += sorted(VENDOR.glob("*.py"))
    for path in paths:
        combined.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return combined.hexdigest()


def runtime_identity(torch, protocol: dict) -> tuple[dict, dict]:
    proof = runtime_proof(protocol)
    if not torch.cuda.is_available():
        raise EvaluationError("CUDA is unavailable; this evaluator requires an H200 GPU")
    if torch.cuda.device_count() != 1:
        raise EvaluationError("expose exactly one GPU to the evaluator container")
    properties = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if "H200" not in name.upper() or capability != [9, 0]:
        raise EvaluationError(f"requires NVIDIA H200 compute capability 9.0, got {name} {capability}")
    if "NVL" in name.upper() or "PCIE" in name.upper():
        raise EvaluationError("H200 NVL/PCIe does not reproduce the required SXM hardware")
    sxm_attested = "SXM" in name.upper() or os.environ.get("MUSE_ATTENTION_BENCH_SXM_ATTESTED") == "1"
    if protocol.get("runtime", {}).get("require_sxm", True) and not sxm_attested:
        raise EvaluationError("GPU name does not establish SXM; require operator SXM attestation")
    uuid = getattr(properties, "uuid", None)
    uuid = str(uuid) if uuid else None
    driver = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=10)
        rows = [line.strip().split(",") for line in result.stdout.splitlines() if line.strip()]
        if len(rows) == 1:
            uuid, driver = (v.strip() for v in rows[0])
    except (OSError, subprocess.SubprocessError):
        pass
    observed_commit, commit_evidence = _observed_sglang_commit()
    source_fingerprint, source_files = _source_fingerprint()
    expected_commit = protocol.get("runtime", {}).get("sglang_commit")
    if observed_commit and expected_commit and observed_commit != expected_commit:
        raise EvaluationError(f"observed SGLang commit {observed_commit} differs from expected {expected_commit}")
    identity = {
        "protocol_sha256": canonical_hash(protocol),
        "image_digest": proof["image_digest"],
        "execution_backend": proof["execution_backend"],
        "gpu_name": name, "gpu_uuid": uuid,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "driver_version": driver, "sglang_commit": observed_commit,
        "sglang_source_sha256": source_fingerprint, "evaluator_sha256": evaluator_fingerprint(),
    }
    details = {"compute_capability": capability, "total_memory_bytes": properties.total_memory,
               "sxm_verified_or_attested": sxm_attested,
               "sxm_evidence": "device name" if "SXM" in name.upper() else "operator environment attestation",
               "declared_sglang_commit": expected_commit,
               "observed_sglang_commit": observed_commit, "sglang_commit_evidence": commit_evidence,
               "sglang_source_files": source_files,
               "sglang_version": importlib.metadata.version("sglang"),
               "declared_base_image_digest": proof["declared_base_image_digest"],
               "image_digest_evidence": proof["image_digest_evidence"],
               "runtime_proof_sha256": proof["runtime_proof_sha256"],
               "runtime_proof_limitations": proof["runtime_proof_limitations"]}
    if "provider_receipt" in proof:
        details["provider_receipt"] = proof["provider_receipt"]
    return identity, details


def collect_environment(protocol: dict) -> dict:
    """GPU preflight without importing or running a candidate implementation."""
    try:
        import torch
        identity, runtime = runtime_identity(torch, protocol)
        return {"status": "ready", "comparison_identity": identity, "runtime": runtime,
                "gpu_kernel_validated": False}
    except Exception as exc:
        return {"status": "unavailable", "error": {"type": type(exc).__name__, "message": str(exc)},
                "gpu_kernel_validated": False}


def _schema(torch, data: dict) -> dict:
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = (tuple(value.shape), str(value.dtype), tuple(value.stride()), str(value.device))
        else:
            result[key] = copy.deepcopy(value)
    return result


def copy_inputs(torch, destination: dict, source: dict):
    """Fresh values into captured addresses, rejecting shape-dependent metadata drift."""
    if _schema(torch, destination) != _schema(torch, source):
        raise EvaluationError("fresh input schema differs from captured graph; cannot silently recapture")
    for key, value in source.items():
        if isinstance(value, torch.Tensor):
            destination[key].copy_(value)


def clone_inputs(torch, source: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in source.items()}


def input_pointers(torch, data: dict) -> dict:
    return {key: value.data_ptr() for key, value in data.items() if isinstance(value, torch.Tensor)}


def validate_mutations(torch, actual: dict, pristine: dict, reference, expected_pointers: dict):
    """Reject changed metadata/input values and writes outside the new KV slots.

    Cache comparison occurs on GPU as raw bytes. The immutable oracle copy is
    independent; checks do not rely on candidate-modified lengths/maps/cache.
    """
    if _schema(torch, actual) != _schema(torch, pristine):
        raise EvaluationError("candidate changed input schema or configuration")
    if input_pointers(torch, actual) != expected_pointers:
        raise EvaluationError("candidate replaced an input tensor")
    for key, value in pristine.items():
        if isinstance(value, torch.Tensor) and key not in {"k_cache", "v_cache"}:
            if not torch.equal(actual[key].contiguous().reshape(-1).view(torch.uint8),
                               value.contiguous().reshape(-1).view(torch.uint8)):
                raise EvaluationError(f"candidate modified immutable input {key}")
    rows = reference.new_slots(pristine).reshape(-1)
    hkv, dim = pristine["config"]["num_kv_heads"], pristine["config"]["head_dim"]
    for key in ("k_cache", "v_cache"):
        got = actual[key].view(torch.uint8).reshape(-1, hkv, dim)
        want = pristine[key].view(torch.uint8).reshape(-1, hkv, dim)
        saved_new = got.index_select(0, rows)
        got.index_copy_(0, rows, want.index_select(0, rows))
        unchanged = torch.equal(got, want)
        got.index_copy_(0, rows, saved_new)
        if not unchanged:
            raise EvaluationError(f"candidate modified old/unallocated {key} rows")


def _check(torch, reference, actual, pristine, output, pointers) -> str:
    validate_mutations(torch, actual, pristine, reference, pointers)
    raw = reference.collect(actual, output).clone()
    ok, message = reference.check_implementation(pristine, raw)
    if not ok:
        raise EvaluationError(message)
    return message


def _seed(base: int, case_id: str, phase: str, repetition: int) -> int:
    data = f"{base}:{case_id}:{phase}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little") & 0x7fffffff


def _write_result(output: Path, report: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def evaluate(candidate: Path, output: Path, protocol: dict, *, correctness_only: bool = False) -> dict:
    """Evaluate one sealed candidate, returning/writing a structured local result.

    The caller must impose the complete 900-second subprocess/container timeout.
    Missing hardware or candidate failure is reported, never scored as zero.
    """
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    started = time.monotonic()
    report = {
        "schema_version": 1, "benchmark": "muse_attention_decode",
        "evaluation_kind": "documented-protocol-reconstruction",
        "official_score": False, "gpu_validated": False, "status": "running",
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "protocol_sha256": canonical_hash(protocol), "speedup": None,
        "correctness_cases": [], "scored_cases": [],
        "source_review": {"approved": False, "required_separately": True},
        "limitations": ["Hosted seeds, anti-cheat, repetition policy and exact overhead/CI calibration are unavailable.",
                        "Candidate and trusted code share a Python process; this is not a hostile-code security boundary."],
    }
    try:
        if os.environ.get("MUSE_ATTENTION_BENCH_CONTAINER") != "1":
            raise EvaluationError("candidate evaluation is container-only; use the container launcher")
        import torch
        identity, runtime = runtime_identity(torch, protocol)
        report["comparison_identity"], report["runtime"] = identity, runtime
        torch.set_grad_enabled(False)
        sys.path.insert(0, str(VENDOR))
        reference = _load_module(VENDOR / "reference.py", "muse_trusted_reference")
        baseline = _load_module(VENDOR / "baseline.py", "muse_trusted_baseline")
        # Trusted module paths are not a sandbox. Container isolation and source
        # review are explicit, separate requirements rather than an import trick.
        submission = _load_module(candidate, "muse_candidate")
        function = getattr(submission, "kernel_fn", None)
        if not callable(function):
            raise EvaluationError("candidate must define callable kernel_fn(data)")
        seed_base = protocol.get("seed")
        if seed_base is None:
            seed_base = secrets.randbits(31)
        report["evaluation_seed"] = seed_base
        timing = protocol.get("timing", {})
        orders = replay_orders(timing.get("repeats", 20))
        cold = ColdL2(torch, timing.get("flush_bytes", 128 * 1024 * 1024))

        for case in protocol["test_cases"]:
            pristine = reference.generate_input(**case["kwargs"], seed=_seed(seed_base, case["id"], "test", 0))
            data = clone_inputs(torch, pristine)
            pointers = input_pointers(torch, data)
            result = function(data)
            torch.cuda.synchronize()
            message = _check(torch, reference, data, pristine, result, pointers)
            # Mandatory tests must also be graph-capturable and respond to new values.
            graph, graph_output = capture(torch, function, data, timing.get("warmup", 2))
            fresh = reference.generate_input(**case["kwargs"], seed=_seed(seed_base, case["id"], "test", 1))
            copy_inputs(torch, data, fresh)
            graph.replay()
            torch.cuda.synchronize()
            graph_message = _check(torch, reference, data, fresh, graph_output, pointers)
            report["correctness_cases"].append({"id": case["id"], "passed": True,
                                                 "eager": message, "changed_input_graph": graph_message})
            del graph, graph_output, result, data, pristine, fresh
            torch.cuda.empty_cache()

        calibration = calibrate_graph_cost(torch, cold, timing.get("calibration_repeats", 50))
        report["graph_cost_calibration"] = calibration
        for case in protocol["scored_cases"]:
            initial = reference.generate_input(**case["kwargs"], seed=_seed(seed_base, case["id"], "capture", 0))
            candidate_data, baseline_data = clone_inputs(torch, initial), clone_inputs(torch, initial)
            cp, bp = input_pointers(torch, candidate_data), input_pointers(torch, baseline_data)
            cg, co = capture(torch, function, candidate_data, timing.get("warmup", 2))
            bg, bo = capture(torch, baseline.kernel_fn, baseline_data, timing.get("warmup", 2))
            observations = []
            repetitions = 2 if correctness_only else len(orders)
            for repetition in range(repetitions):
                pristine = reference.generate_input(**case["kwargs"], seed=_seed(seed_base, case["id"], "replay", repetition))
                copy_inputs(torch, candidate_data, pristine)
                copy_inputs(torch, baseline_data, pristine)
                order = orders[repetition]
                times = {}
                for actor in order:
                    graph = bg if actor == "A" else cg
                    times[actor] = event_replay_us(torch, graph, cold)
                # Both outputs are checked after every fresh timed input pair.
                candidate_message = _check(torch, reference, candidate_data, pristine, co, cp)
                baseline_message = _check(torch, reference, baseline_data, pristine, bo, bp)
                observations.append({"order": order, "baseline_us": times["A"], "candidate_us": times["B"],
                                     "candidate_check": candidate_message, "baseline_check": baseline_message})
                del pristine
            if correctness_only:
                summary = {"id": case["id"], "passed": True, "changed_input_replays": repetitions}
            else:
                summary = {"id": case["id"], "passed": True,
                           **summarize_shape(observations, calibration["median_us"])}
            report["scored_cases"].append(summary)
            del cg, bg, co, bo, candidate_data, baseline_data, initial
            torch.cuda.empty_cache()

        report["gpu_validated"] = True
        if correctness_only:
            report["status"] = "correctness_passed"
        else:
            report["score"] = bootstrap_score(report["scored_cases"], protocol["scored_cases"],
                samples=timing.get("bootstrap_samples", 2000), seed=seed_base,
                interval_floor=timing.get("minimum_relative_interval", timing.get("interval_floor", .015)))
            report["speedup"] = report["score"]["speedup"]
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _write_result(output, report)
    return report
