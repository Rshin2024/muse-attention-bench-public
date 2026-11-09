"""Public task transcription and immutable experiment identities; no Torch import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "configs/documented-competition-v1.json"
VENDOR = ROOT / "third_party/muse_attention_decode"
SGLANG_COMMIT = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
BASE_IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_shape(description: str) -> dict:
    """Translate literal TASK.md shape descriptors without importing a reference."""
    result = {}
    for field in description.strip().split(","):
        key, value = field.strip().split(maxsplit=1)
        if key in {"batch", "ctx", "q_len", "page"}:
            result[key] = int(value)
        elif key == "kv_scale":
            result[key] = float(value)
        elif key in {"heads", "layer", "spread", "dist"}:
            result[key] = value
        else:
            raise ValueError(f"Unknown official shape field: {key}")
    required = {"batch", "ctx", "heads", "layer"}
    if not required <= result.keys():
        raise ValueError(f"Incomplete shape descriptor: {description}")
    return result


def cases_from_task(task: str) -> tuple[list[dict], list[dict]]:
    scored = []
    for line in task.splitlines():
        if not line.startswith("| decode ") and not line.startswith("| verify "):
            continue
        cells = [s.strip() for s in line.strip("|").split("|")]
        load, description, weight, _ = cells
        scored.append({"id": f"scored-{len(scored)+1:02d}", "load": load,
                       "weight": int(weight), "kwargs": parse_shape(description)})
    block = task.split("Before timing, a run must pass these 16 tests", 1)[1].split("```", 2)[1]
    tests = [{"id": f"test-{i+1:02d}", "load": None, "weight": 0, "kwargs": parse_shape(s)}
             for i, s in enumerate(x for x in block.splitlines() if x.strip())]
    if len(tests) != 16 or len(scored) != 12:
        raise ValueError("Unexpected public task; refusing a silent suite change")
    return tests, scored


def default_profile() -> dict:
    tests, scored = cases_from_task((VENDOR / "TASK.md").read_text())
    return {
        "schema_version": 1,
        "name": "documented-competition-v1",
        "problem": "muse_attention_decode",
        "official_equivalent": False,
        "scope": "Reconstruction of the documented public competition protocol; not the hosted judge",
        "seed": 20261107,
        "runtime": {"base_image": BASE_IMAGE, "platform": "linux/amd64",
                    "sglang_commit": SGLANG_COMMIT,
                    "source_commit_status": "Expected from published baseline header; independently record installed runtime"},
        "hardware": {"gpu": "NVIDIA H200 SXM", "compute_capability": [9, 0],
                     "visible_devices": 1, "sxm_operator_attestation_required": True},
        "timing": {"repeats": 20, "warmup": 3, "flush_bytes": 134217728,
                   "bootstrap_samples": 2000, "calibration_repeats": 100,
                   "confidence_level": 0.95, "minimum_relative_interval": 0.015,
                   "cold_l2_each_replay": True,
                   "fixed_replay_cost": "calibrated locally; exact hosted calibration is not public",
                   "run_timeout_seconds": 900},
        "scoring": {"shape_speedup": "median of adjacent swapped-order paired ratios",
                    "load_time": "sum(weight * shape_time)",
                    "overall_speedup": "geometric mean of six baseline_load/candidate_load ratios"},
        "test_cases": tests, "scored_cases": scored,
        "known_gaps": ["Hosted anti-cheat and process isolation are not reproduced",
                       "Organizer private seeds, repeat counts and calibration details are unavailable",
                       "Image digest is the published tag resolved locally, not an organizer-signed historical attestation",
                       "SXM form factor, node exclusivity, clocks and thermals need operator verification",
                       "This repository has not yet been validated on H200 hardware"]}


def load_profile(path: Path | str = DEFAULT_PROFILE) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("problem") != "muse_attention_decode":
        raise ValueError("Expected muse_attention_decode profile")
    if len(data.get("test_cases", [])) != 16 or len(data.get("scored_cases", [])) != 12:
        raise ValueError("A competition profile must retain 16 tests and 12 scored cases")
    tests,scored=cases_from_task((VENDOR/'TASK.md').read_text())
    if data['test_cases']!=tests or data['scored_cases']!=scored:
        raise ValueError('Case definitions/weights differ from the pinned public TASK.md')
    if data.get('runtime',{}).get('base_image')!=BASE_IMAGE:
        raise ValueError('Profile base image differs from the locked competition runtime')
    if data.get("timing", {}).get("repeats", 0) < 4 or data["timing"]["repeats"] % 2:
        raise ValueError("Timing repeats must be even and at least four")
    return data


def verify_vendor(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "third_party/provenance.json").read_text())
    mismatches = []
    for entry in manifest["files"]:
        file = root / entry["path"]
        got = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
        if got != entry["sha256"]:
            mismatches.append(entry["path"])
    return {"ok": not mismatches, "mismatches": mismatches,
            "upstream_commit": manifest["upstream_commit"], "files_checked": len(manifest["files"])}
