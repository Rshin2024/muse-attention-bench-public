"""Runner-neutral preparation, sealing, and comparison of model trials.

This module never calls a model provider, runs candidate code, provisions a GPU,
or claims to sandbox an actor. An external runner must enforce the declared
budget, tool access, and isolation policy. Result admission binds exact artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_WALL_CLOCK_SECONDS = 1800
DEFAULT_DIAGNOSTIC_BUDGET = 3
COMPARISON_IDENTITY_FIELDS = (
    "protocol_sha256", "execution_backend", "image_digest", "gpu_name", "gpu_uuid",
    "torch_version", "cuda_version", "driver_version", "evaluator_sha256",
)
_TRIAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    # O_EXCL prevents accidental replacement, including an existing symlink.
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_new(path: Path, value: Any) -> None:
    _write_new(path, json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False).encode("utf-8") + b"\n")


def _replace_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _check_positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _safe_candidate(trial: Path, candidate: Path | None) -> Path:
    actor = (trial / "actor").resolve()
    selected = candidate or Path("kernel/submission.py")
    selected = Path(selected)
    if not selected.is_absolute():
        selected = actor / selected
    resolved = selected.resolve(strict=True)
    if not resolved.is_relative_to(actor) or selected.is_symlink():
        raise ValueError("The candidate must be a regular .py file inside this trial's actor directory")
    if not resolved.is_file() or resolved.suffix != ".py":
        raise ValueError("The candidate must be a regular .py file")
    return resolved


def prepare_trial(
    root: Path,
    model: str,
    trial_id: str,
    protocol_path: Path,
    *,
    provider: str = "unspecified",
    effort: str = "unspecified",
    tool_policy: dict | None = None,
    context_budget: dict | int | None = None,
    wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS,
    diagnostic_budget: int = DEFAULT_DIAGNOSTIC_BUDGET,
    expose_baseline: bool = True,
    cohort: str = "default",
) -> Path:
    """Create a fresh trial with identical actor inputs for the same conditions.

    ``root`` is the benchmark repository. Trials live in ``root/trials/trial_id``.
    This records declarations; it does not start/enforce an agent's clock.
    """
    root = Path(root).resolve(strict=True)
    if not _TRIAL_ID.fullmatch(trial_id):
        raise ValueError("trial_id must contain 1-128 letters, digits, underscores or hyphens")
    if not isinstance(model, str) or not model.strip() or "\n" in model:
        raise ValueError("An exact nonempty model identifier is required")
    _check_positive(wall_clock_seconds, "wall_clock_seconds")
    if isinstance(diagnostic_budget, bool) or not isinstance(diagnostic_budget, int) or diagnostic_budget < 0:
        raise ValueError("diagnostic_budget must be a nonnegative integer")
    if not isinstance(expose_baseline, bool):
        raise ValueError("expose_baseline must be boolean")
    if tool_policy is not None and not isinstance(tool_policy, dict):
        raise ValueError("tool_policy must be a JSON object")
    if context_budget is not None and (isinstance(context_budget, bool) or not isinstance(context_budget, (int, dict))):
        raise ValueError("context_budget must be a token limit, JSON object, or null")
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol_raw = protocol_path.read_bytes()
    protocol = _load_json(protocol_path)
    source = root / "third_party/muse_attention_decode"
    names = ["submission.py", "TASK.md", "AGENTS.md"]
    if expose_baseline:
        names.append("baseline.py")
    source_bytes = {name: (source / name).read_bytes() for name in names}
    prompt = (root / "prompts/actor.md").read_bytes()
    policy = tool_policy or {
        "runner": "external-unattested",
        "network": "disabled-required",
        "allowed_tools": ["shell", "gpu_diagnostic_bridge"],
        "reference_solutions": "unavailable-required",
    }
    conditions = {
        "prompt_sha256": sha256(prompt).hexdigest(),
        "initial_files": {f"kernel/{n}": sha256(b).hexdigest() for n, b in source_bytes.items()},
        "protocol_sha256": canonical_sha256(protocol),
        "effort": effort,
        "tool_policy": policy,
        "context_budget": context_budget,
        "wall_clock_seconds": wall_clock_seconds,
        "diagnostic_budget": diagnostic_budget,
        "expose_baseline": expose_baseline,
    }
    # Validate all metadata before creating any trial directory.
    condition_hash = canonical_sha256(conditions)
    trial_parent = root / "trials"
    trial_parent.mkdir(exist_ok=True)
    trial = trial_parent / trial_id
    trial.mkdir(exist_ok=False)
    actor = trial / "actor"
    (actor / "kernel").mkdir(parents=True)
    (trial / "evaluator").mkdir()
    for name, body in source_bytes.items():
        _write_new(actor / "kernel" / name, body)
    _write_new(actor / "PROMPT.md", prompt)
    _write_new(trial / "evaluator/protocol.json", protocol_raw)
    (trial / "evaluator/protocol.json").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "muse_attention_decode",
        "trial_id": trial_id,
        "cohort": cohort,
        "model": model,
        "provider": provider,
        "created_at": _utc_now(),
        "state": "prepared",
        "conditions": conditions,
        "comparison_condition_sha256": condition_hash,
        "protocol_sha256": conditions["protocol_sha256"],
        "protocol_file_sha256": sha256(protocol_raw).hexdigest(),
        "budget_enforcement": "external-runner-required; not enforced by prepare/seal",
        "filesystem_isolation": "directory separation only; external sandbox required",
    }
    _write_json_new(trial / "manifest.json", manifest)
    return trial


def mark_trial_started(trial_dir: Path) -> dict:
    """Record an external runner start. This does not launch or police it."""
    trial = Path(trial_dir).resolve(strict=True)
    manifest = _load_json(trial / "manifest.json")
    if manifest["state"] != "prepared":
        raise ValueError("Only a prepared trial can be started")
    manifest.update(state="running", started_at=_utc_now(), started_unix_ns=time.time_ns())
    _replace_json(trial / "manifest.json", manifest)
    return manifest


def seal_trial(trial_dir: Path, candidate: Path | None = None) -> dict:
    """Seal the final source without executing it or overwriting a prior seal.

    Paths are relative to ``actor`` by default. The read-only sealed copy and
    recorded hash detect later changes; filesystem ownership is not a sandbox.
    """
    trial = Path(trial_dir).resolve(strict=True)
    manifest = _load_json(trial / "manifest.json")
    if manifest["state"] not in {"prepared", "running"}:
        raise ValueError("This trial has already been sealed or evaluated")
    selected = _safe_candidate(trial, candidate)
    body = selected.read_bytes()
    protocol_path = trial / "evaluator/protocol.json"
    if file_sha256(protocol_path) != manifest["protocol_file_sha256"]:
        raise ValueError("The sealed evaluation protocol was modified")
    if canonical_sha256(_load_json(protocol_path)) != manifest["protocol_sha256"]:
        raise ValueError("Evaluation protocol identity mismatch")
    sealed = trial / "sealed"
    sealed.mkdir(exist_ok=False)
    final_source = sealed / "submission.py"
    _write_new(final_source, body)
    final_source.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    elapsed = None
    if "started_unix_ns" in manifest:
        elapsed = (time.time_ns() - manifest["started_unix_ns"]) / 1_000_000_000
        if elapsed < 0:
            raise ValueError("Host wall clock moved before the recorded trial start")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "trial_id": manifest["trial_id"],
        "candidate_sha256": sha256(body).hexdigest(),
        "protocol_sha256": manifest["protocol_sha256"],
        "comparison_condition_sha256": manifest["comparison_condition_sha256"],
        "sealed_at": _utc_now(),
        "elapsed_wall_seconds": elapsed,
        "wall_clock_budget_seconds": manifest["conditions"]["wall_clock_seconds"],
        "recorded_within_budget": elapsed is not None and elapsed <= manifest["conditions"]["wall_clock_seconds"],
        "budget_enforcement": "unattested-external-runner",
        "state": "sealed_unmeasured",
    }
    _write_json_new(sealed / "seal.json", seal)
    manifest.update(state="sealed", sealed_at=seal["sealed_at"], candidate_sha256=seal["candidate_sha256"])
    _replace_json(trial / "manifest.json", manifest)
    return seal


def record_result(trial_dir: Path, result_path: Path, *, review_path: Path | None = None) -> dict:
    """Import a measured result bound to this exact source and profile.

    An optional independent source-review JSON must bind the same source hash.
    Numeric correctness does not imply source-rule approval. Imports are append-
    once: neither a previous result nor a prior seal can be replaced.
    """
    trial = Path(trial_dir).resolve(strict=True)
    manifest = _load_json(trial / "manifest.json")
    if manifest["state"] != "sealed":
        raise ValueError("A result can only be imported once into a sealed trial")
    seal = _load_json(trial / "sealed/seal.json")
    if file_sha256(trial / "sealed/submission.py") != seal["candidate_sha256"]:
        raise ValueError("Sealed candidate bytes changed")
    result = _load_json(Path(result_path))
    for field in ("candidate_sha256", "protocol_sha256"):
        if result.get(field) != seal[field]:
            raise ValueError(f"Evaluation {field} does not match the sealed trial")
    identity = result.get("comparison_identity")
    if not isinstance(identity, dict) or identity.get("protocol_sha256") != seal["protocol_sha256"]:
        raise ValueError("Evaluation comparison_identity does not bind the sealed protocol")
    review = None
    if review_path is not None:
        review = _load_json(Path(review_path))
        if review.get("candidate_sha256") != seal["candidate_sha256"]:
            raise ValueError("Source review is for a different candidate")
        if not isinstance(review.get("approved"), bool) or not review.get("reviewer") or not review.get("notes"):
            raise ValueError("Source review needs approved, reviewer, notes, and candidate_sha256")
    admission = {
        "schema_version": SCHEMA_VERSION,
        "imported_at": _utc_now(),
        "trial_id": manifest["trial_id"],
        "candidate_sha256": seal["candidate_sha256"],
        "protocol_sha256": seal["protocol_sha256"],
        "result": result,
        "source_review": review,
        "source_rules_approved": bool(review and review["approved"]),
    }
    _write_json_new(trial / "evaluation.json", admission)
    manifest.update(state="evaluated", imported_at=admission["imported_at"])
    _replace_json(trial / "manifest.json", manifest)
    return admission


def _speedup(result: dict) -> float | None:
    value = result.get("speedup")
    if value is None and isinstance(result.get("score"), dict):
        value = result["score"].get("speedup")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Measured speedup must be a finite positive number")
    return float(value)


def summarize_trials(trial_dirs: Iterable[Path]) -> dict:
    """Compare only matching declared conditions and known evaluation runtime.

    Failed, missing, unreviewed, correctness-only, and over-budget observations
    remain explicit rows with null ranked speedup; they are never zeros.
    """
    trial_paths = [Path(p).resolve(strict=True) for p in trial_dirs]
    if not trial_paths:
        raise ValueError("At least one trial is required")
    manifests = [_load_json(p / "manifest.json") for p in trial_paths]
    conditions = {m["comparison_condition_sha256"] for m in manifests}
    if len(conditions) != 1:
        raise ValueError("Refusing to compare trials with different prompts, profiles, effort, tools or budgets")
    identities = []
    rows = []
    for path, manifest in zip(trial_paths, manifests):
        row = {"trial_id": manifest["trial_id"], "model": manifest["model"],
               "provider": manifest["provider"], "state": manifest["state"],
               "status": "not_evaluated", "measured_speedup": None,
               "ranked_speedup": None, "source_rules_approved": False,
               "exclusion_reasons": []}
        evaluation_path = path / "evaluation.json"
        if not evaluation_path.exists():
            row["exclusion_reasons"].append("not_evaluated")
            rows.append(row)
            continue
        seal = _load_json(path / "sealed/seal.json")
        if file_sha256(path / "sealed/submission.py") != seal["candidate_sha256"]:
            raise ValueError(f"Sealed candidate changed: {manifest['trial_id']}")
        admission = _load_json(evaluation_path)
        result = admission["result"]
        if result.get("candidate_sha256") != seal["candidate_sha256"] or result.get("protocol_sha256") != manifest["protocol_sha256"]:
            raise ValueError(f"Result identity changed: {manifest['trial_id']}")
        identity = result.get("comparison_identity", {})
        missing = [key for key in COMPARISON_IDENTITY_FIELDS if not identity.get(key)]
        if missing:
            raise ValueError(f"Missing evaluation identity for {manifest['trial_id']}: {', '.join(missing)}")
        if not identity.get("sglang_commit") and not identity.get("sglang_source_sha256"):
            raise ValueError(f"Missing SGLang revision/source identity for {manifest['trial_id']}")
        normalized_identity = {k: identity[k] for k in COMPARISON_IDENTITY_FIELDS}
        normalized_identity.update(sglang_commit=identity.get("sglang_commit"),
                                   sglang_source_sha256=identity.get("sglang_source_sha256"))
        identities.append(canonical_sha256(normalized_identity))
        row.update(status=result.get("status", "unknown"),
                   measured_speedup=_speedup(result),
                   source_rules_approved=admission["source_rules_approved"],
                   elapsed_wall_seconds=seal["elapsed_wall_seconds"])
        if row["status"] != "passed":
            row["exclusion_reasons"].append(row["status"])
        if row["measured_speedup"] is None:
            row["exclusion_reasons"].append("no_timing_score")
        if not row["source_rules_approved"]:
            row["exclusion_reasons"].append("source_review_not_approved")
        if not seal["recorded_within_budget"]:
            row["exclusion_reasons"].append("unstarted_or_over_budget")
        if not row["exclusion_reasons"]:
            row["ranked_speedup"] = row["measured_speedup"]
        rows.append(row)
    if len(set(identities)) > 1:
        raise ValueError("Refusing to compare different GPU, runtime, image or SGLang identities")
    return {"schema_version": SCHEMA_VERSION,
            "task": "muse_attention_decode",
            "comparison_condition_sha256": manifests[0]["comparison_condition_sha256"],
            "comparison_identity_sha256": identities[0] if identities else None,
            "rows": rows,
            "scope": "local documented-protocol results; not official competition scores",
            "conditions_enforcement": "declared conditions; external runner attestation required"}
