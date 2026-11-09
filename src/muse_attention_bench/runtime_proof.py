"""Explicit launcher evidence for Docker and provider-managed GPU runtimes.

Receipts are assertions supplied by the trusted launcher, not signatures or an
adversarial security boundary. This module validates structure and the locked
image request; it does not independently authenticate a cloud account or query
the provider. Never put credentials in a receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from collections.abc import Mapping

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECEIPT_BYTES = 1024 * 1024


class RuntimeProofError(ValueError):
    """Malformed/mismatched launcher evidence; error strings contain no secrets."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _locked_image(protocol: dict) -> tuple[str, str]:
    image = protocol.get("runtime", {}).get("base_image")
    if not isinstance(image, str) or "@" not in image:
        raise RuntimeProofError("Profile must lock a container image by digest")
    _, digest = image.rsplit("@", 1)
    if not _DIGEST.fullmatch(digest):
        raise RuntimeProofError("Profile container image digest is malformed")
    return image, digest


def _contains_credentials(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            sensitive = any(part in normalized for part in
                            ("token", "password", "secret", "authorization", "cookie", "api_key"))
            if sensitive and child not in (None, "", [], {}):
                return True
            if _contains_credentials(child):
                return True
    elif isinstance(value, list):
        return any(_contains_credentials(child) for child in value)
    return False


def runtime_proof(protocol: dict, environ: Mapping[str, str] | None = None) -> dict:
    """Validate launcher evidence without Torch or any provider network call.

    ``execution_backend`` and ``image_digest`` are comparison identities. The
    receipt hash and per-job details are audit evidence only: different job IDs
    must not by themselves make otherwise equal cloud trials incomparable.
    """
    env = os.environ if environ is None else environ
    backend = env.get("MUSE_ATTENTION_BENCH_RUNTIME_BACKEND", "docker-inspected")
    locked_image, base_digest = _locked_image(protocol)
    if backend == "docker-inspected":
        image_id = env.get("MUSE_ATTENTION_BENCH_IMAGE_DIGEST", "")
        if not _DIGEST.fullmatch(image_id):
            raise RuntimeProofError("Missing inspected evaluator image ID; launch through evaluate, not a copied dry-run command")
        declared_base = env.get("MUSE_ATTENTION_BENCH_BASE_IMAGE_DIGEST")
        if declared_base is not None and declared_base != base_digest:
            raise RuntimeProofError("Launcher base image digest differs from the locked profile")
        evidence = {"execution_backend": backend, "image_digest": image_id,
                    "declared_base_image_digest": declared_base}
        return {
            **evidence,
            "runtime_proof_sha256": hashlib.sha256(_canonical_bytes(evidence)).hexdigest(),
            "image_digest_evidence": "Launcher-provided Docker inspect ID; launched by immutable ID after base-label/platform check. Not independently re-inspected inside the container.",
            "runtime_proof_limitations": "The trusted launcher supplies inspection evidence; this is not an organizer image attestation.",
        }
    if backend != "huggingface-jobs":
        raise RuntimeProofError("Unsupported runtime evidence backend")

    receipt_path = env.get("MUSE_ATTENTION_BENCH_RUNTIME_RECEIPT")
    if not receipt_path:
        raise RuntimeProofError("Hugging Face runtime requires a provider receipt file")
    path = Path(receipt_path)
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeProofError("Provider receipt must be a regular JSON file")
        if metadata.st_mode & 0o222:
            raise RuntimeProofError("Provider receipt must be read-only before evaluation")
        if metadata.st_size > _MAX_RECEIPT_BYTES:
            raise RuntimeProofError("Provider receipt exceeds the size limit")
        raw = path.read_bytes()
        receipt = json.loads(raw)
    except RuntimeProofError:
        raise
    except (OSError, ValueError, UnicodeError):
        raise RuntimeProofError("Provider receipt could not be read as JSON") from None
    if not isinstance(receipt, dict):
        raise RuntimeProofError("Provider receipt must be a JSON object")
    if _contains_credentials(receipt):
        raise RuntimeProofError("Provider receipt contains a credential field; supply sanitized metadata only")
    if receipt.get("provider") != "huggingface-jobs":
        raise RuntimeProofError("Provider receipt is for a different backend")
    for field in ("job_id", "namespace"):
        if not isinstance(receipt.get(field), str) or not _IDENTIFIER.fullmatch(receipt[field]):
            raise RuntimeProofError("Provider receipt needs valid job and namespace identifiers")
    if receipt.get("image") != locked_image:
        raise RuntimeProofError("Provider requested image differs from the locked profile")
    if receipt.get("hardware") != "h200":
        raise RuntimeProofError("Provider receipt must request a single H200")
    if "api_record_sha256" in receipt and not (
            isinstance(receipt["api_record_sha256"], str) and _SHA256.fullmatch(receipt["api_record_sha256"])):
        raise RuntimeProofError("Provider API record hash is malformed")
    if "observed_image" in receipt and receipt["observed_image"] != locked_image:
        raise RuntimeProofError("Provider observed image differs from the locked profile")

    safe_receipt = {key: receipt[key] for key in
                    ("provider", "job_id", "namespace", "image", "hardware", "receipt_source",
                     "observed_at", "api_record_sha256", "observed_image") if key in receipt}
    return {
        "execution_backend": backend,
        "image_digest": base_digest,
        "declared_base_image_digest": base_digest,
        "runtime_proof_sha256": hashlib.sha256(raw).hexdigest(),
        "provider_receipt": safe_receipt,
        "image_digest_evidence": "Sanitized provider job record supplied by the trusted launcher, recording the requested immutable image digest. Authentication of that record is the launcher's responsibility; this is not Docker inspection.",
        "runtime_proof_limitations": "Receipt validation checks structure and locked request fields, not a signature or the live provider. Installed-source fingerprints and actual GPU checks are recorded separately. No organizer image attestation is claimed.",
    }
