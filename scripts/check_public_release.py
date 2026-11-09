#!/usr/bin/env python3
"""Audit an explicit public file allowlist, including PDF objects and ZIP payloads."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|(?:ak|as|mt)-[A-Za-z0-9_-]{30,}|ya29\.[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,})")
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")
PEM = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{80,}")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
LOCAL = re.compile("/" + "Users" + r"/[A-Za-z0-9_. -]{2,}/|/" + "home" + r"/[A-Za-z0-9_.-]{2,}/")
OBJECT_ID = re.compile(r"\b(?:sb|im|ap|ta|cs)-[A-Za-z0-9]{15,}\b|\bGPU-[0-9a-fA-F-]{20,}\b|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IPV4 = re.compile(r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])")
PROJECT = re.compile(r"\bprojects/[a-z][a-z0-9-]{5,}(?:/|\b)")
PRIVATE_FIELDS = {"access_token", "api_key", "apikey", "authorization", "bearer_token",
    "client_secret", "id_token", "private_key", "refresh_token", "session_token",
    "modal_token_id", "modal_token_secret", "opaque_native_state", "native_provider_state",
    "source_absolute_path", "gpu_uuid", "project_id", "quota_project_id", "instance_id",
    "sandbox_id", "image_id", "creation_nonce", "request_id", "service_account_email",
    "hostname", "local_path"}
SAFE_EMAIL_DOMAINS = {"example.invalid", "example.test", "example.com", "users.noreply.github.com"}


def text_findings(text: str, private_identifiers: tuple[str, ...] = ()) -> list[str]:
    found = []
    for label, pattern in (("credential_pattern", SECRET), ("JWT", JWT),
        ("private_key_material", PEM), ("personal_machine_path", LOCAL),
        ("unique_resource_identity", OBJECT_ID), ("cloud_project_path", PROJECT)):
        if pattern.search(text):
            found.append(label)
    if any(value.rsplit("@", 1)[1].lower() not in SAFE_EMAIL_DOMAINS for value in EMAIL.findall(text)):
        found.append("nonplaceholder_email")
    for value in IPV4.findall(text):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if not address.is_loopback and str(address) != "0.0.0.0":
            found.append("nonloopback_IP_address")
            break
    if any(value and value.casefold() in text.casefold() for value in private_identifiers):
        found.append("private_identifier_dictionary_match")
    return sorted(set(found))


def json_findings(value, excluded_digests: set[str]) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in PRIVATE_FIELDS and item not in (None, "", False, [], {}):
                found.append("nonempty_restricted_JSON_field")
            found.extend(json_findings(item, excluded_digests))
    elif isinstance(value, list):
        for item in value:
            found.extend(json_findings(item, excluded_digests))
    elif isinstance(value, str):
        if hashlib.sha256(value.encode()).hexdigest() in excluded_digests:
            found.append("excluded_source_embedded_as_string")
        if len(value) > 80:
            try:
                decoded = base64.b64decode(value, validate=True)
            except (ValueError, TypeError):
                decoded = b""
            if hashlib.sha256(decoded).hexdigest() in excluded_digests:
                found.append("excluded_source_embedded_as_base64")
    return sorted(set(found))


def inspect_payload(name: str, body: bytes, *, private_identifiers=(), excluded_digests=frozenset(), depth=0) -> tuple[list[dict], dict]:
    problems = []
    counts = {"payloads": 1, "PDFs": 0, "ZIPs": 0, "JSONs": 0}
    def flag(kinds):
        problems.extend({"path": name, "kind": kind} for kind in kinds)
    flag(text_findings(name, private_identifiers))
    if hashlib.sha256(body).hexdigest() in excluded_digests:
        flag(["excluded_source_body"])
    suffix = PurePosixPath(name).suffix.lower()
    if body.startswith(b"%PDF-") and suffix != ".pdf":
        flag(["PDF_magic_suffix_mismatch"])
        suffix = ".pdf"
    elif body.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")) and suffix != ".zip":
        flag(["ZIP_magic_suffix_mismatch"])
        suffix = ".zip"
    if suffix == ".zip":
        counts["ZIPs"] += 1
        if depth > 1:
            flag(["unexpected_nested_archive"])
            return problems, counts
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
                raise ValueError("Release archive expands beyond audit limit")
            if len(archive.namelist()) != len(set(archive.namelist())):
                flag(["duplicate_ZIP_member_names"])
            if archive.testzip() is not None:
                flag(["ZIP_integrity_error"])
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    flag(["unsafe_ZIP_member"])
                if info.is_dir():
                    continue
                nested, subtotal = inspect_payload(name + "::" + info.filename, archive.read(info),
                    private_identifiers=private_identifiers, excluded_digests=excluded_digests, depth=depth + 1)
                problems.extend(nested)
                for key, count in subtotal.items():
                    counts[key] += count
    elif suffix == ".pdf":
        counts["PDFs"] += 1
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted:
            flag(["encrypted_PDF_cannot_be_audited"])
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        flag(text_findings(text, private_identifiers))
        flag(text_findings(str(dict(reader.metadata or {})), private_identifiers))
        if reader.xmp_metadata is not None or list(reader.attachments):
            flag(["PDF_XMP_or_attachment"])
        seen = set()
        def walk(obj):
            identity = id(obj)
            if identity in seen:
                return
            seen.add(identity)
            if hasattr(obj, "get_object"):
                resolved = obj.get_object()
                if resolved is not obj:
                    walk(resolved)
                    return
            if hasattr(obj, "get_data"):
                try:
                    decoded = obj.get_data()
                except Exception:
                    flag(["PDF_stream_could_not_be_audited"])
                else:
                    flag(text_findings(decoded.decode("utf-8", errors="replace"), private_identifiers))
                    if hashlib.sha256(decoded).hexdigest() in excluded_digests:
                        flag(["excluded_source_in_PDF_stream"])
            if isinstance(obj, str):
                flag(text_findings(obj, private_identifiers))
                if obj in {"/JavaScript", "/JS", "/Launch"}:
                    flag(["active_PDF_content"])
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    walk(key)
                    walk(item)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
        walk(reader.trailer)
    else:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            flag(["nontext_payload_with_unrecognized_format"])
            return problems, counts
        flag(text_findings(text, private_identifiers))
        if suffix == ".json":
            counts["JSONs"] += 1
            flag(json_findings(json.loads(text), set(excluded_digests)))
    return problems, counts


def audit(root: Path, paths: list[str], *, private_identifiers=(), excluded_digests=frozenset()) -> dict:
    root = Path(root).resolve()
    problems = []
    counts = {"payloads": 0, "PDFs": 0, "ZIPs": 0, "JSONs": 0}
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate public allowlist path")
    for name in paths:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or any(part in {".git", ".local", "tmp", "__pycache__", "trials", "results"} for part in relative.parts):
            raise ValueError("Unsafe release allowlist path")
        if relative.parts[:2] == ("third_party", "muse_attention_decode"):
            raise ValueError("Locally fetched task source cannot enter a release")
        path = root / name
        if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
            raise ValueError("Missing or symlinked release input")
        findings, subtotal = inspect_payload(name, path.read_bytes(), private_identifiers=private_identifiers,
            excluded_digests=excluded_digests)
        problems.extend(findings)
        for key, value in subtotal.items():
            counts[key] += value
    return {"schema_version": 1, "status": "PASS" if not problems else "FAIL",
        "allowlisted_files": len(paths), "inspected_payloads_including_archive_members": counts,
        "private_identifier_dictionary_checked": bool(private_identifiers),
        "excluded_source_digest_count": len(excluded_digests), "finding_count": len(problems),
        "findings_paths_and_categories_only": problems,
        "scope": "Field/pattern, known private-identifier and excluded-source-body checks; no universal secret-absence guarantee."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--private-identifiers", type=Path)
    parser.add_argument("--excluded-digests", type=Path, default=ROOT / "docs/excluded-source-digests.json")
    args = parser.parse_args(argv)
    private = tuple(json.loads(args.private_identifiers.read_text())) if args.private_identifiers else ()
    excluded = set(json.loads(args.excluded_digests.read_text())["sha256"])
    result = audit(args.root, json.loads(args.allowlist.read_text()), private_identifiers=private, excluded_digests=excluded)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
