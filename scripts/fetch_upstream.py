#!/usr/bin/env python3
"""Plan or explicitly download hash-pinned upstream material into an ignored directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "023046877c6eb209bff662a64ca011fbe7403da7"
NAMES = {"AGENTS.md", "TASK.md", "baseline.py", "checks.py", "reference.py", "submission.py"}
PREFIX = f"https://github.com/rohanarun/Auto-research-agent/blob/{COMMIT}/benchmarks/data/kernel-starters/muse_attention_decode/"
RAW_PREFIX = f"https://raw.githubusercontent.com/rohanarun/Auto-research-agent/{COMMIT}/benchmarks/data/kernel-starters/muse_attention_decode/"
LIMIT = 2 * 1024 * 1024


def validate_manifest(manifest: dict) -> list[dict]:
    if manifest.get("upstream_commit") != COMMIT:
        raise ValueError("Unexpected upstream revision")
    entries = manifest.get("files", [])
    names = set()
    for entry in entries:
        path = PurePosixPath(entry["path"])
        if path.parts[:2] != ("third_party", "muse_attention_decode") or len(path.parts) != 3:
            raise ValueError("Unexpected upstream destination")
        name = path.name
        if name not in NAMES or name in names or entry.get("source_url") != PREFIX + name:
            raise ValueError("Unexpected upstream file or origin")
        for key, size in (("sha256", 64), ("git_blob_sha1", 40)):
            if not re.fullmatch("[0-9a-f]{" + str(size) + "}", entry.get(key, "")):
                raise ValueError("Invalid source digest")
        names.add(name)
    if names != NAMES:
        raise ValueError("Expected all six pinned upstream files")
    return entries


def download(entries: list[dict], root: Path = ROOT) -> list[dict]:
    root = Path(root).resolve()
    vendor = root / "third_party" / "muse_attention_decode"
    if (root / "third_party").is_symlink() or vendor.is_symlink():
        raise ValueError("Local upstream directory must not be a symlink")
    vendor.mkdir(parents=True, exist_ok=True)
    written = []
    for entry in entries:
        destination = root / entry["path"]
        if destination.parent != vendor or destination.is_symlink():
            raise ValueError("Unsafe local destination")
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Existing local material has a different digest; refusing to overwrite")
            written.append({"file": destination.name, "status": "already_verified"})
            continue
        request = urllib.request.Request(RAW_PREFIX + destination.name,
            headers={"User-Agent": "Muse-Attention-Benchmark-Upstream-Fetch"})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.geturl() != RAW_PREFIX + destination.name:
                raise ValueError("Unexpected upstream redirect")
            body = response.read(LIMIT + 1)
        if len(body) > LIMIT:
            raise ValueError("Upstream file exceeds download limit")
        git_blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if hashlib.sha256(body).hexdigest() != entry["sha256"] or git_blob != entry["git_blob_sha1"]:
            raise ValueError("Downloaded upstream bytes differ from the pinned source")
        with tempfile.NamedTemporaryFile(dir=vendor, prefix=".verified-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(body)
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        written.append({"file": destination.name, "status": "downloaded_and_verified"})
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly download local copies; default only prints a plan")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "third_party/provenance.json").read_text())
    entries = validate_manifest(manifest)
    notice = "Original upstream terms apply. This helper grants no license and accepts no competition terms. Local copies remain ignored and excluded from public releases."
    result = {"notice": notice, "network_called": bool(args.download),
              "upstream_commit": COMMIT, "files": [entry["path"] for entry in entries]}
    if args.download:
        result["downloads"] = download(entries)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
