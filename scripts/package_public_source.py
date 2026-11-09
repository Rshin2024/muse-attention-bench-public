#!/usr/bin/env python3
"""Build a source ZIP from an explicit approved public-file allowlist.

The allowlist is a JSON list of repository-relative paths. This program does
not discover files recursively, import candidates, use credentials or make
network calls. Run only after public privacy and licensing review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'output/latex/Muse_Attention_Public_Source.zip'
RECEIPT = ROOT/'output/latex/public-report/source-archive-validation.json'


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def safe_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or '..' in path.parts or str(path) != value or '\\' in value:
        raise ValueError('Archive paths must be normalized repository-relative files')
    if any(part in {'.git', '.local', '.venv', '__pycache__', 'tmp', 'trials', 'results'}
           for part in path.parts):
        raise ValueError('Local/private/build paths are not distributable')
    if path.suffix in {'.zip', '.log', '.pyc', '.pem', '.key'}:
        raise ValueError('Archive, log and credential-like payloads are not allowed')
    result = ROOT/path
    if result.is_symlink() or not result.is_file() or not result.resolve().is_relative_to(ROOT):
        raise ValueError('Archive payload must be an ordinary file in this checkout')
    if result == RECEIPT or path.name in {'release-manifest.json', 'RELEASE_MANIFEST.json', 'SOURCE_ARCHIVE_MANIFEST.json'}:
        raise ValueError('Post-archive hash receipts remain external to avoid cycles')
    if 'muse_attention_decode' in path.parts and 'third_party' in path.parts:
        raise ValueError('Upstream task bodies are intentionally not distributed')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allowlist', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    allowlist = args.allowlist.resolve()
    if not allowlist.is_relative_to(ROOT) or allowlist.is_symlink():
        raise ValueError('The approved allowlist must be a file in this public checkout')
    names = json.loads(allowlist.read_text())
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError('Expected a JSON list of approved paths')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate archive payload path')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.suffix != '.zip':
        raise ValueError('Source archive must be a ZIP inside this public checkout')
    records = []
    payloads = []
    forbidden = set()
    provenance = ROOT/'third_party/provenance.json'
    if provenance.exists():
        forbidden |= {r['sha256'] for r in json.loads(provenance.read_text())['files']}
    dataset = ROOT/'data/results.json'
    if dataset.exists():
        forbidden |= {r['source']['candidate_sha256'] for r in json.loads(dataset.read_text())['models']}
    for name in sorted(names):
        path = safe_path(name)
        body = path.read_bytes()
        if digest(body) in forbidden:
            raise ValueError('An excluded upstream/candidate source body was selected')
        records.append({'path': name, 'bytes': len(body), 'sha256': digest(body)})
        payloads.append((name, body))
    manifest = json.dumps({'schema_version': 1,
        'kind': 'public-source-archive-payload-manifest', 'files': records,
        'archive_manifest_excludes_itself': True,
        'post_archive_release_manifest_external': True,
        'upstream_task_and_candidate_source_bodies_included': False},
        indent=2, sort_keys=True).encode()+b'\n'
    payloads.append(('SOURCE_ARCHIVE_MANIFEST.json', manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, body in payloads:
            info = zipfile.ZipInfo('Muse_Attention_Public/'+name,
                                   date_time=(2026, 11, 9, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(payloads):
            raise ValueError('Source archive integrity check failed')
        for name, body in payloads:
            if archive.read('Muse_Attention_Public/'+name) != body:
                raise ValueError('Source archive payload differs from the approved bytes')
    receipt = {'schema_version': 1, 'kind': 'public-source-archive-integrity-receipt',
        'archive': str(output.relative_to(ROOT)), 'bytes': output.stat().st_size,
        'sha256': digest(output.read_bytes()), 'payload_files': len(records),
        'total_members': len(payloads), 'zip_integrity': 'PASS',
        'all_payload_bytes_verified': True, 'private_workspace_read': False,
        'cloud_model_or_GPU_calls': False, 'symlinks_or_path_traversal': False,
        'release_manifest_intentionally_external': True}
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
