#!/usr/bin/env python3
"""Recheck public numerical data and build a metadata-cleared research PDF.

Requires Tectonic and pypdf==6.2.0 for PDF output. No GPU, provider, cloud or
candidate operation is performed. Only files in this public checkout are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'output/latex/public-report'
PDF = ROOT / 'output/pdf/Muse_Attention_Public_Report.pdf'
EXPECTED = {
    'GPT-6 Astra': 1.1777314777,
    'Claude Opus 5.5': .9734116425,
    'GPT-6.1 Sol': 1.2701674529,
    'GLM 5.3': .002476685,
    'Claude Fable 5.1': 1.156184367,
    'Gemini 4 Argon': .002468858,
    'DeepSeek V4 Pro': .0024715274143491195,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_data() -> dict:
    replay = subprocess.run([sys.executable, str(ROOT/'scripts/reproduce_results.py')],
                            check=True, cwd=ROOT, capture_output=True, text=True)
    if json.loads(replay.stdout).get('status') != 'PASS':
        raise ValueError('The public saved-data replay did not pass')
    data = json.loads((ROOT/'data/results.json').read_text())
    rows = {row['model']: row for row in data['models']}
    if set(rows) != set(EXPECTED):
        raise ValueError('Public report needs exactly the seven recorded model outcomes')
    for model, expected in EXPECTED.items():
        observed = rows[model]['result']['speedup']
        if not math.isclose(observed, expected, rel_tol=2e-6, abs_tol=1e-9):
            raise ValueError('Public report scientific values differ from the selected measurements')
    body = (PROJECT/'main.tex').read_text()
    if body.count(r'\begin{figure}') != 9:
        raise ValueError('Expected nine source-native vector figures')
    if any(value in body for value in ('/Users/', 'gserviceaccount.com', 'GPU-',
            'modal-image:', 'seven-model-20261109', 'muse-deep-recovery')):
        raise ValueError('Private metadata is present in the report source')
    return data


def clear_pdf_metadata(source: Path, target: Path) -> dict:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, TextStringObject

    reader = PdfReader(source)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    # Start a new document info dictionary, rather than preserving TeX's host
    # information or creation timestamps. The page stream remains vector text.
    writer._info = DictionaryObject({NameObject('/Title'): TextStringObject(
        'Muse attention decode: public research report')})
    writer._root_object.pop(NameObject('/Metadata'), None)
    writer._root_object.pop(NameObject('/OpenAction'), None)
    writer._root_object.pop(NameObject('/AA'), None)
    names = writer._root_object.get('/Names')
    if names is not None:
        names = names.get_object()
        names.pop(NameObject('/EmbeddedFiles'), None)
        names.pop(NameObject('/JavaScript'), None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as stream:
        writer.write(stream)
    result = PdfReader(target)
    info = dict(result.metadata or {})
    if set(info) != {'/Title'}:
        raise ValueError('Unexpected retained PDF document metadata')
    if '/Metadata' in result.trailer['/Root'] or result.attachments:
        raise ValueError('Unexpected XMP metadata or embedded PDF attachment')
    for page in result.pages:
        for annotation in page.get('/Annots', []):
            action = annotation.get_object().get('/A')
            if action and action.get('/S') != '/URI':
                raise ValueError('Unexpected active PDF annotation')
    return {'page_count': len(result.pages), 'document_info_keys': sorted(info),
            'XMP_metadata_present': False, 'embedded_attachments': 0,
            'JavaScript_or_document_actions_present': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Validate saved data and paper bindings without compiling')
    args = parser.parse_args(argv)
    validate_data()
    if args.check_only:
        print(json.dumps({'data_and_report_binding': 'PASS', 'PDF_compile_requested': False}))
        return 0
    if not shutil.which('tectonic'):
        raise SystemExit('Install Tectonic to compile the public paper.')
    try:
        import pypdf  # noqa: F401 - checked before starting the compiler
    except ImportError:
        raise SystemExit('Install the report dependency: python -m pip install pypdf==6.2.0')
    with tempfile.TemporaryDirectory(prefix='muse-public-report-') as directory:
        temporary = Path(directory)
        shutil.copyfile(PROJECT/'main.tex', temporary/'main.tex')
        process = subprocess.run(['tectonic', '--keep-logs', 'main.tex'],
                                 cwd=temporary, capture_output=True, text=True)
        log = (process.stdout or '') + (process.stderr or '')
        latex_log = temporary/'main.log'
        if latex_log.exists():
            log += latex_log.read_text(errors='replace')
        if process.returncode:
            # Compiler traces may contain the local temporary-directory path.
            # Do not write them into the distributable checkout.
            raise SystemExit('Public paper compilation failed; no build trace was published.')
        overfull = len(re.findall(r'Overfull \\[hv]box', log))
        if overfull:
            raise SystemExit('Public paper contains an overfull box; repair layout before release.')
        pdf_facts = clear_pdf_metadata(temporary/'main.pdf', PDF)
    receipt = {
        'schema_version': 1,
        'kind': 'public-paper-build-receipt',
        'status': 'BUILT_AND_NUMERICALLY_CHECKED_VISUAL_REVIEW_RECORDED_SEPARATELY',
        'data_and_report_binding': 'PASS', 'Tectonic_compile': 'PASS',
        'overfull_boxes': 0, 'vector_figures': 9,
        'public_projection_not_private_provenance_replay': True,
        'candidate_import_or_execution': False,
        'cloud_model_or_GPU_calls': False,
        'private_workspace_read': False,
        'raw_build_logs_included': False,
        **pdf_facts,
        'artifacts': [
            {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
             'sha256': sha(path)} for path in
            [PROJECT/'main.tex', ROOT/'data/results.json', ROOT/'data/profile.json', PDF]
        ]
    }
    (PROJECT/'build-validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
