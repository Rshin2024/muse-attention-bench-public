# Public research paper

This is the clean public edition of the seven-model Muse attention-decode study.
It retains nine source-native vector figures and the original scientific
measurements. The dataset is a field-whitelisted numerical projection; original
administrative captures and candidate source bodies are not shipped.

From this public checkout, use Python3.11+, Tectonic and the PDF dependency:

```sh
python3 -m pip install pypdf==6.2.0
python3 scripts/build_public_report.py
```

The builder first runs the public CPU numerical replay and checks the seven
reported outcomes. It then compiles this clean TeX source in a temporary
directory and clears PDF document metadata, XMP, attachments and document
actions. It reads no private workspace and makes no model, cloud or GPU call.
The output is `output/pdf/Muse_Attention_Public_Report.pdf`.

`build-validation.json` records numerical binding, compiler outcome, PDF
structure and artifact hashes. `validation.json` separately records actual
rendered-page visual review. Compiler traces and rendered QA images are local
build intermediates and are not part of the public release.

The report describes private source-review observations, not a public replay of
source adjudication or resource ownership. See `docs/LICENSING.md` and
`docs/PRIVACY.md` for the redistribution and projection boundaries.
