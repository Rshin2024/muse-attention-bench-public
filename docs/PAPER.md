# Public paper and source archive

The public paper preserves the scientific outcomes of the original study:
seven first final measurements on a common H200, four authored implementations,
three unchanged-starter controls, calibration and paired timing samples,
research-budget differences and cost qualifications. No new model or GPU
experiment was performed for this edition.

The private audit repository remains separate. This release omits personal
identities, local paths, unique cloud/hardware identifiers, administrative
operations and native conversations. Source bodies with unresolved upstream
redistribution terms are also omitted. Source-review conclusions remain dated
author-reported observations tied to retained scientific hashes; only saved
scalar statistics and public file integrity are independently replayable here.

To check the numerical data and rebuild the PDF:

```sh
python3 scripts/reproduce_results.py
python3 -m pip install pypdf==6.2.0
python3 scripts/build_public_report.py
```

The source archive is assembled only from the explicit public file allowlist.
Its internal manifest binds each payload by size and SHA256. The external final
release manifest and archive receipt are generated after the ZIP to avoid a
self-referential hash cycle. No private audit records are required to rebuild
or verify the public edition.
