# Muse attention-decode benchmark

A public, privacy-filtered release of an H200 model-agent experiment on
`muse_attention_decode`. This repository starts with fresh Git history. It
contains the original evaluation harness, derived numerical observations, and
an illustrated report. It has no operator account identities, cloud resource
receipts, API credentials, provider conversations, or private machine paths.

## Results

All seven selected artifacts received their first final measurement on the same
H200 SXM. Each research opportunity had a 30-minute cap. The unchanged-baseline
control measured **0.996995x**.

| Model | Authored-kernel correctness | Common-H200 speedup | Known API estimate, including readiness |
|---|---|---:|---:|
| GPT-6.1 Sol | 28/28 | **1.2702x** | $0.60 |
| GPT-6 Astra | 28/28 | 1.1777x | $2.11 |
| Claude Fable 5.1 | 28/28 | 1.1562x | $8.25 |
| Claude Opus 5.5 | 28/28 | 0.9734x | $2.17 |
| GLM 5.3 | Starter unchanged | No optimization delivered | $0.52 plus unresolved usage |
| Gemini 4 Argon | Starter unchanged | No optimization delivered | $0.01 plus unresolved usage |
| DeepSeek V4 Pro | Starter unchanged | No optimization delivered | $0.57 plus unresolved usage |

The unchanged starter observations also passed all public numerical cases;
they are excluded from authored-kernel performance points. Four modified
kernels passed the captured current kernel-rule source review. They differ
from the older task's BF16 activation-rounding requirement, and these results
do not certify eligibility for the stricter serving track.

Six sources were retained after an interrupted run. A fresh DeepSeek attempt
used an independent diagnostic GPU while those six sources were replayed on
the common final GPU. The recovery repaired an administrative CPU timeout and
reduced the maximum late-stage CPU allowance by 25 seconds. Final timing and
numerical conditions match; complete source-authoring conditions differ. One
attempt per model does not establish a general model-ability ranking.

Known whole-project public-tariff estimates total **$300.26**, including earlier
engineering. The separate **$392 unresolved reserve is not a confirmed bill**.
The selected seven opportunities have $14.228959724 in known API estimates;
their unresolved reserve subset is $90. Unpriced fees remain unknown.

[Read the illustrated paper](output/pdf/Muse_Attention_Public_Report.pdf),
[edit the LaTeX source](output/latex/public-report/), or inspect the
[derived dataset](data/results.json). The complete
[public source archive](output/latex/Muse_Attention_Public_Source.zip) includes
the harness, dataset, report source and its own payload manifest.

## Replay the saved results without a GPU

Use Python 3.11 or newer. The numerical replay uses only the standard library:

```sh
python3 scripts/reproduce_results.py
PYTHONPATH=src python3 -m unittest discover -s tests
```

The replay checks saved calibration samples, swapped-order timing brackets,
weighted loads, geometric scores and bootstrap intervals for seven artifacts
and two baseline controls. It does not import candidates, contact providers,
use cloud credentials, or rerun GPU correctness. Original event authenticity,
source-review adjudication and cloud cleanup were checked privately; this
derived release does not independently replay those provenance checks.

## Run a new local benchmark

The original harness code is included. Upstream task files and historical
candidate sources are omitted because a redistribution grant for their
retained upstream expression was not established. Read
[the licensing review](docs/LICENSING.md) before obtaining local copies.

Print the pinned upstream download plan without network access:

```sh
python3 scripts/fetch_upstream.py
```

If you choose to obtain those public files subject to their original terms,
download them explicitly into the Git-ignored local task directory:

```sh
python3 scripts/fetch_upstream.py --download
python3 -m pip install -e .
muse-attention-bench verify
muse-attention-bench profile
muse-attention-bench build-image --dry-run
```

The downloader checks the fixed origin, revision, SHA-256 and Git blob IDs. It
does not accept competition terms. Local task files, submissions, credentials
and experiment outputs are excluded from Git and source-release archives.

For a new GPU run, prepare an exclusive H200 SXM and the pinned image. These
commands are examples; running them uses your own compute:

```sh
muse-attention-bench build-image
muse-attention-bench container-doctor --gpu 0 --attest-h200-sxm
muse-attention-bench evaluate /path/to/your/submission.py \
  --output results/example --gpu 0 --attest-h200-sxm
```

The example provider configuration contains environment-variable names and
public endpoints only. Keep your keys in your own trusted local credential
store; never commit them or mount them into model/evaluator sandboxes. The
configuration file and read-only `providers` command launch no model trials.

## Rebuild and audit this public release

See [paper rebuilding](docs/PAPER.md), [privacy and transformation scope](docs/PRIVACY.md),
and [measurement limits](docs/PARITY.md). The public manifest records new hashes
for the transformed release. It does not relabel transformed files as original
producer captures. The complete original audit repository remains private.

Original harness, prose, figures and the derived dataset use the project MIT
license. Upstream task material retains its original terms and is not
distributed here. See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md).
