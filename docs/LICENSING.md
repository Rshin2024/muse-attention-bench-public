# Licensing review

Checked **9 November 2026** against the pinned upstream sources below.

The root MIT license covers the original software and documentation distributed
in this public export. Dependencies and material obtained separately retain
their own terms. This review records the evidence used for the release; it does
not create permission for excluded upstream material.

| Material | Finding | Public release treatment |
|---|---|---|
| Original benchmark harness, analysis, report prose and figures | Original project contributions are released under the root MIT license. | Included with the license and an explicit public-export description. |
| Six mirrored challenge task files | The pinned mirror's ownership notice reserves upstream task terms. Its MIT license does not establish a separate grant for these files. | Source bodies are omitted, including nested copies. The provenance manifest retains public URLs and hashes. |
| Candidate and diagnostic kernel source | The study used the supplied starter as context; output ownership alone does not establish clearance for any upstream expression retained in a candidate. | Source bodies are conservatively omitted. Numerical observations, source hashes, algorithm summaries and recorded review outcomes may remain. |
| SGLang | The baseline's pinned SGLang revision has an Apache-2.0 license. That license does not establish the terms of the organizer's task wrapper. | SGLang source and binaries are not bundled. Dependencies remain separately installed. |
| GPU MODE reference-kernels / kernelbot ancestry | The current reference-kernels license is the June 9 Researcher Reciprocity License. | No code from those evaluators is bundled or relabeled as MIT. |

## Primary sources

The [pinned mirror ownership notice](https://github.com/rohanarun/Auto-research-agent/blob/023046877c6eb209bff662a64ca011fbe7403da7/benchmarks/NOTICE.md)
distinguishes its contributed runner and documentation from upstream problem
statements, starters, checks and baselines. The mirror's
[MIT license](https://github.com/rohanarun/Auto-research-agent/blob/023046877c6eb209bff662a64ca011fbe7403da7/LICENSE)
therefore is not treated as blanket permission for the mirrored task package.

The [SGLang license at the revision identified by the baseline](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/LICENSE)
is Apache-2.0. Any later distribution of its source must preserve applicable
license and attribution notices and identify modifications as required by that
license.

The [reference-kernels license](https://github.com/gpu-mode/reference-kernels/blob/main/LICENSE)
contains its own attribution, redistribution and use conditions. An ancestry
citation does not assign that license to a separate challenge task package.

The [official task](https://openkernelchallenge.dev/problems/muse_attention_decode)
and [rules](https://openkernelchallenge.dev/rules) identify the benchmark.
The official pages did not provide a readable blanket redistribution grant
during this review. No such grant is inferred. The excluded source can be
obtained separately from its upstream location subject to the applicable terms;
the public export does not accept terms or authorize downstream use for a user.

## Reproduction boundary

This release is a derived, privacy-scrubbed publication. The private original
audit repository retains its complete captures and original history. New public
hashes describe the files actually distributed here.

Public numerical replay can check included scalar observations and aggregation.
It cannot repeat a source-body review or execute an omitted candidate. A source
hash records identity, not a license grant or a substitute for the missing body.
Keep separately obtained task files and any private candidate sources outside
the public release archive and its Git history.

Required third-party public copyright and attribution notices must be retained
when their licensed material is distributed. Private email addresses, local
machine paths and cloud identifiers are not required attribution.
