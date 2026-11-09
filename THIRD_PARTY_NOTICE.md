# Third-party provenance and release boundary

The root MIT license applies to the original project contributions distributed
in this public export. It does not relicense independently obtained dependencies
or upstream challenge materials.

The Open Kernel Challenge `muse_attention_decode` task package is **not bundled**.
Its six known public-mirror file identities and locations are recorded in
[third_party/provenance.json](third_party/provenance.json). The
[pinned mirror ownership notice](https://github.com/rohanarun/Auto-research-agent/blob/023046877c6eb209bff662a64ca011fbe7403da7/benchmarks/NOTICE.md)
reserves the original terms of upstream task statements, starters, references,
checks and baselines. The mirror's parent MIT license is not treated as a grant
for those files.

Candidate and diagnostic kernel source bodies are also omitted from this public
release pending separate upstream-expression provenance clearance. Measurements
and recorded source hashes are retained where included; their presence does not
provide source code or permission to redistribute it.

SGLang is an external dependency. Its
[pinned source license](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/LICENSE)
is Apache-2.0. No SGLang source tree, binary package or container image is
redistributed in this repository.

GPU MODE's reference-kernels/kernelbot projects are cited as upstream evaluator
ancestry. This repository does not bundle their evaluator code. The
[reference-kernels license](https://github.com/gpu-mode/reference-kernels/blob/main/LICENSE)
is the June 9 Researcher Reciprocity License; those projects are not relabeled
as MIT or Apache by this release.

See [the licensing review](docs/LICENSING.md), checked 9 November 2026, for the
primary sources, inclusion decisions and reproduction limits.
