# Working on the public benchmark

This repository is a derived public release. Keep personal identities, machine
paths, cloud account/resource identifiers, credentials, provider conversations,
and administrative records out of Git and release archives.

The six upstream task files and all candidate sources are deliberately omitted.
Do not add local downloads to Git or archives. Read docs/LICENSING.md and use the
explicit local download helper only when the operator chooses to obtain them.

The public dataset supports CPU replay of saved scalar measurements. It does not
replay private source adjudication, establish hosted-judge equivalence, or rerun
the GPU experiments. Preserve these qualifications in reports.

Run CPU checks with `PYTHONPATH=src python3 -m unittest discover -s tests`.
Do not run paid APIs, rent GPUs, consume hosted checks, or accept external terms
without an explicit operator action and budget.
