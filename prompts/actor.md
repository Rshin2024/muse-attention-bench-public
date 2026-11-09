# Muse attention optimization trial

Work from `kernel/` and read `TASK.md` and `AGENTS.md`. Optimize the computation
implemented by `submission.py` while preserving its `kernel_fn(data)` interface,
the complete required attention, dtypes, cache mutations, numerical contract,
CUDA-graph compatibility, and readable source requirements. The initial files
are identical for every model assigned the same benchmark conditions.

The trial controller supplies the same wall-clock, context, tool, and GPU
diagnostic limits to every model. Limits and tool access are enforced by the
external runner. Do not infer an unlimited budget from this prompt. Diagnostic
waits and compilation count against research wall time. Stop when instructed
and leave your final implementation in `kernel/submission.py`.

Use only the permitted workspace, installed dependencies, and the designated
GPU diagnostic bridge. No network access, public optimized solutions, unrelated
repositories, prior trial results, account files, or credentials are available
or authorized. Do not use official `okc` services, log in, submit entries,
provision compute, or call providers from the research shell. The imported
upstream AGENTS instructions describe the historical competition workflow;
this local trial uses the controller's diagnostic bridge instead.

Do not modify trusted evaluation files, infer hidden seeds, introspect the
evaluator, replay previous outputs, cache input-derived answers across calls,
patch timing, or lower precision outside the applicable kernel contract. Report
negative results and failures accurately. A successful compile is not a speedup.

When time expires, return a short account of the implementation and the exact
diagnostics you observed. The controller seals the final source and evaluates
it independently on fresh inputs. Numerical correctness, source-rule review,
and timing improvement are separate outcomes.
