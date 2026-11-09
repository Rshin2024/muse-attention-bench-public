# Measurement scope

This harness reconstructs the publicly documented kernel benchmark. It is not
the organizer's hosted judge, and its scores are not organizer-issued results.

The saved experiment used H200 SXM, one visible GPU per evaluator, pinned
software, 16 mandatory correctness cases, 12 scored cases, cold-L2 graph
replays, per-run empty-graph calibration, swapped-order timing, six weighted
loads and geometric aggregation. Each load combines sliding and full attention
with weights 46 and 16. Bootstrap intervals use the frozen pure scoring helper,
2,000 resamples and the documented local interval floor.

The organizer's private seeds, exact repeat/calibration implementation, hidden
reruns and hosted anti-cheat implementation are unavailable. Local source review
and sandboxing do not reproduce that entire system. Hardware/software type is
retained in the public methods, while unique machine and cloud identifiers are
removed for privacy.

The research cap was 1,800 seconds, with at most three public diagnostics and
two public cases per diagnostic, identical initial files/prompt and native high
reasoning settings. The same spelling of an effort setting does not establish
equal internal reasoning compute or spending. Models can finish early or fail.

The final comparison uses the first full measurement of each artifact on the
common final board. Six authoring sources precede the repaired DeepSeek run.
The administrative timeout repair adds a 20-second outer cleanup margin and
changes the late CPU reserve from 45 to 70 seconds. Complete authoring conditions
therefore differ, although the final numerical/timing conditions are matched.

All four authored artifacts passed the privately performed current kernel-rule
source review. None satisfies the older frozen task's BF16 activation-rounding
wording. The serving track has its own stricter requirements. The three unchanged
starter outcomes are controls, not model-produced optimizations.

The public derived dataset permits scalar replay. Source adjudication and
original event/resource provenance remain private and are not independently
replayable from this release.
