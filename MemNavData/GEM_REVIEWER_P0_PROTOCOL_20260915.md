# Reviewer P0 supplement, 2026-09-15

Do not change the manuscript before interpreting the complete evidence.

## Control attribution (local pilot)

Retain the independently verified four-history full-GEM / initial-turn-only experiment.
Freeze 20 additional executions before viewing outcomes: native and a fixed positive
180-degree initial scan on all eight existing Novel/Revisit goals; GEM with explicit
turning disabled on the four Revisit goals. No goal label, goal ground-truth bearing,
retrieval, or PnP may choose the fixed scan direction or duration. The native and scan
arms use the original ImageGoal and the same monocular-depth sidecar. Every scan action
costs one of the same 600 actions, is bounded to 4.5 degrees, and produces a fresh RGB.
The sidecar performs no historical readout for control in those arms. All parameters,
models, history, query RGB, seeds, executor and W64 storage are otherwise frozen.

The fixed half-turn is a deliberate test of the simplest 'turn around' explanation.
It is not a general scene-search system or a reproduction of a published baseline.
Testing Novel as well as Revisit measures the cost of applying it without role labels.
There is no validation-driven choice of turn sign, duration, scene, or random seed.
These four histories were already consumed during development. Report a pilot, not a
fresh independent success-rate estimate; retain failures and direction-invariant results.

Record SR, actual path, total and turning actions, final planar error and per-query
paired outcomes. Verify actual actions, RGB/depth provenance, no historical calls in
memory-free arms, fixed scan duration, and unchanged initial cues for GEM without turns.

## Existing-evidence audit (CPU)

Use sealed main-table and mechanism-table records where available. Derive selected
anchor age, cumulative travel since that anchor, ground-truth anchor–goal displacement,
goal bearing error and current/recent/all-history covisibility. Read the controller's
actual context contract rather than assuming that W64 is a controller observation window.
Report missing fields and denominators. Ground truth is used only in this offline audit.

Predefine descriptive windows of 16/32/64 raw observations, current-view support below
0.1, recent-window maximum support below 0.1, and old support at least 0.5. These descriptive
windows must not be relabeled as the exact controller FIFO without verifying its sampling.
Offset bins are [0,0.5), [0.5,1), [1,2), [2,infinity) meters in the horizontal plane.
Report signed and absolute planar bearing error, median and 90th percentile, and counts
above 30/90 degrees. Accepted and rejected queries have separate denominators; Novel is
not a false-match label. No threshold fitting to navigation outcomes.

Old-history dependence must be established by support availability, not total history
length alone. A recent-only versus all-history readout/navigation comparison remains
separate evidence. Traces stopped at one meter do not establish autonomous stopping or
success at tighter thresholds. Yaw-only queries do not establish positional-offset robustness.
