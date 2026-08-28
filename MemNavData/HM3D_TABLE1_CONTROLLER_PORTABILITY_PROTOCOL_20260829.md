# HM3D Table-1 controller-portability protocol

This experiment is submitted only if the independent construction verifier for
the fresh-query reserve reports both `verified=true` and
`formal_policy_evaluation_authorized=true`. The population is therefore fixed
without reading any query-controller outcome. It may overlap earlier HM3D
scenes, but it excludes every consumed scene/episode query identity.

Each retained actual-online full-monocular Goal-A history contributes one
unsupported Novel query and one standard-support Revisit query. The runtime
never receives either analysis role. All queries use a 1 m success threshold,
600-step budget, frozen checkpoints, and balanced same-process paired arms.

## Within-controller estimands

- **NavDP:** `mono_native` versus `mono_cec`. Both replay the same causal
  full-monocular history and consume the same causal monocular depth stream.
  Rejection must reproduce the unchanged native ImageGoal request exactly.
- **ViNT:** `forced_reject_native` versus `grant`. Rejection runs the same ViNT
  ImageGoal controller. Acceptance must consume the certified bearing through
  bounded zero-translation turns (`<=30 deg` each), with one fresh observation
  and one shadow-memory update per turn, before the unchanged ViNT controller
  receives the verified historical anchor image.

The two primary estimands are CEC-minus-native paired SR within NavDP and within
ViNT. Absolute NavDP-versus-ViNT SR is contextual because the observation and
action interfaces differ; it is not treated as a paired controller-superiority
estimand.

For Novel, Revisit, and Overall, the report includes SR, paired gain/loss,
two-sided exact McNemar, scene-cluster bootstrap 95% CI, SPL, final distance,
path length, steps, takeover coverage, and exact all-reject fallback. The
80-step smoke is an infrastructure gate only. Neither smoke performance nor
partial formal outcomes may alter the population, arms, thresholds, or
analysis.
