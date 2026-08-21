# HM3D actual-online full-monocular mixed-role protocol

Date frozen: 2026-08-20 (Asia/Shanghai), before any outcome from the new
monocular Goal-A collection or any downstream query arm is read.

## Question

Can the complete CEC stack operate from one causal monocular RGB stream when
both the history-producing Goal-A rollout and the subsequent role-unknown
Novel/Revisit queries use the frozen monocular NavDP depth sidecar?

This is a prospective, same-scene HM3D integration experiment.  The nine HM3D
scenes and their generated source episodes have already appeared in the
metric-controller Revisit study, so the experiment is **not** a new-scene
generalization claim.  Its new estimand is the end-to-end sensor/control
contract: no simulator metric-depth read is allowed during Goal-A, replay, or
query execution.

## Frozen source population

- HM3D v0.2 held-out-val assets from the sealed parent manifest with SHA-256
  `62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89`;
- the same nine scene identities frozen by the prior mixed-role protocol;
- all four generated source episodes in each scene, 36 Goal-A sources total;
- source order is inherited byte-for-byte from the parent manifest;
- no source may be dropped because of a new Goal-A or query outcome.

Goal-A is run once with `navdp_depth_source=monocular_sidecar` and
`hybrid_route=native_sidecar`.  The first 40 causal frames use exact zero
depth; a single first-40 RGB-only scale receipt is then frozen.  Every NavDP
plan must report `metric_depth_sensor_consumed=false`.  Its deterministic seed
is `2026082000 + 100 * scene_rank + source_episode_rank`.

## Deterministic construction after Goal-A

Construction is allowed to use only the recorded Goal-A trace and simulator
geometry, never a query-policy result.  A source becomes a materialized
history only when Goal-A succeeded, native-control auditing passes, and the
trace is long enough for the frame-39 plus 16-frame end-margin contract.

For every materialized history, the existing frozen Final14 builder searches:

- Natural Novel: maximum co-visibility over eligible online-A frames `<0.10`;
- standard Revisit: maximum co-visibility in `[0.55,0.90]`;
- query geodesic distance `2--9 m` from the exact mono Goal-A endpoint;
- one Novel/Revisit pair per retained history;
- all four source histories per scene are attempted in frozen order, with no
  post-outcome replacement or population-size backfilling.

HM3D asset lookup must use the explicit, checksummed `.basis.glb` path in the
parent manifest.  Reconstructing an asset path from `scene_id` is forbidden.

## Three paired query arms

Each query independently resets the same server pair and replays the same
actual mono Goal-A RGB trace.  Analysis role is never forwarded to the policy.

1. `mono_native`: native ImageGoal NavDP with the monocular depth sidecar;
2. `mono_raw_fixed`: DINO raw-memory fixed-bearing intervention with the same
   monocular controller;
3. `mono_cec`: frozen CEC certificate and fixed 2.5 m scale-free bearing, with
   exact `mono_native` fallback on rejection.

All three arms share checkpoints, causal RGBs, depth contract, diffusion
seeds, FIFO semantics, 600-step query budget, execution horizon 8, 1.0 m
success radius, and one persistent server pair per scene.  Arm order is
rotated by frozen history index.

## Reporting and claim boundary

Report:

- Goal-A success and trace/scale attrition over all 36 frozen sources;
- constructed histories and scenes before any query rollout;
- role-stratified and balanced overall SR for all three query arms;
- paired gains/losses, exact two-sided McNemar tests, and scene-cluster
  bootstrap intervals for CEC vs native and CEC vs raw fixed;
- Novel certificate takeover, runtime failures, and exact fallback receipts;
- zero metric-depth consumption for Goal-A and every query plan.

Passing the protocol supports a fully monocular, role-free, external-scene
integration claim on reused HM3D scenes.  It does not establish new-scene
generalization, mono-vs-metric non-inferiority, or official GOAT performance.
