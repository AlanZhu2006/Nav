# Monocular height-calibrated route progress

Date: 2026-09-02; updated 2026-09-03 (Asia/Shanghai)  
Status: **prospective index-40 mechanism gate failed; cumulative path-budget route compass rejected; closed-loop SR remains unauthorized**

## 1. Final modality boundary

The deployable method may consume only:

- the causal monocular RGB stream;
- the ImageGoal;
- calibrated camera intrinsics;
- one known, fixed camera mounting height.

The camera height is a calibration prior, not a depth observation.  It may
convert LingBot's relative depth gauge into one metric scale receipt, frozen
from RGB frames `0..39`.  The method may not consume simulator depth,
simulator pose, wheel odometry, an IMU trajectory, a shortest path, a success
label, or the hidden Novel/Revisit role.

Consequently, the Habitat pose-difference SE(2) branch is an analysis upper
bound only.  It is not a candidate paper method and no closed-loop job may be
reported as monocular if it uses that receipt.

## 2. Evidence that constrains the design

Three tempting shortcuts have already been tested and must not be repeated.

1. **Metric endpoint distance does not solve long returns.** A larger or
   dynamically scaled endpoint residual remained worse than the fixed 2.5 m
   bearing.  Long-range error is primarily route topology, not missing chord
   magnitude.
2. **The global LingBot pose curve drifts.** On the medium and long diagnostic
   histories its maximum route cross-track error reached `3.89 m` and
   `4.95 m`.  Multiplying that curve by a camera-height scale does not remove
   its rotational or accumulated translational drift.
3. **Raw LingBot consecutive pose differences are not reliable odometry.** On
   a `22.96 m` controlled return they predicted only `8.60 m` final progress,
   with `14.35 m` final absolute error and `53.25 deg` median route-bearing
   error.  Six turn-only observations hallucinated `2.27 m` of translation.

The height prior therefore belongs at the depth-to-local-geometry boundary;
it must not be used to cosmetically metricize a failed global pose estimate.

## 3. Frozen candidate: one visual witness at two time scales

The same frozen geometric primitive is reused for target proof and local
motion:

```text
causal RGB I_t
    -> LingBot relative depth D_t^rel
    -> first-40 height receipt s_hat
    -> metric monocular depth D_hat_t = s_hat D_t^rel

adjacent RGB pair (I_{t-1}, I_t)
    -> SuperPoint + LightGlue correspondences
    -> lift keypoints with D_hat_{t-1}
    -> PnP-RANSAC in the previous-camera frame
    -> local planar receipt (delta forward, delta left, delta yaw)
```

No global map is constructed.  During the historical traversal, each accepted
adjacent-frame solve is immediately reduced to one local SE(2) edge; only the
edge and its evidence receipt are retained.  The previous frame's depth is
held for one step, so dense depth for every old frame need not be cached.

Frames before the first-40 scale is frozen may retain raw-scale PnP
translations.  The immutable scale multiplies those translations only after
frame 40; no future image is used to estimate the scale.

After the ordinary CEC target certificate selects a historical anchor:

```text
historical local edges -> reverse causal route tape
query adjacent edges   -> live local pose
live pose              -> causal non-expansive projection onto route
route point 2.5 m ahead -> unit bearing only
unit bearing           -> unchanged frozen NavDP
```

The projected coordinate obeys

```text
s_(t-1) <= s_t <= min(L, s_(t-1) + ||delta p_t||).
```

This is a kinematic invariant, not a tuned confidence gate.  It prevents a
nearest-point projection from jumping across a self-intersection or a nearby
later corridor.  The controller still receives a fixed 2.5 m PointGoal; the
metric scale is used to define local route progress and lookahead, not to send
an unbounded endpoint distance to NavDP.

## 4. Runtime failure contract

Normal CEC rejection remains exact native ImageGoal navigation.  Once CEC has
authorized a historical route, there is no distance switch, endpoint branch,
stuck rescue, or native fallback inside the challenger.  If the shared
monocular geometry stream cannot produce a valid adjacent-frame motion
receipt, that is an explicit geometry-stream failure and the experimental arm
stops.  It must not silently substitute simulator motion or nominal actions.

This distinction preserves a total scientific accounting:

- insufficient target-history evidence -> ordinary CEC abstention;
- invalid shared monocular state -> method failure;
- valid state -> one route-conditioned unit bearing.

## 5. Mechanism gate before any SR run

The first test uses already-consumed controlled histories and reads no
navigation success.  It must measure, for short, medium, and long routes:

- adjacent-frame solve coverage and failure reasons;
- false translation during turn-only observations;
- cumulative path-length bias;
- local yaw error;
- route-bearing error, including the fraction within the previously measured
  `30 deg` useful-bearing tolerance;
- progress monotonicity and the non-expansion invariant;
- per-frame latency and retained memory.

Evaluator poses may score these quantities only after inference.  They may not
enter matching, PnP, route construction, scale recovery, or projection.

The mechanism is rejected before closed-loop evaluation if it repeats the
known failure pattern of the raw LingBot pose tape: systematic under-travel,
turn-induced translation, or bearing error outside the useful range.  Passing
schema checks alone is insufficient.

## 6. Index-16 adjacent-motion audit

### 6.1 Original sampled stream: useful diagnosis, invalid chain test

The first immutable audit used the previously generated 78-frame reverse
stream (`77` adjacent pairs). It exposed evaluator pose only after all visual
estimates had been fixed. Its result was:

| Measure | Result |
|---|---:|
| PnP pose available | `76/77` |
| full sparse-CEC certificate accepted | `68/77` |
| translation-vector error, median / P90 | `0.0159 / 0.0906 m` |
| translation-direction error, median / within 30 deg | `0.694 deg / 69 of 71` |
| turn-only false translation, P95 | `0.00249 m` |

All 68 fully certified estimates were well behaved: their maximum translation
direction error was `13.2 deg` and maximum yaw error was `3.23 deg`. The
rejected group contained the catastrophic estimates, including one `109.5
deg` yaw error. This confirms that the existing sparse target certificate is
not decorative.

It also revealed a construction defect in that stream. At route corners the
generator translated approximately `0.30 m` and replaced the camera heading
with the following segment's heading in one adjacent observation. Three such
edges combined translation with `71--168 deg` yaw changes. The PnP chain first
broke at pair 31; therefore this artifact did not pass the chain-level gate.
It is retained unchanged as diagnosis, not reinterpreted as a runtime failure.

Artifact:

```text
.diagnostics/mono_adjacent_motion_20260903/
  index16_attempt1/adjacent_motion_audit.json
SHA-256 8d6e0eb81bdd0310c97969a9a00cf26d4222f9158628cacb328e1c35ac7dee9c
```

### 6.2 Physically continuous stream: local-motion gate passed

A separate v2 construction kept the same frozen history, target, sampled
positions, and `22.9567 m` route. It inserted the observations a real
turn-then-translate controller would produce. Across 148 RGB frames:

- maximum in-place yaw increment was `15 deg`;
- maximum translation increment was `0.337 m`;
- no edge contained simultaneous translation and a heading discontinuity;
- all 148 RGB hashes and the query-set hash were verified before inference.

This audit distinguishes two evidence contracts without changing either one's
numeric thresholds. The open-set CEC target witness keeps hull coverage. A
temporally adjacent motion receipt cannot infer target identity or role; it
uses PnP status, at least 16 inliers, and at most 2 px reprojection RMSE. This
principled distinction was fixed after the v1 failure analysis and before any
index-40 output.

| Measure | Result |
|---|---:|
| PnP pose available | `147/147` |
| adjacent-motion receipt valid | `147/147` |
| full sparse-CEC certificate accepted | `142/147` |
| true / estimated traversed distance | `22.9567 / 22.0206 m` |
| path-length bias | `-0.9361 m` (`-4.08%`) |
| final integrated position error | `0.275 m` |
| translation-vector error, median / P90 | `0.0173 / 0.0359 m` |
| translation-direction error, median / P90 | `0.584 / 2.55 deg` |
| translated edges within 30 deg | `72/72` |
| turn-only false translation, median / P95 / max | `0.00102 / 0.00350 / 0.00554 m` |
| yaw error, median / P90 / max | `0.049 / 0.287 / 3.76 deg` |
| depth request / local match+PnP median latency | `286.6 / 25.6 ms` |

The five full-certificate rejects were all hull-coverage rejects; all five
still satisfied the adjacent-motion contract. No controller or NavDP policy
was loaded by the audit client, and no success, final-distance, role, metric
sensor depth, global pose, or odometry value entered inference.

Artifacts:

```text
.diagnostics/mono_adjacent_motion_20260903/
  dense_reverse_index16_v2/query_set.json
  index16_dense_v2_attempt1/adjacent_motion_audit.json
query SHA-256  aa994858ee3213bb3dc858763294be9f836e25b3a76a894dd605dd14dc334a28
audit SHA-256  0ce8299682c5e54acf004de4992be02c179d363b35fcabf7abd1e8726ad1d701
```

This passes only the **live local-motion half** of the index-16 development
gate. It does not yet show that the actual outgoing A history forms a complete
monocular route tape, that projected route bearings remain accurate, or that
NavDP succeeds. Those claims remain withheld.

### 6.3 Actual-online history route: geometry passes, per-step projection fails

The next audit used the CEC-authorized anchor 26 rather than the scorer's true
source frame. It replayed all 1,206 actual-online A frames. From the authorized
anchor to the history tail it formed 1,166 chronological edges:

- 379 byte-identical consecutive JPEG pairs produced exact identity motion;
- 787 non-identical pairs used height-scaled monocular depth and
  SuperPoint/LightGlue/PnP;
- all `1,166/1,166` adjacent-motion receipts were valid;
- history translation-vector error was `0.00187 m` median and `0.00733 m` P90;
- history yaw error was `0.0118 deg` median and `0.0719 deg` P90.

The reconstructed route remained close in shape but accumulated scale and
orientation error over the full traversal:

| Measure | Predicted / observed |
|---|---:|
| route extent | `20.930 / 23.414 m` |
| median / P90 route-position error | `0.543 / 0.781 m` |
| route endpoint error | `1.026 m` |

When this valid route and the valid dense-return motion were passed through
the v2 *per-step* non-expansion rule, route progress froze at `9.018 m`. Final
progress lag was `13.939 m`; only `34/72` translated readouts were within 30
degrees and median translated bearing error was `39.94 deg`.

This is not a PnP or coordinate-sign failure. The estimated query endpoint was
only `0.275 m` from its scorer pose and the estimated historical route endpoint
was `1.026 m` from its scorer endpoint. The failure came from discarding unused
motion budget at every projection step. A small lag at a corner could never be
recovered because v2 enforced
`s_t <= s_(t-1) + ||delta p_t||` independently at every frame.

Artifact:

```text
.diagnostics/mono_adjacent_motion_20260903/
  route_compass_index16_v1_attempt1/monocular_route_compass_audit.json
SHA-256 4ca9061fc98f62a3fb9e82cf25d7075e61ae416766098e35523c7c34b7dd198e
```

### 6.4 Cumulative path-budget projection

The replacement preserves unused causal motion budget:

```text
s_(t-1) <= s_t <= min(L, q_t),
q_t = sum_(i<=t) ||delta p_i||.
```

The two-dimensional nearest-route projection still decides progress. Total
travel does not itself become route progress; it is only an upper bound. The
rule has no distance bin, learned/tuned gate, endpoint branch, stuck rescue, or
native fallback. At an exact self-intersection tie, the earliest reachable arc
wins.

Replaying the same sealed index-16 motions, without rerunning a model or
controller, produced:

| Measure | Per-step budget | Cumulative budget |
|---|---:|---:|
| final projected progress | `9.018 m` | `20.930 / 20.930 m` |
| translated bearing within 30 deg | `34/72` | `64/72` |
| translated bearing median error | `39.94 deg` | `4.51 deg` |
| translated bearing P90 | not decision-bearing | `31.21 deg` |
| median normalized progress error | -- | `0.0190` |
| monotone / path-budget compliant | yes / yes | yes / yes |

For translated observations with more than 1 m of scorer route remaining,
`64/70` bearings were within 30 degrees, with median/P90 errors of
`4.38/22.94 deg`. The excluded near-goal interval is not a runtime switch: it
is reported separately because a roughly 1 m route-endpoint error makes the
direction between two already-near-goal points ill-conditioned. Indeed, the
12 terminal in-place views had a median bearing error of `176.2 deg`.
Whether the unchanged ImageGoal policy stops rather than following that
ill-conditioned residual is a closed-loop question; this mechanism audit does
not answer it.

Artifact:

```text
.diagnostics/mono_adjacent_motion_20260903/
  path_budgeted_index16_v1/path_budgeted_route_replay.json
SHA-256 0f0d9625f6bd7f02e417418f1a8b69c6a3f42214734da69f2781c51fb9d83692
```

The cumulative rule is now frozen before any output is computed on index 40.
Its index-16 improvement is post-hoc development evidence, not confirmation.

### 6.5 Prospective index-40 decision rule

Index 40 remains the prospective mechanism row fixed by the preceding
action-coordinate protocol: scene `Qpor2mEya8F`, 1,693 history frames, and
CEC-authorized anchor 30. The dense RGB construction must use the unchanged
`0.30 m` position spacing and at most `15 deg` per in-place turn. It may not be
regenerated after observing a model output.

The complete mechanism passes only if all of the following hold:

- both actual-history and dense-return adjacent-motion chains are complete;
- neither history nor query estimated path length differs from its scorer
  length by more than 15%;
- final query-pose error is at most 1.0 m;
- historical route endpoint error is at most 1.25 m. This is the maximum
  lateral displacement compatible with the previously established 30-degree
  bearing tolerance at a 2.5 m lookahead (`2.5 sin 30 deg`), not a fitted
  index-16 quantile;
- at least 80% of translated route readouts are within 30 degrees and their
  median bearing error is at most 20 degrees;
- progress is finite, monotone, and never exceeds cumulative visual path
  length;
- every controller payload is a unit bearing projected to exactly 2.5 m;
- all information-boundary flags remain false for evaluator pose, metric
  sensor depth, global pose, wheel odometry, role, controller outcome, and SR.

Terminal in-place bearing is reported but is not a mechanism pass criterion:
direction is ill-conditioned once both endpoints lie within the navigation
success neighborhood. No corresponding runtime switch or stopping rule is
added. Closed-loop evaluation must still reveal whether frozen NavDP stops or
is pulled away from the goal.

### 6.6 Prospective index-40 result: failed

The index-40 sequence was evaluated once under the rule above. Before the full
run, a readout-only execution mode was added solely to avoid duplicating the
MemNav planning caches that are never read by this mechanism audit. It retains
the unchanged live LingBot aggregator/camera KV state and explicitly disables
planning. On the same index-40 prefix and the same `episode_len=1888`, full and
readout-only modes produced bit-identical depth PNGs, decoded float arrays, and
first-40 scale receipts at frames 40, 127, and 255. Thus this is a memory-use
optimization, not a different depth model.

| Execution-equivalence artifact | SHA-256 |
|---|---|
| full mode | `d1707cf616ad01fd7dd6f5c2d990b501fdfb9a2903ef1f35098b9fab16828cee` |
| readout-only mode | `a8758b84973d4a3736a92bd956e7608437717a4ff548d22631fc7279f25beeb8` |

The dense return remained physically valid: 194 frames, 99 translated route
steps, 94 in-place turns, maximum turn `15 deg`, maximum translation
`0.337 m`, and no edge with simultaneous turn and translation.

| Dense-return local motion | Result | Frozen criterion |
|---|---:|---:|
| PnP / local-motion chain | `193/193` / `193/193` | complete |
| strict CEC certificate | `184/193` | descriptive only |
| predicted / scorer path length | `29.862 / 31.709 m` | relative error `5.82%` (pass) |
| translated step direction within 30 deg | `99/99` | descriptive local evidence |
| translated step direction median / P90 | `0.857 / 3.73 deg` | descriptive local evidence |
| final integrated query-position error | **`2.573 m`** | at most `1.0 m` (**fail**) |
| in-place false translation median / P95 | `1.29 / 3.92 mm` | descriptive |
| per-step yaw error median / P90 | `0.052 / 0.500 deg` | descriptive |

The actual online-A history was not the failing component. From authorized
anchor 30 to the tail, all `1,653/1,653` local edges were valid (`552` exact
RGB identity edges and `1,101` visual edges). The reconstructed historical
route was `29.662 m` versus `32.151 m` for the scorer, a `7.74%` length error,
and its endpoint error was `0.401 m`. Both satisfy their frozen criteria.

The cumulative path-budget projection repaired the old per-step freeze and
reached `29.662/29.662 m` while remaining finite, monotone, and bounded by
cumulative visual travel. It nevertheless failed the directional criterion:

| Route-bearing readout | Per-step budget | Cumulative budget | Frozen criterion |
|---|---:|---:|---:|
| translated bearings within 30 deg | `40/99` | **`57/99`** | at least `80%` (**fail**) |
| translated median error | `57.77 deg` | **`22.45 deg`** | at most `20 deg` (**fail**) |
| translated P90 error | `140.91 deg` | `103.84 deg` | descriptive |
| median / final progress-fraction error | -- | `0.0088 / 0.0138` | descriptive |
| controller residual norm | -- | exactly `2.5 m` for all 194 readouts | pass |

This separates the failure modes. Arc-length phase was accurate, and every
individual translated step pointed within 30 degrees of its scorer motion.
However, integrating small translation and yaw errors over `31.7 m` produced
`2.57 m` query-position drift. The cumulative route projection therefore knew
approximately *where along the route* the query was, but the lookahead chord
was still computed from a laterally drifting query pose. Position error was
`0.534 m` at the median and `2.570 m` at P90; cross-track error was `0.914 m`
at the median and `2.641 m` at P90. In the final route quartile, `0/27`
translated bearings were within 30 degrees. This is accumulated monocular
visual-odometry drift, not failure of adjacent matching, historical route
construction, or cumulative progress.

Sealed result artifacts:

```text
.diagnostics/mono_adjacent_motion_20260903/
  dense_reverse_index40_v2/query_set.json
    SHA-256 8f9f8a39327c4c5fda3c1faa06e2f55c72e682d040604201c29630135bf11cea
  index40_dense_v2_attempt1/adjacent_motion_audit.json
    SHA-256 956bea58d1f7669015e3bbb7c92c521e15e8ba427cf25ac62037e783ab8e64b9
  route_compass_index40_v1_attempt1/monocular_route_compass_audit.json
    SHA-256 8066329817eb21e4c929628bb7f19fe5b22d3d419ea1eee1a42295e5cb87262f
  path_budgeted_index40_v1/path_budgeted_route_replay.json
    SHA-256 37e2a195c874d2521d476e629c966d4eb2af436bf3042fba772862bed7060079
```

Decision: **reject this route-compass formulation and do not run closed-loop
SR**. The index-40 thresholds are not relaxed, and this row is now consumed;
it cannot be reused as confirmation for a revised method.

## 7. Experiment order

1. Generate the index-40 dense physical return using the already frozen
   `0.30 m` translation spacing and `15 deg` maximum turn increment.
2. Run the same local-motion, actual-history route, and cumulative path-budget
   audits once on index 40. No threshold or architecture change is permitted
   after reading that output.
3. Only if that prospective gate passes, integrate the same receipt into the
   causal writer and query stream.
4. Run one SR-hidden runtime gate that verifies information boundaries and
   the cumulative path-budget invariant.
5. Only then freeze a paired endpoint-bearing versus route-bearing NavDP
   evaluation.

No new HPC closed-loop job is authorized by the index-16 result alone.

Execution outcome: steps 1--2 completed. Because the prospective gate failed,
steps 3--5 are prohibited for this formulation.
