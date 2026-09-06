# Long-range Revisit complete audit

Date: 2026-09-04  
Scope: CEC long-range development evidence only  
Status: scale issue largely resolved; route-tangent mechanism positive but not confirmed; a U-turn is a necessary interface repair for a subset, not a complete long-range solution


## 1. Executive conclusion

The current evidence does **not** support the explanation that long-range
Revisit fails mainly because monocular depth lacks metric scale. The frozen
first-40 camera-height receipt recovers range accurately enough for the tested
short/mid-range queries, and passing the resulting metric magnitude to NavDP
does not improve closed-loop success over the fixed 2.5 m conditioning radius.

The long-range 20--30 m failure has two later and largely independent causes:

1. **route-state observability and accumulation:** the current causal visual
   route is an all-edge chain; `14/23` runs stop when one adjacent motion
   witness fails, and valid chains can still accumulate enough error to finish
   the route coordinate several metres before the true goal;
2. **controller support:** every initialized reverse-route tangent begins in
   the rear half-plane, while frozen NavDP clips negative forward PointGoal
   coordinates. This can collapse a valid near-180-degree token almost to
   zero.

A bounded U-turn directly repairs the second cause. It does not reconstruct a
missing visual edge, reset accumulated route-state error, or provide the
return-facing observations absent from a forward traversal. It must therefore
be evaluated as an interface mechanism, not presented as the solution to all
long-range failures.

The canonical paper result remains the high-support 2--9 m regime. The
20--30 m study is a useful pressure test and a precise limitation unless a new
continuous route-state estimator is confirmed on a fresh population.

## 2. Evidence ledger

### 2.1 Short/mid-range results that are already strong

| Population | CEC Revisit result | Interpretation |
|---|---:|---|
| Final14 | `20/21` | near-saturated supported Revisit |
| HM3D full-mono | `25/28` | external-scene full-mono transfer |
| Fresh160 supported Revisit | `112/120` | high-support, large internal population |

These populations are mostly within approximately `2--9 m`. In this regime,
one certified endpoint bearing plus the unchanged ImageGoal is usually enough
for NavDP to act as a local closed-loop controller.

### 2.2 Metric-scale evidence

The first-40 height-calibrated readout was audited on 28 actual-mono histories:

- first-takeover range MAE: `0.361 m`;
- first-takeover median absolute error: `0.254 m`;
- median within-query correlation with evaluator-only ground-truth distance:
  `0.996`;
- final-readout median metric error: `0.195 m`;
- `26/28` final readouts were within `0.5 m`.

The corresponding same-population closed-loop comparison was:

| PointGoal payload | Success |
|---|---:|
| unit bearing at fixed 2.5 m | `25/28` |
| height-scaled full metric displacement | `24/28` |

Paired difference: `+1/-2`, exact McNemar `p=1.0`; cluster interval
`[-16.0,+7.69] pp`. None of the 479 metric takeovers reached NavDP's 10 m
input cap.

**Verdict:** scale has useful geometric information, but magnitude is not the
missing control variable. The 2.5 m vector is a policy-conditioning radius,
not a command to drive along a straight 2.5 m segment. NavDP replans every
eight actions, so a normalized local direction preserves the signal while
keeping the frozen policy in its familiar PointGoal range.

### 2.3 Fresh same-floor long-range result

Frozen population: 23 result-blind causal-RGB histories, eight HM3D scene
clusters, approximately `20.37--24.66 m` initial geodesic distance. The history
is a controlled causal geodesic survey, not an actual NavDP Goal-A rollout, so
this is a mechanism/pressure population rather than the main end-to-end table.

| Arm | Success | Mean final 3-D distance | Mean realized path |
|---|---:|---:|---:|
| mono native | `0/23` | `16.46 m` | `18.50 m` |
| CEC endpoint bearing | `0/23` | `13.24 m` | `10.81 m` |
| CEC route tangent | `4/23` | `13.05 m` | `9.23 m` |

Route tangent versus endpoint: `+4/-0`, exact McNemar `p=0.125`, clustered
risk-difference interval `[+5.56,+31.82] pp`. The direction is encouraging,
but the preregistered significance gate was not met.

Every one of the 23 queries had the same accepted initial CEC target proof.
Consequently this experiment does not diagnose DINO retrieval or the initial
certificate. Its failures occur after target authorization.

## 3. Failure tree

```text
23/23 accepted target identity and initialized a Revisit request
|
+-- 14/23 typed geometry-stream failures
|   +-- 13 PnP status not OK
|   `--  1 fewer than 16 inliers
|
`-- 9/23 retained a live route stream
    +-- 4/9 success
    `-- 5/9 stuck
        +-- 2 persistent rear-token deadlocks with near-zero travel
        `-- 3 turned/moved, but route execution still failed
```

The 14 geometry stops are concentrated on low-parallax, turn-heavy intervals.
The current tracker is a series system: one bad edge invalidates the entire
remaining route. Deleting Fundamental-MAGSAC is not a fix. On the exact 14
failed edges, direct PnP was both valid and accurate only `1/14`; one unfiltered
estimate had `155.6 deg` yaw error.

Denser direct PnP removed the live stop in two selected histories but recovered
`0/2` successes. In one of them, the internal route coordinate reached 100%
while the true goal remained `6.63 m` away, after which cross-track error grew
from `3.15 m` to `7.04 m`. This is direct evidence of accumulated route-state
error rather than missing metric scale.

## 4. Exact U-turn attribution

### 4.1 Why a U-turn is technically justified

Frozen NavDP applies:

```python
clip_goals = goals.clip(-10, 10)
clip_goals[:, 0] = np.clip(clip_goals[:, 0], 0, 10)
```

Thus a rear unit bearing `[forward < 0, left]` loses its negative forward
component. At exactly 180 degrees, the lateral component is also almost zero,
so the nominal 2.5 m token can collapse to a near-zero vector.

All nine route-tangent runs that retained a live geometry stream began with a
rear tangent:

| Group | N | first heading range | median first post-clip norm | median max route progress |
|---|---:|---:|---:|---:|
| success | 4 | `169.4--180.0 deg` | `0.224 m` | `95.35%` |
| stuck | 5 | `177.1--179.5 deg` | `0.043 m` | approximately `0%` |

This establishes an input-support mismatch. A bounded in-place turn followed
by a fresh observation would place the same authenticated tangent in NavDP's
forward support without changing target identity, route, radius, or policy
weights.

### 4.2 Why a U-turn is not sufficient

One successful query began at `179.97 deg`, where NavDP's post-clip token norm
was only `0.0013 m`. Therefore token collapse does not deterministically imply
failure: the ImageGoal branch and diffusion sampling can sometimes initiate a
turn anyway.

The five no-stop stuck histories separate further:

| index | scene | first heading | path | max route progress | interpretation |
|---:|---|---:|---:|---:|---|
| 3 | `b28CWbpQvor` | `178.5 deg` | `2.34 m` | `0%` | weak escape, no route progress |
| 4 | `dVW2D7TDctW` | `179.0 deg` | `7.33 m` | `27.8%` | turned and moved; later route failure |
| 8 | `BHXhpBwSMLh` | `177.1 deg` | `0.58 m` | `0%` | persistent rear-token deadlock |
| 10 | `b28CWbpQvor` | `179.5 deg` | `7.70 m` | `23.8%` | turned and moved; later route failure |
| 21 | `rsggHU7g7dh` | `179.3 deg` | `0.57 m` | approximately `0%` | persistent rear-token deadlock |

Indices 8 and 21 are the clearest U-turn targets: 90--100% of their subsequent
requested headings remain behind and their total translation is below 0.6 m.
Indices 3, 4, and 10 are not explained by the initial U-turn alone.

Even an optimistic rescue of both clear deadlocks would raise the observed
route-tangent count only from `4/23` to `6/23`, while leaving all 14 geometry
failures untouched. This bound is descriptive, not a counterfactual result.

### 4.3 Why the existing bounded-turn code cannot be attached blindly

`cec_initial_bearing_alignment=first_certified_bounded` already implements
at-most-30-degree zero-translation turns, a fresh observation after each turn,
and forced replanning. It was built for the proof-carrying controller
portability path.

The long-range route tracker, however, estimates every route-state update from
adjacent visual PnP. Feeding a six-step pure rotation through that same chain
would create precisely the low-parallax geometry that already fails. A valid
long-range turn must therefore be atomic with respect to route progress:

1. derive the signed turn only from the authenticated route tangent;
2. execute bounded zero-translation turns;
3. hold the route position fixed during the maneuver;
4. update orientation from the issued/verified turn receipt rather than
   interpreting pure rotation as visual translation;
5. take a fresh RGB observation and recompute the forward tangent before any
   translation.

This is a controller-interface mechanism. If implemented with simulator pose
deltas it would be privileged and invalid for the mono claim; a deployable
version must bind the commanded turn plus an onboard completion receipt, or
use a visual/IMU yaw estimate available on the robot.

## 5. Why short range succeeds while long range fails

Short/mid-range CEC asks NavDP to solve a local problem. Supported Revisit
queries retain high co-visibility with an online historical anchor, and an
endpoint direction often agrees with the first feasible path segment. The
frozen ImageGoal branch still supplies appearance and local obstacle context;
the PointGoal contributes only a directional prior. Replanning corrects small
errors before they accumulate.

At 20--30 m, endpoint direction and first route action are often different.
Doorways, corners, and homotopy become essential. Route tangent supplies this
missing topology and explains the four gains, but it also introduces a state
variable that must remain valid over tens or hundreds of local updates. Camera
height fixes the monocular similarity gauge; it cannot remove compounded yaw,
translation, correspondence, or route-coordinate error:

```text
metric scale receipt  -> fixes metres per relative-depth unit
route pose product    -> still composes per-edge translation/yaw errors
rear PointGoal clip   -> still discards a valid backward control request
```

The short-range method therefore does not contradict the long-range failure.
It avoids the two new burdens: persistent route state and repeated reversal
through a forward-only controller interface.

## 6. Recommended long-range architecture

The smallest principled extension is a proof-conditioned visual route filter,
not another distance/radius sweep:

```text
initial CEC proof fixes target identity and one historical route
        |
action/motion prediction maintains a distribution over route coordinate
        |
local visual witnesses correct that distribution when observable
        |
posterior route tangent provides one unit bearing
        |
rear tangent -> bounded atomic alignment -> fresh observation
        |
unchanged frozen NavDP executes the forward local request
```

The state should be a posterior over route coordinate, not an unbounded product
of accepted pairwise poses. Missing one visual edge must increase uncertainty,
not silently invent a pose and not destroy the already authorized target.
Periodic evidence should come from a local temporal window or explicit
return-facing memory; the known reverse-view diagnostic (`top-1 0/12`, `top-8
4/12` within 1 m at 180 degrees) shows that the current forward-only frame
addresses are not enough for reliable global relocalization on the return path.

This extension adds real methodological content, but it is a new long-range
method and requires a fresh population. It should not be folded into the
canonical 2--9 m result after inspecting these 23 outcomes.

## 7. Experimental decision

### Admissible now

- keep the current fixed-2.5 m unit-bearing CEC as the canonical method;
- report the 20--30 m route-tangent result as a pressure test/limitation;
- use the consumed population only for a clearly labelled atomic U-turn and
  route-filter mechanism test;
- require another frozen, result-blind population before any navigation claim.

### Not admissible

- claim that height calibration solved long-range navigation;
- replace the certificate with a weaker threshold;
- delete Fundamental-MAGSAC globally;
- call support projection a physical U-turn;
- use evaluator pose to maintain the route while retaining a mono-only claim;
- tune by distance, stuck state, critic threshold, or observed failure index;
- report a consumed U-turn gain as a paper SR result.

## 8. Submitted proof-bound rear-alignment diagnostic

The implementation audit exposed one additional constraint: pure-rotation
frames cannot simply be fed back into the adjacent visual-PnP route clock.
The deployed mechanism therefore seals a route-specific packet, executes
at-most-30-degree zero-translation turns, binds each new frame to an issued-yaw
receipt, and rotates the existing route compass without advancing route
position or constructing visual edges during the maneuver.

An initial gate built from the later accumulated working-tree runtime stopped
before executing any turn and was discarded as an implementation-confounded
attempt.  The active retry is derived from the independently verified formal
route-tangent source bundle; its visual route estimator and servers are
byte-identical to that parent.  The frozen DAG is:

- gate `16929826` (history index 8);
- paired consumed diagnostic `16929827` (nine histories);
- analysis `16929828`;
- independent verification `16929829`.

This diagnostic has no paper-SR authority.  It can only decide whether to
construct a fresh, outcome-blind long-range alignment population.  Full audit
and source receipts are recorded in
`HM3D_LONGRANGE_REAR_ALIGNMENT_INCIDENT_AND_RETRY_20260904.md`.

## 9. Sources of truth

- `MemNavData/FIRST40_LONGRANGE_METRIC_SCALE_AUDIT_20260831.md`
- `MemNavData/LONG_RANGE_REVISIT_ROOT_CAUSE_AUDIT_20260903.md`
- `MemNavData/HM3D_LONGRANGE_ROUTE_TANGENT_FORMAL_RESULT_20260903.md`
- `MemNavData/HM3D_LONGRANGE_DENSE_QUERY_GATE_STATUS_20260903.md`
- `.diagnostics/hm3d_longrange_route_tangent_formal_20260903/result/summary.json`
- `.diagnostics/hm3d_longrange_route_tangent_formal_20260903/result/independent_verification.json`
- `.diagnostics/hm3d_longrange_route_tangent_formal_20260903/posthoc_diagnostics/failure_audit.json`
