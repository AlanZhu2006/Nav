# Long-range Revisit: continuous episodic path-field protocol

Date frozen: 2026-09-02 (Asia/Shanghai)  
Status: route-information gate passed; global LingBot curve failed the local
deployment-quality gate; local visual re-anchor gate frozen; canonical CEC is
unchanged

## 1. Why a separate extension is necessary

The canonical method localizes a supported historical goal and sends only its
unit bearing, projected to a fixed 2.5 m PointGoal, to frozen NavDP.  On the
main 2--9 m evaluation range this is close to saturated (Final14 Revisit
20/21; HM3D full-mono Revisit 25/28).  It should not be replaced merely because
an intentionally harder length stress test fails.

The consumed HM3D length stress test diagnoses a different regime:

| Revisit geodesic range | native | canonical CEC |
|---|---:|---:|
| approximately 7--10.6 m | 2/16 | 4/16 |
| approximately 20.1--24.9 m | 1/16 | 2/16 |
| approximately 30.1--39.7 m | 2/16 | 0/16 |

Across those same bins, the localized endpoint bearing is close to the direct
Euclidean endpoint ray, while that ray differs from the geodesic initial
tangent by roughly 33--42 degrees.  The missing quantity is therefore not a
longer PointGoal radius.  It is the local route tangent/homotopy along an
already traversed history.

This population and its outcomes are already consumed.  It may guide and
falsify this extension, but no experiment selected on it will be presented as
fresh confirmation.

## 2. Frozen architectural hypothesis

The extension is **not** a topological planner, graph search, second policy,
distance-gated expert, or failure rescue.  Once CEC has certified a Revisit,
the causal pose stream between the query start and certified anchor already
contains one demonstrated route.  Preserve its temporal order as a continuous
polyline:

```text
actual causal tail -> reversed historical pose curve -> certified anchor
                   -> PnP terminal goal hypothesis
```

At every planning step:

1. project the current LingBot translation monotonically onto the frozen
   curve;
2. advance 2.5 m in curve arc length, or to the terminal endpoint if less
   remains;
3. discard the resulting magnitude and express only its unit bearing in the
   current camera frame;
4. send the same fixed 2.5 m PointGoal authority as canonical CEC to the same
   frozen NavDP policy.

Formally, for ordered curve `gamma(s)`, monotone progress `s_t`, and the
already-frozen controller horizon `rho=2.5 m`:

```text
q_t = gamma(min(s_t + rho, S))
b_t = unit(R_t^T [q_t - x_t])
p_t = rho b_t
```

There is no short/long switch.  A short route naturally places `q_t` at the
terminal endpoint, reproducing an endpoint bearing.  A long route naturally
places `q_t` on the local demonstrated path.  The first-40 metric receipt is
used only to measure arc length; estimated endpoint distance never crosses the
memory-to-policy authority boundary.

## 3. Differences from the failed 2026-08-13 graph rescue

The old rescue does not test this hypothesis cleanly.  It:

- activated only after a hand-defined stuck detector;
- guessed a nearest historical start after failure rather than freezing the
  known causal tail at goal switch;
- discretized the path into 1.25 m nodes with a 0.60 m arrival switch;
- passed every different node through a later fixed-radius normalization,
  creating corner overshoot;
- mixed route-start error, pose drift, discrete switching, and controller
  effects in only five failures.

The new readout starts at the known causal tail, has no stuck trigger or
arrival threshold, and moves its reference continuously along one ordered
curve.

## 4. Authority and failure contract

- Scope: certificate-accepted Revisit only.  This is not a Novel direction
  source.
- Mixed-role canonical behavior remains unchanged: certificate rejection
  executes the exact native ImageGoal request.
- Within the accepted long-range extension there is no distance gate and no
  fallback to the endpoint chord.
- A malformed pose stream, invalid immutable scale receipt, or degenerate path
  is a geometry-stream failure and must be surfaced explicitly; it must not be
  silently rewritten into a successful method output.
- On completed route progress, the path contribution is zero and the original
  ImageGoal remains responsible for terminal visual alignment.  This is the
  terminal contract, not failure recovery.

## 5. Two-stage consumed-population mechanism gate

The sealed 48-history result bundle stores the evaluator's exact causal trace,
selected anchor, and controller receipts, but it does **not** store LingBot's
online `cam_pose` stream.  Consequently the mechanism audit is split before
any result is read.  Stage A is an information upper bound; Stage B is the
deployment gate.  Passing Stage A alone never authorizes a method claim or a
closed-loop run.

### Stage A: causal-history route-information upper bound

Run renderer-free on every certificate-accepted Revisit.  Build the ordered
tail-to-anchor curve from the saved evaluator trace and compare, at the frozen
query start:

- canonical endpoint-bearing error versus the manifest's construction-only
  geodesic initial tangent;
- evaluator-route local-bearing error versus the same tangent.

Evaluator positions are metric, so this diagnostic uses scale `1.0`; it must
not mix the evaluator frame with the PnP terminal pose or first-40 LingBot
scale.  Report paired angular changes, `<=30 deg` coverage, all
missing/invalid rows, and each distance bin separately.  Habitat/evaluator
state is analysis-only and cannot enter the runtime implementation.

The route-information gate passes only if all of the following hold:

1. median angular error improves by at least 10 degrees in at least two of
   three bins;
2. no bin's median angular error worsens by more than 5 degrees;
3. overall `<=30 deg` coverage improves by at least 20 percentage points;
4. a coordinate-gauge transform (translations multiplied by `a`, scale
   divided by `a`) changes no unit bearing beyond `1e-6` degrees;
5. evaluator-trace replay produces finite, monotone progress for every
   auditable trajectory.

Failure stops this branch.  It does not authorize a lookahead sweep, distance
gate, oracle correction, or full closed-loop run.

### Stage B: deployable LingBot geometry gate

Only after Stage A passes, replay a small frozen set through the unchanged
LingBot server and compute the same path field from information available at
runtime:

- `cam_pose[:goal_start_frame]` for the causal curve;
- the certificate-selected historical anchor;
- current `cam_pose[-1]` for monotone projection;
- the already-frozen first-40 scale receipt for arc length;
- the PnP terminal witness only in the same LingBot coordinate gauge.

No evaluator pose, Habitat path, geodesic tangent, role label, or nearest-node
oracle may enter this readout.  Stage B must pass schema, finiteness, gauge
invariance, temporal-anchor, and replay-monotonicity checks before any
development closed-loop comparison is launched.

## 6. Closed-loop sequence after a pass

1. Complete Stage B on three predetermined histories: one
   short/straight, one medium/single-turn, one long/multi-turn.  Inspect only
   contract receipts and crashes first.
2. Run a development closed-loop comparison on the consumed length population:
   canonical fixed endpoint bearing versus continuous path-field bearing, with
   identical certificate decisions, ImageGoal, policy, seed, and budget.
3. If direction and stability are favorable, freeze a new disjoint long-range
   HM3D population before reading outcomes.  Only that population can support
   a paper claim.

The formal comparison is paired Revisit SR and SPL.  Secondary diagnostics are
path efficiency, collision/stagnation rate, progress monotonicity, cross-track
distance, and interventions per episode.  No arm may use Habitat paths or
runtime Novel/Revisit labels.

## 7. Claim boundary

Until a disjoint confirmation succeeds, the paper's main method remains the
minimal fixed-bearing CEC.  This branch may be described only as an
architecture-level solution under development for long, non-convex returns.

## 8. Locally re-anchored redesign after the Stage-B failure

The global LingBot curve is not sent to closed loop.  RGB replay showed that
it completes the approximately 11 m route, but loses alignment after 9--18 m
on the medium and long routes.  The historical sequence is therefore retained
as a temporal route address, while metric geometry is made local:

```text
current RGB -> DINO historical address -> LightGlue/PnP certificate
            -> current pose re-anchored in a local historical frame
            -> 2.5 m arc-ahead route tangent -> unit bearing -> frozen NavDP
```

Before implementing control, freeze a 12-query RGB-only mechanism gate on the
medium history (population index 16).  Construction renders a midpoint between
two historical floor poses separated by eight frames, adds the same frozen
0.5 m camera mounting height as the source survey, and applies alternating
+/-15 deg yaw.  Runtime receives only those RGB images and the sealed causal
RGB history; the construction pose and expected frame are analysis-only.

The gate passes only if all conditions hold:

1. strict CEC certificate accepts at least 9/12 queries;
2. at least 9/12 accepted anchors are within 1.0 m of the construction pose;
3. accepted-anchor median position error is at most 0.75 m;
4. in descending return order there is at most one anchor jump forward by
   more than 16 frames;
5. each temporal third contains at least two accepted queries;
6. every accepted proof has a finite PnP pose witness.

These criteria are frozen before runtime results are read.  A pass authorizes
implementation of local re-anchoring, not a navigation or paper claim.  A
failure stops the long-range extension and preserves canonical fixed-bearing
CEC.

Construction repair note: v1 accidentally passed the trace's floor position
directly to the optical-center renderer, omitting the 0.5 m camera-height
offset.  Its 10/12 result is invalidated in full.  The unchanged criteria above
apply to the corrected v2 images; v1 outcomes are not pooled or cited.

## 9. Proof-once route-coordinate gate

If corrected v2 local PnP is viable, replace repeated global projection with
one unified route-coordinate update.  The target's strict CEC certificate is
evaluated once.  It freezes the route and owns all memory authority.  Every
later observation is restricted to the next 2.5 m arc interval of that same
route, exactly matching the controller's maximum residual horizon:

```text
s_t = argmin_{s in [s_(t-1), s_(t-1)+2.5m]}
      || gamma(s) - x_t^PnP ||
b_t = unit(R_t^T [gamma(min(s_t+2.5m,S)) - x_t^PnP])
```

Local PnP is a state observation, not another authority decision.  There is no
distance classifier, endpoint fallback, native fallback, stuck trigger, graph
rescue, or alternate policy.  If a local pose witness cannot be computed, the
geometry stream fails explicitly and execution stops.

The frozen local gate uses 24 corrected same-height RGB queries on history 16,
spaced densely enough that consecutive observations remain inside one control
horizon.  It passes only if:

1. the target strict certificate is accepted exactly once;
2. all 24 observations update the route coordinate without geometry failure;
3. every selected anchor is within 1.0 m of the construction pose and median
   error is at most 0.75 m;
4. progress is monotone and final remaining route is at most 5.0 m;
5. every active update emits a finite bearing;
6. all receipts confirm no fallback, distance gate, local control authority,
   evaluator pose, or runtime role input.

This is still a mechanism gate, not SR.  Only a pass authorizes a NavDP
closed-loop smoke.
