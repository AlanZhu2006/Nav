# Certified relocalization closed-loop protocol v1

Date frozen: 2026-08-12 (Asia/Shanghai)

Status: runtime/transport preflight passed; effectiveness outcomes unread.

## Question

Can causal episodic memory support an unknown ImageGoal with a geometric
certificate, expose only a scale-free bearing to the already validated
NavDP mixed controller, and improve closed-loop success over native ImageGoal
NavDP?

This is not a forced Novel/Revisit classifier. The deployment states are:

1. a history hypothesis passes the complete atomic certificate: supply one
   fixed-radius bearing residual;
2. evidence is absent, malformed, or rejected: `Unknown/unsupported`, execute
   native ImageGoal NavDP on that same planning step.

Failure to localize is never reported as proof that the goal is semantically
Novel.

## Runtime correction discovered before closed loop

The frozen v2 offline confirmation established an eight-true-positive,
zero-false-positive geometry certificate under an externally audited causal
metric-scale artifact. A real runtime smoke on
`rPc6DW4iMge/episode_0001` then exposed a boundary error that the image
certificate cannot detect:

- PnP bearing: `174.28°`; GT bearing: `174.61°`;
- runtime ground scale: `4.100`; scale consistent with GT distance: about
  `1.75`;
- resulting metric request: `15.29 m`; GT relative distance: `6.54 m`.

Therefore the v2 pixel certificate does **not** certify online monocular
metric scale. No closed-loop job may use the experimental
`certified_metric_v2` interface.

The accepted v2 confirmation rows independently support the narrower signal:
all `8/8` accepted estimates have bearing error below `4.45°` (median
`2.35°`). The already frozen B0 controller experiment supplies the action
contract: fixed `2.5 m` bearing and metric geometry each achieved `20/26`
conditional-B successes (`+1/-1`, exact McNemar `p=1.0`). Thus the runtime
method removes scale before effectiveness evaluation:

```text
DINO causal top-8 proposal
  -> SuperPoint + LightGlue + Fundamental-MAGSAC ranking
  -> selected-anchor causal LingBot depth + LightGlue PnP
  -> atomic v2 certificate
       reject -> native ImageGoal NavDP
       accept -> normalize [forward,left] in LingBot raw units
              -> fixed 2.5 m verified_bearing_v1
              -> existing ImageGoal+PointGoal NavDP mixed controller
```

Runtime schema v3 explicitly reports
`metric_distance_certified=false`, `pointgoal_units=lingbot_raw_direction_only`,
and never calls ground-scale recovery.

## Frozen mechanics

- Causal candidates: raw-DINO top 8, minimum anchor 8, temporal gap 4.
- Label-free candidate order: Fundamental inliers, query grid coverage, query
  hull coverage, median LightGlue score, DINO cosine, earlier anchor.
- LightGlue: SuperPoint, 2,048 keypoints, Fundamental-MAGSAC threshold 1.5 px.
- Atomic certificate on the selected PnP hypothesis:
  - at least 16 PnP inliers;
  - query inlier hull coverage at least 5%;
  - reference inlier hull coverage at least 5%;
  - reprojection RMSE at most 2 px.
- Accepted absolute goal pose is computed once per goal and cached. The current
  relative bearing is recomputed after motion. Candidate membership cannot
  change after the goal's causal start frame.
- If the first causal query has no eligible history candidate, that empty
  shortlist is itself frozen as one cached abstention; later goal-session
  frames cannot turn it into a self-match or trigger repeated localization.
- Any exception, changed candidate contract, non-finite/zero vector, wrong
  units, or failed certificate abstains to native ImageGoal NavDP.
- No radius, threshold, controller, or keypoint sweep is permitted.

## Local transport evidence (not SR)

With the exact gatecurr600 MemNav checkpoint and official NavDP checkpoint on
one RTX 4090:

- accepted Revisit smoke: certificate accepted rank-2 DINO anchor, 217 PnP
  inliers, 28.5%/26.1% query/reference coverage, 1.07 px RMSE; first call
  2.09 s, cached call 0.15 ms; mixed NavDP returned 16 candidates and executed
  one horizon;
- moving one causal frame preserved the immutable shortlist and recomputed the
  current-relative bearing from the cached absolute pose;
- reset erased the cache and a goal queried without a causal probe was
  rejected;
- Novel-A first plan had no causal history, automatically abstained, called
  native ImageGoal NavDP, returned 16 candidates, and executed normally;
- The original transport smoke plus the final cached-empty lifecycle test
  passed; the complete frozen submission preflight now contains 63 tests,
  including a synthetic 20-scene/160-episode four-arm summary audit.

## Evaluation universe and role

Use the immutable 160 fresh episodes generated in the completed
`fresh160_v3_attempt600_20260811T2000` run:

- 20 MP3D scene clusters, eight episodes each;
- no overlap with 50 training scenes;
- manifest SHA256
  `8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`;
- development, final-reserved, and blind data are not read.

This pool has already been consumed by the known-Revisit direct-vs-geometry
architecture decision. It is appropriate for a statistically powered internal
closed-loop gate for the newly frozen certificate, but it is not a fresh-scene
or paper-final confirmation. Results cannot be used to retune this method.

## Four arms and causal pairing

Each scene task starts one MemNav process and one NavDP process. It samples
Novel-A exactly once per episode and stores the full RGB/pose/action/seed trace.
Every B arm replays that identical A trace into both long and short memory.

1. `native`: official ImageGoal NavDP;
2. `geometry_router`: existing SIFT/RANSAC hard router + legacy metric mixed
   controller;
3. `known_revisit_direct`: benchmark-role upper reference, raw-DINO top-1 +
   legacy metric mixed controller; this is not deployable when goal kind is
   unknown;
4. `certified_relocalization`: the scale-free runtime method above.

The four arms run through a four-row Williams design repeated across 20
scenes, so every arm occurs five times in every ordinal position and immediate
carry-over is balanced. All arms use the same episode seed and deterministic
per-plan diffusion seed. Primary and paired effects may only be computed from
the same scene task after checking trace SHA and Goal-A equality.

## Outcomes and statistics

Report for every arm:

- Novel-A SR (must be identical by construction);
- Revisit-B SR conditional on shared A success;
- joint `A and B` SR, SPL, and final distance;
- certified takeover/fallback episodes, selected DINO rank, PnP evidence, and
  first-call latency.

Frozen contrasts:

1. primary: `certified_relocalization - native`;
2. secondary: `certified_relocalization - geometry_router`;
3. ceiling gap: `certified_relocalization - known_revisit_direct`;
4. direct/native and geometry/native are integrity replications.

For joint and conditional-B outcomes, report the paired risk difference,
gain/loss counts, two-sided exact McNemar test, and 100,000-resample
scene-cluster bootstrap 95% interval. The primary branch is:

- positive difference, `p < 0.05`, and cluster CI lower bound above zero:
  `certified_router_has_closed_loop_value_seek_fresh_scene_open_set_confirmation`;
- significant negative difference with cluster CI upper bound below zero:
  `reject_certified_router_retain_known_role_system`;
- otherwise:
  `inconclusive_do_not_retune_on_consumed_pool`.

No branch authorizes blind evaluation or a paper-final claim.
