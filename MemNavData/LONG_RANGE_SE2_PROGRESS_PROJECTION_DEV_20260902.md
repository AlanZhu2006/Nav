# Long-range Revisit: local SE(2) progress projection

Date: 2026-09-02 (Asia/Shanghai)  
Status: **privileged diagnostic only; both queued runtime gates cancelled before execution; not a deployable method**

> Superseding audit: the first counterfactual used Habitat pose differences as
> a local-odometry proxy, and its original unrestricted nearest-route
> projection could jump across spatially adjacent later route segments.  It
> therefore does not establish a monocular navigation improvement.  The
> deployable follow-up is frozen separately in
> `MONOCULAR_HEIGHT_CALIBRATED_ROUTE_PROGRESS_PROTOCOL_20260902.md`.

## 1. Why the scalar action coordinate was insufficient

The frozen action-coordinate experiment advances route progress as

```text
s <- min(route_extent, s + executed_translation)
```

This is invariant to monocular scale and turn-only frames, but it discards the
direction of executed translation.  Lateral motion, collision-induced navmesh
projection, and loops therefore all count as forward progress.  In the first
28 completed development traces, the scalar coordinate reached 100% on every
failed query (21/21), even though their final goal distances remained large.

Simply integrating the existing scalar translation and yaw under the nominal
``yaw then forward`` action model did not solve the problem.  On the same
traces it still saturated on 20/21 failures.  The historical route integrated
accurately, but the query rollout accumulated a median 3.88 m position error:
after collision handling, the realized navmesh displacement can differ sharply
from the nominal forward command.

## 2. Upgraded state estimator

After the unchanged initial CEC proof chooses a historical anchor, the new
development arm uses one frame-bound local motion receipt:

```text
delta_t = (forward_t, left_t, delta_yaw_t)
```

The translation is expressed in the previous body frame.  It is a relative
executor/odometry receipt, not a goal-relative or world-frame pose exposed to
the policy.  The method then performs:

```text
certified historical anchor
        -> invert historical local SE(2) receipts
        -> one metric route in the query-start body frame

live local SE(2) receipts
        -> integrate current 2-D pose
        -> monotone orthogonal projection onto the certified route
        -> route point 2.5 m ahead
        -> unit bearing only
        -> unchanged frozen NavDP
```

The bearing is computed from the estimated **live** position to the lookahead,
not from a sampled route point to another route point.  Consequently, an
off-route rollout receives a cross-track correction.  Total travelled distance
cannot by itself advance route state.

There is still one route and one continuous rule.  The estimator contains no:

- distance bin or short/long switch;
- visual update threshold after authorization;
- stuck detector or graph rescue;
- endpoint controller;
- native fallback after CEC acceptance;
- metricized LingBot translation.

Normal CEC rejection before authorization continues to execute the unchanged
native ImageGoal request.  A malformed/missing local receipt after acceptance
is an explicit geometry-stream failure, not permission to change controller.

## 3. Counterfactual diagnostic

Input population: 28 already-completed action-coordinate Revisit query traces
copied from the consumed HM3D length-development run.  This audit replays the
same physical trajectories; it does not resample NavDP and cannot measure a new
navigation SR.

| State estimate on the same traces | Saturated, all 28 | Saturated among 21 failures | Median final cross-track |
|---|---:|---:|---:|
| Original scalar progress | 25 | 21 | n/a |
| Nominal yaw+forward 2-D integration | 24 | 20 | 14.16 m |
| Realized local delta-SE(2) + projection | 2 | 2 | 1.46 m |

Additional scorer-only checks for the realized local receipt:

- median historical-route integration error: `7.63e-15 m`;
- median query-pose integration error: `4.09e-15 m`;
- median final projected route fraction: `0.836`, versus `1.000` for the
  original scalar coordinate.

Artifact:

```text
.diagnostics/se2_projection_replay_20260902/
  se2_local_odometry_counterfactual_audit.json
```

The simulator x/z poses in the completed trace were used both to synthesize
the relative receipt and to score it.  They were never passed as an absolute
world pose to `SE2ProjectedRouteCompass`, but their differences are still
privileged metric motion.  The current Habitat bridge labels this honestly as
`habitat_pose_difference_odometry_proxy_v1`.  It cannot support a monocular
claim and is retained only as an upper-bound diagnostic.

An additional audit then found a second problem in the original v1
projection.  It searched the complete remaining polyline for the nearest
point.  On self-intersecting or spatially adjacent route segments, this could
advance the route coordinate much farther than the robot moved.  Such an
excess jump occurred in `18/28` replayed traces; the largest jump was
`15.82 m`.  The favorable `2/21` failed-query saturation count above is
therefore not a trustworthy method effect.

Projection schema v2 now enforces the causal non-expansion invariant

```text
previous_s <= s_t <= previous_s + ||delta_position_t||.
```

This repairs the state-machine defect, but it does not solve the privileged
motion source.  No closed-loop result is claimed from either version.

## 4. Runtime implementation

New implementation:

- `MemNavData/se2_projected_route_compass.py`;
- `MemNavData/audit_se2_progress_projection.py`;
- guidance mode `se2_route_compass` in
  `NavDP/baselines/memnav/policy_agent.py`;
- frame-bound wire receipt `frame_bound_local_se2_v1` in
  `MemNavData/eval_2leg_habitat.py` and
  `NavDP/baselines/memnav/memnav_server.py`.

The legacy `action_coordinate_compass` implementation and its currently
running 48-history experiment remain unchanged.  The new mode has a separate
cache, schema, diagnostic namespace, and command-line value.

Validation completed locally for the diagnostic component:

- 16 focused SE(2)/gate tests passed, including a self-intersection case;
- 96 combined route, policy, replay, adapter, and graph tests passed;
- Habitat evaluator `--contract_dry_run` accepted the complete full-mono CEC
  configuration for `se2_route_compass`;
- malformed or mixed receipt models fail explicitly.

## 5. Final decision and cancelled gates

The valid part of the diagnostic supports only:

> If accurate local metric motion were available, two-dimensional route state
> would carry information that the scalar travelled-distance coordinate
> discards.

It does **not** show that monocular RGB can supply that motion, nor that the
changed bearings improve closed-loop SR.  Two queued SR-hidden gates
(`16792872` and `16793028`) were cancelled before GPU execution and produced
no method outcome.  The odometry-proxy branch will not be resubmitted as the
paper method.

The next allowed experiment is the offline monocular motion gate specified in
`MONOCULAR_HEIGHT_CALIBRATED_ROUTE_PROGRESS_PROTOCOL_20260902.md`.  It must
verify that:

1. adjacent RGB frames plus height-scaled monocular depth recover local
   translation and yaw without simulator motion;
2. turn-only frames do not hallucinate systematic translation;
3. cumulative progress and route bearing remain useful at medium and long
   spans;
4. projection is finite, monotone, and non-expansive;
5. no role label, evaluator pose, distance regime, endpoint controller, or
   native fallback reaches the authorized route reader.

Only after that gate passes should a paired closed-loop development comparison
be frozen.  The ongoing scalar 48-history job remains unchanged and is not
stopped or reinterpreted by this post-hoc development.
