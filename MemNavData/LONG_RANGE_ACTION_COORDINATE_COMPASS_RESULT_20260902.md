# Long-range Revisit action-coordinate compass result

Date: 2026-09-02 (Asia/Shanghai)  
Protocol: `LONG_RANGE_ACTION_COORDINATE_COMPASS_PROTOCOL_20260902.md`  
Frozen protocol SHA-256: `d5b004825f5af2c6e3faa8485fdc605ceda939229c0a2e27db27a994fd4d1305`  
Status: **SR-hidden runtime gate passed; frozen paired closed-loop evaluation is running under an exact infrastructure retry**

## 1. Decision

The preregistered index-40 holdout passed every route-coordinate and bearing
criterion without changing the method or thresholds:

| Frozen criterion | Required | Observed |
|---|---:|---:|
| translated readouts within 1 m | at least 90% | 99/99 (100%) |
| median position error | at most 0.50 m | 0.145 m |
| final position error | at most 1.0 m | 0.293 m |
| bearings within 30 deg | at least 80% | 99/99 (100%) |
| median bearing error | at most 20 deg | 8.77 deg |
| bearing P90 | descriptive only | 14.96 deg |
| monotone finite progress | required | pass |
| PointGoal norm | exactly 2.5 m | pass |
| post-authorization visual gate | forbidden | absent |
| endpoint/native fallback | forbidden | absent |

The tested route had a 32.15 m executor-action extent. The existing CEC proof
selected historical anchor 30. Runtime consumed the causal LingBot route shape
and scalar executor translation/yaw receipts. Absolute survey positions were
kept outside the method and read only by the scorer.

Result artifact:

```text
.diagnostics/long_range_path_field_20260902/
  action_coordinate_route_compass_index40_holdout_v1/
    action_coordinate_route_compass_audit.json
SHA-256 09934b9075b8d9a52ee9fe5c8b2e0b1b2199dc19921f4370858b88d38e134376
```

## 2. Development routes

The same frozen computation was also audited on the three already-consumed
diagnostic routes. Their role is mechanism development, not confirmation.

| Manifest index | Route extent | CEC anchor | Position within 1 m | Median / final position error | Bearing within 30 deg | Median / P90 bearing error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.49 m | 24 | 31/31 | 0.020 / 0.020 m | 31/31 | 2.01 / 2.49 deg |
| 16 | 23.41 m | 26 | 72/72 | 0.183 / 0.234 m | 60/72 | 15.16 / 36.62 deg |
| 32 | 33.07 m | 25 | 102/102 | 0.217 / 0.308 m | 102/102 | 4.44 / 10.15 deg |

Across these three development routes and the prospective route, one method
covered approximately 10--33 m without a distance label, a short/long switch,
metric scaling of LingBot translation, or a route-localization threshold.

## 3. What changed relative to the rejected designs

The earlier designs assigned progress using either a frame index, a repeated
single-image match, or accumulated LingBot translation. All three coordinates
were unstable for a different reason:

- frame density varies with the number of turn-only observations;
- opposite-facing return views are not reliably addressable by single-frame
  DINO/geometry (`0/12` top-1 and `4/12` top-8 within 1 m);
- monocular pose integration under return-facing views underestimates long
  progress, and changing first-40 to first-64 scale does not repair it.

The action-coordinate compass separates the two quantities that those designs
had conflated:

```text
executor action arc                 -> where along the route
causal LingBot route, arbitrary scale -> which local direction
executor yaw receipt                -> direction in the current body frame
```

Only the direction is normalized and exposed to frozen NavDP. The method is
therefore invariant to a global rescaling of the monocular route and to extra
turn-only frames.

## 4. Chronology audit

The method, prospective index, source hashes, and pass gate were frozen before
the index-40 action-coordinate output was computed. While obtaining the
already-frozen CEC anchor required by the method, a command printed the same
row's previous endpoint/path-field diagnostic as well as the anchor. This
violated the protocol sentence that the previous long-range outcome would
remain unread until sealing, although it did **not** expose the new method's
output and no method parameter, sample, or gate was changed afterward.

Accordingly, this result is reported as a prospective mechanism check with a
chronology caveat, not as an untouched paper test set or a closed-loop result.

## 5. Claim boundary and next gate

The passed test supports the following limited claim:

> Executor action coordinates remove the route-pace failure observed in the
> monocular estimators while preserving a scale-free bearing interface.

It does not establish navigation success. The controlled survey follows the
recorded route and supplies realized motion receipts; a policy rollout may
deviate laterally, collide, or slip. The next required step is therefore:

1. freeze the runtime bridge that records realized translation/yaw with every
   causal RGB observation;
2. run one controller smoke without reading SR;
3. if the runtime contract passes, run the preregistered paired comparison of
   endpoint-bearing CEC versus action-coordinate CEC on identical histories,
   proofs, ImageGoals, seeds, budgets, and success checks.

No endpoint controller or native fallback may be added to the challenger after
CEC has authorized its historical route.

## 6. SR-hidden runtime gate

The runtime bridge was frozen before navigation outcomes were read and then
executed on the preregistered index-40 history.  Slurm job `16780871` completed
on one H100 in 16 min 27 s with exit code zero.  The sealed gate receipt reports:

- `navigation_success_read = false`;
- `navigation_final_distance_read = false`;
- `navigation_sr_computed = false`;
- 10 authorized action-coordinate readouts and 10 frame-bound executor
  receipts during the query;
- 9 positive realized-translation updates, with progress finite and monotone;
- unit-norm bearings and an exactly 2.5 m controller PointGoal;
- no evaluator pose, Habitat path, metric route scale, visual gate, distance
  regime, endpoint fallback, or native fallback after authorization.

This gate authorizes the closed-loop comparison but contains no navigation
outcome.  Its immutable source receipt is
`7f4bb20a451e03fb85a0742117370897fdc799da01ed3632f67a0d8e50390fed`.

## 7. Frozen paired evaluation

The formal development comparison was submitted only after the SR-hidden gate
passed.  It evaluates all 48 frozen histories, with one hidden Novel query and
one hidden Revisit query per history, using two same-process paired arms:

1. canonical CEC with one endpoint bearing;
2. CEC with the action-coordinate route compass.

The arms share the history, initial CEC proof, ImageGoal, seed, and success
contract.  They also receive the same pre-existing Table-3 execution budget
for that history:
`max(600, ceil(2.5 * max_query_geodesic / 0.0376))`, bounded to
`600..3400` steps.  This evaluator budget is fixed before either arm runs.
As in the canonical pipeline, the server reset receives the total episode
horizon and uses it to choose LingBot's shared keyframe-compression flow tier;
this inherited setting is identical for both arms.  The action-coordinate
estimator itself never reads the geodesic distance, length bin, budget, or flow
tier and has no corresponding control branch.  The arm order is balanced.  The
runtime receives no Novel/Revisit label; outcomes are loaded only after both
arms finish.  Within each history's budget the stuck trigger is disabled
(`stuck_window = max_steps + 1`), so the challenger has neither a hidden
recovery branch nor an endpoint/native fallback after authorization.

Submission provenance:

```text
evaluation array: 16781348_[0-47]%3
automatic aggregate: 16781349 (afterany; fails closed on missing records)
independent verifier: 16781350 (afterok aggregate)
run root: /scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_action_coordinate_compass_20260902/pair_b7de41263049e415
source receipt: b7de41263049e415755f3595338f9e064373966e44992d234ebe453e55bb802a
```

This population is already-consumed development data.  Its result may decide
whether the architecture is worth confirmation, but it is not an untouched
paper test.  No result is reported until all 48 paired records are complete
and the independent verifier passes.

### 7.1 File-quota interruption and exact retry

The original array produced 18 sealed paired completions (indices 0--17).
Indices 18--20 then failed while writing runtime-buffer JPEGs with
`OSError: [Errno 122] Disk quota exceeded`; the remaining indices did not
produce records. At diagnosis, the user scratch allocation contained
`5,000,083 / 5,000,000` files. This is a method-independent infrastructure
failure: all three failing shards raised the same exception at the filesystem
write boundary. The failed partials were not interpreted as navigation
outcomes, and no partial SR was computed.

The repair followed the shared-HPC exact-retry contract:

- completion sidecars for retained indices 0--17 were verified without
  parsing their outcomes;
- partial evaluation directories, server logs, and buffers for indices
  18--20 were moved intact to
  `failed_attempts/16781348_disk_quota_exact_retry_20260902`;
- only regenerable runtime RGB buffers belonging to already-sealed indices
  0--17 were removed: 89,942 files and 3,560,282,971 bytes in total;
- completion records, per-arm plans and metrics, and all logs were retained;
- scratch usage after cleanup was `4,925,323 / 5,000,000` files;
- the replacement wrapper invokes the exact immutable scientific wrapper and,
  only after verifying a successful completion sidecar, removes that shard's
  regenerable RGB buffer. It does not change either arm.

The missing set was resubmitted exactly once as `18-47%3`:

```text
exact retry:          16788483
aggregate:            16788486 (afterok exact retry)
independent verifier: 16788487 (afterok aggregate)
repair bundle receipt:
  93789ef8d3d3b6852389ec40ab06f966faa45a61da67e51b798b596a80b4a096
scientific source receipt (unchanged):
  b7de41263049e415755f3595338f9e064373966e44992d234ebe453e55bb802a
```

The repair changes temporary-file retention only. The population, histories,
seeds, arm order, controller code, episode budgets, success contract, and
scientific source receipt remain unchanged. At submission time no navigation
outcome from the 18 completed records had been read.

## 8. Interpretation fixed before reading outcomes

The primary contrast is action-coordinate versus endpoint-bearing CEC on all
48 Revisit queries.  The three 16-history distance strata diagnose where the
effect occurs; they are not runtime modes and will not create a distance gate.
Novel queries check that the common initial certificate and reject path remain
identical rather than providing another optimization target.

The post-run decision is constrained as follows:

- an overall net loss, or a repeatable short-range loss that offsets long-range
  gains, rejects this as the sought unified controller; no distance switch is
  added afterward;
- a positive point estimate whose scene-cluster interval still crosses zero is
  development evidence only and requires a new untouched confirmation;
- a positive overall effect with no contradictory distance stratum justifies
  freezing the architecture for confirmation;
- any pairing, proof-equality, role-hiding, receipt, or information-boundary
  failure invalidates the affected run rather than becoming a navigation loss
  or a fallback completion.

Exact McNemar counts, scene-cluster intervals, SPL, path length, and final
distance will all be reported.  There is no post-hoc SR pass threshold and no
partial-result stopping rule.
