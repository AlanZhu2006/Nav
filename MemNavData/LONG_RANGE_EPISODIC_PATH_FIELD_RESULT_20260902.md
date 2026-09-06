# Long-range Revisit: continuous episodic path-field result ledger

Date: 2026-09-02 (Asia/Shanghai)  
Status: route-information gate passed; global LingBot curve rejected; local
same-heading gates passed; strict reverse-traversal deployment gate failed  
Canonical paper method: unchanged fixed-bearing CEC

## 1. Problem isolated

The consumed 48-history HM3D length stress test showed that increasing or
metricizing the endpoint residual does not solve long returns.  Canonical CEC
was `4/16`, `2/16`, and `0/16` in the approximately 10 m, 20--25 m, and
30--40 m strata.  Its PnP bearing was close to the direct endpoint ray, but
that ray differed from the geodesic initial tangent by roughly 33--42 degrees.

The missing variable is route structure, not endpoint scale: one global chord
can point through a wall even when localization is correct.

## 2. Frozen challenger

The new readout preserves exactly the same policy authority:

```text
certificate-selected Revisit
  -> causal LingBot pose sequence from query tail to historical anchor
  -> one continuous temporally ordered curve
  -> monotone projection of current pose
  -> reference 2.5 m ahead in curve arc length
  -> discard magnitude; retain unit bearing only
  -> the same fixed 2.5 m mixed ImageGoal + PointGoal NavDP request
```

There is no distance classifier, short/long switch, stuck detector, node
arrival threshold, graph search, second planner, or alternate policy.  Short
curves naturally terminate at the endpoint; long curves expose a local route
direction.  The first-40 metric receipt controls only arc-length lookahead.

Implementation:

- `MemNavData/episodic_path_field.py`
- `NavDP/baselines/memnav/policy_agent.py`
- `MemNavData/eval_2leg_habitat.py`

The default guidance mode remains `endpoint_bearing`; the challenger requires
the explicit development flag `episodic_path_field`.

## 3. Stage-A route-information upper bound

Inputs were downloaded read-only from the independently verified, already
consumed 48-history result bundle.  The manifest SHA-256 was independently
recomputed as:

```text
cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
```

All 48 certificate-accepted Revisit queries were auditable; none was excluded.
Evaluator positions were used only in this information diagnostic.  PnP pose
and LingBot scale were deliberately not mixed into the evaluator coordinate
frame.

| Distance stratum | N | Endpoint median error | Ordered-route median error | Improvement | `<=30 deg` endpoint -> route |
|---|---:|---:|---:|---:|---:|
| 0--20 m pool (actual approximately 7--10.6 m) | 16 | 44.59 deg | 6.95 deg | 37.65 deg | 5/16 -> 14/16 |
| 20--30 m pool | 16 | 42.27 deg | 25.93 deg | 16.34 deg | 5/16 -> 8/16 |
| 30--50 m pool | 16 | 54.33 deg | 31.60 deg | 22.73 deg | 5/16 -> 8/16 |
| All | 48 | 44.59 deg | 19.40 deg | paired median 21.12 deg | 15/48 -> 30/48 |

Additional audit:

- route direction improved angular error on 33/48 queries and worsened it on
  15/48;
- all three strata improved in median, but individual alternatives/homotopies
  can differ sharply from the shortest-path tangent;
- coordinate gauge error was at most `3.18e-14 deg`;
- evaluator-trace replay progress was finite and monotone for 48/48.

Every preregistered Stage-A criterion passed.  This establishes only that the
ordered causal history contains substantially more useful local route
information than the endpoint chord.  It is not a deployable result: the
diagnostic used evaluator poses, and the controlled historical route itself is
a previously traversed feasible demonstration rather than a newly predicted
shortest path.

Audit code and generated local result:

- `MemNavData/audit_episodic_path_field_information.py`
- `.diagnostics/long_range_path_field_20260902/episodic_path_field_information_audit.json`

## 4. Stage-B LingBot deployment gate

The runtime implementation now constructs the identical curve only from:

- `cam_pose[:goal_start_frame]`;
- the certificate-selected causal anchor;
- current `cam_pose[-1]`;
- the immutable first-40 scale receipt;
- the PnP terminal witness in the same LingBot gauge.

Geometry failures emit an explicit receipt.  The strict challenger aborts
instead of silently reverting to the endpoint chord.  Certificate rejection
still retains canonical mixed-role native behavior; route completion returns
to the original ImageGoal for terminal visual alignment.

The frozen gate selects the lowest population index in each distance stratum:
`0`, `16`, and `32`.  Each replays the full causal RGB history but executes at
most 80 Revisit query steps.  It reads path receipts and failures only, not
navigation success.

Two pre-model infrastructure attempts are retained and excluded from method
results:

- `16762007` failed in 20--31 seconds because its wrapper invoked an older
  task runner that rejected the `causal_survey` history contract;
- `16762125` failed in 39--51 seconds because the experimental overlay was
  incorrectly also used as `WRAPPER_ROOT`, so the verified lifetime-held port
  allocator was outside the runner's search path.

Neither attempt loaded the models, ran a query, or produced a navigation
result.  The corrected composition uses the verified Table-3 SURVEY bundle as
`WRAPPER_ROOT` and runner, while the content-addressed experiment bundle is
used only for query/server source.  Failed roots are retained and not
overwritten.

The corrected HPC submission was:

```text
job: 16762311_[0,16,32]
requested time: 01:00:00 per element
partitions: h100_tandon,a100_tandon
run root: /scratch/yz11502/Research/Nav-axis-uturn-results/
          hm3d_episodic_path_field_20260902/gate_db1ea94f32b53108
source receipt: db1ea94f32b53108d497c8e22cb7e08a57b8105f0fa51a952e79902c2a762706
```

Before any element left `PENDING`, an exact CLI audit found that the wrapper
still supplied a runtime Revisit role filter under a scope where role filtering
is forbidden.  Job `16762311_[0,16,32]` was cancelled at `00:00` before GPU,
model loading, or outcome generation.  The corrected runner now executes both
hidden-role queries and selects the Revisit receipt only after rollout for
analysis.  No result from this cancelled job is a method observation.

The exact deployment geometry was then closed locally by mirroring three
sealed causal histories and their hash-verified HM3D assets.  Each history was
replayed through the real LingBot/CEC server; Habitat, NavDP, actions, and SR
were not instantiated.  Reversed historical RGB was used only to inspect the
live pose curve under known progress:

| Frozen history | Demonstrated curve | Final progress | Remaining | Max cross-track | Decision |
|---|---:|---:|---:|---:|---|
| index 0, short | 11.00 m | 10.99 m | 0.01 m | 0.24 m | usable |
| index 16, medium | 48.98 m | 9.00 m | 39.98 m | 3.89 m | reject |
| index 32, long | 78.96 m | 17.52 m | 61.44 m | 4.95 m | reject |

All three runs retained finite monotone internal progress and stable scale
receipts, but those weak schema checks were insufficient: the medium and long
curves lost global alignment under accumulated monocular-pose drift.  Because
the replay used the demonstrated observations themselves, this is already an
easier condition than closed-loop navigation.  The global pose curve is
therefore rejected and was never sent to an HPC closed-loop comparison.

Local receipts:

- `.diagnostics/long_range_path_field_20260902/local_rgb_gate_index0_attempt1/result/gate_result.json`
- `.diagnostics/long_range_path_field_20260902/local_rgb_gate_index16_attempt1/result/gate_result.json`
- `.diagnostics/long_range_path_field_20260902/local_rgb_gate_index32_attempt1/result/gate_result.json`

## 5. RGB-only local re-anchor gate

The failure isolated a gauge problem rather than a retrieval problem.  The
redesign retains the historical sequence as a temporal route but repeatedly
re-establishes current pose from local visual evidence.  Twelve non-duplicate
views were rendered along the frozen medium route at an eight-frame midpoint
and alternating `+/-15 deg` yaw.  Construction poses remained analysis-only;
runtime received RGB and the sealed causal history only.

The first local run produced 10/12 strict accepts, but a subsequent image-level
construction audit found that the v1 renderer consumed the trace's navigable
floor position as the optical-center position.  The original survey explicitly
adds its frozen 0.5 m camera height.  V1 therefore rendered a floor-level
camera and is invalidated in full; its numbers below are retained only as an
audit trail and are not mechanism evidence:

| Measure | Frozen gate | Result |
|---|---:|---:|
| strict certificate accepts | at least 9/12 | 10/12 |
| anchors within 1.0 m | at least 9/12 | 10/12 |
| accepted median position error | at most 0.75 m | 0.268 m |
| forward jumps above 16 frames in return order | at most 1 | 0 |
| accepted per temporal third | at least 2 each | 4 / 4 / 2 |
| finite PnP witness for every accept | required | 10/10 |

The two rejected views were the two closest to the causal tail; both failed
closed before authority.  They do not invalidate route startup because the
tail pose is the live causal state before drift has accumulated, but this must
be tested explicitly in the controller implementation.

The unchanged gate was rerun on v2 views constructed from the same floor
midpoint plus the frozen 0.5 m camera-height offset.  Corrected v2 passed every
criterion:

| Measure | Frozen gate | Corrected v2 |
|---|---:|---:|
| strict certificate accepts | at least 9/12 | 12/12 |
| anchors within 1.0 m | at least 9/12 | 12/12 |
| accepted median position error | at most 0.75 m | 0.057 m |
| forward jumps above 16 frames | at most 1 | 0 |
| accepted per temporal third | at least 2 each | 4 / 4 / 4 |
| finite PnP witness for every accept | required | 12/12 |

This establishes local visual addressability under controlled, same-height
viewpoint change.  It is not SR.  It authorizes only the denser proof-once
route-coordinate mechanism gate frozen in the protocol.

Artifacts:

- `MemNavData/generate_hm3d_episodic_reanchor_queries.py`
- `MemNavData/run_local_episodic_reanchor_gate.py`
- `.diagnostics/long_range_path_field_20260902/local_reanchor_gate_index16_attempt1/result/gate_result.json` (invalid v1 audit trail)
- `.diagnostics/long_range_path_field_20260902/local_reanchor_gate_index16_v2_attempt1/result/gate_result.json` (valid corrected v2)

The audit also exposed a latency requirement.  The current single-goal CEC
implementation lazily replays the LingBot stream to obtain exact depth for a
new anchor.  Repeating that operation at every route update is not a deployable
implementation.  The controller version must cache sparse historical depth
and retrieval features during the original causal traversal; accuracy and
latency are separate gates.

## 6. Decision after the local re-anchor pass

- Implement local re-anchoring as an internal proof-derived pose update; do
  not expose evaluator pose, a route role, or an arbitrary external pose API.
- Cache sparse historical depth/features on the first causal traversal so a
  local proof does not replay hundreds of old frames at every decision.
- Run one local/controller smoke without reading SR, then a paired development
  comparison on the consumed 48-history set: canonical endpoint bearing versus
  locally re-anchored path bearing, with identical target proof decisions,
  seeds, ImageGoal, controller, and budget.
- Only a favorable development result permits freezing a new disjoint
  long-range HM3D confirmation population.  The paper cannot claim the
  extension before that confirmation.

## 7. Verification completed locally

- 64 route/runtime regression tests passed after closed-loop integration;
- Python compilation passed;
- repository diff whitespace checks passed;
- exact remote container import and CLI contract dry-run passed before
  submission;
- the source bundle is content-addressed and read-only.
- the invalid v1 local re-anchor run passed all six numeric checks and the GPU
  server exited cleanly, but camera-height audit invalidated the construction;
  corrected v2 was subsequently completed at the proper optical height and
  passed 12/12.

## 8. Proof-once monotone visual-route result

The first 24-query route-coordinate implementation incorrectly used the
`2.5 m` controller horizon as a localization horizon. It updated only 8/24
queries and then failed explicitly with `34.434 m` remaining. The successful
anchors progressively lagged the constructed locations (errors reached
`1.37 m` before failure). This was not a matcher-capacity failure: corrected
global route localization had already achieved 12/12 strict accepts. It was a
coupling error between accumulated monocular route scale and local control.

The corrected design searches only the still-unvisited portion of the single
already-authorized temporal route. The selected visual address may advance the
monotone route coordinate by more than one controller horizon; the policy
still receives only a `2.5 m` arc-ahead unit bearing. Sparse historical depth
is written every eight frames by the existing LingBot stream to CPU memory.
No second KV stream is maintained.

On the same frozen 24 corrected RGB queries:

| Check | Result |
|---|---:|
| route-coordinate state updates | 24/24 |
| selected anchors within 1.0 m | 24/24 |
| median / maximum anchor error | 0.0553 / 0.5405 m |
| initial / final remaining route | 48.9812 / 0.8828 m |
| monotone progress | pass |
| finite active bearings | pass |
| endpoint or native fallback | none |
| distance gate or local control authority | none |

The GPU curve also passed the deployment resource check: memory stabilized at
approximately `24.0 GB` from frames 546 through 1,078, whereas the rejected
second-dense-KV implementation had already reached `39.4 GB` by frame 388.

The result was independently loaded and recomputed from the raw JSON. SHA-256:

```text
2cccd935155ffc5ac0bf1167db82815cec528bb26c2ff0c77936a1f81ff16c94
```

Artifact:

- `.diagnostics/long_range_path_field_20260902/local_route_coordinate_gate_index16_attempt3/result/gate_result.json`

This remains an RGB mechanism result: no NavDP action and no SR was computed.
The next frozen step is the three-history outcome-blind HPC deployment gate in
`LONG_RANGE_MONOTONE_VISUAL_ROUTE_DEV_PROTOCOL_20260902.md`.

## 9. Frozen HPC deployment-gate submission

The minimal runtime bundle was rebuilt after its staging-only self-test found
that an earlier draft had accidentally included a broad legacy regression
test.  That test imported unrelated CDEC and Pi3X branches and therefore did
not describe the path-field runtime closure.  The broad test remains in the
local repository regression suite; the sealed deployment bundle contains only
the modules exercised by this experiment, including the direct colored-geometry
dependency of PnP.  The earlier uploaded draft was never submitted and remains
an infrastructure audit artifact.

The replacement bundle passed, in order:

- 62 local repository regression tests;
- 23 staging-only tests with only the bundle on `PYTHONPATH`;
- SHA-256 and read-only checks after upload;
- the same 23 bundle-only tests under the remote MemNav interpreter;
- direct imports of the PnP and policy runtime;
- the exact Habitat container contract dry-run;
- Slurm lint and `sbatch --test-only`.

The frozen deployment gate was then submitted without reading navigation
outcomes:

```text
job: 16772919_[0,16,32]
requested time: 01:00:00 per element
partitions: h100_tandon,a100_tandon
run root: /scratch/yz11502/Research/Nav-axis-uturn-results/
          hm3d_episodic_path_field_20260902/gate_2692f18a6561eb7a
source bundle: /scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
               hm3d_episodic_path_field_gate_2692f18a6561eb7a
source receipt: 2692f18a6561eb7ad310bb907de9df01a29ee1f2d33268c70cd6684c7cdf6cbe
```

At submission the array was pending under the project QoS GPU-group limit.
This is scheduler state, not a runtime or scientific result.  The gate remains
outcome-blind: it audits proof acceptance, online visual route-coordinate
updates, monotonicity, finite bearings, explicit failure, and absence of
endpoint/native fallback.  It does not use success rate to choose the method.

## 10. Strict deployment-gate result: first design rejected

All three preregistered array elements finished and all three failed the
strict no-fallback runtime contract:

| History | Slurm state / elapsed | First disqualifying event |
|---|---:|---|
| index 0 | failed / 00:10:18 | local route PnP witness unavailable after several successful updates |
| index 16 | failed / 00:17:36 | hard monotone address reached route completion before benchmark position success |
| index 32 | failed / 00:18:15 | local route PnP witness unavailable |

No element emitted a valid gate result and no success rate is reported.  The
already staged 48-query comparison was therefore not submitted.  These are
method failures rather than scheduler or dependency failures: model startup,
the hidden-role Novel rollout, initial target proof, and initial route
construction all completed before the explicit route-coordinate exception.

The two failure signatures isolate separate defects in the first design:

1. repeated local PnP assumes sufficient appearance overlap between the
   current return-facing camera and an outgoing historical frame;
2. selecting a single best address from the complete remaining route permits
   perceptual aliasing to jump the monotone coordinate close to the endpoint.

The failed immutable run is retained at:

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_episodic_path_field_20260902/gate_2692f18a6561eb7a
job 16772919_[0,16,32]
```

## 11. Reverse-view observability audit

The controlled local gates above rendered each query at the historical
heading plus or minus 15 degrees.  A real return along the taught route often
faces approximately 180 degrees away from the outgoing camera.  A new
construction-only audit therefore rendered the same 12 positions with a
180-degree base offset and the same alternating 15-degree perturbation.  The
runtime again received only query RGB and the sealed causal RGB history.

The DINO-only address result was:

| View condition | top-1 within 1 m | top-8 contains a frame within 1 m | median top-1 position error |
|---|---:|---:|---:|
| historical heading +/-15 deg | 12/12 | not required for the earlier gate | 0.130 m |
| reverse heading 180 deg +/-15 deg | 0/12 | 4/12 | 5.295 m |

The reverse audit also produced six large top-1 temporal-order violations.
It invoked neither PnP nor a controller and computed no SR.  This result rules
out threshold tuning as a repair: a hard per-frame DINO address followed by a
mandatory local PnP witness is not an observable return-state estimator under
the actual camera geometry.

Artifact:

```text
.diagnostics/long_range_path_field_20260902/
  local_route_address_index16_reverse180_v3_attempt1/result/address_audit.json
SHA-256 b13cf761ea889cded0cec75a48c66f07c586d207183bb1702e8da21f3ba39068
```

The next admissible design must exploit the state that is genuinely known at
a goal switch: the live camera is continuous with the tail of the causal
history.  It should propagate a continuous route coordinate with short-range
relative motion and use appearance only as a soft sequence observation.  It
must not reinstate a range classifier, a hard per-frame proof gate, endpoint
fallback, or native fallback after the initial target certificate accepts.

## 12. Pure route-tape odometry is also insufficient

The next diagnostic tested the strongest simpler alternative before adding a
sequence model.  It rendered a physically ordered RGB return stream on index
16: six in-place turn frames from the outgoing terminal heading to the return
heading, followed by 72 reverse-route views at approximately 0.30 m spacing.
The demonstrated return covered 22.96 m.  Runtime received only the outgoing
RGB prefix and this RGB stream.  A read-only receipt exposed LingBot's own
pose predictions; evaluator poses were used only after inference to score the
result.

An intrinsic route tape was initialized at the known causal tail and advanced
only by positive model-predicted forward motion.  It had no appearance update,
control action, distance gate, or fallback.  The result was negative:

| Measure | Result |
|---|---:|
| target / predicted final progress | 22.96 / 8.60 m |
| median / final absolute progress error | 6.64 / 14.35 m |
| median route-bearing error | 53.25 deg |
| route bearings within 30 deg | 19/72 |
| spurious translation during the six-frame turn | 2.27 m |

Even though the model-derived historical route length on the sampled tape was
close to the demonstrated length (21.41 versus 22.96 m), the return-facing
images caused predicted motion to become largely lateral or stationary.  Pure
integration therefore cannot replace visual state correction.  This rejects
the second tempting shortcut: neither repeated single-frame relocalization nor
odometry-only route playback is reliable in isolation.

Artifacts:

```text
.diagnostics/long_range_path_field_20260902/
  reverse_route_odometry_index16_v1/query_set.json
  local_route_tape_odometry_index16_v1_attempt1/result/odometry_audit.json
odometry audit SHA-256 fe2d296cd016e3dc6d45267b82ff982ce550cdf74db535f87d4cacd97c45e391
```

The remaining evidence-backed hypothesis is a causal sequence estimator: use
the known tail state as its initial condition, a monotone transition model as
the route dynamics, and DINO similarity as a soft observation over the whole
authorized route.  This is a single continuous belief update, not an
accept/reject gate.  A sequence-information audit must pass before it is wired
to NavDP.

## 13. Action-coordinate redesign: prospective mechanism gate passed

The subsequent sequence audit showed that a soft visual filter still confused
frame density with traveled distance: it passed two consumed routes but drifted
to a 5.16 m final address error on the next route. Extending the monocular
scale receipt from 40 to 64 frames left that failure essentially unchanged.
The causal survey contains many turn-only frames, so frame advance is not a
stable route coordinate.

The replacement keeps LingBot only as an arbitrary-scale route shape and uses
the executor's own recorded translation/yaw receipts as the route coordinate.
It has no distance regime, post-authorization visual threshold, endpoint
controller, or native fallback. On the frozen prospective index-40 route it
obtained:

- address within 1 m: `99/99`;
- median/final address error: `0.145/0.293 m`;
- bearing within 30 degrees: `99/99`;
- median/P90 bearing error: `8.77/14.96 degrees`.

All preregistered mechanism criteria passed. The subsequent SR-hidden runtime
smoke also passed without reading success or final distance (job `16780871`).
The frozen 48-history endpoint-bearing versus action-coordinate comparison is
therefore running as array `16781348`, followed by aggregate job `16781349`
and independent verifier `16781350`. No partial outcome may be reported. Full
protocol, chronology caveat, source receipts, and claim boundary are recorded
in:

- `MemNavData/LONG_RANGE_ACTION_COORDINATE_COMPASS_PROTOCOL_20260902.md`;
- `MemNavData/LONG_RANGE_ACTION_COORDINATE_COMPASS_RESULT_20260902.md`.
