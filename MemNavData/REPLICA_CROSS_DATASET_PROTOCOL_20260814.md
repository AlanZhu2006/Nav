# Replica cross-dataset role-pair protocol

Date: 2026-08-14 (Asia/Shanghai)

## Purpose

Test whether the frozen role-free certified episodic-memory mechanism transfers
from MP3D to the visually and geometrically distinct official Replica-v1 scenes.
This protocol is frozen before any Replica navigation outcome is read.  It is
not used to tune retrieval, certificate, bearing radius, NavDP, or success
thresholds.

## Population and attrition

- Source archive: official Replica v1.0, locally extracted with receipt
  `/home/asus/Research/Pi3/data/replica_v1_full_20260814/EXTRACTION_RECEIPT.json`.
- Compatibility receipt:
  `.diagnostics/replica_full18_compatibility_20260814.json`, SHA-256
  `f6c14266b5aa5462d8a0d21c1219212e482f95b18fe73808eae204f3fe5752cf`.
- Ten of eighteen scenes passed the pre-navigation sensor/NavMesh gate.  Every
  eligible scene is attempted; construction failures remain explicit attrition
  and are never replaced based on policy outcome.
- The first implementation smoke uses `room_0`, selected before navigation
  because it has the largest compatibility-probe geodesic diameter (7.01 m),
  not because of controller performance.
- Because the 600-step `room_0` smoke has now exposed query outcomes, the
  formal run still executes all ten compatible scenes but treats `room_0` as
  a locked pilot.  The sole confirmatory inference population is the other
  nine scenes; a ten-scene aggregate may be shown only as descriptive.  The
  fresh primary target was at least 20 constructible histories over at least
  8 scene clusters.
- Each scene has exactly four deterministic source attempts.  The stratum is
  fixed from the simulator-only compatibility probe: scenes with sampled
  geodesic diameter at least 4.5 m use the 4.5--6.5 m long-history band; all
  others use the 2.0--5.0 m diagnostic band.  This rule is fixed before
  native Goal-A execution and is never changed using navigation outcomes.

### Pre-query constructibility veto

The first two detached attempts produced no policy rollout: attempt 1 failed
while redirecting Python bytecode from a read-only bundle, and attempt 2 failed
because the bundle omitted NavDP's `depth_anything` package.  Attempt 2 did,
however, finish outcome-blind source generation and exposed that the historical
two-leg carrier unnecessarily required a later Revisit-B even though only its
start and Goal A were consumed.

An outcome-blind diagnostic therefore kept every Goal-A condition unchanged
(distance band, 0.40 m turning/clearance radius, controller, smoothness and
collision gates, scene seeds and attempt budgets) and removed only the unused
Revisit-B.  Receipt:
`.diagnostics/replica_goal_a_constructibility_10scene_20260814/constructibility_result.json`,
SHA-256 `d4a74d41a3d4a874323b3efb28da445165233ef6338822fd1be0b0be5bd6782c`.
It yielded 20 fresh histories but only five fresh scene clusters, below the
frozen eight-cluster target.  Thus the planned ten-scene formal confirmation is
vetoed before any query outcome.  The remaining run is authorized only as an
explicitly underpowered cross-dataset stress test; it cannot be reported as a
formal confirmation regardless of its result.

The subsequent frozen stress construction further reduced to 23 successful
Goal-A traces, nine runtime-eligible histories, and three constructible role
pairs.  All three pairs belonged to the already consumed `room_0` pilot, so the
sealed fresh population was empty.  The query process was terminated without
reading any query outcome; partial files are excluded from every statistic.
The population-gate stop receipt is
`.diagnostics/replica_stress_10scene_20260814_attempt3/population_gate_stop_receipt.json`
(SHA-256 `90b228fef4bc2db5f6f39c1a30f76248fc32adba225dbe1e3a02d42c77618482`).

## Causal benchmark construction

1. `generate_twoleg.py --goal_a_source_only` supplies only a frozen start pose
   and Goal-A image.  Its first-leg sampling and execution contract is unchanged.
   A schema placeholder is emitted only so the historical loader can run with
   `--stop_after_leg1`; the evaluator fails closed if a query is attempted on
   this carrier.  No scripted Revisit goal is sampled, replayed or scored.
   Source inputs are attempted in two pre-query strata: a 2.0--5.0 m
   start-to-A diagnostic stratum and a 4.5--6.5 m long-history stratum.  The
   latter increases the chance of obtaining a runtime-eligible memory frame,
   but admission requires only one frame at or after LingBot's frozen frame-39
   anchor margin; it does not impose the double-Revisit two-anchor contract.
   Every generated source and every online-A outcome remains in attrition.
2. Frozen native NavDP navigates online from the sampled start to Goal A under
   the same 600-step, 1.0 m, horizon-8 and deterministic-seed contract as MP3D.
3. Only successful online-A traces with sufficient causal history are
   re-rendered and byte-hash checked on the same scene representation.
4. The paper's corrected single-Revisit V1 builder and matched-Novel builder
   construct one query pair per retained history.  Runtime receives neither
   role label nor construction diagnostics.  This construction correction was
   frozen before any Replica query rollout; it does not change method
   thresholds or read navigation outcomes.
5. Every query arm independently resets and replays the identical online-A
   observation history.

## Frozen method and comparisons

Primary arms are unchanged:

1. native NavDP;
2. raw metric DINO residual;
3. raw fixed 2.5 m bearing;
4. old geometry-fixed router;
5. certified LightGlue + LingBot-depth PnP, fixed 2.5 m bearing, exact native
   fallback on rejection.

The certificate remains: at least 16 PnP inliers, at least 5% query/reference
inlier hull coverage, and at most 2 px reprojection RMSE.  No rescue, oracle,
role label, threshold recalibration, or Replica fine-tuning is permitted.

## Reporting

The implementation smoke proves only contract compatibility.  The stress run
still reports paired SR/SPL, gain/loss counts, exact McNemar tests,
scene-cluster bootstrap intervals, acceptance by role, fallback equality,
latency and full attrition, but its five-scene fresh population is descriptive
and underpowered by construction.  Replica results are reported separately
from MP3D and are never pooled across datasets.
