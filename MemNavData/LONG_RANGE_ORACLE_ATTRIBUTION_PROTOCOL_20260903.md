# Long-range Revisit oracle attribution protocol

Date frozen: 2026-09-03 (Asia/Shanghai)  
Scope: consumed HM3D Table-III development population; diagnostic only.

## Question

The deployable action-coordinate compass reduces final distance on long
Revisit queries but has not produced a large SR gain.  This experiment asks
which boundary is responsible:

1. route-state estimation from executed motion;
2. the reverse historical route itself;
3. simultaneous ImageGoal/PointGoal conditioning in frozen NavDP; or
4. the frozen controller's long-horizon capability.

It does not propose a new method and is not eligible for a paper result.  The
three oracle arms consume evaluator pose, the query goal position, and Habitat
shortest-path geometry.  Every output must retain that disclosure.

## Frozen population

- Source: the existing sealed 48-history HM3D causal-RGB length population.
- Cohort: indices 32--47 only, all in `30_to_50_m` (16 histories).
- Query: Revisit only.  The role is used by the evaluator to select this
  already-consumed diagnostic cohort and is never forwarded to CEC or NavDP.
- History: the byte-identical causal RGB survey replay and its frozen
  first-40 monocular scale receipt.
- Success: Habitat position distance below 1.0 m.
- Budget: the existing per-history length budget, derived without outcomes.
- Execution horizon: eight simulator actions per NavDP plan.

## Four same-process arms

All arms rerun the real DINO proposal, geometry witness, PnP certificate, and
monocular-depth transaction.  An oracle controller sample is permitted only
after the real certificate accepts.  Diffusion seed, history, goal image,
current observation, FIFO, depth receipt, benchmark geometry, and action
executor are paired.

1. `action_coordinate_mixed`: current deployable scale-free route clock and
   mixed ImageGoal/PointGoal NavDP.
2. `oracle_route_mixed`: project the true current evaluator position
   monotonically onto the reverse observed historical route, connect the
   certified anchor to the exact goal by a local Habitat path, take a 2.5 m
   lookahead, discard magnitude, and resample the mixed decoder.
3. `oracle_geodesic_mixed`: recompute the current Habitat shortest path to the
   goal at each planning decision, take a 2.5 m lookahead, discard magnitude,
   and resample the mixed decoder.
4. `oracle_geodesic_point`: use the identical geodesic PointGoal but resample
   the pure PointGoal decoder, removing ImageGoal conditioning only.

Arm order rotates by population index.  The first ordinary CEC/NavDP call
advances the causal memories exactly once.  Oracle replacement uses a
read-only resample endpoint, must preserve FIFO length and content hashes, and
must bind to the same frame-specific monocular-depth transaction and diffusion
seed.

## Interpretation

No numeric pass threshold is defined after seeing the preceding result.
Instead, the paired pattern identifies the next engineering target:

- oracle route approximately matches oracle geodesic and both recover well:
  route-state estimation is the dominant bottleneck;
- oracle geodesic mixed clearly exceeds oracle route mixed: the reverse
  historical homotopy/route is itself inadequate;
- oracle geodesic point clearly exceeds oracle geodesic mixed: mixed
  ImageGoal/PointGoal conditioning causes a controller conflict;
- all oracle arms remain weak: frozen NavDP is not an adequate long-range
  controller for this extension.

These statements are causal diagnostics within a consumed development set,
not generalization claims.  The canonical CEC paper method and its main-table
results remain unchanged.

## Required audit

- immutable source bundle and exact manifest SHA-256;
- all four arms in one array element and one persistent server process;
- identical frozen history replay and first target proof across arms;
- real certificate acceptance at every overridden decision;
- explicit evaluator-pose/goal/path disclosure in every oracle plan receipt;
- 2.5 m PointGoal norm, fixed deterministic seed, and read-only FIFO hashes;
- no partial aggregation; all 16 histories required;
- aggregate followed by an independent raw-completion verifier.

One full 16-history-element gate is run before the full array to measure the
real all-arm wall time.  Production time limit is selected from that measured
gate with the HPC manual's safety margin; it is not inherited from an older
template.
