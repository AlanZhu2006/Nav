# Long-range local-tangent attribution (consumed v2)

## Scope

This is a privileged mechanism diagnosis on the already-consumed HM3D
long-range history at frozen manifest index 32. It is not a deployable method,
not a paper result, and it does not reopen the sealed 48-history population.
No navigation outcome may be read until all three arms finish.

The v1 attribution revealed three evaluation/control confounds on this sample:

1. a 2.5 m planar chord can disagree sharply with the first traversable path
   tangent at a corner;
2. commanded pure-pursuit translation was recorded even when pathfinder snap
   left the agent stationary;
3. planar x/z success is ambiguous for a cross-floor query.

This fork repairs those measurements and changes no production CEC default.

## Frozen population and common contract

- benchmark manifest SHA-256:
  `cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451`;
- one consumed Revisit query: manifest index 32, bin `30_to_50_m`;
- full-monocular causal-survey history and pinned content-addressed NavMesh;
- real DINO proposal, local matching, PnP and strict CEC certificate execute
  before every privileged readout;
- same ImageGoal, FIFO, diffusion seed, 8-action horizon and 2.5 m PointGoal
  radius in all arms;
- task-specific training, graph rescue, stagnation triggers, oracle candidate
  selection and terminal U-turn are disabled;
- every controller replacement is read-only and must preserve FIFO hashes and
  the bound monocular-depth transaction;
- path length is the post-snap realized x/z displacement;
- success requires 3-D Euclidean distance below 1 m to the frozen goal-floor
  position.

## Arms

1. `oracle_chord_mixed_realized`
   - recompute the current Habitat shortest path at every decision;
   - project the 2.5 m planar geodesic chord to a fixed-radius PointGoal;
   - condition frozen NavDP jointly on ImageGoal and PointGoal.

2. `oracle_tangent_mixed_realized`
   - recompute the same shortest path;
   - use the first ordered path point at least 0.30 m away in x/z as the local
     tangent;
   - condition frozen NavDP jointly on ImageGoal and that fixed-radius tangent
     at every decision.

3. `oracle_tangent_then_native_realized`
   - compute the same first local tangent;
   - if its signed local heading residual exceeds 20 degrees, use the mixed
     ImageGoal+PointGoal request;
   - otherwise execute a read-only native ImageGoal resample with the same
     observation, FIFO, depth transaction and diffusion seed;
   - this is a direction-only intervention diagnostic, not a failure-triggered
     fallback and not a deployable source of direction.

## Interpretation gate

- Tangent better than chord isolates local route semantics as a cause.
- Selective tangent better than continuous tangent shows that the direction
  should correct heading rather than replace ImageGoal control indefinitely.
- If neither tangent arm improves realized progress or final 3-D distance, the
  next diagnosis must move downstream to NavDP's conditioned trajectory
  distribution/critic; no additional radius or threshold sweep is authorized.
- A success on this one consumed sample is only a mechanism lead. It cannot be
  promoted to an aggregate claim without a separately frozen replication.

## Failure handling

Runtime/server errors abort the arm and are not navigation failures. Partial
artifacts remain immutable. Reconstructible RGB buffer JPEGs live only under
`$SLURM_TMPDIR`; durable logs, metrics, plans and receipts remain under the run
root.
