# HM3D long-range rear-alignment incident and formal-runtime retry

Date: 2026-09-04  
Scope: consumed mechanism diagnostic only; no navigation claim

## Attempt 1 did not test the U-turn

The first immutable rear-alignment source bundle was
`hm3d_longrange_rear_alignment_1decfde19c1ba5c8`.  Its Slurm chain was:

- technical gate: `16928802` (history index 8);
- formal array: `16928803`;
- analysis: `16928804`;
- independent verification: `16928805`.

The gate stopped after one query step.  The initial target certificate still
accepted with exactly 379 matches, 333 PnP inliers, and 0.8971766 px
reprojection RMSE, but the route initializer then rejected the frame
992-to-998 `history_to_query_bridge` with zero local matches.  Consequently:

- no route-alignment packet was emitted;
- no bounded turn action was executed;
- no proof-bound yaw receipt was consumed;
- the formal and result jobs were dependency-cancelled.

This is therefore neither a U-turn failure nor a navigation outcome.

## Reproduction audit

The failed gate and the independently verified parent run shared:

- the same 998-frame causal Goal-A memory trace, canonical SHA-256
  `07719fcc0c4d08d2674ab662458cf65da05a62339274090a9c6933db13821421`;
- the same Goal-A rollout trace, SHA-256
  `4058392496febda723965318a446ac2647265892c70ae45475f7d89e902f8b36`;
- the same replay metadata digest
  `019d02522ffe2dadf62f3809bf3a0526520acfcd9ecc5c4e67b9d87cea0cae8b`;
- the same query start position and yaw;
- the same initial CEC target proof.

The parent formal source bundle was
`hm3d_longrange_route_tangent_751efdcbb436c99b`, whereas Attempt 1 packaged
the later working-tree runtime.  That working tree also contained subsequent
route-motion/dense-query attribution changes.  Even though their configured
defaults were intended to preserve behavior, using that accumulated runtime
made a result-dependent mechanism comparison needlessly confounded.  The
correct repair is to derive the treatment directly from the sealed parent
runtime, not to select another history that happens to initialize.

## Minimal formal-runtime overlay

The retry bundle is:

`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_longrange_rear_alignment_formal_overlay_a86e3d9b07234fd6`

Receipt SHA-256:

`a86e3d9b07234fd653438b05d1288e295948b2ed9afb0ffbad718a7f5caf74ef`

It is constructed from the sealed formal source receipt
`751efdcbb436c99b5cc92d69633e812e9d1edc381dcf224af9042113ee8e874e`.
The parent `monocular_adjacent_motion.py`, route-tangent runtime,
`memnav_server.py`, NavDP runtime, and full-mono launcher remain byte-identical.
Only the following treatment plumbing is added:

1. seal the accepted route tangent in a route-specific authorization packet;
2. when its forward component is negative, discard the pre-turn controller
   horizon and execute at-most-30-degree zero-translation yaw atoms;
3. bind each fresh observation to the issued yaw action receipt;
4. rotate the existing route compass without sending pure-rotation frames
   through visual PnP;
5. replan with the unchanged frozen NavDP after the turn.

The packet and runtime reject role labels, Habitat pose, evaluator distance,
mixed motion batches, partial turns, translation during alignment, and any
unbound executor source.

Verification:

- isolated local bundle: 36 tests passed;
- exact remote bundle with the production MemNav interpreter: 36 tests passed;
- source receipt and every bundled file passed SHA-256 verification;
- both evaluator contract dry runs and all Slurm test submissions passed.

## Retry DAG

- technical gate, frozen index 8: `16929826`;
- formal nine-history paired array: `16929827`;
- analysis: `16929828`;
- independent verification: `16929829`.

Result root:

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_rear_alignment_20260904/consumed_a86e3d9b07234fd6`

At submission, the gate was pending with `QOSMaxGRESPerUser` because the
authority-spectrum array occupied the current user GPU allowance.  This is a
queue condition, not an evaluator error.  The formal array cannot run unless
the gate exits successfully.

## Scientific boundary

The nine histories and their parent outcomes were read before this mechanism
was designed.  The comparison can establish whether the rear PointGoal
support mismatch is causal on this consumed partition.  It cannot contribute
a fresh SR claim.  A positive mechanism result must be followed by a new,
outcome-blind long-range population before inclusion as a navigation result.

