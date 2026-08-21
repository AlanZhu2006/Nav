# Lifelong NNR runtime repair — 2026-08-21

## Scope

This is an infrastructure-only amendment to the frozen lifelong NNR paired
evaluation.  It does not change the selected population, paired arms, seeds,
controller, thresholds, checkpoints, or outcome definitions.

## Incident

The original support audit (`16121493`) completed and selected the frozen
19-episode population.  Array tasks `16121506_0` through `16121506_7` then
failed before any rollout because the content-addressed source bundle omitted
NavDP's vendored `depth_anything.depth_anything_v2` Python package.  NavDP's
`policy_backbone.py` imports that package for its DINOv2 encoder
implementation.  No query result or SR was produced by these failed tasks.

The affected immutable bundle is:

`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/lifelong_nnr_51adf5128b500d60`

## Repair

The replacement bundle adds only the twelve Python source files required by
`depth_anything.depth_anything_v2`.  It does not add Depth-Anything model
weights and does not introduce a new depth predictor.  The submission script
now imports `policy_backbone` from the staged bundle before hashing, upload,
or Slurm submission.  This makes the previously missing transitive dependency
a local fail-fast condition.

The repaired run must use a new run root and content-addressed source bundle.
The failed array and its never-started downstream jobs are retained in Slurm
accounting as superseded infrastructure evidence.  Evaluation concurrency is
limited to two GPUs to respect the observed account QoS.

## Scientific invariants

- factual-B support selection remains result-blind and unchanged;
- arms remain `all_prior` and `initial_leg_only` in the same job;
- NNR source root and dependency receipt hashes remain unchanged;
- checkpoints, seeds, scene/episode identities, budgets, and success criteria
  remain unchanged;
- the repair may not read rollout outcomes from the failed attempt.
