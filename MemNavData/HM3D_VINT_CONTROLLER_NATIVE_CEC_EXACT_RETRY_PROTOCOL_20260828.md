# HM3D ViNT native-control / ViNT+CEC exact retry protocol

Date frozen: 2026-08-28
Status: **infrastructure-only repair protocol; no repaired outcome read**

## Scope

The frozen formal array `16482393_[0-27]` completed 26 of 28 paired HM3D
histories.  Tasks `23` and `27` terminated with exit code `6:0` while running
the first `forced_reject_native` arm.  Their shell logs report the Habitat
evaluator as `Aborted (core dumped)`.  Neither task produced an arm
`summary.json`, a `controller_native_pair_audit.json`, or a query outcome that
could enter the formal estimator.

This repair reruns exactly those two indices:

```text
23  LEFTm3JecaC  episode_0001
27  58NLZxWBSpk  episode_0001
```

## Frozen treatment

The retry reuses the original immutable source bundle, benchmark, controller
checkpoints, causal histories, arm order, seeds, 600-step budget, success
radius, and one-hour H100/A100 Slurm contract.  It changes no method code,
threshold, checkpoint, query, population, or analysis rule.

- source bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_vint_cec_3c8da4454ad11c64`;
- source receipt SHA-256:
  `735f6e39012bfa5bd02c1ddfcbaa8c0a2e17d0369f892260c1e3709fd16796a9`;
- formal run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_vint_controller_native_cec_20260828/hm3d_vint_cec_table1_20260828`;
- benchmark SHA-256:
  `aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0`.

## Preservation and replacement analysis

Before retry, each incomplete `vint/` cell is moved intact to
`formal/failed_attempts/vint_cec_exact_retry1_20260828/`.  It is never deleted
or treated as a failure outcome.  The replacement array may write only the two
now-missing formal cells; all 26 completed cells remain read-only.

The replacement formal aggregate depends on successful completion of both
retry tasks and still requires all 28 frozen pair audits over 21 scene
clusters.  The independent verifier depends on that aggregate and recounts all
56 raw query rows.  No partial 26-history statistic is a paper result.
