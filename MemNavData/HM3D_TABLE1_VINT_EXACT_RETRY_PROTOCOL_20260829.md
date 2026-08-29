# HM3D Table-1 ViNT exact retry protocol

Date frozen: 2026-08-29  
Status: infrastructure-only exact retry; no partial navigation outcome read

## Incident

The frozen ViNT formal array `16526731_[0-27]` produced one incomplete cell:

```text
index 18  b28CWbpQvor  episode_0001  exit 6:0
```

The Slurm stderr records the Habitat evaluator as `Aborted (core dumped)`.
The process terminated after writing runtime frames but before producing
`controller_native_pair_audit.json`.  The partial directory is therefore an
infrastructure artifact, not a policy failure and not an input to an estimator.
No partial success, distance, or arm-level outcome was read when selecting this
repair.

## Frozen treatment

The retry reuses exactly the original immutable Table-1 bundle, benchmark
manifest, causal history, controller checkpoints, arm order, seed, 600-step
budget, success radius, and H100/A100 execution contract:

- source bundle:
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_controller_portability_c0c373bf2ed63087`;
- source-receipt SHA-256:
  `c0c373bf2ed630873751e72769643c7d52ee0493f17a8a7bece381f9d52ff955`;
- frozen benchmark SHA-256:
  `f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01`;
- formal run root:
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/formal_20260828T231109Z`.

The repair changes no method code, threshold, checkpoint, query, controller,
population, role visibility, or analysis rule.

## Preservation and dependency repair

The incomplete cell is moved intact to
`repairs/vint_exact_retry1_20260829/failed_attempts/` and made read-only. It is never
deleted, overwritten, or counted. Only array index `18` is resubmitted, and it
may write only the now-missing canonical cell.

The retry waits for the retained original array with `afterany:16526731`.
Slurm refused an in-place dependency edit on pending aggregate `16526745`, so
the still-unrun aggregate, verifier, and joint seal were cancelled and replaced
by jobs with the same immutable analysis scripts. The replacement aggregate
depends on the exact retry, and its verifier depends on that aggregate. A later
independent NavDP cache incident required its own exact repair, so the current
joint seal `16541369` waits for both the new NavDP verifier `16541368` and this
ViNT verifier `16540208`. The ViNT aggregate still requires all 28 frozen pair
audits over 21 scene clusters; there is no reduced-denominator result.
