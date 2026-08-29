# MP3D Table-1 ViNT scope allowlist repair（2026-08-29）

## Incident

The independently verified MP3D Table-1 population passed its prospective
construction gate with 42 histories from 25 scene clusters.  The first formal
submission created a NavDP smoke/formal/verification chain, but the ViNT smoke
job `16548403` stopped before policy execution with:

```text
ABORT: ROLE_PAIR_SCOPE must be consumed_integration or paper_heldout
```

The outer submitter, the ViNT sbatch wrapper, the Python evaluator, and the
frozen protocol already used `paper_replication`.  Only the shared portability
shell wrapper retained the older two-value allowlist.  The failed smoke created
no controller outcome and no ViNT evaluation cell.

## Additive repair

The repair adds `paper_replication` to two shell validation sites:

1. the general role-pair provenance allowlist;
2. the complete-population check for bounded certified-bearing alignment.

It does not change the benchmark, query identities, history, checkpoints,
controller parameters, bearing execution, success radius, step budget,
certificate thresholds, or runtime role visibility.  The Python evaluator
already treats `paper_replication` as a provenance label only.

## Resume contract

`SUBMIT_MODE=vint_scope_repair` in
`submit_hm3d_table1_controller_portability_hpc.sh`:

- verifies the exact failed-smoke error and `2:0` exit;
- verifies that no ViNT smoke/formal policy cell exists;
- binds the unchanged construction verifier and benchmark manifest;
- builds a new immutable source bundle containing the additive repair;
- submits only replacement ViNT smoke/formal/aggregate/verifier jobs;
- joins the replacement ViNT verifier with the already submitted NavDP
  verifier before creating the final controller seal;
- records both the failed and replacement bundle/job provenance;
- does not read partial policy outcomes.

The old immutable bundle remains unchanged.  The failed smoke is an
infrastructure receipt, not an experimental row.
