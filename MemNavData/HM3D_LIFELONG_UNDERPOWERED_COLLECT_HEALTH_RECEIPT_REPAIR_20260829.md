# HM3D factual-C health-receipt verifier repair

Frozen on 2026-08-29 after reading only Slurm states, the failed CPU-gate
traceback, structural file presence, `hub_health.json`, and
`compute_identity.json`. No factual-C success, SPL, distance, action decision,
or B2 outcome was read.

## Incident

Collect smoke `16514058` completed on its frozen source node. Gate `16514066`
then rejected the smoke because it expected `reject_policy` and
`reject_controller` in `/healthz`. The immutable legacy HM3D hub never emitted
those two fields. It did emit a healthy initialized NavDP controller receipt,
while the same smoke's hash-bound `compute_identity.json` recorded
`legacy_shared_native_exact` and `shared_native_exact`. Before startup, an AST
audit of the exact frozen hub additionally proved that the same
`ComparisonPlan` hard-codes the NavDP fallback.

This was therefore a verifier-schema mismatch, not an authority mismatch. The
six scientific retry jobs and all B2 descendants were dependency-cancelled
before start. No navigation output was selected or discarded by efficacy.

## Minimal repair

The verifier now supports two receipt forms:

1. a newer health payload may explicitly state `shared_native_exact` and
   `navdp`;
2. the frozen legacy health payload must state schema v2, controller NavDP,
   initialized true, reset-required false, and force-reject false, while the
   compute identity and AST audit carry the exact authority proof.

Any disagreement still fails closed. The smoke is not rerun and never enters
the scientific denominator. Navigation code, hub code, model weights, causal
history, population, nodes, seeds, thresholds, budgets, and the exact six
repair indices remain unchanged.

After the repaired gate passes, the same six indices are replayed on their
frozen factual-B source nodes with at most two concurrent GPUs. A terminal
integrity barrier requires all 22 factual-C receipts and byte-identical hashes
for the retained 16 before population sealing. The existing node-affine B2
smoke, paired evaluation, aggregation, and independent verifier then resume.

The study remains an underpowered 22-history/15-scene external
continual-memory mechanism test. It cannot be relabelled as the failed
40-history powered confirmation.
