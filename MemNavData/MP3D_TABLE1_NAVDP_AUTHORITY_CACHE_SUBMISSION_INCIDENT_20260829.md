# MP3D Table-1 authority/cache repair submission incident

Date: 2026-08-29

This note is an additive execution record. It does not modify the frozen
population, controller, method, thresholds, seeds, query set, budget, success
criterion, or outcome-opening rule in the parent repair protocol.

## What happened

The second NavDP exact repair was submitted successfully as job
`16559033_27`, followed by aggregate `16559034` and independent verifier
`16559035`. The first attempt to submit the joint seal used an `afterok`
dependency containing both the pending NavDP verifier and the already-completed
ViNT verifier `16558669`. Slurm rejected that dependency expression with
`Job dependency problem`. No seal job was created by that call, and the NavDP
repair chain was unaffected.

## Minimal scheduler correction

Before submitting a replacement seal, the completed ViNT verifier was checked
as `COMPLETED 0:0`, its immutable output was hashed without opening its payload,
and the hash was bound into the seal environment:

```text
ViNT verifier job: 16558669
ViNT verifier SHA-256: 9596e85fa78a94070d3cd0e21ce93b5b80207bf1d868db4e7652ca92d116c3bd
replacement joint seal: 16559083
replacement dependency: afterok:16559035
```

The seal therefore waits only for the new NavDP verifier while independently
requiring the already-completed ViNT verifier file to retain the pinned SHA-256.
This is equivalent to the intended two-verifier gate without asking Slurm to
reattach an already-terminal job to a new dependency chain.

The reusable submission script now enforces this order: require the retained
ViNT verifier to be `COMPLETED 0:0`, hash-pin it, submit the seal after only the
new NavDP verifier, and pass the expected ViNT hash to the seal itself.

## Outcome visibility

No success flag, final distance, path metric, SR, or paired contrast was read
while making this scheduler correction. The only inspected fields were job
state/exit code, immutable file hashes, and the previously disclosed failed-cell
exception/runtime-failure fields. Aggregate results remain closed until job
`16559083` completes successfully.

The byte-identical remote submission receipt is mirrored in
`MP3D_TABLE1_NAVDP_AUTHORITY_CACHE_REPAIR_SUBMISSION_20260829.json`.
