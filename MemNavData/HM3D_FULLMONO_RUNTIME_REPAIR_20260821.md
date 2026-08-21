# HM3D Full-Mono runtime repair — 2026-08-21

## Scope

This amendment repairs missing Goal-A collection records in the frozen HM3D
Full-Mono mixed-role run.  It does not change the parent manifest, scene or
episode population, controller, seeds, checkpoint, depth arm, budgets,
construction rule, query arms, thresholds, or success criterion.

Frozen run root:

`/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6`

## Incidents

- Scene index 29 stopped on episode 3 after a single HTTP 409 sidecar JPEG
  digest mismatch.  Episodes 0–2 had already produced complete raw receipts;
  episode 3 left only an empty output directory.
  The failure happened during Goal-A source collection, before role-pair
  construction or any query outcome.
- Scene indices 46, 47, and 53 were cancelled by Slurm with `QOSGrpGRES`
  before creating runtime or episode output.

The original Goal-A array therefore cannot satisfy its `afterok` dependency,
even after all other tasks finish.

## Repair contract

1. Snapshot every pre-existing episode artifact and its SHA-256 before repair.
2. Run only missing episodes for indices 29, 46, 47, and 53, with the original
   source root, manifest, protocol, checkpoints, seeds, and controller.
3. Never overwrite a pre-existing file.  Re-audit every non-empty episode
   directory from raw trace, plans, and metric receipts.  A failed empty
   directory may only be filled additively by its originally assigned episode.
4. Use a distinct runtime-attempt directory and immutable repair bundle.
5. After the original array reaches any terminal state and the repair array
   succeeds, run a CPU barrier that independently checks all 54 scene
   completions, all 196 constructible source episodes, zero metric-depth reads,
   and preservation of the pre-repair hashes.
6. Only the barrier may release fresh construction and query dependencies.

The one-off JPEG mismatch is treated as a transient transport incident unless
it reproduces.  This repair does not alter the MemNav/NavDP depth contract.  A
repeat at the same endpoint is a stop condition requiring a separately tested
transactional transport amendment.

## Reproduced mismatch and transactional amendment

The stop condition was reached.  Repair array `16126593` completed indices
46, 47, and 53, but index 29 reproduced the same HTTP 409 on the same frozen
episode.  The second attempt successfully served the first 20 monocular depth
queries and then rejected the query following observation append 160.  The
MemNav buffer contains all 161 causal JPEGs, so this is not a missing-frame or
model-startup failure.  The old wire protocol nevertheless used two separate
HTTP transactions: append the current JPEG, then ask NavDP to query whatever
the sidecar regarded as latest.  Its error response was not propagated by the
NavDP server, so the surviving logs cannot identify which request carried the
divergent digest.

The v2 amendment removes that ambiguity without changing navigation:

1. a planning append materializes the unchanged LingBot depth in the same
   `/memory_step` transaction;
2. its JPEG SHA-256, frame index, depth-PNG SHA-256, and scale-receipt SHA-256
   are bound into one immutable token;
3. the evaluator verifies the append JPEG against its exact outbound bytes;
4. NavDP can read only the token-bound payload, and rejects a missing,
   superseded, mutated, wrong-frame, or wrong-image transaction;
5. non-planning appends invalidate the previous token, preventing stale reads.

This moves no model computation, changes no depth values, and changes no
checkpoint, threshold, seed, budget, controller, population, or success rule.
It replaces an ambiguous two-call "latest frame" transport with a fail-closed
frame-addressed transaction.  Unit tests cover image, frame, depth, and token
mutation, and the formal repair remains additive at scene index 29 only.

Submission receipt:
`HM3D_FULLMONO_TRANSACTION_REPAIR_SUBMISSION_RECEIPT_20260821.json`.
The submitted chain is repair `16130514`, barrier `16130521`, construction
`16130525`, finalize `16130526`, smoke `16130528`, formal evaluation
`16130535`, summary `16130538`, and independent verification `16130541`.
At submission the repair is pending on `QOSGrpGRES`, not failed or cancelled;
the already-running throttled Lifelong NNR task owns the available group GPU.
