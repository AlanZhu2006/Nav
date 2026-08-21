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

## v2 scheduler cancellation and latent dependency-closure finding

The v2 repair `16130514_29` was allocated on `gh014` at 07:13 EDT and was
cancelled by Slurm root after `00:02:02`; accounting records the reason as
`QOSGrpGRES`.  It created no runtime directory, episode artifact, or Slurm log.
The downstream `afterok` chain `16130521--16130541` was consequently cancelled.
This was a scheduler cancellation, not evidence that the atomic transport
repair failed.

Before resubmission, an exact reconstruction of the formal launch
`PYTHONPATH` exposed a second, previously latent infrastructure defect.  The v2
task bundle contained its updated NavDP `policy_backbone.py`, but not the
vendored `depth_anything.depth_anything_v2` sources imported by that file.  The
sources existed below the immutable base bundle's
`NavDP/baselines/navdp/` directory, which was not itself on the top-level
module search path.  The exact remote import therefore reproduced:

```text
ModuleNotFoundError: No module named 'depth_anything'
```

The QoS cancellation happened before this missing transitive dependency could
appear in a task log.  Reusing the v2 bundle unchanged would therefore have
turned a scheduler repair into a later environment failure.

## v3 dependency-closed, serialized retry

The v3 bundle adds only the twelve vendored DINOv2 Python sources already
required by the frozen NavDP backbone.  It adds no model weights and does not
enable a new depth predictor.  A new outcome-blind exact-runtime preflight now
runs with the same Singularity image, MemNav Python 3.10, Habitat Python 3.9,
base bundle, task bundle, LightGlue path, and import ordering as the formal
job.  It fails unless:

- NavDP `policy_agent`, `policy_backbone`, and `depth_anything_v2.dpt` resolve
  inside the new task bundle;
- MemNav policy modules resolve inside the task bundle;
- the certificate runtime resolves inside one of the two verified source
  roots;
- the three Habitat entry points compile under the pinned Python 3.9 without
  writing into either immutable bundle.

The remote receipt reports `verified=true` and records every resolved module
path.  Its SHA-256 is
`19bbfc6477c7daa34b909bae37ed43fe91fcc8b38f2ee719740cdf217716d567`.

New immutable bundle:

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_fullmono_transaction_repair_27132081e26acfb9
```

Bundle receipt SHA-256:
`deea747eb7c8aad79a3dd76ab6fab6542ad987d1b768d4b7453e30032200da2e`.

The retry is deliberately serialized behind Lifelong task-5 repair
`16130123`; it cannot request a GPU while that frozen repair is still waiting
or running.  The submitted chain is:

| stage | job |
|---|---:|
| Goal-A index-29 repair | `16131068` |
| artifact barrier | `16131071` |
| role-pair construction | `16131072` |
| population finalizer | `16131074` |
| 80-step query smoke | `16131081` |
| formal paired query | `16131090` |
| summary | `16131095` |
| independent verifier | `16131098` |

At submission every job is pending by deliberate dependency.  No query arm
has run and no new SR exists.  The retry still changes no method parameter,
model/depth output, frozen population, seed, budget, or success criterion.
