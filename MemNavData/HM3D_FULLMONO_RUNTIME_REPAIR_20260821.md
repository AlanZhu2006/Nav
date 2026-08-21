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
