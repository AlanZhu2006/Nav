# HM3D continual-memory factual-C repair attempt 2

Frozen on 2026-08-28 after inspecting only Slurm state, startup logs, file
presence, and provenance. The successful factual-C navigation outcomes remain
unread, no attempt-1 navigation outcome exists, and B2 has not started. The
machine-readable authority is
`hm3d_lifelong_underpowered_collect_repair_attempt2_20260828.json`.

## Why attempt 1 stopped

The first exact-repair bundle correctly carried the queue-contract and
node-affinity amendments, but its runtime dependency closure was incomplete.
The Slurm arm launched the repaired overlay runner while still launching
`cec_controller_portability_hub.py` from the older frozen task bundle. The
overlay runner supplied `--reject-policy shared_native_exact`; that older hub
does not declare this CLI option. Jobs 16509621, 16509634, and 16509637
therefore exited with code 2 during hub argument parsing, before the evaluator
or any navigation step began. Their indices were 0, 7, and 11. The other lane
and all downstream jobs were still dependency-held, so jobs 16509627,
16509636, 16509642, 16509644, 16509648, and 16509649 were cancelled exactly at
2026-08-28 11:08:55 EDT. No running task was killed.

This is a runtime packaging defect, not a new scientific failure. The legacy
hub SHA-256 is
`fb249dae4f865bd17e12bd1673156403651592b870ac1a216de1ffca3d36b7d5`.
Although it lacks the CLI flag, its `ComparisonPlan` literally hard-codes
`reject_policy="shared_native_exact"` and `fallback_controller="navdp"`—the
same requested semantics.

## Minimal closure

Attempt 2 does not replace or edit the frozen hub. A dependency-free AST
auditor, `cec_hub_cli_compat.py`, inspects the exact hub source before model
startup:

- A hub that declares `--reject-policy` receives the frozen policy explicitly.
- A legacy hub may omit the redundant option only when one and the same
  `ComparisonPlan` literally proves `shared_native_exact` plus the NavDP
  fallback.
- A different policy, an unproved legacy plan, or a legacy authority/direction
  handoff mode fails closed before model startup.

Thus no controller, hub behavior, weight, observation, action, memory entry,
seed, benchmark, population, navigation threshold, or certificate threshold is
changed. The attempt-1 queue-contract patch and factual-B node-affinity map are
unchanged.

## Preservation and staged gate

The 16 originally completed factual-C directories remain byte-identical to the
pre-repair fingerprint ledger. The six original failed partials remain in the
first immutable incident archive. The three attempt-1 startup directories
(indices 0, 7, and 11) are moved—not deleted—to a second immutable archive;
indices 1, 9, and 13 are proved absent because they never started.

Before any scientific retry, index 0 is run once into a separate smoke root on
its frozen source node `gh005`. A CPU verifier checks only structural
completion, checksums, compute-node identity, hub health, the legacy CLI
contract, and the shared-native policy. It does not read success, SPL,
distance, or any navigation decision, and the smoke output never enters the
scientific denominator.

Only after that verifier succeeds do two deterministic lanes retry the exact
six scientific indices on their already-frozen source nodes. Each lane uses
`afterok` and invalid-dependency cancellation to fail fast; a terminal CPU
barrier still runs after both lane tails and refuses sealing unless all 22
factual-C items are structurally complete and the retained 16 fingerprints are
unchanged.

Population sealing then resumes the existing node-affine B2 launcher: B2 has
its own one-item true-stack smoke, two GPU lanes, aggregation, and independent
raw-file verification. Regardless of outcome, this 22-history/15-scene study
remains an underpowered continual-memory mechanism experiment and cannot be
reported as powered confirmation.
