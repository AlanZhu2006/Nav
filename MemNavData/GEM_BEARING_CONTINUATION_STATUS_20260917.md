# Frozen 70-query bearing study: continuation status

Updated 2026-09-17, China time.

## Current reporting decision

The user has selected the existing verified **53-query reporting cutoff** for
this manuscript revision and withdrawn the request to continue to 70. No local
GPU job is active, and no further collection is scheduled for this revision.
All existing records, including the additional verified index-36 result, remain
archived. The 53-query cutoff is not a claim that only 53 results exist or that
these queries were a prespecified subset. No remote queue was modified or
rechecked for this decision. No manuscript file was edited during this review.

The recovery record below documents work completed before this decision.
The original population, five-arm execution protocol, source snapshot,
checkpoints and pairing rules were unchanged during that recovery.

## Verified local recovery

Extension index 36 (paper index 101, MP3D `r47D5H71a5s/episode_0005`) completed
all five arms and passed the original independent verifier. The runner exited
with code 0. Its four previously completed arms reproduce the interrupted
attempt's success labels, action counts and path lengths. The interrupted
attempts and both user-requested pauses remain archived.

The complete resumed pair is the only version eligible for aggregation.
Native and Full GEM success labels match the original main-reference labels;
execution metrics are not identical to that earlier reference, as recorded
in `result_36.json`.

Combining this new pair with the previously verified 53-query reporting
snapshot gives **54 locally verifiable queries / 270 rollouts**. This is not
a live remote completion count. The original 13-query subgroup is preserved.

The earlier 53-query snapshot comprises **42 HPC queries and 11 local 4090
queries**. Each query's five arms used the same physical GPU. The local
worker downloaded inputs one query at a time; it did not cache all 70 queries.
The new index-36 result raises local executions to 12 and known coverage to 54.

## Local execution readiness and missing inputs

The local 4090, model checkpoints, frozen source snapshot and all nine scene
assets for the remaining 16 queries are available. Searches of the research
workspace, ignored diagnostics, `/data/diagnostics/nav-graph-blind` and
`/data/archive` did not locate those queries' complete frozen inputs. The
available pose export does not include the original goal JPEGs, full history
files, history RGB frames or input receipts.

`remaining16_input_requirements_20260917.json` records the exact identities,
source directories and expected hashes. Obtaining these inputs once permits
local execution; remote compute capacity is not required. No additional GPU
job has been started with incomplete or reconstructed inputs.

The original `alantorch` SSH endpoint currently times out on TCP port 22.
The shared control socket is absent; both ordinary and PTY probes failed to
connect. No remote jobs, broker state or result directories were modified.
The remote status of the other 16 identities from the old reporting cutoff
has not yet been retrieved.

The existing Globus CLI login belongs to a different account from this
experiment. No Globus transfer, login switch or shared configuration change
was made.

If continuation is requested again, the recovery procedure is to import the
verified index-36 result through the original collector, obtain a live
queue/result inventory, and run the original combined reducer on the 13-query
and 57-query runs. Only pending identities with confirmed exclusive ownership
may be claimed for local execution. This procedure is inactive under the
current 53-query reporting decision. Studies B and C remain canceled.

## Records

Local root:
`.diagnostics/gem_reviewer_p0_20260915/local_paper_001/`

- `current_continuation_status.json`: continuation status.
- `prior53_execution_audit_20260917.json`: verified HPC/local breakdown.
- `remaining16_input_requirements_20260917.json`: exact missing inputs.
- `result_36.json`: independently verified completed pair.
- `tasks/36/completion.json`: completion hashes and execution binding.
- `tasks/36/infrastructure_recovery/`: interruption and recovery provenance.
- `resume_20260916_235648/resume_status.json`: successful worker state.
- `resume_20260916_235648/local_known_coverage.json`: 54-query local coverage.
- `resume_20260916_235648/coverage_reconciliation.json`: original 70 identities
  reconciled with the earlier 53-query cutoff.
- `exports/index_36.json`: original collector's transfer envelope after export.

Frozen extension-plan SHA-256:
`c00888e6d4d0ccf1417ea853c2997bb3d63e8dd21c70c380280f9f761b76fb64`.
