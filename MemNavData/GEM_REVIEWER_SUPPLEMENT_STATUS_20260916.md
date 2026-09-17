# Reviewer supplement status

## Four local histories verified; automatic continuation, 2026-09-16 07:16 UTC+08:00

The original 13 histories plus four locally completed histories now give **17/70 verified histories and 85/350 verified rollouts**. All local raw results were imported into their original HPC task directories with plan, archive, completion and independent-verification hashes checked. The remote reducer independently confirms these totals.

| Arm | Interim successes / verified histories |
|---|---:|
| Native controller | 5/17 |
| Fixed half turn | 12/17 |
| GEM without explicit turn | 10/17 |
| Initial memory-directed turn only | 10/17 |
| Full GEM | 15/17 |

Full GEM versus fixed turning has five paired wins and two losses (exact McNemar p = 0.453125). None of the three primary active-control contrasts is significant after Holm correction in this incomplete subset. Do not infer a decisive success advantage of continued bearing or select subsequent histories by these outcomes.

The bounded local queue finished tasks 0–3 at 06:27 and then became idle. A serial automatic worker was started at 07:15 to continue the original frozen population. It claims the lowest remaining PENDING array index, cancels only while PENDING, confirms exclusive ownership and no HPC outputs, transfers byte-identical inputs, executes the unchanged five-arm runner, verifies and imports the full result, then claims the next task. It stops on infrastructure/audit failure rather than rerunning trajectories. GPU task 4 is now running locally; HPC retains 52 tasks (indices 5–56), still pending on QOSGrpGRES at this snapshot. Studies B/C and the real-world stack remain stopped.

No experiment population, model source, checkpoint, intervention, arm order, threshold, action budget or metric changed. Every history runs all five arms on one GPU. All eight local native/full success labels agree with their main-table references; trajectories/action counts are not universally bit-identical across hardware and dependency environments. Local hardware provenance and detailed reference differences are retained.

The four completed local five-arm tasks took 347–580 seconds each (mean 459 seconds). Approximately seven hours for all 53 remaining histories is a rough serial estimate based on that small observed subset, excluding variable input/transport costs; concurrent HPC starts can shorten it.

Evidence: `.diagnostics/gem_reviewer_p0_20260915/local_paper_001/broker_preflight_summary.json`, `result_0.json`–`result_3.json`, `auto_status.json`, `auto_worker.log`, and remote `local_transfers/`. The automatic worker is PID 3556901; its owned remote broker is PID 762671. Read live process receipts for subsequent changes. Remote summaries use separate timestamped output directories to avoid writing concurrently with the original Slurm finalizer.

## First local task verified; continued allocation, 2026-09-16 06:05 UTC+08:00

The first local task passed the unchanged five-arm independent verifier in 396.47 seconds. Its result is native 0 (383 actions), fixed half turn 1 (106), GEM without explicit turn 1 (92), initial guided turn only 1 (103), and full GEM 1 (99). This single additional history does not isolate a success advantage of continued bearing over initial turning. Keep inference at the full predeclared population.

Native/full success labels, SPL and turn counts agree with the original main-table references. Native differs by four actions and +0.18556 m travelled; full GEM has the same 99 actions and a -0.0000155 m path difference. The local execution is independently valid and same-GPU paired, but is not a bit-identical A100 reproduction. Preserve local hardware provenance and these measured differences.

All first-task raw artifacts and verification receipts have been imported into HPC `gem_reviewer_20260916_001/tasks/0` with archive and completion hashes verified. Original task indices 1, 2 and 3 were then removed from the pending array for local serial execution. These ascending indices were selected before the first result, and continuation requires the independent verifier, not any success outcome. Index 1 is running locally; indices 2 and 3 follow automatically. HPC retains indices 4–56 (53 histories / 265 rollouts), plus its original CPU finalizer. Studies B and C remain cancelled.

Slurm may remove an unstarted array element without leaving an individual `scontrol`/`sacct` row. Continuation receipts therefore preserve the pending-state check, pending-only cancellation, empty post-cancellation queue and absence of task output; they do not invent a zero-time accounting record. Local ownership is checked before each task starts.

Local verified-result export and an HPC collector are running for indices 1–3. They copy unchanged raw artifacts into their original task directories only after full verification, then create a separate `combined70_after_local4` summary to avoid concurrent writes with the original finalizer. Transport uses owned loopback HTTP endpoints over the existing SSH master. Process receipts and logs live in local `local_paper_001/` and remote `local_transfers/`; preserve the forwards while collection is active. The first task had one earlier dependency-startup failure with zero navigation rollouts, preserved separately.

## Local execution started, 2026-09-16 05:53 UTC+08:00

The user authorized local execution and stopping the real-world services. The documented `stop_policy_stack.sh` command stopped the managed parked `cec-realworld` session and released ports 18888/8888; GPU use dropped from approximately 7.6 GiB to 256 MiB. Recordings and unrelated sessions were retained.

Extension task 0 (paper index 0, X7gTkoDHViv/episode_0002) was cancelled while pending on HPC, with zero elapsed execution and no task output. All five arms now execute on the local 48 GB RTX 4090. The original array retains tasks 1–56, and its finalizer remains pending. Studies B/C remain cancelled.

The local runner uses a private filesystem namespace with the original absolute paths, original plan bytes and all 1,244 pinned source files. Input hashes, all three checkpoint hashes, and external LingBot/LightGlue source hashes match HPC. A missing unpinned NavDP dependency directory was copied from the HPC bundle; the preceding startup failure consumed zero navigation rollouts and is preserved as infrastructure evidence. Python package version differences are recorded. Do not assume bit-identical cross-GPU trajectories: compare reference metrics and keep hardware provenance.

Evidence and owned process receipts: `.diagnostics/gem_reviewer_p0_20260915/local_paper_001/`. The first five-arm task must pass the original independent verifier before more local tasks start; preparation of additional input files alone does not transfer their scheduler ownership.

## Scheduler diagnosis, 2026-09-16 05:38 UTC+08:00

Only study A remains submitted. Array 17865019 is still PENDING with QOSGrpGRES; finalizer 17865543 is waiting on its dependency. No new A task has started. Studies B and C remain cancelled.

The binding reported limit is the A100 partition's aggregate GPU quota: runtime `scontrol show assoc_mgr flags=qos` reports `QOS=a100_tandon`, `gres/gpu=60(60)` (limit 60, allocated 60), across 57 running jobs. Its CPU and RAM limits have spare capacity. The job-level `gpu48` QOS has no aggregate GPU cap and a 16-GPU per-user cap; our user has zero GPUs running. The earlier physical-node snapshot also showed 167 of 168 GPUs on healthy A100 nodes allocated.

Each array element requests one A100, 12 CPUs, 128 GB RAM and 40 minutes. The array throttle of 12 is a maximum concurrency, not a requirement to allocate 12 GPUs together. The `gpu48` QOS name does not mean that each task requests 48 hours. Completed reference tasks took 8.2–22.7 minutes each; reducing the 40-minute request does not remove the observed shared 60-GPU cap. Slurm can have additional scheduling constraints beyond the single displayed reason.

No resource requests, source bundles, or scheduler jobs were changed during this diagnosis. Other GPU partition QOS entries have some quota headroom in this snapshot, but that does not establish available physical GPUs, immediate start, or validated execution parity. A100 was retained because the completed controls reproduced the original results there; an earlier H100 attempt was stopped for abnormal runtime before producing outcomes.

Receipts: `.diagnostics/gem_reviewer_p0_20260915/hpc_002/scheduler_limits_20260916.json` and `scheduler_runtime_qos_20260916.json`. Slurm semantics: https://slurm.schedmd.com/job_reason_codes.html and https://slurm.schedmd.com/qos.html .

## User update, 2026-09-16 04:50 UTC+08:00

The user also cancelled study B (full versus recent archive access). Cancelled automatic launcher 17865545 first, then GPU gate 17865409; neither had started, no task directory exists, and no remaining-pair or finalizer job had been submitted. Both studies B and C are now cancelled and must not be restarted without a new user request. Their source bundles, plans and receipts are retained.

Only study A remains active: 57 additional histories × five arms = 285 planned rollouts, GPU array 17865019, with CPU finalizer 17865543 to combine them with the completed 13 histories / 65 rollouts. At cancellation, A remains pending on QOSGrpGRES. The existing old-memory subgroup analysis and Table II(c) remain the paper's relevant evidence; no manuscript edits were made.

Study B cancellation receipt: `.diagnostics/gem_reviewer_p0_20260915/retrieval_domain_001/cancelled_by_user.json`. The updates below are historical snapshots superseded by this instruction.

## User update, 2026-09-16 04:46 UTC+08:00

The user cancelled study C (matched position readout). Cancelled its automatic launcher 17865936 first, then GPU gate 17865935. Slurm accounting confirms both CANCELLED with zero elapsed execution; no task directory was created and no remaining-pair or finalizer job had been submitted. Preserve its plans, source bundle and cancellation receipt; do not restart it. Active additional planned coverage is now A: 285 plus B: 26, totaling 311 rollouts. A and the B first pair remain queued for GPU quota.

The user's question about prior old-memory evidence was checked against the active manuscript. The paper already includes the 13-query low-recent-overlap analysis (NavDP 3/13 to 11/13, ViNT 0/13 to 8/13, NoMaD 4/13 to 7/13), as well as Table II(c)'s shared two-leg-history Revisit comparison (8/20 to 17/20). Study B adds a within-GEM archive-access intervention on the same 13 queries. It is additional mechanism evidence, not the first evidence for recall beyond recent observations, not a new independent cohort, and not an unconditional publication requirement. Study B has not been cancelled. No manuscript changes were made.

Cancellation receipt: `.diagnostics/gem_reviewer_p0_20260915/position_readout_001/cancelled_by_user.json`.

## Original submission snapshot, 2026-09-16 04:02:39 UTC+08:00

This is a timestamped execution snapshot. Read the submission receipts and live scheduler for later state. No manuscript or Overleaf changes were made in this implementation phase. No local GPU was used.

## Scope

Freeze the main-table NavDP condition (legacy/W32 geometry, canonical monocular depth, strict certificate, original controller/checkpoints and original shared histories/goals). These experiments attribute the existing reported results; they do not train or select a new memory architecture. Simulator ground truth is restricted to evaluator diagnostics.

| Study | New planned rollouts | Submitted GPU job | State at snapshot |
|---|---:|---|---|
| Remaining 57 main-table Revisit histories, five control arms | 285 | 17865019, array 0–56, max 12 concurrent | Pending QOSGrpGRES; none started |
| Full archive versus last seven historical decision observations, 13 histories | 26 | 17865409, first pair only | Pending QOSGrpGRES; none started |
| PnP goal position versus same verified historical camera, seven offset queries | 14 | 17865935, first pair only | Pending QOSGrpGRES; none started |

There are 325 new planned rollouts. Currently submitted GPU tasks cover 289 of those rollouts (285 + 2 + 2). The remaining 36 are configured for automatic submission only after their corresponding first pair passes independent verification. These first pairs are part of the planned totals, not additional repeated experiments.

## Completed evidence retained

The previous 13 histories / 65 executions remain unchanged and will be merged with the 57-history extension to form 70 histories / 350 executions for control attribution. These are existing main-table histories, not 70 fresh independent scenes. The combined population covers 46 scenes.

| Arm | Previously verified Revisit success |
|---|---:|
| native | 3/13 |
| fixed_half_turn | 8/13 |
| gem_no_turn | 7/13 |
| initial_turn_only | 6/13 |
| full_gem | 11/13 |

The small completed subset does not yet establish a significant advantage over the active control variants. Report all 70 paired histories, paired discordances and scene-cluster uncertainty when complete; do not stop or change membership based on interim success rates.

## Diagnostic boundaries

The archive comparison uses the already identified 13-history old-support subgroup. “Seven” means seven prior controller decision observations, not seven raw frames or the last 64 raw frames. Restriction occurs before temporal candidate suppression and is audited at the actual geometric read. Both arms retain the same causal streaming geometry and dense depth. This tests old external-archive access conditional on shared geometry; it is not a claim that all older neural state has been erased.

The position comparison includes all seven previously accepted main-table queries with at least 1 m offset between the selected historical camera and the goal. Both arms retain the same shortlist, selected support, SP/LightGlue/PnP certificate and original goal image. Only the accepted bearing target differs; historical camera positions are predicted, not oracle. Both execute with a shared 0.3 m Euclidean arrival threshold, the original 600-action budget and original termination behavior. Measure first passage at 1.0/0.5/0.3 m Euclidean and geodesic radii on the new trajectories, endpoint error and bearing error. This is a post-hoc conditional diagnostic with seven pairs, not an autonomous STOP benchmark or fresh generalization study.

## Verification and continuation

- Seven CPU retrieval-domain tests passed, including masking before temporal suppression, unchanged full-domain shortlist, cache binding and actual-read guards. Position-runner syntax, shell syntax and pending-state aggregation checks passed. GPU execution correctness remains to be checked by the submitted first pair.
- All three studies have independent source bundles and hash-bound plans. Seven position-query inputs and all 1,258 pinned source files were verified on HPC before submission.
- Job 17865543 reduces old 13 + new 57 after the attribution array finishes, even if some tasks fail, preserving explicit incomplete coverage.
- Job 17865545 checks the retrieval-domain first pair, requires exact original full-history reproduction, then submits the remaining 12 pairs (max three concurrent) and their finalizer.
- Job 17865936 checks the position-readout first pair, requires matching initial evidence and independently verified consumed bearings, then submits the remaining six pairs (max three concurrent) and their finalizer.
- Infrastructure/audit failures remain distinct from valid navigation failures. Failed first-pair checks stop expansion and preserve raw artifacts; there is no blind automatic rerun or alternate controller path.
- Latest scratch snapshot: 2.16 TB / 5 TB and 4,732,031 / 5,000,000 files. Continue monitoring file count during execution; no existing outputs were deleted.

## Evidence locations

- `.diagnostics/gem_reviewer_p0_20260915/hpc_002/`
- `.diagnostics/gem_reviewer_p0_20260915/retrieval_domain_001/`
- `.diagnostics/gem_reviewer_p0_20260915/position_readout_001/`
- `.diagnostics/gem_reviewer_p0_20260915/supplement_live_status_20260916.json`

Each directory holds the exact plan, source digest list, local/remote preparation receipts and actual submission receipts. Historical preparation records saying “not submitted” describe the state when those records were created; the later submission receipts are authoritative.
