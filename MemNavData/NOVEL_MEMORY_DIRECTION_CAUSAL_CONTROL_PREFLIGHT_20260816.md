# Novel Memory-Direction Causal Control — Preflight

**Date:** 2026-08-16

**Result status:** no causal-control outcome has been read

**Authority:** consumed-development mechanism analysis only; never paper
confirmation or method/threshold selection

## 1. Why the population contract was amended

The complete official MP3D inventory has 90 scenes and is exhausted by four
pairwise-disjoint partitions:

| partition | scenes |
|---|---:|
| train40 | 40 |
| consumed development | 20 |
| consumed Attempt-7/Phase-2 | 16 |
| untouched final | 14 |

The original request for at least 20 fresh scene clusters is therefore not
constructible.  The immutable scene-budget receipt is
`.diagnostics/mp3d_scene_budget_20260816/scene_budget.json`, SHA-256
`779e2d7d63faa0f9b9e735680b1d620f04428c11a57ac83158933306b62407ef`.

The outcome-blind amendment assigns:

- the already consumed Phase-2 population to the four-arm causal mechanism
  analysis;
- all 14 untouched scenes to one later prospective mixed-role confirmation.

No result from the consumed control may be presented as fresh-scene evidence.

## 2. Frozen intervention

Every factual episode runs four arms in a frozen balanced rotation:

1. `native`;
2. `raw_factual_history`;
3. `raw_deranged_history`;
4. `raw_randomized_bearing`.

The deranged arm sends donor RGB history only to the MemNav long-term sidecar;
NavDP receives the factual decision-frame FIFO.  Donors form a no-fixed-point
permutation, prefer another history in the same scene, and otherwise match
history length without using goal, pose, retrieval score, or outcome.

The randomized arm preserves the complete factual proposal path and proposal
availability on its own trajectory.  For each finite non-zero proposal it
replaces only the angle using SHA-256 over
`(20260816, scene, episode, plan_index, "random_bearing")`; the controller
radius remains exactly 2.5 m.

## 3. Runtime invariants

One Slurm task owns one episode, one MemNav server process, one NavDP server
process, one GPU and all four arms.  Each arm resets the servers with the same
episode seed.  The task writes a completion only after verifying:

- identical frozen online-A plans and physical prefix across arms;
- identical factual NavDP FIFO identities and RGB hashes;
- factual and randomized arms use the same long-term sidecar RGB history;
- deranged sidecar identity and RGB aggregate differ from factual history;
- the randomized first-decision factual proposal agrees with the factual arm,
  proposal availability is preserved, and the executed radius is 2.5 m;
- every requested diffusion seed equals the server-echoed seed;
- every zero-takeover arm reproduces the native query rollout exactly;
- role and construction metadata are not forwarded to either controller.

Ground-truth goal position is used only after rollout to compute final
geodesic distance.  It never enters the policy request.

## 4. Metrics and independent verification

Each arm records SR, SPL, initial geodesic, path length, steps, final Euclidean
and geodesic distances, plan count, takeover count, fallback count and wall
time.  Frozen contrasts are reported in this order:

1. factual minus randomized;
2. factual minus deranged;
3. factual minus native;
4. deranged minus native;
5. randomized minus native.

Every contrast includes paired gains/losses, exact two-sided McNemar, paired
risk difference and scene-cluster bootstrap 95% CI.  A separate verifier reads
raw metric CSVs and plan ledgers without importing the evaluator or summarizer
and recomputes the counts and paired statistics.

## 5. Pre-execution validation

- 9/9 focused unit tests pass, including the exact frozen random-hash fixture,
  donor permutation, split replay, synthetic aggregation and independent raw
  verification;
- all Python modules compile;
- all shell scripts pass `bash -n`;
- every embedded Python heredoc compiles;
- no whitespace error is reported by `git diff --check` on the new files.

There is no discardable runtime smoke because no eligible scene can be spent
without violating either the formal population or the final14 seal.  Formal
index 0 is therefore staged first as a denominator-retained record.  Only its
successful contract completion releases indices 1..N through an `afterok`
dependency; source and protocol cannot change after index 0 produces an
outcome.

## 6. Implementation surface

- `novel_memory_direction_control.py`: pure intervention and manifest helpers;
- `freeze_novel_memory_direction_control.py`: outcome-blind population and
  donor freeze;
- `eval_novel_memory_direction_control.py`: thin wrapper over the production
  role-pair evaluator;
- `run_novel_memory_direction_control_episode.sh`: paired four-arm runner and
  completion audit;
- `summarize_novel_memory_direction_control.py`: frozen statistics;
- `independent_verify_novel_memory_direction_control.py`: raw-file verifier;
- `slurm_novel_memory_direction_control.sbatch` and dependent summary/verifier
  jobs;
- `submit_novel_memory_direction_control_hpc.sh`: additive immutable source
  bundle, staged formal head and tail submission.

The production NavDP, MemNav, certificate, checkpoint and controller files are
not modified by this experiment.
