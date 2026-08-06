# Probabilistic Memory + Reverse-Graph Experiment (2026-08-06)

## Starting point

The frozen 20-scene, 40-episode two-leg benchmark produced:

- native NavDP joint SR: `4/40 = 10.0%`;
- DINO + SIFT/RANSAC geometry router joint SR: `19/40 = 47.5%`;
- Goal-A (Novel) success: `31/40 = 77.5%`;
- Revisit success conditioned on Goal-A success: `19/31 = 61.3%`.

The current benchmark therefore has a `31/40` Novel ceiling.  Improving the
memory system cannot recover the nine episodes in which the frozen Novel
controller never reaches Goal A.

## Failure audit

The 12 Goal-B failures among the 31 eligible episodes split as follows:

- seven failures never activate the memory route;
- five failures activate it but do not reach Goal B;
- among the five active failures, at least three have a likely wrong or weak
  anchor (for example, nearest Goal-B frame `26`, selected frame `243`);
- the remaining roughly two failures have a plausible anchor but the direct
  image-conditioned point goal does not yield successful local control.

The inactive failures also show why lowering one hard threshold is unsafe:
some high-co-visibility candidates are rejected by the fixed match/inlier
rules, while several low-co-visibility candidates already pass those rules.

## Architectural change

The intended non-oracle pipeline is:

1. DINO proposes temporally diverse memory nodes.
2. A K+1 set localizer jointly ranks candidates and assigns probability to an
   explicit `no match` node.
3. High-confidence LingBot point-cloud/pose consistency becomes a learned loop
   factor rather than a Boolean RANSAC switch.
4. Once localized, the controller follows short nodes along the recorded
   LingBot pose chain in reverse.
5. After reaching the localized memory anchor, the original image-conditioned
   LingBot goal pose performs final alignment.
6. If no-match probability is high, the system falls back to frozen NavDP
   ImageGoal without changing its Novel policy.

This separates localization, uncertainty, global/topological planning, and
local collision-aware control.  It uses no Habitat pose or episode phase at
inference time.

## Local set-localizer result

`train_neural_set_localizer.py` was run locally on the exact frozen feature
cache from job `15315411`:

- 40 training scenes;
- eight scene-disjoint internal tuning scenes selected only from training;
- ten held-out development scenes evaluated after configuration freeze;
- 118 held-out sessions, 35 with a co-visibility positive;
- 250-epoch maximum with early stopping and three-seed ensemble.

Held-out result:

- candidate recall@1: `30/35 = 85.7%`;
- joint localization accuracy: `101/118 = 85.6%`;
- match ROC-AUC: `0.927`;
- Brier score: `0.095`.

This is better calibrated than the old scalar gate and far above raw DINO
ranking (`24/35`), but it does **not** yet beat the existing linear listwise
ranker (`30/35`) on candidate top-1.  It remains explicitly marked
`deployment_approved=false`; more epochs alone are not evidence for replacing
the geometry fallback.

## Paired closed-loop ablation

`run_graph_router_ablation.sh` evaluates three configurations against the
immutable direct/gap-16 reference, with identical scenes, episodes, seeds,
checkpoints, and success criterion:

- `direct_gap4`: isolates better temporal candidate coverage;
- `graph_gap16`: isolates reverse-memory graph subgoals;
- `graph_gap4`: measures their composition.

The graph spacing is `1.25 m` and the arrival radius is `0.60 m`.  Zero graph
spacing is tested to be an exact direct-goal fallback.  The report includes
Novel SR, conditional Revisit SR, joint SR, activation, per-episode paired
transitions, and exact McNemar tests.

The first development run produced a strong but still preliminary signal:

| controller | Novel | Revisit given Novel | joint |
|---|---:|---:|---:|
| direct, gap 16 | `31/40` | `19/31 = 61.3%` | `19/40 = 47.5%` |
| graph, gap 16 | `31/40` | `25/31 = 80.6%` | `25/40 = 62.5%` |

The six paired gains and no paired losses give an exact two-sided sign-test
`p=0.03125`.  This is evidence that intermediate reverse-memory subgoals can
help local control.  It is not yet the final number: the old evaluator seeded
only once at episode reset, so a different number of earlier diffusion calls
could give later arms different DDPM noise.  The gap-4 arm changing Novel
outcomes exposed that confound even though graph-gap16 happened to retain the
same `31/40` Novel count.

## Strict causal re-evaluation

The replacement protocol removes that confound instead of averaging it away:

1. Run Goal A once with the automatic direct-gap16 baseline router.  This
   makes the reference arm's Novel trajectory factual and the graph arm a
   controlled Goal-B intervention.
2. Save every simulator pose and the SHA256 of its rendered JPEG.
3. For direct and graph arms, re-render every saved pose and require an exact
   JPEG hash match before rebuilding LingBot's full streaming memory.
4. Restore NavDP's eight-frame observation queue only at the original policy
   decision frames, using a replay endpoint that never samples diffusion.
5. Seed every actual diffusion request by `(episode, leg, plan index)` and
   require the server to echo the same seed.
6. Reject the summary unless both arms have the same Goal-A trace hash, Goal-A
   outcome, steps, path length, SPL, and final distance.

This means direct versus graph differs only after the shared Goal-A trajectory
and receives matched DDPM noise at corresponding Goal-B replans.  The strict
development runner is `run_strict_graph_2leg.sh`.

Because LingBot streaming, rather than NavDP diffusion, dominates wall time,
the full runner supports scene-level multi-GPU execution inside one Slurm job.
Each worker receives one visible GPU and uses scene-specific ports, temporary
directories, server processes, and output directories; a failed worker makes
the whole job fail before summary generation.  `SCENE_WORKERS=1` remains the
default.  A formal run may request four GPUs and set `SCENE_WORKERS=4` without
changing any model, seed, episode, or metric.  Smoke mode intentionally uses
one episode from one explicitly selected scene.  It defaults to
`cV4RVeZvu5T/episode_0000`, an existing development episode whose memory route
activated and whose graph arm recovered a direct-arm failure.  The smoke now
fails closed unless both control arms activate memory, share the exact Novel
trace and per-request diffusion seeds, and graph conditioning changes at least
one paired active point-goal.  This avoids treating a short, retrieval-ineligible
episode as evidence that the graph execution path works.

For three-leg diagnosis, `run_graph_conditional_c.sh` replays the exact causal
A/B source prefix into both LingBot and NavDP, then evaluates six logical arms:
native, direct-gap16, graph-gap16, oracle-anchor direct, oracle-anchor graph,
and oracle point-goal.  This separates retrieval-anchor error, graph-control
error, and metric point-goal/control error.  It is a conditional diagnostic,
not an end-to-end three-leg SR claim.

These 20 scenes are now a development ablation because their failures were
inspected.  `build_graph_blind_manifest.py` deterministically selects every
remaining eligible unseen scene using a frozen hash salt, validates two
complete episodes per scene, and records hashes for assets, metadata, parquet,
and goal images.  The resulting 16-scene/32-episode manifest must be generated
and committed before any blind performance result is inspected.  It is not a
blind result until the strict development rerun passes and that one-shot run is
complete.

That freeze is now materialized as
`strict_graph_blind_20260806.json`, SHA256
`b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9`.
No blind navigation result had been run or inspected when this hash was
committed.

## LingBot-native loop-factor expansion

The 50-row feasibility run showed that LingBot-native geometry contains useful
verification signal:

- DINO AUC: `0.546`;
- LingBot cloud-overlap AUC: `0.729`;
- LingBot pose-consistency AUC: `0.700`;
- per-session top-1: DINO `8/13`, cloud overlap `11/13`.

The long expansion samples up to 100 development/training sessions with
balanced hard positives and negatives.  It is a feature/data collection job,
not a claimed closed-loop improvement; its output is intended to train and
calibrate a probabilistic loop-factor head.

The completed expansion contains 93 candidate rows from 25 sessions and 22
scenes.  Cloud overlap remained stronger than DINO at candidate verification
(ROC-AUC `0.743` versus `0.610`).  A read-only exploratory scene-LOSO logistic
fusion of DINO, cloud overlap, pose consensus, and refinement reached candidate
ROC-AUC `0.776`, AP `0.825`, and session top-1 `23/25`, compared with DINO
`18/25`.  This is a development-only structure signal, not a deployable or
blind result.

That collection also exposed two limits in the original diagnostic:

- every retained session was deliberately forced to contain both a positive
  and a negative, so it could not measure a true Novel/no-match decision;
- the CSV discarded the inferred `goal_pose`, so metric translation,
  direction, and rotation errors could not be measured against NavDP labels.

`diag_lingbot_goal_loop_closure.py` now has a separate `deployment` sampler.
It keeps temporal-diverse top-DINO candidate sets, labels strict-positive,
strict-no-match, and ambiguous sessions separately, and enforces an optional
frozen scene role.  It also reconstructs the exact axis-fixed NavDP target from
the candidate episode parquet and query pose, applies per-episode
ground-anchored LingBot scale (with an explicit pooled fallback), and records:

- predicted and target relative point-goals;
- metric position, bearing, distance, and camera-rotation errors;
- the complete inferred goal pose for each nearby-anchor hypothesis;
- the candidate episode path and set-level no-match status.

For cross-episode queries, replay now derives the RGB stream from
`candidate_path`, not `query_path`.  This did not alter the earlier 93-row
`revisit_b` result because those queries and candidates were from the same
episode, but it is required before collecting cross-episode no-match data.
The pure pose/axis, set-label, cross-episode path, and split-leak checks pass as
part of the full 116-test `MemNavData` suite.  A local dependency preflight also
passed against the real 4.4-GB LingBot weight (SHA256
`832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`),
real paired caches, parquet, metadata, and raw RGB data.  No new GPU collection
or closed-loop score is claimed until the scene-disjoint smoke and full job
finish.

## Dependency and provenance checks

Every submitted task must fail before GPU model allocation unless all of the
following hold:

- exact child-repository commit;
- clean task-file diff against that commit;
- frozen manifest and split SHA256;
- exact teacher CSV and feature-cache SHA256;
- exact NavDP and MemNav checkpoint SHA256;
- exact LingBot weight SHA256 and LingBot repository commit;
- readable MP3D asset/episode roots;
- pinned Habitat and MemNav Python imports;
- focused unit tests, Python compilation, and shell syntax checks.

No source file under `/home/asus/Research/Nav` is modified.  This strict
protocol is implemented in the clean child worktree
`/home/asus/Research/Nav-graph-blind`; remote runs must use a clean worktree at
the exact committed revision.
