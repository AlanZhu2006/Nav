# MemNav Forward-Only Terminal U-Turn Experiment (2026-08-02)

## Scope and repository state

- Worktree: `/home/asus/Research/Nav-axis-uturn`
- Branch: `feat/memnav-terminal-uturn-20260802`
- Base: `origin/main@b3dfae9` (`flowgate_precompute`)
- The base already contains `0d9466f fix_axis_bug`.
- Neither `/home/asus/Research/Nav` nor the pre-existing dirty
  `/home/asus/Research/Nav-axis-fix` worktree was modified by this experiment.
- Terminal alignment is evaluation/controller logic. It does not require a new
  training run.

The old `align_turn()` rotates at a fixed position and is not used. The new
controller plans a forward-only Dubins path with the same `r_min=0.40 m`
nonholonomic constraint as the generated leg-transition turns. For the common
same-position 180-degree case, it produces mirrored RLR/LRL teardrop U-turns.
Every yaw-changing sample has non-zero translation. If any sample leaves the
inflated navmesh, the candidate fails closed.

## Modes

`eval_2leg_habitat.py --terminal_uturn ...` supports:

- `off`: original distance-only behavior; this remains the default.
- `oracle`: GT goal position and yaw; geometry/controller upper bound.
- `lingbot_yaw`: GT position plus axis-corrected LingBot yaw; isolates rotation.
- `lingbot_local`: current reached position plus LingBot yaw; recommended
  experimental mode because it does not trust aux metric translation.
- `lingbot`: complete LingBot aux position plus LingBot yaw; diagnostic only.

`--terminal_visual_refine` is deliberately independent of those coarse modes:

- `off`: do not run two-view geometry; default and backward-compatible.
- `verify`: estimate and log the residual yaw, but never move.
- `refine`: execute at most one additional forward-only correction, and only
  when the frozen confidence gates and the 8-degree control deadband pass.

The LingBot relative yaw is recovered from `R_rel` with the same rotation basis
correction used by the axis-fixed trainer:

```text
C = diag(-1, -1, 1)
R_corrected = C @ R_rel @ C.T
relative_yaw = atan2(R_corrected[0,2], R_corrected[2,2])
```

This is deliberately separate from the translation axis conversion.

## Local validation

### Unit and static checks

```bash
cd /home/asus/Research/Nav-axis-uturn
python -m unittest -v MemNavData.test_terminal_uturn
python -m py_compile \
  MemNavData/terminal_uturn.py \
  MemNavData/eval_2leg_habitat.py \
  MemNavData/visual_yaw_refinement.py \
  MemNavData/diag_visual_yaw_calibration.py \
  MemNavData/diag_oracle_retrieval_firsthop.py \
  NavDP/baselines/memnav/pose_alignment.py
/home/asus/miniconda3/envs/memnav/bin/python -m py_compile \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py
git diff --check
```

All twelve unit tests pass. They cover axis-correct yaw extraction, NavDP local-to-
Habitat translation, exact 180-degree RLR/LRL geometry, coupled translation and
yaw, fail-closed collision handling, the staged-turn fallback, and metric
denominator semantics. They also verify the direct visual-yaw sign and that its
control gate rejects low-confidence, deadband, and out-of-range corrections.

### Thirty-episode MP3D navmesh scan

Dataset: three scenes x ten existing 2-leg episodes, navmesh recomputed with
`agent_radius=0.30 m`.

- Natural recorded arrival pose -> exact goal-image pose: `26/30` feasible.
- Exact same-position 180-degree reversal at B: `23/30` feasible.
- Current-position heading loop: `26/30` feasible.
- Simple 0.6-2.5 m forward staging fallback: still `26/30`; it added no coverage
  on this sample and must not be described as solving narrow endpoints.
- Median natural arrival/goal yaw difference was about 150-159 degrees across
  the three scenes, confirming that terminal reversal is the typical geometry.

### Closed-loop single-episode diagnostics

These are stochastic single-episode diagnostics, not SR estimates.

Episode: `1LXtFkjw3qL/episode_0009`.

| Mode / checkpoint | Key result |
|---|---|
| Oracle / checkpoint-2600 | RLR, 2.51 m; GT yaw error `154.49° -> 0.00°`; final position error `0.00 m` |
| LingBot yaw / checkpoint-1500 | LingBot yaw error `2.21°`; RSL, 2.64 m; GT yaw `174.96° -> 2.21°`; raw-DINO cosine `0.825 -> 0.989`; final position error `0.00 m` |
| Full LingBot pose / checkpoint-1500 | Failed closed before motion: aux target position error `1.93 m`, while yaw error was only `4.69°`; no collision-free path to the incorrect predicted position |
| LingBot local / checkpoint-1500 | Did not use aux translation; RLR, 3.18 m; remained `0.967 m` from B; GT yaw `143.80° -> 5.29°`; raw-DINO cosine `0.797 -> 0.944` |

This single-example result shows that LingBot rotation can be accurate enough
to orient the camera for direct ImageGoal verification. In this example, the
full-pose terminal controller failed because of metric translation, not yaw;
the larger evaluation below exposes scene- and gap-dependent yaw tails.

### Fixed-seed thirty-episode closed-loop evaluation

Configuration: `checkpoint-1500`, raw-DINO retrieval, replayed leg A, base seed
`20260802`, ten episodes from each of `17DRP5sb8fy`, `1LXtFkjw3qL`, and
`Uxmj2M2itWa`. Diffusion noise is reset per episode (`base_seed + episode_idx`).
An `off` versus `lingbot_local` preflight produced byte-for-byte equal planning
records through first reach, proving that the terminal arm does not change its
navigation prefix.

| Scene | Reached B | Forward-only path completed | Pose success (<1 m, <15 deg) |
|---|---:|---:|---:|
| `17DRP5sb8fy` | 2/10 | 2/2 | 0/2 |
| `1LXtFkjw3qL` | 5/10 | 5/5 | 5/5 |
| `Uxmj2M2itWa` | 6/10 | 5/6 | 6/6 |
| **Pooled** | **13/30 (43.3%)** | **12/13 (92.3%)** | **11/13 (84.6%)** |

Across the twelve completed maneuvers:

- yaw-error median: `161.57 deg -> 2.86 deg`;
- raw-DINO current/goal cosine mean: `0.8725 -> 0.9393` (`11/12` improved);
- added path: median `2.86 m`, mean `3.46 m`, range `2.55-6.27 m`;
- official distance-only SPL: `0.2829`; SPL including terminal travel: `0.1611`.

Path completion and final-pose success are intentionally different metrics.
One sample had no collision-free forward-only path, but it was already at
`0.987 m` and `6.74 deg`; it therefore fails the maneuver-completion metric but
passes the final-pose metric. The two actual pose failures completed their
curves but LingBot's target yaw missed by `15.42 deg` and `30.59 deg`. The
retrieved anchors were near the GT covisibility
peaks; anchor-index error did not explain the failures. On these thirteen
reached samples, stream pose gap (`current_frame - anchor`) correlated with yaw
error (`Pearson r=0.67`), whereas absolute anchor-index error did not
(`r=-0.19`). This is a small diagnostic sample, so it supports a long-gap pose
reliability hypothesis rather than establishing a universal threshold.

### Direct visual residual diagnostic and independent calibration

The actual post-turn frames retained by the server were compared to their goal
images with OpenCV SIFT correspondences plus a calibrated essential matrix.
An inlier-only diagnostic accepted `10/12` completed maneuvers; on those ten,
the residual-yaw magnitude had `0.54 deg` mean absolute error and `1.52 deg`
maximum error against Habitat GT. Essential geometry alone was not safe enough
for control, however: a controlled sample exposed a high-inlier sign flip. The
final module therefore requires agreement with an independent median horizontal
feature-bearing estimate. With this consensus gate, `8/12` real post-turn views
were accepted. The `30.59 deg` long tail was still correctly accepted: essential
yaw was about `30.2 deg`, bearing yaw about `31.9 deg`, and their disagreement
only about `1.7 deg`.

Confidence thresholds were then selected on 144 controlled views from 16
episodes in two scenes, with known residuals from -35 to +35 degrees and camera
positions 0.15-0.85 m from the goal. The selected fixed gate is:

```text
matches >= 8
pose inliers >= 16
inlier ratio >= 0.50
off-axis rotation <= 15 deg
essential/bearing disagreement <= 5 deg
act only for 8 deg < |correction| <= 45 deg
```

On calibration it accepted `55/144` (38.2%): MAE `0.64 deg`, P95 `1.75 deg`,
maximum `4.72 deg`, with no error over 5 degrees. The same frozen thresholds on
another 144 views from 16 independent episodes in two scenes accepted `75/144`
(52.1%): MAE `1.37 deg`, P95 `6.16 deg`, maximum `8.39 deg`, and no error over
15 degrees. Thus it is not a universal precision estimator; it is a conservative
control gate whose observed accepted errors stayed inside the evaluator's
15-degree tolerance on this validation set.

For a simulated one-action controller, the 8-degree deadband changed
within-15-degree pose success from `64/144` to `90/144` on calibration (32
actions, all improved) and from `67/144` to `103/144` on frozen validation (54
actions, 53 improved; the one slight regression remained successful). It fixed
all 26 and all 36 initially failed samples on which it acted, respectively.

### End-to-end visual correction checks

The calibrated module was integrated after `lingbot_local`; it never changes
the navigation prefix or aux translation. Two fixed-data checks exercised both
control branches:

- `17DRP5sb8fy/episode_0001`: reached B and completed the coarse maneuver, but
  the final current/goal pair had only five ratio-test matches. The controller
  safely abstained, preserving the coarse result rather than guessing.
- `Uxmj2M2itWa/episode_0006`: coarse LingBot alignment left `8.361 deg` GT yaw
  error. The direct estimate was `8.421 deg` (83 matches, 67 inliers, ratio
  0.807, consensus error `0.995 deg`). One forward-only RLR correction completed;
  final GT yaw error was `0.060 deg`, the independent final visual estimate was
  `0.059 deg`, direct image cosine rose `0.865 -> 0.982`, and final distance
  remained successful at `0.972 m`.

These are causal end-to-end checks of the correction mechanism, not a new SR
estimate. A larger fixed-seed evaluation is still required before claiming an
aggregate gain. Raw outputs are under `eval_terminal_ab/visual_refine_e2e/`;
calibration and frozen-validation reports are under
`eval_terminal_ab/visual_yaw_calibration/`.

## Metric semantics

- Official distance-only `spl_B` remains based on the path length at first
  positional success, so it stays comparable with older runs.
- `spl_B_with_terminal` includes the extra U-turn distance.
- `pre_turn_goal_cos` and `post_turn_goal_cos` measure raw DINO cosine.
- `terminal_success` requires final GT distance below `success_dist` and GT yaw
  error below `terminal_yaw_tol_deg`. GT is evaluation-only.
- `loop_closed` is unset unless `--loop_cos_min` is explicitly provided. No
  universal cosine threshold has been calibrated yet.
- `summarize_terminal_uturn.py` reports navigation SR, terminal completion
  conditioned on an attempted maneuver, and pose success with explicit
  denominators. It also writes `aggregate_summary.json` next to the raw CSVs.

## Important remaining limitation

The current Habitat evaluator still uses the benchmark's GT distance
`distance(current, B) < 1 m` to decide when navigation has reached the endpoint
and should hand control to the terminal maneuver. This isolates the U-turn and
visual-verification question, but it is not a deployable stop detector.

Terminal-motion frames are streamed into LingBot memory, but final verification
now uses the stateless `/imagegoal_similarity` endpoint. It neither appends the
final image nor runs retrieval, preventing a trivial near-self retrieval match.
The reported `current_goal_cos` is therefore a direct current-image-to-goal
comparison. This experiment still must not be described as pose-graph loop
closure: a causal loop-closure version should freeze the pre-turn anchor (or
mask terminal frames), then apply an explicit accepted-match pose correction.

The next causal version should trigger terminal alignment from a calibrated
combination of policy stopping evidence, current/goal visual similarity, and
pose uncertainty. It should not use the raw aux translation alone.

## Dependency preflight before any job submission

The clean Git worktree does not contain ignored local dependency directories.
An attempted launch correctly failed with:

```text
ModuleNotFoundError: internnav.model.basemodel.LongCLIP
```

For the local tests, the server therefore used the complete dependency tree in
the synchronized main checkout while running the modified server/controller
from this worktree:

```bash
cd /home/asus/Research/Nav-axis-uturn
/home/asus/miniconda3/envs/memnav/bin/python \
  NavDP/baselines/memnav/memnav_server.py \
  --port 18889 \
  --checkpoint /home/asus/Research/Nav/InternNav/checkpoints/memnav_2leg_axisfix/checkpoint-1500/memnav.ckpt \
  --internnav_root /home/asus/Research/Nav/InternNav \
  --retrieval raw
```

Before submitting any future Slurm evaluation/training task, verify all of the
following on the target node:

1. The selected commit/worktree is printed and clean enough for the intended run.
2. `LongCLIP`, LingBot-Map, model weights, checkpoints, dataset, and feature-cache
   paths exist and are readable.
3. Imports are tested in the exact Conda environment used by the job.
4. A one-sample forward pass succeeds before requesting the long wall-clock job.
5. The output/checkpoint/W&B directories are writable and belong to the intended
   user or shared project.
6. The launched server exposes `goal_rel_yaw` and `current_goal_cos`; an older
   already-running process will not pick up the new source automatically.

Follow-up: MemNav no longer eagerly imports every unrelated encoder when it
imports `navdp_backbone`; the public encoder exports are lazy, so a clean MemNav
worktree does not require the ignored Long-CLIP checkout. The Slurm training
script now runs `scripts/train_memnav/dependency_preflight.py` before the GPU
probe. It verifies the real Python import closure, data/LingBot/weight paths, a
versioned cache pair and its window geometry, optional warm-start checkpoint
schema, and output-directory writability. The pt1 job also explicitly exports
`MEMNAV_REQUIRE_VERSIONED_CACHE=1`, so it cannot silently fall back to legacy
row-equals-frame indexing.
