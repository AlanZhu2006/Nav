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

These 20 scenes are now a development ablation because their failures were
inspected.  Any selected configuration must be frozen before one new blind
scene split is evaluated.

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

No source file under `/home/asus/Research/Nav` is modified.  All source changes
are confined to `/home/asus/Research/Nav-axis-uturn` and remote clean worktrees.
