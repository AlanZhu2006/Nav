# Gatecurr600 direct-decoder strict controller ablation (2026-08-04)

## Conclusion

On the exact five-scene, ten-episode split used by the automatic geometric
router, replacing the frozen NavDP point-goal controller with gatecurr600's
original diffusion decoder did **not change a single success outcome**:

| Metric | gatecurr direct decoder | geometry router + NavDP point-goal |
|---|---:|---:|
| Goal A SR | 9/10 = 0.900 | 9/10 = 0.900 |
| Goal B SR given A | 8/9 = 0.889 | 8/9 = 0.889 |
| Joint two-leg SR | 8/10 = 0.800 | 8/10 = 0.800 |
| Goal B SPL given A | 0.695 | 0.623 |
| Goal B path given A | 4.598 m | 4.530 m |
| Goal B steps given A | 151.6 | 139.1 |

The direct arm's mean SPL was 0.0716 higher, but the paired 95% bootstrap
interval was `[-0.0197, 0.1550]`. Its mean path was 0.0687 m longer with a
95% interval of `[-0.681, 0.907]` m. Neither small ten-route difference
establishes that one final controller is better.

The defensible result is therefore narrower and more useful: the large gain
over pure NavDP (`Goal B|A = 3/9`) comes from supplying trustworthy long-term
memory geometry. It is not currently attributable to replacing the original
gatecurr decoder with the NavDP point-goal decoder. The automatic geometric
router remains necessary because this direct arm is told the A-to-B phase
boundary; it is not a deployable Novel/Revisit detector.

## Controlled protocol

- code: `5b9dc5c0be14d16a44b72a8f50a657cd7ee0b916`;
- scenes: `e9zR4mvMWw7`, `rqfALeAoiTq`, `s8pcmisQ38h`,
  `yqstnuAEVhm`, `zsNo4HB9uLZ`;
- two fixed episodes per scene;
- Goal A: frozen official NavDP, with every observation also streamed to
  MemNav;
- Goal B: gatecurr600's native diffusion waypoint decoder;
- backend: evaluator `hybrid_oracle`; “oracle” here means only the known
  A-to-B phase boundary, not a GT anchor, pose, gate, waypoint, or action;
- MemNav: W32/S8, 16 samples, raw DINO retrieval, `exclude_recent=32`,
  automatic flow gate, complementary fusion;
- `success_dist=1.0 m`, `max_steps=500`, `exec_horizon=8`;
- terminal U-turn and visual refinement disabled;
- Habitat-Sim 0.3.3, matching the geometric-router run.

The Goal A columns were compared against the raw geometric-router output.
`reached_A`, SPL, geodesic distance, executed length, final distance, and step
count were bit-for-bit equal for every episode. This confirms that only the
Goal B controller path changed.

## Dependency identities

| Artifact | SHA256 |
|---|---|
| gatecurr600 | `9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7` |
| official NavDP | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` |
| LingBot-Map long | `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409` |

The runner also validates all five GLB hashes, all ten episode structures,
scene disjointness, Python dependencies, ports, and runtime checkpoint
loading before evaluation.

## Paired details

- Goal B outcomes were identical in all ten rows.
- Both methods failed `e9zR4mvMWw7/episode_0000` after Goal A succeeded.
- Both skipped Goal B for `zsNo4HB9uLZ/episode_0001` because frozen NavDP
  failed Goal A.
- Among the eight shared Goal B successes, direct decoding used 4.166 m on
  average versus 4.348 m for point-goal, but 136.8 versus 132.5 controller
  steps. The differences are episode-dependent rather than a uniform gain.
- Direct SPL was higher in five rows, lower in two, and tied in two among the
  nine executable Goal B rows.

Full per-episode values are in
`MemNavData/GATECURR_DIRECT_CONTROLLER_RESULTS_20260804.csv`.

## Repeated-smoke variance

The earlier isolated `s8pcmisQ38h/episode_0000` direct run succeeded with SPL
0.787; the full-run repetition also succeeded but had SPL 0.314. Goal A,
the first retrieval anchor, raw score, and gate were identical. The current
view already diverged by the second plan, so the difference begins in sampled
waypoints, not retrieval. This is evidence that a single-run SPL estimate has
meaningful diffusion/restart variance even when the nominal episode seed is
fixed. SR is consistent across the two repetitions, but future controller
ranking should use repeated policy seeds.

## Relation to the older 6/10 unseen result

The older gatecurr unseen-scene smoke used replayed Goal A, trained-head
retrieval, `exclude_recent=83`, and a 1200-step budget. This experiment uses
the actually executed NavDP Goal A memory, raw retrieval, `exclude_recent=32`,
and a 500-step budget. The old 6/10 and current 8/9 conditional SR therefore
must not be interpreted as a training improvement.

## Reproduction and raw outputs

- runner: `MemNavData/run_gatecurr_direct_control_ab.sh`;
- Slurm wrapper: `MemNavData/slurm_gatecurr_direct_control_ab.sbatch`;
- raw local output:
  `.diagnostics/gatecurr_direct_control_20260804/results/full_raw_e32/gatecurr600_direct`;
- the queued HPC smoke was cancelled before execution when the matching local
  Habitat 0.3.3 GPU became available; no remote result was mixed into this
  table.
