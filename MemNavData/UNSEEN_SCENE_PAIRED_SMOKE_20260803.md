# Gate curriculum unseen-scene paired Habitat smoke (2026-08-03)

## Scope

This is a paired diagnostic of `flowgate2600` and `gatecurr600` on MP3D
scenes that are absent from the 50-scene MemNav training list. It is not the
old same-scene 20-episode evaluation, and it is not a standard held-out
Habitat leaderboard result.

All generated data, copied assets, checkpoints, logs, and results are under:

```text
/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803
```

The mother worktree `/home/asus/Research/Nav` was not modified.

## Audited inputs

- Scenes: `s8pcmisQ38h`, `e9zR4mvMWw7`, `rqfALeAoiTq`,
  `zsNo4HB9uLZ`, and `yqstnuAEVhm`.
- Scene overlap with the 50 training scenes: zero.
- Episodes: 2 generated 2-leg episodes per scene, 10 total; no episode was
  replaced or filtered after observing its result.
- Frame counts: 213--470.
- Recall gaps: 48--172 frames.
- `flowgate2600` checkpoint SHA256:
  `debd079c6f578e9c6e2c1f0e70f6dc8fc2c2230785c28d6da2fae118a665b38b`.
- `gatecurr600` checkpoint SHA256:
  `9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7`.

The validator rechecked scene disjointness and the SHA256 values of all
assets, episode metadata, and checkpoints after the evaluation completed.

## Evaluation protocol

- Identical scene, episode, start/goal, and policy seed for both checkpoints.
- Leg A: replay.
- Leg B success: distance-only, final goal distance at most 1.0 m.
- Replan interval: 8 simulator frames.
- Maximum policy budget: 1200 frames per leg.
- Retrieval: learned head; gate: automatic.
- 16 DDPM samples and `exclude_recent=83`.
- Terminal U-turn and terminal visual refinement: disabled.

This isolates the learned revisit policy. It does not credit the later
rule-based terminal U-turn/visual loop-closure logic.

## Aggregate result

| Metric | flowgate2600 | gatecurr600 | Change |
|---|---:|---:|---:|
| Successes | 2/10 | 6/10 | +4 |
| SR | 0.200 | 0.600 | +0.400 |
| Mean SPL (failures are zero) | 0.105 | 0.395 | +0.291 |
| Mean final distance | 2.998 m | 1.882 m | -1.116 m |
| Median final distance | 2.961 m | 0.992 m | -1.969 m |
| Mean executed path | 10.172 m | 6.808 m | -3.363 m |
| Mean simulator steps | 275.8 | 219.7 | -56.1 |
| Mean predicted gate | 0.350 | 0.324 | -0.026 |

Gate curriculum had a lower final distance in 8/10 paired episodes. The
paired outcomes were:

- both succeeded: 1;
- flowgate only: 1;
- gatecurr only: 5;
- both failed: 3.

The exact two-sided McNemar p-value is 0.21875. The 95% Wilson intervals for
SR are approximately `[0.057, 0.510]` for flowgate and `[0.313, 0.832]` for
gatecurr. The smoke is therefore strong directional evidence, but 10 routes
are not enough for a statistical claim.

## Per-episode paired result

Rows are sorted by recall gap.

| Scene / episode | Gap | SR, flow -> curr | SPL, flow -> curr | Final distance, flow -> curr | Gate, flow -> curr |
|---|---:|---:|---:|---:|---:|
| `s8pcmisQ38h/0000` | 48 | 1 -> 0 | 0.321 -> 0.000 | 0.985 -> 2.778 m | 0.220 -> 0.210 |
| `zsNo4HB9uLZ/0000` | 48 | 0 -> 1 | 0.000 -> 0.612 | 3.685 -> 0.987 m | 0.041 -> 0.088 |
| `rqfALeAoiTq/0001` | 59 | 0 -> 1 | 0.000 -> 0.566 | 2.963 -> 0.978 m | 0.564 -> 0.554 |
| `e9zR4mvMWw7/0001` | 62 | 0 -> 1 | 0.000 -> 0.643 | 1.900 -> 0.997 m | 0.582 -> 0.582 |
| `yqstnuAEVhm/0001` | 82 | 1 -> 1 | 0.726 -> 0.612 | 0.975 -> 0.987 m | 0.512 -> 0.458 |
| `s8pcmisQ38h/0001` | 100 | 0 -> 1 | 0.000 -> 0.685 | 4.544 -> 0.980 m | 0.070 -> 0.107 |
| `yqstnuAEVhm/0000` | 103 | 0 -> 1 | 0.000 -> 0.835 | 1.806 -> 0.974 m | 0.448 -> 0.386 |
| `e9zR4mvMWw7/0000` | 118 | 0 -> 0 | 0.000 -> 0.000 | 2.959 -> 2.493 m | 0.409 -> 0.281 |
| `rqfALeAoiTq/0000` | 138 | 0 -> 0 | 0.000 -> 0.000 | 4.348 -> 2.087 m | 0.524 -> 0.461 |
| `zsNo4HB9uLZ/0001` | 172 | 0 -> 0 | 0.000 -> 0.000 | 5.815 -> 5.554 m | 0.131 -> 0.116 |

The observed gap split is descriptive, not pre-registered: for gaps at most
109, SR changed from 2/7 to 6/7; all three routes with gaps at least 118
failed for both checkpoints.

## What the traces establish

The first retrieved anchor was exactly equal between checkpoints in 9/10
episodes. Its median absolute distance from the generated covisibility
argmax was 11.5 frames for both checkpoints (mean 14.0 versus 13.8). Thus
gatecurr did not obtain its aggregate gain by making retrieval globally more
accurate.

All five gatecurr-only successes began from exactly the same anchor as
flowgate. On those five episodes, the average predicted gate was also nearly
identical: 0.3413 for flowgate and 0.3434 for gatecurr. A particularly clean
case is `yqstnuAEVhm/0000`: both use anchors `[58, 66]`, while the result
changes from 17.55 m / 470 steps / failure to 4.81 m / 140 steps / success.

The most defensible interpretation is therefore:

1. gate curriculum improved how the diffusion decoder converts an already
   retrieved revisit memory into executable waypoints;
2. the improvement is not explained by a larger inference gate or a
   different initial retrieval anchor;
3. the remaining long-gap failures are not fixed by the curriculum.

In the gap-172 failure, both policies eventually selected frames from the
current return traversal after those frames aged beyond `exclude_recent`.
That is evidence for a source/age ambiguity in the single retrieval pool,
but one episode is not enough to call it the sole long-range cause. This
evaluation also did not log per-plan Habitat-vs-LingBot pose error, so it
cannot by itself attribute the remaining failures to metric scale or pose
drift.

## Long-gap oracle-covis follow-up

The three shared failures were rerun with `gatecurr600`, the original policy
seed, and the same execution settings, while forcing every retrieval to the
metadata covisibility argmax. This changes only the anchor; it does not inject
the GT pose or action.

| Gap | Learned -> oracle anchor | Learned final/path | Oracle final/path | Oracle success |
|---:|---:|---:|---:|---:|
| 118 | 103 -> 89 | 2.493 / 7.329 m | 2.432 / 13.272 m | 0 |
| 138 | 74 -> 39 | 2.087 / 11.833 m | 4.348 / 8.682 m | 0 |
| 172 | `[59, 293]` -> 39 | 5.554 / 15.640 m | 4.975 / 9.930 m | 0 |

Oracle retrieval rescued 0/3 failures. The first two goals had strong peak
covisibility (0.747 and 0.840), yet forcing those peaks did not produce a
successful waypoint trajectory. This rules out learned top-1 anchor error as
the main explanation for those failures. It also shows that the maximum-
covisibility frame is not necessarily the frame the current decoder can use
most effectively.

For gap 172, fixing the anchor prevented the learned policy from later
switching to frame 293 from the return traversal and substantially shortened
the path, but the final distance remained 4.975 m. Retrieval-pool age/source
ambiguity therefore amplifies this long case, but fixing it alone is not
sufficient. The remaining failure is downstream of retrieval and needs
per-plan pose/scale/waypoint instrumentation before it can be assigned to
LingBot drift, metric calibration, or the diffusion decoder.

## Recommended next experiment

1. Run a pre-registered paired validation with at least 50 routes, prioritizing
   more independent unseen scenes rather than repeatedly sampling only these
   five scenes.
2. Ablate a frozen long-term retrieval bank: take a snapshot when the image
   goal is issued, retrieve the revisit anchor only from that bank, and feed
   observations from the current return leg through a separate local-context
   path. This prevents old return-leg frames from silently becoming
   long-term candidates after `exclude_recent` expires.
3. Add per-plan goal-pose error against Habitat GT before assigning any
   remaining long-gap failure to LingBot drift or metric scale.
4. Train or rerank for downstream action utility in addition to covisibility;
   the gap-138 oracle result shows that the visually highest-overlap frame can
   be worse for the current decoder than its learned anchor.

Until the larger run is complete, report this result as an unseen-scene
paired smoke (`6/10` versus `2/10`), not as the final SR of the method and not
as directly comparable to the earlier same-scene `0.35` result.
