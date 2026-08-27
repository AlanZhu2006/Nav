# Raw-DINO Novel Forced-Anchor Attribution

**Date:** 2026-08-16  
**Status:** completed consumed-development mechanism audit; stop gate failed  
**Claim boundary:** no Habitat rollout, no SR result, no final14 access

## 1. Question

Phase-2 raw-fixed unexpectedly reached `9/19` Novel goals versus native `4/19`,
while Attempt 7 had raw `1/9` versus native `2/9`.  The completed-rollout audit
showed that Phase-2 was unusually U-turn-heavy and that the raw head itself
collapsed near the rear.  It did not establish whether DINO-selected history
actually supplied query-specific route information.

This experiment holds fixed, for each consumed Phase-2 query:

- the complete actual-online Goal-A RGB history;
- the first Novel current RGB;
- the Novel ImageGoal;
- the model weights, seed, candidate boundary and current state;
- the LingBot cache and all historical camera poses.

Only the forced historical anchor changes.  The factual arm uses the anchor
selected by raw DINO in the completed Phase-2 run.  Twelve controls are drawn
uniformly without replacement from the exact eligible anchor interval, with an
identity-bound seed frozen before any new model forward.  The production
`/posegoal_step` and `/posegoal_query` paths then recover the final LingBot
`aux_pose` bearing.  This tests the complete DINO-anchor-to-LingBot-bearing
proposal, not merely the physical direction to an anchor.

## 2. Population and immutable artifacts

- 19 consumed Phase-2 Novel queries;
- 12 MP3D scene clusters;
- 12 random controls plus one factual query per history;
- 247 pose-only model queries in total;
- 0 Habitat steps;
- 0 untouched final14 scenes read.

Manifest:

`.diagnostics/raw_novel_forced_anchor_replay_20260816/manifest.json`

SHA-256:

`e168363e04a2ea5a666e1396c5724f9895a5cc16ed8797ac6a4affe51c293c21`

Machine-readable result:

`.diagnostics/raw_novel_forced_anchor_replay_20260816/report.json`

SHA-256:

`b6345eb104fca9c70f7576e9869d081877914b279002ba56d5b3bc1a849db8d4`

Implementation and tests:

- `raw_novel_forced_anchor_replay.py`;
- `test_raw_novel_forced_anchor_replay.py`, `4/4` passed.

Independent arithmetic verification:

`.diagnostics/raw_novel_forced_anchor_replay_20260816/independent_verification.json`

- `verified=true`;
- SHA-256
  `334a17426293ecbf4c174c67bdf7e33564608085a52cdb2cfcbdf2e41e9cb8e4`;
- verifier: `independent_verify_raw_novel_forced_anchor_replay.py`, which does
  not import the runner or summarizer.

## 3. Replay audit

The local RTX 4090 replay reproduced the factual HPC bearing closely:

| diagnostic | value |
|---|---:|
| absolute bearing difference, mean | `1.079 deg` |
| median | `0.804 deg` |
| minimum | `0.116 deg` |
| maximum | `3.656 deg` |

This is small relative to the route errors and the factual-versus-control
spread.  The result is not an artifact of replaying a materially different
proposal function.

## 4. Primary result: shortest-path first segment

Positive advantage means that the DINO-selected anchor produced a smaller
angular error than the mean of its 12 uniformly sampled legal anchors.

| measure | factual DINO anchor | random-anchor expectation |
|---|---:|---:|
| mean angular error | `38.630 deg` | `42.778 deg` |
| median angular error | `27.161 deg` | `31.816 deg` |
| bearings within `30 deg` | `10/19` | `9.75/19` expected |

Factual advantage:

- mean `+4.148 deg`;
- median `+3.308 deg`;
- range `[-20.832,+22.821] deg`;
- positive mean advantage in `10/12` scene clusters;
- 100,000-resample scene-cluster bootstrap 95% CI
  `[-1.357,+8.898] deg`;
- bootstrap probability of a non-positive mean: `0.06516`.

The predeclared promotion rule required the scene-cluster 95% CI lower bound
to be greater than zero.  It failed.  More importantly, the `30 deg` useful
coverage changed by only `0.25` expected query: `10` versus `9.75`.

## 5. Secondary result: exact direct-goal bearing

The independent direct-goal reference is even less favorable:

| measure | factual DINO anchor | random-anchor expectation |
|---|---:|---:|
| mean angular error | `39.577 deg` | `41.746 deg` |
| bearings within `30 deg` | `10/19` | `10.0/19` expected |

- mean advantage `+2.169 deg`;
- scene-cluster 95% CI `[-3.710,+7.069] deg`;
- bootstrap probability of a non-positive mean `0.22026`.

## 6. Relation to the exact physical-anchor audit

Before paying for model replay, every eligible historical pose was scored
against the target direction.  On the same Phase-2 queries, the physical
direction of the DINO-selected anchor had only `+0.048 deg` mean advantage over
all eligible anchors, with scene-cluster CI `[-2.522,+3.531] deg`.  Against the
direct goal, it was worse by `1.652 deg`, with CI `[-3.949,+0.774] deg`.

The GPU result therefore has a precise interpretation:

- DINO does not select a historically better route location;
- LingBot's goal-pose insertion can sometimes shift the final bearing by a few
  useful degrees;
- that correction is heterogeneous, not statistically stable, and does not
  increase `<=30 deg` coverage in this population.

## 7. Causal conclusion

The Phase-2 raw Novel successes cannot be attributed to a reliable
history-specific DINO compass.  The best-supported explanation is a
combination of:

1. a strong rear/U-turn prior in raw proposals;
2. a Phase-2 cohort with the correct route behind in `16/19` queries;
3. goal/current-view and LingBot pose interactions that make small,
   query-dependent corrections;
4. closed-loop sampling sensitivity in a `19`-query, `12`-scene population.

This does not prove that history is always irrelevant to Novel navigation.  It
does show that the current raw-DINO mechanism fails the evidentiary gate needed
to spend a fresh closed-loop population on that claim.

## 8. Frozen decision

`stop_novel_dino_branch_and_preserve_final14_for_cec_confirmation`

Consequences:

- do not run the paused 600-step Novel four-arm control;
- do not run goal-image shuffle after the factual-history gate failed;
- do not use final14 to debug or rescue raw Novel behavior;
- do not claim that DINO retrieves a global Novel route direction;
- retain the oracle-bearing result only as a privileged capability/mechanism
  result;
- make the next prospective experiment the role-free CEC Revisit
  utility/Novel-interference confirmation on the untouched final14, frozen
  before any policy outcome is read.
