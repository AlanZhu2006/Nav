# Retrieval causal diagnostic and long-route next step

Date: 2026-07-21 (Asia/Shanghai)

This report records local, offline diagnostics on cached Habitat episodes. It is not
a closed-loop navigation result. The evaluated fixed set contains 28 samples from 12
cached episodes in two scenes, so small deltas must not be treated as evidence of
generalization.

## Question

The current checkpoint still performs poorly on long 3-leg decisions. This round of
experiments asks three separate questions:

1. Is retrieval itself measurably wrong?
2. If retrieval is made correct, does the generated action improve under otherwise
   identical DDPM sampling?
3. If not, what information is missing after retrieval?

## Frozen evaluation contract

All full-DDPM arms use the same checkpoint, explicit sample indices, diffusion seed,
and within-batch goal shuffle:

- checkpoint: `/home/asus/Research/Nav-axis-fix/.diagnostics/checkpoints/mkf-1371557-step400.ckpt`;
- source commit: `6c48b24768e436fafbc65016502645d7e4353192`;
- fixed indices: `0..16,18..28` (28 rows);
- parent fingerprint: `e60441f1524f88f578fe05f2ab0657e72b8a3566b7ae0eca88af06e395cd499a`;
- baseline evaluation fingerprint: `67e8e611ebf22a51caa48eedefcbc638fc79be30af8340dab6dec4c442aedd81`;
- sampling mode/seed: `fixed_leg`, seed 0;
- full DDPM seed: `104729`, with paired diffusion randomness;
- closed-loop navigation: false.

The baseline report exactly reproduces the previously audited full-DDPM action MSE
of `0.1024612893`.

## Finding 1: retrieval has a real candidate-boundary bug

The historical earliest retrievable anchor is
`num_scale + window - 1 = 8 + 32 - 1 = 39`. This boundary is stricter than goal
insertion actually requires. Some goals have valid co-visible positives before frame
39, so the old boundary removes most or all of their true-positive region before the
retrieval head sees it.

Two concrete cached examples are:

- one episode has 16 positive frames at 24--39, but the old boundary leaves only
  frame 39;
- another has 47 positives at 0--46, but the old boundary leaves only frames 39--46.

The insertion path can replay from the initial scale block at frame 8. An explicit
LingBot goal-insertion test fitted one Sim(3) per full episode and then tried early
anchors. Translation error at frame 8 was 0.127 m and 0.216 m in the two failure
episodes, comparable to the corresponding frame-39 errors of 0.185 m and 0.228 m.
Across the tested early anchors (8--38), error remained approximately 0.13--0.26 m.
This supports frame 8 as an evaluation candidate floor on this cache; it is not yet
a scene-held-out guarantee.

## Finding 2: raw DINO is a better selector on this small set

A 20-seed, no-training screen compares projected retrieval logits, frozen raw-DINO
cosine, blends, top-k cross-reranking, and temporal score mass.

| Candidate floor / selector | Revisit rows | Strict positive | Gray | Negative |
| --- | ---: | ---: | ---: | ---: |
| frame 39 / projected Top-1 | 200 | 80.0% | 22 | 18 |
| frame 39 / raw Top-1 | 200 | 80.0% | 27 | 13 |
| frame 8 / projected Top-1 | 220 | 82.7% | 20 | 18 |
| frame 8 / projected Top-5 then raw rerank | 220 | 93.6% | 0 | 14 |
| frame 8 / raw Top-1 | 220 | 94.1% | 0 | 13 |

Temporal score-mass selection reduced exact accuracy to about 60--70%, so it is
rejected. The current ranking loss also ignores gray candidates while inference
argmax allows them to win; that is a genuine train/inference objective mismatch.

These numbers establish a retrieval-metric problem, but they do not establish an
action-generation problem.

## Finding 3: perfect retrieval does not materially fix action generation

The decisive A/B reruns the entire conditioned DDPM path with paired randomness.
The frame-8 arms contain one additional correctly identified revisit row, hence the
11/17 revisit/novel split instead of 10/18.

| Arm | Strict revisit hit | Full-DDPM action MSE | Change vs baseline | 3-leg Goal-C MSE |
| --- | ---: | ---: | ---: | ---: |
| projected / frame 39 | 8/10 (80%) | 0.102461 | baseline | 0.112371 |
| projected / frame 8 | 9/11 (81.8%) | 0.101775 | -0.67% | 0.112371 |
| raw / frame 8 | 11/11 (100%) | 0.101929 | -0.52% | 0.112462 |

The paired 95% intervals for the overall action-MSE changes cross zero:

- projected/frame-8 minus baseline: approximately
  `[-0.001949, +0.000268]`;
- raw/frame-8 minus baseline: approximately
  `[-0.001852, +0.000504]`.

On the ten revisit rows shared with the old baseline, action MSE is slightly worse,
not better. Most importantly, 3-leg Goal C is unchanged. Therefore increasing exact
retrieval accuracy from 80% to 100% is retrieval hygiene, not the current long-route
solution.

This is consistent with the earlier oracle-positive experiment: forcing a GT-positive
history frame did not improve the auxiliary pose MSE. It is also consistent with the
early-anchor geometry result: covisibility class is an imperfect proxy for how well a
history anchor supports the downstream goal-pose estimate.

## Finding 4: the missing signal is route topology, not another gate threshold

The policy conditions the local diffusion decoder on endpoint/revisit pose and visual
memory, but it does not receive an explicit navigable route or near-term subgoal. In a
maze, the direct endpoint bearing can point through a wall or disagree with the first
segment of the feasible A* route.

An offline geometry diagnostic compares the future 16-frame route tangent against
several inference-like vectors:

| Group | Endpoint-bearing error | Recent 2-frame tangent error | Reverse breadcrumb error |
| --- | ---: | ---: | ---: |
| revisit rows | 30.34 deg mean / 16.42 deg median | 15.28 / 8.43 deg | 155.94 deg mean |
| 3-leg Goal C | 27.98 / 22.59 deg | 15.93 / 15.60 deg | 150.16 deg mean |

The recent tangent is useful in aggregate but is better than endpoint bearing on only
63.6% of revisit rows and 50% of 3-leg Goal-C rows. A fixed "keep going straight"
heuristic is therefore unsafe. Reversing the remembered path is decisively wrong for
these samples.

The current checkpoint is also weakly goal-sensitive on the difficult long group:

- full-set goal-shuffle sensitivity MSE: `0.007840`, ratio `0.208`;
- 3-leg Goal-C sensitivity MSE: `0.001892`, ratio `0.135`;
- its shuffled-goal action penalty is only `0.003575`.

This means the decoder's action changes little when the goal is changed on the exact
group where a long-horizon route decision matters.

To test whether the problem could be solved by merely suppressing an incompatible
revisit, an oracle used the future GT route angle to modify the revisit gate while
holding the prepared condition and DDPM random streams fixed. Even this unavailable
oracle information made action generation worse:

| Oracle gate strategy | All-row MSE | Delta | 3-leg Goal-C MSE | Delta |
| --- | ---: | ---: | ---: | ---: |
| unchanged baseline | 0.102461 | -- | 0.112371 | -- |
| zero every revisit gate | 0.109121 | +0.006659 | 0.129254 | +0.016882 |
| zero gate when route disagreement >=45 deg | 0.105821 | +0.003360 | 0.119502 | +0.007130 |
| half-cosine gate | 0.104583 | +0.002122 | 0.114375 | +0.002004 |

On the two >=45-degree hard rows, zeroing the gate raises MSE from 0.225541 to
0.272579. Therefore the next model must preserve the useful revisit/goal condition
and add route information; replacing or attenuating it is not enough.

## Recommended solution: residual hierarchical route tokens

The next architectural arm should predict a coarse route sketch and append it to the
existing diffusion memory. It should not feed A* ground truth at inference and should
not replace retrieval.

Suggested design:

1. Form inference-safe route-head input from the existing current/global memory,
   current state, endpoint goal code, retrieved goal-pose code, and recent motion.
2. Predict robot-frame route directions and normalized/log ranges at multiple path
   horizons, initially 8, 32, and 96 frames. Direction is represented as a unit 2-D
   vector, not a raw angle, so the -179/179-degree discontinuity cannot reappear.
3. Encode those predictions as three route tokens and append them to the current
   memory tokens consumed by the diffusion decoder.
4. Initialize a learned residual route-token gate at zero. Loading the current
   checkpoint therefore reproduces the current policy exactly before training.
5. Supervise the head from the existing expert/A* trajectory with circular direction
   loss plus Huber loss on normalized range. The decoder must always consume predicted
   route tokens, never teacher-forced GT route tokens.
6. Keep retrieval, gate, aux, and action targets otherwise unchanged for the first
   controlled A/B.

This changes MemNav from "retrieve an endpoint and locally react" into "retrieve a
goal, infer a coarse feasible route, then generate a local trajectory." It remains
different from simply attaching revisit to NavDP, because global memory now supports
an explicit hierarchical planning representation.

## Training and acceptance experiment

Do not launch an 8-hour retrieval-only job. The next production comparison should be
single-variable:

- A: current best checkpoint continued with the accepted baseline data schedule;
- B: same initialization, batches, optimizer budget, and seeds, plus residual route
  head/tokens;
- optionally C only after B works: B plus a goal/route-conditioned multi-candidate
  critic. A previous best-of-32 oracle indicates substantial candidate-selection
  headroom, but it uses unavailable action GT and is not an inference solution.

Before the 8-hour run, require:

- exact checkpoint equivalence with route-token residual gate at zero;
- finite two-step GPU smoke and save/resume round trip;
- route-label axis, scale-normalization, angle-wrap, and padding-mask unit tests;
- a short paired continuation where B improves 3-leg Goal-C and span>=256 action MSE
  without regressing 2-leg by more than 2%;
- increasing 3-leg Goal-C goal-shuffle sensitivity, not merely a lower rank loss.

For the full run, accept B only if two or more seeds show improvement in 3-leg Goal C
and long-span full-DDPM action MSE, retrieval remains non-regressed, and a held-out
scene closed-loop Habitat evaluation improves SPL/success. Offline 28-row MSE is a
screening gate, not the final claim.

## Implemented diagnostic support

The code in this branch intentionally changes no production default. It adds:

- an evaluation-only raw-DINO anchor selector;
- an evaluation-only anchor-floor override;
- explicit evaluator indices for exact paired reruns;
- reusable retrieval-strategy, early-anchor-pose, route-condition, and oracle-gate
  diagnostics;
- unit tests for the new contracts.

Raw reports are under `.diagnostics/retrieval/` in this worktree. They are local
artifacts and are intentionally not source-controlled.

## Dependency and validation record

Run tests from the `InternNav` directory with the repository's `memnav` environment:

```bash
conda run -n memnav python -m pytest tests/unit_test -q
```

Result: `87 passed in 2.60s`. All changed Python diagnostic/evaluator files also pass
`py_compile`. The default base environment has no pytest, so launching tests from it
is a dependency error, not a test failure.
