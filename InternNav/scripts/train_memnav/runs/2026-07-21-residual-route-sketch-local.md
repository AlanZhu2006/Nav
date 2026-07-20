# Residual route sketch local controlled experiment

Date: 2026-07-21 (Asia/Shanghai)

This document records a local, fixed-population diagnostic. It is not a
closed-loop navigation result and it does not justify enabling the route sketch
in production. All repository changes and generated checkpoints were kept in
the child worktree
`/home/asus/Research/Nav/.claude/worktrees/memnav-retrieval-20260721`; the parent
`/home/asus/Research/Nav` remained clean.

## Motivation

The retrieval causal diagnostic showed that moving the candidate floor from
frame 39 to frame 8 and using raw DINO can improve strict revisit retrieval from
80% to 100%, while paired full-DDPM action MSE changes by only -0.52% and 3-leg
Goal C is unchanged. Correctly retrieving an endpoint is therefore not the same
as representing the feasible route to it.

The experimental adapter predicts robot-frame route directions at action
horizons 2, 8, and 24 using only inference-available current/revisit/novel
memory. Expert actions are used only to form label-side unit-vector targets.
The predicted directions are encoded as zero-initialized residuals in existing
current-state slots, preserving decoder memory length and legacy checkpoint
shape. Version 2 additionally multiplies the residual by a predicted curvature
gate:

`gate = 0.5 * (1 - dot(pred_h2, pred_h24))`.

No GT route value enters inference.

## Safety and compatibility checks

- The feature is opt-in and remains disabled by default.
- Loading the audited `mkf-1371557-step400` checkpoint with zero residual was
  bitwise identical to the legacy policy on the first four fixed samples,
  including training-noise output, full DDPM, shuffled-goal DDPM, retrieval,
  gate, and auxiliary fields. Both means were `0.1236555464565754`.
- A real two-step GPU backward was finite, saved 395 non-LingBot tensors, and a
  resume restored model, optimizer, scheduler, and RNG state.
- Legacy migration is allowed only when the whole route namespace is absent.
  A partially present namespace fails closed. Route code version and horizons
  are validated against checkpoint metadata.
- Route auxiliary inputs are detached from the legacy encoders. The diffusion
  action objective is the only path that can open the residual scales.
- Optional adapter construction now preserves the global CPU PyTorch RNG.
  Training logs also record `diffusion_noise_mean/std` and
  `diffusion_timestep_mean` so paired optimization can be audited directly.

## Controlled protocol

All 20-step arms start from the same checkpoint and use:

- dataset root: `/home/asus/Research/Nav/memnav_viz/validate_gated`;
- sparse feature cache:
  `/home/asus/Research/Nav-axis-fix/.diagnostics/sparse_cache_smoke`;
- cache signature:
  `97a7819c4722ebf6e2165538b4908a276d426faf3190f18987332f59889c9afc`;
- sampling: `random_leg`, seed 0, batch size 1;
- 20 optimizer steps, LR `1e-5`, route LR multiplier 10;
- route weight `0.2`, horizons `2,8,24`, curvature emphasis 8;
- fixed full-DDPM indices `0..16,18..28`, seed `104729`, with paired
  within-batch shuffled goals.

The route-off and RNG-isolated route-on runs have exactly equal action-target
means, diffusion-noise mean/std, and diffusion-timestep mean in every 5-step
logging window. This establishes a paired sample and training-noise sequence.

The repeated route-off checkpoint has 379 tensors. Compared with the original
control run, 378 are bitwise equal; the only difference is a `3.73e-9` tail in
the frozen DINO `pos_embed`. The clean route-on checkpoint similarly differs
from the already evaluated v2 checkpoint only by a `9.31e-10` tail in that
frozen tensor. The existing full-DDPM reports are therefore reused instead of
performing a numerically redundant 12-minute evaluation.

Artifacts:

- control checkpoint:
  `checkpoints/route_ab_control_rng20_20260721/ckpts/checkpoint-20`;
- clean route checkpoint:
  `checkpoints/route_ab_route_rng20_20260721/ckpts/checkpoint-20`;
- control full-DDPM report:
  `.diagnostics/route_sketch/control20-full28-ddpm.json`;
- v1 report: `.diagnostics/route_sketch/route20-full28-ddpm.json`;
- v2 report: `.diagnostics/route_sketch/route-v2-20-full28-ddpm.json`;
- paired v2 comparison:
  `.diagnostics/route_sketch/control-vs-route-v2-20-paired.json`.

## Result 1: the head optimizes, but the horizons collapse

The route direction loss falls from `0.6659` at step 5 to `0.1881` at step 20,
and the three residual scales become nonzero. This proves that the optimizer,
gradient, and checkpoint paths work. It does not prove that the learned signal
is useful.

On the fixed 28 rows:

| Group | h2 error | h8 error | h24 error | predicted curvature gate |
| --- | ---: | ---: | ---: | ---: |
| all (28) | 7.59 deg | 17.77 deg | 21.38 deg | 0.00410 |
| Goal C (6) | 6.89 deg | 16.85 deg | 22.32 deg | 0.00196 |
| hard turn (2) | 1.81 deg | 21.23 deg | 99.62 deg | 0.00392 |

For hard turns, the GT h2-to-h24 separation is about 94.2 degrees, while the
predicted separation is only about 5.2 degrees. The curvature gate therefore
closes most strongly on the examples that need a long-horizon correction.

A constant `[1,0]` (always forward) baseline is also stronger than the learned
head on the population as a whole:

| Predictor | h2 error | h8 error | h24 error |
| --- | ---: | ---: | ---: |
| always forward | 4.78 deg | 16.30 deg | 21.31 deg |
| learned route head | 7.59 deg | 17.77 deg | 21.38 deg |

The learned head only beats the constant prior on the two hard rows at h2. It
has not yet learned a generally useful route representation.

## Result 2: full DDPM is worse

| Group | control | route v2 | relative change | improved rows |
| --- | ---: | ---: | ---: | ---: |
| all | 0.107667 | 0.110549 | +2.68% | 4/28 |
| revisit | 0.144502 | 0.147791 | +2.28% | 1/10 |
| novel | 0.087203 | 0.089860 | +3.05% | 3/18 |
| 2-leg | 0.117544 | 0.122306 | +4.05% | 0/11 |
| Goal C | 0.124825 | 0.125239 | +0.33% | 2/6 |
| 3-leg Goal-C revisit | 0.114822 | 0.115805 | +0.86% | 1/5 |
| remaining span >=256 | 0.186156 | 0.186851 | +0.37% | 1/4 |
| hard turn | 0.228850 | 0.230840 | +0.87% | 1/2 |

The all-row paired bootstrap interval for treatment minus control is positive
(`approximately [0.00172, 0.00411]`). Goal sensitivity also falls from
`0.006234` to `0.006148`. The experiment fails every intended reason for a long
run: it does not improve Goal C or span>=256, it regresses 2-leg by more than
2%, and it does not increase goal sensitivity.

The v1 ungated adapter and v2 curvature-gated adapter give nearly identical
action results even though v2's mean gate is only 0.0041. This shows that a
small early residual can still change the short optimization trajectory; it is
not evidence that scale should simply be enlarged.

## Result 3: simple memory traceback is not the missing planner

A read-only diagnostic formed directions from the current LingBot pose to
historical poses at approximately 8, 32, and 96 raw frames behind the current
frame, clamped at the live retrieval anchor. On the ten revisit rows, its mean
h2/h8/h24 errors against the expert route are approximately
`158.1/134.1/100.0` degrees. On five Goal-C revisit rows they are
`150.3/132.3/112.7` degrees.

This is expected geometrically: a past frame is behind the robot, while the
shortest expert path may first continue forward and turn later. In the hardest
example, the target starts forward at h2 and becomes backward by h24. Directly
replaying history in reverse is therefore not a substitute for graph search.

## Diagnosis

Four effects jointly explain the failure:

1. Hard route changes are rare under the 20-step `random_leg` sequence. Only
   two of the twenty sampled rows were marked decision-hard.
2. With batch size 1, the current sample-level curvature reweighting is
   normalized away. It also applies one weight to every horizon, so it does not
   preferentially correct h24 on a bending route.
3. The adapter pools current/revisit/novel tokens but does not consume an
   ordered topological memory or map. Endpoint information alone does not
   identify which side of an obstacle a geodesic route takes.
4. The residual is allowed to affect action before the route predictor beats a
   trivial forward baseline. Zero initialization protects step zero, not the
   first several optimization steps.

## Decision and next experiment

The residual route sketch remains default-off and is rejected for an 8-hour
training submission. No long job should use this configuration.

If the route direction line is revisited, the minimum next protocol is:

1. route-head-only warm-up with residual fixed at exactly zero;
2. hard-turn/long-span sampling plus per-horizon curvature weighting that still
   changes gradients at batch size 1;
3. fixed-set validation against the always-forward baseline before the action
   path can be opened;
4. a ramped residual followed by the same paired full-DDPM acceptance gates.

The lower-risk alternative remains a goal-conditioned candidate ranker. The
existing best-of-8 oracle already reduces all-row candidate MSE from about
`0.10072` to `0.07002`, and best-of-32 reaches about `0.06427` on 3-leg Goal C.
That evidence shows useful trajectories exist in the diffusion distribution;
selection has substantially more measured headroom than the present route
adapter. A production ranker must use only current/revisit/novel/goal condition
and geometric collision cues at inference, with the generator initially
frozen.

## Verification

Final child-worktree checks:

- `PYTHONPATH=InternNav:InternNav/src/diffusion-policy conda run -n memnav
  python -m pytest -q InternNav/tests/unit_test`: `105 passed`;
- focused route/checkpoint/evaluator tests after final edits: `28 passed`;
- `conda run -n memnav python -m pip check`: `No broken requirements found`;
- `python -m py_compile` on every changed Python implementation file: passed;
- `bash -n InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch`: passed;
- `git diff --check`: passed;
- parent repository status: clean `main`.

The 8-hour job is intentionally not submitted because the local acceptance
thresholds failed.
