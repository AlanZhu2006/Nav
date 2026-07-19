# Long-route decision curriculum: local validation

Date: 2026-07-20 (Asia/Shanghai)

This is an offline/local training diagnostic. It is not a closed-loop navigation
result and it does not establish 3-leg success in Habitat.

## Question

The sparse-keyframe interval fix is already active and audited. What still limits
long 3-leg behavior, and what is the smallest inference-safe change worth testing in
the next training run?

## Interval is no longer the primary failure

All 12 local cache pairs are schema-versioned and share precompute signature
`97a7819c4722ebf6e2165538b4908a276d426faf3190f18987332f59889c9afc`.
They use `window=32`, `num_scale=8`, and
`keyframe_interval=ceil(num_frames/320)`, retain the original frame indices for
3-D RoPE, and contain at most 296 sparse anchor frames. Intervals are
`{1: 2, 2: 5, 3: 4, 5: 1}`; the 1329-frame episode uses interval 5 and retains raw
index 1328.

Against the old dense cache, this policy reduced mean 3-leg Sim(2)-aligned ATE from
2.295 m to 0.552 m (76% lower). The remaining 3-leg leg-direction medians are about
2 degrees and the leg-3 distance ratio is about 0.904. Therefore another interval
change is not justified by the current evidence.

## Pose controls and rejected alternatives

The official LingBot `gct_stream_window` reset path was tested on all six local
3-leg episodes with window 128 and overlap 16. Mean ATE changed from 0.5521 m to
0.4690 m (15.1% lower), median from 0.4807 m to 0.3251 m (32.4% lower), and RPE RMSE
improved at gaps 16/64/128/256. It nevertheless lost on two of six episodes,
including the two longest/hardest streams.

Neither tested way of inserting the reset trajectory into the current goal-pose
pipeline was safe:

- stitching overlap by camera-pose scale made the longest episode ATE 1.414 m,
  worse than both continuous sparse pose (0.981 m) and depth-aligned windowing
  (1.079 m);
- transferring the live goal append from the continuous map into the reset map at
  the retrieval anchor increased mean direction error on five Goal-C rows from
  9.55 to 15.53 degrees.

The anchor-to-goal baselines are only 0.19--0.46 m in four of those five rows, so
small cross-map alignment error becomes a large bearing error. Window resets remain
a promising separate mapping experiment, but they must not be coupled to the next
training arm yet.

## Remaining failure is concentrated in long decisions

Best audited checkpoint:
`.diagnostics/checkpoints/mkf-1371557-step400.ckpt`.
Fixed-28 full-DDPM report:
`.diagnostics/range_live_ab/mkf400_full28_ddpm_goalshuffle.json`.

The checkpoint retrieves the correct historical region for all five revisit 3-leg
Goal-C rows. Zeroing the revisit condition increases paired action MSE by about 40%,
so memory is used. Oracle-positive retrieval does not materially improve auxiliary
pose error, and pose direction error does not positively explain action MSE on the
ten revisit rows.

By contrast, full-DDPM action error is strongly concentrated where the feasible
near-term route initially disagrees with the direct endpoint bearing. Identity-based
joining (rather than the stale positional-index join used in an earlier scratch
calculation) finds two such rows on fixed-28. Their mean full-DDPM MSE is 0.2255
versus 0.0930 for the other rows (2.43 times). Across the set, full-DDPM action MSE
correlates with route-disagreement angle at 0.747, with remaining span at 0.578, and
with goal distance at 0.592. The previous 400-step W&B
run was still improving, while its gap-512+ and span-256+ action losses remained
well above the overall action loss.

The hardest long Goal-C condition can also be weakly goal-sensitive even when its
retrieval and pose are correct. Across six 3-leg Goal-C rows, the paired goal-shuffle
sensitivity ratio is 0.135 versus 0.279 on 2-leg rows; the longest Goal-C row is only
0.011. This supports insufficient goal-dependent action training rather than another
global pose-axis correction.

## Implemented training-only sampler

`sampling_mode=decision_curriculum` defines a hard k as:

```text
remaining = goal_step - k >= 128
angle(position[k+16] - position[k], position[goal] - position[k]) >= 45 deg
```

The angle is invariant to the later local-frame rotation. It uses future geometry
only to select a supervised training row. It is never passed to the policy, fixed
evaluation remains unchanged, and inference has no dependency on ground-truth
future poses.

For each uniformly selected goal-sample, the sampler draws from its hard-k pool with
probability 0.5 when the pool is non-empty; otherwise it performs the historical
uniform draw. The remaining 0.5 uniform branch preserves broad coverage and avoids
turning this into a 3-leg-only or revisit-only dataset.

Exact enumeration of all valid k on the local 29-goal/12-episode set gives:

| Equal-goal sampling statistic | Uniform k | 0.5 curriculum |
| --- | ---: | ---: |
| hard fraction, all goals | 0.1346 | 0.3604 |
| hard fraction, 3-leg Goal C | 0.3063 | 0.6531 |
| revisit fraction, all goals | 0.3448 | 0.3448 |
| revisit fraction, 3-leg Goal C | 0.8333 | 0.8333 |
| mean remaining frames, all goals | 97.7 | 123.5 |
| mean remaining frames, 3-leg Goal C | 176.8 | 224.8 |

Thus the intended exposure rises 2.67 times without changing the revisit/novel class
balance. A 232-row check against actual NavDP action labels found that the vectorized
16-frame proxy matches the first-four-action route angle with 0.134 degree MAE,
0.99997 correlation, and 100% hard/easy classification agreement.

## Local execution checks

- Python compilation passed for every changed Python module.
- `bash -n` and `git diff --check` passed.
- The clean submission branch passed every test it contains: `64 passed` through
  pytest and 62 through the documented unittest-discovery entry point. The broader
  integration worktree, which also contains unrelated detached-range evaluator tests,
  passed 71/69 respectively before this change was isolated.
- A real two-optimizer-step GPU smoke used strict local sparse caches, batch size 1,
  workers 0, no W&B, and `decision_curriculum_prob=1.0`. Step 2 sampled a 131.21-degree
  decision row, logged `decision_hard_fraction=1.0`, completed finite action/gate
  backward, and wrote the complete checkpoint/optimizer/scheduler/RNG/trainer state.
- Fixed-28 parent and subset fingerprints were restored exactly to the audited legacy
  values `e60441f...` and `67e8e611...`; curriculum has a distinct fingerprint
  `01e8b1f...`. Dormant curriculum knobs therefore do not block legitimate old-run
  resume.

The first complete evaluator launch was correctly rejected before evaluation because
the personal worktree does not duplicate the frozen DINO checkpoint. All subsequent
runs explicitly set `MEMNAV_DINO_WEIGHTS` to the existing read-only checkpoint at
`/home/asus/Research/Nav/InternNav/checkpoints/depth_anything_v2_vits.pth`; the model
reported that DINO initialization succeeded. This path must be supplied explicitly on
another host rather than relying on a worktree-relative default.

## Paired 20-step continuation screen

All arms below start from the same `mkf-1371557-step400.ckpt`, seed 0, batch size 1,
and strict sparse cache. The short local scheduler is intentionally only a logic
screen: with `max_steps=20` its cosine learning rate reaches nearly zero, so these
numbers do not define the production schedule.

Resetting continuation to learning rate `1e-4` was unsafe. Fixed-28 noise MSE was
0.18281 for uniform continuation and 0.15884 for decision continuation, both much
worse than the untouched checkpoint's 0.09319. The paired low-learning-rate screen
therefore uses `1e-5`.

The decisive comparison is complete DDPM sampling with the same initial diffusion
noise (`seed=104729`) and the same within-batch cyclic goal shuffle:

| Fixed-28 group | Untouched step-400 | Uniform 20-step | Decision 20-step | Decision vs uniform |
| --- | ---: | ---: | ---: | ---: |
| all | 0.10246 | 0.10767 | 0.09967 | -7.43% |
| hard turn | 0.22554 | 0.22885 | 0.21848 | -4.53% |
| remaining span >=128 | 0.18490 | 0.20681 | 0.19317 | -6.59% |
| remaining span >=256 | 0.14779 | 0.18616 | 0.15996 | -14.07% (4/4 rows) |
| 3-leg Goal C | 0.11237 | 0.12482 | 0.10872 | -12.90% (5/6 rows) |
| 3-leg Goal C revisit | 0.10735 | 0.11482 | 0.10288 | -10.40% (4/5 rows) |
| 2-leg | 0.11039 | 0.11754 | 0.11408 | -2.95% |
| revisit | 0.13937 | 0.14450 | 0.13944 | -3.50% |
| novel | 0.08196 | 0.08720 | 0.07758 | -11.04% |

Decision continuation also has higher paired goal-sensitivity ratio than uniform on
every listed long-route grouping: +7.05% on span >=128, +12.58% on span >=256,
+5.57% on 3-leg C, and +3.98% on 3-leg C revisit. Retrieval accuracy and gate
separation are unchanged. Thus the sampler has a real same-budget advantage over
ordinary continuation; the effect is not explained by merely taking 20 more steps.

It has not, however, solved long navigation. Relative to untouched step-400, decision
continuation still regresses 4.47% on span >=128, 8.24% on span >=256, and 3.35% on
2-leg, although it improves all-sample MSE by 2.72% and 3-leg C by 3.25%. These are
offline action diagnostics, not closed-loop success.

## Rejected independent long-path pool

A follow-up three-way sampler used 50% uniform, 25% route-disagreement, and 25%
independent span >=256 sampling. Exact enumeration doubled the all-goal long-row rate
from 6.9% to 13.7% and raised the 3-leg-C long-row rate from 22.4% to 41.1% without
changing revisit balance. Its same-seed 20-step checkpoint nevertheless failed the
cheap fixed-noise gate:

| Fixed-noise group | Untouched | Uniform | Accepted decision | Three-way |
| --- | ---: | ---: | ---: | ---: |
| all | 0.09319 | 0.09248 | 0.08903 | 0.09320 |
| span >=256 | 0.07733 | 0.10313 | 0.09089 | 0.09221 |
| 3-leg C | 0.03703 | 0.04851 | 0.04021 | 0.04658 |
| 3-leg C revisit | 0.03497 | 0.04614 | 0.03878 | 0.04361 |

Because it was 15.83% worse than the accepted decision arm on 3-leg C and 19.24%
worse than untouched step-400 on span >=256, the three-way logic was rejected before
the expensive full-DDPM stage and removed from the code. More exposure to long
straight-looking rows is not by itself sufficient; the supported signal is specifically
long-horizon route disagreement.

## Next controlled experiment and rejection rule

The next long run should change only the sampling mode and its four documented knobs;
range loss and pose-reliability conditioning stay off, anchor teacher forcing stays at
the accepted baseline, and sparse-cache geometry/signature stay fixed. A scientifically
clean comparison starts the uniform-continuation and curriculum-continuation arms from
the same checkpoint and seed, rather than comparing unrelated fresh initializations.

The scientifically justified next experiment is a paired uniform/decision continuation
from the same checkpoint and seed, long enough that the result is not determined by
twenty stochastic samples. Accept the sampler only if fixed full-DDPM evaluation improves long Goal-C and
span-256+ action metrics without a material regression in 2-leg/easy rows, retrieval,
gate recall, or overall action MSE. Training loss alone is not sufficient, and no
offline result should be reported as closed-loop 3-leg navigation success.
