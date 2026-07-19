# MemNav action-safe range gradient diagnostic (2026-07-19)

## Scope

This report diagnoses the 400-step `range + live-anchor` treatment and records the
next controlled experiment. It is an offline training diagnostic, not a Habitat
closed-loop navigation result. Source changes and local artifacts are confined to
the personal worktree `/home/asus/Research/Nav-axis-fix`; the mother checkout
`/home/asus/Research/Nav` remains read-only.

The controlled 400-step runs are:

- baseline: `mkf-train-1371557-400`, checkpoint SHA256
  `a6d66a66d3e316da6835c1d8e835e7f007357a782ffb7f22277da5a41b4333d1`;
- treatment: `range-live-train-8414d2d-400`, checkpoint SHA256
  `9d2feba94c51a886df4dd3b63b1f768360af753701346982e7fab914a6e85d30`.

Both formal runs used the same production sparse cache, training/validation split,
seed, batch size, fixed-64 validation population, and 400-step bound. The treatment
changed range weight from `0` to `0.2` and decayed anchor teacher forcing from
`1.0` to `0.5`.

## Observed 400-step regression

On the identical fixed-64 single-timestep validation population:

| Metric | Baseline | Range + live | Relative change |
|---|---:|---:|---:|
| action noise MSE | 0.095227 | 0.122679 | +28.8% |
| retrieval loss | 0.445679 | 0.409961 | -8.0% |
| direction error | 8.112 deg | 8.287 deg | +2.2% |
| adapted range-code MAE | n/a | 0.170230 | 9.6% below treatment raw code |

Goal A/B/C action losses all regressed, so the failure is not isolated to one long
Goal-C bucket. The treatment successfully learned its range target, but that did
not translate into a better action objective.

The paired full-DDPM check used the same 28 of 29 local fixed-leg samples,
identical per-sample diffusion randomness, and the same checkpoint-400 pair:

| Full-DDPM metric | Baseline | Range + live | Relative change |
|---|---:|---:|---:|
| all action MSE | 0.102461 | 0.137217 | +33.9% |
| revisit action MSE | 0.139368 | 0.183820 | +31.9% |
| novel action MSE | 0.081958 | 0.111326 | +35.8% |
| goal-sensitivity MSE | 0.007840 | 0.002654 | -66.1% |

Only one of 28 samples improved. The paired mean action-MSE delta was
`+0.034756` with a 50,000-resample bootstrap 95% interval of
`[+0.025934, +0.044196]`; all ten revisits regressed. The lower goal sensitivity
and shuffled-goal penalty show that the treatment used the goal less, rather than
merely moving error between action coordinates.

The production fixed-64 evaluator subsequently completed both immutable
`e658fa3` jobs with exit code `0:0`. It used the same held-out fingerprint,
selection indices, strict cache signature, diffusion seed, and paired randomness:

| Fixed-64 full-DDPM metric | Baseline | Range + live | Relative change |
|---|---:|---:|---:|
| all action MSE | 0.084969 | 0.120480 | +41.8% |
| revisit action MSE | 0.094307 | 0.135802 | +44.0% |
| novel action MSE | 0.075631 | 0.105158 | +39.0% |
| x / y / theta MSE | 0.079921 / 0.062499 / 0.112487 | 0.121904 / 0.094274 / 0.145262 | +52.5% / +50.8% / +29.1% |
| goal-sensitivity MSE | 0.003396 | 0.001851 | -45.5% |
| shuffled-goal penalty | 0.006239 | 0.001775 | -71.5% |
| revisit range-code MAE | 0.211636 | 0.196399 | -7.2% |

All 64 paired action deltas were positive. Their mean was `+0.035511`, with a
100,000-resample bootstrap 95% interval of `[+0.030018, +0.041533]`. Revisit and
novel intervals were also strictly positive, and every A/B/C row regressed. This
rules out a small-sample or one-goal-bucket explanation for the treatment failure.

## Shared-gradient audit

`aux_range_code` is coordinate 2 of the exact adapted four-dimensional pose code
consumed by `revisit_head`. Therefore range supervision and action supervision both
update `revisit_merge.rel_adapter`. A temporary read-only diagnostic loaded the
baseline step-400 checkpoint, forced a positive retrieval anchor, used identical
training-noise draws, and computed both gradients on that shared adapter without an
optimizer step.

At batch size 1 over ten fixed revisits:

| Weighted auxiliary gradient | Median norm / action norm | Mean cosine | Negative fraction |
|---|---:|---:|---:|
| direction (`0.2 x loss`) | 0.82 | -0.142 | 60% |
| range (`0.2 x loss`) | 17.44 | -0.082 | 60% |
| direction + range | 18.10 | +0.013 | 50% |

At the formal batch size 4 over all seven local batches containing a valid revisit:

| Weighted auxiliary gradient | Median norm / action norm | Mean cosine | Negative fraction |
|---|---:|---:|---:|
| direction (`0.2 x loss`) | 3.22 | +0.070 | 57.1% |
| range (`0.2 x loss`) | 29.02 | +0.274 | 28.6% |
| direction + range | 29.21 | +0.342 | 28.6% |

Batch averaging makes the mean range cosine positive, but it does not solve the
magnitude problem: even after multiplying by `0.2`, the median range update is
about 29 times the action update on their shared parameters. Two of seven batches
are both very large and adversarial. This directly explains how range MAE can
improve while action quality degrades. It can also perturb global gradient clipping,
although the complete model's logged pre-clipping norm is already large in both
baseline and treatment, so clipping alone is not claimed as the sole cause.

## Action-safe range update

The new mechanism is opt-in through
`MEMNAV_AUX_RANGE_GRAD_CAP_RATIO=rho`. Let `g_a` be the action gradient and `g_r`
the already loss-weighted range gradient on the complete shared `rel_adapter`
parameter group.

First remove only a conflicting component:

```text
g_projected = g_r - min(<g_a, g_r>, 0) / max(||g_a||^2, eps) * g_a
```

Then cap its global norm:

```text
scale = min(1, rho * ||g_a|| / max(||g_projected||, eps))
g_safe = scale * g_projected
```

The trainer adds a zero-valued surrogate whose gradient is
`g_safe - g_r`. Thus the reported scalar objective is unchanged, the adapted
range coordinate still learns, but the range contribution on shared action features is
first-order non-adversarial and bounded. Direction, retrieval, gate, and action
gradients retain their previous definitions.

The implementation logs:

- `range_grad_action_cosine`;
- `range_grad_raw_to_action_norm`;
- `range_grad_corrected_to_action_norm`;
- `range_grad_cap_scale`;
- `range_grad_conflict`.

The default ratio is `0`, which preserves the legacy backward exactly. Checkpoint
metadata records the ratio; old metadata without the field migrates only to the
equivalent default-off value. The current implementation deliberately fails closed
for multi-process DDP because the inner gradient queries are not safe to run through
the DDP reducer twice; the controlled arm is a single-GPU job.

## Next controlled arm

The proposed formal arm changes only range supervision relative to the sparse
baseline:

```text
MEMNAV_W_AUX_RANGE=0.2
MEMNAV_AUX_RANGE_BETA=0.1
MEMNAV_AUX_RANGE_GRAD_CAP_RATIO=0.25
MEMNAV_ANCHOR_TF_START=1.0
MEMNAV_ANCHOR_TF_END=1.0
MEMNAV_ANCHOR_TF_DECAY_STEPS=0
```

The live-anchor schedule is deliberately removed from this arm. The previous run
combined two changes, and exposing action training to a known-negative discrete
anchor is not justified by the observed regression. Evaluation remains live-anchor,
and oracle-positive retrieval remains a diagnostic.

## Validation status

- fail-closed evaluator dependency/cache-contract tests: passed;
- gradient projection and zero-valued gradient-replacement tests: passed;
- all MemNav unit tests: `59 passed`;
- Python compile, Slurm syntax, and `git diff --check`: passed;
- local real-model 20-step backward/checkpoint smoke: passed with no non-finite
  trainer-state value;
- on six revisit batches, raw weighted range/action gradient ratio averaged
  `33.16`, three were conflicting, and every corrected ratio was exactly `0.25`;
- paired local fixed-10, three-repeat full-DDPM action MSE was `0.268001` for the
  baseline and `0.264083` for action-safe range (-1.46% overall); the five-revisit
  slice was `0.304336` versus `0.319791` (+5.1%), with a paired bootstrap interval
  crossing zero, so this is a safety smoke rather than an improvement claim;
- adapted range-code MAE on those five revisits changed only from `0.088754` to
  `0.088724` over 20 steps, as expected under the strong cap;
- production fixed-64 full-DDPM baseline and range-live jobs: both completed
  `0:0`; range-live regressed on all 64 paired samples as reported above.

The 8-hour arm was therefore justified only as a controlled test of whether bounded
range learning avoids the proven 400-step regression. No production default or
navigation-quality claim was made before its paired acceptance evaluation.

## Formal controlled-arm result

The full dependency chain completed successfully:

- zero-step preflight `14215548`: `COMPLETED 0:0`;
- ten-step smoke `14215550`: `COMPLETED 0:0`;
- 400-step H200 training `14215557`: `COMPLETED 0:0` in `03:50:11`;
- fixed-64 paired full-DDPM evaluator `14250526`: `COMPLETED 0:0` in
  `00:37:37`.

The formal training run preserved the expected train, validation, and fixed-64
fingerprints and wrote complete checkpoints at steps 100/200/300/400. Across 40
ten-step windows, the already weighted raw range gradient was `65.41` times the
action gradient in median and `75.95` times in mean on the shared adapter. Mean
conflict fraction was `50.8%`; every corrected ratio was exactly `0.25`, with no
non-finite value. The implementation therefore behaved as designed.

The final single-timestep fixed-64 action loss was `0.115014`: `6.25%` below the
rejected range+live arm but `20.78%` above baseline. The exact paired DDPM test
used the same 64 sample identities, selection fingerprint, cache contract,
diffusion seed `104729`, and shuffled-goal randomness as the two immutable
comparison reports:

| Fixed-64 full-DDPM metric | Baseline | Range + live | Action-safe range |
|---|---:|---:|---:|
| all action MSE | 0.084969 | 0.120480 | 0.112534 |
| revisit action MSE | 0.094307 | 0.135802 | 0.124235 |
| novel action MSE | 0.075631 | 0.105158 | 0.100834 |
| x / y / theta MSE | 0.079921 / 0.062499 / 0.112487 | 0.121904 / 0.094274 / 0.145262 | 0.112683 / 0.085809 / 0.139112 |
| goal-sensitivity MSE | 0.003396 | 0.001851 | 0.001980 |
| shuffled-goal penalty | 0.006239 | 0.001775 | 0.001734 |
| revisit range-code MAE | 0.211636 | 0.196399 | 0.199157 |

Action-safe range recovered part of the old failure (`-6.60%` action MSE versus
range+live), but still regressed `32.44%` versus baseline. Only 3 of 64 paired
samples improved. Mean paired delta was `+0.027565`; its 100,000-resample
bootstrap 95% interval was `[+0.022543, +0.032802]`. Revisit and novel regressed
`31.73%` and `33.32%`, and Goal A/B/C regressed `33.9% / 38.8% / 26.3%`.
All three remaining-path-span buckets also regressed, so the failure is not a
long-range-only or LingBot-drift-only effect.

The controlled arm is rejected. Projection guarantees only non-adversarial
first-order alignment on the current batch; an orthogonal update repeated over
400 steps can still hurt future batches and shared-decoder generalization. The
range label should remain default-off. A stronger follow-up would train range in
a detached calibration branch and expose it to the policy only through a
zero-initialized gate optimized by action loss, so auxiliary supervision cannot
directly rewrite the action representation.
