# MemNav revisit range / metric-scale local diagnostic (2026-07-18)

## Scope

This is a fixed offline diagnostic of the step-400 checkpoint, not a closed-loop
Habitat navigation result. All new source, reports, and checkpoints are under the
personal worktree `/home/asus/Research/Nav-axis-fix`; the mother checkout
`/home/asus/Research/Nav` is only used as a read-only source for frames and frozen
weights.

Checkpoint:

- file: `.diagnostics/checkpoints/mkf-1371557-step400.ckpt`
- SHA256: `a6d66a66d3e316da6835c1d8e835e7f007357a782ffb7f22277da5a41b4333d1`
- source training run: `mkf-train-1371557-400`, step 400 (about 0.404 epoch)

Dataset and cache:

- 29 deterministic fixed-goal samples from 12 local episodes;
- 10 samples are dynamically valid revisits at the selected current frame;
- the revisit split contains 5 Goal-B and 5 Goal-C samples;
- current sparse, versioned LingBot caches are used;
- dataset fingerprint:
  `056f5142470723d313d9f7f9d78d498d6460949c026121aa9c43487effe5c865`.

## Controlled intervention

`scripts/eval/diag_memnav_range_scale.py` encodes each batch once. It keeps the
following values exactly paired:

- current image, goal image, memory, retrieval logits, selected anchor and bearing;
- revisit gate and pose reliability;
- diffusion target, noise, timestep and (for full DDPM) every random seed.

It changes only coordinate 2 of the four-dimensional revisit pose code, and only
on rows labelled revisit:

1. `current`: current LingBot stream-normalized range;
2. `zero_range`: range code is zero;
3. `oracle_stream_range`: GT endpoint distance divided by the median past GT step;
4. `odom_metric_range`: current LingBot range calibrated using past executed
   motion, without looking at the future goal;
5. `oracle_metric_range`: GT endpoint distance in the canonical nominal metric
   scale (upper bound).

The report distinguishes `range_code_*` (raw code produced by the fixed pose
encoder) from `adapted_range_code_*` (raw code plus the trainable residual
adapter). The latter is the value consumed by the policy and supervised by the
new range auxiliary. Conflating the two would incorrectly make a trained
adapter appear unchanged.

The constructed `current` path exactly reproduced the existing evaluator's
4-sample action MSE (`0.1957736862823367`), which checks that the diagnostic does
not change the production condition or random draw.

## Results

### Range quality itself

On the 10 revisit samples:

| Metric | Result |
|---|---:|
| Pearson, current range vs GT range | 0.9781 |
| Spearman, current range vs GT range | 0.9879 |
| Online action-calibrated metric MAE | 0.7009 m |
| Online action-calibrated metric RMSE | 1.1114 m |
| Median relative metric error | 12.60% |
| Optimistic global correction factor | 1.1642 |
| MAE after optimistic global factor | 0.5672 m |
| Leave-one-episode-out scalar MAE | 0.6709 m |

The median past GT step is `0.037462 m/frame`; across these samples it ranges only
from `0.037379` to `0.037573 m/frame`. The generator nominal value is
`0.0376 m/frame`. Therefore the current stream-normalized range is already almost
metric on this dataset, and the online odometry correction is almost the identity.

The residual is distance-dependent:

- Goal B metric MAE: `0.364 m`;
- Goal C metric MAE: `1.038 m`;
- representative long C samples: `8.94 -> 7.70 m` and `11.34 -> 8.24 m`.

Thus long-range underestimation is real, but it is not an arbitrary per-episode
metric gauge failure. A single scalar cannot remove all residual drift.

### Paired single-timestep denoising objective

| Revisit condition | Action MSE | Delta from current |
|---|---:|---:|
| current | 0.077963 | -- |
| zero range | 0.080525 | +0.002562 (+3.29%) |
| oracle stream range | 0.077946 | -0.000017 (-0.02%) |
| online metric range | 0.077961 | -0.000002 (-0.003%) |

### Complete 10-step reverse diffusion, three paired seeds

| Revisit condition | Sampled-action MSE | Delta from current |
|---|---:|---:|
| current | 0.130877 | -- |
| zero range | 0.136575 | +0.005698 (+4.35%) |
| oracle stream range | 0.130637 | -0.000239 (-0.18%) |
| online metric range | 0.130873 | -0.000003 (-0.002%) |

The effect differs by goal leg:

| Group | Current | Zero range delta | Oracle range delta |
|---|---:|---:|---:|
| Goal B (5) | 0.152848 | -0.008867 | +0.000782 |
| Goal C (5) | 0.108905 | +0.020263 | -0.001260 |

For the 6 goals beyond the approximately 96-frame action horizon, zeroing range
increases sampled-action MSE by `0.00903` on average. Exact metric correction is
still negligible. The decoder mainly uses range as a coarse signal that the goal
is still far away; it does not need centimetre-accurate endpoint distance to emit
the next finite action horizon.

### Retrieval-anchor control

Live retrieval selects a labelled positive on 8/10 revisit samples. Forcing a
GT-positive anchor changes only 2 anchors. The mean paired current-action MSE
change is `+0.0000019`, effectively zero. The range/action conclusion therefore
is not caused by a wrong retrieval anchor on this local set.

## Twenty-step local training A/B

Three runs start from the same step-400 checkpoint, use seed 0, batch size 1,
the same 29-sample shuffle and exactly 20 optimizer steps:

| Arm | Range weight | Anchor teacher forcing |
|---|---:|---:|
| baseline | 0.0 | 1.0 |
| range-only | 0.2 | 1.0 |
| range + live anchor | 0.2 | 0.5 |

There are only six revisit training rows in those 20 steps. In the live arm,
two of the six anchors were teacher-forced and five of six selected anchors were
labelled positive. A wrong live anchor still trains action and reliability, but
is excluded from direction/range supervision so the pose adapter is not taught
to repair a semantically unrelated frame.

All checkpoints were evaluated on the same balanced fixed subset (indices
`[2, 7, 8, 10, 12, 15, 19, 20, 21, 28]`), with five revisits and five novel
samples. The subset fingerprint is
`056f5142470723d313d9f7f9d78d498d6460949c026121aa9c43487effe5c865`.
Complete 10-step DDPM sampling uses three paired seeds.

| Metric | Baseline | Range-only | Range + live |
|---|---:|---:|---:|
| Adapted range-code MAE, revisit | 0.088754 | 0.088537 | 0.088513 |
| Full-DDPM action MSE, all | 0.268001 | 0.275417 | 0.267845 |
| Full-DDPM action MSE, revisit | 0.304336 | 0.304410 | 0.303677 |
| Full-DDPM action MSE, Goal B revisit | 0.318517 | 0.312210 | 0.318708 |
| Full-DDPM action MSE, Goal C revisit | 0.283064 | 0.292711 | 0.281132 |

The combined arm improves revisit action MSE by `0.000658` (`0.216%`) and wins
on four of five paired revisit samples, but Goal B is `0.060%` worse. Range-only
slightly improves the adapted-code target yet does not improve aggregate revisit
action MSE. This shows that the loss is connected and learns the intended
coordinate, but the 20-step evidence is too small and heterogeneous to establish
generalization or justify a new production default.

For the baseline, forcing revisit range to zero raises full-DDPM revisit action
MSE from `0.304336` to `0.319458` (`4.97%`), so the range signal is useful.
Replacing it with oracle stream range changes MSE only to `0.304297` (`0.013%`
better), so exact metric scale is not the current action bottleneck.

## Conclusion

Metric scale is not the current primary bottleneck and should not be made a
blocking production change before the next training run:

- the corrected range has very high rank and linear correlation with GT;
- past-action metric calibration is nearly an identity on this generator;
- replacing current range with GT improves complete diffusion by only 0.18%;
- removing range hurts, especially for long Goal-C samples, so range must stay.

The more important unresolved issue is that the checkpoint uses range
inconsistently across B and C. This is compatible with three known facts: training
ended before half an epoch, the action objective only predicts a finite local
horizon, and the auxiliary loss directly supervises bearing but not range usage.

The next controlled training experiment should therefore prioritize:

1. longer exposure with balanced short/long and Goal-B/Goal-C samples;
2. scheduled live-anchor exposure to remove train/eval teacher-forcing mismatch;
3. an explicit coarse/ordinal near-mid-far range objective or separately routed
   range token, while retaining the current gauge-invariant input;
4. evaluation by complete paired diffusion and distance/horizon buckets.

A hard-coded global `x1.164` inference correction is not recommended: it is fitted
on only 10 revisits, differs between B and C, and produced almost no action gain.

The implementation therefore keeps both new knobs default-off/legacy-safe:
`MEMNAV_W_AUX_RANGE=0` and teacher forcing `1 -> 1`. A larger experiment must be
an explicit controlled arm, not a silent change to the production baseline.

## Artifacts

- live paired diagnostic:
  `.diagnostics/range_scale/paired_full_live.json`
- oracle-positive anchor diagnostic:
  `.diagnostics/range_scale/paired_full_oracle_anchor.json`
- full DDPM, three seeds:
  `.diagnostics/range_scale/full_diffusion_full_live_seed104729_r3.json`
- evaluator:
  `scripts/eval/diag_memnav_range_scale.py`
- unit test:
  `tests/unit_test/test_memnav_range_diagnostic.py`
- local 20-step outputs (gitignored):
  `.diagnostics/range_live_ab/`

## Dependency and test audit

The diagnostic uses dependencies already declared by InternNav: NumPy, pandas,
PyArrow, PyTorch and diffusers. It deliberately does not add SciPy. The local
`memnav` environment was missing the pytest tools already declared in
`requirements/core_requirements.txt`; these were installed within the declared
version ranges (`pytest 7.4.4`, `pytest-cov 4.1.0`, `pytest-timeout 2.4.0`).

Checks completed:

- checkpoint SHA256 matches the remote formal-run record;
- `pip check`: no broken requirements;
- Python compile/import checks: passed;
- `python -m pytest -q tests/unit_test/test_memnav*.py`: `57 passed`;
- strict 0-step checkpoint/data/model/Trainer preflight: passed; all 379
  non-LingBot checkpoint tensors loaded and only 2611 frozen LingBot tensors were
  intentionally absent;
- the final preflight used cache schema 2, window 32, num-scale 8, max-frame 4096,
  signature `97a7819c4722ebf6e2165538b4908a276d426faf3190f18987332f59889c9afc`,
  and required the corrected generated Z-up pose convention;
- `.sbatch` syntax and `git diff --check`: passed;
- source files contain UTF-8/LF text, no tabs, no lines over 120 characters.

`black` and `isort` are configured by `pyproject.toml` but are not declared or
installed in this environment, so their CLI checks were not treated as runtime
dependency requirements.
