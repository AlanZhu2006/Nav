# Action-safe range training chain

## Purpose and controlled question

This run tests one question: can range-code supervision improve the relative-pose
feature without overwhelming the diffusion action gradient that uses the same
`revisit_merge.rel_adapter`? It is a controlled training arm, not a production
default and not a closed-loop Habitat result.

The comparison baseline is `mkf-train-1371557-400` (JobID `14124336`, W&B run
`memnav_mkf_1371557_400`). Relative to that baseline, the formal arm changes only:

- `MEMNAV_W_AUX_RANGE=0.2`, SmoothL1 beta `0.1`;
- `MEMNAV_AUX_RANGE_GRAD_CAP_RATIO=0.25`;
- the conflicting range-gradient component is projected away before the range
  adapter-gradient norm is capped at one quarter of the action adapter-gradient
  norm.

Anchor teacher forcing stays constant at `1.0`. Pose-reliability conditioning and
loss stay disabled. This deliberately excludes the live-anchor schedule that was
confounded with range supervision in the rejected treatment.

## Evidence that rejected the previous treatment

The immutable `e658fa3` fixed-64 evaluator completed the baseline and old
range+live checkpoint jobs with exit code `0:0`. Both used evaluation fingerprint
`c5bf140feb86a8863bbb65e46eb55cee39fa36b25a16504497679495328169df`,
the same 64 selection indices, strict cache signature, diffusion seed `104729`,
and paired diffusion randomness.

| Full-DDPM metric | Baseline | Range + live | Change |
|---|---:|---:|---:|
| all action MSE | 0.084969 | 0.120480 | +41.8% |
| revisit action MSE | 0.094307 | 0.135802 | +44.0% |
| novel action MSE | 0.075631 | 0.105158 | +39.0% |
| x / y / theta MSE | 0.079921 / 0.062499 / 0.112487 | 0.121904 / 0.094274 / 0.145262 | +52.5% / +50.8% / +29.1% |
| goal-sensitivity MSE | 0.003396 | 0.001851 | -45.5% |
| shuffled-goal penalty | 0.006239 | 0.001775 | -71.5% |
| revisit range-code MAE | 0.211636 | 0.196399 | -7.2% |

The treatment lost on all 64 paired action samples. Mean paired delta was
`+0.035511`; a 100,000-resample bootstrap 95% interval was
`[+0.030018, +0.041533]`. Therefore the old recipe learned its auxiliary target
while making the policy less goal-conditioned.

Gradient probes at formal batch size four explain the mechanism. The already
weighted range gradient was 29.02 times the action gradient in median on the
shared adapter, and two of seven revisit batches were adversarial. Scalar range
loss and global gradient clipping did not expose or prevent this imbalance.

## New mechanism and local validation

Code commit: `b10f0522149c87b21ce889e453255af3583b6912`.

The gradient replacement has zero forward value, so logged objective values keep
their ordinary meaning. Checkpoint metadata records the cap ratio. A zero ratio
preserves the legacy backward exactly. Multi-process DDP fails closed because the
inner gradient queries are not DDP-reducer safe; this chain requests one GPU.

Validation before submission:

- all `test_memnav*.py` unit tests: `59 passed`;
- Python compilation, Slurm `bash -n`, and `git diff --check`: passed;
- real local 20-step training: completed with checkpoint, optimizer, scheduler,
  RNG, Trainer state, and no non-finite logged value;
- six local revisit batches had mean raw range/action gradient ratio `33.16` and
  three conflicts; every corrected ratio was exactly `0.25`;
- fixed-10, three-repeat paired DDPM action MSE was `0.268001` for the local
  baseline and `0.264083` for action-safe range (-1.46% overall). The five-revisit
  slice was `0.304336` versus `0.319791` (+5.1%) with a bootstrap interval crossing
  zero. This is a safety smoke, not an improvement claim.

## Immutable deployment and file identity

- GitHub fork: `AlanZhu2006/Nav`;
- branch: `fix/memnav-gate-conditioning-20260717`;
- draft PR: `https://github.com/AlanZhu2006/Nav/pull/1`;
- deployed repository:
  `/scratch/yz11502/Research/Nav-memnav-b10f052-deploy`;
- deployed `HEAD`: `b10f0522149c87b21ce889e453255af3583b6912`;
- deployment tracked status: clean;
- `train_memnav_mp3d.sbatch` SHA256:
  `246f22fb826042275e36722534ceca8b8e2ef58508241a331358c82658462b4d`;
- `memnav_trainer.py` SHA256:
  `af3c0f2eef6fb8985fd66701d6771c03f73d287f82e06576884ffb3fdcd050e6`;
- `scripts/train/configs/memnav.py` SHA256:
  `141a165185636bdaf20f2ec21031058347e317a4ffe0d2dd012b6890173e9803`.

All three hashes matched between the personal worktree and deployment before
submission.

## Runtime and data dependency contract

- source overlay:
  `/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf`;
- in-container source root: `/mp3d_revisit_v0/vln_n1/traj_data`;
- sparse feature root:
  `/scratch/yz11502/Research/datasets/mp3d_revisit_v0_feat_kf320_1371557/vln_n1/traj_data`;
- required cache signature:
  `b25fb60ed3abd8bca31da996274eba22e1eec669d968a5b5a9d96fe6487fc809`;
- cache geometry: window / scale / max-frame `32 / 8 / 4096`;
- generated corrected-pose marker: mandatory;
- LingBot repository:
  `/scratch/yz11502/Research/lingbot-map-7ff6f3e-clean`;
- LingBot weights:
  `/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt`;
- DINO weights:
  `/scratch/yz11502/Research/checkpoints/depth_anything_v2_vits.pth`;
- Conda runtime:
  `/scratch/yz11502/conda_envs/memnav-runtime-8414d2d`;
- Conda activation script:
  `/scratch/lg154/miniconda3/etc/profile.d/conda.sh`;
- Apptainer image:
  `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`.

Every job checks the exact code commit, clean code and LingBot worktrees, both
weight files and hashes, `pip check`, CUDA imports, strict versioned caches,
generated pose marker, and dataset fingerprints. Dependencies use `afterok`, so
a failed check prevents the next stage from starting.

## Submitted dependency chain

### 1. Zero-step dependency preflight

- JobID: `14215548`;
- name: `range-safe-pre-b10f052`;
- time: `00:45:00`;
- resources: one GPU, 16 CPUs, 128 GiB RAM;
- batch / epochs / workers: `2 / 0 / 0`;
- fixed validation population: `64`;
- reporting / resume: `none / none`;
- final state: `COMPLETED`, exit code `0:0`, elapsed `00:05:12` on H200
  node `gh118`;
- stdout/stderr:
  `logs/train_memnav/range-safe-pre-14215548.{out,err}`.

### 2. Ten-step backward/checkpoint/evaluation smoke

- JobID: `14215550`;
- dependency: `afterok:14215548`;
- time: `02:00:00`;
- resources: one GPU, 16 CPUs, 128 GiB RAM;
- batch / workers / max steps: `4 / 0 / 10`;
- save / log / fixed-eval steps: `5 / 1 / 10`;
- fixed evaluation samples / batch: `16 / 4`;
- reporting / resume: `none / none`;
- final state: `COMPLETED`, exit code `0:0`, elapsed `00:09:44` on H200
  node `gh118`;
- both step-5 and step-10 checkpoints contain model, optimizer, scheduler, RNG,
  Trainer state, and MemNav metadata recording gradient cap `0.25`;
- seven revisit batches had raw weighted range/action adapter-gradient ratios
  `23.01, 2.42, 7.88, 79.53, 101.00, 24.89, 94.12`; all corrected ratios
  were exactly `0.25`, three batches had a conflicting component, and no
  non-finite diagnostic occurred;
- fixed-16 step-10 smoke metrics were action MSE `0.968933`, retrieval loss
  `1.140624`, gate loss `0.347705`, gate accuracy `0.875`, seen-goal match
  `0.750`, direction error `14.9069 deg`, and range MAE `0.201883`. These are
  deliberately early smoke values, not a performance comparison;
- stdout/stderr:
  `logs/train_memnav/range-safe-smoke-14215550.{out,err}`.

### 3. Eight-hour controlled W&B training

- JobID: `14215557`;
- dependency: `afterok:14215550`;
- time: `08:00:00`;
- partition: `a100_tandon,h100_tandon,h200_tandon`;
- resources: one GPU, 16 CPUs, 128 GiB RAM;
- batch / workers / max steps: `4 / 4 / 400`;
- save / log / fixed-eval steps: `100 / 10 / 100`;
- fixed evaluation samples / batch: `64 / 4`;
- range weight / beta / gradient cap: `0.2 / 0.1 / 0.25`;
- anchor teacher forcing: `1.0 -> 1.0 @ 0`;
- pose-reliability conditioning / loss: `0 / 0.0`;
- seed / split seed / sampling seed: `0 / 0 / 0`;
- BF16 / TF32 / gradient accumulation / max gradient norm:
  `0 / 1 / 1 / 1.0`;
- resume: none, using the same fresh initialization path as the baseline;
- W&B project / run ID:
  `memnav / memnav_range_safe_b10f052_400`;
- W&B URL:
  `https://wandb.ai/yz11502-new-york-university/memnav/runs/memnav_range_safe_b10f052_400`;
- final state: `COMPLETED`, exit code `0:0`, elapsed `03:50:11` on H200 node
  `gh118`;
- all four checkpoints at steps 100/200/300/400 contain model, metadata,
  optimizer, scheduler, RNG, and Trainer state;
- final `memnav.ckpt` SHA256:
  `a9933609c6614f563a9ceea371dbeb6cd5f8cf5a4590a7048a010bcf1392a81c`;
- final metadata SHA256:
  `5aa2e5efd16a0248ef0c3cba83843e694d61d3d996e739b287e53cfc28d53bdd`;
- train / validation / fixed-64 fingerprints matched the controlled baseline:
  `aa4c6d1a1799ac5338f1fa7734404406b88e7afb732d396470b38a43082033a9` /
  `414673cab0f1776e8c2f03c1c4dda60508b3a3735d0ad346e67ea6ea639a02ce` /
  `c5bf140feb86a8863bbb65e46eb55cee39fa36b25a16504497679495328169df`;
- over all 40 ten-step windows, the raw weighted range/action adapter-gradient
  ratio had median `65.41` and mean `75.95`; mean conflict fraction was `50.8%`.
  Every corrected ratio was exactly `0.25`, and no logged value was non-finite;
- stdout/stderr:
  `logs/train_memnav/range-safe-train-14215557.{out,err}`.

The fixed-64 single-timestep trajectory was:

| Step | Action loss | Retrieval loss | Direction error | Range-code MAE |
|---:|---:|---:|---:|---:|
| 100 | 0.202598 | 0.689358 | 5.8070 deg | 0.194685 |
| 200 | 0.191578 | 0.489251 | 6.0820 deg | 0.158404 |
| 300 | 0.129267 | 0.454138 | 7.3492 deg | 0.174353 |
| 400 | 0.115014 | 0.434202 | 9.6527 deg | 0.188160 |

At step 400, action loss was `6.25%` below the rejected range+live arm but
`20.78%` above the sparse baseline. This recovered `27.9%` of the old arm's
baseline gap, but did not satisfy the acceptance criterion. Goal A/B/C action
losses were all above baseline.

### 4. Immutable fixed-64 paired full-DDPM acceptance evaluation

- JobID: `14250526`;
- name: `ddpm64-safe-b10f052`;
- final state: `COMPLETED`, exit code `0:0`, elapsed `00:37:37` on A100 node
  `ga040`;
- evaluated code / checkpoint:
  `b10f0522149c87b21ce889e453255af3583b6912` /
  `checkpoint-400/memnav.ckpt`;
- selection, fingerprint, cache signature, diffusion seed `104729`, oracle
  diagnostic, and cyclic shuffled-goal randomness exactly matched both immutable
  `e658fa3` comparison reports;
- output:
  `/scratch/yz11502/Research/eval_outputs/ddpm64-range-safe-b10f052.json`;
- output SHA256:
  `1067a64052a1ef685c2708040abb0f9d19a643c0233dfc3af378cc1fa2849d59`;
- stdout/stderr:
  `logs/eval_memnav/ddpm64-safe-14250526.{out,err}`.

| Fixed-64 full-DDPM metric | Baseline | Range + live | Action-safe range |
|---|---:|---:|---:|
| all action MSE | 0.084969 | 0.120480 | 0.112534 |
| revisit action MSE | 0.094307 | 0.135802 | 0.124235 |
| novel action MSE | 0.075631 | 0.105158 | 0.100834 |
| x action MSE | 0.079921 | 0.121904 | 0.112683 |
| y action MSE | 0.062499 | 0.094274 | 0.085809 |
| theta action MSE | 0.112487 | 0.145262 | 0.139112 |
| goal-sensitivity MSE | 0.003396 | 0.001851 | 0.001980 |
| shuffled-goal penalty | 0.006239 | 0.001775 | 0.001734 |
| revisit range-code MAE | 0.211636 | 0.196399 | 0.199157 |

Action-safe range improved the rejected arm by `6.60%` overall, but remained
`32.44%` worse than baseline (`31.73%` revisit and `33.32%` novel). Only 3 of
64 paired samples improved over baseline. Mean paired delta was `+0.027565`,
with a 100,000-resample bootstrap 95% interval of
`[+0.022543, +0.032802]`. In contrast, 46 of 64 samples improved over the old
arm; that paired delta was `-0.007946`, interval
`[-0.012797, -0.003343]`.

Goal A/B/C full-DDPM MSE remained `33.9% / 38.8% / 26.3%` above baseline.
Revisit remaining-path buckets `000-127 / 128-255 / 256+` were also all worse
by `47.3% / 16.2% / 18.3%`. Therefore the residual regression is not confined
to long-range or three-leg examples. Range MAE improved only `5.90%`, while
goal sensitivity and shuffled-goal penalty remained `41.7%` and `72.2%` below
baseline.

## Acceptance criteria

The run is not accepted from training loss alone. Required checks are:

- preflight and smoke complete with exit code `0:0`;
- smoke emits finite nonzero `range_grad_*` diagnostics and complete checkpoint
  metadata with cap ratio `0.25`;
- train, validation, and fixed-64 fingerprints match the baseline contract;
- compare steps 100/200/300/400 with baseline on fixed action, retrieval, gate,
  direction, range, and goal A/B/C metrics;
- final fixed-64 paired full-DDPM action MSE must not regress, especially on
  revisit rows, and goal sensitivity must not collapse;
- range-code improvement alone is insufficient;
- any production-default or closed-loop navigation claim requires a separate
  Habitat evaluation.

## Decision

The `0.25` controlled arm is rejected. The implementation correctly bounds and
projects the auxiliary gradient and materially reduces the old treatment's
damage, but a locally non-conflicting update still changes the shared pose code
on every revisit batch and does not protect validation or future-batch action
gradients. Direct range supervision must remain default-off. A future range
experiment should give the auxiliary a detached calibration branch and let an
action-trained, zero-initialized gate decide whether any calibrated range enters
the policy; it should not directly shape the shared action coordinate.
