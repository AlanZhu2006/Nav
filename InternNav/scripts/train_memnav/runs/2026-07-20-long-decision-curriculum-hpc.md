# Long-decision curriculum: paired HPC continuation

Date: 2026-07-20 (Asia/Shanghai)

This is a controlled, paired continuation experiment.  It is not a closed-loop
navigation result.  The two training arms start from the same model checkpoint and
change only the training sampler.

## Question and isolation

The local fixed-28 screen found that route-disagreement sampling beats ordinary
continuation at the same 20-step budget, but that screen is too short to establish a
production improvement.  The HPC comparison therefore runs:

- control: historical `random_leg` sampling;
- treatment: `decision_curriculum` with probability/lookahead/minimum remaining
  span/minimum angle `0.5 / 16 / 128 / 45 degrees`.

Both arms use identical initialization, model, seed, optimizer construction, learning
rate, batch size, step budget, losses, cache generation and fixed evaluation.  Range
loss, range gradient surgery and pose-reliability conditioning stay off.  Candidate
sampling/critic logic from the separate oracle diagnostic is not part of either arm.

## Immutable code and deployment

- GitHub fork: `AlanZhu2006/Nav`;
- branch: `feat/memnav-long-decision-curriculum-20260720`;
- deployed commit: `94eace00e53143a39cd3cd37b51676626d5c6b19`;
- deployment: `/scratch/yz11502/Research/Nav-memnav-94eace0-deploy`;
- deployment is detached at the exact commit and has no tracked modifications;
- deployed/local `train_memnav_mp3d.sbatch` SHA256:
  `43fdfe2f91f84075af1c6938ed6bf9adb331a34c52d5db1cdd1f02618fc21776`.

Local validation before deployment:

- all MemNav pytest: `67 passed`;
- documented unittest discovery: `65 passed`;
- `python -m pip check`: no broken requirements;
- Python compilation, `bash -n` and `git diff --check`: passed;
- real local CUDA zero-step initialization loaded the step-400 checkpoint with
  `lr=1e-5`, seed `0`, `resume=none`, and completed with zero optimizer steps.

## Runtime, data and initialization contract

- source overlay:
  `/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf`;
- in-container source root: `/mp3d_revisit_v0/vln_n1/traj_data`;
- sparse feature root:
  `/scratch/yz11502/Research/datasets/mp3d_revisit_v0_feat_kf320_1371557/vln_n1/traj_data`;
- required cache signature:
  `b25fb60ed3abd8bca31da996274eba22e1eec669d968a5b5a9d96fe6487fc809`;
- aggregator cache count observed before submission: `1944`;
- the formal strict preflights scanned every source-ready train/validation episode and
  accepted all required aggregator/camera cache metadata, geometry and corrected-pose
  markers (the standalone login-node `find` count is intentionally not inferred);
- cache geometry window / scale / max-frame: `32 / 8 / 4096`;
- corrected generated-pose marker: required;
- LingBot checkout:
  `/scratch/yz11502/Research/lingbot-map-7ff6f3e-clean`;
- LingBot weights SHA256:
  `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`;
- DINO weights SHA256:
  `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`;
- Conda runtime: `/scratch/yz11502/conda_envs/memnav-runtime-8414d2d`;
- initialization checkpoint:
  `/scratch/yz11502/Research/Nav-memnav-1371557/InternNav/checkpoints/`
  `mkf-train-1371557-400/ckpts/checkpoint-400/memnav.ckpt`;
- initialization checkpoint SHA256:
  `a6d66a66d3e316da6835c1d8e835e7f007357a782ffb7f22277da5a41b4333d1`;
- source checkpoint train / fixed-eval fingerprints:
  `aa4c6d1a1799ac5338f1fa7734404406b88e7afb732d396470b38a43082033a9` /
  `c5bf140feb86a8863bbb65e46eb55cee39fa36b25a16504497679495328169df`.

The checkpoint is loaded as model initialization, not resumed.  Optimizer, scheduler,
global step and RNG are rebuilt identically in each arm.  The launcher rejects a
simultaneous initialization checkpoint and non-empty resume request.

## Shared training parameters

- batch size / workers: `4 / 4`;
- maximum optimizer steps: `200`;
- learning rate / seed: `1e-5 / 0`;
- maximum wall time per arm: `04:00:00`;
- total bounded treatment+control allocation: `8 GPU-hours`;
- save / log / fixed-eval interval: `50 / 10 / 100` steps;
- fixed evaluation samples / batch: `64 / 4`;
- auxiliary direction weight: `0.2`;
- range weight / gradient cap: `0.0 / 0.0`;
- anchor teacher forcing: constant `1.0`;
- pose-reliability conditioning / loss: `0 / 0.0`;
- W&B project: `memnav`.

The previous 400-step run took `06:02:12`; two 200-step jobs with fewer scheduled
evaluations are expected to fit their separate four-hour limits.  Step count, rather
than wall time or GPU type, defines the paired budget.

## Submitted dependency chains

### Uniform control

- zero-step preflight JobID: `14376464` (`ld-u-pre-94eace0`);
- preflight limit / reporting: `00:45:00 / none`;
- 200-step JobID: `14376466` (`ld-u-s200-94eace0`);
- dependency: `afterok:14376464`;
- W&B run ID: `ld_uniform_s200_94eace0`;
- W&B URL:
  `https://wandb.ai/yz11502-new-york-university/memnav/runs/ld_uniform_s200_94eace0`;
- preflight/training logs: `/home/yz11502/logs/train_memnav/mp3d-14376464.{out,err}` /
  `/home/yz11502/logs/train_memnav/mp3d-14376466.{out,err}`;
- observed state: preflight `COMPLETED` in `00:05:30`, `ExitCode=0:0` on
  `ga012`; training entered `RUNNING` on `ga008` only after that success.

### Decision-curriculum treatment

- zero-step preflight JobID: `14376465` (`ld-d-pre-94eace0`);
- preflight limit / reporting: `00:45:00 / none`;
- 200-step JobID: `14376467` (`ld-d-s200-94eace0`);
- dependency: `afterok:14376465`;
- W&B run ID: `ld_decision_s200_94eace0`;
- W&B URL:
  `https://wandb.ai/yz11502-new-york-university/memnav/runs/ld_decision_s200_94eace0`;
- preflight/training logs: `/home/yz11502/logs/train_memnav/mp3d-14376465.{out,err}` /
  `/home/yz11502/logs/train_memnav/mp3d-14376467.{out,err}`;
- observed state: preflight `COMPLETED` in `00:05:57`, `ExitCode=0:0` on
  `ga016`; training entered `RUNNING` on `ga009` only after that success.

No training job can start unless its own sampling-mode preflight exits successfully.
The preflights must verify the exact code commit, clean code and LingBot worktrees,
package consistency, CUDA, all cache metadata/header contracts, source/feature
coverage, checkpoint and weight hashes, model construction, both dataset populations,
and fixed evaluation fingerprints.

Both completed preflights found `3962` goal samples across `1704` train episodes and
`558` goal samples across `240` validation episodes.  The control/treatment training
fingerprints are respectively
`aa4c6d1a1799ac5338f1fa7734404406b88e7afb732d396470b38a43082033a9` and
`9f7028f8de4a4a3c82ea5fd73d9ed65a5e33b1eda9e852194a7d047d98c1df5e`;
the distinct values are expected because sampling mode is part of the dataset
contract.  Both fixed-validation populations have fingerprint
`414673cab0f1776e8c2f03c1c4dda60508b3a3735d0ad346e67ea6ea639a02ce`, and both
fixed-64 subsets have fingerprint
`c5bf140feb86a8863bbb65e46eb55cee39fa36b25a16504497679495328169df`.

## Launch-health snapshot

Both jobs reached optimizer step 10 with finite gradients and losses, without a
traceback, OOM or NaN:

| Arm | Step-10 total loss | Action | Retrieval | Gate | Aux direction | Hard fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform control | 0.6675 | 0.1440 | 0.1381 | 0.4512 | 0.1645 | 0.15 |
| decision treatment | 0.6521 | 0.1688 | 0.1625 | 0.4282 | 0.0547 | 0.35 |

The hard fraction is computed from all 40 rows in the first ten four-sample batches.
It confirms that the treatment changes the intended sample composition: about 14/40
treatment rows versus 6/40 naturally occurring control rows meet the hard criterion.
The two action losses are therefore not a paired validation comparison, and this
early training window is recorded only as a launch/sampler check.  At step 10 the
treatment hard/easy action losses were `0.1927 / 0.1559`; the control values were
`0.2529 / 0.1248` on different sampled rows.

## Scheduled paired full-DDPM acceptance evaluation

Two immutable evaluator jobs were submitted after launch validation.  Both depend on
*both* training jobs with
`afterok:14376466:14376467`; if either training arm fails, neither half of the pair is
allowed to run.

- uniform JobID: `14376762` (`dd64-u-94eace0`);
- decision JobID: `14376763` (`dd64-d-94eace0`);
- evaluator script SHA256:
  `a8070b8c45e453194468dc8c3107ee53ef3beb80dd2f53688288157804c681e1`;
- partition / wall limit: `a100_tandon / 02:00:00` per arm;
- checkpoints: each arm's `checkpoint-200/memnav.ckpt`;
- selection: validation `fixed_leg`, balanced fixed `64`, batch size `4`;
- seeds: split/sampling/model `0 / 0 / 0`, diffusion `104729`;
- diagnostics: oracle-positive enabled, paired full-DDPM correct/shuffled-goal
  sampling enabled, per-sample rows retained;
- uniform output:
  `/scratch/yz11502/Research/eval_outputs/ddpm64-ld-uniform-94eace0.json`;
- decision output:
  `/scratch/yz11502/Research/eval_outputs/ddpm64-ld-decision-94eace0.json`;
- logs:
  `/home/yz11502/logs/eval_memnav/ddpm64-{uniform,decision}-%j.{out,err}`.

These jobs are acceptance diagnostics, not extra optimizer steps.  Their separate
limits do not change the controlled 200-step training budget.

After both reports complete, run the fail-closed comparator from this branch with
`100000` bootstrap resamples.  It must accept all experiment- and row-level contract
fields before any delta is reported:

```bash
python scripts/eval/compare_memnav_offline.py \
  --control /path/to/ddpm64-ld-uniform-94eace0.json \
  --treatment /path/to/ddpm64-ld-decision-94eace0.json \
  --output /path/to/ddpm64-ld-paired-comparison.json \
  --bootstrap-resamples 100000 --bootstrap-seed 0
```

Before use, the comparator passed four focused tests and the complete 69-test MemNav
unittest suite.  A 100000-resample replay of the earlier fixed-28 pair reproduced the
documented remaining-span >=256 delta exactly (`-14.07%`, all `4/4` rows improved).

## Acceptance and rejection rule

Training loss alone is not sufficient.  After both jobs finish, compare fixed
full-DDPM action metrics at the same final step and paired diffusion randomness.  The
treatment is accepted only if it improves 3-leg Goal C and remaining-span >=256 rows
without a material regression in 2-leg/easy rows, overall action MSE, retrieval, gate
recall or goal sensitivity.  A candidate-oracle upper bound cannot be counted as an
achieved policy improvement.

Current checklist:

- [x] local dependencies, regression tests and real checkpoint initialization passed;
- [x] immutable deployment commit and launcher hashes matched;
- [x] both long jobs have explicit `afterok` preflight dependencies;
- [x] uniform preflight completed with `ExitCode=0:0`;
- [x] decision preflight completed with `ExitCode=0:0`;
- [x] both training jobs started only after their preflights;
- [x] W&B URLs and first finite step-10 logs recorded;
- [x] paired full-DDPM jobs submitted with a joint `afterok` dependency;
- [ ] both jobs wrote complete final step-200 checkpoints;
- [ ] paired full-DDPM final evaluation completed.
