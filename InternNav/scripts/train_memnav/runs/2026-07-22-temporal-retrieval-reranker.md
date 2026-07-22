# Bounded temporal retrieval reranker

Date: 2026-07-22 (Asia/Shanghai)

This is a retrieval-hygiene experiment. It does not claim to solve the previously
measured long-route action-planning bottleneck, and none of the numbers below are a
closed-loop navigation result.

## Why a temporal reranker

On 220 repeated revisit rows from the local fixed diagnostic, frozen raw-DINO
retrieval already achieved strict Top-1 `207/220 = 94.09%`, Recall@5 `96.36%`, and
Recall@10 `100%`. All 13 strict errors came from one sample identity, and every true
positive was still in raw Top-10. The failed Top-1 choices were isolated score peaks,
whereas a real co-visible pass normally produces a short, temporally supported score
plateau.

This motivates reranking only the existing raw Top-10 using the local shape of the
already observed score curve. It does not replace DINO, invent positives outside the
shortlist, use future frames, or use absolute frame IDs.

## Implemented treatment

- Rank mode `raw_temporal` starts from frozen raw-DINO cosine.
- Thirteen label-free features describe score deltas at offsets `-2,-1,+1,+2`,
  neighbor validity, local mean/RMS/min/max, and gap from the row maximum.
- A zero-initialized linear form plus bias produces a `tanh`-bounded residual.
- The residual is applied only to raw Top-10 and is bounded by `±0.02` cosine.
- Zero initialization exactly preserves the raw logits and Top-1.
- Checkpoints persist the mode, Top-K, bound, weights, and bias. Legacy mode 0/1
  checkpoints migrate to a complete zero temporal namespace; a partially written
  mode-2 checkpoint fails strict loading.

The formal training path sets `MEMNAV_RETRIEVAL_ONLY=1`. It skips window images,
GCT/camera pose, novel-branch, and diffusion forward work, while using the exact same
goal DINO, cached memory CLS, candidate masks, and retrieval head as the full policy.
It freezes everything except 13 weights and one bias. Raw temperature remains fixed
at `0.01`, so loss cannot improve merely by calibration.

The objective is all-candidate multi-positive listwise likelihood plus a strict
Top-1 hinge. The hinge requires the best positive cosine to exceed the hardest gray
or negative candidate by `0.005`. W&B reports listwise and margin losses separately,
strict Top-1, negative/gray fractions, Recall@5/10, cosine margin, fixed temperature,
and bounded-residual magnitude.

## Local evidence

### Leave-one-identity-out diagnostic

The only failing identity was entirely excluded from optimization and checkpoint
selection. Training used 160 rows from other identities; selection used 40 rows from
two other identities; the held-out hard identity contributed 20 evaluation rows.

| Metric | Raw DINO | Reranked | Change |
| --- | ---: | ---: | ---: |
| Strict positives, all | 207/220 | 212/220 | +5 |
| Negatives, all | 13/220 | 8/220 | -5 |
| Gray, all | 0/220 | 0/220 | unchanged |
| Held-out hard identity | 7/20 | 12/20 | +5 |
| Other ten identities | 200/200 | 200/200 | unchanged |

The same result held with and without the explicit `0.005` margin term. This is a
small two-scene diagnostic, so it supports launching a scene-held-out experiment but
does not establish generalization.

### Real two-step and resume smoke

Starting from `mkf-1371557-step400.ckpt`, the production training entry completed two
real optimizer steps and fixed evaluation using actual goal images/cache CLS:

- Trainer-reported trainable parameters: exactly 14;
- peak allocated/reserved GPU memory: approximately `4.76/4.83 GiB`;
- raw temperature: exactly `0.01` throughout;
- small fixed eval listwise loss: `0.336594 -> 0.333176`;
- checkpoints 1 and 2 contained model, optimizer, scheduler, trainer state, RNG, and
  metadata;
- checkpoint 2 persisted rank mode code `2`, bound `0.02`, and nonzero temporal
  weights;
- resuming checkpoint 2 restored global step 2 and completed/saved step 3.

The tiny fixed eval contained only one supported revisit row, so its delta validates
the execution path rather than model quality.

## Local verification

- all MemNav pytest tests: `114 passed`;
- focused retrieval/checkpoint/input tests: `38 passed`;
- real checkpoint initialization, backward, save, and resume: passed;
- Python compilation: passed;
- `bash -n train_memnav_mp3d.sbatch`: passed;
- `git diff --check`: passed;
- local `python -m pip check`: no broken requirements.

## Formal scene-held-out job contract

The cluster experiment must initialize from the accepted uniform-continuation
checkpoint, not resume its optimizer. It uses the audited scene split and complete
fixed validation population.

- rank mode / denominator: `raw_temporal / all_candidates`;
- retrieval-only: `1`;
- raw temperature: `0.01`, frozen;
- Top-K / residual bound: `10 / 0.02`;
- cosine margin / weight: `0.005 / 1.0`;
- candidate floor: frame `8`;
- sampling: training `random_leg`, validation `fixed_leg`;
- only 14 reranker scalars train;
- maximum Slurm wall time: 8 hours;
- long job must depend on an identical zero-step preflight via `afterok`.

Acceptance requires a held-out-scene strict Top-1 increase and fewer negatives,
without increased gray fraction or reduced Recall@10. The selected checkpoint must
then pass the existing fixed full-DDPM evaluation with no more than 2% overall action
regression and no 3-leg Goal-C regression. A lower retrieval loss alone is not an
acceptance result.

## Formal scene-held-out training result

The dependent job completed all 3000 optimizer steps in `00:08:48`. The complete
fixed validation set had 558 goals, of which 212 revisit rows had a valid ranking
target after candidate masking. Recall@5 stayed `181/212` and Recall@10 stayed
`189/212` at every recorded evaluation, confirming that training changed only the
ordering inside the raw shortlist.

| Step | Retrieval loss | Strict Top-1 | Negative | Gray | Mean abs residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 0.728327 | 152/212 | 15/212 | 45/212 | 0.001890 |
| 750 | 0.722242 | 155/212 | 15/212 | 42/212 | 0.003806 |
| 1750 | **0.717315** | 155/212 | 15/212 | 42/212 | 0.005946 |
| 2250 | 0.719674 | **156/212** | **14/212** | **42/212** | 0.006147 |
| 2500 | 0.719595 | **156/212** | **14/212** | **42/212** | 0.006184 |
| 3000 | 0.719215 | 155/212 | 14/212 | 43/212 | 0.006268 |

Step 2500 is the provisional classification checkpoint: it ties the best strict
Top-1/negative/gray counts and has slightly lower listwise loss than step 2250. Its
checkpoint SHA256 is
`c5382ccb732af2c8053741be8117258b80f2c190ed1e9427b7d4bfeb3001e3df`.

The exact zero-residual control subsequently completed on the same 558-row population
with base learning rate zero. Of the 221 revisit rows, 212 had a valid ranking target
after candidate masking:

| Metric | Zero residual | Step 2500 | Change |
| --- | ---: | ---: | ---: |
| Retrieval loss | 0.735311 | 0.719595 | -2.14% |
| Strict Top-1 | 153/212 | 156/212 | +3 |
| Negative | 15/212 | 14/212 | -1 |
| Gray | 44/212 | 42/212 | -2 |
| Recall@5 | 181/212 | 181/212 | unchanged |
| Recall@10 | 189/212 | 189/212 | unchanged |

This passes the classification-only gate: the bounded temporal residual improves
held-out-scene Top-1 and reduces both negative and gray selections without changing
the raw shortlist coverage. It is still not an accepted policy improvement until a
paired full-DDPM action evaluation shows no overall/2-leg regression and no 3-leg
Goal-C regression.

## Local paired full-DDPM action screen

While the formal fixed-64 jobs waited for group GPU quota, the two formal checkpoints
were screened on the existing fixed 28-row local population. The raw checkpoint was
downloaded with its SHA256 verified. A remote tensor-by-tensor comparison showed that
step 2500 differs from the raw checkpoint in exactly two tensors and 14 scalar values:
13 `core.retrieval.temporal_weights` and one `core.retrieval.temporal_bias`. All other
tensors are bitwise identical. The local treatment was therefore reconstructed by
changing only those remotely audited values.

Both arms used the same 28 explicit indices, batch size 4, diffusion seed `104729`,
correct/shuffled-goal initial noise, sparse-cache signature
`97a7819c4722ebf6e2165538b4908a276d426faf3190f18987332f59889c9afc`, and
per-sample output. The fail-closed comparator accepted every experiment and row
contract before producing deltas.

| Group | Raw control | Step 2500 | Relative change |
| --- | ---: | ---: | ---: |
| All 28 | 0.093622 | 0.093745 | +0.13% |
| Revisit, 11 | 0.125041 | 0.125038 | -0.00% |
| Novel, 17 | 0.073291 | 0.073497 | +0.28% |
| 2-leg, 11 | 0.099027 | 0.099346 | +0.32% |
| 3-leg, 17 | 0.090124 | 0.090121 | -0.00% |
| 3-leg Goal-C revisit, 5 | 0.102280 | 0.102280 | 0.00% |
| Remaining span >=256, 4 | 0.142828 | 0.142828 | 0.00% |
| Hard turn, 2 | 0.220943 | 0.220943 | 0.00% |

The all-row paired bootstrap 95% interval for treatment minus control is
`[-0.0000074, +0.0003799]`, which crosses zero. Four rows changed match index. Only
one was a revisit row, and it moved from one positive to an adjacent positive while
its action MSE changed by only `-0.0000311`. The other three were novel rows whose
selected history frames remained negatives; the largest action delta was a `+0.003546`
regression on one 2-leg Goal-A row. Thus the local screen passes the predefined 2%
safety bound and has no long-group regression, but shows no meaningful action gain.
The formal fixed-64 evaluation remains the acceptance result.

Local artifacts (gitignored diagnostics):

- raw report: `.diagnostics/retrieval/temporal-reranker-raw-full28.json`, SHA256
  `f4f7df12f9f4d95bf7559898891a667b8ce46f68c52a2afde4970bf12cd55fea`;
- step-2500 report:
  `.diagnostics/retrieval/temporal-reranker-step2500-full28.json`, SHA256
  `4bc2cee95185059fa14607c79c171eaa00c2f7f8220c52987cef06277e83dd09`;
- 100,000-resample comparison:
  `.diagnostics/retrieval/temporal-reranker-raw-vs-step2500-full28-100k.json`,
  SHA256
  `60ab8f54fb3a5660539a35b74f66aa837a17d6fa033c96c98d8c2ef1bb46a057`.

## Formal paired full-DDPM action gate

The formal comparison used 64 balanced fixed rows, including 32 revisit and 32 novel
rows. Both arms ran concurrently on separate GPUs of the same H200 node `gh126` with
the same code, sparse cache, 64 explicit indices, batch size 4, diffusion seed
`104729`, and paired correct/shuffled-goal initial noise. The only model difference
was the remotely audited 14-scalar temporal reranker state. The fail-closed comparator
accepted all experiment-level and row-level contracts before computing 100,000 paired
bootstrap resamples.

| Group | Rows | Raw control | Step 2500 | Relative change | Paired 95% CI for delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 64 | 0.073553 | 0.073424 | -0.18% | `[-0.000504, +0.000145]` |
| Revisit | 32 | 0.076599 | 0.076613 | +0.02% | `[0, +0.0000337]` |
| Novel | 32 | 0.070507 | 0.070235 | -0.39% | `[-0.001011, +0.000277]` |
| 2-leg | 15 | 0.060391 | 0.059757 | -1.05% | `[-0.001922, +0.0000175]` |
| 3-leg | 49 | 0.077583 | 0.077608 | +0.03% | `[-0.000201, +0.000260]` |
| 3-leg Goal-C revisit | 25 | 0.076170 | 0.076185 | +0.0189% | `[0, +0.0000400]` |
| Remaining span >=256 | 6 | 0.095117 | 0.095117 | 0.00% | `[0, 0]` |
| Hard turn | 4 | 0.201410 | 0.201410 | 0.00% | `[0, 0]` |

The apparent all-row improvement is not a reliable policy gain: its confidence
interval crosses zero, and only 4/64 rows improved while 8 worsened and 52 were
bitwise tied. The revisit retrieval counts were exactly unchanged at 23 positive,
8 gray/ignored, and 1 negative. Twelve rows changed match index:

- all three changed revisit rows remained positive-to-positive; none improved its
  full-diffusion action MSE, and their regressions were only `0.0000617`, `0.0000796`,
  and `0.0002803`;
- all nine changed novel rows remained non-positive-to-non-positive, because a novel
  Goal-A row has no retrieval positive to learn;
- one novel 2-leg Goal-A row contributed a `-0.009612` delta by itself and dominated
  the small aggregate improvement, while another novel Goal-A row regressed by
  `+0.003994`.

The result passes the overall 2% regression bound but fails the separately
pre-registered zero-regression requirement for 3-leg Goal-C revisit rows. Therefore
step 2500 is **not accepted as a policy improvement** and the temporal mode remains an
optional retrieval diagnostic rather than a new default. The scene-held-out
classification result is still real: it improves strict retrieval Top-1 inside the
raw Top-10, but this fixed action gate shows that the improvement does not transfer to
long-route or revisit action quality. Future work should not spend another long run
only reducing this listwise loss; it needs a goal-conditioned action-relevance signal
and must avoid applying a revisit-trained temporal residual blindly to novel rows.

Formal artifacts (gitignored diagnostics):

- raw report: `.diagnostics/retrieval/temporal-reranker-raw-full64.json`, SHA256
  `61ceec9d87f06c57c0943e4a38dbab3e2fd6888414f04e10b3a1be721120fda4`;
- step-2500 report:
  `.diagnostics/retrieval/temporal-reranker-step2500-full64.json`, SHA256
  `1ae4504a1a88acd57fdd6f551215fd32b0cd4d2b52c2ab4c0519a56ed8c563dc`;
- 100,000-resample comparison:
  `.diagnostics/retrieval/temporal-reranker-raw-vs-step2500-full64-100k.json`,
  SHA256
  `b67fc722fc2d9137fead6fa28843be1311ea5d3d2b2a123162a5e8e3fc25c832`.

## Submission record

- code commit: `1c7aee077252fa3d23fda6532c5660a21ddca702`;
- immutable deployment:
  `/scratch/yz11502/Research/Nav-memnav-1c7aee0-deploy`;
- LingBot commit: `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`;
- initialization checkpoint:
  `/scratch/yz11502/Research/Nav-memnav-94eace0-deploy/InternNav/checkpoints/ld_uniform_s200_94eace0/ckpts/checkpoint-200/memnav.ckpt`;
- initialization checkpoint SHA256:
  `2dea2dd4531677b42b8f3e8b2205b19de701a9a0e0ffc645eaa29971450d8bf5`;
- DINO SHA256:
  `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`;
- LingBot weights SHA256:
  `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`;
- expected cache signature:
  `b25fb60ed3abd8bca31da996274eba22e1eec669d968a5b5a9d96fe6487fc809`;
- zero-step preflight JobID: `14500381`;
- preflight final state: `COMPLETED, ExitCode=0:0`, elapsed `00:06:39` on
  `h200_tandon`;
- dependent 8-hour JobID: `14500382`, submitted with
  `afterok:14500381`; the dependency was released only after the successful
  preflight; final state `COMPLETED, ExitCode=0:0`, elapsed `00:08:48` on
  `h200_tandon`;
- long-run budget: batch size `8`, maximum `3000` optimizer steps, evaluation every
  `250` steps on all `558` held-out goals, checkpoint every `500` steps with six
  checkpoints retained;
- W&B run:
  `https://wandb.ai/yz11502-new-york-university/memnav/runs/retr-temporal-s3000-1c7aee0`;
- exact zero-residual control: JobID `14502040`, `COMPLETED, ExitCode=0:0` in
  `00:09:29`; one optimizer step at base learning rate zero followed by the complete
  fixed evaluation; checkpoint SHA256
  `d80765be07f696fbcada6911cc6de9f80c8d5ba8ac55f6bfb1f0d91a8dd5a719`;
- paired full-DDPM action-gate smoke jobs: raw `14526133`, temporal `14526134`;
- raw smoke `14526133` completed with `ExitCode=0:0` in `00:04:27`; it
  evaluated the balanced fixed indices `[204, 533]`, wrote
  `ddpm2-retr-raw-1c7aee0.json` (SHA256
  `79aff7a551ad74edd0933909483f058c4b7335b3b050a333f13c46d97bf5c5a4`),
  and passed the runtime dependency, code, model-weight, and cache-signature checks;
- temporal smoke `14526134` completed with `ExitCode=0:0` in `00:03:49` on `ga004`
  after waiting for `QOSGrpGRES`; its report SHA256 is
  `458115e365ae2b8b842c37be96cb5b441a7f236ab03a700eabe55c59afbd5633`;
- the strict two-row smoke comparison was bitwise tied for both match index and all
  primary action groups; comparison SHA256
  `c7ed53391a25175b291226a173248fa4e6ff6253171a7af0b1c2c3e369965ee8`;
- the first fixed-64 raw allocation, JobID `14526135`, passed repository, dependency,
  package, weight, dataset, and cache checks but failed before its first batch on
  `ga040` with `CUDA-capable device(s) is/are busy or unavailable`; final state
  `FAILED, ExitCode=1:0`, elapsed `00:02:40`, and no report was written;
- the original pending temporal JobID `14526136` was cancelled without running so
  that both arms could be repeated on one healthy GPU architecture; the cluster
  rejected an attempted `--exclude=ga040` submission before assigning a JobID;
- retry fixed-64 jobs: raw `14527490`, temporal `14527491`. Because Slurm refused to
  reuse the already-satisfied smoke dependencies, both were submitted only after the
  two successful smoke exits and JSON checks had been recorded. They ran concurrently
  on separate H200 GPUs of node `gh126` and completed with `ExitCode=0:0` in
  `00:21:42` and `00:21:16`, respectively;
- both retry arms used diffusion seed `104729`, balanced fixed selection, paired
  correct/shuffled-goal noise, and immutable evaluator SHA256
  `a8070b8c45e453194468dc8c3107ee53ef3beb80dd2f53688288157804c681e1`;
- retry stdout/stderr:
  `/home/yz11502/logs/eval_memnav/offline-14527490.{out,err}` and
  `/home/yz11502/logs/eval_memnav/offline-14527491.{out,err}`;
- training dataset: `3962` goals / `1704` episodes, fingerprint
  `f4ded662fdd7db7b37c4ebdfb6c94a82e6b8a6bc7fa29f1cd71e4c2c36b483aa`;
- validation dataset: `558` goals / `240` episodes, fingerprint
  `9dd43c170bc9b92d3c786bcfce5519021d2eab8928c9a52bad886ecf3e695442`;
- fixed held-out evaluation subset: all `558/558` goals (`221` revisit, `337`
  novel), fingerprint
  `bf72324a0377bc8e11e4e8873ea55b3391674ef5bf93176e8b39364b5a327707`;
- preflight stdout/stderr:
  `logs/train_memnav/mp3d-14500381.out` and
  `logs/train_memnav/mp3d-14500381.err` under the immutable deployment;
- long-run stdout/stderr:
  `logs/train_memnav/mp3d-14500382.out` and
  `logs/train_memnav/mp3d-14500382.err` under the immutable deployment.
