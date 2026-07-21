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

- all MemNav pytest tests: `108 passed`;
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

## Submission record

- commit / deployment: pending;
- initialization checkpoint SHA256: pending remote verification;
- zero-step preflight JobID / final state: pending;
- dependent long JobID: pending;
- W&B run: pending;
- train/validation fingerprints: pending preflight;
- stdout/stderr: pending.
