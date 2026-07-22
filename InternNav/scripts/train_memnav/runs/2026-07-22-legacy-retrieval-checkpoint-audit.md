# Legacy retrieval checkpoint audit

Date: 2026-07-22 (Asia/Shanghai)

This is an offline retrieval diagnosis. It does not modify the old checkpoint,
training data, cached features, or the Nav policy, and it is not a closed-loop
navigation result.

## Question

The W&B run `mn38_aux000_s0_e1_20260715` (`x58uloib`) trained for one epoch and
logged only random training minibatches. Its 76 retrieval-loss points had mean
`1.203246`, median `1.086294`, minimum `0.106033`, and maximum `2.909040`.
There was no fixed evaluation dataset (`eval_strategy=no`, `do_eval=false`). This
audit asks two separate questions:

1. Is the large train-curve range evidence of optimization instability, or mostly
   the expected variance of batch size four plus a randomly sampled current step
   `k`?
2. Does the learned projected retrieval head actually rank historical frames better
   than the frozen raw-DINO representation?

## Audited checkpoint

- source run: `mn38_aux000_s0_e1_20260715`;
- source code commit: `ed15f6148dc645a28e200f64e7f93ff02e1c7fa5`;
- checkpoint:
  `/scratch/yz11502/Research/Nav/InternNav/checkpoints/mn38_aux000_s0_e1_20260715/ckpts/checkpoint-759/memnav.ckpt`;
- checkpoint SHA256:
  `cf781e82da27f3e64afc8445def7bfdb332401ee7612b888536c122c2df2500c`;
- extracted tensors: `proj_goal`, `proj_mem`, `null_key`, and `log_temp` under
  `core.retrieval`;
- learned temperature: `0.070244506`, effectively unchanged from its `0.07`
  initialization.

The full old policy is intentionally not loaded into the current architecture.
Only the six retrieval tensors are extracted into an exact copy of the old head.
Goal CLS and memory CLS remain the same frozen LingBot DINO representation for every
method being compared.

## Evaluation contracts

All panels use five deterministic current-step seeds (`0..4`). The current scene
slice is `val`, split seed zero. The old run originally trained with
`scene_split=all`, so the legacy panel is a deterministic audit of its training
domain, not a claim of scene-held-out generalization.

### Legacy

This reconstructs the `ed15f61` candidate and label semantics from generator meta:

- candidate region `[anchor_margin, k]`, with anchor margin `39` in all 511 rows;
- no modern 83-frame recent-history exclusion;
- semantic revisit/novel kind is authoritative for covisibility goals;
- weak revisit goals are skipped, matching the old loader (`53` skipped on this
  scene slice);
- goal A uses the old 14/83-frame heuristic;
- the null key participates in the same joint multi-positive softmax as training.

### Current8

This uses the current unified candidate region `[8, k-83]`, dynamic revisit labels,
and all 558 current validation goals. Its dataset fingerprint is
`9dd43c170bc9b92d3c786bcfce5519021d2eab8928c9a52bad886ecf3e695442`; its full
evaluation fingerprint is
`bf72324a0377bc8e11e4e8873ea55b3391674ef5bf93176e8b39364b5a327707`.

### Current39

This changes only the current candidate floor from frame 8 back to frame 39. It
separates the early-anchor change from the projected-head failure.

For frame ranking, both the legacy projected head and raw DINO use the same current
candidate mask and all-candidate multi-positive listwise denominator. Raw DINO uses
the fixed current temperature `0.01`. Ranking metrics are computed only where a row
has at least one positive and at least one non-positive candidate.

## Complete five-seed result

Lower listwise loss is better. Values below are means across the five fixed `k`
populations.

| Panel | Method | Listwise loss | Strict Top-1 | Recall@5 | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Legacy | old projected head | 1.895728 | 30.48% | 50.78% | 62.08% |
| Legacy | raw DINO | **0.982591** | **69.01%** | **84.26%** | **89.31%** |
| Current8 | old projected head | 1.493180 | 39.09% | 58.88% | 66.82% |
| Current8 | raw DINO | **0.722040** | **72.74%** | **86.32%** | **89.46%** |
| Current39 | old projected head | 1.594314 | 40.02% | 59.10% | 68.36% |
| Current39 | raw DINO | **0.780266** | **71.12%** | **87.21%** | **91.62%** |

The old projection is worse in every panel. On Current8 seed zero, raw DINO finds
`157/217` positives while the old head finds only `84/217`. The projection destroys
84 raw-DINO successes and rescues only 11 raw failures; the two methods choose the
same frame in only `4/217` rankable rows. This is a representation/ranking failure,
not merely a different temperature.

Changing frame 8 back to frame 39 does not recover the old head. It also changes
the semantic population (221 revisit rows at frame 8 versus 186 at frame 39), so
the two anchor panels should not be interpreted as a paired accuracy ablation. The
important controlled conclusion is that raw DINO remains much stronger under both
floors.

## Null-slot collapse

Under the exact legacy joint frame-plus-null objective, the five-seed means are:

- fixed loss: `1.066222`;
- reported retrieval accuracy: `61.14%`;
- revisit positive accuracy over the joint argmax: `1.98%`;
- novel null accuracy: `99.81%`;
- mean gate: revisit `0.822`, novel `0.528`.

For seed zero, 314 of 511 rows are novel. Always predicting null therefore scores
`314/511 = 61.45%`; the checkpoint scores `318/511 = 62.23%`, only four rows better.
The same pathology appears under Current8: always-null is `337/558 = 60.39%`, while
the checkpoint is `341/558 = 61.11%`, again only four rows better.

This does not contradict the nontrivial gate separation. The gate is one minus the
*total* null probability, so dozens of individually weak real-frame logits can have
large aggregate mass even while null is the largest single logit. The old revisit
rows contain about 57 positives on average. Consequently, the summed multi-positive
numerator and gate can look acceptable while the actual frame argmax is poor.

## Why the old train curve fluctuated

At the fixed final checkpoint, bootstrapping the exact legacy loss over batches of
four gives:

- mean `1.0646` and standard deviation `0.5000`;
- 5th/50th/95th percentiles `0.3533 / 1.0051 / 1.9745`;
- sampled extrema `0.0338 .. 3.3330` over 10,000 draws.

The observed W&B extrema `0.1060 .. 2.9090` lie inside this fixed-checkpoint batch
distribution. In contrast, the complete-population loss across five deterministic
`k` seeds is `1.0662 +/- 0.0187`; the within-identity projected-loss standard
deviation from changing `k` is about `0.152`.

Therefore the visible train curve is dominated by batch-size-four composition and
random temporal sampling. It is not evidence of an exploding optimizer. The deeper
problem is that the stable final projected head is worse than the raw representation.

## Newly found W&B evaluation omission

The CPU contract check compared all 558 production `MemNav_Dataset.__getitem__`
outputs against the lightweight evaluator:

- fixed `k`: exact on every row;
- positive, negative, and candidate masks: exact on every row;
- revisit rows: `221`;
- rankable rows: `217`;
- mismatches: `0`;
- contract report SHA256:
  `6ba25e5dedc761126c00621521246ad2faab256d395bcfcbdbd03c9661ed68cf`.

The existing W&B raw control reported only 212 rankable rows because
`InternNav/scripts/train/train.py` sets `dataloader_drop_last=True` in
`TrainingArguments`. The formal retrieval run used evaluation batch size 16, so the
558-row set was truncated to the first 544 rows. The omitted final 14 rows contain
five rankable 3-leg Goal-C revisits.

This is demonstrated exactly by applying the same truncation to the new raw-DINO
records:

| Population | Loss | Top-1 | Negative | Gray | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complete 558 / 217 rankable | 0.728060 | 157/217 | 15/217 | 45/217 | 186/217 | 194/217 |
| first 544 / 212 rankable | 0.735311 | 153/212 | 15/212 | 44/212 | 181/212 | 189/212 |
| existing W&B zero control | 0.735311 | 153/212 | 15/212 | 44/212 | 181/212 | 189/212 |

The truncated row reproduces every W&B value. Future fixed evaluation must not use
drop-last. MemNav already overrides its training DataLoader with `drop_last=True` in
`MemNavTrainer.get_train_dataloader`, so disabling the TrainingArguments flag for
MemNav evaluation can preserve training behavior while evaluating every held-out
row. This audit records the fix requirement but does not change production training
code.

## Jobs and artifacts

- `14532567`: failed safely at the original global `pip check`; no checkpoint was
  loaded and no report was written. It exposed five pre-existing conflicts in the
  shared `/scratch/lg154/conda-envs/memnav` environment.
- `14532688`: 32-row, two-seed smoke, `COMPLETED`, `00:03:10`;
  report SHA256
  `187b037e780087d6f75d95687f74d9614d58841d5292b6919e95e6814afe3960`.
- `14532978`: full three-panel, five-seed audit, `COMPLETED`, `00:03:01`;
  report
  `/scratch/yz11502/Research/diagnostics/legacy_retrieval/legacy-ret-full5-v1-14532978.json`,
  SHA256
  `a3f4dac69892ce1b5eedc45390eb2f09de5526e05a6dc29249841510bb92be6c`.
- `14533293`: GPU contract job cancelled while pending; it was replaced by the
  CPU-only check.
- `14533335`: all-558 production dataset contract check, `COMPLETED`, `00:02:12`;
  report
  `/scratch/yz11502/Research/diagnostics/legacy_retrieval/current8-contract-14533335.json`.

The reusable evaluator is `scripts/eval/eval_legacy_memnav_retrieval.py`. Future
submission wrappers default to the personal clean runtime environment
`/scratch/yz11502/conda_envs/memnav-runtime-8414d2d`; Job 14533335 recorded
`No broken requirements found` and passed the actual NumPy, PyArrow, and Torch import
checks. No source or report was written into the parent Nav checkout.

## Conclusion

The old W&B curve is noisy because it is a sequence of random batches, not because
the final retrieval loss is intrinsically unstable. However, the old checkpoint is
not a useful retrieval initialization: its joint null objective nearly collapses to
the majority null class, and its learned 1024-to-256 projections severely damage the
already strong raw-DINO ranking. The current raw-preserving design is therefore the
right direction. The next evaluation hygiene change is to remove drop-last from
fixed validation and re-report current raw/reranker classification on all 558 rows,
especially the five previously omitted 3-leg Goal-C revisits.
