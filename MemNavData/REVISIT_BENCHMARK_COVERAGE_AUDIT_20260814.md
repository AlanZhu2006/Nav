# Revisit benchmark coverage and frozen next test

Date: 2026-08-14

Scope: train/consumed artifacts only. Development and blind are not read; no
threshold or model fitting is performed here.

## Current evidence

| axis | result | interpretation |
|---|---:|---|
| Supported Revisit, fresh160 | certified `112/120` vs native `27/120` | strong closed-loop utility when actual-online support exists |
| Certified vs raw direct | `+9/−3`, McNemar `p=0.146` | safer direction, not a significant superiority claim |
| Fresh160 online support | max co-visibility `≥0.20`: `120/120`; `≥0.50`: `115/120` | valid supported-Revisit test, not a low-overlap stress test |
| Actual-online delayed NNR | certified `16/19` vs native `5/19`; `+11/−0`, `p=0.0009766` | strong internal paired result across 8 scene clusters |
| Graph rescue causal contrast | graph `16/19` vs equal-budget certified `16/19`; `+0/−0` | remove graph rescue from the main architecture |
| Role-free Novel safety | `0/7` false accepts | only a smoke test; one-sided 95% upper bound is still 34.8% |
| Controlled viewpoint sweep | 3,040 retained rows, 8 trajectories, 2 scenes | mechanism evidence, insufficient scene breadth |
| Public/cross-dataset benchmark | none | still missing |

The NNR result is independently recounted and uses byte-identical online A/B
prefixes across all arms. It remains an internal test on a consumed source
pool, not scene-disjoint paper confirmation.

## Why another fresh160 is not useful

The successful-A fresh160 population is almost entirely high-support. Its
actual-online max co-visibility median is `0.898`, and `115/120` episodes exceed
`0.50`. Repeating that distribution would increase the denominator without
testing the current uncertainty: whether a fixed certificate remains safe and
useful near or outside the support boundary.

The existing train40 inventory already supplies that challenge:

| session max support band, audit-only | sessions | scenes | old geometry pass |
|---|---:|---:|---:|
| `≤0.10` | 274 | 40 | 23.7% |
| `(0.10,0.50]` | 51 | 26 | 54.9% |
| `>0.50` | 155 | 40 | 80.0% |

Co-visibility is used only to stratify outcomes. It is not the deployment
classifier and never enters candidate selection, PnP, or certificate decisions.

## Frozen architecture

```text
causal online RGB history + current ImageGoal
  → DINO retrieves a temporally diverse top-8
  → SuperPoint/LightGlue ranks anchors with label-free 2-D support
  → LingBot history depth lifts correspondences to 2-D/3-D
  → PnP estimates a relative metric pose
  → fixed symmetric certificate
       inliers ≥ 16
       query hull coverage ≥ 5%
       reference hull coverage ≥ 5%
       reprojection RMSE ≤ 2 px
  → pass: emit a scale-free residual bearing to frozen NavDP
  → reject: abstain and leave frozen NavDP unchanged
```

The selected method is the **minimal role-free certified residual**. Graph
rescue is excluded because it emitted 92 real graph plans in three NNR episodes
yet changed no success outcome relative to the equal-budget certified arm.

## Exhaustive train40 certificate challenge

### 2026-08-14 reuse resolution

The queued GPU rerun below was cancelled at zero GPU elapsed time after an
identity audit established that the independently verified CDEC collector
already contains the exact same 480-session geometry center hypotheses. Those
rows were mechanically extracted and independently recounted instead.

The exhaustive result is TP/FP/FN/TN `122/9/31/318`, precision `93.13%`
(Wilson 95% CI `87.46--96.34%`) and recall `79.74%`. The frozen zero-FP gate
does not pass; the result therefore motivates a role-free mixed-goal
closed-loop falsification, not a perfect-safety claim or threshold retuning.
Full details and immutable hashes are in
`TRAIN40_CERTIFICATE_REUSE_RESULT_20260814.md`.

The following block is retained as the historical submission receipt; job
`15703087` did not run.

The next offline test runs the frozen pipeline on all 480 train sessions, not a
label-balanced sample. The manifest is a complete sorted session universe;
selection cannot depend on support labels or previous method outcomes.

- manifest: `train40_certificate_challenge_manifest_20260814.json`;
- 480 sessions, 40 train scenes, universe SHA256
  `fe974099a82a411d451fbb32fb85b4f0683fa644b3e8f602d724f0cf49ac6d2f`;
- Slurm job: `15703087`, partition `a100_tandon`, limit 12 hours;
- immutable source bundle SHA256:
  `1690b30c71e1d1bcad6b3e3c9b106dbfe5ad4dc982ffa52b1b9c9c306e274633`;
- expected runtime from the prior 24-session run: about 8.0 GPU-hours
  (`1,444.5 s / 24 × 480`), with session-atomic checkpoint/requeue support.
- independently frozen raw-CSV verifier:
  `independent_verify_train40_certificate_challenge.py`, SHA256
  `a3c31e276edb9618835b28dc3ea2be925cf12ab76c79e7d1844cbe59d4a0c78f`.

The report is frozen to include:

1. overall TP/FP/FN/TN, precision, recall, and Wilson intervals;
2. the same confusion counts by selected-anchor support and session-max support;
3. memory-age bands `≤32`, `33–96`, and `>96` frames, tied to the LingBot
   window rather than fitted quantiles;
4. causal-state strata (`goal_b_t0`, `goal_b_midpoint_t1`, `goal_c_t0`);
5. scene coverage and exact artifact hashes.

No threshold is changed after observing this run. This exhaustive train-only
challenge characterizes the frozen method; it does not itself establish SR or
scene generalization.

## Decision sequence

1. Complete and independently recount job `15703087`.
2. If precision/coverage remain usable without changing the certificate,
   freeze the minimal adapter and run one scene-disjoint, role-free mixed-goal
   closed-loop confirmation.
3. Only after that confirmation, run a public secondary benchmark, with a
   MemoNav-derived MP3D protocol as the first practical candidate.

Explicit non-actions: do not rerun same-population fresh160, do not restore
graph rescue, do not treat co-visibility as a runtime Novel/Revisit label, and
do not open blind16 for architecture or threshold selection.
