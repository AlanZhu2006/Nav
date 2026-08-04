# MemNav decoder-gate + Novel-conditioning fixes — 2026-08-04

Working tree: `/home/asus/Research/Nav` (main). All commits this date; diagnostics
that motivated each fix are committed under `InternNav/scripts/diag_retrieval/` and
their result JSON under `Nav/eval2leg_results/`.

## 1. What was diagnosed (checkpoint-5570, `memnav_mp3d_decgate`, W&B `l3bhs8i8`)

| Finding | Evidence |
|---|---|
| `dec_gate_a/b` do not train: random walk at init (net +0.009/+0.165 over 5.5k steps) while the BCE pair in the SAME 10x-LR group moves monotonically (+1.47/+1.08) | W&B scalars |
| The action loss is **locally optimal in the decoder gate logit z** — flat at ±2, significantly worse at ±4 in BOTH directions, on BOTH revisit and novel rows. Nothing to descend; more steps/LR cannot fix an equilibrium | `diag_decgate_zsweep.py` (`eval2leg_results/zsweep_decgate5570/`) |
| The **novel branch is used but content-dead**: zeroing its tokens hurts (+0.0094 on novel rows) but swapping in a different episode's goal changes the output by ~nothing (RMS 0.0023 vs 0.08) — it collapsed to constant "register" tokens carrying no goal information | `diag_branch_ablation.py` (`eval2leg_results/ablation_decgate5570/`) |
| The revisit branch IS read and content-sensitive (swap hurts more than zero: +0.0141 vs +0.0120) — but amplifying it never helps (saturation) | same |
| A **fresh** DINOv2-init model is goal-sensitive (swap output-RMS 0.169) — the collapse is trained-in by the contrast-free action MSE, not innate | port smoke test |
| Independent confirmation (student worktree `Nav-axis-uturn`): same-state same-seed goal swap moves MemNav candidates by 0.13–3.16% of seed variation vs **NavDP 176.8%**; unseen-scene closed loop NavDP 9/10 SR vs MemNav pure-Novel 2–4/10 | `MemNavData/NOVEL_ROOT_CAUSE_AUDIT_20260804.md`, `NOVEL_NAVDP_PAIRED_EVAL_20260804.md` |

Root cause, one paragraph: NavDP avoids goal-collapse via data-level contrast
(same state, different goals, different expert actions), aux goal-position heads,
and a separate no-goal branch. MemNav's two-leg data has none of that — "continue
forward" explains most frames, so ignoring the goal image is a low-loss shortcut
(classic imitation-learning causal confusion), and the closed-at-init decoder gate
(z = 10·max_cos − 8 ≈ −5 with fresh projections) additionally prevented the decoder
from ever learning to read the teacher-anchored revisit tokens.

## 2. Fixes applied (all on main)

### 2.1 Decoder-gate curriculum + neutral init + selectable fusion
(`decgate_schedule.py`, `memnav_policy.py`, `memnav_trainer.py`)

* **Logit-space teacher curriculum**: decoder uses `z_used = z_pred + r·(±teacher_z − z_pred)`,
  r annealed 1→0; grad to `z_pred` scales by 1−r; BCE-free scaffold; eval path
  untouched at r=0. `teacher_z=3` ≈ cos 0.9 through a converged calibration, NOT
  the ±9.2 rail (the visual branch has value even on GT revisits).
* **Neutral `dec_gate_a/b` init (0,0)** instead of the classifier's (10,−8): a
  router should start agnostic, not closed.
* **Fusion modes** (`MEMNAV_DECGATE_FUSION`, persisted in checkpoint buffers, synced
  on load — legacy checkpoints keep `symmetric`): `symmetric` ±z/2 tilt (current),
  `residual` (revisit +z, novel untouched — visual branch as always-on base policy),
  `value_scale` (revisit token values ×σ(z); gradient scales with readout magnitude,
  not attention weight on a suppressed column; `MEMNAV_DECGATE_SCALE_NOVEL` also
  scales novel by 1−σ(z)).

### 2.2 NavDP warm start for the novel backbone (`warmstart_navdp.py`)

`NovelBranch` wraps the SAME image-goal encoder NavDP trains (DINOv2-S/14,
6-channel early fusion) — verified 174/174 key/shape match. The script remaps
`image_encoder.*` → `core.novel.backbone.*` and writes an init checkpoint for
`MEMNAV_CKPT_TO_LOAD` (loaded strict=False). `MEMNAV_FREEZE_NOVEL_BACKBONE=1`
pins the encoder (a frozen encoder cannot collapse); heads/decoder stay trainable.
Initialization alone does NOT remove the shortcut — pair with 2.3.

### 2.3 Goal-swap counterfactual loss (`goal_swap.py`, dataset, trainer)
Ported from the student worktree (`edca2dd`) onto the decgate architecture.

* Negatives: same scene, same goal type (A↔A, B↔B; type relaxed only on pool
  exhaustion, scene never), per-sampled-k direction filter — most bearing-divergent
  wrong goal, ≥30° and ≥0.5 m (world coords select only, never model inputs).
* Loss: two decodes differing ONLY in the goal image (same state/history/noise/
  timestep/blended gate); hinge `relu(margin − (mse_wrong − mse_correct))`,
  margin 0.05 — the true goal must explain its own expert action better than a
  divergent wrong goal. No action label is invented for the wrong goal. Novel rows
  only (on revisit rows the shared memory pathway makes a similar action legitimate).
* Cost: ~0.1–0.3 s on a measured ~30 s step (<1%).

### 2.4 Early-Novel coverage + no-candidate hygiene (from the same port)

* `MEMNAV_GOAL_A_MIN_K=40` starts Goal-A rows at the real inference boundary
  (k=40) instead of the historical k≥122; E(k) may be empty.
* All-masked rows skip the (expensive, fabricated) revisit goal-pose append in
  `encode_memory` and get an exactly-zero revisit feature in `forward` — no longer
  relying on a small gate tilt to suppress a made-up pose token.

## 3. Switches (all env, read by `scripts/train/configs/memnav.py`)

| Env | Default | Meaning |
|---|---|---|
| `MEMNAV_GOAL_SWAP` | `1` | **master switch** for 2.3 (0 = no negative pool, no second decode) |
| `MEMNAV_GOAL_SWAP_WEIGHT` / `_MARGIN` / `_MIN_ANGLE_DEG` | 0.25 / 0.05 / 30 | tuning when on |
| `MEMNAV_GOAL_A_MIN_K` | unset (=122 behavior) | 40 = train Novel from the inference boundary |
| `MEMNAV_CKPT_TO_LOAD` | unset | warm-start init ckpt (use `warmstart_navdp.py` output) |
| `MEMNAV_FREEZE_NOVEL_BACKBONE` | off | freeze the warm-started encoder |
| `MEMNAV_DECGATE_FUSION` | `symmetric` | `symmetric` \| `residual` \| `value_scale` |
| `MEMNAV_DECGATE_TEACHER_START/END/STEPS/Z` | 1.0 / 0.0 / 500 / 3.0 | gate curriculum (`STEPS=0` = off = control arm) |
| `MEMNAV_DECGATE_INIT_A/B` | 0.0 / 0.0 | decoder-gate affine init |

Recommended first run (one fix per broken link, attributable):

```bash
python scripts/train_memnav/warmstart_navdp.py \
    --navdp_ckpt <navdp_checkpoint.ckpt> \
    --out checkpoints/navdp_warmstart/memnav_novel_init.ckpt
export MEMNAV_CKPT_TO_LOAD=$PWD/checkpoints/navdp_warmstart/memnav_novel_init.ckpt
export MEMNAV_FREEZE_NOVEL_BACKBONE=1
export MEMNAV_GOAL_A_MIN_K=40
# goal swap + gate curriculum are ON by default; MEMNAV_GOAL_SWAP=0 /
# MEMNAV_DECGATE_TEACHER_STEPS=0 are the ablation off-switches
```

## 4. Acceptance metrics (before/after instruments already exist)

* W&B `action/goal_swap_error_gap` → should climb past +0.05;
  `action/goal_swap_output_rms` → from ~0.002 toward ~0.08 (the `zero_novel` level).
* Re-run `diag_branch_ablation.py` on the new checkpoint: `swap_novel` RMS/Δloss
  should approach `zero_novel`'s.
* Re-run `diag_decgate_zsweep.py`: success = revisit rows develop a genuine
  downward slope toward +z while novel rows stay closed (curves separate).
* `action/dec_gate_a`, `dec_gate_b`, `retrieval/gate_*` scalars: dec pair should
  move directionally once the branches carry signal.
* Closed loop: `MemNavData/eval_2leg_habitat.py` paired A/B, and the student's
  unseen-scene pure-Novel protocol vs NavDP (target: close the 2–4/10 vs 9/10 gap).
