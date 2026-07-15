# MemNav Policy — Pose Pipeline Fixes

`memnav_policy.py` (`MemNavNet`/`RevisitMerge`) and `lingbot_stream.py`
(`LingBotStream`) implement the trainable heads over the frozen LingBot-Map
GCT front-end. This document covers a round of fixes to the **pose
pipeline** specifically — `cur_pose`, `goal_pose`, and how `RevisitMerge`
turns them into the revisit/aux-pose signal — found by comparing the
pipeline's own pose estimates against ground truth and against LingBot's
own "official" continuous-stream inference. Diagnostic tooling lives in
`InternNav/scripts/diag_lingbot_pose_accuracy.py`.

---

## 1. What was wrong

`encode_memory`'s original per-sample loop derived **both** `cur_pose` (the
current frame's absolute camera pose) and `goal_pose` (the revisit goal's
absolute camera pose) by re-deriving them from a **cold-started**
`window_forward`/`goal_append` reconstruction: inject the precomputed scale
+ specials-only history, then recompute a local window of raw frames live
from scratch before reading off the pose.

Comparing this against (a) real GT extrinsics and (b) LingBot's own
`GCTStream.inference_streaming`-equivalent continuous pass (captured once,
for free, during precompute — `cam_pose_enc` in `lingbot_cam_cache.npz`)
showed:

- **`cur_pose`** didn't need to be reconstructed at all — `k` is always a
  real trajectory frame, and its exact pose is already sitting in
  `cam_pose_enc[k]`. `window_forward`'s cold start (no real predecessors at
  the start of the recomputed window) was reproducing this with several
  meters of avoidable error at deep `k`, for zero reason — the ground truth
  reconstruction was on disk the whole time.
- **`goal_pose`** genuinely needs *some* live computation (the goal image
  is newly inserted, not a cached frame), but `goal_append`'s cold start at
  the nominal `window` boundary (32 frames) starved it of context. A deeper
  live recompute (`warm=64`) — still bounded, still cheap, doesn't scale
  with `recall_gap` — closes the gap almost entirely to what a true,
  unbroken continuous stream achieves.
- **`RevisitMerge`** was trying to learn the relative pose
  `T_cur^-1 T_goal` from `cur_pose`/`goal_pose` via independently-embedded
  tokens merged by attention — a representation that is architecturally
  incapable of the bilinear cross term the true relative transform requires
  (`t_rel = R_cur^T(t_goal - t_cur)` mixes a rotation derived from one pose
  with a translation difference derived from both). No amount of data
  fixes an architecture that can't represent the target function.
- **`aux_pose_head`**'s `θ` target isn't recoverable from `(cur_pose,
  goal_pose)` at all, regardless of how accurate those poses are. GT `θ` is
  the path's net heading change between departure and arrival — a function
  of the geodesic route's shape (obstacle layout), not of the two endpoint
  poses. Worse: the goal image's own rendered orientation is independent of
  the true arrival heading *by construction* of the data generator
  (`MemNavData/generate_twoleg.py`'s `roll_leg`: "NO terminal orientation
  alignment... arrival heading is the natural approach heading"; goal yaw =
  the historical anchor frame's own heading + random jitter). There is no
  `θ` signal in the inputs to extract, even in principle.

---

## 2. Fixes

### 2.1 `cur_pose` — read from cache, not reconstructed

`MemNavNet._load_cache` now also loads `cam_pose_enc` from
`lingbot_cam_cache.npz`. `encode_memory` reads `cur_pose =
cache["cam_pose_enc"][k]` directly instead of calling
`self.lingbot.camera_pose(ck, cv, k, cur_agg)`. Cheaper (no extra camera-head
forward) and exact by construction — verified to match `cam_pose_enc[k]`
bit-for-bit on a real batch. `window_forward` is still run for `cur`/`dfeat`
(the RGBD/depth Perceiver branches still need it); only the pose readout
changed.

### 2.2 `goal_pose` — deep warm-recompute instead of a cold start

New method `LingBotStream.goal_append_warm(goal_img, cache, m, rgb_dir,
warm, return_agg=False)`: recomputes live from
`max(num_scale, m - warm + 1)` (not `m - window + 1`) before streaming the
goal at `m+1`. `encode_memory` calls this with `self.goal_warm` (default
**64**) instead of `goal_append`.

Validated against a true continuous-stream oracle and real goal GT
positions (`scripts/diag_lingbot_pose_accuracy.py`, `warm_goal_pose` /
`oracle_goal_pose`):

| depth (3-leg, `m=140`, `recall_gap=290`) | error vs. true goal position |
|---|---|
| production (`window=32`, cold start) | 1.464 m |
| `warm=64` | **1.046 m** |
| `warm=128` | 1.106 m (no further gain) |
| oracle (true continuous stream) | 1.101 m |

`warm=64` matches the oracle to within noise; deeper warm-up buys nothing.
Also checked: the model's own KV eviction can stay at the nominal `window`
(32) during the 64-frame warm loop — an "evict back to 32" run scored
1.038 m, statistically the same as never evicting (1.046 m) — so
`goal_append_warm` needed **no** change to `kv_cache_sliding_window`, only
a longer live-recompute range.

Threaded through config: `il.goal_warm` (`MemNavPolicy.__init__` →
`MemNavNet(goal_warm=...)`), default 64, set explicitly in
`scripts/train/configs/memnav.py`.

### 2.3 `RevisitMerge` — analytic relative pose, not learned absolute-pose fusion

`RevisitMerge._relative_pose(cur_pose9, goal_pose9)` computes
`T_cur^-1 T_goal` in closed form:

```
t_rel = R_cur^T (t_goal - t_cur)
R_rel = R_cur^T R_goal
```

via `quat_to_mat` (`lingbot_map.utils.rotation`, lazy import — needs
`lingbot_repo` on `sys.path`, guaranteed by the time this runs since
`LingBotStream.__init__` already did it). `R_rel` is kept as a flattened
3×3 matrix rather than converted back to a quaternion — nothing downstream
needs the compact 4-d form, and `mat_to_quat`'s branch-selection has known
numerical rough edges near 180° rotations that a plain matrix avoids.

- **`revisit_head`**: trainable `Linear(12, n_out·token_dim)` on
  `[t_rel, R_rel.flatten()]`, reshaped to the decoder's revisit tokens.
  Replaces the old `pose_encoder(7,dim) + TokenCompressor` pipeline — no
  attention machinery needed for a single input feature vector
  (`TokenCompressor` degenerates to per-slot linear reads of one token
  anyway).
- **`aux_pose_head`**: `Linear(3, 2)`, output `(x, y)` **only** — `θ` is
  dropped from this auxiliary task (see §1; it belongs to the diffusion
  decoder, which has the depth/visual context needed to reason about
  obstacles, not `RevisitMerge`, which only ever sees two poses).

### 2.4 `aux_pose_head` is a trainable global calibration

`cur_pose`/`goal_pose` come from the frozen camera head under
`torch.no_grad()`, but that does **not** prevent `Linear(3,2)` from learning:
its input can be constant while its own weight and bias still receive
gradients. The previous freeze made `w_aux * aux_loss` a constant offset in
the reported total loss and did not change any parameter update.

The head is now trainable and initialized only with the known signed-axis
mapping:

```python
aux_pose = aux_pose_head(t_rel)
aux_pose_head.weight = [[0, 0, 1], [-1, 0, 0]]
aux_pose_head.bias = [0, 0]
```

The mapping follows from the generated-data camera-mount correction. Legacy
labels represented `[up, -right, back]`; corrected NavDP coordinates are
`[-back, -right, up]`. Combined with the validated LingBot-to-legacy mapping,
the corrected planar coordinates are `[lingbot_z, -lingbot_x]`.

No fixed metric scale is baked in. The former `1 / 0.541` value came from two
episodes evaluated against legacy labels that omitted one horizontal axis, so
it is not a valid calibration target. Training can learn the best dataset-wide
affine scale, but a single linear head still cannot remove LingBot's
sequence-dependent monocular scale or accumulated VO drift. Also, the aux
output is not consumed by the diffusion decoder, so this loss updates the aux
head only; it is a calibration diagnostic rather than additional policy
supervision.

`MemNavTrainer.compute_loss`'s `gt_pose` is now sliced to
`inputs["batch_goal_rel_pose"][..., :2]` to match.

---

## 3. Files touched

| file | change |
|---|---|
| `internnav/model/basemodel/memnav/memnav_policy.py` | `_load_cache` loads `cam_pose_enc`; `cur_pose` reads it directly; `RevisitMerge` uses analytic `_relative_pose` and a trainable corrected-axis aux calibration; `MemNavNet.__init__` gains `goal_warm` |
| `internnav/model/basemodel/memnav/lingbot_stream.py` | new `goal_append_warm` method |
| `internnav/trainer/memnav_trainer.py` | `gt_pose` sliced to `[..., :2]`; aux loss trains only the calibration head |
| `scripts/train/configs/memnav.py` | explicit `goal_warm=64` |
| `scripts/diag_lingbot_pose_accuracy.py` | new diagnostic harness (GT vs. official-continuous-stream vs. ours; `warm_forward`/`warm_goal_pose`/`oracle_goal_pose`) used to find and validate all of the above |

## 4. Open items

- **Precompute still runs at `kv_cache_sliding_window=32`**, not
  LingBot's intended 64 — `cam_pose_enc` itself (hence `cur_pose`, which
  reads it directly) would be more accurate at window=64 (0.35–0.40 m ATE
  measured vs. 0.64–0.65 m at window=32 on the same trajectories). Not yet
  changed — it's a precompute config/cost tradeoff (roughly doubles
  per-trajectory KV work), not a code fix, and out of scope for this round.
- **Global aux calibration has an irreducible residual** because LingBot's
  monocular scale varies by sequence and long trajectories accumulate drift.
  Report per-axis errors and distance-binned errors; do not interpret one
  aggregate aux MSE as a direct policy-quality metric.
- Frozen VO accuracy has a real, separate ceiling on long/turn-heavy
  trajectories (measured 2.5 m ATE on a 744-frame, 2-turn episode even for
  the trusted continuous-stream reference) — not something any of the
  fixes above can close; it's a property of the frozen model itself, not
  the reconstruction path.
