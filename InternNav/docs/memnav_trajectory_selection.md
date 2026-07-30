# MemNav Trajectory Post-Selection (Geometric Collision Filter)

*2026-07-24. Code: `internnav/model/basemodel/memnav/collision_check.py`;
integration: `NavDP/baselines/memnav/policy_agent.py` (`plan()`); validation:
`scripts/diag_collision_geo/diag_collision_geometry.py` + `scripts/train_memnav/diag_collision_geometry.sbatch`.*

## 1. Problem

MemNav's diffusion head samples **N = 16 candidate trajectories** (each 24
waypoints in the current camera planar frame) and must pick one to execute.
Previously the pick was the **endpoint medoid** — no collision reasoning at all
(`all_values` was a `[0.0]*N` placeholder).

NavDP / LoGoPlanner solve this with a **learned critic**, jointly trained
against a GT-mesh collision label (`navdp_dataset.py:372-381`):

```
critic = -5.0 * frac(waypoints with BEV-L1 dist to obstacle < 0.1)
         + 0.5 * (clearance_at_end − clearance_at_start)
no obstacles in view → 2.0
```

At inference they argmax the critic over the 16 samples. MemNav has no critic —
but the frozen **LingBot backbone already predicts full-resolution depth of the
current view** (the same up-to-scale units as its poses, made metric by the
per-episode ground scale). So the *same score* can be computed **geometrically,
per planning step, with zero training**.

Decision (user, 2026-07-23): **LingBot depth only** (not GT sensor depth — the
method stays RGB-only), **filter only** (no critic training for now).

## 2. Pipeline: depth → point cloud → BEV → score → pick

### 2.1 Depth → 3D point cloud (camera frame)

One extra `_predict_depth` call on the current frame's already-computed
aggregated tokens gives `depth [H,W]` + `depth_conf [H,W]` (518×518, LingBot
map units). Each pixel `(u, v)` with depth `d` unprojects by the pinhole model,
with intrinsics derived from the FoV encoded in the pose9 vector
(`fov_v = pose9[7]`, `fov_h = pose9[8]`):

```
fx = (W/2) / tan(fov_h / 2)          fy = (H/2) / tan(fov_v / 2)
x_c = (u − W/2) · d / fx             (right)
y_c = (v − H/2) · d / fy             (down)
z_c = d                              (forward)
```

Pixels are subsampled with `pixel_stride=4` (≈17k rays — plenty for a 0.2 m-scale
BEV query) and filtered by a **per-frame confidence quantile**
(`conf_quantile=0.25`, keep top 75%; conf is relative per frame, not calibrated
across frames/checkpoints).

**No pose transform is needed.** The candidate waypoints are defined in the
*current* camera's planar frame, and the depth we unproject is the *current*
frame's — both sides are already in the same frame. This is the key
simplification of the whole design.

### 2.2 Point cloud → BEV obstacle points ("what counts as an obstacle")

BEV = drop the height axis. But first, height is what decides *whether a point
is an obstacle at all*, so the steps are:

1. **Compute each point's height above the local floor.** The capture camera is
   level (MP3D generation uses pitch 0), so height above floor is simply
   `h_above = (h_est − y_c) · metric_scale` meters, where `h_est` is the
   episode's camera-to-floor distance in LingBot units (`ground_h_est`; in the
   live agent it is recovered from the ground scale as
   `h_est = 1.15 · cam_h / scale`) and `metric_scale` is the ground scale
   (LingBot units → meters).

2. **Keep only points in the obstacle height band** `0.15 m ≤ h_above ≤ 1.2 m`:
   below 0.15 m is floor + depth ripple on the floor plane (would create phantom
   obstacles everywhere); above 1.2 m is ceilings/lintels the Dingo robot passes
   under. If `h_est` is unavailable the band is applied around camera level
   conservatively (floor survives → over-reports collision rather than being blind).

3. **Project the survivors onto the floor plane** — map camera axes to the
   planar frame (x forward, y left) and scale to meters:

   ```
   x_p =  z_c · metric_scale    (forward)
   y_p = −x_c · metric_scale    (left)
   ```

   plus a range cut `x_p ≤ 5 m` (LingBot depth past that is unreliable and the
   24-waypoint horizon never reaches it anyway).

The result is `obs_xy [M, 2]`: a **scattered 2-D obstacle point set**, not a
rasterized occupancy grid. We deliberately skip gridding — with 16×24 waypoints
vs ~10⁴ points, a direct `torch.cdist` min-distance (one GPU matmul-sized op) is
cheaper and exact, whereas a grid adds a resolution parameter and quantization
error for no benefit. Conceptually it *is* the occupied-cell set of a BEV
occupancy map with infinitesimal cells; "distance to nearest obstacle point <
robot radius" is the same test as "waypoint's inflated footprint touches an
occupied cell".

### 2.3 Visibility mask (the "occlusion" part of the BEV view)

A single-view point cloud only describes the **sensed wedge** — nothing behind
the camera, outside the horizontal FoV, or beyond max range. NavDP's critic is
trained on the full GT mesh and doesn't have this problem; we do, and treating
unseen space as free is exactly the corridor-corner failure mode.

So each waypoint is classified visible/unknown:

```
visible(w) = |atan2(y_w, x_w)| ≤ fov_h/2 − 0.05  and  ‖w‖ ≤ 5 m
```

Only **visible** waypoints are scored; unknown ones contribute nothing (their
`min_dist` is +inf, i.e. "no information", not "safe"). Note this wedge test
handles the FoV boundary but **not intra-view occlusion**: space *behind* a
sensed obstacle (past the first surface along a ray) is also unknown, yet a
waypoint there currently still gets scored against the visible points. In
practice that waypoint is near/behind an obstacle so it scores badly anyway,
but a ray-casting visibility check would be the rigorous fix.

### 2.4 Scoring and selection

Per candidate, the NavDP formula with Euclidean distance over visible waypoints:

```
score = −5.0 · frac(visible waypoints with min_dist < 0.30 m)
        + 0.5 · (min_dist at last visible − min_dist at first visible)
```

No obstacle points, or no visible waypoint → `NO_OBSTACLE_SCORE = 2.0` (NavDP's
constant, kept so downstream thresholds transfer). Selection is **argmax**;
ties (e.g. all candidates obstacle-free) break by the **endpoint medoid**, so
obstacle-free views reproduce the old selector exactly.

### 2.5 Integration

`policy_agent.plan()`: env-gated by `MEMNAV_COLLISION_SELECT` (default **on**,
`=0` disables → medoid). Any exception falls back to medoid with a log line.
The per-candidate scores are returned as `all_values` (previously zeros).

## 3. Validation (Phase-0): is LingBot depth good enough?

`scripts/diag_collision_geo/diag_collision_geometry.py` compares LingBot-derived vs GT-sensor-depth
BEV obstacle distances on a grid of would-be waypoint positions (x 0.4–3.6 m,
|y| ≤ 1.6 m, both FoV wedges), on the ~30 episodes whose raw frames survive on
host. 16 episodes, 74k query points:

| metric | value |
|---|---|
| nearest-obstacle distance error, median | **0.16 m** |
| —, p90 | 1.09 m (heavy tail) |
| collision @ r=0.30, `conf_quantile=0.5` (old) | P 0.89, **R 0.49** |
| collision @ r=0.30, `conf_quantile=0.25` (**default**) | **P 0.85, R 0.77** |
| collision @ r=0.40, `conf_quantile=0.0` (paranoid option) | P 0.68, R 0.95 |

The low initial recall was the confidence filter discarding wall pixels, not
depth quality (sweep job 14701113; full table in
`logs/mp3d_gen/diag_collision_geometry_14701113.json`). Reading: **precision =
1 − phantom-veto rate** (how often a vetoed path was actually fine), **recall =
fraction of true collisions the filter sees**; a recall miss degrades to the old
medoid behavior, so the filter is a strict improvement over no filter.

## 4. Known limitations / future work

- **Current-view only**: can't veto paths that collide behind the camera or
  around corners — the gap a Phase-2 *trained* critic (distilled from these
  geometric labels, or NavDP-style from GT mesh) would close.
- **Intra-view occlusion** not ray-cast (see §2.3).
- `h_est` recovery in the agent is exact only when the ground-scale clamp
  `[0.8, 6.0]` didn't bind (>95% of episodes).
- Level-camera assumption inherited from MP3D generation (pitch 0); a pitched
  deployment camera would need the pitch folded into the height computation.
- Not yet A/B-evaluated end-to-end (`MEMNAV_COLLISION_SELECT=1` vs `0`).
