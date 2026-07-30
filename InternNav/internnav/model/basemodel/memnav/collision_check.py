"""Geometric trajectory selection from LingBot's predicted depth (no learned critic).

MemNav's DDPM head samples N candidate trajectories and previously picked the
endpoint medoid (NavDP/baselines/memnav/policy_agent.py). NavDP/LoGoPlanner instead
train a critic whose label is a GT-mesh collision score (navdp_dataset.py:372-381:
    critic = -5.0 * frac(waypoint BEV-L1 dist to obstacle < 0.1)
             + 0.5 * (clearance_end - clearance_start),  no obstacles -> 2.0)
and argmax it at inference. We have no critic, but the frozen LingBot backbone
predicts full-res depth of the CURRENT view (the same map-scale units as its poses,
made metric by the per-episode ground scale) — so the same score can be computed
geometrically, per planning step, with no training.

Frames. Candidate waypoints live in the current camera PLANAR frame (x forward,
y left, theta CCW; policy_agent.plan docstring). The current frame's depth
unprojects into its own OpenCV camera frame (x right, y down, z forward) with
intrinsics only — no pose needed — and maps to planar as x_p = z_c, y_p = -x_c,
up = -y_c. Both sides end up in METERS: waypoints are metric by construction
(naction/4 -> cumsum), obstacle points via metric_scale (lingbot -> m).

Differences from the NavDP label, on purpose:
  * Euclidean BEV distance, not L1 (their 0.1 threshold is in summed-xy units).
  * A VISIBILITY mask: only waypoints inside the sensed FoV wedge and range are
    scored. NavDP's critic sees the whole GT mesh; a single-view point cloud says
    nothing about what is behind the camera or past max range — treating unseen
    as safe is exactly the corridor-corner failure mode.
"""

import math

import torch

# obstacle band above the local floor, meters: below 0.15 is floor/noise (depth
# ripple on the floor plane), above 1.2 clears the Dingo body + camera mount with
# margin while ignoring ceilings/lintels the robot passes under.
OBSTACLE_H_BAND_M = (0.15, 1.2)
ROBOT_RADIUS_M = 0.30
# score of a trajectory when no obstacle is in view — NavDP's no-obstacle constant
# (navdp_dataset.py:377-379), kept so downstream thresholds transfer.
NO_OBSTACLE_SCORE = 2.0


@torch.no_grad()
def obstacle_points_from_depth(depth, conf, fov_v, fov_h, h_est, metric_scale,
                               conf_quantile=0.25, pixel_stride=4,
                               h_band_m=OBSTACLE_H_BAND_M, max_range_m=5.0):
    """Current-view predicted depth -> BEV obstacle points in the planar frame, meters.

      depth, conf  : [H, W] full depth-head output for ONE frame (lingbot units)
      fov_v, fov_h : vertical/horizontal FoV in radians (pose9[7], pose9[8])
      h_est        : episode camera-to-floor distance, lingbot units (ground_h_est);
                     None -> fall back to camera_height/metric_scale is NOT attempted,
                     the height band is applied around y=0 conservatively widened.
      metric_scale : lingbot units -> meters multiplier (ground scale)
    Returns obs_xy [M, 2] (x forward, y left, meters). M can be 0."""
    H, W = depth.shape
    dev = depth.device
    fy = (H / 2.0) / math.tan(fov_v / 2.0)
    fx = (W / 2.0) / math.tan(fov_h / 2.0)
    d = depth[::pixel_stride, ::pixel_stride]
    c = conf[::pixel_stride, ::pixel_stride]
    vs = torch.arange(0, H, pixel_stride, device=dev, dtype=torch.float32)
    us = torch.arange(0, W, pixel_stride, device=dev, dtype=torch.float32)
    v, u = torch.meshgrid(vs, us, indexing="ij")
    x_c = (u - W / 2.0) * d / fx                                  # right
    y_c = (v - H / 2.0) * d / fy                                  # down
    # per-frame conf quantile, as in ground_frame_heights: conf is relative, not
    # calibrated across frames/checkpoints.
    keep = (d > 1e-6) & (c >= torch.quantile(c.reshape(-1), conf_quantile))
    # height above the LOCAL floor, meters (capture camera is level: MP3D gen
    # pitch 0 — same assumption ground_h_est itself rests on).
    if h_est is not None:
        h_above = (float(h_est) - y_c) * metric_scale
        keep &= (h_above >= h_band_m[0]) & (h_above <= h_band_m[1])
    else:
        # no floor estimate: keep everything below camera-level+band top; floor
        # points then survive and make the check conservative (over-reports
        # collision) rather than blind.
        keep &= (-y_c * metric_scale) <= h_band_m[1]
    x_p = d * metric_scale                                        # forward = z_cam
    y_p = -x_c * metric_scale                                     # left = -x_cam
    keep &= x_p <= max_range_m
    return torch.stack([x_p[keep], y_p[keep]], -1)                # [M, 2]


@torch.no_grad()
def score_trajectories(paths, obs_xy, fov_h, robot_radius_m=ROBOT_RADIUS_M,
                       collision_penalty=5.0, clearance_weight=0.5,
                       max_range_m=5.0, fov_margin=0.05):
    """NavDP-style geometric critic over candidate trajectories.

      paths  : [N, T, 3] planar (x, y, theta) waypoints, meters
      obs_xy : [M, 2] planar obstacle points, meters (obstacle_points_from_depth)
      fov_h  : horizontal FoV (radians) — the sensed wedge; waypoints outside it
               (or past max_range_m) are UNKNOWN and excluded from the score.
    Returns (scores [N], min_dist [N, T] with +inf where unseen/no obstacle).
    Score (mirrors navdp_dataset.py:372-381, Euclidean, visible-only):
      -collision_penalty * frac(visible waypoints with dist < robot_radius)
      + clearance_weight * (dist at last visible - dist at first visible)
    No obstacle in view, or no visible waypoint -> NO_OBSTACLE_SCORE."""
    N, T, _ = paths.shape
    dev = paths.device
    xy = paths[..., :2]                                           # [N, T, 2]
    ang = torch.atan2(xy[..., 1], xy[..., 0])
    rng = xy.norm(dim=-1)
    visible = (ang.abs() <= (fov_h / 2.0 - fov_margin)) & (rng <= max_range_m)
    min_dist = torch.full((N, T), float("inf"), device=dev)
    if obs_xy.numel() == 0 or not visible.any():
        return torch.full((N,), NO_OBSTACLE_SCORE, device=dev), min_dist
    d_all = torch.cdist(xy.reshape(N * T, 2), obs_xy).min(-1).values.reshape(N, T)
    min_dist = torch.where(visible, d_all, min_dist)

    coll = ((d_all < robot_radius_m) & visible).float().sum(-1)
    n_vis = visible.float().sum(-1)                               # [N]
    frac = torch.where(n_vis > 0, coll / n_vis.clamp(min=1), torch.zeros_like(coll))
    # clearance gain between the first and last VISIBLE waypoint of each candidate
    first = visible.float().argmax(-1)                            # [N]
    last = T - 1 - visible.flip(-1).float().argmax(-1)
    idx = torch.arange(N, device=dev)
    gain = d_all[idx, last] - d_all[idx, first]
    scores = -collision_penalty * frac + clearance_weight * gain
    scores = torch.where(n_vis > 0,
                         scores, torch.full_like(scores, NO_OBSTACLE_SCORE))
    return scores, min_dist


@torch.no_grad()
def select_trajectory(paths, scores, tie_eps=1e-4):
    """argmax score; ties (e.g. every candidate NO_OBSTACLE_SCORE) break by the
    endpoint medoid — the previous selector, so obstacle-free views keep the old
    behavior exactly. Returns the winning index."""
    best = scores.max()
    tied = (scores >= best - tie_eps).nonzero().flatten()
    if tied.numel() == 1:
        return int(tied[0])
    ends = paths[tied, -1, :2]
    return int(tied[(ends - ends.mean(0)).norm(dim=-1).argmin()])
