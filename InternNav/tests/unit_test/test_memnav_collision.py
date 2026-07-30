"""Unit tests for memnav geometric trajectory selection (collision_check.py).

Synthetic scenes, CPU-only: known obstacle layouts -> known score ordering.
Run: python -m pytest tests/unit_test/test_memnav_collision.py -q
"""

import math

import torch

from internnav.model.basemodel.memnav.collision_check import (
    NO_OBSTACLE_SCORE,
    obstacle_points_from_depth,
    score_trajectories,
    select_trajectory,
)

FOV_H = math.radians(90.0)
FOV_V = math.radians(58.0)


def _straight(n=24, step=0.125, heading=0.0):
    """Straight path along `heading` (planar CCW), [1,T,3]."""
    t = torch.arange(1, n + 1, dtype=torch.float32) * step
    return torch.stack([t * math.cos(heading), t * math.sin(heading),
                        torch.full_like(t, heading)], -1)[None]


def _wall(x, y_lo=-1.0, y_hi=1.0, n=100):
    """Obstacle wall at forward distance x, [n,2]."""
    y = torch.linspace(y_lo, y_hi, n)
    return torch.stack([torch.full_like(y, x), y], -1)


def test_straight_into_wall_scores_below_avoiding():
    obs = _wall(1.5)
    paths = torch.cat([_straight(),                       # drives through the wall
                       _straight(heading=math.radians(40))], 0)  # veers left of it
    scores, min_dist = score_trajectories(paths, obs, FOV_H)
    assert scores[1] > scores[0]
    assert min_dist[0].min() < 0.05                       # waypoints pierce the wall
    assert select_trajectory(paths, scores) == 1


def test_no_obstacles_falls_back_to_medoid():
    paths = torch.cat([_straight(heading=math.radians(a)) for a in (-30, 0, 30)], 0)
    scores, _ = score_trajectories(paths, torch.zeros(0, 2), FOV_H)
    assert (scores == NO_OBSTACLE_SCORE).all()
    # medoid of endpoints at -30/0/30 deg is the middle (straight) candidate
    assert select_trajectory(paths, scores) == 1


def test_out_of_fov_waypoints_are_not_scored():
    # obstacle far to the side, outside the 90-deg wedge; a path curving into it
    # keeps only in-wedge waypoints scored — no phantom collision from unseen space
    obs = torch.tensor([[0.3, 2.0]])                      # ~81 deg off-axis, in front
    behind = _straight(heading=math.radians(120))          # exits wedge immediately
    scores, min_dist = score_trajectories(behind, obs, FOV_H)
    assert scores[0] == NO_OBSTACLE_SCORE                 # nothing visible to score
    assert torch.isinf(min_dist[0]).all()


def test_clearance_term_prefers_retreating_path():
    obs = _wall(0.6, n=40)                                # wall right ahead
    toward = _straight(step=0.02)                          # creeps toward it (no hit)
    away = -_straight(step=0.02)                           # backs off
    away[..., 2] = 0.0
    # backing up leaves the FoV wedge, so compare two forward paths instead:
    # skimming along the wall vs angling away from it
    skim = _straight(step=0.02, heading=0.0)
    ang = _straight(step=0.02, heading=math.radians(35))
    scores, _ = score_trajectories(torch.cat([skim, ang], 0), obs, FOV_H)
    assert scores[1] > scores[0]


def test_obstacle_extraction_height_band_and_frame():
    # synthetic depth image of a fronto-parallel wall at 2.0 (lingbot units) with
    # the floor visible below: floor pixels must be rejected by the height band,
    # wall pixels kept and mapped to x_p ~= 2.0 * scale meters.
    H = W = 120
    h_est, scale = 0.25, 2.0                              # cam->floor 0.25 lb-units
    fy = (H / 2.0) / math.tan(FOV_V / 2.0)
    v = torch.arange(H, dtype=torch.float32)[:, None].expand(H, W).clone()
    depth = torch.full((H, W), 2.0)
    # rows whose ray hits the floor first: y_c = (v - H/2) * d / fy = h_est at
    # d_floor = h_est * fy / (v - H/2); where d_floor < 2.0 the floor is closer
    below = v > H / 2.0
    d_floor = torch.where(below, h_est * fy / (v - H / 2.0).clamp(min=1e-3),
                          torch.full_like(v, 1e9))
    depth = torch.minimum(depth, d_floor)
    conf = torch.ones(H, W)
    obs = obstacle_points_from_depth(depth, conf, FOV_V, FOV_H, h_est, scale,
                                     conf_quantile=0.0, pixel_stride=2)
    assert obs.numel() > 0
    # every kept point is the wall, at metric forward distance 4.0 (= 2.0 * scale)
    assert (obs[:, 0] - 4.0).abs().max() < 1e-4
    # floor points (h_above ~ 0) were all rejected
    band_h = (float(h_est) - 0.0) * scale                 # camera-level pixel height
    assert band_h <= 1.2                                  # sanity on the synthetic setup


def test_conf_quantile_filters_low_conf():
    H = W = 40
    depth = torch.full((H, W), 2.0)
    conf = torch.ones(H, W)
    conf[: H // 2] = 0.0                                  # top half low-conf
    obs_all = obstacle_points_from_depth(depth, conf, FOV_V, FOV_H, None, 1.0,
                                         conf_quantile=0.0, pixel_stride=1)
    obs_flt = obstacle_points_from_depth(depth, conf, FOV_V, FOV_H, None, 1.0,
                                         conf_quantile=0.6, pixel_stride=1)
    assert obs_flt.shape[0] < obs_all.shape[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
