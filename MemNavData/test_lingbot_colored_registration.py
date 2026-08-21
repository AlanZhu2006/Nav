import math

import numpy as np
import pytest
import torch

from lingbot_colored_registration import (
    RegistrationSchedule,
    apply_world_delta_to_pose9,
    colored_world_cloud,
    multiscale_registration,
    pose9_to_matrix,
)


def yaw_transform(yaw: float, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])
    transform[:3, 3] = translation
    return transform


def test_apply_world_delta_to_pose9_left_composes_pose():
    pose = np.array([1.0, 0.2, 2.0, 0.0, 0.0, 0.0, 1.0, 1.1, 1.2])
    delta = yaw_transform(math.radians(20.0), (0.3, -0.1, 0.4))
    refined = apply_world_delta_to_pose9(pose, delta)
    np.testing.assert_allclose(
        pose9_to_matrix(refined), delta @ pose9_to_matrix(pose), atol=1e-7)
    np.testing.assert_allclose(refined[7:], pose[7:])


def test_colored_world_cloud_keeps_xyz_rgb_sampling_aligned(monkeypatch):
    # Avoid importing the vendored LingBot package in this pure helper test.
    import sys
    import types

    rotation = types.ModuleType("lingbot_map.utils.rotation")
    rotation.quat_to_mat = lambda _q: torch.eye(3)
    monkeypatch.setitem(sys.modules, "lingbot_map", types.ModuleType("lingbot_map"))
    monkeypatch.setitem(sys.modules, "lingbot_map.utils", types.ModuleType("lingbot_map.utils"))
    monkeypatch.setitem(sys.modules, "lingbot_map.utils.rotation", rotation)

    depth = torch.ones(4, 4)
    confidence = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    image = torch.zeros(3, 4, 4)
    image[0] = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 15.0
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    points, colors, mean_conf = colored_world_cloud(
        depth, confidence, image, pose,
        pixel_stride=1, confidence_quantile=0.5, max_points=64)
    assert points.shape == colors.shape
    assert points.shape[0] == 8
    assert np.all((0.0 <= colors) & (colors <= 1.0))
    assert mean_conf == pytest.approx(11.5)


@pytest.mark.parametrize("method", ["geometric", "colored"])
def test_multiscale_registration_recovers_small_rigid_delta(method):
    # A non-planar colored surface avoids the tangent-plane ambiguity that
    # makes a synthetic flat ICP test underdetermined.
    grid = np.linspace(-0.8, 0.8, 24)
    x, z = np.meshgrid(grid, grid)
    y = 0.12 * np.sin(2.3 * x) + 0.08 * np.cos(1.7 * z)
    target = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = np.stack([
        (x + 0.8) / 1.6,
        (z + 0.8) / 1.6,
        0.5 + 0.4 * np.sin(2.0 * x + z),
    ], axis=-1).reshape(-1, 3)
    truth = yaw_transform(math.radians(6.0), (0.06, -0.01, -0.04))
    inverse = np.linalg.inv(truth)
    source = (
        target @ inverse[:3, :3].T + inverse[:3, 3]
    )
    schedule = RegistrationSchedule(
        voxel_ratios=(0.08, 0.04, 0.02),
        correspondence_ratios=(0.30, 0.16, 0.08),
        iterations=(50, 35, 25),
    )
    result = multiscale_registration(
        source, colors, target, colors,
        depth_scale=1.0, method=method, schedule=schedule)
    assert result["status"] == "ok"
    assert result["fitness"] > 0.85
    np.testing.assert_allclose(
        np.asarray(result["transform"]), truth,
        atol=0.035 if method == "geometric" else 0.025)

