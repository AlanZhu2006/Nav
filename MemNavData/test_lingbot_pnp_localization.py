import math

import cv2
import numpy as np
import pytest

from lingbot_pnp_localization import (
    SiftPnPConfig,
    correspondence_pnp_localize,
    image_to_rgb_u8,
    intrinsics_from_pose9,
    lift_reference_keypoints,
    map_raw_intrinsic_to_lingbot_pad,
    map_raw_points_to_lingbot_pad,
    solve_camera_pose_pnp,
)
from lingbot_colored_registration import pose9_to_matrix


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([0.0, math.sin(yaw / 2.0), 0.0,
                     math.cos(yaw / 2.0)])


def test_intrinsics_decode_lingbot_fov_order():
    pose = np.r_[np.zeros(3), [0.0, 0.0, 0.0, 1.0],
                 math.radians(60.0), math.radians(90.0)]
    intrinsic = intrinsics_from_pose9(pose, 200, 400)
    assert intrinsic[0, 0] == pytest.approx(200.0)
    assert intrinsic[1, 1] == pytest.approx(100.0 / math.tan(math.radians(30.0)))
    np.testing.assert_allclose(intrinsic[:2, 2], [200.0, 100.0])


def test_lift_reference_keypoints_uses_camera_to_world_pose():
    pose = np.r_[[1.0, 2.0, 3.0], yaw_quaternion(math.pi / 2.0), 1.0, 1.0]
    depth = np.full((4, 4), 2.0)
    confidence = np.ones((4, 4))
    world, valid = lift_reference_keypoints(
        np.array([[2.0, 2.0]]), depth, confidence, pose,
        confidence_quantile=0.0)
    assert valid.tolist() == [True]
    # Optical-axis point [0,0,2] rotates by +90 degrees around world y.
    np.testing.assert_allclose(world[0], [3.0, 2.0, 3.0], atol=1e-7)


def test_solve_camera_pose_pnp_recovers_camera_to_world():
    rng = np.random.default_rng(7)
    truth = np.r_[
        [0.4, -0.15, 0.25],
        yaw_quaternion(math.radians(12.0)),
        math.radians(60.0), math.radians(75.0),
    ]
    transform = pose9_to_matrix(truth)
    camera_points = np.column_stack([
        rng.uniform(-1.2, 1.2, 80),
        rng.uniform(-0.8, 0.8, 80),
        rng.uniform(2.5, 6.0, 80),
    ])
    world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    intrinsic = intrinsics_from_pose9(truth, 360, 640)
    image_points = np.column_stack([
        intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2]
        + intrinsic[0, 2],
        intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2]
        + intrinsic[1, 2],
    ])
    image_points += rng.normal(0.0, 0.15, image_points.shape)
    # Add correspondence outliers to ensure this exercises RANSAC.
    image_points[:12] = rng.uniform([0.0, 0.0], [640.0, 360.0], (12, 2))
    result = solve_camera_pose_pnp(
        world_points, image_points, intrinsic,
        config=SiftPnPConfig(min_correspondences=8), fov_pose9=truth)
    assert result["status"] == "ok"
    assert result["inliers"] >= 60
    assert result["reprojection_rmse_px"] < 0.5
    np.testing.assert_allclose(
        pose9_to_matrix(result["pose9"]), transform, atol=0.006)


def test_image_conversion_preserves_rgb_shape():
    chw = np.zeros((3, 5, 7), dtype=np.float32)
    chw[0] = 1.0
    rgb = image_to_rgb_u8(chw)
    assert rgb.shape == (5, 7, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[..., 0] == 255)
    assert np.all(rgb[..., 1:] == 0)


def test_degenerate_epipolar_correspondences_fail_closed():
    points = np.tile([[4.0, 4.0]], (12, 1))
    pose = np.r_[np.zeros(3), [0.0, 0.0, 0.0, 1.0], 1.0, 1.0]
    result = correspondence_pnp_localize(
        points, points, np.ones((8, 8)), np.ones((8, 8)), pose,
        config=SiftPnPConfig(), epipolar_threshold_px=1.5)
    assert result["status"] in {
        "epipolar_degenerate", "epipolar_ransac_failed",
        "insufficient_epipolar_inliers",
    }
    assert result["inliers"] == 0


def test_raw_landscape_points_map_to_lingbot_pad_coordinates():
    points = np.array([[0.0, 0.0], [320.0, 240.0], [640.0, 480.0]])
    mapped = map_raw_points_to_lingbot_pad(
        points, raw_height=480, raw_width=640,
        target_height=518, target_width=518, patch_size=14)
    # 640x480 -> 518x392 plus 63 pixels of vertical padding on each side.
    np.testing.assert_allclose(mapped[0], [0.0, 63.0])
    np.testing.assert_allclose(mapped[1], [259.0, 259.0])
    np.testing.assert_allclose(mapped[2], [518.0, 455.0])


def test_raw_intrinsic_uses_same_lingbot_pad_transform_as_points():
    intrinsic = np.array([
        [320.0, 0.0, 320.0],
        [0.0, 300.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    mapped = map_raw_intrinsic_to_lingbot_pad(
        intrinsic, raw_height=480, raw_width=640,
        target_height=518, target_width=518, patch_size=14)
    points = np.array([[320.0, 240.0], [400.0, 300.0]])
    expected = map_raw_points_to_lingbot_pad(
        points, raw_height=480, raw_width=640,
        target_height=518, target_width=518, patch_size=14)
    rays = np.column_stack([points, np.ones(len(points))])
    normalized = rays @ np.linalg.inv(intrinsic).T
    projected = normalized @ mapped.T
    projected = projected[:, :2] / projected[:, 2:]
    np.testing.assert_allclose(projected, expected)


def test_correspondence_pnp_uses_distinct_query_intrinsic():
    rng = np.random.default_rng(17)
    height, width = 240, 320
    reference_pose = np.r_[
        np.zeros(3), [0.0, 0.0, 0.0, 1.0],
        math.radians(60.0), math.radians(70.0),
    ]
    reference_intrinsic = intrinsics_from_pose9(
        reference_pose, height, width)
    query_truth = np.r_[
        [0.35, -0.08, 0.18], yaw_quaternion(math.radians(10.0)),
        math.radians(42.0), math.radians(45.0),
    ]
    query_intrinsic = intrinsics_from_pose9(query_truth, height, width)

    reference_depth = np.full((height, width), 4.0)
    confidence = np.ones_like(reference_depth)
    reference_points = np.column_stack([
        rng.uniform(55.0, width - 55.0, 100),
        rng.uniform(45.0, height - 45.0, 100),
    ])
    z = np.full(len(reference_points), 4.0)
    world_points = np.column_stack([
        (reference_points[:, 0] - reference_intrinsic[0, 2])
        * z / reference_intrinsic[0, 0],
        (reference_points[:, 1] - reference_intrinsic[1, 2])
        * z / reference_intrinsic[1, 1],
        z,
    ])
    query_transform = pose9_to_matrix(query_truth)
    query_camera = (
        world_points - query_transform[:3, 3]
    ) @ query_transform[:3, :3]
    query_points = np.column_stack([
        query_intrinsic[0, 0] * query_camera[:, 0] / query_camera[:, 2]
        + query_intrinsic[0, 2],
        query_intrinsic[1, 1] * query_camera[:, 1] / query_camera[:, 2]
        + query_intrinsic[1, 2],
    ])

    corrected = correspondence_pnp_localize(
        reference_points, query_points, reference_depth, confidence,
        reference_pose, config=SiftPnPConfig(),
        query_intrinsic=query_intrinsic)
    assert corrected["status"] == "ok"
    np.testing.assert_allclose(
        pose9_to_matrix(corrected["pose9"]), query_transform, atol=2e-3)

    shared_intrinsic = correspondence_pnp_localize(
        reference_points, query_points, reference_depth, confidence,
        reference_pose, config=SiftPnPConfig())
    assert shared_intrinsic["status"] != "ok" or (
        np.linalg.norm(
            np.asarray(shared_intrinsic["pose9"][:3]) - query_truth[:3]
        ) > 0.1
    )
