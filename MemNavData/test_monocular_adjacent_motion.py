import math

import numpy as np
import pytest

from MemNavData.lingbot_colored_registration import (
    matrix_to_quaternion_xyzw,
)
from MemNavData.monocular_adjacent_motion import (
    PlanarMotionReceipt,
    estimate_adjacent_motion,
    integrate_planar_motion,
    invert_planar_motion,
    local_motion_validity,
    pad_intrinsic_pose9,
    planar_motion_from_pnp_pose,
)


def test_padded_intrinsic_pose_preserves_centered_camera():
    intrinsic = np.asarray([
        [355.81464, 0.0, 240.0],
        [0.0, 351.687, 135.0],
        [0.0, 0.0, 1.0],
    ])
    padded, pose = pad_intrinsic_pose9(
        intrinsic,
        raw_height=270,
        raw_width=480,
        target_height=518,
        target_width=518,
    )
    np.testing.assert_allclose(padded[[0, 1], [2, 2]], [259.0, 259.0])
    assert 0.0 < pose[7] < math.pi
    assert 0.0 < pose[8] < math.pi
    assert pose[6] == pytest.approx(1.0)


def test_pnp_pose_reduces_to_navdp_forward_left_and_yaw():
    yaw = math.radians(30.0)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    # Raw LingBot basis that lingbot_relative_yaw maps to +30 degrees.
    basis = np.diag([-1.0, -1.0, 1.0])
    corrected = np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])
    raw_rotation = basis.T @ corrected @ basis
    pose = np.asarray([
        -0.25, 0.04, 0.80,
        *matrix_to_quaternion_xyzw(raw_rotation),
        1.0, 1.0,
    ])
    motion = planar_motion_from_pnp_pose(pose)
    assert motion.forward_m == pytest.approx(0.80)
    assert motion.left_m == pytest.approx(0.25)
    assert motion.vertical_m == pytest.approx(0.04)
    assert motion.yaw_rad == pytest.approx(yaw)


def test_planar_receipts_compose_in_previous_body_frame():
    first = PlanarMotionReceipt(
        forward_m=1.0, left_m=0.0,
        yaw_rad=math.pi / 2.0, vertical_m=0.0)
    position, yaw = integrate_planar_motion([0.0, 0.0], 0.0, first)
    np.testing.assert_allclose(position, [1.0, 0.0], atol=1e-12)
    assert yaw == pytest.approx(math.pi / 2.0)

    second = PlanarMotionReceipt(
        forward_m=1.0, left_m=0.0,
        yaw_rad=0.0, vertical_m=0.0)
    position, yaw = integrate_planar_motion(position, yaw, second)
    np.testing.assert_allclose(position, [1.0, 1.0], atol=1e-12)
    assert yaw == pytest.approx(math.pi / 2.0)


def test_off_center_intrinsic_is_rejected_by_pose9_interface():
    intrinsic = np.asarray([
        [300.0, 0.0, 10.0],
        [0.0, 300.0, 10.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValueError, match="centered principal point"):
        pad_intrinsic_pose9(
            intrinsic,
            raw_height=270,
            raw_width=480,
            target_height=518,
            target_width=518,
        )


def test_local_motion_validity_omits_open_set_hull_coverage():
    result = local_motion_validity({
        "status": "ok",
        "inliers": 20,
        "reprojection_rmse_px": 1.0,
        "query_inlier_coverage": 0.001,
        "reference_inlier_coverage": 0.001,
        "pose9": [0.0] * 9,
    })
    assert result["accepted"] is True
    assert result["identity_or_role_inference"] is False


def test_local_motion_validity_rejects_weak_pnp_pose():
    result = local_motion_validity({
        "status": "insufficient_inliers",
        "inliers": 6,
        "reprojection_rmse_px": 1.0,
        "pose9": [0.0] * 9,
    })
    assert result["accepted"] is False
    assert "status_ok" in result["failed_checks"]


def test_planar_motion_inverse_composes_to_identity():
    motion = PlanarMotionReceipt(
        forward_m=0.8,
        left_m=-0.3,
        yaw_rad=math.radians(37.0),
        vertical_m=0.02,
    )
    position, yaw = integrate_planar_motion([0.0, 0.0], 0.0, motion)
    position, yaw = integrate_planar_motion(
        position, yaw, invert_planar_motion(motion))
    np.testing.assert_allclose(position, [0.0, 0.0], atol=1e-12)
    assert yaw == pytest.approx(0.0, abs=1e-12)


def test_unfiltered_pnp_shadow_cannot_change_primary_validity(
        tmp_path, monkeypatch):
    reference = tmp_path / "reference.jpg"
    query = tmp_path / "query.jpg"
    reference.write_bytes(b"reference")
    query.write_bytes(b"query")

    class Matcher:
        def match_paths(self, *_args, **_kwargs):
            points = np.stack([
                np.linspace(20.0, 80.0, 20),
                np.linspace(25.0, 75.0, 20),
            ], axis=1)
            return {
                "reference_points": points,
                "query_points": points + 1.0,
                "scores": np.ones(20),
                "reference_keypoints": 20,
                "query_keypoints": 20,
            }

    calls = []

    def fake_pnp(*_args, epipolar_threshold_px, **_kwargs):
        calls.append(epipolar_threshold_px)
        if epipolar_threshold_px is not None:
            return {
                "status": "epipolar_degenerate",
                "inliers": 0,
                "reprojection_rmse_px": None,
            }
        return {
            "status": "ok",
            "inliers": 20,
            "reprojection_rmse_px": 1.0,
            "pose9": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }

    monkeypatch.setattr(
        "MemNavData.monocular_adjacent_motion.correspondence_pnp_localize",
        fake_pnp,
    )
    result = estimate_adjacent_motion(
        reference_path=reference,
        query_path=query,
        reference_metric_depth=np.ones((100, 100)),
        raw_camera_intrinsic=np.asarray([
            [50.0, 0.0, 50.0],
            [0.0, 50.0, 50.0],
            [0.0, 0.0, 1.0],
        ]),
        matcher=Matcher(),
        raw_height=100,
        raw_width=100,
        run_unfiltered_pnp_shadow=True,
    )
    assert calls == [1.5, None]
    assert result["local_motion_validity"]["accepted"] is False
    assert result["motion"] is None
    shadow = result["unfiltered_pnp_shadow"]
    assert shadow["authority"] == "diagnostic_only_no_control_effect"
    assert shadow["local_motion_validity"]["accepted"] is True
    assert shadow["motion"]["translation_m"] == pytest.approx(0.2)


def test_direct_pnp_primary_rejects_redundant_unfiltered_shadow(tmp_path):
    reference = tmp_path / "reference.jpg"
    query = tmp_path / "query.jpg"
    reference.write_bytes(b"reference")
    query.write_bytes(b"query")
    with pytest.raises(ValueError, match="shadow is redundant"):
        estimate_adjacent_motion(
            reference_path=reference,
            query_path=query,
            reference_metric_depth=np.ones((8, 8)),
            raw_camera_intrinsic=np.eye(3),
            matcher=object(),
            raw_height=8,
            raw_width=8,
            epipolar_threshold_px=None,
            run_unfiltered_pnp_shadow=True,
        )
