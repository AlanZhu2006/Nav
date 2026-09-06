"""Height-calibrated monocular motion between adjacent RGB observations.

The frozen LingBot depth head supplies relative depth from one causal RGB
stream.  A camera-height receipt metricizes that depth.  Local image
correspondences then estimate the next camera pose in the previous camera
frame with PnP.  Only the planar ``(forward, left, yaw)`` increment leaves this
module; no global pose, simulator state, or wheel odometry is required.

This module deliberately contains no navigation policy and no success logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from MemNavData.certified_relocalization_runtime import (
    CERTIFIED_EPIPOLAR_THRESHOLD_PX,
    certificate_decision,
)
from MemNavData.lingbot_colored_registration import (
    quaternion_xyzw_to_matrix,
)
from MemNavData.lingbot_pnp_localization import (
    SiftPnPConfig,
    correspondence_pnp_localize,
    map_raw_intrinsic_to_lingbot_pad,
)
from NavDP.baselines.memnav.pose_alignment import lingbot_relative_yaw


MONOCULAR_ADJACENT_MOTION_SCHEMA_VERSION = (
    "height_calibrated_monocular_adjacent_motion_v2_20260903"
)

LOCAL_MOTION_MIN_PNP_INLIERS = 16
LOCAL_MOTION_MAX_REPROJECTION_RMSE_PX = 2.0


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def pad_intrinsic_pose9(
    raw_intrinsic: Sequence[Sequence[float]],
    *,
    raw_height: int,
    raw_width: int,
    target_height: int,
    target_width: int,
    patch_size: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded camera intrinsics and an identity LingBot pose9.

    ``correspondence_pnp_localize`` lifts reference pixels with the pose9 FOV.
    The identity pose makes the PnP output a camera-to-previous-camera
    transform.  The known calibrated intrinsics are mapped through the same
    LingBot resize-and-pad transform as the matched keypoints.
    """

    raw = _finite_array(raw_intrinsic, (3, 3), "raw_intrinsic")
    padded = map_raw_intrinsic_to_lingbot_pad(
        raw,
        raw_height=int(raw_height),
        raw_width=int(raw_width),
        target_height=int(target_height),
        target_width=int(target_width),
        patch_size=int(patch_size),
    )
    expected_center = np.asarray(
        [float(target_width) / 2.0, float(target_height) / 2.0])
    if not np.allclose(
            padded[[0, 1], [2, 2]], expected_center, atol=1.0):
        raise ValueError(
            "the current PnP pose9 interface requires a centered principal "
            "point after LingBot padding")
    fy = float(padded[1, 1])
    fx = float(padded[0, 0])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("padded camera focal lengths must be positive")
    fov_height = 2.0 * math.atan(float(target_height) / (2.0 * fy))
    fov_width = 2.0 * math.atan(float(target_width) / (2.0 * fx))
    pose9 = np.asarray([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        fov_height, fov_width,
    ], dtype=np.float64)
    return padded, pose9


@dataclass(frozen=True)
class PlanarMotionReceipt:
    forward_m: float
    left_m: float
    yaw_rad: float
    vertical_m: float

    @property
    def translation_m(self) -> float:
        return float(math.hypot(self.forward_m, self.left_m))

    def audit_dict(self) -> dict[str, float]:
        return {
            "forward_m": float(self.forward_m),
            "left_m": float(self.left_m),
            "yaw_rad": float(self.yaw_rad),
            "translation_m": self.translation_m,
            "vertical_m": float(self.vertical_m),
        }


def planar_motion_from_pnp_pose(
    pose9: Sequence[float],
) -> PlanarMotionReceipt:
    """Reduce a camera-to-previous-camera PnP pose to NavDP planar axes."""

    pose = _finite_array(pose9, (9,), "pose9")
    rotation = quaternion_xyzw_to_matrix(pose[3:7])
    # LingBot/OpenCV camera coordinates: +z forward, +x right, +y down.
    forward = float(pose[2])
    left = float(-pose[0])
    yaw = _wrap_angle(lingbot_relative_yaw(rotation))
    return PlanarMotionReceipt(
        forward_m=forward,
        left_m=left,
        yaw_rad=yaw,
        vertical_m=float(pose[1]),
    )


def integrate_planar_motion(
    position_forward_left: Sequence[float],
    yaw_rad: float,
    motion: PlanarMotionReceipt,
) -> tuple[np.ndarray, float]:
    """Compose one previous-body planar increment into its initial frame."""

    position = _finite_array(
        position_forward_left, (2,), "position_forward_left")
    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("yaw_rad must be finite")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    displacement = np.asarray([
        cosine * motion.forward_m - sine * motion.left_m,
        sine * motion.forward_m + cosine * motion.left_m,
    ], dtype=np.float64)
    return position + displacement, _wrap_angle(yaw + motion.yaw_rad)


def invert_planar_motion(motion: PlanarMotionReceipt) -> PlanarMotionReceipt:
    """Invert one planar previous-camera to current-camera transform."""

    inverse_yaw = _wrap_angle(-float(motion.yaw_rad))
    cosine, sine = math.cos(inverse_yaw), math.sin(inverse_yaw)
    forward = -(
        cosine * float(motion.forward_m)
        - sine * float(motion.left_m)
    )
    left = -(
        sine * float(motion.forward_m)
        + cosine * float(motion.left_m)
    )
    return PlanarMotionReceipt(
        forward_m=forward,
        left_m=left,
        yaw_rad=inverse_yaw,
        vertical_m=-float(motion.vertical_m),
    )


def local_motion_validity(pnp: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an adjacent pose without open-set identity coverage checks.

    Sparse CEC must prove that independently retrieved images depict the same
    place, so it requires distributed hull coverage. Consecutive RGB frames
    already have a causal adjacency relation. Their local-motion receipt keeps
    the PnP status, inlier, and reprojection checks, but does not reinterpret
    corridor-localized support as an image-identity test.
    """

    status_ok = str(pnp.get("status", "")) == "ok"
    inliers = int(pnp.get("inliers", 0))
    rmse_raw = pnp.get("reprojection_rmse_px")
    rmse = None if rmse_raw is None else float(rmse_raw)
    checks = {
        "status_ok": status_ok,
        "minimum_inliers": inliers >= LOCAL_MOTION_MIN_PNP_INLIERS,
        "maximum_reprojection_rmse": (
            rmse is not None
            and math.isfinite(rmse)
            and rmse <= LOCAL_MOTION_MAX_REPROJECTION_RMSE_PX
        ),
        "pose_available": "pose9" in pnp,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": not failed,
        "checks": checks,
        "failed_checks": failed,
        "reason": "local_motion_valid" if not failed else failed[0],
        "thresholds": {
            "min_pnp_inliers": LOCAL_MOTION_MIN_PNP_INLIERS,
            "max_reprojection_rmse_px": (
                LOCAL_MOTION_MAX_REPROJECTION_RMSE_PX),
        },
        "identity_or_role_inference": False,
    }


def estimate_adjacent_motion(
    *,
    reference_path: Path,
    query_path: Path,
    reference_metric_depth: np.ndarray,
    raw_camera_intrinsic: Sequence[Sequence[float]],
    matcher: Any,
    raw_height: int,
    raw_width: int,
    patch_size: int = 14,
    depth_transport_clip_m: float | None = None,
    config: SiftPnPConfig | None = None,
    epipolar_threshold_px: float | None = (
        CERTIFIED_EPIPOLAR_THRESHOLD_PX),
    run_unfiltered_pnp_shadow: bool = False,
) -> dict[str, Any]:
    """Estimate one adjacent-frame local motion and its strict certificate.

    The depth is already metricized by the immutable camera-height receipt.
    ``depth_transport_clip_m`` is used only when an offline wire format has
    saturated far depth; clipped pixels are marked invalid rather than treated
    as a false plane.  The default preserves the historical Fundamental-MAGSAC
    pre-filter.  Passing ``None`` uses PnP-RANSAC directly, which is a distinct
    causal-adjacency motion model rather than a relaxed open-set certificate.

    ``run_unfiltered_pnp_shadow`` is diagnostic-only.  It solves direct PnP on
    the exact same correspondences and depth after the primary solve, but its
    result never changes the returned primary motion or validity decision.
    """

    reference_path = Path(reference_path)
    query_path = Path(query_path)
    if not reference_path.is_file() or not query_path.is_file():
        raise FileNotFoundError(
            reference_path if not reference_path.is_file() else query_path)
    depth = np.asarray(reference_metric_depth, dtype=np.float64)
    if (depth.ndim != 2 or not np.isfinite(depth).all()
            or np.any(depth < 0.0)):
        raise ValueError("reference_metric_depth must be finite non-negative 2-D")
    target_height, target_width = (int(value) for value in depth.shape)
    if epipolar_threshold_px is not None:
        epipolar_threshold_px = float(epipolar_threshold_px)
        if (not math.isfinite(epipolar_threshold_px)
                or epipolar_threshold_px <= 0.0):
            raise ValueError("epipolar_threshold_px must be positive or None")
    if not isinstance(run_unfiltered_pnp_shadow, bool):
        raise ValueError("run_unfiltered_pnp_shadow must be boolean")
    if run_unfiltered_pnp_shadow and epipolar_threshold_px is None:
        raise ValueError(
            "unfiltered shadow is redundant when primary PnP is unfiltered")
    padded_intrinsic, identity_pose9 = pad_intrinsic_pose9(
        raw_camera_intrinsic,
        raw_height=int(raw_height),
        raw_width=int(raw_width),
        target_height=target_height,
        target_width=target_width,
        patch_size=int(patch_size),
    )
    matched = matcher.match_paths(
        reference_path,
        query_path,
        target_height=target_height,
        target_width=target_width,
        patch_size=int(patch_size),
    )
    confidence = np.ones_like(depth, dtype=np.float64)
    valid_depth = depth > 1e-6
    if depth_transport_clip_m is not None:
        clip = float(depth_transport_clip_m)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("depth_transport_clip_m must be positive")
        valid_depth &= depth < clip - 1e-4
    confidence[~valid_depth] = -1.0
    pnp = correspondence_pnp_localize(
        matched["reference_points"],
        matched["query_points"],
        depth,
        confidence,
        identity_pose9,
        config=SiftPnPConfig() if config is None else config,
        match_scores=matched["scores"],
        epipolar_threshold_px=epipolar_threshold_px,
        query_intrinsic=padded_intrinsic,
    )
    certificate = certificate_decision(pnp)
    motion_validity = local_motion_validity(pnp)
    motion = (
        None if "pose9" not in pnp
        else planar_motion_from_pnp_pose(pnp["pose9"])
    )
    shadow = None
    if run_unfiltered_pnp_shadow:
        shadow_pnp = correspondence_pnp_localize(
            matched["reference_points"],
            matched["query_points"],
            depth,
            confidence,
            identity_pose9,
            config=SiftPnPConfig() if config is None else config,
            match_scores=matched["scores"],
            epipolar_threshold_px=None,
            query_intrinsic=padded_intrinsic,
        )
        shadow_validity = local_motion_validity(shadow_pnp)
        shadow_motion = (
            None if "pose9" not in shadow_pnp
            else planar_motion_from_pnp_pose(shadow_pnp["pose9"])
        )
        shadow = {
            "authority": "diagnostic_only_no_control_effect",
            "motion_model": "direct_depth_pnp_ransac",
            "pnp": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in shadow_pnp.items()
            },
            "local_motion_validity": shadow_validity,
            "motion": (
                None if shadow_motion is None
                else shadow_motion.audit_dict()
            ),
        }
    return {
        "schema_version": MONOCULAR_ADJACENT_MOTION_SCHEMA_VERSION,
        "reference_rgb": str(reference_path),
        "query_rgb": str(query_path),
        "metric_depth_sensor_consumed": False,
        "global_pose_consumed": False,
        "wheel_odometry_consumed": False,
        "camera_height_scale_required": True,
        "primary_motion_model": (
            "fundamental_magsac_then_depth_pnp_ransac"
            if epipolar_threshold_px is not None
            else "direct_depth_pnp_ransac"
        ),
        "epipolar_threshold_px": epipolar_threshold_px,
        "unfiltered_pnp_shadow_enabled": run_unfiltered_pnp_shadow,
        "matches": int(len(matched["reference_points"])),
        "reference_keypoints": int(matched["reference_keypoints"]),
        "query_keypoints": int(matched["query_keypoints"]),
        "valid_depth_fraction": float(np.mean(valid_depth)),
        "pnp": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in pnp.items()
        },
        "certificate": certificate,
        "local_motion_validity": motion_validity,
        "motion": None if motion is None else motion.audit_dict(),
        "unfiltered_pnp_shadow": shadow,
    }
