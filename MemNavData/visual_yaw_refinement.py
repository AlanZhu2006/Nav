"""Direct two-view yaw residual for terminal ImageGoal verification.

LingBot supplies the coarse long-gap rotation.  Once the current view overlaps
the goal image, this module estimates the remaining signed yaw directly from
local image geometry, without reading or mutating LingBot memory.

OpenCV is imported lazily because this is evaluator-side optional logic.  The
Habitat environment used by this repository provides SIFT and calibrated
essential-matrix estimation (verified with OpenCV 4.11.0).
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VisualYawEstimate:
    yaw_correction_rad: Optional[float]
    bearing_correction_rad: Optional[float]
    matches: int
    inliers: int
    inlier_ratio: float
    off_axis_deg: Optional[float]
    bearing_mad_deg: Optional[float]
    consensus_error_deg: Optional[float]
    reliable: bool
    reason: str

    @property
    def yaw_correction_deg(self):
        return (None if self.yaw_correction_rad is None
                else math.degrees(self.yaw_correction_rad))

    @property
    def bearing_correction_deg(self):
        return (None if self.bearing_correction_rad is None
                else math.degrees(self.bearing_correction_rad))

    def to_dict(self):
        result = asdict(self)
        result["yaw_correction_deg"] = self.yaw_correction_deg
        result["bearing_correction_deg"] = self.bearing_correction_deg
        return result


def visual_yaw_action_decision(
        estimate, *, deadband_deg=8.0, max_correction_deg=45.0):
    """Return whether a visual estimate is safe and useful for one action."""
    if estimate is None:
        return False, "visual yaw unavailable"
    if not estimate.reliable:
        return False, f"visual yaw rejected: {estimate.reason}"
    correction_deg = estimate.yaw_correction_deg
    if correction_deg is None or not math.isfinite(correction_deg):
        return False, "visual yaw unavailable"
    if abs(correction_deg) <= deadband_deg:
        return False, "visual yaw within deadband"
    if abs(correction_deg) > max_correction_deg:
        return False, "visual yaw exceeds correction bound"
    return True, "ok"


def yaw_correction_from_rotation(rotation):
    """OpenCV current-camera -> goal-camera rotation to Habitat yaw correction.

    Synthetic Habitat renders at known +/-10, 20, and 30 degree offsets verify
    this sign: positive means increase the evaluator's Habitat yaw.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    return float(math.atan2(rotation[0, 2], rotation[2, 2]))


def _failure(reason, matches=0, inliers=0, yaw=None, bearing=None,
             off_axis=None, bearing_mad=None, consensus=None):
    ratio = inliers / matches if matches else 0.0
    return VisualYawEstimate(
        yaw_correction_rad=yaw,
        bearing_correction_rad=bearing,
        matches=int(matches),
        inliers=int(inliers),
        inlier_ratio=float(ratio),
        off_axis_deg=off_axis,
        bearing_mad_deg=bearing_mad,
        consensus_error_deg=consensus,
        reliable=False,
        reason=reason,
    )


def _essential_candidates(essential):
    essential = np.asarray(essential, dtype=np.float64)
    if essential.shape == (3, 3):
        return [essential]
    if essential.ndim == 2 and essential.shape[1] == 3 \
            and essential.shape[0] % 3 == 0:
        return [essential[i:i + 3] for i in range(0, essential.shape[0], 3)]
    return []


def estimate_visual_yaw(
        current_jpg,
        goal_jpg,
        camera_intrinsic,
        *,
        nfeatures=4000,
        ratio_threshold=0.75,
        ransac_threshold_px=1.5,
        min_matches=8,
        min_inliers=20,
        min_inlier_ratio=0.60,
        max_off_axis_deg=15.0,
        max_consensus_error_deg=5.0):
    """Estimate signed goal yaw relative to the current JPEG image.

    A value is returned even when the confidence gates reject it, allowing
    diagnostics to distinguish missing overlap from an off-axis solution.
    Callers must use ``estimate.reliable`` before controlling the robot.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency preflight path
        return _failure(f"opencv unavailable: {exc}")

    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("camera_intrinsic must be a finite 3x3 matrix")

    def decode(payload):
        encoded = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

    current = decode(current_jpg)
    goal = decode(goal_jpg)
    if current is None or goal is None:
        return _failure("jpeg decode failed")

    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    current_keypoints, current_desc = sift.detectAndCompute(current, None)
    goal_keypoints, goal_desc = sift.detectAndCompute(goal, None)
    if current_desc is None or goal_desc is None:
        return _failure("insufficient image features")

    pairs = cv2.BFMatcher().knnMatch(current_desc, goal_desc, k=2)
    matches = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
            matches.append(pair[0])
    if len(matches) < min_matches:
        return _failure("too few ratio-test matches", matches=len(matches))

    current_points = np.float32(
        [current_keypoints[m.queryIdx].pt for m in matches])
    goal_points = np.float32(
        [goal_keypoints[m.trainIdx].pt for m in matches])
    current_bearing = np.arctan(
        (current_points[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0])
    goal_bearing = np.arctan(
        (goal_points[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0])
    bearing_delta = np.arctan2(
        np.sin(goal_bearing - current_bearing),
        np.cos(goal_bearing - current_bearing),
    )
    bearing = float(np.median(bearing_delta))
    bearing_mad = float(np.degrees(np.median(np.abs(np.arctan2(
        np.sin(bearing_delta - bearing), np.cos(bearing_delta - bearing))))))
    essential, ransac_mask = cv2.findEssentialMat(
        current_points,
        goal_points,
        intrinsic,
        cv2.RANSAC,
        0.999,
        float(ransac_threshold_px),
    )
    candidates = _essential_candidates(essential) if essential is not None else []
    if not candidates:
        return _failure(
            "essential matrix unavailable", matches=len(matches),
            bearing=bearing, bearing_mad=bearing_mad)

    best = None
    for candidate in candidates:
        mask = None if ransac_mask is None else ransac_mask.copy()
        try:
            inliers, rotation, _translation, _pose_mask = cv2.recoverPose(
                candidate, current_points, goal_points, intrinsic, mask=mask)
        except cv2.error:
            continue
        if best is None or int(inliers) > best[0]:
            best = (int(inliers), np.asarray(rotation, dtype=np.float64))
    if best is None:
        return _failure(
            "relative pose recovery failed", matches=len(matches),
            bearing=bearing, bearing_mad=bearing_mad)

    inliers, rotation = best
    yaw = yaw_correction_from_rotation(rotation)
    # For a level camera, the recovered rotation should be almost entirely
    # about optical +Y/-Y. R[1,1] is cos(total off-axis tilt) in that case.
    off_axis = math.degrees(math.acos(float(np.clip(rotation[1, 1], -1.0, 1.0))))
    consensus = math.degrees(abs(math.atan2(
        math.sin(yaw - bearing), math.cos(yaw - bearing))))
    inlier_ratio = inliers / len(matches)
    reasons = []
    if inliers < min_inliers:
        reasons.append("too few pose inliers")
    if inlier_ratio < min_inlier_ratio:
        reasons.append("low pose-inlier ratio")
    if off_axis > max_off_axis_deg:
        reasons.append("off-axis rotation")
    if consensus > max_consensus_error_deg:
        reasons.append("essential/bearing disagreement")

    return VisualYawEstimate(
        yaw_correction_rad=yaw,
        bearing_correction_rad=bearing,
        matches=len(matches),
        inliers=inliers,
        inlier_ratio=float(inlier_ratio),
        off_axis_deg=float(off_axis),
        bearing_mad_deg=bearing_mad,
        consensus_error_deg=float(consensus),
        reliable=not reasons,
        reason="ok" if not reasons else "; ".join(reasons),
    )
