#!/usr/bin/env python3
"""Frozen, deployment-visible contract for open-set goal relocalization.

This module contains only the small deterministic boundary between image
matching and navigation.  DINO proposes a causal history shortlist,
SuperPoint+LightGlue/Fundamental geometry ranks it, and LingBot-depth PnP
produces one pose in LingBot's per-stream scale.  The complete v2 geometry
certificate may authorize that pose as a *bearing source*; it does not certify
the monocular metric scale.  Every malformed or unsupported case abstains.

Ground truth, co-visibility labels, and semantic Novel/Revisit labels are
deliberately absent from this module.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


CERTIFIED_RELOCALIZATION_SCHEMA_VERSION = 3
CERTIFIED_GEOMETRY_CERTIFICATE_VERSION = 2
CERTIFIED_CANDIDATE_TOP_K = 8
CERTIFIED_CANDIDATE_MIN_GAP = 4
CERTIFIED_MINIMUM_ANCHOR = 8
CERTIFIED_EPIPOLAR_THRESHOLD_PX = 1.5
CERTIFICATE_MIN_INLIERS = 16
CERTIFICATE_MIN_QUERY_COVERAGE = 0.05
CERTIFICATE_MIN_REFERENCE_COVERAGE = 0.05
CERTIFICATE_MAX_REPROJECTION_RMSE_PX = 2.0
STRICT_AUTHORITY_POLICY = "strict_certificate"
UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY = "pnp_pose_available"
SUPPORTED_AUTHORITY_POLICIES = (
    STRICT_AUTHORITY_POLICY,
    UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
)


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a cam-to-world rotation for one finite XYZW quaternion."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("pose quaternion must be finite XYZW")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("pose quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def scale_free_relative_xy(
    current_pose9: Sequence[float],
    goal_pose9: Sequence[float],
) -> list[float]:
    """Express goal translation as NavDP ``[forward, left]`` without scale.

    LingBot pose9 stores translation followed by an XYZW cam-to-world
    quaternion.  ``R_cur.T @ (t_goal - t_cur)`` therefore gives translation
    in the current camera frame.  MemNav's audited axis conversion is
    ``[camera_z, -camera_x]``.  Its norm remains in arbitrary LingBot units;
    only its direction is deployment-visible.
    """

    current = np.asarray(current_pose9, dtype=np.float64)
    goal = np.asarray(goal_pose9, dtype=np.float64)
    if (current.shape != (9,) or goal.shape != (9,)
            or not np.isfinite(current).all()
            or not np.isfinite(goal).all()):
        raise ValueError("current and goal must be finite pose9 vectors")
    rotation = _quat_xyzw_to_matrix(current[3:7])
    relative = rotation.T @ (goal[:3] - current[:3])
    point = np.asarray([relative[2], -relative[0]], dtype=np.float64)
    if not np.isfinite(point).all():
        raise ValueError("pose pair produced a non-finite bearing vector")
    return point.tolist()


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _hull_coverage(points: np.ndarray, height: int, width: int) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull) / max(float(height * width), 1.0))


def _grid_coverage(points: np.ndarray, height: int, width: int,
                   grid_size: int = 4) -> float:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return 0.0
    x = np.clip(
        (points[:, 0] / width * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip(
        (points[:, 1] / height * grid_size).astype(int), 0, grid_size - 1)
    return float(len(set(zip(x.tolist(), y.tolist()))) / (grid_size ** 2))


def fundamental_support(
    reference_points: np.ndarray,
    query_points: np.ndarray,
    scores: np.ndarray,
    reference_shape: tuple[int, int],
    query_shape: tuple[int, int],
    *,
    threshold_px: float = CERTIFIED_EPIPOLAR_THRESHOLD_PX,
) -> dict[str, Any]:
    """Return label-free two-view evidence used by the frozen rank order."""

    reference = np.asarray(reference_points, dtype=np.float32)
    query = np.asarray(query_points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[1:] != (2,)
            or query.shape != reference.shape
            or scores.shape != (len(reference),)):
        raise ValueError("aligned reference/query points and scores are required")
    if threshold_px <= 0.0:
        raise ValueError("epipolar threshold must be positive")
    count = len(reference)
    result = {
        "lightglue_matches": int(count),
        "lightglue_score_median": (
            float(np.median(scores)) if count else 0.0),
        "fundamental_inliers": 0,
        "fundamental_inlier_ratio": 0.0,
        "fundamental_query_grid_coverage": 0.0,
        "fundamental_query_hull_coverage": 0.0,
        "fundamental_reference_grid_coverage": 0.0,
        "fundamental_reference_hull_coverage": 0.0,
    }
    if count < 8:
        return result
    cv2.setRNGSeed(0)
    try:
        _fundamental, mask = cv2.findFundamentalMat(
            reference, query, cv2.USAC_MAGSAC,
            float(threshold_px), 0.999, 10000)
    except cv2.error:
        mask = None
    if mask is None:
        return result
    inliers = np.asarray(mask).reshape(-1).astype(bool)
    if len(inliers) != count:
        return result
    ref_height, ref_width = map(int, reference_shape)
    query_height, query_width = map(int, query_shape)
    result.update({
        "fundamental_inliers": int(inliers.sum()),
        "fundamental_inlier_ratio": float(inliers.mean()),
        "fundamental_query_grid_coverage": _grid_coverage(
            query[inliers], query_height, query_width),
        "fundamental_query_hull_coverage": _hull_coverage(
            query[inliers], query_height, query_width),
        "fundamental_reference_grid_coverage": _grid_coverage(
            reference[inliers], ref_height, ref_width),
        "fundamental_reference_hull_coverage": _hull_coverage(
            reference[inliers], ref_height, ref_width),
    })
    return result


def candidate_rank_key(candidate: Mapping[str, Any]) -> tuple:
    """Frozen label-free lexicographic order from the v2 confirmation."""

    fields = (
        "fundamental_inliers",
        "fundamental_query_grid_coverage",
        "fundamental_query_hull_coverage",
        "lightglue_score_median",
        "dino_cosine",
        "anchor",
    )
    if not all(_finite(candidate.get(field)) for field in fields):
        raise ValueError("candidate ranking evidence must be finite")
    return (
        int(candidate["fundamental_inliers"]),
        float(candidate["fundamental_query_grid_coverage"]),
        float(candidate["fundamental_query_hull_coverage"]),
        float(candidate["lightglue_score_median"]),
        float(candidate["dino_cosine"]),
        -int(candidate["anchor"]),
    )


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Return a stable best-first copy without changing shortlist membership."""

    copied = [dict(candidate) for candidate in candidates]
    return sorted(copied, key=candidate_rank_key, reverse=True)


def certificate_decision(pnp: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the atomic v2 certificate and expose every failed invariant."""

    checks = {
        "status_ok": pnp.get("status") == "ok",
        "minimum_inliers": (
            _finite(pnp.get("inliers"))
            and int(pnp["inliers"]) >= CERTIFICATE_MIN_INLIERS),
        "minimum_query_coverage": (
            _finite(pnp.get("query_inlier_coverage"))
            and float(pnp["query_inlier_coverage"])
            >= CERTIFICATE_MIN_QUERY_COVERAGE),
        "minimum_reference_coverage": (
            _finite(pnp.get("reference_inlier_coverage"))
            and float(pnp["reference_inlier_coverage"])
            >= CERTIFICATE_MIN_REFERENCE_COVERAGE),
        "maximum_reprojection_rmse": (
            _finite(pnp.get("reprojection_rmse_px"))
            and float(pnp["reprojection_rmse_px"])
            <= CERTIFICATE_MAX_REPROJECTION_RMSE_PX),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": CERTIFIED_RELOCALIZATION_SCHEMA_VERSION,
        "accepted": not failed,
        "reason": "certificate_accepted" if not failed else failed[0],
        "failed_checks": failed,
        "checks": checks,
        "thresholds": {
            "min_pnp_inliers": CERTIFICATE_MIN_INLIERS,
            "min_query_inlier_coverage": CERTIFICATE_MIN_QUERY_COVERAGE,
            "min_reference_inlier_coverage": (
                CERTIFICATE_MIN_REFERENCE_COVERAGE),
            "max_reprojection_rmse_px": (
                CERTIFICATE_MAX_REPROJECTION_RMSE_PX),
        },
    }


def _finite_pose9(pnp: Mapping[str, Any]) -> bool:
    """Return whether PnP exposed one finite camera pose witness."""

    try:
        pose = np.asarray(pnp.get("pose9"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(pose.shape == (9,) and np.isfinite(pose).all())


def operational_authority_decision(
        pnp: Mapping[str, Any], *,
        policy: str = STRICT_AUTHORITY_POLICY) -> dict[str, Any]:
    """Separate geometric pose production from intervention authority.

    ``strict_certificate`` is the deployed CEC rule.  The diagnostic
    ``pnp_pose_available`` policy deliberately removes only the operational
    certificate thresholds: retrieval, local matching, geometric ranking,
    LingBot-depth lifting, and PnP remain unchanged.  It therefore measures
    the causal value of the certificate boundary without becoming a
    retrieval-only or geometry-free baseline.
    """

    policy = str(policy)
    if policy not in SUPPORTED_AUTHORITY_POLICIES:
        raise ValueError(f"unsupported authority policy: {policy}")
    certificate = certificate_decision(pnp)
    pose_available = _finite_pose9(pnp)
    if policy == STRICT_AUTHORITY_POLICY:
        accepted = bool(certificate["accepted"] and pose_available)
        if accepted:
            reason = "certificate_accepted"
        elif certificate["accepted"]:
            reason = "pnp_pose_unavailable"
        else:
            reason = str(certificate["reason"])
        thresholds_enforced = True
    else:
        accepted = pose_available
        reason = (
            "pnp_pose_available"
            if accepted else "pnp_pose_unavailable"
        )
        thresholds_enforced = False
    return {
        "policy": policy,
        "accepted": accepted,
        "reason": reason,
        "pnp_pose_available": pose_available,
        "certificate_thresholds_enforced": thresholds_enforced,
        # Always retain the strict decision for paired diagnostics.
        "strict_certificate": certificate,
    }


def fundamental_can_reach_certificate(
        evidence: Mapping[str, Any]) -> tuple[bool, str]:
    """Safe early abstention before the expensive LingBot-depth replay.

    PnP consumes the Fundamental-MAGSAC inlier subset.  Its inlier count and
    spatial support therefore cannot exceed these values.  Failing any of
    the three monotone lower bounds makes the v2 certificate impossible and
    can be rejected without estimating depth.
    """

    requirements = (
        ("fundamental_inliers", CERTIFICATE_MIN_INLIERS),
        ("fundamental_query_hull_coverage",
         CERTIFICATE_MIN_QUERY_COVERAGE),
        ("fundamental_reference_hull_coverage",
         CERTIFICATE_MIN_REFERENCE_COVERAGE),
    )
    for field, threshold in requirements:
        value = evidence.get(field)
        if not _finite(value) or float(value) < threshold:
            return False, f"precheck_{field}"
    return True, "precheck_passed"


def runtime_contract() -> dict[str, Any]:
    """JSON-compatible immutable mechanics advertised by the server."""

    return {
        "schema_version": CERTIFIED_RELOCALIZATION_SCHEMA_VERSION,
        "geometry_certificate_version": (
            CERTIFIED_GEOMETRY_CERTIFICATE_VERSION),
        "candidate_top_k": CERTIFIED_CANDIDATE_TOP_K,
        "candidate_min_gap": CERTIFIED_CANDIDATE_MIN_GAP,
        "minimum_anchor": CERTIFIED_MINIMUM_ANCHOR,
        "candidate_lifecycle": "frozen_at_first_goal_query",
        "empty_candidate_semantics": "cached_native_abstention",
        "ranking": [
            "fundamental_inliers",
            "fundamental_query_grid_coverage",
            "fundamental_query_hull_coverage",
            "lightglue_score_median",
            "dino_cosine",
            "earlier_anchor",
        ],
        "epipolar_threshold_px": CERTIFIED_EPIPOLAR_THRESHOLD_PX,
        "certificate": certificate_decision({})["thresholds"],
        "default_authority_policy": STRICT_AUTHORITY_POLICY,
        "diagnostic_authority_policies": [
            UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
        ],
        "output": "scale_free_relative_bearing",
        "pointgoal_units": "lingbot_raw_direction_only",
        "metric_distance_certified": False,
        "controller_adapter": "verified_bearing_v1_fixed_2.5m",
        "fallback": "native_imagegoal",
        "semantic_claim": "certified_history_bearing_or_unsupported_unknown",
    }


__all__ = [
    "CERTIFICATE_MAX_REPROJECTION_RMSE_PX",
    "CERTIFICATE_MIN_INLIERS",
    "CERTIFICATE_MIN_QUERY_COVERAGE",
    "CERTIFICATE_MIN_REFERENCE_COVERAGE",
    "CERTIFIED_CANDIDATE_MIN_GAP",
    "CERTIFIED_CANDIDATE_TOP_K",
    "CERTIFIED_EPIPOLAR_THRESHOLD_PX",
    "CERTIFIED_MINIMUM_ANCHOR",
    "CERTIFIED_RELOCALIZATION_SCHEMA_VERSION",
    "CERTIFIED_GEOMETRY_CERTIFICATE_VERSION",
    "STRICT_AUTHORITY_POLICY",
    "SUPPORTED_AUTHORITY_POLICIES",
    "UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY",
    "candidate_rank_key",
    "certificate_decision",
    "fundamental_can_reach_certificate",
    "fundamental_support",
    "operational_authority_decision",
    "rank_candidates",
    "runtime_contract",
    "scale_free_relative_xy",
]
