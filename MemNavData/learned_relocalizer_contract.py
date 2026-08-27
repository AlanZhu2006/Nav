"""Fail-closed interface for learned goal-to-history relocalization.

The learned model is deliberately kept outside the navigation policy.  It
receives one causal history reference image and the current image goal and may
predict the query camera relative to the reference camera.  This module turns
that prediction into the same scale-free ``[forward, left]`` interface already
consumed by Certified Episodic Compass.

Coordinate convention
---------------------
``rotation_reference_to_query`` and ``translation_reference_to_query`` obey

    X_query = R_reference_to_query @ X_reference + t_reference_to_query

where camera coordinates use +x right, +y down and +z forward.  Both the
reference and current ``pose9`` values are camera-to-world poses in the
LingBot history coordinate system.  A metric learned translation is divided
by ``meters_per_history_unit`` before it is composed with those poses.

Ground-truth role, co-visibility and simulator pose are intentionally absent
from this deployment boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from MemNavData.certified_relocalization_runtime import _quat_xyzw_to_matrix


LEARNED_RELOCALIZER_SCHEMA_VERSION = 1
TRANSLATION_CONVENTION = "x_query=R_reference_to_query*x_reference+t"


def _finite_scalar(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _proper_rotation(value: Any) -> np.ndarray:
    rotation = _finite_array(
        value, (3, 3), "rotation_reference_to_query")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("rotation_reference_to_query must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3):
        raise ValueError("rotation_reference_to_query must have determinant +1")
    return rotation


def _camera_to_world(pose9: Sequence[float]) -> np.ndarray:
    pose = _finite_array(pose9, (9,), "pose9")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_xyzw_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


@dataclass(frozen=True)
class LearnedPairPrediction:
    """Model-neutral pairwise output recorded by every shadow adapter."""

    model_id: str
    status: str
    rotation_reference_to_query: Sequence[Sequence[float]] | None
    translation_reference_to_query_m: Sequence[float] | None
    support_score: float | None
    solver_support: float | None
    latency_ms: float
    reason: str

    def validated(self) -> "LearnedPairPrediction":
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.status not in {"ok", "abstain", "error"}:
            raise ValueError("status must be ok, abstain or error")
        if not _finite_scalar(self.latency_ms) or float(self.latency_ms) < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if not self.reason:
            raise ValueError("reason cannot be empty")
        for name, value in (("support_score", self.support_score),
                            ("solver_support", self.solver_support)):
            if value is not None and not _finite_scalar(value):
                raise ValueError(f"{name} must be finite when present")
        if self.status == "ok":
            _proper_rotation(self.rotation_reference_to_query)
            translation = _finite_array(
                self.translation_reference_to_query_m,
                (3,), "translation_reference_to_query_m")
            if float(np.linalg.norm(translation)) <= 1e-9:
                raise ValueError("an accepted relative translation cannot be zero")
        elif (self.rotation_reference_to_query is not None
              or self.translation_reference_to_query_m is not None):
            raise ValueError("abstain/error predictions must not expose a pose")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "schema_version": LEARNED_RELOCALIZER_SCHEMA_VERSION,
            "model_id": self.model_id,
            "status": self.status,
            "rotation_reference_to_query": (
                None if self.rotation_reference_to_query is None
                else np.asarray(
                    self.rotation_reference_to_query,
                    dtype=np.float64).tolist()),
            "translation_reference_to_query_m": (
                None if self.translation_reference_to_query_m is None
                else np.asarray(
                    self.translation_reference_to_query_m,
                    dtype=np.float64).tolist()),
            "translation_convention": TRANSLATION_CONVENTION,
            "support_score": self.support_score,
            "solver_support": self.solver_support,
            "latency_ms": float(self.latency_ms),
            "reason": self.reason,
        }


def query_camera_to_world(
    reference_pose9: Sequence[float],
    rotation_reference_to_query: Sequence[Sequence[float]],
    translation_reference_to_query_m: Sequence[float],
    *,
    meters_per_history_unit: float,
) -> np.ndarray:
    """Compose a learned relative pose into the causal history frame."""

    if (not _finite_scalar(meters_per_history_unit)
            or float(meters_per_history_unit) <= 0):
        raise ValueError("meters_per_history_unit must be finite and positive")
    reference_c2w = _camera_to_world(reference_pose9)
    rotation = _proper_rotation(rotation_reference_to_query)
    translation_m = _finite_array(
        translation_reference_to_query_m,
        (3,), "translation_reference_to_query_m")

    reference_to_query = np.eye(4, dtype=np.float64)
    reference_to_query[:3, :3] = rotation
    reference_to_query[:3, 3] = (
        translation_m / float(meters_per_history_unit))
    query_c2w = reference_c2w @ np.linalg.inv(reference_to_query)
    if not np.isfinite(query_c2w).all():
        raise ValueError("relative pose composition produced non-finite values")
    return query_c2w


def scale_free_bearing_from_pairwise(
    current_pose9: Sequence[float],
    reference_pose9: Sequence[float],
    prediction: LearnedPairPrediction,
    *,
    meters_per_history_unit: float,
) -> list[float]:
    """Return NavDP ``[forward, left]`` only for one valid learned pose."""

    prediction.validated()
    if prediction.status != "ok":
        raise ValueError("only an ok learned prediction can produce a bearing")
    current_c2w = _camera_to_world(current_pose9)
    query_c2w = query_camera_to_world(
        reference_pose9,
        prediction.rotation_reference_to_query,
        prediction.translation_reference_to_query_m,
        meters_per_history_unit=meters_per_history_unit)
    relative = current_c2w[:3, :3].T @ (
        query_c2w[:3, 3] - current_c2w[:3, 3])
    bearing = np.asarray([relative[2], -relative[0]], dtype=np.float64)
    if not np.isfinite(bearing).all() or float(np.linalg.norm(bearing)) <= 1e-9:
        raise ValueError("learned pose does not define a usable planar bearing")
    return bearing.tolist()


def shadow_contract() -> Mapping[str, Any]:
    """Immutable experiment semantics shared by all candidate adapters."""

    return {
        "schema_version": LEARNED_RELOCALIZER_SCHEMA_VERSION,
        "input": "goal_rgb+causal_history_reference_rgb+intrinsics",
        "relative_pose_convention": TRANSLATION_CONVENTION,
        "output": "relative_pose+support+solver_support+latency",
        "navigation_interface": "scale_free_[forward,left]",
        "runtime_role_labels": False,
        "runtime_ground_truth": False,
        "shadow_only": True,
        "candidate_universe": "frozen_dino_top8",
        "fallback_during_shadow": "unchanged_certified_episodic_compass",
    }


__all__ = [
    "LEARNED_RELOCALIZER_SCHEMA_VERSION",
    "LearnedPairPrediction",
    "TRANSLATION_CONVENTION",
    "query_camera_to_world",
    "scale_free_bearing_from_pairwise",
    "shadow_contract",
]
