"""Pure contracts for using X-NavDP as a Revisit PointGoal controller.

The module deliberately has no Habitat, Flask, or Torch dependency.  It owns
the frozen source/checkpoint identities, the ground-plane frame conversion,
and fail-closed validation shared by the evaluator and policy server.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


OFFICIAL_XNAVDP_COMMIT = "878740a2011856d0e3782dd6ccd880fd2eccd70f"
OFFICIAL_XNAVDP_POSTTRAIN_SHA256 = (
    "267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24")
FROZEN_BASE_NAVDP_SHA256 = (
    "3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947")

REVISIT_CONTROLLERS = (
    "navdp_mixed",
    "navdp_point",
    "xnavdp_point",
)
XNAVDP_ALGO = "x-navdp-revisit-pointgoal"
XNAVDP_EMBODIMENT = "wheeled"
XNAVDP_MODEL_STATE_TENSOR_COUNT = 1329
XNAVDP_CHECKPOINT_TENSOR_COUNT = 1686
XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT = 357


def _finite_vector(value: Sequence[float], size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(size),) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector of shape ({size},)")
    return array


def pointgoal_payload(aux_pose: Sequence[float]) -> dict[str, list[float]]:
    """Return the HTTP PointGoal payload for local [forward, left]."""

    point = _finite_vector(aux_pose, 2, "aux_pose")
    return {
        "goal_x": [float(point[0])],
        "goal_y": [float(point[1])],
    }


def habitat_pose_to_xnavdp(
    position_hab: Sequence[float], yaw_hab_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Habitat x/y/z + camera yaw to X-NavDP xy-ground coordinates.

    Habitat's local waypoint convention in this repository is
    ``[forward, left]`` with

    ``dx = -forward*sin(psi) - left*cos(psi)`` and
    ``dz = -forward*cos(psi) + left*sin(psi)``.

    Mapping the world plane to ``[x_hab, -z_hab]`` and using
    ``theta = pi/2 + psi`` makes a standard z-axis quaternion rotate X-NavDP
    local ``[x, y]`` into exactly those forward/left basis vectors.
    Quaternions are returned in SciPy-compatible xyzw order, matching the
    official X-NavDP server.
    """

    position = _finite_vector(position_hab, 3, "position_hab")
    yaw = float(yaw_hab_rad)
    if not math.isfinite(yaw):
        raise ValueError("yaw_hab_rad must be finite")
    theta = math.atan2(math.sin(math.pi / 2.0 + yaw),
                       math.cos(math.pi / 2.0 + yaw))
    world_position = np.asarray(
        [position[0], -position[2], 0.0], dtype=np.float64)
    quaternion_xyzw = np.asarray(
        [0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)],
        dtype=np.float64,
    )
    return world_position, quaternion_xyzw


def xnavdp_state_payload(
    position_hab: Sequence[float], yaw_hab_rad: float,
) -> dict[str, list[list[float]]]:
    """Build the batch-one robot-state payload consumed by X-NavDP RTC."""

    position, quaternion = habitat_pose_to_xnavdp(
        position_hab, yaw_hab_rad)
    return {
        "robot_pos": [position.tolist()],
        "robot_quat": [quaternion.tolist()],
    }


def validate_reset_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a server that is not the frozen official X-NavDP controller."""

    _validate_frozen_identity(payload)
    history_count = payload.get("history_frame_count")
    if history_count != [0]:
        raise ValueError(
            "X-NavDP reset did not clear batch-one history: "
            f"{history_count!r}")
    return dict(payload)


def _validate_frozen_identity(payload: Mapping[str, Any]) -> None:
    """Validate fields that must remain constant for every server response."""

    required = {
        "algo": XNAVDP_ALGO,
        "official_commit": OFFICIAL_XNAVDP_COMMIT,
        "checkpoint_sha256": OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        "actor_mode": "posttrain",
        "embodiment": XNAVDP_EMBODIMENT,
    }
    for key, expected in required.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"X-NavDP receipt {key}={actual!r}, expected {expected!r}")
    audit = payload.get("checkpoint_load_audit")
    expected_audit = {
        "audited": True,
        "model_tensor_count": XNAVDP_MODEL_STATE_TENSOR_COUNT,
        "checkpoint_tensor_count": XNAVDP_CHECKPOINT_TENSOR_COUNT,
        "missing_count": 0,
        "unexpected_count": XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
        "shape_mismatch_count": 0,
    }
    if audit != expected_audit:
        raise ValueError(
            "X-NavDP checkpoint/model coverage audit differs from the frozen "
            f"contract: {audit!r}")


def validate_history_receipt(
    payload: Mapping[str, Any], *, expected_frame_count: int | None = None,
) -> dict[str, Any]:
    _validate_frozen_identity(payload)
    if payload.get("diffusion_sampled") is not False:
        raise ValueError("history replay sampled diffusion")
    appended = payload.get("frames_appended")
    if appended != 1:
        raise ValueError(f"history replay appended {appended!r} frames, expected 1")
    _validate_history_frame_count(payload, expected_frame_count)
    return dict(payload)


def _validate_history_frame_count(
    payload: Mapping[str, Any], expected_frame_count: int | None,
) -> None:
    history_count = payload.get("history_frame_count")
    if (not isinstance(history_count, list) or len(history_count) != 1
            or isinstance(history_count[0], bool)
            or not isinstance(history_count[0], (int, np.integer))
            or int(history_count[0]) < 0):
        raise ValueError(
            "X-NavDP history_frame_count must be one non-negative integer")
    if (expected_frame_count is not None
            and int(history_count[0]) != int(expected_frame_count)):
        raise ValueError(
            "X-NavDP cumulative history count is "
            f"{history_count[0]!r}, expected {int(expected_frame_count)}")


def normalize_xnavdp_response(
    payload: Mapping[str, Any], *, expected_seed: int | None,
    expected_history_frame_count: int | None = None,
) -> dict[str, Any]:
    """Validate and squeeze a batch-one X-NavDP trajectory response."""

    _validate_frozen_identity(payload)
    if payload.get("controller") != "xnavdp_point_posttrain":
        raise ValueError("X-NavDP response did not use the post-trained actor")
    if payload.get("frames_appended") != 1:
        raise ValueError("X-NavDP PointGoal request did not append exactly one frame")
    _validate_history_frame_count(payload, expected_history_frame_count)
    echoed_seed = payload.get("diffusion_seed")
    if expected_seed is not None and echoed_seed != int(expected_seed):
        raise ValueError(
            f"X-NavDP echoed diffusion seed {echoed_seed!r}, "
            f"expected {int(expected_seed)}")

    trajectory = np.asarray(payload.get("trajectory"), dtype=np.float64)
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if (trajectory.ndim != 2 or trajectory.shape != (24, 3)
            or not np.isfinite(trajectory).all()):
        raise ValueError(
            f"X-NavDP trajectory must be finite [24,3], got {trajectory.shape}")

    candidates = np.asarray(payload.get("all_trajectory"), dtype=np.float64)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if (candidates.ndim != 3 or candidates.shape[1:] != (24, 3)
            or candidates.shape[0] < 2 or not np.isfinite(candidates).all()):
        raise ValueError(
            "X-NavDP candidates must be finite [samples,24,3], got "
            f"{candidates.shape}")

    values = np.asarray(payload.get("all_values"), dtype=np.float64)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if (values.shape != (candidates.shape[0],)
            or not np.isfinite(values).all()):
        raise ValueError(
            "X-NavDP Q values must match the candidate count, got "
            f"{values.shape} for {candidates.shape[0]} candidates")

    result = dict(payload)
    result["trajectory"] = trajectory.tolist()
    result["all_trajectory"] = candidates.tolist()
    result["all_values"] = values.tolist()
    return result


__all__ = [
    "FROZEN_BASE_NAVDP_SHA256",
    "OFFICIAL_XNAVDP_COMMIT",
    "OFFICIAL_XNAVDP_POSTTRAIN_SHA256",
    "REVISIT_CONTROLLERS",
    "XNAVDP_ALGO",
    "XNAVDP_CHECKPOINT_TENSOR_COUNT",
    "XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT",
    "XNAVDP_EMBODIMENT",
    "XNAVDP_MODEL_STATE_TENSOR_COUNT",
    "habitat_pose_to_xnavdp",
    "normalize_xnavdp_response",
    "pointgoal_payload",
    "validate_history_receipt",
    "validate_reset_receipt",
    "xnavdp_state_payload",
]
