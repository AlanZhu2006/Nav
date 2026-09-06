#!/usr/bin/env python3
"""Consumed one-episode diagnosis of long-range local route semantics.

Every arm first executes the same real CEC proposal/certificate transaction.
Only then may this diagnostic fork read evaluator pose and the Habitat shortest
path to resample the frozen controller without mutating its FIFO.  The fork is
privileged, consumed-development-only, and never paper-result eligible.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import requests

import eval_shared_online_role_pairs as evaluator
from longrange_local_tangent_attribution import (
    TANGENT_ALIGNMENT_THRESHOLD_DEG,
    TANGENT_ARMS,
    TANGENT_MIN_PLANAR_M,
    TANGENT_SCHEMA_VERSION,
    first_local_tangent_world,
    realized_planar_translation_m,
    signed_pointgoal_heading_deg,
    tangent_requires_alignment,
)
from longrange_oracle_attribution import (
    OracleRuntimeState,
    fixed_radius_pointgoal,
    geodesic_lookahead_world,
)


base = evaluator.base
ARM = os.environ.get("HM3D_LONGRANGE_TANGENT_ARM", "").strip()
RADIUS_M = 2.5
CHORD_LOOKAHEAD_M = 2.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


STATE = OracleRuntimeState()
ORIGINAL_REPLAY_PREFIX = evaluator.replay_prefix
ORIGINAL_RUNTIME_QUERY = evaluator.runtime_query
ORIGINAL_RUN_POLICY_LEG = base.run_policy_leg
ORIGINAL_SRV_PLAN = base.srv_plan
ORIGINAL_PURSUIT_STEP = base.pursuit_step


def capture_replay_prefix(frozen: dict) -> tuple[dict, dict]:
    STATE.bind_history(frozen["trace"])
    return ORIGINAL_REPLAY_PREFIX(frozen)


def capture_runtime_query(query: dict) -> dict:
    result = ORIGINAL_RUNTIME_QUERY(query)
    STATE.bind_query(np.asarray(result["floor_position"], dtype=np.float64))
    return result


def capture_run_policy_leg(simulator, pathfinder, *args, **kwargs):
    STATE.bind_pathfinder(pathfinder)
    require(STATE.goal_floor is not None,
            "floor-aware query goal was not bound before rollout")
    kwargs["success_goal_position"] = STATE.goal_floor.copy()
    return ORIGINAL_RUN_POLICY_LEG(simulator, pathfinder, *args, **kwargs)


def realized_pursuit_step(position, yaw, path_xz, pathfinder):
    """Preserve motion exactly but account for post-snap displacement."""

    before = np.asarray(position, dtype=np.float64).copy()
    after, next_yaw, _commanded = ORIGINAL_PURSUIT_STEP(
        position, yaw, path_xz, pathfinder)
    realized = realized_planar_translation_m(before, after)
    return after, next_yaw, realized


def request_read_only_controller(
    *,
    pointgoal: np.ndarray,
    image_jpg: bytes,
    goal_jpg: bytes,
    depth: np.ndarray,
    diffusion_seed: int,
    original_response: dict[str, Any],
    pure_pointgoal: bool,
) -> dict[str, Any]:
    """Resample one controller without mutating or advancing NavDP state."""

    receipt = original_response.get("monocular_depth_receipt")
    require(isinstance(receipt, dict),
            "oracle resample requires the original mono-depth receipt")
    token = receipt.get("monocular_depth_transaction_token")
    frame_index = receipt.get("frame_index")
    require(isinstance(token, str) and len(token) == 64,
            "mono-depth transaction token is missing")
    require(isinstance(frame_index, int) and frame_index >= 0,
            "mono-depth transaction frame is missing")
    data = {
        "diffusion_seed": str(int(diffusion_seed)),
        "monocular_depth_transaction_token": token,
        "monocular_depth_frame_index": str(frame_index),
        "goal_data": json.dumps(base.pointgoal_payload(pointgoal)),
    }
    files = {
        "image": ("image.jpg", image_jpg),
        "depth": ("depth.png", base.depth_png_bytes(depth)),
    }
    endpoint = "pointgoal_resample"
    if not pure_pointgoal:
        endpoint = "mixgoal_resample"
        files["image_goal"] = ("goal.jpg", goal_jpg)
    response = requests.post(
        f"{base.NOVEL_BASE}/{endpoint}",
        files=files,
        data=data,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    require(payload.get("memory_mutated") is False,
            f"{endpoint} did not assert read-only FIFO semantics")
    require(payload.get("queue_hashes_before")
            == payload.get("queue_hashes_after"),
            f"{endpoint} changed FIFO content")
    require(isinstance(payload.get("queue_hashes_before"), list)
            and payload["queue_hashes_before"],
            f"{endpoint} omitted FIFO fingerprints")
    require(int(payload.get("diffusion_seed")) == int(diffusion_seed),
            f"{endpoint} seed echo mismatch")
    require(payload.get("metric_depth_sensor_consumed") is False,
            f"{endpoint} consumed simulator metric depth")
    replay_receipt = payload.get("monocular_depth_receipt")
    require(isinstance(replay_receipt, dict)
            and replay_receipt.get("monocular_depth_transaction_token")
            == token
            and replay_receipt.get("image_sha256")
            == receipt.get("image_sha256"),
            f"{endpoint} was not bound to the original observation")
    return base.normalize_navdp_response(payload)


def _native_read_only_controller(
    *,
    image_jpg: bytes,
    goal_jpg: bytes,
    depth: np.ndarray,
    diffusion_seed: int,
    original_response: dict[str, Any],
) -> dict[str, Any]:
    receipt = original_response.get("monocular_depth_receipt")
    require(isinstance(receipt, dict),
            "native resample requires the original mono-depth receipt")
    payload = base.srv_navdp_imagegoal_resample(
        image_jpg,
        goal_jpg,
        depth,
        diffusion_seed,
        monocular_depth_transaction_token=receipt.get(
            "monocular_depth_transaction_token"),
        monocular_depth_frame_index=receipt.get("frame_index"),
    )
    require(payload.get("metric_depth_sensor_consumed") is False,
            "native read-only resample consumed simulator metric depth")
    replay_receipt = payload.get("monocular_depth_receipt")
    require(isinstance(replay_receipt, dict)
            and replay_receipt.get("monocular_depth_transaction_token")
            == receipt.get("monocular_depth_transaction_token")
            and replay_receipt.get("image_sha256")
            == receipt.get("image_sha256"),
            "native resample was not bound to the original observation")
    return base.normalize_navdp_response(payload)


def _oracle_direction(
    position: np.ndarray, yaw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(STATE.pathfinder is not None and STATE.goal_floor is not None,
            "local-tangent oracle lacks pathfinder/query goal")
    ok, distance, path = base.geodesic(
        STATE.pathfinder, position, STATE.goal_floor)
    require(ok and np.isfinite(distance) and len(path) >= 2,
            "current oracle geodesic query failed")
    path_array = np.asarray(path, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "local_tangent_oracle_current_geodesic_m": float(distance),
        "local_tangent_oracle_path_point_count": int(len(path_array)),
        "local_tangent_oracle_path_start_y_m": float(path_array[0, 1]),
        "local_tangent_oracle_path_goal_y_m": float(path_array[-1, 1]),
        "local_tangent_oracle_goal_position_xyz": (
            STATE.goal_floor.tolist()),
        "local_tangent_oracle_goal_vertical_delta_m": float(
            STATE.goal_floor[1] - position[1]),
    }
    if ARM == "oracle_chord_mixed_realized":
        target_xz, path_extent = geodesic_lookahead_world(
            path_array, CHORD_LOOKAHEAD_M)
        diagnostics.update({
            "local_tangent_oracle_direction_source": (
                "planar_2p5m_geodesic_chord"),
            "local_tangent_oracle_path_extent_planar_m": float(path_extent),
            "local_tangent_oracle_reference_world_xz": target_xz.tolist(),
            "local_tangent_oracle_reference_world_xyz": None,
        })
    else:
        target_xyz, tangent = first_local_tangent_world(
            path_array, min_planar_m=TANGENT_MIN_PLANAR_M)
        target_xz = target_xyz[[0, 2]]
        diagnostics.update({
            "local_tangent_oracle_direction_source": (
                "first_traversable_geodesic_tangent"),
            "local_tangent_oracle_reference_world_xz": target_xz.tolist(),
            "local_tangent_oracle_reference_world_xyz": target_xyz.tolist(),
            **{f"local_tangent_oracle_{key}": value
               for key, value in tangent.items()},
        })
    pointgoal = fixed_radius_pointgoal(
        target_xz, position[[0, 2]], yaw, RADIUS_M)
    heading = signed_pointgoal_heading_deg(pointgoal)
    diagnostics.update({
        "local_tangent_oracle_pointgoal": pointgoal.tolist(),
        "local_tangent_oracle_pointgoal_radius_m": float(
            np.linalg.norm(pointgoal)),
        "local_tangent_oracle_signed_heading_deg": heading,
        "local_tangent_oracle_alignment_threshold_deg": (
            TANGENT_ALIGNMENT_THRESHOLD_DEG),
    })
    return pointgoal, diagnostics


def _merge_controller_response(
    original: dict[str, Any], replacement: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(original)
    for key in (
        "trajectory", "all_trajectory", "all_values", "diffusion_seed",
        "critic_max", "critic_min", "navdp_critic_max",
        "navdp_stop_evidence", "depth_source",
        "metric_depth_sensor_consumed", "monocular_depth_receipt",
    ):
        if key in replacement:
            merged[key] = replacement[key]
    return merged


def tangent_srv_plan(*args, **kwargs) -> dict[str, Any]:
    response = ORIGINAL_SRV_PLAN(*args, **kwargs)
    require(response.get("certified_relocalization_ok") is True
            and response.get("certified_relocalization_accepted") is True,
            "local-tangent attribution requires an accepted real CEC proof")
    require(response.get("certified_relocalization_authority_policy")
            == "strict_certificate",
            "local-tangent attribution received another authority policy")
    position = np.asarray(kwargs.get("robot_position"), dtype=np.float64)
    yaw = float(kwargs.get("robot_yaw"))
    seed = kwargs.get("diffusion_seed")
    require(position.shape == (3,) and np.isfinite(position).all()
            and np.isfinite(yaw) and seed is not None,
            "local-tangent attribution requires finite evaluator pose/seed")

    pointgoal, diagnostics = _oracle_direction(position, yaw)
    use_tangent = (
        ARM != "oracle_tangent_then_native_realized"
        or tangent_requires_alignment(
            pointgoal, threshold_deg=TANGENT_ALIGNMENT_THRESHOLD_DEG)
    )
    if use_tangent:
        replacement = request_read_only_controller(
            pointgoal=pointgoal,
            image_jpg=args[0],
            goal_jpg=args[1],
            depth=kwargs.get("depth"),
            diffusion_seed=int(seed),
            original_response=response,
            pure_pointgoal=False,
        )
        controller = "mixed_image_pointgoal"
        selection_reason = (
            "continuous_oracle_direction"
            if ARM != "oracle_tangent_then_native_realized"
            else "tangent_heading_residual_above_threshold")
    else:
        replacement = _native_read_only_controller(
            image_jpg=args[0],
            goal_jpg=args[1],
            depth=kwargs.get("depth"),
            diffusion_seed=int(seed),
            original_response=response,
        )
        controller = "native_imagegoal"
        selection_reason = "tangent_heading_aligned_native_control"

    merged = _merge_controller_response(response, replacement)
    merged.update({
        "pose_controller": f"oracle_local_tangent_{controller}_resample",
        "local_tangent_oracle_schema_version": TANGENT_SCHEMA_VERSION,
        "local_tangent_oracle_arm": ARM,
        "local_tangent_oracle_evaluator_pose_consumed": True,
        "local_tangent_oracle_goal_position_consumed": True,
        "local_tangent_oracle_habitat_path_consumed": True,
        "local_tangent_oracle_read_only_resample": True,
        "local_tangent_oracle_controller": controller,
        "local_tangent_oracle_selection_reason": selection_reason,
        "local_tangent_oracle_tangent_authority_used": bool(use_tangent),
        "local_tangent_oracle_realized_motion_accounting": True,
        "local_tangent_oracle_success_distance_contract": (
            "floor_aware_3d_euclidean_v1"),
        **diagnostics,
    })
    return merged


def main() -> None:
    require(ARM in TANGENT_ARMS,
            "HM3D_LONGRANGE_TANGENT_ARM is missing or invalid")
    require(base.args.role_pair_scope == "table3_longrange_oracle",
            "local-tangent evaluator requires its consumed scope")
    require(base.args.role_pair_query_role == "revisit",
            "local-tangent evaluator is Revisit-only")
    require(base.args.certified_guidance_mode == "endpoint_bearing",
            "local-tangent evaluator requires canonical CEC proof first")
    evaluator.replay_prefix = capture_replay_prefix
    evaluator.runtime_query = capture_runtime_query
    base.run_policy_leg = capture_run_policy_leg
    base.pursuit_step = realized_pursuit_step
    base.srv_plan = tangent_srv_plan
    evaluator.main()


if __name__ == "__main__":
    main()
