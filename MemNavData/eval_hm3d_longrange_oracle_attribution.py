#!/usr/bin/env python3
"""Consumed long-range oracle attribution on the sealed Table-III cohort.

The ordinary evaluator performs the real CEC proposal/certificate transaction
and advances both causal memories exactly once.  Only after that transaction,
this wrapper may replace the controller sample with a read-only resample whose
PointGoal is derived from evaluator pose.  The fork is diagnostic and cannot
support a deployable or paper-method claim.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import requests

import eval_shared_online_role_pairs as evaluator
from longrange_oracle_attribution import (
    ORACLE_ARMS,
    ORACLE_SCHEMA_VERSION,
    OracleRouteProjector,
    OracleRuntimeState,
    fixed_radius_pointgoal,
    geodesic_lookahead_world,
    historical_reverse_route,
)


base = evaluator.base
ARM = os.environ.get("HM3D_LONGRANGE_ATTRIBUTION_ARM", "").strip()
RADIUS_M = 2.5
LOOKAHEAD_M = 2.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


STATE = OracleRuntimeState()
ORIGINAL_REPLAY_PREFIX = evaluator.replay_prefix
ORIGINAL_RUNTIME_QUERY = evaluator.runtime_query
ORIGINAL_RUN_POLICY_LEG = base.run_policy_leg
ORIGINAL_SRV_PLAN = base.srv_plan


def capture_replay_prefix(frozen: dict) -> tuple[dict, dict]:
    STATE.bind_history(frozen["trace"])
    return ORIGINAL_REPLAY_PREFIX(frozen)


def capture_runtime_query(query: dict) -> dict:
    result = ORIGINAL_RUNTIME_QUERY(query)
    floor = np.asarray(result["floor_position"], dtype=np.float64)
    STATE.bind_query(floor)
    return result


def capture_run_policy_leg(simulator, pathfinder, *args, **kwargs):
    STATE.bind_pathfinder(pathfinder)
    return ORIGINAL_RUN_POLICY_LEG(simulator, pathfinder, *args, **kwargs)


def _monocular_transaction_data(
    response: dict[str, Any], diffusion_seed: int,
) -> dict[str, str]:
    receipt = response.get("monocular_depth_receipt")
    require(isinstance(receipt, dict),
            "oracle resample requires the original mono-depth receipt")
    token = receipt.get("monocular_depth_transaction_token")
    frame_index = receipt.get("frame_index")
    require(isinstance(token, str) and len(token) == 64,
            "mono-depth transaction token is missing")
    require(isinstance(frame_index, int) and frame_index >= 0,
            "mono-depth transaction frame is missing")
    return {
        "diffusion_seed": str(int(diffusion_seed)),
        "monocular_depth_transaction_token": token,
        "monocular_depth_frame_index": str(frame_index),
    }


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
    data = _monocular_transaction_data(original_response, diffusion_seed)
    data["goal_data"] = json.dumps(base.pointgoal_payload(pointgoal))
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
    receipt = payload.get("monocular_depth_receipt")
    original_receipt = original_response["monocular_depth_receipt"]
    require(isinstance(receipt, dict)
            and receipt.get("monocular_depth_transaction_token")
            == original_receipt.get("monocular_depth_transaction_token")
            and receipt.get("image_sha256")
            == original_receipt.get("image_sha256"),
            f"{endpoint} was not bound to the original observation")
    return base.normalize_navdp_response(payload)


def _route_pointgoal(
    response: dict[str, Any], position: np.ndarray, yaw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(STATE.trace is not None and STATE.goal_floor is not None,
            "route oracle lacks a frozen trace/query")
    anchor = int(response["anchor"])
    goal_start = int(response["goal_start_frame"])
    if (STATE.route_projector is None or STATE.route_anchor != anchor
            or STATE.route_goal_start != goal_start):
        poses = STATE.trace["poses"]
        route = historical_reverse_route(
            poses,
            query_start_xz=position[[0, 2]],
            goal_start_frame=goal_start,
            target_anchor=anchor,
        )
        anchor_pose = np.asarray([
            poses[anchor]["x"], poses[anchor]["y"], poses[anchor]["z"],
        ], dtype=np.float64)
        ok, _tail_distance, tail = base.geodesic(
            STATE.pathfinder, anchor_pose, STATE.goal_floor)
        require(ok and len(tail) >= 2,
                "oracle route could not connect anchor to query goal")
        route = np.concatenate((
            route,
            np.asarray(tail, dtype=np.float64)[1:, [0, 2]],
        ), axis=0)
        STATE.route_projector = OracleRouteProjector(
            route, lookahead_m=LOOKAHEAD_M, radius_m=RADIUS_M)
        STATE.route_anchor = anchor
        STATE.route_goal_start = goal_start
    projection = STATE.route_projector.update(position[[0, 2]], yaw)
    return projection.pointgoal, {
        "action_coordinate_oracle_route_extent_m": (
            STATE.route_projector.extent_m),
        "action_coordinate_oracle_progress_m": projection.progress_m,
        "action_coordinate_oracle_remaining_m": projection.remaining_m,
        "action_coordinate_oracle_cross_track_m": projection.cross_track_m,
        "action_coordinate_oracle_projection_segment": (
            projection.projection_segment),
        "action_coordinate_oracle_reference_arc_m": (
            projection.reference_arc_m),
        "action_coordinate_oracle_reference_world_xz": (
            projection.reference_world_xz.tolist()),
        "action_coordinate_oracle_progress_jump_m": (
            projection.progress_jump_m),
    }


def _geodesic_pointgoal(
    position: np.ndarray, yaw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(STATE.pathfinder is not None and STATE.goal_floor is not None,
            "geodesic oracle lacks pathfinder/query goal")
    ok, distance, path = base.geodesic(
        STATE.pathfinder, position, STATE.goal_floor)
    require(ok and np.isfinite(distance),
            "current oracle geodesic query failed")
    target, path_extent = geodesic_lookahead_world(path, LOOKAHEAD_M)
    pointgoal = fixed_radius_pointgoal(
        target, position[[0, 2]], yaw, RADIUS_M)
    return pointgoal, {
        "action_coordinate_oracle_current_geodesic_m": float(distance),
        "action_coordinate_oracle_current_path_extent_m": path_extent,
        "action_coordinate_oracle_reference_world_xz": target.tolist(),
    }


def oracle_srv_plan(*args, **kwargs) -> dict[str, Any]:
    response = ORIGINAL_SRV_PLAN(*args, **kwargs)
    if ARM == "action_coordinate_mixed":
        return response
    require(response.get("certified_relocalization_ok") is True
            and response.get("certified_relocalization_accepted") is True,
            "oracle attribution requires an accepted real CEC proof")
    require(response.get("certified_relocalization_authority_policy")
            == "strict_certificate",
            "oracle attribution received another authority policy")
    position = np.asarray(kwargs.get("robot_position"), dtype=np.float64)
    yaw = float(kwargs.get("robot_yaw"))
    seed = kwargs.get("diffusion_seed")
    require(position.shape == (3,) and np.isfinite(position).all()
            and np.isfinite(yaw) and seed is not None,
            "oracle attribution requires finite evaluator pose and seed")
    if ARM == "oracle_route_mixed":
        pointgoal, diagnostics = _route_pointgoal(response, position, yaw)
        pure_pointgoal = False
    else:
        pointgoal, diagnostics = _geodesic_pointgoal(position, yaw)
        pure_pointgoal = ARM == "oracle_geodesic_point"
    resampled = request_read_only_controller(
        pointgoal=pointgoal,
        image_jpg=args[0],
        goal_jpg=args[1],
        depth=kwargs.get("depth"),
        diffusion_seed=int(seed),
        original_response=response,
        pure_pointgoal=pure_pointgoal,
    )
    merged = dict(response)
    for key in (
        "trajectory", "all_trajectory", "all_values", "diffusion_seed",
        "critic_max", "critic_min", "navdp_critic_max",
        "navdp_stop_evidence", "depth_source",
        "metric_depth_sensor_consumed", "monocular_depth_receipt",
    ):
        if key in resampled:
            merged[key] = resampled[key]
    merged.update({
        "pose_controller": (
            "oracle_geodesic_point_resample"
            if pure_pointgoal else "oracle_point_image_resample"),
        "action_coordinate_oracle_schema_version": ORACLE_SCHEMA_VERSION,
        "action_coordinate_oracle_arm": ARM,
        "action_coordinate_oracle_evaluator_pose_consumed": True,
        "action_coordinate_oracle_goal_position_consumed": True,
        "action_coordinate_oracle_habitat_path_consumed": True,
        "action_coordinate_oracle_current_geodesic_replanned": (
            ARM != "oracle_route_mixed"),
        "action_coordinate_oracle_anchor_to_goal_tail_only": (
            ARM == "oracle_route_mixed"),
        "action_coordinate_oracle_historical_route_consumed": (
            ARM == "oracle_route_mixed"),
        "action_coordinate_oracle_read_only_resample": True,
        "action_coordinate_oracle_controller": (
            "pure_pointgoal" if pure_pointgoal else "mixed_image_pointgoal"),
        "action_coordinate_oracle_pointgoal": pointgoal.tolist(),
        "action_coordinate_oracle_pointgoal_radius_m": float(
            np.linalg.norm(pointgoal)),
        **diagnostics,
    })
    return merged


def main() -> None:
    require(ARM in ORACLE_ARMS,
            "HM3D_LONGRANGE_ATTRIBUTION_ARM is missing or invalid")
    require(base.args.role_pair_scope == "table3_longrange_oracle",
            "oracle evaluator requires its explicit consumed scope")
    require(base.args.role_pair_query_role == "revisit",
            "oracle evaluator is Revisit-only")
    expected_guidance = (
        "action_coordinate_compass"
        if ARM == "action_coordinate_mixed" else "endpoint_bearing")
    require(base.args.certified_guidance_mode == expected_guidance,
            "oracle attribution arm received the wrong CEC guidance mode")
    evaluator.replay_prefix = capture_replay_prefix
    evaluator.runtime_query = capture_runtime_query
    base.run_policy_leg = capture_run_policy_leg
    if ARM != "action_coordinate_mixed":
        base.srv_plan = oracle_srv_plan
    evaluator.main()


if __name__ == "__main__":
    main()
