#!/usr/bin/env python3
"""Counterfactual audit of SE(2) progress on completed scalar-compass traces.

This script never reruns navigation.  It replays the frame-bound executor
translation/yaw receipts already produced by a completed action-coordinate
arm, reconstructs the certified historical route, and reports what the new
monotone 2-D projection would have believed along the *same physical path*.
Simulator positions in the trace are used only by the scorer to quantify
dead-reckoning error; they are never passed to the compass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from MemNavData.se2_projected_route_compass import (
    SE2ProjectedRouteCompass,
    reconstruct_reverse_route,
    reconstruct_reverse_route_se2,
)


SCHEMA_VERSION = "se2_progress_counterfactual_audit_v1_20260902"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def world_delta_in_body(delta_xz: np.ndarray, yaw: float) -> np.ndarray:
    """Habitat x/z displacement -> (forward, left) at ``yaw``."""

    dx, dz = (float(value) for value in np.asarray(delta_xz, dtype=np.float64))
    return np.asarray([
        -dx * math.sin(yaw) - dz * math.cos(yaw),
        -dx * math.cos(yaw) + dz * math.sin(yaw),
    ], dtype=np.float64)


def pose_xz(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row["x"]), float(row["z"])], dtype=np.float64)


def receipt_from_query_pose(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row["executed_translation_m_since_previous_frame"]),
        float(row["executed_yaw_rad_since_previous_frame"]),
    )


def relative_se2(
    previous: dict[str, Any], current: dict[str, Any],
) -> tuple[float, float, float]:
    delta = world_delta_in_body(
        pose_xz(current) - pose_xz(previous), float(previous["yaw"]))
    yaw = (float(current["yaw"]) - float(previous["yaw"])
           + math.pi) % (2.0 * math.pi) - math.pi
    return float(delta[0]), float(delta[1]), float(yaw)


def audit_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("arm") == "certified",
            f"{path}: not a certified query trace")
    history = payload["rollout_traces"]["legA"]
    query = payload["query_trace_payload"]["poses"]
    replay_receipts = payload["replay"]["executor_motion_receipts"]
    plans = [
        plan for plan in payload["query_leg"]
        if plan.get("certified_relocalization_accepted") is True
    ]
    require(history and query and plans, f"{path}: missing accepted route data")
    require(len(history) == len(replay_receipts),
            f"{path}: history receipt count mismatch")
    require([int(row["step"]) for row in history]
            == list(range(len(history))),
            f"{path}: history steps are not contiguous")
    require([int(row["step"]) for row in query]
            == list(range(len(query))),
            f"{path}: query steps are not contiguous")

    first = plans[0]
    goal_start = int(first["action_coordinate_goal_start_frame"])
    target = int(first["action_coordinate_target_anchor"])
    require(goal_start == len(history),
            f"{path}: query boundary differs from replayed history length")
    require(0 <= target < goal_start - 1,
            f"{path}: certified target cannot form a route")

    first_query_translation, first_query_yaw = receipt_from_query_pose(query[0])
    translations = [first_query_translation]
    yaw_deltas = [first_query_yaw]
    boundary_forward, boundary_left, boundary_yaw = relative_se2(
        history[-1], query[0])
    local_forwards = [boundary_forward]
    local_lefts = [boundary_left]
    local_yaws = [boundary_yaw]
    for index in range(goal_start - 1, target, -1):
        receipt = replay_receipts[index]
        require(int(receipt["step"]) == index,
                f"{path}: history receipt binding changed")
        translations.append(float(receipt["executed_translation_m"]))
        yaw_deltas.append(float(receipt["executed_yaw_rad"]))
        forward, left, yaw = relative_se2(
            history[index - 1], history[index])
        local_forwards.append(forward)
        local_lefts.append(left)
        local_yaws.append(yaw)

    reconstructed_route, _ = reconstruct_reverse_route(
        translations, yaw_deltas)
    scalar_compass = SE2ProjectedRouteCompass(
        reconstructed_route, lookahead_m=2.5, controller_radius_m=2.5)
    local_route, _ = reconstruct_reverse_route_se2(
        local_forwards, local_lefts, local_yaws)
    local_compass = SE2ProjectedRouteCompass.from_reverse_local_se2_receipts(
        local_forwards,
        local_lefts,
        local_yaws,
        lookahead_m=2.5,
        controller_radius_m=2.5,
    )

    # Scorer-only reconstruction error.  The compass above never receives any
    # row's x/z/yaw field.
    query_origin = pose_xz(query[0])
    query_origin_yaw = float(query[0]["yaw"])
    actual_route_world = [query_origin]
    actual_route_world.extend(
        pose_xz(history[index])
        for index in range(goal_start - 1, target - 1, -1)
    )
    actual_route_local = np.stack([
        world_delta_in_body(point - query_origin, query_origin_yaw)
        for point in actual_route_world
    ])
    require(actual_route_local.shape == reconstructed_route.shape
            and actual_route_local.shape == local_route.shape,
            f"{path}: scorer route shape mismatch")
    scalar_route_errors = np.linalg.norm(
        reconstructed_route - actual_route_local, axis=1)
    local_route_errors = np.linalg.norm(
        local_route - actual_route_local, axis=1)

    plans_by_step = {int(plan["step"]): plan for plan in plans}
    plan_steps = sorted(plans_by_step)
    require(plan_steps[0] == 0 and len(plan_steps) == len(plans),
            f"{path}: accepted plan steps are not uniquely bound at zero")
    maximum_plan_step = plan_steps[-1]
    require(maximum_plan_step < len(query),
            f"{path}: final plan lies outside query trace")

    scalar_readout = scalar_compass.advance(
        executed_translation_m=0.0, executed_yaw_rad=0.0)
    local_readout = local_compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    scalar_readouts = {0: scalar_readout}
    local_readouts = {0: local_readout}
    scalar_odometry_errors = [0.0]
    local_odometry_errors = [0.0]
    scalar_cross_tracks = [float(scalar_readout.cross_track_error_m)]
    local_cross_tracks = [float(local_readout.cross_track_error_m)]
    for step in range(1, maximum_plan_step + 1):
        translation, yaw = receipt_from_query_pose(query[step])
        scalar_readout = scalar_compass.advance(
            executed_translation_m=translation,
            executed_yaw_rad=yaw,
        )
        forward, left, local_yaw = relative_se2(
            query[step - 1], query[step])
        local_readout = local_compass.advance_local_se2(
            executed_forward_m=forward,
            executed_left_m=left,
            executed_yaw_rad=local_yaw,
        )
        actual_local = world_delta_in_body(
            pose_xz(query[step]) - query_origin, query_origin_yaw)
        scalar_odometry_errors.append(float(np.linalg.norm(
            np.asarray(scalar_readout.estimated_position) - actual_local)))
        local_odometry_errors.append(float(np.linalg.norm(
            np.asarray(local_readout.estimated_position) - actual_local)))
        if step in plans_by_step:
            scalar_readouts[step] = scalar_readout
            local_readouts[step] = local_readout
            scalar_cross_tracks.append(float(
                scalar_readout.cross_track_error_m))
            local_cross_tracks.append(float(local_readout.cross_track_error_m))
    require(set(scalar_readouts) == set(plan_steps)
            and set(local_readouts) == set(plan_steps),
            f"{path}: failed to replay every accepted plan observation")

    last_plan = plans_by_step[maximum_plan_step]
    scalar_final = scalar_readouts[maximum_plan_step]
    local_final = local_readouts[maximum_plan_step]
    old_progress = float(last_plan["action_coordinate_progress_m"])
    old_extent = float(last_plan["action_coordinate_route_extent_m"])
    require(old_extent > 0.0, f"{path}: old route has no extent")
    old_fraction = old_progress / old_extent
    reached = bool(payload["query_result"]["reached"])
    return {
        "source": path.name,
        "source_sha256": sha256(path),
        "scene": str(payload["query_trace_payload"]["source_scene"]),
        "episode": str(payload["query_trace_payload"]["episode"]),
        "reached": reached,
        "final_goal_distance_m": float(
            payload["query_result"]["final_goal_dist_m"]),
        "target_anchor": target,
        "history_frames": len(history),
        "query_frames": len(query),
        "accepted_plan_count": len(plans),
        "old_scalar_progress_m": old_progress,
        "old_scalar_extent_m": old_extent,
        "old_scalar_progress_fraction": old_fraction,
        "old_scalar_saturated": old_fraction >= 1.0 - 1e-9,
        "scalar_se2_projected_progress_m": float(
            scalar_final.projected_progress_m),
        "scalar_se2_route_extent_m": float(scalar_compass.route_extent_m),
        "scalar_se2_projected_progress_fraction": float(
            scalar_final.route_progress_fraction),
        "scalar_se2_projected_saturated": bool(
            scalar_final.route_progress_fraction >= 1.0 - 1e-9),
        "scalar_se2_query_path_length_m": float(
            scalar_final.query_path_length_m),
        "scalar_se2_final_cross_track_m": float(
            scalar_final.cross_track_error_m),
        "scalar_se2_median_plan_cross_track_m": median(
            scalar_cross_tracks),
        "local_se2_projected_progress_m": float(
            local_final.projected_progress_m),
        "local_se2_route_extent_m": float(local_compass.route_extent_m),
        "local_se2_projected_progress_fraction": float(
            local_final.route_progress_fraction),
        "local_se2_projected_saturated": bool(
            local_final.route_progress_fraction >= 1.0 - 1e-9),
        "local_se2_query_path_length_m": float(
            local_final.query_path_length_m),
        "local_se2_final_cross_track_m": float(
            local_final.cross_track_error_m),
        "local_se2_median_plan_cross_track_m": median(local_cross_tracks),
        "scalar_route_dead_reckoning_median_error_m_scorer_only": float(
            np.median(scalar_route_errors)),
        "scalar_route_dead_reckoning_final_error_m_scorer_only": float(
            scalar_route_errors[-1]),
        "local_route_dead_reckoning_median_error_m_scorer_only": float(
            np.median(local_route_errors)),
        "local_route_dead_reckoning_final_error_m_scorer_only": float(
            local_route_errors[-1]),
        "scalar_query_dead_reckoning_median_error_m_scorer_only": float(
            np.median(scalar_odometry_errors)),
        "scalar_query_dead_reckoning_final_error_m_scorer_only": float(
            scalar_odometry_errors[-1]),
        "local_query_dead_reckoning_median_error_m_scorer_only": float(
            np.median(local_odometry_errors)),
        "local_query_dead_reckoning_final_error_m_scorer_only": float(
            local_odometry_errors[-1]),
        "simulator_pose_visible_to_compass": False,
        "metric_depth_visible_to_compass": False,
        "visual_observation_visible_after_authorization": False,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [record for record in records if not record["reached"]]

    def values(key: str, rows=records) -> list[float]:
        return [float(row[key]) for row in rows]

    return {
        "records": len(records),
        "successes": sum(record["reached"] for record in records),
        "failures": len(failures),
        "old_scalar_saturated": sum(
            record["old_scalar_saturated"] for record in records),
        "scalar_se2_projected_saturated": sum(
            record["scalar_se2_projected_saturated"] for record in records),
        "local_se2_projected_saturated": sum(
            record["local_se2_projected_saturated"] for record in records),
        "old_scalar_saturated_failures": sum(
            record["old_scalar_saturated"] for record in failures),
        "scalar_se2_projected_saturated_failures": sum(
            record["scalar_se2_projected_saturated"] for record in failures),
        "local_se2_projected_saturated_failures": sum(
            record["local_se2_projected_saturated"] for record in failures),
        "median_old_scalar_progress_fraction": median(values(
            "old_scalar_progress_fraction")),
        "median_scalar_se2_projected_progress_fraction": median(values(
            "scalar_se2_projected_progress_fraction")),
        "median_local_se2_projected_progress_fraction": median(values(
            "local_se2_projected_progress_fraction")),
        "median_scalar_se2_final_cross_track_m": median(values(
            "scalar_se2_final_cross_track_m")),
        "median_local_se2_final_cross_track_m": median(values(
            "local_se2_final_cross_track_m")),
        "median_scalar_route_dead_reckoning_error_m_scorer_only": median(
            values("scalar_route_dead_reckoning_median_error_m_scorer_only")),
        "median_local_route_dead_reckoning_error_m_scorer_only": median(
            values("local_route_dead_reckoning_median_error_m_scorer_only")),
        "median_scalar_query_dead_reckoning_error_m_scorer_only": median(
            values("scalar_query_dead_reckoning_median_error_m_scorer_only")),
        "median_local_query_dead_reckoning_error_m_scorer_only": median(
            values("local_query_dead_reckoning_median_error_m_scorer_only")),
        "claim_boundary": (
            "counterfactual state-estimator replay on completed trajectories; "
            "not closed-loop navigation and not an SR result"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "output already exists")
    paths = sorted({path.resolve() for path in args.inputs})
    require(bool(paths), "no input traces")
    records = [audit_one(path) for path in paths]
    output = {
        "schema_version": SCHEMA_VERSION,
        "summary": summarize(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(output["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
