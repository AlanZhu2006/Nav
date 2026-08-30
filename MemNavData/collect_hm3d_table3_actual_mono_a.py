#!/usr/bin/env python3
"""Collect one frozen Table-III candidate's actual monocular Goal-A history."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from deterministic_eval_protocol import validate_leg1_trace
from generate_twoleg import (
    GOAL_A_SOURCE_PROTOCOL, M_W, cam_to_world_hab, first_path_yaw,
    geodesic, make_sim, render, save_traj, yaw_facing,
)
from hm3d_fullmono_mixed_role import audit_goal_a_plans
from materialize_online_a_traces import native_control_audit


SCHEMA = "hm3d_table3_actual_mono_factual_a_v1_20260830"
CARRIER_SCHEMA = "hm3d_table3_goal_a_source_carrier_v1_20260830"
EXECUTION_SCHEMA = "hm3d_table3_actual_mono_execution_v2_20260830"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_vector(position: np.ndarray) -> list[float]:
    return [float(value) for value in M_W @ np.asarray(position, dtype=float)]


def final_path_yaw(points: list[np.ndarray], goal: np.ndarray) -> float:
    goal_xz = np.asarray(goal, dtype=float)[[0, 2]]
    for point in reversed(points[:-1]):
        delta = goal_xz - np.asarray(point, dtype=float)[[0, 2]]
        if float(np.linalg.norm(delta)) >= 0.30:
            return float(yaw_facing(delta))
    raise RuntimeError("Goal-A path has no nontrivial final segment")


def materialize_carrier(row: dict, destination: Path) -> dict:
    require(not destination.exists(), f"source carrier exists: {destination}")
    asset = Path(row["asset"]["glb_path"])
    navmesh = Path(row["asset"]["navmesh_path"])
    require(asset.is_file() and sha256(asset) == row["asset"]["glb_sha256"],
            "HM3D GLB changed")
    require(navmesh.is_file() and sha256(navmesh) == row["asset"]["navmesh_sha256"],
            "HM3D navmesh changed")
    geometry = row["capacity_geometry"]
    start = np.asarray(geometry["first_goal"], dtype=float)
    goal = np.asarray(geometry["query_start"], dtype=float)
    simulator = make_sim(
        str(asset), str(navmesh), agent_radius=0.30,
        recompute_navmesh=False,
    )
    try:
        # Capacity measured the eventual query direction
        # ``query_start -> first_goal``.  Factual Goal A deliberately traverses
        # the reverse direction to end at query_start and build causal history.
        # Detour endpoint projection can make those two directed measurements
        # differ near polygon boundaries, so verify the frozen capacity receipt
        # in its original direction and use the reverse path only for rollout.
        ok, distance, points = geodesic(simulator.pathfinder, start, goal)
        require(ok and math.isfinite(distance), "factual Goal-A path is invalid")
        capacity_ok, capacity_distance, _ = geodesic(
            simulator.pathfinder, goal, start
        )
        require(
            capacity_ok
            and math.isfinite(capacity_distance)
            and abs(
                float(capacity_distance)
                - float(geometry["first_goal_geodesic_m"])
            )
            <= 0.05,
            "capacity query-direction geodesic changed",
        )
        start_yaw = float(first_path_yaw(points, start))
        goal_yaw = final_path_yaw(points, goal)
        camera_height = 0.5
        start_camera = start + np.asarray([0.0, camera_height, 0.0])
        goal_camera = goal + np.asarray([0.0, camera_height, 0.0])
        start_rgb, start_depth = render(simulator, start_camera, start_yaw)
        goal_rgb, goal_depth = render(simulator, goal_camera, goal_yaw)
    finally:
        simulator.close()
    meta = {
        "scene": asset.name,
        "ep_idx": int(row["history_index"]),
        "generation_seed": int(row["history_index"]),
        "n_frames": 2,
        "n_legs": 2,
        "switch_idx": 2,
        "switches": [2],
        "start": data_vector(start),
        "A": data_vector(goal),
        "goals": [{
            "name": "B", "kind": "source_only_placeholder",
            "pos": data_vector(goal), "yaw_habitat": goal_yaw,
            "covis": 1.0, "covis_argmax": 1, "head_off_deg": 0.0,
            "anchor_frame_limit": 2, "non_anchor_max_covis": 0.0,
            "recall_gap": 0, "covis_curve": [0.0, 1.0],
        }],
        "geo_startA": float(distance),
        "geo_AB": 0.0,
        "geo_BC": None,
        "gen_protocol": CARRIER_SCHEMA,
        "upstream_source_protocol": GOAL_A_SOURCE_PROTOCOL,
        "role_sequence": ["initial_imagegoal", "source_only_placeholder"],
        "source_only_goal_a": True,
        "source_only_usage": "eval_2leg_habitat --stop_after_leg1",
        "query_goal_present": False,
        "initial_yaw_mode": "path_aligned",
        "start_yaw_habitat": start_yaw,
        "start_path_yaw_habitat": start_yaw,
        "start_heading_offset_deg": 0.0,
        "initial_distance_band_m": [float(distance), float(distance)],
        "initial_goal_pose_source": "frozen_capacity_query_start",
        "initial_start_pose_source": "frozen_capacity_first_goal",
        "covis_band": [0.0, 1.0],
        "novel_covis": 0.10,
        "covis_pos_hi": 0.55,
        "covis_pos_lo": 0.10,
        "window": 32,
        "num_scale": 8,
        "anchor_margin": 39,
        "camera_height_m": camera_height,
        "frame_convention": "positions+parquet in data(Zup,M_W); yaw_habitat in render frame",
        "candidate_identity_sha256": row["candidate_identity_sha256"],
    }
    poses = [cam_to_world_hab(start_camera, start_yaw),
             cam_to_world_hab(goal_camera, goal_yaw)]
    save_traj(str(destination), [start_rgb, goal_rgb],
              [start_depth, goal_depth], poses, meta, [goal_rgb])
    goal_path = destination / "videos/chunk-000/observation.images.rgb/1.jpg"
    return {
        "carrier_schema": CARRIER_SCHEMA,
        "carrier_root": str(destination.resolve()),
        "goal_a_sha256": sha256(goal_path),
        "start_yaw_rad": start_yaw,
        "goal_yaw_rad": goal_yaw,
        "geodesic_m": float(distance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--execution-protocol", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution_protocol.read_text())
    require(execution.get("schema_version") == EXECUTION_SCHEMA,
            "Table-III execution protocol changed")
    require(execution.get("runtime_geometry") == {
        "mode": "content_addressed_pinned_navmesh",
        "navmesh_path_source": "candidate_plan.asset.navmesh_path",
        "navmesh_sha256_source": "candidate_plan.asset.navmesh_sha256",
        "runtime_recomputation": False,
        "reason": ("the capacity bins and every rollout must use the "
                   "identical frozen Habitat path graph"),
    }, "Table-III runtime geometry contract changed")
    require(sha256(args.candidate_plan) == execution["candidate_plan"]["sha256"],
            "candidate plan changed")
    plan = json.loads(args.candidate_plan.read_text())
    require(len(plan["episodes"]) == execution["candidate_plan"]["candidate_count"],
            "candidate count changed")
    require(0 <= args.history_index < len(plan["episodes"]),
            "history index outside candidate plan")
    row = plan["episodes"][args.history_index]
    require(int(row["history_index"]) == args.history_index,
            "candidate order changed")
    runtime_episode = f"episode_{row['episode']}"
    label = f"{args.history_index:03d}_{row['scene']}_{runtime_episode}"
    output = args.run_root / "factual_a" / label
    require(not output.exists(), f"factual Goal-A output exists: {output}")
    output.mkdir(parents=True)
    carrier = args.run_root / "carriers" / row["scene"] / runtime_episode
    carrier_receipt = materialize_carrier(row, carrier)
    geometry = row["capacity_geometry"]
    cfg = execution["factual_A"]
    max_steps = max(600, math.ceil(
        2.5 * float(geometry["first_goal_geodesic_m"])
        / float(cfg["v_max_m_per_frame"])
    ))
    require(max_steps <= int(cfg["maximum_steps_cap"]),
            "factual Goal-A budget exceeds frozen cap")
    seed = int(cfg["base_seed"]) + args.history_index
    result = output / "result"
    log = output / "eval.log"
    command = [
        args.hab_python, "-u", str(args.source_root / "MemNavData/eval_2leg_habitat.py"),
        "--episode_root", str(carrier.parent), "--episode_ids", runtime_episode,
        "--scene", row["asset"]["glb_path"], "--scene_identity", row["scene"],
        "--pinned_navmesh", row["asset"]["navmesh_path"],
        "--expected_pinned_navmesh_sha256", row["asset"]["navmesh_sha256"],
        "--host", "127.0.0.1", "--port", str(args.memnav_port),
        "--novel_port", str(args.navdp_port), "--server_backend", "hybrid_pose",
        "--success_dist", str(cfg["success_radius_m"]),
        "--max_steps", str(max_steps), "--exec_horizon", str(cfg["execution_horizon"]),
        "--trajectory_selector", "server", "--trajectory_selector_scope", "all",
        "--leg1_mode", "policy", "--leg1_goal_source", "own",
        "--write_leg1_trace", "--stop_after_leg1", "--seed", str(seed),
        "--terminal_uturn", "off", "--terminal_visual_refine", "off",
        "--deterministic_plan_seeds", "--retrieval_override", "off",
        "--certified_cdec_rescue", "off", "--certified_stagnation_graph", "off",
        "--revisit_controller", "navdp_mixed", "--hybrid_route", "native_sidecar",
        "--revisit_adapter", "legacy_metric", "--navdp_depth_source", "monocular_sidecar",
        "--out", str(result),
    ]
    with log.open("x") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   check=False)
    require(completed.returncode == 0,
            f"factual Goal-A evaluator failed ({completed.returncode})")
    trace_path = result / f"{runtime_episode}_leg1_trace.json"
    plans_path = result / f"{runtime_episode}_plans.json"
    require(trace_path.is_file() and plans_path.is_file(),
            "factual Goal-A trace is missing")
    trace = json.loads(trace_path.read_text())
    validate_leg1_trace(
        trace, expected_episode=runtime_episode, expected_seed=seed,
        expected_goal_sha256=carrier_receipt["goal_a_sha256"],
        expected_source_scene=row["scene"],
    )
    control = native_control_audit(trace)
    require(control["ok"], "factual Goal-A was not native NavDP")
    depth_audit = audit_goal_a_plans(trace["plans"])
    with (result / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, "factual Goal-A metric row is ambiguous")
    reached = bool(trace["reached"])
    require(int(float(rows[0]["reached_A"])) == int(reached),
            "factual Goal-A metric/trace outcome mismatch")
    completion = {
        "schema_version": SCHEMA, "status": "complete",
        "history_index": args.history_index, "scene": row["scene"],
        "episode": runtime_episode, "candidate_episode": row["episode"],
        "bin_name": row["bin_name"],
        "candidate_identity_sha256": row["candidate_identity_sha256"],
        "candidate_plan_sha256": sha256(args.candidate_plan),
        "execution_protocol_sha256": sha256(args.execution_protocol),
        "carrier": carrier_receipt,
        "controller": "frozen_navdp_native_sidecar",
        "depth_source": "monocular_sidecar",
        "runtime_geometry": "content_addressed_pinned_navmesh",
        "runtime_navmesh_sha256": row["asset"]["navmesh_sha256"],
        "metric_depth_sensor_reads": 0,
        "max_steps": max_steps, "seed": seed,
        "reached_A": reached, "history_eligible": bool(
            reached and len(trace["poses"]) >= int(cfg["minimum_successful_history_frames"])),
        "steps_A": int(trace["steps"]),
        "final_goal_dist_A_m": float(trace["final_goal_dist_m"]),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "plans_sha256": sha256(plans_path),
        "native_control_audit": control, "depth_audit": depth_audit,
        "query_policy_outcomes_read": False,
    }
    path = output / "completion.json"
    path.write_text(json.dumps(completion, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n")
    print(json.dumps({"history_index": args.history_index,
                      "reached_A": reached,
                      "history_eligible": completion["history_eligible"],
                      "steps_A": completion["steps_A"]}, sort_keys=True))


if __name__ == "__main__":
    main()
