#!/usr/bin/env python3
"""Collect one factual C prefix for a sealed HM3D full-mono A/B history."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
import eval_3leg_habitat as multigoal
import eval_hm3d_fullmono_lifelong as hm3d
import eval_shared_online_lifelong_nnr as life
import eval_shared_online_role_pairs as role_pair
from deterministic_eval_protocol import file_sha256
from lifelong_shared_c_contract import TRACE_SCHEMA, require, write_trace


args = base.args
RESULT_SCHEMA = "hm3d_lifelong_shared_c_collection_v1_20260825"


def main() -> None:
    hm3d.validate_cli()
    require(args.lifelong_history_scope == "all_prior",
            "HM3D shared-C collection has no B2 treatment")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")
    scene_file = Path(args.scene).resolve()
    scene = base.SCENE_IDENTITY
    scene_root = Path(args.episode_root).resolve()
    wanted = [item.strip() for item in args.episode_ids.split(",") if item.strip()]
    require(len(wanted) == 1, "HM3D shared-C collection requires one episode")
    episode_dir = scene_root / wanted[0]
    benchmark, benchmark_sha = hm3d.load_benchmark(episode_dir, scene)
    source_a = Path(benchmark["source_online_A_episode"])
    receipt_a = json.loads((source_a / "receipt.json").read_text())
    trace_a = json.loads((source_a / "online_a_trace.json").read_text())
    frozen_a = {"source": source_a, "receipt": receipt_a, "trace": trace_a}
    parquet = Path(receipt_a["source_episode"]) / (
        "data/chunk-000/episode_000000.parquet")
    require(file_sha256(parquet) == receipt_a["source_parquet_sha256"],
            "source camera-intrinsic parquet changed")
    rows = pd.read_parquet(parquet)
    intrinsic = np.stack([
        np.asarray(row, dtype=np.float64)
        for row in rows.iloc[0]["observation.camera_intrinsic"]
    ])
    image_b = (episode_dir / benchmark["goals"]["B"]["rgb"]).read_bytes()
    goal_c = benchmark["goals"]["C"]
    image_c = (episode_dir / goal_c["rgb"]).read_bytes()
    c_floor = np.asarray(goal_c["floor_position"], dtype=np.float64)
    c_yaw = float(goal_c["yaw_rad"])
    episode_seed = int(benchmark["episode_seed"])
    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    try:
        base.srv_reset(
            camera_height=float(receipt_a["camera_height_m"]),
            seed=episode_seed,
            episode_len=int(benchmark["online_A_steps"])
            + int(benchmark["online_B_steps"]) + int(args.max_steps),
            camera_intrinsic=intrinsic,
        )
        leg_a, replay_a = role_pair.replay_prefix(frozen_a)
        require(leg_a["reached"] and replay_a["all_rgb_hashes_verified"],
                "online-A replay failed")
        leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
            simulator, episode_dir, episode_dir.name, episode_seed, image_b,
            leg_a["end_pos"], leg_a["end_psi"])
        require(leg_b["reached"], "factual B is not successful")
        require(trace_b_sha == benchmark["online_B_trace_sha256"],
                "factual B trace changed")
        a_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
        b_ceiling = int(leg_b["memory_trace"][-1]["frame_idx"])
        reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)
        leg_c, geo_c = life.run_query(
            simulator, simulator.pathfinder,
            leg_b["end_pos"], leg_b["end_psi"], image_c, c_floor, c_yaw,
            intrinsic, episode_seed, 2, a_ceiling)
        session = life.validate_query_session(
            "C", leg_c["plans"], 1, a_ceiling)
        life.validate_not_forced("C", leg_c["plans"])
        life.ensure_contiguous_memory([
            ("A", leg_a), ("B", leg_b), ("C", leg_c)])
        require(int(session["goal_start_frame"]) == b_ceiling + 1,
                "C did not begin immediately after factual B")
        controllers = {
            str(row["cec_accept_controller"]) for row in leg_c["plans"]
            if row.get("cec_accept_controller") is not None
        }
        require(controllers == {"navdp"}, "HM3D C controller changed")
        trace_payload = {
            "schema_version": TRACE_SCHEMA,
            "scene": scene,
            "episode": episode_dir.name,
            "controller": "navdp",
            "benchmark_sha256": benchmark_sha,
            "online_A_trace_sha256": benchmark[
                "source_online_A_trace_sha256"],
            "online_B_trace_sha256": trace_b_sha,
            "episode_seed": episode_seed,
            "goal_C_sha256": base.bytes_sha256(image_c),
            "online_A_candidate_ceiling": a_ceiling,
            "online_B_candidate_ceiling": b_ceiling,
            "C_goal_start_frame": int(session["goal_start_frame"]),
            "C_candidate_ceiling": int(session["candidate_ceiling"]),
            "runtime_role_visible": False,
            "reached_C": bool(leg_c["reached"]),
            "geodesic_C_m": float(geo_c),
            "path_len_C_m": float(leg_c["path_len"]),
            "steps_C": int(leg_c["steps"]),
            "final_goal_dist_C_m": float(leg_c["final_goal_dist_m"]),
            "termination_reason": leg_c.get("termination_reason"),
            "start_position": np.asarray(leg_b["end_pos"], dtype=float).tolist(),
            "start_yaw": float(leg_b["end_psi"]),
            "end_position": np.asarray(leg_c["end_pos"], dtype=float).tolist(),
            "end_yaw": float(leg_c["end_psi"]),
            "poses": leg_c["rollout_trace"],
            "plans": leg_c["plans"],
            "memory_trace": leg_c["memory_trace"],
            "navdp_short_fifo_reset_receipt": reset_receipt,
            "B2_navigation_outcomes_read": False,
        }
        trace_path = output / f"{episode_dir.name}_shared_C_trace.json"
        trace_sha = write_trace(trace_path, trace_payload)
        metric = {
            "result_schema": RESULT_SCHEMA,
            "scene": scene,
            "episode": episode_dir.name,
            "controller": "navdp",
            "benchmark_sha256": benchmark_sha,
            "shared_C_trace_sha256": trace_sha,
            "reached_C": int(leg_c["reached"]),
            "steps_C": int(leg_c["steps"]),
            "len_C": float(leg_c["path_len"]),
            "geo_C": float(geo_c),
            "final_dist_C": float(leg_c["final_goal_dist_m"]),
            "online_A_candidate_ceiling": a_ceiling,
            "online_B_candidate_ceiling": b_ceiling,
            "B2_outcomes_read": 0,
        }
        with (output / "metric.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric))
            writer.writeheader(); writer.writerow(metric)
        plans = {
            "result_schema": RESULT_SCHEMA,
            "scene": scene, "episode": episode_dir.name,
            "controller": "navdp", "benchmark_sha256": benchmark_sha,
            "shared_C_trace_sha256": trace_sha,
            "memory_traces": {
                "A": leg_a["memory_trace"], "B": leg_b["memory_trace"],
                "C": leg_c["memory_trace"],
            },
            "C": {
                "reached": bool(leg_c["reached"]),
                "steps": int(leg_c["steps"]),
                "path_len_m": float(leg_c["path_len"]),
                "final_goal_dist_m": float(leg_c["final_goal_dist_m"]),
                "termination_reason": leg_c.get("termination_reason"),
                "end_position": np.asarray(
                    leg_c["end_pos"], dtype=float).tolist(),
                "end_yaw": float(leg_c["end_psi"]),
                "plans": leg_c["plans"],
                "rollout_trace": leg_c["rollout_trace"],
                "memory_trace": leg_c["memory_trace"],
            },
        }
        (output / f"{episode_dir.name}_plans.json").write_text(json.dumps(
            plans, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (output / "summary.json").write_text(json.dumps({
            "result_schema": RESULT_SCHEMA,
            "controller": "navdp", "episodes": 1,
            "C_successes": int(leg_c["reached"]),
            "B2_outcomes_read": False,
            "shared_C_trace_sha256": trace_sha,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
