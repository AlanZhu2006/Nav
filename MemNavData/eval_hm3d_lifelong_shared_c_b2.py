#!/usr/bin/env python3
"""HM3D B2-only evaluation after an immutable actual full-mono C prefix."""

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
from eval_lifelong_shared_c_b2 import load_population_item, replay_c
from lifelong_shared_c_contract import ARMS, RESULT_SCHEMA, load_trace, require, sha256_file


args = base.args


def main() -> None:
    hm3d.validate_cli()
    require(args.lifelong_history_scope in ARMS, "unknown B2 treatment")
    require(bool(args.lifelong_shared_c_trace_root),
            "HM3D B2 evaluation requires shared C")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")
    scene_file = Path(args.scene).resolve()
    scene = base.SCENE_IDENTITY
    scene_root = Path(args.episode_root).resolve()
    wanted = [item.strip() for item in args.episode_ids.split(",") if item.strip()]
    require(len(wanted) == 1, "HM3D B2 evaluation requires one episode")
    episode_dir = scene_root / wanted[0]
    benchmark, benchmark_sha = hm3d.load_benchmark(episode_dir, scene)
    population_root = Path(args.lifelong_shared_c_trace_root).resolve()
    population_item, population_sha = load_population_item(
        population_root, scene, episode_dir.name)
    require(population_item["benchmark_sha256"] == benchmark_sha,
            "shared-C benchmark binding changed")
    trace_path = population_root / population_item["shared_C_trace"]
    trace = load_trace(trace_path, expected_sha256=population_item[
        "shared_C_trace_sha256"])
    require(trace["controller"] == "navdp", "HM3D C controller changed")

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
    goal_b = benchmark["goals"]["B"]
    image_b = (episode_dir / goal_b["rgb"]).read_bytes()
    image_c = (episode_dir / benchmark["goals"]["C"]["rgb"]).read_bytes()
    b_floor = np.asarray(goal_b["floor_position"], dtype=np.float64)
    b_yaw = float(goal_b["yaw_rad"])
    episode_seed = int(benchmark["episode_seed"])
    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    try:
        base.srv_reset(
            camera_height=float(receipt_a["camera_height_m"]),
            seed=episode_seed,
            episode_len=int(benchmark["online_A_steps"])
            + int(benchmark["online_B_steps"])
            + int(trace["steps_C"]) + int(args.max_steps),
            camera_intrinsic=intrinsic,
        )
        leg_a, replay_a = role_pair.replay_prefix(frozen_a)
        require(leg_a["reached"] and replay_a["all_rgb_hashes_verified"],
                "online-A replay failed")
        leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
            simulator, episode_dir, episode_dir.name, episode_seed, image_b,
            leg_a["end_pos"], leg_a["end_psi"])
        require(leg_b["reached"]
                and trace_b_sha == trace["online_B_trace_sha256"],
                "factual B replay changed")
        a_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
        b_ceiling = int(leg_b["memory_trace"][-1]["frame_idx"])
        require(a_ceiling == int(trace["online_A_candidate_ceiling"])
                and b_ceiling == int(trace["online_B_candidate_ceiling"]),
                "factual memory boundaries changed")
        reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)
        leg_c, c_session = replay_c(
            simulator, trace, image_c, leg_b["end_pos"], leg_b["end_psi"])
        b2_ceiling = (
            a_ceiling if args.lifelong_history_scope == "initial_leg_only"
            else b_ceiling)
        leg_b2, geo_b2 = life.run_query(
            simulator, simulator.pathfinder,
            leg_c["end_pos"], leg_c["end_psi"], image_b, b_floor, b_yaw,
            intrinsic, episode_seed, 3, b2_ceiling)
        b2_session = life.validate_query_session(
            "B2", leg_b2["plans"], 2, b2_ceiling)
        if args.lifelong_history_scope == "forced_reject_native":
            life.validate_forced_reject("B2", leg_b2["plans"])
        else:
            life.validate_not_forced("B2", leg_b2["plans"])
        depth = hm3d.query_depth_audit(leg_b2["plans"], "B2")
        memory_first, memory_last = life.ensure_contiguous_memory([
            ("A", leg_a), ("B", leg_b), ("C", leg_c), ("B2", leg_b2)])
        stats = life.plan_stats(leg_b2["plans"], a_ceiling, b_ceiling)
        metric = {
            "result_schema": RESULT_SCHEMA,
            "scene": scene, "episode": episode_dir.name,
            "controller": "navdp", "history_scope": args.lifelong_history_scope,
            "benchmark_sha256": benchmark_sha,
            "shared_C_population_sha256": population_sha,
            "shared_C_trace_sha256": sha256_file(trace_path),
            "shared_C_prefix_replayed": 1,
            "shared_C_start_x": float(trace["start_position"][0]),
            "shared_C_start_y": float(trace["start_position"][1]),
            "shared_C_start_z": float(trace["start_position"][2]),
            "shared_C_start_yaw": float(trace["start_yaw"]),
            "B2_start_x": float(leg_c["end_pos"][0]),
            "B2_start_y": float(leg_c["end_pos"][1]),
            "B2_start_z": float(leg_c["end_pos"][2]),
            "B2_start_yaw": float(leg_c["end_psi"]),
            "online_A_candidate_ceiling": a_ceiling,
            "online_B_candidate_ceiling": b_ceiling,
            "B2_candidate_ceiling": b2_ceiling,
            "reached_B2": int(leg_b2["reached"]),
            "steps_B2": int(leg_b2["steps"]),
            "len_B2": float(leg_b2["path_len"]),
            "geo_B2": float(geo_b2),
            "final_dist_B2": float(leg_b2["final_goal_dist_m"]),
            "B2_used_factual_B_anchor": int(stats["used_factual_B_anchor"]),
            "cec_takeovers_B2": int(stats["takeovers"]),
            "cec_shadow_takeovers_B2": int(stats["shadow_takeovers"]),
            "metric_depth_reads_B2": int(depth["metric_depth_reads"]),
            "memory_first_frame": memory_first,
            "memory_last_frame": memory_last,
            "runtime_role_visible": 0,
        }
        with (output / "metric.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric))
            writer.writeheader(); writer.writerow(metric)
        payload = {
            "result_schema": RESULT_SCHEMA,
            "scene": scene, "episode": episode_dir.name,
            "controller": "navdp", "history_scope": args.lifelong_history_scope,
            "shared_C_population_sha256": population_sha,
            "shared_C_trace_sha256": sha256_file(trace_path),
            "C_goal_session_replay": c_session,
            "B2_goal_session": b2_session,
            "navdp_short_fifo_reset_before_C": reset_receipt,
            "frozen_legA": leg_a["plans"],
            "frozen_legB": leg_b["plans"],
            "frozen_legC": trace["plans"],
            "B2": leg_b2["plans"],
            "rollout_traces": {
                "A": leg_a["rollout_trace"], "B": leg_b["rollout_trace"],
                "C": leg_c["rollout_trace"], "B2": leg_b2["rollout_trace"],
            },
            "memory_traces": {
                "A": leg_a["memory_trace"], "B": leg_b["memory_trace"],
                "C": leg_c["memory_trace"], "B2": leg_b2["memory_trace"],
            },
            "depth_audit": {"B2": depth},
        }
        (output / f"{episode_dir.name}_plans.json").write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (output / "summary.json").write_text(json.dumps({
            "result_schema": RESULT_SCHEMA,
            "controller": "navdp", "history_scope": args.lifelong_history_scope,
            "episodes": 1, "B2_success": int(leg_b2["reached"]),
            "shared_C_prefix_replayed": True,
            "metric_depth_reads_B2": int(depth["metric_depth_reads"]),
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
