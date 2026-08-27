#!/usr/bin/env python3
"""Collect one controller-specific factual C prefix before any B2 arm runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
import eval_3leg_habitat as multigoal
import eval_shared_online_lifelong_nnr as lifelong
import eval_shared_online_novel_revisit as nnr
from deterministic_eval_protocol import file_sha256
from lifelong_shared_c_contract import (
    TRACE_SCHEMA,
    require,
    write_trace,
)


args = base.args
RESULT_SCHEMA = "lifelong_shared_c_collection_v1_20260825"


def selected_episode(scene_root: Path) -> tuple[dict, Path, str]:
    manifest = nnr.load_manifest(scene_root)
    manifest_rows = {row["episode"]: row for row in manifest["accepted"]}
    wanted = [
        item.strip() for item in args.episode_ids.split(",") if item.strip()
    ]
    require(len(wanted) == 1, "shared-C collection requires one episode")
    episode_dir = scene_root / wanted[0]
    require((episode_dir / "benchmark.json").is_file(),
            "shared-C benchmark is missing")
    benchmark, benchmark_sha = nnr.load_benchmark(
        episode_dir, manifest_rows)
    return benchmark, episode_dir, benchmark_sha


def main() -> None:
    lifelong.validate_cli()
    require(args.lifelong_history_scope == "all_prior",
            "shared-C collection has no B2 treatment scope")
    require(not args.lifelong_shared_c_trace_root,
            "shared-C collection cannot consume a prior C trace")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")

    scene_file = Path(args.scene).resolve()
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene_file.stem, "scene/benchmark mismatch")
    benchmark, episode_dir, benchmark_sha = selected_episode(scene_root)
    source, image_a, image_b, intrinsic = lifelong.source_assets(benchmark)
    require(file_sha256(scene_file) == benchmark["source_scene_asset_sha256"],
            "scene asset changed")
    trace_root = Path(args.shared_leg1_trace_root).resolve()
    sealed_trace_root = lifelong.remap_sealed_path(benchmark["trace_root"])
    require(trace_root == sealed_trace_root.resolve(),
            "trace root differs from sealed benchmark")
    trace_a_path = trace_root / benchmark["online_a_trace"]
    trace_b_path = trace_root / benchmark["online_b_trace"]
    require(file_sha256(trace_a_path) == benchmark["online_a_trace_sha256"],
            "online-A trace changed")
    require(file_sha256(trace_b_path) == benchmark["online_b_trace_sha256"],
            "online-B trace changed")

    rows = pd.read_parquet(source / "data/chunk-000/episode_000000.parquet")
    start_position, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])
    c_goal = benchmark["goal_c"]
    c_floor = np.asarray(c_goal["floor_position"], dtype=np.float64)
    c_yaw = float(c_goal["yaw_rad"])
    image_c_path = episode_dir / benchmark["goal_c_asset"]["rgb"]
    require(file_sha256(image_c_path)
            == benchmark["goal_c_asset"]["rgb_sha256"], "Goal C image changed")
    image_c = image_c_path.read_bytes()
    episode_seed = int(benchmark["episode_seed"])

    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    try:
        base.srv_reset(
            camera_height=float(benchmark["camera_height_m"]),
            seed=episode_seed,
            episode_len=(int(benchmark["online_a_steps"])
                         + int(benchmark["online_b_steps"])
                         + int(args.max_steps)),
            camera_intrinsic=intrinsic,
        )
        leg_a, trace_a_sha = base.replay_shared_leg1(
            simulator, trace_root, episode_dir.name, episode_seed,
            image_a, start_position, start_yaw)
        require(leg_a["reached"], "sealed online A is not successful")
        leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
            simulator, trace_root, episode_dir.name, episode_seed,
            image_b, leg_a["end_pos"], leg_a["end_psi"])
        require(leg_b["reached"], "sealed online B is not successful")
        require(trace_a_sha == benchmark["online_a_trace_sha256"]
                and trace_b_sha == benchmark["online_b_trace_sha256"],
                "factual A/B trace binding changed")
        a_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
        b_ceiling = int(leg_b["memory_trace"][-1]["frame_idx"])
        require(b_ceiling > a_ceiling, "online B did not extend memory")

        reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)
        leg_c, geo_c = lifelong.run_query(
            simulator, simulator.pathfinder,
            leg_b["end_pos"], leg_b["end_psi"], image_c, c_floor, c_yaw,
            intrinsic, episode_seed, 2, a_ceiling)
        session = lifelong.validate_query_session(
            "C", leg_c["plans"], 1, a_ceiling)
        lifelong.validate_not_forced("C", leg_c["plans"])
        memory_first, memory_last = lifelong.ensure_contiguous_memory([
            ("A", leg_a), ("B", leg_b), ("C", leg_c),
        ])
        require(memory_first == 0 and memory_last == b_ceiling + len(
            leg_c["memory_trace"]), "shared-C memory extent changed")
        require(int(session["goal_start_frame"]) == b_ceiling + 1,
                "C goal session did not start after factual B")
        controllers = {
            str(row["cec_accept_controller"])
            for row in leg_c["plans"]
            if row.get("cec_accept_controller") is not None
        }
        require(len(controllers) == 1,
                "shared-C plans do not identify one controller")
        controller = next(iter(controllers))
        trace_payload = {
            "schema_version": TRACE_SCHEMA,
            "scene": scene_file.stem,
            "episode": episode_dir.name,
            "controller": controller,
            "benchmark_sha256": benchmark_sha,
            "online_A_trace_sha256": trace_a_sha,
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
            "scene": scene_file.stem,
            "episode": episode_dir.name,
            "controller": controller,
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
            writer.writeheader()
            writer.writerow(metric)
        plans = {
            "schema_version": RESULT_SCHEMA,
            "scene": scene_file.stem,
            "episode": episode_dir.name,
            "controller": controller,
            "benchmark_sha256": benchmark_sha,
            "shared_C_trace_sha256": trace_sha,
            "memory_traces": {
                "A": leg_a["memory_trace"],
                "B": leg_b["memory_trace"],
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
        summary = {
            "schema_version": RESULT_SCHEMA,
            "controller": controller,
            "episodes": 1,
            "C_successes": int(leg_c["reached"]),
            "B2_outcomes_read": False,
            "shared_C_trace_sha256": trace_sha,
        }
        (output / "summary.json").write_text(json.dumps(
            summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
