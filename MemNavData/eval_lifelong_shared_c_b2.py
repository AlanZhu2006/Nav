#!/usr/bin/env python3
"""Evaluate B2 after replaying one immutable controller-specific C prefix."""

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
    ARMS,
    POPULATION_SCHEMA,
    RESULT_SCHEMA,
    load_trace,
    require,
    sha256_file,
)


args = base.args


def load_population_item(root: Path, scene: str, episode: str) -> tuple[dict, str]:
    population_path = root / "population.json"
    require((root / "SEALED").is_file(), "shared-C population is not sealed")
    require((root / "population.json.sha256").read_text().split()[0]
            == sha256_file(population_path), "shared-C population hash changed")
    payload = json.loads(population_path.read_text())
    require(payload.get("schema_version") == POPULATION_SCHEMA,
            "shared-C population schema changed")
    require(payload.get("selection_reads_B2_navigation_outcomes") is False,
            "shared-C population was selected after B2")
    matches = [
        row for row in payload["accepted"]
        if row["scene"] == scene and row["episode"] == episode
    ]
    require(len(matches) == 1, "shared-C population identity is ambiguous")
    return matches[0], sha256_file(population_path)


def replay_c(
    simulator, trace: dict, image_c: bytes, expected_start_position,
    expected_start_yaw: float,
) -> tuple[dict, dict]:
    require(base.bytes_sha256(image_c) == trace["goal_C_sha256"],
            "shared-C goal image changed")
    require(np.allclose(
        np.asarray(trace["start_position"], dtype=float),
        np.asarray(expected_start_position, dtype=float),
        rtol=0.0, atol=1e-6), "shared-C start position changed")
    require(abs(base.wrap_angle(
        float(trace["start_yaw"]) - float(expected_start_yaw))) <= 1e-6,
        "shared-C start yaw changed")
    start_frame = int(trace["C_goal_start_frame"])
    session_receipt = base.srv_replay_goal_session(image_c, start_frame)
    plan_steps = {int(row["step"]) for row in trace["plans"]}
    memory_trace = []
    navdp = None
    for pose in trace["poses"]:
        position = np.asarray(
            [pose["x"], pose["y"], pose["z"]], dtype=float)
        rgb, _depth = base.render(
            simulator,
            position + np.asarray([0.0, base.CAM_H, 0.0]),
            float(pose["yaw"]),
        )
        image = base.jpg_bytes(rgb)
        require(base.bytes_sha256(image) == pose["jpg_sha256"],
                f"shared-C RGB changed at step {pose['step']}")
        receipt = base.srv_memory(image)
        expected_frame = start_frame + len(memory_trace)
        require(int(receipt.get("frame_idx", -1)) == expected_frame,
                "shared-C memory replay index changed")
        memory_trace.append({
            "frame_idx": expected_frame,
            "step": int(pose["step"]),
            "x": float(pose["x"]),
            "z": float(pose["z"]),
            "yaw": float(pose["yaw"]),
        })
        if int(pose["step"]) in plan_steps:
            navdp = base.srv_navdp_memory_replay(image)
    require(memory_trace == trace["memory_trace"],
            "shared-C memory replay differs from sealed trace")
    require(navdp is not None, "shared-C replay restored no decision frames")
    memory_size = int(navdp.get("memory_size", -1))
    require(memory_size > 0, "controller replay omitted bounded memory size")
    expected_queue = min(len(plan_steps), memory_size)
    require(navdp.get("queue_lengths") == [expected_queue],
            "shared-C controller queue length changed")
    leg = {
        "reached": True,
        "path_len": float(trace["path_len_C_m"]),
        "steps": int(trace["steps_C"]),
        "termination_reason": trace.get("termination_reason"),
        "plans": trace["plans"],
        "memory_trace": memory_trace,
        "rollout_trace": trace["poses"],
        "end_pos": np.asarray(trace["end_position"], dtype=float),
        "end_psi": float(trace["end_yaw"]),
        "final_goal_dist_m": float(trace["final_goal_dist_C_m"]),
    }
    return leg, session_receipt


def main() -> None:
    lifelong.validate_cli()
    require(args.lifelong_history_scope in ARMS,
            "unknown shared-C B2 treatment")
    require(bool(args.lifelong_shared_c_trace_root),
            "shared-C B2 evaluation requires a sealed C population")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")

    scene_file = Path(args.scene).resolve()
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene_file.stem, "scene/benchmark mismatch")
    manifest = nnr.load_manifest(scene_root)
    manifest_rows = {row["episode"]: row for row in manifest["accepted"]}
    wanted = [item.strip() for item in args.episode_ids.split(",") if item.strip()]
    require(len(wanted) == 1, "shared-C B2 evaluation requires one episode")
    episode_dir = scene_root / wanted[0]
    benchmark, benchmark_sha = nnr.load_benchmark(episode_dir, manifest_rows)
    population_root = Path(args.lifelong_shared_c_trace_root).resolve()
    population_item, population_sha = load_population_item(
        population_root, scene_file.stem, episode_dir.name)
    require(population_item["benchmark_sha256"] == benchmark_sha,
            "shared-C population benchmark changed")
    trace_path = population_root / population_item["shared_C_trace"]
    trace = load_trace(
        trace_path, expected_sha256=population_item["shared_C_trace_sha256"])

    source, image_a, image_b, intrinsic = lifelong.source_assets(benchmark)
    require(file_sha256(scene_file) == benchmark["source_scene_asset_sha256"],
            "scene asset changed")
    trace_root = Path(args.shared_leg1_trace_root).resolve()
    require(trace_root == lifelong.remap_sealed_path(
        benchmark["trace_root"]).resolve(), "factual trace root changed")
    metadata = json.loads((source / "meta/gen_meta.json").read_text())
    rows = pd.read_parquet(source / "data/chunk-000/episode_000000.parquet")
    start_position, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])
    b_goal = metadata["goals"][0]
    b_floor = base.data_to_hab(b_goal["pos"])
    b_yaw = float(b_goal["yaw_habitat"])
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
                         + int(trace["steps_C"]) + int(args.max_steps)),
            camera_intrinsic=intrinsic,
        )
        leg_a, trace_a_sha = base.replay_shared_leg1(
            simulator, trace_root, episode_dir.name, episode_seed,
            image_a, start_position, start_yaw)
        leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
            simulator, trace_root, episode_dir.name, episode_seed,
            image_b, leg_a["end_pos"], leg_a["end_psi"])
        require(leg_a["reached"] and leg_b["reached"],
                "sealed factual A/B prefix is not successful")
        require(trace_a_sha == trace["online_A_trace_sha256"]
                and trace_b_sha == trace["online_B_trace_sha256"],
                "shared-C trace is bound to another A/B prefix")
        a_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
        b_ceiling = int(leg_b["memory_trace"][-1]["frame_idx"])
        require(a_ceiling == int(trace["online_A_candidate_ceiling"])
                and b_ceiling == int(trace["online_B_candidate_ceiling"]),
                "factual memory ceiling changed")
        reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)
        leg_c, c_session_replay = replay_c(
            simulator, trace, image_c, leg_b["end_pos"], leg_b["end_psi"])

        b2_ceiling = (
            a_ceiling if args.lifelong_history_scope == "initial_leg_only"
            else b_ceiling
        )
        leg_b2, geo_b2 = lifelong.run_query(
            simulator, simulator.pathfinder,
            leg_c["end_pos"], leg_c["end_psi"], image_b, b_floor, b_yaw,
            intrinsic, episode_seed, 3, b2_ceiling)
        b2_session = lifelong.validate_query_session(
            "B2", leg_b2["plans"], 2, b2_ceiling)
        if args.lifelong_history_scope == "forced_reject_native":
            lifelong.validate_forced_reject("B2", leg_b2["plans"])
        else:
            lifelong.validate_not_forced("B2", leg_b2["plans"])
        memory_first, memory_last = lifelong.ensure_contiguous_memory([
            ("A", leg_a), ("B", leg_b), ("C", leg_c), ("B2", leg_b2),
        ])
        stats = lifelong.plan_stats(leg_b2["plans"], a_ceiling, b_ceiling)
        metric = {
            "result_schema": RESULT_SCHEMA,
            "scene": scene_file.stem,
            "episode": episode_dir.name,
            "controller": trace["controller"],
            "history_scope": args.lifelong_history_scope,
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
            "memory_first_frame": memory_first,
            "memory_last_frame": memory_last,
            "runtime_role_visible": 0,
        }
        with (output / "metric.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric))
            writer.writeheader()
            writer.writerow(metric)
        payload = {
            "result_schema": RESULT_SCHEMA,
            "scene": scene_file.stem,
            "episode": episode_dir.name,
            "controller": trace["controller"],
            "history_scope": args.lifelong_history_scope,
            "shared_C_population_sha256": population_sha,
            "shared_C_trace_sha256": sha256_file(trace_path),
            "C_goal_session_replay": c_session_replay,
            "B2_goal_session": b2_session,
            "navdp_short_fifo_reset_before_C": reset_receipt,
            "frozen_legA": leg_a["plans"],
            "frozen_legB": leg_b["plans"],
            "frozen_legC": trace["plans"],
            "B2": leg_b2["plans"],
            "rollout_traces": {
                "A": leg_a["rollout_trace"],
                "B": leg_b["rollout_trace"],
                "C": leg_c["rollout_trace"],
                "B2": leg_b2["rollout_trace"],
            },
            "memory_traces": {
                "A": leg_a["memory_trace"],
                "B": leg_b["memory_trace"],
                "C": leg_c["memory_trace"],
                "B2": leg_b2["memory_trace"],
            },
        }
        (output / f"{episode_dir.name}_plans.json").write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        summary = {
            "result_schema": RESULT_SCHEMA,
            "controller": trace["controller"],
            "history_scope": args.lifelong_history_scope,
            "episodes": 1,
            "B2_success": int(leg_b2["reached"]),
            "shared_C_prefix_replayed": True,
        }
        (output / "summary.json").write_text(json.dumps(
            summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
