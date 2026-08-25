#!/usr/bin/env python3
"""CEC accumulation evaluation after a sealed full-mono HM3D A->B prefix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
import eval_3leg_habitat as multigoal
import eval_shared_online_lifelong_nnr as life
import eval_shared_online_novel_revisit as nnr
import eval_shared_online_role_pairs as role_pair
from deterministic_eval_protocol import file_sha256
from hm3d_fullmono_lifelong import PREFIX_SCHEMA, QUERY_NAMES, RESULT_SCHEMA


args = base.args


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_cli() -> None:
    nnr.validate_cli()
    require(args.shared_online_nnr_arm == "cec_portability",
            "full-mono lifelong requires the role-free CEC hub")
    require(args.lifelong_history_scope in (
        "all_prior", "initial_leg_only", "forced_reject_native"
    ), "unknown lifelong history scope")
    require(args.navdp_depth_source == "monocular_sidecar",
            "full-mono query depth source changed")


def load_benchmark(episode_dir: Path, expected_scene: str) -> tuple[dict, str]:
    path = episode_dir / "benchmark.json"
    payload = json.loads(path.read_text())
    require(payload.get("schema_version") == PREFIX_SCHEMA,
            "lifelong benchmark schema changed")
    require(payload["scene"] == expected_scene
            and payload["episode"] == episode_dir.name,
            "lifelong benchmark identity changed")
    require(payload.get("runtime_role_visibility") == "none",
            "runtime role visibility changed")
    require(payload.get("query_outcomes_read") is False,
            "benchmark was constructed after query outcomes")
    source = Path(payload["source_online_A_episode"])
    require(file_sha256(source / "receipt.json")
            == payload["source_online_A_receipt_sha256"],
            "online-A receipt changed")
    require(file_sha256(source / "online_a_trace.json")
            == payload["source_online_A_trace_sha256"],
            "online-A trace changed")
    trace_b = episode_dir / payload["online_B_trace"]
    require(file_sha256(trace_b) == payload["online_B_trace_sha256"],
            "online-B trace changed")
    factual_b = episode_dir / payload["factual_B_completion"]
    require(file_sha256(factual_b)
            == payload["factual_B_completion_sha256"],
            "factual-B full-mono receipt changed")
    factual_b_payload = json.loads(factual_b.read_text())
    require(factual_b_payload.get("controller")
            == "frozen_navdp_native_sidecar"
            and factual_b_payload.get("navdp_depth_source")
            == "monocular_sidecar"
            and int(factual_b_payload.get("metric_depth_sensor_reads", -1)) == 0,
            "factual-B prefix is not full-mono native NavDP")
    for name in ("B", "C"):
        goal = payload["goals"][name]
        require(file_sha256(episode_dir / goal["rgb"]) == goal["rgb_sha256"],
                f"Goal-{name} RGB changed")
        require(file_sha256(episode_dir / goal["depth"]) == goal["depth_sha256"],
                f"Goal-{name} depth changed")
    return payload, file_sha256(path)


def query_depth_audit(plans: list[dict], name: str) -> dict:
    decisions = [row for row in plans if row.get("navdp_depth_source") is not None]
    require(bool(decisions), f"{name}: no NavDP decisions")
    require(all(row.get("navdp_depth_source") == "monocular_sidecar"
                for row in decisions), f"{name}: depth source changed")
    reads = sum(row.get("metric_depth_sensor_consumed") is True
                for row in decisions)
    require(reads == 0, f"{name}: metric depth was consumed")
    return {"navdp_decisions": len(decisions), "metric_depth_reads": reads}


def main() -> None:
    validate_cli()
    if args.contract_dry_run:
        print(
            "[hm3d-fullmono-lifelong] contract_dry_run OK: "
            f"scope={args.lifelong_history_scope} "
            f"depth={args.navdp_depth_source} max_steps={args.max_steps}"
        )
        return
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")
    scene_file = Path(args.scene).resolve()
    scene = base.SCENE_IDENTITY
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene, "scene/benchmark identity mismatch")
    episode_dirs = sorted(
        path for path in scene_root.glob("episode_*")
        if (path / "benchmark.json").is_file()
    )
    if args.episode_ids:
        wanted = {value.strip() for value in args.episode_ids.split(",")
                  if value.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    require(bool(episode_dirs), "no lifelong episodes selected")

    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        for episode_dir in episode_dirs:
            benchmark, benchmark_sha = load_benchmark(episode_dir, scene)
            require(file_sha256(scene_file) == benchmark["source_scene_asset_sha256"],
                    "scene asset changed")
            source_a = Path(benchmark["source_online_A_episode"])
            receipt_a = json.loads((source_a / "receipt.json").read_text())
            trace_a = json.loads((source_a / "online_a_trace.json").read_text())
            frozen_a = {"source": source_a, "receipt": receipt_a, "trace": trace_a}
            parquet = Path(receipt_a["source_episode"]) / (
                "data/chunk-000/episode_000000.parquet"
            )
            require(file_sha256(parquet) == receipt_a["source_parquet_sha256"],
                    "source camera-intrinsic parquet changed")
            rows = pd.read_parquet(parquet)
            intrinsic = np.stack([
                np.asarray(row, dtype=np.float64)
                for row in rows.iloc[0]["observation.camera_intrinsic"]
            ])
            image_a = (source_a / "goal_a.jpg").read_bytes()
            goal_b = benchmark["goals"]["B"]
            goal_c = benchmark["goals"]["C"]
            image_b = (episode_dir / goal_b["rgb"]).read_bytes()
            image_c = (episode_dir / goal_c["rgb"]).read_bytes()
            b_floor = np.asarray(goal_b["floor_position"], dtype=np.float64)
            c_floor = np.asarray(goal_c["floor_position"], dtype=np.float64)
            b_yaw = float(goal_b["yaw_rad"])
            c_yaw = float(goal_c["yaw_rad"])
            episode_seed = int(benchmark["episode_seed"])
            base.srv_reset(
                camera_height=float(receipt_a["camera_height_m"]),
                seed=episode_seed,
                episode_len=(
                    int(benchmark["online_A_steps"])
                    + int(benchmark["online_B_steps"])
                    + 3 * int(args.max_steps)
                ),
                camera_intrinsic=intrinsic,
            )
            leg_a, replay_a = role_pair.replay_prefix(frozen_a)
            require(leg_a["reached"], "sealed online A is not successful")
            require(replay_a["all_rgb_hashes_verified"],
                    "online-A RGB replay failed")
            leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
                simulator,
                episode_dir,
                episode_dir.name,
                episode_seed,
                image_b,
                leg_a["end_pos"],
                leg_a["end_psi"],
            )
            require(leg_b["reached"], "sealed factual B is not successful")
            require(trace_b_sha == benchmark["online_B_trace_sha256"],
                    "factual B trace hash changed")
            a_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
            b_ceiling = int(leg_b["memory_trace"][-1]["frame_idx"])
            require(a_ceiling + 1 == int(benchmark["online_A_steps"]),
                    "online-A memory boundary changed")
            require(b_ceiling + 1 == int(benchmark["online_A_steps"])
                    + int(benchmark["online_B_steps"]),
                    "online-B memory boundary changed")
            reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)

            leg_c, geo_c = life.run_query(
                simulator, pathfinder, leg_b["end_pos"], leg_b["end_psi"],
                image_c, c_floor, c_yaw, intrinsic, episode_seed, 2, a_ceiling,
            )
            legs = {"C": leg_c}
            geodesics = {"C": geo_c}
            receipts = [life.validate_query_session(
                "C", leg_c["plans"], 1, a_ceiling
            )] if leg_c["plans"] else []
            if leg_c["reached"]:
                b2_ceiling = (
                    b_ceiling
                    if args.lifelong_history_scope in (
                        "all_prior", "forced_reject_native"
                    ) else a_ceiling
                )
                leg_b2, geo_b2 = life.run_query(
                    simulator, pathfinder, leg_c["end_pos"], leg_c["end_psi"],
                    image_b, b_floor, b_yaw, intrinsic, episode_seed, 3,
                    b2_ceiling,
                )
                receipts.append(life.validate_query_session(
                    "B2", leg_b2["plans"], 2, b2_ceiling
                ))
            else:
                leg_b2 = multigoal.empty_leg(
                    leg_c["end_pos"], leg_c["end_psi"], b_floor[[0, 2]]
                )
                geo_b2 = float("nan")
            legs["B2"], geodesics["B2"] = leg_b2, geo_b2

            if leg_c["reached"] and leg_b2["reached"]:
                c2_override = (
                    None
                    if args.lifelong_history_scope in (
                        "all_prior", "forced_reject_native"
                    ) else a_ceiling
                )
                leg_c2, geo_c2 = life.run_query(
                    simulator, pathfinder, leg_b2["end_pos"], leg_b2["end_psi"],
                    image_c, c_floor, c_yaw, intrinsic, episode_seed, 4,
                    c2_override,
                )
                expected_c2 = (
                    int(leg_c2["plans"][0]["cec_goal_start_frame"]) - 1
                    if c2_override is None else a_ceiling
                )
                receipts.append(life.validate_query_session(
                    "C2", leg_c2["plans"], 3, expected_c2
                ))
            else:
                leg_c2 = multigoal.empty_leg(
                    leg_b2["end_pos"], leg_b2["end_psi"], c_floor[[0, 2]]
                )
                geo_c2 = float("nan")
            legs["C2"], geodesics["C2"] = leg_c2, geo_c2

            for name in QUERY_NAMES:
                if not legs[name]["plans"]:
                    continue
                if args.lifelong_history_scope == "forced_reject_native":
                    life.validate_forced_reject(name, legs[name]["plans"])
                else:
                    life.validate_not_forced(name, legs[name]["plans"])
            memory_legs = [("A", leg_a), ("B", leg_b)] + [
                (name, legs[name]) for name in QUERY_NAMES
                if legs[name]["memory_trace"]
            ]
            memory_first, memory_last = life.ensure_contiguous_memory(memory_legs)
            stats = {
                name: life.plan_stats(legs[name]["plans"], a_ceiling, b_ceiling)
                for name in QUERY_NAMES
            }
            depth = {
                name: query_depth_audit(legs[name]["plans"], name)
                for name in QUERY_NAMES if legs[name]["plans"]
            }
            reached = [bool(legs[name]["reached"]) for name in QUERY_NAMES]
            completed = 0
            for success in reached:
                if not success:
                    break
                completed += 1
            metric = {
                "result_schema": RESULT_SCHEMA,
                "scene": scene,
                "episode": episode_dir.name,
                "benchmark_sha256": benchmark_sha,
                "history_scope": args.lifelong_history_scope,
                "runtime_role_visible": 0,
                "online_A_trace_sha256": benchmark[
                    "source_online_A_trace_sha256"
                ],
                "online_B_trace_sha256": trace_b_sha,
                "online_A_candidate_ceiling": a_ceiling,
                "online_B_candidate_ceiling": b_ceiling,
                "memory_first_frame": memory_first,
                "memory_last_frame": memory_last,
                "reached_C": int(reached[0]),
                "reached_B2": int(reached[1]),
                "reached_C2": int(reached[2]),
                "evaluated_B2": int(reached[0]),
                "evaluated_C2": int(all(reached[:2])),
                "queries_completed_before_first_failure": completed,
                "query_joint_success": int(all(reached)),
                "B2_used_factual_B_anchor": int(
                    stats["B2"]["used_factual_B_anchor"]
                ),
                "B_goal_max_factual_B_covis": float(
                    goal_b["max_factual_B_covis"]
                ),
                "B_goal_strong_support": int(
                    benchmark["B_goal_strong_support"]
                ),
                "metric_depth_reads_queries": sum(
                    row["metric_depth_reads"] for row in depth.values()
                ),
                "navdp_short_FIFO_reset_before_C": 1,
                "navdp_short_FIFO_reset_receipt": json.dumps(
                    reset_receipt, sort_keys=True
                ),
            }
            for name in QUERY_NAMES:
                leg = legs[name]
                metric[f"geo_{name}"] = geodesics[name]
                metric[f"steps_{name}"] = int(leg["steps"])
                metric[f"len_{name}"] = float(leg["path_len"])
                metric[f"final_dist_{name}"] = float(leg["final_goal_dist_m"])
                metric[f"cec_takeovers_{name}"] = stats[name]["takeovers"]
                metric[f"cec_shadow_takeovers_{name}"] = stats[name][
                    "shadow_takeovers"
                ]
            metrics.append(metric)
            plan_payload = {
                "result_schema": RESULT_SCHEMA,
                "history_scope": args.lifelong_history_scope,
                "runtime_role_visible": False,
                "benchmark_sha256": benchmark_sha,
                "goal_session_receipts": receipts,
                "frozen_legA": leg_a["plans"],
                "frozen_legB": leg_b["plans"],
                "queries": {
                    name: legs[name]["plans"] for name in QUERY_NAMES
                },
                "rollout_traces": {
                    "A": leg_a["rollout_trace"],
                    "B": leg_b["rollout_trace"],
                    **{name: legs[name]["rollout_trace"] for name in QUERY_NAMES},
                },
                "memory_traces": {
                    "A": leg_a["memory_trace"],
                    "B": leg_b["memory_trace"],
                    **{name: legs[name]["memory_trace"] for name in QUERY_NAMES},
                },
                "depth_audit": depth,
            }
            (output / f"{episode_dir.name}_plans.json").write_text(json.dumps(
                plan_payload, indent=2, sort_keys=True, allow_nan=False
            ) + "\n")
            with (output / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{scene}/{episode_dir.name}] "
                f"scope={args.lifelong_history_scope} "
                f"queries={''.join(str(int(value)) for value in reached)}"
            )
        summary = {
            "result_schema": RESULT_SCHEMA,
            "history_scope": args.lifelong_history_scope,
            "episodes": len(metrics),
            "runtime_role_visible": False,
            "frozen_actual_fullmono_prefix": "Novel_A_then_Novel_B",
            "query_sequence": list(QUERY_NAMES),
            "C_success": sum(row["reached_C"] for row in metrics),
            "B2_success_given_C": sum(
                row["reached_B2"] for row in metrics if row["reached_C"]
            ),
            "C2_success_given_CB2": sum(
                row["reached_C2"] for row in metrics
                if row["reached_C"] and row["reached_B2"]
            ),
            "query_joint_success": sum(
                row["query_joint_success"] for row in metrics
            ),
            "metric_depth_reads_queries": sum(
                row["metric_depth_reads_queries"] for row in metrics
            ),
            "claim_scope": (
                "consumed-scene full-mono lifelong accumulation confirmation"
            ),
        }
        (output / "summary.json").write_text(json.dumps(
            summary, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        print(json.dumps(summary, sort_keys=True))
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
