#!/usr/bin/env python3
"""Paired Revisit-C evaluation after a frozen online A->Novel-B prefix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
import eval_3leg_habitat as multigoal
from build_shared_online_novel_revisit import ROLE_SEQUENCE, SCHEMA_VERSION
from deterministic_eval_protocol import file_sha256
from navdp_goal_switch import should_reset_before_leg


args = base.args
RESULT_SCHEMA = "shared_online_novel_revisit_eval_v1_20260813"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_cli() -> None:
    require(args.shared_online_nnr_arm is not None, "--shared_online_nnr_arm is required")
    require(args.leg1_mode == "shared_trace", "frozen A/B requires shared_trace")
    require(bool(args.shared_leg1_trace_root), "shared A/B trace root is required")
    require(not args.write_leg1_trace, "frozen A/B traces cannot be overwritten")
    require(args.leg1_goal_source == "own", "goal swapping is forbidden")
    require(not args.stop_after_leg1 and not args.reset_memory, "prefix must be preserved")
    require(
        args.server_backend in ("hybrid_pose", "cec_portability"),
        "all arms require causal memory replay",
    )
    if args.server_backend == "hybrid_pose":
        require(args.novel_port is not None, "hybrid runtime requires --novel_port")
    require(args.deterministic_plan_seeds, "paired C requires deterministic plan seeds")
    require(args.trajectory_selector == "server", "trajectory oracle is forbidden")
    require(args.retrieval_override == "off", "retrieval oracle is forbidden")
    require(args.gate_override is None, "gate oracle is forbidden")
    require(args.oracle_candidate_seed_count == 1, "candidate pooling is forbidden")
    require(args.oracle_global_subgoal_m == 0.0, "global subgoal oracle is forbidden")
    require(args.oracle_observed_frontier == "off", "frontier oracle is forbidden")
    require(args.terminal_uturn == "off", "position SR forbids terminal U-turn")
    require(args.terminal_visual_refine == "off", "visual refinement is forbidden")
    require(
        args.double_revisit_c_history == "initial_leg_only",
        "C retrieval must stop at the online-A boundary",
    )
    require(
        args.navdp_goal_switch_reset == "before_c",
        "long-memory C requires the frozen pre-C NavDP FIFO reset",
    )
    require(args.exec_horizon == 8, "formal NavDP execution horizon is 8")
    require(args.certified_cdec_rescue == "off", "CDEC rescue is a separate ablation")

    expected_stagnation_mode = {
        "native": "off",
        "known_direct": "off",
        "certified": "off",
        "certified_budget": "budget_control",
        "certified_graph": "rescue",
        "cec_portability": "off",
    }[args.shared_online_nnr_arm]
    require(
        args.certified_stagnation_graph == expected_stagnation_mode,
        "shared-online arm/stagnation mode mismatch",
    )

    if args.shared_online_nnr_arm == "cec_portability":
        require(
            args.server_backend == "cec_portability",
            "CEC portability arm requires the all-CEC hub backend",
        )
        require(
            args.hybrid_route == "phase" and args.revisit_adapter == "legacy_metric",
            "CEC portability routing is owned entirely by the hub",
        )
    elif args.shared_online_nnr_arm in ("native", "known_direct"):
        require(args.hybrid_route == "phase", "native/direct arms require phase route")
        require(args.revisit_adapter == "legacy_metric", "native/direct use legacy adapter config")
    else:
        require(
            args.hybrid_route == "certified_relocalization",
            "certified arm requires certified_relocalization route",
        )
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "certified arm requires the scale-free bearing adapter",
        )
    if args.shared_online_nnr_arm != "cec_portability":
        base.validate_revisit_adapter_configuration(
            mode=args.revisit_adapter,
            server_backend=args.server_backend,
            revisit_controller=args.revisit_controller,
            router_is_automatic_geometry=(args.hybrid_route in base.AUTO_HYBRID_ROUTES),
            router_is_certified_relocalization=(
                args.hybrid_route == "certified_relocalization"
            ),
        )


def controller_backend() -> str:
    return {
        "native": "navdp",
        "known_direct": "navdp_mix",
        "certified": "navdp_auto",
        "certified_budget": "navdp_auto",
        "certified_graph": "navdp_auto",
        "cec_portability": "navdp",
    }[args.shared_online_nnr_arm]


def router_stats(plans: list[dict]) -> dict:
    decisions = []
    for plan in plans:
        decision = plan.get("router_active")
        if decision is None:
            decision = plan.get("cec_takeover")
        if decision is not None:
            decisions.append(bool(decision))
    return {
        "plans": len(decisions),
        "active_plans": sum(decisions),
        "active_episode": bool(any(decisions)),
    }


def load_manifest(scene_root: Path) -> dict:
    manifest_path = scene_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == SCHEMA_VERSION, "wrong benchmark schema")
    require(manifest.get("scene") == Path(args.scene).stem, "manifest scene mismatch")
    require(manifest.get("selected_before_c_navigation") is True, "selection seal missing")
    return manifest


def load_benchmark(
    episode_dir: Path,
    manifest_rows: dict[str, dict],
) -> tuple[dict, str]:
    path = episode_dir / "benchmark.json"
    digest = file_sha256(path)
    row = manifest_rows.get(episode_dir.name)
    require(row is not None, "episode is absent from frozen manifest")
    require(row["benchmark_sha256"] == digest, "benchmark hash changed")
    benchmark = json.loads(path.read_text())
    require(benchmark.get("schema_version") == SCHEMA_VERSION, "wrong episode schema")
    require(tuple(benchmark.get("role_sequence") or ()) == ROLE_SEQUENCE, "role mismatch")
    require(benchmark.get("construction_uses_c_navigation_outcomes") is False, "C leakage")
    return benchmark, digest


def source_assets(benchmark: dict) -> tuple[Path, bytes, bytes, np.ndarray]:
    source = Path(benchmark["source_episode"])
    metadata_path = source / "meta/gen_meta.json"
    parquet_path = source / "data/chunk-000/episode_000000.parquet"
    require(file_sha256(metadata_path) == benchmark["source_metadata_sha256"], "metadata changed")
    require(
        file_sha256(parquet_path)
        == benchmark["source_strict_v4_audit"]["source_parquet_sha256"],
        "parquet changed",
    )
    metadata = json.loads(metadata_path.read_text())
    rows = pd.read_parquet(parquet_path)
    intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
    intrinsic = np.stack(
        [np.asarray(row, dtype=np.float64) for row in intrinsic_raw]
    )
    switch_a = int(metadata["switches"][0])
    rgb_root = source / "videos/chunk-000/observation.images.rgb"
    goal_a = (rgb_root / f"{switch_a - 1}.jpg").read_bytes()
    goal_b = (source / "goal_1.jpg").read_bytes()
    require(base.bytes_sha256(goal_a) == benchmark["goal_a_sha256"], "Goal A changed")
    require(base.bytes_sha256(goal_b) == benchmark["goal_b_sha256"], "Goal B changed")
    return source, goal_a, goal_b, intrinsic


def main() -> None:
    validate_cli()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must be empty")

    scene_file = Path(args.scene).resolve()
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene_file.stem, "episode root must be a scene directory")
    require(file_sha256(scene_file) == json.loads(
        next((scene_root / path.name / "benchmark.json").read_text()
             for path in scene_root.glob("episode_*")
             if (path / "benchmark.json").is_file())
    )["source_scene_asset_sha256"], "scene asset changed")
    manifest = load_manifest(scene_root)
    manifest_rows = {row["episode"]: row for row in manifest["accepted"]}
    episode_dirs = sorted(
        path for path in scene_root.glob("episode_*")
        if (path / "benchmark.json").is_file()
    )
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[: args.episodes]
    require(bool(episode_dirs), "no shared-online NNR episodes selected")

    backend_c = controller_backend()
    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        for episode_dir in episode_dirs:
            benchmark, benchmark_sha = load_benchmark(episode_dir, manifest_rows)
            source, image_a, image_b, camera_intrinsic = source_assets(benchmark)
            trace_root = Path(args.shared_leg1_trace_root).resolve()
            require(str(trace_root) == str(Path(benchmark["trace_root"]).resolve()), "trace root differs")
            trace_a_path = trace_root / benchmark["online_a_trace"]
            trace_b_path = trace_root / benchmark["online_b_trace"]
            require(file_sha256(trace_a_path) == benchmark["online_a_trace_sha256"], "A trace changed")
            require(file_sha256(trace_b_path) == benchmark["online_b_trace_sha256"], "B trace changed")
            episode_seed = int(benchmark["episode_seed"])
            camera_height = float(benchmark["camera_height_m"])
            base.srv_reset(
                camera_height=camera_height,
                seed=episode_seed,
                episode_len=(
                    int(benchmark["online_a_steps"])
                    + int(benchmark["online_b_steps"])
                    + int(args.max_steps)
                ),
                camera_intrinsic=camera_intrinsic,
            )

            metadata = json.loads((source / "meta/gen_meta.json").read_text())
            rows = pd.read_parquet(
                source / "data/chunk-000/episode_000000.parquet"
            )
            start_position, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])
            leg_a, trace_a_sha = base.replay_shared_leg1(
                simulator,
                trace_root,
                episode_dir.name,
                episode_seed,
                image_a,
                start_position,
                start_yaw,
            )
            require(leg_a["reached"], "frozen online A is not successful")
            require(trace_a_sha == benchmark["online_a_trace_sha256"], "A replay hash differs")
            require(
                len(leg_a["memory_trace"]) == int(benchmark["online_a_steps"]),
                "A replay omitted long-memory frames",
            )
            a_candidate_ceiling = int(leg_a["memory_trace"][-1]["frame_idx"])
            leg_b, trace_b_sha = multigoal.replay_shared_leg_b(
                simulator,
                trace_root,
                episode_dir.name,
                episode_seed,
                image_b,
                leg_a["end_pos"],
                leg_a["end_psi"],
            )
            require(leg_b["reached"], "frozen online B is not successful")
            require(trace_b_sha == benchmark["online_b_trace_sha256"], "B replay hash differs")
            require(
                len(leg_b["memory_trace"]) == int(benchmark["online_b_steps"]),
                "B replay omitted long-memory frames",
            )
            require(
                max(item["frame_idx"] for item in leg_b["memory_trace"])
                > a_candidate_ceiling,
                "B replay did not extend memory after the A ceiling",
            )
            b_route_start_anchor = int(leg_b["memory_trace"][-1]["frame_idx"])

            c_goal = benchmark["goal_c"]
            c_floor = np.asarray(c_goal["floor_position"], dtype=np.float64)
            c_xz = c_floor[[0, 2]]
            c_yaw = float(c_goal["yaw_rad"])
            image_c_path = episode_dir / benchmark["goal_c_asset"]["rgb"]
            require(
                file_sha256(image_c_path)
                == benchmark["goal_c_asset"]["rgb_sha256"],
                "Goal C image changed",
            )
            image_c = image_c_path.read_bytes()
            ok, geo_c, _points = base.geodesic(pathfinder, leg_b["end_pos"], c_floor)
            require(ok and np.isfinite(geo_c), "B->C geodesic is invalid")
            require(
                abs(float(geo_c) - float(c_goal["geo_B_to_C_m"])) <= 0.05,
                "stored/measured B->C geodesic mismatch",
            )
            require(
                benchmark.get("navdp_short_fifo_before_c") == "reset",
                "benchmark lacks the pre-C FIFO-reset seal",
            )
            require(
                float(c_goal["online_b_switch_endpoint_covis"])
                <= float(benchmark["contract"][
                    "maximum_online_b_endpoint_covis"
                ]),
                "online B switch endpoint is not a Goal-C hard negative",
            )
            require(
                int(c_goal["max_online_a_covis_frame"])
                < int(benchmark["online_a_steps"]),
                "Goal C anchor is outside online A",
            )
            require(
                should_reset_before_leg(args.navdp_goal_switch_reset, 2),
                "pre-C FIFO reset was not requested",
            )
            reset_receipt = base.srv_reset_navdp_short_memory(env_id=0)
            leg_c = base.run_policy_leg(
                simulator,
                pathfinder,
                leg_b["end_pos"],
                leg_b["end_psi"],
                image_c,
                c_xz,
                float(geo_c),
                None,
                terminal_mode="off",
                goal_yaw=c_yaw,
                camera_intrinsic=camera_intrinsic,
                policy_backend=backend_c,
                episode_seed=episode_seed,
                leg_index=2,
                candidate_ceiling_override=a_candidate_ceiling,
                certified_route_start_anchor=b_route_start_anchor,
            )
            route = router_stats(leg_c["plans"])
            metric = {
                "result_schema": RESULT_SCHEMA,
                "scene": scene_file.stem,
                "episode": episode_dir.name,
                "arm": args.shared_online_nnr_arm,
                "controller_backend_C": (
                    "cec_portability"
                    if args.shared_online_nnr_arm == "cec_portability"
                    else backend_c
                ),
                "benchmark_sha256": benchmark_sha,
                "online_A_trace_sha256": trace_a_sha,
                "online_B_trace_sha256": trace_b_sha,
                "online_A_steps": int(leg_a["steps"]),
                "online_B_steps": int(leg_b["steps"]),
                "online_A_candidate_ceiling": a_candidate_ceiling,
                "online_B_route_start_anchor": b_route_start_anchor,
                "online_B_rollout_max_covis_to_C_descriptive": float(
                    c_goal["online_b_rollout_max_covis"]
                ),
                "online_B_switch_endpoint_covis_to_C": float(
                    c_goal["online_b_switch_endpoint_covis"]
                ),
                "navdp_short_fifo_reset_before_C": 1,
                "navdp_short_fifo_reset_receipt": json.dumps(
                    reset_receipt, sort_keys=True
                ),
                "online_A_max_covis_to_C": float(c_goal["max_online_a_covis"]),
                "geo_C": float(geo_c),
                "reached_C": int(leg_c["reached"]),
                "spl_C": base.spl(
                    leg_c["reached"], float(geo_c),
                    leg_c["path_len_at_reach"]
                    if leg_c.get("path_len_at_reach") is not None
                    else leg_c["path_len"],
                ),
                "steps_C": int(leg_c["steps"]),
                "len_C": float(leg_c["path_len"]),
                "final_dist_C": float(leg_c["final_goal_dist_m"]),
                "termination_C": leg_c.get("termination_reason"),
                "certified_stagnation_intervention_C": int(bool(
                    leg_c.get("certified_stagnation_intervention_attempted")
                )),
                "certified_stagnation_intervention_step_C": leg_c.get(
                    "certified_stagnation_intervention_step"
                ),
                "certified_graph_rescue_active_C": int(bool(
                    leg_c.get("certified_graph_rescue_active")
                )),
                "certified_graph_rescue_step_C": leg_c.get(
                    "certified_graph_rescue_step"
                ),
                "router_plans_C": route["plans"],
                "router_active_plans_C": route["active_plans"],
                "router_active_episode_C": int(route["active_episode"]),
            }
            metrics.append(metric)
            (output / f"{episode_dir.name}_plans.json").write_text(
                json.dumps({
                    "frozen_legA": leg_a["plans"],
                    "frozen_legB": leg_b["plans"],
                    "legC": leg_c["plans"],
                    "rollout_traces": {
                        "legA": leg_a["rollout_trace"],
                        "legB": leg_b["rollout_trace"],
                        "legC": leg_c["rollout_trace"],
                    },
                    "memory_traces": {
                        "legA": leg_a["memory_trace"],
                        "legB": leg_b["memory_trace"],
                        "legC": leg_c["memory_trace"],
                    },
                }, sort_keys=True, allow_nan=False) + "\n"
            )
            with (output / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{episode_dir.name}] arm={args.shared_online_nnr_arm} "
                f"C={int(leg_c['reached'])} final={leg_c['final_goal_dist_m']:.3f}m"
            )
        summary = {
            "result_schema": RESULT_SCHEMA,
            "arm": args.shared_online_nnr_arm,
            "episodes": len(metrics),
            "shared_online_A_success": len(metrics),
            "shared_online_B_success_given_A": len(metrics),
            "C_success_given_frozen_online_AB": sum(row["reached_C"] for row in metrics),
            "SR_C_given_frozen_online_AB": float(np.mean([
                row["reached_C"] for row in metrics
            ])),
            "mean_SPL_C": float(np.mean([row["spl_C"] for row in metrics])),
            "stagnation_intervention_episodes": sum(
                row["certified_stagnation_intervention_C"] for row in metrics
            ),
            "graph_rescue_active_episodes": sum(
                row["certified_graph_rescue_active_C"] for row in metrics
            ),
            "prefix_trace_hashes": sorted({
                (row["online_A_trace_sha256"], row["online_B_trace_sha256"])
                for row in metrics
            }),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
