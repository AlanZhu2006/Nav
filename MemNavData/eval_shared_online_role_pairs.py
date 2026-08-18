#!/usr/bin/env python3
"""Role-free closed-loop evaluation after an exact frozen online-A replay."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
from audit_shared_online_role_pairs import audit as audit_benchmark
from shared_online_double_revisit_runtime import replay_online_a, sha256_file
from shared_online_role_pair_contract import runtime_query


args = base.args
RESULT_SCHEMA = "shared_online_role_pair_closed_loop_v1_20260814"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def resolve_arm() -> tuple[str, str | None]:
    if args.server_backend == "navdp":
        require(args.hybrid_route == "phase", "native arm must use phase label")
        require(args.novel_port is None, "native arm must use one NavDP server")
        return "native", None
    require(args.server_backend == "hybrid_pose", "unsupported server backend")
    require(args.novel_port is not None, "hybrid arm requires --novel_port")
    if args.hybrid_route == "phase":
        if args.revisit_adapter == "legacy_metric":
            return "raw_direct", "navdp_mix"
        if args.revisit_adapter == "raw_fixed_bearing_v1":
            return "raw_fixed_bearing", "navdp_mix"
        raise RuntimeError(
            "phase ablation requires legacy_metric or raw_fixed_bearing_v1"
        )
    if args.hybrid_route == "memory_geometry":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "geometry-fixed arm requires the frozen bearing adapter",
        )
        return "geometry_fixed", "navdp_auto"
    if args.hybrid_route == "certified_relocalization":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "certified arm requires the frozen bearing adapter",
        )
        return "certified", "navdp_auto"
    if args.hybrid_route == "certified_semantic_first":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "semantic-first arm requires the frozen bearing adapter",
        )
        return "semantic_first_certified", "navdp_auto"
    if args.hybrid_route == "learned_pi3x_relocalization":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "learned Pi3X arm requires the frozen bearing adapter",
        )
        return "learned_pi3x_spatial", "navdp_auto"
    raise RuntimeError(
        "role-pair evaluator supports only "
        "native/raw/geometry/certified/semantic-first/learned-pi3x"
    )


def validate_cli() -> tuple[str, str | None]:
    arm, backend = resolve_arm()
    require(args.leg1_mode == "shared_trace", "online A requires shared_trace")
    require(
        not args.shared_leg1_trace_root,
        "online A is bound by role_pairs.json, not legacy trace root",
    )
    require(args.leg1_goal_source == "own", "Goal-A swapping is forbidden")
    require(not args.write_leg1_trace, "frozen online A cannot be rewritten")
    require(not args.stop_after_leg1, "query rollout cannot stop after replay")
    require(not args.reset_memory, "online-A memory must be preserved")
    require(args.terminal_uturn == "off", "position SR forbids terminal U-turn")
    require(
        args.terminal_visual_refine == "off",
        "position SR forbids terminal visual refinement",
    )
    require(args.retrieval_override == "off", "retrieval oracle is forbidden")
    require(args.gate_override is None, "gate oracle is forbidden")
    require(args.trajectory_selector == "server", "trajectory oracle is forbidden")
    require(args.oracle_candidate_seed_count == 1, "candidate oracle is forbidden")
    require(args.oracle_global_subgoal_m == 0.0, "global oracle is forbidden")
    require(args.oracle_observed_frontier == "off", "frontier oracle is forbidden")
    require(args.deterministic_plan_seeds, "paired queries require fixed seeds")
    require(args.agent_radius == 0.30, "benchmark uses a 0.30 m agent radius")
    require(args.exec_horizon == 8, "formal NavDP execution horizon is eight")
    require(args.certified_cdec_rescue == "off", "CDEC rescue is out of scope")
    require(args.certified_stagnation_graph == "off", "graph rescue is out of scope")
    require(args.revisit_controller == "navdp_mixed", "controller must stay frozen")
    if arm == "learned_pi3x_spatial":
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.expected_pi3x_model_sha256)
            is not None,
            "learned arm requires a pinned Pi3X model hash",
        )
        require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                args.expected_pi3x_proof_manifest_sha256,
            )
            is not None,
            "learned arm requires a pinned proof-manifest hash",
        )
    if args.role_pair_query_role != "all":
        require(
            args.role_pair_scope == "consumed_integration",
            "role filtering is permitted only for consumed development",
        )
    base.validate_revisit_adapter_configuration(
        mode=args.revisit_adapter,
        server_backend=args.server_backend,
        revisit_controller=args.revisit_controller,
        router_is_automatic_geometry=(args.hybrid_route in base.AUTO_HYBRID_ROUTES),
        router_is_certified_relocalization=(
            args.hybrid_route in base.SCALE_FREE_RELOCALIZATION_ROUTES
        ),
    )
    return arm, backend


def load_episode(episode_dir: Path, expected_scene: str) -> dict:
    payload = json.loads((episode_dir / "role_pairs.json").read_text())
    require(payload["scene"] == expected_scene, "role-pair scene mismatch")
    source = Path(payload["online_a_episode"])
    require(source.is_dir(), "online-A source is missing")
    require(
        sha256_file(source / "receipt.json") == payload["online_a_receipt_sha256"],
        "online-A receipt hash changed",
    )
    require(
        sha256_file(source / "online_a_trace.json")
        == payload["online_a_trace_sha256"],
        "online-A trace hash changed",
    )
    receipt = json.loads((source / "receipt.json").read_text())
    trace = json.loads((source / "online_a_trace.json").read_text())
    require(trace.get("reached") is True, "frozen online A did not succeed")
    require(
        len(trace["poses"]) == int(payload["online_a_steps"]),
        "online-A trace length changed",
    )
    return {
        "benchmark": payload,
        "source": source,
        "receipt": receipt,
        "trace": trace,
    }


def replay_prefix(frozen: dict) -> tuple[dict, dict]:
    replay = replay_online_a(
        frozen,
        memory_step=base.srv_memory,
        navdp_replay_step=base.srv_navdp_memory_replay,
    )
    trace = frozen["trace"]
    leg = {
        "reached": True,
        "path_len": float(trace["path_len"]),
        "steps": int(trace["steps"]),
        "plans": trace["plans"],
        "memory_trace": replay["memory_trace"],
        "rollout_trace": trace["poses"],
        "end_pos": np.asarray(trace["end_position"], dtype=np.float64),
        "end_psi": float(trace["end_yaw"]),
    }
    return leg, replay


def router_counts(plans: list[dict]) -> dict:
    return {
        "router_active_plans": sum(
            plan.get("router_active") is True for plan in plans
        ),
        "certificate_accept_plans": sum(
            plan.get("certified_relocalization_accepted") is True
            for plan in plans
        ),
        "learned_pi3x_accept_plans": sum(
            plan.get("learned_pi3x_relocalization_accepted") is True
            for plan in plans
        ),
        "learned_pi3x_initial_inference_plans": sum(
            plan.get("learned_pi3x_initial_candidate_selection_cached")
            is False
            for plan in plans
        ),
        "adapter_takeover_plans": sum(
            plan.get("revisit_adapter_takeover") is True for plan in plans
        ),
        "runtime_failure_plans": sum(
            (
                plan.get("certified_relocalization_reason")
                == "certificate_endpoint_failure"
            )
            or (
                plan.get("learned_pi3x_relocalization_ok") is False
            )
            for plan in plans
        ),
    }


def main() -> None:
    arm, policy_backend = validate_cli()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output must be empty")

    scene_file = Path(args.scene).resolve()
    scene = base.SCENE_IDENTITY
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene, "episode root must be a scene directory")
    benchmark_root = scene_root.parent
    benchmark_audit = audit_benchmark(benchmark_root)
    require(benchmark_audit["ok"], "benchmark-wide audit failed")
    episode_dirs = sorted(
        path for path in scene_root.glob("episode_*")
        if (path / "role_pairs.json").is_file()
    )
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[: args.episodes]
    require(bool(episode_dirs), "no role-pair episodes selected")

    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        query_index = 0
        for episode_dir in episode_dirs:
            frozen = load_episode(episode_dir, scene)
            receipt = frozen["receipt"]
            require(
                sha256_file(scene_file) == receipt["source_asset_sha256"],
                "scene asset hash differs from online-A materialization",
            )
            source_parquet = (
                Path(receipt["source_episode"])
                / "data/chunk-000/episode_000000.parquet"
            )
            require(
                sha256_file(source_parquet) == receipt["source_parquet_sha256"],
                "source parquet hash changed",
            )
            rows = pd.read_parquet(source_parquet)
            intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
            camera_intrinsic = np.stack(
                [np.asarray(row, dtype=np.float64) for row in intrinsic_raw]
            )
            camera_height = float(receipt["camera_height_m"])
            require(
                math.isclose(camera_height, float(base.CAM_H), abs_tol=1e-12),
                "online-A camera height differs from evaluator camera height",
            )
            episode_seed = int(frozen["trace"]["episode_seed"])
            for pair in frozen["benchmark"]["pairs"]:
                for stored_query in pair["queries"]:
                    if (args.role_pair_query_role != "all"
                            and stored_query["analysis_role"]
                            != args.role_pair_query_role):
                        continue
                    # The runtime projection deliberately drops analysis_role,
                    # co-visibility and all construction diagnostics.  The role
                    # remains available below only for stratified scoring.
                    query = runtime_query(stored_query)
                    goal_rgb_path = episode_dir / query["goal_rgb"]
                    goal_depth_path = episode_dir / query["goal_depth"]
                    require(
                        sha256_file(goal_rgb_path) == query["goal_rgb_sha256"]
                        and sha256_file(goal_depth_path)
                        == query["goal_depth_sha256"],
                        "runtime query asset hash changed",
                    )
                    goal_jpg = goal_rgb_path.read_bytes()
                    goal_floor = np.asarray(query["floor_position"], dtype=np.float64)
                    goal_xz = goal_floor[[0, 2]]
                    goal_yaw = float(query["yaw_rad"])

                    base.srv_reset(
                        camera_height=camera_height,
                        seed=episode_seed,
                        episode_len=int(frozen["benchmark"]["online_a_steps"])
                        + int(args.max_steps),
                        camera_intrinsic=camera_intrinsic,
                    )
                    leg_a, replay = replay_prefix(frozen)
                    position = np.asarray(leg_a["end_pos"], dtype=np.float64)
                    yaw = float(leg_a["end_psi"])
                    ok, geo_distance, _path = base.geodesic(
                        pathfinder, position, goal_floor
                    )
                    require(ok and np.isfinite(geo_distance), "query geodesic failed")
                    require(
                        abs(
                            float(geo_distance)
                            - float(stored_query["geodesic_from_a_end_m"])
                        )
                        <= 0.05,
                        "stored/measured query geodesic mismatch",
                    )
                    leg = base.run_policy_leg(
                        simulator,
                        pathfinder,
                        position,
                        yaw,
                        goal_jpg,
                        goal_xz,
                        float(geo_distance),
                        None,
                        terminal_mode="off",
                        goal_yaw=goal_yaw,
                        camera_intrinsic=camera_intrinsic,
                        policy_backend=policy_backend,
                        episode_seed=episode_seed,
                        leg_index=1,
                    )
                    counts = router_counts(leg["plans"])
                    metric = {
                        "scene": scene,
                        "episode": episode_dir.name,
                        "pair_id": pair["pair_id"],
                        "query_id": query["query_id"],
                        "analysis_role": stored_query["analysis_role"],
                        "arm": arm,
                        "seed": episode_seed,
                        "shared_A_frames": replay["online_frames"],
                        "shared_A_decision_frames": replay["decision_frames"],
                        "shared_A_hashes_ok": int(replay["all_rgb_hashes_verified"]),
                        "shared_A_diffusion_samples": replay[
                            "diffusion_samples_during_replay"
                        ],
                        "reached": int(bool(leg["reached"])),
                        "geodesic_m": float(geo_distance),
                        "path_len_m": float(leg["path_len"]),
                        "steps": int(leg["steps"]),
                        "final_goal_dist_m": float(leg["final_goal_dist_m"]),
                        "termination_reason": leg.get("termination_reason"),
                        **counts,
                    }
                    metrics.append(metric)
                    plan_path = output / f"{episode_dir.name}_{query['query_id']}_plans.json"
                    plan_path.write_text(
                        json.dumps(
                            {
                                "schema_version": RESULT_SCHEMA,
                                "arm": arm,
                                "query_runtime_fields": sorted(query),
                                "analysis_role_not_forwarded": True,
                                "replay": replay,
                                "legA": leg_a["plans"],
                                "query_leg": leg["plans"],
                                "memory_traces": {
                                    "legA": leg_a["memory_trace"],
                                    "query": leg["memory_trace"],
                                },
                                "rollout_traces": {
                                    "legA": leg_a["rollout_trace"],
                                    "query": leg["rollout_trace"],
                                },
                            },
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    query_index += 1
                    print(
                        f"[{scene}/{episode_dir.name}/{pair['pair_id']}/"
                        f"{stored_query['analysis_role']}/{arm}] "
                        f"success={metric['reached']} steps={metric['steps']}"
                    )
                    with (output / "metric.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                        writer.writeheader()
                        writer.writerows(metrics)

        roles = {
            role: [row for row in metrics if row["analysis_role"] == role]
            for role in ("novel", "revisit")
        }
        scope_map = {
            "consumed_integration": (
                "consumed-scene integration unless externally promoted"
            ),
            "paper_heldout": "paper held-out role-pair evaluation",
            "replica_cross_dataset": (
                "Replica cross-dataset role-pair evaluation"
            ),
        }
        summary = {
            "schema_version": RESULT_SCHEMA,
            "scope": scope_map[args.role_pair_scope],
            "role_pair_scope": args.role_pair_scope,
            "role_pair_query_role": args.role_pair_query_role,
            "arm": arm,
            "server_backend": args.server_backend,
            "hybrid_route": args.hybrid_route,
            "revisit_adapter": args.revisit_adapter,
            "revisit_controller": args.revisit_controller,
            "benchmark_manifest_sha256": benchmark_audit["manifest_sha256"],
            "scene": scene,
            "queries": len(metrics),
            "role_counts": {role: len(rows) for role, rows in roles.items()},
            "SR": mean_or_none([row["reached"] for row in metrics]),
            "SR_by_role": {
                role: mean_or_none([row["reached"] for row in rows])
                for role, rows in roles.items()
            },
            "router_active_episodes_by_role": {
                role: sum(row["router_active_plans"] > 0 for row in rows)
                for role, rows in roles.items()
            },
            "certificate_accept_episodes_by_role": {
                role: sum(row["certificate_accept_plans"] > 0 for row in rows)
                for role, rows in roles.items()
            },
            "learned_pi3x_accept_episodes_by_role": {
                role: sum(
                    row["learned_pi3x_accept_plans"] > 0 for row in rows
                )
                for role, rows in roles.items()
            },
            "adapter_takeover_episodes_by_role": {
                role: sum(row["adapter_takeover_plans"] > 0 for row in rows)
                for role, rows in roles.items()
            },
            "runtime_failure_plans": sum(
                row["runtime_failure_plans"] for row in metrics
            ),
            "shared_A_all_hashes_ok": all(
                row["shared_A_hashes_ok"] for row in metrics
            ),
            "shared_A_total_diffusion_samples": sum(
                row["shared_A_diffusion_samples"] for row in metrics
            ),
            "runtime_role_visibility": "none",
            "deterministic_plan_seeds": bool(args.deterministic_plan_seeds),
            "max_steps": int(args.max_steps),
            "exec_horizon": int(args.exec_horizon),
            "certified_cdec_rescue": args.certified_cdec_rescue,
            "certified_stagnation_graph": args.certified_stagnation_graph,
            "memnav_server_info": dict(base.MEMNAV_SERVER_INFO),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print("[shared-online-role-pair] done", summary)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
