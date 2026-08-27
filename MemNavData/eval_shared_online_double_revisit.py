#!/usr/bin/env python3
"""Closed-loop B->C evaluation on a frozen, genuinely online NavDP A prefix.

The evaluator never regenerates or re-executes Goal A.  It restores the exact
audited RGB stream into MemNav and restores only the original decision frames
into NavDP's local FIFO, then runs two Revisit goals.  Under the natural
``carry`` contract, the complete factual B rollout is checked for visual
co-visibility with C and a recent-memory shortcut censors C.  The explicit
``before_c`` causal arm clears only NavDP's bounded short FIFO at the switch;
then the B endpoint must remain a visual hard negative while MemNav retrieval
is still capped at the online-A boundary.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import eval_2leg_habitat as base
from audit_shared_online_double_revisit import audit as audit_benchmark
from generate_twoleg import (
    backproject,
    cam_to_world_hab,
    covis_frac,
    to_world,
)
from multigoal_policy_contract import three_leg_policy_backends
from navdp_goal_switch import should_reset_before_leg
from shared_online_double_revisit_runtime import (
    load_frozen_episode,
    replay_online_a,
    sha256_file,
    summarize_c_tail,
)


args = base.args
ROLE_SEQUENCE = ("initial_imagegoal", "revisit", "revisit")
RESULT_SCHEMA = "shared_online_double_revisit_closed_loop_v3_leg_scope_20260813"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def empty_leg(position: np.ndarray, yaw: float, goal_xz: np.ndarray, reason: str) -> dict:
    return {
        "reached": False,
        "path_len": 0.0,
        "path_len_at_reach": None,
        "step_at_reach": None,
        "steps": 0,
        "diagnostic_steps": 0,
        "termination_reason": reason,
        "blocked_step_count": 0,
        "plans": [],
        "memory_trace": [],
        "rollout_trace": [],
        "end_pos": np.asarray(position, dtype=float),
        "end_psi": float(yaw),
        "final_goal_dist_m": float(
            np.linalg.norm(np.asarray(position, dtype=float)[[0, 2]] - goal_xz)
        ),
    }


def router_stats(plans: list[dict]) -> dict:
    decisions = [
        bool(plan["router_active"])
        for plan in plans
        if plan.get("router_active") is not None
    ]
    return {
        "plans": len(decisions),
        "active_plans": sum(decisions),
        "active_episode": bool(any(decisions)),
    }


def goal_asset(episode_dir: Path, frozen: dict, role: str) -> tuple[bytes, Path]:
    asset = frozen["variant"]["assets"][role]
    root = episode_dir / args.shared_online_variant
    rgb = root / asset["rgb"]
    depth = root / asset["depth"]
    return rgb.read_bytes(), depth


def habitat_goal(frozen: dict, role: str) -> tuple[np.ndarray, float]:
    goal = frozen["variant"]["goals"][role]
    return (
        np.asarray(goal["floor_position"], dtype=np.float64),
        float(goal["yaw_rad"]),
    )


def measured_geodesic(pathfinder, first: np.ndarray, second: np.ndarray) -> float:
    ok, distance, _path = base.geodesic(pathfinder, first, second)
    require(ok and np.isfinite(distance), "shared-online geodesic is invalid")
    return float(distance)


def audit_online_b_against_c(
    simulator,
    rollout_trace: list[dict],
    c_goal: dict,
    c_depth_path: Path,
    *,
    camera_height: float,
) -> dict:
    """Render the factual B prefix and test whether it exposes Goal C."""
    require(bool(rollout_trace), "Goal-B rollout has no observations")
    depth = np.asarray(Image.open(c_depth_path), dtype=np.float32) / 10000.0
    goal_camera = np.asarray(c_goal["camera_position"], dtype=np.float64)
    goal_yaw = float(c_goal["yaw_rad"])
    goal_points = to_world(
        backproject(depth, stride=6),
        cam_to_world_hab(goal_camera, goal_yaw),
    )
    curve = []
    for pose in rollout_trace:
        floor = np.asarray(
            [pose["x"], pose["y"], pose["z"]], dtype=np.float64
        )
        camera = floor + np.asarray([0.0, camera_height, 0.0])
        _rgb, rendered_depth = base.render(
            simulator, camera, float(pose["yaw"])
        )
        curve.append(
            covis_frac(
                goal_points,
                cam_to_world_hab(camera, float(pose["yaw"])),
                rendered_depth,
            )
        )
    return summarize_c_tail(
        curve,
        maximum_allowed=float(args.shared_online_c_tail_max_covis),
    )


def validate_cli() -> None:
    require(
        args.shared_online_variant is not None,
        "--shared_online_variant is required",
    )
    require(args.leg1_mode == "shared_trace", "online A requires shared_trace mode")
    require(
        not args.shared_leg1_trace_root,
        "online A is bound by benchmark.json; do not pass legacy trace root",
    )
    require(args.leg1_goal_source == "own", "Goal-A swapping is forbidden")
    require(not args.write_leg1_trace, "frozen online A cannot be rewritten")
    require(not args.stop_after_leg1, "B/C evaluation cannot stop after A")
    require(not args.reset_memory, "double-Revisit must preserve A memory")
    require(args.terminal_uturn == "off", "position SR forbids terminal U-turn")
    require(
        args.terminal_visual_refine == "off",
        "position SR forbids terminal visual refinement",
    )
    require(args.retrieval_override == "off", "retrieval oracle is forbidden")
    require(args.gate_override is None, "gate oracle is forbidden")
    require(args.trajectory_selector == "server", "trajectory oracle is forbidden")
    require(
        args.oracle_candidate_seed_count == 1,
        "candidate-pooling oracle is forbidden",
    )
    require(args.oracle_global_subgoal_m == 0.0, "global subgoal oracle is forbidden")
    require(args.oracle_observed_frontier == "off", "frontier oracle is forbidden")
    require(
        args.double_revisit_c_history == "initial_leg_only",
        "Goal C must be restricted to the online-A memory boundary",
    )
    require(args.deterministic_plan_seeds, "paired B/C requires deterministic seeds")
    require(args.server_backend in ("navdp", "hybrid_pose"), "unsupported backend")
    require(args.agent_radius == 0.30, "benchmark was built with 0.30 m radius")
    require(args.exec_horizon == 8, "formal NavDP execution horizon is 8")
    require(
        math.isfinite(float(args.shared_online_c_tail_max_covis))
        and 0.0 <= args.shared_online_c_tail_max_covis <= 1.0,
        "C-tail threshold must be in [0,1]",
    )
    if args.server_backend == "hybrid_pose":
        require(args.novel_port is not None, "hybrid_pose requires --novel_port")
    else:
        require(args.novel_port is None, "native NavDP must not receive --novel_port")
        require(args.hybrid_route == "phase", "native NavDP uses phase route label")
    if args.shared_online_known_revisit_scope == "b_only":
        require(
            args.server_backend == "hybrid_pose" and args.hybrid_route == "phase",
            "b_only requires the known-role hybrid phase route",
        )
    base.validate_revisit_adapter_configuration(
        mode=args.revisit_adapter,
        server_backend=args.server_backend,
        revisit_controller=args.revisit_controller,
        router_is_automatic_geometry=(args.hybrid_route in base.AUTO_HYBRID_ROUTES),
        router_is_certified_relocalization=(
            args.hybrid_route == "certified_relocalization"
        ),
    )
    if args.hybrid_route == "certified_relocalization":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "certified relocalization requires verified_bearing_v1",
        )
    if args.certified_stagnation_graph != "off":
        require(
            args.server_backend == "hybrid_pose"
            and args.hybrid_route == "certified_relocalization",
            "certified stagnation intervention requires the certified hybrid route",
        )


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
        "path_len_at_reach": trace.get("path_len_at_reach"),
        "step_at_reach": trace.get("step_at_reach"),
        "steps": int(trace["steps"]),
        "termination_reason": trace.get("termination_reason"),
        "blocked_step_count": int(trace.get("blocked_step_count", 0)),
        "plans": trace["plans"],
        "memory_trace": replay["memory_trace"],
        "rollout_trace": trace["poses"],
        "end_pos": np.asarray(trace["end_position"], dtype=np.float64),
        "end_psi": float(trace["end_yaw"]),
        "final_goal_dist_m": float(trace["final_goal_dist_m"]),
    }
    return leg, replay


def main() -> None:
    validate_cli()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(
        not any(output.iterdir()),
        "output directory must be empty for a fail-closed run",
    )

    scene_file = Path(args.scene).resolve()
    scene = scene_file.stem
    scene_root = Path(args.episode_root).resolve()
    require(scene_root.name == scene, "episode root must be the selected scene directory")
    benchmark_root = scene_root.parent
    benchmark_audit = audit_benchmark(benchmark_root)
    require(benchmark_audit["ok"], "benchmark-wide audit failed")

    episode_dirs = sorted(
        path for path in scene_root.glob("episode_*")
        if (path / "benchmark.json").is_file()
    )
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[: args.episodes]
    require(bool(episode_dirs), "no shared-online episodes selected")

    known_revisit_leg_indices = (
        {1} if args.shared_online_known_revisit_scope == "b_only" else None
    )
    backends = three_leg_policy_backends(
        server_backend=args.server_backend,
        hybrid_route=args.hybrid_route,
        automatic_routes=base.AUTO_HYBRID_ROUTES,
        role_sequence=ROLE_SEQUENCE,
        known_revisit_leg_indices=known_revisit_leg_indices,
    )
    c_uses_long_memory = backends[2] in ("navdp_mix", "navdp_auto")
    simulator = base.make_sim(str(scene_file), "", agent_radius=args.agent_radius)
    pathfinder = simulator.pathfinder
    metrics = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            frozen = load_frozen_episode(
                episode_dir / "benchmark.json",
                variant=args.shared_online_variant,
                expected_scene=scene,
            )
            receipt = frozen["receipt"]
            require(
                sha256_file(scene_file) == receipt["source_asset_sha256"],
                "evaluator scene asset hash differs from materialization asset",
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
            episode_seed = int(frozen["trace"]["episode_seed"])
            require(
                episode_seed == int(args.seed) + episode_index,
                "CLI seed does not reproduce the frozen online-A seed",
            )
            camera_height = float(receipt["camera_height_m"])
            base.srv_reset(
                camera_height=camera_height,
                seed=episode_seed,
                episode_len=int(frozen["benchmark"]["online_a_steps"])
                + 2 * int(args.max_steps),
                camera_intrinsic=camera_intrinsic,
            )
            leg_a, replay = replay_prefix(frozen)
            position = np.asarray(leg_a["end_pos"], dtype=np.float64)
            yaw = float(leg_a["end_psi"])
            b_floor, b_yaw = habitat_goal(frozen, "B")
            c_floor, c_yaw = habitat_goal(frozen, "C")
            b_xz = b_floor[[0, 2]]
            c_xz = c_floor[[0, 2]]
            image_b, _b_depth_path = goal_asset(episode_dir, frozen, "B")
            image_c, c_depth_path = goal_asset(episode_dir, frozen, "C")

            geo_a_b = measured_geodesic(pathfinder, position, b_floor)
            geo_b_c = measured_geodesic(pathfinder, b_floor, c_floor)
            stored_geodesics = frozen["variant"]["leg_geodesics_m"]
            require(
                abs(geo_a_b - float(stored_geodesics["A_to_B"])) <= 0.05,
                "stored/measured A->B geodesic mismatch",
            )
            require(
                abs(geo_b_c - float(stored_geodesics["B_to_C"])) <= 0.05,
                "stored/measured B->C geodesic mismatch",
            )
            if args.server_backend == "hybrid_pose":
                require(
                    len(replay["memory_trace"]) == replay["online_frames"],
                    "hybrid replay did not restore every long-memory frame",
                )
                a_candidate_ceiling = int(replay["memory_trace"][-1]["frame_idx"])
            else:
                a_candidate_ceiling = None

            reset_receipts = {"before_B": None, "before_C": None}
            if should_reset_before_leg(args.navdp_goal_switch_reset, 1):
                reset_receipts["before_B"] = base.srv_reset_navdp_short_memory(
                    env_id=0
                )
            leg_b = base.run_policy_leg(
                simulator,
                pathfinder,
                position,
                yaw,
                image_b,
                b_xz,
                geo_a_b,
                None,
                terminal_mode="off",
                goal_yaw=b_yaw,
                camera_intrinsic=camera_intrinsic,
                policy_backend=backends[1],
                episode_seed=episode_seed,
                leg_index=1,
            )
            position = np.asarray(leg_b["end_pos"], dtype=np.float64)
            yaw = float(leg_b["end_psi"])
            previous_certified_anchor = next((
                int(plan["router_selected_anchor"])
                for plan in leg_b["plans"]
                if plan.get("certified_relocalization_accepted") is True
                and plan.get("router_selected_anchor") is not None
            ), None)
            c_tail = audit_online_b_against_c(
                simulator,
                leg_b["rollout_trace"],
                frozen["variant"]["goals"]["C"],
                c_depth_path,
                camera_height=camera_height,
            )
            reset_before_c = should_reset_before_leg(
                args.navdp_goal_switch_reset, 2
            )
            c_effective_input_ok = bool(c_tail["ok"])
            if reset_before_c:
                c_effective_input_ok = (
                    float(c_tail["endpoint_covisibility"])
                    <= float(args.shared_online_c_tail_max_covis)
                )
                if leg_b["reached"]:
                    reset_receipts["before_C"] = (
                        base.srv_reset_navdp_short_memory(env_id=0)
                    )
            c_tail["navdp_short_memory_reset_before_c"] = reset_before_c
            c_tail["effective_input_contract_ok"] = c_effective_input_ok
            c_tail["effective_input_contract"] = (
                "A-bounded long memory + current B endpoint only"
                if reset_before_c
                else "A-bounded long memory + carried B short FIFO"
            )
            c_tail_path = output / f"{episode_dir.name}_c_tail_audit.json"
            c_tail_path.write_text(
                json.dumps(c_tail, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )

            if not leg_b["reached"]:
                leg_c = empty_leg(position, yaw, c_xz, "censored_goal_b_failure")
            elif not c_effective_input_ok:
                leg_c = empty_leg(
                    position,
                    yaw,
                    c_xz,
                    (
                        "censored_current_observation_contamination"
                        if reset_before_c
                        else "censored_recent_memory_contamination"
                    ),
                )
            else:
                leg_c = base.run_policy_leg(
                    simulator,
                    pathfinder,
                    position,
                    yaw,
                    image_c,
                    c_xz,
                    geo_b_c,
                    None,
                    terminal_mode="off",
                    goal_yaw=c_yaw,
                    camera_intrinsic=camera_intrinsic,
                    policy_backend=backends[2],
                    episode_seed=episode_seed,
                    leg_index=2,
                    candidate_ceiling_override=(
                        a_candidate_ceiling if c_uses_long_memory else None
                    ),
                    certified_route_start_anchor=(
                        previous_certified_anchor
                        if args.hybrid_route == "certified_relocalization"
                        else None
                    ),
                )
                if a_candidate_ceiling is not None and c_uses_long_memory:
                    reported_ceilings = [
                        int(plan["candidate_ceiling"])
                        for plan in leg_c["plans"]
                        if plan.get("candidate_ceiling") is not None
                    ]
                    require(bool(reported_ceilings), "Goal-C plans omitted ceiling")
                    require(
                        all(value <= a_candidate_ceiling for value in reported_ceilings),
                        "Goal-C retrieved beyond online-A memory boundary",
                    )

            reached_b = bool(leg_b["reached"])
            reached_c = bool(leg_c["reached"])
            valid_joint = reached_b and c_effective_input_ok and reached_c
            route_b = router_stats(leg_b["plans"])
            route_c = router_stats(leg_c["plans"])
            metric = {
                "scene": scene,
                "episode": episode_dir.name,
                "seed": episode_seed,
                "variant": args.shared_online_variant,
                "server_backend": args.server_backend,
                "hybrid_route": args.hybrid_route,
                "certified_stagnation_graph": args.certified_stagnation_graph,
                "known_revisit_scope": args.shared_online_known_revisit_scope,
                "policy_backend_B": backends[1],
                "policy_backend_C": backends[2],
                "C_long_memory_enabled": int(c_uses_long_memory),
                "shared_online_A": 1,
                "shared_A_frames": replay["online_frames"],
                "shared_A_decision_frames": replay["decision_frames"],
                "shared_A_hashes_ok": int(replay["all_rgb_hashes_verified"]),
                "shared_A_replay_diffusion_samples": replay[
                    "diffusion_samples_during_replay"
                ],
                "A_candidate_ceiling": a_candidate_ceiling,
                "reached_A": 1,
                "reached_B": int(reached_b),
                "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
                "navdp_short_reset_before_B": int(
                    reset_receipts["before_B"] is not None
                ),
                "navdp_short_reset_before_C": int(
                    reset_receipts["before_C"] is not None
                ),
                "c_tail_contract_ok": int(c_tail["ok"]),
                "c_tail_max_covis": c_tail["maximum_covisibility"],
                "c_tail_argmax_B_frame": c_tail["argmax_b_frame"],
                "c_tail_endpoint_covis": c_tail["endpoint_covisibility"],
                "c_tail_frames": c_tail["frames"],
                "c_effective_input_contract_ok": int(c_effective_input_ok),
                "C_evaluated": int(reached_b and c_effective_input_ok),
                "reached_C": int(reached_c),
                "joint_success": int(valid_joint),
                "geo_A_to_B": geo_a_b,
                "geo_B_to_C": geo_b_c,
                "len_A": leg_a["path_len"],
                "len_B": leg_b["path_len"],
                "len_C": leg_c["path_len"],
                "steps_A": leg_a["steps"],
                "steps_B": leg_b["steps"],
                "steps_C": leg_c["steps"],
                "final_dist_B": leg_b["final_goal_dist_m"],
                "final_dist_C": leg_c["final_goal_dist_m"],
                "termination_B": leg_b.get("termination_reason"),
                "termination_C": leg_c.get("termination_reason"),
                "certified_graph_rescue_attempted_B": int(
                    leg_b.get("certified_graph_rescue_attempted", False)
                ),
                "certified_graph_rescue_attempted_C": int(
                    leg_c.get("certified_graph_rescue_attempted", False)
                ),
                "certified_stagnation_intervention_attempted_B": int(
                    leg_b.get(
                        "certified_stagnation_intervention_attempted", False)
                ),
                "certified_stagnation_intervention_attempted_C": int(
                    leg_c.get(
                        "certified_stagnation_intervention_attempted", False)
                ),
                "certified_stagnation_intervention_step_B": leg_b.get(
                    "certified_stagnation_intervention_step"
                ),
                "certified_stagnation_intervention_step_C": leg_c.get(
                    "certified_stagnation_intervention_step"
                ),
                "certified_graph_rescue_step_B": leg_b.get(
                    "certified_graph_rescue_step"
                ),
                "certified_graph_rescue_step_C": leg_c.get(
                    "certified_graph_rescue_step"
                ),
                "certified_route_start_anchor_C": leg_c.get(
                    "certified_route_start_anchor"
                ),
                "certified_graph_active_plans_B": sum(
                    plan.get("certified_graph_rescue_active") is True
                    for plan in leg_b.get("plans", [])
                ),
                "certified_graph_active_plans_C": sum(
                    plan.get("certified_graph_rescue_active") is True
                    for plan in leg_c.get("plans", [])
                ),
                "router_plans_B": route_b["plans"],
                "router_plans_C": route_c["plans"],
                "router_active_plans_B": route_b["active_plans"],
                "router_active_plans_C": route_c["active_plans"],
                "router_active_episode_B": int(route_b["active_episode"]),
                "router_active_episode_C": int(route_c["active_episode"]),
            }
            metrics.append(metric)
            (output / f"{episode_dir.name}_plans.json").write_text(
                json.dumps(
                    {
                        "schema_version": RESULT_SCHEMA,
                        "replay": replay,
                        "navdp_short_memory_reset_receipts": reset_receipts,
                        "legA": leg_a["plans"],
                        "legB": leg_b["plans"],
                        "legC": leg_c["plans"],
                        "memory_traces": {
                            "legA": leg_a["memory_trace"],
                            "legB": leg_b["memory_trace"],
                            "legC": leg_c["memory_trace"],
                        },
                        "rollout_traces": {
                            "legA": leg_a["rollout_trace"],
                            "legB": leg_b["rollout_trace"],
                            "legC": leg_c["rollout_trace"],
                        },
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            with (output / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{scene}/{episode_dir.name}/{args.shared_online_variant}] "
                f"B={int(reached_b)} C_tail={int(c_tail['ok'])} "
                f"C_input={int(c_effective_input_ok)} C={int(reached_c)}"
            )

        c_evaluated = [row for row in metrics if row["C_evaluated"]]
        summary = {
            "schema_version": RESULT_SCHEMA,
            "scope": "pipeline pilot; no statistical claim",
            "episodes": len(metrics),
            "scene": scene,
            "variant": args.shared_online_variant,
            "server_backend": args.server_backend,
            "hybrid_route": args.hybrid_route,
            "certified_stagnation_graph": args.certified_stagnation_graph,
            "known_revisit_scope": args.shared_online_known_revisit_scope,
            "policy_backends": {"B": backends[1], "C": backends[2]},
            "C_long_memory_enabled": c_uses_long_memory,
            "benchmark_manifest_sha256": benchmark_audit["manifest_sha256"],
            "deterministic_plan_seeds": bool(args.deterministic_plan_seeds),
            "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
            "shared_A_all_hashes_ok": all(row["shared_A_hashes_ok"] for row in metrics),
            "shared_A_total_diffusion_samples": sum(
                row["shared_A_replay_diffusion_samples"] for row in metrics
            ),
            "SR_B": mean_or_none([row["reached_B"] for row in metrics]),
            "C_tail_contract_rate": mean_or_none(
                [row["c_tail_contract_ok"] for row in metrics]
            ),
            "C_effective_input_contract_rate": mean_or_none(
                [row["c_effective_input_contract_ok"] for row in metrics]
            ),
            "SR_C_given_B_and_valid_input": mean_or_none(
                [row["reached_C"] for row in c_evaluated]
            ),
            "joint_SR_with_valid_input_contract": mean_or_none(
                [row["joint_success"] for row in metrics]
            ),
            "certified_graph_rescue_episode_count_B": sum(
                row["certified_graph_rescue_attempted_B"] for row in metrics
            ),
            "certified_graph_rescue_episode_count_C": sum(
                row["certified_graph_rescue_attempted_C"] for row in metrics
            ),
            "certified_stagnation_intervention_episode_count_B": sum(
                row["certified_stagnation_intervention_attempted_B"]
                for row in metrics
            ),
            "certified_stagnation_intervention_episode_count_C": sum(
                row["certified_stagnation_intervention_attempted_C"]
                for row in metrics
            ),
            "certified_graph_active_plan_count_B": sum(
                row["certified_graph_active_plans_B"] for row in metrics
            ),
            "certified_graph_active_plan_count_C": sum(
                row["certified_graph_active_plans_C"] for row in metrics
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (output / "run_contract.json").write_text(
            json.dumps(
                {
                    "benchmark_root": str(benchmark_root),
                    "benchmark_manifest_sha256": benchmark_audit[
                        "manifest_sha256"
                    ],
                    "scene_asset": str(scene_file),
                    "scene_asset_sha256": sha256_file(scene_file),
                    "variant": args.shared_online_variant,
                    "role_sequence": list(ROLE_SEQUENCE),
                    "C_history": "initial_leg_only",
                    "known_revisit_scope": args.shared_online_known_revisit_scope,
                    "C_long_memory_enabled": c_uses_long_memory,
                    "C_candidate_ceiling": (
                        "online_A_boundary" if c_uses_long_memory else "disabled"
                    ),
                    "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
                    "C_effective_short_memory": (
                        "current_B_endpoint_only"
                        if should_reset_before_leg(
                            args.navdp_goal_switch_reset, 2
                        )
                        else "carried_B_FIFO"
                    ),
                    "C_tail_max_covis": args.shared_online_c_tail_max_covis,
                    "certified_stagnation_graph": (
                        args.certified_stagnation_graph
                    ),
                    "graph_subgoal_spacing_m": base.MEMNAV_SERVER_INFO.get(
                        "graph_subgoal_spacing_m"
                    ),
                    "graph_subgoal_arrival_m": base.MEMNAV_SERVER_INFO.get(
                        "graph_subgoal_arrival_m"
                    ),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        print("[shared-online-double-revisit] done", summary)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
