#!/usr/bin/env python3
"""True three-leg Habitat evaluation built on the audited 2-leg controller.

The existing evaluator owns all HTTP, waypoint, collision and router logic.
This entry point deliberately reuses those functions but executes the complete
start->A->B->C protocol: A and B are Novel goals, while C revisits leg A.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
from conditional_c_protocol import world_goal_to_local
from global_subgoal_protocol import polyline_subgoal
from navdp_goal_switch import should_reset_before_leg


args = base.args


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def empty_leg(pos: np.ndarray, psi: float, goal_xz: np.ndarray) -> dict:
    return {
        "reached": False,
        "path_len": 0.0,
        "path_len_at_reach": None,
        "step_at_reach": None,
        "steps": 0,
        "plans": [],
        "end_pos": np.asarray(pos, dtype=float),
        "end_psi": float(psi),
        "final_goal_dist_m": float(
            np.linalg.norm(np.asarray(pos, dtype=float)[[0, 2]] - goal_xz)
        ),
    }


def policy_backend() -> str | None:
    if args.server_backend == "navdp":
        return None
    if args.server_backend == "hybrid_pose":
        return "navdp_auto"
    raise ValueError("3-leg evaluation supports only navdp and hybrid_pose")


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


def leg_spl(leg: dict, geodesic_m: float) -> float:
    path = (
        leg["path_len_at_reach"]
        if leg.get("path_len_at_reach") is not None
        else leg["path_len"]
    )
    return base.spl(leg["reached"], geodesic_m, path)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_oracle_global_subgoal_leg(
    sim,
    pathfinder,
    start_position: np.ndarray,
    start_yaw: float,
    goal_jpg: bytes,
    goal_xz: np.ndarray,
    episode_seed: int,
) -> dict:
    """Run frozen NavDP behind a privileged Habitat shortest-path subgoal.

    This arm changes only the high-level target supplied on Novel-B.  NavDP
    still produces and collision-scores every local trajectory, and the same
    pure-pursuit executor applies it.  It is an upper bound, not a deployable
    policy, because the final metric goal and Habitat pathfinder are privileged.
    """
    position = np.asarray(start_position, dtype=np.float64).copy()
    yaw = float(start_yaw)
    path_len = 0.0
    path_len_at_reach = None
    way_world = None
    plans: list[dict] = []
    history: list[np.ndarray] = []
    success_dist = float(args.success_dist)

    def result(reached: bool, steps: int) -> dict:
        return {
            "reached": bool(reached),
            "path_len": float(path_len),
            "path_len_at_reach": path_len_at_reach,
            "step_at_reach": int(steps) if reached else None,
            "steps": int(steps),
            "plans": plans,
            "end_pos": position.copy(),
            "end_psi": float(yaw),
            "final_goal_dist_m": float(np.linalg.norm(
                position[[0, 2]] - goal_xz)),
        }

    if np.linalg.norm(position[[0, 2]] - goal_xz) < success_dist:
        path_len_at_reach = 0.0
        return result(True, 0)

    for step in range(args.max_steps):
        rgb, depth = base.render(
            sim, position + np.asarray([0.0, base.CAM_H, 0.0]), yaw)
        frame = base.jpg_bytes(rgb)

        if step % args.exec_horizon == 0:
            goal3 = np.asarray(
                [goal_xz[0], position[1], goal_xz[1]], dtype=np.float64)
            ok, remaining_m, path_points = base.geodesic(
                pathfinder, position, goal3)
            require(
                ok and np.isfinite(remaining_m) and bool(path_points),
                "oracle global-subgoal geodesic is invalid",
            )
            subgoal = polyline_subgoal(
                path_points, args.oracle_global_subgoal_m)
            local_goal = world_goal_to_local(
                subgoal[[0, 2]], position[[0, 2]], yaw)
            request_seed = base.diffusion_plan_seed(
                int(episode_seed), 1, len(plans))
            response = base.requests.post(
                f"{base.BASE}/navdp_step_ip_mixgoal",
                files={
                    "image": ("image.jpg", frame),
                    "image_goal": ("goal.jpg", goal_jpg),
                    "depth": ("depth.png", base.depth_png_bytes(depth)),
                },
                data={
                    "goal_data": json.dumps({
                        "goal_x": [float(local_goal[0])],
                        "goal_y": [float(local_goal[1])],
                    }),
                    "diffusion_seed": str(request_seed),
                },
            )
            response.raise_for_status()
            response_json = response.json()
            if int(response_json.get("diffusion_seed", -1)) != request_seed:
                raise RuntimeError(
                    "NavDP server did not echo global-subgoal diffusion seed")
            plan = base.normalize_navdp_response(response_json)
            way, selector = base.select_plan_trajectory(
                plan,
                position,
                yaw,
                pathfinder,
                goal_xz,
                trajectory_selector="server",
            )
            way_world = base.waypoints_to_world(
                way, position[[0, 2]], yaw)
            plans.append({
                "step": int(step),
                "current_x": float(position[0]),
                "current_z": float(position[2]),
                "current_yaw": float(yaw),
                "current_goal_dist_m": float(np.linalg.norm(
                    position[[0, 2]] - goal_xz)),
                "remaining_geodesic_m": float(remaining_m),
                "oracle_subgoal_world": subgoal.tolist(),
                "oracle_subgoal_local": local_goal.tolist(),
                "oracle_subgoal_distance_m": float(np.linalg.norm(
                    subgoal - position)),
                "pose_controller": "oracle_habitat_geodesic_subgoal",
                "router_active": None,
                "router_reason": "privileged_global_subgoal",
                "diffusion_seed": plan.get("diffusion_seed"),
                "requested_diffusion_seed": request_seed,
                "navdp_critic_max": plan.get("navdp_critic_max"),
                **selector,
            })

        if way_world is not None:
            position, yaw, distance = base.pursuit_step(
                position, yaw, way_world, pathfinder)
            path_len += distance
        history.append(position[[0, 2]].copy())
        if np.linalg.norm(position[[0, 2]] - goal_xz) < success_dist:
            path_len_at_reach = float(path_len)
            return result(True, step + 1)
        if (len(history) > args.stuck_window
                and np.linalg.norm(
                    history[-1] - history[-args.stuck_window]
                ) < args.stuck_dist):
            return result(False, step + 1)
    return result(False, args.max_steps)


def main() -> None:
    require(args.leg1_mode == "policy", "3-leg benchmark requires policy leg 1")
    require(args.leg1_goal_source == "own", "3-leg benchmark forbids goal swapping")
    require(not args.stop_after_leg1, "3-leg benchmark cannot stop after leg 1")
    require(not args.reset_memory, "3-leg benchmark must preserve memory across goals")
    require(args.retrieval_override == "off", "3-leg benchmark forbids GT retrieval")
    require(args.terminal_uturn == "off", "3-leg position SR disables terminal U-turn")
    require(
        args.terminal_visual_refine == "off",
        "3-leg position SR disables visual terminal refinement",
    )
    if args.server_backend == "hybrid_pose":
        require(args.novel_port is not None, "hybrid_pose requires --novel_port")
        require(
            args.hybrid_route in base.AUTO_HYBRID_ROUTES,
            "3-leg hybrid evaluation requires automatic routing",
        )
    if args.navdp_goal_switch_reset != "carry":
        require(
            args.deterministic_plan_seeds,
            "goal-switch reset ablations require --deterministic_plan_seeds",
        )
    if args.oracle_candidate_seed_count > 1:
        require(
            args.server_backend == "navdp",
            "multi-seed candidate pooling currently supports native NavDP only",
        )
    if args.oracle_global_subgoal_m > 0:
        require(
            args.server_backend == "navdp",
            "oracle global subgoals currently support native NavDP only",
        )
        require(
            args.deterministic_plan_seeds,
            "oracle global subgoals require deterministic plan seeds",
        )
        require(
            args.trajectory_selector == "server",
            "oracle global subgoals cannot be mixed with trajectory selection",
        )
        require(
            args.navdp_goal_switch_reset == "carry",
            "oracle global subgoals require carried NavDP short memory",
        )

    os.makedirs(args.out, exist_ok=True)
    episode_dirs = sorted(
        path
        for path in glob.glob(os.path.join(args.episode_root, "episode_*"))
        if os.path.isfile(os.path.join(path, "meta", "gen_meta.json"))
    )
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",") if item.strip()}
        episode_dirs = [
            path for path in episode_dirs if os.path.basename(path) in wanted
        ]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    require(bool(episode_dirs), "no 3-leg episodes selected")

    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    print(
        f"[eval3leg] episodes={len(episode_dirs)} backend={args.server_backend} "
        f"route={args.hybrid_route} "
        f"navdp_goal_switch_reset={args.navdp_goal_switch_reset}"
    )
    metrics = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            with open(os.path.join(episode_dir, "meta", "gen_meta.json")) as handle:
                metadata = json.load(handle)
            require(int(metadata.get("n_legs", -1)) == 3, "selected episode is not 3-leg")
            require(len(metadata.get("switches", [])) == 2, "3-leg switches are invalid")
            require(len(metadata.get("goals", [])) == 2, "3-leg goals are invalid")
            goal_b, goal_c = metadata["goals"]
            require(goal_b.get("kind") == "novel", "Goal B must be Novel")
            require(goal_c.get("kind") == "revisit", "Goal C must be Revisit")

            rows = pd.read_parquet(
                os.path.join(
                    episode_dir, "data/chunk-000/episode_000000.parquet"
                )
            )
            require(len(rows) == int(metadata["n_frames"]), "parquet frame count mismatch")
            intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
            camera_intrinsic = np.stack(
                [np.asarray(row, dtype=np.float64) for row in intrinsic_raw]
            )
            switch_a, _switch_b = [int(value) for value in metadata["switches"]]
            rgb_root = os.path.join(
                episode_dir, "videos/chunk-000/observation.images.rgb"
            )

            a_hab = base.data_to_hab(metadata["A"])
            b_hab = base.data_to_hab(goal_b["pos"])
            c_hab = base.data_to_hab(goal_c["pos"])
            a_xz, b_xz, c_xz = a_hab[[0, 2]], b_hab[[0, 2]], c_hab[[0, 2]]
            start_floor, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])

            def floor_point(xz: np.ndarray) -> np.ndarray:
                return np.asarray([xz[0], start_floor[1], xz[1]], dtype=float)

            geodesics = []
            for source, target in (
                (start_floor, floor_point(a_xz)),
                (floor_point(a_xz), floor_point(b_xz)),
                (floor_point(b_xz), floor_point(c_xz)),
            ):
                ok, distance, _path = base.geodesic(pathfinder, source, target)
                require(ok and np.isfinite(distance), "three-leg geodesic is invalid")
                geodesics.append(float(distance))
            geo_a, geo_b, geo_c = geodesics

            with open(os.path.join(rgb_root, f"{switch_a - 1}.jpg"), "rb") as handle:
                image_a = handle.read()
            with open(os.path.join(episode_dir, "goal_1.jpg"), "rb") as handle:
                image_b = handle.read()
            with open(os.path.join(episode_dir, "goal_2.jpg"), "rb") as handle:
                image_c = handle.read()

            episode_seed = args.seed + episode_index
            base.srv_reset(
                camera_height=float(metadata.get("camera_height_m", base.CAM_H)),
                seed=episode_seed,
                episode_len=int(metadata["n_frames"]),
                camera_intrinsic=camera_intrinsic,
            )
            backend = policy_backend()
            leg_a = base.run_policy_leg(
                sim,
                pathfinder,
                start_floor,
                start_yaw,
                image_a,
                a_xz,
                geo_a,
                None,
                terminal_mode="off",
                forced_gate=args.gate_override,
                policy_backend=backend,
                success_dist=args.leg1_success_dist,
                episode_seed=episode_seed,
                leg_index=0,
            )
            position, yaw = leg_a["end_pos"], leg_a["end_psi"]

            leg_b = empty_leg(position, yaw, b_xz)
            reset_before_b = False
            if leg_a["reached"]:
                if should_reset_before_leg(args.navdp_goal_switch_reset, 1):
                    base.srv_reset_navdp_short_memory(env_id=0)
                    reset_before_b = True
                if args.oracle_global_subgoal_m > 0:
                    leg_b = run_oracle_global_subgoal_leg(
                        sim,
                        pathfinder,
                        position,
                        yaw,
                        image_b,
                        b_xz,
                        episode_seed,
                    )
                else:
                    leg_b = base.run_policy_leg(
                        sim,
                        pathfinder,
                        position,
                        yaw,
                        image_b,
                        b_xz,
                        geo_b,
                        None,
                        terminal_mode="off",
                        goal_yaw=float(goal_b["yaw_habitat"]),
                        camera_intrinsic=camera_intrinsic,
                        forced_gate=args.gate_override,
                        policy_backend=backend,
                        episode_seed=episode_seed,
                        leg_index=1,
                    )
            position, yaw = leg_b["end_pos"], leg_b["end_psi"]

            leg_c = empty_leg(position, yaw, c_xz)
            reset_before_c = False
            if leg_a["reached"] and leg_b["reached"]:
                if should_reset_before_leg(args.navdp_goal_switch_reset, 2):
                    base.srv_reset_navdp_short_memory(env_id=0)
                    reset_before_c = True
                leg_c = base.run_policy_leg(
                    sim,
                    pathfinder,
                    position,
                    yaw,
                    image_c,
                    c_xz,
                    geo_c,
                    None,
                    terminal_mode="off",
                    goal_yaw=float(goal_c["yaw_habitat"]),
                    camera_intrinsic=camera_intrinsic,
                    forced_gate=args.gate_override,
                    policy_backend=backend,
                    episode_seed=episode_seed,
                    leg_index=2,
                )

            route_a = router_stats(leg_a["plans"])
            route_b = router_stats(leg_b["plans"])
            route_c = router_stats(leg_c["plans"])
            reached_a = bool(leg_a["reached"])
            reached_b = bool(leg_b["reached"])
            reached_c = bool(leg_c["reached"])
            joint = reached_a and reached_b and reached_c
            total_path = sum(leg["path_len"] for leg in (leg_a, leg_b, leg_c))
            metric = {
                "episode": os.path.basename(episode_dir),
                "seed": episode_seed,
                "server_backend": args.server_backend,
                "hybrid_route": args.hybrid_route,
                "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
                "trajectory_selector": args.trajectory_selector,
                "trajectory_selector_scope": args.trajectory_selector_scope,
                "oracle_selector_horizon": (
                    args.oracle_selector_horizon
                    if args.trajectory_selector == "oracle_geodesic" else None
                ),
                "oracle_candidate_seed_count": args.oracle_candidate_seed_count,
                "oracle_global_subgoal_m": args.oracle_global_subgoal_m,
                "navdp_reset_before_B": int(reset_before_b),
                "navdp_reset_before_C": int(reset_before_c),
                "reached_A": int(reached_a),
                "reached_B": int(reached_b),
                "reached_C": int(reached_c),
                "joint_success": int(joint),
                "spl_A": leg_spl(leg_a, geo_a),
                "spl_B": leg_spl(leg_b, geo_b),
                "spl_C": leg_spl(leg_c, geo_c),
                "joint_spl": base.spl(joint, geo_a + geo_b + geo_c, total_path),
                "geo_A": geo_a,
                "geo_B": geo_b,
                "geo_C": geo_c,
                "len_A": leg_a["path_len"],
                "len_B": leg_b["path_len"],
                "len_C": leg_c["path_len"],
                "steps_A": leg_a["steps"],
                "steps_B": leg_b["steps"],
                "steps_C": leg_c["steps"],
                "final_dist_A": leg_a["final_goal_dist_m"],
                "final_dist_B": leg_b["final_goal_dist_m"],
                "final_dist_C": leg_c["final_goal_dist_m"],
                "c_recall_gap": int(goal_c["recall_gap"]),
                "c_gt_covis_anchor": int(goal_c["covis_argmax"]),
                "router_plans_A": route_a["plans"],
                "router_plans_B": route_b["plans"],
                "router_plans_C": route_c["plans"],
                "router_active_plans_A": route_a["active_plans"],
                "router_active_plans_B": route_b["active_plans"],
                "router_active_plans_C": route_c["active_plans"],
                "router_active_episode_A": int(route_a["active_episode"]),
                "router_active_episode_B": int(route_b["active_episode"]),
                "router_active_episode_C": int(route_c["active_episode"]),
            }
            metrics.append(metric)
            with open(
                os.path.join(args.out, metric["episode"] + "_plans.json"), "w"
            ) as handle:
                json.dump(
                    {"legA": leg_a["plans"], "legB": leg_b["plans"], "legC": leg_c["plans"]},
                    handle,
                )
            with open(os.path.join(args.out, "metric.csv"), "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{metric['episode']}] A={int(reached_a)} B={int(reached_b)} "
                f"C={int(reached_c)} gap={metric['c_recall_gap']}"
            )

        reached_a_rows = [row for row in metrics if row["reached_A"]]
        reached_ab_rows = [
            row for row in reached_a_rows if row["reached_B"]
        ]
        summary = {
            "episodes": len(metrics),
            "server_backend": args.server_backend,
            "hybrid_route": args.hybrid_route,
            "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
            "trajectory_selector": args.trajectory_selector,
            "trajectory_selector_scope": args.trajectory_selector_scope,
            "oracle_selector_horizon": (
                args.oracle_selector_horizon
                if args.trajectory_selector == "oracle_geodesic" else None
            ),
            "oracle_candidate_seed_count": args.oracle_candidate_seed_count,
            "oracle_global_subgoal_m": args.oracle_global_subgoal_m,
            "SR_A": mean_or_none([row["reached_A"] for row in metrics]),
            "SR_B_given_A": mean_or_none([row["reached_B"] for row in reached_a_rows]),
            "SR_C_given_AB": mean_or_none([row["reached_C"] for row in reached_ab_rows]),
            "joint_SR": mean_or_none([row["joint_success"] for row in metrics]),
            "mean_joint_spl": mean_or_none([row["joint_spl"] for row in metrics]),
            "novel_A_router_false_activation_rate": mean_or_none(
                [row["router_active_episode_A"] for row in metrics]
            ),
            "novel_B_router_false_activation_rate_given_A": mean_or_none(
                [row["router_active_episode_B"] for row in reached_a_rows]
            ),
            "revisit_C_router_activation_rate_given_AB": mean_or_none(
                [row["router_active_episode_C"] for row in reached_ab_rows]
            ),
        }
        with open(os.path.join(args.out, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        print("[eval3leg] done", summary)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
