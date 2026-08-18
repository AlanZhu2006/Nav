#!/usr/bin/env python3
"""True three-leg Habitat evaluation built on the audited 2-leg controller.

The existing evaluator owns all HTTP, waypoint, collision and router logic.
This entry point deliberately reuses those functions but executes the complete
start->A->B->C protocol.  It supports both strict Novel-then-Revisit data and
strict double-Revisit data; causal roles come from the audited metadata.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
from conditional_c_protocol import world_goal_to_local
from global_subgoal_protocol import polyline_subgoal
from navdp_goal_switch import should_reset_before_leg
from observed_frontier import CoverageResidualTrigger, ObservedFrontierGrid
from multigoal_benchmark_contract import (
    DOUBLE_REVISIT_PROTOCOL,
    DOUBLE_REVISIT_SEQUENCE,
    DoubleRevisitObservation,
    ROLE_SYMMETRIC_PROTOCOL,
    ROLE_SEQUENCE,
    RoleSymmetryObservation,
    validate_double_revisit_contract,
    validate_role_symmetric_contract,
)
from multigoal_policy_contract import three_leg_policy_backends
from double_revisit_diagnostics import online_path_nearest_anchor


args = base.args


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def replay_shared_leg_b(
        sim, trace_root, episode, episode_seed, goal_jpg,
        expected_start_position, expected_start_yaw):
    """Replay a frozen online Goal-B rollout into both memory systems.

    The trace uses the same audited schema as ``base.replay_shared_leg1``.
    Every RGB observation is re-rendered and hash checked before it is added
    to long-term memory; only the originally sampled decision frames are
    restored to NavDP's bounded FIFO.  This makes A->B a factual shared prefix
    and lets paired arms diverge only when Goal C begins.
    """
    trace_path = Path(trace_root) / f"{episode}_legB_trace.json"
    payload, trace_sha = base.load_leg1_trace(
        trace_path,
        expected_episode=episode,
        expected_seed=int(episode_seed),
        expected_goal_sha256=base.bytes_sha256(goal_jpg),
        expected_source_scene=Path(args.scene).stem,
    )
    if payload["goal_source_episode"] != episode:
        raise RuntimeError("shared trace does not use the episode's own Goal B")
    base.validate_shared_trace_source(payload)
    if payload["source_retrieval_candidate_min_gap"] != 16:
        raise RuntimeError("shared trace source candidate gap is not 16")
    if not np.isclose(
            float(payload["source_graph_subgoal_spacing_m"]), 0.0,
            rtol=0.0, atol=1e-12):
        raise RuntimeError("shared trace source is not the direct controller")
    if not np.isclose(
            float(payload["source_graph_subgoal_arrival_m"]), 0.60,
            rtol=0.0, atol=1e-12):
        raise RuntimeError("shared trace source graph arrival changed")

    poses = payload["poses"]
    if poses:
        first = poses[0]
        first_position = np.asarray(
            [first["x"], first["y"], first["z"]], dtype=float)
        if not np.allclose(
                first_position, expected_start_position,
                rtol=0.0, atol=1e-6):
            raise RuntimeError("shared Goal-B trace start position mismatch")
        if abs(base.wrap_angle(
                float(first["yaw"]) - expected_start_yaw)) > 1e-6:
            raise RuntimeError("shared Goal-B trace start yaw mismatch")

    memory_trace = []
    plan_steps = [int(plan["step"]) for plan in payload["plans"]]
    if len(plan_steps) != len(set(plan_steps)):
        raise RuntimeError("shared Goal-B trace contains duplicate plan steps")
    plan_step_set = set(plan_steps)
    navdp = None
    navdp_queue_lengths = None
    for pose in poses:
        floor_position = np.asarray(
            [pose["x"], pose["y"], pose["z"]], dtype=float)
        rgb, _depth = base.render(
            sim,
            floor_position + np.asarray([0.0, base.CAM_H, 0.0]),
            float(pose["yaw"]),
        )
        frame = base.jpg_bytes(rgb)
        if base.bytes_sha256(frame) != pose.get("jpg_sha256"):
            raise RuntimeError("shared Goal-B trace rendered RGB mismatch")
        response = base.srv_memory(frame)
        frame_idx = response.get("frame_idx")
        if frame_idx is not None:
            memory_trace.append({
                "frame_idx": int(frame_idx),
                "step": int(pose["step"]),
                "x": float(pose["x"]),
                "z": float(pose["z"]),
                "yaw": float(pose["yaw"]),
            })
        if int(pose["step"]) in plan_step_set:
            navdp = base.srv_navdp_memory_replay(frame)
            navdp_queue_lengths = navdp.get("queue_lengths")

    if plan_steps:
        if navdp is None:
            raise RuntimeError("Goal-B NavDP replay did not execute")
        memory_size = int(navdp.get("memory_size", -1))
        if memory_size <= 0:
            raise RuntimeError("NavDP replay endpoint omitted memory size")
        expected_length = min(len(plan_steps), memory_size)
        if navdp_queue_lengths != [expected_length]:
            raise RuntimeError(
                "NavDP replay queue length does not match frozen plan count")

    return {
        "reached": bool(payload["reached"]),
        "path_len": float(payload["path_len"]),
        "path_len_at_reach": payload.get("path_len_at_reach"),
        "step_at_reach": payload.get("step_at_reach"),
        "steps": int(payload["steps"]),
        "termination_reason": payload.get("termination_reason"),
        "blocked_step_count": int(payload.get("blocked_step_count", 0)),
        "plans": payload["plans"],
        "memory_trace": memory_trace,
        "rollout_trace": poses,
        "end_pos": np.asarray(payload["end_position"], dtype=float),
        "end_psi": float(payload["end_yaw"]),
        "final_goal_dist_m": float(payload["final_goal_dist_m"]),
        "navdp_replayed_plan_steps": plan_steps,
    }, trace_sha


def empty_leg(pos: np.ndarray, psi: float, goal_xz: np.ndarray) -> dict:
    return {
        "reached": False,
        "path_len": 0.0,
        "path_len_at_reach": None,
        "step_at_reach": None,
        "steps": 0,
        "termination_reason": "causally_censored",
        "plans": [],
        # A downstream leg can be causally censored when an earlier goal was
        # not reached.  Keep the trace schema total so observation-only
        # collectors can record that censoring instead of crashing while
        # serializing the episode.
        "memory_trace": [],
        "rollout_trace": [],
        "end_pos": np.asarray(pos, dtype=float),
        "end_psi": float(psi),
        "final_goal_dist_m": float(
            np.linalg.norm(np.asarray(pos, dtype=float)[[0, 2]] - goal_xz)
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


def seed_observed_frontier(
    sim,
    rollout_trace: list[dict],
    camera_intrinsic: np.ndarray,
) -> ObservedFrontierGrid:
    """Reconstruct the sensor observations actually available on Goal A."""
    frontier = ObservedFrontierGrid()
    for trace_index in range(0, len(rollout_trace), args.exec_horizon):
        pose = rollout_trace[trace_index]
        floor_position = np.asarray(
            [pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        _rgb, depth = base.render(
            sim,
            floor_position + np.asarray([0.0, base.CAM_H, 0.0]),
            float(pose["yaw"]),
        )
        frontier.integrate_depth(
            depth,
            floor_position,
            float(pose["yaw"]),
            camera_intrinsic,
        )
    return frontier


def run_observed_frontier_leg(
    sim,
    pathfinder,
    start_position: np.ndarray,
    start_yaw: float,
    goal_jpg: bytes,
    goal_xz: np.ndarray,
    episode_seed: int,
    camera_intrinsic: np.ndarray,
    frontier: ObservedFrontierGrid,
) -> dict:
    """Explore Novel-B using only observed free-space frontiers.

    ``goal_xz`` is used only for benchmark success/final-distance accounting.
    Frontier ranking and NavDP point-goals never receive it.  Evaluator pose,
    metric Habitat depth and navmesh reachability still make this an explicit
    feasibility upper bound rather than a deployable LingBot implementation.
    """
    position = np.asarray(start_position, dtype=np.float64).copy()
    yaw = float(start_yaw)
    path_len = 0.0
    path_len_at_reach = None
    way_world = None
    plans: list[dict] = []
    history: list[np.ndarray] = []
    rejected_targets: list[np.ndarray] = []
    target_xz: np.ndarray | None = None
    target_diag: dict | None = None
    success_dist = float(args.success_dist)
    frontier_arrival_m = 0.60
    frontier_subgoal_m = 1.25
    residual_trigger = CoverageResidualTrigger(
        threshold_m=0.60, confirm_plans=3)
    frontier_mode = args.oracle_observed_frontier

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

    def select_frontier_target() -> tuple[np.ndarray | None, dict | None]:
        candidates = frontier.ranked_frontiers(
            position[[0, 2]], excluded_world_xz=rejected_targets)
        for rank, candidate in enumerate(candidates):
            raw = np.asarray(candidate.world_xz, dtype=np.float64)
            snapped = np.asarray(pathfinder.snap_point(
                [raw[0], position[1], raw[1]]), dtype=np.float64)
            if (not np.isfinite(snapped).all()
                    or np.linalg.norm(snapped[[0, 2]] - raw) > 0.35):
                continue
            ok, distance_m, path_points = base.geodesic(
                pathfinder, position, snapped)
            if (not ok or not np.isfinite(distance_m)
                    or distance_m <= frontier_arrival_m or not path_points):
                continue
            diagnostics = candidate.to_dict()
            diagnostics.update(
                candidate_rank=int(rank),
                snapped_world=snapped.tolist(),
                target_geodesic_m=float(distance_m),
            )
            return snapped[[0, 2]], diagnostics
        return None, None

    if np.linalg.norm(position[[0, 2]] - goal_xz) < success_dist:
        path_len_at_reach = 0.0
        return result(True, 0)

    for step in range(args.max_steps):
        rgb, depth = base.render(
            sim, position + np.asarray([0.0, base.CAM_H, 0.0]), yaw)
        frame = base.jpg_bytes(rgb)

        if step % args.exec_horizon == 0:
            frontier.integrate_depth(
                depth, position, yaw, camera_intrinsic)
            if (target_xz is not None
                    and np.linalg.norm(position[[0, 2]] - target_xz)
                    <= frontier_arrival_m):
                rejected_targets.append(target_xz.copy())
                target_xz, target_diag = None, None
                residual_trigger.reset()

            target_path = None
            target_remaining = None
            if target_xz is not None:
                target3 = np.asarray(
                    [target_xz[0], position[1], target_xz[1]],
                    dtype=np.float64,
                )
                ok, target_remaining, target_path = base.geodesic(
                    pathfinder, position, target3)
                if not ok or not np.isfinite(target_remaining) or not target_path:
                    rejected_targets.append(target_xz.copy())
                    target_xz, target_diag = None, None
                    target_path, target_remaining = None, None

            # The native ImageGoal request always runs first and is the only
            # request allowed to append this observation to NavDP's FIFO.
            # A residual frontier proposal, when needed, is sampled from that
            # exact FIFO through the read-only endpoint below.
            request_seed = base.diffusion_plan_seed(
                int(episode_seed), 1, len(plans))
            native_response = base.requests.post(
                f"{base.BASE}/imagegoal_step",
                files={
                    "image": ("image.jpg", frame),
                    "goal": ("goal.jpg", goal_jpg),
                    "depth": ("depth.png", base.depth_png_bytes(depth)),
                },
                data={"diffusion_seed": str(request_seed)},
            )
            native_response.raise_for_status()
            native_json = native_response.json()
            if int(native_json.get("diffusion_seed", -1)) != request_seed:
                raise RuntimeError(
                    "NavDP server did not echo native diffusion seed")
            native_plan = base.normalize_navdp_response(native_json)
            native_way, native_selector = base.select_plan_trajectory(
                native_plan,
                position,
                yaw,
                pathfinder,
                position[[0, 2]],
                trajectory_selector="server",
            )
            native_way_world = base.waypoints_to_world(
                native_way, position[[0, 2]], yaw)
            native_endpoint_novelty_m = (
                frontier.distance_to_visited(native_way_world[-1])
                if len(native_way_world) else None)
            native_repetition_confirmed = residual_trigger.observe(
                native_endpoint_novelty_m)

            should_activate = (
                frontier_mode == "always"
                or (
                    frontier_mode == "residual"
                    and native_repetition_confirmed
                )
            )
            if should_activate and target_xz is None:
                target_xz, target_diag = select_frontier_target()
                if target_xz is not None:
                    target3 = np.asarray(
                        [target_xz[0], position[1], target_xz[1]],
                        dtype=np.float64,
                    )
                    ok, target_remaining, target_path = base.geodesic(
                        pathfinder, position, target3)
                    require(ok and np.isfinite(target_remaining) and target_path,
                            "selected frontier target became unreachable")

            residual_active = target_xz is not None and bool(target_path)
            if residual_active:
                subgoal = polyline_subgoal(target_path, frontier_subgoal_m)
                local_goal = world_goal_to_local(
                    subgoal[[0, 2]], position[[0, 2]], yaw)
                response = base.requests.post(
                    f"{base.BASE}/mixgoal_resample",
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
                if response_json.get("memory_mutated") is not False:
                    raise RuntimeError(
                        "frontier residual mutated NavDP observation FIFO")
                controller = "observed_frontier_residual_image_point_mix"
            else:
                subgoal = None
                local_goal = None
                response_json = native_json
                controller = "observed_frontier_native_imagegoal"
            if int(response_json.get("diffusion_seed", -1)) != request_seed:
                raise RuntimeError(
                    "NavDP server did not echo observed-frontier diffusion seed")
            plan = base.normalize_navdp_response(response_json)
            selector_goal = (
                target_xz if target_xz is not None else position[[0, 2]])
            way, selector = base.select_plan_trajectory(
                plan,
                position,
                yaw,
                pathfinder,
                selector_goal,
                trajectory_selector="server",
            )
            way_world = base.waypoints_to_world(
                way, position[[0, 2]], yaw)
            selected_endpoint_novelty_m = frontier.distance_to_visited(
                way_world[-1]) if len(way_world) else None
            ranked_frontiers = frontier.ranked_frontiers(
                position[[0, 2]], excluded_world_xz=rejected_targets)
            plans.append({
                "step": int(step),
                "current_x": float(position[0]),
                "current_z": float(position[2]),
                "current_yaw": float(yaw),
                "evaluation_gt_goal_distance_m": float(np.linalg.norm(
                    position[[0, 2]] - goal_xz)),
                "frontier_target_world": (
                    target_xz.tolist() if target_xz is not None else None),
                "frontier_target": target_diag,
                "frontier_target_geodesic_m": (
                    float(target_remaining)
                    if target_remaining is not None else None),
                "frontier_subgoal_world": (
                    subgoal.tolist() if subgoal is not None else None),
                "frontier_subgoal_local": (
                    local_goal.tolist() if local_goal is not None else None),
                "frontier_map": frontier.summary(),
                "frontier_mode": frontier_mode,
                "selected_endpoint_world": (
                    way_world[-1].tolist() if len(way_world) else None),
                "selected_endpoint_novelty_m": selected_endpoint_novelty_m,
                "native_endpoint_world": (
                    native_way_world[-1].tolist()
                    if len(native_way_world) else None),
                "native_endpoint_novelty_m": native_endpoint_novelty_m,
                "native_low_novelty_streak": int(residual_trigger.streak),
                "native_novelty_threshold_m": residual_trigger.threshold_m,
                "residual_confirm_plans": residual_trigger.confirm_plans,
                "frontier_residual_active": bool(residual_active),
                "frontier_top_shadow": (
                    ranked_frontiers[0].to_dict()
                    if ranked_frontiers else None),
                "frontier_rejected_targets": len(rejected_targets),
                "pose_controller": controller,
                "router_active": None,
                "router_reason": "goal_blind_observed_frontier",
                "diffusion_seed": plan.get("diffusion_seed"),
                "requested_diffusion_seed": request_seed,
                "navdp_critic_max": plan.get("navdp_critic_max"),
                "native_navdp_critic_max": native_plan.get("navdp_critic_max"),
                "native_trajectory_selector": native_selector,
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
    require(
        args.leg1_mode in ("policy", "shared_trace"),
        "3-leg benchmark requires policy or shared_trace leg 1",
    )
    if args.leg1_mode == "shared_trace":
        require(
            bool(args.shared_leg1_trace_root),
            "shared_trace requires --shared_leg1_trace_root",
        )
        require(
            not args.write_leg1_trace,
            "shared_trace cannot overwrite its frozen source traces",
        )
        require(
            args.oracle_observed_frontier == "off"
            and args.oracle_global_subgoal_m == 0,
            "shared_trace replays factual A and B and forbids oracle B arms",
        )
    else:
        require(
            not args.shared_leg1_trace_root,
            "--shared_leg1_trace_root is valid only with shared_trace",
        )
    require(args.leg1_goal_source == "own", "3-leg benchmark forbids goal swapping")
    require(not args.stop_after_leg1, "3-leg benchmark cannot stop after leg 1")
    require(not args.reset_memory, "3-leg benchmark must preserve memory across goals")
    require(
        args.retrieval_override in ("off", "gt_path_nearest"),
        "3-leg benchmark only permits the explicit C path-nearest oracle",
    )
    if args.retrieval_override == "gt_path_nearest":
        require(
            args.server_backend == "hybrid_pose"
            and args.hybrid_route == "phase"
            and args.double_revisit_c_history == "initial_leg_only",
            "3-leg C path-nearest oracle requires known-role hybrid_pose and "
            "strict initial_leg_only history",
        )
    require(args.terminal_uturn == "off", "3-leg position SR disables terminal U-turn")
    require(
        args.terminal_visual_refine == "off",
        "3-leg position SR disables visual terminal refinement",
    )
    if args.server_backend == "hybrid_pose":
        require(args.novel_port is not None, "hybrid_pose requires --novel_port")
    base.validate_revisit_adapter_configuration(
        mode=args.revisit_adapter,
        server_backend=args.server_backend,
        revisit_controller=args.revisit_controller,
        router_is_automatic_geometry=(
            args.hybrid_route in base.AUTO_HYBRID_ROUTES),
        router_is_certified_relocalization=(
            args.hybrid_route == "certified_relocalization"),
    )
    if args.hybrid_route == "certified_relocalization":
        require(
            args.revisit_adapter == "verified_bearing_v1",
            "certified_relocalization requires verified_bearing_v1",
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
    if args.oracle_observed_frontier != "off":
        require(
            args.server_backend == "navdp",
            "observed-frontier diagnostics currently support native NavDP only",
        )
        require(
            args.deterministic_plan_seeds,
            "observed-frontier diagnostics require deterministic plan seeds",
        )
        require(
            args.trajectory_selector == "server",
            "observed frontiers cannot be mixed with trajectory selection",
        )
        require(
            args.navdp_goal_switch_reset == "carry",
            "observed frontiers require carried NavDP short memory",
        )
        require(
            args.oracle_global_subgoal_m == 0,
            "observed frontier and GT global-subgoal arms are mutually exclusive",
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

    episode_contracts = []
    for episode_dir in episode_dirs:
        with open(os.path.join(
                episode_dir, "meta", "gen_meta.json")) as handle:
            episode_contracts.append(json.load(handle))
    role_sequences = {
        tuple(metadata.get("role_sequence") or ())
        for metadata in episode_contracts
    }
    generation_protocols = {
        metadata.get("gen_protocol") for metadata in episode_contracts
    }
    require(
        len(role_sequences) == 1 and len(generation_protocols) == 1,
        "one evaluation run cannot mix multi-goal protocols or role sequences",
    )
    role_sequence = next(iter(role_sequences))
    generation_protocol = next(iter(generation_protocols))
    require(
        (generation_protocol, role_sequence) in {
            (ROLE_SYMMETRIC_PROTOCOL, ROLE_SEQUENCE),
            (DOUBLE_REVISIT_PROTOCOL, DOUBLE_REVISIT_SEQUENCE),
        },
        "selected episodes do not satisfy a supported strict 3-leg protocol",
    )
    backends = three_leg_policy_backends(
        server_backend=args.server_backend,
        hybrid_route=args.hybrid_route,
        automatic_routes=base.AUTO_HYBRID_ROUTES,
        role_sequence=role_sequence,
    )

    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    print(
        f"[eval3leg] episodes={len(episode_dirs)} backend={args.server_backend} "
        f"route={args.hybrid_route} "
        f"leg_backends={backends} "
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
            require(
                tuple(metadata.get("role_sequence") or ()) == role_sequence,
                "episode role sequence differs within one evaluation run",
            )
            require(
                goal_b.get("kind") == role_sequence[1],
                f"Goal B must be {role_sequence[1]}",
            )
            require(
                goal_c.get("kind") == role_sequence[2],
                f"Goal C must be {role_sequence[2]}",
            )

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
            switch_a, switch_b = [int(value) for value in metadata["switches"]]
            rgb_root = os.path.join(
                episode_dir, "videos/chunk-000/observation.images.rgb"
            )

            a_hab = base.data_to_hab(metadata["A"])
            b_hab = base.data_to_hab(goal_b["pos"])
            c_hab = base.data_to_hab(goal_c["pos"])
            a_xz, b_xz, c_xz = a_hab[[0, 2]], b_hab[[0, 2]], c_hab[[0, 2]]
            start_floor, start_yaw = base.parquet_pose_hab(rows.iloc[0]["action"])

            geodesics = []
            for source, target in (
                (start_floor, a_hab),
                (a_hab, b_hab),
                (b_hab, c_hab),
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
            with open(os.path.join(rgb_root, f"{switch_b - 1}.jpg"), "rb") as handle:
                image_b_terminal = handle.read()

            a_terminal, _a_terminal_yaw = base.parquet_pose_hab(
                rows.iloc[switch_a - 1]["action"])
            b_terminal, b_terminal_yaw = base.parquet_pose_hab(
                rows.iloc[switch_b - 1]["action"])
            common_contract = {
                "geo_a_m": float(geo_a),
                "geo_b_m": float(geo_b),
                "initial_pose_error_m": float(np.linalg.norm(
                    start_floor - base.data_to_hab(metadata["start"])
                )),
                "a_terminal_pose_error_m": float(np.linalg.norm(
                    a_terminal - a_hab
                )),
            }
            if generation_protocol == DOUBLE_REVISIT_PROTOCOL:
                camera_height = float(metadata.get(
                    "camera_height_m", base.CAM_H))
                rendered_b, _ = base.render(
                    sim,
                    b_hab + np.asarray([0.0, camera_height, 0.0]),
                    float(goal_b["yaw_habitat"]),
                )
                rendered_c, _ = base.render(
                    sim,
                    c_hab + np.asarray([0.0, camera_height, 0.0]),
                    float(goal_c["yaw_habitat"]),
                )
                contract_observation = DoubleRevisitObservation(
                    **common_contract,
                    geo_c_m=float(geo_c),
                    goal_b_matches_render=(
                        image_b == base.jpg_bytes(rendered_b)),
                    goal_c_matches_render=(
                        image_c == base.jpg_bytes(rendered_c)),
                )
                multigoal_contract = validate_double_revisit_contract(
                    metadata, contract_observation)
            else:
                contract_observation = RoleSymmetryObservation(
                    **common_contract,
                    b_terminal_pose_error_m=float(np.linalg.norm(
                        b_terminal - b_hab)),
                    b_terminal_yaw_error_deg=abs(float(np.degrees(
                        base.wrap_angle(
                            b_terminal_yaw
                            - float(goal_b["yaw_habitat"])
                        )
                    ))),
                    goal_b_matches_terminal_rgb=(image_b == image_b_terminal),
                )
                multigoal_contract = validate_role_symmetric_contract(
                    metadata, contract_observation)
            if (not multigoal_contract["ok"]
                    and not args.allow_legacy_multigoal_data):
                raise RuntimeError(
                    "strict 3-leg data contract failed; pass "
                    "--allow_legacy_multigoal_data only to reproduce the old "
                    "confounded benchmark: "
                    + "; ".join(multigoal_contract["issues"]))

            episode_seed = args.seed + episode_index
            base.srv_reset(
                camera_height=float(metadata.get("camera_height_m", base.CAM_H)),
                seed=episode_seed,
                episode_len=int(metadata["n_frames"]),
                camera_intrinsic=camera_intrinsic,
            )
            episode_name = os.path.basename(episode_dir)
            leg_a_trace_sha256 = None
            leg_b_trace_sha256 = None
            if args.leg1_mode == "shared_trace":
                leg_a, leg_a_trace_sha256 = base.replay_shared_leg1(
                    sim,
                    args.shared_leg1_trace_root,
                    episode_name,
                    episode_seed,
                    image_a,
                    start_floor,
                    start_yaw,
                )
            else:
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
                    policy_backend=backends[0],
                    success_dist=args.leg1_success_dist,
                    episode_seed=episode_seed,
                    leg_index=0,
                )
                if args.write_leg1_trace:
                    leg_a_trace_sha256 = base.write_leg1_trace(
                        Path(args.out) / f"{episode_name}_leg1_trace.json",
                        base.leg1_trace_payload(
                            episode=episode_name,
                            episode_seed=episode_seed,
                            goal_jpg=image_a,
                            goal_source_episode=episode_name,
                            source_scene=Path(args.scene).stem,
                            leg=leg_a,
                        ),
                    )
            position, yaw = leg_a["end_pos"], leg_a["end_psi"]
            c_candidate_ceiling = None
            c_forced_anchor = None
            c_path_nearest_dist_m = None
            if (generation_protocol == DOUBLE_REVISIT_PROTOCOL
                    and args.double_revisit_c_history == "initial_leg_only"
                    and args.server_backend == "hybrid_pose"):
                a_memory_frames = [
                    int(item["frame_idx"])
                    for item in leg_a["memory_trace"]
                    if item.get("frame_idx") is not None
                ]
                require(
                    bool(a_memory_frames),
                    "strict double-Revisit C requires an online leg-A memory trace",
                )
                require(
                    a_memory_frames == list(range(
                        a_memory_frames[0], a_memory_frames[-1] + 1)),
                    "online leg-A memory frame indices are not contiguous",
                )
                c_candidate_ceiling = int(a_memory_frames[-1])
                if args.retrieval_override == "gt_path_nearest":
                    nearest = online_path_nearest_anchor(
                        leg_a["memory_trace"],
                        c_xz,
                        candidate_ceiling=c_candidate_ceiling,
                    )
                    c_forced_anchor = int(nearest["frame_idx"])
                    c_path_nearest_dist_m = float(nearest["distance_m"])

            if args.retrieval_override == "gt_path_nearest":
                require(
                    generation_protocol == DOUBLE_REVISIT_PROTOCOL,
                    "3-leg C path-nearest oracle is double-Revisit only",
                )
                require(
                    c_forced_anchor is not None,
                    "C path-nearest oracle did not resolve an online-A anchor",
                )

            leg_b = empty_leg(position, yaw, b_xz)
            reset_before_b = False
            if leg_a["reached"]:
                if should_reset_before_leg(args.navdp_goal_switch_reset, 1):
                    base.srv_reset_navdp_short_memory(env_id=0)
                    reset_before_b = True
                if args.leg1_mode == "shared_trace":
                    leg_b, leg_b_trace_sha256 = replay_shared_leg_b(
                        sim,
                        args.shared_leg1_trace_root,
                        episode_name,
                        episode_seed,
                        image_b,
                        position,
                        yaw,
                    )
                elif args.oracle_observed_frontier != "off":
                    frontier = seed_observed_frontier(
                        sim, leg_a["rollout_trace"], camera_intrinsic)
                    leg_b = run_observed_frontier_leg(
                        sim,
                        pathfinder,
                        position,
                        yaw,
                        image_b,
                        b_xz,
                        episode_seed,
                        camera_intrinsic,
                        frontier,
                    )
                elif args.oracle_global_subgoal_m > 0:
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
                        policy_backend=backends[1],
                        episode_seed=episode_seed,
                        leg_index=1,
                    )
                    if args.write_leg1_trace:
                        leg_b_trace_sha256 = base.write_leg1_trace(
                            Path(args.out) / f"{episode_name}_legB_trace.json",
                            base.leg1_trace_payload(
                                episode=episode_name,
                                episode_seed=episode_seed,
                                goal_jpg=image_b,
                                goal_source_episode=episode_name,
                                source_scene=Path(args.scene).stem,
                                leg=leg_b,
                            ),
                        )
            position, yaw = leg_b["end_pos"], leg_b["end_psi"]

            leg_c = empty_leg(position, yaw, c_xz)
            reset_before_c = False
            prefix_recording_only = bool(
                args.leg1_mode == "policy" and args.write_leg1_trace
            )
            if prefix_recording_only:
                leg_c["termination_reason"] = "shared_prefix_recording_only"
            if (leg_a["reached"] and leg_b["reached"]
                    and not prefix_recording_only):
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
                    policy_backend=backends[2],
                    episode_seed=episode_seed,
                    leg_index=2,
                    forced_anchor=c_forced_anchor,
                    candidate_ceiling_override=c_candidate_ceiling,
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
                "policy_backend_A": backends[0],
                "policy_backend_B": backends[1],
                "policy_backend_C": backends[2],
                "shared_prefix_mode": args.leg1_mode,
                "leg_A_trace_sha256": leg_a_trace_sha256,
                "leg_B_trace_sha256": leg_b_trace_sha256,
                "generation_protocol": metadata.get("gen_protocol"),
                "role_sequence": json.dumps(metadata.get("role_sequence")),
                "multigoal_contract_ok": int(multigoal_contract["ok"]),
                "multigoal_contract_issues": json.dumps(
                    multigoal_contract["issues"]),
                "legacy_multigoal_data_allowed": int(
                    args.allow_legacy_multigoal_data),
                "start_heading_offset_deg": metadata.get(
                    "start_heading_offset_deg"),
                "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
                "trajectory_selector": args.trajectory_selector,
                "trajectory_selector_scope": args.trajectory_selector_scope,
                "oracle_selector_horizon": (
                    args.oracle_selector_horizon
                    if args.trajectory_selector == "oracle_geodesic" else None
                ),
                "oracle_candidate_seed_count": args.oracle_candidate_seed_count,
                "oracle_global_subgoal_m": args.oracle_global_subgoal_m,
                "oracle_observed_frontier": args.oracle_observed_frontier,
                "navdp_reset_before_B": int(reset_before_b),
                "navdp_reset_before_C": int(reset_before_c),
                "double_revisit_c_history": args.double_revisit_c_history,
                "double_revisit_c_candidate_ceiling": c_candidate_ceiling,
                "retrieval_override": args.retrieval_override,
                "double_revisit_c_forced_anchor": c_forced_anchor,
                "double_revisit_c_path_nearest_dist_m": (
                    c_path_nearest_dist_m),
                "reached_A": int(reached_a),
                "reached_B": int(reached_b),
                "reached_C": int(reached_c),
                "C_evaluated": int(
                    reached_a and reached_b and not prefix_recording_only
                ),
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
                "b_recall_gap": (
                    int(goal_b["recall_gap"])
                    if goal_b.get("kind") == "revisit" else None
                ),
                "b_gt_covis_anchor": (
                    int(goal_b["covis_argmax"])
                    if goal_b.get("kind") == "revisit" else None
                ),
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
                    {
                        "legA": leg_a["plans"],
                        "legB": leg_b["plans"],
                        "legC": leg_c["plans"],
                        # Evaluation-only provenance for an offline causal
                        # teacher.  These traces never enter srv_plan or the
                        # executed controller; they only bind each query step
                        # and retrieved memory frame to the natural rollout
                        # pose and rendered-image hash.
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
                    },
                    handle,
                )
            with open(os.path.join(args.out, "metric.csv"), "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{metric['episode']}] A={int(reached_a)} B={int(reached_b)} "
                f"C={'not_run' if prefix_recording_only else int(reached_c)} "
                f"gap={metric['c_recall_gap']}"
            )

        reached_a_rows = [row for row in metrics if row["reached_A"]]
        reached_ab_rows = [
            row for row in reached_a_rows if row["reached_B"]
        ]
        evaluated_c_rows = [row for row in reached_ab_rows if row["C_evaluated"]]
        prefix_recording_only = bool(
            args.leg1_mode == "policy" and args.write_leg1_trace
        )
        summary = {
            "episodes": len(metrics),
            "server_backend": args.server_backend,
            "hybrid_route": args.hybrid_route,
            "policy_backends": {
                "A": backends[0],
                "B": backends[1],
                "C": backends[2],
            },
            "multigoal_contract": generation_protocol,
            "legacy_multigoal_data_allowed": bool(
                args.allow_legacy_multigoal_data),
            "contract_valid_episodes": sum(
                row["multigoal_contract_ok"] for row in metrics),
            "role_labels": {
                "A": role_sequence[0],
                "B": role_sequence[1],
                "C": role_sequence[2],
            },
            "navdp_goal_switch_reset": args.navdp_goal_switch_reset,
            "trajectory_selector": args.trajectory_selector,
            "trajectory_selector_scope": args.trajectory_selector_scope,
            "oracle_selector_horizon": (
                args.oracle_selector_horizon
                if args.trajectory_selector == "oracle_geodesic" else None
            ),
            "oracle_candidate_seed_count": args.oracle_candidate_seed_count,
            "oracle_global_subgoal_m": args.oracle_global_subgoal_m,
            "oracle_observed_frontier": args.oracle_observed_frontier,
            "double_revisit_c_history": args.double_revisit_c_history,
            "retrieval_override": args.retrieval_override,
            "shared_prefix_recording_only": prefix_recording_only,
            "SR_A": mean_or_none([row["reached_A"] for row in metrics]),
            "SR_B_given_A": mean_or_none([row["reached_B"] for row in reached_a_rows]),
            "SR_C_given_AB": mean_or_none(
                [row["reached_C"] for row in evaluated_c_rows]
            ),
            "joint_SR": (
                None if prefix_recording_only else
                mean_or_none([row["joint_success"] for row in metrics])
            ),
            "mean_joint_spl": (
                None if prefix_recording_only else
                mean_or_none([row["joint_spl"] for row in metrics])
            ),
            "novel_A_router_false_activation_rate": mean_or_none(
                [row["router_active_episode_A"] for row in metrics]
            ),
            "initial_A_router_activation_rate": mean_or_none(
                [row["router_active_episode_A"] for row in metrics]
            ),
            "router_activation_rate_B_given_A": mean_or_none(
                [row["router_active_episode_B"] for row in reached_a_rows]
            ),
            "router_activation_rate_C_given_AB": mean_or_none(
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
