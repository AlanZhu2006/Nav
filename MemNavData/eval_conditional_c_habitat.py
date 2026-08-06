#!/usr/bin/env python3
"""Causal conditional-C evaluation for generated three-leg episodes.

The source A/B RGB prefix is replayed through MemNav up to the frame directly
before Goal C.  Control begins only at that recorded B endpoint.  This isolates
revisit localization and local control from the separate Novel-A/B exploration
bottleneck while preserving a causal streaming memory.

This is a diagnostic protocol, not end-to-end three-leg SR.  ``oracle_anchor``
and ``oracle_point`` are explicitly privileged upper bounds.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_c_protocol import (
    CONDITIONAL_C_MODES,
    infer_mode,
    prefix_last_frame,
    world_goal_to_local,
)


# ``eval_2leg_habitat`` owns the audited common CLI.  Remove this evaluator's
# one additional flag before importing it so there is still a single source of
# truth for controller and router arguments.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument(
    "--conditional_c_mode",
    choices=CONDITIONAL_C_MODES,
    default="auto",
)
_conditional, _remaining = _pre_parser.parse_known_args()
sys.argv = [sys.argv[0], *_remaining]

import eval_2leg_habitat as base  # noqa: E402  (CLI must be filtered first)


args = base.args


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def replay_prefix(rgb_root: Path, rows: pd.DataFrame,
                  last_frame: int) -> tuple[list[dict], list[int]]:
    """Restore LingBot's full stream and NavDP's decision-frame queue."""
    trace = []
    navdp_steps = []
    for frame in range(last_frame + 1):
        image = read_bytes(rgb_root / f"{frame}.jpg")
        response = base.srv_memory(image)
        frame_idx = response.get("frame_idx")
        if frame_idx is not None and (
                frame in (0, last_frame) or frame % 32 == 0):
            position, yaw = base.parquet_pose_hab(rows.iloc[frame]["action"])
            trace.append({
                "source_frame": frame,
                "memory_frame": int(frame_idx),
                "x": float(position[0]),
                "z": float(position[2]),
                "yaw": float(yaw),
            })
        if frame % args.exec_horizon == 0:
            base.srv_navdp_memory_replay(image)
            navdp_steps.append(frame)
    return trace, navdp_steps


def oracle_point_leg(sim, pathfinder, start_position: np.ndarray,
                     start_yaw: float, goal_jpg: bytes,
                     goal_xz: np.ndarray, geodesic_m: float,
                     episode_seed: int) -> dict:
    """Run frozen NavDP with the privileged exact relative point-goal.

    The image-goal input remains present.  Only the LingBot-estimated metric
    point is replaced, which cleanly measures the controller upper bound.
    """
    position = np.asarray(start_position, dtype=np.float64).copy()
    yaw = float(start_yaw)
    path_len = 0.0
    way_world = None
    plans = []
    memory_trace = []
    success_dist = float(args.success_dist)

    def result(reached: bool, steps: int, path_at_reach=None) -> dict:
        return {
            "reached": bool(reached),
            "path_len": float(path_len),
            "path_len_at_reach": path_at_reach,
            "step_at_reach": steps if reached else None,
            "steps": int(steps),
            "plans": plans,
            "memory_trace": memory_trace,
            "end_pos": position.copy(),
            "end_psi": float(yaw),
            "final_goal_dist_m": float(np.linalg.norm(
                position[[0, 2]] - goal_xz)),
        }

    if np.linalg.norm(position[[0, 2]] - goal_xz) < success_dist:
        return result(True, 0, 0.0)

    for step in range(args.max_steps):
        rgb, depth = base.render(
            sim, position + np.asarray([0.0, base.CAM_H, 0.0]), yaw)
        frame = base.jpg_bytes(rgb)
        memory = base.srv_memory(frame)
        memory_frame = memory.get("frame_idx")
        if memory_frame is not None:
            memory_trace.append({
                "frame_idx": int(memory_frame),
                "step": int(step),
                "x": float(position[0]),
                "z": float(position[2]),
                "yaw": float(yaw),
            })

        if step % args.exec_horizon == 0:
            local_goal = world_goal_to_local(
                goal_xz, position[[0, 2]], yaw)
            request_seed = (
                base.diffusion_plan_seed(
                    int(episode_seed), 2, len(plans))
                if args.deterministic_plan_seeds else None
            )
            nav_data = {"goal_data": json.dumps({
                "goal_x": [float(local_goal[0])],
                "goal_y": [float(local_goal[1])],
            })}
            if request_seed is not None:
                nav_data["diffusion_seed"] = str(request_seed)
            response = base.requests.post(
                f"{base.NOVEL_BASE}/navdp_step_ip_mixgoal",
                files={
                    "image": ("image.jpg", frame),
                    "image_goal": ("goal.jpg", goal_jpg),
                    "depth": ("depth.png", base.depth_png_bytes(depth)),
                },
                data=nav_data,
            )
            response.raise_for_status()
            response_json = response.json()
            if request_seed is not None and int(
                    response_json.get("diffusion_seed", -1)) != request_seed:
                raise RuntimeError(
                    "NavDP server did not echo oracle-point diffusion seed")
            plan = base.normalize_navdp_response(response_json)
            way, selector = base.select_plan_trajectory(
                plan, position, yaw, pathfinder, goal_xz)
            way_world = base.waypoints_to_world(
                way, position[[0, 2]], yaw)
            plans.append({
                "step": int(step),
                "current_x": float(position[0]),
                "current_z": float(position[2]),
                "current_yaw": float(yaw),
                "current_goal_dist_m": float(np.linalg.norm(
                    position[[0, 2]] - goal_xz)),
                "aux_pose": local_goal.tolist(),
                "pose_controller": "oracle_gt_image_point_mix",
                "router_active": None,
                "router_reason": "privileged_oracle_point",
                "diffusion_seed": plan.get("diffusion_seed"),
                "requested_diffusion_seed": request_seed,
                "memory_frame_idx": memory_frame,
                **selector,
            })

        if way_world is not None:
            position, yaw, distance = base.pursuit_step(
                position, yaw, way_world, pathfinder)
            path_len += distance
        if np.linalg.norm(position[[0, 2]] - goal_xz) < success_dist:
            return result(True, step + 1, float(path_len))
    return result(False, args.max_steps)


def route_stats(plans: list[dict]) -> dict:
    active = [
        bool(plan["router_active"])
        for plan in plans if plan.get("router_active") is not None
    ]
    ranks = [
        int(plan["router_selected_candidate_rank"])
        for plan in plans
        if plan.get("router_selected_candidate_rank") is not None
    ]
    return {
        "router_plans": len(active),
        "router_active_plans": sum(active),
        "router_active_episode": int(any(active)),
        "selected_candidate_rank_min": min(ranks, default=None),
        "selected_candidate_rank_max": max(ranks, default=None),
    }


def leg_spl(leg: dict, geodesic_m: float) -> float:
    path = (
        leg["path_len_at_reach"]
        if leg.get("path_len_at_reach") is not None
        else leg["path_len"]
    )
    return base.spl(leg["reached"], geodesic_m, path)


def main() -> None:
    mode = infer_mode(
        _conditional.conditional_c_mode,
        args.server_backend,
        args.router_verify_top_k,
    )
    require(args.server_backend in ("navdp", "hybrid_pose"),
            "conditional-C supports only navdp and hybrid_pose")
    require((mode == "native") == (args.server_backend == "navdp"),
            "native mode must use navdp; memory modes must use hybrid_pose")
    require(args.retrieval_override == "off",
            "conditional-C oracle modes are selected explicitly, not by override")
    require(args.terminal_uturn == "off", "conditional-C position SR disables U-turn")
    require(args.terminal_visual_refine == "off",
            "conditional-C position SR disables visual refinement")
    require(args.trajectory_selector == "server",
            "conditional-C formal protocol forbids privileged trajectory selection")
    if args.server_backend == "hybrid_pose":
        require(args.novel_port is not None, "hybrid_pose requires --novel_port")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "metric.csv").exists(),
            f"conditional-C output already contains metrics: {output}")
    episode_dirs = sorted(
        Path(path) for path in glob.glob(os.path.join(args.episode_root, "episode_*"))
        if os.path.isfile(os.path.join(path, "meta", "gen_meta.json"))
    )
    if args.episode_ids:
        wanted = {value.strip() for value in args.episode_ids.split(",") if value.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    require(bool(episode_dirs), "no conditional-C episodes selected")

    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    metrics = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            metadata = json.loads((episode_dir / "meta/gen_meta.json").read_text())
            require(int(metadata.get("n_legs", -1)) == 3,
                    "selected episode is not three-leg")
            require(len(metadata.get("goals", [])) == 2,
                    "three-leg goal metadata is invalid")
            goal_c = metadata["goals"][1]
            require(goal_c.get("kind") == "revisit", "Goal C must be Revisit")
            rows = pd.read_parquet(
                episode_dir / "data/chunk-000/episode_000000.parquet")
            require(len(rows) == int(metadata["n_frames"]),
                    "parquet frame count mismatch")
            last_prefix = prefix_last_frame(
                metadata["switches"], int(metadata["n_frames"]))
            rgb_root = episode_dir / "videos/chunk-000/observation.images.rgb"
            intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
            camera_intrinsic = np.stack([
                np.asarray(row, dtype=np.float64) for row in intrinsic_raw
            ])
            start_position, start_yaw = base.parquet_pose_hab(
                rows.iloc[last_prefix]["action"])
            goal_hab = base.data_to_hab(goal_c["pos"])
            goal_xz = goal_hab[[0, 2]]
            goal_floor = np.asarray(
                [goal_xz[0], start_position[1], goal_xz[1]], dtype=float)
            ok, geodesic_c, _ = base.geodesic(
                pathfinder, start_position, goal_floor)
            require(ok and np.isfinite(geodesic_c),
                    "conditional-C geodesic is invalid")
            goal_jpg = read_bytes(episode_dir / "goal_2.jpg")

            episode_seed = args.seed + episode_index
            base.srv_reset(
                camera_height=float(metadata.get("camera_height_m", base.CAM_H)),
                seed=episode_seed,
                episode_len=int(metadata["n_frames"]),
                camera_intrinsic=camera_intrinsic,
            )
            prefix_trace = []
            navdp_prefix_steps = []
            prefix_trace, navdp_prefix_steps = replay_prefix(
                rgb_root, rows, last_prefix)

            if mode == "native":
                leg = base.run_policy_leg(
                    sim, pathfinder, start_position, start_yaw,
                    goal_jpg, goal_xz, geodesic_c,
                    terminal_mode="off",
                    goal_yaw=float(goal_c["yaw_habitat"]),
                    camera_intrinsic=camera_intrinsic,
                    policy_backend=None,
                    episode_seed=episode_seed, leg_index=2,
                )
            elif mode in ("geometry_top1", "geometry_topk"):
                leg = base.run_policy_leg(
                    sim, pathfinder, start_position, start_yaw,
                    goal_jpg, goal_xz, geodesic_c,
                    terminal_mode="off",
                    goal_yaw=float(goal_c["yaw_habitat"]),
                    camera_intrinsic=camera_intrinsic,
                    policy_backend="navdp_auto",
                    episode_seed=episode_seed, leg_index=2,
                )
            elif mode == "oracle_anchor":
                leg = base.run_policy_leg(
                    sim, pathfinder, start_position, start_yaw,
                    goal_jpg, goal_xz, geodesic_c,
                    terminal_mode="off",
                    goal_yaw=float(goal_c["yaw_habitat"]),
                    camera_intrinsic=camera_intrinsic,
                    forced_anchor=int(goal_c["covis_argmax"]),
                    policy_backend="navdp_mix",
                    episode_seed=episode_seed, leg_index=2,
                )
            else:
                leg = oracle_point_leg(
                    sim, pathfinder, start_position, start_yaw,
                    goal_jpg, goal_xz, float(geodesic_c), episode_seed)

            route = route_stats(leg["plans"])
            metric = {
                "episode": episode_dir.name,
                "seed": episode_seed,
                "mode": mode,
                "server_backend": args.server_backend,
                "deterministic_plan_seeds": bool(
                    args.deterministic_plan_seeds),
                "retrieval_candidate_min_gap": (
                    base.MEMNAV_SERVER_INFO.get(
                        "retrieval_candidate_min_gap")
                    if args.server_backend == "hybrid_pose" else None),
                "graph_subgoal_spacing_m": (
                    base.MEMNAV_SERVER_INFO.get("graph_subgoal_spacing_m")
                    if args.server_backend == "hybrid_pose" else 0.0),
                "graph_subgoal_arrival_m": (
                    base.MEMNAV_SERVER_INFO.get("graph_subgoal_arrival_m")
                    if args.server_backend == "hybrid_pose" else None),
                "reached_C": int(bool(leg["reached"])),
                "spl_C": leg_spl(leg, float(geodesic_c)),
                "geo_C": float(geodesic_c),
                "len_C": float(leg["path_len"]),
                "steps_C": int(leg["steps"]),
                "final_dist_C": float(leg["final_goal_dist_m"]),
                "prefix_last_source_frame": int(last_prefix),
                "prefix_source_frames": int(last_prefix + 1),
                "memory_prefix_frames": (
                    int(last_prefix + 1)
                    if args.server_backend == "hybrid_pose" else 0),
                "navdp_prefix_decision_frames": len(navdp_prefix_steps),
                "c_recall_gap": int(goal_c["recall_gap"]),
                "c_gt_covis_anchor": int(goal_c["covis_argmax"]),
                **route,
            }
            metrics.append(metric)
            (output / f"{episode_dir.name}_plans.json").write_text(json.dumps({
                "protocol": "conditional_C_after_causal_source_AB_replay",
                "mode": mode,
                "prefix_trace": prefix_trace,
                "navdp_prefix_steps": navdp_prefix_steps,
                "legC": leg["plans"],
            }, indent=2))
            with (output / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            print(
                f"[{episode_dir.name}] mode={mode} C={metric['reached_C']} "
                f"final={metric['final_dist_C']:.3f}m")

        successes = sum(row["reached_C"] for row in metrics)
        summary = {
            "protocol": "conditional_C_after_causal_source_AB_replay",
            "diagnostic_not_end_to_end_sr": True,
            "mode": mode,
            "episodes": len(metrics),
            "successes": successes,
            "conditional_C_SR": successes / len(metrics),
            "mean_SPL_C": float(np.mean([row["spl_C"] for row in metrics])),
            "mean_final_dist_C_m": float(np.mean([
                row["final_dist_C"] for row in metrics])),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("[conditional-C] done", summary)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
