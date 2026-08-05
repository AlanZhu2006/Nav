#!/usr/bin/env python3
"""Audit whether a closed-loop online memory really contains an ImageGoal.

The generated two-leg metadata labels a goal against the *scripted* leg-A
trajectory.  A closed-loop NavDP rollout follows a different trajectory, so a
metadata Revisit is not automatically a Revisit of the memory that the policy
actually collected.  This diagnostic reconstructs depth at every recorded
online memory pose and measures directional goal-surface co-visibility.

It also reports a geometry-only memory-graph upper bound: follow the recorded
trajectory backwards to the best co-visible node, then use a local geodesic
connector to the goal.  This is not navigation SR and is deliberately kept
separate from the learned policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# Habitat(Y-up) -> stored data(Z-up), duplicated here so pure CPU helpers and
# their tests do not import Habitat/quaternion.  Rendering geometry is imported
# lazily inside main() under the pinned habitat environment.
M_W = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def data_to_hab(position: Iterable[float]) -> np.ndarray:
    return M_W.T @ np.asarray(position, dtype=np.float64)


def image_mae(rendered: np.ndarray, path: Path) -> float:
    stored = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    rendered = np.asarray(rendered, dtype=np.float32)
    if stored.shape != rendered.shape:
        raise ValueError(
            f"render/image shape mismatch {rendered.shape} != {stored.shape}: {path}")
    return float(np.mean(np.abs(rendered - stored)))


def snap_floor_position(pathfinder, x: float, z: float,
                        reference_y: float) -> np.ndarray:
    """Recover the omitted online trace height from the same-floor navmesh."""
    snapped = np.asarray(
        pathfinder.snap_point([float(x), float(reference_y), float(z)]),
        dtype=np.float64,
    )
    if (snapped.shape != (3,) or not np.isfinite(snapped).all()
            or np.linalg.norm(snapped[[0, 2]] - [x, z]) > 0.08):
        raise RuntimeError(
            f"cannot recover online floor height at ({x:.3f}, {z:.3f})")
    return snapped


def trace_path_length(trace: list[dict], first: int, last: int) -> float:
    """Ground-plane length along trace indices, inclusive at both ends."""
    if not 0 <= first <= last < len(trace):
        raise ValueError("invalid trace interval")
    if first == last:
        return 0.0
    points = np.asarray(
        [[trace[index]["x"], trace[index]["z"]]
         for index in range(first, last + 1)],
        dtype=np.float64,
    )
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def adaptive_keyframes(trace: list[dict], distance_m: float,
                       yaw_deg: float) -> list[int]:
    """Deterministic motion-aware keyframes; always retain both endpoints."""
    if distance_m <= 0.0 or yaw_deg <= 0.0:
        raise ValueError("adaptive keyframe thresholds must be positive")
    if not trace:
        return []
    selected = [0]
    yaw_threshold = np.deg2rad(yaw_deg)
    for index in range(1, len(trace) - 1):
        previous = trace[selected[-1]]
        current = trace[index]
        distance = np.linalg.norm([
            float(current["x"]) - float(previous["x"]),
            float(current["z"]) - float(previous["z"]),
        ])
        yaw_delta = abs(wrap_angle(
            float(current["yaw"]) - float(previous["yaw"])))
        if distance >= distance_m or yaw_delta >= yaw_threshold:
            selected.append(index)
    if selected[-1] != len(trace) - 1:
        selected.append(len(trace) - 1)
    return selected


def uniform_keyframes(count: int, cap: int) -> list[int]:
    if count < 0 or cap < 1:
        raise ValueError("invalid keyframe count/cap")
    if count <= cap:
        return list(range(count))
    return sorted({
        int(round(value))
        for value in np.linspace(0, count - 1, num=cap)
    })


def positive_retained(candidate_rows: list[dict], selected_trace_indices: list[int],
                      threshold: float) -> bool:
    selected = set(selected_trace_indices)
    return any(
        int(row["trace_index"]) in selected
        and float(row["teacher_covis"]) >= threshold
        for row in candidate_rows
    )


def goal_index(episode_root: Path) -> dict[str, tuple[str, str, Path]]:
    index: dict[str, tuple[str, str, Path]] = {}
    for goal in sorted(episode_root.glob("*/episode_*/goal_1.jpg")):
        scene, episode = goal.parents[1].name, goal.parent.name
        digest = sha256(goal)
        if digest in index:
            raise RuntimeError(f"duplicate goal image hash: {goal} and {index[digest][2]}")
        index[digest] = (scene, episode, goal)
    if not index:
        raise RuntimeError(f"no goal_1.jpg files below {episode_root}")
    return index


def match_online_buffers(buffer_root: Path, episode_root: Path) -> list[dict]:
    goals = goal_index(episode_root)
    matches = []
    for buffer_dir in sorted(buffer_root.glob("ep_*")):
        goal = buffer_dir / "_goal.jpg"
        if not goal.is_file():
            continue
        matched = goals.get(sha256(goal))
        if matched is None:
            continue
        scene, episode, goal_path = matched
        matches.append({
            "scene": scene,
            "episode": episode,
            "goal_path": goal_path,
            "buffer_dir": buffer_dir,
        })
    identities = [(item["scene"], item["episode"]) for item in matches]
    if len(identities) != len(set(identities)):
        raise RuntimeError("multiple online buffers matched one episode")
    return matches


def summarize_episode(candidate_rows: list[dict], trace: list[dict], *,
                      positive_threshold: float, uniform_cap: int,
                      adaptive_distance_m: float,
                      adaptive_yaw_deg: float) -> dict:
    if not candidate_rows:
        raise ValueError("episode has no candidate rows")
    best = max(
        candidate_rows,
        key=lambda row: (float(row["teacher_covis"]),
                         -int(row["candidate_frame"])),
    )
    positives = [
        row for row in candidate_rows
        if float(row["teacher_covis"]) >= positive_threshold
    ]
    uniform = uniform_keyframes(len(trace), uniform_cap)
    adaptive = adaptive_keyframes(
        trace, adaptive_distance_m, adaptive_yaw_deg)
    return {
        "memory_frames": len(trace),
        "evaluated_frames": len(candidate_rows),
        "positive_frames": len(positives),
        "online_memory_has_positive": bool(positives),
        "max_covisibility": float(best["teacher_covis"]),
        "best_frame": int(best["candidate_frame"]),
        "best_trace_index": int(best["trace_index"]),
        "best_node_goal_distance_m": float(best["goal_distance_m"]),
        "best_node_goal_yaw_error_deg": float(best["goal_yaw_error_deg"]),
        "uniform_keyframes": len(uniform),
        "uniform_retains_positive": positive_retained(
            candidate_rows, uniform, positive_threshold),
        "adaptive_keyframes": len(adaptive),
        "adaptive_retains_positive": positive_retained(
            candidate_rows, adaptive, positive_threshold),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--pose-stride", type=int, default=1)
    parser.add_argument("--depth-stride", type=int, default=6)
    parser.add_argument("--depth-tolerance", type=float, default=0.3)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--uniform-cap", type=int, default=320)
    parser.add_argument("--adaptive-distance-m", type=float, default=0.50)
    parser.add_argument("--adaptive-yaw-deg", type=float, default=20.0)
    parser.add_argument("--max-render-mae", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from MemNavData import generate_twoleg as geometry
    except ModuleNotFoundError:  # direct script invocation
        import generate_twoleg as geometry  # type: ignore
    for path in (args.episode_root, args.asset_root, args.results_root,
                 args.buffer_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.report.exists() or args.candidate_csv.exists():
        raise FileExistsError("report/candidate output already exists")
    if (args.pose_stride < 1 or args.depth_stride < 1
            or args.max_episodes < 0 or args.uniform_cap < 1):
        raise ValueError("stride/count/cap arguments are invalid")
    if (not 0.0 < args.positive_threshold <= 1.0
            or args.depth_tolerance <= 0.0
            or args.max_render_mae <= 0.0):
        raise ValueError("threshold/tolerance arguments are invalid")

    matches = match_online_buffers(args.buffer_root, args.episode_root)
    selected_scenes = set(args.scene)
    if selected_scenes:
        matches = [item for item in matches if item["scene"] in selected_scenes]
    matches.sort(key=lambda item: (item["scene"], item["episode"]))
    if args.max_episodes:
        matches = matches[:args.max_episodes]
    if not matches:
        raise RuntimeError("no online buffer matched a generated Goal B")

    all_rows: list[dict] = []
    episodes = []
    current_scene = None
    sim = None
    try:
        for episode_number, item in enumerate(matches, 1):
            scene, episode = item["scene"], item["episode"]
            episode_dir = args.episode_root / scene / episode
            meta_path = episode_dir / "meta" / "gen_meta.json"
            plans_path = args.results_root / scene / f"{episode}_plans.json"
            asset = args.asset_root / f"{scene}.glb"
            for required in (meta_path, plans_path, asset):
                if not required.is_file():
                    raise FileNotFoundError(required)

            if scene != current_scene:
                if sim is not None:
                    sim.close()
                sim = geometry.make_sim(str(asset), None)
                current_scene = scene

            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            with open(plans_path, encoding="utf-8") as handle:
                plans = json.load(handle)
            trace = plans.get("legA_memory_trace", [])
            if not trace:
                raise RuntimeError(f"empty online leg-A memory trace: {scene}/{episode}")
            frame_ids = [int(row["frame_idx"]) for row in trace]
            if len(frame_ids) != len(set(frame_ids)) or frame_ids != sorted(frame_ids):
                raise RuntimeError(f"invalid memory frame order: {scene}/{episode}")

            goal = meta["goals"][0]
            floor_goal = data_to_hab(goal["pos"])
            camera_height = float(meta.get("camera_height_m", 0.5))
            floor_y = float(floor_goal[1])
            goal_camera = floor_goal + np.asarray([0.0, camera_height, 0.0])
            goal_yaw = float(goal["yaw_habitat"])
            goal_rgb, goal_depth = geometry.render(
                sim, goal_camera, goal_yaw)
            goal_render_mae = image_mae(goal_rgb, item["goal_path"])
            if goal_render_mae > args.max_render_mae:
                raise RuntimeError(
                    f"goal render mismatch {goal_render_mae:.3f}: {scene}/{episode}")
            goal_pose = geometry.cam_to_world_hab(goal_camera, goal_yaw)
            goal_points = geometry.to_world(
                geometry.backproject(goal_depth, stride=args.depth_stride),
                goal_pose)
            if not len(goal_points):
                raise RuntimeError(f"goal depth has no valid points: {scene}/{episode}")

            sampled_indices = list(range(0, len(trace), args.pose_stride))
            if sampled_indices[-1] != len(trace) - 1:
                sampled_indices.append(len(trace) - 1)
            rows = []
            for trace_index in sampled_indices:
                point = trace[trace_index]
                frame = int(point["frame_idx"])
                image_path = item["buffer_dir"] / f"{frame}.jpg"
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                candidate_floor = snap_floor_position(
                    sim.pathfinder, float(point["x"]), float(point["z"]),
                    floor_y)
                camera = candidate_floor + np.asarray(
                    [0.0, camera_height, 0.0])
                yaw = float(point["yaw"])
                rendered_rgb, depth = geometry.render(sim, camera, yaw)
                render_mae = image_mae(rendered_rgb, image_path)
                if render_mae > args.max_render_mae:
                    raise RuntimeError(
                        f"online render mismatch {render_mae:.3f}: "
                        f"{scene}/{episode}/frame{frame}")
                pose = geometry.cam_to_world_hab(camera, yaw)
                covis = geometry.covis_frac(
                    goal_points, pose, depth, tol=args.depth_tolerance)
                row = {
                    "session_id": f"{scene}/{episode}/online_goal_b",
                    "scene": scene,
                    "episode": episode,
                    "buffer_episode": item["buffer_dir"].name,
                    "goal_path": str(item["goal_path"].resolve()),
                    "candidate_path": str(image_path.resolve()),
                    "candidate_frame": frame,
                    "trace_index": trace_index,
                    "teacher_covis": float(covis),
                    "teacher_positive": int(covis >= args.positive_threshold),
                    "goal_distance_m": float(np.linalg.norm(
                        camera[[0, 2]] - goal_camera[[0, 2]])),
                    "goal_yaw_error_deg": float(np.degrees(abs(
                        wrap_angle(yaw - goal_yaw)))),
                    "render_rgb_mae": render_mae,
                }
                rows.append(row)
                all_rows.append(row)

            summary = summarize_episode(
                rows, trace,
                positive_threshold=args.positive_threshold,
                uniform_cap=args.uniform_cap,
                adaptive_distance_m=args.adaptive_distance_m,
                adaptive_yaw_deg=args.adaptive_yaw_deg,
            )
            row_by_frame = {
                int(row["candidate_frame"]): row for row in rows
            }
            leg_b_plans = plans.get("legB", [])
            first_retrieval = next((
                plan for plan in leg_b_plans
                if plan.get("retrieved_anchor") is not None
            ), None)
            first_retrieved_frame = (
                int(first_retrieval["retrieved_anchor"])
                if first_retrieval is not None else None)
            first_retrieved_row = row_by_frame.get(first_retrieved_frame)
            first_active = next((
                plan for plan in leg_b_plans
                if plan.get("router_active") and plan.get("anchor") is not None
            ), None)
            active_anchor_frame = (
                int(first_active["anchor"]) if first_active is not None else None)
            active_anchor_row = row_by_frame.get(active_anchor_frame)
            summary.update({
                "raw_dino_top1_frame": first_retrieved_frame,
                "raw_dino_top1_covisibility": (
                    float(first_retrieved_row["teacher_covis"])
                    if first_retrieved_row is not None else None),
                "raw_dino_top1_positive": (
                    bool(float(first_retrieved_row["teacher_covis"])
                         >= args.positive_threshold)
                    if first_retrieved_row is not None else None),
                "router_activated": first_active is not None,
                "active_anchor_frame": active_anchor_frame,
                "active_anchor_covisibility": (
                    float(active_anchor_row["teacher_covis"])
                    if active_anchor_row is not None else None),
                "active_anchor_positive": (
                    bool(float(active_anchor_row["teacher_covis"])
                         >= args.positive_threshold)
                    if active_anchor_row is not None else None),
            })
            best_index = int(summary["best_trace_index"])
            end = trace[-1]
            best = trace[best_index]
            end_floor = snap_floor_position(
                sim.pathfinder, float(end["x"]), float(end["z"]), floor_y)
            best_floor = snap_floor_position(
                sim.pathfinder, float(best["x"]), float(best["z"]), floor_y)
            goal_floor = np.asarray([goal_camera[0], floor_y, goal_camera[2]])
            ok_direct, direct_geo, _ = geometry.geodesic(
                sim.pathfinder, end_floor, goal_floor)
            ok_connector, connector_geo, _ = geometry.geodesic(
                sim.pathfinder, best_floor, goal_floor)
            reverse_length = trace_path_length(trace, best_index, len(trace) - 1)
            graph_length = (
                reverse_length + float(connector_geo)
                if ok_connector and np.isfinite(connector_geo) else None)
            summary.update({
                "scene": scene,
                "episode": episode,
                "buffer_episode": item["buffer_dir"].name,
                "goal_render_rgb_mae": goal_render_mae,
                "candidate_render_rgb_mae_median": float(np.median(
                    [row["render_rgb_mae"] for row in rows])),
                "candidate_render_rgb_mae_max": float(max(
                    row["render_rgb_mae"] for row in rows)),
                "direct_goal_geodesic_m": (
                    float(direct_geo) if ok_direct and np.isfinite(direct_geo)
                    else None),
                "reverse_memory_path_m": reverse_length,
                "best_node_goal_geodesic_m": (
                    float(connector_geo)
                    if ok_connector and np.isfinite(connector_geo) else None),
                "oracle_graph_route_m": graph_length,
                "oracle_graph_stretch": (
                    graph_length / float(direct_geo)
                    if graph_length is not None and ok_direct
                    and np.isfinite(direct_geo) and direct_geo > 1e-6 else None),
            })
            episodes.append(summary)
            print(json.dumps({
                "progress": f"{episode_number}/{len(matches)}",
                "scene": scene,
                "episode": episode,
                "online_memory_has_positive": summary[
                    "online_memory_has_positive"],
                "max_covisibility": summary["max_covisibility"],
                "best_frame": summary["best_frame"],
            }), flush=True)
    finally:
        if sim is not None:
            sim.close()

    fieldnames = list(all_rows[0])
    args.candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.candidate_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    positive_episodes = sum(
        int(item["online_memory_has_positive"]) for item in episodes)
    evaluated_top1 = [
        item for item in episodes if item["raw_dino_top1_positive"] is not None
    ]
    positive_top1 = sum(
        int(item["raw_dino_top1_positive"]) for item in evaluated_top1)
    activated = [item for item in episodes if item["router_activated"]]
    positive_active = sum(
        int(item["active_anchor_positive"])
        for item in activated if item["active_anchor_positive"] is not None)
    report = {
        "purpose": "closed-loop online-memory oracle coverage and graph feasibility",
        "deployment_approved": False,
        "not_navigation_sr": True,
        "inputs": {
            "episode_root": str(args.episode_root.resolve()),
            "asset_root": str(args.asset_root.resolve()),
            "results_root": str(args.results_root.resolve()),
            "buffer_root": str(args.buffer_root.resolve()),
        },
        "settings": {
            "pose_stride": args.pose_stride,
            "depth_stride": args.depth_stride,
            "depth_tolerance": args.depth_tolerance,
            "positive_threshold": args.positive_threshold,
            "uniform_cap": args.uniform_cap,
            "adaptive_distance_m": args.adaptive_distance_m,
            "adaptive_yaw_deg": args.adaptive_yaw_deg,
        },
        "matched_online_episodes": len(episodes),
        "episodes_with_online_positive": positive_episodes,
        "online_positive_rate": positive_episodes / len(episodes),
        "raw_dino_top1_evaluated": len(evaluated_top1),
        "raw_dino_top1_positive": positive_top1,
        "raw_dino_top1_recall_on_these_revisits": (
            positive_top1 / len(evaluated_top1) if evaluated_top1 else None),
        "router_activated_episodes": len(activated),
        "active_anchor_positive": positive_active,
        "candidate_rows": len(all_rows),
        "episodes": episodes,
        "candidate_csv": str(args.candidate_csv.resolve()),
        "candidate_csv_sha256": sha256(args.candidate_csv),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "report": str(args.report),
        "matched_online_episodes": len(episodes),
        "episodes_with_online_positive": positive_episodes,
        "online_positive_rate": report["online_positive_rate"],
        "raw_dino_top1_positive": positive_top1,
        "raw_dino_top1_evaluated": len(evaluated_top1),
        "router_activated_episodes": len(activated),
        "candidate_rows": len(all_rows),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
