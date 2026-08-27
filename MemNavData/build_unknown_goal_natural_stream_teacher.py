#!/usr/bin/env python3
"""Build causal co-visibility labels for a natural NavDP planning stream.

The input trace is produced by ``eval_3leg_habitat.py`` with the memory router
forced to remain inactive.  Every server frame is first bound to the natural
rollout pose and to the exact RGB bytes stored by MemNav.  Habitat depth is
then rendered *offline* at those recorded poses.  Consequently this program
cannot change the controller, its observations, or its action sequence.

The emitted target is deliberately local: whether the deployable top-k
shortlist contains a goal-co-visible anchor.  It is not a privileged Novel vs
Revisit phase label and it makes no claim about positives outside the
shortlist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

try:
    from MemNavData.covisibility_teacher import (
        backproject_world,
        covisibility_label,
        projected_covisibility,
    )
except ImportError:  # pragma: no cover - direct script execution
    from covisibility_teacher import (  # type: ignore
        backproject_world,
        covisibility_label,
        projected_covisibility,
    )


LEGS = ("legA", "legB", "legC")
M_W = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
SCHEMA_VERSION = "unknown_goal_natural_stream_covisibility_teacher_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def matrix(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.stack([np.asarray(row, dtype=np.float64) for row in value])
    require(result.shape == shape and np.isfinite(result).all(),
            f"{label} is malformed")
    return result


def camera_to_world_habitat(position: np.ndarray, yaw: float) -> np.ndarray:
    """OpenGL-camera to Habitat-world transform for a Y-axis yaw."""

    position = np.asarray(position, dtype=np.float64)
    require(position.shape == (3,) and np.isfinite(position).all(),
            "camera position is malformed")
    require(math.isfinite(float(yaw)), "camera yaw is malformed")
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0],
         [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    transform[:3, 3] = position
    return transform


@dataclass(frozen=True)
class NaturalFrame:
    frame_idx: int
    leg: str
    step: int
    floor_position: np.ndarray
    yaw: float
    rgb_path: Path
    rgb_sha256: str

    def camera_position(self, camera_height_m: float) -> np.ndarray:
        return self.floor_position + np.asarray(
            [0.0, float(camera_height_m), 0.0], dtype=np.float64)


@dataclass(frozen=True)
class GoalView:
    leg: str
    rgb_path: Path
    camera_position: np.ndarray
    yaw: float
    pose_source: str


def _pose_close(left: object, right: object) -> bool:
    return finite_number(left) and finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def index_natural_frames(
    payload: Mapping[str, object],
    buffer_episode_root: Path,
    *,
    allow_censored_legs: bool = False,
) -> dict[int, NaturalFrame]:
    """Fail-closed binding from MemNav frame id to causal rollout pose/RGB."""

    rollout_traces = payload.get("rollout_traces")
    memory_traces = payload.get("memory_traces")
    require(isinstance(rollout_traces, Mapping), "rollout traces are absent")
    require(isinstance(memory_traces, Mapping), "memory traces are absent")
    require(buffer_episode_root.is_dir(), "RGB buffer episode root is absent")
    result: dict[int, NaturalFrame] = {}
    for leg in LEGS:
        rollout = rollout_traces.get(leg)
        memory = memory_traces.get(leg)
        require(isinstance(rollout, list)
                and (bool(rollout) or allow_censored_legs),
                f"{leg} rollout trace is absent")
        require(isinstance(memory, list)
                and (bool(memory) or allow_censored_legs),
                f"{leg} memory trace is absent")
        require(bool(rollout) == bool(memory),
                f"{leg} rollout/memory censoring disagrees")
        rollout_by_step: dict[int, Mapping[str, object]] = {}
        for item in rollout:
            require(isinstance(item, Mapping), f"{leg} rollout item is malformed")
            step = item.get("step")
            require(isinstance(step, int) and step not in rollout_by_step,
                    f"{leg} rollout step is duplicate/invalid")
            require(all(finite_number(item.get(key))
                        for key in ("x", "y", "z", "yaw")),
                    f"{leg} rollout pose is malformed")
            digest = item.get("jpg_sha256")
            require(isinstance(digest, str) and len(digest) == 64,
                    f"{leg} rollout RGB digest is malformed")
            rollout_by_step[step] = item
        for item in memory:
            require(isinstance(item, Mapping), f"{leg} memory item is malformed")
            frame_idx, step = item.get("frame_idx"), item.get("step")
            require(isinstance(frame_idx, int) and frame_idx >= 0,
                    f"{leg} memory frame is invalid")
            require(frame_idx not in result, "memory frame id is duplicated")
            require(isinstance(step, int) and step in rollout_by_step,
                    f"{leg} memory step has no rollout pose")
            pose = rollout_by_step[step]
            require(all(_pose_close(item.get(key), pose.get(key))
                        for key in ("x", "z", "yaw")),
                    f"{leg} memory/rollout pose mismatch at frame {frame_idx}")
            rgb_path = buffer_episode_root / f"{frame_idx}.jpg"
            require(rgb_path.is_file() and not rgb_path.is_symlink(),
                    f"buffer RGB is absent for frame {frame_idx}")
            rgb_sha = sha256_file(rgb_path)
            require(rgb_sha == pose["jpg_sha256"],
                    f"buffer/rollout RGB mismatch at frame {frame_idx}")
            result[frame_idx] = NaturalFrame(
                frame_idx=frame_idx,
                leg=leg,
                step=step,
                floor_position=np.asarray(
                    [pose["x"], pose["y"], pose["z"]], dtype=np.float64),
                yaw=float(pose["yaw"]),
                rgb_path=rgb_path.resolve(),
                rgb_sha256=rgb_sha,
            )
    require(bool(result), "natural frame index is empty")
    ordered = sorted(result)
    require(ordered == list(range(ordered[-1] + 1)),
            "natural memory frame ids are not a complete causal prefix")
    return result


def _action_goal_pose(action: object) -> tuple[np.ndarray, float]:
    transform_data = matrix(action, (4, 4), "goal-A action")
    camera_position = M_W.T @ transform_data[:3, 3]
    rotation_habitat = M_W.T @ transform_data[:3, :3]
    yaw = float(math.atan2(
        float(rotation_habitat[0, 2]), float(rotation_habitat[2, 2])))
    expected = camera_to_world_habitat(camera_position, yaw)[:3, :3]
    # Episode actions were serialized through float32 parquet tensors.  Keep
    # this much tighter than navigation precision while admitting the observed
    # ~2e-8 orthogonality/round-trip error from that serialization.
    require(np.allclose(rotation_habitat, expected, rtol=0.0, atol=1e-6),
            "goal-A action is not a planar Habitat camera pose")
    return camera_position, yaw


def goal_views(
    episode_root: Path,
    metadata: Mapping[str, object],
    parquet_rows,
    camera_height_m: float,
) -> dict[str, GoalView]:
    switches = metadata.get("switches")
    goals = metadata.get("goals")
    require(isinstance(switches, list) and len(switches) == 2,
            "3-leg switch metadata is malformed")
    require(isinstance(goals, list) and len(goals) == 2,
            "3-leg goal metadata is malformed")
    switch_a = int(switches[0])
    require(1 <= switch_a <= len(parquet_rows), "Goal-A frame is invalid")
    camera_a, yaw_a = _action_goal_pose(
        parquet_rows.iloc[switch_a - 1]["action"])
    rgb_root = (episode_root / "videos" / "chunk-000"
                / "observation.images.rgb")
    views = {
        "legA": GoalView(
            leg="legA",
            rgb_path=(rgb_root / f"{switch_a - 1}.jpg").resolve(),
            camera_position=camera_a,
            yaw=yaw_a,
            pose_source="expert_action_at_switch_a_minus_1",
        )
    }
    for leg, goal_index in (("legB", 0), ("legC", 1)):
        goal = goals[goal_index]
        require(isinstance(goal, Mapping), f"{leg} goal metadata is malformed")
        position_data = np.asarray(goal.get("pos"), dtype=np.float64)
        yaw = goal.get("yaw_habitat")
        require(position_data.shape == (3,) and np.isfinite(position_data).all(),
                f"{leg} goal position is malformed")
        require(finite_number(yaw), f"{leg} goal yaw is malformed")
        floor_habitat = M_W.T @ position_data
        views[leg] = GoalView(
            leg=leg,
            rgb_path=(episode_root / f"goal_{goal_index + 1}.jpg").resolve(),
            camera_position=(floor_habitat + np.asarray(
                [0.0, float(camera_height_m), 0.0], dtype=np.float64)),
            yaw=float(yaw),
            pose_source=f"goal_{goal_index + 1}_metadata_pose",
        )
    for view in views.values():
        require(view.rgb_path.is_file() and not view.rgb_path.is_symlink(),
                f"goal RGB is absent: {view.rgb_path}")
    return views


def shortlist_support_label(labels: list[int]) -> tuple[int, str]:
    """Return positive / strict-negative / ambiguous actionable support."""

    require(all(label in (-1, 0, 1) for label in labels),
            "candidate labels are invalid")
    if any(label == 1 for label in labels):
        return 1, "positive_candidate_in_shortlist"
    if not labels:
        return 0, "no_eligible_candidate"
    if all(label == 0 for label in labels):
        return 0, "all_shortlist_candidates_strict_negative"
    return -1, "shortlist_contains_only_negative_or_ambiguous_candidates"


class HabitatDepthRenderer:
    """Single-scene, depth-only Habitat renderer with no NavMesh use."""

    def __init__(self, scene: Path, intrinsic: np.ndarray,
                 height: int, width: int) -> None:
        import habitat_sim
        import magnum as mn

        require(scene.is_file() and not scene.is_symlink(), "scene GLB is absent")
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all(),
                "camera intrinsic is malformed")
        require(height >= 1 and width >= 1, "render resolution is invalid")
        hfov = math.degrees(
            2.0 * math.atan(float(intrinsic[0, 2]) / float(intrinsic[0, 0])))
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene.resolve())
        backend.enable_physics = False
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "depth"
        sensor.sensor_type = habitat_sim.SensorType.DEPTH
        sensor.resolution = [height, width]
        sensor.hfov = hfov
        sensor.position = mn.Vector3(0, 0, 0)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        self._habitat_sim = habitat_sim
        self._simulator = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [agent]))
        self.identity = {
            "kind": "offline_recorded_pose_habitat_depth",
            "habitat_sim_version": getattr(habitat_sim, "__version__", None),
            "scene_path": str(scene.resolve()),
            "scene_sha256": sha256_file(scene),
            "height": int(height),
            "width": int(width),
            "hfov_deg": float(hfov),
            "navmesh_used": False,
        }

    def render(self, camera_position: np.ndarray, yaw: float) -> np.ndarray:
        import quaternion

        state = self._habitat_sim.agent.AgentState()
        state.position = np.asarray(camera_position, dtype=np.float64)
        state.rotation = quaternion.from_rotation_vector(
            [0.0, float(yaw), 0.0])
        self._simulator.get_agent(0).set_state(state)
        observations = self._simulator.get_sensor_observations()
        require("depth" in observations, "Habitat depth output is absent")
        depth = np.asarray(observations["depth"], dtype=np.float32)
        require(depth.ndim == 2 and np.isfinite(depth).all(),
                "rendered depth is malformed")
        return depth.copy()

    def close(self) -> None:
        self._simulator.close()


def build_teacher(
    *,
    plan_trace: Path,
    episode_root: Path,
    scene: Path,
    buffer_episode_root: Path,
    stride: int = 6,
    tolerance_m: float = 0.3,
    positive_threshold: float = 0.5,
    negative_threshold: float = 0.1,
    allow_censored_legs: bool = False,
) -> dict[str, object]:
    import pandas as pd

    require(plan_trace.is_file() and not plan_trace.is_symlink(),
            "plan trace is absent")
    require(episode_root.is_dir() and not episode_root.is_symlink(),
            "episode root is absent")
    require(stride >= 1 and tolerance_m > 0.0,
            "geometry configuration is invalid")
    require(0.0 <= negative_threshold < positive_threshold <= 1.0,
            "teacher thresholds are invalid")
    payload = json.loads(plan_trace.read_text(encoding="utf-8"))
    require(isinstance(payload, Mapping), "plan trace root is malformed")
    metadata_path = episode_root / "meta" / "gen_meta.json"
    parquet_path = (episode_root / "data" / "chunk-000"
                    / "episode_000000.parquet")
    require(metadata_path.is_file() and parquet_path.is_file(),
            "episode metadata/parquet is absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(isinstance(metadata, Mapping)
            and int(metadata.get("n_legs", -1)) == 3,
            "episode is not a true 3-leg episode")
    rows = pd.read_parquet(parquet_path)
    require(len(rows) == int(metadata["n_frames"]),
            "episode parquet length changed")
    intrinsic = matrix(
        rows.iloc[0]["observation.camera_intrinsic"], (3, 3), "intrinsic")
    camera_height_m = float(metadata.get("camera_height_m", 0.5))
    require(math.isfinite(camera_height_m) and camera_height_m > 0.0,
            "camera height is invalid")
    frames = index_natural_frames(
        payload,
        buffer_episode_root,
        allow_censored_legs=allow_censored_legs,
    )
    goals = goal_views(episode_root, metadata, rows, camera_height_m)
    with Image.open(next(iter(frames.values())).rgb_path) as image:
        width, height = image.size
    require(width >= 1 and height >= 1, "RGB resolution is invalid")
    renderer = HabitatDepthRenderer(scene, intrinsic, height, width)
    depth_cache: dict[int, np.ndarray] = {}
    depth_hashes: dict[int, str] = {}
    goal_points: dict[str, np.ndarray] = {}
    goal_records: dict[str, dict[str, object]] = {}
    try:
        for leg, goal in goals.items():
            depth = renderer.render(goal.camera_position, goal.yaw)
            require(depth.shape == (height, width), "goal depth shape changed")
            points = backproject_world(
                depth.astype(np.float64), intrinsic,
                camera_to_world_habitat(goal.camera_position, goal.yaw),
                stride=stride,
            )
            require(points.ndim == 2 and points.shape[1:] == (3,)
                    and len(points) > 0, f"{leg} goal surface is empty")
            goal_points[leg] = points
            goal_records[leg] = {
                "rgb_path": str(goal.rgb_path),
                "rgb_sha256": sha256_file(goal.rgb_path),
                "camera_position_habitat": goal.camera_position.tolist(),
                "yaw_habitat_rad": float(goal.yaw),
                "pose_source": goal.pose_source,
                "depth_float32_sha256": sha256_bytes(
                    np.asarray(depth, dtype="<f4").tobytes(order="C")),
                "surface_point_count": int(len(points)),
            }

        def candidate_depth(frame_idx: int) -> np.ndarray:
            if frame_idx not in depth_cache:
                frame = frames[frame_idx]
                depth = renderer.render(
                    frame.camera_position(camera_height_m), frame.yaw)
                require(depth.shape == (height, width),
                        "candidate depth shape changed")
                depth_cache[frame_idx] = depth
                depth_hashes[frame_idx] = sha256_bytes(
                    np.asarray(depth, dtype="<f4").tobytes(order="C"))
            return depth_cache[frame_idx]

        records = []
        global_decision_index = 0
        for leg in LEGS:
            plans = payload.get(leg)
            require(isinstance(plans, list)
                    and (bool(plans) or allow_censored_legs),
                    f"{leg} plans are absent")
            for leg_plan_index, plan in enumerate(plans):
                require(isinstance(plan, Mapping), f"{leg} plan is malformed")
                require(plan.get("router_active") is False,
                        "teacher input contains an active router decision")
                require(plan.get("revisit_adapter_takeover") is False,
                        "teacher input contains an adapter takeover")
                current_frame = plan.get("frame_idx")
                step = plan.get("step")
                require(isinstance(current_frame, int)
                        and current_frame in frames,
                        f"{leg} plan current frame is absent")
                current = frames[current_frame]
                require(current.leg == leg and current.step == step,
                        f"{leg} plan pose binding changed")
                trials = plan.get("router_candidate_trials")
                pool = plan.get("router_candidate_pool_size")
                considered = plan.get("router_candidates_considered")
                require(isinstance(trials, list)
                        and isinstance(pool, int)
                        and isinstance(considered, int)
                        and len(trials) == pool == considered
                        and 0 <= pool <= 8,
                        f"{leg} plan shortlist contract changed")
                candidates = []
                for trial_index, trial in enumerate(trials):
                    require(isinstance(trial, Mapping),
                            f"{leg} candidate trial is malformed")
                    anchor = trial.get("anchor")
                    require(isinstance(anchor, int) and anchor in frames
                            and anchor < current_frame,
                            f"{leg} candidate is non-causal or absent")
                    candidate = frames[anchor]
                    depth = candidate_depth(anchor)
                    score = projected_covisibility(
                        goal_points[leg], depth, intrinsic,
                        camera_to_world_habitat(
                            candidate.camera_position(camera_height_m),
                            candidate.yaw),
                        tolerance=tolerance_m,
                    )
                    label = covisibility_label(
                        score,
                        positive_threshold=positive_threshold,
                        negative_threshold=negative_threshold,
                    )
                    candidates.append({
                        "trial_index": int(trial_index),
                        "rank": trial.get("rank"),
                        "dino_rank": trial.get("dino_rank"),
                        "anchor": int(anchor),
                        "anchor_leg": candidate.leg,
                        "anchor_step": int(candidate.step),
                        "dino_cosine": trial.get("score"),
                        "matches": trial.get("matches"),
                        "inliers": trial.get("inliers"),
                        "inlier_ratio": trial.get("inlier_ratio"),
                        "geometry_passed_under_collection_thresholds": (
                            trial.get("passed")),
                        "covisibility": float(score),
                        "label": int(label),
                        "candidate_rgb_sha256": candidate.rgb_sha256,
                        "candidate_depth_float32_sha256": depth_hashes[anchor],
                    })
                labels = [int(candidate["label"]) for candidate in candidates]
                support_label, support_reason = shortlist_support_label(labels)
                positive_ranks = [
                    int(candidate["rank"])
                    for candidate in candidates
                    if candidate["label"] == 1
                    and isinstance(candidate.get("rank"), int)
                ]
                records.append({
                    "decision_index": int(global_decision_index),
                    "leg": leg,
                    "leg_plan_index": int(leg_plan_index),
                    "step": int(step),
                    "current_frame": int(current_frame),
                    "current_rgb_sha256": current.rgb_sha256,
                    "candidate_pool_size": int(pool),
                    "topk_support_label": int(support_label),
                    "topk_support_reason": support_reason,
                    "best_covisibility": (
                        float(max(candidate["covisibility"]
                                  for candidate in candidates))
                        if candidates else 0.0),
                    "first_positive_rank": (
                        int(min(positive_ranks)) if positive_ranks else None),
                    "candidates": candidates,
                })
                global_decision_index += 1
    finally:
        renderer.close()

    label_counts = {"positive": 0, "negative": 0, "ambiguous": 0}
    candidate_label_counts = {"positive": 0, "negative": 0, "ambiguous": 0}
    by_leg = {}
    for leg in LEGS:
        subset = [record for record in records if record["leg"] == leg]
        leg_counts = {"positive": 0, "negative": 0, "ambiguous": 0}
        for record in subset:
            key = {1: "positive", 0: "negative", -1: "ambiguous"}[
                record["topk_support_label"]]
            label_counts[key] += 1
            leg_counts[key] += 1
            for candidate in record["candidates"]:
                candidate_key = {1: "positive", 0: "negative", -1: "ambiguous"}[
                    candidate["label"]]
                candidate_label_counts[candidate_key] += 1
        by_leg[leg] = {"plans": len(subset), "topk_labels": leg_counts}

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": (
            "offline causal teacher smoke; no policy intervention, SR, or "
            "deployable-method claim"
        ),
        "inputs": {
            "plan_trace": str(plan_trace.resolve()),
            "plan_trace_sha256": sha256_file(plan_trace),
            "episode_root": str(episode_root.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "parquet_sha256": sha256_file(parquet_path),
            "buffer_episode_root": str(buffer_episode_root.resolve()),
            "scene": str(scene.resolve()),
            "scene_sha256": sha256_file(scene),
        },
        "teacher": {
            "target": "usable_goal_anchor_exists_in_deployable_top8_shortlist",
            "forbidden_input": "A/B/C goal role",
            "label_source": "offline_recorded_pose_depth_reprojection",
            "backprojection_stride": int(stride),
            "depth_tolerance_m": float(tolerance_m),
            "positive_threshold": float(positive_threshold),
            "negative_threshold": float(negative_threshold),
            "camera_height_m": float(camera_height_m),
            "censored_downstream_legs_allowed": bool(allow_censored_legs),
            "intrinsic": intrinsic.tolist(),
            "resolution": [int(height), int(width)],
            "renderer": renderer.identity,
        },
        "goal_views": goal_records,
        "summary": {
            "natural_frames": len(frames),
            "plans": len(records),
            "candidate_trials": sum(len(record["candidates"])
                                    for record in records),
            "unique_rendered_candidate_frames": len(depth_cache),
            "topk_label_counts": label_counts,
            "candidate_label_counts": candidate_label_counts,
            "by_leg": by_leg,
        },
        "records": records,
    }
    result["content_sha256_without_self"] = sha256_bytes(
        canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-trace", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--buffer-episode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--tolerance-m", type=float, default=0.3)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.1)
    parser.add_argument(
        "--allow-censored-legs",
        action="store_true",
        help=("accept explicit empty downstream traces after an earlier leg "
              "failed; empty legs emit no training records"),
    )
    args = parser.parse_args()
    require(not args.out.exists(), f"output already exists: {args.out}")
    result = build_teacher(
        plan_trace=args.plan_trace,
        episode_root=args.episode_root,
        scene=args.scene,
        buffer_episode_root=args.buffer_episode_root,
        stride=args.stride,
        tolerance_m=args.tolerance_m,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        allow_censored_legs=args.allow_censored_legs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(result))
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
