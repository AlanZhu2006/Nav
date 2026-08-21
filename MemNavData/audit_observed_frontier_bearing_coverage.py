#!/usr/bin/env python3
"""Measure whether observed-space frontiers contain the Novel-A bearing.

This is an architecture diagnostic, not a navigation result.  It re-renders
the RGB-D observations at the frozen, paired Novel-A rollout poses, builds the
same goal-blind ``ObservedFrontierGrid`` used by the evaluator, and compares
the *reachable frontier headings* with Habitat's privileged geodesic bearing.

The two decision units are kept separate:

* ``candidate_oracle``: the smallest angular error among all reachable
  frontier candidates (does the provider expose a useful direction?);
* ``fixed_top1``: the angular error of the existing goal-blind utility ranker
  (does the current selector choose it?).

Goal coordinates are used only after candidate generation, for diagnostic
labels.  This consumed development cohort must not select a checkpoint or be
reported as a deployable closed-loop gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from MemNavData.observed_frontier import ObservedFrontierGrid


M_W = np.array([[1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0]], dtype=np.float64)
THRESHOLDS_DEG = (15, 30, 45, 60)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def data_to_hab(point: Sequence[float]) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("stored point must be a finite 3-vector")
    return M_W.T @ value


def matrix_from_nested(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.stack([
        np.asarray(row, dtype=np.float64) for row in value
    ])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera intrinsic must be a finite 3x3 matrix")
    return matrix


def parquet_floor_pose(row_action: Sequence[Sequence[float]],
                       camera_height_m: float) -> tuple[np.ndarray, float]:
    transform = np.stack([
        np.asarray(row, dtype=np.float64) for row in row_action
    ]).reshape(4, 4)
    camera_hab = M_W.T @ transform[:3, 3]
    rotation_hab = M_W.T @ transform[:3, :3]
    yaw = float(np.arctan2(rotation_hab[0, 2], rotation_hab[2, 2]))
    floor = camera_hab - np.asarray([0.0, camera_height_m, 0.0])
    return floor, yaw


def circular_error_deg(first_rad: float, second_rad: float) -> float:
    """Smallest unsigned separation between two angles."""

    delta = math.atan2(
        math.sin(float(first_rad) - float(second_rad)),
        math.cos(float(first_rad) - float(second_rad)),
    )
    return abs(math.degrees(delta))


def path_initial_bearing(path_points: Iterable[Sequence[float]],
                         start: Sequence[float],
                         *, min_offset_m: float = 0.30) -> float | None:
    """Habitat yaw of the first non-degenerate horizontal path segment."""

    origin = np.asarray(start, dtype=np.float64)
    points = [np.asarray(point, dtype=np.float64) for point in path_points]
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("path start must be a finite 3-vector")
    valid = [point for point in points
             if point.shape == (3,) and np.isfinite(point).all()]
    if not valid:
        return None
    selected = None
    for point in valid:
        if np.linalg.norm(point[[0, 2]] - origin[[0, 2]]) >= min_offset_m:
            selected = point
            break
    if selected is None:
        selected = valid[-1]
    delta = selected[[0, 2]] - origin[[0, 2]]
    if np.linalg.norm(delta) < 1e-6:
        return None
    # Habitat camera forward at yaw psi is (-sin(psi), -cos(psi)).
    return float(np.arctan2(-delta[0], -delta[1]))


def heading_resultant(headings_rad: Sequence[float]) -> float | None:
    if not headings_rad:
        return None
    headings = np.asarray(headings_rad, dtype=np.float64)
    return float(np.hypot(np.cos(headings).mean(), np.sin(headings).mean()))


def summarize_state_rows(rows: list[dict], thresholds=THRESHOLDS_DEG) -> dict:
    valid = [row for row in rows if row["oracle_bearing_deg"] is not None]
    output: dict[str, object] = {
        "states": len(rows),
        "oracle_bearing_valid_states": len(valid),
        "states_with_reachable_candidates": sum(
            row["reachable_candidate_count"] > 0 for row in valid),
        "mean_reachable_candidates": (
            float(np.mean([row["reachable_candidate_count"] for row in valid]))
            if valid else None),
        "mean_heading_resultant": (
            float(np.mean([row["candidate_heading_resultant"] for row in valid
                           if row["candidate_heading_resultant"] is not None]))
            if any(row["candidate_heading_resultant"] is not None
                   for row in valid) else None),
    }
    for threshold in thresholds:
        oracle_hits = [
            row["candidate_oracle_error_deg"] is not None
            and row["candidate_oracle_error_deg"] <= threshold
            for row in valid
        ]
        top1_hits = [
            row["fixed_top1_error_deg"] is not None
            and row["fixed_top1_error_deg"] <= threshold
            for row in valid
        ]
        current_hits = [
            row["current_heading_error_deg"] is not None
            and row["current_heading_error_deg"] <= threshold
            for row in valid
        ]
        output[f"candidate_oracle_within_{threshold}_count"] = int(sum(oracle_hits))
        output[f"candidate_oracle_within_{threshold}_rate"] = (
            float(np.mean(oracle_hits)) if oracle_hits else None)
        output[f"fixed_top1_within_{threshold}_count"] = int(sum(top1_hits))
        output[f"fixed_top1_within_{threshold}_rate"] = (
            float(np.mean(top1_hits)) if top1_hits else None)
        output[f"current_heading_within_{threshold}_count"] = int(
            sum(current_hits))
        output[f"current_heading_within_{threshold}_rate"] = (
            float(np.mean(current_hits)) if current_hits else None)
        output[f"fixed_top1_vs_current_within_{threshold}_gains"] = int(sum(
            top1 and not current
            for top1, current in zip(top1_hits, current_hits)))
        output[f"fixed_top1_vs_current_within_{threshold}_losses"] = int(sum(
            current and not top1
            for top1, current in zip(top1_hits, current_hits)))
    return output


def summarize_episode_rows(rows: list[dict], thresholds=THRESHOLDS_DEG) -> dict:
    output: dict[str, object] = {"episodes": len(rows)}
    for threshold in thresholds:
        oracle_any = [bool(row[f"candidate_oracle_any_within_{threshold}"])
                      for row in rows]
        top1_any = [bool(row[f"fixed_top1_any_within_{threshold}"])
                    for row in rows]
        oracle_fraction = [
            float(row[f"candidate_oracle_within_{threshold}_fraction"])
            for row in rows if row["states_with_valid_oracle"] > 0
        ]
        top1_fraction = [
            float(row[f"fixed_top1_within_{threshold}_fraction"])
            for row in rows if row["states_with_valid_oracle"] > 0
        ]
        output[f"candidate_oracle_any_within_{threshold}_episodes"] = int(
            sum(oracle_any))
        output[f"candidate_oracle_any_within_{threshold}_rate"] = (
            float(np.mean(oracle_any)) if oracle_any else None)
        output[f"candidate_oracle_mean_state_coverage_within_{threshold}"] = (
            float(np.mean(oracle_fraction)) if oracle_fraction else None)
        output[f"fixed_top1_any_within_{threshold}_episodes"] = int(sum(top1_any))
        output[f"fixed_top1_any_within_{threshold}_rate"] = (
            float(np.mean(top1_any)) if top1_any else None)
        output[f"fixed_top1_mean_state_coverage_within_{threshold}"] = (
            float(np.mean(top1_fraction)) if top1_fraction else None)
    return output


def episode_record(scene: str, episode: str, native_reached_a: bool,
                   rows: list[dict], thresholds=THRESHOLDS_DEG) -> dict:
    valid = [row for row in rows if row["oracle_bearing_deg"] is not None]
    record: dict[str, object] = {
        "scene": scene,
        "episode": episode,
        "native_reached_A": native_reached_a,
        "states": len(rows),
        "states_with_valid_oracle": len(valid),
        "states_with_reachable_candidates": sum(
            row["reachable_candidate_count"] > 0 for row in valid),
    }
    for threshold in thresholds:
        oracle_hit_indices = [
            index for index, row in enumerate(valid)
            if row["candidate_oracle_error_deg"] is not None
            and row["candidate_oracle_error_deg"] <= threshold
        ]
        top1_hit_indices = [
            index for index, row in enumerate(valid)
            if row["fixed_top1_error_deg"] is not None
            and row["fixed_top1_error_deg"] <= threshold
        ]
        current_hit_indices = [
            index for index, row in enumerate(valid)
            if row["current_heading_error_deg"] is not None
            and row["current_heading_error_deg"] <= threshold
        ]
        denominator = len(valid)
        record[f"candidate_oracle_any_within_{threshold}"] = bool(
            oracle_hit_indices)
        record[f"candidate_oracle_within_{threshold}_states"] = len(
            oracle_hit_indices)
        record[f"candidate_oracle_within_{threshold}_fraction"] = (
            len(oracle_hit_indices) / denominator if denominator else 0.0)
        record[f"candidate_oracle_first_within_{threshold}_plan"] = (
            int(valid[oracle_hit_indices[0]]["plan_index"])
            if oracle_hit_indices else None)
        record[f"fixed_top1_any_within_{threshold}"] = bool(top1_hit_indices)
        record[f"fixed_top1_within_{threshold}_states"] = len(top1_hit_indices)
        record[f"fixed_top1_within_{threshold}_fraction"] = (
            len(top1_hit_indices) / denominator if denominator else 0.0)
        record[f"current_heading_any_within_{threshold}"] = bool(
            current_hit_indices)
        record[f"current_heading_within_{threshold}_states"] = len(
            current_hit_indices)
        record[f"current_heading_within_{threshold}_fraction"] = (
            len(current_hit_indices) / denominator if denominator else 0.0)
    return record


def summarize_early_windows(rows: list[dict], *, threshold: int = 30,
                            windows=(1, 4, 8, 16)) -> dict:
    """Coverage before a fixed number of policy decisions.

    Failed rollouts are much longer than successful ones.  Reporting only a
    whole-trajectory state average would therefore over-weight late, already
    stuck states.  These windows make the intervention timing explicit.
    """

    records: dict[str, dict] = {}
    episodes = sorted({(str(row["scene"]), str(row["episode"])) for row in rows})
    for window in windows:
        selected = [
            row for row in rows
            if int(row["plan_index"]) < window
            and row["oracle_bearing_deg"] is not None
        ]
        by_episode = {
            key: [row for row in selected
                  if (row["scene"], row["episode"]) == key]
            for key in episodes
        }

        def hit(row: dict, field: str) -> bool:
            value = row[field]
            return value is not None and float(value) <= threshold

        oracle_hits = [hit(row, "candidate_oracle_error_deg")
                       for row in selected]
        top1_hits = [hit(row, "fixed_top1_error_deg") for row in selected]
        current_hits = [hit(row, "current_heading_error_deg")
                        for row in selected]
        records[f"first_{window}_plans"] = {
            "episodes": len(episodes),
            "states": len(selected),
            "candidate_oracle_state_rate": (
                float(np.mean(oracle_hits)) if oracle_hits else None),
            "fixed_top1_state_rate": (
                float(np.mean(top1_hits)) if top1_hits else None),
            "current_heading_state_rate": (
                float(np.mean(current_hits)) if current_hits else None),
            "candidate_oracle_episode_any_count": sum(
                any(hit(row, "candidate_oracle_error_deg") for row in episode_rows)
                for episode_rows in by_episode.values()),
            "fixed_top1_episode_any_count": sum(
                any(hit(row, "fixed_top1_error_deg") for row in episode_rows)
                for episode_rows in by_episode.values()),
            "current_heading_episode_any_count": sum(
                any(hit(row, "current_heading_error_deg") for row in episode_rows)
                for episode_rows in by_episode.values()),
        }
    return records


def cluster_bootstrap_episode_mean(rows: list[dict], value_key: str, *,
                                   resamples: int, seed: int) -> dict | None:
    scenes = sorted({str(row["scene"]) for row in rows})
    if len(scenes) < 2 or resamples < 1:
        return None
    by_scene = {scene: [row for row in rows if row["scene"] == scene]
                for scene in scenes}
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.choice(scenes, len(scenes), replace=True)
        sample = [row for scene in selected for row in by_scene[str(scene)]]
        values[index] = float(np.mean([float(row[value_key]) for row in sample]))
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "scene_clusters": len(scenes),
        "resamples": resamples,
        "seed": seed,
        "lower_95": float(low),
        "median": float(median),
        "upper_95": float(high),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_native_metric(path: Path) -> dict[str, bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["episode"]: float(row["reached_A"]) > 0.5
            for row in csv.DictReader(handle)
        }


def _reachable_candidate_bearings(pathfinder, current: np.ndarray,
                                  frontier: ObservedFrontierGrid) -> list[dict]:
    from MemNavData.generate_twoleg import geodesic

    reachable = []
    for rank, candidate in enumerate(
            frontier.ranked_frontiers(current[[0, 2]])):
        raw = np.asarray(candidate.world_xz, dtype=np.float64)
        snapped = np.asarray(pathfinder.snap_point(
            [raw[0], current[1], raw[1]]), dtype=np.float64)
        if (not np.isfinite(snapped).all()
                or np.linalg.norm(snapped[[0, 2]] - raw) > 0.35
                or abs(float(snapped[1] - current[1])) > 0.80):
            continue
        ok, distance_m, path = geodesic(pathfinder, current, snapped)
        if not ok or not np.isfinite(distance_m) or distance_m <= 0.60:
            continue
        bearing = path_initial_bearing(path, current)
        if bearing is None:
            continue
        reachable.append({
            "rank": rank,
            "bearing_rad": bearing,
            "target_geodesic_m": distance_m,
            "candidate": candidate,
            "snapped": snapped,
        })
    return reachable


def audit_episode(*, sim, scene: str, episode: str, episode_dir: Path,
                  plans_path: Path, native_reached_a: bool,
                  frozen_goal: dict,
                  frozen_episode: dict) -> tuple[list[dict], dict]:
    import pandas as pd
    from MemNavData.generate_twoleg import geodesic, render

    meta_path = episode_dir / "meta" / "gen_meta.json"
    parquet_path = episode_dir / "data" / "chunk-000" / "episode_000000.parquet"
    goal_path = (episode_dir / "videos" / "chunk-000" /
                 "observation.images.rgb" / f"{frozen_goal['frame_index']}.jpg")
    for required in (meta_path, parquet_path, goal_path, plans_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    for label, path in (("metadata", meta_path), ("parquet", parquet_path)):
        expected = frozen_episode["files"][label]
        if (path.stat().st_size != int(expected["bytes"])
                or sha256_file(path) != expected["sha256"]):
            raise RuntimeError(
                f"frozen {label} mismatch: {scene}/{episode}")
    if goal_path.stat().st_size != int(frozen_goal["bytes"]):
        raise RuntimeError(f"frozen Goal-A byte mismatch: {scene}/{episode}")
    if sha256_file(goal_path) != frozen_goal["sha256"]:
        raise RuntimeError(f"frozen Goal-A hash mismatch: {scene}/{episode}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    rows = pd.read_parquet(parquet_path)
    camera_height = float(meta.get("camera_height_m", 0.5))
    intrinsic = matrix_from_nested(rows.iloc[0]["observation.camera_intrinsic"])
    initial_floor, _initial_yaw = parquet_floor_pose(
        rows.iloc[0]["action"], camera_height)
    goal = data_to_hab(meta["A"])
    if abs(float(goal[1] - initial_floor[1])) > 0.80:
        raise RuntimeError(f"Goal-A is on a different floor: {scene}/{episode}")
    goal[1] = initial_floor[1]

    trace_by_frame = {
        int(row["frame_idx"]): row for row in plans["legA_memory_trace"]
    }
    plan_frames = [int(row["frame_idx"]) for row in plans["legA"]]
    if len(trace_by_frame) != len(plans["legA_memory_trace"]):
        raise RuntimeError(f"duplicate rollout frame: {scene}/{episode}")
    missing = sorted(set(plan_frames) - set(trace_by_frame))
    if missing:
        raise RuntimeError(
            f"plan frames absent from rollout trace {scene}/{episode}: {missing}")

    frontier = ObservedFrontierGrid()
    state_rows: list[dict] = []
    for plan_index, frame_index in enumerate(plan_frames):
        trace = trace_by_frame[frame_index]
        current = np.asarray([
            float(trace["x"]), initial_floor[1], float(trace["z"])
        ], dtype=np.float64)
        _rgb, depth = render(
            sim, current + np.asarray([0.0, camera_height, 0.0]),
            float(trace["yaw"]))
        frontier.integrate_depth(depth, current, float(trace["yaw"]), intrinsic)

        ok, remaining_m, goal_path_points = geodesic(
            sim.pathfinder, current, goal)
        oracle_bearing = (
            path_initial_bearing(goal_path_points, current) if ok else None)
        candidates = _reachable_candidate_bearings(
            sim.pathfinder, current, frontier)
        errors = (
            [circular_error_deg(item["bearing_rad"], oracle_bearing)
             for item in candidates]
            if oracle_bearing is not None else [])
        best_index = int(np.argmin(errors)) if errors else None
        summary = frontier.summary()
        state_rows.append({
            "scene": scene,
            "episode": episode,
            "native_reached_A": native_reached_a,
            "plan_index": plan_index,
            "frame_index": frame_index,
            "current_x": float(current[0]),
            "current_y": float(current[1]),
            "current_z": float(current[2]),
            "current_yaw_deg": math.degrees(float(trace["yaw"])),
            "goal_remaining_geodesic_m": (
                float(remaining_m) if ok and np.isfinite(remaining_m) else None),
            "oracle_bearing_deg": (
                math.degrees(oracle_bearing)
                if oracle_bearing is not None else None),
            "current_heading_error_deg": (
                circular_error_deg(float(trace["yaw"]), oracle_bearing)
                if oracle_bearing is not None else None),
            "raw_frontier_count": len(frontier.ranked_frontiers(
                current[[0, 2]])),
            "reachable_candidate_count": len(candidates),
            "candidate_oracle_error_deg": (
                float(errors[best_index]) if best_index is not None else None),
            "candidate_oracle_rank": (
                int(candidates[best_index]["rank"])
                if best_index is not None else None),
            "fixed_top1_error_deg": (float(errors[0]) if errors else None),
            "candidate_heading_resultant": heading_resultant(
                [item["bearing_rad"] for item in candidates]),
            "free_cells": summary["free_cells"],
            "obstacle_cells": summary["obstacle_cells"],
            "frontier_cells": summary["frontier_cells"],
        })
    return state_rows, episode_record(
        scene, episode, native_reached_a, state_rows)


def run(args: argparse.Namespace) -> dict:
    from MemNavData.generate_twoleg import make_sim

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    overlay = json.loads(args.input_overlay.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(args.manifest)
    if overlay["parent_manifest_sha256"] != manifest_sha:
        raise RuntimeError("Goal-A overlay does not pin the supplied manifest")
    scenes = list(manifest["selection"]["selected_scenes"])
    if args.scene:
        unknown = sorted(set(args.scene) - set(scenes))
        if unknown:
            raise ValueError(f"scene not in frozen manifest: {unknown}")
        scenes = [scene for scene in scenes if scene in set(args.scene)]
    if args.scene_limit:
        scenes = scenes[:args.scene_limit]

    state_rows: list[dict] = []
    episode_rows: list[dict] = []
    for scene in scenes:
        scene_index = manifest["selection"]["selected_scenes"].index(scene)
        result_root = args.plans_root / f"{scene_index:02d}_{scene}"
        native = _read_native_metric(result_root / "navdp_native" / "metric.csv")
        glb = args.asset_root / scene / f"{scene}.glb"
        navmesh = args.asset_root / scene / f"{scene}.navmesh"
        if not glb.is_file() or not navmesh.is_file():
            raise FileNotFoundError(f"missing scene asset for {scene}")
        expected_asset = manifest["assets"][scene]
        if (glb.stat().st_size != int(expected_asset["bytes"])
                or sha256_file(glb) != expected_asset["sha256"]):
            raise RuntimeError(f"frozen GLB mismatch: {scene}")
        episode_root = (
            args.legacy_episode_root
            if scene in set(manifest["selection"]["anchor_scenes"])
            else args.expanded_episode_root
        )
        sim = make_sim(str(glb), str(navmesh))
        try:
            episodes = list(manifest["episodes"][scene])
            if args.episode_limit:
                episodes = episodes[:args.episode_limit]
            for episode_item in episodes:
                episode = episode_item["episode"]
                if episode not in native:
                    raise RuntimeError(f"missing native metric: {scene}/{episode}")
                rows, summary = audit_episode(
                    sim=sim,
                    scene=scene,
                    episode=episode,
                    episode_dir=episode_root / scene / episode,
                    plans_path=(result_root / "geometry_router" /
                                f"{episode}_plans.json"),
                    native_reached_a=native[episode],
                    frozen_goal=overlay["goal_a_images"][scene][episode],
                    frozen_episode=episode_item,
                )
                state_rows.extend(rows)
                episode_rows.append(summary)
                print(
                    f"[{scene}/{episode}] native_A={int(native[episode])} "
                    f"states={len(rows)} "
                    f"oracle@30={summary['candidate_oracle_within_30_fraction']:.3f} "
                    f"top1@30={summary['fixed_top1_within_30_fraction']:.3f}",
                    flush=True,
                )
        finally:
            sim.close()

    failed_states = [row for row in state_rows if not row["native_reached_A"]]
    failed_episodes = [row for row in episode_rows if not row["native_reached_A"]]
    report = {
        "scope": (
            "architecture diagnostic on consumed 20-scene development cohort; "
            "Habitat goal is used only to label goal-blind candidate headings; "
            "deployment_approved=false"
        ),
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "input_overlay": str(args.input_overlay.resolve()),
            "input_overlay_sha256": sha256_file(args.input_overlay),
            "plans_root": str(args.plans_root.resolve()),
            "asset_root": str(args.asset_root.resolve()),
            "legacy_episode_root": str(args.legacy_episode_root.resolve()),
            "expanded_episode_root": str(args.expanded_episode_root.resolve()),
        },
        "definitions": {
            "candidate_oracle": (
                "minimum heading error among reachable candidates; not deployable"),
            "fixed_top1": (
                "first reachable candidate under the existing goal-blind utility"),
            "angular_thresholds_deg": list(THRESHOLDS_DEG),
            "frontier_parameters": {
                "resolution_m": ObservedFrontierGrid().resolution_m,
                "obstacle_clearance_m": (
                    ObservedFrontierGrid().obstacle_clearance_m),
                "min_component_cells": (
                    ObservedFrontierGrid().min_component_cells),
                "min_novelty_m": ObservedFrontierGrid().min_novelty_m,
            },
        },
        "all": {
            "states": summarize_state_rows(state_rows),
            "episodes": summarize_episode_rows(episode_rows),
        },
        "native_A_failures": {
            "states": summarize_state_rows(failed_states),
            "episodes": summarize_episode_rows(failed_episodes),
            "early_window_at_30_deg": summarize_early_windows(failed_states),
            "candidate_oracle_state_coverage_30_scene_bootstrap": (
                cluster_bootstrap_episode_mean(
                    failed_episodes,
                    "candidate_oracle_within_30_fraction",
                    resamples=args.bootstrap_resamples,
                    seed=args.bootstrap_seed)),
            "fixed_top1_state_coverage_30_scene_bootstrap": (
                cluster_bootstrap_episode_mean(
                    failed_episodes,
                    "fixed_top1_within_30_fraction",
                    resamples=args.bootstrap_resamples,
                    seed=args.bootstrap_seed + 1)),
        },
        "counts": {
            "scenes": len({row["scene"] for row in episode_rows}),
            "episodes": len(episode_rows),
            "native_A_failure_episodes": len(failed_episodes),
            "states": len(state_rows),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "states.csv", state_rows)
    _write_csv(args.out / "episodes.csv", episode_rows)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("MemNavData/expanded_navdp_router_eval_20260805.json"))
    parser.add_argument(
        "--input-overlay", type=Path,
        default=Path("MemNavData/novel_a_bearing_inputs_20260808.json"))
    parser.add_argument(
        "--plans-root", type=Path,
        default=Path(".diagnostics/twentyscene_local_20260808"))
    parser.add_argument(
        "--asset-root", type=Path,
        default=Path("/home/asus/Research/datasets/mp3d_20scene/assets"))
    parser.add_argument(
        "--legacy-episode-root", type=Path,
        default=Path(
            "/home/asus/Research/Nav-axis-uturn/.diagnostics/"
            "unseen_scene_eval_20260803/episodes"))
    parser.add_argument(
        "--expanded-episode-root", type=Path,
        default=Path("/home/asus/Research/datasets/mp3d_20scene/episodes"))
    parser.add_argument(
        "--out", type=Path,
        default=Path(".diagnostics/observed_frontier_bearing_coverage_20260809"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()
    if args.scene_limit < 0 or args.episode_limit < 0:
        parser.error("limits must be non-negative")
    if args.bootstrap_resamples < 1:
        parser.error("bootstrap resamples must be positive")
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
