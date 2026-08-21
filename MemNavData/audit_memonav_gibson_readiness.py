#!/usr/bin/env python3
"""Audit whether the released MemoNav/Gibson assets define a runnable score."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SPLITS = (
    "val_4200",
    "multi_goal_val/1goal",
    "multi_goal_val/2goal",
    "multi_goal_val/3goal",
    "multi_goal_val/4goal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audit(repo: Path, asset_roots: list[Path]) -> dict[str, Any]:
    dataset_root = repo / "image-goal-nav-dataset/gibson"
    if not dataset_root.is_dir():
        raise RuntimeError(f"MemoNav Gibson dataset is absent: {dataset_root}")
    source_files = [path for path in repo.rglob("*.py")
                    if ".git" not in path.parts]
    split_reports: dict[str, Any] = {}
    scene_ids: set[str] = set()
    for split in SPLITS:
        files = sorted((dataset_root / split).glob("*.json.gz"))
        if len(files) != 14:
            raise RuntimeError(f"{split}: expected 14 scene files, got {len(files)}")
        episodes: list[dict[str, Any]] = []
        file_receipts = []
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise RuntimeError(f"{path}: top-level payload is not a list")
            episodes.extend(payload)
            file_receipts.append({
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "episodes": len(payload),
            })
        goal_count = rotation_count = exact_adjacent_repeats = 0
        adjacent_pairs = 0
        final_nearest_previous: list[float] = []
        for episode in episodes:
            scene = Path(str(episode["scene_id"])).stem
            scene_ids.add(scene)
            goals = episode.get("goals")
            if not isinstance(goals, list) or not goals:
                raise RuntimeError(f"{split}: episode has no goals")
            goal_count += len(goals)
            rotation_count += sum("rotation" in goal for goal in goals)
            for left, right in zip(goals, goals[1:]):
                adjacent_pairs += 1
                exact_adjacent_repeats += int(
                    left.get("position") == right.get("position"))
            if len(goals) > 1:
                final = goals[-1]["position"]
                final_nearest_previous.append(min(
                    math.dist(final, goal["position"])
                    for goal in goals[:-1]))
        split_reports[split] = {
            "scene_files": len(files),
            "episodes": len(episodes),
            "goals": goal_count,
            "goal_rotation_fields": rotation_count,
            "adjacent_goal_pairs": adjacent_pairs,
            "exact_adjacent_position_repeats": exact_adjacent_repeats,
            "final_to_nearest_previous_goal_euclidean_m": {
                "n": len(final_nearest_previous),
                "minimum": quantile(final_nearest_previous, 0.0),
                "median": quantile(final_nearest_previous, 0.5),
                "maximum": quantile(final_nearest_previous, 1.0),
                "fraction_le_1m": (
                    sum(value <= 1.0 for value in final_nearest_previous) /
                    len(final_nearest_previous)
                    if final_nearest_previous else None),
                "fraction_le_2m": (
                    sum(value <= 2.0 for value in final_nearest_previous) /
                    len(final_nearest_previous)
                    if final_nearest_previous else None),
            },
            "files": file_receipts,
        }

    found_assets: dict[str, list[str]] = {scene: [] for scene in scene_ids}
    for root in asset_roots:
        if not root.exists():
            continue
        for scene in sorted(scene_ids):
            for path in root.rglob(f"{scene}.glb"):
                if path.is_file():
                    found_assets[scene].append(str(path.resolve()))
    available = sorted(scene for scene, paths in found_assets.items() if paths)
    missing = sorted(scene for scene, paths in found_assets.items() if not paths)
    all_goal_rotations_absent = all(
        row["goal_rotation_fields"] == 0 for row in split_reports.values())
    return {
        "schema_version": "memonav_gibson_readiness_audit_v1_20260816",
        "repository": str(repo.resolve()),
        "readme_sha256": sha256_file(repo / "README.md"),
        "released_python_source_files": [
            str(path.relative_to(repo)) for path in sorted(source_files)],
        "released_training_or_evaluation_code": bool(source_files),
        "splits": split_reports,
        "scene_ids": sorted(scene_ids),
        "scene_count": len(scene_ids),
        "asset_roots_checked": [str(path.resolve()) for path in asset_roots],
        "available_scene_assets": available,
        "missing_scene_assets": missing,
        "all_goal_rotations_absent": all_goal_rotations_absent,
        "official_score_reproduction_ready": (
            bool(source_files) and not missing and not all_goal_rotations_absent),
        "cec_compatibility_eval_ready": (
            not missing and not all_goal_rotations_absent),
        "blockers": [
            "official repository does not release training/evaluation source"
            if not source_files else None,
            "required Gibson GLB assets are absent"
            if missing else None,
            "episode JSON contains goal positions but no goal rotations/images"
            if all_goal_rotations_absent else None,
            "default Habitat ImageGoalSensor addresses goals[0] and therefore "
            "does not define sequential goal switching"
            if all_goal_rotations_absent else None,
        ],
        "honest_next_state": (
            "not runnable as an official MemoNav score; after licensed scene "
            "assets and a preregistered goal-render/switching contract, it can "
            "be used only as a Gibson compatibility transfer evaluation"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args.repo.resolve(), [path.resolve()
                                         for path in args.asset_root])
    payload["blockers"] = [value for value in payload["blockers"] if value]
    if args.out is not None:
        out = args.out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
