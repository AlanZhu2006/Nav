#!/usr/bin/env python3
"""Summarize paired candidate-gap and reverse-memory-graph ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from MemNavData.summarize_expanded_navdp_router_eval import (
        arm_summary,
        load_arm,
        paired_summary,
        require,
    )
except ModuleNotFoundError:  # direct script invocation
    from summarize_expanded_navdp_router_eval import (  # type: ignore
        arm_summary,
        load_arm,
        paired_summary,
        require,
    )


def load_geometry_run(manifest: dict, root: Path) -> tuple[
        dict[tuple[str, str], dict], set[tuple[str, str]]]:
    scenes = manifest["selection"]["selected_scenes"]
    expected = {
        (scene, row["episode"])
        for scene in scenes for row in manifest["episodes"][scene]
    }
    rows: dict[tuple[str, str], dict] = {}
    for index, scene in enumerate(scenes):
        scene_root = root / "scenes" / f"{index:02d}_{scene}"
        rows.update(load_arm(scene_root, "geometry_router", scene))
    require(set(rows) == expected, f"result keys differ from manifest: {root}")
    return rows, expected


def load_complete_scene_intersection(
        manifest: dict, roots: dict[str, Path]) -> tuple[
            dict[str, dict[tuple[str, str], dict]],
            set[tuple[str, str]], list[dict]]:
    """Load only scenes with complete results in every paired arm.

    Missing scene metric files are treated as interrupted work. Once a metric
    file exists, however, all expected rows and plan traces remain mandatory so
    corrupt or half-written results cannot silently enter the report.
    """
    runs: dict[str, dict[tuple[str, str], dict]] = {
        name: {} for name in roots
    }
    included: set[tuple[str, str]] = set()
    excluded = []
    scenes = manifest["selection"]["selected_scenes"]
    for index, scene in enumerate(scenes):
        expected = {
            (scene, record["episode"])
            for record in manifest["episodes"][scene]
        }
        scene_roots = {
            name: root / "scenes" / f"{index:02d}_{scene}"
            for name, root in roots.items()
        }
        missing = [
            name for name, scene_root in scene_roots.items()
            if not (scene_root / "geometry_router" / "metric.csv").is_file()
        ]
        if missing:
            excluded.append({
                "index": index,
                "scene": scene,
                "missing_configurations": sorted(missing),
            })
            continue
        loaded = {
            name: load_arm(scene_root, "geometry_router", scene)
            for name, scene_root in scene_roots.items()
        }
        for name, rows in loaded.items():
            require(
                set(rows) == expected,
                f"incomplete existing scene result: {name} {scene}",
            )
            runs[name].update(rows)
        included.update(expected)
    require(bool(included), "no complete paired scenes found")
    return runs, included, excluded


def summarize_ablation(
        manifest: dict, reference_root: Path,
        configurations: dict[str, Path], *,
        complete_scene_intersection: bool = False) -> dict:
    if not configurations:
        raise ValueError("at least one configuration is required")
    reference_name = "direct_gap16_reference"
    require(reference_name not in configurations,
            f"duplicate configuration name: {reference_name}")
    excluded_scenes = []
    if complete_scene_intersection:
        roots = {reference_name: reference_root, **configurations}
        runs, expected, excluded_scenes = load_complete_scene_intersection(
            manifest, roots)
    else:
        reference, expected = load_geometry_run(manifest, reference_root)
        runs = {reference_name: reference}
        for name, path in configurations.items():
            require(name not in runs, f"duplicate configuration name: {name}")
            rows, run_expected = load_geometry_run(manifest, path)
            require(run_expected == expected, f"manifest mismatch: {name}")
            runs[name] = rows

    comparisons = {}
    for name in configurations:
        comparisons[f"{reference_name}_vs_{name}"] = paired_summary(
            reference_name, name, runs[reference_name], runs[name], expected)
    names = list(configurations)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            comparisons[f"{left}_vs_{right}"] = paired_summary(
                left, right, runs[left], runs[right], expected)

    return {
        "audit": {
            "status": (
                "partial_complete_scene_intersection"
                if complete_scene_intersection else "ok"
            ),
            "planned_scenes": len(manifest["selection"]["selected_scenes"]),
            "scenes": len({scene for scene, _ in expected}),
            "episodes": len(expected),
            "excluded_scenes": excluded_scenes,
            "not_a_full_manifest_result": bool(excluded_scenes),
            "policy_training_overlap": sorted(
                {scene for scene, _ in expected}
                & set(manifest["training_scenes"])),
            "interpretation": (
                "interrupted development ablation over the complete paired "
                "scene intersection; this is not the planned full-manifest "
                "result"
                if complete_scene_intersection else
                "development ablation: candidate gap and graph control are "
                "paired independently; freeze before a new blind scene split"
            ),
        },
        "configurations": {
            name: arm_summary([rows[key] for key in sorted(expected)])
            for name, rows in runs.items()
        },
        "pairwise": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--config", action="append", required=True,
        help="repeatable NAME=RUN_ROOT entry",
    )
    parser.add_argument(
        "--complete-scene-intersection", action="store_true",
        help=("summarize only scenes complete in every arm; intended for "
              "transparent analysis of interrupted runs"),
    )
    args = parser.parse_args()
    configurations = {}
    for item in args.config:
        if "=" not in item:
            raise ValueError(f"invalid --config {item!r}; expected NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or name in configurations:
            raise ValueError(f"invalid or duplicate configuration: {name!r}")
        configurations[name] = Path(raw_path)
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(summarize_ablation(
        manifest, args.reference_root, configurations,
        complete_scene_intersection=args.complete_scene_intersection),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
