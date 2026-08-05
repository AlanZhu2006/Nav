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


def summarize_ablation(
        manifest: dict, reference_root: Path,
        configurations: dict[str, Path]) -> dict:
    if not configurations:
        raise ValueError("at least one configuration is required")
    reference, expected = load_geometry_run(manifest, reference_root)
    runs = {"direct_gap16_reference": reference}
    for name, path in configurations.items():
        require(name not in runs, f"duplicate configuration name: {name}")
        rows, run_expected = load_geometry_run(manifest, path)
        require(run_expected == expected, f"manifest mismatch: {name}")
        runs[name] = rows

    comparisons = {}
    reference_name = "direct_gap16_reference"
    for name in configurations:
        comparisons[f"{reference_name}_vs_{name}"] = paired_summary(
            reference_name, name, reference, runs[name], expected)
    names = list(configurations)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            comparisons[f"{left}_vs_{right}"] = paired_summary(
                left, right, runs[left], runs[right], expected)

    return {
        "audit": {
            "status": "ok",
            "scenes": len(manifest["selection"]["selected_scenes"]),
            "episodes": len(expected),
            "policy_training_overlap": sorted(
                set(manifest["selection"]["selected_scenes"])
                & set(manifest["training_scenes"])),
            "interpretation": (
                "development ablation: candidate gap and graph control are "
                "paired independently; freeze before a new blind scene split"),
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
        manifest, args.reference_root, configurations),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
