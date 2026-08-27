#!/usr/bin/env python3
"""Fail closed unless a strict smoke actually exercises graph control."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


NOVEL_FIELDS = (
    "reached_A", "spl_A", "geo_A", "len_A", "final_dist_A", "steps_A",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def singleton(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    require(len(paths) == 1, (
        f"expected one {pattern} below {root}, found {len(paths)}"))
    return paths[0]


def read_metric(root: Path) -> dict[str, str]:
    path = singleton(root, "scenes/*/geometry_router/metric.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"strict smoke needs one metric row: {path}")
    return rows[0]


def read_plans(root: Path) -> dict:
    path = singleton(
        root, "scenes/*/geometry_router/episode_*_plans.json")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"plans must be an object: {path}")
    for leg in ("legA", "legB"):
        require(isinstance(value.get(leg), list), f"missing {leg}: {path}")
    return value


def checked_seed_sequence(rows: list[dict], label: str) -> list[int]:
    seeds: list[int] = []
    for index, row in enumerate(rows):
        requested = row.get("requested_diffusion_seed")
        echoed = row.get("diffusion_seed")
        require(isinstance(requested, int) and isinstance(echoed, int),
                f"{label}[{index}] lacks an integer diffusion seed echo")
        require(requested == echoed,
                f"{label}[{index}] seed request/echo mismatch")
        seeds.append(requested)
    return seeds


def pose_distance(left: object, right: object) -> float:
    require(isinstance(left, list) and isinstance(right, list),
            "active plans must contain aux_pose lists")
    require(len(left) == len(right) and len(left) >= 2,
            "active aux_pose shapes do not match")
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    require(all(math.isfinite(value) for value in delta),
            "active aux_pose contains a non-finite value")
    return math.sqrt(sum(value * value for value in delta))


def validate(source_root: Path, direct_root: Path,
             graph_root: Path) -> dict:
    metrics = {
        "source": read_metric(source_root),
        "direct": read_metric(direct_root),
        "graph": read_metric(graph_root),
    }
    episode = metrics["source"].get("episode")
    require(episode and all(row.get("episode") == episode
                            for row in metrics.values()),
            "source/direct/graph episode mismatch")
    trace_sha = metrics["source"].get("leg1_trace_sha256")
    require(trace_sha and len(trace_sha) == 64, "invalid shared trace SHA256")
    require(all(row.get("leg1_trace_sha256") == trace_sha
                for row in metrics.values()),
            "source/direct/graph do not share the same Novel trace")
    require(all(row.get("deterministic_plan_seeds") == "True"
                for row in metrics.values()),
            "deterministic plan seeding is not enabled in every arm")
    for field in NOVEL_FIELDS:
        require(metrics["direct"].get(field) == metrics["graph"].get(field),
                f"direct/graph Novel prefix differs at {field}")
    require(float(metrics["direct"]["graph_subgoal_spacing_m"]) == 0.0,
            "direct smoke unexpectedly enables graph subgoals")
    require(float(metrics["graph"]["graph_subgoal_spacing_m"]) > 0.0,
            "graph smoke does not enable graph subgoals")

    direct = read_plans(direct_root)
    graph = read_plans(graph_root)
    direct_a = checked_seed_sequence(direct["legA"], "direct.legA")
    graph_a = checked_seed_sequence(graph["legA"], "graph.legA")
    require(direct_a == graph_a, "direct/graph Novel plan seeds differ")
    direct_b = checked_seed_sequence(direct["legB"], "direct.legB")
    graph_b = checked_seed_sequence(graph["legB"], "graph.legB")
    common = min(len(direct_b), len(graph_b))
    require(common > 0 and direct_b[:common] == graph_b[:common],
            "direct/graph Revisit plan seeds differ on their common prefix")

    direct_active = sum(bool(row.get("router_active"))
                        for row in direct["legB"])
    graph_active = sum(bool(row.get("router_active"))
                       for row in graph["legB"])
    require(direct_active > 0 and graph_active > 0,
            "strict smoke never activated memory in both control arms")
    paired_pose_deltas = [
        pose_distance(left.get("aux_pose"), right.get("aux_pose"))
        for left, right in zip(direct["legB"], graph["legB"])
        if left.get("router_active") and right.get("router_active")
    ]
    require(paired_pose_deltas,
            "direct/graph have no paired active Revisit plan")
    max_pose_delta = max(paired_pose_deltas)
    require(max_pose_delta > 1e-6,
            "graph did not change any active point-goal")
    return {
        "status": "ok",
        "episode": episode,
        "shared_leg1_trace_sha256": trace_sha,
        "matched_novel_fields": list(NOVEL_FIELDS),
        "direct_active_revisit_plans": direct_active,
        "graph_active_revisit_plans": graph_active,
        "paired_active_plans": len(paired_pose_deltas),
        "max_direct_graph_aux_pose_delta_m": max_pose_delta,
        "common_revisit_seeded_plans": common,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_root, args.direct_root, args.graph_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
