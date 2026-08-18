#!/usr/bin/env python3
"""Renderer-free independent audit of final14 construction populations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from audit_shared_online_role_pairs import audit as audit_role_pairs
from final14_role_pair_contract import (
    POPULATION_SCHEMA,
    assigned_direction_stratum,
    direction_in_stratum,
    goal_yaw_bin,
    goal_yaw_radians,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def angle_error_degrees(first: float, second: float) -> float:
    delta = (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def query_by_role(episode: dict) -> dict[str, dict[str, Any]]:
    pairs = episode["pairs"]
    require(len(pairs) == 1, "final14 episode must contain exactly one pair")
    queries = pairs[0]["queries"]
    require(len(queries) == 2, "final14 instrumentation pair must have two queries")
    result = {str(query["analysis_role"]): query for query in queries}
    require(set(result) == {"novel", "revisit"}, "role set changed")
    return result


def audit(root: Path) -> dict[str, Any]:
    protocol_audits = {
        protocol: audit_role_pairs(root / protocol)
        for protocol in ("natural_direction", "hard_support")
    }
    manifests = {
        protocol: json.loads((root / protocol / "manifest.json").read_text())
        for protocol in ("natural_direction", "hard_support")
    }
    natural_rows = {
        (str(row["scene"]), str(row["episode"])): row
        for row in manifests["natural_direction"]["episodes"]
    }
    hard_rows = {
        (str(row["scene"]), str(row["episode"])): row
        for row in manifests["hard_support"]["episodes"]
    }
    require(bool(natural_rows), "natural/standard population is empty")
    require(bool(hard_rows), "hard-support population is empty")
    require(set(hard_rows).issubset(natural_rows), "hard population is not a subset")
    per_scene = Counter(scene for scene, _episode in natural_rows)
    require(max(per_scene.values()) <= 3, "per-scene cap exceeded")

    natural_novel_assets = {}
    standard_support = []
    hard_support = []
    strata = Counter()
    for identity, episode in natural_rows.items():
        queries = query_by_role(episode)
        novel = queries["novel"]
        revisit = queries["revisit"]
        scene_rank = int(episode["final14_scene_rank"])
        episode_rank = int(episode["final14_source_episode_rank"])
        expected_stratum = assigned_direction_stratum(scene_rank, episode_rank)
        relative = float(
            novel["initial_path_direction_relative_to_a_end_deg"]
        )
        require(
            novel["construction_support_band"] == "unsupported_novel",
            f"{identity}: Novel support label changed",
        )
        require(
            float(novel["max_online_a_covis"]) < 0.10,
            f"{identity}: Novel has historical support",
        )
        require(
            novel["assigned_direction_stratum"] == expected_stratum
            and direction_in_stratum(relative, expected_stratum),
            f"{identity}: Novel direction stratum changed",
        )
        require(
            int(novel["goal_world_yaw_bin"])
            == goal_yaw_bin(*identity),
            f"{identity}: identity-bound goal yaw bin changed",
        )
        require(
            angle_error_degrees(
                float(novel["yaw_rad"]), goal_yaw_radians(*identity)
            ) <= 1e-9,
            f"{identity}: goal yaw is not the identity-bound world bin",
        )
        require(
            float(novel["paired_revisit_separation_m"]) >= 1.0,
            f"{identity}: Novel/Revisit target separation is too small",
        )
        require(
            revisit["construction_support_band"] == "standard"
            and 0.55 <= float(revisit["max_online_a_covis"]) <= 0.90
            and int(revisit["argmax_gap_frames"]) <= 24
            and 0.20 <= float(revisit["translation_from_source_m"]) <= 0.80
            and 12.0 <= float(revisit["yaw_delta_from_source_deg"]) <= 45.0
            and float(revisit["pixel_mae_from_source"]) >= 5.0,
            f"{identity}: standard Revisit contract changed",
        )
        require(
            int(revisit["source_online_frame"]) >= 39
            and int(revisit["source_online_frame"])
            <= int(episode["online_a_steps"]) - 17,
            f"{identity}: standard source frame is outside causal margins",
        )
        standard_support.append(float(revisit["max_online_a_covis"]))
        strata[expected_stratum] += 1
        natural_novel_assets[identity] = (
            novel["goal_rgb_sha256"], novel["goal_depth_sha256"],
            novel["floor_position"], float(novel["yaw_rad"]),
        )

    for identity, episode in hard_rows.items():
        queries = query_by_role(episode)
        novel = queries["novel"]
        revisit = queries["revisit"]
        require(
            (
                novel["goal_rgb_sha256"], novel["goal_depth_sha256"],
                novel["floor_position"], float(novel["yaw_rad"]),
            ) == natural_novel_assets[identity],
            f"{identity}: hard instrumentation did not reuse the natural query",
        )
        require(
            revisit["construction_support_band"] == "hard"
            and 0.25 <= float(revisit["max_online_a_covis"]) < 0.55
            and int(revisit["argmax_gap_frames"]) <= 32
            and 0.30 <= float(revisit["translation_from_source_m"]) <= 1.00
            and 18.0 <= float(revisit["yaw_delta_from_source_deg"]) <= 60.0
            and float(revisit["pixel_mae_from_source"]) >= 5.0,
            f"{identity}: hard Revisit contract changed",
        )
        hard_support.append(float(revisit["max_online_a_covis"]))

    population = None
    population_path = root / "population_receipt.json"
    if population_path.is_file():
        population = json.loads(population_path.read_text())
        require(
            population.get("schema_version")
            == POPULATION_SCHEMA,
            "population receipt schema changed",
        )
        require(population["policy_outcomes_read"] is False, "query outcome leak")
        require(
            int(population["populations"]["natural_standard"]["histories"])
            == len(natural_rows),
            "natural population count changed",
        )
        require(
            int(population["populations"]["hard_support"]["histories"])
            == len(hard_rows),
            "hard population count changed",
        )

    return {
        "ok": True,
        "scope": "renderer-free final14 construction audit; no policy rollout",
        "natural_standard_histories": len(natural_rows),
        "natural_standard_scenes": len({scene for scene, _ in natural_rows}),
        "hard_support_histories": len(hard_rows),
        "hard_support_scenes": len({scene for scene, _ in hard_rows}),
        "per_scene_natural_standard": dict(sorted(per_scene.items())),
        "direction_strata": dict(sorted(strata.items())),
        "standard_support_range": [min(standard_support), max(standard_support)],
        "hard_support_range": [min(hard_support), max(hard_support)],
        "hard_subset_of_natural_standard": True,
        "hard_novel_instrumentation_reuses_natural_query": True,
        "runtime_role_visibility": "none",
        "policy_outcomes_read": False,
        "protocol_audits": protocol_audits,
        "population_receipt_present": population is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
