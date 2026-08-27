#!/usr/bin/env python3
"""Build the pre-navigation parent manifest for fresh Full-Mono HM3D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_fresh_fullmono_parent_manifest_v1_20260820"
PROTOCOL_SCHEMA = "hm3d_fresh_fullmono_mixed_role_protocol_v1_20260820"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"non-finite {label}")
    return result


def checked_receipt(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing receipt {path}")
    receipt = path.with_name(path.name + ".sha256")
    require(receipt.is_file(), f"missing receipt hash {receipt}")
    require(sha256(path) == receipt.read_text().split()[0],
            f"receipt hash changed {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def episode_receipt(root: Path, scene: str, index: int, seed: int,
                    generation: dict[str, Any]) -> dict[str, Any]:
    episode = f"episode_{index:04d}"
    episode_root = root / episode
    paths = {
        "metadata": episode_root / "meta/gen_meta.json",
        "parquet": episode_root / "data/chunk-000/episode_000000.parquet",
        "goal": episode_root / "goal_image.jpg",
    }
    for kind, path in paths.items():
        require(path.is_file() and not path.is_symlink(),
                f"missing physical {kind}: {scene}/{episode}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    require(metadata.get("ep_idx") == index, f"episode index changed {scene}")
    require(metadata.get("generation_seed") == seed,
            f"generation seed changed {scene}/{episode}")
    require(metadata.get("scene") == f"{scene}.basis.glb",
            f"scene identity changed {scene}/{episode}")
    require(metadata.get("n_legs") == 2 and
            metadata.get("role_sequence") == ["initial_imagegoal", "revisit"],
            f"source role schema changed {scene}/{episode}")
    require(metadata.get("gen_protocol") == "multileg_v2_symmetric_20260807",
            f"generator protocol changed {scene}/{episode}")
    goals = metadata.get("goals")
    require(isinstance(goals, list) and len(goals) == 1 and
            goals[0].get("kind") == "revisit",
            f"source Revisit carrier missing {scene}/{episode}")
    covis = finite(goals[0].get("covis"), "covisibility")
    heading = abs(finite(goals[0].get("head_off_deg"), "heading"))
    covis_lo, covis_hi = generation["revisit_covisibility_band"]
    require(float(covis_lo) <= covis <= float(covis_hi),
            f"covisibility changed {scene}/{episode}")
    require(heading <= float(generation["goal_heading_max_deg"]) + 1e-9,
            f"heading changed {scene}/{episode}")
    require(int(metadata.get("n_frames", 0)) > 39 and
            int(metadata.get("anchor_margin", -1)) == 39,
            f"history length contract changed {scene}/{episode}")
    # This generated Revisit is only a legacy carrier that makes the source
    # compatible with the historical two-leg file schema.  The next stage
    # invokes eval_2leg_habitat with --stop_after_leg1 and constructs the
    # actual Novel/Revisit queries exclusively from the resulting online-A
    # trace.  Consequently its recall gap is useful provenance, but cannot be
    # an eligibility gate for Goal-A.  The constructed query population has
    # its own frozen online-history support/gap checks downstream.
    recall_gap = int(goals[0].get("recall_gap", -1))
    require(recall_gap >= 0,
            f"missing carrier recall gap {scene}/{episode}")
    minimum_recall_gap = int(generation["minimum_recall_gap_frames"])
    distance = finite(metadata.get("geo_startA"), "Goal-A geodesic")
    low, high = generation["distance_band_m"]
    require(float(low) <= distance <= float(high),
            f"Goal-A distance changed {scene}/{episode}")
    return {
        "episode": episode,
        "generation_seed": seed,
        "n_frames": int(metadata["n_frames"]),
        "goal_a_geodesic_m": distance,
        "covisibility": covis,
        "heading_offset_deg": heading,
        "recall_gap_frames": recall_gap,
        "carrier_recall_gap_meets_protocol_minimum": (
            recall_gap >= minimum_recall_gap),
        "carrier_used_by_downstream_query": False,
        "files": {
            kind: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                   "sha256": sha256(path)}
            for kind, path in paths.items()
        },
    }


def build(protocol_path: Path, asset_root: Path, asset_receipt_path: Path,
          selection_receipt_path: Path, generated_root: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "fresh protocol schema changed")
    dataset = protocol["dataset"]
    generation = protocol["source_generation"]
    scenes = dataset["scenes"]
    require(len(scenes) == dataset["fresh_scene_count"] == 54,
            "fresh scene population changed")
    require([int(row["rank"]) for row in scenes] == list(range(54)),
            "fresh scene ranks changed")

    asset_receipt = json.loads(asset_receipt_path.read_text(encoding="utf-8"))
    selection_receipt = json.loads(
        selection_receipt_path.read_text(encoding="utf-8"))
    require(asset_receipt.get("complete") is True and
            asset_receipt.get("scene_count") == 54,
            "asset receipt incomplete")
    require(asset_receipt.get("protocol_sha256") == sha256(protocol_path),
            "asset receipt protocol changed")
    require(asset_receipt.get("selection_receipt_sha256") ==
            sha256(selection_receipt_path), "selection receipt changed")
    require(selection_receipt.get("verified") is True and
            selection_receipt.get("fresh_scene_count") == 54 and
            selection_receipt.get("query_outcome_blind") is True,
            "selection audit invalid")
    require(set(asset_receipt["assets"]) ==
            {str(row["scene_id"]) for row in scenes},
            "asset identities changed")

    assets: dict[str, Any] = {}
    episodes: dict[str, list[dict[str, Any]]] = {}
    attrition = []
    complete_scene_indices = []
    count = int(generation["episodes_per_scene"])
    base_seed = int(generation["base_seed"])
    keys: set[tuple[str, str]] = set()
    carrier_gap_diagnostics: list[dict[str, Any]] = []
    for spec in scenes:
        rank = int(spec["rank"])
        scene = str(spec["scene_id"])
        directory = str(spec["directory"])
        directory_root = asset_root / directory
        glb = directory_root / f"{scene}.basis.glb"
        navmesh = directory_root / f"{scene}.basis.navmesh"
        for path in (glb, navmesh):
            require(path.is_file() and not path.is_symlink(),
                    f"missing physical asset {path}")
        asset_record = asset_receipt["assets"][scene]
        require(glb.stat().st_size == int(asset_record["glb_bytes"]) and
                navmesh.stat().st_size == int(asset_record["navmesh_bytes"]),
                f"asset bytes changed {scene}")
        assets[scene] = {
            "directory": directory,
            "glb_path": str(glb.resolve()),
            "glb_bytes": glb.stat().st_size,
            "glb_sha256": sha256(glb),
            "navmesh_path": str(navmesh.resolve()),
            "navmesh_bytes": navmesh.stat().st_size,
            "navmesh_sha256": sha256(navmesh),
        }

        root = generated_root / scene
        receipt = checked_receipt(root / "scene_generation_receipt.json")
        summary_path = root / "generation_summary.json"
        require(receipt["scene"] == scene and int(receipt["scene_rank"]) == rank,
                f"generation identity changed {scene}")
        require(receipt["generation_summary_sha256"] == sha256(summary_path),
                f"generation summary changed {scene}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("requested_episodes") == count and
                summary.get("n_legs") == 2 and
                summary.get("protocol") == "multileg_v2_symmetric_20260807",
                f"generation contract changed {scene}")
        generated = int(summary.get("generated_episodes", -1))
        if receipt["status"] != "complete":
            require(summary.get("complete") is False and generated < count and
                    receipt["partial_episodes_excluded"] is True,
                    f"invalid constructibility attrition {scene}")
            episodes[scene] = []
            attrition.append({
                "scene": scene, "scene_rank": rank,
                "reason": "fixed_attempt_source_generation_incomplete",
                "requested_episodes": count, "generated_episodes": generated,
                "generation_summary_sha256": sha256(summary_path),
                "scene_generation_receipt_sha256": sha256(
                    root / "scene_generation_receipt.json"),
                "retry_or_parameter_change": False,
            })
            continue
        require(summary.get("complete") is True and generated == count,
                f"complete generation count changed {scene}")
        seed = base_seed + rank
        scene_episodes = [
            episode_receipt(root, scene, index, seed, generation)
            for index in range(count)
        ]
        for row in scene_episodes:
            key = (scene, row["episode"])
            require(key not in keys, f"duplicate source episode {key}")
            keys.add(key)
            if not row["carrier_recall_gap_meets_protocol_minimum"]:
                carrier_gap_diagnostics.append({
                    "scene": scene,
                    "episode": row["episode"],
                    "recall_gap_frames": row["recall_gap_frames"],
                    "minimum_recall_gap_frames": int(
                        generation["minimum_recall_gap_frames"]),
                    "used_by_downstream_query": False,
                })
        episodes[scene] = scene_episodes
        complete_scene_indices.append(rank)

    return {
        "schema_version": SCHEMA,
        "scope": protocol["scope"],
        "query_outcomes_read": False,
        "fresh_scene_generalization": True,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "selection_receipt_path": str(selection_receipt_path.resolve()),
        "selection_receipt_sha256": sha256(selection_receipt_path),
        "asset_receipt_path": str(asset_receipt_path.resolve()),
        "asset_receipt_sha256": sha256(asset_receipt_path),
        "scenes": [str(row["scene_id"]) for row in scenes],
        "scene_specs": scenes,
        "scene_count": len(scenes),
        "episodes_per_scene": count,
        "target_episode_count": len(scenes) * count,
        "episode_count": len(keys),
        "constructible_scene_count": len(complete_scene_indices),
        "evaluation_scene_indices": complete_scene_indices,
        "assets": assets,
        "episodes": episodes,
        "paths": {"generated_root": str(generated_root.resolve()),
                  "asset_root": str(asset_root.resolve())},
        "source_generation": generation,
        "legacy_source_carrier_audit": {
            "used_by_downstream_query": False,
            "goal_a_collection_stops_after_leg1": True,
            "below_minimum_recall_gap_count": len(
                carrier_gap_diagnostics),
            "below_minimum_recall_gap": carrier_gap_diagnostics,
            "query_population_has_independent_online_support_checks": True,
        },
        "construction_attrition": {
            "scene_level": attrition,
            "scene_count": len(attrition),
            "partial_episodes_excluded": True,
            "retry_or_parameter_change": False,
            "navigation_or_query_outcomes_read": False,
        },
        "audit": {
            "status": "ok", "physical_assets": True,
            "physical_episode_files": True, "unique_episode_keys": True,
            "fresh_scene_overlap_with_prior_outcomes": 0,
            "query_outcomes_read": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.protocol.resolve(), args.asset_root.resolve(),
                    args.asset_receipt.resolve(),
                    args.selection_receipt.resolve(),
                    args.generated_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": "complete", "scenes": payload["scene_count"],
        "constructible_scenes": payload["constructible_scene_count"],
        "source_episodes": payload["episode_count"],
        "attrition_scenes": payload["construction_attrition"]["scene_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
