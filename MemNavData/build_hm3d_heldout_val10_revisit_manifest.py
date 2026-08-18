#!/usr/bin/env python3
"""Build the immutable identity manifest for HM3D held-out val10 Revisit transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA_V1 = "hm3d_heldout_val10_causal_revisit_protocol_v1_20260816"
PROTOCOL_SCHEMA_V2 = "hm3d_heldout_val10_causal_revisit_protocol_v2_20260816"
MANIFEST_SCHEMA_V1 = "hm3d_heldout_val10_causal_revisit_manifest_v1_20260816"
MANIFEST_SCHEMA_V2 = "hm3d_heldout_val10_causal_revisit_manifest_v2_20260816"
DEPENDENCY_NAMES = (
    "gatecurr600",
    "navdp_checkpoint",
    "lingbot_map_long",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_number(value: Any) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite numeric value: {value!r}")
    return parsed


def validate_protocol(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    schema = protocol.get("schema_version")
    require(schema in {PROTOCOL_SCHEMA_V1, PROTOCOL_SCHEMA_V2},
            "unexpected protocol schema")
    dataset = protocol.get("dataset", {})
    require(dataset.get("name") == "HM3D", "dataset must be HM3D")
    require(dataset.get("release") == "v0.2", "HM3D release changed")
    require(dataset.get("split") == "heldout_val_asset_subset",
            "split must be the frozen held-out val-asset subset")
    require(dataset.get("archive_scene_count") == 100 and
            dataset.get("prior_consumed_scene_count") == 36 and
            dataset.get("unconsumed_scene_count_before_selection") == 64,
            "scene-selection population changed")
    require(dataset.get("scene_overlap_with_consumed_hm3d_evaluations") == 0,
            "selected scenes overlap prior HM3D outcomes")
    scenes = protocol.get("scenes")
    require(isinstance(scenes, list) and len(scenes) == 10,
            "protocol must contain ten held-out scenes")
    require([row.get("index") for row in scenes] == list(range(10)),
            "scene indices must be contiguous 0..9")
    scene_ids = [str(row.get("scene_id", "")) for row in scenes]
    directories = [str(row.get("directory", "")) for row in scenes]
    require(all(scene_ids) and len(set(scene_ids)) == 10,
            "scene IDs must be non-empty and unique")
    require(all(directories) and len(set(directories)) == 10,
            "scene directories must be non-empty and unique")
    require(all(directory.endswith(scene_id)
                for directory, scene_id in zip(directories, scene_ids)),
            "scene directory/ID mapping changed")
    generation = protocol.get("generation", {})
    require(generation.get("episodes_per_scene") == 4,
            "formal protocol requires four episodes per scene")
    require(generation.get("legs") == 2, "protocol must remain two-leg")
    guards = protocol.get("frozen_guards", {})
    for name in (
        "no_mp3d_evaluation",
        "no_scene_or_episode_filtering_after_outcomes",
        "no_threshold_tuning_on_hm3d_heldout_val10",
        "goal_a_failures_retained",
        "goal_b_not_executed_after_goal_a_failure",
        "certificate_thresholds_unchanged_from_mp3d_development",
        "selected_scenes_disjoint_from_all_prior_hm3d_outcomes",
    ):
        require(guards.get(name) is True, f"protocol guard is absent: {name}")
    if schema == PROTOCOL_SCHEMA_V2:
        attrition = protocol.get("construction_attrition", {})
        require(attrition.get("selected_scene_count") == 10 and
                attrition.get("target_episode_count") == 40,
                "construction target changed")
        require(attrition.get("constructible_scene_indices") ==
                [0, 1, 2, 3, 4, 5, 6, 7, 9],
                "constructible scene identities changed")
        require(attrition.get("constructible_scene_count") == 9 and
                attrition.get("constructible_episode_count") == 36,
                "constructible population changed")
        failed = attrition.get("failed_scenes")
        require(isinstance(failed, list) and len(failed) == 1 and
                failed[0].get("index") == 8 and
                failed[0].get("scene_id") == "q3hn1WQ12rz",
                "construction attrition identity changed")
        require(attrition.get("target_met") is False and
                attrition.get("underpowered") is True and
                attrition.get("scene_replacement") is False and
                attrition.get("generation_retry") is False and
                attrition.get("navigation_outcomes_generated") is False and
                attrition.get("navigation_outcomes_read") is False,
                "construction attrition is not outcome blind")
        for name in (
            "construction_attrition_frozen_before_navigation_evaluation",
            "failed_scene_retained_as_explicit_attrition",
            "original_scene_indices_and_arm_orders_preserved",
            "no_scene_replacement",
            "no_generation_retry_or_constraint_change",
        ):
            require(guards.get(name) is True,
                    f"attrition guard is absent: {name}")
    return scenes


def validate_parent_protocol(
    protocol: dict[str, Any], protocol_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    """Return the asset-binding protocol for V1 or its sealed V2 parent."""
    if protocol.get("schema_version") == PROTOCOL_SCHEMA_V1:
        return protocol_path, protocol, sha256_file(protocol_path)
    parent_record = protocol.get("parent_protocol", {})
    parent_path = protocol_path.with_name(str(parent_record.get("path", "")))
    require(parent_path.is_file() and not parent_path.is_symlink(),
            f"missing physical parent protocol: {parent_path}")
    parent_sha = sha256_file(parent_path)
    require(parent_sha == parent_record.get("sha256"),
            "parent protocol hash changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    require(parent.get("schema_version") == PROTOCOL_SCHEMA_V1,
            "attrition parent is not the frozen V1 protocol")
    require(parent.get("scenes") == protocol.get("scenes"),
            "attrition amendment changed selected scenes or indices")
    for field in ("dataset", "generation", "evaluation"):
        require(parent.get(field) == protocol.get(field),
                f"attrition amendment changed frozen {field}")
    return parent_path, parent, parent_sha


def validate_selection_audit(
    protocol: dict[str, Any], protocol_path: Path,
) -> tuple[Path, dict[str, Any]]:
    dataset = protocol["dataset"]
    audit_path = protocol_path.with_name(str(dataset["selection_audit"]))
    require(audit_path.is_file() and not audit_path.is_symlink(),
            f"missing physical scene-selection audit: {audit_path}")
    audit_sha = sha256_file(audit_path)
    require(audit_sha == dataset.get("selection_audit_sha256"),
            "scene-selection audit hash changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("schema_version") ==
            "hm3d_consumed_scene_audit_v1_20260816" and
            audit.get("status") == "ok",
            "scene-selection audit is not valid")
    require(audit.get("consumed_scene_count") == 36 and
            audit.get("unconsumed_scene_count") == 64 and
            audit.get("outcome_fields_read_for_selection") is False,
            "scene-selection audit population changed")
    require(audit.get("selected_overlap_with_consumed") == [],
            "scene-selection audit reports overlap")
    require(audit.get("selected_scenes") == protocol.get("scenes"),
            "protocol scenes differ from the audited deterministic selection")
    consumed = set(audit.get("consumed_scene_ids", []))
    selected = {row["scene_id"] for row in protocol["scenes"]}
    require(len(consumed) == 36 and not (consumed & selected),
            "selected/consumed scene sets are not disjoint")
    return audit_path, audit


def dependency_receipt(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"dependency must be a physical file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_manifest(
    *,
    protocol_path: Path,
    generated_root: Path,
    data_root: Path,
    asset_receipt: Path,
    base_source_receipt: Path,
    expected_base_source_receipt_sha: str,
    dependencies: dict[str, Path],
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scenes = validate_protocol(protocol)
    parent_protocol_path, _parent_protocol, asset_protocol_sha = (
        validate_parent_protocol(protocol, protocol_path))
    selection_audit_path, selection_audit = validate_selection_audit(
        protocol, protocol_path)
    require(sha256_file(base_source_receipt) ==
            expected_base_source_receipt_sha,
            "base source receipt changed")
    asset_payload = json.loads(asset_receipt.read_text(encoding="utf-8"))
    require(asset_payload.get("complete") is True and
            asset_payload.get("scene_count") == 10 and
            asset_payload.get("schema_version") ==
            "hm3d_heldout_val10_asset_receipt_v1_20260816",
            "HM3D held-out val10 asset receipt is incomplete")
    require(asset_payload.get("protocol_sha256") == asset_protocol_sha,
            "asset receipt references a different protocol")
    require(set(asset_payload.get("assets", {})) ==
            {str(row["scene_id"]) for row in scenes},
            "asset receipt scene identities differ from the protocol")

    episode_count = int(protocol["generation"]["episodes_per_scene"])
    base_seed = int(protocol["generation"]["base_seed"])
    assets: dict[str, Any] = {}
    episodes: dict[str, list[dict[str, Any]]] = {}
    scene_ids: list[str] = []
    all_episode_keys: set[tuple[str, str]] = set()
    attrition = protocol.get("construction_attrition", {})
    if protocol.get("schema_version") == PROTOCOL_SCHEMA_V2:
        frozen_generated_root = (
            Path(str(attrition["parent_run_root"])) / "data/hm3d_2leg")
        require(generated_root.resolve() == frozen_generated_root.resolve(),
                "V2 generated root differs from the preserved parent run")
    failed_by_index = {
        int(record["index"]): record
        for record in attrition.get("failed_scenes", [])
    }
    construction_receipts: list[dict[str, Any]] = []

    for row in scenes:
        index = int(row["index"])
        scene = str(row["scene_id"])
        directory = str(row["directory"])
        scene_ids.append(scene)
        asset_dir = (data_root / "data/scene_datasets/hm3d/heldout_val10" /
                     directory)
        glb = asset_dir / f"{scene}.basis.glb"
        navmesh = asset_dir / f"{scene}.basis.navmesh"
        for path in (glb, navmesh):
            require(path.is_file() and not path.is_symlink(),
                    f"missing physical HM3D asset: {path}")
        assets[scene] = {
            "directory": directory,
            "glb_path": str(glb.resolve()),
            "glb_bytes": glb.stat().st_size,
            "glb_sha256": sha256_file(glb),
            "navmesh_path": str(navmesh.resolve()),
            "navmesh_bytes": navmesh.stat().st_size,
            "navmesh_sha256": sha256_file(navmesh),
        }

        scene_root = generated_root / scene
        generation_summary = scene_root / "generation_summary.json"
        require(generation_summary.is_file(),
                f"missing generation summary for {scene}")
        summary = json.loads(generation_summary.read_text(encoding="utf-8"))
        if index in failed_by_index:
            failed = failed_by_index[index]
            require(sha256_file(generation_summary) ==
                    failed.get("summary_sha256"),
                    f"failed generation receipt changed for {scene}")
            for field in (
                "complete", "requested_episodes", "generated_episodes",
                "outer_attempt_budget", "outer_attempts_used",
            ):
                require(summary.get(field) == failed.get(field),
                        f"failed generation field changed: {scene}/{field}")
            require(summary.get("n_legs") == 2 and
                    summary.get("protocol") ==
                    "multileg_v2_symmetric_20260807",
                    f"failed generation contract changed for {scene}")
            require(not any(scene_root.glob("episode_*")),
                    f"failed scene contains unregistered episodes: {scene}")
            episodes[scene] = []
            construction_receipts.append({
                "scene": scene,
                "scene_index": index,
                "generation_summary_path": str(generation_summary.resolve()),
                "generation_summary_sha256": sha256_file(generation_summary),
                "summary": summary,
            })
            continue
        require(summary.get("complete") is True,
                f"generation is incomplete for {scene}")
        require(summary.get("n_legs") == 2 and
                summary.get("requested_episodes") == episode_count and
                summary.get("generated_episodes") == episode_count,
                f"generation count/leg contract changed for {scene}")
        require(summary.get("protocol") == "multileg_v2_symmetric_20260807",
                f"generator protocol changed for {scene}")

        scene_episodes: list[dict[str, Any]] = []
        for episode_index in range(episode_count):
            episode = f"episode_{episode_index:04d}"
            episode_root = scene_root / episode
            paths = {
                "metadata": episode_root / "meta/gen_meta.json",
                "parquet": episode_root /
                    "data/chunk-000/episode_000000.parquet",
                "goal": episode_root / "goal_image.jpg",
            }
            for kind, path in paths.items():
                require(path.is_file() and not path.is_symlink(),
                        f"missing physical {kind}: {path}")
            metadata = json.loads(paths["metadata"].read_text(
                encoding="utf-8"))
            require(metadata.get("ep_idx") == episode_index,
                    f"episode index mismatch: {scene}/{episode}")
            require(metadata.get("generation_seed") == base_seed + index,
                    f"generation seed mismatch: {scene}/{episode}")
            require(metadata.get("scene") == f"{scene}.basis.glb",
                    f"scene identity mismatch: {scene}/{episode}")
            require(metadata.get("n_legs") == 2 and
                    metadata.get("role_sequence") ==
                    ["initial_imagegoal", "revisit"],
                    f"role sequence mismatch: {scene}/{episode}")
            require(metadata.get("gen_protocol") ==
                    "multileg_v2_symmetric_20260807",
                    f"episode protocol mismatch: {scene}/{episode}")
            goals = metadata.get("goals")
            require(isinstance(goals, list) and len(goals) == 1 and
                    goals[0].get("kind") == "revisit",
                    f"missing Revisit-B goal: {scene}/{episode}")
            covis = finite_number(goals[0].get("covis"))
            heading = abs(finite_number(goals[0].get("head_off_deg")))
            require(0.20 <= covis <= 1.0,
                    f"covisibility outside frozen band: {scene}/{episode}")
            require(heading <= 45.0 + 1e-9,
                    f"heading outside frozen band: {scene}/{episode}")
            require(int(metadata.get("n_frames", 0)) > 39 and
                    int(metadata.get("anchor_margin", -1)) == 39,
                    f"history is too short: {scene}/{episode}")
            recall_gap = int(goals[0].get("recall_gap", -1))
            require(recall_gap >= int(
                protocol["generation"]["minimum_recall_gap_frames"]),
                f"recall gap is below contract: {scene}/{episode}")
            distance = finite_number(metadata.get("geo_startA"))
            distance_lo, distance_hi = protocol["generation"][
                "distance_band_m"]
            require(float(distance_lo) <= distance <= float(distance_hi),
                    f"Goal-A geodesic is outside contract: {scene}/{episode}")
            require(metadata.get("initial_distance_band_m") ==
                    protocol["generation"]["distance_band_m"],
                    f"generation distance band changed: {scene}/{episode}")
            require(metadata.get("covis_band") ==
                    protocol["generation"]["revisit_covisibility_band"],
                    f"generation covisibility band changed: {scene}/{episode}")
            require(metadata.get("window") == 32 and
                    metadata.get("num_scale") == 8,
                    f"history sampling scale changed: {scene}/{episode}")
            key = (scene, episode)
            require(key not in all_episode_keys, f"duplicate episode key {key}")
            all_episode_keys.add(key)
            scene_episodes.append({
                "episode": episode,
                "generation_seed": base_seed + index,
                "n_frames": int(metadata["n_frames"]),
                "goal_a_geodesic_m": distance,
                "covisibility": covis,
                "heading_offset_deg": heading,
                "recall_gap_frames": recall_gap,
                "files": {
                    kind: {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for kind, path in paths.items()
                },
            })
        episodes[scene] = scene_episodes

    dependency_records = {
        name: dependency_receipt(dependencies[name])
        for name in DEPENDENCY_NAMES
    }
    schema = protocol.get("schema_version")
    constructible_indices = [
        index for index, scene in enumerate(scene_ids) if episodes[scene]
    ]
    if schema == PROTOCOL_SCHEMA_V1:
        require(constructible_indices == list(range(10)) and
                len(all_episode_keys) == 40,
                "V1 manifest lost part of its frozen population")
    else:
        require(constructible_indices ==
                attrition["constructible_scene_indices"] and
                len(all_episode_keys) ==
                attrition["constructible_episode_count"],
                "V2 constructible population differs from amendment")
        require(len(construction_receipts) == len(failed_by_index),
                "V2 attrition receipts are incomplete")
    return {
        "schema_version": (
            MANIFEST_SCHEMA_V1 if schema == PROTOCOL_SCHEMA_V1
            else MANIFEST_SCHEMA_V2),
        "scope": protocol["scope"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "parent_protocol_path": str(parent_protocol_path.resolve()),
        "parent_protocol_sha256": asset_protocol_sha,
        "selection_audit_path": str(selection_audit_path.resolve()),
        "selection_audit_sha256": sha256_file(selection_audit_path),
        "prior_consumed_scene_count": selection_audit[
            "consumed_scene_count"],
        "selected_overlap_with_consumed": [],
        "asset_receipt_path": str(asset_receipt.resolve()),
        "asset_receipt_sha256": sha256_file(asset_receipt),
        "base_source_receipt_path": str(base_source_receipt.resolve()),
        "base_source_receipt_sha256": expected_base_source_receipt_sha,
        "scenes": scene_ids,
        "scene_count": len(scene_ids),
        "selected_scene_count": len(scene_ids),
        "constructible_scene_count": len(constructible_indices),
        "evaluation_scene_indices": constructible_indices,
        "episode_count": len(all_episode_keys),
        "episodes_per_scene": episode_count,
        "paths": {
            "generated_root": str(generated_root.resolve()),
            "data_root": str(data_root.resolve()),
        },
        "assets": assets,
        "episodes": episodes,
        "dependencies": dependency_records,
        "generation": protocol["generation"],
        "evaluation": protocol["evaluation"],
        "analysis": protocol["analysis"],
        "frozen_guards": protocol["frozen_guards"],
        "construction_attrition": {
            "target_scene_count": len(scene_ids),
            "target_episode_count": (
                len(scene_ids) * episode_count),
            "constructible_scene_count": len(constructible_indices),
            "constructible_episode_count": len(all_episode_keys),
            "target_met": len(constructible_indices) == len(scene_ids),
            "underpowered": len(constructible_indices) != len(scene_ids),
            "evaluation_scene_indices": constructible_indices,
            "receipts": construction_receipts,
            "navigation_outcomes_read": False,
        },
        "audit": {
            "status": "ok",
            "physical_assets": True,
            "physical_episode_files": True,
            "unique_episode_keys": True,
            "scene_balanced": len(constructible_indices) == len(scene_ids),
            "scene_balanced_among_constructible": all(
                len(episodes[scene_ids[index]]) == episode_count
                for index in constructible_indices),
            "construction_attrition_explicit": bool(construction_receipts),
            "no_mp3d_evaluation": True,
            "outcome_fields_read": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--base-source-receipt", type=Path, required=True)
    parser.add_argument("--expected-base-source-receipt-sha", required=True)
    parser.add_argument("--gatecurr-checkpoint", type=Path, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--lingbot-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        protocol_path=args.protocol.resolve(),
        generated_root=args.generated_root.resolve(),
        data_root=args.data_root.resolve(),
        asset_receipt=args.asset_receipt.resolve(),
        base_source_receipt=args.base_source_receipt.resolve(),
        expected_base_source_receipt_sha=args.expected_base_source_receipt_sha,
        dependencies={
            "gatecurr600": args.gatecurr_checkpoint.resolve(),
            "navdp_checkpoint": args.navdp_checkpoint.resolve(),
            "lingbot_map_long": args.lingbot_weights.resolve(),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(args.out, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "status": "complete",
        "manifest": str(args.out.resolve()),
        "scenes": manifest["scene_count"],
        "episodes": manifest["episode_count"],
        "protocol_sha256": manifest["protocol_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
