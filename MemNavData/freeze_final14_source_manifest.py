#!/usr/bin/env python3
"""Freeze the result-blind Final14 Goal-A source manifest.

The parent protocol fixes all 14 scenes through the immutable MP3D scene
budget and asks for the lexicographically first eight *available* source
episodes per scene.  This freezer is the only component allowed to translate
that ledger into the paper collection manifest.  It reads no navigation
outcome and reports only aggregate counts on stdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from build_graph_blind_manifest import episode_record, sha_record


SCHEMA_VERSION = "final14_paper_source_manifest_v1_20260817"
EXPECTED_SCENE_BUDGET_SHA256 = (
    "779e2d7d63faa0f9b9e735680b1d620f04428c11a57ac83158933306b62407ef"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9"
)
PARENT_PROTOCOL_SHA256 = (
    "3d1ebc6ef429fd16df4d550eda52eceb55d7b15fd181a5c00c0b8f971f7aaa32"
)
SOURCE_EPISODE_TARGET = 8
FINAL_SCENE_COUNT = 14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def freeze(
    *,
    base_manifest_path: Path,
    scene_budget_path: Path,
    out: Path,
    asset_root_override: Path | None = None,
    episode_root_override: Path | None = None,
    expected_base_manifest_sha256: str = EXPECTED_BASE_MANIFEST_SHA256,
    expected_scene_budget_sha256: str = EXPECTED_SCENE_BUDGET_SHA256,
) -> dict[str, Any]:
    """Build an immutable manifest without consulting any policy outcome."""

    require(not out.exists(), f"output already exists: {out}")
    base_sha = sha256_file(base_manifest_path)
    budget_sha = sha256_file(scene_budget_path)
    require(base_sha == expected_base_manifest_sha256, "base manifest changed")
    require(budget_sha == expected_scene_budget_sha256, "scene budget changed")

    base = json.loads(base_manifest_path.read_text())
    budget = json.loads(scene_budget_path.read_text())
    require(
        budget.get("schema_version") == "mp3d_scene_budget_v1_20260816",
        "scene budget schema changed",
    )
    require(budget.get("freeze_precedes_new_control_outcomes") is True,
            "scene budget is not prospective")
    partitions = budget["partitions"]
    scenes = [str(value) for value in partitions["untouched_final14"]]
    require(len(scenes) == FINAL_SCENE_COUNT, "Final14 scene count changed")
    require(len(set(scenes)) == len(scenes), "Final14 scenes are not unique")
    require(scenes == sorted(scenes), "Final14 ledger order changed")

    train = set(map(str, partitions["train40"]))
    development = set(map(str, partitions["consumed_development20"]))
    consumed_blind = set(map(str, partitions["consumed_blind16"]))
    final = set(scenes)
    require(
        not (train & development or train & consumed_blind or train & final
             or development & consumed_blind or development & final
             or consumed_blind & final),
        "scene partitions overlap",
    )
    require(
        len(train | development | consumed_blind | final) == 90,
        "MP3D scene partition union changed",
    )

    asset_root = asset_root_override or Path(base["paths"]["asset_root"])
    episode_root = episode_root_override or Path(
        base["paths"]["expanded_episode_root"]
    )
    assets: dict[str, dict[str, Any]] = {}
    episodes: dict[str, list[dict[str, Any]]] = {}
    attrition: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for scene in scenes:
        asset = asset_root / scene / f"{scene}.glb"
        require(asset.is_file(), f"missing Final14 scene asset: {scene}")
        assets[scene] = sha_record(asset)

        candidates = sorted(
            path for path in (episode_root / scene).glob("episode_*")
            if path.is_dir()
        )
        selected: list[dict[str, Any]] = []
        examined = 0
        for candidate in candidates:
            if len(selected) >= SOURCE_EPISODE_TARGET:
                break
            examined += 1
            try:
                record = episode_record(candidate, scene)
            except Exception as error:
                attrition.append({
                    "scene": scene,
                    "episode": candidate.name,
                    "stage": "source_asset_validation",
                    "reason": f"{type(error).__name__}: {error}",
                })
            else:
                selected.append(record)
        episodes[scene] = selected
        counts[scene] = len(selected)
        if len(selected) < SOURCE_EPISODE_TARGET:
            attrition.append({
                "scene": scene,
                "episode": None,
                "stage": "source_episode_shortage",
                "reason": (
                    f"available_{len(selected)}_of_{SOURCE_EPISODE_TARGET};"
                    f"examined_{examined}"
                ),
            })

    payload = copy.deepcopy(base)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "created_at": "2026-08-17",
        "purpose": (
            "single prospective Final14 role-free CEC plus learned Pi3X "
            "confirmation; source membership frozen before Goal-A rollout"
        ),
        "training_scenes": sorted(train),
        "assets": assets,
        "episodes": episodes,
        "source_attrition": attrition,
        "final14_source": {
            "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
            "scene_budget_path": str(scene_budget_path.resolve()),
            "scene_budget_sha256": budget_sha,
            "base_manifest_path": str(base_manifest_path.resolve()),
            "base_manifest_sha256": base_sha,
            "policy_outcomes_read": False,
            "scene_replacement": False,
            "episode_selection": (
                "lexicographically_first_eight_available_per_scene"
            ),
            "source_episode_target_per_scene": SOURCE_EPISODE_TARGET,
            "source_episode_counts_by_scene": counts,
            "source_asset_attrition_count": len(attrition),
        },
    })
    payload["paths"] = copy.deepcopy(base["paths"])
    payload["paths"]["asset_root"] = str(asset_root)
    payload["paths"]["expanded_episode_root"] = str(episode_root)
    payload["selection"] = {
        "method": "exact_untouched_final14_ledger_order",
        "selected_scene_count": FINAL_SCENE_COUNT,
        "additional_scene_count": FINAL_SCENE_COUNT,
        "anchor_scenes": [],
        "selected_scenes": scenes,
        "eligible_unseen_scenes": scenes,
        "salt": "none_ledger_identity_is_already_frozen",
    }
    payload["evaluation"] = copy.deepcopy(base["evaluation"])
    payload["evaluation"].pop("episodes_per_scene", None)
    payload["evaluation"].update({
        "episode_target_per_scene": SOURCE_EPISODE_TARGET,
        "episode_counts_by_scene": counts,
        "success_distance_m": 1.0,
        "max_steps_per_leg": 600,
        "execution_horizon": 8,
        "terminal_uturn": "off",
        "terminal_visual_refine": "off",
        "shared_novel_prefix": True,
        "deterministic_per_plan_seed": True,
    })
    payload["frozen_source"] = {
        "scene_budget_sha256": budget_sha,
        "base_manifest_sha256": base_sha,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "query_outcomes_read_before_freeze": False,
    }

    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(out)
    digest = sha256_file(out)
    out.with_name(out.name + ".sha256").write_text(
        f"{digest}  {out.name}\n"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(out.resolve()),
        "sha256": digest,
        "scenes": len(scenes),
        "source_episode_target": SOURCE_EPISODE_TARGET * len(scenes),
        "source_episodes_available": sum(counts.values()),
        "scenes_below_target": sum(
            count < SOURCE_EPISODE_TARGET for count in counts.values()
        ),
        "source_asset_attrition_count": len(attrition),
        "policy_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--scene-budget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--episode-root", type=Path)
    args = parser.parse_args()
    result = freeze(
        base_manifest_path=args.base_manifest,
        scene_budget_path=args.scene_budget,
        out=args.out,
        asset_root_override=args.asset_root,
        episode_root_override=args.episode_root,
    )
    if args.receipt is not None:
        require(not args.receipt.exists(), f"receipt exists: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        key: result[key]
        for key in (
            "sha256", "scenes", "source_episode_target",
            "source_episodes_available", "scenes_below_target",
            "source_asset_attrition_count", "policy_outcomes_read",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
