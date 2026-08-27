#!/usr/bin/env python3
"""Build a metadata-only scene-role receipt for the 3-leg episode pool.

The audit intentionally never opens an episode file.  It reads only scene
directory names, counts directories named ``episode_*``, and consumes the
scene-role fields of already frozen manifests.  The output contains counts and
digests rather than scene identifiers so held-out membership is not exposed in
the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "three_leg_scene_role_receipt_v1_20260813"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _string_set(value: Any, field: str) -> set[str]:
    require(isinstance(value, list), f"{field} must be a list")
    result = {str(item) for item in value}
    require(len(result) == len(value), f"{field} contains duplicates")
    return result


def inventory_episode_directories(root: Path) -> dict[str, int]:
    require(root.is_dir(), f"3-leg root is missing: {root}")
    counts: dict[str, int] = {}
    for scene_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not scene_dir.is_dir():
            continue
        count = sum(
            child.is_dir() and child.name.startswith("episode_")
            for child in scene_dir.iterdir()
        )
        if count:
            counts[scene_dir.name] = count
    require(counts, "3-leg root contains no episode directories")
    return counts


def build_receipt(
    *,
    three_leg_root: Path,
    consumed_manifest: Path,
    role_split: Path,
    blind_role_manifest: Path,
    expected_episode_count: int | None = None,
) -> dict[str, Any]:
    counts_by_scene = inventory_episode_directories(three_leg_root)
    pool = set(counts_by_scene)

    consumed_payload = json.loads(consumed_manifest.read_text())
    split_payload = json.loads(role_split.read_text())
    blind_payload = json.loads(blind_role_manifest.read_text())

    consumed = _string_set(consumed_payload.get("scenes"), "consumed.scenes")
    train = _string_set(split_payload.get("train"), "split.train")
    development = _string_set(
        split_payload.get("development"), "split.development"
    )
    final_reserved = _string_set(
        split_payload.get("final_reserved"), "split.final_reserved"
    )
    selection = blind_payload.get("selection")
    require(isinstance(selection, dict), "blind.selection must be an object")
    blind = _string_set(selection.get("selected_scenes"), "blind.selected_scenes")

    require(not (train & development), "train/development role overlap")
    require(not (train & final_reserved), "train/final role overlap")
    require(not (development & final_reserved), "development/final role overlap")

    episode_count = sum(counts_by_scene.values())
    if expected_episode_count is not None:
        require(
            episode_count == expected_episode_count,
            f"episode count changed: {episode_count} != {expected_episode_count}",
        )

    remaining = pool - consumed - train - development - final_reserved
    remaining_blind = remaining & blind
    remaining_outside_blind = remaining - blind
    decision = (
        "stop_before_blind_confirmation"
        if remaining and remaining == remaining_blind
        else "nonblind_scene_disjoint_candidates_exist"
        if remaining_outside_blind
        else "no_scene_disjoint_candidates"
    )

    scene_rows = "\n".join(sorted(pool)).encode()
    count_rows = "\n".join(
        f"{scene}\t{counts_by_scene[scene]}" for scene in sorted(pool)
    ).encode()
    consumed_rows = "\n".join(sorted(consumed)).encode()

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "episode_target_or_outcome_read": False,
        "inventory_scope": {
            "fields_read": [
                "3-leg episode-directory names and counts",
                "consumed data_manifest.scenes",
                "frozen split scene-role lists",
            ],
            "three_leg_episode_root": str(three_leg_root.resolve()),
        },
        "source_receipts": {
            "consumed20_manifest_sha256": sha256_file(consumed_manifest),
            "consumed20_scene_name_digest": sha256_bytes(consumed_rows),
            "frozen_role_split_sha256": sha256_file(role_split),
            "frozen_blind_role_manifest_sha256": sha256_file(
                blind_role_manifest
            ),
            "three_leg_pool_scene_episode_count_digest": sha256_bytes(count_rows),
            "three_leg_pool_scene_name_digest": sha256_bytes(scene_rows),
        },
        "counts": {
            "three_leg_episodes": episode_count,
            "three_leg_nonempty_scene_clusters": len(pool),
            "intersection_consumed20": len(pool & consumed),
            "intersection_train40": len(pool & train),
            "intersection_development10": len(pool & development),
            "intersection_final_reserved4": len(pool & final_reserved),
            "remaining_after_consumed_train_development_final": len(remaining),
            "remaining_intersection_blind16": len(remaining_blind),
            "remaining_outside_blind": len(remaining_outside_blind),
        },
        "interpretation": [
            "The nominal pool size is an episode count, not an independent scene-cluster count.",
            "No episode file, target, policy outcome, or blind result was read.",
            "A held-out confirmation requires an explicit one-shot authorization.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-leg-root", type=Path, required=True)
    parser.add_argument("--consumed-manifest", type=Path, required=True)
    parser.add_argument("--role-split", type=Path, required=True)
    parser.add_argument("--blind-role-manifest", type=Path, required=True)
    parser.add_argument("--expected-episode-count", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_receipt(
        three_leg_root=args.three_leg_root,
        consumed_manifest=args.consumed_manifest,
        role_split=args.role_split,
        blind_role_manifest=args.blind_role_manifest,
        expected_episode_count=args.expected_episode_count,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(rendered)
    partial.replace(args.output)


if __name__ == "__main__":
    main()
