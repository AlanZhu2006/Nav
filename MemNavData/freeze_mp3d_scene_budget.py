#!/usr/bin/env python3
"""Freeze the 90-scene MP3D evidence budget before new control outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novel_memory_direction_control import sha256_file


SCHEMA = "mp3d_scene_budget_v1_20260816"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def freeze(
    *,
    asset_root: Path,
    train_manifest: Path,
    development_manifest: Path,
    consumed_blind_manifest: Path,
    out: Path,
) -> dict:
    require(not out.exists(), f"output already exists: {out}")
    all_scenes = {path.stem for path in asset_root.glob("*/*.glb")}
    require(len(all_scenes) == 90, "official MP3D asset inventory is not 90 scenes")

    train_payload = json.loads(train_manifest.read_text())
    train = {
        str(session).split("/")[1]
        for session in train_payload["sessions"]
    }
    development_payload = json.loads(development_manifest.read_text())
    development = set(development_payload["selection"]["selected_scenes"])
    blind_payload = json.loads(consumed_blind_manifest.read_text())
    consumed_blind = set(blind_payload["selection"]["selected_scenes"])

    require(len(train) == 40, "train scene count changed")
    require(len(development) == 20, "development scene count changed")
    require(len(consumed_blind) == 16, "consumed blind scene count changed")
    require(not (train & development), "train/development scene overlap")
    require(not (train & consumed_blind), "train/blind scene overlap")
    require(not (development & consumed_blind), "development/blind scene overlap")
    used = train | development | consumed_blind
    untouched = all_scenes - used
    require(len(used) == 76 and len(untouched) == 14, "MP3D scene ledger changed")

    payload = {
        "schema_version": SCHEMA,
        "frozen_date": "2026-08-16",
        "freeze_precedes_new_control_outcomes": True,
        "official_asset_scene_count": len(all_scenes),
        "partitions": {
            "train40": sorted(train),
            "consumed_development20": sorted(development),
            "consumed_blind16": sorted(consumed_blind),
            "untouched_final14": sorted(untouched),
        },
        "partition_counts": {
            "train40": len(train),
            "consumed_development20": len(development),
            "consumed_blind16": len(consumed_blind),
            "untouched_final14": len(untouched),
        },
        "pairwise_disjoint": True,
        "union_scene_count": len(all_scenes),
        "novel_control_allocation": (
            "consumed Phase-2 blind16 only; development-stage mechanism evidence, never confirmation"
        ),
        "final_confirmation_allocation": (
            "all untouched_final14 scenes; no development outcome may be read"
        ),
        "fresh_twenty_scene_protocol_constructible": False,
        "source_receipts": {
            "asset_root": str(asset_root.resolve()),
            "train_manifest": {
                "path": str(train_manifest.resolve()),
                "sha256": sha256_file(train_manifest),
            },
            "development_manifest": {
                "path": str(development_manifest.resolve()),
                "sha256": sha256_file(development_manifest),
            },
            "consumed_blind_manifest": {
                "path": str(consumed_blind_manifest.resolve()),
                "sha256": sha256_file(consumed_blind_manifest),
            },
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out.with_name(out.name + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--consumed-blind-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        asset_root=args.asset_root,
        train_manifest=args.train_manifest,
        development_manifest=args.development_manifest,
        consumed_blind_manifest=args.consumed_blind_manifest,
        out=args.out,
    )
    print(json.dumps(result["partition_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
