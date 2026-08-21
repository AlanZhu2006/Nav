#!/usr/bin/env python3
"""Aggregate the frozen Replica source/online-A construction population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def summarize(root: Path, freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    require(freeze["query_outcomes_read"] is False, "freeze read query outcomes")
    rows = []
    materialized = []
    materialization_attrition = []
    source_requested = 0
    source_generated = 0
    for spec in freeze["scenes"]:
        index = int(spec["index"]); scene = str(spec["scene"])
        scene_root = root / "traces" / f"{index:02d}_{scene}"
        receipt_path = scene_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt["scene_index"] == index, "scene receipt index differs")
        require(receipt["scene"] == scene, "scene receipt identity differs")
        require(receipt["query_outcomes_read"] is False, "query outcome leak")
        require(
            receipt["freeze_manifest_sha256"] == sha256_file(freeze_path),
            "freeze manifest changed",
        )
        source_requested += int(receipt["source_attempts"])
        source_generated += int(receipt["source_generated"])
        online = json.loads((scene_root / "online_a/manifest.json").read_text())
        materialized.extend(online["episodes"])
        materialization_attrition.extend(online["attrition"])
        for trace in receipt["traces"]:
            path = scene_root / "native_a" / f"{trace['episode']}_leg1_trace.json"
            require(sha256_file(path) == trace["sha256"], "trace hash changed")
            rows.append({
                "scene_index": index,
                "scene": scene,
                "analysis_status": receipt["analysis_status"],
                "episode": trace["episode"],
                "reached": bool(trace["reached"]),
                "steps": int(trace["steps"]),
                "trace": str(path.resolve()),
                "trace_sha256": trace["sha256"],
            })
    successes = [row for row in rows if row["reached"]]
    return {
        "schema_version": "replica_formal_online_a_inventory_v1_20260814",
        "scope": "Replica source/native Goal-A construction; no memory query evaluated",
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(freeze_path),
        "source_scenes": len(freeze["scenes"]),
        "source_episodes": source_requested,
        "generated_source_episodes": source_generated,
        "source_generation_attrition_count": source_requested - source_generated,
        "goal_a_successes": len(successes),
        "goal_a_failures": len(rows) - len(successes),
        "successful_scene_count": len({row["scene"] for row in successes}),
        "materialized_histories": len(materialized),
        "materialized_scene_count": len({row["scene"] for row in materialized}),
        "materialization_attrition_count": len(materialization_attrition),
        "materialization_attrition": materialization_attrition,
        "pilot_scene_excluded_from_primary": freeze[
            "pilot_scene_excluded_from_primary"
        ],
        "query_outcomes_read": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.freeze_manifest)
    require(not args.out.exists(), f"output exists: {args.out}")
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: result[key] for key in (
            "source_scenes", "source_episodes", "generated_source_episodes",
            "goal_a_successes", "materialized_histories",
            "query_outcomes_read",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
