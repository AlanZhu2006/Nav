#!/usr/bin/env python3
"""Write a policy-query-blind receipt for one Replica online-A fragment."""

from __future__ import annotations

import argparse
import csv
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


def write_receipt(
    run_root: Path, freeze_path: Path, scene_index: int, out: Path
) -> dict:
    freeze = json.loads(freeze_path.read_text())
    require(0 <= scene_index < len(freeze["scenes"]), "scene index out of range")
    spec = freeze["scenes"][scene_index]
    scene = str(spec["scene"])
    source_root = run_root / "source_episodes" / scene
    trace_root = run_root / "traces" / f"{scene_index:02d}_{scene}"
    generation = json.loads((source_root / "generation_summary.json").read_text())
    episode_ids = sorted(path.name for path in source_root.glob("episode_*"))
    require(
        len(episode_ids) == int(generation["generated_episodes"]),
        "generated source episode count differs",
    )

    traces = []
    if episode_ids:
        summary = json.loads((trace_root / "native_a/summary.json").read_text())
        require(int(summary["episodes"]) == len(episode_ids), "native-A count differs")
        require(summary["leg1_policy_backend"] == "navdp", "native-A backend changed")
        require(summary["stop_after_leg1"] is True, "native-A did not stop after A")
        with (trace_root / "native_a/metric.csv").open(newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        require(
            [row["episode"] for row in metric_rows] == episode_ids,
            "native-A identity/order differs",
        )
        for episode, metric in zip(episode_ids, metric_rows):
            path = trace_root / "native_a" / f"{episode}_leg1_trace.json"
            payload = json.loads(path.read_text())
            require(payload["source_scene"] == scene, "trace scene identity differs")
            require(payload["episode"] == episode, "trace episode identity differs")
            require(bool(payload["reached"]) == bool(int(float(metric["reached_A"]))),
                    "trace reach flag differs")
            traces.append({
                "episode": episode,
                "reached": bool(payload["reached"]),
                "steps": int(payload["steps"]),
                "sha256": sha256_file(path),
            })

    online_path = trace_root / "online_a/manifest.json"
    online = json.loads(online_path.read_text())
    require(online["schema_version"] == "shared_online_a_materialized_v1",
            "online-A materialization schema changed")
    require(int(online["source_trace_count"]) == len(episode_ids),
            "online-A source trace count differs")
    construction = json.loads(
        (trace_root / "role_pairs/construction_receipt.json").read_text()
    )
    require(construction["policy_outcomes_read"] is False, "query outcome leak")
    receipt = {
        "schema_version": "replica_formal_online_a_scene_receipt_v1_20260814",
        "scene_index": scene_index,
        "scene": scene,
        "analysis_status": spec["analysis_status"],
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "source_stratum": spec["source_stratum"],
        "source_attempts": int(spec["source_attempts"]),
        "source_generated": len(episode_ids),
        "source_generation_complete": bool(generation["complete"]),
        "source_generation_summary_sha256": sha256_file(
            source_root / "generation_summary.json"
        ),
        "policy": "frozen_native_navdp_imagegoal",
        "query_outcomes_read": False,
        "traces": traces,
        "materialization": online["selection"],
        "materialized_episode_count": len(online["episodes"]),
        "materialization_attrition": online["attrition"],
        "role_pair_construction": construction,
    }
    require(not out.exists(), f"receipt already exists: {out}")
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = write_receipt(
        args.run_root, args.freeze_manifest, args.scene_index, args.out
    )
    print(json.dumps({
        "scene": receipt["scene"],
        "source_generated": receipt["source_generated"],
        "materialized": receipt["materialized_episode_count"],
        "constructible": receipt["role_pair_construction"][
            "constructible_histories"
        ],
        "query_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
