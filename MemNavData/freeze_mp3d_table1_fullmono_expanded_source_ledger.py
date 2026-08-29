#!/usr/bin/env python3
"""Freeze the 20+16 scene MP3D full-mono source ledger without outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from MemNavData.deterministic_eval_protocol import validate_leg1_trace
from MemNavData.mp3d_table1_new_query_contract import (
    SOURCE_LEDGER_EXPANSION_SCHEMA,
    SOURCE_LEDGER_SCHEMA,
    require,
)


PROTOCOL_SCHEMA = "mp3d_table1_fullmono_source_expansion_protocol_v1_20260829"
COLLECTION_SCHEMA = "mp3d_table1_fullmono_goal_a_scene_v1_20260829"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def consumed_identity(query: dict[str, Any], *, provenance: str) -> dict[str, Any]:
    return {
        "goal_rgb_sha256": str(query["goal_rgb_sha256"]),
        "floor_position": [float(value) for value in query["floor_position"]],
        "yaw_rad": float(query["yaw_rad"]),
        "provenance": provenance,
    }


def identity_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["goal_rgb_sha256"]),
        *(round(float(value), 7) for value in row["floor_position"]),
        round(float(row["yaw_rad"]), 7),
    )


def read_consumed_query_manifests(
    manifest_specs: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipts = []
    for spec in manifest_specs:
        path = Path(spec["path"])
        require(path.is_file() and sha256_file(path) == spec["sha256"],
                f"consumed-query manifest changed: {path}")
        payload = json.loads(path.read_text())
        count = 0
        for history in payload["episodes"]:
            scene = str(history["scene"])
            for pair in history["pairs"]:
                for query in pair["queries"]:
                    goal_path = Path(query["goal_rgb"])
                    if not goal_path.is_absolute():
                        goal_path = (
                            path.parent / scene / str(history["episode"]) /
                            goal_path
                        ).resolve()
                    require(goal_path.is_file()
                            and sha256_file(goal_path)
                            == str(query["goal_rgb_sha256"]),
                            f"consumed query image changed: {goal_path}")
                    by_scene[scene].append(consumed_identity(
                        query, provenance=f"old_query_manifest:{path}",
                    ))
                    count += 1
        receipts.append({
            "path": str(path.resolve()),
            "sha256": str(spec["sha256"]),
            "query_identities": count,
        })
    for scene, rows in by_scene.items():
        unique = {identity_key(row): row for row in rows}
        by_scene[scene] = [unique[key] for key in sorted(unique)]
    return dict(by_scene), receipts


def freeze(*, base_ledger_path: Path, protocol_path: Path,
           expansion_manifest_path: Path, collection_root: Path,
           out: Path) -> dict[str, Any]:
    require(not out.exists(), f"output already exists: {out}")
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "source-expansion protocol changed")
    base_spec = protocol["base_population"]
    require(sha256_file(base_ledger_path) == base_spec["source_ledger_sha256"],
            "base source ledger changed")
    base = json.loads(base_ledger_path.read_text())
    require(base.get("schema_version") == SOURCE_LEDGER_SCHEMA,
            "base source-ledger schema changed")
    require(base.get("previous_goal_b_policy_outcomes_read") is False
            and base.get("query_policy_outcomes_read") is False,
            "base source ledger read a policy outcome")

    expansion_spec = protocol["expansion_source"]
    require(sha256_file(expansion_manifest_path)
            == expansion_spec["manifest_sha256"],
            "phase-2 source manifest changed")
    manifest = json.loads(expansion_manifest_path.read_text())
    expansion_scenes = [
        str(value) for value in manifest["selection"]["selected_scenes"]
    ]
    require(len(expansion_scenes) == int(expansion_spec["scenes"])
            and len(expansion_scenes) == len(set(expansion_scenes)),
            "expansion scene set changed")
    base_scenes = [str(row["scene"]) for row in base["scenes"]]
    require(set(base_scenes).isdisjoint(expansion_scenes),
            "base and expansion scenes overlap")

    old_queries, old_query_receipts = read_consumed_query_manifests(
        protocol["consumed_query_exclusion"]["old_query_manifests"],
    )
    declared_expansion = protocol["dataset"]["scenes"]
    require([str(row["scene_id"]) for row in declared_expansion]
            == expansion_scenes, "protocol expansion scene order changed")

    rows = []
    for index, source_scene in enumerate(base["scenes"]):
        row = copy.deepcopy(source_scene)
        require(int(row["scene_index"]) == index,
                "base scene order changed")
        for episode in row["episodes"]:
            episode["consumed_queries"] = [copy.deepcopy(
                episode["consumed_goal_b"]
            )]
        rows.append(row)

    expansion_episode_ids = list(expansion_spec["episode_ids"])
    protocol_sha = sha256_file(protocol_path)
    manifest_sha = sha256_file(expansion_manifest_path)
    collection_receipts = []
    for expansion_rank, scene in enumerate(expansion_scenes):
        scene_index = len(rows)
        require(scene_index == len(base_scenes) + expansion_rank,
                "expanded scene rank changed")
        scene_root = (collection_root / "goal_a/scenes" /
                      f"{expansion_rank:02d}_{scene}")
        completion_path = scene_root / "completion.json"
        require(completion_path.is_file(),
                f"Goal-A completion missing: {scene}")
        completion = json.loads(completion_path.read_text())
        require(completion.get("schema_version") == COLLECTION_SCHEMA
                and completion.get("status") == "complete"
                and completion.get("formal") is True,
                f"{scene}: Goal-A collection is not formal/complete")
        require(int(completion["scene_index"]) == expansion_rank
                and str(completion["scene"]) == scene,
                f"{scene}: Goal-A completion identity changed")
        require(completion.get("protocol_sha256") == protocol_sha
                and completion.get("source_manifest_sha256") == manifest_sha,
                f"{scene}: Goal-A source binding changed")
        require(int(completion["metric_depth_sensor_reads"]) == 0
                and completion.get("query_outcomes_generated") is False
                and completion.get("query_outcomes_read") is False,
                f"{scene}: Goal-A collection violated outcome/depth contract")
        records = list(completion["records"])
        require([str(record["episode"]) for record in records]
                == expansion_episode_ids,
                f"{scene}: Goal-A episode order changed")

        asset = (Path(manifest["paths"]["asset_root"]) /
                 scene / f"{scene}.glb")
        require(asset.is_file() and sha256_file(asset)
                == manifest["assets"][scene]["sha256"],
                f"{scene}: source asset changed")
        episode_root = Path(manifest["paths"]["expanded_episode_root"])
        source_rows = {
            str(row["episode"]): row for row in manifest["episodes"][scene]
        }
        episodes = []
        for episode_rank, record in enumerate(records):
            episode = str(record["episode"])
            source = source_rows[episode]
            trace = Path(record["trace_path"])
            require(trace.is_file()
                    and sha256_file(trace) == record["trace_sha256"],
                    f"{scene}/{episode}: Goal-A trace changed")
            trace_payload = json.loads(trace.read_text())
            validate_leg1_trace(trace_payload)
            require(str(trace_payload["source_scene"]) == scene
                    and str(trace_payload["episode"]) == episode,
                    f"{scene}/{episode}: Goal-A trace identity changed")

            source_episode = episode_root / scene / episode
            metadata_path = source_episode / "meta/gen_meta.json"
            parquet_path = (source_episode / "data/chunk-000" /
                            "episode_000000.parquet")
            goal_path = source_episode / "goal_image.jpg"
            expected = source["files"]
            require(metadata_path.is_file()
                    and sha256_file(metadata_path)
                    == expected["metadata"]["sha256"],
                    f"{scene}/{episode}: metadata changed")
            require(parquet_path.is_file() and sha256_file(parquet_path)
                    == expected["parquet"]["sha256"],
                    f"{scene}/{episode}: parquet changed")
            require(goal_path.is_file() and sha256_file(goal_path)
                    == expected["goal"]["sha256"],
                    f"{scene}/{episode}: source Goal-B changed")
            metadata = json.loads(metadata_path.read_text())
            goals = metadata.get("goals")
            require(isinstance(goals, list) and len(goals) == 1
                    and str(goals[0].get("name")) == "B",
                    f"{scene}/{episode}: source Goal-B identity changed")
            goal = goals[0]
            source_goal = {
                "goal_rgb_path": str(goal_path.resolve()),
                "goal_rgb_sha256": sha256_file(goal_path),
                "floor_position": [float(value) for value in goal["pos"]],
                "yaw_rad": float(goal["yaw_habitat"]),
            }
            consumed = [copy.deepcopy(source_goal), *copy.deepcopy(
                old_queries.get(scene, [])
            )]
            unique = {identity_key(row): row for row in consumed}
            episodes.append({
                "episode": episode,
                "episode_rank": episode_rank,
                "trace_path": str(trace.resolve()),
                "trace_sha256": sha256_file(trace),
                "source_metadata_path": str(metadata_path.resolve()),
                "source_metadata_sha256": sha256_file(metadata_path),
                "source_parquet_path": str(parquet_path.resolve()),
                "source_parquet_sha256": sha256_file(parquet_path),
                "consumed_goal_b": source_goal,
                "consumed_queries": [unique[key] for key in sorted(unique)],
            })
        rows.append({
            "scene": scene,
            "scene_index": scene_index,
            "source_population": "pre_result_phase2_fullmono_reroll",
            "asset_path": str(asset.resolve()),
            "asset_sha256": sha256_file(asset),
            "episode_root": str(episode_root.resolve()),
            "episodes": episodes,
        })
        collection_receipts.append({
            "scene": scene,
            "scene_index": scene_index,
            "completion_path": str(completion_path.resolve()),
            "completion_sha256": sha256_file(completion_path),
            "source_histories": len(records),
        })

    require(len(rows) == len(base_scenes) + len(expansion_scenes),
            "expanded source scene count changed")
    payload = {
        "schema_version": SOURCE_LEDGER_EXPANSION_SCHEMA,
        "scope": (
            "outcome-blind full-mono MP3D source expansion for new-query "
            "Table-1 construction"
        ),
        "source_expansion_protocol": str(protocol_path.resolve()),
        "source_expansion_protocol_sha256": protocol_sha,
        "base_source_ledger": str(base_ledger_path.resolve()),
        "base_source_ledger_sha256": sha256_file(base_ledger_path),
        "expansion_manifest": str(expansion_manifest_path.resolve()),
        "expansion_manifest_sha256": manifest_sha,
        "collection_root": str(collection_root.resolve()),
        "collection_receipts": collection_receipts,
        "consumed_query_manifest_receipts": old_query_receipts,
        "scene_count": len(rows),
        "base_scene_count": len(base_scenes),
        "expansion_scene_count": len(expansion_scenes),
        "source_history_count": sum(len(row["episodes"]) for row in rows),
        "previous_goal_b_policy_outcomes_read": False,
        "old_query_policy_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "history_selection_rule": (
            "base_all_first_two_plus_pre_result_phase2_all_episode_0002_0005"
        ),
        "scenes": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source-ledger", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expansion-manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        base_ledger_path=args.base_source_ledger.resolve(),
        protocol_path=args.protocol.resolve(),
        expansion_manifest_path=args.expansion_manifest.resolve(),
        collection_root=args.collection_root.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "scene_count": result["scene_count"],
        "source_history_count": result["source_history_count"],
        "policy_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
