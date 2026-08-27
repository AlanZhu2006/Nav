#!/usr/bin/env python3
"""Freeze result-blind, one-hour factual-B execution shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hm3d_fullmono_lifelong_natural_v4_b_shards_v1_20260827"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(manifest: dict, *, manifest_sha256: str,
          maximum_histories_per_shard: int) -> dict:
    require(maximum_histories_per_shard > 0, "shard size must be positive")
    grouped: dict[int, list[int]] = {}
    scenes: dict[int, str] = {}
    source_dependencies = set()
    for index, item in enumerate(manifest["episodes"]):
        scene_index = int(item["final14_scene_rank"])
        scene = str(item["scene"])
        require(scene_index not in scenes or scenes[scene_index] == scene,
                "scene rank maps to multiple scenes")
        scenes[scene_index] = scene
        grouped.setdefault(scene_index, []).append(index)
        construction = item["lifelong_construction"]
        source_dependencies.add((scene, construction["recipient_episode"]))
    shards = []
    for scene_index in sorted(grouped):
        indices = grouped[scene_index]
        for offset in range(0, len(indices), maximum_histories_per_shard):
            selected = indices[offset:offset + maximum_histories_per_shard]
            shards.append({
                "shard_index": len(shards),
                "scene_index": scene_index,
                "scene": scenes[scene_index],
                "history_indices": selected,
                "history_count": len(selected),
                "navigation_outcomes_read": False,
            })
    flattened = [index for shard in shards for index in shard["history_indices"]]
    require(sorted(flattened) == list(range(len(manifest["episodes"]))),
            "shards do not exactly partition the manifest")
    require(len(flattened) == len(set(flattened)),
            "factual-B shard contains duplicate histories")
    return {
        "schema_version": SCHEMA,
        "scope": "result-blind factual mono B execution scheduling only",
        "benchmark_manifest_sha256": manifest_sha256,
        "candidate_histories": len(flattened),
        "source_recipient_histories": len(source_dependencies),
        "scene_clusters": len(grouped),
        "maximum_histories_per_shard": maximum_histories_per_shard,
        "shard_count": len(shards),
        "all_candidates_partitioned_once": True,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-histories-per-shard", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.manifest.with_name(args.manifest.name + ".sha256")
    require(sidecar.is_file(), "benchmark manifest sidecar is missing")
    fields = sidecar.read_text().split()
    digest = sha256(args.manifest)
    require(len(fields) == 2 and fields[0] == digest
            and fields[1] == args.manifest.name,
            "benchmark manifest sidecar changed")
    result = build(
        json.loads(args.manifest.read_text()),
        manifest_sha256=digest,
        maximum_histories_per_shard=args.maximum_histories_per_shard,
    )
    require(not args.out.exists(), "factual-B shard manifest exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps({
        "candidate_histories": result["candidate_histories"],
        "scene_clusters": result["scene_clusters"],
        "shard_count": result["shard_count"],
        "maximum_histories_per_shard": result[
            "maximum_histories_per_shard"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
