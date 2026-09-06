#!/usr/bin/env python3
"""Aggregate the sealed 16-history long-range attribution fork."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.longrange_oracle_attribution import ORACLE_ARMS
from MemNavData.run_hm3d_longrange_oracle_attribution import (
    FROZEN_INDICES,
    SCHEMA_VERSION as EPISODE_SCHEMA,
)
from MemNavData.analyze_hm3d_action_coordinate_compass_pair import (
    contrast,
    spl,
)


SCHEMA_VERSION = "hm3d_longrange_oracle_attribution_result_v1_20260903"
ACTION, ROUTE, GEODESIC_MIXED, GEODESIC_POINT = ORACLE_ARMS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file()
            and sidecar.read_text(encoding="utf-8").split()
            == [digest, path.name],
            f"invalid SHA receipt: {path}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "aggregate result already exists")
    require(sha256(arguments.manifest)
            == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    require(len(manifest["episodes"]) == 48,
            "sealed length population must contain 48 histories")

    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for index in FROZEN_INDICES:
        item = manifest["episodes"][index]
        require(item["bin_name"] == "30_to_50_m",
                f"index {index} escaped the long-range stratum")
        root = (arguments.run_root / "development" /
                "longrange_oracle_attribution" /
                f"{index:03d}_{item['scene']}_{item['episode']}")
        path = root / "completion.json"
        digest = verify_sidecar(path)
        source_hashes[str(path.relative_to(arguments.run_root))] = digest
        completion = json.loads(path.read_text(encoding="utf-8"))
        require(
            completion.get("schema_version") == EPISODE_SCHEMA
            and int(completion["history_index"]) == index
            and completion["scene"] == item["scene"]
            and completion["episode"] == item["episode"]
            and completion["arms"] == list(ORACLE_ARMS)
            and completion.get("paper_result_eligible") is False
            and completion.get("prefix_equality") is True
            and completion.get("target_proof_equality") is True,
            f"episode contract changed at index {index}",
        )
        row: dict[str, Any] = {
            "population_index": index,
            "scene": item["scene"],
            "episode": item["episode"],
            "bin_name": item["bin_name"],
        }
        for arm in ORACLE_ARMS:
            row[arm] = int(completion["outcomes"][arm])
            row[f"{arm}_geodesic_m"] = float(
                completion["geodesic_m"][arm])
            row[f"{arm}_path_len_m"] = float(
                completion["path_len_m"][arm])
            row[f"{arm}_final_distance_m"] = float(
                completion["final_distance_m"][arm])
            row[f"{arm}_steps"] = int(completion["query_steps"][arm])
            row[f"{arm}_spl"] = spl(
                row[arm], row[f"{arm}_geodesic_m"],
                row[f"{arm}_path_len_m"])
        records.append(row)

    contrasts = {
        "oracle_route_vs_action_coordinate": contrast(
            records, left=ROUTE, right=ACTION, seed=60931),
        "oracle_geodesic_mixed_vs_action_coordinate": contrast(
            records, left=GEODESIC_MIXED, right=ACTION, seed=60932),
        "oracle_geodesic_mixed_vs_oracle_route": contrast(
            records, left=GEODESIC_MIXED, right=ROUTE, seed=60933),
        "oracle_geodesic_point_vs_oracle_geodesic_mixed": contrast(
            records, left=GEODESIC_POINT, right=GEODESIC_MIXED, seed=60934),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "consumed-development privileged attribution only",
        "paper_result_eligible": False,
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "histories": len(records),
        "scene_clusters": len({row["scene"] for row in records}),
        "distance_bin": "30_to_50_m",
        "arms": list(ORACLE_ARMS),
        "contrasts": contrasts,
        "mean_metrics_by_arm": {
            arm: {
                "SR": float(np.mean([row[arm] for row in records])),
                "SPL": float(np.mean([
                    row[f"{arm}_spl"] for row in records])),
                "path_length_m": float(np.mean([
                    row[f"{arm}_path_len_m"] for row in records])),
                "final_distance_m": float(np.mean([
                    row[f"{arm}_final_distance_m"] for row in records])),
            }
            for arm in ORACLE_ARMS
        },
        "records": records,
        "completion_sha256": source_hashes,
        "partial_results_reported": False,
        "fallback_completion_used": False,
        "post_hoc_pass_threshold_defined": False,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    arguments.out.with_name(arguments.out.name + ".sha256").write_text(
        f"{sha256(arguments.out)}  {arguments.out.name}\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete", "contrasts": contrasts,
        "out": str(arguments.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
