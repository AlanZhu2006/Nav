#!/usr/bin/env python3
"""Summarize the missing Final14 zero-depth row with frozen reference arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.final14_mono_factorial import (
    paired_contrast,
    require,
    scene_cluster_interval,
)
from MemNavData.final14_zero_depth import ARM, DEPTH_SOURCE
from MemNavData.run_final14_mono_factorial_episode import sha256


SCHEMA = "final14_zero_depth_summary_v1_20260828"
EPISODE_SCHEMA = "final14_zero_depth_episode_v1_20260828"
REFERENCE_ARMS = ("metric_native", "mono_native", "metric_cec", "mono_cec")
TABLE_ARMS = ("metric_native", ARM, "mono_native", "metric_cec", "mono_cec")
BOOTSTRAP_SEED = 2026082801
BOOTSTRAP_RESAMPLES = 100_000
REFERENCE_SUMMARY_SHA = "ae24138b3cd7fecb737dffe8454eb91aaae8f5aa19d4e679e67e039d793e17b7"
REFERENCE_VERIFY_SHA = "7bf7e496c1a9cc53f3dc9ef0ff0194cce61f03d554a84678210320f699fbb35f"


def verify_receipt(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"receipt missing: {path}")
    fields = sidecar.read_text().split()
    digest = sha256(path)
    require(fields == [digest, path.name], f"receipt mismatch: {path}")
    return digest


def read_metric(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def record(index: int, arm: str, row: dict[str, str]) -> dict[str, Any]:
    reached = int(row["reached"])
    distance = float(row["final_goal_dist_m"])
    geodesic = float(row["geodesic_m"])
    path_len = float(row["path_len_m"])
    require(reached == int(distance < 1.0),
            f"{index}/{arm}: success-distance mismatch")
    spl = reached * geodesic / max(geodesic, path_len, 1e-12)
    return {
        "history_index": index,
        "scene": str(row["scene"]),
        "episode": str(row["episode"]),
        "query_id": str(row["query_id"]),
        "role": str(row["analysis_role"]),
        "arm": arm,
        "reached": reached,
        "geodesic_m": geodesic,
        "path_len_m": path_len,
        "final_distance_m": distance,
        "spl": spl,
    }


def load_records(run_root: Path, reference_root: Path,
                 manifest: dict, manifest_sha: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    posthoc = reference_root / "POSTHOC"
    require(sha256(posthoc / "final14_mono_factorial_summary.json")
            == REFERENCE_SUMMARY_SHA, "reference summary changed")
    verify_path = posthoc / "final14_mono_factorial_independent_verification.json"
    require(sha256(verify_path) == REFERENCE_VERIFY_SHA,
            "reference independent verification changed")
    verified = json.loads(verify_path.read_text())
    require(verified.get("verified") is True
            and verified.get("authorized") is True,
            "reference factorial is not independently authorized")

    for index, item in enumerate(manifest["episodes"]):
        scene, episode = str(item["scene"]), str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        zero_root = run_root / "evaluation/natural_direction" / label
        zero_completion_path = zero_root / "completion.json"
        verify_receipt(zero_completion_path)
        zero_completion = json.loads(zero_completion_path.read_text())
        require(zero_completion.get("schema_version") == EPISODE_SCHEMA
                and zero_completion.get("status") == "complete"
                and zero_completion.get("history_index") == index
                and zero_completion.get("scene") == scene
                and zero_completion.get("episode") == episode
                and zero_completion.get("benchmark_manifest_sha256")
                == manifest_sha
                and zero_completion.get(
                    "prefix_equality_to_verified_factorial") is True,
                f"{label}: zero-depth completion contract changed")

        reference = reference_root / "evaluation/natural_direction" / label
        reference_completion_sha = verify_receipt(
            reference / "completion.json"
        )
        require(zero_completion.get("reference_factorial_completion_sha256")
                == reference_completion_sha,
                f"{label}: reference factorial binding changed")

        zero_rows = read_metric(zero_root / ARM / "metric.csv")
        require(len(zero_rows) == 2
                and {row["analysis_role"] for row in zero_rows}
                == {"novel", "revisit"},
                f"{label}: zero-depth role population changed")
        for row in zero_rows:
            require(row["navdp_depth_source"] == DEPTH_SOURCE
                    and int(row["metric_depth_sensor_consumed_any"]) == 0
                    and int(row["monocular_receipt_plans"]) == 0,
                    f"{label}: zero-depth receipt contract changed")
            role = row["analysis_role"]
            require(int(row["reached"])
                    == int(zero_completion["outcomes"][role]),
                    f"{label}/{role}: completion outcome changed")
            records.append(record(index, ARM, row))

        for arm in REFERENCE_ARMS:
            rows = read_metric(reference / arm / "metric.csv")
            require(len(rows) == 2
                    and {row["analysis_role"] for row in rows}
                    == {"novel", "revisit"},
                    f"{label}/{arm}: reference role population changed")
            records.extend(record(index, arm, row) for row in rows)
    require(len(records) == len(manifest["episodes"]) * len(TABLE_ARMS) * 2,
            "unified depth table row count changed")
    return records


def arm_summary(records: list[dict[str, Any]], arm: str,
                role: str) -> dict[str, Any]:
    rows = [row for row in records
            if row["arm"] == arm and (role == "all" or row["role"] == role)]
    return {
        "n": len(rows),
        "successes": sum(row["reached"] for row in rows),
        "sr": sum(row["reached"] for row in rows) / len(rows),
        "mean_spl": sum(row["spl"] for row in rows) / len(rows),
        "mean_path_m": sum(row["path_len_m"] for row in rows) / len(rows),
        "mean_final_distance_m": (
            sum(row["final_distance_m"] for row in rows) / len(rows)),
    }


def statistical_rows(records: list[dict[str, Any]], role: str) -> list[dict]:
    selected = records if role == "all" else [
        row for row in records if row["role"] == role
    ]
    return [{
        "scene": row["scene"],
        "episode": (f"{row['episode']}/{row['role']}"
                    if role == "all" else row["episode"]),
        "arm": row["arm"],
        "reached": row["reached"],
    } for row in selected]


def summarize(run_root: Path, reference_root: Path, bench_root: Path,
              expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_path = bench_root / "manifest.json"
    require(sha256(manifest_path) == expected_manifest_sha256,
            "Final14 manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 21,
            "Final14 history population changed")
    records = load_records(
        run_root, reference_root, manifest, expected_manifest_sha256
    )
    results = {}
    for role in ("novel", "revisit", "all"):
        rows = statistical_rows(records, role)
        contrasts = {}
        for reference in ("metric_native", "mono_native"):
            key = f"{ARM}_minus_{reference}"
            value = paired_contrast(rows, ARM, reference)
            value["scene_cluster_bootstrap_95"] = scene_cluster_interval(
                rows, ARM, reference, seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            )
            contrasts[key] = value
        results[role] = {
            "arms": {
                arm: arm_summary(records, arm, role) for arm in TABLE_ARMS
            },
            "zero_depth_contrasts": contrasts,
        }
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": "consumed_final14_query_controller_depth_attribution",
        "fresh_confirmation": False,
        "benchmark_manifest_sha256": expected_manifest_sha256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA,
        "reference_independent_verification_sha256": REFERENCE_VERIFY_SHA,
        "histories": 21,
        "scene_count": len({row["scene"] for row in records}),
        "queries_per_arm": 42,
        "table_arms": list(TABLE_ARMS),
        "runtime_role_visibility": "none",
        "shared_history_policy": "original_metric_navdp_goal_a_rgb_replay",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "results": results,
        "interpretation_boundary": (
            "Same consumed Final14 RGB histories and paired queries; this "
            "isolates query-leg depth and does not relabel Goal-A as full mono."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.run_root, args.reference_root, args.bench_root,
        args.expected_manifest_sha256,
    )
    require(not args.out.exists(), "zero-depth summary already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_name(args.out.name + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + f"  {args.out.name}\n"
    )
    print(json.dumps(result["results"], sort_keys=True))


if __name__ == "__main__":
    main()
