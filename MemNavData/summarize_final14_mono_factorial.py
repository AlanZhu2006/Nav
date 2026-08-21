#!/usr/bin/env python3
"""Fail-closed summary for the consumed Final14 mono factorial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.final14_mono_factorial import (
    ARMS,
    DEPTH_SOURCE,
    PRIMARY_CONTRASTS,
    interaction_difference,
    paired_contrast,
    require,
    rotated_arm_order,
    scene_cluster_interval,
)


SUMMARY_SCHEMA = "final14_mono_factorial_summary_v1_20260819"
BOOTSTRAP_SEED = 2026081902
BOOTSTRAP_RESAMPLES = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(run_root: Path, manifest: dict[str, Any],
              manifest_sha: str) -> tuple[list[dict[str, Any]], list[dict]]:
    records: list[dict[str, Any]] = []
    completions: list[dict] = []
    for index, item in enumerate(manifest["episodes"]):
        scene = str(item["scene"])
        episode = str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "evaluation" / "natural_direction" / label
        completion_path = root / "completion.json"
        require(completion_path.is_file(), f"missing completion: {label}")
        receipt = (root / "completion.json.sha256").read_text().split()[0]
        require(sha256(completion_path) == receipt,
                f"completion hash mismatch: {label}")
        completion = json.loads(completion_path.read_text())
        require(completion.get("fresh_confirmation") is False,
                f"{label}: consumed scope was relabeled")
        require(completion.get("benchmark_manifest_sha256") == manifest_sha,
                f"{label}: manifest SHA changed")
        require(completion.get("arm_order") == list(rotated_arm_order(index)),
                f"{label}: arm order changed")
        require(completion.get("prefix_equality") is True,
                f"{label}: shared replay equality failed")
        require(int(completion.get("online_a_steps", -1)) >= 40,
                f"{label}: insufficient mono prefix")
        completions.append(completion)

        for arm in ARMS:
            metric_path = root / arm / "metric.csv"
            require(metric_path.is_file(), f"{label}/{arm}: metric missing")
            with metric_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2 and
                    {row["analysis_role"] for row in rows} ==
                    {"novel", "revisit"},
                    f"{label}/{arm}: role population changed")
            for row in rows:
                role = row["analysis_role"]
                reached = int(row["reached"])
                distance = float(row["final_goal_dist_m"])
                require(reached == int(distance < 1.0),
                        f"{label}/{arm}/{role}: success-distance mismatch")
                require(row["navdp_depth_source"] == DEPTH_SOURCE[arm],
                        f"{label}/{arm}/{role}: depth source mismatch")
                metric_consumed = int(row["metric_depth_sensor_consumed_any"])
                mono_receipts = int(row["monocular_receipt_plans"])
                mono_active = int(row["monocular_active_receipt_plans"])
                if DEPTH_SOURCE[arm] == "monocular_sidecar":
                    require(metric_consumed == 0,
                            f"{label}/{arm}/{role}: metric sensor consumed")
                    require(mono_receipts > 0 and mono_active == mono_receipts,
                            f"{label}/{arm}/{role}: mono receipts incomplete")
                    require(int(row["monocular_scale_receipt_hashes"]) == 1,
                            f"{label}/{arm}/{role}: scale receipt drift")
                else:
                    require(metric_consumed == 1 and mono_receipts == 0,
                            f"{label}/{arm}/{role}: metric receipt invalid")
                require(reached ==
                        int(completion["outcomes"][arm][role]),
                        f"{label}/{arm}/{role}: completion outcome mismatch")
                require(abs(distance - float(
                    completion["final_distance_m"][arm][role])) < 1e-9,
                    f"{label}/{arm}/{role}: completion distance mismatch")
                records.append({
                    "history_index": index,
                    "scene": scene,
                    "episode": episode,
                    "query_id": row["query_id"],
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "final_distance_m": distance,
                    "certificate_accept_plans": int(
                        row["certificate_accept_plans"]),
                    "adapter_takeover_plans": int(
                        row["adapter_takeover_plans"]),
                    "runtime_failure_plans": int(
                        row["runtime_failure_plans"]),
                    "metric_depth_sensor_consumed_any": metric_consumed,
                    "monocular_receipt_plans": mono_receipts,
                    "monocular_active_receipt_plans": mono_active,
                })
    expected = len(manifest["episodes"]) * len(ARMS) * 2
    require(len(records) == expected, "factorial result row count changed")
    return records, completions


def statistic_rows(records: list[dict[str, Any]],
                   role: str) -> list[dict[str, Any]]:
    selected = records if role == "all" else [
        row for row in records if row["role"] == role
    ]
    return [{
        "scene": row["scene"],
        "episode": (
            f"{row['episode']}/{row['role']}" if role == "all"
            else row["episode"]
        ),
        "arm": row["arm"],
        "reached": row["reached"],
    } for row in selected]


def arm_summary(records: list[dict[str, Any]], arm: str,
                role: str) -> dict[str, Any]:
    rows = [
        row for row in records
        if row["arm"] == arm and (role == "all" or row["role"] == role)
    ]
    return {
        "n": len(rows),
        "successes": sum(row["reached"] for row in rows),
        "sr": (sum(row["reached"] for row in rows) / len(rows)
               if rows else None),
        "mean_final_distance_m": (
            sum(row["final_distance_m"] for row in rows) / len(rows)
            if rows else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    bench_root = Path(args.bench_root)
    manifest_path = bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Final14 natural manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 21,
            "Final14 natural population changed")
    records, completions = load_rows(
        run_root, manifest, args.expected_manifest_sha256
    )

    role_summaries: dict[str, Any] = {}
    for role in ("novel", "revisit", "all"):
        rows = statistic_rows(records, role)
        contrasts: dict[str, Any] = {}
        for treatment, reference in PRIMARY_CONTRASTS:
            key = f"{treatment}_minus_{reference}"
            contrasts[key] = paired_contrast(rows, treatment, reference)
            contrasts[key]["scene_cluster_bootstrap_95"] = (
                scene_cluster_interval(
                    rows, treatment, reference,
                    seed=BOOTSTRAP_SEED,
                    resamples=BOOTSTRAP_RESAMPLES,
                )
            )
        role_summaries[role] = {
            "arms": {
                arm: arm_summary(records, arm, role) for arm in ARMS
            },
            "contrasts": contrasts,
            "interaction": interaction_difference(
                rows,
                seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            ),
        }

    cec_behavior: dict[str, Any] = {}
    for arm in ("mono_cec", "metric_cec"):
        cec_rows = [row for row in records if row["arm"] == arm]
        cec_behavior[arm] = {
            "query_count": len(cec_rows),
            "accepted_queries": sum(
                row["certificate_accept_plans"] > 0 for row in cec_rows
            ),
            "accepted_queries_by_role": {
                role: sum(
                    row["role"] == role and
                    row["certificate_accept_plans"] > 0
                    for row in cec_rows
                )
                for role in ("novel", "revisit")
            },
            "fully_rejected_exact_native_by_role": {
                role: sum(
                    completion["fully_rejected_exact_native"][arm][role]
                    for completion in completions
                )
                for role in ("novel", "revisit")
            },
            "runtime_failure_plans": sum(
                row["runtime_failure_plans"] for row in cec_rows
            ),
        }

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scope": "consumed_final14_query_controller_depth_attribution",
        "fresh_confirmation": False,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "histories": len(manifest["episodes"]),
        "scene_count": len({row["scene"] for row in records}),
        "queries_per_arm": 2 * len(manifest["episodes"]),
        "rows": len(records),
        "arms": list(ARMS),
        "runtime_role_visibility": "none",
        "shared_history_policy": "original_metric_navdp_goal_a_rgb_replay",
        "primary_estimands": [
            "mono_cec-minus-mono_native by role and balanced all",
            "mono_cec-minus-mono_raw_fixed by role and balanced all",
        ],
        "sensor_attribution_estimands": [
            "mono_native-minus-metric_native",
            "mono_cec-minus-metric_cec",
            "(mono_cec-mono_native)-(metric_cec-metric_native)",
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "results": role_summaries,
        "cec_behavior": cec_behavior,
        "depth_audit": {
            "mono_metric_sensor_consumed_queries": sum(
                row["metric_depth_sensor_consumed_any"]
                for row in records
                if DEPTH_SOURCE[row["arm"]] == "monocular_sidecar"
            ),
            "mono_queries_with_active_receipts": sum(
                row["monocular_active_receipt_plans"] > 0
                for row in records
                if DEPTH_SOURCE[row["arm"]] == "monocular_sidecar"
            ),
            "mono_query_count": sum(
                DEPTH_SOURCE[row["arm"]] == "monocular_sidecar"
                for row in records
            ),
        },
        "interpretation_boundary": (
            "Same consumed Final14 RGB histories and queries; query-controller "
            "depth attribution only. Not a fresh or fully-mono Goal-A "
            "confirmation."
        ),
    }
    out = Path(args.out)
    require(not out.exists(), "summary output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
