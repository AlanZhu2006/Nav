#!/usr/bin/env python3
"""Fail-closed summary for the paired Final14 CEC authority ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.final14_authority_ablation import (
    ARMS,
    AUTHORITY_POLICY,
    DEPTH_SOURCE,
    paired_contrast,
    require,
    rotated_arm_order,
    scene_cluster_interval,
)


SUMMARY_SCHEMA = "final14_cec_authority_ablation_summary_v1_20260828"
BOOTSTRAP_SEED = 2026082801
BOOTSTRAP_RESAMPLES = 100_000
TREATMENT = "mono_cec"
REFERENCE = "mono_unthresholded_witness"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(
        run_root: Path, manifest: dict[str, Any], manifest_sha: str,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["episodes"]):
        scene = str(item["scene"])
        episode = str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "evaluation" / "natural_direction" / label
        completion_path = root / "completion.json"
        hash_path = root / "completion.json.sha256"
        require(completion_path.is_file() and hash_path.is_file(),
                f"{label}: completion receipt missing")
        require(sha256(completion_path) == hash_path.read_text().split()[0],
                f"{label}: completion hash mismatch")
        completion = json.loads(completion_path.read_text())
        require(completion.get("benchmark_manifest_sha256") == manifest_sha,
                f"{label}: benchmark manifest changed")
        require(completion.get("arm_order") == list(rotated_arm_order(index)),
                f"{label}: arm order changed")
        require(completion.get("prefix_equality") is True,
                f"{label}: shared Goal-A replay changed")
        require(completion.get("initial_proposal_equality") is True,
                f"{label}: proposal pairing failed")
        require(completion.get("runtime_role_visibility") == "none",
                f"{label}: role leaked into runtime")
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
                require(row["navdp_depth_source"] == DEPTH_SOURCE,
                        f"{label}/{arm}/{role}: depth arm changed")
                require(int(row["metric_depth_sensor_consumed_any"]) == 0,
                        f"{label}/{arm}/{role}: metric depth consumed")
                mono_plans = int(row["monocular_receipt_plans"])
                require(mono_plans > 0 and
                        int(row["monocular_active_receipt_plans"]) == mono_plans,
                        f"{label}/{arm}/{role}: mono receipts incomplete")
                require(int(row["monocular_scale_receipt_hashes"]) == 1,
                        f"{label}/{arm}/{role}: mono scale drift")
                require(int(row["runtime_failure_plans"]) == 0,
                        f"{label}/{arm}/{role}: runtime failure")
                require(reached == int(completion["outcomes"][arm][role]),
                        f"{label}/{arm}/{role}: completion outcome mismatch")
                initial = completion["initial_proposal_audit"][role]
                require(initial.get("proposal_fields_equal") is True,
                        f"{label}/{role}: proposal fields changed")
                accepted = int(row["certificate_accept_plans"]) > 0
                require(accepted == bool(
                    initial[
                        "strict_accepted" if arm == "mono_cec"
                        else "unthresholded_witness_accepted"]),
                    f"{label}/{arm}/{role}: initial/cached authority mismatch")
                records.append({
                    "history_index": index,
                    "scene": scene,
                    "episode": episode,
                    "query_id": row["query_id"],
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "final_distance_m": distance,
                    "accepted": int(accepted),
                    "accept_plans": int(row["certificate_accept_plans"]),
                })
    require(len(records) == len(manifest["episodes"]) * len(ARMS) * 2,
            "authority ablation row count changed")
    return records, completions


def statistic_rows(records: list[dict[str, Any]], role: str) -> list[dict]:
    selected = records if role == "all" else [
        row for row in records if row["role"] == role]
    return [{
        "scene": row["scene"],
        "episode": (
            f"{row['episode']}/{row['role']}"
            if role == "all" else row["episode"]),
        "arm": row["arm"],
        "reached": row["reached"],
    } for row in selected]


def arm_summary(records: list[dict[str, Any]], arm: str,
                role: str) -> dict[str, Any]:
    rows = [row for row in records
            if row["arm"] == arm and
            (role == "all" or row["role"] == role)]
    return {
        "n": len(rows),
        "successes": sum(row["reached"] for row in rows),
        "sr": sum(row["reached"] for row in rows) / len(rows),
        "accepted_queries": sum(row["accepted"] for row in rows),
        "accept_rate": sum(row["accepted"] for row in rows) / len(rows),
        "mean_final_distance_m": (
            sum(row["final_distance_m"] for row in rows) / len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest_path = Path(args.bench_root) / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Final14 natural manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 21,
            "Final14 natural population changed")
    records, completions = load_records(
        run_root, manifest, args.expected_manifest_sha256)

    results: dict[str, Any] = {}
    for role in ("novel", "revisit", "all"):
        rows = statistic_rows(records, role)
        contrast = paired_contrast(rows, TREATMENT, REFERENCE)
        contrast["scene_cluster_bootstrap_95"] = scene_cluster_interval(
            rows, TREATMENT, REFERENCE,
            seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES)
        results[role] = {
            "arms": {
                arm: arm_summary(records, arm, role) for arm in ARMS},
            "strict_cec_minus_unthresholded_witness": contrast,
        }

    discordance = {
        role: sum(
            completion["initial_proposal_audit"][role][
                "authority_discordant"]
            for completion in completions)
        for role in ("novel", "revisit")
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scope": "consumed_final14_cec_authority_only_ablation",
        "fresh_confirmation": False,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "histories": len(manifest["episodes"]),
        "scene_count": len({row["scene"] for row in records}),
        "queries_per_arm": 2 * len(manifest["episodes"]),
        "arms": list(ARMS),
        "authority_policies": AUTHORITY_POLICY,
        "runtime_role_visibility": "none",
        "shared_history_policy": "original_metric_navdp_goal_a_rgb_replay",
        "query_controller_depth": DEPTH_SOURCE,
        "initial_proposal_equality_histories": len(completions),
        "authority_discordant_queries_by_role": discordance,
        "results": results,
        "table_fields": {
            arm: {
                "novel_success": results["novel"]["arms"][arm]["successes"],
                "revisit_success": results["revisit"]["arms"][arm]["successes"],
                "all_success": results["all"]["arms"][arm]["successes"],
                "novel_false_accept": results["novel"]["arms"][arm][
                    "accepted_queries"],
                "revisit_false_reject": (
                    21 - results["revisit"]["arms"][arm]["accepted_queries"]),
            }
            for arm in ARMS
        },
        "interpretation_boundary": (
            "This is a closed-loop operational-authority ablation. The "
            "unthresholded arm still uses DINO retrieval, local geometry, "
            "LingBot historical depth, and PnP; it is not retrieval-only and "
            "not geometry-free."),
    }
    out = Path(args.out)
    require(not out.exists(), "summary output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
