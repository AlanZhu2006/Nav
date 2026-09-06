#!/usr/bin/env python3
"""Fail-closed summary of the sealed HM3D Table-I authority spectrum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.hm3d_table1_authority_spectrum import (
    ARMS,
    AUTHORITY_POLICY,
    DEPTH_SOURCE,
    PRIMARY_CONTRASTS,
    selected_arm_order,
)
from MemNavData.mdtec_raw_depth_gate_d import (
    paired_contrast,
    require,
    scene_cluster_interval,
)


SCHEMA = "hm3d_table1_authority_spectrum_summary_v1_20260904"
BOOTSTRAP_SEED = 2026090401
BOOTSTRAP_RESAMPLES = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(
    run_root: Path, manifest: dict[str, Any], manifest_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["episodes"]):
        label = f'{index:03d}_{item["scene"]}_{item["episode"]}'
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
        require(completion.get("arm_order") ==
                list(selected_arm_order(index, ARMS)),
                f"{label}: arm order changed")
        require(completion.get("prefix_equality") is True,
                f"{label}: shared history replay changed")
        require(completion.get("initial_proposal_equality") is True,
                f"{label}: strict/witness proposal mismatch")
        require(completion.get("runtime_role_visibility") == "none",
                f"{label}: role leaked into runtime")
        require(completion.get("fresh_confirmation") is False,
                f"{label}: retrospective scope mislabeled")
        require(completion.get("outcomes_known_before_ablation_design") ==
                ["mono_native", "mono_cec"],
                f"{label}: prior-outcome disclosure changed")
        for role in ("novel", "revisit"):
            proposal = completion["initial_proposal_audit"][role]
            require(proposal.get("proposal_fields_equal") is True,
                    f"{label}/{role}: matched proposal audit failed")

        for arm in ARMS:
            metric_path = root / arm / "metric.csv"
            require(metric_path.is_file(), f"{label}/{arm}: metric missing")
            with metric_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2 and
                    {row["analysis_role"] for row in rows} ==
                    {"novel", "revisit"},
                    f"{label}/{arm}: query population changed")
            for row in rows:
                role = row["analysis_role"]
                reached = int(row["reached"])
                distance = float(row["final_goal_dist_m"])
                require(reached == int(distance < 1.0),
                        f"{label}/{arm}/{role}: success mismatch")
                require(row["navdp_depth_source"] == DEPTH_SOURCE[arm],
                        f"{label}/{arm}/{role}: depth source changed")
                require(int(row["metric_depth_sensor_consumed_any"]) == 0,
                        f"{label}/{arm}/{role}: metric depth consumed")
                require(int(row["runtime_failure_plans"]) == 0,
                        f"{label}/{arm}/{role}: runtime failure")
                require(reached ==
                        int(completion["outcomes"][arm][role]),
                        f"{label}/{arm}/{role}: outcome receipt mismatch")
                rows_out.append({
                    "history_index": index,
                    "scene": str(item["scene"]),
                    "episode": str(item["episode"]),
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "final_distance_m": distance,
                    "accepted": int(row["certificate_accept_plans"]) > 0,
                    "accept_plans": int(row["certificate_accept_plans"]),
                })
        completions.append(completion)
    require(len(rows_out) == len(manifest["episodes"]) * len(ARMS) * 2,
            "authority-spectrum row count changed")
    return rows_out, completions


def _selected(records: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return records if role == "all" else [
        row for row in records if row["role"] == role]


def _arm_summary(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [row for row in records if row["arm"] == arm]
    return {
        "n": len(rows),
        "successes": sum(row["reached"] for row in rows),
        "sr": sum(row["reached"] for row in rows) / len(rows),
        "authorized_queries": sum(row["accepted"] for row in rows),
        "authorization_rate": sum(row["accepted"] for row in rows) / len(rows),
        "mean_final_distance_m": (
            sum(row["final_distance_m"] for row in rows) / len(rows)
        ),
    }


def _contrast_rows(records: list[dict[str, Any]], role: str) -> list[dict]:
    return [{
        "scene": row["scene"],
        "episode": (
            f'{row["episode"]}/{row["role"]}'
            if role == "all" else row["episode"]
        ),
        "arm": row["arm"],
        "reached": row["reached"],
    } for row in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-histories", type=int, default=28)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == args.expected_histories,
            "frozen history denominator changed")
    records, completions = _records(
        args.run_root, manifest, args.expected_manifest_sha256
    )

    results = {}
    for role in ("novel", "revisit", "all"):
        selected = _selected(records, role)
        contrast_input = _contrast_rows(selected, role)
        contrasts = {}
        for treatment, reference in PRIMARY_CONTRASTS:
            result = paired_contrast(contrast_input, treatment, reference)
            result["scene_cluster_bootstrap_95"] = scene_cluster_interval(
                contrast_input, treatment, reference,
                seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES,
            )
            contrasts[f"{treatment}_minus_{reference}"] = result
        results[role] = {
            "arms": {arm: _arm_summary(selected, arm) for arm in ARMS},
            "contrasts": contrasts,
        }

    discordance = {
        role: sum(bool(row["initial_proposal_audit"][role]
                       ["authority_discordant"])
                  for row in completions)
        for role in ("novel", "revisit")
    }
    payload = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": "sealed_hm3d_table1_retrospective_authority_spectrum",
        "fresh_confirmation": False,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "histories": len(manifest["episodes"]),
        "scene_clusters": len({row["scene"] for row in records}),
        "queries_per_arm": 2 * len(manifest["episodes"]),
        "arms": list(ARMS),
        "authority_policies": AUTHORITY_POLICY,
        "runtime_role_visibility": "none",
        "query_controller_depth": "monocular_sidecar",
        "outcomes_known_before_ablation_design": ["mono_native", "mono_cec"],
        "raw_and_witness_outcomes_known_before_submission": False,
        "strict_witness_proposal_pairs_verified": 2 * len(completions),
        "strict_witness_authority_discordance": discordance,
        "results": results,
        "interpretation_boundary": (
            "The fixed query population was selected without outcomes, but "
            "native and strict-CEC outcomes were known before this four-arm "
            "ablation was designed. Results are therefore retrospective "
            "causal ablations, not fresh confirmation."
        ),
    }
    require(not args.out.exists(), "summary output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
