#!/usr/bin/env python3
"""Dependency-light verification of the long-range attribution aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


EPISODE_SCHEMA = "hm3d_longrange_oracle_attribution_episode_v1_20260903"
SUMMARY_SCHEMA = "hm3d_longrange_oracle_attribution_result_v1_20260903"
VERIFY_SCHEMA = "hm3d_longrange_oracle_attribution_verification_v1_20260903"
INDICES = tuple(range(32, 48))
ARMS = (
    "action_coordinate_mixed",
    "oracle_route_mixed",
    "oracle_geodesic_mixed",
    "oracle_geodesic_point",
)
CONTRASTS = {
    "oracle_route_vs_action_coordinate": (ARMS[1], ARMS[0]),
    "oracle_geodesic_mixed_vs_action_coordinate": (ARMS[2], ARMS[0]),
    "oracle_geodesic_mixed_vs_oracle_route": (ARMS[2], ARMS[1]),
    "oracle_geodesic_point_vs_oracle_geodesic_mixed": (ARMS[3], ARMS[2]),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gains, losses) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def verify_contrast(reported: dict, rows: list[dict], left: str,
                    right: str, label: str) -> None:
    gains = sum(row[left] == 1 and row[right] == 0 for row in rows)
    losses = sum(row[left] == 0 and row[right] == 1 for row in rows)
    expected = {
        "episodes": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "left_successes": sum(row[left] for row in rows),
        "right_successes": sum(row[right] for row in rows),
        "paired_gains": gains,
        "paired_losses": losses,
        "paired_net_gain": gains - losses,
    }
    for field, value in expected.items():
        require(int(reported[field]) == int(value),
                f"{label}: {field} mismatch")
    require(math.isclose(float(reported["left_SR"]),
                         expected["left_successes"] / len(rows),
                         abs_tol=1e-12, rel_tol=0.0)
            and math.isclose(float(reported["right_SR"]),
                             expected["right_successes"] / len(rows),
                             abs_tol=1e-12, rel_tol=0.0)
            and math.isclose(float(reported["mcnemar_exact_two_sided_p"]),
                             mcnemar(gains, losses),
                             abs_tol=1e-12, rel_tol=0.0),
            f"{label}: rates/statistic mismatch")
    interval = reported.get(
        "scene_cluster_bootstrap_risk_difference_pp_95ci")
    require(isinstance(interval, list) and len(interval) == 2
            and all(math.isfinite(float(value)) for value in interval),
            f"{label}: cluster interval invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "verification output already exists")
    require(sha256(arguments.manifest)
            == arguments.expected_manifest_sha256,
            "sealed manifest changed")
    require(arguments.summary.is_file(), "aggregate summary is missing")
    sidecar = arguments.summary.with_name(arguments.summary.name + ".sha256")
    require(sidecar.is_file() and sidecar.read_text().split()
            == [sha256(arguments.summary), arguments.summary.name],
            "aggregate summary receipt is invalid")
    manifest = json.loads(arguments.manifest.read_text())
    summary = json.loads(arguments.summary.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA
            and summary.get("benchmark_manifest_sha256")
            == arguments.expected_manifest_sha256
            and summary.get("arms") == list(ARMS)
            and summary.get("paper_result_eligible") is False
            and summary.get("partial_results_reported") is False
            and summary.get("fallback_completion_used") is False,
            "aggregate identity/scope changed")

    rows = []
    source_hashes = {}
    for index in INDICES:
        item = manifest["episodes"][index]
        require(item["bin_name"] == "30_to_50_m",
                "verifier index escaped long-range stratum")
        path = (arguments.run_root / "development" /
                "longrange_oracle_attribution" /
                f"{index:03d}_{item['scene']}_{item['episode']}" /
                "completion.json")
        digest = sha256(path)
        require(path.with_name(path.name + ".sha256").read_text().split()
                == [digest, path.name],
                f"completion sidecar changed at {index}")
        relative = str(path.relative_to(arguments.run_root))
        require(summary["completion_sha256"].get(relative) == digest,
                f"aggregate source hash changed at {index}")
        source_hashes[relative] = digest
        completion = json.loads(path.read_text())
        require(completion.get("schema_version") == EPISODE_SCHEMA
                and completion.get("history_index") == index
                and completion.get("arms") == list(ARMS)
                and completion.get("paper_result_eligible") is False
                and completion.get("prefix_equality") is True
                and completion.get("target_proof_equality") is True,
                f"completion contract changed at {index}")
        for arm in ARMS[1:]:
            audit = completion["runtime_audits"][arm]["route"]
            require(audit["evaluator_pose_consumed"] is True
                    and audit["fixed_controller_radius_m"] == 2.5,
                    f"oracle disclosure changed at {index}/{arm}")
        rows.append({
            "population_index": index,
            "scene": item["scene"],
            "episode": item["episode"],
            **{arm: int(completion["outcomes"][arm]) for arm in ARMS},
        })
    require(len(rows) == 16 and len(source_hashes) == 16,
            "completion population is incomplete")
    reported_records = summary.get("records")
    require(isinstance(reported_records, list) and len(reported_records) == 16,
            "aggregate record population changed")
    reported = {row["population_index"]: row for row in reported_records}
    for row in rows:
        target = reported.get(row["population_index"])
        require(target is not None, "aggregate omitted one population index")
        for field in ("scene", "episode", *ARMS):
            require(target[field] == row[field],
                    f"aggregate raw row mismatch: {field}")
    for label, (left, right) in CONTRASTS.items():
        verify_contrast(summary["contrasts"][label], rows, left, right, label)

    result = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "paper_result_eligible": False,
        "summary_sha256": sha256(arguments.summary),
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "histories_verified": 16,
        "completion_hashes_verified": source_hashes,
        "paired_outcomes_recomputed": True,
        "mcnemar_recomputed": True,
        "privileged_attribution_disclosure_verified": True,
        "partial_results_used": False,
    }
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(arguments.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
    arguments.out.with_name(arguments.out.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {arguments.out.name}\n")
    print(json.dumps({"verified": True, "out": str(arguments.out)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
