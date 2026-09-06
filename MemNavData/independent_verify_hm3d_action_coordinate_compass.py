#!/usr/bin/env python3
"""Dependency-light verification of the action-coordinate aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


EPISODE_SCHEMA = "hm3d_action_coordinate_compass_pair_v1_20260902"
SUMMARY_SCHEMA = "hm3d_action_coordinate_compass_result_v1_20260902"
VERIFY_SCHEMA = "hm3d_action_coordinate_compass_verification_v1_20260902"
ARMS = ("mono_cec_endpoint", "mono_cec_action_coordinate")
ACTION = ARMS[1]
ENDPOINT = ARMS[0]
ROLES = ("novel", "revisit")


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


def compare_float(left: float, right: float, name: str) -> None:
    require(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12),
            f"aggregate float mismatch: {name}")


def verify_contrast(
    reported: dict,
    rows: list[dict],
    *,
    label: str,
) -> None:
    gains = sum(row[ACTION] == 1 and row[ENDPOINT] == 0 for row in rows)
    losses = sum(row[ACTION] == 0 and row[ENDPOINT] == 1 for row in rows)
    expected = {
        "episodes": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "left_successes": sum(row[ACTION] for row in rows),
        "right_successes": sum(row[ENDPOINT] for row in rows),
        "paired_gains": gains,
        "paired_losses": losses,
        "paired_net_gain": gains - losses,
    }
    for key, value in expected.items():
        require(int(reported[key]) == int(value),
                f"{label}: {key} mismatch")
    compare_float(
        reported["left_SR"], expected["left_successes"] / len(rows),
        f"{label}/left_SR")
    compare_float(
        reported["right_SR"], expected["right_successes"] / len(rows),
        f"{label}/right_SR")
    compare_float(
        reported["mcnemar_exact_two_sided_p"], mcnemar(gains, losses),
        f"{label}/mcnemar")
    interval = reported.get(
        "scene_cluster_bootstrap_risk_difference_pp_95ci")
    require(isinstance(interval, list) and len(interval) == 2
            and all(math.isfinite(float(value)) for value in interval)
            and float(interval[0]) <= float(interval[1]),
            f"{label}: invalid cluster interval")


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
    summary_sidecar = arguments.summary.with_name(
        arguments.summary.name + ".sha256")
    require(summary_sidecar.is_file()
            and summary_sidecar.read_text().split()
            == [sha256(arguments.summary), arguments.summary.name],
            "aggregate summary receipt is invalid")
    manifest = json.loads(arguments.manifest.read_text())
    summary = json.loads(arguments.summary.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "aggregate schema changed")
    require(summary.get("benchmark_manifest_sha256")
            == arguments.expected_manifest_sha256,
            "aggregate is bound to another manifest")
    require(summary.get("arms") == list(ARMS), "aggregate arms changed")
    require(summary.get("partial_results_reported") is False
            and summary.get("fallback_completion_used") is False,
            "aggregate admitted partial or fallback output")
    require(summary.get("post_hoc_pass_threshold_defined") is False,
            "aggregate introduced a post-hoc pass threshold")

    rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    for index, item in enumerate(manifest["episodes"]):
        root = (arguments.run_root / "development" /
                "action_coordinate_compass" /
                f"{index:03d}_{item['scene']}_{item['episode']}")
        path = root / "completion.json"
        digest = sha256(path)
        sidecar = path.with_name(path.name + ".sha256")
        require(sidecar.is_file()
                and sidecar.read_text().split() == [digest, path.name],
                f"completion receipt changed at index {index}")
        relative = str(path.relative_to(arguments.run_root))
        source_hashes[relative] = digest
        require(summary["completion_sha256"].get(relative) == digest,
                f"aggregate source hash mismatch at index {index}")
        completion = json.loads(path.read_text())
        require(completion.get("schema_version") == EPISODE_SCHEMA
                and completion.get("history_index") == index
                and completion.get("scene") == item["scene"]
                and completion.get("episode") == item["episode"]
                and completion.get("arms") == list(ARMS),
                f"completion identity changed at index {index}")
        require(completion.get("prefix_equality") is True
                and completion.get("target_proof_equality") is True
                and completion.get("runtime_role_visibility") == "none",
                f"pairing or role-hiding failed at index {index}")
        require(completion.get("metric_scale_consumed_by_route") is False
                and completion.get("visual_gate_after_authorization") is False
                and completion.get("distance_regime_present") is False
                and completion.get("stuck_trigger_within_budget") is False
                and completion.get(
                    "endpoint_or_native_fallback_after_authorization") is False,
                f"forbidden route mechanism at index {index}")
        for role in ROLES:
            route = completion["runtime_audits"][ACTION][role]["route"]
            if route is not None:
                require(route["progress_finite_monotone"] is True
                        and route["unit_bearing_norm_verified"] is True
                        and route["fixed_controller_radius_m"] == 2.5
                        and route["visual_gate_after_authorization"] is False
                        and route["distance_regime_present"] is False
                        and route["endpoint_fallback_available"] is False
                        and route[
                            "native_fallback_after_authorization_available"]
                        is False
                        and route["evaluator_pose_consumed"] is False
                        and route["habitat_path_consumed"] is False
                        and route["metric_scale_consumed"] is False,
                        f"action-coordinate audit failed at {index}/{role}")
            rows.append({
                "population_index": index,
                "scene": item["scene"],
                "episode": item["episode"],
                "bin_name": item["bin_name"],
                "role": role,
                ENDPOINT: int(completion["outcomes"][ENDPOINT][role]),
                ACTION: int(completion["outcomes"][ACTION][role]),
            })

    require(len(rows) == 96 and len(source_hashes) == 48,
            "raw completion population is incomplete")
    reported_records = summary.get("records")
    require(isinstance(reported_records, list) and len(reported_records) == 96,
            "aggregate record population changed")
    by_key = {
        (row["population_index"], row["role"]): row
        for row in reported_records
    }
    for row in rows:
        reported = by_key.get((row["population_index"], row["role"]))
        require(reported is not None, "aggregate omitted a raw role row")
        for key in ("scene", "episode", "bin_name", ENDPOINT, ACTION):
            require(reported[key] == row[key],
                    f"aggregate raw record mismatch: {key}")

    revisit = [row for row in rows if row["role"] == "revisit"]
    novel = [row for row in rows if row["role"] == "novel"]
    primary = summary["primary_action_coordinate_vs_endpoint_revisit"]
    verify_contrast(primary["all"], revisit, label="revisit/all")
    for name in ("0_to_20_m", "20_to_30_m", "30_to_50_m"):
        subset = [row for row in revisit if row["bin_name"] == name]
        require(len(subset) == 16, f"{name}: stratum population changed")
        verify_contrast(primary[name], subset, label=f"revisit/{name}")
    verify_contrast(
        summary["novel_safety_action_coordinate_vs_endpoint"],
        novel,
        label="novel/all",
    )

    result = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "summary_sha256": sha256(arguments.summary),
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "histories_verified": 48,
        "hidden_role_queries_verified": 96,
        "completion_hashes_verified": source_hashes,
        "paired_outcomes_recomputed": True,
        "mcnemar_recomputed": True,
        "route_information_boundary_verified": True,
        "partial_results_used": False,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(arguments.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
    arguments.out.with_name(arguments.out.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {arguments.out.name}\n")
    print(json.dumps({
        "verified": True,
        "summary_sha256": result["summary_sha256"],
        "out": str(arguments.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
