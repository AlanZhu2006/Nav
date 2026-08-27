#!/usr/bin/env python3
"""Independent raw-receipt verifier for the Final14 authority ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "final14_cec_authority_ablation_independent_verification_v1_20260828"
ARMS = ("mono_cec", "mono_unthresholded_witness")
POLICIES = {
    "mono_cec": "strict_certificate",
    "mono_unthresholded_witness": "pnp_pose_available",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(
        min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest_path = Path(args.bench_root) / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 21, "history population changed")
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    require(summary.get("status") == "complete", "summary is incomplete")
    require(summary.get("benchmark_manifest_sha256") ==
            args.expected_manifest_sha256, "summary manifest changed")

    raw: dict[tuple[int, str, str], dict[str, Any]] = {}
    proposal_pairs = 0
    source_files: list[Path] = [manifest_path, summary_path]
    for index, item in enumerate(manifest["episodes"]):
        scene = str(item["scene"])
        episode = str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "evaluation" / "natural_direction" / label
        completion_path = root / "completion.json"
        completion_hash = root / "completion.json.sha256"
        source_files.extend([completion_path, completion_hash])
        require(sha256(completion_path) ==
                completion_hash.read_text().split()[0],
                f"{label}: completion hash mismatch")
        completion = json.loads(completion_path.read_text())
        require(completion.get("prefix_equality") is True,
                f"{label}: prefix mismatch")
        require(completion.get("initial_proposal_equality") is True,
                f"{label}: proposal mismatch")
        expected_order = list(ARMS[index % 2:] + ARMS[:index % 2])
        require(completion.get("arm_order") == expected_order,
                f"{label}: arm order mismatch")

        plans_by_arm_role: dict[tuple[str, str], dict] = {}
        for arm in ARMS:
            metric_path = root / arm / "metric.csv"
            source_files.append(metric_path)
            with metric_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2, f"{label}/{arm}: wrong query count")
            for row in rows:
                role = row["analysis_role"]
                require(role in ("novel", "revisit"),
                        f"{label}/{arm}: invalid role")
                reached = int(row["reached"])
                distance = float(row["final_goal_dist_m"])
                require(reached == int(distance < 1.0),
                        f"{label}/{arm}/{role}: SR mismatch")
                require(row["navdp_depth_source"] == "monocular_sidecar",
                        f"{label}/{arm}/{role}: depth source mismatch")
                require(int(row["metric_depth_sensor_consumed_any"]) == 0,
                        f"{label}/{arm}/{role}: sensor depth consumed")
                require(int(row["runtime_failure_plans"]) == 0,
                        f"{label}/{arm}/{role}: runtime failure")
                plans_path = root / arm / f"{episode}_{row['query_id']}_plans.json"
                source_files.append(plans_path)
                payload = json.loads(plans_path.read_text())
                require(payload.get("analysis_role_not_forwarded") is True,
                        f"{label}/{arm}/{role}: role was forwarded")
                plans = payload.get("query_leg")
                require(isinstance(plans, list) and plans,
                        f"{label}/{arm}/{role}: plans missing")
                first = plans[0]
                require(first.get(
                    "certified_relocalization_authority_policy") ==
                    POLICIES[arm],
                    f"{label}/{arm}/{role}: policy mismatch")
                require(first.get("certified_relocalization_proposal_order") ==
                        "geometry_first",
                        f"{label}/{arm}/{role}: proposal order mismatch")
                accepted = int(row["certificate_accept_plans"]) > 0
                require(accepted == bool(first.get(
                    "certified_relocalization_accepted") is True),
                    f"{label}/{arm}/{role}: cached authority mismatch")
                raw[(index, role, arm)] = {
                    "scene": scene,
                    "episode": episode,
                    "reached": reached,
                    "accepted": int(accepted),
                }
                plans_by_arm_role[(arm, role)] = first

        for role in ("novel", "revisit"):
            strict = plans_by_arm_role[("mono_cec", role)]
            witness = plans_by_arm_role[("mono_unthresholded_witness", role)]
            for field in (
                "router_candidate_order_dino",
                "router_candidate_order_used",
                "router_selected_anchor",
                "router_selected_candidate_dino_rank",
            ):
                require(strict.get(field) == witness.get(field),
                        f"{label}/{role}: proposal field {field} differs")
            require(not (
                strict.get("certified_relocalization_accepted") is True and
                witness.get("certified_relocalization_accepted") is not True),
                f"{label}/{role}: authority monotonicity failed")
            proposal_pairs += 1

    require(len(raw) == 21 * 2 * 2, "raw query count mismatch")
    recomputed: dict[str, Any] = {}
    for role in ("novel", "revisit", "all"):
        indices = [(index, selected_role)
                   for index in range(21)
                   for selected_role in ("novel", "revisit")
                   if role == "all" or role == selected_role]
        arms = {}
        for arm in ARMS:
            rows = [raw[(index, selected_role, arm)]
                    for index, selected_role in indices]
            arms[arm] = {
                "n": len(rows),
                "successes": sum(row["reached"] for row in rows),
                "accepted_queries": sum(row["accepted"] for row in rows),
            }
        gains = losses = 0
        for index, selected_role in indices:
            strict = raw[(index, selected_role, "mono_cec")]["reached"]
            witness = raw[(index, selected_role,
                           "mono_unthresholded_witness")]["reached"]
            gains += strict == 1 and witness == 0
            losses += strict == 0 and witness == 1
        recomputed[role] = {
            "arms": arms,
            "strict_minus_unthresholded": {
                "gains": gains,
                "losses": losses,
                "risk_difference_pp": 100.0 * (gains - losses) / len(indices),
                "exact_mcnemar_two_sided_p": exact_mcnemar(gains, losses),
            },
        }
        reported = summary["results"][role]
        for arm in ARMS:
            for field in ("n", "successes", "accepted_queries"):
                require(reported["arms"][arm][field] ==
                        arms[arm][field],
                        f"summary {role}/{arm}/{field} mismatch")
        contrast = reported["strict_cec_minus_unthresholded_witness"]
        for field in ("gains", "losses", "risk_difference_pp",
                      "exact_mcnemar_two_sided_p"):
            require(abs(float(contrast[field]) - float(
                recomputed[role]["strict_minus_unthresholded"][field])) < 1e-12,
                f"summary {role} contrast {field} mismatch")

    receipt = {
        "schema_version": SCHEMA,
        "verified": True,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "summary_sha256": sha256(summary_path),
        "histories": 21,
        "queries_per_arm": 42,
        "proposal_pairs_verified": proposal_pairs,
        "runtime_role_visibility": "none",
        "metric_depth_sensor_reads": 0,
        "recomputed": recomputed,
        "source_file_count": len(source_files),
        "source_digest_sha256": hashlib.sha256("".join(
            f"{sha256(path)}  {path}\n" for path in sorted(
                source_files, key=str)).encode()).hexdigest(),
    }
    out = Path(args.out)
    require(not out.exists(), "verification output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
