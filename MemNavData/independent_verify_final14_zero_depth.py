#!/usr/bin/env python3
"""Independent raw-file verifier for the Final14 zero-depth row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from MemNavData.final14_mono_factorial import exact_mcnemar_two_sided, require
from MemNavData.final14_zero_depth import ARM, audit_zero_depth_plans
from MemNavData.run_final14_mono_factorial_episode import sha256
from MemNavData.summarize_final14_zero_depth import (
    EPISODE_SCHEMA,
    REFERENCE_SUMMARY_SHA,
    REFERENCE_VERIFY_SHA,
    SCHEMA,
)


VERIFY_SCHEMA = "independent_final14_zero_depth_verification_v1_20260828"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify(run_root: Path, reference_root: Path, bench_root: Path,
           expected_manifest_sha256: str, summary_path: Path) -> dict:
    manifest_path = bench_root / "manifest.json"
    require(sha256(manifest_path) == expected_manifest_sha256,
            "benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 21, "history count changed")
    require(sha256(reference_root / "POSTHOC/final14_mono_factorial_summary.json")
            == REFERENCE_SUMMARY_SHA, "reference summary changed")
    reference_verify = (reference_root /
        "POSTHOC/final14_mono_factorial_independent_verification.json")
    require(sha256(reference_verify) == REFERENCE_VERIFY_SHA,
            "reference verifier changed")
    reference_verified = json.loads(reference_verify.read_text())
    require(reference_verified.get("verified") is True
            and reference_verified.get("authorized") is True,
            "reference verifier did not authorize the population")

    counts = {"novel": 0, "revisit": 0}
    paired = {
        role: {"zero": [], "metric": [], "mono": []}
        for role in ("novel", "revisit")
    }
    plan_count = 0
    for index, item in enumerate(manifest["episodes"]):
        scene, episode = str(item["scene"]), str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "evaluation/natural_direction" / label
        completion_path = root / "completion.json"
        sidecar = completion_path.with_name("completion.json.sha256")
        require(sidecar.read_text().split()
                == [sha256(completion_path), "completion.json"],
                f"{label}: completion receipt changed")
        completion = json.loads(completion_path.read_text())
        require(completion.get("schema_version") == EPISODE_SCHEMA
                and completion.get("status") == "complete"
                and completion.get("history_index") == index
                and completion.get("benchmark_manifest_sha256")
                == expected_manifest_sha256
                and completion.get("runtime_role_visibility") == "none",
                f"{label}: completion contract changed")
        zero_rows = rows(root / ARM / "metric.csv")
        reference = reference_root / "evaluation/natural_direction" / label
        metric_rows = {row["analysis_role"]: row
                       for row in rows(reference / "metric_native/metric.csv")}
        mono_rows = {row["analysis_role"]: row
                     for row in rows(reference / "mono_native/metric.csv")}
        require(len(zero_rows) == 2, f"{label}: zero row count changed")
        for row in zero_rows:
            role = row["analysis_role"]
            require(role in counts and row["navdp_depth_source"] == "zero"
                    and int(row["metric_depth_sensor_consumed_any"]) == 0
                    and int(row["monocular_receipt_plans"]) == 0,
                    f"{label}/{role}: zero receipt changed")
            reached = int(row["reached"])
            require(reached == int(float(row["final_goal_dist_m"]) < 1.0),
                    f"{label}/{role}: success-distance mismatch")
            payload_path = root / ARM / f"{episode}_{row['query_id']}_plans.json"
            payload = json.loads(payload_path.read_text())
            require(payload.get("analysis_role_not_forwarded") is True,
                    f"{label}/{role}: role leak")
            plan_count += audit_zero_depth_plans(
                payload["query_leg"])["plan_count"]
            counts[role] += reached
            paired[role]["zero"].append(reached)
            paired[role]["metric"].append(int(metric_rows[role]["reached"]))
            paired[role]["mono"].append(int(mono_rows[role]["reached"]))

    summary = json.loads(summary_path.read_text())
    require(summary.get("schema_version") == SCHEMA
            and summary.get("status") == "complete"
            and summary.get("benchmark_manifest_sha256")
            == expected_manifest_sha256,
            "summary contract changed")
    for role in ("novel", "revisit"):
        arm = summary["results"][role]["arms"][ARM]
        require(int(arm["n"]) == 21
                and int(arm["successes"]) == counts[role],
                f"summary {role} count differs from raw files")
        for reference, values in (("metric_native", "metric"),
                                  ("mono_native", "mono")):
            gains = sum(a == 1 and b == 0 for a, b in zip(
                paired[role]["zero"], paired[role][values]))
            losses = sum(a == 0 and b == 1 for a, b in zip(
                paired[role]["zero"], paired[role][values]))
            contrast = summary["results"][role]["zero_depth_contrasts"][
                f"{ARM}_minus_{reference}"]
            require(int(contrast["gains"]) == gains
                    and int(contrast["losses"]) == losses
                    and abs(float(contrast["exact_mcnemar_two_sided_p"])
                            - exact_mcnemar_two_sided(gains, losses)) < 1e-12,
                    f"summary {role}/{reference} contrast changed")
    require(int(summary["results"]["all"]["arms"][ARM]["successes"])
            == counts["novel"] + counts["revisit"],
            "summary all count differs from raw files")
    return {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "authorized": True,
        "histories": 21,
        "queries": 42,
        "successes_by_role": counts,
        "zero_depth_plan_count": plan_count,
        "metric_sensor_plan_count": 0,
        "monocular_receipt_plan_count": 0,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA,
        "reference_verification_sha256": REFERENCE_VERIFY_SHA,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_root, args.reference_root, args.bench_root,
        args.expected_manifest_sha256, args.summary,
    )
    require(not args.out.exists(), "verification output exists")
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_name(args.out.name + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + f"  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
