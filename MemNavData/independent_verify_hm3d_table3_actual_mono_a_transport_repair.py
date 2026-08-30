#!/usr/bin/env python3
"""Independently verify an outcome-blind Table-III factual-A repair plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freeze_hm3d_table3_actual_mono_a_transport_repair import (
    ARCHIVE_SCHEMA, SCHEMA, factual_label, inventory, sha256, verify_sidecar,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--expected-candidate-plan-sha256", required=True)
    parser.add_argument("--repair-plan", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"verification exists: {args.out}")
    verify_sidecar(args.repair_plan)
    repair = json.loads(args.repair_plan.read_text())
    require(repair.get("schema_version") == SCHEMA,
            "repair schema changed")
    candidate_sha = sha256(args.candidate_plan)
    require(candidate_sha == args.expected_candidate_plan_sha256
            == repair.get("candidate_plan_sha256"),
            "candidate plan identity mismatch")
    episodes = json.loads(args.candidate_plan.read_text())["episodes"]
    require(len(episodes) == repair.get("candidate_count") == 125,
            "candidate count mismatch")

    archive_path = args.archive_root / "archive_receipt.json"
    archive_sha = verify_sidecar(archive_path)
    require(archive_sha == repair.get("archive_receipt_sha256"),
            "archive receipt mismatch")
    archive = json.loads(archive_path.read_text())
    require(archive.get("schema_version") == ARCHIVE_SCHEMA,
            "archive schema changed")
    archived_indices = set()
    for entry in archive["entries"]:
        source = Path(entry["source"])
        destination = Path(entry["destination"])
        require(not source.exists(), f"partial source remains: {source}")
        require(destination.is_dir() and not destination.is_symlink(),
                f"archive destination missing: {destination}")
        require(inventory(destination) == entry["files"],
                f"archive inventory changed: {destination}")
        archived_indices.add(int(entry["history_index"]))

    completed = []
    missing = []
    factual_root = args.run_root / "factual_a"
    for index, row in enumerate(episodes):
        completion = factual_root / factual_label(index, row) / "completion.json"
        if completion.is_file():
            # Verify bytes only; do not deserialize the outcome-bearing receipt.
            verify_sidecar(completion)
            completed.append(index)
        else:
            missing.append(index)
    require(completed == repair.get("completed_history_indices"),
            "completed membership changed")
    require(missing == repair.get("missing_history_indices"),
            "missing membership changed")
    require(archived_indices.issubset(set(missing)),
            "archive contains a completed identity")
    require([int(row["history_index"])
             for row in repair.get("repair_identities", [])] == missing,
            "repair identities changed")
    for payload in (repair, archive):
        require(payload.get("completion_payloads_deserialized") is False,
                "completion payload was read")
        require(payload.get("navigation_outcomes_read") is False,
                "navigation outcome was read")
        require(payload.get("scientific_thresholds_changed") is False,
                "scientific threshold changed")
    require(repair.get("fallback_completion_allowed") is False,
            "fallback completion was allowed")

    result = {
        "schema_version": (
            "hm3d_table3_actual_mono_a_transport_repair_verification_v1_20260830"
        ),
        "verified": True,
        "candidate_plan_sha256": candidate_sha,
        "repair_plan_sha256": sha256(args.repair_plan),
        "archive_receipt_sha256": archive_sha,
        "completed_history_count": len(completed),
        "missing_history_count": len(missing),
        "missing_history_indices": missing,
        "completion_payloads_deserialized": False,
        "navigation_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "scientific_thresholds_changed": False,
        "fallback_completion_allowed": False,
    }
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps({"verified": True, "missing": len(missing)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
