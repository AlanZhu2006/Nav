#!/usr/bin/env python3
"""Independently verify an outcome-blind factual-B transport repair plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing sidecar: {path}")
    require(sidecar.read_text().split() == [digest, path.name],
            f"bad sidecar: {path}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--repair-plan", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "verification output exists")
    plan_sha = verify_sidecar(args.repair_plan)
    plan = json.loads(args.repair_plan.read_text())
    benchmark = json.loads(args.benchmark_manifest.read_text())
    episodes = benchmark["episodes"]
    require(plan["candidate_histories"] == len(episodes) == 84,
            "population size mismatch")
    missing = [int(value) for value in plan["missing_history_indices"]]
    flattened = [
        int(index)
        for group in plan["repair_groups"]
        for index in group["history_indices"]
    ]
    require(sorted(flattened) == missing and len(flattened) == len(set(flattened)),
            "repair groups do not exactly partition missing identities")
    require(plan["repair_group_count"] == len(plan["repair_groups"]),
            "repair group count mismatch")
    require(plan["completion_payloads_deserialized"] is False
            and plan["navigation_outcomes_read"] is False
            and plan["query_policy_outcomes_read"] is False,
            "repair selection was not result-blind")
    require(plan["model_or_controller_changed"] is False
            and plan["scientific_thresholds_changed"] is False
            and plan["step_budget_changed"] is False
            and plan["fallback_completion_allowed"] is False,
            "repair changed the scientific contract")
    archive = args.archive_root / "archive_receipt.json"
    archive_sha = verify_sidecar(archive)
    require(archive_sha == plan["archive_receipt_sha256"],
            "archive receipt binding mismatch")
    archive_payload = json.loads(archive.read_text())
    require(archive_payload["navigation_outcomes_read"] is False
            and archive_payload["completion_payloads_deserialized"] is False,
            "partial archive inspected outcomes")
    for index in missing:
        row = episodes[index]
        label = f"{index:03d}_{row['scene']}_{row['episode']}"
        require(not (args.run_root / "factual_b" / label).exists(),
                f"repair target was not cleared: {index}")
    result = {
        "schema_version": "hm3d_natural_b_transport_repair_verification_v1_20260830",
        "verified": True,
        "repair_plan_sha256": plan_sha,
        "missing_history_count": len(missing),
        "repair_group_count": len(plan["repair_groups"]),
        "completion_payloads_deserialized": False,
        "navigation_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "scientific_contract_unchanged": True,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )


if __name__ == "__main__":
    main()
