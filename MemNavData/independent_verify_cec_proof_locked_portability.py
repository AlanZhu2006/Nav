#!/usr/bin/env python3
"""Independent raw-receipt verification of the portability pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


SCHEMA = "cec_proof_locked_portability_pilot_verification_v1_20260827"
CONTROLLERS = {"navdp", "vint", "iplanner"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run_root: Path, summary_path: Path,
           query_manifest: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    require(summary.get("verified") is True, "summary is not verified")
    require(summary.get("performance_used_as_gate") is False,
            "pilot improperly used performance as a gate")
    require(summary.get("query_manifest_sha256") == digest(query_manifest),
            "summary/query-manifest binding changed")
    paths = sorted((run_root / "evaluation").glob(
        "*/*/authority_pair_audit.json"))
    require(len(paths) == 12, "raw pilot cell count changed")
    rows = [json.loads(path.read_text()) for path in paths]
    require(all(row.get("verified") is True
                and row.get("same_process_pair") is True
                and row.get("handoff_packet_verified") is True
                and row.get("source_accepted_manifest_match") is True
                for row in rows), "a raw causal/packet audit failed")
    require({row.get("controller") for row in rows} == CONTROLLERS,
            "raw controller set changed")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["episode"], row["query_id"])].append(row)
    require(len(grouped) == 4, "raw history count changed")
    for identity, group in grouped.items():
        require({row["controller"] for row in group} == CONTROLLERS,
                f"incomplete executor triad: {identity}")
        for field in ("first_handoff_proof_sha256", "first_handoff_anchor",
                      "handoff_packet_sha256"):
            require(len({row[field] for row in group}) == 1,
                    f"cross-controller {field} changed: {identity}")

    for controller in sorted(CONTROLLERS):
        group = [row for row in rows if row["controller"] == controller]
        reported = summary["controller_results"][controller]
        require(int(reported["n"]) == 4,
                f"{controller}: reported denominator changed")
        checks = {
            "grant_success": sum(int(row["grant_success"]) for row in group),
            "forced_reject_success": sum(
                int(row["forced_reject_success"]) for row in group),
            "paired_gain": sum(int(row["paired_gain"]) for row in group),
            "paired_loss": sum(int(row["paired_loss"]) for row in group),
        }
        require(all(int(reported[key]) == value
                    for key, value in checks.items()),
                f"{controller}: reported outcomes differ from raw receipts")

    hashes = summary.get("raw_audit_sha256", {})
    require(len(hashes) == 12, "summary raw-hash inventory changed")
    require(all(hashes.get(str(path)) == digest(path) for path in paths),
            "a raw authority audit changed after aggregation")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "summary_sha256": digest(summary_path),
        "query_manifest_sha256": digest(query_manifest),
        "raw_cells": 12,
        "histories": 4,
        "controllers": sorted(CONTROLLERS),
        "same_packet_across_controller_triads": True,
        "navigation_utility_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root.resolve(), args.summary.resolve(),
                    args.query_manifest.resolve())
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": True, "raw_cells": 12}, sort_keys=True))


if __name__ == "__main__":
    main()
