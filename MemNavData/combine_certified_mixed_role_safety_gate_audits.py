#!/usr/bin/env python3
"""Combine disjoint partial receipts from the strict-v4 safety gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from MemNavData.audit_certified_mixed_role_safety_gate import SCHEMA_VERSION


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def leg_executed(record: dict[str, Any], leg: str) -> bool:
    """Support both the original receipt and the corrected executed-leg field."""
    explicit = record.get("novel_leg_executed")
    if isinstance(explicit, dict) and leg in explicit:
        return bool(explicit[leg])
    return int(record["certificate"][leg].get("plans", 0)) > 0


def combine(paths: list[Path], *, expected_scenes: int | None = None) -> dict[str, Any]:
    require(bool(paths), "no audit receipts supplied")
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records = []
    seen = set()
    reasons: Counter[str] = Counter()
    for path, receipt in zip(paths, receipts):
        require(receipt.get("schema_version") == SCHEMA_VERSION,
                f"schema changed: {path}")
        require(receipt.get("audit_ok") is True, f"partial audit failed: {path}")
        require(receipt.get("scope") ==
                "strict-v4 implementation/causal safety gate; not an SR estimate",
                f"scope changed: {path}")
        for record in receipt.get("records", []):
            identity = (record["scene"], record["episode"], int(record["seed"]))
            require(identity not in seen, f"duplicate episode across receipts: {identity}")
            seen.add(identity)
            records.append(record)
        reasons.update(receipt.get("independent_certificate_reasons", {}))
    if expected_scenes is not None:
        require(len(records) == expected_scenes,
                f"expected {expected_scenes} scenes, found {len(records)}")
    return {
        "schema_version": "certified_mixed_role_safety_gate_combined_v1_20260813",
        "scope": "strict-v4 implementation/causal safety gate; not an SR estimate",
        "audit_ok": True,
        "partial_receipts": [str(path.resolve()) for path in paths],
        "scenes": len(records),
        "novel_legs_audited": sum(
            leg_executed(record, leg)
            for record in records for leg in ("A", "B")
        ),
        "all_novel_prefixes_exact": all(
            record["prefix_exact"][leg]["all_exact"]
            for record in records for leg in ("A", "B")
            if leg_executed(record, leg)
        ),
        "novel_certificate_accepts": sum(
            record["certificate"][leg]["accepted_requests"]
            for record in records for leg in ("A", "B")
        ),
        "novel_adapter_takeovers": sum(
            record["certificate"][leg]["takeovers"]
            for record in records for leg in ("A", "B")
        ),
        "certificate_runtime_failures": sum(
            record["certificate"][leg]["runtime_failures"]
            for record in records for leg in ("A", "B", "C")
        ),
        "revisit_positive_controls": sum(
            record["revisit_positive_control_eligible"] for record in records
        ),
        "revisit_positive_control_activations": sum(
            record["revisit_positive_control_activated"] for record in records
        ),
        "revisit_positive_control_successes": sum(
            record["revisit_positive_control_activated"]
            and record["certified_success"]["C"] for record in records
        ),
        "independent_certificate_reasons": dict(sorted(reasons.items())),
        "records": sorted(
            records, key=lambda row: (row["scene"], row["episode"], row["seed"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = combine(args.receipts, expected_scenes=args.expected_scenes)
    require(result["all_novel_prefixes_exact"], "combined Novel prefixes differ")
    require(result["novel_certificate_accepts"] == 0,
            "combined receipt contains a Novel certificate accept")
    require(result["novel_adapter_takeovers"] == 0,
            "combined receipt contains a Novel adapter takeover")
    require(result["certificate_runtime_failures"] == 0,
            "combined receipt contains a runtime failure")
    require(result["revisit_positive_control_activations"] >= 1,
            "combined receipt has no Revisit positive-control activation")
    require(result["revisit_positive_control_successes"] >= 1,
            "combined receipt has no successful Revisit positive control")
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
