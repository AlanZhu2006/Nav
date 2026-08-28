#!/usr/bin/env python3
"""Aggregate the five consumed ViNT/CEC bearing-alignment mechanism cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "vint_cec_bearing_alignment_loss5_summary_v1_20260828"
CELL_SCHEMA = "vint_cec_bearing_alignment_cell_audit_v1_20260828"
ARMS = (
    "anchor_unaligned",
    "native_bearing_aligned",
    "anchor_bearing_aligned",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(run_root: Path, expected_cells: int = 5) -> dict:
    paths = sorted(run_root.glob("evaluation/*/vint/direction_triple_audit.json"))
    require(len(paths) == expected_cells,
            f"expected {expected_cells} cell audits, found {len(paths)}")
    cells = []
    identities = set()
    for path in paths:
        row = json.loads(path.read_text())
        require(row.get("schema_version") == CELL_SCHEMA
                and row.get("verified") is True, f"invalid cell audit {path}")
        identity = (str(row["scene"]), str(row["episode"]))
        require(identity not in identities, "duplicate mechanism cell")
        identities.add(identity)
        cells.append({**row, "audit_path": str(path),
                      "audit_sha256": sha256_file(path)})
    arm_summary = {}
    for arm in ARMS:
        rows = [cell["arms"][arm] for cell in cells]
        arm_summary[arm] = {
            "n": len(rows),
            "success": sum(int(row["success"]) for row in rows),
            "first_horizon_moved_closer": sum(
                bool(row["moved_closer"]) for row in rows),
            "first_horizon_heading_within_30_deg": sum(
                float(row["bearing_execution_error_deg"]) <= 30.0
                for row in rows),
            "mean_distance_change_m": sum(
                float(row["distance_change_m"]) for row in rows) / len(rows),
            "alignment_count": sum(int(row["alignment_count"]) for row in rows),
        }
    aligned = ("native_bearing_aligned", "anchor_bearing_aligned")
    primary_gate = {
        arm: (
            arm_summary[arm]["alignment_count"] == expected_cells
            and arm_summary[arm]["first_horizon_heading_within_30_deg"] >= 4
            and arm_summary[arm]["first_horizon_moved_closer"] >= 4
        )
        for arm in aligned
    }
    endpoint_gate = {
        arm: arm_summary[arm]["success"] >= 3 for arm in aligned
    }
    return {
        "schema_version": SCHEMA,
        "verified_cells": len(cells),
        "scope": "consumed outcome-aware mechanism; not a paper SR result",
        "arm_summary": arm_summary,
        "primary_direction_consumption_gate": primary_gate,
        "exploratory_endpoint_gate": endpoint_gate,
        "any_aligned_arm_passed_both_gates": any(
            primary_gate[arm] and endpoint_gate[arm] for arm in aligned),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    payload = aggregate(args.run_root.resolve(), args.expected_cells)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
