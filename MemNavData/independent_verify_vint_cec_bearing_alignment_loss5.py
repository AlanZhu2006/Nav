#!/usr/bin/env python3
"""Independently recount raw ViNT/CEC Loss-5 mechanism outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from MemNavData.cec_handoff_contract import verify_handoff_packet_envelope


SCHEMA = "vint_cec_bearing_alignment_loss5_verification_v1_20260828"
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


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def verify(run_root: Path, summary_path: Path, selection_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    require(
        summary.get("schema_version")
        == "vint_cec_bearing_alignment_loss5_summary_v1_20260828",
        "mechanism summary schema changed",
    )
    selection = json.loads(selection_path.read_text())
    expected = {
        (str(row["scene"]), str(row["episode"]))
        for row in selection["queries"]
    }
    cells = sorted(run_root.glob("evaluation/*/vint"))
    require(len(cells) == 5, "raw mechanism cell count changed")
    observed = set()
    recount = {
        arm: {
            "n": 0,
            "success": 0,
            "first_horizon_moved_closer": 0,
            "first_horizon_heading_within_30_deg": 0,
            "distance_change_sum_m": 0.0,
            "alignment_count": 0,
        }
        for arm in ARMS
    }
    raw_hashes = []
    for cell in cells:
        contract = json.loads((cell / "direction_triple_contract.json").read_text())
        identity = (str(contract["scene"]), str(contract["episode"]))
        require(identity not in observed, "duplicate raw cell identity")
        observed.add(identity)
        for arm in ARMS:
            result = cell / arm / "result"
            metric_path = result / "metric.csv"
            with metric_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 1, "raw arm does not contain one query")
            metric = rows[0]
            plan_paths = list(result.glob("*_revisit_plans.json"))
            require(len(plan_paths) == 1, "raw arm plan count changed")
            plan_path = plan_paths[0]
            payload = json.loads(plan_path.read_text())
            plans = payload["query_leg"]
            rollout = payload["rollout_traces"]["query"]
            first = plans[0]
            step8 = [row for row in plans if int(row["step"]) == 8]
            require(len(step8) == 1 and len(rollout) > 8,
                    "raw first-horizon evidence is incomplete")
            proof = verify_handoff_packet_envelope(first["cec_handoff_packet"])
            direction = proof["direction_vector"]
            bearing = math.degrees(math.atan2(direction[1], direction[0]))
            yaw = float(rollout[0]["yaw"])
            dx = float(rollout[8]["x"]) - float(rollout[0]["x"])
            dz = float(rollout[8]["z"]) - float(rollout[0]["z"])
            forward = -math.sin(yaw) * dx - math.cos(yaw) * dz
            left = -math.cos(yaw) * dx + math.sin(yaw) * dz
            executed = math.degrees(math.atan2(left, forward))
            error = abs(wrap_deg(executed - bearing))
            change = (
                float(step8[0]["evaluation_gt_goal_distance_m"])
                - float(first["evaluation_gt_goal_distance_m"])
            )
            row = recount[arm]
            row["n"] += 1
            row["success"] += int(metric["reached"])
            row["first_horizon_moved_closer"] += int(change < -1e-6)
            row["first_horizon_heading_within_30_deg"] += int(error <= 30.0)
            row["distance_change_sum_m"] += change
            row["alignment_count"] += int(
                metric["cec_initial_bearing_alignment_count"])
            raw_hashes.extend((sha256_file(metric_path), sha256_file(plan_path)))
    require(observed == expected, "raw cells differ from frozen Loss-5 selection")
    normalized = {}
    for arm, row in recount.items():
        normalized[arm] = {
            key: value for key, value in row.items()
            if key != "distance_change_sum_m"
        }
        normalized[arm]["mean_distance_change_m"] = (
            row["distance_change_sum_m"] / row["n"])
        reported = summary["arm_summary"][arm]
        for key, value in normalized[arm].items():
            if isinstance(value, float):
                require(math.isclose(float(reported[key]), value,
                                     rel_tol=0.0, abs_tol=1e-12),
                        f"summary mismatch for {arm}.{key}")
            else:
                require(reported[key] == value,
                        f"summary mismatch for {arm}.{key}")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "selection_sha256": sha256_file(selection_path),
        "raw_file_hash_count": len(raw_hashes),
        "recount": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    payload = verify(
        args.run_root.resolve(), args.summary.resolve(), args.selection.resolve())
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
