#!/usr/bin/env python3
"""Audit whether a ViNT+CEC rollout physically consumes the certified bearing.

The formal ViNT controller-native experiment projects an accepted CEC proof to
an historical anchor ImageGoal.  This audit does not rescore success and never
uses an oracle to choose an action.  It compares the bearing already present in
the proof packet with the direction of the first executed rollout horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


SCHEMA = "vint_cec_direction_consumption_audit_v1_20260828"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_pair(value: Any, label: str) -> tuple[float, float]:
    require(isinstance(value, list) and len(value) == 2,
            f"{label} must contain two values")
    pair = (float(value[0]), float(value[1]))
    require(all(math.isfinite(item) for item in pair),
            f"{label} must be finite")
    require(math.hypot(*pair) > 1e-9, f"{label} must be non-zero")
    return pair


def first_proof_direction(plan: dict[str, Any]) -> tuple[float, float]:
    packet = plan.get("cec_handoff_packet")
    require(isinstance(packet, dict), "first plan lacks CEC handoff packet")
    proof = packet.get("public_proof")
    require(isinstance(proof, dict), "handoff packet lacks public proof")
    require(proof.get("accepted") is True, "first proof was not accepted")
    require(plan.get("cec_takeover") is True,
            "first accepted proof did not authorize takeover")
    require(proof.get("pointgoal_units") == "lingbot_raw_direction_only",
            "CEC direction units changed")
    return finite_pair(proof.get("direction_vector"), "direction_vector")


def local_motion(
    start: dict[str, Any], end: dict[str, Any]
) -> tuple[float, float]:
    yaw = float(start["yaw"])
    dx = float(end["x"]) - float(start["x"])
    dz = float(end["z"]) - float(start["z"])
    # Same [forward, left] convention as eval_2leg_habitat.waypoints_to_world.
    forward = -math.sin(yaw) * dx - math.cos(yaw) * dz
    left = -math.cos(yaw) * dx + math.sin(yaw) * dz
    return forward, left


def audit_file(path: Path, exec_horizon: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    plans = payload.get("query_leg")
    rollout = payload.get("rollout_traces", {}).get("query")
    require(isinstance(plans, list) and plans, f"{path}: no query plans")
    require(isinstance(rollout, list) and len(rollout) >= 2,
            f"{path}: insufficient rollout poses")
    first = plans[0]
    require(int(first.get("step", -1)) == 0, f"{path}: first plan is not step 0")
    direction = first_proof_direction(first)
    bearing_deg = math.degrees(math.atan2(direction[1], direction[0]))

    start = rollout[0]
    end_index = min(exec_horizon, len(rollout) - 1)
    end = rollout[end_index]
    forward, left = local_motion(start, end)
    displacement_m = math.hypot(forward, left)
    require(displacement_m > 1e-6, f"{path}: first horizon did not translate")
    executed_heading_deg = math.degrees(math.atan2(left, forward))
    heading_error_deg = abs(wrap_deg(executed_heading_deg - bearing_deg))

    current_distance = float(first["evaluation_gt_goal_distance_m"])
    next_plans = [
        plan for plan in plans
        if int(plan.get("step", -1)) == exec_horizon
    ]
    require(len(next_plans) == 1,
            f"{path}: expected one plan at step {exec_horizon}")
    horizon_distance = float(next_plans[0]["evaluation_gt_goal_distance_m"])

    cell = path.parents[3].name
    return {
        "cell": cell,
        "plans_path": str(path),
        "plans_sha256": sha256_file(path),
        "cec_bearing_deg": bearing_deg,
        "cec_bearing_abs_deg": abs(bearing_deg),
        "executed_heading_deg": executed_heading_deg,
        "bearing_execution_error_deg": heading_error_deg,
        "first_horizon_displacement_m": displacement_m,
        "first_horizon_forward_m": forward,
        "first_horizon_left_m": left,
        "distance_before_m": current_distance,
        "distance_after_horizon_m": horizon_distance,
        "distance_change_m": horizon_distance - current_distance,
        "moved_away": horizon_distance > current_distance + 1e-6,
    }


def aggregate(records: list[dict[str, Any]], run_root: Path,
              exec_horizon: int) -> dict[str, Any]:
    errors = [float(row["bearing_execution_error_deg"]) for row in records]
    changes = [float(row["distance_change_m"]) for row in records]
    return {
        "schema_version": SCHEMA,
        "source_run_root": str(run_root),
        "audit_scope": (
            "first executed horizon of every formal grant-arm Revisit; "
            "diagnostic only, no action or outcome selection"
        ),
        "exec_horizon": exec_horizon,
        "histories": len(records),
        "first_plan_cec_takeovers": len(records),
        "first_horizon_moved_away": sum(bool(row["moved_away"])
                                         for row in records),
        "bearing_execution_error_deg": {
            "mean": mean(errors),
            "median": median(errors),
            "minimum": min(errors),
            "maximum": max(errors),
            "within_30_deg": sum(value <= 30.0 for value in errors),
            "at_least_135_deg": sum(value >= 135.0 for value in errors),
        },
        "first_horizon_distance_change_m": {
            "mean": mean(changes),
            "median": median(changes),
            "minimum": min(changes),
            "maximum": max(changes),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-histories", type=int, default=28)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    require(args.expected_histories > 0, "expected histories must be positive")
    require(args.exec_horizon > 0, "execution horizon must be positive")
    run_root = args.run_root.resolve()
    pattern = "formal/evaluation/*/vint/grant/result/*_revisit_plans.json"
    paths = sorted(run_root.glob(pattern))
    require(len(paths) == args.expected_histories,
            f"expected {args.expected_histories} Revisit files, found {len(paths)}")
    records = [audit_file(path, args.exec_horizon) for path in paths]
    require(len({row["cell"] for row in records}) == len(records),
            "duplicate formal cells found")
    result = aggregate(records, run_root, args.exec_horizon)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        require(not args.out.exists(), f"refusing to overwrite {args.out}")
        args.out.write_text(encoded)
        print(args.out)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
