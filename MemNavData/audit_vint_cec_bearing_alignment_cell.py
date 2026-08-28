#!/usr/bin/env python3
"""Audit one same-process ViNT/CEC initial-bearing mechanism triple."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from MemNavData.cec_handoff_contract import verify_handoff_packet_envelope


SCHEMA = "vint_cec_bearing_alignment_cell_audit_v1_20260828"
ARMS = (
    "anchor_unaligned",
    "native_bearing_aligned",
    "anchor_bearing_aligned",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"{path}: expected one selected query")
    return rows[0]


def local_motion(start: dict[str, Any], end: dict[str, Any]) -> tuple[float, float]:
    yaw = float(start["yaw"])
    dx = float(end["x"]) - float(start["x"])
    dz = float(end["z"]) - float(start["z"])
    return (
        -math.sin(yaw) * dx - math.cos(yaw) * dz,
        -math.cos(yaw) * dx + math.sin(yaw) * dz,
    )


def measure_arm(root: Path, arm: str, exec_horizon: int = 8) -> dict[str, Any]:
    require(arm in ARMS, f"unknown arm {arm}")
    result = root / arm / "result"
    summary_path = result / "summary.json"
    metric_path = result / "metric.csv"
    identity_path = root / arm / "compute_identity.json"
    require(summary_path.is_file() and metric_path.is_file(),
            f"{arm}: result files missing")
    plan_paths = sorted(result.glob("*_revisit_plans.json"))
    require(len(plan_paths) == 1, f"{arm}: expected one Revisit plan file")
    plan_path = plan_paths[0]
    payload = json.loads(plan_path.read_text())
    plans = payload.get("query_leg")
    rollout = payload.get("rollout_traces", {}).get("query")
    require(isinstance(plans, list) and len(plans) >= 2,
            f"{arm}: insufficient plans")
    require(isinstance(rollout, list) and len(rollout) > exec_horizon,
            f"{arm}: insufficient rollout")
    first = plans[0]
    require(int(first.get("step", -1)) == 0, f"{arm}: first plan is not step 0")
    at_horizon = [p for p in plans if int(p.get("step", -1)) == exec_horizon]
    require(len(at_horizon) == 1, f"{arm}: missing plan at first horizon")
    packet = first.get("cec_handoff_packet")
    proof = verify_handoff_packet_envelope(packet)
    direction = proof.get("direction_vector")
    require(isinstance(direction, list) and len(direction) == 2,
            f"{arm}: proof direction missing")
    bearing_deg = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    forward, left = local_motion(rollout[0], rollout[exec_horizon])
    require(math.hypot(forward, left) > 1e-6,
            f"{arm}: first horizon did not translate")
    executed_deg = math.degrees(math.atan2(left, forward))
    heading_error = abs(wrap_deg(executed_deg - bearing_deg))
    before = float(first["evaluation_gt_goal_distance_m"])
    after = float(at_horizon[0]["evaluation_gt_goal_distance_m"])
    summary = json.loads(summary_path.read_text())
    metric = read_one_csv(metric_path)
    identity = json.loads(identity_path.read_text())
    alignment_count = int(metric["cec_initial_bearing_alignment_count"])
    if arm == "anchor_unaligned":
        require(alignment_count == 0, "unaligned arm executed an alignment")
        require(first.get("cec_takeover") is True,
                "unaligned anchor arm did not take over")
        require(first.get("cec_forced_reject_native") is False,
                "unaligned anchor arm used forced rejection")
        require(first.get("cec_initial_bearing_alignment_executed") is False,
                "unaligned first plan reports an alignment")
    elif arm == "native_bearing_aligned":
        require(alignment_count == 1, "native aligned arm did not align once")
        require(first.get("cec_takeover") is False
                and first.get("cec_shadow_takeover") is True,
                "native aligned arm lost its shadow proof contract")
        require(first.get("cec_forced_reject_native") is True,
                "native aligned arm did not preserve original-goal fallback")
        require(first.get("cec_initial_bearing_alignment_executed") is True,
                "native aligned first plan did not execute alignment")
    else:
        require(alignment_count == 1, "anchor aligned arm did not align once")
        require(first.get("cec_takeover") is True,
                "anchor aligned arm did not take over")
        require(first.get("cec_forced_reject_native") is False,
                "anchor aligned arm used forced rejection")
        require(first.get("cec_initial_bearing_alignment_executed") is True,
                "anchor aligned first plan did not execute alignment")
    require(summary.get("queries") == 1, f"{arm}: summary query count changed")
    require(summary.get("role_counts") == {"novel": 0, "revisit": 1},
            f"{arm}: selected query is not exactly one Revisit")
    return {
        "arm": arm,
        "success": int(metric["reached"]),
        "steps": int(metric["steps"]),
        "final_goal_dist_m": float(metric["final_goal_dist_m"]),
        "alignment_count": alignment_count,
        "bearing_deg": bearing_deg,
        "executed_first_horizon_heading_deg": executed_deg,
        "bearing_execution_error_deg": heading_error,
        "distance_before_m": before,
        "distance_after_horizon_m": after,
        "distance_change_m": after - before,
        "moved_closer": after < before - 1e-6,
        "gpu_uuid": identity.get("gpu_uuid"),
        "host": identity.get("host"),
        "processes": {
            key: identity.get(key)
            for key in ("memnav", "navdp", "accepted_controller", "controller_proxy")
        },
        "files": {
            "summary_sha256": sha256_file(summary_path),
            "metric_sha256": sha256_file(metric_path),
            "plans_sha256": sha256_file(plan_path),
            "compute_identity_sha256": sha256_file(identity_path),
        },
    }


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    contract_path = root / "direction_triple_contract.json"
    require(contract_path.is_file(), "direction triple contract missing")
    contract = json.loads(contract_path.read_text())
    require(
        contract.get("schema_version")
        == "vint_cec_direction_triple_contract_v1_20260828",
        "direction triple contract schema changed",
    )
    order = contract.get("arm_order")
    require(isinstance(order, list) and set(order) == set(ARMS)
            and len(order) == len(ARMS), "triple arm order is invalid")
    arms = {arm: measure_arm(root, arm) for arm in ARMS}
    hosts = {row["host"] for row in arms.values()}
    gpus = {row["gpu_uuid"] for row in arms.values()}
    require(len(hosts) == 1 and None not in hosts, "arms used different hosts")
    require(len(gpus) == 1 and None not in gpus, "arms used different GPUs")
    for process_key in ("memnav", "navdp", "accepted_controller", "controller_proxy"):
        identities = {
            json.dumps(row["processes"][process_key], sort_keys=True)
            for row in arms.values()
        }
        require(len(identities) == 1,
                f"arms did not share {process_key} process identity")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "scope": contract["scope"],
        "scene": contract["scene"],
        "episode": contract["episode"],
        "arm_order": order,
        "same_loaded_processes": True,
        "contract_sha256": sha256_file(contract_path),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    result = audit(args.root)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
