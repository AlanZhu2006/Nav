#!/usr/bin/env python3
"""Score stored first-decision bearings, never infer counterfactual SR."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from MemNavData.certified_relocalization_runtime import (
    COVERAGE_ABLATION_AUTHORITY_POLICY, STRICT_AUTHORITY_POLICY,
    UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
    fundamental_can_reach_certificate, operational_authority_decision,
    scale_free_relative_xy,
)


def unit(vector):
    if vector is None:
        return None
    v = np.asarray(vector, dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-12:
        return None
    return v / np.linalg.norm(v)


def endpoint_bearing(state, goal):
    """GT scoring only: Habitat yaw -> NavDP [forward, left]."""
    dx, dz = float(goal[0]) - state["x"], float(goal[2]) - state["z"]
    yaw = float(state["yaw"])
    return unit([-dx * math.sin(yaw) - dz * math.cos(yaw),
                 -dx * math.cos(yaw) + dz * math.sin(yaw)])


def angle(a, b):
    a, b = unit(a), unit(b)
    return None if a is None or b is None else math.degrees(
        math.atan2(abs(a[0]*b[1]-a[1]*b[0]), float(a @ b)))


def score_row(row):
    cec, raw = row["arms"]["cec"], row["arms"]["raw_fixed"]
    cp, rp = cec["plan"], raw["plan"]
    gt = endpoint_bearing(cec["state"], row["goal_floor_position"])
    pnp = cp.get("certified_relocalization_pnp") or {}
    strict = operational_authority_decision(pnp)
    no_cov = operational_authority_decision(pnp, policy=COVERAGE_ABLATION_AUTHORITY_POLICY)
    finite = operational_authority_decision(pnp, policy=UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY)
    if strict["accepted"] != bool(cp["certified_relocalization_accepted"]):
        raise ValueError(f"Original strict decision not reproduced: task {row['task']}")
    pnp_vector = None
    if finite["pnp_pose_available"]:
        pnp_vector = unit(scale_free_relative_xy(cec["camera_pose9"], pnp["pose9"]))
    recorded = cp.get("memory_bearing_unit")
    if strict["accepted"] and (recorded is None or angle(pnp_vector, recorded) > 1e-4):
        raise ValueError(f"Reconstructed bearing differs from accepted runtime: {row['task']}")
    candidates = cp.get("router_candidate_trials") or []
    selected = next((c for c in candidates if c["anchor"] == cp["router_selected_anchor"]), {})
    precheck_strict = fundamental_can_reach_certificate(selected)[0]
    precheck_no_cov = fundamental_can_reach_certificate(selected, require_coverage=False)[0]
    precheck_finite = int(selected.get("lightglue_matches", 0)) >= 8
    never_solved = str(pnp.get("status", "")).startswith("precheck_")
    distance = math.hypot(row["goal_floor_position"][0]-cec["state"]["x"],
                          row["goal_floor_position"][2]-cec["state"]["z"])
    if abs(distance - cp["evaluation_gt_goal_distance_m"]) > 1e-5:
        raise ValueError("Goal/current-position scoring convention mismatch")
    return {
        "task": row["task"], "history": row["history"], "scene": row["scene"],
        "group": row["group"], "q_eligible": row["q_eligible"],
        "initial_distance_m": distance, "old_reached": row["original_reached"],
        "raw_anchor": rp["anchor"], "pnp_anchor": cp["router_selected_anchor"],
        "raw_error_deg": angle(rp.get("memory_bearing_unit"), gt),
        "pnp_error_deg": angle(pnp_vector, gt),
        "pnp_status": pnp.get("status"), "pnp_inliers": pnp.get("inliers"),
        "pnp_available": finite["pnp_pose_available"],
        "pnp_query_coverage": pnp.get("query_inlier_coverage"),
        "pnp_reference_coverage": pnp.get("reference_inlier_coverage"),
        "strict_accept": strict["accepted"],
        "no_coverage_existing_accept": bool(precheck_no_cov and no_cov["accepted"]),
        "finite_existing_accept": bool(precheck_finite and finite["accepted"]),
        "new_no_coverage_existing_accept": bool(not strict["accepted"] and precheck_no_cov and no_cov["accepted"]),
        "needs_no_coverage_pnp": bool(never_solved and precheck_no_cov),
        "needs_finite_pnp": bool(never_solved and precheck_finite),
        "original_precheck_pass": precheck_strict,
        "same_state_rgb_sha256": cec["image_sha256"],
        "accepted_runtime_bearing_crosschecked": bool(strict["accepted"]),
    }


def error_summary(values):
    values = [float(v) for v in values if v is not None]
    return {"available": len(values),
            "median_deg": float(np.median(values)) if values else None,
            "p90_deg": float(np.percentile(values, 90)) if values else None,
            **{f"within_{t}_deg": sum(v <= t for v in values) for t in (15, 30, 45)},
            "over_90_deg": sum(v > 90 for v in values)}


def summarize(rows):
    groups = {}
    for group in sorted({r["group"] for r in rows}):
        subset = [r for r in rows if r["group"] == group]
        newly = [r for r in subset if r["new_no_coverage_existing_accept"]]
        groups[group] = {
            "queries": len(subset), "histories": len({r["history"] for r in subset}),
            "scenes": len({r["scene"] for r in subset}),
            "raw": error_summary(r["raw_error_deg"] for r in subset),
            "pnp_all_available": error_summary(r["pnp_error_deg"] for r in subset),
            "pnp_strict_accepted": error_summary(r["pnp_error_deg"] for r in subset if r["strict_accept"]),
            "pnp_new_no_coverage": error_summary(r["pnp_error_deg"] for r in newly),
            "new_no_coverage_tasks": [r["task"] for r in newly],
            "needs_no_coverage_pnp": [r["task"] for r in subset if r["needs_no_coverage_pnp"]],
            "needs_finite_pnp": [r["task"] for r in subset if r["needs_finite_pnp"]],
        }
    return groups


def load_evidence(path):
    header = final = None
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("EVIDENCE_HEADER "):
            header = json.loads(line.split(" ", 1)[1])
        elif line.startswith("EVIDENCE_ROW "):
            rows.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("EVIDENCE_FINAL "):
            final = json.loads(line.split(" ", 1)[1])
    if not header or not final or not final["completed"]:
        raise ValueError("Evidence extraction is incomplete")
    if len(rows) != header["expected_rows"] or len(rows) != final["rows"]:
        raise ValueError("Evidence row count mismatch")
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task in evidence")
    return header, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    header, source = load_evidence(args.evidence)
    rows = [score_row(r) for r in source]
    payload = {"schema": "low_covis_first_bearing_audit_v1", "completed": True,
        "evidence": str(args.evidence.resolve()),
        "evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "source": header, "queries": len(rows), "new_rollouts": 0,
        "interpretation": "Stored-pose diagnostic only; missing PnP not imputed; no new SR",
        "groups": summarize(rows), "rows": rows}
    with args.out.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"queries": len(rows), "groups": payload["groups"]}, indent=2))


if __name__ == "__main__":
    main()
