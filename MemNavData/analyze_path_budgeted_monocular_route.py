#!/usr/bin/env python3
"""Replay a sealed monocular route with cumulative path-budget projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from MemNavData.monocular_adjacent_motion import PlanarMotionReceipt
from MemNavData.path_budgeted_route_compass import (
    from_local_route,
    tangent_from_local_route,
)
from MemNavData.run_local_monocular_adjacent_motion_audit import require, sha256
from MemNavData.run_local_monocular_route_compass_audit import (
    angle_between_degrees,
    identity_motion,
    motion_from_payload,
    route_from_chronological,
)


SCHEMA_VERSION = "path_budgeted_monocular_route_replay_v1_20260903"
TANGENT_SCHEMA_VERSION = (
    "tangent_path_budgeted_monocular_route_replay_v1_20260903"
)
SOURCE_SCHEMA_VERSION = "monocular_route_compass_mechanism_audit_v1_20260903"
QUERY_MOTION_SCHEMA_VERSION = "local_monocular_adjacent_motion_audit_v2_20260903"


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--expected-route-audit-sha", required=True)
    parser.add_argument("--query-motion-audit", type=Path, required=True)
    parser.add_argument("--expected-query-motion-sha", required=True)
    parser.add_argument(
        "--guidance-mode",
        choices=("pose-chord", "route-tangent"),
        default="pose-chord",
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(not arguments.out.exists(), "output already exists")
    require(sha256(arguments.route_audit)
            == arguments.expected_route_audit_sha,
            "route audit changed")
    require(sha256(arguments.query_motion_audit)
            == arguments.expected_query_motion_sha,
            "query motion audit changed")
    source = json.loads(arguments.route_audit.read_text(encoding="utf-8"))
    query = json.loads(
        arguments.query_motion_audit.read_text(encoding="utf-8"))
    require(source.get("schema_version") == SOURCE_SCHEMA_VERSION,
            "wrong route-audit schema")
    require(query.get("schema_version") == QUERY_MOTION_SCHEMA_VERSION,
            "wrong query-motion schema")
    require(source.get("history_index") == query.get("history_index"),
            "source histories differ")
    require(source.get("history_local_motion_chain_intact") is True
            and query.get("local_motion_chain_intact") is True,
            "source local-motion chain is incomplete")

    history_motions = [
        motion_from_payload(edge["motion"])
        for edge in source["history_edges"]
    ]
    require(all(motion is not None for motion in history_motions),
            "history route contains an absent motion")
    route = route_from_chronological([
        motion for motion in history_motions if motion is not None])
    compass = (
        tangent_from_local_route(route)
        if arguments.guidance_mode == "route-tangent"
        else from_local_route(route)
    )
    query_motions: list[PlanarMotionReceipt] = [identity_motion()]
    query_motions.extend(
        motion_from_payload(row["visual"].get("motion"))
        for row in query["pairs"])
    require(all(motion is not None for motion in query_motions),
            "query route contains an absent motion")
    scorer_rows = source["compass_rows"]
    require(len(scorer_rows) == len(query_motions),
            "source scorer and query motion counts differ")

    rows: list[dict[str, Any]] = []
    bearing_errors: list[float] = []
    translated_errors: list[float] = []
    terminal_errors: list[float] = []
    progress_fraction_errors: list[float] = []
    monotone = True
    budget_respected = True
    previous_progress = 0.0
    for motion, scorer in zip(query_motions, scorer_rows):
        require(motion is not None, "query motion disappeared")
        readout = compass.advance_local_se2(
            executed_forward_m=motion.forward_m,
            executed_left_m=motion.left_m,
            executed_yaw_rad=motion.yaw_rad,
        )
        oracle = scorer["ground_truth_analysis_only"]
        bearing_error = angle_between_degrees(
            np.asarray(readout.unit_bearing),
            np.asarray(oracle["unit_bearing"]))
        progress_fraction_error = abs(
            float(readout.route_progress_fraction)
            - float(oracle["route_progress_fraction"]))
        bearing_errors.append(bearing_error)
        kind = str(scorer["query_kind"])
        if kind == "reverse_route":
            translated_errors.append(bearing_error)
        if kind == "terminal_turn":
            terminal_errors.append(bearing_error)
        progress_fraction_errors.append(progress_fraction_error)
        monotone &= readout.projected_progress_m + 1e-9 >= previous_progress
        budget_respected &= (
            readout.projected_progress_m
            <= readout.query_path_length_m + 1e-9
            and readout.progress_increment_m
            <= readout.progress_budget_m + 1e-9)
        previous_progress = readout.projected_progress_m
        rows.append({
            "query_index": int(scorer["query_index"]),
            "query_kind": kind,
            "readout": readout.audit_dict(),
            "bearing_error_deg_analysis_only": bearing_error,
            "route_progress_fraction_error_analysis_only": (
                progress_fraction_error),
        })

    old = source["route_compass"]
    result = {
        "schema_version": (
            TANGENT_SCHEMA_VERSION
            if arguments.guidance_mode == "route-tangent"
            else SCHEMA_VERSION),
        "claim_boundary": (
            "post-hoc projection-state replay on sealed monocular motion "
            "receipts; no model, controller, navigation outcome, or SR"),
        "history_index": int(source["history_index"]),
        "source_route_audit_sha256": sha256(arguments.route_audit),
        "source_query_motion_audit_sha256": sha256(
            arguments.query_motion_audit),
        "projection_rule": (
            "previous_s <= s_t <= min(route_extent, cumulative_query_path_t)"),
        "guidance_mode": arguments.guidance_mode,
        "live_position_used_for_bearing": (
            arguments.guidance_mode == "pose-chord"),
        "distance_regime_present": False,
        "visual_gate_present": False,
        "endpoint_fallback_present": False,
        "native_fallback_present": False,
        "projection_monotone": bool(monotone),
        "cumulative_path_budget_respected": bool(budget_respected),
        "route_extent_m": float(compass.route_extent_m),
        "final_projected_progress_m": float(compass.projected_progress_m),
        "query_path_length_m": float(rows[-1]["readout"]["query_path_length_m"]),
        "bearing_count": len(bearing_errors),
        "bearing_error_median_deg_analysis_only": float(
            median(bearing_errors)),
        "bearing_error_p90_deg_analysis_only": percentile(
            bearing_errors, 0.90),
        "bearing_within_30deg_count": sum(
            value <= 30.0 for value in bearing_errors),
        "translated_bearing_count": len(translated_errors),
        "translated_bearing_error_median_deg_analysis_only": float(
            median(translated_errors)),
        "translated_bearing_error_p90_deg_analysis_only": percentile(
            translated_errors, 0.90),
        "translated_bearing_within_30deg_count": sum(
            value <= 30.0 for value in translated_errors),
        "terminal_turn_bearing_count": len(terminal_errors),
        "terminal_turn_bearing_error_median_deg_analysis_only": float(
            median(terminal_errors)),
        "route_progress_fraction_error_median_analysis_only": float(
            median(progress_fraction_errors)),
        "route_progress_fraction_error_final_analysis_only": float(
            progress_fraction_errors[-1]),
        "per_step_projection_baseline": {
            "translated_bearing_count": int(old["translated_bearing_count"]),
            "translated_bearing_error_median_deg_analysis_only": float(
                old["translated_bearing_error_median_deg_analysis_only"]),
            "translated_bearing_within_30deg_count": int(
                old["translated_bearing_within_30deg_count"]),
            "progress_error_final_m_analysis_only": float(
                old["progress_error_final_m_analysis_only"]),
        },
        "rows": rows,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = arguments.out / "path_budgeted_route_replay.json"
    output.write_bytes(encoded)
    (arguments.out / "path_budgeted_route_replay.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  path_budgeted_route_replay.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": int(source["history_index"]),
        "translated_bearing": {
            "within_30deg": result["translated_bearing_within_30deg_count"],
            "count": result["translated_bearing_count"],
            "median_deg": result[
                "translated_bearing_error_median_deg_analysis_only"],
            "p90_deg": result[
                "translated_bearing_error_p90_deg_analysis_only"],
        },
        "final_progress_m": result["final_projected_progress_m"],
        "route_extent_m": result["route_extent_m"],
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
