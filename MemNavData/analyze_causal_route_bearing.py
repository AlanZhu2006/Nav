#!/usr/bin/env python3
"""Score action-integrated route bearings after a route-filter inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from MemNavData.conditional_c_protocol import world_goal_to_local
from MemNavData.episodic_route_filter import ActionIntegratedRouteBearing


SCHEMA_VERSION = "causal_route_bearing_analysis_v1_20260902"
LOOKAHEAD_M = 2.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.zeros(1, dtype=np.float64),
        np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
    ])


def sample_polyline(
    points: np.ndarray, cumulative: np.ndarray, arc_m: float,
) -> np.ndarray:
    arc = float(np.clip(arc_m, 0.0, float(cumulative[-1])))
    if arc >= float(cumulative[-1]):
        return points[-1].copy()
    segment = int(np.searchsorted(cumulative, arc, side="right") - 1)
    segment = min(max(segment, 0), len(points) - 2)
    while (segment < len(points) - 2
           and cumulative[segment + 1] <= cumulative[segment] + 1e-12):
        segment += 1
    low, high = float(cumulative[segment]), float(cumulative[segment + 1])
    if high <= low + 1e-12:
        return points[segment].copy()
    alpha = (arc - low) / (high - low)
    return ((1.0 - alpha) * points[segment]
            + alpha * points[segment + 1])


def angle_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-filter-result", type=Path, required=True)
    parser.add_argument("--runtime-receipts", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "bearing-analysis output exists")

    result = json.loads(arguments.route_filter_result.read_text(encoding="utf-8"))
    query = json.loads(arguments.query_set.read_text(encoding="utf-8"))
    trace = json.loads(arguments.history_trace.read_text(encoding="utf-8"))
    runtime = np.load(arguments.runtime_receipts)
    pose9 = runtime["pose9"].astype(np.float64)
    route_indices = runtime["route_indices"].astype(np.int64)
    history_count = len(trace["poses"])
    require(pose9.shape[0] == history_count + len(query["queries"]),
            "runtime pose receipt and query length differ")
    require(result["source_query_set_sha256"] == sha256(arguments.query_set),
            "route-filter result belongs to another query set")
    require(result["runtime_evaluator_pose_visible"] is False
            and result["observation_acceptance_gate_present"] is False,
            "route-filter information contract changed")

    raw_route = pose9[:history_count][route_indices][:, (0, 2)]
    raw_mean_lag1 = float(np.mean(np.linalg.norm(
        raw_route[1:] - raw_route[:-1], axis=1)))
    metric_mean_lag1 = float(
        result["transition"]["mean_chord_m_by_lag"][0])
    require(raw_mean_lag1 > 1e-12 and metric_mean_lag1 > 0.0,
            "cannot recover the immutable route scale")
    scale = metric_mean_lag1 / raw_mean_lag1
    model_route = raw_route * scale
    bearing = ActionIntegratedRouteBearing(
        model_route, pose9[history_count - 1], lookahead_m=LOOKAHEAD_M)

    evaluator_route = np.asarray([
        [float(trace["poses"][int(index)]["x"]),
         float(trace["poses"][int(index)]["z"])]
        for index in route_indices
    ], dtype=np.float64)
    evaluator_arc = cumulative_distance(evaluator_route)
    state_by_source = {
        int(source): state for state, source in enumerate(route_indices)}
    readout_by_query = {
        int(row["query_index"]): row
        for row in result["readouts"]
    }

    initial_yaw = float(trace["poses"][-1]["yaw"])
    previous_yaw = initial_yaw
    rows = []
    for query_row in query["queries"]:
        current_yaw = float(query_row["construction_yaw_rad"])
        delta_yaw = (current_yaw - previous_yaw + math.pi) % (
            2.0 * math.pi) - math.pi
        bearing.advance_executor_yaw(delta_yaw)
        previous_yaw = current_yaw
        if query_row["kind"] != "reverse_route":
            continue
        readout = readout_by_query[int(query_row["query_index"])]
        prediction = bearing.guidance(int(readout["state_index"]))
        true_source = int(
            query_row["history_reference_frame_analysis_only"])
        true_state = state_by_source[true_source]
        reference = sample_polyline(
            evaluator_route,
            evaluator_arc,
            min(float(evaluator_arc[-1]),
                float(evaluator_arc[true_state]) + LOOKAHEAD_M),
        )
        floor = query_row["construction_floor_position"]
        target = world_goal_to_local(
            reference, [float(floor[0]), float(floor[2])], current_yaw)
        if float(np.linalg.norm(target)) <= 1e-9:
            # At the exact route endpoint the benchmark arrival contract, not
            # a route tangent, determines success.  Keep it in the receipt but
            # do not manufacture a direction label.
            error = None
        else:
            error = angle_error_deg(
                np.asarray(prediction.unit_bearing), target)
        rows.append({
            "query_index": int(query_row["query_index"]),
            "predicted_state": int(readout["state_index"]),
            "target_state_analysis_only": int(true_state),
            "executor_delta_yaw_deg": math.degrees(delta_yaw),
            "predicted_unit_bearing": list(prediction.unit_bearing),
            "target_unit_bearing_analysis_only": (
                None if error is None
                else (target / np.linalg.norm(target)).tolist()),
            "bearing_error_deg_analysis_only": error,
            "terminal_tangent_held": prediction.terminal_tangent_held,
        })

    errors = [float(row["bearing_error_deg_analysis_only"])
              for row in rows
              if row["bearing_error_deg_analysis_only"] is not None]
    output = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "post-inference route-bearing mechanism analysis; construction "
            "yaw deltas stand in only for executor efference; no control/SR"),
        "source_route_filter_result_sha256": sha256(
            arguments.route_filter_result),
        "source_runtime_receipts_sha256": sha256(arguments.runtime_receipts),
        "source_query_set_sha256": sha256(arguments.query_set),
        "source_history_trace_sha256": sha256(arguments.history_trace),
        "runtime_habitat_translation_visible": False,
        "runtime_source_address_visible": False,
        "executor_relative_yaw_visible": True,
        "lookahead_m": LOOKAHEAD_M,
        "distance_regime_present": False,
        "bearing_acceptance_gate_present": False,
        "endpoint_fallback_present": False,
        "native_fallback_present": False,
        "scored_bearing_count": len(errors),
        "within_30deg_count": sum(error <= 30.0 for error in errors),
        "within_45deg_count": sum(error <= 45.0 for error in errors),
        "median_bearing_error_deg": median(errors),
        "p90_bearing_error_deg": float(np.percentile(errors, 90.0)),
        "rows": rows,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode()
    path = arguments.out / "route_bearing_analysis.json"
    path.write_bytes(encoded)
    (arguments.out / "route_bearing_analysis.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  route_bearing_analysis.json\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "n": len(errors),
        "within_30deg": output["within_30deg_count"],
        "within_45deg": output["within_45deg_count"],
        "median_deg": output["median_bearing_error_deg"],
        "p90_deg": output["p90_bearing_error_deg"],
        "out": str(path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
