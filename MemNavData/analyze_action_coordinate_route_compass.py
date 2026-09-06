#!/usr/bin/env python3
"""Audit a scale-free action-coordinate compass on a reverse route.

For the controlled causal survey, position differences are the construction
commands that generated each RGB frame.  They are converted once into scalar
translation/yaw receipts; absolute simulator coordinates are retained only by
the scorer.  A deployed runner records the same receipts directly from its
executor.  The compass consumes LingBot route shape plus those action receipts,
never simulator pose, metric depth, DINO thresholds, or success labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from MemNavData.conditional_c_protocol import world_goal_to_local
from MemNavData.episodic_route_filter import ActionCoordinateRouteCompass


SCHEMA_VERSION = "action_coordinate_route_compass_audit_v1_20260902"
QUERY_SCHEMA_VERSION = "hm3d_reverse_route_odometry_queries_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
LOOKAHEAD_M = 2.5
CONTROLLER_RADIUS_M = 2.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_path(mirror_root: Path, remote_path: str) -> Path:
    path = Path(remote_path)
    require(path.is_absolute() and path.parts[:2] == ("/", "scratch"),
            "source path is outside the sealed scratch mirror")
    return mirror_root.joinpath(*path.parts[1:])


def sample_polyline(
    points: np.ndarray, cumulative: np.ndarray, arc_m: float,
) -> np.ndarray:
    arc = float(np.clip(arc_m, 0.0, float(cumulative[-1])))
    if arc >= float(cumulative[-1]):
        return points[-1].copy()
    segment = int(np.searchsorted(cumulative, arc, side="right") - 1)
    segment = min(max(segment, 0), len(points) - 2)
    while (segment < len(points) - 2
           and cumulative[segment + 1]
           <= cumulative[segment] + 1e-12):
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


def load_history_pose9(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as payload:
            pose9 = payload["pose9"].astype(np.float64)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pose9 = np.asarray(payload["history_pose9"], dtype=np.float64)
    require(pose9.ndim == 2 and pose9.shape[1] == 9
            and np.isfinite(pose9).all(), "model pose receipt is malformed")
    return pose9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--authorized-anchor", type=int, required=True)
    parser.add_argument("--model-pose-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(not args.out.exists(), "output exists")
    require(sha256(args.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    item = manifest["episodes"][args.history_index]
    history = mirror_path(args.mirror_root.resolve(), item["online_a_episode"])
    trace_path = history / "online_a_trace.json"
    receipt_path = history / "receipt.json"
    require(sha256(trace_path) == item["online_a_trace_sha256"]
            and sha256(receipt_path) == item["online_a_receipt_sha256"],
            "causal history changed")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey",
            "action-receipt reconstruction is valid only for the survey")
    survey = receipt["survey_contract"]
    maximum_translation = float(survey["translation_step_m"])
    maximum_yaw = math.radians(float(survey["maximum_yaw_step_deg"]))

    query = json.loads(args.query_set.read_text(encoding="utf-8"))
    require(query.get("schema_version") == QUERY_SCHEMA_VERSION
            and int(query.get("history_index", -1)) == args.history_index,
            "wrong reverse-route query set")
    require(query.get("runtime_evaluator_pose_visible") is False,
            "query runtime boundary changed")
    poses = trace["poses"]
    history_count = len(poses)
    target = int(args.authorized_anchor)
    require(0 <= target <= int(query["target_history_frame_analysis_only"]),
            "authorized anchor does not cover the constructed return route")
    route_indices = np.arange(
        history_count - 1, target - 1, -1, dtype=np.int64)

    evaluator_positions = np.asarray([
        [float(poses[index]["x"]), float(poses[index]["z"])]
        for index in route_indices
    ], dtype=np.float64)
    action_edges = np.linalg.norm(
        np.diff(evaluator_positions, axis=0), axis=1)
    require(float(np.max(action_edges)) <= maximum_translation + 1e-5,
            "survey translation command exceeds its frozen contract")
    for edge, newer, older in zip(
            action_edges, route_indices[:-1], route_indices[1:]):
        if float(edge) <= 1e-9:
            delta_yaw = (float(poses[int(newer)]["yaw"])
                         - float(poses[int(older)]["yaw"]) + math.pi) % (
                             2.0 * math.pi) - math.pi
            require(abs(delta_yaw) <= maximum_yaw + 1e-6,
                    "survey yaw command exceeds its frozen contract")
    action_arc = np.concatenate([
        np.zeros(1, dtype=np.float64), np.cumsum(action_edges),
    ])

    pose9 = load_history_pose9(args.model_pose_receipt)
    require(pose9.shape[0] >= history_count,
            "model pose receipt is shorter than causal history")
    model_route = pose9[:history_count][route_indices][:, (0, 2)]
    compass = ActionCoordinateRouteCompass(
        model_route,
        action_arc,
        pose9[history_count - 1],
        lookahead_m=LOOKAHEAD_M,
        controller_radius_m=CONTROLLER_RADIUS_M,
    )

    state_by_source = {
        int(source): state for state, source in enumerate(route_indices)}
    previous_floor = np.asarray([
        float(poses[-1]["x"]), float(poses[-1]["z"]),
    ], dtype=np.float64)
    previous_yaw = float(poses[-1]["yaw"])
    rows = []
    position_errors = []
    bearing_errors = []
    for query_row in query["queries"]:
        current_yaw = float(query_row["construction_yaw_rad"])
        yaw_receipt = (current_yaw - previous_yaw + math.pi) % (
            2.0 * math.pi) - math.pi
        previous_yaw = current_yaw
        current_floor = np.asarray(
            query_row["construction_floor_position"], dtype=np.float64)[[0, 2]]
        if query_row["kind"] == "reverse_route":
            translation_receipt = float(np.linalg.norm(
                current_floor - previous_floor))
            previous_floor = current_floor
        else:
            translation_receipt = 0.0
        readout = compass.advance(
            executed_translation_m=translation_receipt,
            executed_yaw_rad=yaw_receipt,
        )
        record = {
            **readout.audit_dict(),
            "query_index": int(query_row["query_index"]),
            "action_class": (
                "translation" if translation_receipt > 0.0 else "turn"),
            "executed_translation_receipt_m": translation_receipt,
            "executed_yaw_receipt_rad": yaw_receipt,
        }
        if query_row["kind"] == "reverse_route":
            predicted_floor = sample_polyline(
                evaluator_positions, action_arc,
                readout.action_progress_m)
            position_error = float(np.linalg.norm(
                predicted_floor - current_floor))
            true_source = int(
                query_row["history_reference_frame_analysis_only"])
            true_state = state_by_source[true_source]
            true_arc = float(action_arc[true_state])
            reference = sample_polyline(
                evaluator_positions, action_arc,
                min(float(action_arc[-1]), true_arc + LOOKAHEAD_M),
            )
            target_local = world_goal_to_local(
                reference, current_floor, current_yaw)
            if float(np.linalg.norm(target_local)) <= 1e-9:
                bearing_error = None
            else:
                bearing_error = angle_error_deg(
                    np.asarray(readout.unit_bearing), target_local)
                bearing_errors.append(float(bearing_error))
            position_errors.append(position_error)
            record.update({
                "target_source_index_analysis_only": true_source,
                "position_error_m_analysis_only": position_error,
                "bearing_error_deg_analysis_only": bearing_error,
            })
        rows.append(record)

    require(position_errors and bearing_errors, "audit has no scored route rows")
    output = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "controlled-survey action-coordinate mechanism audit; absolute "
            "construction poses create and score scalar executor receipts but "
            "are not visible to the compass; no control or SR"),
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_history_trace_sha256": sha256(trace_path),
        "source_query_set_sha256": sha256(args.query_set),
        "source_model_pose_receipt_sha256": sha256(args.model_pose_receipt),
        "history_index": int(args.history_index),
        "authorized_anchor": target,
        "scene": str(item["scene"]),
        "episode": str(item["episode"]),
        "route_action_extent_m": float(action_arc[-1]),
        "query_executed_translation_m": float(sum(
            row["executed_translation_receipt_m"] for row in rows)),
        "monocular_metric_scale_consumed": False,
        "metric_depth_sensor_consumed": False,
        "dino_similarity_consumed": False,
        "distance_regime_present": False,
        "visual_acceptance_gate_present_after_initialization": False,
        "endpoint_fallback_present": False,
        "native_fallback_present": False,
        "absolute_pose_visible_to_compass": False,
        "executor_translation_and_yaw_receipts_visible": True,
        "position_count": len(position_errors),
        "within_1m_count": sum(value <= 1.0 for value in position_errors),
        "median_position_error_m": median(position_errors),
        "final_position_error_m": position_errors[-1],
        "maximum_position_error_m": max(position_errors),
        "bearing_count": len(bearing_errors),
        "within_30deg_count": sum(value <= 30.0 for value in bearing_errors),
        "within_45deg_count": sum(value <= 45.0 for value in bearing_errors),
        "median_bearing_error_deg": median(bearing_errors),
        "p90_bearing_error_deg": float(np.percentile(
            bearing_errors, 90.0)),
        "rows": rows,
    }
    args.out.mkdir(parents=True)
    encoded = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode()
    result = args.out / "action_coordinate_route_compass_audit.json"
    result.write_bytes(encoded)
    (args.out / "action_coordinate_route_compass_audit.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  action_coordinate_route_compass_audit.json\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "history_index": int(args.history_index),
        "position": {
            "within_1m": output["within_1m_count"],
            "n": output["position_count"],
            "median_m": output["median_position_error_m"],
            "final_m": output["final_position_error_m"],
        },
        "bearing": {
            "within_30deg": output["within_30deg_count"],
            "n": output["bearing_count"],
            "median_deg": output["median_bearing_error_deg"],
            "p90_deg": output["p90_bearing_error_deg"],
        },
        "out": str(result.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
