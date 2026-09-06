#!/usr/bin/env python3
"""Audit an odometry-propagated episodic route tape on RGB only.

The runtime sees the sealed outgoing RGB history followed by a rendered
reverse-facing RGB stream.  Habitat poses and source-frame addresses remain in
the construction receipt and are used only after model inference to measure
progress and bearing error.  No controller, action, success check, fallback,
or distance-dependent branch is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import requests


SCHEMA_VERSION = "local_route_tape_odometry_audit_v1_20260902"
QUERY_SCHEMA_VERSION = "hm3d_reverse_route_odometry_queries_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
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


def mirror_path(mirror_root: Path, remote_path: str) -> Path:
    path = Path(remote_path)
    require(path.is_absolute() and path.parts[:2] == ("/", "scratch"),
            "source path is outside the sealed scratch mirror")
    return mirror_root.joinpath(*path.parts[1:])


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    result = response.json()
    require(isinstance(result, dict), f"{label} returned non-object JSON")
    return result


def rotation_from_pose9(pose9: np.ndarray) -> np.ndarray:
    x, y, z, w = pose9[3:7]
    norm = float(np.linalg.norm([x, y, z, w]))
    require(math.isfinite(norm) and norm > 1e-12,
            "model pose has an invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def model_heading(pose9: np.ndarray) -> float:
    forward = rotation_from_pose9(pose9)[:, 2]
    return math.atan2(float(forward[0]), float(forward[2]))


def model_local_translation(
    first: np.ndarray, second: np.ndarray, scale: float
) -> np.ndarray:
    local = rotation_from_pose9(first).T @ (second[:3] - first[:3])
    return float(scale) * np.asarray([local[2], -local[0]], dtype=np.float64)


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2,
            "polyline must have shape [N>=2,2]")
    edges = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([np.zeros(1), np.cumsum(edges)])


def sample_polyline(points: np.ndarray, cumulative: np.ndarray,
                    arc_m: float) -> np.ndarray:
    arc = float(np.clip(arc_m, 0.0, float(cumulative[-1])))
    if arc >= float(cumulative[-1]):
        return points[-1].copy()
    index = int(np.searchsorted(cumulative, arc, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    lo, hi = float(cumulative[index]), float(cumulative[index + 1])
    if hi <= lo + 1e-12:
        return points[index].copy()
    alpha = (arc - lo) / (hi - lo)
    return (1.0 - alpha) * points[index] + alpha * points[index + 1]


def tangent_heading(points: np.ndarray, cumulative: np.ndarray,
                    progress_m: float) -> float:
    start = sample_polyline(points, cumulative, progress_m)
    end = sample_polyline(
        points, cumulative,
        min(float(cumulative[-1]), float(progress_m) + LOOKAHEAD_M))
    delta = end - start
    if float(np.linalg.norm(delta)) <= 1e-9:
        delta = points[-1] - points[-2]
    require(float(np.linalg.norm(delta)) > 1e-9,
            "route tangent is degenerate")
    return math.atan2(float(delta[0]), float(delta[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in (0, 16, 32, 40),
            "odometry audit is sealed to indices 0,16,32,40")
    require(not arguments.out.exists(), "odometry-audit output exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]
    history = mirror_path(arguments.mirror_root.resolve(),
                          item["online_a_episode"])
    receipt_path = history / "receipt.json"
    trace_path = history / "online_a_trace.json"
    require(sha256(receipt_path) == item["online_a_receipt_sha256"]
            and sha256(trace_path) == item["online_a_trace_sha256"],
            "causal history changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evaluator_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    evaluator_poses = evaluator_trace["poses"]
    require(len(evaluator_poses) == int(item["online_a_steps"]),
            "causal history length changed")
    history_rgb = []
    for row, expected in zip(
            evaluator_poses, receipt["rgb_frame_hashes"]):
        path = history / "rgb" / f"{int(row['step']):06d}.jpg"
        require(sha256(path) == expected == row["jpg_sha256"],
                f"history RGB changed at {row['step']}")
        history_rgb.append(path)

    query_set = json.loads(arguments.query_set.read_text(encoding="utf-8"))
    require(query_set.get("schema_version") == QUERY_SCHEMA_VERSION
            and int(query_set.get("history_index", -1))
            == arguments.history_index,
            "wrong reverse-route query set")
    require(query_set.get("runtime_evaluator_pose_visible") is False,
            "query construction leaked evaluator pose to runtime")
    queries = query_set["queries"]
    require(len(queries) >= 20, "reverse route is too sparse to audit")
    query_root = arguments.query_set.parent
    for row in queries:
        path = query_root / row["rgb"]
        require(sha256(path) == row["rgb_sha256"],
                f"query RGB changed: {row['rgb']}")

    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(history_rgb) + len(queries) + 1,
        }, timeout=120), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")
    for frame_index, frame in enumerate(history_rgb):
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=120), f"history frame {frame_index}")
        require(int(response.get("frame_idx", -1)) == frame_index,
                "history replay index diverged")
    history_receipt = checked_json(session.post(
        f"{base}/causal_pose_trace_query", timeout=120), "history pose trace")
    history_pose9 = np.asarray(history_receipt["pose9"], dtype=np.float64)
    require(history_pose9.shape == (len(history_rgb), 9),
            "model history pose trace has the wrong shape")

    for query_offset, row in enumerate(queries):
        path = query_root / row["rgb"]
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", path.read_bytes(), "image/jpeg")},
            timeout=120), f"query frame {query_offset}")
        require(int(response.get("frame_idx", -1))
                == len(history_rgb) + query_offset,
                "query replay index diverged")
    final_receipt = checked_json(session.post(
        f"{base}/causal_pose_trace_query", timeout=120), "final pose trace")
    all_pose9 = np.asarray(final_receipt["pose9"], dtype=np.float64)
    require(all_pose9.shape == (len(history_rgb) + len(queries), 9),
            "final model pose trace has the wrong shape")
    np.testing.assert_allclose(all_pose9[:len(history_rgb)], history_pose9)
    query_pose9 = all_pose9[len(history_rgb):]
    scale_receipt = final_receipt["longrange_metric_scale"]
    require(scale_receipt.get("available") is True,
            "first-40 model-pose scale is unavailable")
    scale = float(scale_receipt["metric_scale_m_per_raw"])
    require(math.isfinite(scale) and scale > 0.0, "invalid pose scale")

    indices = [int(value) for value in
               query_set["sampled_history_indices_analysis_only"]]
    target = int(query_set["target_history_frame_analysis_only"])
    require(indices[0] == len(history_rgb) - 1 and indices[-1] == target,
            "route index endpoints changed")
    model_route = history_pose9[indices][:, (0, 2)] * scale
    model_route_arc = cumulative_distance(model_route)
    gt_route = np.asarray([
        [float(evaluator_poses[index]["x"]),
         float(evaluator_poses[index]["z"])]
        for index in indices
    ], dtype=np.float64)
    gt_route_arc = cumulative_distance(gt_route)

    previous_pose = history_pose9[-1]
    progress = 0.0
    route_rows = []
    turn_translation = 0.0
    for row, current_pose in zip(queries, query_pose9):
        local_motion = model_local_translation(previous_pose, current_pose, scale)
        forward_advance = max(0.0, float(local_motion[0]))
        if row["kind"] == "turn_in_place":
            turn_translation += float(np.linalg.norm(local_motion))
        progress = min(float(model_route_arc[-1]), progress + forward_advance)
        if row["kind"] == "reverse_route":
            source_index = int(row["history_reference_frame_analysis_only"])
            position_in_route = indices.index(source_index)
            gt_progress = float(gt_route_arc[position_in_route])
            desired_model = tangent_heading(
                model_route, model_route_arc, progress)
            predicted_bearing = wrap(
                desired_model - model_heading(current_pose))
            desired_gt = tangent_heading(
                gt_route, gt_route_arc, gt_progress)
            actual_forward_heading = wrap(
                float(row["construction_yaw_rad"]) + math.pi)
            target_bearing = wrap(desired_gt - actual_forward_heading)
            bearing_error = abs(wrap(predicted_bearing - target_bearing))
            route_rows.append({
                "query_index": int(row["query_index"]),
                "history_reference_frame_analysis_only": source_index,
                "predicted_progress_m": progress,
                "target_progress_m_analysis_only": gt_progress,
                "progress_error_m_analysis_only": progress - gt_progress,
                "model_forward_advance_m": forward_advance,
                "model_lateral_motion_m": float(local_motion[1]),
                "predicted_route_bearing_deg": math.degrees(predicted_bearing),
                "target_route_bearing_deg_analysis_only": math.degrees(
                    target_bearing),
                "bearing_error_deg_analysis_only": math.degrees(
                    bearing_error),
            })
        previous_pose = current_pose

    progress_errors = [abs(row["progress_error_m_analysis_only"])
                       for row in route_rows]
    bearing_errors = [row["bearing_error_deg_analysis_only"]
                      for row in route_rows]
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "RGB-only short-range model-motion audit; evaluator poses score "
            "the readout after inference and no controller/actions/SR run"),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(arguments.query_set),
        "runtime_evaluator_pose_visible": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "distance_gate_present": False,
        "fallback_present": False,
        "metric_scale_m_per_raw": scale,
        "history_route_length_model_m": float(model_route_arc[-1]),
        "history_route_length_analysis_m": float(gt_route_arc[-1]),
        "turn_induced_translation_m": turn_translation,
        "route_observation_count": len(route_rows),
        "final_predicted_progress_m": route_rows[-1]["predicted_progress_m"],
        "final_target_progress_m_analysis_only": route_rows[-1][
            "target_progress_m_analysis_only"],
        "median_absolute_progress_error_m": median(progress_errors),
        "final_absolute_progress_error_m": progress_errors[-1],
        "median_route_bearing_error_deg": median(bearing_errors),
        "route_bearing_within_30deg_count": sum(
            error <= 30.0 for error in bearing_errors),
        "route_bearing_within_45deg_count": sum(
            error <= 45.0 for error in bearing_errors),
        "queries": route_rows,
    }
    arguments.out.mkdir(parents=True)
    raw = {
        "schema_version": "local_route_tape_model_pose_receipt_v1_20260902",
        "source_query_set_sha256": sha256(arguments.query_set),
        "runtime_evaluator_pose_visible": False,
        "history_pose9": history_pose9.tolist(),
        "query_pose9": query_pose9.tolist(),
        "metric_scale_receipt": scale_receipt,
    }
    raw_encoded = (json.dumps(raw, separators=(",", ":"),
                              sort_keys=True) + "\n").encode()
    raw_path = arguments.out / "model_pose_receipt.json"
    raw_path.write_bytes(raw_encoded)
    result["model_pose_receipt_sha256"] = hashlib.sha256(
        raw_encoded).hexdigest()
    result_encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    result_path = arguments.out / "odometry_audit.json"
    result_path.write_bytes(result_encoded)
    (arguments.out / "odometry_audit.json.sha256").write_text(
        hashlib.sha256(result_encoded).hexdigest()
        + "  odometry_audit.json\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "observations": len(route_rows),
        "median_progress_error_m": result[
            "median_absolute_progress_error_m"],
        "final_progress_error_m": result[
            "final_absolute_progress_error_m"],
        "median_bearing_error_deg": result[
            "median_route_bearing_error_deg"],
        "within_30deg": result["route_bearing_within_30deg_count"],
        "out": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
