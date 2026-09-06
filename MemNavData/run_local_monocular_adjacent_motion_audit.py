#!/usr/bin/env python3
"""Audit height-calibrated adjacent-frame visual motion without navigation.

One frozen MemNav/LingBot server causally replays an immutable history and a
controlled reverse-facing RGB sequence.  The client requests only the current
monocular depth payload, estimates adjacent-frame motion with
SuperPoint+LightGlue+PnP, and records the evidence.  Construction poses are
used only after each visual estimate to score it; neither model receives them.

No NavDP controller, navigation action, success threshold, role label, global
pose, wheel odometry, or metric-depth sensor participates in inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Any

import numpy as np
from PIL import Image
import requests

from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
from MemNavData.monocular_route_tangent_contract import (
    SEALED_DENSE_HISTORY_INDICES,
)
from MemNavData.monocular_adjacent_motion import (
    estimate_adjacent_motion,
    integrate_planar_motion,
    PlanarMotionReceipt,
)
from MemNavData.monocular_depth_runtime import (
    decode_monocular_depth_payload,
)


SCHEMA_VERSION = "local_monocular_adjacent_motion_audit_v2_20260903"
QUERY_SCHEMA_VERSIONS = {
    "hm3d_reverse_route_odometry_queries_v1_20260902",
    "hm3d_dense_reverse_route_queries_v2_20260903",
}
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
DEPTH_PNG_TRANSPORT_CLIP_M = 6.5535


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_path(mirror_root: Path, source: str) -> Path:
    path = Path(source)
    require(path.is_absolute() and path.parts[:2] == ("/", "scratch"),
            "source path is outside the frozen scratch mirror")
    return mirror_root.joinpath(*path.parts[1:])


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    result = response.json()
    require(isinstance(result, dict), f"{label} returned non-object JSON")
    require("error" not in result, f"{label} failed: {result.get('error')}")
    return result


def wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def pose_xz(row: dict[str, Any]) -> np.ndarray:
    position = row["construction_floor_position"]
    return np.asarray([float(position[0]), float(position[2])])


def world_delta_in_body(delta_xz: np.ndarray, yaw_rad: float) -> np.ndarray:
    dx, dz = (float(value) for value in np.asarray(delta_xz, dtype=np.float64))
    yaw = float(yaw_rad)
    return np.asarray([
        -dx * math.sin(yaw) - dz * math.cos(yaw),
        -dx * math.cos(yaw) + dz * math.sin(yaw),
    ], dtype=np.float64)


def angular_error(first: float, second: float) -> float:
    return abs(wrap_angle(float(first) - float(second)))


def safe_median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def percentile(values: list[float], quantile: float) -> float | None:
    return (None if not values else
            float(np.quantile(np.asarray(values, dtype=np.float64), quantile)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--lightglue-dependency-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in SEALED_DENSE_HISTORY_INDICES,
            "adjacent-motion audit history is not sealed")
    require(not arguments.out.exists(), "audit output already exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen length manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]
    history = mirror_path(
        arguments.mirror_root.resolve(), item["online_a_episode"])
    receipt_path = history / "receipt.json"
    trace_path = history / "online_a_trace.json"
    require(sha256(receipt_path) == item["online_a_receipt_sha256"]
            and sha256(trace_path) == item["online_a_trace_sha256"],
            "causal history changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_rows = trace["poses"]
    require(len(trace_rows) == int(item["online_a_steps"]),
            "causal history length changed")
    history_rgb: list[Path] = []
    for row, expected_hash in zip(
            trace_rows, receipt["rgb_frame_hashes"]):
        frame = history / "rgb" / f"{int(row['step']):06d}.jpg"
        require(sha256(frame) == expected_hash == row["jpg_sha256"],
                f"history RGB changed at frame {row['step']}")
        history_rgb.append(frame)

    query_payload = json.loads(
        arguments.query_set.read_text(encoding="utf-8"))
    require(query_payload.get("schema_version") in QUERY_SCHEMA_VERSIONS
            and int(query_payload.get("history_index", -1))
            == arguments.history_index,
            "wrong reverse-route query set")
    require(query_payload.get("runtime_evaluator_pose_visible") is False,
            "query-set contract exposed evaluator pose to runtime")
    # Only these fields cross the inference boundary.  Construction positions,
    # yaws, and source addresses remain scorer-only below.
    runtime_queries = [{
        "query_index": int(row["query_index"]),
        "kind": str(row["kind"]),
        "path": arguments.query_set.parent / str(row["rgb"]),
        "sha256": str(row["rgb_sha256"]),
    } for row in query_payload["queries"]]
    require(len(runtime_queries) >= 20, "query sequence is too short")
    for row in runtime_queries:
        require(sha256(row["path"]) == row["sha256"],
                f"query RGB changed at {row['query_index']}")

    matcher = LightGluePointMatcher(
        arguments.lightglue_repo,
        dependency_root=arguments.lightglue_dependency_root,
        device=arguments.device,
        max_keypoints=2048,
        reference_cache_size=4,
    )
    base = f"http://127.0.0.1:{arguments.port}"
    session = requests.Session()
    reset = checked_json(session.post(
        f"{base}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": receipt["camera_intrinsic"],
            "seed": int(receipt["episode_seed"]),
            "episode_len": len(history_rgb) + len(runtime_queries) + 1,
        }, timeout=180), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")

    replay_started = time.perf_counter()
    for frame_index, frame in enumerate(history_rgb):
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=180), f"history frame {frame_index}")
        require(int(response.get("frame_idx", -1)) == frame_index,
                "history replay index diverged")
    history_replay_s = time.perf_counter() - replay_started

    estimates: list[dict[str, Any]] = []
    previous_runtime: dict[str, Any] | None = None
    previous_depth: np.ndarray | None = None
    depth_request_ms: list[float] = []
    visual_motion_ms: list[float] = []
    for query_offset, runtime_row in enumerate(runtime_queries):
        frame_index = len(history_rgb) + query_offset
        request_started = time.perf_counter()
        append = checked_json(session.post(
            f"{base}/memory_step",
            data={"materialize_monocular_depth": "1"},
            files={"image": (
                "image.jpg", runtime_row["path"].read_bytes(), "image/jpeg")},
            timeout=240), f"query frame {query_offset}")
        require(int(append.get("frame_idx", -1)) == frame_index,
                "query replay index diverged")
        require(append.get("image_sha256") == runtime_row["sha256"],
                "server query-image hash changed")
        token = append.get("monocular_depth_transaction_token")
        require(isinstance(token, str), "query append returned no depth token")
        depth_payload = checked_json(session.post(
            f"{base}/monocular_depth_query",
            data={
                "expected_image_sha256": runtime_row["sha256"],
                "expected_frame_index": str(frame_index),
                "monocular_depth_transaction_token": token,
            }, timeout=240), f"query depth {query_offset}")
        depth, depth_metadata = decode_monocular_depth_payload(
            depth_payload, expected_image_sha256=runtime_row["sha256"])
        require(depth_metadata.get("metric_depth_sensor_consumed") is False,
                "depth payload consumed a metric sensor")
        require(depth_metadata.get("scale_state") == "raw_lingbot_metric_depth"
                and depth_metadata.get("scale_valid") is True,
                "query depth lacks a valid height-derived scale")
        depth_request_ms.append(
            1000.0 * (time.perf_counter() - request_started))

        if previous_runtime is not None and previous_depth is not None:
            with Image.open(previous_runtime["path"]) as image:
                raw_width, raw_height = image.size
            estimate_started = time.perf_counter()
            visual = estimate_adjacent_motion(
                reference_path=previous_runtime["path"],
                query_path=runtime_row["path"],
                reference_metric_depth=previous_depth,
                raw_camera_intrinsic=receipt["camera_intrinsic"],
                matcher=matcher,
                raw_height=raw_height,
                raw_width=raw_width,
                patch_size=14,
                depth_transport_clip_m=DEPTH_PNG_TRANSPORT_CLIP_M,
            )
            runtime_ms = 1000.0 * (time.perf_counter() - estimate_started)
            visual_motion_ms.append(runtime_ms)
            estimates.append({
                "pair_index": query_offset - 1,
                "reference_query_index": previous_runtime["query_index"],
                "query_index": runtime_row["query_index"],
                "reference_kind": previous_runtime["kind"],
                "query_kind": runtime_row["kind"],
                "runtime_ms": runtime_ms,
                "visual": visual,
            })
        previous_runtime = runtime_row
        previous_depth = depth

    # Scoring begins only after all model outputs have been fixed above.
    scorer_rows = query_payload["queries"]
    require(len(estimates) == len(scorer_rows) - 1,
            "one visual estimate per adjacent pair was not produced")
    pnp_position = np.zeros(2, dtype=np.float64)
    pnp_yaw = 0.0
    strict_position = np.zeros(2, dtype=np.float64)
    strict_yaw = 0.0
    local_position = np.zeros(2, dtype=np.float64)
    local_yaw = 0.0
    pnp_chain_intact = True
    strict_chain_intact = True
    local_chain_intact = True
    first_pnp_failure = None
    first_strict_reject = None
    first_local_reject = None
    scored: list[dict[str, Any]] = []
    for estimate, previous_gt, current_gt in zip(
            estimates, scorer_rows[:-1], scorer_rows[1:]):
        gt_translation = world_delta_in_body(
            pose_xz(current_gt) - pose_xz(previous_gt),
            float(previous_gt["construction_yaw_rad"]),
        )
        gt_yaw = wrap_angle(
            float(current_gt["construction_yaw_rad"])
            - float(previous_gt["construction_yaw_rad"])
        )
        true_distance = float(np.linalg.norm(gt_translation))
        visual = estimate["visual"]
        motion_payload = visual.get("motion")
        motion = None
        if isinstance(motion_payload, dict):
            motion = PlanarMotionReceipt(
                forward_m=float(motion_payload["forward_m"]),
                left_m=float(motion_payload["left_m"]),
                yaw_rad=float(motion_payload["yaw_rad"]),
                vertical_m=float(motion_payload["vertical_m"]),
            )
        pnp_pose_available = motion is not None
        strict_accepted = bool(visual["certificate"]["accepted"])
        local_accepted = bool(visual["local_motion_validity"]["accepted"])
        if pnp_chain_intact and motion is not None:
            pnp_position, pnp_yaw = integrate_planar_motion(
                pnp_position, pnp_yaw, motion)
        else:
            if pnp_chain_intact:
                first_pnp_failure = int(estimate["pair_index"])
            pnp_chain_intact = False
        if strict_chain_intact and strict_accepted and motion is not None:
            strict_position, strict_yaw = integrate_planar_motion(
                strict_position, strict_yaw, motion)
        else:
            if strict_chain_intact:
                first_strict_reject = int(estimate["pair_index"])
            strict_chain_intact = False
        if local_chain_intact and local_accepted and motion is not None:
            local_position, local_yaw = integrate_planar_motion(
                local_position, local_yaw, motion)
        else:
            if local_chain_intact:
                first_local_reject = int(estimate["pair_index"])
            local_chain_intact = False

        predicted = (
            None if motion is None else
            np.asarray([motion.forward_m, motion.left_m], dtype=np.float64)
        )
        direction_error_deg = None
        if (predicted is not None and true_distance > 0.05
                and float(np.linalg.norm(predicted)) > 1e-6):
            cosine = float(np.dot(predicted, gt_translation)
                           / (np.linalg.norm(predicted) * true_distance))
            direction_error_deg = math.degrees(
                math.acos(float(np.clip(cosine, -1.0, 1.0))))
        scored.append({
            **estimate,
            "ground_truth_analysis_only": {
                "forward_m": float(gt_translation[0]),
                "left_m": float(gt_translation[1]),
                "translation_m": true_distance,
                "yaw_rad": float(gt_yaw),
            },
            "pnp_pose_available": pnp_pose_available,
            "strict_certificate_accepted": strict_accepted,
            "local_motion_accepted": local_accepted,
            "translation_vector_error_m_analysis_only": (
                None if predicted is None else
                float(np.linalg.norm(predicted - gt_translation))
            ),
            "translation_magnitude_error_m_analysis_only": (
                None if motion is None else
                abs(float(motion.translation_m) - true_distance)
            ),
            "yaw_error_deg_analysis_only": (
                None if motion is None else
                math.degrees(angular_error(motion.yaw_rad, gt_yaw))
            ),
            "translation_direction_error_deg_analysis_only": (
                direction_error_deg),
        })

    turn_rows = [row for row in scored
                 if row["ground_truth_analysis_only"]["translation_m"] <= 0.01]
    route_rows = [row for row in scored
                  if row["ground_truth_analysis_only"]["translation_m"] > 0.05]
    pnp_rows = [row for row in scored if row["pnp_pose_available"]]
    strict_rows = [row for row in scored
                   if row["strict_certificate_accepted"]]
    local_rows = [row for row in scored if row["local_motion_accepted"]]
    pnp_route = [row for row in route_rows if row["pnp_pose_available"]]
    strict_route = [row for row in route_rows
                    if row["strict_certificate_accepted"]]
    local_route = [row for row in route_rows
                   if row["local_motion_accepted"]]
    turn_false_translation = [
        float(row["visual"]["motion"]["translation_m"])
        for row in turn_rows if row["pnp_pose_available"]
    ]
    translation_vector_errors = [
        float(row["translation_vector_error_m_analysis_only"])
        for row in pnp_route
    ]
    yaw_errors = [float(row["yaw_error_deg_analysis_only"])
                  for row in pnp_rows]
    direction_errors = [
        float(row["translation_direction_error_deg_analysis_only"])
        for row in pnp_route
        if row["translation_direction_error_deg_analysis_only"] is not None
    ]
    initial_gt = scorer_rows[0]
    final_gt = scorer_rows[-1]
    true_final_position = world_delta_in_body(
        pose_xz(final_gt) - pose_xz(initial_gt),
        float(initial_gt["construction_yaw_rad"]),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "height-calibrated monocular adjacent-motion mechanism audit; "
            "no controller, action, navigation success, role label, metric "
            "depth sensor, global pose, or odometry enters inference"),
        "history_index": int(arguments.history_index),
        "scene": str(item["scene"]),
        "episode": str(item["episode"]),
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(arguments.query_set),
        "runtime_evaluator_pose_visible": False,
        "metric_depth_sensor_consumed": False,
        "global_pose_consumed": False,
        "wheel_odometry_consumed": False,
        "camera_height_m": float(receipt["camera_height_m"]),
        "camera_height_prior_consumed": True,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "history_frame_count": len(history_rgb),
        "query_frame_count": len(runtime_queries),
        "adjacent_pair_count": len(scored),
        "pnp_pose_count": len(pnp_rows),
        "strict_certificate_accept_count": len(strict_rows),
        "local_motion_accept_count": len(local_rows),
        "turn_pair_count": len(turn_rows),
        "route_pair_count": len(route_rows),
        "strict_route_accept_count": len(strict_route),
        "local_route_accept_count": len(local_route),
        "pnp_chain_intact": pnp_chain_intact,
        "strict_chain_intact": strict_chain_intact,
        "local_motion_chain_intact": local_chain_intact,
        "first_pnp_failure_pair": first_pnp_failure,
        "first_strict_reject_pair": first_strict_reject,
        "first_local_motion_reject_pair": first_local_reject,
        "turn_false_translation_median_m": safe_median(
            turn_false_translation),
        "turn_false_translation_p95_m": percentile(
            turn_false_translation, 0.95),
        "turn_false_translation_max_m": (
            None if not turn_false_translation else
            float(max(turn_false_translation))),
        "route_translation_vector_error_median_m": safe_median(
            translation_vector_errors),
        "route_translation_vector_error_p90_m": percentile(
            translation_vector_errors, 0.90),
        "yaw_error_median_deg": safe_median(yaw_errors),
        "yaw_error_p90_deg": percentile(yaw_errors, 0.90),
        "route_direction_error_median_deg": safe_median(direction_errors),
        "route_direction_within_30deg_count": sum(
            error <= 30.0 for error in direction_errors),
        "route_direction_scored_count": len(direction_errors),
        "pnp_final_position_error_m_analysis_only": (
            None if not pnp_chain_intact else
            float(np.linalg.norm(pnp_position - true_final_position))
        ),
        "strict_final_position_error_m_analysis_only": (
            None if not strict_chain_intact else
            float(np.linalg.norm(strict_position - true_final_position))
        ),
        "local_motion_final_position_error_m_analysis_only": (
            None if not local_chain_intact else
            float(np.linalg.norm(local_position - true_final_position))
        ),
        "ground_truth_path_length_m_analysis_only": float(sum(
            row["ground_truth_analysis_only"]["translation_m"]
            for row in scored)),
        "local_motion_predicted_path_length_m": float(sum(
            row["visual"]["motion"]["translation_m"]
            for row in local_rows)),
        "local_motion_scored_path_length_m_analysis_only": float(sum(
            row["ground_truth_analysis_only"]["translation_m"]
            for row in local_rows)),
        "history_replay_s": float(history_replay_s),
        "depth_request_median_ms": safe_median(depth_request_ms),
        "visual_motion_median_ms": safe_median(visual_motion_ms),
        "visual_motion_p90_ms": percentile(visual_motion_ms, 0.90),
        "depth_transport": (
            "uint16_0.1mm_clipped_pixels_excluded_from_pnp"),
        "pairs": scored,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    result_path = arguments.out / "adjacent_motion_audit.json"
    result_path.write_bytes(encoded)
    (arguments.out / "adjacent_motion_audit.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  adjacent_motion_audit.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": int(arguments.history_index),
        "pairs": len(scored),
        "pnp_pose_count": len(pnp_rows),
        "strict_accept_count": len(strict_rows),
        "local_motion_accept_count": len(local_rows),
        "local_motion_chain_intact": local_chain_intact,
        "turn_false_translation_p95_m": result[
            "turn_false_translation_p95_m"],
        "route_translation_vector_error_median_m": result[
            "route_translation_vector_error_median_m"],
        "route_direction_error_median_deg": result[
            "route_direction_error_median_deg"],
        "out": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
