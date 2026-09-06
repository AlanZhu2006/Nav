#!/usr/bin/env python3
"""Audit a complete monocular history route and route-conditioned bearing.

The runtime half replays one immutable outgoing RGB history. Starting when the
first-40 camera-height receipt becomes available, adjacent RGB/depth-PnP
estimates form a local SE(2) route tape. Exact duplicate JPEGs contribute an
identity edge. If the CEC-authorized anchor predates frame 40, frame 40 depth
is used once in the reverse direction and the resulting transform is inverted.

The already sealed dense-return motion audit supplies live local increments.
Only after both tapes are fixed are construction poses read to score route
shape, monotone projection, and bearing. No controller or navigation outcome
is computed here.
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
    invert_planar_motion,
    PlanarMotionReceipt,
)
from MemNavData.monocular_depth_runtime import decode_monocular_depth_payload
from MemNavData.run_local_monocular_adjacent_motion_audit import (
    DEPTH_PNG_TRANSPORT_CLIP_M,
    FROZEN_MANIFEST_SHA256,
    angular_error,
    checked_json,
    mirror_path,
    pose_xz,
    require,
    sha256,
    world_delta_in_body,
    wrap_angle,
)
from MemNavData.se2_projected_route_compass import (
    reconstruct_reverse_route_se2,
    SE2_LOCAL_ODOMETRY_MODEL,
    SE2ProjectedRouteCompass,
)


SCHEMA_VERSION = "monocular_route_compass_mechanism_audit_v1_20260903"
QUERY_SCHEMA_VERSION = "hm3d_dense_reverse_route_queries_v2_20260903"
QUERY_MOTION_SCHEMA_VERSION = "local_monocular_adjacent_motion_audit_v2_20260903"
FIRST_METRIC_FRAME = 40
LOOKAHEAD_M = 2.5
CONTROLLER_RADIUS_M = 2.5


def history_route_contract(
    *, anchor: int, frame_count: int,
) -> dict[str, int | bool]:
    """Return the fixed anchor-to-tail reconstruction schedule.

    A pre-receipt anchor needs one inverse bridge from frame 40.  An anchor at
    or after frame 40 is itself the first metric route state and needs no
    bridge.  Frame 40 is still queried once to attest the scale receipt.
    """

    if not 0 <= int(anchor) < int(frame_count) - 1:
        raise ValueError("authorized anchor is outside the causal history")
    route_start = max(int(anchor), FIRST_METRIC_FRAME)
    bridge = int(anchor) < FIRST_METRIC_FRAME
    expected_edges = (
        int(frame_count) - FIRST_METRIC_FRAME
        if bridge else int(frame_count) - int(anchor) - 1
    )
    return {
        "route_start_frame": route_start,
        "pre_scale_anchor_bridge": bridge,
        "expected_edge_count": expected_edges,
    }


def motion_from_payload(payload: dict[str, Any] | None) -> PlanarMotionReceipt | None:
    if not isinstance(payload, dict):
        return None
    return PlanarMotionReceipt(
        forward_m=float(payload["forward_m"]),
        left_m=float(payload["left_m"]),
        yaw_rad=float(payload["yaw_rad"]),
        vertical_m=float(payload["vertical_m"]),
    )


def identity_motion() -> PlanarMotionReceipt:
    return PlanarMotionReceipt(
        forward_m=0.0,
        left_m=0.0,
        yaw_rad=0.0,
        vertical_m=0.0,
    )


def angle_between_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first.shape != (2,) or second.shape != (2,):
        raise ValueError("bearings must have shape [2]")
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise ValueError("bearings must be nonzero")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def true_motion(previous: dict[str, Any], current: dict[str, Any]) -> PlanarMotionReceipt:
    translation = world_delta_in_body(
        pose_xz(current) - pose_xz(previous),
        float(previous["construction_yaw_rad"]),
    )
    return PlanarMotionReceipt(
        forward_m=float(translation[0]),
        left_m=float(translation[1]),
        yaw_rad=wrap_angle(
            float(current["construction_yaw_rad"])
            - float(previous["construction_yaw_rad"])),
        vertical_m=0.0,
    )


def trace_motion(previous: dict[str, Any], current: dict[str, Any]) -> PlanarMotionReceipt:
    translation = world_delta_in_body(
        pose_xz({"construction_floor_position": [
            float(current["x"]), float(current["y"]), float(current["z"])]})
        - pose_xz({"construction_floor_position": [
            float(previous["x"]), float(previous["y"]), float(previous["z"])]}),
        float(previous["yaw"]),
    )
    return PlanarMotionReceipt(
        forward_m=float(translation[0]),
        left_m=float(translation[1]),
        yaw_rad=wrap_angle(float(current["yaw"]) - float(previous["yaw"])),
        vertical_m=0.0,
    )


def route_from_chronological(
    motions: list[PlanarMotionReceipt],
) -> np.ndarray:
    return reconstruct_reverse_route_se2(
        [motion.forward_m for motion in reversed(motions)],
        [motion.left_m for motion in reversed(motions)],
        [motion.yaw_rad for motion in reversed(motions)],
    )[0]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def safe_median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--query-motion-audit", type=Path, required=True)
    parser.add_argument("--expected-query-motion-sha", required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--lightglue-dependency-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in SEALED_DENSE_HISTORY_INDICES,
            "route audit history is not sealed")
    require(not arguments.out.exists(), "audit output already exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    require(sha256(arguments.query_motion_audit)
            == arguments.expected_query_motion_sha,
            "query-motion audit changed")
    require(sha256(arguments.authorization_receipt)
            == arguments.expected_authorization_sha,
            "authorization receipt changed")

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
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_rows = trace_payload["poses"]
    require(len(trace_rows) == int(item["online_a_steps"]),
            "causal history length changed")
    history_rgb: list[dict[str, Any]] = []
    for row, expected_hash in zip(trace_rows, receipt["rgb_frame_hashes"]):
        frame = history / "rgb" / f"{int(row['step']):06d}.jpg"
        require(sha256(frame) == expected_hash == row["jpg_sha256"],
                f"history RGB changed at frame {row['step']}")
        history_rgb.append({
            "index": int(row["step"]),
            "path": frame,
            "sha256": str(expected_hash),
        })

    query_payload = json.loads(arguments.query_set.read_text(encoding="utf-8"))
    require(query_payload.get("schema_version") == QUERY_SCHEMA_VERSION
            and int(query_payload.get("history_index", -1))
            == arguments.history_index,
            "wrong dense reverse-route query set")
    require(query_payload.get("runtime_evaluator_pose_visible") is False,
            "query set exposed evaluator pose")
    query_rows = query_payload["queries"]
    for row in query_rows:
        path = arguments.query_set.parent / str(row["rgb"])
        require(sha256(path) == row["rgb_sha256"],
                f"query RGB changed at {row['query_index']}")

    query_motion_payload = json.loads(
        arguments.query_motion_audit.read_text(encoding="utf-8"))
    require(query_motion_payload.get("schema_version")
            == QUERY_MOTION_SCHEMA_VERSION,
            "wrong query-motion audit schema")
    require(int(query_motion_payload.get("history_index", -1))
            == arguments.history_index,
            "query-motion audit belongs to another history")
    require(query_motion_payload.get("source_query_set_sha256")
            == sha256(arguments.query_set),
            "query-motion audit used another query set")
    require(query_motion_payload.get("local_motion_chain_intact") is True,
            "query local-motion chain is incomplete")
    require(query_motion_payload.get("runtime_evaluator_pose_visible") is False
            and query_motion_payload.get("metric_depth_sensor_consumed") is False
            and query_motion_payload.get("global_pose_consumed") is False
            and query_motion_payload.get("wheel_odometry_consumed") is False,
            "query-motion audit crossed its information boundary")

    authorization = json.loads(
        arguments.authorization_receipt.read_text(encoding="utf-8"))
    require(int(authorization.get("history_index", -1))
            == arguments.history_index,
            "authorization receipt belongs to another history")
    anchor = int(authorization["authorized_anchor"])
    route_contract = history_route_contract(
        anchor=anchor, frame_count=len(history_rgb))
    route_start_frame = int(route_contract["route_start_frame"])
    pre_scale_anchor_bridge = bool(
        route_contract["pre_scale_anchor_bridge"])

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
            "episode_len": len(history_rgb) + len(query_rows) + 1,
        }, timeout=180), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")

    history_edges: list[dict[str, Any]] = []
    previous_depth: np.ndarray | None = None
    history_depth_request_ms: list[float] = []
    history_visual_motion_ms: list[float] = []
    replay_started = time.perf_counter()
    for frame_index, runtime_row in enumerate(history_rgb):
        # Frame 40 attests the immutable height-derived scale. The route tape
        # itself begins at max(anchor, 40), so a post-receipt anchor never
        # causes irrelevant earlier motion to enter the reconstructed route.
        materialize = (
            frame_index == FIRST_METRIC_FRAME
            or frame_index >= route_start_frame)
        request_started = time.perf_counter()
        append = checked_json(session.post(
            f"{base}/memory_step",
            data={"materialize_monocular_depth": "1" if materialize else "0"},
            files={"image": (
                "image.jpg", runtime_row["path"].read_bytes(), "image/jpeg")},
            timeout=240), f"history frame {frame_index}")
        require(int(append.get("frame_idx", -1)) == frame_index,
                "history replay index diverged")
        require(append.get("image_sha256") == runtime_row["sha256"],
                "server history-image hash changed")
        if not materialize:
            continue

        token = append.get("monocular_depth_transaction_token")
        require(isinstance(token, str), "history append returned no depth token")
        depth_payload = checked_json(session.post(
            f"{base}/monocular_depth_query",
            data={
                "expected_image_sha256": runtime_row["sha256"],
                "expected_frame_index": str(frame_index),
                "monocular_depth_transaction_token": token,
            }, timeout=240), f"history depth {frame_index}")
        current_depth, depth_metadata = decode_monocular_depth_payload(
            depth_payload, expected_image_sha256=runtime_row["sha256"])
        require(depth_metadata.get("metric_depth_sensor_consumed") is False,
                "history depth consumed a metric sensor")
        require(depth_metadata.get("scale_state")
                == "raw_lingbot_metric_depth"
                and depth_metadata.get("scale_valid") is True,
                "history depth lacks a valid height-derived scale")
        history_depth_request_ms.append(
            1000.0 * (time.perf_counter() - request_started))

        if frame_index == FIRST_METRIC_FRAME and pre_scale_anchor_bridge:
            reference = runtime_row
            query = history_rgb[anchor]
            with Image.open(reference["path"]) as reference_image:
                raw_width, raw_height = reference_image.size
            estimate_started = time.perf_counter()
            visual = estimate_adjacent_motion(
                reference_path=reference["path"],
                query_path=query["path"],
                reference_metric_depth=current_depth,
                raw_camera_intrinsic=receipt["camera_intrinsic"],
                matcher=matcher,
                raw_height=raw_height,
                raw_width=raw_width,
                patch_size=14,
                depth_transport_clip_m=DEPTH_PNG_TRANSPORT_CLIP_M,
            )
            history_visual_motion_ms.append(
                1000.0 * (time.perf_counter() - estimate_started))
            reverse_motion = motion_from_payload(visual.get("motion"))
            edge_motion = (
                None if reverse_motion is None
                else invert_planar_motion(reverse_motion))
            history_edges.append({
                "previous_index": anchor,
                "current_index": frame_index,
                "source": "inverse_frame40_depth_bridge",
                "accepted": bool(
                    visual["local_motion_validity"]["accepted"]),
                "motion": None if edge_motion is None else edge_motion.audit_dict(),
                "visual": visual,
            })
        elif frame_index == route_start_frame:
            # A metric-range anchor is the route origin, not an edge.
            previous_depth = current_depth
            continue
        elif frame_index > route_start_frame:
            previous = history_rgb[frame_index - 1]
            if previous["sha256"] == runtime_row["sha256"]:
                edge_motion = identity_motion()
                history_edges.append({
                    "previous_index": frame_index - 1,
                    "current_index": frame_index,
                    "source": "exact_rgb_identity",
                    "accepted": True,
                    "motion": edge_motion.audit_dict(),
                    "visual": None,
                })
            else:
                require(previous_depth is not None,
                        "previous metric depth is absent")
                with Image.open(previous["path"]) as previous_image:
                    raw_width, raw_height = previous_image.size
                estimate_started = time.perf_counter()
                visual = estimate_adjacent_motion(
                    reference_path=previous["path"],
                    query_path=runtime_row["path"],
                    reference_metric_depth=previous_depth,
                    raw_camera_intrinsic=receipt["camera_intrinsic"],
                    matcher=matcher,
                    raw_height=raw_height,
                    raw_width=raw_width,
                    patch_size=14,
                    depth_transport_clip_m=DEPTH_PNG_TRANSPORT_CLIP_M,
                )
                history_visual_motion_ms.append(
                    1000.0 * (time.perf_counter() - estimate_started))
                edge_motion = motion_from_payload(visual.get("motion"))
                history_edges.append({
                    "previous_index": frame_index - 1,
                    "current_index": frame_index,
                    "source": "adjacent_rgb_depth_pnp",
                    "accepted": bool(
                        visual["local_motion_validity"]["accepted"]),
                    "motion": (
                        None if edge_motion is None
                        else edge_motion.audit_dict()),
                    "visual": visual,
                })
        else:
            # For a post-receipt anchor, this isolated frame-40 request proves
            # the scale state but is not part of the anchor-to-tail route.
            require(frame_index == FIRST_METRIC_FRAME,
                    "unexpected metric-depth frame before route origin")
            previous_depth = None
            continue
        previous_depth = current_depth
    history_replay_s = time.perf_counter() - replay_started

    # All model outputs are fixed above. Construction poses enter only here.
    require(len(history_edges) == int(route_contract["expected_edge_count"]),
            "history edge count changed")
    predicted_history: list[PlanarMotionReceipt] = []
    true_history: list[PlanarMotionReceipt] = []
    history_vector_errors: list[float] = []
    history_yaw_errors: list[float] = []
    first_history_failure = None
    for edge in history_edges:
        motion = motion_from_payload(edge["motion"])
        if not edge["accepted"] or motion is None:
            if first_history_failure is None:
                first_history_failure = int(edge["current_index"])
            continue
        previous_gt = trace_rows[int(edge["previous_index"])]
        current_gt = trace_rows[int(edge["current_index"])]
        truth = trace_motion(previous_gt, current_gt)
        predicted_history.append(motion)
        true_history.append(truth)
        history_vector_errors.append(float(np.linalg.norm(
            np.asarray([motion.forward_m, motion.left_m])
            - np.asarray([truth.forward_m, truth.left_m]))))
        history_yaw_errors.append(math.degrees(
            angular_error(motion.yaw_rad, truth.yaw_rad)))

    history_chain_intact = first_history_failure is None
    query_edges = query_motion_payload["pairs"]
    predicted_query = [
        motion_from_payload(row["visual"].get("motion"))
        for row in query_edges
    ]
    require(all(row["local_motion_accepted"] for row in query_edges)
            and all(motion is not None for motion in predicted_query),
            "sealed query-motion chain changed")
    predicted_query = [motion for motion in predicted_query if motion is not None]
    true_query = [
        true_motion(previous, current)
        for previous, current in zip(query_rows, query_rows[1:])
    ]
    require(len(predicted_query) == len(true_query),
            "query motion count changed")

    route_summary: dict[str, Any] = {
        "available": False,
        "reason": (
            None if history_chain_intact else
            "history_local_motion_chain_incomplete"),
    }
    compass_rows: list[dict[str, Any]] = []
    if history_chain_intact:
        predicted_route = route_from_chronological(predicted_history)
        true_route = route_from_chronological(true_history)
        require(predicted_route.shape == true_route.shape,
                "predicted and scorer routes differ in shape")
        route_position_errors = np.linalg.norm(
            predicted_route - true_route, axis=1)
        predicted_compass = SE2ProjectedRouteCompass(
            predicted_route,
            lookahead_m=LOOKAHEAD_M,
            controller_radius_m=CONTROLLER_RADIUS_M,
            motion_model=SE2_LOCAL_ODOMETRY_MODEL,
        )
        scorer_compass = SE2ProjectedRouteCompass(
            true_route,
            lookahead_m=LOOKAHEAD_M,
            controller_radius_m=CONTROLLER_RADIUS_M,
            motion_model=SE2_LOCAL_ODOMETRY_MODEL,
        )
        pairs = [(identity_motion(), identity_motion(), query_rows[0])]
        pairs.extend(zip(predicted_query, true_query, query_rows[1:]))
        bearing_errors: list[float] = []
        translated_bearing_errors: list[float] = []
        progress_errors: list[float] = []
        monotone = True
        nonexpansive = True
        previous_progress = 0.0
        for query_index, (predicted, truth, query_row) in enumerate(pairs):
            predicted_readout = predicted_compass.advance_local_se2(
                executed_forward_m=predicted.forward_m,
                executed_left_m=predicted.left_m,
                executed_yaw_rad=predicted.yaw_rad,
            )
            scorer_readout = scorer_compass.advance_local_se2(
                executed_forward_m=truth.forward_m,
                executed_left_m=truth.left_m,
                executed_yaw_rad=truth.yaw_rad,
            )
            bearing_error = angle_between_degrees(
                np.asarray(predicted_readout.unit_bearing),
                np.asarray(scorer_readout.unit_bearing))
            progress_error = abs(
                predicted_readout.projected_progress_m
                - scorer_readout.projected_progress_m)
            bearing_errors.append(bearing_error)
            if str(query_row["kind"]) == "reverse_route":
                translated_bearing_errors.append(bearing_error)
            progress_errors.append(progress_error)
            monotone &= (
                predicted_readout.projected_progress_m + 1e-9
                >= previous_progress)
            nonexpansive &= (
                predicted_readout.progress_increment_m
                <= predicted_readout.progress_budget_m + 1e-9)
            previous_progress = predicted_readout.projected_progress_m
            compass_rows.append({
                "query_index": query_index,
                "query_kind": str(query_row["kind"]),
                "predicted": predicted_readout.audit_dict(),
                "ground_truth_analysis_only": scorer_readout.audit_dict(),
                "bearing_error_deg_analysis_only": bearing_error,
                "progress_error_m_analysis_only": progress_error,
            })
        route_summary = {
            "available": True,
            "reason": None,
            "route_state_count": int(len(predicted_route)),
            "predicted_route_extent_m": float(
                np.linalg.norm(np.diff(predicted_route, axis=0), axis=1).sum()),
            "ground_truth_route_extent_m_analysis_only": float(
                np.linalg.norm(np.diff(true_route, axis=0), axis=1).sum()),
            "route_position_error_median_m_analysis_only": float(
                np.median(route_position_errors)),
            "route_position_error_p90_m_analysis_only": float(
                np.quantile(route_position_errors, 0.90)),
            "route_endpoint_error_m_analysis_only": float(
                route_position_errors[-1]),
            "bearing_count": len(bearing_errors),
            "bearing_error_median_deg_analysis_only": safe_median(
                bearing_errors),
            "bearing_error_p90_deg_analysis_only": percentile(
                bearing_errors, 0.90),
            "bearing_within_30deg_count": sum(
                value <= 30.0 for value in bearing_errors),
            "translated_bearing_count": len(translated_bearing_errors),
            "translated_bearing_error_median_deg_analysis_only": safe_median(
                translated_bearing_errors),
            "translated_bearing_error_p90_deg_analysis_only": percentile(
                translated_bearing_errors, 0.90),
            "translated_bearing_within_30deg_count": sum(
                value <= 30.0 for value in translated_bearing_errors),
            "progress_error_median_m_analysis_only": safe_median(
                progress_errors),
            "progress_error_final_m_analysis_only": float(progress_errors[-1]),
            "projection_monotone": bool(monotone),
            "projection_nonexpansive": bool(nonexpansive),
            "controller_pointgoal_norm_m": CONTROLLER_RADIUS_M,
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "height-calibrated monocular history-route and dense-return "
            "mechanism audit; no controller, navigation outcome, role label, "
            "metric depth sensor, global pose, or odometry enters inference"),
        "history_index": int(arguments.history_index),
        "scene": str(item["scene"]),
        "episode": str(item["episode"]),
        "authorized_anchor": anchor,
        "history_route_start_frame": route_start_frame,
        "pre_scale_anchor_bridge": pre_scale_anchor_bridge,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_history_trace_sha256": sha256(trace_path),
        "source_query_set_sha256": sha256(arguments.query_set),
        "source_query_motion_audit_sha256": sha256(
            arguments.query_motion_audit),
        "source_authorization_receipt_sha256": sha256(
            arguments.authorization_receipt),
        "runtime_evaluator_pose_visible": False,
        "metric_depth_sensor_consumed": False,
        "global_pose_consumed": False,
        "wheel_odometry_consumed": False,
        "camera_height_prior_consumed": True,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "history_frame_count": len(history_rgb),
        "history_edge_count": len(history_edges),
        "history_identity_edge_count": sum(
            edge["source"] == "exact_rgb_identity" for edge in history_edges),
        "history_visual_edge_count": sum(
            edge["source"] != "exact_rgb_identity" for edge in history_edges),
        "history_local_motion_accept_count": sum(
            edge["accepted"] for edge in history_edges),
        "history_local_motion_chain_intact": history_chain_intact,
        "first_history_local_motion_failure_frame": first_history_failure,
        "history_translation_vector_error_median_m_analysis_only": (
            safe_median(history_vector_errors)),
        "history_translation_vector_error_p90_m_analysis_only": (
            percentile(history_vector_errors, 0.90)),
        "history_yaw_error_median_deg_analysis_only": safe_median(
            history_yaw_errors),
        "history_yaw_error_p90_deg_analysis_only": percentile(
            history_yaw_errors, 0.90),
        "history_replay_s": float(history_replay_s),
        "history_depth_request_median_ms": safe_median(
            history_depth_request_ms),
        "history_visual_motion_median_ms": safe_median(
            history_visual_motion_ms),
        "history_visual_motion_p90_ms": percentile(
            history_visual_motion_ms, 0.90),
        "query_motion_chain_intact": True,
        "route_compass": route_summary,
        "history_edges": history_edges,
        "compass_rows": compass_rows,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    result_path = arguments.out / "monocular_route_compass_audit.json"
    result_path.write_bytes(encoded)
    (arguments.out / "monocular_route_compass_audit.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  monocular_route_compass_audit.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": int(arguments.history_index),
        "history_edges": len(history_edges),
        "history_accept": result["history_local_motion_accept_count"],
        "history_chain_intact": history_chain_intact,
        "route_compass": route_summary,
        "out": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
