#!/usr/bin/env python3
"""Diagnose a fixed-cadence sparse monocular history route.

This is a post-confirmation development diagnostic.  It changes only the
outgoing history representation: instead of integrating every non-identical
RGB pair, it estimates one direct local SE(2) edge at the already established
eight-frame sparse-depth cadence.  Query motion and the route-tangent reader
remain sealed inputs.

The fixed cadence is selected without evaluator geometry.  Habitat poses are
read only after every visual edge has been estimated, solely to score route
shape and bearing.  No controller, navigation outcome, distance regime,
runtime gate, or fallback is present.
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
from MemNavData.monocular_adjacent_motion import (
    estimate_adjacent_motion,
    invert_planar_motion,
    PlanarMotionReceipt,
)
from MemNavData.monocular_depth_runtime import decode_monocular_depth_payload
from MemNavData.monocular_route_tangent_contract import (
    SEALED_DENSE_HISTORY_INDICES,
)
from MemNavData.path_budgeted_route_compass import tangent_from_local_route
from MemNavData.run_local_monocular_adjacent_motion_audit import (
    checked_json,
    DEPTH_PNG_TRANSPORT_CLIP_M,
    FROZEN_MANIFEST_SHA256,
    mirror_path,
    require,
    sha256,
)
from MemNavData.run_local_monocular_route_compass_audit import (
    angle_between_degrees,
    history_route_contract,
    identity_motion,
    motion_from_payload,
    percentile,
    route_from_chronological,
    safe_median,
    trace_motion,
)


SCHEMA_VERSION = "sparse_monocular_history_route_diagnostic_v1_20260903"
SOURCE_ROUTE_SCHEMA_VERSION = (
    "monocular_route_compass_mechanism_audit_v1_20260903"
)
QUERY_MOTION_SCHEMA_VERSION = "local_monocular_adjacent_motion_audit_v2_20260903"
FIRST_METRIC_FRAME = 40
DEFAULT_FRAME_STRIDE = 8
LOOKAHEAD_M = 2.5
CONTROLLER_RADIUS_M = 2.5


def fixed_keyframe_indices(
    *, anchor: int, frame_count: int, frame_stride: int,
) -> list[int]:
    """Return an evaluator-independent absolute-frame keyframe schedule."""

    anchor = int(anchor)
    frame_count = int(frame_count)
    stride = int(frame_stride)
    if not 0 <= anchor < frame_count - 1:
        raise ValueError("anchor is outside the causal history")
    if stride < 2:
        raise ValueError("frame_stride must be at least two")
    route_start = max(anchor, FIRST_METRIC_FRAME)
    result = [anchor]
    if route_start != anchor:
        result.append(route_start)
    next_index = ((route_start // stride) + 1) * stride
    result.extend(range(next_index, frame_count, stride))
    if result[-1] != frame_count - 1:
        result.append(frame_count - 1)
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise RuntimeError("fixed keyframe schedule is not strictly increasing")
    return result


def _motion_route_with_yaw(
    motions: list[PlanarMotionReceipt],
) -> tuple[np.ndarray, np.ndarray]:
    from MemNavData.se2_projected_route_compass import (
        reconstruct_reverse_route_se2,
    )

    return reconstruct_reverse_route_se2(
        [motion.forward_m for motion in reversed(motions)],
        [motion.left_m for motion in reversed(motions)],
        [motion.yaw_rad for motion in reversed(motions)],
    )


def _wrap(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-route-audit", type=Path, required=True)
    parser.add_argument("--expected-source-route-sha", required=True)
    parser.add_argument("--query-motion-audit", type=Path, required=True)
    parser.add_argument("--expected-query-motion-sha", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--lightglue-dependency-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in SEALED_DENSE_HISTORY_INDICES,
            "history is outside the consumed diagnostic set")
    require(not arguments.out.exists(), "diagnostic output already exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    require(sha256(arguments.source_route_audit)
            == arguments.expected_source_route_sha,
            "source route audit changed")
    require(sha256(arguments.query_motion_audit)
            == arguments.expected_query_motion_sha,
            "query-motion audit changed")

    source = json.loads(
        arguments.source_route_audit.read_text(encoding="utf-8"))
    query_motion = json.loads(
        arguments.query_motion_audit.read_text(encoding="utf-8"))
    require(source.get("schema_version") == SOURCE_ROUTE_SCHEMA_VERSION,
            "wrong source route schema")
    require(query_motion.get("schema_version") == QUERY_MOTION_SCHEMA_VERSION,
            "wrong query-motion schema")
    require(int(source.get("history_index", -1)) == arguments.history_index
            and int(query_motion.get("history_index", -1))
            == arguments.history_index,
            "diagnostic inputs belong to another history")
    require(source.get("query_motion_chain_intact") is True
            and query_motion.get("local_motion_chain_intact") is True,
            "sealed query motion is incomplete")

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
    # Only frame address and image hash are exposed while visual estimates are
    # produced.  Position/yaw fields remain scorer-only below.
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_rows = trace_payload["poses"]
    require(len(trace_rows) == int(item["online_a_steps"]),
            "causal history length changed")
    history_rgb: list[dict[str, Any]] = []
    for row, expected_hash in zip(trace_rows, receipt["rgb_frame_hashes"]):
        frame_index = int(row["step"])
        frame = history / "rgb" / f"{frame_index:06d}.jpg"
        require(sha256(frame) == expected_hash == row["jpg_sha256"],
                f"history RGB changed at frame {frame_index}")
        history_rgb.append({
            "index": frame_index,
            "path": frame,
            "sha256": str(expected_hash),
        })

    anchor = int(source["authorized_anchor"])
    route_contract = history_route_contract(
        anchor=anchor, frame_count=len(history_rgb))
    route_start = int(route_contract["route_start_frame"])
    bridge = bool(route_contract["pre_scale_anchor_bridge"])
    keyframes = fixed_keyframe_indices(
        anchor=anchor,
        frame_count=len(history_rgb),
        frame_stride=arguments.frame_stride,
    )
    keyframe_set = set(keyframes)
    require(keyframes[0] == anchor and keyframes[-1] == len(history_rgb) - 1,
            "keyframe schedule changed its endpoints")

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
            "episode_len": (
                len(history_rgb) + int(query_motion["query_frame_count"]) + 1),
        }, timeout=180), "reset")
    require(reset.get("algo") == "memnav", "wrong runtime server")

    edges: list[dict[str, Any]] = []
    previous_keyframe: dict[str, Any] | None = None
    previous_depth: np.ndarray | None = None
    depth_request_ms: list[float] = []
    visual_motion_ms: list[float] = []
    replay_started = time.perf_counter()
    for frame_index, runtime_row in enumerate(history_rgb):
        materialize = (
            frame_index == FIRST_METRIC_FRAME
            or (frame_index in keyframe_set and frame_index >= route_start)
        )
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
        depth_request_ms.append(
            1000.0 * (time.perf_counter() - request_started))

        if frame_index == FIRST_METRIC_FRAME and bridge:
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
            visual_motion_ms.append(
                1000.0 * (time.perf_counter() - estimate_started))
            reverse = motion_from_payload(visual.get("motion"))
            motion = None if reverse is None else invert_planar_motion(reverse)
            edges.append({
                "previous_index": anchor,
                "current_index": frame_index,
                "source": "inverse_frame40_depth_bridge",
                "accepted": bool(visual["local_motion_validity"]["accepted"]),
                "motion": None if motion is None else motion.audit_dict(),
                "visual": visual,
            })
            previous_keyframe = runtime_row
            previous_depth = current_depth
            continue

        if frame_index not in keyframe_set:
            # The isolated frame-40 receipt initializes scale for a later
            # post-receipt anchor but is not part of that route.
            require(frame_index == FIRST_METRIC_FRAME and not bridge,
                    "unexpected materialized non-keyframe")
            continue

        if previous_keyframe is None:
            require(frame_index == route_start,
                    "first sparse keyframe is not the route origin")
            previous_keyframe = runtime_row
            previous_depth = current_depth
            continue

        require(previous_depth is not None, "previous keyframe depth is absent")
        if previous_keyframe["sha256"] == runtime_row["sha256"]:
            motion = identity_motion()
            edges.append({
                "previous_index": int(previous_keyframe["index"]),
                "current_index": frame_index,
                "source": "exact_rgb_identity",
                "accepted": True,
                "motion": motion.audit_dict(),
                "visual": None,
            })
        else:
            with Image.open(previous_keyframe["path"]) as reference_image:
                raw_width, raw_height = reference_image.size
            estimate_started = time.perf_counter()
            visual = estimate_adjacent_motion(
                reference_path=previous_keyframe["path"],
                query_path=runtime_row["path"],
                reference_metric_depth=previous_depth,
                raw_camera_intrinsic=receipt["camera_intrinsic"],
                matcher=matcher,
                raw_height=raw_height,
                raw_width=raw_width,
                patch_size=14,
                depth_transport_clip_m=DEPTH_PNG_TRANSPORT_CLIP_M,
            )
            visual_motion_ms.append(
                1000.0 * (time.perf_counter() - estimate_started))
            motion = motion_from_payload(visual.get("motion"))
            edges.append({
                "previous_index": int(previous_keyframe["index"]),
                "current_index": frame_index,
                "source": "fixed_stride_rgb_depth_pnp",
                "accepted": bool(visual["local_motion_validity"]["accepted"]),
                "motion": None if motion is None else motion.audit_dict(),
                "visual": visual,
            })
        previous_keyframe = runtime_row
        previous_depth = current_depth

    replay_s = time.perf_counter() - replay_started
    require(len(edges) == len(keyframes) - 1,
            "sparse edge count differs from the fixed schedule")
    first_failure = next(
        (int(edge["current_index"]) for edge in edges
         if not edge["accepted"] or edge["motion"] is None),
        None,
    )
    chain_intact = first_failure is None

    # Inference is now fixed.  Construction poses enter only below.
    predicted_motions = [
        motion_from_payload(edge["motion"]) for edge in edges
    ]
    true_sparse_motions = [
        trace_motion(trace_rows[int(edge["previous_index"])],
                     trace_rows[int(edge["current_index"])])
        for edge in edges
    ]
    dense_indices = [anchor, route_start] if bridge else [anchor]
    dense_indices.extend(range(route_start + 1, len(trace_rows)))
    true_dense_motions = [
        trace_motion(trace_rows[previous], trace_rows[current])
        for previous, current in zip(dense_indices, dense_indices[1:])
    ]
    true_dense_route = route_from_chronological(true_dense_motions)
    true_dense_extent = float(np.linalg.norm(
        np.diff(true_dense_route, axis=0), axis=1).sum())

    route_summary: dict[str, Any] = {
        "available": False,
        "reason": None if chain_intact else "sparse_history_chain_incomplete",
    }
    rows: list[dict[str, Any]] = []
    if chain_intact:
        predicted = [motion for motion in predicted_motions if motion is not None]
        predicted_route, predicted_yaw = _motion_route_with_yaw(predicted)
        true_sparse_route, true_sparse_yaw = _motion_route_with_yaw(
            true_sparse_motions)
        require(predicted_route.shape == true_sparse_route.shape,
                "sparse scorer route shape changed")
        position_errors = np.linalg.norm(
            predicted_route - true_sparse_route, axis=1)
        endpoint_yaw_error = math.degrees(abs(_wrap(
            float(predicted_yaw[-1]) - float(true_sparse_yaw[-1]))))
        compass = tangent_from_local_route(
            predicted_route,
            lookahead_m=LOOKAHEAD_M,
            controller_radius_m=CONTROLLER_RADIUS_M,
        )
        query_motions: list[PlanarMotionReceipt] = [identity_motion()]
        query_motions.extend(
            motion_from_payload(row["visual"].get("motion"))
            for row in query_motion["pairs"])
        require(all(motion is not None for motion in query_motions),
                "query motion disappeared")
        scorer_rows = source["compass_rows"]
        require(len(query_motions) == len(scorer_rows),
                "query/scorer length changed")
        bearing_errors: list[float] = []
        translated_errors: list[float] = []
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
            error = angle_between_degrees(
                np.asarray(readout.unit_bearing),
                np.asarray(oracle["unit_bearing"]))
            bearing_errors.append(error)
            if str(scorer["query_kind"]) == "reverse_route":
                translated_errors.append(error)
            monotone &= readout.projected_progress_m + 1e-9 >= previous_progress
            budget_respected &= (
                readout.projected_progress_m
                <= readout.query_path_length_m + 1e-9
                and readout.progress_increment_m
                <= readout.progress_budget_m + 1e-9)
            previous_progress = readout.projected_progress_m
            rows.append({
                "query_index": int(scorer["query_index"]),
                "query_kind": str(scorer["query_kind"]),
                "bearing_error_deg_analysis_only": error,
                "readout": readout.audit_dict(),
            })
        predicted_extent = float(np.linalg.norm(
            np.diff(predicted_route, axis=0), axis=1).sum())
        true_sparse_extent = float(np.linalg.norm(
            np.diff(true_sparse_route, axis=0), axis=1).sum())
        route_summary = {
            "available": True,
            "reason": None,
            "predicted_route_extent_m": predicted_extent,
            "true_sparse_route_extent_m_analysis_only": true_sparse_extent,
            "true_dense_route_extent_m_analysis_only": true_dense_extent,
            "dense_path_length_bias_fraction": (
                predicted_extent / true_dense_extent - 1.0),
            "keyframe_position_error_median_m_analysis_only": float(
                np.median(position_errors)),
            "keyframe_position_error_p90_m_analysis_only": float(
                np.quantile(position_errors, 0.90)),
            "route_endpoint_error_m_analysis_only": float(
                position_errors[-1]),
            "route_endpoint_yaw_error_deg_analysis_only": endpoint_yaw_error,
            "translated_bearing_count": len(translated_errors),
            "translated_bearing_within_30deg_count": sum(
                value <= 30.0 for value in translated_errors),
            "translated_bearing_error_median_deg_analysis_only": safe_median(
                translated_errors),
            "translated_bearing_error_p90_deg_analysis_only": percentile(
                translated_errors, 0.90),
            "bearing_count": len(bearing_errors),
            "bearing_within_30deg_count": sum(
                value <= 30.0 for value in bearing_errors),
            "bearing_error_median_deg_analysis_only": safe_median(
                bearing_errors),
            "bearing_error_p90_deg_analysis_only": percentile(
                bearing_errors, 0.90),
            "final_projected_progress_m": float(compass.projected_progress_m),
            "query_path_length_m": float(
                rows[-1]["readout"]["query_path_length_m"]),
            "projection_monotone": bool(monotone),
            "cumulative_path_budget_respected": bool(budget_respected),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "post-failure development diagnostic on a consumed history; "
            "fixed-cadence forward-history PnP only; no controller or SR"),
        "history_index": int(arguments.history_index),
        "scene": str(item["scene"]),
        "episode": str(item["episode"]),
        "authorized_anchor": anchor,
        "route_start_frame": route_start,
        "pre_scale_anchor_bridge": bridge,
        "frame_stride": int(arguments.frame_stride),
        "selection_rule": "absolute causal frame index modulo fixed stride",
        "keyframe_count": len(keyframes),
        "keyframe_indices": keyframes,
        "edge_count": len(edges),
        "visual_edge_count": sum(edge["visual"] is not None for edge in edges),
        "identity_edge_count": sum(edge["visual"] is None for edge in edges),
        "accepted_edge_count": sum(edge["accepted"] for edge in edges),
        "chain_intact": chain_intact,
        "first_failure_frame": first_failure,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_history_trace_sha256": sha256(trace_path),
        "source_route_audit_sha256": sha256(arguments.source_route_audit),
        "source_query_motion_audit_sha256": sha256(
            arguments.query_motion_audit),
        "runtime_evaluator_pose_visible": False,
        "metric_depth_sensor_consumed": False,
        "global_pose_consumed": False,
        "wheel_odometry_consumed": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "distance_regime_present": False,
        "visual_runtime_gate_present": False,
        "endpoint_fallback_present": False,
        "native_fallback_present": False,
        "history_replay_s": float(replay_s),
        "depth_request_median_ms": safe_median(depth_request_ms),
        "visual_motion_median_ms": safe_median(visual_motion_ms),
        "route_summary": route_summary,
        "edges": edges,
        "rows": rows,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = arguments.out / "sparse_monocular_history_route.json"
    output.write_bytes(encoded)
    (arguments.out / "sparse_monocular_history_route.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  sparse_monocular_history_route.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": int(arguments.history_index),
        "frame_stride": int(arguments.frame_stride),
        "keyframes": len(keyframes),
        "accepted_edges": result["accepted_edge_count"],
        "chain_intact": chain_intact,
        "route_summary": route_summary,
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
