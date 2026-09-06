#!/usr/bin/env python3
"""Audit an action-conditioned visual route filter on a reverse RGB stream.

Runtime receives a sealed causal RGB history, the subsequent RGB observations,
and whether the previous action chunk translated or only turned.  Habitat pose
and the historical source address score the readout only after inference.  No
controller, success detector, distance regime, intervention gate, or fallback
is executed here.
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

from MemNavData.episodic_route_filter import (
    MonotoneRouteFilter,
    derive_route_transition,
)


SCHEMA_VERSION = "local_causal_route_filter_audit_v1_20260902"
QUERY_SCHEMA_VERSION = "hm3d_reverse_route_odometry_queries_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
NOMINAL_TRANSLATION_M = 0.30
TRANSITION_SEARCH_FRAMES = 64


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


def checked_json(response: requests.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    result = response.json()
    require(isinstance(result, dict), f"{label} returned non-object JSON")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in (0, 16, 32),
            "route-filter audit is sealed to indices 0,16,32")
    require(not arguments.out.exists(), "route-filter output exists")
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
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    poses = trace["poses"]
    history_rgb = []
    for row, expected in zip(poses, receipt["rgb_frame_hashes"]):
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
            "query construction leaked evaluator pose")
    require(abs(float(query_set["spacing_m"])
                - NOMINAL_TRANSLATION_M) <= 1e-9,
            "query cadence differs from the frozen control update")
    queries = query_set["queries"]
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
    for index, frame in enumerate(history_rgb):
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", frame.read_bytes(), "image/jpeg")},
            timeout=120), f"history frame {index}")
        require(int(response.get("frame_idx", -1)) == index,
                "history replay index diverged")
    for offset, row in enumerate(queries):
        path = query_root / row["rgb"]
        response = checked_json(session.post(
            f"{base}/memory_step",
            files={"image": ("image.jpg", path.read_bytes(), "image/jpeg")},
            timeout=120), f"query frame {offset}")
        require(int(response.get("frame_idx", -1))
                == len(history_rgb) + offset,
                "query replay index diverged")

    similarity_receipt = checked_json(session.post(
        f"{base}/causal_visual_similarity_query",
        data={"history_count": str(len(history_rgb))}, timeout=180),
        "DINO similarity")
    similarity = np.asarray(
        similarity_receipt["similarity"], dtype=np.float64)
    require(similarity.shape == (len(queries), len(history_rgb)),
            "DINO similarity matrix has the wrong shape")
    pose_receipt = checked_json(session.post(
        f"{base}/causal_pose_trace_query", timeout=180), "pose trace")
    pose9 = np.asarray(pose_receipt["pose9"], dtype=np.float64)
    require(pose9.shape == (len(history_rgb) + len(queries), 9),
            "model pose trace has the wrong shape")
    scale_receipt = pose_receipt["longrange_metric_scale"]
    require(scale_receipt.get("available") is True,
            "causal first-40 scale is unavailable")
    scale = float(scale_receipt["metric_scale_m_per_raw"])
    require(math.isfinite(scale) and scale > 0.0, "invalid causal scale")

    target = int(query_set["target_history_frame_analysis_only"])
    route_indices = np.arange(
        len(history_rgb) - 1, target - 1, -1, dtype=np.int64)
    route_positions = pose9[:len(history_rgb)][route_indices][:, (0, 2)] * scale
    transition = derive_route_transition(
        route_positions,
        nominal_motion_m=NOMINAL_TRANSLATION_M,
        maximum_search_frames=TRANSITION_SEARCH_FRAMES,
    )
    tracker = MonotoneRouteFilter(
        len(route_indices), transition, source_indices=route_indices)

    readouts = []
    scored = []
    for row, observation in zip(queries, similarity[:, route_indices]):
        action_class = str(row["kind"])
        require(action_class in ("turn_in_place", "reverse_route"),
                "unknown construction action class")
        readout = tracker.update(
            observation, translated=(action_class == "reverse_route"))
        record = {
            **readout.audit_dict(),
            "query_index": int(row["query_index"]),
            "action_class": (
                "translation" if action_class == "reverse_route" else "turn"),
        }
        readouts.append(record)
        if action_class == "reverse_route":
            anchor = int(readout.source_index)
            pose = poses[anchor]
            floor = row["construction_floor_position"]
            error = math.hypot(
                float(pose["x"]) - float(floor[0]),
                float(pose["z"]) - float(floor[2]),
            )
            scored.append({
                **record,
                "target_source_index_analysis_only": int(
                    row["history_reference_frame_analysis_only"]),
                "position_error_m_analysis_only": error,
            })

    errors = [row["position_error_m_analysis_only"] for row in scored]
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "causal RGB and executed-action-class route-state audit only; "
            "evaluator pose scores readouts after inference; no control or SR"),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_query_set_sha256": sha256(arguments.query_set),
        "runtime_evaluator_pose_visible": False,
        "runtime_historical_source_address_visible": False,
        "runtime_executed_action_class_visible": True,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "distance_regime_present": False,
        "observation_acceptance_gate_present": False,
        "endpoint_fallback_present": False,
        "native_fallback_present": False,
        "nominal_translation_m": NOMINAL_TRANSLATION_M,
        "transition": transition.audit_dict(),
        "route_observation_count": len(scored),
        "within_1m_count": sum(error <= 1.0 for error in errors),
        "within_2m_count": sum(error <= 2.0 for error in errors),
        "median_position_error_m": median(errors),
        "final_position_error_m": errors[-1],
        "maximum_position_error_m": max(errors),
        "readouts": readouts,
        "scored_route_readouts_analysis_only": scored,
    }
    arguments.out.mkdir(parents=True)
    matrix_path = arguments.out / "runtime_receipts.npz"
    np.savez_compressed(
        matrix_path,
        similarity=similarity.astype(np.float16),
        pose9=pose9.astype(np.float32),
        route_indices=route_indices,
    )
    result["runtime_receipts_npz_sha256"] = sha256(matrix_path)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output = arguments.out / "route_filter_audit.json"
    output.write_bytes(encoded)
    (arguments.out / "route_filter_audit.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  route_filter_audit.json\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "history_index": arguments.history_index,
        "transition": transition.audit_dict(),
        "within_1m": result["within_1m_count"],
        "observations": len(scored),
        "median_m": result["median_position_error_m"],
        "final_m": result["final_position_error_m"],
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
