#!/usr/bin/env python3
"""Render a physically continuous reverse-route RGB mechanism audit.

The earlier reverse-route diagnostic sampled positions approximately every
0.30 m and rendered each sample facing the *following* route segment. At a
corner, one adjacent RGB pair could therefore contain both a translation and
an instantaneous 60--170 degree heading change. A real NavDP stream observes
the intervening turn frames. This generator keeps the same frozen route and
adds those observations explicitly: turn in place at the current position,
then translate while holding the new heading.

Habitat poses remain construction/scorer-only. Runtime receives only the
ordered RGB stream and the already sealed outgoing causal history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from MemNavData.generate_hm3d_reverse_route_odometry_queries import (
    FROZEN_MANIFEST_SHA256,
    mirror_path,
    position,
    require,
    revisit_query,
    sample_reverse_indices,
    sha256,
    wrap,
)
from MemNavData.generate_twoleg import make_sim, render, yaw_facing
from MemNavData.monocular_route_tangent_contract import (
    SEALED_DENSE_HISTORY_INDICES,
)


SCHEMA_VERSION = "hm3d_dense_reverse_route_queries_v2_20260903"


def shortest_turn_samples(
    start_yaw: float,
    target_yaw: float,
    *,
    max_step_deg: float,
) -> list[float]:
    """Return an endpoint-inclusive shortest turn with bounded increments."""

    maximum = math.radians(float(max_step_deg))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_step_deg must be finite and positive")
    delta = wrap(float(target_yaw) - float(start_yaw))
    if abs(delta) <= 1e-9:
        return []
    count = max(1, int(math.ceil(abs(delta) / maximum)))
    return [float(start_yaw) + delta * step / count
            for step in range(1, count + 1)]


def dense_reverse_states(
    poses: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    target_yaw: float,
    max_turn_step_deg: float,
) -> list[dict[str, Any]]:
    """Build a turn-then-translate camera sequence on a frozen reverse route."""

    if len(indices) < 2:
        raise ValueError("reverse route needs at least two sampled positions")
    if int(indices[0]) != len(poses) - 1:
        raise ValueError("reverse route must begin at the causal-history tail")
    if any(int(first) <= int(second)
           for first, second in zip(indices, indices[1:])):
        raise ValueError("reverse route indices must decrease strictly")

    states: list[dict[str, Any]] = []
    current_index = int(indices[0])
    current_floor = position(poses[current_index])
    current_yaw = float(poses[current_index]["yaw"])
    states.append({
        "kind": "route_origin",
        "floor": current_floor,
        "yaw": current_yaw,
        "source_index": current_index,
    })

    for next_index_raw in indices[1:]:
        next_index = int(next_index_raw)
        next_floor = position(poses[next_index])
        delta = next_floor[[0, 2]] - current_floor[[0, 2]]
        if float(np.linalg.norm(delta)) <= 1e-6:
            raise ValueError("sampled reverse route contains zero translation")
        heading = float(yaw_facing(delta))
        for yaw in shortest_turn_samples(
                current_yaw, heading, max_step_deg=max_turn_step_deg):
            states.append({
                "kind": "turn_in_place",
                "floor": current_floor.copy(),
                "yaw": float(yaw),
                "source_index": current_index,
            })
        states.append({
            "kind": "reverse_route",
            "floor": next_floor,
            "yaw": heading,
            "source_index": next_index,
        })
        current_index = next_index
        current_floor = next_floor
        current_yaw = heading

    for yaw in shortest_turn_samples(
            current_yaw, float(target_yaw), max_step_deg=max_turn_step_deg):
        states.append({
            "kind": "terminal_turn",
            "floor": current_floor.copy(),
            "yaw": float(yaw),
            "source_index": current_index,
        })

    maximum = math.radians(float(max_turn_step_deg)) + 1e-9
    for first, second in zip(states, states[1:]):
        translation = float(np.linalg.norm(
            np.asarray(second["floor"])[[0, 2]]
            - np.asarray(first["floor"])[[0, 2]]))
        yaw_delta = abs(wrap(float(second["yaw"]) - float(first["yaw"])))
        if translation <= 1e-6 and yaw_delta > maximum:
            raise RuntimeError("densified in-place turn exceeds its bound")
        if translation > 1e-6 and yaw_delta > 1e-9:
            raise RuntimeError("translation and turn were collapsed into one edge")
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.30)
    parser.add_argument("--max-turn-step-deg", type=float, default=15.0)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in SEALED_DENSE_HISTORY_INDICES,
            "dense mechanism audit history is not sealed")
    require(math.isfinite(arguments.spacing_m)
            and 0.20 <= arguments.spacing_m <= 0.50,
            "spacing must lie in [0.20,0.50] m")
    require(math.isfinite(arguments.max_turn_step_deg)
            and 5.0 <= arguments.max_turn_step_deg <= 30.0,
            "turn step must lie in [5,30] degrees")
    require(not arguments.out.exists(), "query output already exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]
    history = mirror_path(arguments.mirror_root.resolve(),
                          item["online_a_episode"])
    trace_path = history / "online_a_trace.json"
    receipt_path = history / "receipt.json"
    require(sha256(trace_path) == item["online_a_trace_sha256"]
            and sha256(receipt_path) == item["online_a_receipt_sha256"],
            "causal history changed")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    poses = trace["poses"]
    require(len(poses) == int(item["online_a_steps"]),
            "causal history length changed")

    target = revisit_query(item)
    target_index = int(target["source_online_frame"])
    indices = sample_reverse_indices(
        poses, target_index=target_index, spacing_m=arguments.spacing_m)
    states = dense_reverse_states(
        poses,
        indices,
        target_yaw=float(target["yaw_rad"]),
        max_turn_step_deg=float(arguments.max_turn_step_deg),
    )

    scene = mirror_path(arguments.mirror_root.resolve(),
                        receipt["source_asset"])
    navmesh = mirror_path(arguments.mirror_root.resolve(),
                          item["runtime_navmesh"])
    require(sha256(scene) == receipt["source_asset_sha256"]
            and sha256(navmesh) == item["runtime_navmesh_sha256"],
            "scene geometry changed")

    arguments.out.mkdir(parents=True)
    image_root = arguments.out / "rgb"
    image_root.mkdir()
    simulator = make_sim(
        str(scene), str(navmesh), agent_radius=0.30, recompute_navmesh=False)
    rows: list[dict[str, Any]] = []
    camera_height = float(receipt["camera_height_m"])
    try:
        for query_index, state in enumerate(states):
            floor = np.asarray(state["floor"], dtype=np.float64)
            camera = floor + np.asarray([0.0, camera_height, 0.0])
            rgb, _ = render(simulator, camera, float(state["yaw"]))
            output = image_root / f"query_{query_index:04d}.jpg"
            Image.fromarray(rgb).save(output, format="JPEG", quality=95)
            rows.append({
                "query_index": query_index,
                "kind": str(state["kind"]),
                "rgb": output.relative_to(arguments.out).as_posix(),
                "rgb_sha256": sha256(output),
                "construction_floor_position": [
                    float(value) for value in floor],
                "construction_camera_height_m": camera_height,
                "construction_yaw_rad": float(state["yaw"]),
                "history_reference_frame_analysis_only": int(
                    state["source_index"]),
            })
    finally:
        simulator.close()

    floor_positions = [position(row) for row in poses]
    route_length = sum(
        float(np.linalg.norm(
            floor_positions[first][[0, 2]]
            - floor_positions[second][[0, 2]]))
        for first, second in zip(indices, indices[1:])
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "construction-only physically continuous reverse-route RGB "
            "stream; runtime receives neither Habitat pose nor historical "
            "source address"),
        "history_index": int(arguments.history_index),
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_trace_sha256": item["online_a_trace_sha256"],
        "source_scene_sha256": receipt["source_asset_sha256"],
        "source_navmesh_sha256": item["runtime_navmesh_sha256"],
        "target_history_frame_analysis_only": target_index,
        "spacing_m": float(arguments.spacing_m),
        "max_turn_step_deg": float(arguments.max_turn_step_deg),
        "sampled_history_indices_analysis_only": [
            int(value) for value in indices],
        "sampled_route_length_m_analysis_only": float(route_length),
        "runtime_evaluator_pose_visible": False,
        "simultaneous_turn_translation_edges": 0,
        "queries": rows,
        "query_count": len(rows),
    }
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = arguments.out / "query_set.json"
    output.write_bytes(encoded)
    (arguments.out / "query_set.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  query_set.json\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "history_index": int(arguments.history_index),
        "query_count": len(rows),
        "turn_frames": sum(
            str(row["kind"]) in ("turn_in_place", "terminal_turn")
            for row in rows),
        "route_frames": sum(
            str(row["kind"]) == "reverse_route" for row in rows),
        "route_length_m": float(route_length),
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
