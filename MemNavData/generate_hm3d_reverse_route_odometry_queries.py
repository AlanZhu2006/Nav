#!/usr/bin/env python3
"""Render one dense reverse traversal for a LingBot route-tape audit.

Habitat state is used only to construct and later score this mechanism
diagnostic.  Runtime receives the sealed outgoing RGB history followed by the
rendered RGB stream; it never receives the source indices or poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from MemNavData.generate_twoleg import make_sim, render, yaw_facing


SCHEMA_VERSION = "hm3d_reverse_route_odometry_queries_v1_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)


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


def position(row: dict) -> np.ndarray:
    return np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64)


def revisit_query(item: dict) -> dict:
    candidates = [
        query
        for pair in item["pairs"]
        for query in pair["queries"]
        if query["analysis_role"] == "revisit"
    ]
    require(len(candidates) == 1, "population item must have one Revisit query")
    return candidates[0]


def sample_reverse_indices(
    poses: list[dict], *, target_index: int, spacing_m: float
) -> list[int]:
    require(0 <= target_index < len(poses) - 1, "invalid target index")
    selected = [len(poses) - 1]
    accumulated = 0.0
    for index in range(len(poses) - 2, target_index - 1, -1):
        accumulated += float(np.linalg.norm(
            position(poses[index + 1])[[0, 2]]
            - position(poses[index])[[0, 2]]))
        if accumulated + 1e-9 >= spacing_m:
            selected.append(index)
            accumulated = 0.0
    if selected[-1] != target_index:
        selected.append(target_index)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.30)
    parser.add_argument("--turn-step-deg", type=float, default=30.0)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(arguments.history_index in (0, 16, 32, 40),
            "diagnostic is sealed to indices 0,16,32,40")
    require(math.isfinite(arguments.spacing_m)
            and 0.20 <= arguments.spacing_m <= 0.50,
            "spacing must lie in [0.20,0.50] m")
    require(math.isfinite(arguments.turn_step_deg)
            and 10.0 <= arguments.turn_step_deg <= 45.0,
            "turn step must lie in [10,45] degrees")
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
            "causal trace length changed")

    target = revisit_query(item)
    target_index = int(target["source_online_frame"])
    indices = sample_reverse_indices(
        poses, target_index=target_index, spacing_m=arguments.spacing_m)
    require(indices[0] == len(poses) - 1 and indices[-1] == target_index,
            "reverse route endpoints changed")

    scene = mirror_path(arguments.mirror_root.resolve(),
                        receipt["source_asset"])
    navmesh = mirror_path(arguments.mirror_root.resolve(),
                          item["runtime_navmesh"])
    require(sha256(scene) == receipt["source_asset_sha256"]
            and sha256(navmesh) == item["runtime_navmesh_sha256"],
            "scene geometry changed")

    floor_positions = [position(row) for row in poses]
    first_delta = (
        floor_positions[indices[1]][[0, 2]]
        - floor_positions[indices[0]][[0, 2]])
    require(float(np.linalg.norm(first_delta)) > 1e-6,
            "reverse route starts with zero translation")
    initial_yaw = float(poses[-1]["yaw"])
    reverse_yaw = float(yaw_facing(first_delta))
    turn_delta = wrap(reverse_yaw - initial_yaw)
    turn_count = max(
        1, int(math.ceil(abs(math.degrees(turn_delta))
                         / arguments.turn_step_deg)))

    arguments.out.mkdir(parents=True)
    image_root = arguments.out / "rgb"
    image_root.mkdir()
    simulator = make_sim(
        str(scene), str(navmesh), agent_radius=0.30, recompute_navmesh=False)
    rows: list[dict] = []
    camera_height = float(receipt["camera_height_m"])

    def write_frame(kind: str, floor: np.ndarray, yaw: float,
                    source_index: int | None) -> None:
        camera = floor + np.asarray([0.0, camera_height, 0.0])
        rgb, _ = render(simulator, camera, yaw)
        output = image_root / f"query_{len(rows):04d}.jpg"
        Image.fromarray(rgb).save(output, format="JPEG", quality=95)
        rows.append({
            "query_index": len(rows),
            "kind": kind,
            "rgb": output.relative_to(arguments.out).as_posix(),
            "rgb_sha256": sha256(output),
            "construction_floor_position": [float(value) for value in floor],
            "construction_camera_height_m": camera_height,
            "construction_yaw_rad": float(yaw),
            "history_reference_frame_analysis_only": source_index,
        })

    try:
        tail = floor_positions[-1]
        for step in range(1, turn_count + 1):
            alpha = step / turn_count
            write_frame(
                "turn_in_place", tail,
                initial_yaw + alpha * turn_delta,
                len(poses) - 1,
            )
        # The final turn frame is the route origin.  Start with the first
        # translated sample to avoid duplicating an identical image.
        for offset, source_index in enumerate(indices[1:], start=1):
            floor = floor_positions[source_index]
            if offset + 1 < len(indices):
                next_floor = floor_positions[indices[offset + 1]]
                delta = next_floor[[0, 2]] - floor[[0, 2]]
                yaw = (float(yaw_facing(delta))
                       if float(np.linalg.norm(delta)) > 1e-6
                       else float(poses[source_index]["yaw"] + math.pi))
            else:
                yaw = float(target["yaw_rad"])
            write_frame("reverse_route", floor, yaw, source_index)
    finally:
        simulator.close()

    route_length = sum(
        float(np.linalg.norm(
            floor_positions[first][[0, 2]]
            - floor_positions[second][[0, 2]]))
        for first, second in zip(indices, indices[1:])
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "construction-only reverse-route RGB stream; runtime receives "
            "neither Habitat pose nor historical source address"),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_trace_sha256": item["online_a_trace_sha256"],
        "source_scene_sha256": receipt["source_asset_sha256"],
        "source_navmesh_sha256": item["runtime_navmesh_sha256"],
        "target_history_frame_analysis_only": target_index,
        "spacing_m": float(arguments.spacing_m),
        "turn_step_deg": float(arguments.turn_step_deg),
        "turn_delta_deg": math.degrees(turn_delta),
        "sampled_history_indices_analysis_only": indices,
        "sampled_route_length_m_analysis_only": route_length,
        "runtime_evaluator_pose_visible": False,
        "queries": rows,
        "query_count": len(rows),
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output = arguments.out / "query_set.json"
    output.write_bytes(encoded)
    (arguments.out / "query_set.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  query_set.json\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "query_count": len(rows),
        "turn_frames": turn_count,
        "route_frames": len(indices) - 1,
        "route_length_m": route_length,
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
