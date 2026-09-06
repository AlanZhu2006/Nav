#!/usr/bin/env python3
"""Render deterministic non-duplicate views for a local path re-anchor gate.

Evaluator poses are used only to construct this diagnostic query set.  The
runtime re-anchor client receives RGB bytes and the frozen causal history; it
does not receive the source index, perturbation, or Habitat pose.  These views
therefore test local visual addressability under controlled view change, not
navigation efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from MemNavData.generate_twoleg import make_sim, render


SCHEMA_VERSION = "hm3d_episodic_reanchor_query_set_v3_20260902"
FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror_path(mirror_root: Path, remote_path: str) -> Path:
    path = Path(remote_path)
    require(path.is_absolute() and path.parts[:2] == ("/", "scratch"),
            "query source must be a sealed /scratch path")
    return mirror_root.joinpath(*path.parts[1:])


def interpolate_yaw(first: float, second: float, alpha: float) -> float:
    delta = (float(second) - float(first) + math.pi) % (2 * math.pi) - math.pi
    return float(first) + float(alpha) * delta


def optical_center_from_floor(
    floor_position: np.ndarray,
    camera_height_m: float,
) -> np.ndarray:
    floor = np.asarray(floor_position, dtype=np.float64)
    height = float(camera_height_m)
    require(floor.shape == (3,) and np.isfinite(floor).all(),
            "floor position must be a finite 3-vector")
    require(math.isfinite(height) and height > 0.0,
            "camera height must be finite and positive")
    return floor + np.asarray([0.0, height, 0.0], dtype=np.float64)


def query_yaw_from_history(
    base_yaw_rad: float,
    *,
    base_yaw_offset_deg: float,
    perturbation_deg: float,
) -> float:
    """Construct one diagnostic view relative to the taught heading."""

    values = (base_yaw_rad, base_yaw_offset_deg, perturbation_deg)
    require(all(math.isfinite(float(value)) for value in values),
            "query-yaw inputs must be finite")
    require(abs(float(base_yaw_offset_deg)) <= 180.0,
            "base yaw offset must lie in [-180,180] degrees")
    return (float(base_yaw_rad)
            + math.radians(float(base_yaw_offset_deg))
            + math.radians(float(perturbation_deg)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--baseline-frames", type=int, default=8)
    parser.add_argument("--yaw-offset-deg", type=float, default=15.0)
    parser.add_argument(
        "--base-yaw-offset-deg", type=float, default=0.0,
        help=(
            "construction-only shift relative to the historical camera; "
            "180 degrees diagnoses reverse traversal visibility"),
    )
    arguments = parser.parse_args()

    require(arguments.history_index in (0, 16, 32),
            "diagnostic is sealed to indices 0,16,32")
    require(arguments.queries >= 3 and arguments.baseline_frames >= 2,
            "query count/baseline is too small")
    require(0.0 < arguments.yaw_offset_deg <= 30.0,
            "yaw perturbation must be in (0,30] degrees")
    require(math.isfinite(arguments.base_yaw_offset_deg)
            and abs(arguments.base_yaw_offset_deg) <= 180.0,
            "base yaw offset must lie in [-180,180] degrees")
    require(not arguments.out.exists(), "query output already exists")
    mirror = arguments.mirror_root.resolve()
    manifest_path = arguments.manifest.resolve()
    require(sha256(manifest_path) == FROZEN_MANIFEST_SHA256,
            "frozen manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest["episodes"][arguments.history_index]
    history = mirror_path(mirror, item["online_a_episode"])
    trace_path = history / "online_a_trace.json"
    require(sha256(trace_path) == item["online_a_trace_sha256"],
            "causal trace changed")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    poses = trace["poses"]
    require(len(poses) == int(item["online_a_steps"]),
            "causal trace length changed")

    receipt = json.loads((history / "receipt.json").read_text(encoding="utf-8"))
    scene = mirror_path(mirror, receipt["source_asset"])
    navmesh = mirror_path(mirror, item["runtime_navmesh"])
    require(
        sha256(scene) == receipt["source_asset_sha256"]
        and sha256(navmesh) == item["runtime_navmesh_sha256"],
        "mirrored scene geometry changed",
    )

    margin = max(64, arguments.baseline_frames + 1)
    last = len(poses) - arguments.baseline_frames - 17
    require(last > margin, "history is too short for the query protocol")
    raw_indices = np.linspace(margin, last, arguments.queries)
    indices: list[int] = []
    for value in raw_indices:
        index = int(round(float(value)))
        if indices and index <= indices[-1]:
            index = indices[-1] + 1
        indices.append(index)
    require(len(indices) == len(set(indices)) == arguments.queries,
            "deterministic query indices collided")

    arguments.out.mkdir(parents=True)
    image_root = arguments.out / "rgb"
    image_root.mkdir()
    simulator = make_sim(
        str(scene), str(navmesh), agent_radius=0.30, recompute_navmesh=False)
    rows = []
    try:
        for query_index, source_index in enumerate(indices):
            target_index = source_index + arguments.baseline_frames
            first, second = poses[source_index], poses[target_index]
            first_position = np.asarray(
                [first["x"], first["y"], first["z"]], dtype=np.float64)
            second_position = np.asarray(
                [second["x"], second["y"], second["z"]], dtype=np.float64)
            # The sealed trace stores navigable floor positions.  Habitat's
            # ``render`` helper expects the optical-center position, exactly
            # as the original survey constructor does.  Keeping these two
            # quantities explicit prevents a floor-level diagnostic camera.
            floor_position = 0.5 * (first_position + second_position)
            camera_position = optical_center_from_floor(
                floor_position, float(receipt["camera_height_m"]))
            base_yaw = interpolate_yaw(first["yaw"], second["yaw"], 0.5)
            sign = -1.0 if query_index % 2 else 1.0
            perturbation_deg = sign * arguments.yaw_offset_deg
            query_yaw = query_yaw_from_history(
                base_yaw,
                base_yaw_offset_deg=arguments.base_yaw_offset_deg,
                perturbation_deg=perturbation_deg,
            )
            rgb, _depth = render(simulator, camera_position, query_yaw)
            image_path = image_root / f"query_{query_index:02d}.jpg"
            Image.fromarray(rgb).save(image_path, format="JPEG", quality=95)
            rows.append({
                "query_index": query_index,
                "rgb": image_path.relative_to(arguments.out).as_posix(),
                "rgb_sha256": sha256(image_path),
                "construction_source_interval": [source_index, target_index],
                "construction_expected_center_frame": 0.5 * (
                    source_index + target_index),
                "construction_floor_position": [
                    float(value) for value in floor_position],
                "construction_camera_position": [
                    float(value) for value in camera_position],
                "construction_camera_height_m": float(
                    receipt["camera_height_m"]),
                "construction_base_yaw_rad": base_yaw,
                "construction_query_yaw_rad": query_yaw,
                "construction_base_yaw_offset_deg": float(
                    arguments.base_yaw_offset_deg),
                "construction_yaw_offset_deg": perturbation_deg,
            })
    finally:
        simulator.close()

    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "construction-only Habitat poses; runtime receives RGB and sealed "
            "history only"
        ),
        "history_index": arguments.history_index,
        "scene": item["scene"],
        "episode": item["episode"],
        "bin_name": item["bin_name"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_trace_sha256": item["online_a_trace_sha256"],
        "source_scene_sha256": receipt["source_asset_sha256"],
        "source_navmesh_sha256": item["runtime_navmesh_sha256"],
        "query_count": arguments.queries,
        "baseline_frames": arguments.baseline_frames,
        "base_yaw_offset_deg": float(arguments.base_yaw_offset_deg),
        "absolute_yaw_offset_deg": arguments.yaw_offset_deg,
        "runtime_evaluator_pose_visible": False,
        "queries": rows,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    receipt_path = arguments.out / "query_set.json"
    receipt_path.write_bytes(encoded)
    (arguments.out / "query_set.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  query_set.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": arguments.history_index,
        "queries": arguments.queries,
        "out": str(arguments.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}")
        raise
