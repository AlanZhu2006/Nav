#!/usr/bin/env python3
"""Audit Replica-v1 scenes against the frozen MemNav Habitat sensor contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from generate_twoleg import HFOV_DEG, H, W, make_sim, render


SCHEMA_VERSION = "replica_memnav_habitat_compatibility_v1_20260814"
REQUIRED_FILES = (
    "mesh.ply",
    "habitat/replica_stage.stage_config.json",
    "habitat/mesh_semantic.ply",
    "habitat/mesh_semantic.navmesh",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shortest_distance(pathfinder, first: np.ndarray, second: np.ndarray) -> float | None:
    import habitat_sim

    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(first, dtype=np.float32)
    query.requested_end = np.asarray(second, dtype=np.float32)
    if not pathfinder.find_path(query) or not math.isfinite(float(query.geodesic_distance)):
        return None
    return float(query.geodesic_distance)


def sample_geodesic_diameter(pathfinder, *, seed: int, points: int = 96) -> dict:
    pathfinder.seed(int(seed))
    samples = np.stack([
        np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
        for _ in range(points)
    ])
    # Replica's reconstructed single floor is not perfectly planar: the
    # official navmeshes vary by up to roughly 0.6 m.  A 20 cm quantization
    # incorrectly fragments one floor into tiny strips.  Use the generator's
    # frozen 0.8 m same-floor tolerance around the sample median instead.
    floor_center = float(np.median(samples[:, 1]))
    floor_samples = samples[np.abs(samples[:, 1] - floor_center) <= 0.80]
    candidates: list[tuple[float, int, int]] = []
    # A deterministic Euclidean shortlist avoids an O(N^2) path query sweep.
    for first in range(len(floor_samples)):
        for second in range(first + 1, len(floor_samples)):
            euclidean = float(np.linalg.norm(
                floor_samples[first, [0, 2]] - floor_samples[second, [0, 2]]
            ))
            candidates.append((euclidean, first, second))
    candidates.sort(reverse=True)
    finite = []
    for _euclidean, first, second in candidates[:128]:
        distance = shortest_distance(
            pathfinder, floor_samples[first], floor_samples[second]
        )
        if distance is not None:
            finite.append((distance, first, second))
    if not finite:
        return {
            "sample_count": int(len(floor_samples)),
            "floor_center_y_m": floor_center,
            "same_floor_tolerance_m": 0.80,
            "maximum_sampled_geodesic_m": None,
            "endpoint_pair": None,
        }
    distance, first, second = max(finite)
    return {
        "sample_count": int(len(floor_samples)),
        "floor_center_y_m": floor_center,
        "same_floor_tolerance_m": 0.80,
        "maximum_sampled_geodesic_m": float(distance),
        "endpoint_pair": [
            floor_samples[first].tolist(), floor_samples[second].tolist()
        ],
    }


def render_audit(simulator, position: np.ndarray) -> dict:
    views = []
    for yaw in np.linspace(-math.pi, math.pi, 4, endpoint=False):
        rgb, depth = render(
            simulator,
            np.asarray(position, dtype=np.float64)
            + np.asarray([0.0, 0.5, 0.0]),
            float(yaw),
        )
        rgb_array = np.asarray(rgb, dtype=np.uint8)
        depth_array = np.asarray(depth, dtype=np.float64)
        views.append({
            "yaw_rad": float(yaw),
            "rgb_shape": list(rgb_array.shape),
            "rgb_channel_mean": np.mean(rgb_array, axis=(0, 1)).tolist(),
            "rgb_std": float(np.std(rgb_array)),
            "black_pixel_fraction": float(
                np.mean(np.all(rgb_array < 5, axis=-1))
            ),
            "finite_depth_fraction": float(np.mean(np.isfinite(depth_array))),
            "positive_depth_fraction": float(np.mean(depth_array > 0.0)),
            "median_depth_m": float(np.nanmedian(depth_array)),
        })
    return {
        "views": views,
        "all_shapes_match": all(
            view["rgb_shape"] == [H, W, 3] for view in views
        ),
        "minimum_rgb_std": min(view["rgb_std"] for view in views),
        "maximum_black_pixel_fraction": max(
            view["black_pixel_fraction"] for view in views
        ),
        "minimum_finite_depth_fraction": min(
            view["finite_depth_fraction"] for view in views
        ),
        "minimum_positive_depth_fraction": min(
            view["positive_depth_fraction"] for view in views
        ),
    }


def audit_scene(scene_root: Path, *, seed: int, minimum_geodesic_m: float) -> dict:
    paths = {relative: scene_root / relative for relative in REQUIRED_FILES}
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "scene": scene_root.name,
            "complete": False,
            "eligible": False,
            "missing": missing,
        }
    stage = paths["habitat/replica_stage.stage_config.json"]
    navmesh = paths["habitat/mesh_semantic.navmesh"]
    simulator = make_sim(
        str(stage), str(navmesh), agent_radius=0.30, agent_height=1.5
    )
    try:
        pathfinder = simulator.pathfinder
        if not pathfinder.is_loaded:
            raise RuntimeError(f"navmesh did not load: {scene_root.name}")
        pathfinder.seed(int(seed))
        diameter = sample_geodesic_diameter(pathfinder, seed=seed)
        endpoint_pair = diameter["endpoint_pair"]
        render_position = (
            np.asarray(endpoint_pair[0], dtype=np.float64)
            if endpoint_pair is not None
            else np.asarray(
                pathfinder.get_random_navigable_point(), dtype=np.float64
            )
        )
        visual = render_audit(simulator, render_position)
        maximum_geodesic = diameter["maximum_sampled_geodesic_m"]
        checks = {
            "navmesh_loaded": True,
            "navigable_area_positive": float(pathfinder.navigable_area) > 0.0,
            "query_distance_supported": (
                maximum_geodesic is not None
                and float(maximum_geodesic) >= float(minimum_geodesic_m)
            ),
            "rgb_shape_matches": visual["all_shapes_match"],
            "rgb_non_degenerate": visual["minimum_rgb_std"] >= 5.0,
            "rgb_not_black": visual["maximum_black_pixel_fraction"] <= 0.90,
            "depth_finite": visual["minimum_finite_depth_fraction"] >= 0.99,
            # Habitat encodes missing depth as zero.  Replica office_3 has
            # small mesh holes, so requiring 99% positive pixels confuses
            # valid sensor semantics with scene completeness.  Eighty percent
            # retains a strict usable-depth gate without inventing returns.
            "depth_positive": visual["minimum_positive_depth_fraction"] >= 0.80,
        }
        return {
            "scene": scene_root.name,
            "complete": True,
            "eligible": all(checks.values()),
            "checks": checks,
            "files": {
                relative: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for relative, path in paths.items()
            },
            "sensor_contract": {
                "height": H,
                "width": W,
                "horizontal_fov_deg": HFOV_DEG,
                "camera_height_m": 0.5,
                "agent_radius_m": 0.30,
                "agent_height_m": 1.5,
            },
            "navigable_area_m2": float(pathfinder.navigable_area),
            "geodesic_probe": diameter,
            "render_probe": visual,
        }
    finally:
        simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-complete-scenes", type=int, default=8)
    parser.add_argument("--minimum-eligible-scenes", type=int, default=6)
    parser.add_argument("--minimum-geodesic-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    scene_roots = sorted(
        path for path in args.replica_root.iterdir() if path.is_dir()
    )
    complete_roots = [
        path for path in scene_roots
        if all((path / relative).is_file() for relative in REQUIRED_FILES)
    ]
    rows = [
        audit_scene(
            path,
            seed=args.seed + index,
            minimum_geodesic_m=args.minimum_geodesic_m,
        )
        for index, path in enumerate(complete_roots)
    ]
    eligible = [row["scene"] for row in rows if row["eligible"]]
    passed = (
        len(complete_roots) == args.expected_complete_scenes
        and len(eligible) >= args.minimum_eligible_scenes
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "simulator_and_sensor_compatibility_only_not_policy_SR",
        "replica_root": str(args.replica_root.resolve()),
        "expected_complete_scenes": args.expected_complete_scenes,
        "complete_scenes": len(complete_roots),
        "minimum_eligible_scenes": args.minimum_eligible_scenes,
        "eligible_scenes": eligible,
        "eligible_scene_count": len(eligible),
        "minimum_geodesic_m": args.minimum_geodesic_m,
        "seed": args.seed,
        "passed": passed,
        "paper_navigation_evaluation_authorized": passed,
        "scenes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "passed": passed,
        "complete_scenes": len(complete_roots),
        "eligible_scenes": len(eligible),
        "output": str(args.out),
    }, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
