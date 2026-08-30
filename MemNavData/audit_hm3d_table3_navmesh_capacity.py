#!/usr/bin/env python3
"""Audit long-range paired-query capacity without rendering or policy outcomes.

For each HM3D navmesh this program samples a deterministic set of points on
the largest navigable island.  It looks for a query start with two goal
hypotheses in the same requested distance bin.  The two goals must be
distance-matched, spatially distinct, and directionally separated.  These
triads are only geometry proposals for the later rendered Novel/Revisit
constructor; they do not authorize a policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "hm3d_table3_navmesh_capacity_scene_v1_20260830"
SUMMARY_SCHEMA = "hm3d_table3_navmesh_capacity_summary_v1_20260830"
PROTOCOL_SCHEMA = "hm3d_table3_navmesh_capacity_protocol_v1_20260830"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite {path}")
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n")


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(payload.get("schema_version") == PROTOCOL_SCHEMA,
            "Table-3 capacity protocol schema changed")
    require(payload["authority_boundary"]["query_policy_outcomes_read"] is False,
            "protocol permits query-outcome access")
    require(payload["authority_boundary"]["navigation_outcomes_read"] is False,
            "protocol permits navigation-outcome access")
    require(payload["authority_boundary"][
        "this_audit_authorizes_policy_evaluation"] is False,
        "capacity audit may not authorize policy evaluation")
    return payload


def bin_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = protocol["length_definition"]["bins_m"]
    require([row["name"] for row in rows] == [
        "0_to_20_m", "20_to_30_m", "30_to_50_m"
    ], "trajectory-length bins changed")
    return rows


def in_bin(distance: float, spec: dict[str, Any]) -> bool:
    lower = float(spec["lower_inclusive"])
    upper = float(spec["upper"])
    return distance >= lower and (
        distance <= upper if bool(spec["upper_inclusive"]) else distance < upper
    )


def circular_separation_degrees(first: float, second: float) -> float:
    delta = (math.degrees(first - second) + 180.0) % 360.0 - 180.0
    return abs(delta)


def select_scene_triads(
    points: list[list[float]],
    distances: list[list[float | None]],
    bearings: list[list[float | None]],
    protocol: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Select deterministic geometry triads from an already measured graph."""

    pair = protocol["paired_geometry"]
    max_mismatch = float(pair["maximum_goal_distance_mismatch"])
    min_bearing = float(pair["minimum_initial_bearing_separation_deg"])
    min_goal_sep = float(pair["minimum_goal_to_goal_geodesic_m"])
    maximum = int(pair["maximum_candidates_saved_per_scene_per_bin"])
    selected: dict[str, list[dict[str, Any]]] = {}
    for spec in bin_specs(protocol):
        centre = (float(spec["lower_inclusive"]) + float(spec["upper"])) / 2.0
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for start in range(len(points)):
            targets = [
                target for target in range(len(points))
                if target != start
                and distances[start][target] is not None
                and in_bin(float(distances[start][target]), spec)
            ]
            best_for_start: tuple[tuple[Any, ...], dict[str, Any]] | None = None
            for first, second in itertools.combinations(targets, 2):
                first_d = float(distances[start][first])
                second_d = float(distances[start][second])
                if abs(first_d - second_d) > max_mismatch:
                    continue
                mutual = distances[first][second]
                if mutual is None or float(mutual) < min_goal_sep:
                    continue
                first_b = bearings[start][first]
                second_b = bearings[start][second]
                if first_b is None or second_b is None:
                    continue
                bearing_sep = circular_separation_degrees(
                    float(first_b), float(second_b)
                )
                if bearing_sep < min_bearing:
                    continue
                ranking = (
                    abs(first_d - centre) + abs(second_d - centre),
                    abs(first_d - second_d),
                    -bearing_sep,
                    start,
                    first,
                    second,
                )
                row = {
                    "query_start_sample": start,
                    "first_goal_sample": first,
                    "second_goal_sample": second,
                    "query_start": points[start],
                    "first_goal": points[first],
                    "second_goal": points[second],
                    "first_goal_geodesic_m": first_d,
                    "second_goal_geodesic_m": second_d,
                    "goal_distance_mismatch": abs(first_d - second_d),
                    "goal_to_goal_geodesic_m": float(mutual),
                    "initial_bearing_separation_deg": bearing_sep,
                    "ranking": list(ranking),
                }
                if best_for_start is None or ranking < best_for_start[0]:
                    best_for_start = (ranking, row)
            if best_for_start is not None:
                candidates.append(best_for_start)
        candidates.sort(key=lambda item: item[0])
        selected[str(spec["name"])] = [row for _rank, row in candidates[:maximum]]
    return selected


def _measure_graph(pathfinder, points: list[np.ndarray]):
    import habitat_sim

    count = len(points)
    distances: list[list[float | None]] = [[None] * count for _ in range(count)]
    bearings: list[list[float | None]] = [[None] * count for _ in range(count)]
    for index in range(count):
        distances[index][index] = 0.0
    reachable_pairs = 0
    for first in range(count):
        for second in range(first + 1, count):
            request = habitat_sim.ShortestPath()
            request.requested_start = points[first]
            request.requested_end = points[second]
            if not pathfinder.find_path(request):
                continue
            distance = float(request.geodesic_distance)
            if not math.isfinite(distance):
                continue
            route = np.asarray(request.points, dtype=np.float64)
            if len(route) < 2:
                continue
            forward = route[1] - route[0]
            reverse = route[-2] - route[-1]
            first_bearing = math.atan2(-float(forward[0]), -float(forward[2]))
            second_bearing = math.atan2(-float(reverse[0]), -float(reverse[2]))
            distances[first][second] = distances[second][first] = distance
            bearings[first][second] = first_bearing
            bearings[second][first] = second_bearing
            reachable_pairs += 1
    return distances, bearings, reachable_pairs


def audit_scene(
    *, parent_manifest: Path, expected_parent_sha256: str,
    protocol_path: Path, scene_index: int, out: Path,
) -> dict[str, Any]:
    import habitat_sim

    protocol = load_protocol(protocol_path)
    require(sha256_file(parent_manifest) == expected_parent_sha256,
            "parent manifest SHA-256 changed")
    require(expected_parent_sha256 == protocol["parent"]["manifest_sha256"],
            "parent hash differs from protocol")
    parent = json.loads(parent_manifest.read_text())
    scenes = parent["scenes"]
    require(len(scenes) == int(protocol["parent"]["expected_scenes"]),
            "parent scene count changed")
    require(0 <= scene_index < len(scenes), "scene index out of range")
    scene = str(scenes[scene_index])
    asset = parent["assets"][scene]
    navmesh = Path(asset["navmesh_path"])
    require(navmesh.is_file(), f"missing navmesh for {scene}")
    require(sha256_file(navmesh) == asset["navmesh_sha256"],
            f"navmesh hash changed for {scene}")

    pathfinder = habitat_sim.PathFinder()
    require(pathfinder.load_nav_mesh(str(navmesh)),
            f"failed to load navmesh for {scene}")
    require(pathfinder.num_islands > 0, f"no navigable island for {scene}")
    areas = [float(pathfinder.island_area(index))
             for index in range(pathfinder.num_islands)]
    island = int(np.argmax(np.asarray(areas)))
    sampling = protocol["sampling"]
    pathfinder.seed(int(sampling["base_seed"]) + scene_index)
    required = int(sampling["points_per_scene"])
    maximum_draws = int(sampling["maximum_draws_per_scene"])
    decimals = int(sampling["deduplicate_rounding_decimals"])
    points: list[np.ndarray] = []
    identities: set[tuple[float, float, float]] = set()
    draws = 0
    while len(points) < required and draws < maximum_draws:
        draws += 1
        point = np.asarray(
            pathfinder.get_random_navigable_point(100, island), dtype=np.float64
        )
        if point.shape != (3,) or not np.isfinite(point).all():
            continue
        identity = tuple(round(float(value), decimals) for value in point)
        if identity in identities:
            continue
        identities.add(identity)
        points.append(point)
    require(len(points) == required,
            f"{scene}: sampled {len(points)}/{required} unique points")

    distances, bearings, reachable = _measure_graph(pathfinder, points)
    serial_points = [[float(value) for value in point] for point in points]
    triads = select_scene_triads(serial_points, distances, bearings, protocol)
    finite_distances = [
        float(distances[first][second])
        for first in range(len(points))
        for second in range(first + 1, len(points))
        if distances[first][second] is not None
    ]
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scene": scene,
        "scene_index": scene_index,
        "protocol_sha256": sha256_file(protocol_path),
        "parent_manifest_sha256": expected_parent_sha256,
        "navmesh_path": str(navmesh),
        "navmesh_sha256": asset["navmesh_sha256"],
        "largest_island_index": island,
        "largest_island_area_m2": areas[island],
        "sample_seed": int(sampling["base_seed"]) + scene_index,
        "sample_draws": draws,
        "sample_points": serial_points,
        "reachable_sample_pairs": reachable,
        "sampled_geodesic_maximum_m": max(finite_distances),
        "candidate_triads": triads,
        "candidate_counts": {name: len(rows) for name, rows in triads.items()},
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "rendered_images_read": False,
        "policy_evaluation_authorized": False,
    }
    atomic_json(out, result)
    return result


def _select_population(
    fragments: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    gate = protocol["prospective_power_gate"]
    target = int(gate["minimum_histories_per_bin"])
    minimum_scenes = int(gate["minimum_scene_clusters_per_bin"])
    maximum_per_scene = int(gate["maximum_selected_histories_per_scene_per_bin"])
    selections: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec in bin_specs(protocol):
        name = str(spec["name"])
        by_scene = {
            row["scene"]: list(row["candidate_triads"][name])
            for row in fragments if row["candidate_triads"][name]
        }
        chosen: list[dict[str, Any]] = []
        # One per scene first guarantees breadth before a second within-scene row.
        for scene in sorted(by_scene):
            candidate = dict(by_scene[scene][0])
            candidate.update({"scene": scene, "scene_candidate_rank": 0})
            chosen.append(candidate)
        for rank in range(1, maximum_per_scene):
            for scene in sorted(by_scene):
                if len(by_scene[scene]) <= rank:
                    continue
                candidate = dict(by_scene[scene][rank])
                candidate.update({"scene": scene, "scene_candidate_rank": rank})
                chosen.append(candidate)
        selected = chosen[:target]
        selected_scenes = len({row["scene"] for row in selected})
        authorized = len(selected) >= target and selected_scenes >= minimum_scenes
        selections[name] = selected
        diagnostics[name] = {
            "eligible_scene_clusters": len(by_scene),
            "available_candidate_triads": sum(
                min(len(rows), maximum_per_scene) for rows in by_scene.values()
            ),
            "selected_histories": len(selected),
            "selected_scene_clusters": selected_scenes,
            "geometry_capacity_gate_passed": authorized,
        }
    return selections, diagnostics


def finalize(
    *, scene_root: Path, parent_manifest: Path,
    expected_parent_sha256: str, protocol_path: Path, out: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    require(sha256_file(parent_manifest) == expected_parent_sha256,
            "parent manifest SHA-256 changed")
    parent = json.loads(parent_manifest.read_text())
    expected = int(protocol["parent"]["expected_scenes"])
    require(len(parent["scenes"]) == expected, "parent scene count changed")
    fragments: list[dict[str, Any]] = []
    fragment_ledger = []
    for index, scene in enumerate(parent["scenes"]):
        path = scene_root / f"{index:02d}_{scene}" / "capacity.json"
        sidecar = path.with_name(path.name + ".sha256")
        require(path.is_file() and sidecar.is_file(),
                f"missing scene fragment {index}:{scene}")
        tokens = sidecar.read_text().strip().split()
        digest = sha256_file(path)
        require(tokens == [digest, path.name],
                f"invalid scene sidecar {index}:{scene}")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == SCHEMA
                and row.get("status") == "complete",
                f"invalid scene fragment {index}:{scene}")
        require(row["scene"] == scene and row["scene_index"] == index,
                f"scene identity changed {index}:{scene}")
        require(row["query_policy_outcomes_read"] is False
                and row["navigation_outcomes_read"] is False
                and row["policy_evaluation_authorized"] is False,
                f"scene fragment crossed authority boundary {index}:{scene}")
        fragments.append(row)
        fragment_ledger.append({
            "scene": scene, "scene_index": index,
            "path": str(path.resolve()), "sha256": digest,
        })
    selections, diagnostics = _select_population(fragments, protocol)
    all_passed = all(
        row["geometry_capacity_gate_passed"] for row in diagnostics.values()
    )
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "verified_construction_stage": True,
        "scope": "navmesh geometry capacity only",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": expected_parent_sha256,
        "scene_fragments": fragment_ledger,
        "scene_count": len(fragments),
        "bin_diagnostics": diagnostics,
        "selected_geometry_proposals": selections,
        "all_geometry_capacity_gates_passed": all_passed,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "rendered_support_verified": False,
        "policy_evaluation_authorized": False,
        "next_required_gate": protocol["authority_boundary"]["next_required_gate"],
    }
    atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scene = sub.add_parser("scene")
    scene.add_argument("--parent-manifest", type=Path, required=True)
    scene.add_argument("--expected-parent-sha256", required=True)
    scene.add_argument("--protocol", type=Path, required=True)
    scene.add_argument("--scene-index", type=int, required=True)
    scene.add_argument("--out", type=Path, required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--scene-root", type=Path, required=True)
    finish.add_argument("--parent-manifest", type=Path, required=True)
    finish.add_argument("--expected-parent-sha256", required=True)
    finish.add_argument("--protocol", type=Path, required=True)
    finish.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scene":
        result = audit_scene(
            parent_manifest=args.parent_manifest,
            expected_parent_sha256=args.expected_parent_sha256,
            protocol_path=args.protocol,
            scene_index=args.scene_index,
            out=args.out,
        )
    else:
        result = finalize(
            scene_root=args.scene_root,
            parent_manifest=args.parent_manifest,
            expected_parent_sha256=args.expected_parent_sha256,
            protocol_path=args.protocol,
            out=args.out,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
