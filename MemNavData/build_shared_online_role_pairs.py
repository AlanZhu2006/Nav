#!/usr/bin/env python3
"""Build matched Novel/Revisit queries after a frozen native online-A prefix.

Revisit goals are reused from the already-audited controlled-pose V1 builder.
For each such goal, a Novel goal is deterministically sampled on the same floor,
at a matched geodesic distance from the exact online-A endpoint, and is accepted
only when its full co-visibility curve over online A stays below the frozen
Novel ceiling.  The two queries are evaluated independently after resetting and
replaying the same online-A history; this builder performs no policy rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audit_shared_online_double_revisit import audit as audit_revisit_source
from build_shared_online_double_revisit import (
    V1_NAME,
    goal_world_points,
    jpeg_bytes,
    load_online_history,
    sha256_file,
    write_depth_png,
)
from generate_twoleg import (
    covis_curve,
    first_path_yaw,
    geodesic,
    make_sim,
    max_covis,
    render,
    wrap_degrees,
    yaw_facing,
)
from shared_online_role_pair_contract import (
    SCHEMA_VERSION,
    validate_manifest,
)


def stable_seed(seed: int, *parts: str) -> int:
    material = "/".join([str(seed), *parts]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def json_vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float)]


def measured_geodesic(pathfinder, first: np.ndarray, second: np.ndarray) -> float:
    ok, distance, _path = geodesic(pathfinder, first, second)
    if not ok or not np.isfinite(distance):
        raise RuntimeError("query has no finite geodesic from online-A endpoint")
    return float(distance)


def optional_geodesic(
    pathfinder, first: np.ndarray, second: np.ndarray
) -> float | None:
    """Return a finite geodesic, or None for a disconnected random sample."""
    ok, distance, _path = geodesic(pathfinder, first, second)
    if not ok or not np.isfinite(distance):
        return None
    return float(distance)


def query_geometry(
    pathfinder, first: np.ndarray, second: np.ndarray
) -> tuple[float, float] | None:
    """Return geodesic distance and first non-trivial path bearing."""
    ok, distance, path = geodesic(pathfinder, first, second)
    if not ok or not np.isfinite(distance) or not path:
        return None
    bearing = float(first_path_yaw(path, first))
    if not np.isfinite(bearing):
        return None
    return float(distance), bearing


def copy_asset(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return sha256_file(destination)


def write_novel_asset(
    rgb: np.ndarray,
    depth: np.ndarray,
    rgb_path: Path,
    depth_path: Path,
) -> tuple[str, str]:
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_bytes(jpeg_bytes(rgb))
    write_depth_png(depth_path, depth)
    return sha256_file(rgb_path), sha256_file(depth_path)


def sample_novel_query(
    simulator,
    history: dict,
    endpoint: np.ndarray,
    *,
    desired_geodesic_m: float,
    desired_initial_path_bearing_rad: float,
    contract: dict,
    seed: int,
    excluded_positions: list[np.ndarray],
) -> dict:
    pathfinder = simulator.pathfinder
    if not hasattr(pathfinder, "seed"):
        raise RuntimeError("Habitat PathFinder has no deterministic seed method")
    pathfinder.seed(int(seed))
    floor_y = float(endpoint[1])
    diagnostics = {
        "attempts": 0,
        "floor_or_clearance_rejects": 0,
        "geodesic_rejects": 0,
        "initial_path_bearing_rejects": 0,
        "duplicate_rejects": 0,
        "stride_covis_rejects": 0,
        "full_covis_rejects": 0,
    }
    for attempt in range(1, int(contract["novel_candidate_attempts"]) + 1):
        diagnostics["attempts"] = attempt
        position = np.asarray(pathfinder.get_random_navigable_point(), dtype=float)
        if (
            not pathfinder.is_navigable(position)
            or abs(float(position[1] - floor_y))
            > float(contract["same_floor_tolerance_m"])
            or float(pathfinder.distance_to_closest_obstacle(position))
            < float(contract["minimum_clearance_m"])
        ):
            diagnostics["floor_or_clearance_rejects"] += 1
            continue
        geometry = query_geometry(pathfinder, endpoint, position)
        if geometry is None:
            diagnostics["geodesic_rejects"] += 1
            continue
        distance, initial_path_bearing = geometry
        if (
            distance < float(contract["minimum_query_geodesic_m"])
            or distance > float(contract["maximum_query_geodesic_m"])
            or abs(distance - float(desired_geodesic_m))
            > float(contract["maximum_role_distance_error_m"])
        ):
            diagnostics["geodesic_rejects"] += 1
            continue
        bearing_error_deg = abs(wrap_degrees(np.degrees(
            initial_path_bearing - float(desired_initial_path_bearing_rad)
        )))
        if bearing_error_deg > float(
            contract["maximum_role_initial_path_bearing_error_deg"]
        ):
            diagnostics["initial_path_bearing_rejects"] += 1
            continue
        previous_distances = [
            optional_geodesic(pathfinder, previous, position)
            for previous in excluded_positions
        ]
        if any(
            distance_to_previous is not None
            and distance_to_previous
            < float(contract["minimum_novel_pair_separation_m"])
            for distance_to_previous in previous_distances
        ):
            diagnostics["duplicate_rejects"] += 1
            continue

        yaw = float(yaw_facing((position - endpoint)[[0, 2]]))
        camera_height = float(
            history["camera_positions"][0][1]
            - history["floor_positions"][0][1]
        )
        rgb, depth = render(
            simulator,
            position + np.asarray([0.0, camera_height, 0.0]),
            yaw,
        )
        points = goal_world_points(
            depth,
            position + np.asarray([0.0, camera_height, 0.0]),
            yaw,
        )
        cheap, _anchor = max_covis(
            points,
            history["transforms"],
            history["depths"],
            stride=int(contract["novel_covis_stride"]),
            tol=float(contract["covis_depth_tolerance_m"]),
        )
        ceiling = float(contract["novel_max_online_a_covis_exclusive"])
        if float(cheap) >= ceiling:
            diagnostics["stride_covis_rejects"] += 1
            continue
        curve = covis_curve(
            points,
            history["transforms"],
            history["depths"],
            tol=float(contract["covis_depth_tolerance_m"]),
        )
        maximum = float(curve.max()) if len(curve) else 0.0
        if maximum >= ceiling:
            diagnostics["full_covis_rejects"] += 1
            continue
        return {
            "position": position,
            "yaw": yaw,
            "rgb": rgb,
            "depth": depth,
            "geodesic_m": distance,
            "initial_path_bearing_rad": initial_path_bearing,
            "initial_path_bearing_error_deg": bearing_error_deg,
            "max_online_a_covis": maximum,
            "max_online_a_covis_frame": (
                int(np.argmax(curve)) if len(curve) else None
            ),
            "covis_curve": [float(value) for value in curve],
            "sampling_seed": int(seed),
            "sampling_diagnostics": diagnostics,
            "goal_yaw_contract": "online_a_endpoint_to_goal_approach_heading",
        }
    raise RuntimeError(
        "no distance-matched Novel query passed the full online-A support gate: "
        + json.dumps(diagnostics, sort_keys=True)
    )


def build_episode(
    online_episode: Path,
    revisit_root: Path,
    revisit_episode: dict,
    destination: Path,
    *,
    contract: dict,
    global_seed: int,
) -> dict:
    scene = str(revisit_episode["scene"])
    episode = str(revisit_episode["episode"])
    receipt_path = online_episode / "receipt.json"
    trace_path = online_episode / "online_a_trace.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["scene"] != scene or receipt["episode"] != episode:
        raise RuntimeError("online-A identity differs from Revisit source")
    if sha256_file(receipt_path) != revisit_episode["source_online_receipt_sha256"]:
        raise RuntimeError("online-A receipt hash differs from Revisit source")
    if sha256_file(trace_path) != revisit_episode["source_online_trace_sha256"]:
        raise RuntimeError("online-A trace hash differs from Revisit source")
    history = load_online_history(online_episode, receipt)
    endpoint = np.asarray(history["trace"]["end_position"], dtype=float)
    endpoint_yaw = float(history["trace"]["end_yaw"])
    source_variant = revisit_episode["variants"][V1_NAME]
    source_episode_dir = revisit_root / scene / episode / V1_NAME
    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
    selected = []
    excluded_novel_positions: list[np.ndarray] = []
    source_attempts = []
    try:
        source_roles = tuple(sorted(source_variant["goals"]))
        if not source_roles:
            raise RuntimeError("Revisit source has no controlled V1 goals")
        for source_role in source_roles:
            if len(selected) >= int(contract["pairs_per_online_history"]):
                break
            revisit_goal = source_variant["goals"][source_role]
            revisit_position = np.asarray(
                revisit_goal["floor_position"], dtype=float
            )
            if (
                abs(float(revisit_position[1] - endpoint[1]))
                > float(contract["same_floor_tolerance_m"])
            ):
                continue
            revisit_geometry = query_geometry(
                simulator.pathfinder, endpoint, revisit_position
            )
            if revisit_geometry is None:
                continue
            revisit_distance, revisit_initial_path_bearing = revisit_geometry
            if not (
                float(contract["minimum_query_geodesic_m"])
                <= revisit_distance
                <= float(contract["maximum_query_geodesic_m"])
            ):
                continue
            pair_index = len(selected)
            pair_id = f"pair_{pair_index:02d}"
            source_seed = stable_seed(
                global_seed, scene, episode, source_role, "novel"
            )
            try:
                novel = sample_novel_query(
                    simulator,
                    history,
                    endpoint,
                    desired_geodesic_m=revisit_distance,
                    desired_initial_path_bearing_rad=(
                        revisit_initial_path_bearing
                    ),
                    contract=contract,
                    seed=source_seed,
                    excluded_positions=excluded_novel_positions,
                )
            except RuntimeError as error:
                source_attempts.append({
                    "source_controlled_revisit_role": source_role,
                    "constructible": False,
                    "sampling_seed": int(source_seed),
                    "reason": str(error),
                })
                continue
            source_attempts.append({
                "source_controlled_revisit_role": source_role,
                "constructible": True,
                "sampling_seed": int(source_seed),
                "reason": "matched_novel_query_found",
            })
            excluded_novel_positions.append(novel["position"])

            pair_root = destination / pair_id
            revisit_rgb = source_episode_dir / source_variant["assets"][source_role]["rgb"]
            revisit_depth = source_episode_dir / source_variant["assets"][source_role]["depth"]
            revisit_rgb_out = pair_root / "revisit" / "goal.jpg"
            revisit_depth_out = pair_root / "revisit" / "goal_depth.png"
            revisit_rgb_sha = copy_asset(revisit_rgb, revisit_rgb_out)
            revisit_depth_sha = copy_asset(revisit_depth, revisit_depth_out)
            if revisit_rgb_sha != source_variant["assets"][source_role]["rgb_sha256"]:
                raise RuntimeError("copied Revisit RGB hash changed")
            if revisit_depth_sha != source_variant["assets"][source_role]["depth_sha256"]:
                raise RuntimeError("copied Revisit depth hash changed")

            novel_rgb_out = pair_root / "novel" / "goal.jpg"
            novel_depth_out = pair_root / "novel" / "goal_depth.png"
            novel_rgb_sha, novel_depth_sha = write_novel_asset(
                novel["rgb"],
                novel["depth"],
                novel_rgb_out,
                novel_depth_out,
            )
            queries = [
                {
                    "query_id": f"{pair_id}_novel",
                    "analysis_role": "novel",
                    "goal_rgb": str(novel_rgb_out.relative_to(destination)),
                    "goal_rgb_sha256": novel_rgb_sha,
                    "goal_depth": str(novel_depth_out.relative_to(destination)),
                    "goal_depth_sha256": novel_depth_sha,
                    "floor_position": json_vector(novel["position"]),
                    "yaw_rad": float(novel["yaw"]),
                    "geodesic_from_a_end_m": float(novel["geodesic_m"]),
                    "initial_path_bearing_rad": float(
                        novel["initial_path_bearing_rad"]
                    ),
                    "max_online_a_covis": float(novel["max_online_a_covis"]),
                    "max_online_a_covis_frame": novel[
                        "max_online_a_covis_frame"
                    ],
                    "covis_curve": novel["covis_curve"],
                    "sampling_seed": novel["sampling_seed"],
                    "sampling_diagnostics": novel["sampling_diagnostics"],
                    "goal_yaw_contract": novel["goal_yaw_contract"],
                },
                {
                    "query_id": f"{pair_id}_revisit",
                    "analysis_role": "revisit",
                    "goal_rgb": str(revisit_rgb_out.relative_to(destination)),
                    "goal_rgb_sha256": revisit_rgb_sha,
                    "goal_depth": str(revisit_depth_out.relative_to(destination)),
                    "goal_depth_sha256": revisit_depth_sha,
                    "floor_position": json_vector(revisit_position),
                    "yaw_rad": float(revisit_goal["yaw_rad"]),
                    "geodesic_from_a_end_m": float(revisit_distance),
                    "initial_path_bearing_rad": float(
                        revisit_initial_path_bearing
                    ),
                    "max_online_a_covis": float(
                        revisit_goal["max_online_a_covis"]
                    ),
                    "max_online_a_covis_frame": int(
                        revisit_goal["max_online_a_covis_frame"]
                    ),
                    "eligible_online_a_frame_floor": int(
                        revisit_goal["eligible_online_a_frame_floor"]
                    ),
                    "covis_curve": [
                        float(value) for value in revisit_goal["covis_curve"]
                    ],
                    "source_controlled_revisit_role": source_role,
                    "source_online_frame": int(
                        revisit_goal["source_online_frame"]
                    ),
                    "translation_from_source_m": float(
                        revisit_goal["translation_from_source_m"]
                    ),
                    "yaw_delta_from_source_deg": float(
                        revisit_goal["yaw_delta_from_source_deg"]
                    ),
                    "pixel_mae_from_source": float(
                        revisit_goal["pixel_mae_from_source"]
                    ),
                },
            ]
            selected.append(
                {
                    "pair_id": pair_id,
                    "role_distance_error_m": abs(
                        float(novel["geodesic_m"]) - revisit_distance
                    ),
                    "role_initial_path_bearing_error_deg": float(
                        novel["initial_path_bearing_error_deg"]
                    ),
                    "queries": queries,
                }
            )
    finally:
        simulator.close()

    if len(selected) != int(contract["pairs_per_online_history"]):
        raise RuntimeError(
            f"only {len(selected)} independent role pairs could be built for "
            f"{scene}/{episode}; required {contract['pairs_per_online_history']}; "
            f"source attempts={json.dumps(source_attempts, sort_keys=True)}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "episode": episode,
        "online_a_episode": str(online_episode.resolve()),
        "online_a_receipt_sha256": sha256_file(receipt_path),
        "online_a_trace_sha256": sha256_file(trace_path),
        "online_a_steps": len(history["poses"]),
        "online_a_endpoint": {
            "floor_position": json_vector(endpoint),
            "yaw_rad": endpoint_yaw,
        },
        "source_revisit_benchmark_sha256": sha256_file(
            revisit_root / scene / episode / "benchmark.json"
        ),
        "constructibility_attempts": source_attempts,
        "pairs": selected,
    }
    metadata_path = destination / "role_pairs.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    payload["role_pairs_sha256"] = sha256_file(metadata_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--revisit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs-per-history", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--minimum-query-geodesic", type=float, default=2.0)
    parser.add_argument("--maximum-query-geodesic", type=float, default=9.0)
    parser.add_argument("--distance-match-tolerance", type=float, default=0.50)
    parser.add_argument("--initial-path-bearing-tolerance-deg", type=float, default=30.0)
    parser.add_argument("--novel-covis", type=float, default=0.10)
    parser.add_argument("--minimum-clearance", type=float, default=0.30)
    parser.add_argument("--same-floor-tolerance", type=float, default=0.20)
    parser.add_argument("--minimum-novel-separation", type=float, default=1.0)
    parser.add_argument("--novel-candidate-attempts", type=int, default=5000)
    parser.add_argument("--covis-stride", type=int, default=4)
    parser.add_argument("--covis-depth-tolerance", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    if not 1 <= args.pairs_per_history <= 2:
        raise ValueError("pairs-per-history must be one or two for B/C V1 sources")
    source_audit = audit_revisit_source(args.revisit_root)
    if not source_audit["ok"]:
        raise RuntimeError("controlled Revisit source audit failed")
    online_manifest = json.loads((args.online_root / "manifest.json").read_text())
    if online_manifest.get("schema_version") != "shared_online_a_materialized_v1":
        raise RuntimeError("online-A source schema changed")
    online_index = {
        (str(row["scene"]), str(row["episode"])): row
        for row in online_manifest["episodes"]
    }
    revisit_manifest = json.loads((args.revisit_root / "manifest.json").read_text())
    revisit_contract = revisit_manifest["contract"]
    contract = {
        "online_history": "frozen_native_navdp_goal_a_rgb_depth_pose_trace",
        "query_execution": "independent_reset_and_exact_online_a_replay",
        "runtime_role_visibility": "none",
        "analysis_role_location": "sidecar_only_never_forwarded_to_policy",
        "pairs_per_online_history": int(args.pairs_per_history),
        "minimum_query_geodesic_m": float(args.minimum_query_geodesic),
        "maximum_query_geodesic_m": float(args.maximum_query_geodesic),
        "maximum_role_distance_error_m": float(args.distance_match_tolerance),
        "maximum_role_initial_path_bearing_error_deg": float(
            args.initial_path_bearing_tolerance_deg
        ),
        "novel_max_online_a_covis_exclusive": float(args.novel_covis),
        "revisit_min_online_a_covis_inclusive": float(
            revisit_contract["v1_min_max_online_a_covis"]
        ),
        "revisit_max_online_a_covis_inclusive": float(
            revisit_contract["v1_max_max_online_a_covis"]
        ),
        "minimum_clearance_m": float(args.minimum_clearance),
        "same_floor_tolerance_m": float(args.same_floor_tolerance),
        "minimum_novel_pair_separation_m": float(
            args.minimum_novel_separation
        ),
        "novel_candidate_attempts": int(args.novel_candidate_attempts),
        "novel_covis_stride": int(args.covis_stride),
        "covis_depth_tolerance_m": float(args.covis_depth_tolerance),
    }
    if not (
        0.0
        <= contract["novel_max_online_a_covis_exclusive"]
        < contract["revisit_min_online_a_covis_inclusive"]
    ):
        raise ValueError("Novel and Revisit support bands overlap")
    if not (
        0.0 < contract["minimum_query_geodesic_m"]
        < contract["maximum_query_geodesic_m"]
    ):
        raise ValueError("invalid query distance band")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=args.out.name + ".tmp.", dir=args.out.parent)
    )
    episodes = []
    try:
        for revisit_episode in revisit_manifest["episodes"]:
            key = (
                str(revisit_episode["scene"]),
                str(revisit_episode["episode"]),
            )
            if key not in online_index:
                raise RuntimeError(f"missing online-A source for {key}")
            online_episode = args.online_root / key[0] / key[1]
            destination = temporary / key[0] / key[1]
            destination.mkdir(parents=True)
            episodes.append(
                build_episode(
                    online_episode,
                    args.revisit_root,
                    revisit_episode,
                    destination,
                    contract=contract,
                    global_seed=int(args.seed),
                )
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "matched role-free Novel/Revisit queries after an exact frozen "
                "native online-A history"
            ),
            "source_online_root": str(args.online_root.resolve()),
            "source_online_manifest_sha256": sha256_file(
                args.online_root / "manifest.json"
            ),
            "source_revisit_root": str(args.revisit_root.resolve()),
            "source_revisit_manifest_sha256": source_audit["manifest_sha256"],
            "construction_seed": int(args.seed),
            "contract": contract,
            "episodes": episodes,
        }
        validate_manifest(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        (temporary / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        temporary.replace(args.out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.out),
                "episodes": len(episodes),
                "pairs": sum(len(row["pairs"]) for row in episodes),
                "queries": 2 * sum(len(row["pairs"]) for row in episodes),
                "scenes": sorted({row["scene"] for row in episodes}),
                "manifest_sha256": sha256_file(args.out / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
