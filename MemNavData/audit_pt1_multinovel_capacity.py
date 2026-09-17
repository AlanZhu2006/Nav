"""Outcome-blind PT1 scene-capacity diagnostic for three Novel stages.

`extract` reads saved expert poses, not policy results. `spatial` rebakes each
scene at the current constructor radius and measures candidate supply around
the recorded starts. Distance from the traversed path is a spatial diagnostic,
NOT visual novelty. Neither command constructs a formal population or runs a
navigation policy. Existing datasets and running protocols remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from MemNavData.final14_role_pair_contract import (
    STRATA, direction_in_stratum, relative_direction_degrees, stable_u32,
)
from MemNavData.table2_novel_sampling import initial_a_yaw, goal_yaw

SCHEMA = "pt1_multinovel_capacity_diagnostic_20260911_v1"
DISTANCE_BINS = ("2_to_4", "4_to_6", "6_to_9")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    if not np.isfinite(a).all():
        raise ValueError("Distribution contains non-finite measurements")
    return dict(n=len(a), **dict(zip(
        ("min", "median", "p90", "max"),
        map(float, np.quantile(a, (0, .5, .9, 1))),
    )))


def distance_bin(distance):
    if not math.isfinite(distance) or not 2 <= distance <= 9:
        return None
    return DISTANCE_BINS[0] if distance < 4 else DISTANCE_BINS[1] if distance < 6 else DISTANCE_BINS[2]


def distances_to_recorded_path(points, history):
    """Euclidean 3-D distance to saved camera-floor positions, not covisibility."""
    points, history = np.asarray(points, dtype=float), np.asarray(history, dtype=float)
    if len(history) == 0:
        return np.full(len(points), np.inf)
    return np.concatenate([
        np.linalg.norm(chunk[:, None] - history[None], axis=-1).min(axis=1)
        for chunk in np.array_split(points, max(1, math.ceil(len(points) / 128)))
    ])


def trajectory_metrics(positions, switches):
    """Split every recorded transition exactly once; no unsaved initial step."""
    p = np.asarray(positions, dtype=float)
    a, b = map(int, switches)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or not 1 <= a < b < len(p):
        raise ValueError("Expected a finite three-leg pose sequence")
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    # switches are first frame indices of the next leg. Boundary displacement
    # belongs to the arriving leg B/C, rather than being silently dropped.
    slices = (slice(0, a - 1), slice(a - 1, b - 1), slice(b - 1, None))
    lengths = [float(step[s].sum()) for s in slices]
    weighted = {}
    for label, start, end in (("B", a, b), ("C", b, len(p))):
        separation = distances_to_recorded_path(p[start:end], p[:start])
        movement = step[start - 1:end - 1]
        length = float(movement.sum())
        weighted[label] = {
            f"path_fraction_outside_prior_{radius:g}m_tube":
                float(movement[separation > radius].sum() / length) if length else None
            for radius in (1., 2.)
        }
    return dict(recorded_path_m=float(step.sum()), leg_path_m=lengths,
                max_step_m=float(step.max()),
                spatial_cell_count_0_5m=len({tuple(v) for v in np.floor(p / .5).astype(int)}),
                endpoint_displacement_m=float(np.linalg.norm(p[-1] - p[0])),
                relative_to_prior_path=weighted)


def extract(supervision_root, asset_root, out):
    import pandas as pd
    from MemNavData.habitat_rollout_primitives import parquet_data_pose_to_habitat

    paths = sorted(Path(supervision_root).glob("*/episode_*/meta/gen_meta.json"))
    if not paths:
        raise ValueError("No source metadata")
    histories = []
    for path in paths:
        meta = json.loads(path.read_text())
        scene, episode = path.parents[2].name, path.parents[1].name
        parquet = path.parent.parent / "data/chunk-000/episode_000000.parquet"
        frame = pd.read_parquet(parquet, columns=["action", "observation.camera_extrinsic"])
        if len(frame) != meta["n_frames"] or meta["n_legs"] != 3:
            raise ValueError(f"Unexpected episode structure: {path}")
        poses = []
        for _, row in frame.iterrows():
            pose = parquet_data_pose_to_habitat(
                np.asarray(row["action"].tolist(), dtype=float).reshape(4, 4),
                np.asarray(row["observation.camera_extrinsic"].tolist(), dtype=float).reshape(4, 4),
                camera_height_m=float(meta.get("camera_height_m", .5)),
                frame_convention=meta["frame_convention"],
            )
            poses.append([*pose.position.tolist(), pose.yaw_rad])
        positions = np.asarray(poses)[:, :3]
        asset = Path(asset_root) / f"{scene}.glb"
        source_id = f"{scene}/{episode}"
        a, b = map(int, meta["switches"])
        states = [
            dict(stage="A_start", frame=0, history_frames=0, yaw=initial_a_yaw(source_id),
                 yaw_origin="independent new A initialization; no rollout executed"),
            dict(stage="B_after_A", frame=a - 1, history_frames=a, yaw=poses[a - 1][3],
                 yaw_origin="recorded expert A terminal orientation"),
            dict(stage="C_after_AB", frame=b - 1, history_frames=b, yaw=poses[b - 1][3],
                 yaw_origin="recorded expert B terminal orientation"),
        ]
        for state in states:
            state["position"] = positions[state["frame"]].tolist()
        histories.append(dict(
            source_id=source_id, scene=scene, episode=episode,
            role_sequence=["initial_imagegoal"] + [g["kind"] for g in meta["goals"]],
            n_frames=len(frame), switches=[a, b],
            old_A_geodesic_m=float(meta["geo_startA"]),
            asset=str(asset.resolve()), asset_available=asset.is_file(),
            metadata=str(path.resolve()), metadata_sha256=digest(path),
            parquet=str(parquet.resolve()), parquet_sha256=digest(parquet),
            camera_height_m=float(meta.get("camera_height_m", .5)),
            camera_height_origin="metadata" if "camera_height_m" in meta else "legacy generator fixed 0.5 m",
            states=states, positions=positions.tolist(),
            **trajectory_metrics(positions, meta["switches"]),
        ))
    result = dict(
        schema=SCHEMA, phase="expert_trajectory_inventory", code_sha256=digest(__file__),
        population="local PT1 / audited-gapfill supervision subset, not the complete PT1 pool",
        expert_history=True, actual_online_history=False, policy_outcomes_read=False,
        visual_support_measured=False, navigation_rollouts=0,
        formal_population_created=False, scene_count=len({r["scene"] for r in histories}),
        summary={
            "history_count": len(histories),
            "recorded_path_m": distribution([r["recorded_path_m"] for r in histories]),
            "old_A_geodesic_m": distribution([r["old_A_geodesic_m"] for r in histories]),
            "leg_path_m": [distribution([r["leg_path_m"][i] for r in histories]) for i in range(3)],
            "total_over_30m": sum(r["recorded_path_m"] > 30 for r in histories),
        }, histories=histories,
    )
    save(out, result)
    print(json.dumps(result["summary"], indent=2))


def shortest(pathfinder, first, second):
    import habitat_sim
    request = habitat_sim.ShortestPath()
    request.requested_start, request.requested_end = np.asarray(first), np.asarray(second)
    if not pathfinder.find_path(request) or not math.isfinite(request.geodesic_distance):
        return None
    return float(request.geodesic_distance), np.asarray(request.points, dtype=float)


def route_on_floor(route, floor_y, tolerance=.20):
    return bool(np.all(np.abs(np.asarray(route)[:, 1] - floor_y) <= tolerance))


def spatial_supply(pf, state, history, key, attempts):
    from MemNavData.generate_twoleg import first_path_yaw
    start = np.asarray(state["position"], dtype=float)
    snapped = np.asarray(pf.snap_point(start), dtype=float)
    snap_distance = float(np.linalg.norm(snapped - start)) if np.isfinite(snapped).all() else None
    # Do not move a historical start to improve its measured supply.
    if snap_distance is None or snap_distance > .05:
        return dict(state=state, status="recorded_start_off_current_navmesh",
                    snap_distance_m=snap_distance, candidates=[], counts={}, visual_support_measured=False)
    pf.seed(stable_u32(SCHEMA, key))
    counts, candidates, seen = Counter(), [], set()
    for attempt in range(attempts):
        counts["attempts"] += 1
        target = np.asarray(pf.get_random_navigable_point(), dtype=float)
        if not np.isfinite(target).all() or not pf.is_navigable(target):
            counts["non_navigable"] += 1
            continue
        if abs(target[1] - start[1]) > .20:
            counts["other_floor"] += 1
            continue
        if pf.distance_to_closest_obstacle(target) < .30:
            counts["clearance"] += 1
            continue
        measured = shortest(pf, start, target)
        if measured is None:
            counts["unreachable"] += 1
            continue
        distance, route = measured
        band = distance_bin(distance)
        if band is None:
            counts["distance_band"] += 1
            continue
        if not route_on_floor(route, start[1]):
            counts["route_leaves_floor"] += 1
            continue
        cell = tuple(np.floor(target[[0, 2]] / .5).astype(int))
        if cell in seen:
            counts["duplicate_spatial_cell"] += 1
            continue
        seen.add(cell)
        relative = relative_direction_degrees(first_path_yaw(route, start), state["yaw"])
        direction = next(s for s in STRATA if direction_in_stratum(relative, s))
        counts["retained_cells"] += 1
        counts[f"retained_{direction}"] += 1
        counts[f"retained_{band}"] += 1
        candidates.append(dict(proposal_index=attempt, floor_position=target.tolist(),
            yaw_rad=goal_yaw(key, attempt), geodesic_m=distance,
            distance_bin=band, direction=direction, relative_route_angle_deg=relative))
    separations = distances_to_recorded_path(
        np.asarray([r["floor_position"] for r in candidates]).reshape(-1, 3), history,
    ) if candidates else []
    for row, distance in zip(candidates, separations):
        row["distance_to_prior_path_m"] = float(distance) if math.isfinite(distance) else None
        for radius in (1., 2.):
            if distance > radius:
                counts[f"outside_prior_{radius:g}m_tube"] += 1
                counts[f"outside_prior_{radius:g}m_{row['distance_bin']}_{row['direction']}"] += 1
    return dict(state=state, status="measured", snap_distance_m=snap_distance,
        counts=dict(counts), candidates=candidates, visual_support_measured=False,
        warning="outside a spatial history tube does not imply unsupported Novel ImageGoal")


def spatial(inventory_path, out_dir, attempts, scenes=None):
    import habitat_sim
    data = json.loads(Path(inventory_path).read_text())
    if data["schema"] != SCHEMA:
        raise ValueError("Wrong inventory schema")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    all_scenes = sorted({r["scene"] for r in data["histories"]})
    for scene in all_scenes:
        if scenes and scene not in scenes:
            continue
        histories = [r for r in data["histories"] if r["scene"] == scene]
        asset = Path(histories[0]["asset"])
        if not asset.is_file():
            rows.append(dict(scene=scene, status="asset_missing", asset=str(asset)))
            continue
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(asset)
        cfg.enable_physics = cfg.create_renderer = cfg.load_semantic_mesh = False
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = []
        with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius, settings.agent_height = .30, 1.5
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError(f"NavMesh recomputation failed: {scene}")
            navmesh = out_dir / f"{scene}.navmesh"
            sim.pathfinder.save_nav_mesh(str(navmesh))
            results = []
            for source in histories:
                positions = np.asarray(source["positions"])
                states = [spatial_supply(sim.pathfinder, state,
                    positions[:state["history_frames"]], source["source_id"] + "/" + state["stage"], attempts)
                    for state in source["states"]]
                ab = shortest(sim.pathfinder, source["states"][1]["position"], source["states"][2]["position"])
                results.append(dict(source_id=source["source_id"], states=states,
                    recorded_path_m=source["recorded_path_m"],
                    expert_AB_terminal_geodesic_m=None if ab is None else ab[0],
                    expert_AB_in_2_to_9m=ab is not None and distance_bin(ab[0]) is not None))
            row = dict(scene=scene, status="measured", asset=str(asset), asset_sha256=digest(asset),
                navmesh=str(navmesh.resolve()), navmesh_sha256=digest(navmesh),
                agent_radius_m=.30, agent_height_m=1.5,
                all_floors_navigable_area_m2=float(sim.pathfinder.navigable_area),
                source_results=results)
            save(out_dir / f"{scene}.json", row)
            rows.append(row)
            print(scene, [(r["source_id"], [s["counts"].get("retained_cells", 0)
                  for s in r["states"]]) for r in results], flush=True)
    result = dict(schema=SCHEMA, phase="offline_spatial_supply", expert_history=True,
        actual_online_history=False, policy_outcomes_read=False, visual_support_measured=False,
        navigation_rollouts=0, formal_population_created=False,
        inventory_sha256=digest(inventory_path), code_sha256=digest(__file__),
        protocol=dict(attempts_per_state=attempts, target_distance_m=[2, 9],
            maximum_floor_height_difference_m=.20, whole_route_same_floor=True,
            target_clearance_m=.30, spatial_cell_width_m=.5,
            history_tubes_m=[1, 2], history_tubes_are_not_novelty_thresholds=True),
        scene_results=rows)
    save(out_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("extract")
    inventory.add_argument("--supervision-root", type=Path, required=True)
    inventory.add_argument("--asset-root", type=Path, required=True)
    inventory.add_argument("--out", type=Path, required=True)
    geometry = commands.add_parser("spatial")
    geometry.add_argument("--inventory", type=Path, required=True)
    geometry.add_argument("--out-dir", type=Path, required=True)
    geometry.add_argument("--attempts", type=int, default=4000)
    geometry.add_argument("--scene", action="append")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.supervision_root, args.asset_root, args.out)
    else:
        if args.attempts <= 0:
            parser.error("--attempts must be positive")
        spatial(args.inventory, args.out_dir, args.attempts, args.scene)


if __name__ == "__main__":
    main()
