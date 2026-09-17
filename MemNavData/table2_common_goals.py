"""Common next goals measured against every surviving arm's actual history.

Evaluator only: this module renders task images and measures benchmark support.
It never calls a policy. Arm names do not determine proposal or selection order.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_local import load, dump, sha, history_at, revisit_source_frames, support_sources
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA
from MemNavData.table2_balanced_sampling import (
    BINS, SCHEMA, SPATIAL_ATTEMPTS, NOVEL_VIEWS_PER_CELL, REVISIT_VISUAL_BUDGET,
    novel_supported_as_task,
)
from MemNavData.final14_role_pair_contract import (
    stable_u32, relative_direction_degrees, direction_in_stratum, STRATA, DEPTH_TOLERANCE_M,
)
from MemNavData.audit_pt1_multinovel_capacity import distance_bin, shortest
from MemNavData.table2_novel_sampling import goal_yaw


def witness_frames(poses, position, yaw):
    """Original source-frame age, displacement and view-jitter rules, per arm."""
    candidates = []
    for index in revisit_source_frames(len(poses)):
        pose = poses[index]
        origin = np.asarray([pose[k] for k in "xyz"])
        distance = float(np.linalg.norm((np.asarray(position) - origin)[[0, 2]]))
        angle = abs(relative_direction_degrees(yaw, pose["yaw"]))
        if .20 <= distance <= .80 and 12 <= angle <= 45:
            candidates.append((index, distance, angle))
    return candidates


def common_band(geometries):
    """A symmetric distance address; each arm's actual distance stays in its query."""
    distances = sorted(float(g["geodesic_m"]) for g in geometries.values())
    return distance_bin(math.fsum(distances) / len(distances))


def choose_common(menu, key, distance_counts=None):
    counts = distance_counts or {}
    candidates = menu["candidates"]
    if not candidates:
        return None
    return min(candidates, key=lambda c: (
        counts.get(c["distance_band"], 0),
        stable_u32(SCHEMA, key, c["distance_band"]), c["goal_rgb_sha256"]))


def source_address(pose):
    # Independent of controller name and of its position in a Python dict.
    return hashlib.sha256(repr(tuple(pose[k] for k in ("x", "y", "z", "yaw", "jpg_sha256"))).encode()).hexdigest()


def construct_common(source, *, stage, prefixes, role, out):
    from MemNavData.generate_twoleg import make_sim, render, covis_curve, covis_frac, cam_to_world_hab, first_path_yaw
    from MemNavData.build_shared_online_double_revisit import goal_world_points, jpeg_bytes, pixel_mae
    from MemNavData.build_final14_role_pair_scene import deterministic_pose_grid

    if stage not in ("B", "C") or role not in ("novel", "revisit") or not prefixes:
        raise ValueError("A common successor requires B/C, a role, and surviving histories")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    histories = {a: history_at(p) for a, p in prefixes.items()}
    if not all(h["trace"]["reached"] for h in histories.values()):
        raise ValueError("A failed arm cannot be required to execute a later task")
    height = float(source["camera_height_m"])
    key = f"{source['scene']}/{source['episode']}/{stage}/history_before_{stage}"
    sim = make_sim(source["asset"], "", agent_radius=.30)
    pf = sim.pathfinder
    counts, checked, retained, candidates = Counter(), [], set(), []
    states = {}
    for arm, h in histories.items():
        t = h["trace"]
        position = np.asarray(t["end_position"])
        _, depth = render(sim, position + [0, height, 0], t["end_yaw"])
        states[arm] = dict(position=position, yaw=t["end_yaw"], current_depth=depth,
                           current_t=cam_to_world_hab(position + [0, height, 0], t["end_yaw"]))

    def geometry(position):
        measured = {}
        for arm, state in states.items():
            start = state["position"]
            path = shortest(pf, start, position)
            if (path is None or distance_bin(path[0]) is None
                    or np.max(np.abs(path[1][:, 1] - start[1])) > .20):
                return None
            angle = relative_direction_degrees(first_path_yaw(path[1], start), state["yaw"])
            if role == "novel" and abs(angle) > 60:
                return None
            straight = float(np.linalg.norm((position - start)[[0, 2]]))
            direction = next(s for s in STRATA if direction_in_stratum(angle, s))
            measured[arm] = dict(geodesic_m=float(path[0]), straight_distance_m=straight,
                route_ratio=float(path[0]) / straight, initial_relative_route_angle_deg=angle,
                direction_stratum=direction, cell=f"{distance_bin(path[0])}/{direction}")
        return measured

    def inspect(position, yaw, geometries, proposal):
        band = common_band(geometries)
        if band in retained:
            return
        rgb, depth = render(sim, position + [0, height, 0], yaw)
        points = goal_world_points(depth, position + [0, height, 0], yaw)
        counts["visual_candidates"] += 1
        observations, audit = {}, {}
        for arm, h in histories.items():
            curve = covis_curve(points, h["transforms"], h["depths"], tol=DEPTH_TOLERANCE_M)
            state = states[arm]
            current = float(covis_frac(points, state["current_t"], state["current_depth"], tol=DEPTH_TOLERANCE_M))
            maximum = float(curve.max()) if len(curve) else 0.
            eligible = float(curve[8:].max()) if len(curve) > 8 else 0.
            witness = None
            if role == "revisit":
                for frame, shift, angle in witness_frames(h["poses"], position, yaw):
                    mae = float(pixel_mae(rgb, h["rgbs"][frame]))
                    if mae >= 5:
                        witness = dict(source_online_frame=frame, translation_from_source_m=shift,
                                       yaw_delta_from_source_deg=angle, pixel_mae_from_source=mae)
                        break
                accepted = bool(len(points) > 0 and current < .10 and .55 <= maximum <= .90
                                and eligible >= .55 and witness is not None)
            else:
                accepted = bool(novel_supported_as_task(len(points), curve, current))
            observations[arm] = dict(covis_curve=curve.tolist(), max_history_covis=maximum,
                max_runtime_eligible_covis=eligible, runtime_eligible_frame_floor=8,
                construction_source_frame_floor=39, history_frames=len(curve),
                current_view_covis=current, goal_surface_points=len(points),
                **(witness or {}), **support_sources(curve, h["trace"].get("prefix_A_steps", len(curve))))
            audit[arm] = dict(accepted=accepted, geodesic_m=geometries[arm]["geodesic_m"],
                initial_route_angle_deg=geometries[arm]["initial_relative_route_angle_deg"],
                max_history_covis=maximum, current_view_covis=current, perturbation_witness=witness)
        accepted = all(v["accepted"] for v in audit.values())
        checked.append(dict(proposal=proposal, distance_band=band, arms=audit, accepted=accepted))
        if not accepted:
            return
        directory = out / band
        directory.mkdir()
        image = directory / "goal.jpg"
        image.write_bytes(jpeg_bytes(rgb))
        queries = {}
        for arm, h in histories.items():
            q = dict(deepcopy(source), schema=FORWARD_SCHEMA, stage_number="ABC".index(stage),
                start_position=states[arm]["position"].tolist(), start_yaw=states[arm]["yaw"],
                prefix_root=str(Path(prefixes[arm]).resolve()),
                prefix_trace_sha256=sha(Path(prefixes[arm]) / "online_a_trace.json"),
                analysis_role=role, floor_position=position.tolist(), yaw_rad=yaw,
                goal_rgb=str(image), goal_rgb_sha256=sha(image),
                **geometries[arm], **observations[arm])
            path = directory / f"query_{arm}.json"
            dump(path, q)
            queries[arm] = str(path)
        candidates.append(dict(distance_band=band, goal_rgb_sha256=sha(image), queries=queries))
        retained.add(band)

    try:
        if role == "novel":
            pf.seed(stable_u32(SCHEMA, "spatial", key))
            seen, spatial = set(), Counter()
            for proposal in range(SPATIAL_ATTEMPTS):
                counts["spatial_attempts"] += 1
                position = np.asarray(pf.get_random_navigable_point(), dtype=float)
                if not np.isfinite(position).all() or not pf.is_navigable(position):
                    continue
                address = tuple(np.floor(position[[0, 2]] / .25).astype(int))
                if address in seen or pf.distance_to_closest_obstacle(position) < .30:
                    continue
                seen.add(address)
                geometries = geometry(position)
                if geometries is None:
                    continue
                band = common_band(geometries)
                if spatial[band] >= NOVEL_VIEWS_PER_CELL:
                    continue
                spatial[band] += 1
                inspect(position, goal_yaw(key, proposal), geometries, proposal)
                if len(retained) == len(BINS):
                    break
        else:
            # Unique visual source addresses, not "native first". Exchanging
            # arm labels leaves this proposal set and order unchanged.
            addresses = {}
            for h in histories.values():
                for frame in revisit_source_frames(len(h["poses"])):
                    pose = h["poses"][frame]
                    addresses.setdefault(source_address(pose), pose)
            grids = [(address, addresses[address], list(deterministic_pose_grid(key + "/" + address)))
                     for address in sorted(addresses)]
            seen = set()
            for attempt in range(max((len(g) for _, _, g in grids), default=0)):
                for address, pose, grid in grids:
                    if attempt >= len(grid):
                        continue
                    radius, angle, dyaw = grid[attempt]
                    if radius > .8 or not 12 <= abs(dyaw) <= 45:
                        continue
                    origin = np.asarray([pose[k] for k in "xyz"])
                    raw = origin + radius * np.asarray([math.cos(angle), 0, math.sin(angle)])
                    position = np.asarray(pf.snap_point(raw))
                    counts["pose_proposals"] += 1
                    if (not np.isfinite(position).all() or not pf.is_navigable(position)
                            or np.linalg.norm((position-raw)[[0, 2]]) > .20
                            or not .20 <= np.linalg.norm((position-origin)[[0, 2]]) <= .80):
                        continue
                    yaw = (pose["yaw"] + math.radians(dyaw) + math.pi) % (2*math.pi) - math.pi
                    identity = (tuple(position.tolist()), yaw)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    geometries = geometry(position)
                    if geometries is not None:
                        inspect(position, yaw, geometries, dict(address=address, attempt=attempt))
                    if counts["visual_candidates"] >= REVISIT_VISUAL_BUDGET or len(retained) == len(BINS):
                        break
                if counts["visual_candidates"] >= REVISIT_VISUAL_BUDGET or len(retained) == len(BINS):
                    break
        result = dict(schema="table2_common_goals_20260911_v1", stage=stage, role=role,
            prefixes={a: str(Path(p).resolve()) for a, p in prefixes.items()},
            candidates=candidates, counts=dict(counts), checked=checked,
            scope="common next-task construction; no policy scores or navigation outcomes",
            role_runtime_visible=False, live_arms=sorted(prefixes), distance_address="mean across live arms")
        dump(out / "construction.json", result)
        return result
    finally:
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    req = load(args.request)
    result = construct_common(req["source"], stage=req["stage"], prefixes=req["prefixes"], role=req["role"], out=args.out)
    print({"stage": result["stage"], "role": result["role"], "common_distance_bands": [c["distance_band"] for c in result["candidates"]]}, flush=True)
