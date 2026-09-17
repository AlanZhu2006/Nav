"""One outcome-blind Novel view recipe for Table-II A/B/C construction.

This is an offline constructor, not a policy or a role classifier. The local
CLI checks spatial supply on recorded starts without rendering or inference.
Spatial candidates are NOT certified Novel tasks until the full causal
history has been checked by select_novel_view(). Old benchmark assets and
running model services are never changed.
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
    wrap_radians,
)

SCHEMA = "table2_common_novel_sampling_local_20260911_v1"


def goal_yaw(query_key, proposal_index):
    """The same eight-world-direction recipe at every stage; no route yaw."""
    index = stable_u32(SCHEMA, "goal_view", query_key, int(proposal_index)) % 8
    return wrap_radians(index * math.pi / 4.0)


def initial_a_yaw(source_key):
    """An independent initial yaw, not the first segment of a shortest path."""
    return -math.pi + 2.0 * math.pi * (
        stable_u32(SCHEMA, "initial_a_yaw", source_key) / 2**32)


def geometry_pool(pathfinder, position, yaw, query_key, *, per_direction=16,
                  max_attempts=5000):
    """Enumerate common 2--9 m, same-floor candidates in all three directions.

    NavMesh is used only by this offline task constructor. Goal bearings and
    positions in the returned metadata must not enter a policy request.
    Missing directions stay missing instead of silently changing the quota.
    """
    from MemNavData.build_shared_online_role_pairs import query_geometry

    if per_direction < 1 or max_attempts < 1:
        raise ValueError("Positive candidate and proposal budgets required")
    start = np.asarray(position, dtype=float)
    if start.shape != (3,) or not np.isfinite(start).all() or not math.isfinite(yaw):
        raise ValueError("A finite recorded start is required")
    pathfinder.seed(stable_u32(SCHEMA, "positions", query_key))
    counts, retained, seen = Counter(), Counter(), set()
    candidates = []
    for proposal_index in range(max_attempts):
        counts["attempts"] += 1
        target = np.asarray(pathfinder.get_random_navigable_point(), dtype=float)
        if not np.isfinite(target).all() or not pathfinder.is_navigable(target):
            counts["non_navigable"] += 1
            continue
        identity = tuple(round(float(v), 4) for v in target)
        if identity in seen:
            counts["duplicate"] += 1
            continue
        seen.add(identity)
        if abs(float(target[1] - start[1])) > .20:
            counts["other_floor"] += 1
            continue
        if float(pathfinder.distance_to_closest_obstacle(target)) < .30:
            counts["clearance"] += 1
            continue
        geometry = query_geometry(pathfinder, start, target)
        if geometry is None:
            counts["unreachable"] += 1
            continue
        distance, bearing = geometry
        if not 2.0 <= distance <= 9.0:
            counts["distance_band"] += 1
            continue
        relative = relative_direction_degrees(bearing, yaw)
        stratum = next(s for s in STRATA if direction_in_stratum(relative, s))
        counts["spatial_" + stratum] += 1
        if retained[stratum] == per_direction:
            continue
        straight = float(np.linalg.norm((target - start)[[0, 2]]))
        candidates.append(dict(
            proposal_index=proposal_index, floor_position=target.tolist(),
            yaw_rad=goal_yaw(query_key, proposal_index), direction_stratum=stratum,
            geodesic_m=float(distance), straight_distance_m=straight,
            route_ratio=None if straight == 0 else float(distance / straight),
            initial_relative_route_angle_deg=relative,
            visual_support_measured=False,
        ))
        retained[stratum] += 1
        if all(retained[s] == per_direction for s in STRATA):
            break
    return dict(candidates=candidates, counts=dict(counts),
                retained_by_direction={s: retained[s] for s in STRATA},
                per_direction_cap=per_direction, max_attempts=max_attempts,
                visual_support_measured=False)


def select_novel_view(candidates, stratum, render_and_measure, *, history_frames):
    """Select by full-history visual support, never by policy/CEC outcomes.

    render_and_measure(candidate) returns (goal_jpeg, full_history_covis).
    The caller supplies ALL pre-query frames, not just retrievable/top-K frames.
    Empty history is valid for initial A only; B/C callers must provide their
    recorded positive history length. There is no cross-direction fallback.
    """
    if stratum not in STRATA or history_frames < 0:
        raise ValueError("Invalid direction or history length")
    checked = []
    for candidate in candidates:
        if candidate["direction_stratum"] != stratum:
            continue
        jpeg, curve = render_and_measure(candidate)
        curve = np.asarray(curve, dtype=float)
        if (curve.shape != (history_frames,) or not np.isfinite(curve).all()
                or np.any((curve < 0) | (curve > 1))):
            raise ValueError("Co-visibility must cover every causal history frame")
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError("A rendered goal image is required")
        maximum = float(curve.max()) if history_frames else 0.0
        checked.append(dict(proposal_index=candidate["proposal_index"], max_covis=maximum))
        if maximum < .10:
            return dict(candidate, goal_jpeg=jpeg, max_covis=maximum,
                        support_history_frames=history_frames,
                        visual_support_measured=True), checked
    return None, checked


def balanced_c_sources(rows):
    """An outcome-blind equal prefix mix, before either C query is evaluated.

    Input order is the predeclared source order. B outcomes legitimately define
    the successful-prefix population; no C outcome is read. Unselected rows
    stay in the caller's source ledger.
    """
    if any(row["collector"] != "native" for row in rows):
        raise ValueError("The shared C comparison requires the declared native-B collector")
    groups = {role: [row for row in rows if row["b_role"] == role
                    and row["b_reached"] and row["both_c_constructed"]]
              for role in ("novel", "revisit")}
    size = min(map(len, groups.values()))
    return [row for pair in zip(groups["novel"][:size], groups["revisit"][:size])
            for row in pair]


def _load(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_geometry_audit(root, *, per_direction=16):
    """Use two already-consumed local scenes; NOT newly executed A histories."""
    import habitat_sim

    manifest = _load(root / "manifest.json")
    expected = ("gxdoqLR6rwA", "pLe4wQe7qrG")
    if tuple(source["scene"] for source in manifest["sources"]) != expected:
        raise ValueError("The predeclared local source order changed")
    starts, exclusions, input_files = [], [], [root / "manifest.json"]
    for source in manifest["sources"]:
        scene, episode = source["scene"], source["episode"]
        directory = root / "goal_a" / scene / episode
        trace_path = directory / f"{episode}_leg1_trace.json"
        trace = _load(trace_path)
        navmesh = directory / "execution.navmesh"
        input_files += [trace_path, navmesh]
        first = trace["poses"][0]
        source_key = f"{scene}/{episode}"
        starts.append(dict(scene=scene, episode=episode, stage="A", navmesh=str(navmesh),
            position=[first[axis] for axis in "xyz"], yaw=initial_a_yaw(source_key),
            inherited_actual_yaw=False, old_yaw=first["yaw"],
            origin="old physical start, NEW independent initial yaw; rollout not executed"))
        if not trace["reached"]:
            exclusions.append(dict(scene=scene, stage="B/C", reason="old_actual_A_failed"))
            continue
        starts.append(dict(scene=scene, episode=episode, stage="B_N", navmesh=str(navmesh),
            position=trace["end_position"], yaw=trace["end_yaw"], inherited_actual_yaw=True,
            origin="OLD actual mono-A endpoint; not a new balanced-A history"))
        for role in ("novel", "revisit"):
            terminal_path = root / "evaluation" / scene / role / "native/terminal_measurements.json"
            if not terminal_path.is_file():
                exclusions.append(dict(scene=scene, stage="C_after_" + role,
                                       reason="native_B_record_missing"))
                continue
            input_files.append(terminal_path)
            terminal, = _load(terminal_path)
            if not terminal["reached"]:
                exclusions.append(dict(scene=scene, stage="C_after_" + role,
                                       reason="native_B_failed"))
                continue
            starts.append(dict(scene=scene, episode=episode, stage="C_N_after_" + role,
                navmesh=str(navmesh), position=terminal["end_position"],
                yaw=terminal["end_yaw_rad"], inherited_actual_yaw=True,
                origin="OLD actual native-B endpoint; visual support not measured"))
    rows = []
    for start in starts:
        pf = habitat_sim.PathFinder()
        if not pf.load_nav_mesh(start["navmesh"]):
            raise ValueError("Saved execution NavMesh could not be loaded")
        key = f"{start['scene']}/{start['episode']}/{start['stage']}"
        pool = geometry_pool(pf, start["position"], start["yaw"], key,
                             per_direction=per_direction)
        rows.append(dict(start=start, query_key=key, **pool))
        print(key, pool["retained_by_direction"], flush=True)
    return dict(schema=SCHEMA, scope="CPU spatial-supply check on OLD actual states; no SR",
        input_sha256={str(path): _sha(path) for path in input_files},
        constructor_sha256=_sha(__file__),
        all_source_A_starts_retained=True, new_A_rollouts=0, query_rollouts=0,
        goal_images_rendered=0, visual_support_measured=False,
        formal_population_created=False, rows=rows, exclusions=exclusions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-local-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = local_geometry_audit(args.old_local_root.resolve())
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
