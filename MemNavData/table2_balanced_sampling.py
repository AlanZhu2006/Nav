"""Prospective Table-II query menus; no policy scores or results are inputs.

One menu keeps the first valid view in each distance/direction cell. A later
population-level assignment minimizes squared cell counts without dropping
constructible sources. Scarce cells are not silently relabelled. All role and
GT-support measurements belong to construction, not the navigation server.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from MemNavData.audit_pt1_multinovel_capacity import distance_bin, shortest
from MemNavData.final14_role_pair_contract import (
    STRATA, DEPTH_TOLERANCE_M, direction_in_stratum,
    relative_direction_degrees, stable_u32,
)
from MemNavData.table2_mixed_local import (
    dump, load, sha, runtime_spec, history_at, revisit_source_frames, support_sources,
)
from MemNavData.table2_novel_sampling import goal_yaw
from MemNavData.table2_sampling_profiles import (
    FORWARD_SCHEMA, role_cells, verify_query_direction,
)

SCHEMA = "table2_balanced_actual_mono_20260911_v2"
BINS = ("2_to_4", "4_to_6", "6_to_9")
CELLS = tuple(f"{b}/{s}" for b in BINS for s in STRATA)
SPATIAL_ATTEMPTS = 10000
NOVEL_VIEWS_PER_CELL = 12
REVISIT_VISUAL_BUDGET = 384


def cell_of(candidate):
    band = distance_bin(float(candidate["geodesic_m"]))
    direction = candidate["direction_stratum"]
    if band is None or direction not in STRATA:
        raise ValueError("Query is outside the declared 2--9 m / direction cells")
    return f"{band}/{direction}"


def novel_supported_as_task(surface_points, curve, current_covis):
    values = np.asarray(curve, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or not math.isfinite(current_covis) or not 0 <= current_covis <= 1):
        raise ValueError("Invalid full-history support measurement")
    return (surface_points > 0 and current_covis < .10
            and (len(values) == 0 or float(values.max()) < .10))


def balanced_assignment(menus, *, key, cells=CELLS):
    """One candidate per nonempty source, globally minimal sum(cell_count**2).

    Unit-capacity min-cost flow handles scarce cells by reassignment rather
    than a source-order greedy quota. The only inputs are source IDs and
    precomputed cell-address menus; costs never contain navigation outcomes.
    """
    cells = tuple(cells)
    if not cells or len(cells) != len(set(cells)) or not set(cells).issubset(CELLS):
        raise ValueError("Assignment cells must be a nonempty subset of declared cells")
    identities = [m["source_id"] for m in menus]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate source in a population assignment")
    active = [m for m in menus if m["candidates"]]
    n = len(active)
    graph = [[] for _ in range(n + len(cells) + 2)]
    start, sink = 0, len(graph) - 1

    def edge(a, b, capacity, cost):
        forward = [b, len(graph[b]), capacity, cost]
        reverse = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(reverse)
        return forward

    edges = {}
    tie_cells = sorted(cells, key=lambda c: (stable_u32(SCHEMA, key, c), c))
    for i, menu in enumerate(active):
        edge(start, i + 1, 1, 0)
        candidates = {c["cell"]: c for c in menu["candidates"]}
        if len(candidates) != len(menu["candidates"]) or not set(candidates).issubset(cells):
            raise ValueError("Expected at most one candidate per declared cell")
        for cell in tie_cells:
            if cell in candidates:
                edges[i, cell] = edge(i + 1, n + 1 + cells.index(cell), 1, 0)
    for j, cell in enumerate(cells):
        # Parallel unit edges implement the increments 1,3,5,... of k^2.
        for k in range(n):
            edge(n + 1 + j, sink, 1, 2 * k + 1)
    for _ in range(n):
        dist, parent, queued = [math.inf] * len(graph), [None] * len(graph), {start}
        dist[start] = 0
        queue = deque([start])
        while queue:
            at = queue.popleft()
            queued.remove(at)
            for j, (to, _, capacity, cost) in enumerate(graph[at]):
                if capacity and dist[to] > dist[at] + cost:
                    dist[to], parent[to] = dist[at] + cost, (at, j)
                    if to not in queued:
                        queue.append(to)
                        queued.add(to)
        if parent[sink] is None:
            raise RuntimeError("A nonempty source was not assigned")
        at = sink
        while at != start:
            prev, j = parent[at]
            e = graph[prev][j]
            e[2] -= 1
            graph[at][e[1]][2] += 1
            at = prev
    selected = []
    for i, menu in enumerate(active):
        chosen = [c for c in menu["candidates"] if edges[i, c["cell"]][2] == 0]
        if len(chosen) != 1:
            raise RuntimeError("Assignment is not one-to-one")
        selected.append(dict(source_id=menu["source_id"], **chosen[0]))
    counts = Counter(r["cell"] for r in selected)
    return dict(selected=selected, counts={c: counts[c] for c in cells},
                empty_sources=[m["source_id"] for m in menus if not m["candidates"]],
                objective="minimize sum of squared distance-by-direction cell counts",
                exact_equal_cells=len(set(counts[c] for c in cells)) <= 1,
                navigation_outcomes_read=False)


def spatial_menu(pf, start, yaw, key, *, cells=CELLS):
    """Fixed geometric budget, same floor along the entire route, unique cells."""
    from MemNavData.generate_twoleg import first_path_yaw
    cells = tuple(cells)
    if not cells or not set(cells).issubset(CELLS):
        raise ValueError("Spatial menu has invalid declared cells")
    pf.seed(stable_u32(SCHEMA, "spatial", key))
    seen, counts, retained = set(), Counter(), Counter()
    candidates = []
    for proposal in range(SPATIAL_ATTEMPTS):
        counts["attempts"] += 1
        pos = np.asarray(pf.get_random_navigable_point(), dtype=float)
        if not np.isfinite(pos).all() or not pf.is_navigable(pos):
            continue
        grid = tuple(np.floor(pos[[0, 2]] / .25).astype(int))
        if grid in seen or abs(pos[1] - start[1]) > .20:
            continue
        seen.add(grid)
        if pf.distance_to_closest_obstacle(pos) < .30:
            continue
        path = shortest(pf, start, pos)
        if path is None:
            continue
        distance, points = path
        if distance_bin(distance) is None or np.max(np.abs(points[:, 1] - start[1])) > .20:
            continue
        bearing = first_path_yaw(points, start)
        relative = relative_direction_degrees(bearing, yaw)
        direction = next(s for s in STRATA if direction_in_stratum(relative, s))
        cell = f"{distance_bin(distance)}/{direction}"
        counts["spatial_" + cell] += 1
        if cell not in cells:
            counts["outside_role_direction"] += 1
            continue
        if retained[cell] >= NOVEL_VIEWS_PER_CELL:
            continue
        straight = float(np.linalg.norm((pos - start)[[0, 2]]))
        candidates.append(dict(proposal_index=proposal, floor_position=pos.tolist(),
            yaw_rad=goal_yaw(key, proposal), direction_stratum=direction,
            geodesic_m=distance, straight_distance_m=straight, route_ratio=distance / straight,
            initial_relative_route_angle_deg=relative, cell=cell))
        retained[cell] += 1
        if all(retained[c] == NOVEL_VIEWS_PER_CELL for c in cells):
            break
    return dict(candidates=candidates, counts=dict(counts),
                retained={c: retained[c] for c in cells})


def construct(source, *, stage, prefix, out):
    from MemNavData.generate_twoleg import make_sim, render, covis_curve, covis_frac, cam_to_world_hab
    from MemNavData.build_shared_online_double_revisit import (
        goal_world_points, jpeg_bytes, write_depth_png, pixel_mae,
    )
    from MemNavData.build_final14_role_pair_scene import deterministic_pose_grid
    from MemNavData.build_shared_online_role_pairs import query_geometry

    spec = deepcopy(source)
    history = history_at(prefix) if prefix else None
    if history:
        trace = history["trace"]
        if not trace["reached"]:
            raise ValueError("Shared histories must come from successful native prefixes")
        spec.update(start_position=trace["end_position"], start_yaw=trace["end_yaw"],
            prefix_root=str(Path(prefix).resolve()),
            prefix_trace_sha256=sha(Path(prefix) / "online_a_trace.json"))
    schema = FORWARD_SCHEMA if spec.get("schema") == FORWARD_SCHEMA else SCHEMA
    spec.update(schema=schema, stage_number={"A": 0, "B": 1, "C": 2}[stage])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sim = make_sim(spec["asset"], "", agent_radius=.30)
    pf, height = sim.pathfinder, spec["camera_height_m"]
    start = np.asarray(spec["start_position"])
    key = f"{spec['scene']}/{spec['episode']}/{stage}/{Path(prefix).name if prefix else 'empty'}"
    menus = {r: [] for r in ("novel", "revisit")}
    diagnostics = {}
    try:
        pf.save_nav_mesh(str(out / "construction.navmesh"))
        _, current_depth = render(sim, start + [0, height, 0], spec["start_yaw"])
        current_t = cam_to_world_hab(start + [0, height, 0], spec["start_yaw"])

        def measure(position, yaw):
            camera = np.asarray(position) + [0, height, 0]
            rgb, depth = render(sim, camera, yaw)
            points = goal_world_points(depth, camera, yaw)
            curve = covis_curve(points, history["transforms"], history["depths"],
                tol=DEPTH_TOLERANCE_M) if history else np.empty(0)
            visible = float(covis_frac(points, current_t, current_depth, tol=DEPTH_TOLERANCE_M))
            return rgb, depth, curve, visible, len(points)

        def save(role, candidate, observed):
            rgb, depth, curve, visible, surface_points = observed
            folder = out / role / candidate["cell"].replace("/", "_")
            folder.mkdir(parents=True)
            (folder / "goal.jpg").write_bytes(jpeg_bytes(rgb))
            write_depth_png(folder / "goal_depth_evaluator_only.png", depth)
            n = len(curve)
            a_steps = history["trace"].get("prefix_A_steps", n) if history else 0
            row = dict(spec, **candidate, analysis_role=role,
                goal_rgb=str((folder / "goal.jpg").resolve()), goal_rgb_sha256=sha(folder / "goal.jpg"),
                covis_curve=curve.tolist(), max_history_covis=float(curve.max()) if n else 0.,
                max_runtime_eligible_covis=float(curve[8:].max()) if n > 8 else 0.,
                runtime_eligible_frame_floor=8, construction_source_frame_floor=39,
                history_frames=n, current_view_covis=visible, goal_surface_points=surface_points,
                **support_sources(curve, a_steps))
            verify_query_direction(row, schema)
            dump(folder / "query.json", row)
            dump(folder / "runtime.json", runtime_spec(row))
            menus[role].append(dict(cell=candidate["cell"], query=str((folder / "query.json").resolve()),
                query_sha256=sha(folder / "query.json"),
                geodesic_m=candidate["geodesic_m"], current_view_covis=visible,
                direction_stratum=candidate["direction_stratum"]))

        pool = spatial_menu(pf, start, spec["start_yaw"], key,
                            cells=role_cells(schema, "novel"))
        chosen, checked = set(), []
        for candidate in pool["candidates"]:
            if candidate["cell"] in chosen:
                continue
            observed = measure(candidate["floor_position"], candidate["yaw_rad"])
            accepted = novel_supported_as_task(observed[4], observed[2], observed[3])
            checked.append(dict(proposal_index=candidate["proposal_index"], cell=candidate["cell"],
                surface_points=observed[4], current_view_covis=observed[3],
                max_history_covis=float(observed[2].max()) if len(observed[2]) else 0., accepted=accepted))
            if accepted:
                save("novel", candidate, observed)
                chosen.add(candidate["cell"])
        diagnostics["novel"] = dict(spatial=pool, checked=checked)
        if history:
            counts, chosen, checked = Counter(), set(), []
            frames = revisit_source_frames(len(history["poses"]))
            # Interleave spatial perturbations over the WHOLE A/B history;
            # exhaust neither the first anchor nor just the most recent leg.
            grids = [list(deterministic_pose_grid(key + f"/{frame}")) for frame in frames]
            exhausted = False
            for attempt in range(max(map(len, grids), default=0)):
                for frame, grid in zip(frames, grids):
                    if attempt >= len(grid):
                        continue
                    radius, angle, dyaw = grid[attempt]
                    if radius > .8 or not 12 <= abs(dyaw) <= 45:
                        continue
                    origin = history["floor_positions"][frame]
                    raw = origin + radius * np.asarray([math.cos(angle), 0, math.sin(angle)])
                    pos = np.asarray(pf.snap_point(raw))
                    counts["pose_proposals"] += 1
                    if (not np.isfinite(pos).all() or not pf.is_navigable(pos)
                        or abs(pos[1] - start[1]) > .20
                        or np.linalg.norm((pos - raw)[[0, 2]]) > .20
                        or not .20 <= np.linalg.norm((pos - origin)[[0, 2]]) <= .80):
                        continue
                    path = shortest(pf, start, pos)
                    geometry = query_geometry(pf, start, pos)
                    if (path is None or geometry is None or distance_bin(geometry[0]) is None
                            or np.max(np.abs(path[1][:, 1] - start[1])) > .20):
                        continue
                    relative = relative_direction_degrees(geometry[1], spec["start_yaw"])
                    direction = next(s for s in STRATA if direction_in_stratum(relative, s))
                    cell = f"{distance_bin(geometry[0])}/{direction}"
                    if cell in chosen:
                        continue
                    yaw = (history["poses"][frame]["yaw"] + math.radians(dyaw) + math.pi) % (2*math.pi) - math.pi
                    observed = measure(pos, yaw)
                    counts["visual_checks"] += 1
                    curve = observed[2]
                    mae = pixel_mae(observed[0], history["rgbs"][frame])
                    maximum = float(curve.max()) if len(curve) else 0.
                    eligible = float(curve[8:].max()) if len(curve) > 8 else 0.
                    accepted = (observed[4] > 0 and observed[3] < .10 and mae >= 5.
                                and .55 <= maximum <= .90 and eligible >= .55)
                    checked.append(dict(frame=frame, attempt=attempt, cell=cell, accepted=accepted,
                        max_history_covis=maximum, max_eligible_covis=eligible,
                        current_view_covis=observed[3], surface_points=observed[4]))
                    if accepted:
                        straight = float(np.linalg.norm((pos-start)[[0, 2]]))
                        save("revisit", dict(cell=cell, floor_position=pos.tolist(), yaw_rad=yaw,
                            geodesic_m=geometry[0], source_online_frame=frame, source_pose_attempt=attempt,
                            translation_from_source_m=float(np.linalg.norm((pos-origin)[[0, 2]])),
                            yaw_delta_from_source_deg=abs(dyaw), pixel_mae_from_source=mae,
                            initial_relative_route_angle_deg=relative, direction_stratum=direction,
                            straight_distance_m=straight, route_ratio=geometry[0]/straight), observed)
                        chosen.add(cell)
                    if counts["visual_checks"] >= REVISIT_VISUAL_BUDGET or len(chosen) == len(CELLS):
                        exhausted = True
                        break
                if exhausted:
                    break
            diagnostics["revisit"] = dict(counts=dict(counts), source_frames=frames, checked=checked)
        result = dict(schema=schema, stage=stage, menus=menus, diagnostics=diagnostics,
            both_constructed=all(menus.values()), navigation_outcomes_read=False)
        dump(out / "construction.json", result)
        return result
    finally:
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("construct",))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    construct(load(args.source), stage=args.stage, prefix=args.prefix, out=args.out)
