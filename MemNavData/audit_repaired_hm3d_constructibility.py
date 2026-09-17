"""CPU-only audit of saved A trajectories and the frozen query constructor.

Never runs a policy or renders an image.  For the already-failed Novel search,
replay its spatial filters on the saved NavMesh; replace only its final visual
test with a rejecting sentinel so every original proposal is counted.  The
sentinel is NOT a measured co-visibility score or a new query population.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import build_final14_role_pair_scene as builder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_history(trace):
    return {"trace": trace, "poses": trace["poses"],
            "floor_positions": [np.array([p["x"], p["y"], p["z"]], float)
                                for p in trace["poses"]],
            "depths": [], "transforms": []}


def revisit_frame_audit(simulator, history):
    endpoint, _ = builder.online_endpoint(history)
    distances = []
    for frame, position in enumerate(history["floor_positions"]):
        try:
            distance = builder.history_tools.goal_distance(simulator.pathfinder, endpoint, position)
        except RuntimeError:
            distance = None
        distances.append({"frame": frame, "geodesic_m": distance})
    candidates = builder.source_frame_candidates(simulator, history)
    indices = list(range(builder.ELIGIBLE_FRAME_FLOOR,
                         len(history["poses"]) - builder.END_MARGIN_FRAMES,
                         builder.SOURCE_FRAME_STRIDE))
    inspected = [distances[i] for i in indices]
    available = [r["geodesic_m"] for r in inspected if r["geodesic_m"] is not None]
    return {"pose_count": len(distances), "inspected_frame_indices": indices,
            "inspected_distances": inspected, "source_candidates": candidates,
            "max_inspected_geodesic_m": max(available) if available else None,
            "all_frames_in_2_to_9m_band": [r["frame"] for r in distances
                if r["geodesic_m"] is not None and 2.0 <= r["geodesic_m"] <= 9.0]}


def restore_revisit_position(pathfinder, history, selected, scene, episode):
    frame = selected["source_frame"]
    grid = builder.deterministic_pose_grid(f"{scene}/{episode}/{frame}")
    radius, direction, _ = grid[selected["render_attempt"] - 1]
    raw = history["floor_positions"][frame] + np.array([
        radius * math.cos(direction), 0., radius * math.sin(direction)])
    position = np.asarray(pathfinder.snap_point(raw), dtype=float)
    distance = builder.history_tools.goal_distance(
        pathfinder, builder.online_endpoint(history)[0], position)
    if abs(distance - selected["query_geodesic_m"]) > 1e-5:
        raise ValueError("Saved execution NavMesh does not reproduce the selected query distance")
    return position


def geometry_replay(simulator, history, attempt, scene, episode):
    selected = attempt["selected"]["standard"]
    error = attempt["natural_error"]
    if selected is None or not error:
        return None
    saved = json.loads(error.split(": ", 1)[1])
    target = restore_revisit_position(simulator.pathfinder, history, selected, scene, episode)
    original_geometry = builder.pair_tools.query_geometry
    direction_counts = Counter()
    support_inputs = []

    def observed_geometry(pathfinder, first, second):
        result = original_geometry(pathfinder, first, second)
        if result is not None and 2.0 <= result[0] <= 9.0:
            degrees = builder.relative_direction_degrees(
                result[1], builder.online_endpoint(history)[1])
            for stratum in builder.STRATA:
                if builder.direction_in_stratum(degrees, stratum):
                    direction_counts[stratum] += 1
        return result

    def no_render(_simulator, position, yaw):
        support_inputs.append({"camera_position": np.asarray(position).tolist(), "yaw": float(yaw)})
        return None, None

    # Geometry only: intentionally reject the visual stage without evaluating it.
    # This preserves the original 5000 proposals of a failed search, including RNG.
    with patch.object(builder.pair_tools, "query_geometry", observed_geometry), \
         patch.object(builder, "render", no_render), \
         patch.object(builder.history_tools, "goal_world_points", return_value=None), \
         patch.object(builder, "covis_curve", return_value=np.array([1.0])):
        try:
            builder.sample_natural_novel(
                simulator, history, scene=scene, episode=episode,
                scene_rank=attempt["scene_rank"], episode_rank=attempt["source_episode_rank"],
                paired_revisit_position=target, camera_height=.5)
        except builder.NaturalNovelConstructionError as rejected:
            replayed = rejected.diagnostics
        else:
            raise AssertionError("The geometry-only sentinel must not construct a Novel query")
    if replayed != saved:
        raise ValueError(f"Geometry replay differs from saved construction: {replayed} != {saved}")
    return {"original_geometry_counts_reproduced": True,
            "saved_rejection_counts": saved,
            "spatially_valid_direction_counts_before_stratum_filter": dict(direction_counts),
            "candidates_reaching_visual_support": len(support_inputs),
            "visual_support_recomputed": False,
            "selected_revisit_position": target.tolist()}


def audit(root):
    import habitat_sim
    report_path = root / "construction_summary.json"
    reports = json.loads(report_path.read_text())["reports"]
    results = []
    for report in reports:
        scene = report["scene"]
        construction = report["construction"]
        for attempt in construction["attempts"]:
            episode = attempt["episode"]
            folder = root / "goal_a" / scene / episode
            trace_path = folder / f"{episode}_leg1_trace.json"
            mesh = folder / "execution.navmesh"
            trace = json.loads(trace_path.read_text())
            history = trace_history(trace)
            pathfinder = habitat_sim.PathFinder()
            if not pathfinder.load_nav_mesh(str(mesh)):
                raise ValueError(f"Cannot load recorded NavMesh: {mesh}")
            simulator = SimpleNamespace(pathfinder=pathfinder)
            frames = revisit_frame_audit(simulator, history)
            if frames["source_candidates"] != attempt["revisit_diagnostics"]["source_frames_considered"]:
                raise ValueError("Saved source-frame candidates were not reproduced")
            results.append({"scene": scene, "episode": episode,
                "source_candidates_reproduced": True,
                "trace_sha256": digest(trace_path), "navmesh_sha256": digest(mesh),
                "navmesh_area_m2": float(pathfinder.navigable_area),
                "frame_audit": frames,
                "novel_geometry_replay": geometry_replay(simulator, history, attempt, scene, episode)})
    return {"scope": "posthoc construction audit only; no model, render, visual score, or new SR",
            "construction_summary_sha256": digest(report_path), "records": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
