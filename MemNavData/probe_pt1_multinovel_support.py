"""Render a preselected PT1 capacity probe; no navigation or model inference.

This uses expert A/AB prefixes ONLY to test offline scene constructibility.
An accepted candidate is visually unsupported by that expert prefix; it is not
yet a valid query for a different, actual NavDP history. Formal evaluation
must reconstruct its queries after actual mono A/B collection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from MemNavData.audit_pt1_multinovel_capacity import digest, save, distribution

SCHEMA = "pt1_multinovel_visual_capacity_probe_20260911_v1"


def select_candidates(candidates, per_cell=2):
    """First proposals per distance/direction cell, BEFORE visual scoring."""
    used, selected = Counter(), []
    for row in sorted(candidates, key=lambda r: r["proposal_index"]):
        key = (row["distance_bin"], row["direction"])
        if used[key] < per_cell:
            selected.append(row)
            used[key] += 1
    return selected


def is_unsupported(surface_points, curve):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 1 or not np.isfinite(curve).all() or np.any((curve < 0) | (curve > 1)):
        raise ValueError("Invalid complete co-visibility curve")
    return surface_points > 0 and (not len(curve) or float(curve.max()) < .10)


def prepare(inventory_path, spatial_path, sources, out, *, all_sources=False):
    inventory = json.loads(Path(inventory_path).read_text())
    spatial = json.loads(Path(spatial_path).read_text())
    history_by_id = {r["source_id"]: r for r in inventory["histories"]}
    capacity_by_id = {r["source_id"]: r for s in spatial["scene_results"] for r in s.get("source_results", [])}
    if all_sources:
        sources = sorted(history_by_id)
    rows = []
    for source in sources:
        history, capacity = history_by_id[source], capacity_by_id[source]
        scene = next(s for s in spatial["scene_results"] if s["scene"] == history["scene"])
        if any(s["status"] != "measured" for s in capacity["states"]):
            raise ValueError("All three geometric states must be measured")
        rows.append(dict(source_id=source, history=history,
            navmesh=scene["navmesh"], navmesh_sha256=scene["navmesh_sha256"],
            asset_sha256=scene["asset_sha256"],
            stages=[dict(state=r["state"], candidates=select_candidates(r["candidates"]))
                    for r in capacity["states"]]))
    save(out, dict(schema=SCHEMA, expert_history=True, actual_online_history=False,
        source_selection=("complete local inventory in source-id order; earlier two-source probe is included, not fresh confirmation"
                          if all_sources else "explicit geometric shortlist; no visual or policy scores used"),
        candidate_selection="first two proposals per nonempty distance/direction cell; no resampling after scoring",
        inventory_sha256=digest(inventory_path), spatial_sha256=digest(spatial_path),
        policy_outcomes_read=False, navigation_rollouts=0, formal_population_created=False,
        sources=rows))


def run(plan_path, out):
    import pandas as pd
    from PIL import Image
    from MemNavData.generate_twoleg import (
        K, make_sim, render, backproject, to_world, cam_to_world_hab, covis_curve, covis_frac,
    )
    from MemNavData.habitat_rollout_primitives import parquet_data_pose_to_habitat
    from MemNavData.final14_role_pair_contract import DEPTH_TOLERANCE_M

    plan = json.loads(Path(plan_path).read_text())
    if plan["schema"] != SCHEMA:
        raise ValueError("Wrong visual probe plan")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    all_results = []
    started = time.monotonic()
    for source in plan["sources"]:
        h = source["history"]
        for file, expected in ((h["parquet"], h["parquet_sha256"]),
                               (h["metadata"], h["metadata_sha256"]),
                               (h["asset"], source["asset_sha256"]),
                               (source["navmesh"], source["navmesh_sha256"])):
            if digest(file) != expected:
                raise ValueError(f"Source changed: {file}")
        meta = json.loads(Path(h["metadata"]).read_text())
        frame = pd.read_parquet(h["parquet"])
        if not np.allclose(np.asarray(frame.iloc[0]["observation.camera_intrinsic"].tolist()), K, atol=1e-3):
            raise ValueError("Probe must use the original camera intrinsic")
        height = h["camera_height_m"]
        count = max(s["state"]["history_frames"] for s in source["stages"])
        transforms, depths = [], []
        sim = make_sim(h["asset"], source["navmesh"], recompute_navmesh=False)
        try:
            for _, row in frame.iloc[:count].iterrows():
                pose = parquet_data_pose_to_habitat(
                    np.asarray(row["action"].tolist(), dtype=float).reshape(4, 4),
                    np.asarray(row["observation.camera_extrinsic"].tolist(), dtype=float).reshape(4, 4),
                    camera_height_m=height, frame_convention=meta["frame_convention"])
                position = pose.position + [0., height, 0.]
                _, depth = render(sim, position, pose.yaw_rad)
                transforms.append(cam_to_world_hab(position, pose.yaw_rad))
                depths.append(depth)
            for stage in source["stages"]:
                state = stage["state"]
                n = state["history_frames"]
                current_position = np.asarray(state["position"]) + [0., height, 0.]
                current_transform = cam_to_world_hab(current_position, state["yaw"])
                _, current_depth = render(sim, current_position, state["yaw"])
                records = []
                for candidate in stage["candidates"]:
                    position = np.asarray(candidate["floor_position"]) + [0., height, 0.]
                    rgb, depth = render(sim, position, candidate["yaw_rad"])
                    transform = cam_to_world_hab(position, candidate["yaw_rad"])
                    points = to_world(backproject(depth, stride=6), transform)
                    curve = covis_curve(points, transforms[:n], depths[:n], tol=DEPTH_TOLERANCE_M)
                    image_path = out / h["source_id"] / state["stage"] / f"{candidate['proposal_index']:04d}.jpg"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(image_path, quality=95)
                    records.append(dict(candidate, goal_rgb=str(image_path.resolve()),
                        surface_points=len(points), max_history_covis=float(curve.max()) if n else 0.,
                        covis_curve=curve.tolist(), history_frames_checked=len(curve),
                        current_view_covis=covis_frac(points, current_transform, current_depth, tol=DEPTH_TOLERANCE_M),
                        self_covis=covis_frac(points, transform, depth, tol=DEPTH_TOLERANCE_M),
                        unsupported_by_this_expert_prefix=is_unsupported(len(points), curve)))
                accepted = [r for r in records if r["unsupported_by_this_expert_prefix"]]
                result = dict(source_id=h["source_id"], stage=state["stage"],
                    history_frames=n, candidates=len(records), unsupported=len(accepted),
                    unsupported_by_direction=dict(Counter(r["direction"] for r in accepted)),
                    unsupported_by_distance_bin=dict(Counter(r["distance_bin"] for r in accepted)),
                    current_view_covis=distribution([r["current_view_covis"] for r in records]),
                    records=records)
                all_results.append(result)
                save(out / h["source_id"] / (state["stage"] + ".json"), result)
                print(h["source_id"], state["stage"], f"unsupported {len(accepted)}/{len(records)}",
                      result["unsupported_by_direction"], flush=True)
        finally:
            sim.close()
    save(out / "summary.json", dict(schema=SCHEMA, plan_sha256=digest(plan_path),
        code_sha256=digest(__file__), constructor_sha256=digest(Path(__file__).with_name("generate_twoleg.py")),
        expert_history=True, actual_online_history=False, visual_support_measured=True,
        policy_outcomes_read=False, navigation_rollouts=0, formal_population_created=False,
        depth_authority="simulator depth for OFFLINE co-visibility measurement only; no policy invoked",
        complete_prefix_scored=True, depth_tolerance_m=DEPTH_TOLERANCE_M,
        wall_seconds=time.monotonic()-started, results=all_results))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--inventory", type=Path, required=True)
    planning.add_argument("--spatial", type=Path, required=True)
    source_args = planning.add_mutually_exclusive_group(required=True)
    source_args.add_argument("--source", action="append")
    source_args.add_argument("--all-sources", action="store_true")
    planning.add_argument("--out", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "plan":
        prepare(args.inventory, args.spatial, args.source, args.out, all_sources=args.all_sources)
    else:
        run(args.plan, args.out)
