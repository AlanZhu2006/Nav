"""Outcome-blind source inventory and CPU-only query-construction diagnostics.

Reads task definitions or actual A traces, never query-policy outcomes. The
spatial probe deliberately does NOT measure visual support: it enumerates
geometrically eligible proposals in each direction with a rejecting sentinel.
Its output is a design diagnostic, not a new benchmark or navigation result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

PARENT_SHA = "a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory(parent):
    rows = []
    for scene_rank, scene in enumerate(parent["scenes"]):
        for episode_rank, source in enumerate(parent["episodes"][scene]):
            rows.append({"scene_rank": scene_rank, "scene": scene,
                         "episode_rank": episode_rank, "episode": source["episode"],
                         "task_goal_a_geodesic_m": float(source["goal_a_geodesic_m"])})
    if not rows or len(rows) != parent["episode_count"]:
        raise ValueError("Source count differs from parent")
    bins = []
    for lower, upper in ((3., 4.), (4., 5.), (5., 6.), (6., 9.)):
        group = [r for r in rows if lower <= r["task_goal_a_geodesic_m"] < upper]
        bins.append({"range_m": [lower, upper], "episodes": len(group),
                     "scenes": len({r["scene"] for r in group})})
    if sum(b["episodes"] for b in bins) != len(rows):
        raise ValueError("A task distance is outside the declared inventory bins")
    first = [r for r in rows if r["episode_rank"] == 0]
    return {"scope": "task definitions, not executed A length or query distance",
            "query_outcomes_read": False, "selection_performed": False,
            "episodes": len(rows), "scenes": len({r["scene"] for r in rows}),
            "goal_a_geodesic_median_m": statistics.median(r["task_goal_a_geodesic_m"] for r in rows),
            "distance_inventory": bins,
            "episode_zero_only_count": len(first),
            "episode_zero_only_median_m": statistics.median(r["task_goal_a_geodesic_m"] for r in first),
            "sources": rows}


def spatial_probe(trace_path, navmesh_path, selected, *, scene, episode, scene_rank, episode_rank):
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import habitat_sim
    import audit_repaired_hm3d_constructibility as previous
    b = previous.builder
    trace = json.loads(Path(trace_path).read_text())
    if not trace["reached"]:
        raise ValueError("Probe requires a successful factual A")
    history = previous.trace_history(trace)
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path)):
        raise ValueError("Cannot load the recorded execution NavMesh")
    simulator = SimpleNamespace(pathfinder=pathfinder)
    frames = previous.revisit_frame_audit(simulator, history)
    target = previous.restore_revisit_position(pathfinder, history, selected, scene, episode)
    probes = []
    for stratum in b.STRATA:
        candidates = []

        def no_render(_sim, camera_position, yaw):
            candidates.append({"camera_position": np.asarray(camera_position).tolist(), "yaw": float(yaw)})
            return None, None

        with patch.object(b, "render", no_render), \
             patch.object(b.history_tools, "goal_world_points", return_value=None), \
             patch.object(b, "covis_curve", return_value=np.array([1.])):
            try:
                b.sample_natural_novel(simulator, history, scene=scene, episode=episode,
                    scene_rank=scene_rank, episode_rank=episode_rank, paired_revisit_position=target,
                    camera_height=.5, direction_stratum=stratum,
                    sampling_seed_namespace="repaired_fullmono_spatial_design_20260909")
            except b.NaturalNovelConstructionError as error:
                counts = error.diagnostics
            else:
                raise AssertionError("Unmeasured visual support must not authorize a query")
        if counts["support_rejects"] != len(candidates):
            raise ValueError("Spatial count does not match sentinel count")
        probes.append({"direction_stratum": stratum, "counts": counts,
                       "spatial_candidate_count": len(candidates), "candidates": candidates})
    return {"scope": "posthoc geometry-only design diagnostic; not visual support or SR",
            "query_outcomes_read": False, "visual_support_measured": False,
            "navigation_rollouts": 0, "scene": scene, "episode": episode,
            "trace_sha256": sha(trace_path), "navmesh_sha256": sha(navmesh_path),
            "frame_audit": frames, "strata": probes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("sources")
    inventory.add_argument("parent", type=Path)
    spatial = commands.add_parser("spatial")
    spatial.add_argument("integration_root", type=Path)
    spatial.add_argument("scene")
    spatial.add_argument("--episode", default="episode_0000")
    for cmd in (inventory, spatial):
        cmd.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "sources":
        if sha(args.parent) != PARENT_SHA:
            raise ValueError("Unexpected source parent")
        result = source_inventory(json.loads(args.parent.read_text()))
        result["parent_sha256"] = PARENT_SHA
    else:
        construction = json.loads((args.integration_root / "construction_summary.json").read_text())
        report = next(r for r in construction["reports"] if r["scene"] == args.scene)
        attempt = next(a for a in report["construction"]["attempts"] if a["episode"] == args.episode)
        root = args.integration_root / "goal_a" / args.scene / args.episode
        result = spatial_probe(root / f"{args.episode}_leg1_trace.json", root / "execution.navmesh",
            attempt["selected"]["standard"], scene=args.scene, episode=args.episode,
            scene_rank=attempt["scene_rank"], episode_rank=attempt["source_episode_rank"])
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    compact = {k: v for k, v in result.items() if k not in ("sources", "strata", "frame_audit")}
    if "strata" in result:
        compact["spatial_counts"] = {s["direction_stratum"]: s["spatial_candidate_count"] for s in result["strata"]}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
