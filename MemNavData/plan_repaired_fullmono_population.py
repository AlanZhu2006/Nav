"""Prepare task identities and result-blind prefix selection for new mono-A.

Preparation is not submission or query-population sealing. No task is selected
by old A success, query SR, controller confidence, or trajectory length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_repaired_fullmono_design import PARENT_SHA, sha

PREFIXES = (30, 36, 42, 48, 54)
MIN_HISTORIES, MIN_SCENES = 24, 15


def prepare(parent):
    sources = []
    for scene_rank, scene in enumerate(parent["scenes"]):
        for episode_rank, source in enumerate(parent["episodes"][scene]):
            sources.append({"source_index": len(sources), "scene": scene,
                "episode": source["episode"], "scene_rank": scene_rank,
                "episode_rank": episode_rank, "seed": 2026082200 + 100 * scene_rank + episode_rank,
                "asset": parent["assets"][scene], "task_files": source["files"],
                "source_episode": str(Path(parent["paths"]["generated_root"]) / scene / source["episode"]),
                "task_goal_a_geodesic_m": source["goal_a_geodesic_m"]})
    if len(sources) != parent["episode_count"] or len(parent["scenes"]) != 54:
        raise ValueError("Parent population shape changed")
    if len({(r["scene"], r["episode"]) for r in sources}) != len(sources):
        raise ValueError("Duplicate source")
    stages = []
    previous = 0
    for prefix in PREFIXES:
        included = [r for r in sources if r["scene_rank"] < prefix]
        stages.append({"scene_prefix": prefix, "source_count": len(included),
                       "nonempty_scenes": len({r["scene"] for r in included}),
                       "new_source_indices": [r["source_index"] for r in included if r["scene_rank"] >= previous]})
        previous = prefix
    return {"schema": "repaired_fullmono_source_plan_20260909_v1",
        "status": "prepared_not_submitted_not_query_sealed", "parent_sha256": PARENT_SHA,
        "scope": "new actual mono-A on consumed HM3D source tasks; not independent scene confirmation",
        "source_selection": "all parent tasks in original order; no outcome or distance filter",
        "sources": sources, "stages": stages, "target_histories": MIN_HISTORIES,
        "target_scene_clusters": MIN_SCENES,
        "prefix_rule": "smallest complete predefined prefix reaching both targets; otherwise all 54",
        "base_runtime_sha256": "c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08",
        "model_and_execution_changes_allowed": False, "query_outcomes_read": False,
        "new_a_required": True, "old_a_results_for_selection": False,
        "construction_candidate_protocol": "REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_PROTOCOL_20260909.md",
        "formal_prerequisites": ["two-history visual construction validation", "generalized A-only collector preflight",
                                 "new-source physical files/Goal-A image hash binding", "complete prefix construction receipts",
                                 "frozen query population and all-arm runtime gate"]}


def choose_prefix(plan, reports):
    """Read only complete A/construction receipts, not navigation-query outcomes."""
    allowed = {r["source_index"]: r for r in plan["sources"]}
    by_index = {}
    seen = set()
    for r in reports:
        index = r["source_index"]
        if index not in allowed or index in seen:
            raise ValueError("Unknown or repeated source report")
        seen.add(index)
        if r["scene"] != allowed[index]["scene"] or r["episode"] != allowed[index]["episode"]:
            raise ValueError("Source identity changed")
        if r.get("query_rollouts", 0) != 0:
            raise ValueError("Prefix must be chosen before query rollouts")
        if not r.get("verified") or not r.get("construction_complete"):
            continue
        if not isinstance(r.get("pair_constructible"), bool):
            raise ValueError("A constructibility boolean is required")
        by_index[index] = r
    for stage in plan["stages"]:
        prefix = stage["scene_prefix"]
        expected = [r for r in plan["sources"] if r["scene_rank"] < prefix]
        missing = [r["source_index"] for r in expected if r["source_index"] not in by_index]
        if missing:
            return {"state": "await_complete_prefix", "next_prefix": prefix, "missing_source_indices": missing,
                    "selected_prefix": None, "ready_for_query": False}
        retained = [r for r in expected if by_index[r["source_index"]]["pair_constructible"]]
        scenes = len({r["scene"] for r in retained})
        enough = len(retained) >= plan["target_histories"] and scenes >= plan["target_scene_clusters"]
        if enough or prefix == PREFIXES[-1]:
            return {"state": "target_reached" if enough else "maximum_prefix_underpowered",
                    "selected_prefix": prefix, "histories": len(retained), "scenes": scenes,
                    "retained_source_indices": [r["source_index"] for r in retained],
                    "ready_for_query": False,
                    "remaining_requirement": "independently seal actual query assets and runtime before evaluation"}
    raise AssertionError("No terminal prefix")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if sha(args.parent) != PARENT_SHA:
        raise ValueError("Wrong frozen parent")
    result = prepare(json.loads(args.parent.read_text()))
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "source_count": len(result["sources"]),
                      "stages": [{k: v for k, v in s.items() if k != "new_source_indices"} for s in result["stages"]]}, indent=2))
