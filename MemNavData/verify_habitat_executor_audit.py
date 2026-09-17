#!/usr/bin/env python3
"""Recompute local executor evidence from saved commands, NavMesh and poses."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import habitat_sim
import numpy as np

from habitat_executor_audit import compare_step, dump, sha, summarize_steps


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(root):
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "summary.json").read_text())
    assert result["completed"] and not result["changed_source_files"]
    assert len(result["results"]) == 4*len(manifest["histories"])
    for path, expected in manifest["code_and_weights"].items():
        assert sha(path) == expected, path
    records, by_key = [], {}
    for record in result["results"]:
        folder = root / "evaluation" / record["scene"] / f"{record['policy']}__{record['mode']}"
        files = list(folder.glob("*_plans.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        actions = read_rows(folder / "executor_actions.jsonl")
        plans = read_rows(folder / "full_plan_outputs.jsonl")
        trace = payload["rollout_traces"]["query"]
        end = payload["query_result"]
        points = np.array([[r[k] for k in ("x", "y", "z")] for r in trace]
                          + [end["end_position"]])
        assert len(actions) == len(trace) == record["steps"]
        np.testing.assert_allclose([r["executed_position"] for r in actions], points[1:], atol=1e-9, rtol=0)
        length = float(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1).sum())
        assert abs(length-record["actual_path_len_m"]) < 1e-8
        distance = float(np.linalg.norm(points[-1, [0, 2]]-record["goal_xz_evaluator_only"]))
        assert abs(distance-record["final_goal_dist_m"]) < 1e-8
        assert int(distance < 1.0) == record["reached"] == int(end["reached"])
        pf = habitat_sim.PathFinder()
        assert pf.load_nav_mesh(str(folder / "execution.navmesh"))
        for action in actions:
            endings, yaw, recomputed = compare_step(action["position_before"], action["yaw_before"],
                                                    action["world_path"], pf)
            np.testing.assert_allclose(endings[record["mode"]], action["executed_position"], atol=1e-9, rtol=0)
            assert recomputed["legacy_branch"] == action["legacy_branch"]
            assert abs(recomputed["legacy_vs_try_step_m"]-action["legacy_vs_try_step_m"]) < 1e-9
        assert summarize_steps(actions)["actions"] == record["executor"]["actions"]
        value = dict(scene=record["scene"], policy=record["policy"], mode=record["mode"],
                     reached=record["reached"], steps=record["steps"], actual_path_len_m=length,
                     final_distance_m=distance, **summarize_steps(actions))
        records.append(value)
        by_key[(record["scene"], record["policy"], record["mode"])] = (payload, plans, actions)
    pairs = []
    for history in manifest["histories"]:
        for policy in ("mono_native", "mono_cec"):
            left = by_key[(history["scene"], policy, "legacy_snap")]
            right = by_key[(history["scene"], policy, "try_step")]
            assert left[0]["replay"] == right[0]["replay"]
            assert left[0]["rollout_traces"]["legA"] == right[0]["rollout_traces"]["legA"]
            assert left[0]["rollout_traces"]["query"][0] == right[0]["rollout_traces"]["query"][0]
            assert left[1][0]["selected_trajectory"] == right[1][0]["selected_trajectory"]
            n = min(len(left[2]), len(right[2]))
            errors = [float(np.linalg.norm(np.array(a["executed_position"])-b["executed_position"]))
                      for a, b in zip(left[2], right[2])]
            pairs.append(dict(scene=history["scene"], policy=policy, shared_A_equal=True,
                              first_plan_equal=True, compared_prefix_actions=n,
                              first_position_difference_gt_1um=next((i for i, e in enumerate(errors) if e > 1e-6), None),
                              max_prefix_position_difference_m=max(errors, default=0.),
                              all_selected_plans_equal=(len(left[1]) == len(right[1]) and all(
                                  a["selected_trajectory"] == b["selected_trajectory"] for a, b in zip(left[1], right[1])))))
    report = dict(verified=True, scope=manifest["scope"], records=records, pairs=pairs,
                  manifest_sha256=sha(root / "manifest.json"), summary_sha256=sha(root / "summary.json"))
    dump(root / "independent_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
