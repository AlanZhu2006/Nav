"""Independent measurements for the local lifecycle smoke, not formal SR."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_local import load, dump, sha
from MemNavData.executed_path_metrics import planar_path_length, spl


def jsonl(path):
    import json
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def verify(root, out):
    manifest = load(root / "manifest.json")
    run = root / "evaluation"
    summary = load(run / "chain_summary.json")
    actions = jsonl(run / "executor_actions.jsonl")
    http = jsonl(run / "navdp_http_receipts.jsonl")
    memory = jsonl(run / "memory_http_boundary.jsonl")
    formal = bool(manifest["formal_population"])
    common = bool(manifest.get("common_online_goals", False))
    assert manifest["paired_navigation_comparison"] is common
    assert summary["formal_population"] is formal
    if formal:
        from MemNavData.table2_continuous_population import task_at
        frozen = load(Path(manifest["common_root"]) / "frozen_task.json")
        assert frozen == task_at(frozen["population_file"], frozen["task_index"])
        assert manifest["population_sha256"] == frozen["population_sha256"]
        assert manifest["source"] == frozen["source"] and manifest["sequence"] == frozen["sequence"]
        assert load(manifest["a_query"]) == frozen["a_query"]
        assert common
    assert manifest["runtime_profile"] == "bounded_standard/rgb_v1/source_rgb/heading_on"
    assert sum(row["path"] == "/navigator_reset" for row in http) == 1
    assert not any(row["path"] == "/navigator_reset_env" for row in http)
    assert sum(row["path"] == "/navigator_reset" for row in memory) == 1
    assert not any(row["path"] == "/goal_session_replay" for row in memory)
    assert all(not set(row["sent_fields"]) & {
        "analysis_role", "role", "start_position", "start_yaw", "floor_position",
        "executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m",
    } for row in memory)
    receipts = [r["monocular_depth_receipt"] for r in http if r.get("monocular_depth_receipt")]
    assert receipts and all(r["depth_source"] == "monocular_sidecar" and
                            r["metric_depth_sensor_consumed"] is False for r in receipts)
    active = [r for r in receipts if r["frame_index"] >= 40]
    scale_ids = {r["scale_receipt_sha256"] for r in active}
    assert len(scale_ids) <= 1
    for r in active:
        scale = r["scale_receipt"]
        assert scale["scale_prefix_first_frame"] == 0 and scale["scale_prefix_last_frame"] == 39
        assert scale["whole_episode_ground_cache_consumed"] is False
    pose, yaw = manifest["source"]["start_position"], manifest["source"]["start_yaw"]
    previous_index, offset, checks, failed = None, 0, [], False
    for stage in "ABC":
        directory = run / f"leg_{stage}"
        if not directory.exists():
            continue
        assert not failed, "A later leg was executed after navigation failure"
        trace = load(directory / "actual_trace.json")
        measure = load(directory / "measurement.json")
        continuity = load(directory / "continuity.json")
        query = load(run / f"selected_{stage}.json")
        if common:
            permit = load(Path(manifest["common_root"]) / f"permit_{stage}_{manifest['arm']}.json")
            assert permit["status"] == "run" and sha(permit["query"]) == permit["query_sha256"]
            assert query == load(permit["query"])
        assert sha(query["goal_rgb"]) == trace["goal_sha256"] == query["goal_rgb_sha256"]
        np.testing.assert_allclose(continuity["start_position"], pose, rtol=0, atol=1e-8)
        assert abs(math.atan2(math.sin(continuity["start_yaw"]-yaw), math.cos(continuity["start_yaw"]-yaw))) < 1e-8
        if previous_index is not None:
            assert continuity["first_memory_index"] == previous_index + 1
        previous_index = continuity["last_memory_index"]
        ticks = actions[offset:offset + trace["steps"]]
        assert len(ticks) == trace["steps"]
        for t, observation in zip(ticks, trace["poses"]):
            np.testing.assert_allclose(t["position_before"], [observation[k] for k in "xyz"], atol=1e-8, rtol=0)
            assert t["selected_mode"] == "bounded_standard"
        if ticks:
            np.testing.assert_allclose(ticks[-1]["actual_position"], trace["end_position"], atol=1e-8, rtol=0)
        length = planar_path_length(trace["poses"], steps=trace["steps"], end_position=trace["end_position"])
        displacement = math.fsum(t["actual_translation_m"] for t in ticks)
        assert abs(length - displacement) < 1e-7
        assert abs(length - measure["actual_path_len_m"]) < 1e-7
        assert abs(spl(int(trace["reached"]), query["geodesic_m"], length) - measure["spl"]) < 1e-8
        if query["analysis_role"] == "novel":
            assert abs(query["initial_relative_route_angle_deg"]) <= 60. + 1e-9
            assert query["max_history_covis"] < .10 and query["current_view_covis"] < .10
        else:
            assert .55 <= query["max_history_covis"] <= .90 and query["current_view_covis"] < .10
        checks.append(dict(stage=stage, reached=trace["reached"], steps=trace["steps"],
                           actual_path_m=length, spl=measure["spl"], first_frame=continuity["first_memory_index"],
                           last_frame=continuity["last_memory_index"]))
        pose, yaw = trace["end_position"], trace["end_yaw"]
        offset += trace["steps"]
        failed = not trace["reached"]
    assert offset == len(actions)
    if summary["status"] == "lifecycle_finished":
        success_count = sum(row["reached"] for row in checks)
        assert summary["goals_completed"] == success_count
        assert summary["cumulative_success"] == [int(success_count > i) for i in range(3)]
        for record in summary["legs"][len(checks):]:
            assert record["attempted"] is False and record["reached"] is None
    else:
        assert summary["status"] == "task_construction_blocked" and summary["no_navigation_failure_imputed"]
    scope = ("frozen continuous common-goal task" if formal else
             "local common-goal lifecycle; not a formal population" if common else
             "local lifecycle; NOT a paired/formal SR result")
    result = dict(verified=True, scope=scope, formal_population=formal, status=summary["status"],
                  root=str(root), per_leg=checks, one_episode_reset=True, no_prefix_replay=True,
                  original_scale_receipt_ids=sorted(scale_ids), actual_actions=len(actions),
                  all_goals_completed=len(checks) == 3 and all(r["reached"] for r in checks))
    dump(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.run.resolve(), args.out.resolve()))
