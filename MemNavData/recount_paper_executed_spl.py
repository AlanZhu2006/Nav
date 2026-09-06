#!/usr/bin/env python3
"""Additive SPL correction for the frozen Table-II/III populations.

Reads only existing raw records. Does not overwrite any scientific result,
select an episode, rerun a policy, or infer a missing terminal position.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from executed_path_metrics import spl, spl_from_trace


EXPANSION = ("hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/"
             "formal_20260830T045416Z_1f4979a7")
TABLE2 = EXPANSION + "/table2_leg3_power/policy_authority_closure_repair_v3"
PARENT = ("hm3d_fresh_fullmono_mixed_role_20260820/"
          "formal_20260820T143609Z_e6dd44c6")
FACTORIAL = ("final14_mono_factorial_20260819/"
             "formal_20260819T124820Z_5690569a")
ZERO = "final14_zero_depth_20260828/formal_4c061bd6b86da365"


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def record(group, metric, row, trace_path, poses, steps, endpoint,
           *, bound=None, success_key="reached", geo_key="geodesic_m",
           path_key="path_len_m"):
    success, geo = int(float(row[success_key])), float(row[geo_key])
    result = spl_from_trace(success, geo, poses, steps=steps,
                            end_position=endpoint,
                            missing_action_bound_m=bound)
    return {
        "group": group, "episode": row["episode"],
        "query_id": row.get("query_id"), "success": success,
        "geodesic_m": geo, "logged_path_m": float(row[path_key]),
        "logged_spl": spl(success, geo, float(row[path_key])),
        "metric": str(metric), "metric_sha256": sha(metric),
        "trace": str(trace_path), "trace_sha256": sha(trace_path),
        "missing_final_action_bound_m": bound, **result,
    }


def prefix_records(root):
    # Reuse the sealed verifier to bind the ORIGINAL populations and hashes;
    # the correction below independently integrates the recorded positions.
    from independent_verify_hm3d_table2_stage_spl import verify
    parent_path = root / PARENT / "sealed_inputs/parent_manifest.json"
    union_root = root / EXPANSION / "table2_source_union"
    original = verify(
        meeting_verification_path=root / TABLE2 / "meeting_result" /
        "hm3d_table2_meeting_result_independent_verification.json",
        parent_manifest_path=parent_path, source_union_root=union_root)
    results = []
    parent = read_json(parent_path)
    for scene_index, scene in enumerate(parent["scenes"]):
        output = root / PARENT / "goal_a/scenes" / f"{scene_index:02d}_{scene}"
        completion = output / "completion.json"
        if not completion.is_file():
            continue  # Empty source scenes are bound by the verifier above.
        for item in read_json(completion).get("records", []):
            trace_path = Path(item["trace_path"])
            trace = read_json(trace_path)
            metric = trace_path.parent / "metric.csv"
            row = metric_rows(metric)[0]
            steps = trace.get("step_at_reach") if trace["reached"] else None
            steps = trace["steps"] if steps is None else steps
            results.append(record(
                "table2/A", metric, row, trace_path, trace["poses"],
                int(steps), trace.get("end_position"),
                success_key="reached_A", geo_key="geo_A", path_key="len_A"))
    union = read_json(union_root / "population/population.json")
    for source in union["source_populations"]:
        source_root = Path(source["run_root"])
        manifest = read_json(source_root / "ab_population/role_pairs/manifest.json")
        for index, item in enumerate(manifest["episodes"]):
            label = f'{index:03d}_{item["scene"]}_{item["episode"]}'
            output = source_root / "factual_b" / label
            completion = read_json(output / "completion.json")
            trace_path = Path(completion["B_trace_path"])
            trace = read_json(trace_path)
            metric = output / "result/metric.csv"
            row = metric_rows(metric)[0]
            steps = trace.get("step_at_reach") if trace["reached"] else None
            steps = trace["steps"] if steps is None else steps
            results.append(record("table2/B", metric, row, trace_path,
                                  trace["poses"], int(steps),
                                  trace.get("end_position")))
    for group, key in (("table2/A", "leg1_novel"), ("table2/B", "leg2_novel")):
        rows = [row for row in results if row["group"] == group]
        assert len(rows) == original[key]["attempts"]
        assert sum(row["success"] for row in rows) == original[key]["successes"]
    return results, original


def query_records(root, relative, group_prefix, arms, *, bound=None):
    results = []
    evaluation = root / relative / "evaluation/natural_direction"
    for history in sorted(evaluation.iterdir()):
        if not history.is_dir():
            continue
        for arm in arms:
            metric = history / arm / "metric.csv"
            for row in metric_rows(metric):
                trace_path = metric.parent / (
                    row["episode"] + "_" + row["query_id"] + "_plans.json")
                payload = read_json(trace_path)
                endpoint = None
                if all(row.get(k) not in (None, "") for k in
                       ("end_x_m", "end_y_m", "end_z_m")):
                    endpoint = [float(row[k]) for k in
                                ("end_x_m", "end_y_m", "end_z_m")]
                results.append(record(
                    f'{group_prefix}/{arm}/{row["analysis_role"]}', metric,
                    row, trace_path, payload["rollout_traces"]["query"],
                    int(row["steps"]), endpoint, bound=bound))
    return results


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
        if row["group"].startswith("table3/"):
            groups[row["group"].rsplit("/", 1)[0] + "/all"].append(row)
    result = {}
    for group, items in sorted(groups.items()):
        mean = lambda key: (None if any(r[key] is None for r in items) else
                            math.fsum(r[key] for r in items) / len(items))
        result[group] = {
            "n": len(items), "successes": sum(r["success"] for r in items),
            "logged_spl": mean("logged_spl"), "corrected_spl": mean("spl"),
            "spl_lower": mean("spl_lower"), "spl_upper": mean("spl_upper"),
            "terminal_positions_available": sum(
                r["terminal_position_available"] for r in items),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, original = prefix_records(args.results_root)
    rows += query_records(args.results_root, TABLE2 + "/formal/navdp",
                          "table2/C", ("mono_native", "mono_cec"))
    # Frozen pursuit: v <= 0.0376 m, accepted snap offset <= 0.06 m.
    # Triangle inequality bounds the ONE unrecorded terminal displacement.
    # This is an interval calculation, not an imputed endpoint or new SPL.
    for relative, arms in (
        (FACTORIAL, ("metric_native", "mono_native", "metric_cec", "mono_cec")),
        (ZERO, ("zero_native",)),
    ):
        rows += query_records(args.results_root, relative, "table3", arms,
                              bound=0.0376 + 0.06)
    groups = summarize(rows)
    expected = {"table2/A": (196, 131), "table2/B": (183, 54),
                "table2/C/mono_native/novel": (20, 4),
                "table2/C/mono_cec/novel": (20, 4),
                "table2/C/mono_native/revisit": (20, 8),
                "table2/C/mono_cec/revisit": (20, 17),
                "table3/metric_native/all": (42, 11),
                "table3/zero_native/all": (42, 4),
                "table3/mono_native/all": (42, 10),
                "table3/metric_cec/all": (42, 26),
                "table3/mono_cec/all": (42, 28)}
    for group, (n, successes) in expected.items():
        assert (groups[group]["n"], groups[group]["successes"]) == (n, successes), group
    payload = {"schema_version": "paper_executed_spl_correction_20260906_v1",
               "success_and_population_unchanged": True,
               "path_convention": "sum of actual consecutive x-z displacements",
               "original_stage_verification": original,
               "groups": groups, "rows": rows}
    with args.out.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "sha256": sha(args.out),
                      "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
