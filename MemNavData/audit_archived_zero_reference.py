#!/usr/bin/env python3
"""Read-only exposure audit of archived plan diagnostics and measured motion.

Old logs do not retain full candidate coordinates. We therefore report the
literal observation 'all candidate endpoint norms were zero', not a recovered
trajectory or a counterfactual success label. No model or simulator is run.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def inspect(plans, trace):
    positions = {int(r["step"]): (float(r["x"]), float(r["z"])) for r in trace}
    if len(positions) != len(trace):
        raise ValueError("Duplicate recorded motion step")
    events = []
    for row in plans:
        if (row.get("candidate_endpoint_length_mean") != 0.
                or row.get("candidate_endpoint_length_std") != 0.
                or not row.get("trajectory_candidate_count")):
            continue
        step = int(row["step"])
        critic = row.get("navdp_critic_max")
        pointgoal = row.get("memory_controller_pointgoal")
        first_motion = None
        if step in positions and step + 1 in positions:
            first_motion = math.dist(positions[step], positions[step + 1])
        events.append({
            "step": step, "candidate_count": row["trajectory_candidate_count"],
            "candidate_endpoint_mean_m": row["candidate_endpoint_length_mean"],
            "candidate_endpoint_std_m": row["candidate_endpoint_length_std"],
            "critic_max": critic,
            "critic_at_least_minus_half": bool(critic is not None and math.isfinite(critic) and critic >= -.5),
            "memory_pointgoal": pointgoal,
            "pointgoal_behind": bool(pointgoal and pointgoal[0] < 0.),
            "navdp_stop_evidence": row.get("navdp_stop_evidence"),
            "next_action_observed_displacement_m": first_motion,
            "next_action_moved_gt_1um": None if first_motion is None else first_motion > 1e-6,
            "selected_trajectory_sha256": row.get("selected_trajectory_sha256"),
        })
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("population", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.population.glob("*/*/*_plans.json"))
    if not files:
        raise ValueError("No archived query plans")
    source_digests, histories, queries = {}, {}, []
    for path in files:
        raw = path.read_bytes()
        payload = json.loads(raw)
        source_digests[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        history, arm = path.parent.parent.name, path.parent.name
        role = path.stem.rsplit("_", 2)[-2]
        if role not in ("novel", "revisit"):
            raise ValueError(f"Unrecognized query role in archived filename: {path}")
        # Roles are read AFTER evaluation, for stratification only.
        prefix = (payload["legA"], payload["rollout_traces"]["legA"])
        if history in histories and prefix != histories[history]["prefix"]:
            raise ValueError(f"Shared A prefix differs: {history}")
        histories.setdefault(history, {"prefix": prefix, "zero_endpoint_events": inspect(*prefix)})
        queries.append({"history": history, "arm": arm, "analysis_role": role,
                        "source": str(path.resolve()),
                        "plans": len(payload["query_leg"]),
                        "zero_endpoint_events": inspect(payload["query_leg"], payload["rollout_traces"]["query"])})
    grouped = defaultdict(list)
    for row in queries:
        grouped[(row["arm"], row["analysis_role"])].append(row)
    counts = []
    for (arm, role), rows in sorted(grouped.items()):
        events = [e for r in rows for e in r["zero_endpoint_events"]]
        eligible = [e for e in events if e["critic_at_least_minus_half"]]
        counts.append({
            "arm": arm, "analysis_role": role, "queries": len(rows),
            "queries_with_zero_endpoint_plan": sum(bool(r["zero_endpoint_events"]) for r in rows),
            "zero_endpoint_plans": len(events),
            "zero_endpoint_plans_critic_at_least_minus_half": len(eligible),
            "such_plans_with_observed_first_motion": sum(e["next_action_moved_gt_1um"] is True for e in eligible),
            "such_plans_missing_next_position": sum(e["next_action_moved_gt_1um"] is None for e in eligible),
            "such_plans_with_behind_memory_pointgoal": sum(e["pointgoal_behind"] for e in eligible),
            "queries_with_such_observed_motion": sum(any(e["critic_at_least_minus_half"] and e["next_action_moved_gt_1um"] is True
                                                         for e in r["zero_endpoint_events"]) for r in rows),
        })
    report = {
        "schema": "archived_endpoint_zero_motion_exposure_v1",
        "population": str(args.population.resolve()), "query_files": len(files),
        "distinct_histories": len(histories), "counts": counts, "queries": queries,
        "unique_source_A": [{"history": h, "plans": len(value["prefix"][0]),
                              "zero_endpoint_events": value["zero_endpoint_events"]} for h, value in sorted(histories.items())],
        "source_sha256": source_digests,
        "limitations": [
            "Archived logs contain endpoint statistics and selected-trajectory hashes, not full trajectories.",
            "Critic >= -0.5 is an offline descriptive split, not a control change or a complete source-bundle binding.",
            "Only a recorded next position is counted; a missing last endpoint is not invented.",
            "Exposure does not predict repaired SR, identify all zero references, or justify changing old success labels.",
            "Selected successful A histories are not an unbiased source-A population.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("query_files", "distinct_histories", "counts")}, indent=2))


if __name__ == "__main__":
    main()
