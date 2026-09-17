"""Summarize verified matched-support trajectories without treating errors as failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from summarize_gem_reviewer_supplement import bearing_metrics, collect, dump, sha


def reduce(run):
    plan, export = collect(run)
    assert plan["schema"] == "gem_position_readout_v1"
    arms = ["pnp_goal", "historical_camera"]
    assert plan["arms"] == arms and plan["success_radius_m"] == .3
    assert len(plan["cells"]) == 7 and len(export["cells"]) == 7
    rows, pending = [], []
    for item in export["cells"]:
        cell = plan["cells"][item["index"]]
        if not item.get("completion", {}).get("completed"):
            pending.append(dict(index=item["index"], scene=cell["scene"], episode=cell["episode"],
                                failure=item.get("failure"), progress=item.get("progress")))
            continue
        complete, summary, audit = (item[name] for name in ("completion", "summary", "independent_verification"))
        assert complete["plan_sha256"] == summary["plan_sha256"] == export["plan_sha256"]
        assert complete["summary_sha256"] == item["summary_sha256"] == audit["summary_sha256"]
        assert complete["independent_verification_sha256"] == item["independent_verification_sha256"]
        assert summary["completed"] and audit["verified"] and summary["cell"] == cell
        assert audit["first_certificate_and_pnp_identical"] and audit["full_causal_prefix_identical"]
        assert audit["predicted_historical_bearing_verified"]
        metrics = {r["arm"]: r for r in audit["metrics"]}
        assert set(metrics) == set(arms) and {r["arm"] for r in summary["records"]} == set(arms)
        for record in summary["records"]:
            m = metrics[record["arm"]]
            assert m["reached"] == record["reached"] and m["steps"] == record["steps"]
            assert m["execution_context"]["mesh_bytes_identical"]
            rows.append(dict(index=cell["index"], paper_index=cell["paper_index"], dataset=cell["dataset"],
                scene=cell["scene"], episode=cell["episode"], arm=record["arm"],
                selection_offset_m=cell["selection_offset_m"], selection_anchor=cell["selection_anchor"],
                reached=record["reached"], steps=record["steps"], turn_actions=m["turn_actions"],
                endpoint_error_m=m["endpoint_error_m"], minimum_error_m=m["minimum_error_m"],
                final_geodesic_m=m["final_geodesic_m"], passages=m["passages"],
                bearing=bearing_metrics(Path(record["directory"]), record, cell["history_frames"]),
                gpu=complete["gpu"], wall_seconds=record["wall_seconds"]))
    ids = sorted({r["index"] for r in rows})
    by = {(r["index"], r["arm"]): r for r in rows}
    assert len(by) == len(rows) == len(ids) * 2
    thresholds = []
    for radius in (1., .5, .3):
        for distance in ("euclidean", "geodesic"):
            field = distance + "_first_action"
            hit = {(i, arm): next(p for p in by[i, arm]["passages"] if p["radius_m"] == radius)[field]
                   for i in ids for arm in arms}
            thresholds.append(dict(radius_m=radius, distance=distance, histories=len(ids),
                successes={arm: sum(hit[i, arm] is not None for i in ids) for arm in arms},
                pnp_only_successes=sum(hit[i, "pnp_goal"] is not None and hit[i, "historical_camera"] is None for i in ids),
                historical_only_successes=sum(hit[i, "historical_camera"] is not None and hit[i, "pnp_goal"] is None for i in ids)))
    groups = []
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        errors = [r["bearing"]["initial_error_deg"] for r in selected if r["bearing"] and r["bearing"]["initial_error_deg"] is not None]
        groups.append(dict(arm=arm, histories=len(selected),
            success_03m=sum(r["reached"] for r in selected),
            median_endpoint_error_m=statistics.median(r["endpoint_error_m"] for r in selected) if selected else None,
            median_initial_bearing_error_deg=statistics.median(errors) if errors else None))
    result = dict(complete=not pending, verified_histories=len(ids), verified_rollouts=len(rows),
        scheduled_histories=7, thresholds=thresholds, groups=groups, rows=rows, pending=pending,
        source_sha256={str(run / "plan.json"): sha(run / "plan.json"),
                       str(run / "status_export.json"): sha(run / "status_export.json")}, time=time.time(),
        scope="All seven preselected accepted main-table recalls with >=1 m goal-support offset. Post-hoc conditional diagnostic; paired descriptive results. Both arms automatically stop at 0.3 m Euclidean distance or at the original budget/termination. Geodesic and looser Euclidean thresholds are first passages on these same new trajectories, not autonomous STOP trials. Failures of infrastructure or verification are listed separately from valid navigation failures.")
    dump(run / "reduced.json", result)
    lines = ["# Matched-support position readout", "", f"Verified: {len(ids)}/7 histories, {len(rows)}/14 rollouts.", "",
             "| Distance | Radius | PnP goal | Historical camera | PnP-only / historical-only |",
             "|---|---:|---:|---:|---:|"]
    for row in thresholds:
        lines.append(f"| {row['distance']} | {row['radius_m']:.1f} m | {row['successes']['pnp_goal']}/{len(ids)} | {row['successes']['historical_camera']}/{len(ids)} | {row['pnp_only_successes']} / {row['historical_only_successes']} |")
    lines.extend(["", result["scope"], ""])
    (run / "RESULT.md").write_text("\n".join(lines))
    with (run / "positions.csv").open("w", newline="") as stream:
        fields = ["index", "paper_index", "dataset", "scene", "episode", "arm", "selection_offset_m", "reached", "steps", "turn_actions", "endpoint_error_m", "minimum_error_m"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in reduce(args.run).items() if k not in ("rows", "pending")}))
