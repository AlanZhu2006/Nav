"""Read verified experimental receipts; distinguish scheduled and valid coverage."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def exact(wins, losses):
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses)+1)) / 2**n) if n else 1.


def collect(run):
    plan = load(run / "plan.json")
    if not (run / "tasks").is_dir():
        export = load(run / "status_export.json")
        assert export["plan_sha256"] == sha(run / "plan.json")
        return plan, export
    export = dict(run=str(run), plan_sha256=sha(run / "plan.json"), queried_unix=time.time(), cells=[])
    for cell in plan["cells"]:
        folder = run / "tasks" / str(cell["index"])
        row = dict(index=cell["index"], scene=cell["scene"], episode=cell["episode"])
        for name in ("progress", "failure", "completion", "summary", "independent_verification", "paired_gpu_binding"):
            path = folder / (name + ".json")
            if path.exists():
                row[name] = load(path)
                row[name + "_sha256"] = sha(path)
        path = folder / "evaluation/initial_turn_only/bearing_intervention.json"
        if path.exists():
            row["initial_turn_only_intervention"] = load(path)
        export["cells"].append(row)
    export["complete"] = all(c.get("completion", {}).get("completed") and c.get("independent_verification", {}).get("verified") for c in export["cells"])
    dump(run / "status_export.json", export)
    return plan, export


def bearing_metrics(folder, terminal, history_frames):
    if not (folder / "query_plans.json").exists():
        return None
    query = load(folder / "query_plans.json")
    plans = [json.loads(line) for line in (folder / "full_plan_outputs.jsonl").read_text().splitlines()]
    assert len(plans) == len(query["query_leg"])
    points = []
    for p, actual in zip(query["query_leg"], plans):
        assert p["frame_idx"] == actual["receipt"]["memory_frame_idx"]
        bearing = p.get("memory_bearing_unit")
        if bearing is None:
            continue
        dx = terminal["goal_xz_evaluator_only"][0] - actual["position"][0]
        dz = terminal["goal_xz_evaluator_only"][1] - actual["position"][2]
        truth = math.atan2(-dx, -dz) - actual["yaw"]
        angle = math.atan2(bearing[1], bearing[0]) - truth
        error = abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle))))
        points.append(dict(action=actual["next_action_index"], distance_m=math.hypot(dx, dz),
                           absolute_error_deg=error, bearing=bearing, position=actual["position"],
                           yaw=actual["yaw"], goal_xz=terminal["goal_xz_evaluator_only"]))
    first = query["query_leg"][0]
    accepted = first["certified_relocalization_accepted"] is True
    anchor = first.get("router_selected_anchor") if accepted else None
    walked = None
    if anchor is not None:
        poses = query["rollout_traces"]["legA"][anchor:]
        locations = [(p["x"], p["z"]) for p in poses] + [(plans[0]["position"][0], plans[0]["position"][2])]
        walked = math.fsum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(locations, locations[1:]))
    errors = [p["absolute_error_deg"] for p in points]
    return dict(accepted=accepted, selected_anchor=anchor,
                age_raw_frames=None if anchor is None else history_frames-anchor,
                cumulative_walk_since_anchor_m=walked,
                initial_error_deg=errors[0] if errors else None,
                median_error_deg=statistics.median(errors) if errors else None,
                p90_error_deg=float(np.quantile(errors, .9)) if errors else None,
                maximum_error_deg=max(errors) if errors else None, points=points)


def reduce(runs, out, mode, reference=None):
    arms = (["native", "fixed_half_turn", "gem_no_turn", "initial_turn_only", "full_gem"]
            if mode == "controls" else ["full_history", "recent_seven"])
    treatment = "full_gem" if mode == "controls" else "full_history"
    controls = ["fixed_half_turn", "initial_turn_only", "gem_no_turn"] if mode == "controls" else ["recent_seven"]
    rows, pending, sources, identities, all_ids = [], [], {}, set(), set()
    settings = None
    for run in runs:
        plan, export = collect(run)
        sources[str(run / "plan.json")] = sha(run / "plan.json")
        sources[str(run / "status_export.json")] = sha(run / "status_export.json")
        assert plan["arms"] == arms
        fixed = tuple(plan[k] for k in ("memory_mode", "dense_window", "authority_policy", "max_steps", "success_radius_m", "exec_horizon"))
        assert settings is None or settings == fixed
        settings = fixed
        assert len(export["cells"]) == len(plan["cells"])
        assert {c["index"] for c in export["cells"]} == set(range(len(plan["cells"])))
        for item in export["cells"]:
            cell = plan["cells"][item["index"]]
            identity = (cell["dataset"], cell["scene"], cell["episode"])
            assert identity not in all_ids
            all_ids.add(identity)
            if not item.get("completion", {}).get("completed"):
                pending.append(dict(identity=identity, run=str(run), index=item["index"],
                                    failure=item.get("failure"), progress=item.get("progress")))
                continue
            complete, summary, audit = item["completion"], item["summary"], item["independent_verification"]
            for name in ("completion", "summary", "independent_verification"):
                serialized = (json.dumps(item[name], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
                assert hashlib.sha256(serialized).hexdigest() == item[name + "_sha256"]
            assert complete["plan_sha256"] == export["plan_sha256"] == summary["plan_sha256"]
            assert complete["summary_sha256"] == item["summary_sha256"]
            assert complete["independent_verification_sha256"] == item["independent_verification_sha256"]
            assert summary["completed"] and audit["verified"] and summary["cell"] == cell
            assert len(summary["records"]) == len(arms)
            assert {r["arm"] for r in summary["records"]} == set(arms)
            checks = {r["arm"]: r for r in audit["checks"]}
            identities.add(identity)
            for record in summary["records"]:
                arm = record["arm"]
                check = checks[arm]
                assert record["steps"] == check["steps"] and record["reached"] == check["reached"]
                row = dict(identity=list(identity), dataset=cell["dataset"], scene=cell["scene"],
                           episode=cell["episode"], paper_index=cell["paper_index"], arm=arm,
                           cohort=("remaining57" if plan["schema"] == "gem_reviewer_paper_controls_v2"
                                   else "old_support13"),
                           run=str(run), index=item["index"], reached=record["reached"], steps=record["steps"],
                           path_m=record["actual_path_len_m"], spl=record["spl"],
                           final_distance_m=record["final_goal_dist_m"],
                           turn_actions=check["heading"]["turn_actions"],
                           turn_sequences=1 if arm == "fixed_half_turn" else check["heading"]["turn_sequences"],
                           initial_guided_turn=item.get("initial_turn_only_intervention", {}).get("initial_rearward"),
                           gpu=complete["gpu"], wall_seconds=record["wall_seconds"])
                if arm == treatment or mode == "domains":
                    row["bearing"] = bearing_metrics(Path(record["directory"]), record, cell["history_frames"])
                rows.append(row)
    by = {(tuple(r["identity"]), r["arm"]): r for r in rows}
    assert len(by) == len(rows) == len(identities) * len(arms)
    groups = []
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        if selected:
            groups.append(dict(arm=arm, queries=len(selected), success=sum(r["reached"] for r in selected),
                               mean_actions=statistics.mean(r["steps"] for r in selected),
                               mean_turn_actions=statistics.mean(r["turn_actions"] for r in selected),
                               mean_spl=statistics.mean(r["spl"] for r in selected)))
    contrasts = []
    for control in controls:
        wins = sum(by[i, treatment]["reached"] > by[i, control]["reached"] for i in identities)
        losses = sum(by[i, treatment]["reached"] < by[i, control]["reached"] for i in identities)
        scenes = sorted({i[:2] for i in identities})
        ci = None
        if len(scenes) > 1:
            sums = np.array([sum(by[i, treatment]["reached"]-by[i, control]["reached"] for i in identities if i[:2] == s) for s in scenes])
            counts = np.array([sum(i[:2] == s for i in identities) for s in scenes])
            pick = np.random.default_rng(20260916).integers(0, len(scenes), size=(10000, len(scenes)))
            draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
            ci = np.quantile(draws, [.025, .975]).tolist()
        shared = [i for i in identities if by[i, treatment]["reached"] and by[i, control]["reached"]]
        contrasts.append(dict(comparison=treatment + " versus " + control, paired_wins=wins, paired_losses=losses,
                              exact_mcnemar_p=exact(wins, losses), scene_cluster_percentile95=ci,
                              shared_successes=len(shared), shared_success_mean_action_difference=
                              statistics.mean(by[i, treatment]["steps"]-by[i, control]["steps"] for i in shared) if shared else None))
    adjusted = 0.
    for rank, contrast in enumerate(sorted(contrasts, key=lambda c: c["exact_mcnemar_p"])):
        adjusted = max(adjusted, min(1., (len(contrasts)-rank)*contrast["exact_mcnemar_p"]))
        contrast["holm_p"] = adjusted
    reproduction = []
    if reference is not None:
        sources[str(reference)] = sha(reference)
        refs = {c["paper_index"]: c for c in load(reference)["cells"]}
        for r in rows:
            if r["arm"] not in ("native", "full_gem"):
                continue
            old = refs[r["paper_index"]][r["arm"]]
            reproduction.append(dict(paper_index=r["paper_index"], arm=r["arm"],
                                     success_matches=r["reached"] == old["reached"], steps_match=r["steps"] == old["steps"],
                                     path_delta_m=r["path_m"]-old["actual_path_m"], spl_delta=r["spl"]-old["spl"],
                                     turns_match=r["turn_actions"] == old["heading"]["turn_actions"]))
    cohorts = []
    for cohort in sorted({r["cohort"] for r in rows}):
        selected = [r for r in rows if r["cohort"] == cohort]
        cohorts.append(dict(cohort=cohort, histories=len(selected)//len(arms),
                            groups=[dict(arm=arm, success=sum(r["reached"] for r in selected if r["arm"] == arm)) for arm in arms]))
    turn_strata = []
    if mode == "controls":
        for present in (True, False):
            ids = [i for i in identities if by[i, treatment]["initial_guided_turn"] is present]
            if ids:
                turn_strata.append(dict(initial_guided_turn=present, histories=len(ids),
                    groups=[dict(arm=arm, success=sum(by[i, arm]["reached"] for i in ids)) for arm in arms],
                    later_full_turn_sequences=sum(max(0, by[i, treatment]["turn_sequences"]-int(present)) for i in ids)))
    result = dict(complete=not pending, mode=mode, scheduled_histories=len(all_ids), verified_histories=len(identities),
                  scenes=len({i[:2] for i in identities}), verified_rollouts=len(rows), groups=groups,
                  cohorts=cohorts, actual_initial_turn_strata=turn_strata,
                  primary_contrasts=contrasts, main_reference_reproduction=reproduction,
                  pending=pending, rows=rows, source_sha256=sources, queried_unix=time.time(),
                  scope="Paired existing histories; decisions and arms are not independent queries. Report all scheduled coverage. Automatic 1 m arrival. Scene-cluster intervals are descriptive; nonsignificance is not equivalence. Mean actions include early failures; shared-success action differences condition on both arms succeeding.")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "reduced.json", result)
    text = ["# Reviewer supplement: " + mode, "", f"Verified histories: {len(identities)}/{len(all_ids)}; verified rollouts: {len(rows)}.", "", "| Arm | Success | Mean actions | Mean turn actions |", "|---|---:|---:|---:|"]
    for g in groups:
        text.append(f"| {g['arm']} | {g['success']}/{g['queries']} | {g['mean_actions']:.1f} | {g['mean_turn_actions']:.1f} |")
    text.extend(["", "| Contrast | Paired wins | Paired losses | Exact p | Holm p |", "|---|---:|---:|---:|---:|"])
    for c in contrasts:
        text.append(f"| {c['comparison']} | {c['paired_wins']} | {c['paired_losses']} | {c['exact_mcnemar_p']:.4g} | {c['holm_p']:.4g} |")
    if cohorts:
        text.extend(["", "Cohorts are reported separately from the pooled comparison:"])
        for cohort in cohorts:
            text.append("- " + cohort["cohort"] + ": " + str(cohort["histories"]) + " histories; " + ", ".join(g["arm"] + " " + str(g["success"]) for g in cohort["groups"]) + ".")
    for stratum in turn_strata:
        text.append("- Actual initial guided turn " + str(stratum["initial_guided_turn"]) + ": " + str(stratum["histories"]) + " histories; " + str(stratum["later_full_turn_sequences"]) + " later full-GEM turn sequences.")
    text.extend(["", result["scope"], "", "Complete scheduled coverage." if result["complete"] else "Incomplete coverage: these are interim results.", ""])
    (out / "RESULT.md").write_text("\n".join(text))
    if rows:
        columns = [k for k in rows[0] if k not in ("bearing", "gpu", "identity")]
        with (out / "controls.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: row[k] for k in columns} for row in rows)
    print(json.dumps({k: result[k] for k in ("complete", "mode", "scheduled_histories", "verified_histories", "verified_rollouts", "groups", "primary_contrasts")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controls", "domains"), required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reduce(args.runs, args.out, args.mode, args.reference)
