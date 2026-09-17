"""Summarize complete new-A role pairs without merging source/query denominators.

No rollout or source selection occurs here. The paired SR/SPL statistics reuse
the frozen covisibility analysis; this entry changes only population accounting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from repaired_covisibility_eval import load, require, sha
from seal_repaired_fullmono_queries import ARMS, SCHEMA, SCOPE, read_scene
from construct_repaired_fullmono_queries import save
from summarize_covisibility_repaired import group_stats


def source_accounting(population):
    plan = load(population["source_plan"])
    require(sha(population["source_plan"]) == population["source_plan_sha256"], "Source plan changed")
    prefix = population["construction_decision"]["selected_prefix"]
    expected = {s["source_index"] for s in plan["sources"] if s["scene_rank"] < prefix}
    rows, reports = [], []
    for binding in population["source_scenes"]:
        r, a, current = read_scene(Path(binding["root"]), population["source_plan_sha256"])
        require(current == binding, "Source collection/construction receipt changed")
        rows.extend(a)
        reports.extend(r)
    require(len(rows) == len(expected) and {r["source_index"] for r in rows} == expected,
            "Source A denominator changed")
    pairs = [r for r in reports if r["pair_constructible"]]
    waterfall = dict(planned_A=len(expected), completed_A=len(rows),
        successful_A=sum(bool(r["reached"]) for r in rows),
        constructible_role_pairs=len(pairs), pair_scene_clusters=len({r["scene"] for r in pairs}))
    require(waterfall == population["source_waterfall"], "Sealed source waterfall changed")
    return dict(waterfall, A_success_rate=waterfall["successful_A"] / len(rows),
        source_A_records=rows, construction_records=reports,
        interpretation="A success and query constructibility are separate from conditional query SR")


def read_query(task, population, population_sha, folder):
    receipt = folder / "archive_receipt.json"
    if not receipt.exists():
        return None, "archive_receipt_missing"
    archive = load(receipt)
    if not archive["completed"] or archive["exit_code"] != 0:
        return None, "paired_task_or_archive_incomplete"
    verified = load(folder / "independent_verification.json")
    manifest = load(folder / "manifest.json")
    require(archive["all_member_hashes_readback_verified"]
            and sha(archive["archive"]) == archive["archive_sha256"], "Raw archive changed")
    require(archive["verification"] == verified and verified["verified"], "Verification copies differ")
    require(verified["schema"] == manifest["schema"] == SCHEMA, "Wrong runtime population schema")
    require(verified["task"] == manifest["task"] == task
            and verified["population_sha256"] == manifest["population_sha256"] == population_sha,
            "Task/population pairing changed")
    for key in ("runtime_source_receipt", "evaluation_source_receipt"):
        require(manifest[key] == population[key], "Query source version changed")
    payload = load(task["construction"])
    require(sha(task["construction"]) == task["construction_sha256"], "Query construction changed")
    selected = [q for q in payload["queries"] if q["query_id"] == task["query_id"]]
    require(len(selected) == 1 and task["query_id"] in payload["selected_query_ids"], "Unselected target")
    role = selected[0]["analysis_role"]
    require(role in ("novel", "revisit") and manifest["analysis_role"] == role
            and manifest["scene"] == payload["scene"], "Analysis role/source changed")
    records = verified["records"]
    require([r["arm"] for r in records] == task["arm_order"]
            and set(task["arm_order"]) == set(ARMS), "Incomplete or reordered paired arms")
    arms = {r["arm"]: r for r in records}
    return dict(task_index=task["task_index"], history_index=task["history_index"],
        scene=manifest["scene"], query_id=task["query_id"], analysis_role=role, arms=arms,
        cec_takeover_plans=verified["cec_takeover_plans"],
        no_takeover_exact_native=verified["no_takeover_exact_native"]), None


def summarize(population_path, population_sha, evaluation_root):
    require(sha(population_path) == population_sha, "Population changed")
    population = load(population_path)
    require(population["schema"] == SCHEMA and population["phase"] == "sealed_before_query_evaluation",
            "Not a sealed actual-A population")
    require(population["arms"] == list(ARMS), "Arm family changed")
    for key in ("runtime_source_receipt", "evaluation_source_receipt"):
        require(sha(population[key]["path"]) == population[key]["sha256"], "Frozen source receipt changed")
    tasks = population["tasks"]
    require([t["task_index"] for t in tasks] == list(range(len(tasks)))
            and len(tasks) == population["query_count"] == 2 * population["histories"]
            and len(tasks) * 3 == population["query_arm_count"], "Query count/order changed")
    sources = source_accounting(population)
    rows, missing = [], []
    for task in tasks:
        row, reason = read_query(task, population, population_sha,
                                 evaluation_root / f"task_{task['task_index']:03d}")
        if reason is not None:
            missing.append(dict(task_index=task["task_index"], reason=reason))
        else:
            rows.append(row)
    if missing:
        return dict(complete=False, expected_queries=len(tasks), verified_queries=len(rows),
            missing_or_failed_tasks=missing,
            note="No final SR: incomplete infrastructure is not a navigation failure")
    for index in population["construction_decision"]["retained_source_indices"]:
        pair = [r for r in rows if r["history_index"] == index]
        require(len(pair) == 2 and {r["analysis_role"] for r in pair} == {"novel", "revisit"},
                "History lacks a complete role pair")
    groups = {role: group_stats([r for r in rows if r["analysis_role"] == role])
              for role in ("novel", "revisit")}
    for role in groups:
        selected = [r for r in rows if r["analysis_role"] == role]
        groups[role].update(cec_takeover_queries=sum(r["cec_takeover_plans"] > 0 for r in selected),
            exact_native_queries=sum(r["no_takeover_exact_native"] for r in selected))
    return dict(complete=True, scope=SCOPE, population_sha256=population_sha,
        query_count=len(rows), query_arm_count=len(rows) * 3, source_accounting=sources,
        roles=groups, balanced_role_mixture=group_stats(rows), paired_records=rows,
        query_metrics_scope=population["query_metrics_scope"],
        interpretation="Equal Novel/Revisit query mix conditional on A and construction; not unconditional source joint SR",
        comparisons="GEM versus native and raw memory; role-specific and balanced-mixture estimates reported together",
        summarizer_sha256=sha(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--population-sha256", required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to overwrite a summary")
    result = summarize(args.population, args.population_sha256, args.evaluation_root)
    save(args.output, result)
    if not result["complete"]:
        raise SystemExit("Population incomplete; missing task list saved, no final SR")
    print(f"SEALED {result['query_count']} queries / {result['query_arm_count']} rollouts")
