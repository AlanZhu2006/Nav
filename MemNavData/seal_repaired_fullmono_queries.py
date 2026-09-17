"""Seal paired queries from a complete predefined prefix of new mono-A.

Only A collection and construction receipts are read. All legal histories in
the selected prefix are retained, including those beyond the target count.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from construct_repaired_fullmono_queries import SCHEMA as CONSTRUCTION_SCHEMA, read, require, save
from plan_repaired_fullmono_population import choose_prefix
from probe_repaired_fullmono_construction import sha

SCHEMA = "repaired_actual_fullmono_role_queries_20260909_v1"
ARMS = ("native", "raw_fixed", "cec")
SCOPE = "new actual repaired mono-A and paired queries on consumed HM3D source tasks"


def read_scene(root, plan_sha):
    summary_path, verifier_path = root / "summary.json", root / "independent_verification.json"
    summary, verifier = read(summary_path), read(verifier_path)
    inputs = read(root / "a_input_receipt.json")
    require(summary["schema"] == verifier["schema"] == CONSTRUCTION_SCHEMA,
            "Wrong scene construction schema")
    require(summary["completed"] and verifier["verified"] and summary["query_rollouts"] == 0
            and verifier["query_rollouts"] == 0, "Scene is incomplete or query outcomes were used")
    require(summary["source_plan_sha256"] == verifier["source_plan_sha256"] == plan_sha,
            "Scene used another source plan")
    require(sha(summary_path) == verifier["summary_sha256"], "Scene summary changed")
    require(len(summary["reports"]) == len(verifier["reports"]) == summary["source_count"],
            "Scene source count differs")
    require(all(dict(a, verified=True) == b for a, b in zip(summary["reports"], verifier["reports"])),
            "Scene verification does not cover the reported sources")
    collection = Path(inputs["collection"])
    require(sha(collection / "manifest.json") == inputs["manifest_sha256"]
            and sha(collection / "independent_verification.json") == inputs["verification_sha256"],
            "Actual A receipts changed")
    manifest = read(collection / "manifest.json")
    a_summary = read(collection / "summary.json")
    require(manifest == inputs["manifest"] and manifest["source_plan_sha256"] == plan_sha,
            "A input differs from the construction receipt")
    require(a_summary["completed"] and not a_summary["queries"], "Incomplete A source")
    require(len(a_summary["goal_a"]) == summary["source_count"], "Missing A outcome")
    source_indices = [r["source_index"] for r in summary["reports"]]
    require([r["source_index"] for r in a_summary["goal_a"]] == source_indices,
            "A/source order changed")
    binding = {"root": str(root), "summary_sha256": sha(summary_path),
        "verification_sha256": sha(verifier_path), "a_input_sha256": sha(root / "a_input_receipt.json"),
        "source_scene_rank": summary["source_scene_rank"], "source_indices": source_indices,
        "collection": str(collection), "a_summary_sha256": sha(collection / "summary.json"),
        "a_runtime_sha256": manifest["runtime_source_receipt"]["sha256"]}
    return verifier["reports"], a_summary["goal_a"], binding


def paired_tasks(selected, reports, bindings):
    """No score, confidence, success label or role is used for execution order."""
    by_source = {r["source_index"]: r for r in reports}
    owners = {index: binding for binding in bindings for index in binding["source_indices"]}
    tasks = []
    for index in selected:
        row = by_source[index]
        path = Path(row["construction"])
        require(sha(path) == row["construction_sha256"], "Selected construction changed")
        payload = read(path)
        require(payload["schema"] == CONSTRUCTION_SCHEMA and payload["pair_constructible"]
                and payload["completed"] and payload["navigation_rollouts"] == 0
                and not payload["query_outcomes_read"], "Not a complete query-free pair")
        require(payload["source_index"] == payload["history_index"] == index, "Wrong actual A identity")
        chosen = payload["selected_query_ids"]
        queries = [q for q in payload["queries"] if q["query_id"] in chosen]
        require(len(chosen) == len(queries) == 2 and len(set(chosen)) == 2
                and {q["analysis_role"] for q in queries} == {"novel", "revisit"}, "Incomplete role pair")
        online = Path(payload["online_a_episode"])
        require(sha(online / "online_a_trace.json") == payload["online_a_trace_sha256"]
                and sha(online / "receipt.json") == payload["online_a_receipt_sha256"], "Actual A history changed")
        for q in queries:
            for key in ("goal_rgb", "goal_depth"):
                require(sha(path.parent / q[key]) == q[key + "_sha256"], "Selected goal changed")
            tasks.append({"history_index": index, "query_id": q["query_id"],
                "construction": str(path), "construction_sha256": sha(path),
                "scene_construction_root": owners[index]["root"]})
    tasks.sort(key=lambda t: hashlib.sha256(
        f"repaired_fullmono_schedule_20260909/{t['history_index']}/{t['query_id']}".encode()).hexdigest())
    orders = list(itertools.permutations(ARMS))
    for index, task in enumerate(tasks):
        task.update(task_index=index, arm_order=list(orders[index % len(orders)]))
    return tasks


def seal(args):
    require(sha(args.plan) == args.plan_sha256, "Source plan changed")
    plan = read(args.plan)
    require(sha(args.runtime_receipt) == plan["base_runtime_sha256"], "Wrong runtime")
    roots = [root.resolve() for root in args.scenes]
    require(len(roots) == len(set(roots)), "Duplicate scene root")
    reports, a_rows, bindings = [], [], []
    for root in roots:
        r, a, binding = read_scene(root, args.plan_sha256)
        require(binding["a_runtime_sha256"] == plan["base_runtime_sha256"], "A used an old executor")
        reports.extend(r)
        a_rows.extend(a)
        bindings.append(binding)
    decision = choose_prefix(plan, reports)
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "construction_decision.json", decision)
    if decision["selected_prefix"] is None:
        return decision
    selected = decision["retained_source_indices"]
    if not selected:
        return dict(decision, state="empty_population", ready_for_query=False)
    tasks = paired_tasks(selected, reports, bindings)
    prefix = decision["selected_prefix"]
    expected = {r["source_index"] for r in plan["sources"] if r["scene_rank"] < prefix}
    covered_a = [r for r in a_rows if r["source_index"] in expected]
    require(len(covered_a) == len(expected), "Source A denominator differs")
    result = {"schema": SCHEMA, "scope": SCOPE, "phase": "sealed_before_query_evaluation",
        "source_plan": str(args.plan.resolve()), "source_plan_sha256": args.plan_sha256,
        "runtime_source_receipt": {"path": str(args.runtime_receipt.resolve()),
                                   "sha256": sha(args.runtime_receipt)},
        "evaluation_source_receipt": {"path": str(args.evaluation_receipt.resolve()),
                                      "sha256": sha(args.evaluation_receipt)},
        "construction_decision": decision,
        "source_scenes": [b for b in bindings if b["source_scene_rank"] < prefix],
        "source_waterfall": {"planned_A": len(expected), "completed_A": len(covered_a),
            "successful_A": sum(bool(r["reached"]) for r in covered_a),
            "constructible_role_pairs": len(selected), "pair_scene_clusters": decision["scenes"]},
        "histories": len(selected), "tasks": tasks, "query_count": len(tasks),
        "query_arm_count": len(tasks) * len(ARMS), "arms": list(ARMS),
        "query_navigation_rollouts_at_seal": 0,
        "query_metrics_scope": "conditional on successful A and legal paired query construction",
        "source_A_and_construction_attrition_reported_separately": True}
    save(args.out / "population.json", result)
    return {"population": str(args.out / "population.json"),
            "sha256": sha(args.out / "population.json"), "source_waterfall": result["source_waterfall"],
            "query_count": len(tasks), "query_arm_count": result["query_arm_count"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--scenes", type=Path, nargs="+", required=True)
    p.add_argument("--runtime-receipt", type=Path, required=True)
    p.add_argument("--evaluation-receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(seal(p.parse_args()), indent=2))
