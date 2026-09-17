"""Prespecified paired statistics; refuse to seal an incomplete population."""
import argparse
import math
from pathlib import Path

import numpy as np

from repaired_covisibility_eval import ARMS, POPULATION_SHA, SCOPE, dump, load, require, sha

BINS = ("c10_30", "c30_50", "c50_70", "c70_90", "c90_100")


def exact_mcnemar(gain, loss):
    n = gain + loss
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(gain, loss)+1)) / 2**n) if n else 1.


def holm(values):
    adjusted, maximum = [0.]*len(values), 0.
    for rank, i in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        maximum = max(maximum, min(1., (len(values)-rank)*values[i]))
        adjusted[i] = maximum
    return adjusted


def group_stats(tasks):
    require(bool(tasks), "empty analysis group")
    n = len(tasks)
    scenes = sorted({t["scene"] for t in tasks})
    result = {"queries": n, "scenes": len(scenes), "arms": {}, "comparisons": {}}
    for arm in ARMS:
        rows = [t["arms"][arm] for t in tasks]
        success = sum(r["reached"] for r in rows)
        result["arms"][arm] = {"successes": success, "sr": success/n, "spl": float(np.mean([r["spl"] for r in rows])),
            **{f"mean_{field}": float(np.mean([r[field] for r in rows]))
               for field in ("actual_path_m", "steps", "total_turn_deg", "wall_seconds")}}
    for baseline in ("native", "raw_fixed"):
        effects = [int(t["arms"]["cec"]["reached"])-int(t["arms"][baseline]["reached"]) for t in tasks]
        gain, loss = effects.count(1), effects.count(-1)
        totals = np.array([sum(e for t, e in zip(tasks, effects) if t["scene"] == s) for s in scenes])
        counts = np.array([sum(t["scene"] == s for t in tasks) for s in scenes])
        sampled = np.random.default_rng(20260909).integers(0, len(scenes), (20000, len(scenes)))
        estimates = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        result["comparisons"][baseline] = {"gain": gain, "loss": loss, "risk_difference": sum(effects)/n,
            "exact_mcnemar_p": exact_mcnemar(gain, loss),
            "scene_cluster_bootstrap_ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "bootstrap_replicates": 20000, "bootstrap_seed": 20260909}
    return result


def summarize(population_path, evaluation_root, output):
    require(sha(population_path) == POPULATION_SHA, "population changed")
    population = load(population_path)
    rows, missing = [], []
    for task in population["tasks"]:
        folder = evaluation_root / f"task_{task['task_index']:03d}"
        path = folder / "archive_receipt.json"
        if not path.exists():
            missing.append({"task_index": task["task_index"], "reason": "archive_receipt_missing"})
            continue
        archive = load(path)
        if not archive["completed"]:
            missing.append({"task_index": task["task_index"], "reason": "task_not_complete", "exit_code": archive["exit_code"]})
            continue
        verified, manifest = archive["verification"], load(folder / "manifest.json")
        require(verified["verified"] and verified["task"] == task
                and verified["population_sha256"] == POPULATION_SHA, "task verification mismatch")
        require(archive["all_member_hashes_readback_verified"] and sha(archive["archive"]) == archive["archive_sha256"],
                "durable raw archive changed")
        require(verified == load(folder / "independent_verification.json"), "verification copies differ")
        arms = {r["arm"]: r for r in verified["records"]}
        require(len(verified["records"]) == 3 and set(arms) == set(ARMS), "unpaired task")
        rows.append({"task_index": task["task_index"], "history_index": task["history_index"],
            "scene": manifest["scene"], "query_id": task["query_id"], "arms": arms,
            "cec_takeover_plans": verified["cec_takeover_plans"],
            "no_takeover_exact_native": verified["no_takeover_exact_native"]})
    if missing:
        dump(output, {"complete": False, "expected_queries": population["query_count"],
            "verified_queries": len(rows), "missing_or_failed_tasks": missing,
            "note": "Incomplete tasks are infrastructure attrition, not counted as navigation failures or a final SR."})
        raise RuntimeError(f"Incomplete population: {len(missing)} missing/failed tasks; see {output}")
    groups = {label: group_stats([r for r in rows if r["query_id"] == label]) for label in BINS}
    for baseline in ("native", "raw_fixed"):
        corrected = holm([groups[b]["comparisons"][baseline]["exact_mcnemar_p"] for b in BINS])
        for label, p in zip(BINS, corrected):
            groups[label]["comparisons"][baseline]["holm_five_bins_p"] = p
    common = set(population["complete_five_bin_history_indices"])
    novel = [r for r in rows if r["query_id"] == "natural_novel"]
    result = {"complete": True, "scope": SCOPE, "population_sha256": POPULATION_SHA,
        "query_count": len(rows), "query_arm_count": len(rows)*3, "bins": groups,
        "natural_novel_control": dict(group_stats(novel),
            cec_takeover_queries=sum(r["cec_takeover_plans"] > 0 for r in novel),
            exact_native_queries=sum(r["no_takeover_exact_native"] for r in novel),
            annotation_caveat="Original history 22 control has corrected q=0.11667666466706658; retained, not relabelled."),
        "common_five_bin_subset": {b: group_stats([r for r in rows if r["query_id"] == b and r["history_index"] in common]) for b in BINS},
        "construction": population["bins"], "paired_records": rows,
        "interpretation": "No pooled natural deployment SR; across-bin variation also changes goal yaw. All final motion/SPL and archived artifact hashes verified.",
        "summarizer_sha256": sha(__file__)}
    dump(output, result)
    print(f"SEALED {len(rows)} paired targets / {len(rows)*3} rollouts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to overwrite a summary")
    summarize(args.population, args.evaluation_root, args.output)
