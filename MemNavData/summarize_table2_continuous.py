"""Summarize a declared batch, retaining technical failures and construction attrition."""
import argparse
from collections import Counter
from pathlib import Path

from MemNavData.table2_mixed_local import load, dump, sha


def arm_counts(chain):
    observed = {r["stage"]: r["measurement"] for r in chain["measurements"]}
    cumulative, failed, complete = [], False, 0
    for stage in "ABC":
        if stage in observed:
            assert not failed
            reached = bool(observed[stage]["reached"])
            complete += int(reached)
            failed = not reached
            cumulative.append(int(not failed))
        elif failed:
            cumulative.append(0)
        else:
            assert chain["status"] == "task_construction_blocked"
            cumulative.append(None)
    return dict(observed=observed, cumulative=cumulative, completed_goals=complete)


def summarize(population, run, count):
    declared = load(population)
    if count < 1 or count > len(declared["tasks"]):
        raise ValueError("Invalid frozen batch size")
    expected = declared["tasks"][:count]
    rows, incomplete = [], []
    for task in expected:
        directory = run / "tasks" / f"{task['task_index']:03d}"
        receipt = directory / "archive_receipt.json"
        if not receipt.exists() or not load(receipt)["completed"]:
            incomplete.append(dict(task_index=task["task_index"], source_id=task["source_id"],
                                   status="missing_or_failed_runtime", navigation_failure_imputed=False))
            continue
        verified = load(directory / "independent_verification.json")
        summary = load(directory / "summary.json")
        assert verified["verified"] and verified["formal_population"]
        assert summary["population_sha256"] == sha(population)
        assert summary["task_index"] == task["task_index"] and summary["sequence"] == task["sequence"]
        rows.append(dict(task_index=task["task_index"], scene=task["scene"], sequence=task["sequence"],
            arms={a: arm_counts(summary["arms"][a]) for a in ("native", "cec")}))
    totals = {}
    for arm in ("native", "cec"):
        totals[arm] = {}
        for i, stage in enumerate("ABC"):
            values = [r["arms"][arm]["cumulative"][i] for r in rows]
            per_leg = [r["arms"][arm]["observed"][stage] for r in rows if stage in r["arms"][arm]["observed"]]
            lower = sum(v == 1 for v in values)
            unobserved = sum(v is None for v in values) + len(incomplete)
            totals[arm][stage] = dict(attempted=len(per_leg), reached=sum(bool(m["reached"]) for m in per_leg),
                cumulative_success_lower_count=lower, cumulative_success_upper_count=lower+unobserved,
                source_denominator=count, construction_unobserved=sum(v is None for v in values),
                technical_missing=len(incomplete))
    return dict(population_sha256=sha(population), declared_population=len(declared["tasks"]),
        batch_source_count=count, completed_verified_pairs=len(rows), incomplete=incomplete,
        technical_batch_complete=len(rows)==count,
        stage_coverage={stage: sum(stage in r["arms"][a]["observed"] for r in rows for a in ("native", "cec")) for stage in "ABC"},
        sequence_counts=dict(Counter(t["sequence"] for t in expected)), totals=totals, tasks=rows,
        release_remaining_automatically=False, success_rate_used_as_release_gate=False,
        note="Construction losses remain unobserved, not imputed navigation failures. First twelve are a subset of the frozen 97.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.population, a.run, a.count)
    if a.out.exists():
        raise FileExistsError("Use a new summary receipt; do not overwrite a previous result")
    dump(a.out, result)
    print({k:v for k,v in result.items() if k != "tasks"})
    raise SystemExit(0 if result["technical_batch_complete"] else 2)
