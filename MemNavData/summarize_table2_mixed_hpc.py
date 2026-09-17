"""Summarize the finite integration cohort, retaining all construction losses."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from MemNavData.table2_mixed_hpc import (
    validate_plan, read_completed, population_at, durable_root, load, dump, sha, formal,
)
from MemNavData.summarize_table1_repaired import statistics


def summarize(plan_path, run, out):
    plan = validate_plan(load(plan_path))
    population = population_at(plan_path, run)
    n = len(plan["sources"])
    stages = {"A": n, "B": 2*n, "C": population["c_query_tasks"]}
    rows, a_rows, attrition, receipts = [], [], [], []
    for stage, count in stages.items():
        for index in range(count):
            summary = read_completed(run, stage, index, sha(plan_path))
            folder = durable_root(run, stage, index)
            receipt = load(folder/"archive_receipt.json")
            verifier = load(folder/"independent_verification.json")
            if receipt["verification"] != verifier or sha(receipt["archive"]) != receipt["archive_sha256"]:
                raise ValueError("Verifier/archive changed")
            receipts.append(dict(stage=stage, index=index,
                summary_sha256=sha(folder/"summary.json"), archive_sha256=receipt["archive_sha256"]))
            attrition.extend(dict(stage=stage, source_id=summary["source_id"], reason=r)
                             for r in summary["attrition"])
            if stage == "A":
                a_rows.extend(summary["records"])
                continue
            for pair in verifier["pairs"]:
                arms = {r["arm"]: r for r in summary["records"]}
                assert set(arms) == {"native", "cec"}
                rows.append(dict(scene=arms["native"]["scene"], source_id=summary["source_id"],
                    stage=stage, phase=pair["phase"], role=pair["role"], pair=pair, **arms))
    groups = {}
    for stage in ("B", "C"):
        groups[stage] = {}
        for role in ("novel", "revisit"):
            selected = [r for r in rows if r["stage"] == stage and r["role"] == role]
            groups[stage][role] = statistics(selected) if selected else None
    c_strata = {}
    for before in ("novel", "revisit"):
        c_strata[before] = {}
        for role in ("novel", "revisit"):
            selected = [r for r in rows if r["phase"] == "C_after_"+before and r["role"] == role]
            c_strata[before][role] = statistics(selected) if selected else None
    supply = {}
    for stage in ("A", "B", "C"):
        for role in ("novel", "revisit"):
            records = a_rows if stage == "A" and role == "novel" else [
                r["native"] for r in rows if r["stage"] == stage and r["role"] == role]
            if stage == "A" and role == "revisit":
                records = []
            distances = [r["geodesic_m"] for r in records]
            supply[f"{stage}_{role}"] = dict(queries=len(records),
                mean_geodesic_m=float(np.mean(distances)) if distances else None,
                directions=dict(Counter(r["geometry"]["direction_stratum"] for r in records)),
                support_sources=dict(Counter(r["geometry"]["support_source"] for r in records)))
            if formal(plan):
                from MemNavData.table2_balanced_sampling import cell_of
                supply[f"{stage}_{role}"]["distance_direction_cells"] = dict(Counter(
                    cell_of(dict(geodesic_m=r["geodesic_m"],direction_stratum=r["geometry"]["direction_stratum"])) for r in records))
                supply[f"{stage}_{role}"]["current_view_covis_max"] = max(
                    (r["geometry"]["current_view_covis"] for r in records),default=None)
    dump(out, dict(completed=True, formal_result=formal(plan), plan_sha256=sha(plan_path),
        source_scenes=len({s["scene"] for s in plan["sources"]}), source_budget=n,
        A=dict(declared_sources=n, constructed=len(a_rows), successes=sum(r["reached"] for r in a_rows),
               mean_spl=float(np.mean([r["spl"] for r in a_rows])) if a_rows else None),
        groups=groups, C_by_native_B_role=c_strata, C_reference_mix=population["counts"],
        attrition=attrition, supply=supply, paired_rows=rows, tasks=receipts,
        interpretation=("Stage-conditional paired queries on actual mono prefixes, not autonomous joint SR; "
            "menus balanced as far as supply allows; inspect actual covariate distributions."
            if formal(plan) else "Interface/supply pilot; stage-conditional paired queries, not autonomous joint SR; no formal matching claimed.")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    summarize(a.plan, a.run, a.out)
