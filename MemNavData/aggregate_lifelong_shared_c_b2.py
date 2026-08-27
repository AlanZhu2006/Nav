#!/usr/bin/env python3
"""Hash-bind raw shared-C B2 arms into one immutable aggregate input set."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from lifelong_shared_c_contract import ARMS, POPULATION_SCHEMA, require, sha256_file


SCHEMA = "lifelong_shared_c_b2_aggregate_v1_20260825"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    population = json.loads(args.population.read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "shared-C population schema changed")
    require(population.get("selection_reads_B2_navigation_outcomes") is False,
            "shared-C population was not frozen before B2")
    require(not args.out.exists(), f"aggregate output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.out.name + ".tmp.",
                                      dir=args.out.parent))
    arm_inputs = {arm: [] for arm in ARMS}
    try:
        for arm in ARMS:
            copied = temporary / arm
            copied.mkdir(parents=True)
            metrics = []
            for item in population["accepted"]:
                index = int(item["population_index"])
                scene, episode = item["scene"], item["episode"]
                label = f"{index:03d}_{scene}_{episode}"
                run = args.evaluation_root / label / arm
                result = run / "result"
                metric_path = result / "metric.csv"
                plans_path = result / f"{episode}_plans.json"
                summary_path = result / "summary.json"
                compute_path = run / "compute_identity.json"
                for path in (metric_path, plans_path, summary_path, compute_path):
                    require(path.is_file(), f"{label}/{arm}: missing {path.name}")
                with metric_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                require(len(rows) == 1, f"{label}/{arm}: metric rows changed")
                require(rows[0]["scene"] == scene
                        and rows[0]["episode"] == episode,
                        f"{label}/{arm}: identity changed")
                metrics.append(rows[0])
                arm_inputs[arm].append({
                    "population_index": index,
                    "scene": scene,
                    "episode": episode,
                    "run_root": str(run.resolve()),
                    "metric_sha256": sha256_file(metric_path),
                    "plans_sha256": sha256_file(plans_path),
                    "summary_sha256": sha256_file(summary_path),
                    "compute_identity_sha256": sha256_file(compute_path),
                })
            with (copied / "metric.csv").open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
        receipt = {
            "schema_version": SCHEMA,
            "population": str(args.population.resolve()),
            "population_sha256": sha256_file(args.population),
            "episodes": len(population["accepted"]),
            "scenes": len({row["scene"] for row in population["accepted"]}),
            "controller": population["controller"],
            "selection_reads_B2_navigation_outcomes": False,
            "arms": arm_inputs,
        }
        receipt_path = temporary / "aggregate_inputs.json"
        receipt_path.write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (temporary / "aggregate_inputs.json.sha256").write_text(
            sha256_file(receipt_path) + "  aggregate_inputs.json\n")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        with (temporary / "FILES.sha256").open("x") as handle:
            for path in files:
                handle.write(f"{sha256_file(path)}  {path.relative_to(temporary)}\n")
        temporary.replace(args.out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "controller": receipt["controller"],
        "episodes": receipt["episodes"],
        "scenes": receipt["scenes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
