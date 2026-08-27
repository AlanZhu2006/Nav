#!/usr/bin/env python3
"""Seal C-success factual prefixes before any B2 treatment is evaluated."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from lifelong_shared_c_contract import (
    POPULATION_SCHEMA,
    load_trace,
    require,
    sha256_file,
)


def finalize(
    *, source_population: Path, collection_root: Path, controller: str,
    run_root: Path, out: Path,
) -> dict:
    require(source_population.is_file(), "source population is missing")
    source = json.loads(source_population.read_text())
    rows = source.get("accepted")
    require(isinstance(rows, list) and rows, "source population is empty")
    for evaluation_name in ("evaluation", "shared_c_evaluation"):
        require(not (run_root / evaluation_name).exists(),
                "B2 evaluation exists before shared-C population freeze")
    require(not out.exists(), f"shared-C population exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    accepted = []
    attrition = []
    inputs = []
    try:
        for index, item in enumerate(rows):
            scene, episode = str(item["scene"]), str(item["episode"])
            label = f"{index:03d}_{scene}_{episode}"
            result = collection_root / label / "result"
            metric_path = result / "metric.csv"
            trace_path = result / f"{episode}_shared_C_trace.json"
            plans_path = result / f"{episode}_plans.json"
            summary_path = result / "summary.json"
            compute_path = collection_root / label / "compute_identity.json"
            for path in (metric_path, trace_path, plans_path, summary_path,
                         compute_path):
                require(path.is_file(), f"{label}: missing {path.name}")
            with metric_path.open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            require(len(metrics) == 1, f"{label}: metric row count changed")
            metric = metrics[0]
            trace = load_trace(trace_path)
            require(metric["scene"] == scene and metric["episode"] == episode,
                    f"{label}: metric identity changed")
            require(trace["scene"] == scene and trace["episode"] == episode,
                    f"{label}: trace identity changed")
            require(metric["controller"] == controller
                    and trace["controller"] == controller,
                    f"{label}: controller changed")
            require(trace["benchmark_sha256"] == item["benchmark_sha256"],
                    f"{label}: benchmark binding changed")
            require(metric["shared_C_trace_sha256"] == sha256_file(trace_path),
                    f"{label}: metric trace hash changed")
            require(int(metric["B2_outcomes_read"]) == 0
                    and trace["B2_navigation_outcomes_read"] is False,
                    f"{label}: B2 outcomes were read before freeze")
            input_receipt = {
                "source_population_index": index,
                "scene": scene,
                "episode": episode,
                "collection_run_root": str((collection_root / label).resolve()),
                "metric_sha256": sha256_file(metric_path),
                "trace_sha256": sha256_file(trace_path),
                "plans_sha256": sha256_file(plans_path),
                "summary_sha256": sha256_file(summary_path),
                "compute_identity_sha256": sha256_file(compute_path),
            }
            inputs.append(input_receipt)
            if not bool(trace["reached_C"]):
                attrition.append({
                    "source_population_index": index,
                    "scene": scene,
                    "episode": episode,
                    "reason": "factual_shared_C_failed",
                    "trace_sha256": sha256_file(trace_path),
                })
                continue
            trace_out = temporary / "traces" / scene / trace_path.name
            trace_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trace_path, trace_out)
            accepted.append({
                "population_index": len(accepted),
                "source_population_index": index,
                "scene": scene,
                "episode": episode,
                "controller": controller,
                "benchmark_sha256": item["benchmark_sha256"],
                "shared_C_trace": str(trace_out.relative_to(temporary)),
                "shared_C_trace_sha256": sha256_file(trace_out),
                "C_goal_start_frame": int(trace["C_goal_start_frame"]),
                "online_A_candidate_ceiling": int(
                    trace["online_A_candidate_ceiling"]),
                "online_B_candidate_ceiling": int(
                    trace["online_B_candidate_ceiling"]),
            })
        require(bool(accepted), "no factual shared-C prefix reached C")
        payload = {
            "schema_version": POPULATION_SCHEMA,
            "controller": controller,
            "source_population": str(source_population.resolve()),
            "source_population_sha256": sha256_file(source_population),
            "source_histories": len(rows),
            "accepted_histories": len(accepted),
            "scene_clusters": len({row["scene"] for row in accepted}),
            "selection_rule": "factual_C_success_only_before_any_B2_run",
            "selection_reads_B2_navigation_outcomes": False,
            "shared_C_replayed_identically_before_B2": True,
            "accepted": accepted,
            "attrition": attrition,
            "collection_inputs": inputs,
        }
        population_path = temporary / "population.json"
        population_path.write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (temporary / "population.json.sha256").write_text(
            sha256_file(population_path) + "  population.json\n")
        files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"FILES.sha256", "SEALED"}
        )
        with (temporary / "FILES.sha256").open("x") as handle:
            for path in files:
                handle.write(
                    f"{sha256_file(path)}  {path.relative_to(temporary)}\n")
        (temporary / "SEALED").write_text(
            "sealed before any B2 treatment navigation\n")
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-population", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize(
        source_population=args.source_population,
        collection_root=args.collection_root,
        controller=args.controller,
        run_root=args.run_root,
        out=args.out,
    )
    print(json.dumps({
        "controller": payload["controller"],
        "accepted_histories": payload["accepted_histories"],
        "scene_clusters": payload["scene_clusters"],
        "selection_reads_B2_navigation_outcomes": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
