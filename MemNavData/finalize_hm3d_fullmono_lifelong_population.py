#!/usr/bin/env python3
"""Seal the factual A/B population before any C/B2/C2 evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from hm3d_fullmono_lifelong import PREFIX_SCHEMA, load_protocol, require, sha256_file


SCHEMA = "hm3d_fullmono_lifelong_population_v1_20260824"


def finalize(*, protocol_path: Path, ab_root: Path, fragments: Path, out: Path) -> dict:
    protocol = load_protocol(protocol_path)
    ab_manifest_path = ab_root / "role_pairs/manifest.json"
    ab_population_path = ab_root / "population_receipt.json"
    require((ab_root / "SEALED").is_file(), "A/B construction is not sealed")
    ab_population = json.loads(ab_population_path.read_text())
    require(sha256_file(ab_manifest_path)
            == ab_population["benchmark_manifest_sha256"],
            "A/B manifest changed")
    source = json.loads(ab_manifest_path.read_text())
    require(not out.exists(), f"lifelong population exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    accepted = []
    attrition = []
    try:
        benchmark_root = temporary / "benchmark"
        for index, item in enumerate(source["episodes"]):
            scene, episode = str(item["scene"]), str(item["episode"])
            label = f"{index:03d}_{scene}_{episode}"
            fragment = fragments / label
            completion_path = fragment / "completion.json"
            require(completion_path.is_file(), f"{label}: prefix fragment missing")
            completion = json.loads(completion_path.read_text())
            require(completion.get("status") == "complete", f"{label}: incomplete")
            require(completion.get("query_navigation_outcomes_read") is False,
                    f"{label}: prefix construction read query outcomes")
            require(completion["protocol_sha256"] == sha256_file(protocol_path),
                    f"{label}: protocol changed")
            require(completion["AB_manifest_sha256"] == sha256_file(ab_manifest_path),
                    f"{label}: A/B population changed")
            if not completion["eligible"]:
                attrition.append({
                    "history_index": index,
                    "scene": scene,
                    "episode": episode,
                    "reason": completion["attrition_reason"],
                    "completion_sha256": sha256_file(completion_path),
                })
                continue
            source_benchmark = fragment / "benchmark"
            benchmark_path = source_benchmark / "benchmark.json"
            require(sha256_file(benchmark_path) == completion["benchmark_sha256"],
                    f"{label}: benchmark changed")
            benchmark = json.loads(benchmark_path.read_text())
            require(benchmark.get("schema_version") == PREFIX_SCHEMA,
                    f"{label}: benchmark schema changed")
            require(benchmark["scene"] == scene and benchmark["episode"] == episode,
                    f"{label}: benchmark identity changed")
            destination = benchmark_root / scene / episode
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_benchmark, destination)
            accepted.append({
                "population_index": len(accepted),
                "source_AB_history_index": index,
                "scene": scene,
                "episode": episode,
                "benchmark": str((destination / "benchmark.json").relative_to(temporary)),
                "benchmark_sha256": sha256_file(destination / "benchmark.json"),
                "online_B_trace_sha256": benchmark["online_B_trace_sha256"],
                "B_goal_max_factual_B_covis": completion[
                    "B_goal_max_factual_B_covis"
                ],
                "B_goal_strong_support": completion["B_goal_strong_support"],
                "actual_B_end_to_C_geodesic_m": completion[
                    "actual_B_end_to_C_geodesic_m"
                ],
                "prefix_completion_sha256": sha256_file(completion_path),
            })
        scenes = {row["scene"] for row in accepted}
        target_histories = int(protocol["population"]["minimum_target_histories"])
        target_scenes = int(protocol["population"]["minimum_target_scene_clusters"])
        population = {
            "schema_version": SCHEMA,
            "scope": protocol["scope"],
            "protocol_sha256": sha256_file(protocol_path),
            "AB_population_receipt_sha256": sha256_file(ab_population_path),
            "AB_manifest_sha256": sha256_file(ab_manifest_path),
            "intention_to_collect_B": len(source["episodes"]),
            "supported_population": len(accepted),
            "scene_clusters": len(scenes),
            "strong_support_histories": sum(
                row["B_goal_strong_support"] for row in accepted
            ),
            "target_histories": target_histories,
            "target_scene_clusters": target_scenes,
            "target_met": len(accepted) >= target_histories
            and len(scenes) >= target_scenes,
            "underpowered": not (
                len(accepted) >= target_histories and len(scenes) >= target_scenes
            ),
            "selection_reads_C_B2_C2_navigation_outcomes": False,
            "runtime_role_visibility": "none",
            "accepted": accepted,
            "attrition": attrition,
        }
        population_path = temporary / "population.json"
        population_path.write_text(json.dumps(
            population, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "population.json.sha256").write_text(
            sha256_file(population_path) + "  population.json\n"
        )
        files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"POPULATION_FILES.sha256", "SEALED"}
        )
        with (temporary / "POPULATION_FILES.sha256").open("w") as handle:
            for path in files:
                handle.write(f"{sha256_file(path)}  {path.relative_to(temporary)}\n")
        (temporary / "SEALED").write_text(
            "sealed before C/B2/C2 navigation outcomes\n"
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ab-root", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        protocol_path=args.protocol,
        ab_root=args.ab_root,
        fragments=args.fragments,
        out=args.out,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "intention_to_collect_B", "supported_population", "scene_clusters",
            "strong_support_histories", "target_met", "underpowered",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
