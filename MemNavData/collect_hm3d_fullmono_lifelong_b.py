#!/usr/bin/env python3
"""Run and seal one actual-mono Novel-B factual rollout."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from deterministic_eval_protocol import validate_leg1_trace, write_leg1_trace
from final14_mono_factorial import audit_depth_plans
from hm3d_fullmono_lifelong import load_protocol, require, sha256_file


SCHEMA = "hm3d_fullmono_lifelong_b_collection_v1_20260824"


def run(command: list[str], log: Path) -> None:
    with log.open("x") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    require(result.returncode == 0,
            f"Novel-B evaluator failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    require(args.max_steps == int(protocol["factual_b_collection"]["maximum_steps"]),
            "formal B step budget changed")
    manifest_path = args.bench_root / "manifest.json"
    require(sha256_file(manifest_path) == args.expected_manifest_sha256,
            "sealed A/B manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(0 <= args.history_index < len(manifest["episodes"]),
            "history index outside sealed A/B population")
    item = manifest["episodes"][args.history_index]
    scene = str(item["scene"])
    episode = str(item["episode"])
    source = Path(item["online_a_episode"])
    receipt = json.loads((source / "receipt.json").read_text())
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file()
            and sha256_file(scene_file) == receipt["source_asset_sha256"],
            "explicit HM3D scene asset changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "factual_b" / label
    require(not output.exists(), f"Novel-B output exists: {output}")
    result_root = output / "result"
    logs = output / "logs"
    logs.mkdir(parents=True)
    command = [
        args.hab_python, "-u",
        str(args.source_root / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(args.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", "127.0.0.1",
        "--port", str(args.memnav_port),
        "--novel_port", str(args.navdp_port),
        "--server_backend", "hybrid_pose",
        "--success_dist", "1.0",
        "--max_steps", str(args.max_steps),
        "--exec_horizon", "8",
        "--trajectory_selector", "server",
        "--trajectory_selector_scope", "all",
        "--leg1_mode", "shared_trace",
        "--leg1_goal_source", "own",
        "--seed", "0",
        "--terminal_uturn", "off",
        "--terminal_visual_refine", "off",
        "--deterministic_plan_seeds",
        "--retrieval_override", "off",
        "--certified_cdec_rescue", "off",
        "--certified_stagnation_graph", "off",
        "--revisit_controller", "navdp_mixed",
        "--role_pair_scope", "consumed_integration",
        "--role_pair_query_role", "novel",
        "--hybrid_route", "native_sidecar",
        "--revisit_adapter", "legacy_metric",
        "--navdp_depth_source", "monocular_sidecar",
        "--out", str(result_root),
    ]
    run(command, logs / "eval.log")
    with (result_root / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1 and rows[0]["analysis_role"] == "novel",
            "Novel-B evaluator returned the wrong query population")
    row = rows[0]
    plan_files = list(result_root.glob(f"{episode}_*_plans.json"))
    require(len(plan_files) == 1, "Novel-B plans file is missing or ambiguous")
    plans = json.loads(plan_files[0].read_text())
    require(plans.get("analysis_role_not_forwarded") is True,
            "Novel-B role label leaked into runtime")
    trace = plans["query_trace_payload"]
    validate_leg1_trace(
        trace,
        expected_episode=episode,
        expected_seed=int(trace["episode_seed"]),
        expected_goal_sha256=next(
            query["goal_rgb_sha256"]
            for query in item["pairs"][0]["queries"]
            if query["analysis_role"] == "novel"
        ),
        expected_source_scene=scene,
    )
    require(trace["source_hybrid_route"] == "native_sidecar",
            "factual B was not controlled by native NavDP")
    depth_audit = audit_depth_plans("mono_native", trace["plans"])
    require(int(depth_audit["metric_sensor_plan_count"]) == 0,
            "factual B consumed metric depth")
    trace_path = output / f"{episode}_legB_trace.json"
    trace_sha = write_leg1_trace(trace_path, trace)
    completion = {
        "schema_version": SCHEMA,
        "status": "complete",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "protocol_sha256": sha256_file(args.protocol),
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "role_pair_sidecar_sha256": item["role_pairs_sha256"],
        "runtime_role_visible": False,
        "controller": "frozen_navdp_native_sidecar",
        "navdp_depth_source": "monocular_sidecar",
        "metric_depth_sensor_reads": 0,
        "reached_B": bool(trace["reached"]),
        "steps_B": int(trace["steps"]),
        "path_len_B_m": float(trace["path_len"]),
        "final_goal_dist_B_m": float(trace["final_goal_dist_m"]),
        "end_position": trace["end_position"],
        "end_yaw_rad": float(trace["end_yaw"]),
        "online_A_replay_frames": int(row["shared_A_frames"]),
        "online_A_rgb_hashes_verified": bool(int(row["shared_A_hashes_ok"])),
        "B_trace_sha256": trace_sha,
        "B_trace_path": str(trace_path.resolve()),
        "depth_audit": depth_audit,
        "result_metric_sha256": sha256_file(result_root / "metric.csv"),
        "result_plans_sha256": sha256_file(plan_files[0]),
    }
    completion_path = output / "completion.json"
    completion_path.write_text(json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    (output / "completion.json.sha256").write_text(
        sha256_file(completion_path) + "  completion.json\n"
    )
    print(json.dumps({
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "reached_B": completion["reached_B"],
        "steps_B": completion["steps_B"],
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
