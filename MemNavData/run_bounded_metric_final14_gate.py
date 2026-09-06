#!/usr/bin/env python3
"""Run one consumed Final14 Revisit under fixed and bounded CEC residuals.

This is an integration gate, not a paper-population SR experiment.  Both arms
replay the same sealed online-A history in one server pair and differ only in
the source-to-controller radius adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


SCHEMA = "cec_bounded_metric_final14_gate_v1_20260831"
ARMS = ("fixed_2p5m", "bounded_first40_metric")
ADAPTER = {
    "fixed_2p5m": "verified_bearing_v1",
    "bounded_first40_metric": "verified_bounded_metric_v1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> float:
    started = time.perf_counter()
    with log.open("x") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0,
            f"evaluator failed ({result.returncode}); see {log}")
    return time.perf_counter() - started


def read_arm(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    with (root / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, "gate must contain exactly one Revisit query")
    row = rows[0]
    require(row.get("analysis_role") == "revisit", "gate role changed")
    plans = list(root.glob("*_plans.json"))
    require(len(plans) == 1, "gate must contain exactly one plans artifact")
    return row, json.loads(plans[0].read_text())


def first_takeover(payload: dict[str, Any]) -> dict[str, Any]:
    accepted = [
        plan for plan in payload["query_leg"]
        if plan.get("revisit_adapter_takeover") is True
    ]
    require(bool(accepted), "certificate never authorized a controller input")
    return accepted[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    bench_root = Path(args.bench_root).resolve()
    manifest_path = bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Final14 manifest changed")
    histories = json.loads(manifest_path.read_text())["episodes"]
    require(0 <= args.history_index < len(histories),
            "history index outside Final14")
    item = histories[args.history_index]
    require(int(item["online_a_steps"]) >= 40,
            "history cannot freeze the first-40 scale")
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    receipt = json.loads((source_episode / "receipt.json").read_text())
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file(), "source scene missing")
    require(sha256(scene_file) == receipt["source_asset_sha256"],
            "source scene changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = run_root / "gate" / label
    require(not output.exists(), f"gate output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = ARMS if args.history_index % 2 == 0 else tuple(reversed(ARMS))
    contract = {
        "schema": SCHEMA,
        "scope": "consumed integration gate; no aggregate SR claim",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "online_a_steps": int(item["online_a_steps"]),
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "arms": list(ARMS),
        "arm_order": list(order),
        "same_server_pair": True,
        "role_visible_to_runtime": False,
        "depth_source": "monocular_sidecar",
        "max_steps": args.max_steps,
        "success_distance_m": 1.0,
    }
    (output / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n")

    common = [
        args.hab_python, "-u",
        str(source_root / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", "127.0.0.1",
        "--port", str(args.memnav_port),
        "--novel_port", str(args.navdp_port),
        "--server_backend", "hybrid_pose",
        "--hybrid_route", "certified_relocalization",
        "--navdp_depth_source", "monocular_sidecar",
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
        "--role_pair_query_role", "revisit",
    ]

    elapsed: dict[str, float] = {}
    rows: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for arm in order:
        arm_root = output / arm
        arm_root.mkdir()
        elapsed[arm] = run(
            common + [
                "--revisit_adapter", ADAPTER[arm],
                "--out", str(arm_root),
            ],
            output / "logs" / f"eval_{arm}.log",
        )
        rows[arm], payloads[arm] = read_arm(arm_root)
        require(payloads[arm].get("analysis_role_not_forwarded") is True,
                f"{arm}: role visibility contract failed")

    fixed_payload = payloads["fixed_2p5m"]
    bounded_payload = payloads["bounded_first40_metric"]
    require(fixed_payload["legA"] == bounded_payload["legA"],
            "paired arms changed sealed Goal-A plans")
    require(fixed_payload["rollout_traces"]["legA"] ==
            bounded_payload["rollout_traces"]["legA"],
            "paired arms changed Goal-A physical replay")
    require(fixed_payload["memory_traces"]["legA"] ==
            bounded_payload["memory_traces"]["legA"],
            "paired arms changed causal memory replay")

    fixed = first_takeover(fixed_payload)
    bounded = first_takeover(bounded_payload)
    for field in (
        "router_selected_anchor",
        "memory_unbounded_pointgoal",
        "memory_bearing_unit",
        "certified_relocalization_certificate",
    ):
        require(fixed.get(field) == bounded.get(field),
                f"first takeover proof drifted in {field}")
    require(abs(float(fixed["memory_controller_pointgoal_distance_m"])
                - 2.5) <= 1e-9,
            "canonical arm no longer emits 2.5 m")
    bounded_radius = float(
        bounded["memory_controller_pointgoal_distance_m"])
    require(0.0 < bounded_radius <= 2.5,
            "bounded arm escaped the frozen authority cap")
    require(float(bounded["memory_metric_scale_m_per_raw"]) > 0.0,
            "bounded arm omitted the first-40 metric scale")
    require(float(bounded["memory_pointgoal_radius_cap_m"]) == 2.5,
            "bounded arm cap drifted")

    completion = {
        **contract,
        "wall_time_seconds": elapsed,
        "prefix_equality": True,
        "outcomes": {arm: int(rows[arm]["reached"]) for arm in ARMS},
        "final_distance_m": {
            arm: float(rows[arm]["final_goal_dist_m"]) for arm in ARMS
        },
        "first_takeover": {
            "fixed_2p5m": {
                "anchor": fixed["router_selected_anchor"],
                "bearing": fixed["memory_bearing_unit"],
                "controller_distance_m": fixed[
                    "memory_controller_pointgoal_distance_m"],
            },
            "bounded_first40_metric": {
                "anchor": bounded["router_selected_anchor"],
                "bearing": bounded["memory_bearing_unit"],
                "metric_scale_m_per_raw": bounded[
                    "memory_metric_scale_m_per_raw"],
                "unbounded_metric_distance_m": bounded[
                    "memory_unbounded_pointgoal_distance_m"],
                "controller_distance_m": bounded[
                    "memory_controller_pointgoal_distance_m"],
                "radius_cap_m": bounded[
                    "memory_pointgoal_radius_cap_m"],
            },
        },
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    (output / "completion.json").write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n")
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "outcomes": completion["outcomes"],
        "first_takeover": completion["first_takeover"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
