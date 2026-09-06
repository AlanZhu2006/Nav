#!/usr/bin/env python3
"""Run one sealed long-range history through four attribution arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from MemNavData.final14_mono_factorial import audit_depth_plans
from MemNavData.longrange_oracle_attribution import (
    ORACLE_ARMS,
    ORACLE_SCHEMA_VERSION,
)
from MemNavData.run_hm3d_action_coordinate_compass_gate import (
    audit_action_coordinate_plans,
)
from MemNavData.run_hm3d_action_coordinate_compass_pair import (
    first_target_proof,
)
from MemNavData.run_hm3d_fullmono_query_history import (
    audit_history_contract,
)


SCHEMA_VERSION = "hm3d_longrange_oracle_attribution_episode_v1_20260903"
FROZEN_INDICES = tuple(range(32, 48))
ARM_GUIDANCE = {
    "action_coordinate_mixed": "action_coordinate_compass",
    "oracle_route_mixed": "endpoint_bearing",
    "oracle_geodesic_mixed": "endpoint_bearing",
    "oracle_geodesic_point": "endpoint_bearing",
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


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(ORACLE_ARMS)
    return ORACLE_ARMS[offset:] + ORACLE_ARMS[:offset]


def run_command(
    command: list[str], log_path: Path, *, arm: str,
) -> float:
    environment = dict(os.environ)
    environment["HM3D_LONGRANGE_ATTRIBUTION_ARM"] = arm
    start = time.perf_counter()
    with log_path.open("x") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    elapsed = time.perf_counter() - start
    require(result.returncode == 0,
            f"{arm} evaluator failed ({result.returncode}); see {log_path}")
    return elapsed


def load_single_result(
    root: Path, episode: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    with (root / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1 and rows[0]["analysis_role"] == "revisit",
            "oracle attribution must emit one Revisit row")
    row = rows[0]
    path = root / f"{episode}_{row['query_id']}_plans.json"
    require(path.is_file(), f"query plan payload is missing: {path}")
    return row, json.loads(path.read_text(encoding="utf-8"))


def compare_frozen_inputs(
    reference_row: dict[str, str], reference: dict[str, Any],
    row: dict[str, str], payload: dict[str, Any], arm: str,
) -> None:
    for field in (
        "scene", "episode", "pair_id", "query_id", "analysis_role", "seed",
        "shared_A_frames", "shared_A_decision_frames", "geodesic_m",
    ):
        require(reference_row[field] == row[field],
                f"{arm}: paired field {field} changed")
    require(reference["legA"] == payload["legA"],
            f"{arm}: frozen Goal-A plans changed")
    require(reference["rollout_traces"]["legA"]
            == payload["rollout_traces"]["legA"],
            f"{arm}: frozen Goal-A physical trace changed")
    require(reference["memory_traces"]["legA"]
            == payload["memory_traces"]["legA"],
            f"{arm}: replayed causal memory changed")
    for field in (
        "all_rgb_hashes_verified", "decision_frames", "decision_steps",
        "diffusion_samples_during_replay", "navdp_memory_size",
        "navdp_queue_lengths", "online_frames",
    ):
        require(reference["replay"][field] == payload["replay"][field],
                f"{arm}: replay field {field} changed")


def audit_oracle_plans(
    arm: str, plans: list[dict[str, Any]],
) -> dict[str, Any]:
    requests = [
        plan for plan in plans
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(requests and all(
        plan.get("certified_relocalization_ok") is True
        and plan.get("certified_relocalization_accepted") is True
        for plan in requests
    ), f"{arm}: every attribution decision requires a real accepted proof")
    oracle = [
        plan for plan in requests
        if plan.get("action_coordinate_oracle_arm") is not None
    ]
    require(len(oracle) == len(requests),
            f"{arm}: an accepted request missed its oracle receipt")
    require(all(
        plan.get("action_coordinate_oracle_schema_version")
        == ORACLE_SCHEMA_VERSION
        and plan.get("action_coordinate_oracle_arm") == arm
        and plan.get("action_coordinate_oracle_evaluator_pose_consumed") is True
        and plan.get("action_coordinate_oracle_goal_position_consumed") is True
        and plan.get("action_coordinate_oracle_read_only_resample") is True
        and abs(math.hypot(*plan["action_coordinate_oracle_pointgoal"])
                - 2.5) <= 1e-6
        for plan in oracle
    ), f"{arm}: oracle contract changed")
    expected_controller = (
        "pure_pointgoal" if arm == "oracle_geodesic_point"
        else "mixed_image_pointgoal")
    require(all(
        plan.get("action_coordinate_oracle_controller") == expected_controller
        for plan in oracle
    ), f"{arm}: controller conditioning changed")
    result = {
        "request_count": len(requests),
        "certificate_accept_count": len(requests),
        "oracle_readout_count": len(oracle),
        "controller": expected_controller,
        "fixed_controller_radius_m": 2.5,
        "evaluator_pose_consumed": True,
    }
    if arm == "oracle_route_mixed":
        progress = [
            float(plan["action_coordinate_oracle_progress_m"])
            for plan in oracle
        ]
        require(all(current + 1e-9 >= previous
                    for previous, current in zip(progress, progress[1:])),
                "oracle route progress regressed")
        result.update({
            "historical_route_consumed": True,
            "initial_progress_m": progress[0],
            "final_progress_m": progress[-1],
            "max_cross_track_m": max(float(
                plan["action_coordinate_oracle_cross_track_m"])
                for plan in oracle),
        })
    else:
        require(all(
            float(plan["action_coordinate_oracle_current_geodesic_m"]) > 0.0
            for plan in oracle
        ), f"{arm}: current geodesic receipt is invalid")
        result["current_geodesic_replanned_each_decision"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluator-source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--history-contract", required=True)
    parser.add_argument("--role-pair-scope", required=True)
    arguments = parser.parse_args()

    require(arguments.history_contract == "causal_survey",
            "oracle attribution requires the causal-survey history")
    require(arguments.role_pair_scope == "table3_longrange_oracle",
            "oracle runner requires its explicit consumed scope")
    require(arguments.history_index in FROZEN_INDICES,
            "oracle attribution is frozen to the 16 long-range histories")
    require(600 <= arguments.max_steps <= 3400,
            "frozen query budget lies outside 600..3400")
    manifest_path = arguments.bench_root / "manifest.json"
    require(sha256(manifest_path) == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(manifest["episodes"]) == 48,
            "sealed length population must contain 48 histories")
    item = manifest["episodes"][arguments.history_index]
    require(item["bin_name"] == "30_to_50_m",
            "oracle attribution received a non-long-range history")

    source_episode = Path(item["online_a_episode"])
    require(sha256(source_episode / "receipt.json")
            == item["online_a_receipt_sha256"],
            "causal-history receipt changed")
    require(sha256(source_episode / "online_a_trace.json")
            == item["online_a_trace_sha256"],
            "causal-history trace changed")
    receipt = json.loads(
        (source_episode / "receipt.json").read_text(encoding="utf-8"))
    trace = json.loads(
        (source_episode / "online_a_trace.json").read_text(encoding="utf-8"))
    require(all(plan.get("navdp_depth_source") == "monocular_sidecar"
                and plan.get("metric_depth_sensor_consumed") is False
                for plan in trace["plans"]),
            "causal history is not full-monocular")
    prefix_a_steps, prefix_b_steps, history_policy = audit_history_contract(
        receipt, trace, arguments.history_contract)
    require(prefix_a_steps >= 40 and prefix_b_steps == 0,
            "causal survey cannot establish the first-40 receipt")

    scene = str(item["scene"])
    episode = str(item["episode"])
    scene_file = Path(receipt["source_asset"])
    navmesh = Path(item["runtime_navmesh"])
    require(scene_file.is_file()
            and sha256(scene_file) == receipt["source_asset_sha256"],
            "HM3D scene asset changed")
    require(item.get("runtime_geometry")
            == "content_addressed_pinned_navmesh"
            and navmesh.is_file()
            and sha256(navmesh) == item["runtime_navmesh_sha256"],
            "pinned runtime navmesh changed")

    label = f"{arguments.history_index:03d}_{scene}_{episode}"
    output = (arguments.run_root / "development" /
              "longrange_oracle_attribution" / label)
    require(not output.exists(), f"oracle output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = rotated_arm_order(arguments.history_index)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "consumed-development attribution only",
        "paper_result_eligible": False,
        "history_index": arguments.history_index,
        "scene": scene,
        "episode": episode,
        "distance_bin": item["bin_name"],
        "online_a_steps": int(item["online_a_steps"]),
        "history_policy": history_policy,
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "runtime_navmesh_sha256": item["runtime_navmesh_sha256"],
        "arms": list(ORACLE_ARMS),
        "arm_order": list(order),
        "query_role": "revisit",
        "max_steps": arguments.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "pointgoal_radius_m": 2.5,
        "privileged_inputs": [
            "current evaluator pose",
            "query goal position",
            "Habitat shortest path for geodesic arms and route tail",
        ],
        "navigation_outcomes_read_after_all_arms": True,
    }
    (output / "episode_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    common = [
        arguments.hab_python, "-u",
        str(arguments.evaluator_source_root /
            "MemNavData/eval_hm3d_longrange_oracle_attribution.py"),
        "--episode_root", str(arguments.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", "127.0.0.1",
        "--port", str(arguments.memnav_port),
        "--novel_port", str(arguments.navdp_port),
        "--server_backend", "hybrid_pose",
        "--hybrid_route", "certified_relocalization",
        "--revisit_adapter", "verified_bearing_v1",
        "--success_dist", "1.0",
        "--max_steps", str(arguments.max_steps),
        "--stuck_window", str(arguments.max_steps + 1),
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
        "--role_pair_scope", "table3_longrange_oracle",
        "--role_pair_query_role", "revisit",
        "--navdp_depth_source", "monocular_sidecar",
        "--pinned_navmesh", str(navmesh),
        "--expected_pinned_navmesh_sha256",
        item["runtime_navmesh_sha256"],
    ]

    rows: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    elapsed: dict[str, float] = {}
    for arm in order:
        arm_root = output / arm
        arm_root.mkdir()
        command = common + [
            "--certified_guidance_mode", ARM_GUIDANCE[arm],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, output / "logs" / f"eval_{arm}.log", arm=arm)
        summary = json.loads(
            (arm_root / "summary.json").read_text(encoding="utf-8"))
        require(summary.get("queries") == 1
                and summary.get("role_counts") == {"novel": 0, "revisit": 1}
                and summary.get("arm") == "certified"
                and summary.get("role_pair_scope")
                == "table3_longrange_oracle",
                f"{arm}: evaluator population/arm changed")
        row, payload = load_single_result(arm_root, episode)
        require(int(row.get("runtime_failure_plans", -1)) == 0,
                f"{arm}: runtime failure is not a navigation outcome")
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{arm}: role-hiding receipt missing")
        plans = payload["query_leg"]
        depth = audit_depth_plans("mono_cec", plans)
        proof = first_target_proof(plans)
        route = (
            audit_action_coordinate_plans(plans)
            if arm == "action_coordinate_mixed"
            else audit_oracle_plans(arm, plans)
        )
        rows[arm] = row
        payloads[arm] = payload
        audits[arm] = {"depth": depth, "target_proof": proof, "route": route}

    reference_arm = "action_coordinate_mixed"
    for arm in ORACLE_ARMS:
        compare_frozen_inputs(
            rows[reference_arm], payloads[reference_arm],
            rows[arm], payloads[arm], arm)
        require(audits[arm]["target_proof"]
                == audits[reference_arm]["target_proof"],
                f"{arm}: target proposal/certificate proof changed")

    def metric(field: str, cast) -> dict[str, Any]:
        return {arm: cast(rows[arm][field]) for arm in ORACLE_ARMS}

    completion = {
        **contract,
        "prefix_equality": True,
        "target_proof_equality": True,
        "wall_time_seconds": elapsed,
        "outcomes": metric("reached", int),
        "final_distance_m": metric("final_goal_dist_m", float),
        "geodesic_m": metric("geodesic_m", float),
        "path_len_m": metric("path_len_m", float),
        "query_steps": metric("steps", int),
        "certificate_accept_plans": metric(
            "certificate_accept_plans", int),
        "runtime_audits": audits,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    (output / "completion.json").write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": arguments.history_index,
        "outcomes": completion["outcomes"],
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}", file=sys.stderr)
        raise
