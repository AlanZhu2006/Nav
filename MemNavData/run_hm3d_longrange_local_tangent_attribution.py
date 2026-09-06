#!/usr/bin/env python3
"""Run one sealed long-range history through three local-direction arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from MemNavData.final14_mono_factorial import audit_depth_plans
from MemNavData.longrange_local_tangent_attribution import (
    TANGENT_ALIGNMENT_THRESHOLD_DEG,
    TANGENT_ARMS,
    TANGENT_SCHEMA_VERSION,
    floor_aware_goal_distance_m,
)
from MemNavData.run_hm3d_action_coordinate_compass_pair import (
    first_target_proof,
)
from MemNavData.run_hm3d_fullmono_query_history import (
    audit_history_contract,
)
from MemNavData.run_hm3d_longrange_oracle_attribution import (
    compare_frozen_inputs,
    load_single_result,
    require,
    sha256,
)


SCHEMA_VERSION = "hm3d_longrange_local_tangent_episode_v2_20260903"
FROZEN_HISTORY_INDEX = 32


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(TANGENT_ARMS)
    return TANGENT_ARMS[offset:] + TANGENT_ARMS[:offset]


def run_command(command: list[str], log_path: Path, *, arm: str) -> float:
    environment = dict(os.environ)
    environment["HM3D_LONGRANGE_TANGENT_ARM"] = arm
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


def audit_tangent_plans(
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
    ), f"{arm}: every privileged readout requires a real accepted proof")
    oracle = [
        plan for plan in requests
        if plan.get("local_tangent_oracle_arm") is not None
    ]
    require(len(oracle) == len(requests),
            f"{arm}: an accepted request missed its v2 oracle receipt")
    require(all(
        plan.get("local_tangent_oracle_schema_version")
        == TANGENT_SCHEMA_VERSION
        and plan.get("local_tangent_oracle_arm") == arm
        and plan.get("local_tangent_oracle_evaluator_pose_consumed") is True
        and plan.get("local_tangent_oracle_goal_position_consumed") is True
        and plan.get("local_tangent_oracle_habitat_path_consumed") is True
        and plan.get("local_tangent_oracle_read_only_resample") is True
        and plan.get("local_tangent_oracle_realized_motion_accounting") is True
        and plan.get("local_tangent_oracle_success_distance_contract")
        == "floor_aware_3d_euclidean_v1"
        and abs(float(plan["local_tangent_oracle_pointgoal_radius_m"])
                - 2.5) <= 1e-6
        for plan in oracle
    ), f"{arm}: privileged v2 contract changed")

    headings = np.asarray([
        float(plan["local_tangent_oracle_signed_heading_deg"])
        for plan in oracle
    ])
    controllers = [
        str(plan["local_tangent_oracle_controller"]) for plan in oracle
    ]
    if arm == "oracle_chord_mixed_realized":
        require(all(
            plan.get("local_tangent_oracle_direction_source")
            == "planar_2p5m_geodesic_chord"
            and controller == "mixed_image_pointgoal"
            for plan, controller in zip(oracle, controllers)
        ), "chord arm changed geometry or controller")
    elif arm == "oracle_tangent_mixed_realized":
        require(all(
            plan.get("local_tangent_oracle_direction_source")
            == "first_traversable_geodesic_tangent"
            and controller == "mixed_image_pointgoal"
            for plan, controller in zip(oracle, controllers)
        ), "continuous tangent arm changed geometry or controller")
    else:
        require(all(
            plan.get("local_tangent_oracle_direction_source")
            == "first_traversable_geodesic_tangent"
            for plan in oracle
        ), "selective tangent arm stopped using the first local tangent")
        for plan, heading, controller in zip(oracle, headings, controllers):
            expected = (
                "mixed_image_pointgoal"
                if abs(float(heading)) > TANGENT_ALIGNMENT_THRESHOLD_DEG
                else "native_imagegoal"
            )
            require(controller == expected,
                    "selective tangent authorization violated its threshold")

    return {
        "request_count": len(requests),
        "certificate_accept_count": len(requests),
        "oracle_readout_count": len(oracle),
        "mixed_tangent_plan_count": controllers.count(
            "mixed_image_pointgoal"),
        "native_imagegoal_plan_count": controllers.count("native_imagegoal"),
        "median_abs_heading_deg": float(np.median(np.abs(headings))),
        "p90_abs_heading_deg": float(np.percentile(np.abs(headings), 90)),
        "floor_transition_plan_count": int(sum(
            abs(float(plan["local_tangent_oracle_goal_vertical_delta_m"]))
            >= 1.0 for plan in oracle)),
        "fixed_controller_radius_m": 2.5,
        "evaluator_pose_consumed": True,
    }


def audit_realized_motion(
    payload: dict[str, Any], row: dict[str, str],
) -> dict[str, Any]:
    trace = payload["rollout_traces"]["query"]
    require(trace, "query rollout trace is empty")
    positions = np.asarray([
        [frame["x"], frame["y"], frame["z"]] for frame in trace
    ], dtype=np.float64)
    receipts = np.asarray([
        float(frame["executed_translation_m_since_previous_frame"])
        for frame in trace
    ], dtype=np.float64)
    realized_between = np.linalg.norm(
        np.diff(positions[:, [0, 2]], axis=0), axis=1)
    require(abs(receipts[0]) <= 1e-9,
            "first rollout observation has a nonzero prior motion receipt")
    require(np.allclose(receipts[1:], realized_between, atol=1e-7, rtol=0.0),
            "frame-bound motion receipts differ from realized snapped motion")
    end = np.asarray(payload["query_result"]["end_position"], dtype=np.float64)
    final_delta = float(np.linalg.norm(end[[0, 2]] - positions[-1, [0, 2]]))
    realized_total = float(realized_between.sum() + final_delta)
    reported_total = float(row["path_len_m"])
    require(abs(realized_total - reported_total) <= 1e-6,
            "reported path length is not realized post-snap motion")

    first_plan = next(
        plan for plan in payload["query_leg"]
        if plan.get("local_tangent_oracle_goal_position_xyz") is not None
    )
    goal = np.asarray(
        first_plan["local_tangent_oracle_goal_position_xyz"],
        dtype=np.float64,
    )
    distance_3d, planar, vertical = floor_aware_goal_distance_m(end, goal)
    require(abs(distance_3d - float(row["final_goal_dist_m"])) <= 1e-6,
            "reported final distance is not floor-aware 3-D distance")
    reached = bool(int(row["reached"]))
    require(reached == (distance_3d < 1.0),
            "success bit violates floor-aware 1 m distance contract")
    return {
        "reported_path_m": reported_total,
        "recomputed_realized_path_m": realized_total,
        "zero_translation_receipt_count": int(np.sum(receipts <= 1e-12)),
        "nonzero_translation_receipt_count": int(np.sum(receipts > 1e-12)),
        "final_distance_3d_m": distance_3d,
        "final_distance_planar_m": planar,
        "final_vertical_error_m": vertical,
        "floor_aware_success": reached,
    }


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

    require(arguments.history_index == FROZEN_HISTORY_INDEX,
            "v2 is frozen to the one already-consumed failure")
    require(arguments.history_contract == "causal_survey",
            "v2 requires the causal-survey history")
    require(arguments.role_pair_scope == "table3_longrange_oracle",
            "v2 requires its explicit consumed scope")
    require(600 <= arguments.max_steps <= 3400,
            "query budget lies outside the frozen 600..3400 range")
    manifest_path = arguments.bench_root / "manifest.json"
    require(sha256(manifest_path) == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(manifest["episodes"]) == 48,
            "sealed length population must contain 48 histories")
    item = manifest["episodes"][arguments.history_index]
    require(item["bin_name"] == "30_to_50_m",
            "v2 received a non-long-range history")

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
              "longrange_local_tangent_attribution" / label)
    require(not output.exists(), f"v2 output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = rotated_arm_order(arguments.history_index)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "consumed one-failure mechanism diagnosis only",
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
        "arms": list(TANGENT_ARMS),
        "arm_order": list(order),
        "query_role": "revisit",
        "max_steps": arguments.max_steps,
        "success_distance_m": 1.0,
        "success_distance_contract": "floor_aware_3d_euclidean_v1",
        "path_length_contract": "post_snap_realized_planar_motion_v1",
        "exec_horizon": 8,
        "pointgoal_radius_m": 2.5,
        "tangent_min_planar_m": 0.30,
        "alignment_threshold_deg": TANGENT_ALIGNMENT_THRESHOLD_DEG,
        "privileged_inputs": [
            "current evaluator pose",
            "query goal 3-D position",
            "current Habitat shortest path",
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
            "MemNavData/eval_hm3d_longrange_local_tangent_attribution.py"),
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
        "--certified_guidance_mode", "endpoint_bearing",
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
        command = common + ["--out", str(arm_root)]
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
        rows[arm] = row
        payloads[arm] = payload
        audits[arm] = {
            "depth": audit_depth_plans("mono_cec", plans),
            "target_proof": first_target_proof(plans),
            "direction": audit_tangent_plans(arm, plans),
            "motion_and_success": audit_realized_motion(payload, row),
        }

    reference_arm = "oracle_chord_mixed_realized"
    for arm in TANGENT_ARMS:
        compare_frozen_inputs(
            rows[reference_arm], payloads[reference_arm],
            rows[arm], payloads[arm], arm)
        require(audits[arm]["target_proof"]
                == audits[reference_arm]["target_proof"],
                f"{arm}: target proposal/certificate proof changed")

    def metric(field: str, cast) -> dict[str, Any]:
        return {arm: cast(rows[arm][field]) for arm in TANGENT_ARMS}

    completion = {
        **contract,
        "prefix_equality": True,
        "target_proof_equality": True,
        "wall_time_seconds": elapsed,
        "outcomes": metric("reached", int),
        "final_distance_3d_m": metric("final_goal_dist_m", float),
        "initial_geodesic_m": metric("geodesic_m", float),
        "realized_path_len_m": metric("path_len_m", float),
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
