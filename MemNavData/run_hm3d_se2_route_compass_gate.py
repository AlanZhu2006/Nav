#!/usr/bin/env python3
"""Run an SR-hidden runtime gate for the local-SE(2) route compass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from MemNavData.run_hm3d_action_coordinate_compass_gate import (
    FROZEN_DISTANCE_BIN,
    FROZEN_HISTORY_INDEX,
    QUERY_STEP_GUARD,
    _finite_pair,
    require,
    sha256,
)


SCHEMA_VERSION = "hm3d_se2_route_compass_gate_v1_20260902"
SE2_SCHEMA_VERSION = "se2_projected_route_compass_v2_20260902"
SE2_RECEIPT_CONTRACT = "frame_bound_local_se2_v1"
SE2_ODOMETRY_SOURCE = "habitat_pose_difference_odometry_proxy_v1"


def audit_se2_route_plans(
    plans: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate information and motion contracts without reading outcomes."""

    require(bool(plans), "query emitted no planning receipts")
    require(all(plan.get("role_label_visible") is not True for plan in plans),
            "runtime role label became visible")
    requests = [
        plan for plan in plans
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(requests, "query emitted no certificate request")
    require(all(plan.get("certified_relocalization_ok") is True
                for plan in requests),
            "certificate endpoint had a runtime failure")
    accepted = [
        plan for plan in requests
        if plan.get("certified_relocalization_accepted") is True
    ]
    require(accepted, "prospective Revisit was never certificate-authorized")
    require(all(
        plan.get("certified_relocalization_guidance_mode")
        == "se2_route_compass" for plan in accepted
    ), "accepted request used another guidance mode")
    require(all(
        plan.get("se2_route_compass_status") in {"active", "terminal_tangent"}
        and plan.get("se2_route_compass_error") is None
        and plan.get("se2_route_compass_error_type") is None
        and plan.get("se2_route_schema_version") == SE2_SCHEMA_VERSION
        for plan in accepted
    ), "accepted request lacks a valid SE(2) route readout")
    require(all(
        plan.get("se2_route_evaluator_pose_consumed_as_odometry_proxy") is True
        and plan.get("se2_route_world_pose_consumed_by_policy") is False
        and plan.get("se2_route_executor_odometry_required") is True
        and plan.get("se2_route_executor_odometry_source")
        == SE2_ODOMETRY_SOURCE
        and plan.get("se2_route_habitat_path_consumed") is False
        and plan.get("se2_route_lingbot_translation_consumed") is False
        and plan.get("se2_route_metric_scale_consumed") is False
        for plan in accepted
    ), "SE(2) readout consumed forbidden global or metric geometry")
    require(all(
        plan.get("se2_route_visual_gate_present_after_initialization") is False
        and plan.get("se2_route_distance_regime_present") is False
        and plan.get("se2_route_endpoint_fallback_available") is False
        and plan.get("se2_route_native_fallback_available") is False
        for plan in accepted
    ), "SE(2) readout exposed a gate, distance regime, or fallback")

    progress = [float(plan["se2_route_projected_progress_m"])
                for plan in accepted]
    fractions = [float(plan["se2_route_progress_fraction"])
                 for plan in accepted]
    path_lengths = [float(plan["se2_route_query_path_length_m"])
                    for plan in accepted]
    cross_tracks = [float(plan["se2_route_cross_track_error_m"])
                    for plan in accepted]
    extents = {float(plan["se2_route_extent_m"]) for plan in accepted}
    require(all(math.isfinite(value) for value in (
        *progress, *fractions, *path_lengths, *cross_tracks, *extents,
    )), "SE(2) route receipt contains a non-finite scalar")
    require(all(current + 1e-9 >= previous
                for previous, current in zip(progress, progress[1:])),
            "projected route progress regressed")
    require(all(current + 1e-9 >= previous
                for previous, current in zip(path_lengths, path_lengths[1:])),
            "query path length regressed")
    require(all(0.0 <= value <= 1.0 + 1e-9 for value in fractions),
            "route fraction escaped [0,1]")
    require(all(value >= -1e-9 for value in cross_tracks),
            "cross-track distance became negative")
    require(len(extents) == 1 and next(iter(extents)) > 0.0,
            "certified route extent changed or is empty")
    require(path_lengths[-1] > 1e-6,
            "runtime gate never exercised translational motion")

    for name in (
            "se2_route_estimated_position", "se2_route_projected_position",
            "se2_route_reference_position"):
        for plan in accepted:
            _finite_pair(plan.get(name), name)
    bearings = [
        _finite_pair(plan.get("se2_route_unit_bearing"), "unit bearing")
        for plan in accepted
    ]
    goals = [
        _finite_pair(plan.get("se2_route_controller_pointgoal"),
                     "controller PointGoal")
        for plan in accepted
    ]
    require(all(abs(math.hypot(*value) - 1.0) <= 1e-6
                for value in bearings),
            "SE(2) output is not a unit bearing")
    require(all(abs(math.hypot(*value) - 2.5) <= 1e-6
                for value in goals),
            "SE(2) controller payload changed fixed radius")
    require(all(abs(float(plan[
        "memory_controller_pointgoal_distance_m"]) - 2.5) <= 1e-6
                for plan in accepted),
            "NavDP did not receive the fixed 2.5 m residual")

    history_hashes = {
        str(plan["se2_route_history_receipt_sha256"]) for plan in accepted
    }
    history_counts = {
        int(plan["se2_route_history_receipt_count"]) for plan in accepted
    }
    goal_starts = {int(plan["se2_route_goal_start_frame"])
                   for plan in accepted}
    anchors = {int(plan["se2_route_target_anchor"])
               for plan in accepted}
    require(len(history_hashes) == len(history_counts) == 1,
            "historical local receipt changed during the query")
    require(len(goal_starts) == len(anchors) == 1,
            "proof-bound route endpoints changed during the query")

    append_receipts = [
        plan.get("executor_motion_receipt") for plan in plans
        if plan.get("executor_motion_receipt") is not None
    ]
    require(append_receipts, "frame-bound executor receipts were not logged")
    local_receipts = [receipt.get("local_se2")
                      for receipt in append_receipts]
    require(all(
        isinstance(receipt, dict)
        and receipt.get("contract") == SE2_RECEIPT_CONTRACT
        and receipt.get("source") == SE2_ODOMETRY_SOURCE
        and math.isfinite(float(receipt["executed_forward_m"]))
        and math.isfinite(float(receipt["executed_left_m"]))
        and math.isfinite(float(receipt["executed_yaw_rad"]))
        for receipt in local_receipts
    ), "frame-bound local SE(2) wire contract changed")
    require(any(math.hypot(
        float(receipt["executed_forward_m"]),
        float(receipt["executed_left_m"]),
    ) > 1e-6 for receipt in local_receipts),
        "wire gate never observed a non-zero local translation")

    return {
        "request_count": len(requests),
        "certificate_accept_count": len(accepted),
        "active_readout_count": sum(
            plan.get("se2_route_compass_status") == "active"
            for plan in accepted),
        "terminal_tangent_count": sum(
            plan.get("se2_route_compass_status") == "terminal_tangent"
            for plan in accepted),
        "goal_start_frame": next(iter(goal_starts)),
        "target_anchor": next(iter(anchors)),
        "history_receipt_sha256": next(iter(history_hashes)),
        "history_receipt_count": next(iter(history_counts)),
        "route_extent_m": next(iter(extents)),
        "initial_projected_progress_m": progress[0],
        "final_projected_progress_m": progress[-1],
        "final_query_path_length_m": path_lengths[-1],
        "maximum_cross_track_error_m": max(cross_tracks),
        "frame_bound_local_se2_receipt_count": len(local_receipts),
        "progress_finite_monotone": True,
        "unit_bearing_norm_verified": True,
        "fixed_controller_radius_m": 2.5,
        "runtime_geometry": "local_executor_se2_plus_monotone_projection",
        "executor_odometry_source": SE2_ODOMETRY_SOURCE,
        "simulator_pose_difference_used_as_odometry_proxy": True,
        "world_pose_consumed_by_policy": False,
        "visual_gate_after_authorization": False,
        "distance_regime_present": False,
        "endpoint_fallback_available": False,
        "native_fallback_after_authorization_available": False,
        "lingbot_translation_consumed": False,
        "metric_scale_consumed": False,
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
            "runtime gate is sealed to history index 40")
    require(arguments.history_contract == "causal_survey",
            "runtime gate requires causal-survey history")
    require(arguments.role_pair_scope == "table3_length",
            "runtime gate requires the length-population contract")
    require(arguments.max_steps >= 600,
            "outer frozen length budget receipt changed")

    manifest_path = arguments.bench_root / "manifest.json"
    require(sha256(manifest_path) == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(manifest["episodes"]) == 48,
            "length population size changed")
    item = manifest["episodes"][arguments.history_index]
    require(item["bin_name"] == FROZEN_DISTANCE_BIN,
            "prospective history distance bin changed")
    source_episode = Path(item["online_a_episode"])
    require(sha256(source_episode / "receipt.json")
            == item["online_a_receipt_sha256"],
            "causal-history receipt changed")
    require(sha256(source_episode / "online_a_trace.json")
            == item["online_a_trace_sha256"],
            "causal-history trace changed")
    receipt = json.loads(
        (source_episode / "receipt.json").read_text(encoding="utf-8"))
    require(receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey",
            "history source changed")
    scene_asset = Path(receipt["source_asset"])
    require(scene_asset.is_file()
            and sha256(scene_asset) == receipt["source_asset_sha256"],
            "scene asset changed")
    pinned_navmesh = Path(item["runtime_navmesh"])
    require(pinned_navmesh.is_file()
            and sha256(pinned_navmesh) == item["runtime_navmesh_sha256"],
            "pinned navmesh changed")

    scene = str(item["scene"])
    episode = str(item["episode"])
    label = f"{arguments.history_index:03d}_{scene}_{episode}"
    output = arguments.run_root / "runtime_gate" / label
    require(not output.exists(), f"gate output already exists: {output}")
    arm_root = output / "mono_cec_se2_route"
    logs = output / "logs"
    arm_root.mkdir(parents=True)
    logs.mkdir()

    command = [
        arguments.hab_python, "-u",
        str(arguments.evaluator_source_root
            / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(arguments.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_asset),
        "--scene_identity", scene,
        "--host", "127.0.0.1",
        "--port", str(arguments.memnav_port),
        "--novel_port", str(arguments.navdp_port),
        "--server_backend", "hybrid_pose",
        "--success_dist", "1.0",
        "--max_steps", str(QUERY_STEP_GUARD),
        "--stuck_window", str(QUERY_STEP_GUARD + 1),
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
        "--certified_guidance_mode", "se2_route_compass",
        "--revisit_controller", "navdp_mixed",
        "--role_pair_scope", "table3_length",
        "--navdp_depth_source", "monocular_sidecar",
        "--hybrid_route", "certified_relocalization",
        "--revisit_adapter", "verified_bearing_v1",
        "--pinned_navmesh", str(pinned_navmesh),
        "--expected_pinned_navmesh_sha256", item["runtime_navmesh_sha256"],
        "--out", str(arm_root),
    ]
    started = time.monotonic()
    log_path = logs / "eval_mono_cec_se2_route.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            f"SE(2) route runtime gate failed; inspect {log_path}")

    revisit_paths = sorted(arm_root.glob("*_revisit_plans.json"))
    novel_paths = sorted(arm_root.glob("*_novel_plans.json"))
    require(len(revisit_paths) == 1 and len(novel_paths) == 1,
            "expected one complete hidden Novel/Revisit pair")
    revisit_payload = json.loads(revisit_paths[0].read_text(encoding="utf-8"))
    novel_payload = json.loads(novel_paths[0].read_text(encoding="utf-8"))
    require(
        revisit_payload.get("analysis_role_not_forwarded") is True
        and novel_payload.get("analysis_role_not_forwarded") is True,
        "role-hiding receipt is absent",
    )
    audit = audit_se2_route_plans(revisit_payload["query_leg"])
    novel_requests = [
        plan for plan in novel_payload["query_leg"]
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(novel_requests, "Novel query emitted no certificate request")
    require(not any(
        plan.get("certified_relocalization_accepted") is True
        for plan in novel_requests
    ), "prospective Novel query was falsely authorized")

    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "runtime_gate_passed",
        "claim_boundary": (
            "controller-interface gate only; navigation success and final "
            "goal distance were not read"),
        "history_index": arguments.history_index,
        "scene": scene,
        "episode": episode,
        "bin_name": item["bin_name"],
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "query_step_guard": QUERY_STEP_GUARD,
        "se2_route_audit": audit,
        "runtime_role_filter_used": False,
        "hidden_role_queries_executed": ["novel", "revisit"],
        "novel_certificate_request_count": len(novel_requests),
        "novel_certificate_accept_count": 0,
        "source_revisit_plan_sha256": sha256(revisit_paths[0]),
        "source_novel_plan_sha256": sha256(novel_paths[0]),
        "wall_time_seconds": time.monotonic() - started,
        "navigation_success_read": False,
        "navigation_final_distance_read": False,
        "navigation_sr_computed": False,
        "closed_loop_comparison_authorized": True,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    completion_path = output / "gate_completion.json"
    completion_path.write_bytes(encoded)
    (output / "gate_completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  gate_completion.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": completion["status"],
        "history_index": arguments.history_index,
        "audit": audit,
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ABORT: {type(error).__name__}: {error}", file=sys.stderr)
        raise
