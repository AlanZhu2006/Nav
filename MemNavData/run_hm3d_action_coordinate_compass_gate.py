#!/usr/bin/env python3
"""Run the preregistered SR-hidden action-coordinate controller gate.

The frozen index-40 history passed the offline prospective geometry gate.  This
runner exercises the same history through the real MemNav/NavDP/Habitat wire
for at most 80 query steps, but deliberately never reads or reports navigation
success.  It authorizes the later paired SR experiment only when the runtime
receipts prove that one initial CEC witness is followed by a continuous,
scale-free action-coordinate readout with no distance regime or post-accept
fallback.
"""

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


SCHEMA_VERSION = (
    "hm3d_action_coordinate_compass_deployment_gate_v1_20260902"
)
ACTION_COORDINATE_SCHEMA_VERSION = (
    "action_coordinate_route_compass_v1_20260902"
)
FROZEN_HISTORY_INDEX = 40
FROZEN_DISTANCE_BIN = "30_to_50_m"
QUERY_STEP_GUARD = 80


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    require(isinstance(value, list) and len(value) == 2,
            f"{name} must be a two-vector")
    pair = (float(value[0]), float(value[1]))
    require(all(math.isfinite(item) for item in pair),
            f"{name} is non-finite")
    return pair


def audit_action_coordinate_plans(
    plans: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the deployment contract without looking at SR or distance."""

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
        == "action_coordinate_compass"
        for plan in accepted
    ), "accepted request used another guidance mode")
    require(all(
        plan.get("action_coordinate_compass_status")
        in {"active", "terminal_tangent"}
        for plan in accepted
    ), "accepted request lacks a valid action-coordinate state")
    require(all(
        plan.get("action_coordinate_compass_error") is None
        and plan.get("action_coordinate_compass_error_type") is None
        for plan in accepted
    ), "action-coordinate failure was silently retained")
    require(all(
        plan.get("action_coordinate_schema_version")
        == ACTION_COORDINATE_SCHEMA_VERSION
        for plan in accepted
    ), "runtime action-coordinate schema changed")
    require(all(
        plan.get("action_coordinate_evaluator_pose_consumed") is False
        and plan.get("action_coordinate_habitat_path_consumed") is False
        and plan.get("action_coordinate_metric_scale_consumed") is False
        for plan in accepted
    ), "route readout consumed forbidden evaluator or metric geometry")
    require(all(
        plan.get("action_coordinate_visual_gate_present_after_initialization")
        is False
        and plan.get("action_coordinate_distance_regime_present") is False
        and plan.get("action_coordinate_endpoint_fallback_available") is False
        and plan.get("action_coordinate_native_fallback_available") is False
        for plan in accepted
    ), "route readout exposed a visual gate, distance regime, or fallback")

    progress = [
        float(plan["action_coordinate_progress_m"]) for plan in accepted
    ]
    fraction = [
        float(plan["action_coordinate_route_progress_fraction"])
        for plan in accepted
    ]
    update_translation = [
        float(plan["action_coordinate_update_translation_m"])
        for plan in accepted
    ]
    update_yaw = [
        float(plan["action_coordinate_update_yaw_rad"])
        for plan in accepted
    ]
    extents = {
        float(plan["action_coordinate_route_extent_m"])
        for plan in accepted
    }
    require(all(math.isfinite(value) for value in (
        *progress, *fraction, *update_translation, *update_yaw, *extents,
    )), "action-coordinate receipt contains a non-finite scalar")
    require(all(value >= -1e-9 for value in progress),
            "action progress became negative")
    require(all(current + 1e-9 >= previous
                for previous, current in zip(progress, progress[1:])),
            "action progress regressed")
    require(all(0.0 <= value <= 1.0 + 1e-9 for value in fraction),
            "route progress fraction escaped [0,1]")
    require(all(value >= -1e-9 for value in update_translation),
            "executor translation update became negative")
    require(len(extents) == 1 and next(iter(extents)) > 0.0,
            "authorized route extent changed or is empty")
    require(any(value > 1e-6 for value in update_translation),
            "bounded controller gate never exercised translation progress")

    bearings = [
        _finite_pair(plan.get("action_coordinate_unit_bearing"),
                     "unit bearing")
        for plan in accepted
    ]
    controller_goals = [
        _finite_pair(plan.get("action_coordinate_controller_pointgoal"),
                     "controller PointGoal")
        for plan in accepted
    ]
    require(all(abs(math.hypot(*value) - 1.0) <= 1e-6
                for value in bearings),
            "action-coordinate output is not a unit bearing")
    require(all(abs(math.hypot(*value) - 2.5) <= 1e-6
                for value in controller_goals),
            "action-coordinate controller payload changed fixed radius")
    require(all(abs(float(plan.get(
        "memory_controller_pointgoal_distance_m")) - 2.5) <= 1e-6
                for plan in accepted),
            "NavDP adapter did not receive the fixed 2.5 m residual")

    history_hashes = {
        str(plan["action_coordinate_history_receipt_sha256"])
        for plan in accepted
    }
    history_counts = {
        int(plan["action_coordinate_history_receipt_count"])
        for plan in accepted
    }
    goal_starts = {
        int(plan["action_coordinate_goal_start_frame"])
        for plan in accepted
    }
    anchors = {
        int(plan["action_coordinate_target_anchor"])
        for plan in accepted
    }
    require(len(history_hashes) == len(history_counts) == 1,
            "historical executor receipt changed during the query")
    require(len(goal_starts) == len(anchors) == 1,
            "proof-bound route endpoints changed during the query")
    append_receipts = [
        plan.get("executor_motion_receipt") for plan in plans
        if plan.get("executor_motion_receipt") is not None
    ]
    require(append_receipts, "frame-bound executor receipts were not logged")
    require(all(
        receipt.get("contract")
        == "frame_bound_realized_executor_motion_v1"
        and math.isfinite(float(receipt["executed_translation_m"]))
        and float(receipt["executed_translation_m"]) >= 0.0
        and math.isfinite(float(receipt["executed_yaw_rad"]))
        for receipt in append_receipts
    ), "frame-bound executor receipt contract changed")

    return {
        "request_count": len(requests),
        "certificate_accept_count": len(accepted),
        "active_readout_count": sum(
            plan.get("action_coordinate_compass_status") == "active"
            for plan in accepted),
        "terminal_tangent_count": sum(
            plan.get("action_coordinate_compass_status")
            == "terminal_tangent" for plan in accepted),
        "goal_start_frame": next(iter(goal_starts)),
        "target_anchor": next(iter(anchors)),
        "history_receipt_sha256": next(iter(history_hashes)),
        "history_receipt_count": next(iter(history_counts)),
        "route_action_extent_m": next(iter(extents)),
        "initial_action_progress_m": progress[0],
        "final_action_progress_m": progress[-1],
        "positive_translation_update_count": sum(
            value > 1e-6 for value in update_translation),
        "maximum_abs_yaw_update_rad": max(abs(value) for value in update_yaw),
        "frame_bound_executor_receipt_count": len(append_receipts),
        "progress_finite_monotone": True,
        "unit_bearing_norm_verified": True,
        "fixed_controller_radius_m": 2.5,
        "runtime_geometry": (
            "scale_free_lingbot_route_shape_plus_executor_motion_receipts"),
        "visual_gate_after_authorization": False,
        "distance_regime_present": False,
        "endpoint_fallback_available": False,
        "native_fallback_after_authorization_available": False,
        "evaluator_pose_consumed": False,
        "habitat_path_consumed": False,
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
            "controller gate is sealed to prospective history index 40")
    require(arguments.history_contract == "causal_survey",
            "controller gate requires causal-survey history")
    require(arguments.role_pair_scope == "table3_length",
            "controller gate requires the length-population contract")
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
    output = arguments.run_root / "deployment_gate" / label
    require(not output.exists(), f"gate output already exists: {output}")
    arm_root = output / "mono_cec_action_coordinate"
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
        "--certified_guidance_mode", "action_coordinate_compass",
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
    log_path = logs / "eval_mono_cec_action_coordinate.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            f"action-coordinate controller gate failed; inspect {log_path}")

    revisit_paths = sorted(arm_root.glob("*_revisit_plans.json"))
    novel_paths = sorted(arm_root.glob("*_novel_plans.json"))
    require(len(revisit_paths) == 1 and len(novel_paths) == 1,
            "expected one complete hidden Novel/Revisit pair")
    revisit_payload = json.loads(
        revisit_paths[0].read_text(encoding="utf-8"))
    novel_payload = json.loads(
        novel_paths[0].read_text(encoding="utf-8"))
    require(
        revisit_payload.get("analysis_role_not_forwarded") is True
        and novel_payload.get("analysis_role_not_forwarded") is True,
        "role-hiding receipt is absent",
    )
    audit = audit_action_coordinate_plans(revisit_payload["query_leg"])
    novel_requests = [
        plan for plan in novel_payload["query_leg"]
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(novel_requests, "Novel query emitted no certificate requests")
    require(not any(
        plan.get("certified_relocalization_accepted") is True
        for plan in novel_requests
    ), "prospective Novel query was falsely authorized")

    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "deployment_gate_passed",
        "claim_boundary": (
            "controller-interface gate only; navigation success and final "
            "goal distance were not read"),
        "history_selection": (
            "index 40 frozen before prospective offline mechanism output"),
        "history_index": arguments.history_index,
        "scene": scene,
        "episode": episode,
        "bin_name": item["bin_name"],
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "query_step_guard": QUERY_STEP_GUARD,
        "stuck_trigger_within_guard": False,
        "wall_time_seconds": time.monotonic() - started,
        "action_coordinate_audit": audit,
        "runtime_role_filter_used": False,
        "hidden_role_queries_executed": ["novel", "revisit"],
        "novel_certificate_request_count": len(novel_requests),
        "novel_certificate_accept_count": 0,
        "source_revisit_plan_sha256": sha256(revisit_paths[0]),
        "source_novel_plan_sha256": sha256(novel_paths[0]),
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
        "bin_name": item["bin_name"],
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
