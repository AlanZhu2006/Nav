#!/usr/bin/env python3
"""Run one strict LingBot-coordinate replay gate for episodic path guidance.

The gate intentionally stops at 80 query steps and does not aggregate or emit
navigation success.  It verifies that an accepted certificate can construct
and update a continuous path field using only deployment-visible LingBot
geometry before any long closed-loop comparison is authorized.
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


SCHEMA_VERSION = "hm3d_episodic_path_field_deployment_gate_v2_20260902"
FROZEN_HISTORY_INDICES = (0, 16, 32)
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


def audit_path_field_plans(plans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate strict runtime receipts without reading a success outcome."""

    require(bool(plans), "query emitted no planning receipts")
    require(all(plan.get("role_label_visible") is not True for plan in plans),
            "runtime role label became visible")
    requests = [
        plan for plan in plans
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(requests, "query emitted no certificate requests")
    require(all(plan.get("certified_relocalization_ok") is True
                for plan in requests),
            "certificate request had a runtime failure")
    accepted = [
        plan for plan in requests
        if plan.get("certified_relocalization_accepted") is True
    ]
    require(accepted, "no certificate was accepted during the deployment gate")
    require(all(plan.get("certified_relocalization_guidance_mode")
                == "episodic_path_field" for plan in accepted),
            "accepted request used another guidance mode")
    require(all(plan.get("episodic_path_field_status") in {"active", "complete"}
                for plan in accepted),
            "accepted request lacks a valid path-field state")
    require(all(plan.get("episodic_path_field_error") is None
                and plan.get("episodic_path_field_error_type") is None
                for plan in accepted),
            "path-field geometry failure was silently retained")
    require(all(plan.get("episodic_path_field_evaluator_pose_consumed") is False
                and plan.get("episodic_path_field_habitat_path_consumed") is False
                for plan in accepted),
            "deployment readout consumed evaluator geometry")
    route_updates = [
        plan for plan in accepted
        if plan.get("route_coordinate_state_updated") is not None
    ]
    require(route_updates,
            "accepted target proof never entered visual route tracking")
    require(all(plan.get("route_coordinate_state_updated") is True
                for plan in route_updates),
            "a visual route-coordinate observation failed")
    require(all(
        plan.get("route_coordinate_endpoint_fallback_available") is False
        and plan.get("route_coordinate_native_fallback_available") is False
        and plan.get("route_coordinate_distance_gate_present") is False
        for plan in route_updates
    ), "route tracking exposed a fallback or distance gate")

    active = [
        plan for plan in accepted
        if plan.get("episodic_path_field_status") == "active"
    ]
    require(active, "path field never emitted an active local bearing")
    progress = [float(plan["path_progress_m"]) for plan in active]
    remaining = [float(plan["path_remaining_m"]) for plan in active]
    cross_track = [float(plan["path_cross_track_m"]) for plan in active]
    require(all(math.isfinite(value) for value in (
        *progress, *remaining, *cross_track)),
        "path-field receipt contains a non-finite value")
    require(all(current + 1e-9 >= previous
                for previous, current in zip(progress, progress[1:])),
            "path-field progress regressed")
    require(all(value >= -1e-9 for value in remaining),
            "path-field remaining arc is negative")
    require(all(isinstance(plan.get("path_unit_bearing"), list)
                and len(plan["path_unit_bearing"]) == 2
                and all(math.isfinite(float(value))
                        for value in plan["path_unit_bearing"])
                for plan in active),
            "active path field omitted a finite unit bearing")
    scale_hashes = {
        str(plan["episodic_path_field_scale_receipt_sha256"])
        for plan in accepted
        if plan.get("episodic_path_field_scale_receipt_sha256") is not None
    }
    require(len(scale_hashes) == 1, "first-40 scale receipt was not immutable")
    goal_starts = {
        int(plan["episodic_path_field_goal_start_frame"])
        for plan in accepted
    }
    anchors = {
        int(plan["episodic_path_field_target_anchor"])
        for plan in accepted
    }
    require(len(goal_starts) == 1 and len(anchors) == 1,
            "frozen path contract changed during replay")
    return {
        "request_count": len(requests),
        "certificate_accept_count": len(accepted),
        "active_path_readout_count": len(active),
        "complete_path_readout_count": sum(
            plan.get("episodic_path_field_status") == "complete"
            for plan in accepted),
        "visual_route_update_count": len(route_updates),
        "goal_start_frame": next(iter(goal_starts)),
        "target_anchor": next(iter(anchors)),
        "scale_receipt_sha256": next(iter(scale_hashes)),
        "initial_progress_m": progress[0],
        "final_progress_m": progress[-1],
        "maximum_cross_track_m": max(cross_track),
        "initial_remaining_m": remaining[0],
        "final_remaining_m": remaining[-1],
        "progress_finite_monotone": True,
        "runtime_geometry": (
            "target_proof_once_plus_monotone_visual_route_coordinate"),
        "route_localization_horizon": "all_remaining_authorized_route",
        "route_control_horizon_m": 2.5,
        "endpoint_fallback_available": False,
        "native_fallback_after_authorization_available": False,
        "distance_gate_present": False,
        "evaluator_pose_consumed": False,
        "habitat_path_consumed": False,
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

    require(arguments.history_index in FROZEN_HISTORY_INDICES,
            "deployment gate is sealed to history indices 0,16,32")
    require(arguments.history_contract == "causal_survey",
            "deployment gate requires the causal-survey history")
    require(arguments.role_pair_scope == "table3_length",
            "deployment gate requires the length-population contract")
    require(arguments.max_steps >= 600,
            "outer frozen length budget receipt changed")

    manifest_path = arguments.bench_root / "manifest.json"
    require(sha256(manifest_path) == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(manifest["episodes"]) == 48,
            "length population size changed")
    item = manifest["episodes"][arguments.history_index]
    expected_bin = {
        0: "0_to_20_m", 16: "20_to_30_m", 32: "30_to_50_m",
    }[arguments.history_index]
    require(item["bin_name"] == expected_bin,
            "frozen one-per-bin selection changed")
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
    arm_root = output / "mono_cec_path_field"
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
        "--certified_guidance_mode", "episodic_path_field",
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
    log_path = logs / "eval_mono_cec_path_field.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            f"strict path-field replay failed; inspect {log_path}")

    # Formal Table-III evaluation deliberately forbids a runtime role filter.
    # Execute the complete hidden-role pair and select the Revisit receipt only
    # after rollout, for analysis.  This keeps the deployment gate on the same
    # role-free interface as the eventual paired experiment.
    revisit_paths = sorted(arm_root.glob("*_revisit_plans.json"))
    novel_paths = sorted(arm_root.glob("*_novel_plans.json"))
    require(len(revisit_paths) == 1 and len(novel_paths) == 1,
            "expected one complete hidden Novel/Revisit plan pair")
    revisit_payload = json.loads(
        revisit_paths[0].read_text(encoding="utf-8"))
    novel_payload = json.loads(
        novel_paths[0].read_text(encoding="utf-8"))
    require(
        revisit_payload.get("analysis_role_not_forwarded") is True
        and novel_payload.get("analysis_role_not_forwarded") is True,
        "role-hiding receipt is absent",
    )
    audit = audit_path_field_plans(revisit_payload["query_leg"])
    novel_requests = [
        plan for plan in novel_payload["query_leg"]
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(novel_requests, "Novel query emitted no certificate requests")
    require(not any(
        plan.get("certified_relocalization_accepted") is True
        for plan in novel_requests
    ), "deployment gate Novel query was falsely authorized")
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "deployment_gate_passed",
        "claim_boundary": (
            "LingBot-coordinate replay gate only; navigation SR not read"),
        "history_selection": "lowest population index in each distance bin",
        "history_index": arguments.history_index,
        "scene": scene,
        "episode": episode,
        "bin_name": item["bin_name"],
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "query_step_guard": QUERY_STEP_GUARD,
        "wall_time_seconds": time.monotonic() - started,
        "path_field_audit": audit,
        "runtime_role_filter_used": False,
        "hidden_role_queries_executed": ["novel", "revisit"],
        "novel_certificate_request_count": len(novel_requests),
        "novel_certificate_accept_count": 0,
        "source_revisit_plan_sha256": sha256(revisit_paths[0]),
        "source_novel_plan_sha256": sha256(novel_paths[0]),
        "navigation_success_read": False,
        "navigation_sr_computed": False,
        "closed_loop_comparison_authorized": True,
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
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
