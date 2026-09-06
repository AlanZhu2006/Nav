#!/usr/bin/env python3
"""Run one consumed dense-query route-motion mechanism gate history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from MemNavData.hm3d_longrange_route_tangent_experiment import (
    audit_mono_depth,
    audit_route_tangent,
    initial_proof,
    require,
)
from MemNavData.run_final14_mono_factorial_episode import (
    load_payloads,
    load_rows,
    run_command,
)


SCHEMA = "hm3d_longrange_dense_live_query_gate_history_v2_20260903"
PROTOCOL_SCHEMA = (
    "hm3d_longrange_dense_live_query_gate_protocol_v2_20260903")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discrete_proof(proof: dict | None) -> dict | None:
    """Remove cross-device floating values while retaining CEC authority."""

    if proof is None:
        return None
    authority = proof.get("authority") or {}
    strict = authority.get("strict_certificate") or {}
    certificate = proof.get("certificate") or {}
    return {
        "selected_anchor": proof.get("selected_anchor"),
        "candidate_order_dino": proof.get("candidate_order_dino"),
        "candidate_order_used": proof.get("candidate_order_used"),
        "proposal_source": proof.get("proposal_source"),
        "authority_policy": authority.get("policy"),
        "authority_accepted": authority.get("accepted"),
        "authority_reason": authority.get("reason"),
        "strict_checks": strict.get("checks"),
        "certificate_checks": certificate.get("checks"),
        "certificate_thresholds": certificate.get("thresholds"),
    }


def dense_sampling_audit(plans: list[dict]) -> dict:
    accepted = [
        plan for plan in plans
        if plan.get("certified_relocalization_accepted") is True
        and plan.get("geometry_stream_stop") is not True
    ]
    if not accepted:
        stopped = [
            plan for plan in plans
            if plan.get("geometry_stream_stop") is True
        ]
        require(stopped,
                "dense-query arm produced neither an active route nor a "
                "typed geometry stop")
        for plan in stopped:
            require(plan.get("local_tangent_route_motion_model")
                    == "direct_pnp_dense_query",
                    "server did not expose the dense-query motion model")
            require(plan.get("local_tangent_historical_motion_model")
                    == "fundamental_then_pnp"
                    and plan.get("local_tangent_live_query_motion_model")
                    == "direct_pnp",
                    "dense-query stage-specific estimator contract changed")
        return {
            "accepted_route_plans": 0,
            "completed_query_intervals": 0,
            "per_action_dense_intervals": 0,
            "query_motion_edges": 0,
            "all_completed_intervals_dense": True,
            "failure_before_or_at_route_initialization": True,
            "runtime_role_or_distance_gate_present": False,
        }
    previous_frame = None
    interval_count = 0
    dense_intervals = 0
    edge_count = 0
    for plan in accepted:
        require(plan.get("local_tangent_route_motion_model")
                == "direct_pnp_dense_query",
                "server did not expose the dense-query direct-PnP model")
        require(plan.get("local_tangent_query_motion_sampling")
                == "per_action_dense",
                "dense-query sampling receipt changed")
        require(plan.get("local_tangent_historical_motion_model")
                == "fundamental_then_pnp"
                and plan.get("local_tangent_live_query_motion_model")
                == "direct_pnp",
                "dense-query stage-specific estimator contract changed")
        require(plan.get("local_tangent_evaluator_pose_consumed") is False
                and plan.get("local_tangent_executor_odometry_consumed")
                is False
                and plan.get("local_tangent_metric_depth_sensor_consumed")
                is False,
                "dense-query route consumed a forbidden motion source")
        frame = int(plan["frame_idx"])
        update_edges = int(plan.get("local_tangent_update_edge_count", -1))
        receipts = plan.get("local_tangent_update_edge_receipts")
        require(isinstance(receipts, list)
                and len(receipts) == update_edges,
                "dense-query update receipts are incomplete")
        if previous_frame is None:
            require(update_edges == 0,
                    "initial route plan unexpectedly consumed query motion")
        else:
            interval_count += 1
            expected = frame - previous_frame
            require(expected > 0,
                    "accepted dense-query planning frame did not advance")
            if update_edges == expected:
                dense_intervals += 1
            edge_count += update_edges
            require(update_edges == expected,
                    "query interval was not decomposed into every causal "
                    "frame transition")
            expected_pairs = [
                (value, value + 1)
                for value in range(previous_frame, frame)
            ]
            actual_pairs = [
                (int(row["match_reference_frame"]),
                 int(row["match_query_frame"]))
                for row in receipts
            ]
            require(actual_pairs == expected_pairs,
                    "dense-query motion edges are not consecutive")
        previous_frame = frame
    return {
        "accepted_route_plans": len(accepted),
        "completed_query_intervals": interval_count,
        "per_action_dense_intervals": dense_intervals,
        "query_motion_edges": edge_count,
        "all_completed_intervals_dense": dense_intervals == interval_count,
        "failure_before_or_at_route_initialization": False,
        "runtime_role_or_distance_gate_present": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--arms", default="mono_cec_route_tangent")
    parser.add_argument("--history-contract", default="causal_survey")
    parser.add_argument(
        "--role-pair-scope", default="table3_longrange_route_tangent")
    parser.add_argument(
        "--protocol", type=Path,
        default=(Path(os.environ["PROTOCOL"])
                 if os.environ.get("PROTOCOL") else None))
    parser.add_argument(
        "--expected-protocol-sha256",
        default=os.environ.get("EXPECTED_PROTOCOL_SHA256"))
    parser.add_argument(
        "--formal-run-root", type=Path,
        default=(Path(os.environ["FORMAL_RUN_ROOT"])
                 if os.environ.get("FORMAL_RUN_ROOT") else None))
    args = parser.parse_args()

    require(args.arms == "mono_cec_route_tangent",
            "dense-query gate executes one route-tangent arm")
    require(args.history_contract == "causal_survey"
            and args.role_pair_scope == "table3_longrange_route_tangent",
            "dense-query gate benchmark scope changed")
    require(args.protocol is not None
            and args.expected_protocol_sha256 is not None
            and args.formal_run_root is not None,
            "dense-query gate inputs are incomplete")
    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "dense-query protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "dense-query protocol schema changed")
    require(args.history_index in protocol["selected_history_indices"],
            "history is outside the frozen dense-query gate")

    parent = protocol["formal_parent"]
    require(str(args.formal_run_root) == parent["run_root"],
            "formal route-tangent parent changed")
    summary_path = args.formal_run_root / "result" / "summary.json"
    verification_path = (
        args.formal_run_root / "result" / "independent_verification.json")
    require(sha256_file(summary_path) == parent["summary_sha256"]
            and sha256_file(verification_path)
            == parent["independent_verification_sha256"],
            "formal parent result changed")
    require(json.loads(verification_path.read_text()).get("verified") is True,
            "formal parent is not independently verified")

    manifest_path = args.bench_root / "manifest.json"
    require(sha256_file(manifest_path) == args.expected_manifest_sha256
            == parent["benchmark_manifest_sha256"],
            "formal benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 23
            and 0 <= args.history_index < 23,
            "formal population size changed")
    item = manifest["episodes"][args.history_index]
    scene, episode = str(item["scene"]), str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    receipt = json.loads((source_episode / "receipt.json").read_text())
    scene_file = Path(receipt["source_asset"])
    navmesh = Path(item["runtime_navmesh"])
    require(scene_file.is_file()
            and sha256_file(scene_file) == receipt["source_asset_sha256"],
            "HM3D scene asset changed")
    require(navmesh.is_file()
            and sha256_file(navmesh) == item["runtime_navmesh_sha256"],
            "runtime navmesh changed")

    formal_matches = sorted((args.formal_run_root / "evaluation").glob(
        f"{args.history_index:03d}_*/completion.json"))
    require(len(formal_matches) == 1,
            "formal history completion is missing or ambiguous")
    formal = json.loads(formal_matches[0].read_text())
    require(formal["scene"] == scene and formal["episode"] == episode
            and formal["outcomes"]["mono_cec_route_tangent"] == 0
            and formal["termination_reason"]["mono_cec_route_tangent"]
            == "geometry_stream_failure",
            "selected parent history is not a geometry-stream failure")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "evaluation" / label
    require(not output.exists(), f"dense-query output exists: {output}")
    (output / "logs").mkdir(parents=True)
    arm_root = output / "mono_cec_dense_query"
    arm_root.mkdir()
    command = [
        args.hab_python, "-u",
        str(args.evaluator_source_root
            / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(args.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--pinned_navmesh", str(navmesh),
        "--expected_pinned_navmesh_sha256", item["runtime_navmesh_sha256"],
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
        "--role_pair_scope", "table3_longrange_route_tangent",
        "--role_pair_query_role", "revisit",
        "--navdp_depth_source", "monocular_sidecar",
        "--hybrid_route", "certified_relocalization",
        "--revisit_adapter", "verified_bearing_v1",
        "--certified_guidance_mode", "monocular_route_tangent",
        "--out", str(arm_root),
    ]
    elapsed = run_command(command, output / "logs" / "eval.log")
    rows = load_rows(arm_root)
    require(len(rows) == 1 and rows[0]["analysis_role"] == "revisit",
            "dense-query gate did not run one hidden-role Revisit")
    payload = load_payloads(arm_root, episode, rows)["revisit"]
    require(payload.get("analysis_role_not_forwarded") is True,
            "analysis role reached the runtime")
    plans = payload["query_leg"]
    depth_audit = audit_mono_depth(plans, arm="mono_cec_route_tangent")
    route_audit = audit_route_tangent(plans)
    sampling_audit = dense_sampling_audit(plans)
    proof = initial_proof(plans)
    proof_equal = discrete_proof(proof) == discrete_proof(
        formal.get("initial_cec_proof"))
    require(proof_equal,
            "dense-query candidate changed the initial CEC authority")
    geometry_stops = int(rows[0]["geometry_stream_stop_plans"])
    passed = geometry_stops == 0
    completion = {
        "schema_version": SCHEMA,
        "claim_scope": protocol["claim_scope"],
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "protocol_sha256": args.expected_protocol_sha256,
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "formal_parent_completion_sha256": sha256_file(formal_matches[0]),
        "formal_parent_geometry_stream_failure": True,
        "initial_cec_discrete_proof_equal": proof_equal,
        "initial_cec_discrete_proof": discrete_proof(proof),
        "motion_model": "direct_pnp_dense_query",
        "historical_route_motion_model": "fundamental_then_pnp",
        "live_query_motion_model": "direct_pnp",
        "live_query_sampling": "per_action_dense",
        "controller_adapter": "verified_bearing_v1_fixed_2.5m",
        "wall_time_seconds": elapsed,
        "outcome_descriptive_only": int(rows[0]["reached"]),
        "termination_reason": rows[0]["termination_reason"],
        "final_goal_3d_dist_m": float(rows[0]["final_goal_3d_dist_m"]),
        "path_len_m": float(rows[0]["path_len_m"]),
        "steps": int(rows[0]["steps"]),
        "geometry_stream_stop_plans": geometry_stops,
        "mechanism_gate_passed": passed,
        "depth_audit": depth_audit,
        "route_audit": route_audit,
        "dense_sampling_audit": sampling_audit,
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    completion_path = output / "completion.json"
    completion_path.write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  completion.json\n")
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "mechanism_gate_passed": passed,
        "geometry_stream_stop_plans": geometry_stops,
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
