#!/usr/bin/env python3
"""Run three paired arms on one fresh same-floor long-range Revisit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from MemNavData.hm3d_longrange_route_tangent_experiment import (
    ARMS,
    ARM_CONFIG,
    audit_mono_depth,
    audit_route_tangent,
    initial_proof,
    require,
    rotated_arm_order,
)
from MemNavData.run_final14_mono_factorial_episode import (
    audit_fully_rejected_fallback,
    load_payloads,
    load_rows,
    run_command,
)


COMPLETION_SCHEMA = (
    "hm3d_longrange_route_tangent_history_result_v1_20260903"
)
POPULATION_SCHEMA = (
    "hm3d_longrange_route_tangent_population_v2_20260903"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_replay(reference: dict, candidate: dict, arm: str) -> None:
    for field in (
            "all_rgb_hashes_verified", "decision_frames", "decision_steps",
            "diffusion_samples_during_replay", "navdp_memory_size",
            "navdp_queue_lengths", "online_frames"):
        require(reference["replay"][field] == candidate["replay"][field],
                f"{arm}: replay field {field} changed")
    require(reference["legA"] == candidate["legA"],
            f"{arm}: frozen causal history plan receipts changed")
    require(reference["rollout_traces"]["legA"]
            == candidate["rollout_traces"]["legA"],
            f"{arm}: frozen causal history poses changed")
    require(reference["memory_traces"]["legA"]
            == candidate["memory_traces"]["legA"],
            f"{arm}: frozen causal RGB replay changed")


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
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--history-contract", default="causal_survey")
    parser.add_argument(
        "--role-pair-scope",
        default="table3_longrange_route_tangent",
    )
    args = parser.parse_args()
    require(tuple(part.strip() for part in args.arms.split(",") if part.strip())
            == ARMS, "formal route-tangent arm set changed")
    require(args.history_contract == "causal_survey",
            "route-tangent confirmation requires the causal survey")
    require(args.role_pair_scope == "table3_longrange_route_tangent",
            "route-tangent confirmation scope changed")

    manifest_path = args.bench_root / "manifest.json"
    require(sha256_file(manifest_path) == args.expected_manifest_sha256,
            "fresh benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    population_path = args.bench_root.parent / "population_receipt.json"
    verification_path = args.bench_root.parent / "independent_verification.json"
    population = json.loads(population_path.read_text())
    verification = json.loads(verification_path.read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "fresh population receipt changed")
    require(verification.get("verified") is True
            and verification.get("formal_policy_evaluation_authorized") is True
            and verification.get("benchmark_manifest_sha256")
            == args.expected_manifest_sha256,
            "fresh population is not independently authorized")
    require(0 <= args.history_index < len(manifest["episodes"]),
            "history index is outside the frozen population")
    item = manifest["episodes"][args.history_index]
    require(item.get("longrange_route_tangent_fresh") is True,
            "history is not in the fresh route-tangent population")
    require(float(item["revisit_vertical_error_m"]) <= 0.5 + 1e-12,
            "history escaped the same-floor stratum")
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    require(sha256_file(source_episode / "receipt.json")
            == item["online_a_receipt_sha256"],
            "causal-history receipt changed")
    require(sha256_file(source_episode / "online_a_trace.json")
            == item["online_a_trace_sha256"],
            "causal-history trace changed")
    receipt = json.loads((source_episode / "receipt.json").read_text())
    trace = json.loads((source_episode / "online_a_trace.json").read_text())
    require(receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey"
            and trace.get("source_hybrid_route") == "causal_survey",
            "history is not the frozen causal RGB survey")
    require(int(trace.get("metric_depth_sensor_reads", -1)) == 0,
            "history consumed simulator metric depth")
    require(all(plan.get("navdp_depth_source") == "monocular_sidecar"
                for plan in trace["plans"]),
            "history replay contains a non-monocular policy plan")
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file()
            and sha256_file(scene_file) == receipt["source_asset_sha256"],
            "HM3D scene asset changed")
    navmesh = Path(item["runtime_navmesh"])
    require(item["runtime_geometry"] == "content_addressed_pinned_navmesh"
            and navmesh.is_file()
            and sha256_file(navmesh) == item["runtime_navmesh_sha256"],
            "pinned runtime navmesh changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "evaluation" / label
    require(not output.exists(), f"history output already exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = rotated_arm_order(args.history_index)
    contract = {
        "schema_version": COMPLETION_SCHEMA,
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "candidate_identity_sha256": item["candidate_identity_sha256"],
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "runtime_navmesh_sha256": item["runtime_navmesh_sha256"],
        "revisit_vertical_error_m": float(item[
            "revisit_vertical_error_m"]),
        "arms": list(ARMS),
        "arm_order": list(order),
        "runtime_role_visibility": "none",
        "evaluator_selection_role": "revisit",
        "success_distance_m": 1.0,
        "success_geometry": "three_dimensional_euclidean",
        "route_depth_stride_frames": 8,
        "max_steps": int(args.max_steps),
        "deterministic_plan_seeds": True,
    }
    (output / "episode_contract.json").write_text(json.dumps(
        contract, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")

    common = [
        args.hab_python, "-u",
        str(args.evaluator_source_root
            / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(args.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--pinned_navmesh", str(navmesh),
        "--expected_pinned_navmesh_sha256", item[
            "runtime_navmesh_sha256"],
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
    ]
    rows_by_arm = {}
    payload_by_arm = {}
    audits = {}
    elapsed = {}
    for arm in order:
        config = ARM_CONFIG[arm]
        arm_root = output / arm
        arm_root.mkdir()
        command = common + [
            "--hybrid_route", config["hybrid_route"],
            "--revisit_adapter", config["revisit_adapter"],
            "--certified_guidance_mode", config["guidance_mode"],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, output / "logs" / f"eval_{arm}.log")
        summary = json.loads((arm_root / "summary.json").read_text())
        require(summary.get("queries") == 1
                and summary.get("role_counts")
                == {"novel": 0, "revisit": 1},
                f"{arm}: evaluator did not run exactly one Revisit")
        require(summary.get("arm") == config["evaluator_arm"],
                f"{arm}: evaluator arm changed")
        require(summary.get("runtime_role_visibility") == "none",
                f"{arm}: analysis role reached a policy")
        rows = load_rows(arm_root)
        require(len(rows) == 1 and rows[0]["analysis_role"] == "revisit",
                f"{arm}: metric row coverage changed")
        payloads = load_payloads(arm_root, episode, rows)
        payload = payloads["revisit"]
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{arm}: role forwarding audit failed")
        plans = payload["query_leg"]
        arm_audit = {"depth": audit_mono_depth(plans, arm=arm)}
        if arm == "mono_cec_route_tangent":
            arm_audit["route_tangent"] = audit_route_tangent(plans)
            if int(rows[0]["geometry_stream_stop_plans"]) > 0:
                require(rows[0]["termination_reason"]
                        == "geometry_stream_failure"
                        and int(rows[0]["reached"]) == 0,
                        "geometry stop was not scored as a navigation failure")
        rows_by_arm[arm] = rows[0]
        payload_by_arm[arm] = payload
        audits[arm] = arm_audit

    reference = payload_by_arm["mono_native"]
    for arm in ARMS:
        compare_replay(reference, payload_by_arm[arm], arm)

    endpoint_proof = initial_proof(
        payload_by_arm["mono_cec_endpoint"]["query_leg"])
    tangent_proof = initial_proof(
        payload_by_arm["mono_cec_route_tangent"]["query_leg"])
    require(endpoint_proof == tangent_proof,
            "endpoint and route-tangent arms did not share the initial proof")
    fully_rejected_exact_native = {}
    for arm in ("mono_cec_endpoint", "mono_cec_route_tangent"):
        fully_rejected_exact_native[arm] = audit_fully_rejected_fallback(
            arm=arm,
            role="revisit",
            cec_row=rows_by_arm[arm],
            cec_payload=payload_by_arm[arm],
            native_payload=payload_by_arm["mono_native"],
        )

    completion = {
        **contract,
        "prefix_equality": True,
        "initial_cec_proof_equal_between_methods": True,
        "initial_cec_proof": endpoint_proof,
        "wall_time_seconds": elapsed,
        "outcomes": {
            arm: int(rows_by_arm[arm]["reached"]) for arm in ARMS
        },
        "initial_geodesic_m": {
            arm: float(rows_by_arm[arm]["geodesic_m"]) for arm in ARMS
        },
        "final_goal_3d_dist_m": {
            arm: float(rows_by_arm[arm]["final_goal_3d_dist_m"])
            for arm in ARMS
        },
        "final_goal_planar_dist_m": {
            arm: float(rows_by_arm[arm]["final_goal_planar_dist_m"])
            for arm in ARMS
        },
        "final_goal_vertical_error_m": {
            arm: float(rows_by_arm[arm]["final_goal_vertical_error_m"])
            for arm in ARMS
        },
        "path_len_m": {
            arm: float(rows_by_arm[arm]["path_len_m"]) for arm in ARMS
        },
        "steps": {
            arm: int(rows_by_arm[arm]["steps"]) for arm in ARMS
        },
        "termination_reason": {
            arm: rows_by_arm[arm]["termination_reason"] for arm in ARMS
        },
        "certificate_accept_plans": {
            arm: int(rows_by_arm[arm]["certificate_accept_plans"])
            for arm in ARMS
        },
        "geometry_stream_stop_plans": int(
            rows_by_arm["mono_cec_route_tangent"][
                "geometry_stream_stop_plans"]),
        "fully_rejected_exact_native": fully_rejected_exact_native,
        "audits": audits,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode()
    completion_path = output / "completion.json"
    completion_path.write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  completion.json\n"
    )
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "outcomes_sealed_in_completion": True,
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
