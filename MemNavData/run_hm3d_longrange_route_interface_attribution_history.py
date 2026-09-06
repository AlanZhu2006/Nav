#!/usr/bin/env python3
"""Run the frozen two-arm consumed long-range route-interface attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from MemNavData.hm3d_longrange_route_interface_attribution import (
    ARMS,
    ARM_CONFIG,
    audit_controller_support,
    rotated_arm_order,
)
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


SCHEMA = "hm3d_longrange_route_interface_attribution_history_v1_20260903"
PROTOCOL_SCHEMA = (
    "hm3d_longrange_route_interface_attribution_protocol_v1_20260903"
)
POPULATION_SCHEMA = "hm3d_longrange_route_tangent_population_v2_20260903"


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
        "--role-pair-scope", default="table3_longrange_route_tangent")
    parser.add_argument(
        "--protocol", type=Path,
        default=(Path(os.environ["PROTOCOL"])
                 if os.environ.get("PROTOCOL") else None),
    )
    parser.add_argument(
        "--expected-protocol-sha256",
        default=os.environ.get("EXPECTED_PROTOCOL_SHA256"),
    )
    args = parser.parse_args()

    require(tuple(part.strip() for part in args.arms.split(",")
                  if part.strip()) == ARMS,
            "route-interface arm set changed")
    require(args.history_contract == "causal_survey",
            "route-interface attribution requires the causal survey")
    require(args.role_pair_scope == "table3_longrange_route_tangent",
            "route-interface role-pair scope changed")
    require(args.protocol is not None
            and args.expected_protocol_sha256 is not None,
            "frozen attribution protocol was not supplied")
    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "route-interface attribution protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "route-interface attribution protocol schema changed")

    parent_diagnostic = protocol["required_parent_diagnostic"]
    diagnostic_root = Path(parent_diagnostic["run_root"])
    diagnostic_summary = diagnostic_root / "result" / "summary.json"
    diagnostic_verification = (
        diagnostic_root / "result" / "independent_verification.json")
    for path in (diagnostic_summary, diagnostic_verification):
        require(path.is_file(), f"required parent diagnostic missing: {path}")
        sidecar = path.with_name(path.name + ".sha256")
        require(sidecar.is_file(), f"parent diagnostic sidecar missing: {path}")
        words = sidecar.read_text().strip().split()
        require(len(words) == 2 and words[0] == sha256_file(path)
                and words[1] == path.name,
                f"parent diagnostic hash changed: {path}")
    diagnostic = json.loads(diagnostic_summary.read_text())
    diagnostic_check = json.loads(diagnostic_verification.read_text())
    require(diagnostic.get("protocol_sha256")
            == parent_diagnostic["protocol_sha256"],
            "parent diagnostic protocol binding changed")
    require(diagnostic_check.get("verified") is True
            and diagnostic_check.get("supports_epipolar_degeneracy") is True
            and diagnostic.get("supports_epipolar_degeneracy") is True,
            "no-authority shadow gate did not authorize direct-PnP attribution")

    manifest_path = args.bench_root / "manifest.json"
    require(sha256_file(manifest_path) == args.expected_manifest_sha256,
            "formal benchmark manifest changed")
    require(protocol["formal_parent"]["benchmark_manifest_sha256"]
            == args.expected_manifest_sha256,
            "protocol points to another formal benchmark")
    manifest = json.loads(manifest_path.read_text())
    population = json.loads((
        args.bench_root.parent / "population_receipt.json").read_text())
    population_verification = json.loads((
        args.bench_root.parent / "independent_verification.json").read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "formal population receipt changed")
    require(population_verification.get("verified") is True
            and population_verification.get(
                "formal_policy_evaluation_authorized") is True,
            "formal population is not independently authorized")
    require(0 <= args.history_index < len(manifest["episodes"]) == 23,
            "history index is outside the consumed formal population")
    item = manifest["episodes"][args.history_index]
    require(item.get("longrange_route_tangent_fresh") is True
            and float(item["revisit_vertical_error_m"]) <= 0.5 + 1e-12,
            "history escaped the formal same-floor long-range stratum")
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
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file()
            and sha256_file(scene_file) == receipt["source_asset_sha256"],
            "HM3D scene asset changed")
    navmesh = Path(item["runtime_navmesh"])
    require(navmesh.is_file()
            and sha256_file(navmesh) == item["runtime_navmesh_sha256"],
            "pinned runtime navmesh changed")

    formal_root = Path(protocol["formal_parent"]["run_root"])
    formal_matches = sorted((formal_root / "evaluation").glob(
        f"{args.history_index:03d}_*/completion.json"))
    require(len(formal_matches) == 1,
            "sealed formal completion is missing or ambiguous")
    formal = json.loads(formal_matches[0].read_text())
    require(formal["scene"] == scene and formal["episode"] == episode,
            "sealed formal completion identity changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "evaluation" / label
    require(not output.exists(), f"attribution output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = rotated_arm_order(args.history_index)
    contract = {
        "schema_version": SCHEMA,
        "claim_scope": "consumed mechanism attribution only",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "candidate_identity_sha256": item["candidate_identity_sha256"],
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "parent_diagnostic_summary_sha256": sha256_file(
            diagnostic_summary),
        "parent_diagnostic_verification_sha256": sha256_file(
            diagnostic_verification),
        "formal_parent_completion_sha256": sha256_file(formal_matches[0]),
        "formal_parent_route_tangent_outcome": int(
            formal["outcomes"]["mono_cec_route_tangent"]),
        "arms": list(ARMS),
        "arm_order": list(order),
        "adjacent_route_motion_model": "direct_depth_pnp_ransac",
        "runtime_role_visibility": "none",
        "success_distance_m": 1.0,
        "success_geometry": "three_dimensional_euclidean",
        "controller_radius_m": 2.5,
        "max_steps": int(args.max_steps),
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
    }
    (output / "episode_contract.json").write_text(json.dumps(
        contract, indent=2, sort_keys=True, allow_nan=False) + "\n")

    common = [
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
        "--certified_guidance_mode", "monocular_route_tangent",
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
            "--revisit_adapter", config["revisit_adapter"],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, output / "logs" / f"eval_{arm}.log")
        summary = json.loads((arm_root / "summary.json").read_text())
        require(summary.get("queries") == 1
                and summary.get("role_counts") == {"novel": 0, "revisit": 1},
                f"{arm}: evaluator did not run exactly one Revisit")
        require(summary.get("arm") == config["evaluator_arm"],
                f"{arm}: evaluator arm changed")
        rows = load_rows(arm_root)
        require(len(rows) == 1 and rows[0]["analysis_role"] == "revisit",
                f"{arm}: metric row coverage changed")
        payload = load_payloads(arm_root, episode, rows)["revisit"]
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{arm}: role forwarding audit failed")
        plans = payload["query_leg"]
        depth_audit = audit_mono_depth(
            plans, arm="mono_cec_route_tangent")
        route_audit = audit_route_tangent(plans)
        support_audit = audit_controller_support(plans, arm=arm)
        active = [plan for plan in plans
                  if plan.get("geometry_stream_stop") is not True]
        require(all(plan.get("local_tangent_route_motion_model")
                    == "direct_pnp" for plan in active
                    if plan.get("certified_relocalization_accepted") is True),
                f"{arm}: server did not use the direct-PnP route model")
        if int(rows[0]["geometry_stream_stop_plans"]) > 0:
            require(rows[0]["termination_reason"] == "geometry_stream_failure"
                    and int(rows[0]["reached"]) == 0,
                    f"{arm}: geometry failure was not atomic")
        rows_by_arm[arm] = rows[0]
        payload_by_arm[arm] = payload
        audits[arm] = {
            "depth": depth_audit,
            "route_tangent": route_audit,
            "controller_support": support_audit,
        }

    reference = payload_by_arm[ARMS[0]]
    compare_replay(reference, payload_by_arm[ARMS[1]], ARMS[1])
    proofs = {arm: initial_proof(payload_by_arm[arm]["query_leg"])
              for arm in ARMS}
    require(all(proof is not None for proof in proofs.values())
            and proofs[ARMS[0]] == proofs[ARMS[1]],
            "two interface arms did not share one accepted initial proof")

    completion = {
        **contract,
        "prefix_equality": True,
        "initial_cec_proof_equal": True,
        "initial_cec_proof": proofs[ARMS[0]],
        "wall_time_seconds": elapsed,
        "outcomes": {arm: int(rows_by_arm[arm]["reached"])
                     for arm in ARMS},
        "final_goal_3d_dist_m": {
            arm: float(rows_by_arm[arm]["final_goal_3d_dist_m"])
            for arm in ARMS},
        "path_len_m": {arm: float(rows_by_arm[arm]["path_len_m"])
                       for arm in ARMS},
        "steps": {arm: int(rows_by_arm[arm]["steps"]) for arm in ARMS},
        "termination_reason": {
            arm: rows_by_arm[arm]["termination_reason"] for arm in ARMS},
        "geometry_stream_stop_plans": {
            arm: int(rows_by_arm[arm]["geometry_stream_stop_plans"])
            for arm in ARMS},
        "audits": audits,
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
        "outcomes_sealed_in_completion": True,
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
