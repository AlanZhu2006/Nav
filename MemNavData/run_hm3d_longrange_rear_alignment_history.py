#!/usr/bin/env python3
"""Run the paired consumed rear-alignment diagnostic on one history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from MemNavData.hm3d_longrange_rear_alignment import (
    ARMS,
    ARM_ALIGNMENT,
    PARENT_STUCK_INDICES,
    PARENT_SUCCESS_INDICES,
    SELECTED_HISTORY_INDICES,
    audit_rear_aligned,
    audit_unaligned,
    initial_route_packet,
    require,
    rotated_arm_order,
)
from MemNavData.hm3d_longrange_route_tangent_experiment import (
    audit_mono_depth,
    audit_route_tangent,
    initial_proof,
)
from MemNavData.run_final14_mono_factorial_episode import (
    load_payloads,
    load_rows,
    run_command,
)


SCHEMA = "hm3d_longrange_rear_alignment_history_v1_20260904"
PROTOCOL_SCHEMA = "hm3d_longrange_rear_alignment_protocol_v1_20260904"
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
            f"{arm}: frozen causal-history plan receipts changed")
    require(reference["rollout_traces"]["legA"]
            == candidate["rollout_traces"]["legA"],
            f"{arm}: frozen causal-history poses changed")
    require(reference["memory_traces"]["legA"]
            == candidate["memory_traces"]["legA"],
            f"{arm}: frozen causal RGB replay changed")


def _parent_partition(summary: dict) -> tuple[set[int], set[int]]:
    rows = [row for row in summary.get("rows", [])
            if row.get("arm") == "mono_cec_route_tangent"]
    require(len(rows) == 23, "formal parent route rows changed")
    selected = {int(row["history_index"]) for row in rows
                if row.get("termination_reason")
                != "geometry_stream_failure"}
    stuck = {int(row["history_index"]) for row in rows
             if row.get("termination_reason") == "stuck"}
    success = {int(row["history_index"]) for row in rows
               if row.get("termination_reason") == "success"
               and int(row.get("reached", 0)) == 1}
    require(selected == set(SELECTED_HISTORY_INDICES)
            and stuck.intersection(selected) == set(PARENT_STUCK_INDICES)
            and success.intersection(selected) == set(PARENT_SUCCESS_INDICES),
            "consumed parent partition changed")
    return stuck, success


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
    parser.add_argument("--role-pair-scope",
                        default="table3_longrange_route_tangent")
    parser.add_argument(
        "--protocol", type=Path,
        default=(Path(os.environ["PROTOCOL"])
                 if os.environ.get("PROTOCOL") else None))
    parser.add_argument(
        "--expected-protocol-sha256",
        default=os.environ.get("EXPECTED_PROTOCOL_SHA256"))
    args = parser.parse_args()

    require(tuple(part.strip() for part in args.arms.split(",")
                  if part.strip()) == ARMS,
            "rear-alignment arm set changed")
    require(args.history_contract == "causal_survey"
            and args.role_pair_scope == "table3_longrange_route_tangent",
            "rear-alignment history/scope changed")
    require(args.protocol is not None
            and args.expected_protocol_sha256 is not None,
            "frozen rear-alignment protocol was not supplied")
    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "rear-alignment protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "rear-alignment protocol schema changed")
    require(tuple(protocol.get("selected_history_indices", ()))
            == SELECTED_HISTORY_INDICES
            and tuple(protocol.get("arms", ())) == ARMS,
            "rear-alignment frozen population/arms changed")
    require(args.history_index in SELECTED_HISTORY_INDICES,
            "history is outside the frozen consumed partition")

    formal_root = Path(protocol["formal_parent"]["run_root"])
    formal_summary_path = formal_root / "result" / "summary.json"
    formal_verification_path = (
        formal_root / "result" / "independent_verification.json")
    require(formal_summary_path.is_file()
            and sha256_file(formal_summary_path)
            == protocol["formal_parent"]["summary_sha256"],
            "formal parent summary changed")
    require(formal_verification_path.is_file()
            and sha256_file(formal_verification_path)
            == protocol["formal_parent"]["independent_verification_sha256"],
            "formal parent verification changed")
    verification = json.loads(formal_verification_path.read_text())
    require(verification.get("verified") is True,
            "formal parent is not independently verified")
    formal_summary = json.loads(formal_summary_path.read_text())
    parent_stuck, parent_success = _parent_partition(formal_summary)

    manifest_path = args.bench_root / "manifest.json"
    require(sha256_file(manifest_path) == args.expected_manifest_sha256
            == protocol["formal_parent"]["benchmark_manifest_sha256"],
            "formal benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    population = json.loads((
        args.bench_root.parent / "population_receipt.json").read_text())
    population_verification = json.loads((
        args.bench_root.parent / "independent_verification.json").read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA
            and population_verification.get("verified") is True
            and population_verification.get(
                "formal_policy_evaluation_authorized") is True,
            "parent population is not authorized")
    item = manifest["episodes"][args.history_index]
    require(item.get("longrange_route_tangent_fresh") is True
            and float(item["revisit_vertical_error_m"]) <= 0.5 + 1e-12,
            "history escaped the parent same-floor long-range population")
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    require(sha256_file(source_episode / "receipt.json")
            == item["online_a_receipt_sha256"]
            and sha256_file(source_episode / "online_a_trace.json")
            == item["online_a_trace_sha256"],
            "frozen causal history changed")
    history_receipt = json.loads((source_episode / "receipt.json").read_text())
    history_trace = json.loads((
        source_episode / "online_a_trace.json").read_text())
    require(history_receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey"
            and history_trace.get("source_hybrid_route") == "causal_survey"
            and int(history_trace.get("metric_depth_sensor_reads", -1)) == 0,
            "history is not the frozen monocular causal survey")
    scene_file = Path(history_receipt["source_asset"])
    navmesh = Path(item["runtime_navmesh"])
    require(scene_file.is_file()
            and sha256_file(scene_file)
            == history_receipt["source_asset_sha256"]
            and navmesh.is_file()
            and sha256_file(navmesh) == item["runtime_navmesh_sha256"],
            "HM3D scene or pinned navmesh changed")

    formal_matches = sorted((formal_root / "evaluation").glob(
        f"{args.history_index:03d}_*/completion.json"))
    require(len(formal_matches) == 1,
            "formal parent completion is missing or ambiguous")
    formal_completion = json.loads(formal_matches[0].read_text())
    formal_reason = formal_completion["termination_reason"][
        "mono_cec_route_tangent"]
    formal_outcome = int(formal_completion["outcomes"][
        "mono_cec_route_tangent"])
    require(formal_completion["scene"] == scene
            and formal_completion["episode"] == episode
            and formal_reason != "geometry_stream_failure"
            and ((args.history_index in parent_stuck
                  and formal_reason == "stuck" and formal_outcome == 0)
                 or (args.history_index in parent_success
                     and formal_reason == "success" and formal_outcome == 1)),
            "formal parent completion left the frozen partition")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "evaluation" / label
    require(not output.exists(), f"rear-alignment output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = rotated_arm_order(args.history_index)
    contract = {
        "schema_version": SCHEMA,
        "claim_scope": "consumed controller-support mechanism only",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "candidate_identity_sha256": item["candidate_identity_sha256"],
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "formal_parent_completion_sha256": sha256_file(formal_matches[0]),
        "formal_parent_outcome": formal_outcome,
        "formal_parent_termination_reason": formal_reason,
        "formal_parent_partition": (
            "stuck" if args.history_index in parent_stuck else "success"),
        "arms": list(ARMS),
        "arm_order": list(order),
        "runtime_role_visibility": "none",
        "success_distance_m": 1.0,
        "success_geometry": "three_dimensional_euclidean",
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
        "--hybrid_route", "certified_relocalization",
        "--revisit_adapter", "verified_bearing_v1",
        "--certified_guidance_mode", "monocular_route_tangent",
    ]
    rows_by_arm = {}
    payload_by_arm = {}
    audits = {}
    elapsed = {}
    for arm in order:
        arm_root = output / arm
        arm_root.mkdir()
        command = common + [
            "--cec_initial_bearing_alignment", ARM_ALIGNMENT[arm],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, output / "logs" / f"eval_{arm}.log")
        summary = json.loads((arm_root / "summary.json").read_text())
        require(summary.get("queries") == 1
                and summary.get("role_counts") == {"novel": 0, "revisit": 1}
                and summary.get("arm") == "certified",
                f"{arm}: evaluator arm/coverage changed")
        rows = load_rows(arm_root)
        require(len(rows) == 1 and rows[0]["analysis_role"] == "revisit",
                f"{arm}: metric row coverage changed")
        payload = load_payloads(arm_root, episode, rows)["revisit"]
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{arm}: analysis role reached the runtime")
        plans = payload["query_leg"]
        depth_audit = audit_mono_depth(
            plans, arm="mono_cec_route_tangent")
        route_audit = audit_route_tangent(plans)
        alignment_audit = (
            audit_unaligned(row=rows[0], payload=payload)
            if arm == ARMS[0]
            else audit_rear_aligned(row=rows[0], payload=payload)
        )
        if int(rows[0]["geometry_stream_stop_plans"]) > 0:
            require(rows[0]["termination_reason"]
                    == "geometry_stream_failure"
                    and int(rows[0]["reached"]) == 0,
                    f"{arm}: geometry failure was not atomic")
        rows_by_arm[arm] = rows[0]
        payload_by_arm[arm] = payload
        audits[arm] = {
            "depth": depth_audit,
            "route_tangent": route_audit,
            "rear_alignment": alignment_audit,
        }

    compare_replay(payload_by_arm[ARMS[0]], payload_by_arm[ARMS[1]], ARMS[1])
    proofs = {arm: initial_proof(payload_by_arm[arm]["query_leg"])
              for arm in ARMS}
    packets = {arm: initial_route_packet(
        payload_by_arm[arm]["query_leg"]) for arm in ARMS}
    require(proofs[ARMS[0]] is not None
            and proofs[ARMS[0]] == proofs[ARMS[1]],
            "paired arms did not share the initial CEC proof")
    require(packets[ARMS[0]] == packets[ARMS[1]],
            "paired arms did not share the initial route packet")

    completion = {
        **contract,
        "prefix_equality": True,
        "initial_cec_proof_equal": True,
        "initial_route_packet_equal": True,
        "initial_cec_proof": proofs[ARMS[0]],
        "initial_route_packet_sha256": packets[ARMS[0]]["packet_sha256"],
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
