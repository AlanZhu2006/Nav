#!/usr/bin/env python3
"""Run one sealed HM3D history under the long-range CEC development arms.

The target proof is identical in the two CEC arms.  Only the readout after an
accept differs: the canonical arm keeps the endpoint bearing, while the
challenger repeatedly observes one monotone coordinate on the already-proved
visual route.  The evaluator never receives the Novel/Revisit role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from MemNavData.final14_mono_factorial import audit_depth_plans
from MemNavData.run_final14_mono_factorial_episode import (
    audit_fully_rejected_fallback,
    compare_shared_replay,
    load_payloads,
    load_rows,
    run_command,
)
from MemNavData.run_hm3d_episodic_path_field_gate import (
    audit_path_field_plans,
)
from MemNavData.run_hm3d_fullmono_query_history import (
    audit_history_contract,
)


SCHEMA_VERSION = "hm3d_monotone_visual_route_dev_v1_20260902"
ARMS = (
    "mono_native",
    "mono_cec_endpoint",
    "mono_cec_visual_route",
)
ARM_CONFIG = {
    "mono_native": {
        "hybrid_route": "native_sidecar",
        "revisit_adapter": "legacy_metric",
        "guidance_mode": "endpoint_bearing",
        "evaluator_arm": "native_sidecar",
        "depth_audit_arm": "mono_native",
    },
    "mono_cec_endpoint": {
        "hybrid_route": "certified_relocalization",
        "revisit_adapter": "verified_bearing_v1",
        "guidance_mode": "endpoint_bearing",
        "evaluator_arm": "certified",
        "depth_audit_arm": "mono_cec",
    },
    "mono_cec_visual_route": {
        "hybrid_route": "certified_relocalization",
        "revisit_adapter": "verified_bearing_v1",
        "guidance_mode": "episodic_path_field",
        "evaluator_arm": "certified",
        "depth_audit_arm": "mono_cec",
    },
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
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def first_target_proof(plans: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [
        plan for plan in plans
        if plan.get("certified_relocalization_ok") is not None
    ]
    require(requests, "CEC query emitted no target-proof request")
    first = requests[0]
    return {
        "ok": first.get("certified_relocalization_ok"),
        "accepted": first.get("certified_relocalization_accepted"),
        "reason": first.get("certified_relocalization_reason"),
        "selected_anchor": first.get("anchor"),
        "certificate": first.get("certified_relocalization_certificate"),
        "proposal_order": first.get(
            "certified_relocalization_proposal_order"),
        "pointgoal_units": first.get(
            "certified_relocalization_pointgoal_units"),
    }


def guidance_receipt_audit(
    arm: str,
    role: str,
    plans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit the readout without converting target rejection into failure."""

    config = ARM_CONFIG[arm]
    depth = audit_depth_plans(config["depth_audit_arm"], plans)
    if arm == "mono_native":
        return {"depth": depth, "target_proof": None, "route": None}

    proof = first_target_proof(plans)
    accepted = [
        plan for plan in plans
        if plan.get("certified_relocalization_accepted") is True
    ]
    if not accepted:
        return {
            "depth": depth,
            "target_proof": proof,
            "route": None,
            "fully_rejected": True,
        }

    expected_mode = config["guidance_mode"]
    require(all(
        plan.get("certified_relocalization_guidance_mode") == expected_mode
        for plan in accepted
    ), f"{arm}/{role}: accepted request changed guidance mode")
    route = None
    if arm == "mono_cec_visual_route":
        route = audit_path_field_plans(plans)
    else:
        require(not any(
            plan.get("episodic_path_field_requested") is True
            for plan in accepted
        ), f"{arm}/{role}: endpoint arm entered route tracking")
    return {
        "depth": depth,
        "target_proof": proof,
        "route": route,
        "fully_rejected": False,
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

    require(arguments.history_contract == "causal_survey",
            "visual-route development requires causal-survey history")
    require(arguments.role_pair_scope == "table3_length",
            "visual-route development requires the length population")
    require(600 <= arguments.max_steps <= 3400,
            "frozen length budget lies outside 600..3400")
    manifest_path = arguments.bench_root / "manifest.json"
    require(sha256(manifest_path) == arguments.expected_manifest_sha256,
            "sealed length manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    histories = manifest["episodes"]
    require(len(histories) == 48,
            "consumed development population must contain 48 histories")
    require(0 <= arguments.history_index < len(histories),
            "history index outside the sealed population")
    item = histories[arguments.history_index]
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
                for plan in trace["plans"]),
            "causal history contains a non-monocular plan")
    require(all(plan.get("metric_depth_sensor_consumed") is False
                for plan in trace["plans"]),
            "causal history consumed simulator depth")
    prefix_a_steps, prefix_b_steps, history_policy = audit_history_contract(
        receipt, trace, arguments.history_contract)
    require(prefix_a_steps >= 40 and prefix_b_steps == 0,
            "causal survey cannot establish the first-40 receipt")

    scene = str(item["scene"])
    episode = str(item["episode"])
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file()
            and sha256(scene_file) == receipt["source_asset_sha256"],
            "HM3D scene asset changed")
    navmesh = Path(item["runtime_navmesh"])
    require(item.get("runtime_geometry")
            == "content_addressed_pinned_navmesh"
            and navmesh.is_file()
            and sha256(navmesh) == item["runtime_navmesh_sha256"],
            "pinned runtime navmesh changed")

    label = f"{arguments.history_index:03d}_{scene}_{episode}"
    output = (arguments.run_root / "development" /
              "monotone_visual_route" / label)
    require(not output.exists(), f"development output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = rotated_arm_order(arguments.history_index)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "consumed-population development only",
        "history_index": arguments.history_index,
        "scene": scene,
        "episode": episode,
        "distance_bin": item["bin_name"],
        "online_a_steps": int(item["online_a_steps"]),
        "prefix_A_steps": prefix_a_steps,
        "prefix_B_steps": prefix_b_steps,
        "history_policy": history_policy,
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "benchmark_manifest_sha256": arguments.expected_manifest_sha256,
        "runtime_navmesh_sha256": item["runtime_navmesh_sha256"],
        "runtime_role_visibility": "none",
        "arms": list(ARMS),
        "arm_order": list(order),
        "max_steps": arguments.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "route_localization_horizon": "all_remaining_authorized_route",
        "route_control_horizon_m": 2.5,
        "distance_gate_present": False,
        "endpoint_or_native_fallback_after_authorization": False,
    }
    (output / "episode_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    common = [
        arguments.hab_python, "-u",
        str(arguments.evaluator_source_root /
            "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(arguments.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", "127.0.0.1",
        "--port", str(arguments.memnav_port),
        "--novel_port", str(arguments.navdp_port),
        "--server_backend", "hybrid_pose",
        "--success_dist", "1.0",
        "--max_steps", str(arguments.max_steps),
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
        "--role_pair_scope", "table3_length",
        "--navdp_depth_source", "monocular_sidecar",
        "--pinned_navmesh", str(navmesh),
        "--expected_pinned_navmesh_sha256",
        item["runtime_navmesh_sha256"],
    ]

    elapsed: dict[str, float] = {}
    rows_by_arm: dict[str, dict[str, dict[str, str]]] = {}
    payloads_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    audits_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
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
        summary = json.loads(
            (arm_root / "summary.json").read_text(encoding="utf-8"))
        require(summary.get("queries") == 2,
                f"{arm}: expected one hidden Novel/Revisit pair")
        require(summary.get("arm") == config["evaluator_arm"],
                f"{arm}: evaluator arm changed")
        require(summary.get("runtime_role_visibility") == "none",
                f"{arm}: role leaked into runtime")
        require(summary.get("navdp_depth_source") == "monocular_sidecar",
                f"{arm}: query depth source changed")
        rows = load_rows(arm_root)
        require(len(rows) == 2
                and {row["analysis_role"] for row in rows}
                == {"novel", "revisit"},
                f"{arm}: hidden-role population changed")
        by_role = {row["analysis_role"]: row for row in rows}
        require(all(int(row.get("runtime_failure_plans", -1)) == 0
                    for row in by_role.values()),
                f"{arm}: runtime failure is not a navigation outcome")
        payloads = load_payloads(arm_root, episode, rows)
        audits = {}
        for role in ("novel", "revisit"):
            require(payloads[role].get("analysis_role_not_forwarded") is True,
                    f"{arm}/{role}: role-hiding receipt missing")
            audits[role] = guidance_receipt_audit(
                arm, role, payloads[role]["query_leg"])
        rows_by_arm[arm] = by_role
        payloads_by_arm[arm] = payloads
        audits_by_arm[arm] = audits

    native_rows = rows_by_arm["mono_native"]
    native_payloads = payloads_by_arm["mono_native"]
    for arm in ARMS:
        compare_shared_replay(
            native_rows, native_payloads,
            rows_by_arm[arm], payloads_by_arm[arm], arm)

    fallback: dict[str, dict[str, bool]] = {}
    for arm in ("mono_cec_endpoint", "mono_cec_visual_route"):
        fallback[arm] = {}
        for role in ("novel", "revisit"):
            fallback[arm][role] = audit_fully_rejected_fallback(
                arm=arm,
                role=role,
                cec_row=rows_by_arm[arm][role],
                cec_payload=payloads_by_arm[arm][role],
                native_payload=native_payloads[role],
            )

    for role in ("novel", "revisit"):
        left = audits_by_arm["mono_cec_endpoint"][role]["target_proof"]
        right = audits_by_arm["mono_cec_visual_route"][role]["target_proof"]
        require(left == right,
                f"{role}: endpoint and route arms changed target proof")

    def metric(field: str, cast) -> dict[str, dict[str, Any]]:
        return {
            arm: {
                role: cast(rows_by_arm[arm][role][field])
                for role in ("novel", "revisit")
            }
            for arm in ARMS
        }

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
        "fully_rejected_exact_native": fallback,
        "runtime_audits": audits_by_arm,
        "navigation_outcomes_read_after_all_arms": True,
    }
    encoded = (json.dumps(
        completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    completion_path = output / "completion.json"
    completion_path.write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "history_index": arguments.history_index,
        "distance_bin": item["bin_name"],
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
