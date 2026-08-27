#!/usr/bin/env python3
"""Run one Final14 history under paired strict/unthresholded CEC authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from MemNavData.final14_authority_ablation import (
    ARMS,
    AUTHORITY_POLICY,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    require,
    rotated_arm_order,
)
from MemNavData.final14_mono_factorial import audit_depth_plans
from MemNavData.run_final14_mono_factorial_episode import (
    compare_shared_replay,
    load_payloads,
    load_rows,
    run_command,
    sha256,
)


SCHEMA = "final14_cec_authority_ablation_episode_v1_20260828"


def first_query_plan(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    plans = payload.get("query_leg")
    require(isinstance(plans, list) and bool(plans),
            f"{label}: query emitted no plans")
    require(isinstance(plans[0], dict), f"{label}: first plan is invalid")
    return plans[0]


def audit_initial_proposal_pair(
        strict: dict[str, Any], witness: dict[str, Any], *,
        label: str) -> dict[str, Any]:
    """Verify both arms enter authority with the same first proposal."""

    strict_plan = first_query_plan(strict, label=f"{label}/strict")
    witness_plan = first_query_plan(witness, label=f"{label}/witness")
    for plan, expected, arm in (
        (strict_plan, "strict_certificate", "strict"),
        (witness_plan, "pnp_pose_available", "witness"),
    ):
        require(plan.get("certified_relocalization_authority_policy") == expected,
                f"{label}/{arm}: authority policy changed")
        require(plan.get("certified_relocalization_proposal_order") ==
                "geometry_first",
                f"{label}/{arm}: proposal order changed")
    fields = (
        "router_candidate_order_dino",
        "router_candidate_order_used",
        "router_selected_anchor",
        "router_selected_candidate_dino_rank",
    )
    for field in fields:
        require(strict_plan.get(field) == witness_plan.get(field),
                f"{label}: first proposal field {field} differs")

    strict_accept = bool(
        strict_plan.get("certified_relocalization_accepted") is True)
    witness_accept = bool(
        witness_plan.get("certified_relocalization_accepted") is True)
    require(not (strict_accept and not witness_accept),
            f"{label}: unthresholded witness rejected a strict acceptance")
    if strict_accept:
        require(strict_plan.get("certified_relocalization_pnp") ==
                witness_plan.get("certified_relocalization_pnp"),
                f"{label}: accepted PnP witness changed")
        require(strict_plan.get("aux_pose") == witness_plan.get("aux_pose"),
                f"{label}: accepted bearing changed")
    return {
        "proposal_fields_equal": True,
        "strict_accepted": strict_accept,
        "unthresholded_witness_accepted": witness_accept,
        "authority_discordant": witness_accept and not strict_accept,
        "selected_anchor": strict_plan.get("router_selected_anchor"),
        "strict_reason": strict_plan.get("certified_relocalization_reason"),
        "witness_reason": witness_plan.get("certified_relocalization_reason"),
        "strict_certificate": strict_plan.get(
            "certified_relocalization_certificate"),
        "witness_pnp": witness_plan.get("certified_relocalization_pnp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    bench_root = Path(args.bench_root).resolve()
    manifest_path = bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Final14 natural manifest changed")
    manifest = json.loads(manifest_path.read_text())
    histories = manifest["episodes"]
    require(0 <= args.history_index < len(histories),
            "history index outside frozen population")
    item = histories[args.history_index]
    require(int(item["online_a_steps"]) >= 40,
            "Final14 history cannot activate first-40 mono depth")
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    receipt = json.loads((source_episode / "receipt.json").read_text())
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file(), "source scene asset missing")
    require(sha256(scene_file) == receipt["source_asset_sha256"],
            "source scene asset changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    episode_root = run_root / "evaluation" / "natural_direction" / label
    require(not episode_root.exists(), f"output already exists: {episode_root}")
    (episode_root / "logs").mkdir(parents=True)
    order = rotated_arm_order(args.history_index)
    contract = {
        "schema_version": SCHEMA,
        "scope": "consumed_final14_cec_authority_only_ablation",
        "fresh_confirmation": False,
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "online_a_steps": int(item["online_a_steps"]),
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "arms": list(ARMS),
        "arm_order": list(order),
        "runtime_role_visibility": "none",
        "shared_history_policy": "original_metric_navdp_goal_a_rgb_replay",
        "query_controller_depth": DEPTH_SOURCE,
        "fixed_components": [
            "dino_temporally_diverse_top8",
            "superpoint_lightglue",
            "fundamental_magsac_ranking",
            "lingbot_historical_depth",
            "pnp_ransac",
            "verified_bearing_v1_fixed_2.5m",
            "frozen_navdp",
        ],
        "sole_intervention": "operational_authority_policy",
        "max_steps": args.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "deterministic_plan_seeds": True,
        "smoke": bool(args.smoke),
    }
    (episode_root / "episode_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n")

    common = [
        args.hab_python, "-u",
        str(source_root / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", args.host,
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
        "--role_pair_scope", "consumed_integration",
        "--revisit_adapter", REVISIT_ADAPTER,
        "--navdp_depth_source", DEPTH_SOURCE,
    ]

    elapsed: dict[str, float] = {}
    arm_rows: dict[str, dict[str, dict[str, str]]] = {}
    arm_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    arm_audits: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in order:
        arm_root = episode_root / arm
        arm_root.mkdir()
        command = common + [
            "--hybrid_route", HYBRID_ROUTE[arm],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, episode_root / "logs" / f"eval_{arm}.log")
        summary = json.loads((arm_root / "summary.json").read_text())
        require(summary.get("queries") == 2,
                f"{arm}: evaluator query count changed")
        require(summary.get("arm") == EVALUATOR_ARM[arm],
                f"{arm}: evaluator arm mismatch")
        require(summary.get("navdp_depth_source") == DEPTH_SOURCE,
                f"{arm}: summary depth source mismatch")
        require(summary.get("runtime_role_visibility") == "none",
                f"{arm}: role leaked into runtime")
        rows = load_rows(arm_root)
        require(len(rows) == 2 and
                {row["analysis_role"] for row in rows} == {"novel", "revisit"},
                f"{arm}: paired role population changed")
        by_role = {row["analysis_role"]: row for row in rows}
        payloads = load_payloads(arm_root, episode, rows)
        audits: dict[str, dict[str, Any]] = {}
        for role in ("novel", "revisit"):
            payload = payloads[role]
            require(payload.get("analysis_role_not_forwarded") is True,
                    f"{arm}/{role}: analysis role forwarding audit failed")
            audits[role] = audit_depth_plans(
                "mono_cec", payload["query_leg"])
            require(int(by_role[role]["runtime_failure_plans"]) == 0,
                    f"{arm}/{role}: runtime failure observed")
        arm_rows[arm] = by_role
        arm_payloads[arm] = payloads
        arm_audits[arm] = audits

    reference_rows = arm_rows["mono_cec"]
    reference_payloads = arm_payloads["mono_cec"]
    for arm in ARMS:
        compare_shared_replay(
            reference_rows, reference_payloads,
            arm_rows[arm], arm_payloads[arm], arm)

    proposal_audit = {
        role: audit_initial_proposal_pair(
            arm_payloads["mono_cec"][role],
            arm_payloads["mono_unthresholded_witness"][role],
            label=f"{label}/{role}",
        )
        for role in ("novel", "revisit")
    }
    completion = {
        **contract,
        "prefix_equality": True,
        "initial_proposal_equality": True,
        "wall_time_seconds": elapsed,
        "outcomes": {
            arm: {
                role: int(arm_rows[arm][role]["reached"])
                for role in ("novel", "revisit")
            }
            for arm in ARMS
        },
        "final_distance_m": {
            arm: {
                role: float(arm_rows[arm][role]["final_goal_dist_m"])
                for role in ("novel", "revisit")
            }
            for arm in ARMS
        },
        "operational_accept_plans": {
            arm: {
                role: int(arm_rows[arm][role]["certificate_accept_plans"])
                for role in ("novel", "revisit")
            }
            for arm in ARMS
        },
        "initial_proposal_audit": proposal_audit,
        "depth_audits": arm_audits,
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    (episode_root / "completion.json").write_bytes(encoded)
    (episode_root / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n")
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "outcomes": completion["outcomes"],
        "authority_discordance": {
            role: proposal_audit[role]["authority_discordant"]
            for role in ("novel", "revisit")
        },
        "output": str(episode_root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
