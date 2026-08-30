#!/usr/bin/env python3
"""Run one constructed HM3D mono history under three paired query arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from MemNavData.hm3d_fullmono_mixed_role import (
    ARMS,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    audit_query_arm,
    require,
    selected_arm_order,
)
from MemNavData.run_final14_mono_factorial_episode import (
    audit_fully_rejected_fallback,
    compare_shared_replay,
    load_payloads,
    load_rows,
    run_command,
)


SCHEMAS = {
    "goal_a": "hm3d_fullmono_mixed_role_history_v1_20260820",
    "actual_ab": "hm3d_table2_leg3_history_v1_20260829",
    "causal_survey": "hm3d_table3_causal_survey_history_v1_20260830",
}
SCHEMA = SCHEMAS["goal_a"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_history_contract(
    receipt: dict, trace: dict, history_contract: str,
) -> tuple[int, int, str]:
    """Validate and name the immutable history that precedes each query."""

    require(history_contract in SCHEMAS, "unsupported history contract")
    if history_contract == "causal_survey":
        require(
            receipt.get("schema_version")
            == "hm3d_table3_causal_survey_materialized_v1_20260830",
            "Table-III survey receipt schema changed",
        )
        require(
            receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey"
            and trace.get("schema_version")
            == "hm3d_table3_causal_survey_trace_v1_20260830"
            and trace.get("source_hybrid_route") == "causal_survey",
            "Table-III history is not the frozen causal survey",
        )
        survey = receipt.get("survey_contract")
        intrinsic = np.asarray(receipt.get("camera_intrinsic"), dtype=float)
        require(
            isinstance(survey, dict)
            and survey.get("runtime_memory_input") == "RGB only"
            and survey.get("construction_only_simulator_depth") is True
            and survey.get("metric_depth_for_query_control_or_CEC") is False
            and int(trace.get("metric_depth_sensor_reads", -1)) == 0,
            "Table-III survey leaked simulator depth at runtime",
        )
        require(
            intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
            and int(receipt.get("episode_seed", -1))
            == int(trace.get("episode_seed", -2)) >= 0,
            "Table-III survey camera/seed receipt changed",
        )
        return (
            len(trace["poses"]), 0,
            "controlled_causal_rgb_geodesic_survey_replay",
        )
    if history_contract == "actual_ab":
        require(
            receipt.get("prefix_receipt_schema")
            == "hm3d_table2_actual_mono_ab_prefix_v1_20260829",
            "Table-2 history lacks the frozen A/B prefix receipt",
        )
        require(
            receipt.get("prefix_semantics")
            == "actual_mono_Novel_A_then_Novel_B",
            "Table-2 history prefix semantics changed",
        )
        prefix_a_steps = int(receipt.get("prefix_A_steps", -1))
        prefix_b_steps = int(receipt.get("prefix_B_steps", -1))
        require(
            prefix_a_steps > 0 and prefix_b_steps > 0
            and prefix_a_steps + prefix_b_steps == len(trace["poses"]),
            "Table-2 A/B segment lengths do not reproduce",
        )
        require(
            trace.get("prefix_semantics")
            == "exact_actual_mono_A_then_B_observation_concat",
            "Table-2 trace is not an exact actual A/B concatenation",
        )
        return (
            prefix_a_steps,
            prefix_b_steps,
            "actual_mono_navdp_novel_A_then_novel_B_rgb_replay",
        )
    require(
        receipt.get("prefix_receipt_schema") is None,
        "ordinary Goal-A evaluation received a Table-2 A/B prefix",
    )
    return (
        len(trace["poses"]), 0, "actual_mono_navdp_goal_a_rgb_replay",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluator-source-root", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--history-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--history-contract",
        choices=tuple(SCHEMAS),
        default="goal_a",
        help=(
            "goal_a replays one actual Novel-A history; actual_ab replays "
            "one hash-bound actual Novel-A then Novel-B causal prefix"
        ),
    )
    parser.add_argument(
        "--role-pair-scope",
        choices=("consumed_integration", "paper_heldout", "paper_replication",
                 "table3_length"),
        default="consumed_integration",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluator_source_root = args.evaluator_source_root or args.source_root

    manifest_path = args.bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "sealed HM3D full-mono manifest changed")
    manifest = json.loads(manifest_path.read_text())
    population = json.loads(
        (args.bench_root.parent / "population_receipt.json").read_text()
    )
    selected_arms = tuple(part.strip() for part in args.arms.split(",")
                          if part.strip())
    histories = manifest["episodes"]
    require(0 <= args.history_index < len(histories),
            "history index outside sealed population")
    item = histories[args.history_index]
    require(int(item["online_a_steps"]) >= 40,
            "constructed history cannot activate first-40 mono depth")
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    require(
        sha256(source_episode / "receipt.json")
        == item["online_a_receipt_sha256"],
        "sealed online-A receipt changed",
    )
    require(
        sha256(source_episode / "online_a_trace.json")
        == item["online_a_trace_sha256"],
        "sealed online-A trace changed",
    )
    receipt = json.loads((source_episode / "receipt.json").read_text())
    trace = json.loads((source_episode / "online_a_trace.json").read_text())
    if args.history_contract == "causal_survey":
        require(
            receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey",
            "history is outside the sealed causal-survey source",
        )
    else:
        control_audit = receipt.get("online_a_control_audit")
        require(
            isinstance(control_audit, dict) and control_audit.get("ok") is True,
            "history was not generated under audited native Goal-A control",
        )
        require(trace.get("source_hybrid_route") in {"native_sidecar", "phase"},
                "history route is outside audited mono native Goal-A sources")
    require(all(plan.get("navdp_depth_source") == "monocular_sidecar"
                for plan in trace["plans"]),
            "history Goal-A contains a non-mono plan")
    require(all(plan.get("metric_depth_sensor_consumed") is False
                for plan in trace["plans"]),
            "history Goal-A consumed simulator metric depth")
    prefix_a_steps, prefix_b_steps, shared_history_policy = (
        audit_history_contract(receipt, trace, args.history_contract)
    )
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file() and
            sha256(scene_file) == receipt["source_asset_sha256"],
            "explicit HM3D source asset changed")
    pinned_args: list[str] = []
    runtime_geometry = "runtime_recomputed_navmesh"
    runtime_navmesh_sha256 = None
    if args.role_pair_scope == "table3_length":
        pinned_navmesh = Path(item.get("runtime_navmesh", ""))
        require(
            item.get("runtime_geometry") == "content_addressed_pinned_navmesh"
            and pinned_navmesh.is_file()
            and sha256(pinned_navmesh) == item.get("runtime_navmesh_sha256"),
            "Table-III runtime navmesh receipt changed",
        )
        runtime_geometry = "content_addressed_pinned_navmesh"
        runtime_navmesh_sha256 = item["runtime_navmesh_sha256"]
        pinned_args = [
            "--pinned_navmesh", str(pinned_navmesh),
            "--expected_pinned_navmesh_sha256", runtime_navmesh_sha256,
        ]

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = args.run_root / "evaluation" / "natural_direction" / label
    require(not output.exists(), f"history output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = selected_arm_order(args.history_index, selected_arms)
    contract = {
        "schema_version": SCHEMAS[args.history_contract],
        "scope": population["scope"],
        "fresh_scene_generalization": bool(
            population.get("fresh_scene_generalization", False)),
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "online_a_steps": int(item["online_a_steps"]),
        "history_contract": args.history_contract,
        "prefix_A_steps": prefix_a_steps,
        "prefix_B_steps": prefix_b_steps,
        "online_a_trace_sha256": sha256(
            source_episode / "online_a_trace.json"
        ),
        "online_a_depth_source": "monocular_sidecar",
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "arms": list(selected_arms),
        "arm_order": list(order),
        "runtime_role_visibility": "none",
        "shared_history_policy": shared_history_policy,
        "runtime_geometry": runtime_geometry,
        "runtime_navmesh_sha256": runtime_navmesh_sha256,
        "max_steps": args.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "deterministic_plan_seeds": True,
        "smoke": bool(args.smoke),
    }
    (output / "episode_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    common = [
        args.hab_python, "-u",
        str(evaluator_source_root /
            "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(args.bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
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
        "--role_pair_scope", args.role_pair_scope,
        "--navdp_depth_source", "monocular_sidecar",
    ] + pinned_args

    elapsed = {}
    arm_rows = {}
    arm_payloads = {}
    arm_audits = {}
    for arm in order:
        arm_root = output / arm
        arm_root.mkdir()
        command = common + [
            "--hybrid_route", HYBRID_ROUTE[arm],
            "--revisit_adapter", REVISIT_ADAPTER[arm],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, output / "logs" / f"eval_{arm}.log"
        )
        summary = json.loads((arm_root / "summary.json").read_text())
        require(summary.get("queries") == 2,
                f"{arm}: expected one Novel/Revisit query pair")
        require(summary.get("arm") == EVALUATOR_ARM[arm],
                f"{arm}: evaluator arm mismatch")
        require(summary.get("navdp_depth_source") == DEPTH_SOURCE[arm],
                f"{arm}: query depth source changed")
        require(summary.get("runtime_role_visibility") == "none",
                f"{arm}: role leaked into runtime")
        rows = load_rows(arm_root)
        require(len(rows) == 2 and
                {row["analysis_role"] for row in rows} == {"novel", "revisit"},
                f"{arm}: paired role population changed")
        by_role = {row["analysis_role"]: row for row in rows}
        for role, row in by_role.items():
            require(int(row.get("runtime_failure_plans", -1)) == 0,
                    f"{arm}/{role}: certificate runtime failure is not a "
                    "valid policy outcome")
        payloads = load_payloads(arm_root, episode, rows)
        audits = {}
        for role in ("novel", "revisit"):
            payload = payloads[role]
            require(payload.get("analysis_role_not_forwarded") is True,
                    f"{arm}/{role}: role forwarding audit failed")
            audits[role] = audit_query_arm(arm, payload["query_leg"])
        arm_rows[arm] = by_role
        arm_payloads[arm] = payloads
        arm_audits[arm] = audits

    reference_rows = arm_rows["mono_native"]
    reference_payloads = arm_payloads["mono_native"]
    for arm in selected_arms:
        compare_shared_replay(
            reference_rows, reference_payloads,
            arm_rows[arm], arm_payloads[arm], arm,
        )
    fallback = {}
    for role in ("novel", "revisit"):
        fallback[role] = audit_fully_rejected_fallback(
            arm="mono_cec",
            role=role,
            cec_row=arm_rows["mono_cec"][role],
            cec_payload=arm_payloads["mono_cec"][role],
            native_payload=arm_payloads["mono_native"][role],
        )

    completion = {
        **contract,
        "prefix_equality": True,
        "wall_time_seconds": elapsed,
        "outcomes": {
            arm: {role: int(arm_rows[arm][role]["reached"])
                  for role in ("novel", "revisit")}
            for arm in selected_arms
        },
        "final_distance_m": {
            arm: {role: float(arm_rows[arm][role]["final_goal_dist_m"])
                  for role in ("novel", "revisit")}
            for arm in selected_arms
        },
        "geodesic_m": {
            arm: {role: float(arm_rows[arm][role]["geodesic_m"])
                  for role in ("novel", "revisit")}
            for arm in selected_arms
        },
        "path_len_m": {
            arm: {role: float(arm_rows[arm][role]["path_len_m"])
                  for role in ("novel", "revisit")}
            for arm in selected_arms
        },
        "query_steps": {
            arm: {role: int(arm_rows[arm][role]["steps"])
                  for role in ("novel", "revisit")}
            for arm in selected_arms
        },
        "certificate_accept_plans": {
            role: int(arm_rows["mono_cec"][role]["certificate_accept_plans"])
            for role in ("novel", "revisit")
        },
        "fully_rejected_exact_native": fallback,
        "depth_audits": arm_audits,
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    path = output / "completion.json"
    path.write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )
    print(json.dumps({
        "status": "complete", "history_index": args.history_index,
        "scene": scene, "episode": episode,
        "outcomes": completion["outcomes"], "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
