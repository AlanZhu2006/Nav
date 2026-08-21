#!/usr/bin/env python3
"""Run one consumed Final14 history under the five depth/controller arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from MemNavData.final14_mono_factorial import (
    ARMS,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    audit_depth_plans,
    require,
    rotated_arm_order,
)


SCHEMA = "final14_mono_factorial_episode_v1_20260819"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], log_path: Path) -> float:
    start = time.perf_counter()
    with log_path.open("x") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    require(result.returncode == 0,
            f"evaluator failed ({result.returncode}); see {log_path}")
    return elapsed


def load_rows(root: Path) -> list[dict[str, str]]:
    with (root / "metric.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_payloads(root: Path, episode: str,
                  rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = row["analysis_role"]
        path = root / f"{episode}_{row['query_id']}_plans.json"
        require(path.is_file(), f"missing query plans: {path}")
        output[role] = json.loads(path.read_text())
    return output


def compare_shared_replay(
    reference_rows: dict[str, dict[str, str]],
    reference_payloads: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, str]],
    payloads: dict[str, dict[str, Any]],
    arm: str,
) -> None:
    fields = (
        "scene", "episode", "pair_id", "query_id", "analysis_role", "seed",
        "shared_A_frames", "shared_A_decision_frames", "geodesic_m",
    )
    replay_fields = (
        "all_rgb_hashes_verified", "decision_frames", "decision_steps",
        "diffusion_samples_during_replay", "navdp_memory_size",
        "navdp_queue_lengths", "online_frames",
    )
    for role in ("novel", "revisit"):
        left_row = reference_rows[role]
        right_row = rows[role]
        for field in fields:
            require(left_row[field] == right_row[field],
                    f"{arm}/{role}: paired field {field} changed")
        left = reference_payloads[role]
        right = payloads[role]
        require(left["legA"] == right["legA"],
                f"{arm}/{role}: frozen Goal-A plans changed")
        require(left["rollout_traces"]["legA"] ==
                right["rollout_traces"]["legA"],
                f"{arm}/{role}: frozen Goal-A rollout changed")
        require(left["memory_traces"]["legA"] ==
                right["memory_traces"]["legA"],
                f"{arm}/{role}: replayed MemNav history changed")
        for field in replay_fields:
            require(left["replay"][field] == right["replay"][field],
                    f"{arm}/{role}: replay field {field} changed")


def audit_fully_rejected_fallback(
    *,
    arm: str,
    role: str,
    cec_row: dict[str, str],
    cec_payload: dict[str, Any],
    native_payload: dict[str, Any],
) -> bool:
    if int(cec_row["certificate_accept_plans"]) > 0:
        return False
    cec_plans = cec_payload["query_leg"]
    native_plans = native_payload["query_leg"]
    require(len(cec_plans) == len(native_plans),
            f"{arm}/{role}: fully rejected plan count differs from native")
    fields = (
        "requested_diffusion_seed", "diffusion_seed",
        "selected_trajectory_sha256",
    )
    for index, (cec_plan, native_plan) in enumerate(
            zip(cec_plans, native_plans)):
        for field in fields:
            require(cec_plan.get(field) == native_plan.get(field),
                    f"{arm}/{role}/plan{index}: reject field {field} changed")
    require(cec_payload["rollout_traces"]["query"] ==
            native_payload["rollout_traces"]["query"],
            f"{arm}/{role}: fully rejected physical trace differs")
    return True


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
        "scope": "consumed_final14_query_controller_depth_attribution",
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
        "intervention": "query_controller_depth_x_cec_authorization",
        "max_steps": args.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "deterministic_plan_seeds": True,
        "smoke": bool(args.smoke),
    }
    (episode_root / "episode_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

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
            "--revisit_adapter", REVISIT_ADAPTER[arm],
            "--navdp_depth_source", DEPTH_SOURCE[arm],
            "--out", str(arm_root),
        ]
        elapsed[arm] = run_command(
            command, episode_root / "logs" / f"eval_{arm}.log"
        )
        summary = json.loads((arm_root / "summary.json").read_text())
        require(summary.get("queries") == 2,
                f"{arm}: evaluator query count changed")
        require(summary.get("arm") == EVALUATOR_ARM[arm],
                f"{arm}: evaluator arm mismatch")
        require(summary.get("navdp_depth_source") == DEPTH_SOURCE[arm],
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
            audits[role] = audit_depth_plans(arm, payload["query_leg"])
        arm_rows[arm] = by_role
        arm_payloads[arm] = payloads
        arm_audits[arm] = audits

    reference_rows = arm_rows["mono_native"]
    reference_payloads = arm_payloads["mono_native"]
    for arm in ARMS:
        compare_shared_replay(
            reference_rows, reference_payloads,
            arm_rows[arm], arm_payloads[arm], arm,
        )

    fallback: dict[str, dict[str, bool]] = {}
    for cec_arm, native_arm in (
        ("mono_cec", "mono_native"),
        ("metric_cec", "metric_native"),
    ):
        fallback[cec_arm] = {}
        for role in ("novel", "revisit"):
            fallback[cec_arm][role] = audit_fully_rejected_fallback(
                arm=cec_arm,
                role=role,
                cec_row=arm_rows[cec_arm][role],
                cec_payload=arm_payloads[cec_arm][role],
                native_payload=arm_payloads[native_arm][role],
            )

    completion = {
        **contract,
        "prefix_equality": True,
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
        "certificate_accept_plans": {
            arm: {
                role: int(arm_rows[arm][role]["certificate_accept_plans"])
                for role in ("novel", "revisit")
            }
            for arm in ("mono_cec", "metric_cec")
        },
        "fully_rejected_exact_native": fallback,
        "depth_audits": arm_audits,
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    (episode_root / "completion.json").write_bytes(encoded)
    (episode_root / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "outcomes": completion["outcomes"],
        "output": str(episode_root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
