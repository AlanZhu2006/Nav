#!/usr/bin/env python3
"""Run the HM3D Table-1 Revisit query under fixed and full-metric CEC.

This is a consumed-population promotion gate, not a fresh confirmation.  Each
history is replayed through one persistent server pair.  The two arms share
the exact Goal-A prefix, goal, certificate stack, seeds, and controller; only
the verified source-to-controller radius differs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from MemNavData.revisit_bearing_adapter import (
    NAVDP_POINTGOAL_RADIUS_MAX_M,
    VERIFIED_BEARING_RADIUS_M,
)


SCHEMA = "hm3d_table1_metric_distance_revisit_pair_v1_20260901"
ARMS = ("fixed_2p5m", "full_metric")
ADAPTER = {
    "fixed_2p5m": "verified_bearing_v1",
    "full_metric": "verified_metric_v1",
}

# These fields record which adapter was requested and how long its rejected
# proof attempt took.  They are expected to differ between arms even when the
# certificate rejects and the exact same native action is executed.  They are
# not part of the fallback control request.
REJECTED_FALLBACK_DIAGNOSTIC_FIELDS = frozenset({
    "certified_relocalization_ms",
    "certified_relocalization_uncached_ms",
    "full_metric_adapter_schema_version",
    "memory_pointgoal_fixed_radius_m",
    "revisit_adapter_mode",
    "router_overlap_uncached_verification_ms",
    "router_overlap_verification_ms",
    "router_verification_total_ms",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> float:
    started = time.perf_counter()
    with log.open("x") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    require(result.returncode == 0,
            f"evaluator failed ({result.returncode}); see {log}")
    return time.perf_counter() - started


def read_arm(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    with (root / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, "arm must contain exactly one Revisit query")
    row = rows[0]
    require(row.get("analysis_role") == "revisit", "query role changed")
    require(int(row.get("runtime_failure_plans", "-1")) == 0,
            "runtime failure is not a policy outcome")
    paths = list(root.glob("*_plans.json"))
    require(len(paths) == 1, "arm must contain exactly one plans artifact")
    payload = json.loads(paths[0].read_text())
    require(payload.get("analysis_role_not_forwarded") is True,
            "runtime received the analysis role")
    return row, payload


def takeovers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        plan for plan in payload["query_leg"]
        if plan.get("revisit_adapter_takeover") is True
    ]


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance,
    )


def rejected_fallback_control_projection(
    plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove arm identity and wall-clock diagnostics from rejected plans.

    The remaining receipt still includes the native request, selected
    trajectory, diffusion seed, certificate rejection, and causal-state
    hashes.  Physical replay is checked separately below.
    """
    projected = copy.deepcopy(plans)
    for plan in projected:
        for field in REJECTED_FALLBACK_DIAGNOSTIC_FIELDS:
            plan.pop(field, None)
        depth_receipt = plan.get("monocular_depth_receipt")
        if isinstance(depth_receipt, dict):
            depth_receipt.pop("first40_scale_freeze_ms", None)
    return projected


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
    parser.add_argument("--history-contract", default="goal_a")
    parser.add_argument("--role-pair-scope", default="consumed_integration")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    require(args.history_contract == "goal_a",
            "metric-distance gate requires the Table-1 Goal-A history")
    require(args.role_pair_scope == "consumed_integration",
            "Revisit-only role filtering is a consumed ablation")
    selected_arms = tuple(
        item.strip() for item in args.arms.split(",") if item.strip()
    )
    require(selected_arms == ARMS, "metric-distance arm set changed")
    require(args.max_steps == (80 if args.smoke else 600),
            "step budget changed")

    source_root = args.source_root.resolve()
    evaluator_root = (
        args.evaluator_source_root or args.source_root
    ).resolve()
    run_root = args.run_root.resolve()
    bench_root = args.bench_root.resolve()
    manifest_path = bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Table-1 manifest changed")
    manifest = json.loads(manifest_path.read_text())
    histories = manifest.get("episodes")
    require(isinstance(histories, list) and len(histories) == 28,
            "expected the frozen 28-history HM3D Table-1 population")
    require(len({str(item["scene"]) for item in histories}) == 21,
            "expected 21 frozen HM3D scene clusters")
    require(0 <= args.history_index < len(histories),
            "history index outside frozen population")

    item = histories[args.history_index]
    require(int(item["online_a_steps"]) >= 40,
            "history cannot produce the first-40 scale receipt")
    scene = str(item["scene"])
    episode = str(item["episode"])
    source_episode = Path(item["online_a_episode"])
    require(sha256(source_episode / "receipt.json")
            == item["online_a_receipt_sha256"],
            "online-A receipt changed")
    require(sha256(source_episode / "online_a_trace.json")
            == item["online_a_trace_sha256"],
            "online-A trace changed")
    receipt = json.loads((source_episode / "receipt.json").read_text())
    trace = json.loads((source_episode / "online_a_trace.json").read_text())
    control = receipt.get("online_a_control_audit")
    require(isinstance(control, dict) and control.get("ok") is True,
            "Goal-A was not generated under audited online control")
    require(all(plan.get("navdp_depth_source") == "monocular_sidecar"
                for plan in trace["plans"]),
            "Goal-A contains a non-monocular plan")
    require(all(plan.get("metric_depth_sensor_consumed") is False
                for plan in trace["plans"]),
            "Goal-A consumed simulator metric depth")
    scene_file = Path(receipt["source_asset"])
    require(scene_file.is_file() and
            sha256(scene_file) == receipt["source_asset_sha256"],
            "HM3D source asset changed")

    label = f"{args.history_index:03d}_{scene}_{episode}"
    output = run_root / "evaluation" / "revisit_metric_distance" / label
    require(not output.exists(), f"history output exists: {output}")
    (output / "logs").mkdir(parents=True)
    order = ARMS if args.history_index % 2 == 0 else tuple(reversed(ARMS))
    contract = {
        "schema_version": SCHEMA,
        "scope": "consumed HM3D Table-1 Revisit-only promotion gate",
        "fresh_confirmation": False,
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "online_a_steps": int(item["online_a_steps"]),
        "online_a_trace_sha256": item["online_a_trace_sha256"],
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "arms": list(ARMS),
        "arm_order": list(order),
        "same_server_pair": True,
        "runtime_role_visibility": "none",
        "analysis_query_role": "revisit",
        "depth_source": "monocular_sidecar",
        "max_steps": args.max_steps,
        "success_distance_m": 1.0,
        "exec_horizon": 8,
        "metric_radius_support_cap_m": NAVDP_POINTGOAL_RADIUS_MAX_M,
        "smoke": bool(args.smoke),
    }
    (output / "episode_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    common = [
        args.hab_python, "-u",
        str(evaluator_root / "MemNavData/eval_shared_online_role_pairs.py"),
        "--episode_root", str(bench_root / scene),
        "--episode_ids", episode,
        "--scene", str(scene_file),
        "--scene_identity", scene,
        "--host", "127.0.0.1",
        "--port", str(args.memnav_port),
        "--novel_port", str(args.navdp_port),
        "--server_backend", "hybrid_pose",
        "--hybrid_route", "certified_relocalization",
        "--navdp_depth_source", "monocular_sidecar",
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
        "--role_pair_query_role", "revisit",
    ]

    elapsed: dict[str, float] = {}
    rows: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for arm in order:
        arm_root = output / arm
        arm_root.mkdir()
        elapsed[arm] = run(
            common + [
                "--revisit_adapter", ADAPTER[arm],
                "--out", str(arm_root),
            ],
            output / "logs" / f"eval_{arm}.log",
        )
        rows[arm], payloads[arm] = read_arm(arm_root)

    fixed_payload = payloads["fixed_2p5m"]
    metric_payload = payloads["full_metric"]
    require(fixed_payload["legA"] == metric_payload["legA"],
            "paired arms changed sealed Goal-A plans")
    require(fixed_payload["rollout_traces"]["legA"]
            == metric_payload["rollout_traces"]["legA"],
            "paired arms changed Goal-A physical replay")
    require(fixed_payload["memory_traces"]["legA"]
            == metric_payload["memory_traces"]["legA"],
            "paired arms changed causal memory replay")

    fixed_takeovers = takeovers(fixed_payload)
    metric_takeovers = takeovers(metric_payload)
    require(bool(fixed_takeovers) == bool(metric_takeovers),
            "metric-scale availability changed first-step authorization")
    first_receipt: dict[str, Any] | None = None
    if fixed_takeovers:
        fixed = fixed_takeovers[0]
        metric = metric_takeovers[0]
        for field in (
            "router_selected_anchor",
            "memory_unbounded_pointgoal",
            "memory_bearing_unit",
            "certified_relocalization_certificate",
        ):
            require(fixed.get(field) == metric.get(field),
                    f"first takeover proof drifted in {field}")
        require(close(
            fixed["memory_controller_pointgoal_distance_m"],
            VERIFIED_BEARING_RADIUS_M,
        ), "canonical arm no longer emits 2.5 m")
        scale = float(metric["memory_metric_scale_m_per_raw"])
        raw_norm = float(metric["memory_unbounded_pointgoal_norm"])
        unbounded = float(metric["memory_unbounded_pointgoal_distance_m"])
        controller = float(metric["memory_controller_pointgoal_distance_m"])
        require(scale > 0.0 and close(unbounded, scale * raw_norm),
                "metric arm did not expose first-40 metric distance")
        require(close(controller, min(
            unbounded, NAVDP_POINTGOAL_RADIUS_MAX_M)),
            "metric controller radius violates its native support contract")
        first_receipt = {
            "anchor": metric["router_selected_anchor"],
            "bearing": metric["memory_bearing_unit"],
            "metric_scale_m_per_raw": scale,
            "raw_norm": raw_norm,
            "unbounded_metric_distance_m": unbounded,
            "controller_distance_m": controller,
            "differs_from_fixed_2p5m": not close(
                controller, VERIFIED_BEARING_RADIUS_M),
        }
    else:
        require(
            fixed_payload["query_result"] == metric_payload["query_result"],
            "fully rejected arms changed the native query result",
        )
        require(
            fixed_payload["rollout_traces"]["query"]
            == metric_payload["rollout_traces"]["query"],
            "fully rejected arms changed the physical fallback replay",
        )
        require(
            rejected_fallback_control_projection(fixed_payload["query_leg"])
            == rejected_fallback_control_projection(
                metric_payload["query_leg"]
            ),
            "fully rejected arms changed the native control receipt",
        )

    metric_distances = [
        float(plan["memory_controller_pointgoal_distance_m"])
        for plan in metric_takeovers
    ]
    cap_hits = sum(close(
        plan["memory_controller_pointgoal_distance_m"],
        NAVDP_POINTGOAL_RADIUS_MAX_M,
    ) and float(plan["memory_unbounded_pointgoal_distance_m"])
        > NAVDP_POINTGOAL_RADIUS_MAX_M
        for plan in metric_takeovers)
    completion = {
        **contract,
        "prefix_equality": True,
        "wall_time_seconds": elapsed,
        "outcomes": {arm: int(rows[arm]["reached"]) for arm in ARMS},
        "final_distance_m": {
            arm: float(rows[arm]["final_goal_dist_m"]) for arm in ARMS
        },
        "path_len_m": {
            arm: float(rows[arm]["path_len_m"]) for arm in ARMS
        },
        "steps": {arm: int(rows[arm]["steps"]) for arm in ARMS},
        "geodesic_m": {
            arm: float(rows[arm]["geodesic_m"]) for arm in ARMS
        },
        "takeover_plans": {
            "fixed_2p5m": len(fixed_takeovers),
            "full_metric": len(metric_takeovers),
        },
        "first_takeover": first_receipt,
        "metric_controller_distances_m": metric_distances,
        "metric_native_support_cap_hits": cap_hits,
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    (output / "completion.json").write_bytes(encoded)
    (output / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )
    print(json.dumps({
        "status": "complete",
        "history_index": args.history_index,
        "scene": scene,
        "episode": episode,
        "outcomes": completion["outcomes"],
        "first_takeover": first_receipt,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ABORT: {error}", file=sys.stderr)
        raise
