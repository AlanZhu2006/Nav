#!/usr/bin/env python3
"""Independently verify a learned Pi3X closed-loop transport smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_metric(root: Path, arm: str) -> dict[str, str]:
    with (root / arm / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{arm} must contain exactly one metric row")
    return rows[0]


def _as_int(value: Any) -> int:
    return int(float(value or 0))


def _as_bool(value: Any) -> bool:
    return float(value) > 0.5


def verify(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt_line = (root / "receipt.json.sha256").read_text().split()
    if not receipt_line or receipt_line[0] != _sha256(receipt_path):
        raise ValueError("receipt SHA does not match receipt.json")
    schema = receipt.get("schema_version")
    if schema not in {
        "pi3x_learned_closed_loop_smoke_v1_20260817",
        "pi3x_learned_closed_loop_smoke_v2_20260817",
    }:
        raise ValueError("unexpected receipt schema")
    if receipt.get("passed") is not True:
        raise ValueError("runner did not mark the transport smoke passed")
    source_bundle = receipt.get("source_bundle")
    if schema.endswith("v2_20260817"):
        if not isinstance(source_bundle, dict):
            raise ValueError("v2 receipt omitted source bundle identity")
        if (source_bundle.get("manifest_sha256")
                != args.expected_source_receipt_sha256):
            raise ValueError("source bundle receipt differs")
    elif args.expected_source_receipt_sha256 is not None:
        raise ValueError("v1 result cannot prove source bundle identity")
    if (args.expected_goal_b_image_sha256 is not None
            and receipt.get("goal_b_image_sha256")
            != args.expected_goal_b_image_sha256):
        raise ValueError("Goal-B image identity differs")

    trace = _one_metric(root, "trace_source")
    native = _one_metric(root, "native")
    learned = _one_metric(root, "learned_pi3x")
    episode = receipt["episode_id"]
    if any(row.get("episode") != episode for row in (trace, native, learned)):
        raise ValueError("episode identity differs across arms")
    trace_sha = trace.get("leg1_trace_sha256")
    if (not trace_sha or trace_sha != receipt.get("shared_goal_a_trace_sha256")
            or any(row.get("leg1_trace_sha256") != trace_sha
                   for row in (native, learned))):
        raise ValueError("shared Goal-A trace identity differs")
    if not _as_bool(trace["reached_A"]):
        raise ValueError("Goal A did not succeed")
    if any(row["reached_A"] != trace["reached_A"]
           for row in (native, learned)):
        raise ValueError("Goal-A outcome differs across arms")
    if _as_int(native["steps_B"]) <= 0 or _as_int(learned["steps_B"]) <= 0:
        raise ValueError("a Goal-B arm did not execute")

    requests = _as_int(
        learned["learned_pi3x_relocalization_request_count"])
    initial = _as_int(
        learned["learned_pi3x_relocalization_initial_inference_count"])
    accepts = _as_int(learned["learned_pi3x_relocalization_accept_count"])
    failures = _as_int(
        learned["learned_pi3x_relocalization_runtime_failure_count"])
    takeovers = _as_int(learned["revisit_adapter_takeover_plan_count"])
    if requests <= 0 or initial != 1 or failures != 0:
        raise ValueError("learned runtime request lifecycle is invalid")
    if args.require_accept and (accepts <= 0 or accepts != takeovers):
        raise ValueError("accepted plans did not all reach the adapter")
    if _as_int(learned["certified_relocalization_request_count"]) != 0:
        raise ValueError("hand-written certificate was invoked")

    summary = json.loads((root / "learned_pi3x" / "summary.json").read_text())
    contract = summary.get("learned_pi3x_relocalization_server")
    if not isinstance(contract, dict) or contract.get("enabled") is not True:
        raise ValueError("learned server contract is missing")
    expected_contract = {
        "method": "dino_top8_pi3x_b16_spatial_proof_v1",
        "model_sha256": args.expected_model_sha256,
        "proof_manifest_sha256": args.expected_proof_manifest_sha256,
        "candidate_top_k": 8,
        "candidate_min_gap": 4,
        "bridge_frames": 16,
        "spatial_proof_member_count": 4,
        "spatial_proof_consensus_required": 2,
        "controller_adapter": "verified_bearing_v1_fixed_2.5m",
        "fallback": "native_imagegoal",
        "certificate_components_consumed": False,
        "simulator_pose_or_depth_consumed": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"server contract differs at {key}")

    plans_payload = json.loads(
        (root / "learned_pi3x" / f"{episode}_plans.json").read_text()
    )
    plans = plans_payload.get("legB")
    if not isinstance(plans, list) or len(plans) != requests:
        raise ValueError("learned plan count differs from metric")
    selected = []
    for index, plan in enumerate(plans):
        if plan.get("learned_pi3x_relocalization_ok") is not True:
            raise ValueError("learned plan is not runtime-valid")
        cached = plan.get("learned_pi3x_initial_candidate_selection_cached")
        considered = plan.get("router_candidates_considered")
        if index == 0:
            if cached is not False or considered != 8:
                raise ValueError("first plan did not evaluate frozen top-8")
        elif accepts > 0:
            if cached is not True or considered != 1:
                raise ValueError("tracking plan did not reuse one fixed anchor")
        elif cached is not True or considered != 8:
            raise ValueError("sticky abstention did not reuse the initial proof")
        if plan.get("router_candidate_pool_size") != 8:
            raise ValueError("DINO candidate pool changed")
        if plan.get("learned_pi3x_relocalization_accepted") is True:
            if (plan.get("learned_pi3x_pointgoal_units")
                    != "pi3x_current_camera_direction_only"):
                raise ValueError("accepted bearing units changed")
            if plan.get("revisit_adapter_takeover") is not True:
                raise ValueError("accepted plan did not take over")
            radius = float(plan.get("memory_controller_pointgoal_distance_m"))
            norm = float(plan.get("memory_unbounded_pointgoal_norm"))
            if (not math.isclose(radius, 2.5, abs_tol=1e-9)
                    or not math.isclose(norm, 1.0, abs_tol=1e-9)):
                raise ValueError("scale-free bearing adapter changed")
            selected.append((
                int(plan["learned_pi3x_selected_anchor"]),
                int(plan["learned_pi3x_selected_dino_rank"]),
            ))
    if len(selected) != accepts:
        raise ValueError("accepted plan count differs from metric")
    if selected and len(set(selected)) != 1:
        raise ValueError("selected anchor changed after authorization")

    exact_native_fallback = None
    if args.require_exact_native_fallback:
        if accepts != 0 or takeovers != 0:
            raise ValueError("fallback-equivalence run unexpectedly took over")
        native_payload = json.loads(
            (root / "native" / f"{episode}_plans.json").read_text()
        )
        native_plans = native_payload.get("legB")
        native_rollout = native_payload.get("legB_rollout_trace")
        learned_rollout = plans_payload.get("legB_rollout_trace")
        if not isinstance(native_plans, list) or len(native_plans) != len(plans):
            raise ValueError("native/learned plan counts differ under abstention")
        if native_rollout != learned_rollout:
            raise ValueError("native/learned rollout traces differ under abstention")
        equality_keys = (
            "step",
            "requested_diffusion_seed",
            "diffusion_seed",
            "server_selected_idx",
            "trajectory_candidate_count",
            "selected_trajectory_sha256",
        )
        if any(
            native_plan.get(key) != learned_plan.get(key)
            for native_plan, learned_plan in zip(native_plans, plans)
            for key in equality_keys
        ):
            raise ValueError("native NavDP plans differ under learned abstention")
        exact_native_fallback = True

    runner_learned = receipt["learned_pi3x"]
    if (runner_learned["request_count"] != requests
            or runner_learned["accept_count"] != accepts
            or runner_learned["runtime_failure_count"] != failures
            or runner_learned["takeover_plan_count"] != takeovers):
        raise ValueError("runner receipt differs from raw metrics")

    report = {
        "schema_version": "pi3x_learned_closed_loop_verification_v1_20260817",
        "verified": True,
        "scope": receipt["scope"],
        "run_root": str(root),
        "receipt_sha256": _sha256(receipt_path),
        "episode_id": episode,
        "shared_goal_a_trace_sha256": trace_sha,
        "requests": requests,
        "initial_top8_inferences": initial,
        "accepts": accepts,
        "runtime_failures": failures,
        "takeovers": takeovers,
        "fixed_selected_anchor_and_rank": (
            list(selected[0]) if selected else None),
        "exact_native_fallback_verified": exact_native_fallback,
        "native_reached_b": _as_bool(native["reached_B"]),
        "learned_reached_b": _as_bool(learned["reached_B"]),
        "native_steps_b": _as_int(native["steps_B"]),
        "learned_steps_b": _as_int(learned["steps_B"]),
        "model_sha256": args.expected_model_sha256,
        "proof_manifest_sha256": args.expected_proof_manifest_sha256,
        "goal_b_image_sha256": receipt.get("goal_b_image_sha256"),
    }
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-proof-manifest-sha256", required=True)
    parser.add_argument("--expected-source-receipt-sha256")
    parser.add_argument("--expected-goal-b-image-sha256")
    parser.add_argument("--require-accept", action="store_true")
    parser.add_argument("--require-exact-native-fallback", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(verify(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
