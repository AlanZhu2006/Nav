#!/usr/bin/env python3
"""Apply the frozen route-tangent mechanism gate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.monocular_route_tangent_contract import (
    CONTROLLER_RADIUS_M,
    MAX_ABSOLUTE_PATH_LENGTH_BIAS_FRACTION,
    MAX_HISTORY_ENDPOINT_ERROR_M,
    MAX_TRANSLATED_BEARING_MEDIAN_DEG,
    MIN_TRANSLATED_BEARING_WITHIN_30_FRACTION,
    ROUTE_TANGENT_CONFIRMATION_HISTORY_INDEX,
)


SCHEMA_VERSION = "route_tangent_confirmation_gate_v1_20260903"
FREEZE_SCHEMA_VERSION = "route_tangent_confirmation_freeze_v1_20260903"
QUERY_SCHEMA_VERSION = "local_monocular_adjacent_motion_audit_v2_20260903"
ROUTE_SCHEMA_VERSION = "monocular_route_compass_mechanism_audit_v1_20260903"
TANGENT_SCHEMA_VERSION = (
    "tangent_path_budgeted_monocular_route_replay_v1_20260903"
)
QUERY_SET_SCHEMA_VERSION = "hm3d_dense_reverse_route_queries_v2_20260903"
FROZEN_PROTOCOL_SHA256 = (
    "4200787139e73c9d27354292eccd25a8337282a31a8f1c49ae78fceba7c9a3d4"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def relative_bias(predicted: float, reference: float) -> float:
    require(finite(predicted) and finite(reference) and float(reference) > 0.0,
            "path length is absent or invalid")
    return (float(predicted) - float(reference)) / float(reference)


def evaluate_gate(
    freeze: dict[str, Any],
    query: dict[str, Any],
    route: dict[str, Any],
    tangent: dict[str, Any],
) -> dict[str, Any]:
    index = ROUTE_TANGENT_CONFIRMATION_HISTORY_INDEX
    require(freeze.get("schema_version") == FREEZE_SCHEMA_VERSION,
            "wrong confirmation-freeze schema")
    require(query.get("schema_version") == QUERY_SCHEMA_VERSION,
            "wrong query-motion schema")
    require(route.get("schema_version") == ROUTE_SCHEMA_VERSION,
            "wrong history-route schema")
    require(tangent.get("schema_version") == TANGENT_SCHEMA_VERSION,
            "wrong route-tangent schema")
    require(all(int(payload.get("history_index", -1)) == index
                for payload in (freeze, query, route, tangent)),
            "gate inputs do not belong to the frozen confirmation history")

    route_summary = route.get("route_compass")
    require(isinstance(route_summary, dict), "history route has no summary")
    translated_count = int(tangent.get("translated_bearing_count", 0))
    translated_within = int(
        tangent.get("translated_bearing_within_30deg_count", -1))
    require(translated_count > 0 and 0 <= translated_within <= translated_count,
            "translated bearing population is invalid")

    query_path_bias = relative_bias(
        float(query["local_motion_predicted_path_length_m"]),
        float(query["ground_truth_path_length_m_analysis_only"]),
    )
    history_path_bias = relative_bias(
        float(route_summary["predicted_route_extent_m"]),
        float(route_summary["ground_truth_route_extent_m_analysis_only"]),
    )
    controller_norms = [
        float(np.linalg.norm(np.asarray(
            row["readout"]["controller_pointgoal"], dtype=np.float64)))
        for row in tangent.get("rows", [])
    ]
    require(controller_norms, "route-tangent replay has no controller payloads")

    no_forbidden_runtime_inputs = all([
        query.get("runtime_evaluator_pose_visible") is False,
        query.get("metric_depth_sensor_consumed") is False,
        query.get("global_pose_consumed") is False,
        query.get("wheel_odometry_consumed") is False,
        query.get("navigation_controller_executed") is False,
        query.get("navigation_sr_computed") is False,
        route.get("runtime_evaluator_pose_visible") is False,
        route.get("metric_depth_sensor_consumed") is False,
        route.get("global_pose_consumed") is False,
        route.get("wheel_odometry_consumed") is False,
        route.get("navigation_controller_executed") is False,
        route.get("navigation_sr_computed") is False,
        freeze.get("analysis_role_not_forwarded") is True,
        freeze.get("navigation_outcome_read") is False,
        freeze.get("sr_read") is False,
    ])
    checks = {
        "history_motion_chain_complete": (
            route.get("history_local_motion_chain_intact") is True),
        "query_motion_chain_complete": (
            query.get("local_motion_chain_intact") is True
            and route.get("query_motion_chain_intact") is True),
        "no_forbidden_runtime_inputs": no_forbidden_runtime_inputs,
        "query_path_length_bias_within_15pct": (
            abs(query_path_bias)
            <= MAX_ABSOLUTE_PATH_LENGTH_BIAS_FRACTION),
        "history_path_length_bias_within_15pct": (
            abs(history_path_bias)
            <= MAX_ABSOLUTE_PATH_LENGTH_BIAS_FRACTION),
        "history_endpoint_error_within_1p25m": (
            route_summary.get("available") is True
            and finite(route_summary.get(
                "route_endpoint_error_m_analysis_only"))
            and float(route_summary[
                "route_endpoint_error_m_analysis_only"])
            <= MAX_HISTORY_ENDPOINT_ERROR_M),
        "translated_bearing_coverage_at_least_80pct": (
            translated_within / translated_count
            >= MIN_TRANSLATED_BEARING_WITHIN_30_FRACTION),
        "translated_bearing_median_at_most_20deg": (
            finite(tangent.get(
                "translated_bearing_error_median_deg_analysis_only"))
            and float(tangent[
                "translated_bearing_error_median_deg_analysis_only"])
            <= MAX_TRANSLATED_BEARING_MEDIAN_DEG),
        "projection_monotone_and_path_budgeted": (
            tangent.get("projection_monotone") is True
            and tangent.get("cumulative_path_budget_respected") is True),
        "controller_radius_exactly_2p5m": all(
            abs(value - CONTROLLER_RADIUS_M) <= 1e-9
            for value in controller_norms),
        "no_regime_gate_or_fallback": all([
            tangent.get("guidance_mode") == "route-tangent",
            tangent.get("live_position_used_for_bearing") is False,
            tangent.get("distance_regime_present") is False,
            tangent.get("visual_gate_present") is False,
            tangent.get("endpoint_fallback_present") is False,
            tangent.get("native_fallback_present") is False,
        ]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "measurements": {
            "query_path_length_bias_fraction": query_path_bias,
            "history_path_length_bias_fraction": history_path_bias,
            "history_endpoint_error_m": route_summary.get(
                "route_endpoint_error_m_analysis_only"),
            "translated_bearing_within_30deg_count": translated_within,
            "translated_bearing_count": translated_count,
            "translated_bearing_within_30deg_fraction": (
                translated_within / translated_count),
            "translated_bearing_error_median_deg": tangent.get(
                "translated_bearing_error_median_deg_analysis_only"),
            "translated_bearing_error_p90_deg": tangent.get(
                "translated_bearing_error_p90_deg_analysis_only"),
            "query_final_position_error_m_descriptive": query.get(
                "local_motion_final_position_error_m_analysis_only"),
            "query_cross_track_error_final_m_descriptive": (
                tangent["rows"][-1]["readout"].get("cross_track_error_m")),
            "terminal_turn_bearing_error_median_deg_descriptive": tangent.get(
                "terminal_turn_bearing_error_median_deg_analysis_only"),
            "controller_norm_min_m": min(controller_norms),
            "controller_norm_max_m": max(controller_norms),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha", required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--expected-query-set-sha", required=True)
    parser.add_argument("--query-motion", type=Path, required=True)
    parser.add_argument("--expected-query-motion-sha", required=True)
    parser.add_argument("--history-route", type=Path, required=True)
    parser.add_argument("--expected-history-route-sha", required=True)
    parser.add_argument("--tangent-replay", type=Path, required=True)
    parser.add_argument("--expected-tangent-replay-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(not arguments.out.exists(), "gate output already exists")
    require(sha256(arguments.protocol) == FROZEN_PROTOCOL_SHA256,
            "frozen route-tangent protocol changed")
    for path, expected, label in [
        (arguments.freeze, arguments.expected_freeze_sha, "freeze receipt"),
        (arguments.query_set, arguments.expected_query_set_sha, "query set"),
        (arguments.query_motion, arguments.expected_query_motion_sha,
         "query-motion receipt"),
        (arguments.history_route, arguments.expected_history_route_sha,
         "history-route receipt"),
        (arguments.tangent_replay, arguments.expected_tangent_replay_sha,
         "route-tangent replay"),
    ]:
        require(sha256(path) == expected, f"{label} changed")

    freeze = json.loads(arguments.freeze.read_text(encoding="utf-8"))
    query_set = json.loads(arguments.query_set.read_text(encoding="utf-8"))
    query = json.loads(arguments.query_motion.read_text(encoding="utf-8"))
    route = json.loads(arguments.history_route.read_text(encoding="utf-8"))
    tangent = json.loads(arguments.tangent_replay.read_text(encoding="utf-8"))
    require(query_set.get("schema_version") == QUERY_SET_SCHEMA_VERSION
            and int(query_set.get("history_index", -1))
            == ROUTE_TANGENT_CONFIRMATION_HISTORY_INDEX
            and query_set.get("runtime_evaluator_pose_visible") is False,
            "wrong or information-leaking dense query set")
    require(query.get("source_query_set_sha256")
            == sha256(arguments.query_set)
            and route.get("source_query_set_sha256")
            == sha256(arguments.query_set),
            "motion receipts used another dense query set")
    require(route.get("source_query_motion_audit_sha256")
            == sha256(arguments.query_motion),
            "history route used another query-motion receipt")
    require(route.get("source_authorization_receipt_sha256")
            == sha256(arguments.freeze),
            "history route used another CEC authorization receipt")
    require(tangent.get("source_route_audit_sha256")
            == sha256(arguments.history_route)
            and tangent.get("source_query_motion_audit_sha256")
            == sha256(arguments.query_motion),
            "route-tangent replay used another motion receipt")

    gate = evaluate_gate(freeze, query, route, tangent)
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "preregistered single-history mechanism confirmation; authorizes "
            "fresh paired closed-loop construction only, never a paper SR claim"
        ),
        "history_index": ROUTE_TANGENT_CONFIRMATION_HISTORY_INDEX,
        "source_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "source_freeze_sha256": sha256(arguments.freeze),
        "source_query_set_sha256": sha256(arguments.query_set),
        "source_query_motion_sha256": sha256(arguments.query_motion),
        "source_history_route_sha256": sha256(arguments.history_route),
        "source_tangent_replay_sha256": sha256(arguments.tangent_replay),
        "thresholds": {
            "max_absolute_path_length_bias_fraction": (
                MAX_ABSOLUTE_PATH_LENGTH_BIAS_FRACTION),
            "max_history_endpoint_error_m": MAX_HISTORY_ENDPOINT_ERROR_M,
            "min_translated_bearing_within_30_fraction": (
                MIN_TRANSLATED_BEARING_WITHIN_30_FRACTION),
            "max_translated_bearing_median_deg": (
                MAX_TRANSLATED_BEARING_MEDIAN_DEG),
            "controller_radius_m": CONTROLLER_RADIUS_M,
        },
        **gate,
        "fresh_closed_loop_construction_authorized": bool(gate["passed"]),
        "closed_loop_sr_computed": False,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = arguments.out / "route_tangent_confirmation_gate.json"
    output.write_bytes(encoded)
    (arguments.out / "route_tangent_confirmation_gate.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  route_tangent_confirmation_gate.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass" if gate["passed"] else "fail",
        "history_index": ROUTE_TANGENT_CONFIRMATION_HISTORY_INDEX,
        "checks": gate["checks"],
        "out": str(output),
    }, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
