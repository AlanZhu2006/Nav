import copy
import json
from pathlib import Path

from MemNavData.verify_route_tangent_confirmation import (
    evaluate_gate,
    main,
    sha256,
)


def valid_payloads():
    freeze = {
        "schema_version": "route_tangent_confirmation_freeze_v1_20260903",
        "history_index": 39,
        "analysis_role_not_forwarded": True,
        "navigation_outcome_read": False,
        "sr_read": False,
    }
    query = {
        "schema_version": "local_monocular_adjacent_motion_audit_v2_20260903",
        "history_index": 39,
        "local_motion_chain_intact": True,
        "local_motion_predicted_path_length_m": 34.0,
        "ground_truth_path_length_m_analysis_only": 35.0,
        "local_motion_final_position_error_m_analysis_only": 3.0,
        "runtime_evaluator_pose_visible": False,
        "metric_depth_sensor_consumed": False,
        "global_pose_consumed": False,
        "wheel_odometry_consumed": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
    }
    route = {
        "schema_version": "monocular_route_compass_mechanism_audit_v1_20260903",
        "history_index": 39,
        "history_local_motion_chain_intact": True,
        "query_motion_chain_intact": True,
        "runtime_evaluator_pose_visible": False,
        "metric_depth_sensor_consumed": False,
        "global_pose_consumed": False,
        "wheel_odometry_consumed": False,
        "navigation_controller_executed": False,
        "navigation_sr_computed": False,
        "route_compass": {
            "available": True,
            "predicted_route_extent_m": 34.0,
            "ground_truth_route_extent_m_analysis_only": 35.0,
            "route_endpoint_error_m_analysis_only": 0.5,
        },
    }
    tangent = {
        "schema_version": (
            "tangent_path_budgeted_monocular_route_replay_v1_20260903"),
        "history_index": 39,
        "translated_bearing_count": 10,
        "translated_bearing_within_30deg_count": 8,
        "translated_bearing_error_median_deg_analysis_only": 19.0,
        "translated_bearing_error_p90_deg_analysis_only": 31.0,
        "terminal_turn_bearing_error_median_deg_analysis_only": 8.0,
        "projection_monotone": True,
        "cumulative_path_budget_respected": True,
        "guidance_mode": "route-tangent",
        "live_position_used_for_bearing": False,
        "distance_regime_present": False,
        "visual_gate_present": False,
        "endpoint_fallback_present": False,
        "native_fallback_present": False,
        "rows": [{
            "readout": {
                "controller_pointgoal": [-2.5, 0.0],
                "cross_track_error_m": 2.0,
            },
        }],
    }
    return freeze, query, route, tangent


def test_gate_passes_exact_frozen_boundary():
    result = evaluate_gate(*valid_payloads())
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_gate_fails_bearing_coverage_without_changing_thresholds():
    payloads = list(valid_payloads())
    payloads[3] = copy.deepcopy(payloads[3])
    payloads[3]["translated_bearing_within_30deg_count"] = 7
    result = evaluate_gate(*payloads)
    assert result["passed"] is False
    assert result["checks"][
        "translated_bearing_coverage_at_least_80pct"] is False


def test_absolute_query_drift_is_descriptive_after_tangent_readout():
    payloads = list(valid_payloads())
    payloads[1] = copy.deepcopy(payloads[1])
    payloads[1]["local_motion_final_position_error_m_analysis_only"] = 100.0
    result = evaluate_gate(*payloads)
    assert result["passed"] is True
    assert result["measurements"][
        "query_final_position_error_m_descriptive"] == 100.0


def test_cli_verifies_all_content_addressed_links(tmp_path, monkeypatch):
    freeze, query, route, tangent = valid_payloads()
    query_set = {
        "schema_version": "hm3d_dense_reverse_route_queries_v2_20260903",
        "history_index": 39,
        "runtime_evaluator_pose_visible": False,
    }

    def write(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        return path

    freeze_path = write("freeze.json", freeze)
    query_set_path = write("query_set.json", query_set)
    query["source_query_set_sha256"] = sha256(query_set_path)
    query_path = write("query.json", query)
    route["source_query_set_sha256"] = sha256(query_set_path)
    route["source_query_motion_audit_sha256"] = sha256(query_path)
    route["source_authorization_receipt_sha256"] = sha256(freeze_path)
    route_path = write("route.json", route)
    tangent["source_query_motion_audit_sha256"] = sha256(query_path)
    tangent["source_route_audit_sha256"] = sha256(route_path)
    tangent_path = write("tangent.json", tangent)
    output = tmp_path / "gate"
    protocol = (
        Path(__file__).parent
        / "MONOCULAR_ROUTE_TANGENT_COMPASS_PROTOCOL_20260903.md")
    monkeypatch.setattr("sys.argv", [
        "verify_route_tangent_confirmation.py",
        "--protocol", str(protocol),
        "--freeze", str(freeze_path),
        "--expected-freeze-sha", sha256(freeze_path),
        "--query-set", str(query_set_path),
        "--expected-query-set-sha", sha256(query_set_path),
        "--query-motion", str(query_path),
        "--expected-query-motion-sha", sha256(query_path),
        "--history-route", str(route_path),
        "--expected-history-route-sha", sha256(route_path),
        "--tangent-replay", str(tangent_path),
        "--expected-tangent-replay-sha", sha256(tangent_path),
        "--out", str(output),
    ])
    assert main() == 0
    result = json.loads(
        (output / "route_tangent_confirmation_gate.json").read_text())
    assert result["passed"] is True
    assert result["fresh_closed_loop_construction_authorized"] is True
