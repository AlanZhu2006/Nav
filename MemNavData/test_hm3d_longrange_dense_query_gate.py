import hashlib
import json

import pytest

from MemNavData.analyze_hm3d_longrange_dense_query_gate import build_summary
from MemNavData.run_hm3d_longrange_dense_query_gate_history import (
    dense_sampling_audit,
    discrete_proof,
)


def proof(anchor=11, pose=1.0):
    checks = {
        "status_ok": True,
        "minimum_inliers": True,
        "minimum_query_coverage": True,
        "minimum_reference_coverage": True,
        "maximum_reprojection_rmse": True,
    }
    thresholds = {"min_pnp_inliers": 16}
    return {
        "selected_anchor": anchor,
        "candidate_order_dino": [13, 11],
        "candidate_order_used": [11, 13],
        "proposal_source": "geometry",
        "authority": {
            "policy": "strict_certificate",
            "accepted": True,
            "reason": "certificate_accepted",
            "strict_certificate": {"checks": checks},
        },
        "certificate": {"checks": checks, "thresholds": thresholds},
        "pnp": {"pose9": [pose] * 9},
    }


def plan(frame, previous=None):
    receipts = [] if previous is None else [
        {
            "match_reference_frame": index,
            "match_query_frame": index + 1,
        }
        for index in range(previous, frame)
    ]
    return {
        "frame_idx": frame,
        "certified_relocalization_accepted": True,
        "geometry_stream_stop": False,
        "local_tangent_route_motion_model": "direct_pnp_dense_query",
        "local_tangent_historical_motion_model": "fundamental_then_pnp",
        "local_tangent_live_query_motion_model": "direct_pnp",
        "local_tangent_query_motion_sampling": "per_action_dense",
        "local_tangent_evaluator_pose_consumed": False,
        "local_tangent_executor_odometry_consumed": False,
        "local_tangent_metric_depth_sensor_consumed": False,
        "local_tangent_update_edge_count": len(receipts),
        "local_tangent_update_edge_receipts": receipts,
    }


def test_discrete_proof_ignores_cross_device_pose_floats():
    assert discrete_proof(proof(pose=1.0)) == discrete_proof(proof(pose=2.0))
    assert discrete_proof(proof(anchor=11)) != discrete_proof(proof(anchor=12))


def test_dense_sampling_requires_each_causal_transition():
    audit = dense_sampling_audit([plan(64), plan(72, 64), plan(78, 72)])
    assert audit["completed_query_intervals"] == 2
    assert audit["per_action_dense_intervals"] == 2
    assert audit["query_motion_edges"] == 14
    assert audit["all_completed_intervals_dense"] is True


def test_dense_sampling_rejects_one_coarse_edge():
    rows = [plan(64), plan(72, 64)]
    rows[1]["local_tangent_update_edge_count"] = 1
    rows[1]["local_tangent_update_edge_receipts"] = [
        {"match_reference_frame": 64, "match_query_frame": 72}
    ]
    with pytest.raises(RuntimeError, match="every causal frame transition"):
        dense_sampling_audit(rows)


def test_dense_sampling_serializes_typed_initialization_failure():
    audit = dense_sampling_audit([{
        "geometry_stream_stop": True,
        "local_tangent_route_motion_model": "direct_pnp_dense_query",
        "local_tangent_historical_motion_model": "fundamental_then_pnp",
        "local_tangent_live_query_motion_model": "direct_pnp",
    }])
    assert audit["accepted_route_plans"] == 0
    assert audit["failure_before_or_at_route_initialization"] is True


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n")


def test_summary_recounts_the_frozen_three_history_gate(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "selected_history_indices": [0, 1, 6],
        "claim_scope": "consumed",
    }
    protocol_path.write_text(json.dumps(protocol))
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for index in (0, 1, 6):
        write_json(
            tmp_path / "evaluation" / f"{index:03d}_scene_ep"
            / "completion.json",
            {
                "schema_version": (
                    "hm3d_longrange_dense_live_query_gate_history_v2_20260903"),
                "history_index": index,
                "scene": f"scene{index % 2}",
                "protocol_sha256": protocol_sha,
                "initial_cec_discrete_proof_equal": True,
                "motion_model": "direct_pnp_dense_query",
                "historical_route_motion_model": "fundamental_then_pnp",
                "live_query_motion_model": "direct_pnp",
                "live_query_sampling": "per_action_dense",
                "dense_sampling_audit": {
                    "all_completed_intervals_dense": True,
                    "runtime_role_or_distance_gate_present": False,
                    "query_motion_edges": 8 + index,
                },
                "navigation_claim_allowed": False,
                "fresh_confirmation_required": True,
                "geometry_stream_stop_plans": 0,
                "mechanism_gate_passed": True,
                "outcome_descriptive_only": int(index == 6),
            },
        )
    summary = build_summary(tmp_path, protocol_path)
    assert summary["histories"] == 3
    assert summary["scene_clusters"] == 2
    assert summary["geometry_stream_failures"] == 0
    assert summary["query_motion_edges"] == 31
    assert summary["mechanism_gate_passed"] is True
