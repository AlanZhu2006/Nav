import math

import numpy as np
import pytest

from MemNavData.monocular_adjacent_motion import PlanarMotionReceipt
from MemNavData.monocular_route_tangent_runtime import (
    accepted_planar_motion,
    build_route_compass,
    historical_route_schedule,
    query_update_schedule,
)
from MemNavData.route_alignment_contract import (
    build_route_alignment_packet,
    route_alignment_executor_source,
    verify_route_alignment_packet,
)


def motion(forward: float, left: float = 0.0, yaw: float = 0.0):
    return PlanarMotionReceipt(
        forward_m=forward,
        left_m=left,
        yaw_rad=yaw,
        vertical_m=0.0,
    )


def test_historical_schedule_uses_only_cached_depth_and_current_query():
    result = historical_route_schedule(
        target_anchor=44,
        goal_start_frame=66,
        cached_depth_frames=[40, 48, 56, 64, 72],
    )
    assert result["nodes_chronological"] == [44, 48, 56, 64, 66]
    assert [edge["depth_reference_frame"] for edge in
            result["edges_chronological"]] == [44, 48, 56, 64]
    assert result["edges_chronological"][-1]["edge_kind"] == (
        "history_to_query_bridge")
    assert result["world_pose_consumed"] is False


def test_early_non_stride_anchor_uses_its_canonical_cec_depth():
    result = historical_route_schedule(
        target_anchor=9,
        goal_start_frame=57,
        cached_depth_frames=[8, 16, 24, 32, 40, 48, 56],
    )
    assert result["nodes_chronological"] == [9, 16, 24, 32, 40, 48, 56, 57]
    assert result["edges_chronological"][0] == {
        "from_frame": 9,
        "to_frame": 16,
        "depth_reference_frame": 9,
        "match_query_frame": 16,
        "invert_estimate": False,
        "edge_kind": "historical_adjacent_sample",
    }
    assert result["anchor_depth_cached"] is False
    assert result["anchor_depth_available"] is True
    assert result["reference_depth_bridge"] is False
    assert result["pre_metric_anchor_bridge"] is False


def test_first_certifiable_stride_anchor_is_a_valid_route_start():
    result = historical_route_schedule(
        target_anchor=8,
        goal_start_frame=41,
        cached_depth_frames=[8, 16, 24, 32, 40],
    )
    assert result["nodes_chronological"] == [8, 16, 24, 32, 40, 41]
    assert result["anchor_depth_cached"] is True


def test_schedule_rejects_anchor_before_cec_boundary():
    with pytest.raises(ValueError, match="certification boundary"):
        historical_route_schedule(
            target_anchor=7,
            goal_start_frame=57,
            cached_depth_frames=[8, 16, 24, 32, 40, 48, 56],
        )


def test_query_schedule_keeps_intermediate_depth_receipts_in_order():
    assert query_update_schedule(
        previous_frame=65,
        current_frame=82,
        cached_depth_frames=[64, 72, 80, 88],
    ) == [72, 80, 82]
    assert query_update_schedule(
        previous_frame=82,
        current_frame=82,
        cached_depth_frames=[80],
    ) == []


def test_local_motion_must_pass_adjacency_proof():
    estimate = {
        "local_motion_validity": {"accepted": True},
        "motion": {
            "forward_m": 0.3,
            "left_m": -0.1,
            "yaw_rad": 0.2,
            "vertical_m": 0.0,
        },
    }
    receipt = accepted_planar_motion(estimate)
    assert receipt.forward_m == pytest.approx(0.3)
    with pytest.raises(RuntimeError, match="rejected"):
        accepted_planar_motion({
            "local_motion_validity": {
                "accepted": False,
                "reason": "minimum_inliers",
            },
            "motion": estimate["motion"],
        })


def test_route_compass_uses_first_local_tangent_at_a_corner():
    # Chronological route: target -> corner -> query.  Reversing it yields a
    # query-origin route whose first direction is straight back to the corner.
    compass = build_route_compass([
        motion(0.0, 0.5, -math.pi / 2.0),
        motion(0.5, 0.0, 0.0),
    ])
    readout = compass.advance_local_se2(
        executed_forward_m=0.0,
        executed_left_m=0.0,
        executed_yaw_rad=0.0,
    )
    assert np.linalg.norm(readout.unit_bearing) == pytest.approx(1.0)
    assert np.linalg.norm(readout.controller_pointgoal) == pytest.approx(2.5)
    assert readout.reference_arc_m <= 0.51


def route_alignment_packet():
    proof = {
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 40,
        "selected_anchor_image_sha256": "a" * 64,
        "certificate": {"accepted": True, "inliers": 20},
        "authority": {"policy": "strict_certificate"},
    }
    return build_route_alignment_packet(
        authority_proof=proof,
        current_frame=64,
        goal_start_frame=64,
        target_anchor=40,
        current_rgb_sha256="b" * 64,
        goal_image_sha256="c" * 64,
        anchor_image_sha256="a" * 64,
        history_edge_receipt_sha256="d" * 64,
        unit_bearing=(-1.0, 0.0),
        tangent_baseline_m=0.30,
        controller_radius_m=2.5,
    )


def test_route_alignment_packet_binds_proof_route_and_rear_turn():
    packet = verify_route_alignment_packet(route_alignment_packet())
    assert packet["guidance_mode"] == "monocular_route_tangent"
    assert abs(abs(packet["required_turn_rad"]) - math.pi) < 1e-12
    assert packet["simulator_pose_consumed"] is False
    assert packet["role_label_visible"] is False
    assert route_alignment_executor_source(packet["packet_sha256"]).endswith(
        packet["packet_sha256"])


def test_route_alignment_packet_rejects_tampered_direction():
    packet = route_alignment_packet()
    packet["unit_bearing"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="digest"):
        verify_route_alignment_packet(packet)


def test_route_alignment_packet_rejects_role_leakage():
    proof = {
        "ok": True,
        "accepted": True,
        "selected_anchor": 40,
        "selected_anchor_image_sha256": "a" * 64,
        "certificate": {"accepted": True},
        "analysis_role": "revisit",
    }
    with pytest.raises(ValueError, match="privileged"):
        build_route_alignment_packet(
            authority_proof=proof,
            current_frame=64,
            goal_start_frame=64,
            target_anchor=40,
            current_rgb_sha256="b" * 64,
            goal_image_sha256="c" * 64,
            anchor_image_sha256="a" * 64,
            history_edge_receipt_sha256="d" * 64,
            unit_bearing=(-1.0, 0.0),
            tangent_baseline_m=0.30,
            controller_radius_m=2.5,
        )
