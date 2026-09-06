import copy
import hashlib
import json
import math

import pytest

from MemNavData.hm3d_longrange_rear_alignment import (
    SELECTED_HISTORY_INDICES,
    audit_rear_aligned,
    audit_unaligned,
    rotated_arm_order,
)
from MemNavData.route_alignment_contract import (
    build_route_alignment_packet,
    route_alignment_executor_source,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _packet():
    return build_route_alignment_packet(
        authority_proof={
            "ok": True,
            "accepted": True,
            "selected_anchor": 8,
            "selected_anchor_image_sha256": _sha("anchor"),
            "certificate": {"accepted": True},
            "authority": "strict_certificate",
            "pnp": {"ok": True},
        },
        current_frame=48,
        goal_start_frame=40,
        target_anchor=8,
        current_rgb_sha256=_sha("current"),
        goal_image_sha256=_sha("goal"),
        anchor_image_sha256=_sha("anchor"),
        history_edge_receipt_sha256=_sha("route"),
        unit_bearing=(-1.0, 0.0),
        tangent_baseline_m=0.3,
        controller_radius_m=2.5,
    )


def _plan(packet, *, consumed=False):
    edges = []
    if consumed:
        edges = [{
            "edge_kind": "proof_bound_atomic_yaw_control",
            "frame_index": 49 + index,
            "packet_sha256": packet["packet_sha256"],
            "executed_yaw_rad": math.radians(30.0),
            "translation_m": 0.0,
        } for index in range(6)]
    return {
        "certified_relocalization_accepted": True,
        "certified_relocalization_guidance_mode": "monocular_route_tangent",
        "local_tangent_alignment_packet": packet,
        "local_tangent_alignment_packet_sha256": packet["packet_sha256"],
        "local_tangent_target_anchor": 8,
        "local_tangent_history_edge_receipt_sha256": _sha("route"),
        "local_tangent_unit_bearing": [-1.0, 0.0],
        "local_tangent_alignment_executor_receipt_consumed": consumed,
        "local_tangent_alignment_control_edge_count": 6 if consumed else 0,
        "local_tangent_update_edge_receipts": edges,
        "local_tangent_query_edge_count": 0,
    }


def _aligned_payload():
    packet = _packet()
    trace = []
    rollout = []
    yaw = 0.0
    source = route_alignment_executor_source(packet["packet_sha256"])
    for index in range(6):
        before = yaw
        yaw += math.radians(30.0)
        trace.append({
            "action_index": index,
            "step": index,
            "packet_sha256": packet["packet_sha256"],
            "observation_jpg_sha256": _sha(f"frame-{index}"),
            "memory_frame_idx": 48 + index,
            "yaw_before_rad": before,
            "yaw_after_rad": math.atan2(math.sin(yaw), math.cos(yaw)),
            "turn_delta_deg": 30.0,
            "remaining_after_deg": 150.0 - 30.0 * index,
            "translation_m": 0.0,
            "fresh_observation_required_before_next_action": True,
        })
        rollout.append({
            "step": index,
            "jpg_sha256": _sha(f"frame-{index}"),
            "executor_local_se2_source_since_previous_frame": (
                None if index == 0 else source),
            "executed_translation_m_since_previous_frame": 0.0,
            "executed_yaw_rad_since_previous_frame": (
                0.0 if index == 0 else math.radians(30.0)),
        })
    rollout.append({
        "step": 6,
        "jpg_sha256": _sha("frame-6"),
        "executor_local_se2_source_since_previous_frame": source,
        "executed_translation_m_since_previous_frame": 0.0,
        "executed_yaw_rad_since_previous_frame": math.radians(30.0),
    })
    return {
        "query_leg": [_plan(packet), _plan(packet, consumed=True)],
        "cec_initial_bearing_alignment_trace": trace,
        "rollout_traces": {"query": rollout},
    }


def test_arm_order_is_balanced_over_consumed_indices():
    assert set(SELECTED_HISTORY_INDICES) == {3, 4, 5, 8, 10, 11, 13, 15, 21}
    first = [rotated_arm_order(index)[0]
             for index in SELECTED_HISTORY_INDICES]
    assert first.count("route_tangent_unaligned") == 5
    assert first.count("route_tangent_rear_aligned") == 4


def test_unaligned_audit_rejects_any_control_consumption():
    packet = _packet()
    payload = {
        "query_leg": [_plan(packet)],
        "cec_initial_bearing_alignment_trace": [],
    }
    receipt = audit_unaligned(row={
        "cec_initial_bearing_alignment_mode": "off",
        "cec_initial_bearing_alignment_count": "0",
        "cec_initial_bearing_alignment_action_count": "0",
    }, payload=payload)
    assert receipt["initial_heading_deg"] == pytest.approx(180.0)
    bad = copy.deepcopy(payload)
    bad["query_leg"][0][
        "local_tangent_alignment_executor_receipt_consumed"] = True
    with pytest.raises(RuntimeError):
        audit_unaligned(row={
            "cec_initial_bearing_alignment_mode": "off",
            "cec_initial_bearing_alignment_count": "0",
            "cec_initial_bearing_alignment_action_count": "0",
        }, payload=bad)


def test_aligned_audit_requires_proof_bound_atomic_fresh_turn():
    payload = _aligned_payload()
    receipt = audit_rear_aligned(row={
        "cec_initial_bearing_alignment_mode": (
            "first_route_tangent_rear_bounded"),
        "cec_initial_bearing_alignment_count": "1",
        "cec_initial_bearing_alignment_action_count": "6",
    }, payload=payload)
    assert receipt["alignment_turn_deg"] == pytest.approx(180.0)
    assert receipt["visual_pnp_edges_during_alignment"] == 0
    bad = copy.deepcopy(payload)
    bad["query_leg"][1]["local_tangent_update_edge_receipts"][2][
        "edge_kind"] = "live_query_adjacent_sample"
    with pytest.raises(RuntimeError):
        audit_rear_aligned(row={
            "cec_initial_bearing_alignment_mode": (
                "first_route_tangent_rear_bounded"),
            "cec_initial_bearing_alignment_count": "1",
            "cec_initial_bearing_alignment_action_count": "6",
        }, payload=bad)


def test_packet_privileged_tamper_fails_closed():
    payload = _aligned_payload()
    payload["query_leg"][0]["local_tangent_alignment_packet"][
        "habitat_pose"] = [0, 0, 0]
    with pytest.raises(ValueError):
        audit_rear_aligned(row={
            "cec_initial_bearing_alignment_mode": (
                "first_route_tangent_rear_bounded"),
            "cec_initial_bearing_alignment_count": "1",
            "cec_initial_bearing_alignment_action_count": "6",
        }, payload=payload)
