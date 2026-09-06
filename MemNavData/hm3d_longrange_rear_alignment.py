"""Pure contracts for the consumed long-range rear-alignment diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping

from MemNavData.cec_bearing_alignment import (
    certified_route_alignment_turn,
    validate_bounded_turn_trace,
)
from MemNavData.route_alignment_contract import (
    route_alignment_executor_source,
    verify_route_alignment_packet,
)


ARMS = (
    "route_tangent_unaligned",
    "route_tangent_rear_aligned",
)
ARM_ALIGNMENT = {
    "route_tangent_unaligned": "off",
    "route_tangent_rear_aligned": "first_route_tangent_rear_bounded",
}
SELECTED_HISTORY_INDICES = (3, 4, 5, 8, 10, 11, 13, 15, 21)
PARENT_STUCK_INDICES = (3, 4, 8, 10, 21)
PARENT_SUCCESS_INDICES = (5, 11, 13, 15)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    require(history_index in SELECTED_HISTORY_INDICES,
            "history is outside the consumed no-stop partition")
    offset = SELECTED_HISTORY_INDICES.index(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _first_accepted(plans: list[dict[str, Any]]) -> dict[str, Any]:
    plan = next((row for row in plans
                 if row.get("certified_relocalization_accepted") is True),
                None)
    require(isinstance(plan, dict), "route arm has no accepted initial proof")
    return plan


def initial_route_packet(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Return and validate the packet emitted before either treatment."""

    plan = _first_accepted(plans)
    packet = verify_route_alignment_packet(
        plan.get("local_tangent_alignment_packet"))
    require(plan.get("local_tangent_alignment_packet_sha256")
            == packet["packet_sha256"],
            "initial plan/packet digest binding changed")
    require(plan.get("local_tangent_target_anchor")
            == packet["target_anchor"],
            "initial plan/packet target anchor changed")
    require(plan.get("local_tangent_history_edge_receipt_sha256")
            == packet["history_edge_receipt_sha256"],
            "initial plan/packet route receipt changed")
    source = tuple(float(value) for value in plan[
        "local_tangent_unit_bearing"])
    sealed = tuple(float(value) for value in packet["unit_bearing"])
    require(all(math.isclose(left, right, abs_tol=1e-8)
                for left, right in zip(source, sealed)),
            "initial plan/packet tangent changed")
    require(source[0] < 0.0,
            "consumed rear-alignment history no longer has a rear tangent")
    return packet


def audit_unaligned(
    *, row: Mapping[str, Any], payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that the baseline executes the route tangent unchanged."""

    plans = payload.get("query_leg")
    trace = payload.get("cec_initial_bearing_alignment_trace")
    require(isinstance(plans, list) and isinstance(trace, list),
            "unaligned payload is incomplete")
    packet = initial_route_packet(plans)
    require(row.get("cec_initial_bearing_alignment_mode") == "off"
            and int(row.get("cec_initial_bearing_alignment_count", -1)) == 0
            and int(row.get(
                "cec_initial_bearing_alignment_action_count", -1)) == 0
            and not trace,
            "unaligned arm executed a bearing alignment")
    require(not any(plan.get(
        "local_tangent_alignment_executor_receipt_consumed") is True
                    for plan in plans),
            "unaligned arm consumed an alignment control edge")
    return {
        "packet_sha256": packet["packet_sha256"],
        "initial_heading_deg": math.degrees(float(
            packet["required_turn_rad"])),
        "alignment_actions": 0,
        "control_edges_consumed": 0,
    }


def audit_rear_aligned(
    *, row: Mapping[str, Any], payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one proof-bound, bounded, observation-refreshing rear turn."""

    plans = payload.get("query_leg")
    trace = payload.get("cec_initial_bearing_alignment_trace")
    rollout = payload.get("rollout_traces", {}).get("query")
    require(isinstance(plans, list) and isinstance(trace, list)
            and isinstance(rollout, list),
            "aligned payload is incomplete")
    first = _first_accepted(plans)
    packet = initial_route_packet(plans)
    alignment = certified_route_alignment_turn(first)
    require(alignment is not None,
            "rear packet did not authorize the bounded alignment")
    require(alignment.packet_sha256 == packet["packet_sha256"],
            "alignment consumed another packet")
    require(row.get("cec_initial_bearing_alignment_mode")
            == "first_route_tangent_rear_bounded"
            and int(row.get("cec_initial_bearing_alignment_count", -1)) == 1,
            "aligned arm did not execute exactly one initial alignment")
    receipt = validate_bounded_turn_trace(
        trace, expected_turn_rad=alignment.turn_rad, max_step_deg=30.0)
    require(int(row.get(
        "cec_initial_bearing_alignment_action_count", -1))
            == receipt["action_count"],
            "metric/alignment action counts disagree")
    require(receipt["packet_sha256"] == packet["packet_sha256"],
            "bounded actions are not packet-bound")

    consumed = [plan for plan in plans if plan.get(
        "local_tangent_alignment_executor_receipt_consumed") is True]
    require(bool(consumed),
            "policy never consumed the alignment control edges")
    first_consumed = consumed[0]
    control_edges = first_consumed.get(
        "local_tangent_update_edge_receipts")
    require(isinstance(control_edges, list)
            and len(control_edges) == receipt["action_count"]
            and int(first_consumed.get(
                "local_tangent_alignment_control_edge_count", -1))
            == receipt["action_count"],
            "policy control-edge count differs from executed turn")
    require(all(edge.get("edge_kind")
                == "proof_bound_atomic_yaw_control"
                and edge.get("packet_sha256") == packet["packet_sha256"]
                and abs(float(edge.get("translation_m", math.inf))) <= 1e-12
                for edge in control_edges),
            "alignment interval contains a visual or translating route edge")
    require(int(first_consumed.get("local_tangent_query_edge_count", -1)) == 0,
            "alignment frames entered the visual-PnP query edge count")

    source = route_alignment_executor_source(packet["packet_sha256"])
    step_by_index = {int(item["step"]): item for item in rollout}
    for action in trace:
        step = int(action["step"])
        require(step in step_by_index,
                "alignment action lacks its rendered observation")
        before = step_by_index[step]
        require(before.get("jpg_sha256")
                == action["observation_jpg_sha256"],
                "alignment trace does not bind the executed observation")
        following = step_by_index.get(step + 1)
        require(isinstance(following, dict)
                and following.get(
                    "executor_local_se2_source_since_previous_frame")
                == source
                and abs(float(following.get(
                    "executed_translation_m_since_previous_frame", math.inf)))
                <= 1e-12
                and math.isclose(float(following.get(
                    "executed_yaw_rad_since_previous_frame")),
                    math.radians(float(action["turn_delta_deg"])),
                    abs_tol=1e-9),
                "fresh post-turn observation lacks the issued-action receipt")

    return {
        "packet_sha256": packet["packet_sha256"],
        "initial_heading_deg": math.degrees(alignment.turn_rad),
        "alignment_actions": receipt["action_count"],
        "alignment_turn_deg": receipt["total_turn_deg"],
        "max_abs_action_deg": receipt["max_abs_action_deg"],
        "fresh_observation_receipts": receipt[
            "fresh_observation_receipts"],
        "control_edges_consumed": len(control_edges),
        "visual_pnp_edges_during_alignment": 0,
        "zero_translation": receipt["zero_translation"],
        "simulator_pose_receipt_used": False,
    }


__all__ = [
    "ARMS", "ARM_ALIGNMENT", "PARENT_STUCK_INDICES",
    "PARENT_SUCCESS_INDICES", "SELECTED_HISTORY_INDICES",
    "audit_rear_aligned", "audit_unaligned", "initial_route_packet",
    "require", "rotated_arm_order",
]
