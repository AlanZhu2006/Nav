"""Pure arm and audit contracts for the fresh route-tangent experiment."""

from __future__ import annotations

import math
from typing import Any

from MemNavData.certified_relocalization_contract import (
    CERTIFIED_MINIMUM_ANCHOR,
)


ARMS = (
    "mono_native",
    "mono_cec_endpoint",
    "mono_cec_route_tangent",
)
ARM_CONFIG = {
    "mono_native": {
        "hybrid_route": "native_sidecar",
        "revisit_adapter": "legacy_metric",
        "guidance_mode": "endpoint_bearing",
        "evaluator_arm": "native_sidecar",
    },
    "mono_cec_endpoint": {
        "hybrid_route": "certified_relocalization",
        "revisit_adapter": "verified_bearing_v1",
        "guidance_mode": "endpoint_bearing",
        "evaluator_arm": "certified",
    },
    "mono_cec_route_tangent": {
        "hybrid_route": "certified_relocalization",
        "revisit_adapter": "verified_bearing_v1",
        "guidance_mode": "monocular_route_tangent",
        "evaluator_arm": "certified",
    },
}
PRIMARY_CONTRAST = (
    "mono_cec_route_tangent", "mono_cec_endpoint",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def audit_mono_depth(plans: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    require(arm in ARMS, f"unknown route-tangent arm {arm}")
    require(bool(plans), f"{arm}: no planning receipt")
    controller_plans = [
        plan for plan in plans if plan.get("geometry_stream_stop") is not True
    ]
    stop_plans = [
        plan for plan in plans if plan.get("geometry_stream_stop") is True
    ]
    require(len(stop_plans) <= 1, f"{arm}: repeated geometry stop")
    if stop_plans:
        require(arm == "mono_cec_route_tangent"
                and plans[-1] is stop_plans[0],
                f"{arm}: geometry stop is not the terminal route event")
    for plan in plans:
        require(plan.get("navdp_depth_source") == "monocular_sidecar",
                f"{arm}: query depth source changed")
        require(plan.get("metric_depth_sensor_consumed") is not True,
                f"{arm}: simulator metric depth was consumed")
    receipts = [
        plan.get("monocular_depth_receipt") for plan in controller_plans
    ]
    require(all(isinstance(receipt, dict) for receipt in receipts),
            f"{arm}: a controller call omitted its monocular depth receipt")
    scale_hashes = set()
    for receipt in receipts:
        require(receipt.get("depth_contract")
                == "raw_lingbot_depth_first40_v1",
                f"{arm}: monocular depth contract changed")
        require(receipt.get("metric_depth_sensor_consumed") is False,
                f"{arm}: depth receipt reports a metric sensor")
        require(int(receipt.get("frame_index", -1)) >= 40
                and receipt.get("scale_active") is True,
                f"{arm}: first-40 scale is inactive during the query")
        scale_hashes.add(str(receipt.get("scale_receipt_sha256")))
    require(len(scale_hashes) <= 1,
            f"{arm}: monocular scale changed within the query")
    return {
        "controller_plans": len(controller_plans),
        "geometry_stream_stop_plans": len(stop_plans),
        "monocular_depth_receipts": len(receipts),
        "scale_receipt_hashes": len(scale_hashes),
    }


def audit_route_tangent(plans: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        plan for plan in plans
        if plan.get("certified_relocalization_accepted") is True
    ]
    active = []
    stops = []
    history_hashes = set()
    for plan in accepted:
        require(int(plan.get("local_tangent_target_anchor", -1))
                >= CERTIFIED_MINIMUM_ANCHOR,
                "route tangent escaped the canonical CEC anchor boundary")
        if plan.get("geometry_stream_stop") is True:
            # A fail-closed packet can be emitted before a route schedule is
            # materialized, or after an update edge fails.  It therefore need
            # not contain a successful-readout receipt.  An explicit true
            # bridge is still forbidden, while absence is part of the atomic
            # stop contract rather than evidence of a hidden route.
            require(plan.get("local_tangent_pre_metric_anchor_bridge")
                    in (None, False),
                    "geometry stop reports a forbidden pre-metric bridge")
            require(plan.get("geometry_stream_controller_called") is False
                    and plan.get("geometry_stream_native_fallback_executed")
                    is False
                    and plan.get("geometry_stream_endpoint_fallback_executed")
                    is False,
                    "route geometry failure did not stop atomically")
            require(plan.get("local_tangent_status") == "geometry_failure",
                    "geometry stop lost its local-tangent failure receipt")
            stops.append(plan)
            continue
        require(plan.get("local_tangent_pre_metric_anchor_bridge") is False,
                "active route tangent escaped the canonical CEC depth boundary")
        require(plan.get("local_tangent_status")
                in ("active", "terminal_tangent"),
                "accepted proof has no total route-tangent readout")
        require(plan.get("local_tangent_evaluator_pose_consumed") is False
                and plan.get("local_tangent_habitat_path_consumed") is False
                and plan.get("local_tangent_executor_odometry_consumed")
                is False
                and plan.get("local_tangent_metric_depth_sensor_consumed")
                is False,
                "route tangent consumed privileged runtime geometry")
        require(plan.get("local_tangent_distance_regime_present") is False
                and plan.get("local_tangent_stuck_trigger_present") is False
                and plan.get("local_tangent_endpoint_fallback_available")
                is False
                and plan.get("local_tangent_native_fallback_available")
                is False,
                "route tangent gained a forbidden branch")
        require(math.isclose(
            float(plan["local_tangent_tangent_baseline_m"]), 0.30,
            abs_tol=1e-12), "local tangent baseline changed")
        require(math.isclose(
            float(plan["local_tangent_controller_radius_m"]), 2.5,
            abs_tol=1e-12), "NavDP conditioning radius changed")
        unit = plan.get("local_tangent_unit_bearing")
        point = plan.get("local_tangent_controller_pointgoal")
        require(isinstance(unit, list) and len(unit) == 2
                and isinstance(point, list) and len(point) == 2,
                "route tangent omitted its directional payload")
        require(math.isclose(math.hypot(*unit), 1.0, abs_tol=1e-6)
                and math.isclose(math.hypot(*point), 2.5, abs_tol=1e-6),
                "route tangent is not scale-free with fixed conditioning norm")
        history_hashes.add(str(plan[
            "local_tangent_history_edge_receipt_sha256"]))
        active.append(plan)
    require(len(history_hashes) <= 1,
            "authorized historical route changed within the query")
    return {
        "certificate_accept_plans": len(accepted),
        "active_route_plans": len(active),
        "geometry_stream_stop_plans": len(stops),
        "history_route_receipt_hashes": len(history_hashes),
        "canonical_anchor_boundary_verified": all(
            int(plan.get("local_tangent_target_anchor", -1))
            >= CERTIFIED_MINIMUM_ANCHOR
            and (
                plan.get("local_tangent_pre_metric_anchor_bridge") is False
                or (
                    plan.get("geometry_stream_stop") is True
                    and plan.get("local_tangent_pre_metric_anchor_bridge")
                    is None
                )
            )
            for plan in accepted
        ),
    }


def initial_proof(plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    plan = next((row for row in plans
                 if row.get("certified_relocalization_accepted") is True), None)
    if plan is None:
        return None
    return {
        "selected_anchor": plan.get("router_selected_anchor"),
        "candidate_order_dino": plan.get("router_candidate_order_dino"),
        "candidate_order_used": plan.get("router_candidate_order_used"),
        "certificate": plan.get("certified_relocalization_certificate"),
        "authority": plan.get("certified_relocalization_authority"),
        "pnp": plan.get("certified_relocalization_pnp"),
        "proposal_source": plan.get(
            "certified_relocalization_selected_proposal_source"),
    }


__all__ = [
    "ARMS", "ARM_CONFIG", "PRIMARY_CONTRAST", "audit_mono_depth",
    "audit_route_tangent", "initial_proof", "require", "rotated_arm_order",
]
