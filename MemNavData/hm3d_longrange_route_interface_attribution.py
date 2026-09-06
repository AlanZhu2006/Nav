"""Pure contracts for the consumed long-range route-interface attribution."""

from __future__ import annotations

import math
from typing import Any

from MemNavData.revisit_bearing_adapter import (
    VERIFIED_BEARING_RADIUS_M,
    project_unit_bearing_to_navdp_support,
)


ARMS = (
    "direct_pnp_canonical_bearing",
    "direct_pnp_support_projected_bearing",
)
ARM_CONFIG = {
    "direct_pnp_canonical_bearing": {
        "revisit_adapter": "verified_bearing_v1",
        "evaluator_arm": "certified",
    },
    "direct_pnp_support_projected_bearing": {
        "revisit_adapter": "verified_navdp_support_projection_v1",
        "evaluator_arm": "certified_support_projected",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    require(isinstance(value, list) and len(value) == 2,
            f"{name} must be a two-vector")
    pair = float(value[0]), float(value[1])
    require(all(math.isfinite(component) for component in pair),
            f"{name} must be finite")
    return pair


def audit_controller_support(
    plans: list[dict[str, Any]], *, arm: str,
) -> dict[str, Any]:
    """Verify the only difference between the two controller interfaces."""

    require(arm in ARMS, f"unknown interface arm {arm}")
    active = [
        plan for plan in plans
        if plan.get("geometry_stream_stop") is not True
        and plan.get("certified_relocalization_accepted") is True
    ]
    projected_count = 0
    rear_count = 0
    for plan in active:
        require(plan.get("revisit_adapter_takeover") is True,
                f"{arm}: accepted route did not reach the controller")
        source = _finite_pair(plan.get("memory_bearing_unit"),
                              "source bearing")
        tangent = _finite_pair(plan.get("local_tangent_unit_bearing"),
                               "route tangent")
        require(math.isclose(source[0], tangent[0], abs_tol=1e-8)
                and math.isclose(source[1], tangent[1], abs_tol=1e-8),
                f"{arm}: adapter changed the source route tangent")
        require(math.isclose(math.hypot(*source), 1.0, abs_tol=1e-6),
                f"{arm}: source bearing is not unit norm")
        controller = _finite_pair(plan.get("memory_controller_pointgoal"),
                                  "controller PointGoal")
        require(math.isclose(
            math.hypot(*controller), VERIFIED_BEARING_RADIUS_M,
            abs_tol=1e-6), f"{arm}: controller token norm changed")
        is_rear = source[0] < 0.0
        rear_count += int(is_rear)
        if arm == "direct_pnp_canonical_bearing":
            expected = (
                source[0] * VERIFIED_BEARING_RADIUS_M,
                source[1] * VERIFIED_BEARING_RADIUS_M,
            )
            require(plan.get("memory_navdp_support_projection_applied") is None,
                    "canonical arm emitted a support-projection receipt")
        else:
            unit, projected = project_unit_bearing_to_navdp_support(source)
            expected = (
                unit[0] * VERIFIED_BEARING_RADIUS_M,
                unit[1] * VERIFIED_BEARING_RADIUS_M,
            )
            require(plan.get("navdp_support_projection_schema_version") == 1,
                    "support-projection schema changed")
            require(plan.get("memory_navdp_support_projection_applied")
                    is projected,
                    "support-projection decision disagrees with source bearing")
            projected_count += int(projected)
        require(math.isclose(controller[0], expected[0], abs_tol=1e-8)
                and math.isclose(controller[1], expected[1], abs_tol=1e-8),
                f"{arm}: controller PointGoal is not the specified transform")
    if arm == "direct_pnp_support_projected_bearing":
        require(projected_count == rear_count,
                "support projection was not exactly the rear-half-plane set")
    else:
        require(projected_count == 0,
                "canonical arm unexpectedly projected a bearing")
    return {
        "active_controller_plans": len(active),
        "rear_source_bearing_plans": rear_count,
        "support_projection_plans": projected_count,
        "fixed_radius_verified": True,
        "role_distance_stuck_critic_gate_present": False,
        "direct_executor_action_present": False,
    }


__all__ = [
    "ARMS", "ARM_CONFIG", "audit_controller_support", "rotated_arm_order",
]
