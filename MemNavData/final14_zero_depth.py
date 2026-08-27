"""Contracts for the missing Final14 zero-depth native control arm."""

from __future__ import annotations

from typing import Any

from MemNavData.final14_mono_factorial import require


ARM = "zero_native"
DEPTH_SOURCE = "zero"
HYBRID_ROUTE = "native_sidecar"
REVISIT_ADAPTER = "legacy_metric"
EVALUATOR_ARM = "native_sidecar"


def audit_zero_depth_plans(plans: list[dict[str, Any]]) -> dict[str, int]:
    """Require an explicit all-zero NavDP payload on every policy call."""

    require(bool(plans), "zero-depth query produced no NavDP plans")
    for index, plan in enumerate(plans):
        require(plan.get("navdp_depth_source") == DEPTH_SOURCE,
                f"zero-depth plan {index} used another depth source")
        require(plan.get("metric_depth_sensor_consumed") is False,
                f"zero-depth plan {index} consumed metric sensor depth")
        require(plan.get("monocular_depth_receipt") is None,
                f"zero-depth plan {index} consumed a monocular receipt")
    return {
        "plan_count": len(plans),
        "metric_sensor_plan_count": 0,
        "monocular_receipt_plan_count": 0,
        "explicit_zero_depth_plan_count": len(plans),
    }


__all__ = [
    "ARM",
    "DEPTH_SOURCE",
    "EVALUATOR_ARM",
    "HYBRID_ROUTE",
    "REVISIT_ADAPTER",
    "audit_zero_depth_plans",
]
