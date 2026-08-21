"""Pure contracts and paired statistics for the MDTEC monocular x CEC
composition experiment.

This experiment does NOT re-decide the raw-depth substitution (Gate D) or the
CEC certificate (Final14).  It changes exactly one variable across paired
Goal-B rollouts that all share one causal monocular Goal-A trace: whether the
long-horizon readout is plain native ImageGoal (``raw_native``) or the
existing certified relocalization bearing (``raw_cec``).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

try:  # re-exported, not redefined
    from mdtec_raw_depth_gate_d import (
        exact_mcnemar_two_sided,
        paired_contrast,
        require,
        scene_cluster_interval,
    )
except ImportError:
    from MemNavData.mdtec_raw_depth_gate_d import (
        exact_mcnemar_two_sided,
        paired_contrast,
        require,
        scene_cluster_interval,
    )

ARMS = ("raw_native", "raw_cec")
HYBRID_ROUTE = {
    "raw_native": "native_sidecar",
    "raw_cec": "certified_relocalization",
}
DEPTH_SOURCE = "monocular_sidecar"  # identical for both arms; not the IV here


def rotated_arm_order(scene_index: int, episode_index: int) -> tuple[str, ...]:
    offset = (int(scene_index) + int(episode_index)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def audit_shared_leg_a(outcome: dict[str, Any]) -> None:
    """The single shared Goal-A rollout: same checks Gate D applied to its
    ``raw_first40`` arm, since Goal-A always runs under the monocular
    depth contract regardless of which B arm will consume the trace."""
    plans = outcome["plans"]
    require(bool(plans), "shared Goal-A leg produced no plans")
    require(outcome.get("navdp_depth_source") == DEPTH_SOURCE,
            "shared Goal-A leg depth source mismatch")
    require(not bool(outcome.get("metric_depth_sensor_consumed_any")),
            "shared Goal-A leg consumed simulator metric depth")
    receipts = [
        plan.get("monocular_depth_receipt") for plan in plans
        if isinstance(plan.get("monocular_depth_receipt"), dict)
    ]
    require(len(receipts) == len(plans),
            "shared Goal-A leg omitted one or more monocular receipts")
    scale_hashes: set[str] = set()
    for receipt in receipts:
        require(receipt.get("metric_depth_sensor_consumed") is False,
                "shared Goal-A receipt reports metric sensor consumption")
        frame_index = int(receipt["frame_index"])
        if frame_index < 40:
            require(float(receipt.get("depth_nonzero_fraction", -1.0)) == 0.0,
                    "shared Goal-A bootstrap depth was not exactly zero")
            require(receipt.get("scale_active") is False,
                    "shared Goal-A scale activated before frame 40")
        else:
            require(receipt.get("scale_active") is True,
                    "shared Goal-A depth did not activate at/after frame 40")
            scale = receipt.get("scale_receipt")
            require(isinstance(scale, dict), "shared Goal-A receipt omitted scale")
            require(receipt.get("scale_receipt_sha256"),
                    "shared Goal-A scale receipt omitted SHA")
            scale_hashes.add(str(receipt["scale_receipt_sha256"]))
    require(len(scale_hashes) <= 1, "shared Goal-A leg froze scale more than once")


def audit_arm_leg_b(arm: str, outcome: dict[str, Any],
                    *, native_outcome: dict[str, Any] | None) -> dict[str, Any]:
    """Per-B-arm audit: depth contract identical to Goal-A's, plus the
    CEC-specific reject-must-match-native and zero-runtime-failure checks."""
    plans = outcome["plans"]
    require(outcome.get("navdp_depth_source") == DEPTH_SOURCE,
            f"{arm}: Goal-B depth source mismatch")
    require(not bool(outcome.get("metric_depth_sensor_consumed_any")),
            f"{arm}: Goal-B consumed simulator metric depth")

    certified_requests = [
        plan for plan in plans
        if plan.get("certified_relocalization_ok") is not None]
    certified_runtime_failures = [
        plan for plan in certified_requests
        if plan.get("certified_relocalization_ok") is not True]
    if arm == "raw_cec":
        require(not certified_runtime_failures,
                "raw_cec: certificate runtime/transport failure (fail-closed, "
                "audit invalid)")
        if native_outcome is not None:
            native_plans = native_outcome["plans"]
            for i, plan in enumerate(plans):
                if plan.get("certified_relocalization_accepted") is True:
                    continue  # accepted: allowed to diverge from native
                require(i < len(native_plans),
                        "raw_cec: reject-path plan count exceeds native")
                nat = native_plans[i]
                require(plan.get("requested_diffusion_seed")
                        == nat.get("requested_diffusion_seed"),
                        "raw_cec reject: requested seed diverges from native")
                require(plan.get("diffusion_seed") == nat.get("diffusion_seed"),
                        "raw_cec reject: returned seed diverges from native")
                require(plan.get("selected_trajectory_sha256")
                        == nat.get("selected_trajectory_sha256"),
                        "raw_cec reject: selected trajectory diverges from native")
    else:
        require(not certified_requests,
                "raw_native: unexpected certified_relocalization activity")

    return {
        "certified_request_count": len(certified_requests),
        "certified_accept_count": sum(
            1 for plan in certified_requests
            if plan.get("certified_relocalization_accepted") is True),
        "certified_runtime_failure_count": len(certified_runtime_failures),
    }


__all__ = [
    "ARMS", "HYBRID_ROUTE", "DEPTH_SOURCE", "audit_shared_leg_a",
    "audit_arm_leg_b", "exact_mcnemar_two_sided", "paired_contrast",
    "require", "rotated_arm_order", "scene_cluster_interval",
]
