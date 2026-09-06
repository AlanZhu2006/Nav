#!/usr/bin/env python3
"""Frozen contracts for the HM3D Table-I memory-authority spectrum.

The four arms share the same sealed causal RGB history, monocular controller
depth, frozen NavDP checkpoint, query budget, and role-hidden runtime.  They
progressively add memory proposal, a finite geometric witness, and the strict
operational certificate.  This is a retrospective ablation on the already
evaluated Table-I query population; it is not a fresh confirmation experiment.
"""

from __future__ import annotations

from MemNavData.final14_mono_factorial import audit_depth_plans
from MemNavData.mdtec_raw_depth_gate_d import require


ARMS = (
    "mono_native",
    "mono_raw_fixed",
    "mono_unthresholded_witness",
    "mono_cec",
)

DEPTH_SOURCE = {arm: "monocular_sidecar" for arm in ARMS}

HYBRID_ROUTE = {
    "mono_native": "native_sidecar",
    "mono_raw_fixed": "phase",
    "mono_unthresholded_witness": "certified_unthresholded_witness",
    "mono_cec": "certified_relocalization",
}

REVISIT_ADAPTER = {
    "mono_native": "legacy_metric",
    "mono_raw_fixed": "raw_fixed_bearing_v1",
    "mono_unthresholded_witness": "verified_bearing_v1",
    "mono_cec": "verified_bearing_v1",
}

EVALUATOR_ARM = {
    "mono_native": "native_sidecar",
    "mono_raw_fixed": "raw_fixed_bearing",
    "mono_unthresholded_witness": "unthresholded_witness",
    "mono_cec": "certified",
}

AUTHORITY_POLICY = {
    "mono_unthresholded_witness": "pnp_pose_available",
    "mono_cec": "strict_certificate",
}

PRIMARY_CONTRASTS = (
    ("mono_cec", "mono_native"),
    ("mono_raw_fixed", "mono_native"),
    ("mono_unthresholded_witness", "mono_native"),
    ("mono_cec", "mono_raw_fixed"),
    ("mono_cec", "mono_unthresholded_witness"),
)


def selected_arm_order(
    history_index: int, selected: tuple[str, ...]
) -> tuple[str, ...]:
    """Return a cyclic four-arm order and reject partial formal spectra."""

    require(len(selected) == len(set(selected)), "selected arms contain duplicates")
    require(selected == ARMS, "authority spectrum requires all four frozen arms")
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def audit_query_arm(arm: str, plans: list[dict]) -> dict:
    """Reuse the frozen monocular-depth audit for every spectrum arm."""

    require(arm in ARMS, f"unknown authority-spectrum arm {arm!r}")
    # The unthresholded witness changes only the authority rule; its depth
    # contract is identical to strict mono CEC.
    audit_name = "mono_cec" if arm == "mono_unthresholded_witness" else arm
    return audit_depth_plans(audit_name, plans)


__all__ = [
    "ARMS",
    "AUTHORITY_POLICY",
    "DEPTH_SOURCE",
    "EVALUATOR_ARM",
    "HYBRID_ROUTE",
    "PRIMARY_CONTRASTS",
    "REVISIT_ADAPTER",
    "audit_query_arm",
    "selected_arm_order",
]
