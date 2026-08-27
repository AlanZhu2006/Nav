#!/usr/bin/env python3
"""Frozen contracts for the Final14 CEC authority-only ablation.

Both arms use the same causal RGB replay, monocular controller depth, DINO
shortlist, LightGlue/Fundamental ranking, LingBot historical depth, PnP, and
fixed-bearing NavDP adapter.  Only the operational authorization rule differs.
"""

from __future__ import annotations

from MemNavData.final14_mono_factorial import (
    exact_mcnemar_two_sided,
    paired_contrast,
    require,
    scene_cluster_interval,
)


ARMS = (
    "mono_cec",
    "mono_unthresholded_witness",
)

HYBRID_ROUTE = {
    "mono_cec": "certified_relocalization",
    "mono_unthresholded_witness": "certified_unthresholded_witness",
}

EVALUATOR_ARM = {
    "mono_cec": "certified",
    "mono_unthresholded_witness": "unthresholded_witness",
}

AUTHORITY_POLICY = {
    "mono_cec": "strict_certificate",
    "mono_unthresholded_witness": "pnp_pose_available",
}

DEPTH_SOURCE = "monocular_sidecar"
REVISIT_ADAPTER = "verified_bearing_v1"


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


__all__ = [
    "ARMS",
    "AUTHORITY_POLICY",
    "DEPTH_SOURCE",
    "EVALUATOR_ARM",
    "HYBRID_ROUTE",
    "REVISIT_ADAPTER",
    "exact_mcnemar_two_sided",
    "paired_contrast",
    "require",
    "rotated_arm_order",
    "scene_cluster_interval",
]
