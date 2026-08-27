"""Pure contracts for the consumed Final14 controller-depth factorial.

The experiment reuses one already-consumed natural-direction Final14 history
and its two frozen queries.  It does not reconstruct a new population.  Its
estimand is query-controller depth and CEC authorization under an identical
causal RGB replay.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:
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


ARMS = (
    "mono_native",
    "mono_raw_fixed",
    "mono_cec",
    "metric_native",
    "metric_cec",
)

DEPTH_SOURCE = {
    "mono_native": "monocular_sidecar",
    "mono_raw_fixed": "monocular_sidecar",
    "mono_cec": "monocular_sidecar",
    "metric_native": "metric_request",
    "metric_cec": "metric_request",
}

HYBRID_ROUTE = {
    "mono_native": "native_sidecar",
    "mono_raw_fixed": "phase",
    "mono_cec": "certified_relocalization",
    "metric_native": "native_sidecar",
    "metric_cec": "certified_relocalization",
}

REVISIT_ADAPTER = {
    "mono_native": "legacy_metric",
    "mono_raw_fixed": "raw_fixed_bearing_v1",
    "mono_cec": "verified_bearing_v1",
    "metric_native": "legacy_metric",
    "metric_cec": "verified_bearing_v1",
}

EVALUATOR_ARM = {
    "mono_native": "native_sidecar",
    "mono_raw_fixed": "raw_fixed_bearing",
    "mono_cec": "certified",
    "metric_native": "native_sidecar",
    "metric_cec": "certified",
}

PRIMARY_CONTRASTS = (
    ("mono_cec", "mono_native"),
    ("mono_cec", "mono_raw_fixed"),
    ("metric_cec", "metric_native"),
    ("mono_native", "metric_native"),
    ("mono_cec", "metric_cec"),
)


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def audit_depth_plans(arm: str, plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate every actual NavDP query against the selected sensor arm."""

    require(arm in ARMS, f"unknown factorial arm {arm!r}")
    require(bool(plans), f"{arm}: query produced no NavDP plans")
    expected = DEPTH_SOURCE[arm]
    for plan in plans:
        require(
            plan.get("navdp_depth_source") == expected,
            f"{arm}: per-plan depth source changed",
        )

    sensor_reads = sum(
        plan.get("metric_depth_sensor_consumed") is True for plan in plans
    )
    receipts = [
        plan.get("monocular_depth_receipt")
        for plan in plans
        if isinstance(plan.get("monocular_depth_receipt"), dict)
    ]
    if expected == "metric_request":
        require(sensor_reads == len(plans),
                f"{arm}: metric arm omitted a sensor-depth read")
        require(not receipts, f"{arm}: metric arm exposed mono receipts")
        return {
            "metric_sensor_plan_count": sensor_reads,
            "monocular_receipt_plan_count": 0,
            "monocular_scale_hash_count": 0,
        }

    require(sensor_reads == 0, f"{arm}: mono arm consumed metric depth")
    require(len(receipts) == len(plans),
            f"{arm}: mono arm omitted one or more depth receipts")
    hashes: set[str] = set()
    for receipt in receipts:
        require(receipt.get("depth_contract") ==
                "raw_lingbot_depth_first40_v1",
                f"{arm}: mono depth contract changed")
        require(receipt.get("metric_depth_sensor_consumed") is False,
                f"{arm}: mono receipt reports metric consumption")
        require(int(receipt.get("frame_index", -1)) >= 40,
                f"{arm}: Final14 query unexpectedly used bootstrap depth")
        require(receipt.get("scale_active") is True,
                f"{arm}: mono scale was inactive after history replay")
        scale = receipt.get("scale_receipt")
        require(isinstance(scale, dict), f"{arm}: scale receipt missing")
        require(scale.get("scale_evidence_contract") ==
                "causal_first_prefix_rgb_only_v1",
                f"{arm}: scale evidence contract changed")
        require(scale.get("whole_episode_ground_cache_consumed") is False,
                f"{arm}: scale consumed future history")
        require(receipt.get("scale_receipt_sha256"),
                f"{arm}: scale receipt hash missing")
        hashes.add(str(receipt["scale_receipt_sha256"]))
    require(len(hashes) == 1, f"{arm}: scale was not immutable within query")
    return {
        "metric_sensor_plan_count": 0,
        "monocular_receipt_plan_count": len(receipts),
        "monocular_scale_hash_count": len(hashes),
    }


def interaction_difference(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Scene-cluster interval for the binary difference-in-differences.

    Positive values mean CEC's gain over native is larger with monocular depth
    than with metric request depth.  This is descriptive attribution, not a
    McNemar test.
    """

    by_unit: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["episode"]))
        by_unit.setdefault(key, {})[str(row["arm"])] = int(row["reached"])
    required = {"mono_cec", "mono_native", "metric_cec", "metric_native"}
    require(bool(by_unit), "interaction has no paired rows")
    require(all(required.issubset(values) for values in by_unit.values()),
            "interaction arm coverage is incomplete")

    by_scene: dict[str, list[float]] = {}
    for (scene, _episode), values in by_unit.items():
        delta = (
            values["mono_cec"] - values["mono_native"]
            - values["metric_cec"] + values["metric_native"]
        )
        by_scene.setdefault(scene, []).append(float(delta))
    point = float(np.mean([x for values in by_scene.values() for x in values]))
    scenes = sorted(by_scene)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        chosen = rng.integers(0, len(scenes), size=len(scenes))
        values: list[float] = []
        for scene_index in chosen:
            values.extend(by_scene[scenes[int(scene_index)]])
        samples[index] = np.mean(values)
    interval = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    return {
        "estimand": (
            "(mono_cec-mono_native)-(metric_cec-metric_native)"
        ),
        "n": len(by_unit),
        "scene_count": len(scenes),
        "difference_in_differences": point,
        "difference_in_differences_pp": 100.0 * point,
        "scene_cluster_bootstrap_95": interval,
        "scene_cluster_bootstrap_95_pp": [100.0 * x for x in interval],
        "seed": int(seed),
        "resamples": int(resamples),
    }


__all__ = [
    "ARMS",
    "DEPTH_SOURCE",
    "EVALUATOR_ARM",
    "HYBRID_ROUTE",
    "PRIMARY_CONTRASTS",
    "REVISIT_ADAPTER",
    "audit_depth_plans",
    "exact_mcnemar_two_sided",
    "interaction_difference",
    "paired_contrast",
    "require",
    "rotated_arm_order",
    "scene_cluster_interval",
]
