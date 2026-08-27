"""Pure contracts and paired statistics for MDTEC raw-depth Gate D."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


ARMS = ("metric_teacher", "zero_depth", "raw_first40")
DEPTH_SOURCE = {
    "metric_teacher": "metric_request",
    "zero_depth": "zero",
    "raw_first40": "monocular_sidecar",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rotated_arm_order(scene_index: int, episode_index: int) -> tuple[str, ...]:
    offset = (int(scene_index) + int(episode_index)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def audit_arm_contract(arm: str, outcome: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on the per-rollout depth-source audit receipts."""
    plans = outcome["plans"]
    expected_source = DEPTH_SOURCE[arm]
    require(outcome.get("navdp_depth_source") == expected_source,
            f"{arm}: depth source result mismatch")
    require(bool(plans), f"{arm}: no policy plans")
    for plan in plans:
        require(plan.get("navdp_depth_source") == expected_source,
                f"{arm}: per-plan depth source mismatch")
    sensor_any = bool(outcome.get("metric_depth_sensor_consumed_any"))
    if arm == "metric_teacher":
        require(sensor_any, "metric teacher did not consume request depth")
    else:
        require(not sensor_any, f"{arm}: simulator metric depth was consumed")

    receipts = [
        plan.get("monocular_depth_receipt") for plan in plans
        if isinstance(plan.get("monocular_depth_receipt"), dict)
    ]
    if arm != "raw_first40":
        require(not receipts, f"{arm}: unexpected monocular receipt")
        return {
            "bootstrap_plan_count": 0,
            "active_plan_count": 0,
            "receipt_contract_valid": True,
        }

    require(len(receipts) == len(plans),
            "raw arm omitted one or more monocular receipts")
    bootstrap = []
    active = []
    scale_hashes: set[str] = set()
    for receipt in receipts:
        require(receipt.get("depth_contract") ==
                "raw_lingbot_depth_first40_v1",
                "raw arm depth contract changed")
        require(receipt.get("metric_depth_sensor_consumed") is False,
                "raw receipt reports metric sensor consumption")
        require(receipt.get("image_sha256") and receipt.get("depth_png_sha256"),
                "raw receipt omitted current-image/depth hash")
        frame_index = int(receipt["frame_index"])
        if frame_index < 40:
            require(float(receipt.get("depth_nonzero_fraction", -1.0)) == 0.0,
                    "raw bootstrap depth was not exactly zero")
            require(receipt.get("scale_active") is False,
                    "raw scale activated before frame 40")
            bootstrap.append(receipt)
        else:
            require(receipt.get("scale_active") is True,
                    "raw depth did not activate at/after frame 40")
            scale = receipt.get("scale_receipt")
            require(isinstance(scale, dict), "raw active receipt omitted scale")
            require(scale.get("scale_evidence_contract") ==
                    "causal_first_prefix_rgb_only_v1",
                    "raw scale evidence contract changed")
            require(scale.get("whole_episode_ground_cache_consumed") is False,
                    "raw scale consumed whole-episode ground cache")
            require(receipt.get("scale_receipt_sha256"),
                    "raw scale receipt omitted SHA")
            scale_hashes.add(str(receipt["scale_receipt_sha256"]))
            if bool(scale.get("scale_valid")):
                require(float(receipt.get("depth_nonzero_fraction", 0.0)) > 0.0,
                        "valid raw scale silently continued zero depth")
            active.append(receipt)
    require(len(scale_hashes) <= 1, "raw arm froze scale more than once")
    survived = bool(outcome.get("monocular_frame40_survived"))
    require(survived == bool(active), "frame-40 survival accounting mismatch")
    return {
        "bootstrap_plan_count": len(bootstrap),
        "active_plan_count": len(active),
        "receipt_contract_valid": True,
        "scale_receipt_unique_count": len(scale_hashes),
    }


def exact_mcnemar_two_sided(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(
        0, min(int(gains), int(losses)) + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * tail))


def paired_contrast(rows: Iterable[dict[str, Any]], treatment: str,
                    reference: str) -> dict[str, Any]:
    by_unit: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["episode"]))
        by_unit.setdefault(key, {})[str(row["arm"])] = int(row["reached"])
    require(bool(by_unit), "paired contrast has no rows")
    require(all(treatment in arms and reference in arms
                for arms in by_unit.values()), "paired arm coverage is incomplete")
    gains = sum(arms[treatment] == 1 and arms[reference] == 0
                for arms in by_unit.values())
    losses = sum(arms[treatment] == 0 and arms[reference] == 1
                 for arms in by_unit.values())
    risk_difference = float(np.mean([
        arms[treatment] - arms[reference] for arms in by_unit.values()]))
    return {
        "treatment": treatment,
        "reference": reference,
        "n": len(by_unit),
        "gains": int(gains),
        "losses": int(losses),
        "ties": int(len(by_unit) - gains - losses),
        "risk_difference": risk_difference,
        "risk_difference_pp": 100.0 * risk_difference,
        "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(gains, losses),
    }


def scene_cluster_interval(rows: Iterable[dict[str, Any]], treatment: str,
                           reference: str, *, seed: int,
                           resamples: int) -> list[float]:
    by_scene: dict[str, list[tuple[int, int]]] = {}
    by_unit: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["episode"]))
        by_unit.setdefault(key, {})[str(row["arm"])] = int(row["reached"])
    for (scene, _episode), arms in by_unit.items():
        require(treatment in arms and reference in arms,
                "cluster interval arm coverage is incomplete")
        by_scene.setdefault(scene, []).append(
            (arms[treatment], arms[reference]))
    scenes = sorted(by_scene)
    require(bool(scenes), "cluster interval has no scenes")
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        sampled = rng.integers(0, len(scenes), size=len(scenes))
        deltas = []
        for scene_index in sampled:
            deltas.extend(a - b for a, b in by_scene[scenes[int(scene_index)]])
        values[index] = np.mean(deltas)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


__all__ = [
    "ARMS", "DEPTH_SOURCE", "audit_arm_contract", "exact_mcnemar_two_sided",
    "paired_contrast", "require", "rotated_arm_order", "scene_cluster_interval",
]
