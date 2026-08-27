"""Pure-Python constants and identity rules for final14 query construction."""

from __future__ import annotations

import hashlib
import math


SCENE_BUILD_SCHEMA = "final14_role_pair_scene_build_v1_20260817"
POPULATION_SCHEMA = "final14_role_pair_population_v1_20260817"
PROTOCOLS = ("natural_direction", "hard_support")
STRATA = ("front", "side", "rear")
FROZEN_SOURCE_EPISODES_PER_SCENE = 8


def stable_u32(*parts: object) -> int:
    material = "/".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def wrap_radians(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def assigned_direction_stratum(scene_rank: int, episode_rank: int) -> str:
    """Cycle over the lexicographically flattened 14 x 8 source ledger."""

    global_source_rank = (
        int(scene_rank) * FROZEN_SOURCE_EPISODES_PER_SCENE
        + int(episode_rank)
    )
    return STRATA[global_source_rank % len(STRATA)]


def goal_yaw_bin(scene: str, episode: str) -> int:
    return stable_u32("final14_goal_yaw", scene, episode) % 8


def goal_yaw_radians(scene: str, episode: str) -> float:
    return wrap_radians(goal_yaw_bin(scene, episode) * (2.0 * math.pi / 8.0))


def relative_direction_degrees(
    initial_path_bearing_rad: float, endpoint_yaw_rad: float
) -> float:
    return float(math.degrees(wrap_radians(
        float(initial_path_bearing_rad) - float(endpoint_yaw_rad)
    )))


def direction_in_stratum(relative_degrees: float, stratum: str) -> bool:
    magnitude = abs(float(relative_degrees))
    if stratum == "front":
        return magnitude <= 60.0 + 1e-9
    if stratum == "side":
        return 60.0 < magnitude <= 120.0 + 1e-9
    if stratum == "rear":
        return 120.0 < magnitude <= 180.0 + 1e-9
    raise ValueError(f"unknown direction stratum: {stratum}")


def support_band(max_covis: float, argmax_gap: int) -> str | None:
    value = float(max_covis)
    gap = int(argmax_gap)
    if 0.55 <= value <= 0.90 and gap <= 24:
        return "standard"
    if 0.25 <= value < 0.55 and gap <= 32:
        return "hard"
    return None


__all__ = [
    "FROZEN_SOURCE_EPISODES_PER_SCENE",
    "POPULATION_SCHEMA",
    "PROTOCOLS",
    "SCENE_BUILD_SCHEMA",
    "STRATA",
    "assigned_direction_stratum",
    "direction_in_stratum",
    "goal_yaw_bin",
    "goal_yaw_radians",
    "relative_direction_degrees",
    "stable_u32",
    "support_band",
    "wrap_radians",
]
