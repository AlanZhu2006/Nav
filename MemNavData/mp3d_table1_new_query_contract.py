"""Frozen identity and power rules for the MP3D Table-1 replication."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

try:
    from final14_role_pair_contract import STRATA, assigned_direction_stratum, stable_u32
except ImportError:
    from MemNavData.final14_role_pair_contract import (
        STRATA,
        assigned_direction_stratum,
        stable_u32,
    )


SOURCE_LEDGER_SCHEMA = "mp3d_table1_source_ledger_v1_20260829"
SCENE_SCHEMA = "mp3d_table1_new_query_scene_v1_20260829"
POPULATION_SCHEMA = "mp3d_table1_new_query_population_v1_20260829"
VERIFICATION_SCHEMA = "mp3d_table1_new_query_verification_v1_20260829"
CONSTRUCTION_SEED = 20260829
SEED_NAMESPACE = "mp3d_table1_new_query_novel_v1"
SCENE_COUNT = 20
EPISODES_PER_SCENE = 2
TARGET_HISTORIES = 20
TARGET_SCENES = 12
MINIMUM_PER_STRATUM = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stratum_order(scene_rank: int, episode_rank: int, scene: str,
                  episode: str) -> list[str]:
    """Frozen preferred-first direction order independent of imagery/results."""

    preferred = assigned_direction_stratum(scene_rank, episode_rank)
    remaining = [value for value in STRATA if value != preferred]
    if stable_u32("mp3d_table1_stratum_fallback", scene, episode) % 2:
        remaining.reverse()
    return [preferred, *remaining]


def novel_query(row: dict[str, Any]) -> dict[str, Any]:
    queries = [query for pair in row["pairs"] for query in pair["queries"]]
    matches = [query for query in queries if query["analysis_role"] == "novel"]
    require(len(matches) == 1, "history must contain one Novel query")
    return matches[0]


def query_reuses_source_goal(query: dict[str, Any],
                             source_goal: dict[str, Any]) -> bool:
    """Reject byte-identical or pose-identical reuse of the consumed Goal-B."""

    if str(query["goal_rgb_sha256"]) == str(source_goal["goal_rgb_sha256"]):
        return True
    query_position = [float(value) for value in query["floor_position"]]
    source_position = [float(value) for value in source_goal["floor_position"]]
    position_error = math.sqrt(sum(
        (first - second) ** 2
        for first, second in zip(query_position, source_position)
    ))
    yaw_error = abs(
        (float(query["yaw_rad"]) - float(source_goal["yaw_rad"]) + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )
    return position_error <= 1e-4 and yaw_error <= 1e-4


def assert_new_query_identity(row: dict[str, Any],
                              source_goal: dict[str, Any]) -> None:
    for pair in row["pairs"]:
        for query in pair["queries"]:
            require(
                not query_reuses_source_goal(query, source_goal),
                "new-query population reused the consumed source Goal-B",
            )


def selected_stratum(row: dict[str, Any]) -> str:
    value = str(novel_query(row)["assigned_direction_stratum"])
    require(value in STRATA, "Novel query has an invalid direction stratum")
    return value


def power(rows: list[dict[str, Any]], *, target_histories: int,
          target_scenes: int, minimum_per_stratum: int) -> dict[str, Any]:
    strata = Counter(selected_stratum(row) for row in rows)
    scenes = {str(row["scene"]) for row in rows}
    return {
        "histories": len(rows),
        "scene_clusters": len(scenes),
        "direction_strata": {name: int(strata[name]) for name in STRATA},
        "target_histories": int(target_histories),
        "target_scene_clusters": int(target_scenes),
        "minimum_histories_per_direction_stratum": int(minimum_per_stratum),
        "target_met": (
            len(rows) >= target_histories
            and len(scenes) >= target_scenes
            and all(strata[name] >= minimum_per_stratum for name in STRATA)
        ),
    }


__all__ = [
    "CONSTRUCTION_SEED",
    "EPISODES_PER_SCENE",
    "MINIMUM_PER_STRATUM",
    "POPULATION_SCHEMA",
    "SCENE_COUNT",
    "SCENE_SCHEMA",
    "SEED_NAMESPACE",
    "SOURCE_LEDGER_SCHEMA",
    "TARGET_HISTORIES",
    "TARGET_SCENES",
    "VERIFICATION_SCHEMA",
    "assert_new_query_identity",
    "novel_query",
    "power",
    "query_reuses_source_goal",
    "require",
    "selected_stratum",
    "stratum_order",
]
