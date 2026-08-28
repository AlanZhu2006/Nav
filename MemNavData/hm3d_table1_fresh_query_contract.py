"""Pure-Python identity and power rules for the HM3D Table-1 reserve."""

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from final14_role_pair_contract import (
        STRATA,
        assigned_direction_stratum,
        stable_u32,
    )
except ImportError:
    from MemNavData.final14_role_pair_contract import (
        STRATA,
        assigned_direction_stratum,
        stable_u32,
    )


SCENE_SCHEMA = "hm3d_table1_fresh_query_scene_v1_20260829"
POPULATION_SCHEMA = "hm3d_table1_fresh_query_population_v1_20260829"
CONSTRUCTION_SEED = 20260829
SEED_NAMESPACE = "hm3d_table1_fresh_query_novel_v1"
PREFIX_SCHEDULE = (30, 36, 42, 48, 54)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def identity_set(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    rows = manifest.get("episodes")
    require(isinstance(rows, list), "consumed manifest has no episode ledger")
    identities = {(str(row["scene"]), str(row["episode"])) for row in rows}
    require(len(identities) == len(rows), "consumed manifest duplicates identities")
    return identities


def stratum_order(scene_rank: int, episode_rank: int, scene: str,
                  episode: str) -> list[str]:
    """Return a frozen preferred-first order without reading imagery/outcomes."""

    preferred = assigned_direction_stratum(scene_rank, episode_rank)
    remaining = [value for value in STRATA if value != preferred]
    if stable_u32("hm3d_table1_stratum_fallback", scene, episode) % 2:
        remaining.reverse()
    return [preferred, *remaining]


def selected_stratum(row: dict[str, Any]) -> str:
    queries = row["pairs"][0]["queries"]
    novel = next(query for query in queries if query["analysis_role"] == "novel")
    value = str(novel["assigned_direction_stratum"])
    require(value in STRATA, "retained Novel has an invalid direction stratum")
    return value


def power(rows: list[dict[str, Any]], *, target_histories: int,
          target_scenes: int, minimum_per_stratum: int) -> dict[str, Any]:
    strata = Counter(selected_stratum(row) for row in rows)
    scenes = {str(row["scene"]) for row in rows}
    return {
        "histories": len(rows),
        "scene_clusters": len(scenes),
        "direction_strata": {name: int(strata[name]) for name in STRATA},
        "target_histories": target_histories,
        "target_scene_clusters": target_scenes,
        "minimum_histories_per_direction_stratum": minimum_per_stratum,
        "target_met": (
            len(rows) >= target_histories
            and len(scenes) >= target_scenes
            and all(strata[name] >= minimum_per_stratum for name in STRATA)
        ),
    }


__all__ = [
    "CONSTRUCTION_SEED",
    "POPULATION_SCHEMA",
    "PREFIX_SCHEDULE",
    "SCENE_SCHEMA",
    "SEED_NAMESPACE",
    "identity_set",
    "power",
    "require",
    "selected_stratum",
    "stratum_order",
]
