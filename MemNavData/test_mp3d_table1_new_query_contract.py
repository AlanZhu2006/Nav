from __future__ import annotations

import copy

import pytest

from MemNavData.final14_role_pair_contract import STRATA
from MemNavData.mp3d_table1_new_query_contract import (
    assert_new_query_identity,
    power,
    query_reuses_source_goal,
    stratum_order,
)


def history(scene: str, episode: str, stratum: str) -> dict:
    return {
        "scene": scene,
        "episode": episode,
        "pairs": [{
            "queries": [{
                "analysis_role": "novel",
                "goal_rgb_sha256": "a" * 64,
                "floor_position": [1.0, 0.0, 2.0],
                "yaw_rad": 0.2,
                "assigned_direction_stratum": stratum,
            }, {
                "analysis_role": "revisit",
                "goal_rgb_sha256": "b" * 64,
                "floor_position": [3.0, 0.0, 4.0],
                "yaw_rad": -0.4,
            }],
        }],
    }


def test_consumed_goal_reuse_detects_hash_or_pose():
    old = {
        "goal_rgb_sha256": "c" * 64,
        "floor_position": [8.0, 0.0, 9.0],
        "yaw_rad": 1.2,
    }
    query = {
        "goal_rgb_sha256": "c" * 64,
        "floor_position": [0.0, 0.0, 0.0],
        "yaw_rad": 0.0,
    }
    assert query_reuses_source_goal(query, old)
    query["goal_rgb_sha256"] = "d" * 64
    query["floor_position"] = list(old["floor_position"])
    query["yaw_rad"] = old["yaw_rad"]
    assert query_reuses_source_goal(query, old)
    query["floor_position"][0] += 0.01
    assert not query_reuses_source_goal(query, old)


def test_assert_new_query_identity_checks_both_roles():
    row = history("scene", "episode", "front")
    old = {
        "goal_rgb_sha256": "z" * 64,
        "floor_position": [8.0, 0.0, 9.0],
        "yaw_rad": 1.2,
    }
    assert_new_query_identity(row, old)
    reused = copy.deepcopy(row)
    reused["pairs"][0]["queries"][1]["goal_rgb_sha256"] = "z" * 64
    with pytest.raises(RuntimeError, match="reused"):
        assert_new_query_identity(reused, old)


def test_assert_new_query_identity_checks_every_consumed_query():
    row = history("scene", "episode", "front")
    forbidden = [{
        "goal_rgb_sha256": "z" * 64,
        "floor_position": [8.0, 0.0, 9.0],
        "yaw_rad": 1.2,
    }, {
        "goal_rgb_sha256": "y" * 64,
        "floor_position": [7.0, 0.0, 6.0],
        "yaw_rad": -0.8,
    }]
    assert_new_query_identity(row, forbidden)
    reused = copy.deepcopy(row)
    reused["pairs"][0]["queries"][0]["floor_position"] = [7.0, 0.0, 6.0]
    reused["pairs"][0]["queries"][0]["yaw_rad"] = -0.8
    with pytest.raises(RuntimeError, match="consumed query"):
        assert_new_query_identity(reused, forbidden)


def test_direction_order_is_deterministic_complete_and_preferred_first():
    first = stratum_order(3, 1, "scene", "episode")
    assert first == stratum_order(3, 1, "scene", "episode")
    assert set(first) == set(STRATA)
    assert len(first) == len(set(first))


def test_power_gate_requires_size_scenes_and_direction_coverage():
    rows = [
        history(f"scene_{index % 12}", f"episode_{index}",
                STRATA[index % len(STRATA)])
        for index in range(20)
    ]
    result = power(
        rows, target_histories=20, target_scenes=12,
        minimum_per_stratum=4,
    )
    assert result["target_met"] is True
    without_rear = [
        row for row in rows
        if row["pairs"][0]["queries"][0]["assigned_direction_stratum"]
        != "rear"
    ]
    result = power(
        without_rear, target_histories=12, target_scenes=8,
        minimum_per_stratum=4,
    )
    assert result["target_met"] is False
