from copy import deepcopy
import stat
from pathlib import Path

import pytest

from MemNavData.final14_role_pair_contract import role_contract
from MemNavData.finalize_hm3d_table2_leg3_mixed_role import (
    remove_temporary_tree,
)
from MemNavData.hm3d_table2_leg3_mixed_role import (
    compose_actual_ab_trace,
    load_protocol,
    power,
    stratum_order,
)


PROTOCOL = Path(__file__).with_name(
    "hm3d_table2_leg3_mixed_role_protocol_20260829.json"
)


def trace(name: str, *, start: int = 0, count: int = 2) -> dict:
    poses = [
        {
            "step": index,
            "x": float(start + index), "y": 0.0, "z": 0.0,
            "yaw": 0.0, "jpg_sha256": f"{index + start + 1:064x}",
        }
        for index in range(count)
    ]
    return {
        "schema_version": 1,
        "episode": name,
        "episode_seed": 7,
        "goal_sha256": "a" * 64,
        "source_scene": "scene",
        "source_backend": "hybrid_pose",
        "source_hybrid_route": "native_sidecar",
        "source_retrieval_candidate_min_gap": 16,
        "source_graph_subgoal_spacing_m": 0.0,
        "source_graph_subgoal_arrival_m": 0.6,
        "goal_source_episode": name,
        "reached": True,
        "steps": count,
        "poses": poses,
        "plans": [{
            "step": 0,
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed": False,
        }],
        "path_len": float(count),
        "path_len_at_reach": float(count),
        "step_at_reach": count - 1,
        "final_goal_dist_m": 0.2,
        "end_position": [float(start + count - 1), 0.0, 0.0],
        "end_yaw": 0.0,
    }


def row(stratum: str, scene: str, episode: str) -> dict:
    return {
        "scene": scene,
        "episode": episode,
        "pairs": [{
            "queries": [{
                "analysis_role": "novel",
                "assigned_direction_stratum": stratum,
            }, {"analysis_role": "revisit"}],
        }],
    }


def test_protocol_is_frozen_and_self_consistent() -> None:
    payload = load_protocol(PROTOCOL)
    assert payload["population_gate"]["minimum_histories"] == 16
    assert payload["leg3_queries"]["both_query_identities_are_new"] is True


def test_role_contract_is_pure_and_matches_the_frozen_standard_band() -> None:
    contract = role_contract(support="standard")
    assert contract["runtime_role_visibility"] == "none"
    assert contract["novel_max_online_a_covis_exclusive"] == 0.10
    assert contract["revisit_min_online_a_covis_inclusive"] == 0.55
    assert contract["revisit_max_online_a_covis_inclusive"] == 0.90
    assert contract["novel_candidate_attempts"] == 5000
    assert contract["covis_depth_tolerance_m"] == 0.30
    with pytest.raises(ValueError, match="invalid Revisit support band"):
        role_contract(support="unknown")


def test_read_only_temporary_tree_is_removable(tmp_path: Path) -> None:
    tree = tmp_path / "population.tmp.test"
    child = tree / "scene" / "role_pairs.json"
    child.parent.mkdir(parents=True)
    child.write_text("{}")
    child.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    child.parent.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
        | stat.S_IROTH | stat.S_IXOTH
    )
    remove_temporary_tree(tree)
    assert not tree.exists()


def test_stratum_order_balances_the_preferred_direction() -> None:
    orders = [stratum_order(index, "scene", f"episode_{index:04d}")
              for index in range(9)]
    assert [order[0] for order in orders] == [
        "front", "side", "rear", "front", "side", "rear",
        "front", "side", "rear",
    ]
    assert all(set(order) == {"front", "side", "rear"} for order in orders)


def test_actual_ab_trace_is_dense_and_preserves_receipts() -> None:
    a = trace("episode_0001", start=0, count=3)
    b = trace("episode_0001", start=3, count=2)
    result = compose_actual_ab_trace(a, b, episode="episode_0001")
    assert result["steps"] == 5
    assert [pose["step"] for pose in result["poses"]] == list(range(5))
    assert [plan["step"] for plan in result["plans"]] == [0, 3]
    assert result["prefix_A_steps"] == 3
    assert result["prefix_B_steps"] == 2
    assert result["path_len"] == 5.0


def test_actual_ab_trace_rejects_metric_depth() -> None:
    a = trace("episode_0001")
    b = trace("episode_0001", start=2)
    b = deepcopy(b)
    b["plans"][0]["metric_depth_sensor_consumed"] = True
    with pytest.raises(RuntimeError, match="fully monocular"):
        compose_actual_ab_trace(a, b, episode="episode_0001")


def test_power_requires_histories_scenes_and_each_direction() -> None:
    rows = [
        row(("front", "side", "rear")[index % 3], f"scene_{index % 10}",
            f"episode_{index:04d}")
        for index in range(18)
    ]
    observed = power(
        rows, target_histories=16, target_scenes=10,
        minimum_per_stratum=3,
    )
    assert observed["target_met"] is True
    without_rear = [item for item in rows
                    if item["pairs"][0]["queries"][0][
                        "assigned_direction_stratum"] != "rear"]
    assert power(
        without_rear, target_histories=1, target_scenes=1,
        minimum_per_stratum=1,
    )["target_met"] is False
