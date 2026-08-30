from __future__ import annotations

import copy

import pytest

from MemNavData.hm3d_table3_length_contract import (
    SCHEMA_VERSION, runtime_query, validate_manifest,
)
from MemNavData.finalize_hm3d_table3_actual_mono_population import select_powered


SHA = "a" * 64


def query(role: str, distance: float, bearing: float, covis: float) -> dict:
    return {
        "query_id": f"q_{role}", "analysis_role": role,
        "goal_rgb": f"{role}/goal.jpg", "goal_rgb_sha256": SHA,
        "goal_depth": f"{role}/goal.png", "goal_depth_sha256": SHA,
        "floor_position": [1.0, 0.0, 2.0], "yaw_rad": 0.2,
        "geodesic_from_a_end_m": distance,
        "initial_path_bearing_rad": bearing,
        "max_online_a_covis": covis,
    }


def manifest() -> dict:
    novel = query("novel", 24.0, 0.0, 0.05)
    revisit = query("revisit", 25.0, 1.2, 0.70)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "online_history": "actual_frozen_navdp_goal_a_causal_rgb_trace",
            "query_execution": "independent_reset_and_exact_online_a_replay",
            "runtime_role_visibility": "none",
            "bins_m": [
                {"name": "20_to_30_m", "lower_inclusive": 20.0,
                 "upper": 30.0, "upper_inclusive": False},
            ],
            "novel_max_covis_exclusive": 0.10,
            "revisit_min_covis_inclusive": 0.55,
            "maximum_role_distance_mismatch_m": 2.0,
            "minimum_initial_bearing_separation_deg": 60.0,
        },
        "episodes": [{
            "schema_version": SCHEMA_VERSION, "scene": "scene",
            "episode": "episode_table3", "bin_name": "20_to_30_m",
        "candidate_identity_sha256": SHA,
        "runtime_geometry": "content_addressed_pinned_navmesh",
        "runtime_navmesh": "/frozen/scene.navmesh",
        "runtime_navmesh_sha256": SHA,
        "online_a_receipt_sha256": SHA, "online_a_trace_sha256": SHA,
            "online_a_steps": 100,
            "online_a_endpoint": {"floor_position": [0, 0, 0], "yaw_rad": 0},
            "pairs": [{
                "pair_id": "pair_00", "role_distance_error_m": 1.0,
                "role_initial_path_bearing_separation_deg": 68.75493541569878,
                "queries": [novel, revisit],
            }],
        }],
    }


def test_table3_contract_accepts_distance_matched_direction_separated_roles():
    payload = manifest()
    assert validate_manifest(payload) is payload
    projected = runtime_query(payload["episodes"][0]["pairs"][0]["queries"][0])
    assert "analysis_role" not in projected
    assert "max_online_a_covis" not in projected


def test_table3_contract_rejects_a_bearing_matched_pair():
    payload = copy.deepcopy(manifest())
    pair = payload["episodes"][0]["pairs"][0]
    pair["queries"][1]["initial_path_bearing_rad"] = 0.2
    pair["role_initial_path_bearing_separation_deg"] = 11.459155902616466
    with pytest.raises(ValueError, match="insufficiently separated"):
        validate_manifest(payload)


def test_table3_contract_rejects_role_distance_escape():
    payload = copy.deepcopy(manifest())
    pair = payload["episodes"][0]["pairs"][0]
    pair["queries"][1]["geodesic_from_a_end_m"] = 28.0
    pair["role_distance_error_m"] = 4.0
    with pytest.raises(ValueError, match="not matched"):
        validate_manifest(payload)


def test_population_selection_establishes_scene_breadth_before_second_rows():
    rows = [
        {"scene": f"scene{i % 10}", "identity": i}
        for i in range(20)
    ]
    selected = select_powered(rows, histories=16, scenes=10,
                              maximum_per_scene=2)
    assert len(selected) == 16
    assert len({row["scene"] for row in selected}) == 10
    counts = {scene: sum(row["scene"] == scene for row in selected)
              for scene in {row["scene"] for row in selected}}
    assert max(counts.values()) <= 2
