from __future__ import annotations

import copy
import json
from pathlib import Path

from audit_hm3d_table3_navmesh_capacity import (
    _select_population,
    circular_separation_degrees,
    in_bin,
    load_protocol,
    select_scene_triads,
)


PROTOCOL = Path(__file__).with_name(
    "hm3d_table3_navmesh_capacity_protocol_20260830.json"
)


def test_boundaries_are_disjoint_and_cover_requested_edges() -> None:
    protocol = load_protocol(PROTOCOL)
    specs = protocol["length_definition"]["bins_m"]
    assert in_bin(19.999, specs[0])
    assert not in_bin(20.0, specs[0]) and in_bin(20.0, specs[1])
    assert not in_bin(30.0, specs[1]) and in_bin(30.0, specs[2])
    assert in_bin(50.0, specs[2]) and not in_bin(50.001, specs[2])


def test_circular_bearing_separation() -> None:
    assert circular_separation_degrees(0.0, 0.0) == 0.0
    assert abs(circular_separation_degrees(3.13, -3.13) - 1.3289) < 0.01
    assert circular_separation_degrees(0.0, 3.141592653589793) == 180.0


def test_triad_requires_matched_distance_and_separated_bearing() -> None:
    protocol = load_protocol(PROTOCOL)
    points = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]
    distances = [
        [0.0, 10.0, 10.5],
        [10.0, 0.0, 20.5],
        [10.5, 20.5, 0.0],
    ]
    bearings = [
        [None, 0.0, 3.141592653589793],
        [3.141592653589793, None, 3.141592653589793],
        [0.0, 0.0, None],
    ]
    rows = select_scene_triads(points, distances, bearings, protocol)
    assert len(rows["0_to_20_m"]) == 1
    assert rows["0_to_20_m"][0]["query_start_sample"] == 0
    broken = copy.deepcopy(distances)
    broken[0][2] = broken[2][0] = 13.0
    assert not select_scene_triads(points, broken, bearings, protocol)["0_to_20_m"]


def _fragment(scene: str, bins: tuple[str, ...]) -> dict:
    base = {
        "query_start_sample": 0,
        "first_goal_sample": 1,
        "second_goal_sample": 2,
        "query_start": [0.0, 0.0, 0.0],
        "first_goal": [1.0, 0.0, 0.0],
        "second_goal": [-1.0, 0.0, 0.0],
        "first_goal_geodesic_m": 10.0,
        "second_goal_geodesic_m": 10.0,
        "goal_distance_mismatch": 0.0,
        "goal_to_goal_geodesic_m": 2.0,
        "initial_bearing_separation_deg": 180.0,
        "ranking": [0, 0, -180, 0, 1, 2],
    }
    return {
        "scene": scene,
        "candidate_triads": {
            name: ([dict(base), dict(base, query_start_sample=3)] if name in bins else [])
            for name in ("0_to_20_m", "20_to_30_m", "30_to_50_m")
        },
    }


def test_population_selection_prioritizes_scene_breadth() -> None:
    protocol = load_protocol(PROTOCOL)
    fragments = [_fragment(f"scene_{index:02d}", ("0_to_20_m",))
                 for index in range(12)]
    selected, diagnostics = _select_population(fragments, protocol)
    assert len(selected["0_to_20_m"]) == 16
    assert diagnostics["0_to_20_m"]["selected_scene_clusters"] == 12
    assert diagnostics["0_to_20_m"]["geometry_capacity_gate_passed"] is True
    assert diagnostics["20_to_30_m"]["geometry_capacity_gate_passed"] is False


def test_protocol_is_valid_json_and_geometry_stage_never_authorizes_policy() -> None:
    payload = json.loads(PROTOCOL.read_text())
    assert payload["authority_boundary"]["this_audit_authorizes_policy_evaluation"] is False
    assert payload["authority_boundary"]["threshold_relaxation_after_audit"] is False
