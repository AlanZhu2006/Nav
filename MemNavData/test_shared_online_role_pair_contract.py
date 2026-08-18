import copy

import pytest

from shared_online_role_pair_contract import (
    RUNTIME_VISIBLE_QUERY_FIELDS,
    SCHEMA_VERSION,
    runtime_query,
    validate_manifest,
)


SHA = "a" * 64


def query(role: str, distance: float, covis: float) -> dict:
    return {
        "query_id": f"pair_00_{role}",
        "analysis_role": role,
        "goal_rgb": f"pair_00/{role}/goal.jpg",
        "goal_rgb_sha256": SHA,
        "goal_depth": f"pair_00/{role}/goal_depth.png",
        "goal_depth_sha256": SHA,
        "floor_position": [1.0, 0.0, 2.0],
        "yaw_rad": 0.2,
        "geodesic_from_a_end_m": distance,
        "initial_path_bearing_rad": 0.30 if role == "novel" else 0.20,
        "max_online_a_covis": covis,
    }


def manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "online_history": "frozen_native_navdp_goal_a_rgb_depth_pose_trace",
            "query_execution": "independent_reset_and_exact_online_a_replay",
            "runtime_role_visibility": "none",
            "minimum_query_geodesic_m": 2.0,
            "maximum_role_distance_error_m": 0.5,
            "maximum_role_initial_path_bearing_error_deg": 30.0,
            "novel_max_online_a_covis_exclusive": 0.10,
            "revisit_min_online_a_covis_inclusive": 0.50,
            "revisit_max_online_a_covis_inclusive": 0.98,
        },
        "episodes": [
            {
                "schema_version": SCHEMA_VERSION,
                "scene": "scene",
                "episode": "episode_0000",
                "online_a_receipt_sha256": SHA,
                "online_a_trace_sha256": SHA,
                "online_a_endpoint": {
                    "floor_position": [0.0, 0.0, 0.0],
                    "yaw_rad": 0.0,
                },
                "pairs": [
                    {
                        "pair_id": "pair_00",
                        "role_distance_error_m": 0.2,
                        "role_initial_path_bearing_error_deg": 5.7295779513082365,
                        "queries": [
                            query("novel", 3.2, 0.02),
                            query("revisit", 3.0, 0.70),
                        ],
                    }
                ],
            }
        ],
    }


def test_valid_manifest_and_runtime_projection_hides_role():
    payload = manifest()
    assert validate_manifest(payload) is payload
    projected = runtime_query(payload["episodes"][0]["pairs"][0]["queries"][0])
    assert set(projected) == set(RUNTIME_VISIBLE_QUERY_FIELDS)
    assert "analysis_role" not in projected
    assert "max_online_a_covis" not in projected


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["episodes"][0]["pairs"][0]["queries"][0].update(
            max_online_a_covis=0.20), "Novel query has online-A visual support"),
        (lambda data: data["episodes"][0]["pairs"][0]["queries"][1].update(
            max_online_a_covis=0.40), "Revisit query is outside"),
        (lambda data: data["episodes"][0]["pairs"][0]["queries"][0].update(
            geodesic_from_a_end_m=4.0), "geodesics are not matched"),
        (lambda data: data["episodes"][0]["pairs"][0]["queries"][0].update(
            initial_path_bearing_rad=2.0), "path bearings are not matched"),
        (lambda data: data["contract"].update(runtime_role_visibility="role"),
         "runtime must not receive"),
    ],
)
def test_contract_fails_closed(mutation, message):
    payload = copy.deepcopy(manifest())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_manifest(payload)


def test_exclusive_revisit_upper_bound_is_enforced():
    payload = manifest()
    payload["contract"].update({
        "revisit_min_online_a_covis_inclusive": 0.25,
        "revisit_max_online_a_covis_inclusive": 0.55,
        "revisit_max_online_a_covis_is_exclusive": True,
    })
    revisit = payload["episodes"][0]["pairs"][0]["queries"][1]
    revisit["max_online_a_covis"] = 0.549999
    assert validate_manifest(payload) is payload
    revisit["max_online_a_covis"] = 0.55
    with pytest.raises(ValueError, match="Revisit query is outside"):
        validate_manifest(payload)
