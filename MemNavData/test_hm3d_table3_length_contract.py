from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from MemNavData.audit_hm3d_table3_length_role_pairs import audit
from MemNavData.finalize_hm3d_table3_causal_survey_population import (
    select_powered,
)
from MemNavData.hm3d_table3_length_contract import (
    SCHEMA_VERSION, runtime_query, validate_manifest,
)
from MemNavData.hm3d_table3_causal_survey_contract import survey_frames


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


def test_table3_contract_accepts_disclosed_causal_survey_history():
    payload = manifest()
    payload["contract"]["online_history"] = (
        "controlled_causal_rgb_geodesic_survey"
    )
    assert validate_manifest(payload) is payload


def test_causal_survey_respects_translation_and_yaw_steps():
    points = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, -0.20]),
        np.asarray([-0.20, 0.0, -0.20]),
    ]
    frames = survey_frames(points, step_m=0.0376, maximum_yaw_step_deg=4.5)
    assert len(frames) > 12
    for (first_position, first_yaw), (second_position, second_yaw) in zip(
        frames[:-1], frames[1:]
    ):
        assert np.linalg.norm(second_position - first_position) <= 0.03761
        delta = (second_yaw - first_yaw + np.pi) % (2 * np.pi) - np.pi
        assert abs(np.degrees(delta)) <= 4.500001


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


def test_independent_audit_executes_the_causal_survey_branch(tmp_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    online = tmp_path / "survey"
    rgb_root = online / "rgb"
    rgb_root.mkdir(parents=True)
    poses = []
    frame_hashes = []
    for step in range(40):
        path = rgb_root / f"{step:06d}.jpg"
        path.write_bytes(f"rgb-{step}".encode())
        frame_hashes.append(digest(path))
        poses.append({"step": step, "jpg_sha256": digest(path)})
    trace = {
        "schema_version": "hm3d_table3_causal_survey_trace_v1_20260830",
        "source_hybrid_route": "causal_survey", "reached": True,
        "poses": poses, "metric_depth_sensor_reads": 0, "episode_seed": 7,
    }
    (online / "online_a_trace.json").write_text(
        json.dumps(trace, sort_keys=True) + "\n"
    )
    receipt = {
        "schema_version": "hm3d_table3_causal_survey_materialized_v1_20260830",
        "history_source": "controlled_causal_rgb_geodesic_survey",
        "camera_intrinsic": [[355.8, 0.0, 240.0],
                             [0.0, 351.7, 135.0], [0.0, 0.0, 1.0]],
        "episode_seed": 7,
        "survey_contract": {
            "runtime_memory_input": "RGB only",
            "construction_only_simulator_depth": True,
            "metric_depth_for_query_control_or_CEC": False,
        },
        "rgb_frame_hashes": frame_hashes,
    }
    (online / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )
    navmesh = tmp_path / "scene.navmesh"
    navmesh.write_bytes(b"pinned-navmesh")
    role_root = tmp_path / "population" / "role_pairs"
    episode_root = role_root / "scene" / "episode_table3"
    for role in ("novel", "revisit"):
        goal_root = episode_root / role
        goal_root.mkdir(parents=True)
        (goal_root / "goal.jpg").write_bytes(f"{role}-rgb".encode())
        (goal_root / "goal.png").write_bytes(f"{role}-depth".encode())
    payload = manifest()
    payload["contract"].update({
        "online_history": "controlled_causal_rgb_geodesic_survey",
        "minimum_histories_per_bin": 1,
        "minimum_scene_clusters_per_bin": 1,
    })
    episode = payload["episodes"][0]
    episode.update({
        "runtime_navmesh": str(navmesh),
        "runtime_navmesh_sha256": digest(navmesh),
        "online_a_episode": str(online),
        "online_a_receipt_sha256": digest(online / "receipt.json"),
        "online_a_trace_sha256": digest(online / "online_a_trace.json"),
        "online_a_steps": 40,
    })
    for role, query_payload in zip(
        ("novel", "revisit"), episode["pairs"][0]["queries"]
    ):
        query_payload["goal_rgb"] = f"{role}/goal.jpg"
        query_payload["goal_rgb_sha256"] = digest(
            episode_root / role / "goal.jpg"
        )
        query_payload["goal_depth"] = f"{role}/goal.png"
        query_payload["goal_depth_sha256"] = digest(
            episode_root / role / "goal.png"
        )
        value = 0.05 if role == "novel" else 0.70
        query_payload.update({
            "covis_curve": [value] * 40,
            "eligible_online_a_frame_floor": 1,
            "eligible_online_a_end_margin_frames": 1,
            "max_online_a_covis_frame": 1,
        })
    sidecar = episode_root / "role_pairs.json"
    sidecar.write_text(json.dumps(episode, sort_keys=True) + "\n")
    episode["role_pairs_sha256"] = digest(sidecar)
    manifest_path = role_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    (role_root / "manifest.json.sha256").write_text(
        f"{digest(manifest_path)}  manifest.json\n"
    )

    result = audit(role_root)
    assert result["ok"] is True
    assert result["online_history"] == (
        "controlled_causal_rgb_geodesic_survey"
    )
