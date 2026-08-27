import json

from MemNavData.build_monocular_geometry_shards import (
    balanced_episode_subset,
    discover_episode_pairs,
    frame_schedule,
    goal_path_for_frame,
    load_teacher_depth_audit,
    validate_causal_scale_receipt,
)


def _episode(root, scene, name, *, feature=False, group=""):
    episode = root / group / scene / name
    chunk = episode / "videos/chunk-000"
    chunk.mkdir(parents=True)
    if feature:
        (chunk / "lingbot_cache.npz").touch()
        (chunk / "lingbot_cam_cache.npz").touch()
    else:
        (episode / "meta").mkdir()
        (episode / "meta/gen_meta.json").write_text("{}")
        (chunk / "observation.images.rgb").mkdir()
        (chunk / "observation.images.depth").mkdir()
    return episode


def test_discovery_joins_by_scene_episode_across_different_prefixes(tmp_path):
    raw = tmp_path / "raw"
    features = tmp_path / "features/prefix"
    _episode(raw, "scene_a", "episode_0")
    _episode(features, "scene_a", "episode_0", feature=True)
    _episode(raw, "scene_b", "episode_0")
    rows = discover_episode_pairs(raw, tmp_path / "features", {"scene_a"})
    assert [(row["scene"], row["episode_name"]) for row in rows] == [
        ("scene_a", "episode_0")
    ]


def test_discovery_keeps_same_episode_name_in_two_pt1_groups(tmp_path):
    raw = tmp_path / "raw"
    features = tmp_path / "features"
    for group in ("mp3d_2leg", "mp3d_3leg"):
        _episode(raw, "scene_a", "episode_0000", group=group)
        _episode(features, "scene_a", "episode_0000", group=group, feature=True)
    rows = discover_episode_pairs(raw, features, {"scene_a"})
    assert [(row["group"], row["scene"], row["episode_name"]) for row in rows] == [
        ("mp3d_2leg", "scene_a", "episode_0000"),
        ("mp3d_3leg", "scene_a", "episode_0000"),
    ]


def test_balanced_subset_caps_each_scene_deterministically():
    rows = [
        {"scene": scene, "episode_name": f"episode_{index}"}
        for scene in ("a", "b")
        for index in range(5)
    ]
    first = balanced_episode_subset(rows, 2)
    second = balanced_episode_subset(list(reversed(rows)), 2)
    assert first == second
    assert sum(row["scene"] == "a" for row in first) == 2
    assert sum(row["scene"] == "b" for row in first) == 2


def test_frame_schedule_is_causal_unique_and_bounded():
    frames = frame_schedule({"n_frames": 101}, 90, 8)
    assert frames == sorted(set(frames))
    assert frames[0] == 40
    assert frames[-1] == 88


def test_goal_path_tracks_active_leg(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    for name in ("goal_image.jpg", "goal_1.jpg", "goal_2.jpg"):
        (episode / name).touch()
    meta = {"switches": [100, 200]}
    assert goal_path_for_frame(episode, meta, 99)[1] == "goal_a"
    assert goal_path_for_frame(episode, meta, 100)[1] == "goal_1"
    assert goal_path_for_frame(episode, meta, 200)[1] == "goal_2"


def test_causal_scale_receipt_rejects_whole_episode_cache():
    receipt = {
        "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
        "whole_episode_ground_cache_consumed": False,
        "scale_prefix_frames": 40,
        "scale_prefix_first_frame": 0,
        "scale_prefix_last_frame": 39,
    }
    validate_causal_scale_receipt(receipt, 40)
    receipt["whole_episode_ground_cache_consumed"] = True
    import pytest
    with pytest.raises(RuntimeError, match="whole-episode"):
        validate_causal_scale_receipt(receipt, 40)


def test_teacher_depth_audit_binds_only_explicit_all_zero_state(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({
        "schema": "monocular_geometry_teacher_depth_population_audit_v1_20260818",
        "status": "complete",
        "input_quality_only_not_model_selection": True,
        "population_unchanged": True,
        "selected_state_count": 4,
        "valid_state_count": 3,
        "invalid_state_count": 1,
        "invalid_reason_counts": {"all_zero_depth": 1},
        "invalid_states": [{
            "group": "mp3d_2leg",
            "scene": "scene_a",
            "episode_name": "episode_0",
            "frame": 240,
            "depth": {"valid": False, "reason": "all_zero_depth"},
        }],
    }))
    keys, receipt = load_teacher_depth_audit(path)
    assert keys == {("mp3d_2leg", "scene_a", "episode_0", 240)}
    assert receipt["invalid_state_count"] == 1
