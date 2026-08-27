import pytest

from MemNavData.build_pi3x_shadow_input_manifest import required_paths


def test_required_paths_are_minimal_causal_and_include_query_metadata() -> None:
    rows = [{
        "scene": "scene",
        "episode": "episode_0000",
        "candidate_frame": "10",
        "decision_frame": "21",
        "query_relative_path": "scene/episode_0001/goal_1.jpg",
    }]
    paths = required_paths(rows, bridge_frames=3, anchor_offsets=(-2, 0, 2))
    assert "scene/episode_0000/videos/chunk-000/observation.images.rgb/21.jpg" in paths
    assert "scene/episode_0000/videos/chunk-000/observation.images.rgb/10.jpg" in paths
    assert "scene/episode_0000/data/chunk-000/episode_000000.parquet" in paths
    assert "scene/episode_0001/goal_1.jpg" in paths
    assert "scene/episode_0001/meta/gen_meta.json" in paths
    assert len(paths) == len(set(paths))


def test_required_paths_reject_future_anchor() -> None:
    row = {
        "scene": "scene",
        "episode": "episode_0000",
        "candidate_frame": "22",
        "decision_frame": "21",
        "query_relative_path": "scene/episode_0001/goal_1.jpg",
    }
    with pytest.raises(ValueError, match="non-causal"):
        required_paths([row], bridge_frames=3, anchor_offsets=(0,))
