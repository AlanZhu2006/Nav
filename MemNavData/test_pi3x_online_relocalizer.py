import numpy as np

from MemNavData.pi3x_online_relocalizer import (
    Pi3XOnlineRelocalizer,
    _best_view_overlap_20cm,
    _spatial_geometry,
    causal_bridge_frames,
    pack_candidate_evidence,
)
from MemNavData.diag_pi3x_multiview_consistency import (
    _causal_bridge_frames as diagnostic_bridge_frames,
    _overlap_metrics as diagnostic_overlap_metrics,
    _scale_free_spatial_geometry as diagnostic_spatial_geometry,
)


def _evidence(views: int, descriptor_dim: int = 8):
    return {
        "descriptors": np.ones((views, descriptor_dim), dtype=np.float16),
        "roles": [0, *([1] * (views - 2)), 3],
        "relative_age": np.linspace(0, 1, views).tolist(),
        "world_points_in_current": np.ones(
            (views, 3, 4, 3), dtype=np.float16
        ),
        "local_points": np.ones((views, 3, 4, 3), dtype=np.float16),
        "confidence": np.ones((views, 3, 4), dtype=np.float16),
        "poses_in_current": np.ones((views, 3, 4), dtype=np.float16),
    }


def test_b16_bridge_is_causal_and_contains_anchor_support():
    frames, support = causal_bridge_frames(20, 101)
    assert frames == sorted(set(frames), reverse=True)
    assert len(frames) >= 16
    assert max(frames) == 100
    assert min(frames) >= 0
    assert 20 in frames
    assert support == [12, 20, 28]
    assert all(frame < 101 for frame in frames)


def test_pack_candidate_evidence_pads_only_view_axis():
    packed = pack_candidate_evidence([_evidence(4), _evidence(6)])
    assert packed["descriptors"].shape == (2, 6, 8)
    assert packed["world_points_in_current"].shape == (2, 6, 3, 4, 3)
    assert packed["confidence"].shape == (2, 6, 3, 4, 1)
    assert packed["poses_in_current"].shape == (2, 6, 3, 4)
    assert packed["valid"].tolist() == [
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
    ]
    assert packed["roles"][0, 4:].tolist() == [-1, -1]


def test_online_feature_geometry_matches_frozen_diagnostic():
    rng = np.random.default_rng(3)
    views, height, width = 5, 126, 224
    poses = np.repeat(np.eye(4)[None], views, axis=0)
    poses[:, :3, 3] = rng.normal(size=(views, 3))
    local = rng.normal(size=(views, height, width, 3))
    local[..., 2] = np.abs(local[..., 2]) + 0.1
    world = local + poses[:, None, None, :3, 3]
    confidence = rng.uniform(size=(views, height, width))
    expected = diagnostic_spatial_geometry(
        poses.copy(), world.copy(), local.copy(), confidence.copy(),
        patch_size=14,
    )
    actual = _spatial_geometry(
        poses.copy(), world.copy(), local.copy(), confidence.copy(),
        patch_size=14,
    )
    assert actual.keys() == expected.keys()
    for key in actual:
        assert np.allclose(actual[key], expected[key])


def test_online_bridge_and_overlap_match_frozen_diagnostic():
    expected_frames, expected_support = diagnostic_bridge_frames(
        20, 101, 101, bridge_count=16, anchor_offsets=(-8, 0, 8)
    )
    assert causal_bridge_frames(20, 101) == (
        expected_frames, expected_support
    )
    rng = np.random.default_rng(9)
    goal = rng.normal(size=(100, 3))
    history = [goal + rng.normal(scale=0.05, size=goal.shape), rng.normal(size=(80, 3))]
    expected = diagnostic_overlap_metrics(goal, history)["best_view_f1_20cm"]
    assert np.isclose(_best_view_overlap_20cm(goal, history), expected)


def test_candidate_contract_is_top8_causal_and_temporally_diverse():
    candidates = [
        {"anchor": 8 + 4 * index, "score": 0.9 - 0.01 * index}
        for index in range(8)
    ]
    canonical = Pi3XOnlineRelocalizer._canonical_candidates(candidates, 100)
    assert [item["dino_rank"] for item in canonical] == list(range(1, 9))
