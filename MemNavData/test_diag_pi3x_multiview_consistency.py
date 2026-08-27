import numpy as np
import pytest
import torch

from MemNavData.diag_pi3x_multiview_consistency import (
    _history_frames,
    _inference_dtype,
    _pack_view_descriptors,
    _pack_spatial_geometry,
    _causal_bridge_frames,
    _overlap_metrics,
    _planar_angle_deg,
    _scale_free_bearing_from_c2w,
    _scale_free_spatial_geometry,
    _true_bearing_from_generator_pose,
    _transform,
    _umeyama,
    _verify_source_images,
)


def test_umeyama_recovers_similarity() -> None:
    source = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotation = np.asarray([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    target = _transform(source, 2.5, rotation, np.asarray([3.0, -2.0, 1.0]))
    scale, recovered_rotation, translation = _umeyama(source, target)
    assert scale == pytest.approx(2.5)
    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-8)
    np.testing.assert_allclose(
        _transform(source, scale, recovered_rotation, translation), target, atol=1e-8
    )


def test_history_frames_remain_causal_and_distinct() -> None:
    assert _history_frames(3, 6, 20, (-16, -8, 0, 8, 16)) == [0, 3, 5]
    assert _history_frames(15, 30, 20, (-8, 0, 8)) == [7, 15, 19]


def test_causal_bridge_connects_current_to_anchor_and_keeps_local_support() -> None:
    frames, support = _causal_bridge_frames(
        20, 101, 200, bridge_count=5, anchor_offsets=(-4, 0, 4)
    )
    assert frames == sorted(frames, reverse=True)
    assert frames[0] == 100
    assert frames[-1] == 16
    assert support == [16, 20, 24]
    assert set(support).issubset(frames)


def test_planar_angle_ignores_height() -> None:
    assert _planar_angle_deg(np.asarray([1.0, 0.0, 20.0]), np.asarray([0.0, 1.0, -4.0])) == pytest.approx(90.0)


def test_predicted_and_generator_bearing_conventions_agree() -> None:
    current_cv = np.eye(4)
    goal_cv = np.eye(4)
    goal_cv[:3, 3] = [1.0, 0.0, 2.0]
    np.testing.assert_allclose(
        _scale_free_bearing_from_c2w(current_cv, goal_cv), [2.0, -1.0]
    )

    current_gl = np.eye(4)
    true_world_goal = np.asarray([1.0, 0.0, -2.0])
    np.testing.assert_allclose(
        _true_bearing_from_generator_pose(current_gl, true_world_goal),
        [2.0, -1.0],
    )


def test_overlap_separates_aligned_and_distant_clouds() -> None:
    goal = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    aligned = _overlap_metrics(goal, [goal.copy()])
    distant = _overlap_metrics(goal, [goal + 5.0])
    assert aligned["best_view_f1_05cm"] == pytest.approx(1.0)
    assert distant["best_view_f1_50cm"] == pytest.approx(0.0)


def test_inference_dtype_can_force_float32_without_autocast() -> None:
    assert _inference_dtype(torch.device("cuda"), "float32") is None
    assert _inference_dtype(torch.device("cuda"), "bfloat16") is torch.bfloat16
    assert _inference_dtype(torch.device("cuda"), "float16") is torch.float16
    assert _inference_dtype(torch.device("cpu"), "auto") is None


def test_pack_view_descriptors_preserves_roles_mask_and_row_ids() -> None:
    packed = _pack_view_descriptors([
        {
            "row_index": 7,
            "descriptors": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "roles": [0, 3],
            "relative_age": [0.0, -1.0],
        },
        {
            "row_index": 2,
            "descriptors": np.asarray([[5.0, 6.0]]),
            "roles": [2],
            "relative_age": [0.5],
        },
    ])
    assert packed["view_descriptors"].shape == (2, 2, 2)
    assert packed["row_indices"].tolist() == [7, 2]
    assert packed["view_counts"].tolist() == [2, 1]
    assert packed["view_roles"].tolist() == [[0, 3], [2, -1]]
    assert packed["view_valid"].tolist() == [[True, True], [True, False]]


def test_scale_free_spatial_geometry_uses_current_gauge_without_gt() -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[0, :3, 3] = [2.0, 0.0, 0.0]
    poses[1, :3, 3] = [4.0, 0.0, 0.0]
    local = np.zeros((2, 14, 14, 3), dtype=np.float64)
    local[..., 2] = 2.0
    world = local.copy()
    world[0, ..., 0] += 2.0
    world[1, ..., 0] += 4.0
    confidence = np.ones((2, 14, 14), dtype=np.float64)
    result = _scale_free_spatial_geometry(
        poses, world, local, confidence, patch_size=14
    )
    assert result["normalization_scale"] == pytest.approx(2.0)
    np.testing.assert_allclose(result["poses_in_current"][0], np.eye(4)[:3])
    assert result["world_points_in_current"].shape == (2, 1, 1, 3)
    assert float(result["poses_in_current"][1, 0, 3]) == pytest.approx(1.0)


def test_pack_spatial_geometry_pads_views_and_keeps_roles() -> None:
    def item(row, views):
        return {
            "row_index": row,
            "world_points_in_current": np.ones((views, 1, 2, 3)),
            "local_points": np.ones((views, 1, 2, 3)) * 2,
            "confidence": np.ones((views, 1, 2)),
            "poses_in_current": np.ones((views, 3, 4)),
            "normalization_scale": 3.0,
            "roles": [0] + [3] * (views - 1),
            "relative_age": [0.0] + [-1.0] * (views - 1),
        }
    packed = _pack_spatial_geometry([item(7, 2), item(2, 1)])
    assert packed["view_world_points_in_current"].shape == (2, 2, 1, 2, 3)
    assert packed["row_indices"].tolist() == [7, 2]
    assert packed["view_roles"].tolist() == [[0, 3], [0, -1]]
    assert packed["view_valid"].tolist() == [[True, True], [True, False]]


def test_verify_source_images_rejects_stale_same_named_data(tmp_path) -> None:
    import hashlib

    query = tmp_path / "scene/episode_0001/goal_1.jpg"
    candidate = tmp_path / (
        "scene/episode_0000/videos/chunk-000/observation.images.rgb/7.jpg"
    )
    query.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    query.write_bytes(b"query")
    candidate.write_bytes(b"candidate")
    row = {
        "scene": "scene",
        "episode": "episode_0000",
        "candidate_frame": "7",
        "query_relative_path": "scene/episode_0001/goal_1.jpg",
        "query_content_sha256": hashlib.sha256(b"query").hexdigest(),
        "candidate_rgb_content_sha256": hashlib.sha256(b"candidate").hexdigest(),
    }
    assert _verify_source_images(tmp_path, [row]) == 2
    row["candidate_rgb_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source image content mismatch"):
        _verify_source_images(tmp_path, [row])
