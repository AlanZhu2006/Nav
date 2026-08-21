from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit_lightglue_open_set_localization import (
    geometric_support,
    label_from_covis,
    select_rows,
    temporal_topk,
)


def test_label_from_covis_preserves_unknown_band():
    assert label_from_covis(0.8, 0.5, 0.2) == 1
    assert label_from_covis(0.1, 0.5, 0.2) == 0
    assert label_from_covis(0.3, 0.5, 0.2) == -1


def test_temporal_topk_is_dino_ordered_and_diverse():
    frame = pd.DataFrame({
        "dino_cosine": [0.9, 0.8, 0.7, 0.6],
        "candidate_frame": [10, 12, 20, 30],
    })
    assert list(temporal_topk(frame, top_k=3, minimum_gap=4)) == [0, 2, 3]


def test_geometric_support_recovers_consistent_epipolar_matches():
    rng = np.random.default_rng(4)
    points0 = rng.uniform([20, 20], [620, 340], (80, 2)).astype(np.float32)
    # A horizontal stereo displacement obeys a valid fundamental geometry.
    points1 = points0 + np.column_stack([
        rng.uniform(-25, -8, 80), rng.normal(0, 0.1, 80)
    ]).astype(np.float32)
    result = geometric_support(
        points0, points1, np.ones(80), (360, 640), (360, 640), 1.5)
    assert result["fundamental_inliers"] >= 70
    assert result["fundamental_inlier_ratio"] > 0.85
    assert result["fundamental_query_grid_coverage"] > 0.5


def test_geometric_support_fails_closed_on_degenerate_matches():
    points = np.repeat([[100.0, 100.0]], 16, axis=0).astype(np.float32)
    result = geometric_support(
        points, points, np.ones(16), (360, 640), (360, 640), 1.5)
    assert result["fundamental_inliers"] == 0
    assert result["fundamental_query_grid_coverage"] == 0.0


def test_static_selection_matches_executable_teacher_universe(tmp_path):
    chunk = (tmp_path / "mp3d_2leg" / "scene" / "episode" / "videos"
             / "chunk-000")
    chunk.mkdir(parents=True)
    np.savez(chunk / "lingbot_cam_cache.npz",
             cam_pose_enc=np.zeros((20, 9), dtype=np.float32))
    rows = []
    for frame, covis in ((8, 0.8), (17, 0.7), (18, 0.9), (12, np.nan)):
        rows.append({
            "session_id": "scene/episode/goal",
            "scene": "scene",
            "episode": "episode",
            "kind": "cross_episode_train",
            "query_path": "/data/mp3d_2leg/scene/query/16.jpg",
            "candidate_path": (
                f"/data/mp3d_2leg/scene/episode/videos/chunk-000/"
                f"observation.images.rgb/{frame}.jpg"),
            "candidate_frame": frame,
            "dino_cosine": 1.0 - frame / 100.0,
            "teacher_covis": covis,
        })
    selected = select_rows(
        pd.DataFrame(rows), feature_root=tmp_path,
        kind="cross_episode_train", scenes=(), sessions=(), top_k=8,
        minimum_gap=1, minimum_anchor=8, positive=0.5, negative=0.2)
    # len(cam_pose_enc)-2 is 18: anchor+1 must exist.  The NaN teacher row is
    # outside the shared collector universe even though its image is present.
    assert selected["candidate_frame"].tolist() == [8, 17, 18]
