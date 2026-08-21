import numpy as np
import pandas as pd

from MemNavData.train_cdec_scene_oof import (
    SessionTable,
    calibration_scenes,
    decision_metrics,
    teacher_top_index,
    zero_empirical_fp_threshold,
)


def toy_table():
    return SessionTable(
        session_id=np.asarray(["p", "n", "m"]),
        scene=np.asarray(["a", "b", "c"]),
        session_label=np.asarray([1, 0, 1]),
        candidate_label=np.asarray([[0, 1], [0, 0], [0, 0]]),
        features=np.zeros((3, 2, 4), dtype=np.float32),
        certificate_pass=np.zeros((3, 2), dtype=bool),
        teacher_top_index=np.asarray([1, 0, 0]),
        dino_cosine=np.zeros((3, 2), dtype=np.float32),
        candidate_frame=np.asarray([[1, 2], [1, 2], [1, 2]]),
    )


def test_decision_metrics_distinguish_shortlist_miss():
    data = toy_table()
    result = decision_metrics(
        data, np.arange(3), np.asarray([1, 0, 0]),
        np.asarray([True, False, False]))
    assert result["correct_anchor"] == 1
    assert result["strict_negative_false_activation"] == 0
    assert result["shortlist_miss_false_activation"] == 0
    assert result["exact_safe_action"] == 3


def test_zero_fp_threshold_is_strictly_above_negative_max():
    margin = np.asarray([0.2, 0.8, 3.0])
    labels = np.asarray([0, 0, 1])
    threshold = zero_empirical_fp_threshold(margin, labels)
    assert threshold > 0.8
    assert not np.any(margin[labels == 0] > threshold)


def test_calibration_scene_split_is_deterministic_and_proper():
    scenes = np.asarray([f"s{i}" for i in range(12)])
    first = calibration_scenes(scenes, 2, 4)
    second = calibration_scenes(scenes[::-1], 2, 4)
    assert first == second
    assert len(first) == 4


def test_teacher_rank_is_lexicographic():
    frame = pd.DataFrame({
        "fundamental_inliers": [50, 51, 51],
        "fundamental_query_grid_coverage": [1.0, 0.2, 0.3],
        "fundamental_query_hull_coverage": [1.0, 1.0, 0.5],
        "lightglue_score_median": [1.0, 1.0, 1.0],
        "dino_cosine": [1.0, 1.0, 1.0],
        "candidate_frame": [1, 2, 3],
    })
    assert teacher_top_index(frame) == 2
