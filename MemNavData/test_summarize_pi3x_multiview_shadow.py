import math

from MemNavData.summarize_pi3x_multiview_shadow import (
    OBSERVABLE_FEATURES,
    _nested_scene_oof,
    choose_threshold,
    evaluate_picks,
)


def _pick(session_id, session_label, candidate_label, score, bearing=5.0):
    return {
        "session_id": session_id,
        "scene": session_id.split("_")[0],
        "session_label": session_label,
        "selected_candidate_label": candidate_label,
        "score": score,
        "bearing_error_deg": bearing,
    }


def test_evaluate_picks_counts_wrong_positive_selection_as_error() -> None:
    picks = [
        _pick("a_1", 1, 1, 0.9),
        _pick("b_1", 1, 0, 0.8),
        _pick("c_1", 0, 0, 0.7, 120.0),
        _pick("d_1", -1, -1, 0.95),
    ]
    result = evaluate_picks(picks, 0.75)
    assert result["correct_positive_accepts"] == 1
    assert result["wrong_candidate_accepts_in_positive_sessions"] == 1
    assert result["strict_negative_false_accepts"] == 0
    assert result["ambiguous_sessions_excluded"] == 1
    assert result["accepted_precision"] == 0.5


def test_choose_threshold_is_training_only_risk_constraint() -> None:
    picks = [
        _pick("a_1", 1, 1, 0.95),
        _pick("b_1", 1, 1, 0.90),
        _pick("c_1", 0, 0, 0.85),
        _pick("d_1", 0, 0, 0.10),
    ]
    threshold, result = choose_threshold(
        picks, minimum_precision=1.0, maximum_fpr=0.0
    )
    assert threshold == 0.90
    assert result["correct_positive_accepts"] == 2
    assert result["strict_negative_false_accepts"] == 0


def test_empty_acceptance_is_fail_closed() -> None:
    result = evaluate_picks([_pick("a_1", 1, 1, 0.2)], math.inf)
    assert result["accepted_known_sessions"] == 0
    assert result["accepted_precision"] == 1.0
    assert result["positive_session_recall"] == 0.0


def test_nested_oof_never_drops_a_scene_or_candidate() -> None:
    rows = []
    row_index = 0
    for scene_index in range(8):
        scene = f"scene_{scene_index}"
        for session_kind in ("positive", "negative"):
            session_label = 1 if session_kind == "positive" else 0
            for rank in range(2):
                candidate_label = int(session_label == 1 and rank == 0)
                value = 0.9 if candidate_label else 0.1
                row = {
                    feature: value for feature in OBSERVABLE_FEATURES
                }
                row.update({
                    "row_index": row_index,
                    "scene": scene,
                    "session_id": f"{scene}/{session_kind}",
                    "session_label": session_label,
                    "candidate_label": candidate_label,
                    "navigation_action_label": candidate_label,
                    "candidate_rank": rank,
                    "goal_bearing_error_deg_reporting_only": (
                        2.0 if candidate_label else 120.0
                    ),
                })
                rows.append(row)
                row_index += 1
    scores, picks, folds = _nested_scene_oof(
        rows,
        outer_splits=4,
        inner_splits=3,
        minimum_precision=0.9,
        maximum_fpr=0.1,
        target_key="candidate_label",
        correctness_key="selected_candidate_label",
    )
    assert set(scores) == set(range(len(rows)))
    assert len(picks) == 16
    assert len(folds) == 4
