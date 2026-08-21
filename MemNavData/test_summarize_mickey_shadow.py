from MemNavData.summarize_mickey_shadow import (
    choose_threshold,
    group_sessions,
    threshold_metrics,
)


def candidate(session, scene, session_label, candidate_label, score, rank):
    return {
        "pair_index": rank,
        "scene": scene,
        "session_id": session,
        "candidate_rank": rank,
        "candidate_label": candidate_label,
        "session_label": session_label,
        "dino_cosine": 1.0 - rank / 10,
        "fundamental_inliers": 0,
        "score": score,
        "pose_valid": True,
        "latency_ms": 1.0,
    }


def test_group_sessions_selects_support_and_tracks_correct_anchor():
    sessions = group_sessions([
        candidate("p", "a", 1, 0, 1.0, 0),
        candidate("p", "a", 1, 1, 3.0, 1),
        candidate("n", "b", 0, 0, 2.0, 0),
    ])
    positive = next(row for row in sessions if row["session_id"] == "p")
    assert positive["selected_candidate_label"] == 1
    assert positive["max_score"] == 3.0


def test_threshold_metrics_count_wrong_positive_anchor_as_incorrect_accept():
    sessions = [
        {"session_label": 1, "selected_candidate_label": 1,
         "max_score": 3.0, "selected_pose_valid": True},
        {"session_label": 1, "selected_candidate_label": 0,
         "max_score": 2.5, "selected_pose_valid": True},
        {"session_label": 0, "selected_candidate_label": 0,
         "max_score": 2.0, "selected_pose_valid": True},
    ]
    metric = threshold_metrics(sessions, 2.4)
    assert metric["accepted"] == 2
    assert metric["correct_accepted"] == 1
    assert metric["accepted_precision"] == 0.5
    assert metric["strict_negative_false_accepts"] == 0


def test_threshold_choice_maximizes_correct_coverage_under_constraints():
    sessions = [
        {"session_label": 1, "selected_candidate_label": 1,
         "max_score": 4.0, "selected_pose_valid": True},
        {"session_label": 1, "selected_candidate_label": 1,
         "max_score": 3.0, "selected_pose_valid": True},
        {"session_label": 0, "selected_candidate_label": 0,
         "max_score": 2.0, "selected_pose_valid": True},
    ]
    threshold, metric = choose_threshold(
        sessions, minimum_precision=1.0, maximum_negative_fpr=0.0)
    assert threshold == 3.0
    assert metric["correct_accepted"] == 2
