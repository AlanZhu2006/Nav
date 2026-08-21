import copy

from MemNavData.train_pi3x_viewtoken_reliability_crossfit_oof import (
    _ensemble_picks,
    _within_session_borda,
)


def _rows():
    return [
        {"session_id": "s0", "scene": "scene", "session_label": 1,
         "candidate_rank": 0, "candidate_label": 0,
         "navigation_action_label": 0, "bearing_error_deg": 100.0,
         "row_index": 0},
        {"session_id": "s0", "scene": "scene", "session_label": 1,
         "candidate_rank": 1, "candidate_label": 1,
         "navigation_action_label": 1, "bearing_error_deg": 2.0,
         "row_index": 1},
        {"session_id": "s1", "scene": "scene", "session_label": 0,
         "candidate_rank": 0, "candidate_label": 0,
         "navigation_action_label": 0, "bearing_error_deg": 80.0,
         "row_index": 2},
        {"session_id": "s1", "scene": "scene", "session_label": 0,
         "candidate_rank": 1, "candidate_label": 0,
         "navigation_action_label": 0, "bearing_error_deg": 70.0,
         "row_index": 3},
    ]


def test_borda_is_invariant_to_monotone_member_scale() -> None:
    rows = _rows()
    scores = {0: 0.1, 1: 0.9, 2: 0.3, 3: 0.2}
    transformed = {index: 10.0 + 100.0 * value for index, value in scores.items()}
    assert _within_session_borda(rows, scores, {"scene"}) == _within_session_borda(
        rows, transformed, {"scene"}
    )


def test_ensemble_keeps_member_thresholds_bound_and_uses_consensus() -> None:
    rows = _rows()
    member_scores = [
        {0: 0.2, 1: 0.9, 2: 0.8, 3: 0.1},
        {0: 0.1, 1: 0.7, 2: 0.6, 3: 0.2},
        {0: 0.4, 1: 0.8, 2: 0.55, 3: 0.2},
        {0: 0.3, 1: 0.6, 2: 0.51, 3: 0.2},
    ]
    thresholds = [0.85, 0.65, 0.75, 0.65]
    picks, _scores, votes = _ensemble_picks(
        rows, member_scores, thresholds, {"scene"}, consensus=3
    )
    by_session = {pick["session_id"]: pick for pick in picks}
    assert by_session["s0"]["selected_row_index"] == 1
    assert votes[1] == 3
    assert by_session["s0"]["accepted"] is True
    assert by_session["s1"]["selected_row_index"] == 2
    assert votes[2] == 0
    assert by_session["s1"]["accepted"] is False


def test_consensus_change_only_changes_authorization_not_ranking() -> None:
    rows = _rows()
    members = [{0: 0.1, 1: 0.9, 2: 0.7, 3: 0.2}] * 4
    thresholds = [0.8, 0.8, 0.6, 0.6]
    loose, loose_scores, _ = _ensemble_picks(
        rows, members, thresholds, {"scene"}, consensus=2
    )
    strict, strict_scores, _ = _ensemble_picks(
        copy.deepcopy(rows), members, thresholds, {"scene"}, consensus=4
    )
    assert loose_scores == strict_scores
    assert [p["selected_row_index"] for p in loose] == [
        p["selected_row_index"] for p in strict
    ]
    assert [p["accepted"] for p in loose] != [p["accepted"] for p in strict]
