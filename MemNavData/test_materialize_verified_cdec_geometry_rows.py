import pytest

from materialize_verified_cdec_geometry_rows import (
    EXPECTED_ORIGIN,
    select_geometry_rows,
)


def test_select_geometry_rows_keeps_exact_manifest_universe():
    rows = [
        {"session_id": "s1", "scene": "a",
         "candidate_selection_origin": EXPECTED_ORIGIN},
        {"session_id": "s1", "scene": "a",
         "candidate_selection_origin": "cdec_scene_oof_pairwise_rank_v1"},
        {"session_id": "s2", "scene": "b",
         "candidate_selection_origin": EXPECTED_ORIGIN},
        {"session_id": "s2", "scene": "b",
         "candidate_selection_origin": "cdec_scene_oof_pairwise_rank_v1"},
    ]
    selected, counts = select_geometry_rows(rows, {"s1", "s2"})
    assert [row["session_id"] for row in selected] == ["s1", "s2"]
    assert counts == {
        "cdec_scene_oof_pairwise_rank_v1": 2,
        EXPECTED_ORIGIN: 2,
    }


def test_select_geometry_rows_rejects_missing_manifest_session():
    rows = [{
        "session_id": "s1", "scene": "a",
        "candidate_selection_origin": EXPECTED_ORIGIN,
    }]
    with pytest.raises(RuntimeError, match="frozen session universe"):
        select_geometry_rows(rows, {"s1", "s2"})


def test_select_geometry_rows_rejects_duplicate_geometry_session():
    row = {
        "session_id": "s1", "scene": "a",
        "candidate_selection_origin": EXPECTED_ORIGIN,
    }
    with pytest.raises(RuntimeError, match="duplicate sessions"):
        select_geometry_rows([row, row], {"s1"})
