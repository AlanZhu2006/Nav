from __future__ import annotations

from freeze_hm3d_table3_actual_mono_plan import round_robin_prefix


def test_round_robin_prefix_prioritizes_scene_breadth() -> None:
    rows = {
        "b": [{"value": "b0"}, {"value": "b1"}],
        "a": [{"value": "a0"}, {"value": "a1"}],
        "c": [{"value": "c0"}],
    }
    selected = round_robin_prefix(rows, 4)
    assert [(row["value"], row["scene"], row["capacity_candidate_rank"])
            for row in selected] == [
        ("a0", "a", 0), ("b0", "b", 0), ("c0", "c", 0),
        ("a1", "a", 1),
    ]


def test_round_robin_prefix_rejects_insufficient_capacity() -> None:
    try:
        round_robin_prefix({"a": [{}]}, 2)
    except RuntimeError as error:
        assert "only 1/2" in str(error)
    else:
        raise AssertionError("insufficient capacity was accepted")
