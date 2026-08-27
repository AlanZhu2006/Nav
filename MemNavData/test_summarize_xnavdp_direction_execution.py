from MemNavData.summarize_xnavdp_direction_execution import paired_counts


def test_paired_counts_are_directional():
    result = paired_counts(
        [True, True, False, False],
        [False, True, True, False],
    )
    assert result["left_hits"] == 2
    assert result["right_hits"] == 2
    assert result["gains"] == 1
    assert result["losses"] == 1
    assert result["exact_mcnemar_p"] == 1.0
