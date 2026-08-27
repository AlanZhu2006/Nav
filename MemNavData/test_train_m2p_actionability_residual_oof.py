import pandas as pd

from MemNavData.train_m2p_actionability_residual_oof import (
    exact_mcnemar_p,
    top1_outcomes,
)


def test_exact_mcnemar():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(3, 0) == 0.25


def test_top1_tie_is_stable():
    table = pd.DataFrame({
        "session_id": ["s", "s"],
        "scene": ["a", "a"],
        "candidate_frame": [3, 7],
        "actionable": [True, False],
        "relative_position_direction_error_deg_center": [1.0, 90.0],
    })
    result = top1_outcomes(table, scores=[1.0, 1.0], sessions={"s"})
    assert int(result.iloc[0]["candidate_frame"]) == 3
