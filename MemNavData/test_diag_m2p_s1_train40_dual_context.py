import math

import pandas as pd

from MemNavData.diag_m2p_s1_train40_dual_context import (
    _boolean_series,
    binary_auc,
    bootstrap_positive_cdf30_delta,
    circular_difference_degrees,
)


def test_circular_difference_wraps_at_180():
    assert circular_difference_degrees(179.0, -179.0) == 2.0
    assert circular_difference_degrees(-170.0, 170.0) == 20.0


def test_binary_auc_handles_ties_exactly():
    assert binary_auc([1, 1, 0, 0], [3.0, 2.0, 1.0, 0.0]) == 1.0
    assert binary_auc([1, 0], [1.0, 1.0]) == 0.5
    assert math.isnan(binary_auc([1], [1.0]))


def test_boolean_series_does_not_treat_false_string_as_true():
    parsed = _boolean_series(
        pd.Series(["True", "False", "1", "0"]), label="test")
    assert parsed.tolist() == [True, False, True, False]


def test_positive_cdf_bootstrap_reports_paired_point_delta():
    rows = [
        {"scene": "a", "session_label": 1,
         "global_direction_error_deg": 5.0,
         "local_direction_error_deg": 5.0},
        {"scene": "a", "session_label": 1,
         "global_direction_error_deg": 5.0,
         "local_direction_error_deg": 50.0},
        {"scene": "b", "session_label": 1,
         "global_direction_error_deg": 5.0,
         "local_direction_error_deg": 5.0},
        {"scene": "b", "session_label": 1,
         "global_direction_error_deg": 5.0,
         "local_direction_error_deg": 50.0},
    ]
    result = bootstrap_positive_cdf30_delta(
        rows, resamples=100, seed=7)
    assert result["point_delta"] == 0.5
    assert result["ci95"] == [0.5, 0.5]
