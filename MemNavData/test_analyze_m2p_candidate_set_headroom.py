import pandas as pd

from MemNavData.analyze_m2p_candidate_set_headroom import (
    error_summary,
)


def test_error_summary_reports_frozen_cdfs():
    summary = error_summary(pd.Series([1.0, 20.0, 40.0, 90.0]))
    assert summary["n"] == 4
    assert summary["cdf_le_15"] == 1
    assert summary["cdf_le_30"] == 2
    assert summary["cdf_le_45"] == 3
