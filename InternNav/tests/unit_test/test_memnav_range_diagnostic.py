import unittest

import numpy as np

from scripts.eval.diag_memnav_range_scale import (
    _average_ranks,
    _finite_correlation,
    _range_summary,
    _robust_step_m,
)


class MemNavRangeDiagnosticTest(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        np.testing.assert_allclose(
            _average_ranks([3.0, 1.0, 1.0, 2.0]),
            [3.0, 0.5, 0.5, 2.0],
        )

    def test_correlations_use_only_finite_rows(self):
        pearson = _finite_correlation([1.0, 2.0, 3.0, np.nan], [2.0, 4.0, 6.0, 8.0])
        spearman = _finite_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], rank=True)
        self.assertAlmostEqual(pearson["statistic"], 1.0)
        self.assertAlmostEqual(spearman["statistic"], -1.0)

    def test_past_step_scale_ignores_stationary_frames(self):
        positions = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [6.0, 0.0],
            ]
        )
        prefix, recent = _robust_step_m(positions, cur_step=4, recent_window=2)
        self.assertEqual(prefix, 2.0)
        self.assertEqual(recent, 2.5)

    def test_range_summary_distinguishes_raw_and_adapted_codes(self):
        rows = [
            {
                "episode": "episode_a",
                "current_metric_m": 1.0,
                "goal_distance": 2.0,
                "current_range_steps": 10.0,
                "oracle_stream_steps": 20.0,
                "range_code_current": 0.2,
                "adapted_range_code_current": 0.4,
                "range_code_oracle_stream_range": 0.5,
            },
            {
                "episode": "episode_b",
                "current_metric_m": 2.0,
                "goal_distance": 4.0,
                "current_range_steps": 20.0,
                "oracle_stream_steps": 40.0,
                "range_code_current": 0.4,
                "adapted_range_code_current": 0.8,
                "range_code_oracle_stream_range": 1.0,
            },
        ]
        summary = _range_summary(rows)["range_code_error"]
        self.assertAlmostEqual(summary["raw_before_adapter"]["mae"], 0.45)
        self.assertAlmostEqual(
            summary["adapted_consumed_by_policy"]["mae"], 0.15
        )


if __name__ == "__main__":
    unittest.main()
