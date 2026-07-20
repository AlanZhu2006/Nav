import unittest

import torch

from scripts.eval.diag_memnav_candidate_oracle import (
    _repeat_sampling_condition,
    attach_candidate_metrics,
    summarize_candidate_group,
)


class MemNavCandidateOracleTest(unittest.TestCase):
    def test_repeat_condition_preserves_sample_major_candidate_order(self):
        condition = {
            'current_state': torch.tensor([[1.0], [2.0]]),
            'revisit': torch.tensor([[3.0], [4.0]]),
            'novel': torch.tensor([[5.0], [6.0]]),
            'effective_revisit_gate': torch.tensor([0.25, 0.75]),
            'unused_diagnostic': torch.tensor([9.0, 9.0]),
        }
        repeated = _repeat_sampling_condition(condition, 3)
        self.assertEqual(
            repeated['current_state'].squeeze(-1).tolist(),
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        )
        self.assertEqual(
            repeated['effective_revisit_gate'].tolist(),
            [0.25, 0.25, 0.25, 0.75, 0.75, 0.75],
        )
        self.assertNotIn('unused_diagnostic', repeated)

    def test_attach_and_summarize_candidate_oracle_metrics(self):
        # Two rows, four candidates, two action steps.  Every coordinate within
        # a candidate is constant, so its MSE is simply constant**2.
        actions = torch.tensor([
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            ],
            [
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
        ])
        target = torch.zeros(2, 2, 3)
        records = [{}, {}]
        attach_candidate_metrics(records, actions, target)

        self.assertEqual(records[0]['candidate_best_index'], 1)
        self.assertEqual(records[1]['candidate_best_index'], 3)
        self.assertAlmostEqual(records[0]['candidate_first_mse'], 1.0)
        self.assertAlmostEqual(records[0]['candidate_best_of_1_mse'], 1.0)
        self.assertAlmostEqual(records[0]['candidate_best_of_2_mse'], 0.0)
        self.assertAlmostEqual(records[1]['candidate_best_of_4_mse'], 0.0)
        self.assertAlmostEqual(records[0]['candidate_mean_mse_x'], 3.5)
        self.assertAlmostEqual(records[0]['candidate_best_mse_theta'], 0.0)

        summary = summarize_candidate_group(records)
        self.assertEqual(summary['count'], 2)
        self.assertAlmostEqual(summary['candidate_best_mse'], 0.0)
        self.assertAlmostEqual(summary['candidate_best_of_4_mse'], 0.0)
        self.assertAlmostEqual(summary['oracle_reduction_vs_group_mean'], 1.0)

    def test_attach_rejects_horizon_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'horizon'):
            attach_candidate_metrics(
                [{}], torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 3)
            )


if __name__ == '__main__':
    unittest.main()
