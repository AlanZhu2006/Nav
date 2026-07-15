import unittest

import torch

from internnav.model.basemodel.memnav.metrics import (
    compute_memnav_batch_records,
    summarize_memnav_records,
)


def _batch():
    return {
        'batch_is_revisit': torch.tensor([1.0, 0.0]),
        'batch_null_pos': torch.tensor([False, True]),
        'batch_cand_mask': torch.tensor([[True, True], [True, True]]),
        'batch_pos_mask': torch.tensor([[True, False], [False, False]]),
        'batch_neg_mask': torch.tensor([[False, True], [True, True]]),
        'batch_goal_rel_pose': torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
        'batch_goal_rel_rotation': torch.eye(3).repeat(2, 1, 1),
        'cur_steps': [319, 1024],
        'goal_steps': [400, 1100],
        'cache_paths': ['revisit.npz', 'novel.npz'],
    }


def _outputs():
    return {
        'ret_logits': torch.tensor([[3.0, 0.0], [0.0, 2.0]]),
        'match_idx': torch.tensor([0, 1]),
        'anchor_idx': torch.tensor([0, 1]),
        'revisit_gate': torch.tensor([0.9, 0.2]),
        'gate_feature': torch.tensor([0.95, 0.60]),
        'noise': torch.zeros(2, 1, 3),
        'noise_pred': torch.tensor([[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]]),
        'aux_pose': torch.tensor([[1.0, 2.0], [9.0, 9.0]]),
        'R_rel': torch.eye(3).repeat(2, 1, 1),
    }


class MemNavMetricsTest(unittest.TestCase):
    def test_current_architecture_metrics_and_oracle_delta(self):
        outputs = _outputs()
        oracle = {key: value.clone() if torch.is_tensor(value) else value
                  for key, value in outputs.items()}
        oracle['noise_pred'][0] = 0.5
        records = compute_memnav_batch_records(outputs, _batch(), oracle)
        metrics = summarize_memnav_records(records)

        self.assertEqual(records[0]['match_outcome'], 'positive')
        self.assertEqual(records[1]['match_outcome'], 'negative')
        self.assertEqual(records[0]['memory_length'], 320)
        self.assertAlmostEqual(metrics['action_mse'], 2.5)
        self.assertAlmostEqual(metrics['action_noise_mse_theta'], 2.5)
        self.assertAlmostEqual(metrics['revisit_match_accuracy'], 1.0)
        self.assertAlmostEqual(metrics['gate_revisit'], 0.9, places=6)
        self.assertAlmostEqual(metrics['gate_novel'], 0.2, places=6)
        self.assertAlmostEqual(metrics['gate_separation'], 0.7, places=6)
        self.assertAlmostEqual(metrics['aux_mse_x_revisit'], 0.0)
        self.assertLess(metrics['aux_direction_error_deg_revisit'], 0.03)
        self.assertAlmostEqual(metrics['oracle_action_mse_revisit'], 0.25)
        self.assertAlmostEqual(metrics['oracle_action_delta_revisit'], -0.75)
        self.assertAlmostEqual(metrics['rotation_error_converted_deg_revisit'], 0.0)
        self.assertAlmostEqual(
            metrics['gate_threshold_sweep']['best']['balanced_accuracy'], 1.0
        )

    def test_rejects_dynamic_label_mismatch(self):
        batch = _batch()
        batch['batch_null_pos'][0] = True
        with self.assertRaisesRegex(ValueError, 'null_pos'):
            compute_memnav_batch_records(_outputs(), batch)

    def test_oracle_summary_handles_a_novel_only_subset(self):
        batch = _batch()
        batch['batch_is_revisit'] = torch.zeros(2)
        batch['batch_null_pos'] = torch.ones(2, dtype=torch.bool)
        batch['batch_pos_mask'] = torch.zeros(2, 2, dtype=torch.bool)
        batch['batch_neg_mask'] = torch.ones(2, 2, dtype=torch.bool)
        outputs = _outputs()
        outputs['match_idx'] = torch.tensor([0, 0])
        oracle = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in outputs.items()
        }
        records = compute_memnav_batch_records(outputs, batch, oracle)
        metrics = summarize_memnav_records(records)
        self.assertIsNone(metrics['oracle_action_delta_revisit'])
        self.assertIsNone(metrics['oracle_aux_mse_y_delta_revisit'])


if __name__ == '__main__':
    unittest.main()
