import copy
import unittest

from scripts.eval.compare_memnav_offline import compare_reports, validate_and_pair


def _row(identity, mse, sensitivity, penalty, **overrides):
    row = {
        'sample_index': int(identity[-1]),
        'sample_identity': identity,
        'cache_path': f'/cache/{identity}',
        'goal_label': 'B',
        'remaining_path_span': 64,
        'decision_curriculum_hard': False,
        'is_revisit': True,
        'full_diffusion_action_mse': mse,
        'full_diffusion_action_mse_x': mse,
        'full_diffusion_action_mse_y': mse,
        'full_diffusion_action_mse_theta': mse,
        'full_diffusion_goal_sensitivity_mse': sensitivity,
        'full_diffusion_shuffled_goal_penalty': penalty,
        'shuffled_goal_source_batch_index': (int(identity[-1]) + 1) % 4,
        'shuffled_goal_source_identity': f'source-{identity}',
        'gate': 0.8,
        'gate_bce': 0.2,
        'match_correct': True,
    }
    row.update(overrides)
    return row


def _report(rows):
    return {
        'evaluation_type': 'fixed-offline-training-diagnostic',
        'closed_loop_navigation': False,
        'git_commit': 'abc123',
        'root_dir': '/data',
        'feature_root': '/features',
        'cache_contract': {'strict_feature_coverage': True},
        'data_split': 'val',
        'validation_fraction': 0.1,
        'split_seed': 0,
        'sampling_mode': 'fixed_leg',
        'sampling_seed': 0,
        'random_seed': 0,
        'dataset_fingerprint': 'dataset',
        'dataset_size': 100,
        'subset_mode': 'balanced-fixed',
        'selection_indices': [0, 1, 2, 3],
        'eval_dataset_fingerprint': 'fixed-four',
        'evaluated_samples': 4,
        'retrieval_anchor_mode': 'projected',
        'original_anchor_margins': [8],
        'anchor_margin_override': None,
        'oracle_positive': True,
        'full_diffusion_goal_shuffle': True,
        'diffusion_seed': 104729,
        'goal_shuffle_scope': 'within_batch_cyclic_derangement',
        'paired_diffusion_randomness': True,
        'per_sample': rows,
    }


class MemNavOfflineCompareTest(unittest.TestCase):
    def setUp(self):
        self.control_rows = [
            _row('mp3d_2leg/a0', 1.0, 0.1, 0.01),
            _row(
                'mp3d_2leg/a1', 2.0, 0.2, 0.02,
                is_revisit=False, match_correct=None, gate=0.2,
            ),
            _row(
                'mp3d_3leg/a2', 4.0, 0.3, 0.03,
                goal_label='C', remaining_path_span=300,
                decision_curriculum_hard=True,
            ),
            _row('mp3d_3leg/a3', 8.0, 0.4, 0.04),
        ]
        treatment_mse = [0.5, 3.0, 2.0, 8.0]
        self.treatment_rows = []
        for row, mse in zip(self.control_rows, treatment_mse):
            changed = copy.deepcopy(row)
            for key in (
                'full_diffusion_action_mse',
                'full_diffusion_action_mse_x',
                'full_diffusion_action_mse_y',
                'full_diffusion_action_mse_theta',
            ):
                changed[key] = mse
            changed['full_diffusion_goal_sensitivity_mse'] += 0.05
            changed['full_diffusion_shuffled_goal_penalty'] += 0.01
            self.treatment_rows.append(changed)

    def test_pairs_by_identity_and_reports_stratified_deltas(self):
        control = _report(self.control_rows)
        treatment = _report(list(reversed(self.treatment_rows)))
        result = compare_reports(
            control, treatment, bootstrap_resamples=128, bootstrap_seed=7
        )

        metric = result['groups']['all']['metrics']['full_diffusion_action_mse']
        self.assertEqual(metric['count'], 4)
        self.assertAlmostEqual(metric['control_mean'], 3.75)
        self.assertAlmostEqual(metric['treatment_mean'], 3.375)
        self.assertAlmostEqual(metric['mean_delta_treatment_minus_control'], -0.375)
        self.assertEqual(metric['num_improved'], 2)
        self.assertEqual(metric['num_worsened'], 1)
        self.assertEqual(metric['num_tied'], 1)
        self.assertEqual(len(metric['paired_bootstrap_95_ci']), 2)

        goal_c = result['groups']['three_leg_goal_C_revisit']
        self.assertEqual(goal_c['count'], 1)
        self.assertAlmostEqual(
            goal_c['metrics']['full_diffusion_action_mse'][
                'mean_delta_treatment_minus_control'
            ],
            -2.0,
        )
        self.assertEqual(result['groups']['remaining_span_ge_256']['count'], 1)
        sensitivity = result['groups']['all']['metrics'][
            'full_diffusion_goal_sensitivity_mse'
        ]
        self.assertFalse(sensitivity['lower_is_better'])
        self.assertEqual(sensitivity['num_improved'], 4)

    def test_rejects_evaluation_contract_mismatch(self):
        control = _report(self.control_rows)
        treatment = _report(self.treatment_rows)
        treatment['diffusion_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'diffusion_seed'):
            validate_and_pair(control, treatment)

    def test_rejects_retrieval_anchor_contract_mismatch(self):
        mismatches = {
            'retrieval_anchor_mode': 'raw',
            'original_anchor_margins': [16],
            'anchor_margin_override': 16,
        }
        for key, value in mismatches.items():
            with self.subTest(key=key):
                control = _report(self.control_rows)
                treatment = _report(self.treatment_rows)
                treatment[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_and_pair(control, treatment)

    def test_rejects_row_contract_mismatch(self):
        control = _report(self.control_rows)
        treatment = _report(copy.deepcopy(self.treatment_rows))
        treatment['per_sample'][2]['remaining_path_span'] = 301
        with self.assertRaisesRegex(ValueError, 'remaining_path_span'):
            validate_and_pair(control, treatment)

    def test_rejects_duplicate_sample_identity(self):
        control = _report(self.control_rows)
        treatment = _report(copy.deepcopy(self.treatment_rows))
        treatment['per_sample'][1]['sample_identity'] = treatment['per_sample'][0][
            'sample_identity'
        ]
        with self.assertRaisesRegex(ValueError, 'duplicate sample_identity'):
            validate_and_pair(control, treatment)


if __name__ == '__main__':
    unittest.main()
