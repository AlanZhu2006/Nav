import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval.eval_memnav_offline import (
    _attach_route_sketch_records,
    _dataset_cache_contract,
    _summarize_route_sketch,
    _validate_route_checkpoint_metadata,
)
from internnav.model.basemodel.memnav.route_sketch import ResidualRouteSketch


class MemNavOfflineEvalContractTest(unittest.TestCase):
    def test_versioned_cache_contract_is_forwarded_without_weakening(self):
        config = SimpleNamespace(il=SimpleNamespace(
            strict_feature_coverage=True,
            require_versioned_cache=True,
            expected_cache_signature='audited-signature',
            require_generated_pose_convention=True,
        ))

        self.assertEqual(_dataset_cache_contract(config), {
            'strict_feature_coverage': True,
            'require_versioned_cache': True,
            'expected_cache_signature': 'audited-signature',
            'require_generated_pose_convention': True,
        })

    def test_legacy_local_contract_remains_explicit(self):
        config = SimpleNamespace(il=SimpleNamespace())

        self.assertEqual(_dataset_cache_contract(config), {
            'strict_feature_coverage': True,
            'require_versioned_cache': False,
            'expected_cache_signature': '',
            'require_generated_pose_convention': False,
        })

    def test_route_diagnostics_share_training_target_convention(self):
        records = [
            {'decision_curriculum_hard': True, 'goal_label': 'C'},
            {'decision_curriculum_hard': False, 'goal_label': 'A'},
        ]
        actions = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ])
        inv_sqrt_two = 2.0 ** -0.5
        outputs = {
            'route_direction': torch.tensor([
                [[1.0, 0.0], [inv_sqrt_two, inv_sqrt_two]],
                [[0.0, 1.0], [inv_sqrt_two, inv_sqrt_two]],
            ]),
            'route_raw_direction_norm': torch.ones(2, 2),
            'route_curvature_gate': torch.tensor([0.5, 0.25]),
            'route_residual_scale': torch.tensor([0.01, -0.02]),
        }
        enabled = _attach_route_sketch_records(
            records, outputs, {'batch_labels': actions}, (1, 2)
        )
        self.assertTrue(enabled)
        self.assertAlmostEqual(records[0]['route_h1_error_deg'], 0.0)
        self.assertAlmostEqual(records[1]['route_h2_residual_scale'], -0.02)
        summary = _summarize_route_sketch(records, (1, 2), enabled)
        self.assertTrue(summary['enabled'])
        self.assertEqual(summary['groups']['hard_turn']['h1_valid_count'], 1)
        self.assertAlmostEqual(summary['residual_scale']['h1'], 0.01)

    def test_route_checkpoint_metadata_is_required_and_versioned(self):
        model = SimpleNamespace(core=SimpleNamespace(
            route_sketch=ResidualRouteSketch(8, horizons=(1, 2))
        ))
        state = {'core.route_sketch.residual_scale': torch.zeros(2)}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'memnav.ckpt'
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, 'missing memnav_metadata'):
                _validate_route_checkpoint_metadata(model, checkpoint, state)

            metadata = {
                'route_sketch_code': 'wrong-version',
                'training_objective': {'route_horizons': [1, 2]},
            }
            (checkpoint.parent / 'memnav_metadata.json').write_text(
                json.dumps(metadata), encoding='utf-8'
            )
            with self.assertRaisesRegex(ValueError, 'version mismatch'):
                _validate_route_checkpoint_metadata(model, checkpoint, state)

            metadata['route_sketch_code'] = model.core.route_sketch.CODE_VERSION
            (checkpoint.parent / 'memnav_metadata.json').write_text(
                json.dumps(metadata), encoding='utf-8'
            )
            validated = _validate_route_checkpoint_metadata(
                model, checkpoint, state
            )
            self.assertEqual(validated, metadata)

            disabled_model = SimpleNamespace(core=SimpleNamespace(
                route_sketch=None
            ))
            with self.assertRaisesRegex(ValueError, 'route sketch disabled'):
                _validate_route_checkpoint_metadata(
                    disabled_model, checkpoint, state
                )


if __name__ == '__main__':
    unittest.main()
