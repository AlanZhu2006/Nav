import unittest

import torch

from internnav.model.basemodel.memnav.revisit_pose import (
    GaugeInvariantRevisitPose,
)


class GaugeInvariantRevisitPoseTest(unittest.TestCase):
    @staticmethod
    def _context(step_scale=2.0):
        return {
            'step_scale': torch.tensor([step_scale]),
            'step_scale_drift': torch.tensor([0.3]),
            'anchor_gap': torch.tensor([512.0]),
            'goal_anchor_steps': torch.tensor([20.0]),
            'semantic_score_z': torch.tensor([1.5]),
        }

    def test_corrected_bearing_and_uniform_scale_invariance(self):
        encoder = GaugeInvariantRevisitPose(reliability_init=0.95)
        rotation = torch.eye(3).unsqueeze(0)
        first = encoder(
            torch.tensor([[2.0, 0.0, 4.0]]), rotation, self._context(2.0)
        )
        scaled = encoder(
            torch.tensor([[20.0, 0.0, 40.0]]), rotation, self._context(20.0)
        )

        expected = torch.tensor([[4.0, -2.0]])
        expected = torch.nn.functional.normalize(expected, dim=-1)
        torch.testing.assert_close(first['raw_direction'], expected)
        torch.testing.assert_close(first['pose_code'], scaled['pose_code'])
        torch.testing.assert_close(
            first['reliability_features'], scaled['reliability_features']
        )
        self.assertAlmostEqual(
            float(first['reliability'].detach()), 0.95, places=5
        )

    def test_zero_translation_is_finite_and_has_no_fake_bearing(self):
        encoder = GaugeInvariantRevisitPose()
        output = encoder(
            torch.zeros(2, 3),
            torch.eye(3).repeat(2, 1, 1),
            {
                'step_scale': torch.ones(2),
                'step_scale_drift': torch.zeros(2),
                'anchor_gap': torch.zeros(2),
                'goal_anchor_steps': torch.zeros(2),
                'semantic_score_z': torch.zeros(2),
            },
        )
        self.assertTrue(torch.isfinite(output['pose_code']).all())
        torch.testing.assert_close(output['raw_direction'], torch.zeros(2, 2))
        torch.testing.assert_close(output['range_steps'], torch.zeros(2))

    def test_reliability_cues_have_expected_shape_and_receive_gradient(self):
        encoder = GaugeInvariantRevisitPose(reliability_hidden=8)
        translation = torch.tensor([[1.0, 0.5, 3.0]], requires_grad=True)
        rotation = torch.eye(3).unsqueeze(0)
        output = encoder(translation, rotation, self._context())
        self.assertEqual(output['pose_code'].shape, (1, 4))
        self.assertEqual(
            output['reliability_features'].shape,
            (1, len(encoder.RELIABILITY_FEATURES)),
        )
        output['pose_code'].sum().backward()
        self.assertTrue(torch.isfinite(translation.grad).all())
        self.assertIsNotNone(encoder.reliability_head[-1].bias.grad)

    def test_context_shape_mismatch_fails_closed(self):
        encoder = GaugeInvariantRevisitPose()
        context = self._context()
        context['anchor_gap'] = torch.tensor([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, 'anchor_gap'):
            encoder(torch.ones(1, 3), torch.eye(3).unsqueeze(0), context)


if __name__ == '__main__':
    unittest.main()
