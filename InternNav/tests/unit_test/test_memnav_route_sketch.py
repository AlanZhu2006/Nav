import unittest

import torch

from internnav.model.basemodel.memnav.route_sketch import (
    ResidualRouteSketch,
    build_residual_route_sketch,
    route_curvature_gate,
    route_direction_targets,
)


class MemNavRouteSketchTest(unittest.TestCase):
    @staticmethod
    def _inputs(batch=2, dim=16):
        torch.manual_seed(3)
        return (
            torch.randn(batch, 8, dim),
            torch.randn(batch, 4, dim),
            torch.randn(batch, 4, dim),
            torch.tensor([0.2, 0.8])[:batch],
        )

    def test_zero_initialized_residual_is_exact_identity(self):
        module = ResidualRouteSketch(16, horizons=(2, 8, 24))
        current, revisit, novel, gate = self._inputs()
        output = module(current, revisit, novel, gate)
        self.assertTrue(torch.equal(output['current_state'], current))
        torch.testing.assert_close(
            output['residual_scale'], torch.zeros(3)
        )
        self.assertEqual(output['direction'].shape, (2, 3, 2))
        torch.testing.assert_close(
            torch.linalg.vector_norm(output['direction'], dim=-1),
            torch.ones(2, 3),
        )

    def test_optional_adapter_construction_preserves_global_rng(self):
        torch.manual_seed(83)
        expected = torch.rand(16)
        torch.manual_seed(83)
        module = build_residual_route_sketch(16, (2, 8, 24))
        actual = torch.rand(16)
        self.assertIsInstance(module, ResidualRouteSketch)
        self.assertTrue(torch.equal(actual, expected))

    def test_nonzero_scale_changes_only_reserved_current_slots(self):
        module = ResidualRouteSketch(16, horizons=(2, 8, 24))
        current, revisit, novel, gate = self._inputs()
        with torch.no_grad():
            module.residual_scale.fill_(0.2)
        output = module(current, revisit, novel, gate)['current_state']
        self.assertFalse(torch.equal(output[:, :3], current[:, :3]))
        self.assertTrue(torch.equal(output[:, 3:], current[:, 3:]))

    def test_invalid_horizon_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            ResidualRouteSketch(8, horizons=(8, 2))
        with self.assertRaisesRegex(ValueError, 'unique'):
            ResidualRouteSketch(8, horizons=(2, 2))
        with self.assertRaisesRegex(ValueError, 'at least two'):
            ResidualRouteSketch(8, horizons=(2,))

    def test_curvature_gate_suppresses_straight_and_opens_for_uturn(self):
        direction = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ])
        torch.testing.assert_close(
            route_curvature_gate(direction), torch.tensor([0.0, 1.0, 0.5])
        )

    def test_route_targets_sum_deltas_without_using_theta(self):
        actions = torch.tensor([[[1.0, 0.0, 99.0], [0.0, 2.0, -99.0]]])
        target, valid = route_direction_targets(actions, (1, 2))
        torch.testing.assert_close(target[0, 0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(
            target[0, 1], torch.tensor([1.0, 2.0]) / (5.0 ** 0.5)
        )
        self.assertTrue(bool(valid.all()))

    def test_zero_motion_target_is_masked(self):
        target, valid = route_direction_targets(
            torch.zeros(1, 3, 3), (1, 3)
        )
        self.assertFalse(bool(valid.any()))
        torch.testing.assert_close(target, torch.zeros_like(target))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            route_direction_targets(torch.zeros(1, 3, 3), (4,))

    def test_route_supervision_and_action_residual_have_gradients(self):
        module = ResidualRouteSketch(16, horizons=(2, 8, 24))
        current, revisit, novel, gate = self._inputs()
        output = module(current, revisit, novel, gate)
        target = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(2, 3, 2)
        direction_loss = 1.0 - (output['direction'] * target).sum(-1).mean()
        action_proxy = output['current_state'][:, :3].sum()
        (direction_loss + action_proxy).backward()
        self.assertTrue(torch.isfinite(module.residual_scale.grad).all())
        self.assertGreater(float(module.residual_scale.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(module.direction_head.weight.grad).all())
        self.assertGreater(
            float(module.direction_head.weight.grad.abs().sum()), 0.0
        )

    def test_route_auxiliary_gradient_does_not_modify_legacy_inputs(self):
        module = ResidualRouteSketch(16, horizons=(2, 8, 24))
        current, revisit, novel, gate = self._inputs()
        current.requires_grad_()
        revisit.requires_grad_()
        novel.requires_grad_()
        gate.requires_grad_()
        output = module(current, revisit, novel, gate)
        output['direction'].square().sum().backward()
        for value in (current, revisit, novel, gate):
            self.assertIsNone(value.grad)
        self.assertIsNotNone(module.direction_head.weight.grad)


if __name__ == '__main__':
    unittest.main()
