import unittest

import torch

from scripts.eval.diag_memnav_route_pose_oracle import (
    circular_blend_direction,
    route_direction_from_actions,
)


class MemNavRoutePoseOracleTest(unittest.TestCase):
    def test_route_direction_sums_local_action_displacements(self):
        actions = torch.tensor([[[1.0, 0.0, 9.0], [0.0, 2.0, -4.0]]])
        fallback = torch.tensor([[0.0, -1.0]])
        direction = route_direction_from_actions(actions, 2, fallback)
        torch.testing.assert_close(
            direction, torch.tensor([[1.0, 2.0]]) / (5.0 ** 0.5)
        )

    def test_zero_displacement_uses_endpoint_fallback(self):
        actions = torch.zeros(1, 3, 3)
        fallback = torch.tensor([[0.0, -1.0]])
        direction = route_direction_from_actions(actions, 3, fallback)
        torch.testing.assert_close(direction, fallback)

    def test_circular_blend_uses_short_arc(self):
        endpoint_angle = torch.deg2rad(torch.tensor([170.0]))
        route_angle = torch.deg2rad(torch.tensor([-170.0]))
        endpoint = torch.stack(
            (torch.cos(endpoint_angle), torch.sin(endpoint_angle)), dim=-1
        )
        route = torch.stack(
            (torch.cos(route_angle), torch.sin(route_angle)), dim=-1
        )
        midpoint = circular_blend_direction(endpoint, route, 0.5)
        self.assertLess(float(midpoint[0, 0]), -0.999)
        self.assertLess(abs(float(midpoint[0, 1])), 1e-5)


if __name__ == '__main__':
    unittest.main()
