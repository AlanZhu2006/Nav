import unittest

import numpy as np

from MemNavData.observed_frontier import (
    CoverageResidualTrigger,
    ObservedFrontierGrid,
    depth_endpoints_world,
)


class ObservedFrontierTest(unittest.TestCase):
    def test_depth_projection_matches_habitat_heading(self):
        depth = np.ones((1, 1), dtype=np.float64)
        intrinsic = np.asarray([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
        endpoint, _ = depth_endpoints_world(
            depth, [0.0, 0.0, 0.0], 0.0, intrinsic,
            pixel_stride=1, obstacle_height_m=(0.0, 2.0))
        np.testing.assert_allclose(endpoint[0], [0.0, -1.0], atol=1e-12)
        endpoint, _ = depth_endpoints_world(
            depth, [0.0, 0.0, 0.0], np.pi / 2.0, intrinsic,
            pixel_stride=1, obstacle_height_m=(0.0, 2.0))
        np.testing.assert_allclose(endpoint[0], [-1.0, 0.0], atol=1e-12)

    def test_ray_marks_free_space_and_surface_obstacle(self):
        grid = ObservedFrontierGrid(
            resolution_m=0.5,
            obstacle_clearance_m=0.0,
            min_component_cells=1,
            min_novelty_m=0.0,
        )
        grid.integrate_rays(
            [0.1, 0.1], np.asarray([[2.1, 0.1]]), [True])
        self.assertIn((0, 0), grid.visited)
        self.assertIn((2, 0), grid.free)
        self.assertIn((4, 0), grid.obstacle)
        self.assertNotIn((4, 0), grid.free)
        self.assertAlmostEqual(
            grid.distance_to_visited([0.25, 0.25]), 0.0)

    def test_frontier_ranking_is_goal_blind_and_deterministic(self):
        grid = ObservedFrontierGrid(
            resolution_m=1.0,
            obstacle_clearance_m=0.0,
            min_component_cells=1,
            min_novelty_m=0.0,
        )
        # Observe a three-ray fan without any occupied endpoint. Its outer free
        # boundary is a frontier even though no navigation goal was supplied.
        grid.integrate_rays(
            [0.1, 0.1],
            np.asarray([[3.1, -1.1], [3.1, 0.1], [3.1, 1.1]]),
            [False, False, False],
        )
        first = grid.ranked_frontiers([0.1, 0.1])
        second = grid.ranked_frontiers([0.1, 0.1])
        self.assertTrue(first)
        self.assertEqual(first, second)
        excluded = grid.ranked_frontiers(
            [0.1, 0.1], excluded_world_xz=[first[0].world_xz])
        self.assertTrue(
            not excluded or excluded[0].world_xz != first[0].world_xz)

    def test_invalid_configuration_fails_closed(self):
        with self.assertRaises(ValueError):
            ObservedFrontierGrid(resolution_m=0.0)
        with self.assertRaises(ValueError):
            depth_endpoints_world(
                np.ones((2, 2)), [0.0, 0.0], 0.0, np.eye(3))

    def test_residual_trigger_requires_consecutive_repetition(self):
        trigger = CoverageResidualTrigger(
            threshold_m=0.60, confirm_plans=3)
        self.assertFalse(trigger.observe(0.59))
        self.assertFalse(trigger.observe(0.90))
        self.assertEqual(trigger.streak, 0)
        self.assertFalse(trigger.observe(0.30))
        self.assertFalse(trigger.observe(0.20))
        self.assertTrue(trigger.observe(0.10))
        self.assertEqual(trigger.streak, 3)
        trigger.reset()
        self.assertEqual(trigger.streak, 0)

    def test_residual_trigger_fails_closed_on_invalid_signal(self):
        with self.assertRaises(ValueError):
            CoverageResidualTrigger(threshold_m=0.0)
        with self.assertRaises(ValueError):
            CoverageResidualTrigger(confirm_plans=0)
        trigger = CoverageResidualTrigger()
        with self.assertRaises(ValueError):
            trigger.observe(float("nan"))


if __name__ == "__main__":
    unittest.main()
