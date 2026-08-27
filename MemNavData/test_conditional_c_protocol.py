import unittest

import numpy as np

from MemNavData.conditional_c_protocol import (
    infer_mode,
    prefix_last_frame,
    world_goal_to_local,
)


class ConditionalCProtocolTest(unittest.TestCase):
    def test_prefix_stops_immediately_before_c(self):
        self.assertEqual(prefix_last_frame([184, 417], 575), 416)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            prefix_last_frame([184], 575)
        with self.assertRaisesRegex(ValueError, "outside"):
            prefix_last_frame([184, 575], 575)

    def test_mode_inference_is_fail_closed(self):
        self.assertEqual(infer_mode("auto", "navdp", 8), "native")
        self.assertEqual(
            infer_mode("auto", "hybrid_pose", 1), "geometry_top1")
        self.assertEqual(
            infer_mode("auto", "hybrid_pose", 8), "geometry_topk")
        self.assertEqual(
            infer_mode("oracle_anchor", "hybrid_pose", 8),
            "oracle_anchor")
        with self.assertRaisesRegex(ValueError, "requires"):
            infer_mode("auto", "memnav", 8)

    def test_world_goal_inverse_for_forward_and_left(self):
        # yaw=0 faces Habitat -Z. One metre toward -Z is forward, while one
        # metre toward -X is left under the audited NavDP convention.
        np.testing.assert_allclose(
            world_goal_to_local([0.0, -1.0], [0.0, 0.0], 0.0),
            [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            world_goal_to_local([-1.0, 0.0], [0.0, 0.0], 0.0),
            [0.0, 1.0], atol=1e-12)

    def test_world_local_round_trip(self):
        current = np.asarray([2.5, -1.25])
        goal = np.asarray([-0.4, 3.1])
        yaw = 0.73
        forward, left = world_goal_to_local(goal, current, yaw)
        reconstructed = current + np.asarray([
            -forward * np.sin(yaw) - left * np.cos(yaw),
            -forward * np.cos(yaw) + left * np.sin(yaw),
        ])
        np.testing.assert_allclose(reconstructed, goal, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
