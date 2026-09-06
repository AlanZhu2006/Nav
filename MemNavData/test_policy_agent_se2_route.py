import math
import unittest

import numpy as np

from NavDP.baselines.memnav.policy_agent import MemNavAgent


def receipt(index: int, translation: float, yaw: float = 0.0) -> dict:
    return {
        "frame_index": index,
        "executed_translation_m": translation,
        "executed_yaw_rad": yaw,
        "contract": "frame_bound_realized_executor_motion_v1",
        "local_se2": {
            "contract": "frame_bound_local_se2_v1",
            "executed_forward_m": translation * math.cos(yaw),
            "executed_left_m": translation * math.sin(yaw),
            "executed_yaw_rad": yaw,
            "source": "unit_test_local_odometry_v1",
        },
    }


class PolicyAgentSE2RouteTest(unittest.TestCase):
    @staticmethod
    def agent():
        agent = MemNavAgent.__new__(MemNavAgent)
        # Frames 0..2 move forward for two metres.  Frame 3 opens the query at
        # the unchanged causal tail; target anchor 0 is therefore behind it.
        agent.executor_motion_receipts = [
            receipt(0, 0.0),
            receipt(1, 1.0),
            receipt(2, 1.0),
            receipt(3, 0.0),
        ]
        agent.n = 4
        agent._certified_se2_route_compasses = {}
        return agent

    def test_route_is_built_once_from_receipts_and_projected_in_se2(self):
        agent = self.agent()
        direction, first = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        np.testing.assert_allclose(direction, [-1.0, 0.0], atol=1e-12)
        self.assertEqual(first["se2_route_compass_status"], "active")
        self.assertEqual(first["se2_route_history_receipt_count"], 3)
        self.assertAlmostEqual(first["se2_route_extent_m"], 2.0)
        self.assertTrue(first[
            "se2_route_evaluator_pose_consumed_as_odometry_proxy"])
        self.assertFalse(first["se2_route_world_pose_consumed_by_policy"])
        self.assertTrue(first["se2_route_executor_odometry_required"])
        self.assertEqual(first["se2_route_executor_odometry_source"],
                         "unit_test_local_odometry_v1")
        self.assertFalse(first["se2_route_lingbot_translation_consumed"])
        self.assertFalse(first["se2_route_metric_scale_consumed"])
        self.assertFalse(first["se2_route_native_fallback_available"])

        agent.executor_motion_receipts.append(receipt(4, 0.75, math.pi))
        agent.n = 5
        direction, second = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertAlmostEqual(second["se2_route_projected_progress_m"], 0.75)
        self.assertAlmostEqual(second["se2_route_query_path_length_m"], 0.75)
        self.assertAlmostEqual(second["se2_route_cross_track_error_m"], 0.0)
        self.assertEqual(second["se2_route_query_receipt_count"], 1)
        self.assertEqual(second["se2_route_last_consumed_frame"], 4)
        np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-12)

    def test_pending_receipts_are_composed_in_order_not_summed(self):
        agent = self.agent()
        agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)
        agent.executor_motion_receipts.extend([
            receipt(4, 1.0, math.pi / 2.0),
            receipt(5, 1.0, math.pi),
        ])
        agent.n = 6

        _, output = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        np.testing.assert_allclose(
            output["se2_route_estimated_position"], [0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(output["se2_route_projected_progress_m"], 0.0)
        self.assertAlmostEqual(output["se2_route_query_path_length_m"], 2.0)
        self.assertEqual(output["se2_route_query_receipt_count"], 2)

    def test_local_displacement_not_legacy_distance_controls_projection(self):
        agent = self.agent()
        agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)
        sideways = receipt(4, 1.0, 0.0)
        sideways["local_se2"].update(
            executed_forward_m=0.0,
            executed_left_m=1.0,
            source="unit_test_local_odometry_v1",
        )
        agent.executor_motion_receipts.append(sideways)
        agent.n = 5

        _, output = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertAlmostEqual(output["se2_route_projected_progress_m"], 0.0)
        self.assertAlmostEqual(output["se2_route_cross_track_error_m"], 1.0)
        np.testing.assert_allclose(
            output["se2_route_estimated_position"], [0.0, 1.0], atol=1e-12)

    def test_missing_local_se2_history_receipt_fails_explicitly(self):
        agent = self.agent()
        agent.executor_motion_receipts[2].pop("local_se2")

        direction, output = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertIsNone(direction)
        self.assertIn("lacks a local SE(2) receipt", output[
            "se2_route_compass_error"])

    def test_missing_history_receipt_fails_explicitly(self):
        agent = self.agent()
        agent.executor_motion_receipts[2] = None

        direction, output = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertIsNone(direction)
        self.assertEqual(output["se2_route_compass_status"], "geometry_failure")
        self.assertIn("unbound executor receipt", output["se2_route_compass_error"])

    def test_query_cannot_switch_odometry_source(self):
        agent = self.agent()
        agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)
        changed = receipt(4, 0.5, math.pi)
        changed["local_se2"]["source"] = "different_odometry_v1"
        agent.executor_motion_receipts.append(changed)
        agent.n = 5

        direction, output = agent._certified_se2_route_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertIsNone(direction)
        self.assertIn("differs from history", output[
            "se2_route_compass_error"])


if __name__ == "__main__":
    unittest.main()
