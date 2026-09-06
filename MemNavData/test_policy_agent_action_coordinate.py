import unittest

import numpy as np
import torch

from NavDP.baselines.memnav.policy_agent import MemNavAgent


def pose9(x: float, z: float) -> torch.Tensor:
    value = np.zeros(9, dtype=np.float32)
    value[0] = x
    value[2] = z
    value[6] = 1.0
    return torch.from_numpy(value)


def receipt(index: int, translation: float, yaw: float = 0.0) -> dict:
    return {
        "frame_index": index,
        "executed_translation_m": translation,
        "executed_yaw_rad": yaw,
        "contract": "frame_bound_realized_executor_motion_v1",
    }


class PolicyAgentActionCoordinateTest(unittest.TestCase):
    @staticmethod
    def agent():
        agent = MemNavAgent.__new__(MemNavAgent)
        # Frames 0..2 are the outbound route. Frame 3 is the first query at
        # the same causal tail, so its zero edge is explicitly represented.
        agent.cam_pose = [
            pose9(0.0, 0.0),
            pose9(0.0, -1.0),
            pose9(0.0, -2.0),
            pose9(0.0, -2.0),
        ]
        agent.executor_motion_receipts = [
            receipt(0, 0.0),
            receipt(1, 1.0),
            receipt(2, 1.0),
            receipt(3, 0.0),
        ]
        agent.n = 4
        agent._certified_action_coordinate_routes = {}
        return agent

    def test_proof_once_route_uses_frame_bound_action_coordinate(self):
        agent = self.agent()
        direction, first = agent._certified_action_coordinate_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-7)
        self.assertEqual(first["action_coordinate_compass_status"], "active")
        self.assertEqual(first["action_coordinate_history_receipt_count"], 3)
        self.assertAlmostEqual(first["action_coordinate_route_extent_m"], 2.0)
        self.assertFalse(first["action_coordinate_metric_scale_consumed"])
        self.assertFalse(
            first["action_coordinate_visual_gate_present_after_initialization"])
        self.assertFalse(
            first["action_coordinate_endpoint_fallback_available"])
        self.assertFalse(first["action_coordinate_native_fallback_available"])

        agent.cam_pose.append(pose9(10.0, 10.0))  # never used for progress
        agent.executor_motion_receipts.append(receipt(4, 0.75, np.pi / 2.0))
        agent.n = 5
        _, second = agent._certified_action_coordinate_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertAlmostEqual(
            second["action_coordinate_progress_m"], 0.75)
        self.assertAlmostEqual(
            np.linalg.norm(second["action_coordinate_unit_bearing"]), 1.0)
        self.assertAlmostEqual(
            np.linalg.norm(second[
                "action_coordinate_controller_pointgoal"]), 2.5)
        self.assertAlmostEqual(
            second["cumulative_executor_yaw_rad"], np.pi / 2.0)
        self.assertEqual(second["action_coordinate_last_consumed_frame"], 4)

    def test_missing_history_receipt_is_explicit_failure(self):
        agent = self.agent()
        agent.executor_motion_receipts[2] = None
        direction, output = agent._certified_action_coordinate_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertIsNone(direction)
        self.assertEqual(
            output["action_coordinate_compass_status"], "geometry_failure")
        self.assertIn(
            "unbound executor receipt",
            output["action_coordinate_compass_error"],
        )

    def test_turn_only_update_does_not_advance_route(self):
        agent = self.agent()
        agent._certified_action_coordinate_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)
        agent.cam_pose.append(pose9(50.0, -50.0))
        agent.executor_motion_receipts.append(receipt(4, 0.0, -0.5))
        agent.n = 5

        _, output = agent._certified_action_coordinate_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3)

        self.assertEqual(output["action_coordinate_progress_m"], 0.0)
        self.assertEqual(output["route_state_index"], 1)
        self.assertAlmostEqual(output["cumulative_executor_yaw_rad"], -0.5)


if __name__ == "__main__":
    unittest.main()
