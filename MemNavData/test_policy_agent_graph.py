import unittest

import torch

from NavDP.baselines.memnav.policy_agent import MemNavAgent


class _RelativePoseCore:
    @staticmethod
    def build_revisit(current, target, metric_scale):
        delta = (target[:, 0] - current[:, 0]) * metric_scale
        aux = torch.stack([delta, torch.zeros_like(delta)], dim=-1)
        return torch.zeros((1, 1, 1)), aux, torch.eye(3)[None]


class GraphConditionedPoseTest(unittest.TestCase):
    def make_agent(self):
        agent = object.__new__(MemNavAgent)
        agent.graph_subgoal_spacing_m = 2.0
        agent.graph_subgoal_arrival_m = 0.60
        agent._graph_routes = {}
        agent.core = _RelativePoseCore()
        agent.device = torch.device("cpu")
        return agent

    def query(self, agent, current_x):
        poses = torch.zeros((7, 9), dtype=torch.float32)
        poses[:, 0] = torch.arange(7, dtype=torch.float32)
        current = torch.zeros((1, 9), dtype=torch.float32)
        current[:, 0] = current_x
        return agent._graph_conditioned_pose(
            goal_key="goal",
            cache={"cam_pose_enc": poses},
            current_pose=current,
            goal_aux_pose=torch.tensor([[0.2, 0.0]]),
            anchor=0,
            goal_start_frame=7,
            metric_scale=torch.tensor([1.0]),
        )

    def test_follows_nodes_then_returns_to_image_goal(self):
        agent = self.make_agent()
        target, diag = self.query(agent, 6.0)
        self.assertEqual(target.tolist(), [[-2.0, 0.0]])
        self.assertEqual(diag["graph_subgoal_node"], 4)

        target, diag = self.query(agent, 4.0)
        self.assertEqual(target.tolist(), [[-2.0, 0.0]])
        self.assertEqual(diag["graph_subgoal_node"], 2)

        target, diag = self.query(agent, 2.0)
        self.assertEqual(target.tolist(), [[-2.0, 0.0]])
        self.assertEqual(diag["graph_subgoal_node"], 0)

        target, diag = self.query(agent, 0.0)
        self.assertAlmostEqual(target[0, 0].item(), 0.2, places=6)
        self.assertTrue(diag["graph_subgoal_complete"])

    def test_zero_spacing_is_exact_direct_goal(self):
        agent = self.make_agent()
        agent.graph_subgoal_spacing_m = 0.0
        target, diag = self.query(agent, 6.0)
        self.assertAlmostEqual(target[0, 0].item(), 0.2, places=6)
        self.assertFalse(diag["graph_subgoal_enabled"])


if __name__ == "__main__":
    unittest.main()
