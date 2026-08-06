import sys
import unittest
from pathlib import Path

import numpy as np


NAVDP_ROOT = (
    Path(__file__).resolve().parents[1] / "NavDP/baselines/navdp"
)
sys.path.insert(0, str(NAVDP_ROOT))
from policy_agent import NavDP_Agent  # noqa: E402


class NavDPMemoryReplayTest(unittest.TestCase):
    @staticmethod
    def agent(memory_size=2, batch_size=1):
        agent = object.__new__(NavDP_Agent)
        agent.memory_size = memory_size
        agent.memory_queue = [[] for _ in range(batch_size)]
        agent.process_image = lambda images: np.asarray(images, dtype=float)
        return agent

    def test_replay_has_same_bounded_fifo_semantics_as_policy_steps(self):
        agent = self.agent(memory_size=2)
        self.assertEqual(agent.append_observation(np.asarray([[1.0]])), [1])
        self.assertEqual(agent.append_observation(np.asarray([[2.0]])), [2])
        self.assertEqual(agent.append_observation(np.asarray([[3.0]])), [2])
        self.assertEqual(
            [value.tolist() for value in agent.memory_queue[0]],
            [[2.0], [3.0]],
        )

    def test_replay_rejects_batch_mismatch(self):
        agent = self.agent(batch_size=2)
        with self.assertRaisesRegex(ValueError, "batch differs"):
            agent.append_observation(np.asarray([[1.0]]))

    def test_imagegoal_resample_does_not_mutate_fifo(self):
        agent = self.agent(memory_size=2)
        original = np.ones((2, 2, 3), dtype=float)
        agent.memory_queue = [[original.copy()]]
        agent.stop_threshold = -1e9
        agent.process_depth = lambda depths: np.asarray(depths, dtype=float)
        agent.project_trajectory = lambda images, trajectories, values: None

        class Former:
            @staticmethod
            def predict_imagegoal_action(goals, images, depths):
                trajectories = np.zeros((1, 2, 3, 3), dtype=float)
                values = np.asarray([[1.0, 0.0]], dtype=float)
                return trajectories, values, trajectories.copy(), trajectories.copy()

        agent.navi_former = Former()
        before = [value.copy() for value in agent.memory_queue[0]]
        agent.resample_imagegoal(
            np.zeros((1, 2, 2, 3), dtype=float),
            np.zeros((1, 2, 2, 3), dtype=float),
            np.zeros((1, 2, 2, 1), dtype=float),
        )
        self.assertEqual(len(agent.memory_queue[0]), len(before))
        for actual, expected in zip(agent.memory_queue[0], before):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
