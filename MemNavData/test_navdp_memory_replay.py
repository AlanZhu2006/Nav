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


if __name__ == "__main__":
    unittest.main()
