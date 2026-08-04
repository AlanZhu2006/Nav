import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from NavDP.baselines.memnav.policy_agent import MemNavAgent


class RetrievalVerificationCacheTest(unittest.TestCase):
    def make_agent(self, root):
        agent = object.__new__(MemNavAgent)
        agent.n = 1
        agent.rgb_dir = str(root)
        agent.camera_intrinsic = np.array([
            [355.81464, 0.0, 240.0],
            [0.0, 351.687, 135.0],
            [0.0, 0.0, 1.0],
        ])
        agent._retrieval_verification_cache = {}
        return agent

    def test_identical_goal_anchor_uses_cached_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.circle(image, (240, 135), 60, (255, 255, 255), 4)
            cv2.line(image, (80, 40), (400, 230), (180, 180, 180), 3)
            self.assertTrue(cv2.imwrite(str(root / "0.jpg"), image))
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            goal = encoded.tobytes()
            agent = self.make_agent(root)

            first = agent.verify_retrieval_overlap(goal, 0)
            second = agent.verify_retrieval_overlap(goal, 0)

            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            for key in ("matches", "inliers", "inlier_ratio", "error"):
                self.assertEqual(first[key], second[key])
            self.assertEqual(len(agent._retrieval_verification_cache), 1)
            self.assertIn("uncached_verification_ms", second)
            self.assertLessEqual(
                second["verification_ms"], first["verification_ms"])

    def test_invalid_anchor_fails_without_polluting_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(Path(directory))
            result = agent.verify_retrieval_overlap(b"not-an-image", 2)
            self.assertFalse(result["cached"])
            self.assertIn("outside", result["error"])
            self.assertEqual(agent._retrieval_verification_cache, {})


if __name__ == "__main__":
    unittest.main()
