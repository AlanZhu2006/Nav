import unittest
from types import SimpleNamespace
import numpy as np
from probe_repaired_fullmono_construction import choose_direction, measured_projection


class ConstructionProbeTests(unittest.TestCase):
    def test_selection_is_independent_of_candidate_order(self):
        self.assertEqual(choose_direction("s", "e", ["front", "rear", "side"]),
                         choose_direction("s", "e", ["side", "front", "rear"]))

    def test_empty_and_singleton(self):
        self.assertIsNone(choose_direction("s", "e", []))
        self.assertEqual(choose_direction("s", "e", ["rear"]), "rear")

    def test_invalid_direction_rejected(self):
        for value in (["rear", "rear"], ["best_critic"]):
            with self.assertRaises(ValueError):
                choose_direction("s", "e", value)

    def test_projection_context_is_local_and_restored(self):
        old = lambda *args, **kwargs: None
        b = SimpleNamespace(history_tools=SimpleNamespace(goal_world_points=old), covis_frac=old, covis_curve=old)
        with measured_projection(b, np.array([[10., 0., 8.], [0., 10., 8.], [0., 0., 1.]])):
            points = b.history_tools.goal_world_points(np.full((16, 16), 10., np.float32), np.zeros(3), 0.)
            self.assertTrue(np.all(np.isfinite(points)))
            self.assertTrue(np.any(np.abs(points) > 6.5535))
            with self.assertRaises(ValueError):
                b.covis_curve(points, [np.eye(4)], [])
        self.assertIs(b.history_tools.goal_world_points, old)
        self.assertIs(b.covis_curve, old)


if __name__ == "__main__":
    unittest.main()
