import unittest

import numpy as np

from MemNavData.double_revisit_diagnostics import online_path_nearest_anchor


class OnlinePathNearestAnchorTest(unittest.TestCase):
    def test_selects_nearest_frame_under_strict_ceiling(self):
        trace = [
            {"frame_idx": 3, "x": -1.0, "z": 0.0},
            {"frame_idx": 4, "x": 2.0, "z": 0.0},
            {"frame_idx": 5, "x": 1.1, "z": 0.0},
        ]
        result = online_path_nearest_anchor(
            trace, np.asarray([1.0, 0.0]), candidate_ceiling=4)
        self.assertEqual(result["frame_idx"], 4)
        self.assertAlmostEqual(result["distance_m"], 1.0)

    def test_tie_breaks_by_earlier_frame(self):
        trace = [
            {"frame_idx": 9, "x": 1.0, "z": 0.0},
            {"frame_idx": 7, "x": -1.0, "z": 0.0},
        ]
        result = online_path_nearest_anchor(
            trace, np.asarray([0.0, 0.0]), candidate_ceiling=9)
        self.assertEqual(result["frame_idx"], 7)

    def test_fails_closed_without_causal_candidate(self):
        with self.assertRaisesRegex(ValueError, "no causal"):
            online_path_nearest_anchor(
                [{"frame_idx": 5, "x": 0.0, "z": 0.0}],
                np.asarray([0.0, 0.0]),
                candidate_ceiling=4,
            )

    def test_rejects_duplicate_frame_indices(self):
        trace = [
            {"frame_idx": 2, "x": 0.0, "z": 0.0},
            {"frame_idx": 2, "x": 1.0, "z": 0.0},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            online_path_nearest_anchor(
                trace, np.asarray([0.0, 0.0]), candidate_ceiling=2)


if __name__ == "__main__":
    unittest.main()
