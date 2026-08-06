import unittest

import numpy as np

from MemNavData.global_subgoal_protocol import polyline_subgoal


class GlobalSubgoalProtocolTest(unittest.TestCase):
    def test_interpolates_across_multiple_segments(self):
        path = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]]
        np.testing.assert_allclose(
            polyline_subgoal(path, 1.5), [1.0, 0.0, 0.5], atol=1e-12)

    def test_clamps_to_endpoint_and_skips_duplicates(self):
        path = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        np.testing.assert_allclose(
            polyline_subgoal(path, 9.0), [0.0, 0.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(
            polyline_subgoal([[2.0, 3.0, 4.0]], 1.0),
            [2.0, 3.0, 4.0],
            atol=1e-12,
        )

    def test_rejects_invalid_inputs(self):
        for path in ([], [[0.0, 1.0]], [[0.0, np.nan, 1.0]]):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    polyline_subgoal(path, 1.0)
        for distance in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(distance=distance):
                with self.assertRaises(ValueError):
                    polyline_subgoal([[0.0, 0.0, 0.0]], distance)


if __name__ == "__main__":
    unittest.main()
