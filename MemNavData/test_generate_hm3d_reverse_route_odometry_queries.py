import unittest

from MemNavData.generate_hm3d_reverse_route_odometry_queries import (
    sample_reverse_indices,
    wrap,
)


class ReverseRouteQueryTest(unittest.TestCase):
    def test_sampling_is_reverse_monotone_and_reaches_target(self):
        poses = [
            {"x": float(index) * 0.1, "y": 0.0, "z": 0.0}
            for index in range(21)
        ]
        indices = sample_reverse_indices(
            poses, target_index=2, spacing_m=0.25)
        self.assertEqual(indices[0], 20)
        self.assertEqual(indices[-1], 2)
        self.assertTrue(all(a > b for a, b in zip(indices, indices[1:])))

    def test_wrap_uses_short_turn(self):
        self.assertAlmostEqual(wrap(3.5), 3.5 - 2.0 * 3.141592653589793)


if __name__ == "__main__":
    unittest.main()
