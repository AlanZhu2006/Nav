import unittest

from audit_final14_support_band_constructibility import (
    classify_support,
    deterministic_grid,
)


class Final14SupportBandConstructibilityTest(unittest.TestCase):
    def test_support_bands_are_disjoint(self):
        self.assertEqual(classify_support(0.55, 24), "standard")
        self.assertEqual(classify_support(0.549999, 24), "hard")
        self.assertEqual(classify_support(0.25, 32), "hard")
        self.assertIsNone(classify_support(0.90, 25))
        self.assertIsNone(classify_support(0.249, 2))

    def test_grid_is_deterministic_and_bounded(self):
        first = deterministic_grid("scene/episode/39")
        second = deterministic_grid("scene/episode/39")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6 * 12 * 8)
        self.assertTrue(all(0.20 <= radius <= 1.00 for radius, _, _ in first))
        self.assertTrue(all(12.0 <= abs(offset) <= 60.0 for _, _, offset in first))


if __name__ == "__main__":
    unittest.main()
