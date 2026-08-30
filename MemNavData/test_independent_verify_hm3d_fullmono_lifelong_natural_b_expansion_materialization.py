#!/usr/bin/env python3

import unittest

from independent_verify_hm3d_fullmono_lifelong_natural_b_expansion_materialization import (
    planar_distance,
)


class ExpansionMaterializationVerifierTest(unittest.TestCase):
    def test_planar_distance_ignores_height(self):
        self.assertAlmostEqual(
            planar_distance([0.0, 7.0, 0.0], [3.0, -2.0, 4.0]),
            5.0,
        )

    def test_planar_distance_rejects_bad_shape(self):
        with self.assertRaisesRegex(RuntimeError, "three coordinates"):
            planar_distance([0.0, 0.0], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
