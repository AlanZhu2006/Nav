#!/usr/bin/env python3

import unittest

from materialize_hm3d_fullmono_lifelong_natural_ab import (
    equivalent,
    require_planar_separation,
)


class NaturalABMaterializationTest(unittest.TestCase):
    def test_recursive_equivalence_tolerates_only_tiny_float_noise(self):
        equivalent(
            {"a": [1, 2.0], "b": {"c": "x"}},
            {"a": [1.0, 2.0 + 1e-10], "b": {"c": "x"}},
        )
        with self.assertRaisesRegex(RuntimeError, "numeric value changed"):
            equivalent({"a": 2.0}, {"a": 2.0 + 1e-5})

    def test_planar_separation_is_strictly_recomputed(self):
        require_planar_separation([
            {"_position": [0.0, 0.0, 0.0]},
            {"_position": [2.0, 0.0, 0.0]},
        ])
        with self.assertRaisesRegex(RuntimeError, "2 m"):
            require_planar_separation([
                {"_position": [0.0, 0.0, 0.0]},
                {"_position": [1.99, 0.0, 0.0]},
            ])


if __name__ == "__main__":
    unittest.main()
