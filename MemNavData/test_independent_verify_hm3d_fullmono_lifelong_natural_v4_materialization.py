#!/usr/bin/env python3

import unittest

from independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization import (
    planar_separation,
)


class NaturalV4IndependentVerifierTest(unittest.TestCase):
    def test_planar_separation_recounts_all_within_recipient_pairs(self):
        groups = {
            ("s0", "e0"): [
                [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]
            ],
            ("s0", "e1"): [[1.0, 0.0, 1.0]],
        }
        self.assertEqual(planar_separation(groups), 3)

    def test_planar_separation_rejects_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "below 2 m"):
            planar_separation({
                ("s0", "e0"): [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            })


if __name__ == "__main__":
    unittest.main()
