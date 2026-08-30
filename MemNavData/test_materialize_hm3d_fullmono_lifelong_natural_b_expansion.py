#!/usr/bin/env python3

import unittest

from materialize_hm3d_fullmono_lifelong_natural_b_expansion import (
    _audit_candidate_without_receipt_fields,
    require_expansion_separation,
)


class NaturalBExpansionMaterializationTest(unittest.TestCase):
    def test_receipt_only_candidate_fields_are_removed(self):
        row = {
            "candidate_slot": 4,
            "candidate_identity": "episode_0000__natural_b_04",
            "goal_floor_position": [2.0, 0.0, 0.0],
            "max_online_a_covis": 0.01,
        }
        self.assertEqual(
            _audit_candidate_without_receipt_fields(row),
            {"max_online_a_covis": 0.01},
        )

    def test_separation_includes_original_and_new_candidates(self):
        require_expansion_separation(
            [[0.0, 0.0, 0.0]],
            [
                {"_position": [2.0, 0.0, 0.0]},
                {"_position": [4.0, 0.0, 0.0]},
            ],
            2.0,
        )
        with self.assertRaisesRegex(RuntimeError, "frozen planar separation"):
            require_expansion_separation(
                [[0.0, 0.0, 0.0]],
                [{"_position": [1.99, 0.0, 0.0]}],
                2.0,
            )
        with self.assertRaisesRegex(RuntimeError, "frozen planar separation"):
            require_expansion_separation(
                [],
                [
                    {"_position": [2.0, 0.0, 0.0]},
                    {"_position": [3.99, 0.0, 0.0]},
                ],
                2.0,
            )


if __name__ == "__main__":
    unittest.main()
