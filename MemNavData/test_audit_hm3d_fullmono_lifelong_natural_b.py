#!/usr/bin/env python3

import unittest

from audit_hm3d_fullmono_lifelong_natural_b import (
    SCHEMA,
    aggregate_payloads,
)


class NaturalBAuditTest(unittest.TestCase):
    @staticmethod
    def payload(scene, histories, rows):
        return {
            "schema_version": SCHEMA,
            "scene": scene,
            "source_materialized_A_histories": histories,
            "query_policy_outcomes_read": False,
            "navigation_outcomes_read": False,
            "recipients": rows,
        }

    @staticmethod
    def row(scene, episode, status, candidates=()):
        return {
            "scene": scene,
            "episode": episode,
            "status": status,
            "candidates": [
                {
                    "assigned_direction_stratum": stratum,
                    "max_online_a_covis": covis,
                }
                for stratum, covis in candidates
            ],
        }

    def test_aggregate_counts_histories_and_scene_clusters(self):
        result = aggregate_payloads([
            self.payload("s0", 3, [
                self.row("s0", "e0", "constructible", [
                    ("front", 0.01), ("side", 0.03)
                ]),
                self.row("s0", "e1", "no_natural_B_candidate"),
            ]),
            self.payload("s1", 2, [
                self.row("s1", "e0", "constructible", [("rear", 0.09)]),
            ]),
        ])
        self.assertEqual(result["source_materialized_A_histories"], 5)
        self.assertEqual(result["controlled_revisit_constructible_histories"], 3)
        self.assertEqual(result["natural_B_constructible_recipients"], 2)
        self.assertEqual(result["natural_B_candidate_histories"], 3)
        self.assertEqual(result["natural_B_constructible_scene_clusters"], 2)
        self.assertEqual(
            result["direction_strata"], {"front": 1, "rear": 1, "side": 1}
        )
        self.assertEqual(result["candidate_max_online_A_covis"]["maximum"], 0.09)
        self.assertFalse(result["evaluation_authorized"])

    def test_aggregate_rejects_outcome_access(self):
        payload = self.payload("s0", 0, [])
        payload["navigation_outcomes_read"] = True
        with self.assertRaisesRegex(RuntimeError, "navigation outcomes"):
            aggregate_payloads([payload])


if __name__ == "__main__":
    unittest.main()
