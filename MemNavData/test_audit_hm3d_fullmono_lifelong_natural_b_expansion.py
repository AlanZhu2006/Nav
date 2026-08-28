#!/usr/bin/env python3

import unittest

from audit_hm3d_fullmono_lifelong_natural_b_expansion import (
    SCHEMA,
    aggregate_payloads,
)


class NaturalBExpansionAuditTest(unittest.TestCase):
    @staticmethod
    def payload(scene, source_histories, rows):
        return {
            "schema_version": SCHEMA,
            "scene": scene,
            "source_materialized_A_histories": source_histories,
            "query_policy_outcomes_read": False,
            "navigation_outcomes_read": False,
            "evaluation_authorized": False,
            "recipients": rows,
        }

    @staticmethod
    def row(scene, episode, original, candidates=()):
        return {
            "scene": scene,
            "episode": episode,
            "status": (
                "constructible" if candidates
                else "no_additional_natural_B_candidate"
            ),
            "original_candidate_count": original,
            "candidates": [
                {
                    "assigned_direction_stratum": stratum,
                    "max_online_a_covis": covis,
                }
                for stratum, covis in candidates
            ],
        }

    def test_aggregate_counts_only_additional_candidates(self):
        result = aggregate_payloads([
            self.payload("s0", 3, [
                self.row("s0", "e0", 2, [
                    ("front", 0.01), ("rear", 0.03)
                ]),
                self.row("s0", "e1", 0),
            ]),
            self.payload("s1", 2, [
                self.row("s1", "e0", 1, [("side", 0.09)]),
            ]),
        ])
        self.assertEqual(result["source_materialized_A_histories"], 5)
        self.assertEqual(result["controlled_revisit_constructible_histories"], 3)
        self.assertEqual(result["original_candidate_histories_referenced"], 3)
        self.assertEqual(result["expansion_constructible_recipients"], 2)
        self.assertEqual(result["expansion_candidate_histories"], 3)
        self.assertEqual(result["expansion_scene_clusters"], 2)
        self.assertEqual(
            result["direction_strata"], {"front": 1, "rear": 1, "side": 1}
        )
        self.assertFalse(result["evaluation_authorized"])

    def test_aggregate_rejects_outcome_access(self):
        payload = self.payload("s0", 0, [])
        payload["navigation_outcomes_read"] = True
        with self.assertRaisesRegex(RuntimeError, "navigation outcomes"):
            aggregate_payloads([payload])

    def test_aggregate_rejects_evaluation_authority(self):
        payload = self.payload("s0", 0, [])
        payload["evaluation_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "authorized navigation"):
            aggregate_payloads([payload])


if __name__ == "__main__":
    unittest.main()
