import unittest

from audit_hm3d_fullmono_lifelong_constructibility import (
    SCHEMA,
    STAGES,
    aggregate_payloads,
    classify_measurement,
    summarize_recipient_measurements,
)


class ConstructibilityAuditTest(unittest.TestCase):
    def test_candidate_waterfall_stops_at_first_failed_contract(self):
        passed, rejection = classify_measurement(
            floor_delta_m=0.0,
            a_to_b_reachable=True,
            b_to_c_reachable=True,
            a_to_b_geodesic_m=4.0,
            b_to_c_geodesic_m=3.0,
            max_recipient_a_covis=0.2,
            same_floor_tolerance_m=0.2,
            a_to_b_band_m=(2.0, 9.0),
            b_to_c_band_m=(2.0, 9.0),
            maximum_a_covis=0.1,
        )
        self.assertEqual(rejection, "recipient_history_support_not_novel")
        self.assertEqual(passed[-1], "b_to_c_in_band")
        self.assertNotIn("novel_support", passed)

    def test_eligible_candidate_reaches_every_stage(self):
        passed, rejection = classify_measurement(
            floor_delta_m=0.2,
            a_to_b_reachable=True,
            b_to_c_reachable=True,
            a_to_b_geodesic_m=2.0,
            b_to_c_geodesic_m=9.0,
            max_recipient_a_covis=0.099,
            same_floor_tolerance_m=0.2,
            a_to_b_band_m=(2.0, 9.0),
            b_to_c_band_m=(2.0, 9.0),
            maximum_a_covis=0.1,
        )
        self.assertEqual(rejection, "eligible")
        self.assertEqual(tuple(passed), STAGES)

    def test_summary_counts_candidates_and_recipient_coverage(self):
        measurements = [
            {"passed_stages": list(STAGES), "first_rejection": "eligible"},
            {
                "passed_stages": list(STAGES[:4]),
                "first_rejection": "a_to_b_outside_band",
            },
        ]
        result = summarize_recipient_measurements(measurements)
        self.assertEqual(result["candidate_stage_counts"]["temporal_proposal"], 2)
        self.assertEqual(result["candidate_stage_counts"]["novel_support"], 1)
        self.assertTrue(result["recipient_reaches_stage"]["novel_support"])
        self.assertEqual(result["candidate_first_rejection"]["eligible"], 1)

    def test_aggregate_is_result_blind_and_clustered(self):
        recipient = {
            "controlled_revisit_source_status": "constructible",
            "candidate_stage_counts": {stage: 1 for stage in STAGES},
            "candidate_first_rejection": {"eligible": 1},
            "recipient_reaches_stage": {stage: True for stage in STAGES},
            "sealed_selected_candidates": 1,
            "sealed_selection_reproduced": True,
        }
        payloads = []
        for index, scene in enumerate(("a", "b")):
            payloads.append({
                "schema_version": SCHEMA,
                "scene": scene,
                "scene_index": index,
                "query_policy_outcomes_read": False,
                "navigation_outcomes_read": False,
                "sealed_selection_reproduced": True,
                "selected_candidate_count": 1,
                "selected_recipient_count": 1,
                "recipients": [dict(recipient)],
            })
        result = aggregate_payloads(payloads)
        self.assertEqual(result["source_materialized_A_histories"], 2)
        self.assertEqual(result["sealed_selected_scene_clusters"], 2)
        self.assertEqual(result["candidate_stage_counts"]["novel_support"], 2)
        self.assertFalse(result["navigation_outcomes_read"])


if __name__ == "__main__":
    unittest.main()
