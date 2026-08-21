import unittest

from MemNavData.summarize_revisit_benchmark_coverage import (
    nnr_summary,
    zero_event_upper,
)


class ZeroEventUpperTest(unittest.TestCase):
    def test_matches_exact_binomial_bound(self):
        self.assertAlmostEqual(zero_event_upper(7), 0.3481636551311609)

    def test_rule_of_three_scale(self):
        self.assertLess(zero_event_upper(60), 0.05)
        self.assertLess(zero_event_upper(150), 0.02)

    def test_empty_denominator(self):
        self.assertIsNone(zero_event_upper(0))


class NnrSummaryTest(unittest.TestCase):
    def test_extracts_paired_architecture_decision(self):
        report = {
            "schema_version": (
                "shared_online_novel_revisit_paired_report_v1_20260814"),
            "scope": "internal",
            "source_population_size": 22,
            "constructible_population_size": 2,
            "construction_rejections": [{}, {}],
            "scene_clusters": 2,
            "all_shared_prefixes_equal": True,
            "all_treatment_prefixes_equal": True,
            "intervention_episodes": 1,
            "actual_graph_plan_count": 4,
            "records": [{}, {}],
            "arms": {
                name: {
                    "successes": successes,
                    "episodes": 2,
                    "SR_C_given_frozen_online_AB": successes / 2,
                }
                for name, successes in {
                    "native": 0, "known_direct": 1, "certified": 2,
                    "certified_budget": 2, "certified_graph": 2,
                }.items()
            },
            "contrasts": {
                name: {
                    "gains": gains, "losses": losses,
                    "exact_mcnemar_two_sided_p": 1.0,
                    "risk_difference_pp": 0.0,
                    "scene_cluster_bootstrap_95ci_pp": [0.0, 0.0],
                }
                for name, gains, losses in (
                    ("certified_minus_native", 2, 0),
                    ("known_direct_minus_native", 1, 0),
                    ("certified_graph_minus_certified_budget", 0, 0),
                )
            },
        }
        result = nnr_summary(report)
        self.assertEqual(result["arms"]["certified"]["successes"], 2)
        self.assertEqual(
            result["contrasts"][
                "certified_graph_minus_certified_budget"]["gains"], 0)


if __name__ == "__main__":
    unittest.main()
