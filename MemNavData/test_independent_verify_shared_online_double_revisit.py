import unittest

from MemNavData.independent_verify_shared_online_double_revisit import (
    cluster_interval,
    exact_pair,
)


class IndependentDoubleRevisitStatisticsTest(unittest.TestCase):
    def test_independent_exact_pair_counts_gain_direction(self):
        result = exact_pair(
            [True, True, False, True], [False, True, False, False])
        self.assertEqual(result, {
            "N": 4,
            "first_success": 3,
            "second_success": 1,
            "risk_difference_pp": 50.0,
            "gain": 2,
            "loss": 0,
            "discordant": 2,
            "exact_mcnemar_p": 0.5,
        })

    def test_independent_cluster_bootstrap_is_deterministic_and_clustered(self):
        rows = [
            ("scene_a", True, False),
            ("scene_a", False, False),
            ("scene_b", True, True),
        ]
        first = cluster_interval(rows, seed=11, resamples=2000)
        second = cluster_interval(rows, seed=11, resamples=2000)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)
        self.assertEqual(first["episodes"], 3)
        self.assertAlmostEqual(first["risk_difference_pp"], 100.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
