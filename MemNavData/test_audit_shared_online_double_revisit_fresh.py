import unittest

from MemNavData.audit_shared_online_double_revisit_fresh import (
    exact_mcnemar,
    scene_cluster_bootstrap,
)


class FreshAuditStatisticsTest(unittest.TestCase):
    def test_exact_mcnemar_counts_direction(self):
        result = exact_mcnemar(
            [True, True, False, True], [False, True, False, False]
        )
        self.assertEqual(result["gain"], 2)
        self.assertEqual(result["loss"], 0)
        self.assertEqual(result["risk_difference_pp"], 50.0)

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [("a", True, False), ("a", False, False), ("b", True, True)]
        first = scene_cluster_bootstrap(rows, seed=7, resamples=1000)
        second = scene_cluster_bootstrap(rows, seed=7, resamples=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)


if __name__ == "__main__":
    unittest.main()
