import unittest

from MemNavData.independent_verify_certified_graph_rescue_expansion import (
    contrast,
    exact_p,
)


class IndependentExpansionStatisticsTest(unittest.TestCase):
    def test_exact_small_sample_mcnemar(self):
        self.assertEqual(exact_p(0, 0), 1.0)
        self.assertEqual(exact_p(3, 0), 0.25)
        self.assertEqual(exact_p(1, 1), 1.0)

    def test_raw_contrast_is_scene_clustered(self):
        rows = []
        for index in range(4):
            rows.append({
                "scene": f"scene_{index // 2}",
                "rescue": {"joint": index in (0, 1, 2)},
                "direct": {"joint": index == 0},
            })
        value = contrast(
            rows, "rescue", "direct", "joint",
            seed=3, resamples=1_000)
        self.assertEqual(value["gain"], 2)
        self.assertEqual(value["loss"], 0)
        self.assertEqual(value["scene_cluster_bootstrap"]["clusters"], 2)


if __name__ == "__main__":
    unittest.main()
