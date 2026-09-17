import unittest
from plan_repaired_fullmono_population import choose_prefix, prepare


class PopulationPlanTests(unittest.TestCase):
    def setUp(self):
        scenes = [f"s{i}" for i in range(54)]
        p = {"scenes": scenes, "episode_count": 108,
             "episodes": {s: [{"episode": f"ep{j}", "files": {}, "goal_a_geodesic_m": 3.+j} for j in range(2)] for s in scenes},
             "assets": {s: {} for s in scenes}, "paths": {"generated_root": "/source"}}
        self.plan = prepare(p)

    def reports(self, prefix, supports=True):
        return [{"source_index": s["source_index"], "scene": s["scene"], "episode": s["episode"],
                 "verified": True, "construction_complete": True, "pair_constructible": supports,
                 "query_rollouts": 0} for s in self.plan["sources"] if s["scene_rank"] < prefix]

    def test_preserve_episode_ranks_in_seed(self):
        self.assertEqual([s["seed"] for s in self.plan["sources"][:4]], [2026082200,2026082201,2026082300,2026082301])

    def test_all_histories_in_minimum_complete_prefix_retained(self):
        r = choose_prefix(self.plan, self.reports(30))
        self.assertEqual(r["selected_prefix"], 30)
        self.assertEqual(r["histories"], 60)  # no top-24 cherry picking
        self.assertFalse(r["ready_for_query"])

    def test_cannot_stop_on_partial_prefix_even_with_enough_histories(self):
        r = choose_prefix(self.plan, self.reports(29))
        self.assertIsNone(r["selected_prefix"])
        self.assertEqual(r["next_prefix"], 30)

    def test_insufficient_population_advances_only_predeclared_prefix(self):
        r = choose_prefix(self.plan, self.reports(30, supports=False))
        self.assertEqual(r["next_prefix"], 36)

    def test_all_empty_is_underpowered_not_navigation_failure(self):
        r = choose_prefix(self.plan, self.reports(54, supports=False))
        self.assertEqual(r["state"], "maximum_prefix_underpowered")
        self.assertEqual(r["histories"], 0)

    def test_duplicate_or_outcome_read_is_rejected(self):
        rs = self.reports(30)
        with self.assertRaises(ValueError):
            choose_prefix(self.plan, rs+[rs[0]])
        rs[0]["query_rollouts"] = 1
        with self.assertRaises(ValueError):
            choose_prefix(self.plan, rs)


if __name__ == "__main__":
    unittest.main()
