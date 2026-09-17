"""Outcome-blind design inventory tests; no GPU, renderer or training data."""
import unittest
from audit_repaired_fullmono_design import source_inventory


class DesignTests(unittest.TestCase):
    def parent(self):
        return {"scenes": ["a", "b", "empty"], "episode_count": 4,
                "episodes": {"a": [{"episode": "episode_0000", "goal_a_geodesic_m": 3.},
                                     {"episode": "episode_0001", "goal_a_geodesic_m": 6.}],
                             "b": [{"episode": "episode_0000", "goal_a_geodesic_m": 4.},
                                     {"episode": "episode_0001", "goal_a_geodesic_m": 5.}],
                             "empty": []}}

    def test_inventory_does_not_select_or_require_outcomes(self):
        result = source_inventory(self.parent())
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["scenes"], 2)
        self.assertFalse(result["selection_performed"])
        self.assertFalse(result["query_outcomes_read"])
        self.assertEqual([b["episodes"] for b in result["distance_inventory"]], [1, 1, 1, 1])

    def test_original_scene_and_episode_ranks_are_preserved(self):
        result = source_inventory(self.parent())
        self.assertEqual([(r["scene_rank"], r["episode_rank"]) for r in result["sources"]],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_declared_count_must_match(self):
        p = self.parent()
        p["episode_count"] = 5
        with self.assertRaises(ValueError):
            source_inventory(p)

    def test_does_not_drop_out_of_range_tasks(self):
        p = self.parent()
        p["episodes"]["a"][0]["goal_a_geodesic_m"] = 20.
        with self.assertRaises(ValueError):
            source_inventory(p)


if __name__ == "__main__":
    unittest.main()
