import ast
from pathlib import Path
import unittest

from collect_repaired_fullmono_a import scene_sources


class CollectorTests(unittest.TestCase):
    def plan(self):
        return {"status": "prepared_not_submitted_not_query_sealed", "new_a_required": True,
                "old_a_results_for_selection": False,
                "sources": [{"source_index": i, "scene_rank": 5, "scene": "s", "episode": f"ep{i}",
                             "episode_rank": i, "seed": 2026082700+i} for i in range(4)]}

    def test_all_four_tasks_are_kept_with_distinct_seeds(self):
        rows = scene_sources(self.plan(), 5)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["seed"] for r in rows}), 4)

    def test_empty_scene_cannot_silently_collect_another(self):
        with self.assertRaises(ValueError):
            scene_sources(self.plan(), 6)

    def test_replay_and_wrong_episode_rank_rejected(self):
        p = self.plan()
        p["new_a_required"] = False
        with self.assertRaises(ValueError):
            scene_sources(p, 5)
        p = self.plan()
        p["sources"][1]["seed"] -= 1
        with self.assertRaises(ValueError):
            scene_sources(p, 5)

    def test_collector_does_not_invoke_query_or_construction(self):
        tree = ast.parse(Path(__file__).with_name("collect_repaired_fullmono_a.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "evaluator_command"]
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0].keywords)  # original A-only native defaults
        self.assertEqual(len(calls[0].args), 4)


if __name__ == "__main__":
    unittest.main()
