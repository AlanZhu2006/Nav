import copy
import json
import unittest
from pathlib import Path

from MemNavData.summarize_expanded_navdp_router_eval import arm_summary, truth
from MemNavData.validate_expanded_navdp_router_eval import validate_selection
from MemNavData.validate_frozen_router_blind import compare_value


ROOT = Path(__file__).resolve().parent


class ExpandedBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "expanded_navdp_router_eval_20260805.json").read_text()
        )

    def test_frozen_selection_is_scene_disjoint(self):
        selected = validate_selection(self.manifest)
        self.assertEqual(len(selected), 20)
        self.assertFalse(set(selected) & set(self.manifest["training_scenes"]))

    def test_training_scene_leak_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        leaked = broken["training_scenes"][0]
        broken["selection"]["selected_scenes"][0] = leaked
        with self.assertRaisesRegex(RuntimeError, "selected scene outside eligible pool"):
            validate_selection(broken)

    def test_summary_separates_novel_revisit_and_joint(self):
        base = {
            "spl_a": 0.5,
            "spl_b": 0.4,
            "final_dist_a": 0.5,
            "final_dist_b": 0.7,
            "router_active_episode_a": False,
            "router_active_episode_b": True,
            "geometry_verification_ms": [10.0],
        }
        rows = [
            dict(base, reached_a=True, reached_b=True, joint=True),
            dict(base, reached_a=True, reached_b=False, joint=False),
            dict(base, reached_a=False, reached_b=False, joint=False,
                 router_active_episode_b=False),
        ]
        summary = arm_summary(rows)
        self.assertEqual(summary["novel"]["successes"], 2)
        self.assertEqual(summary["revisit_given_novel_success"]["eligible"], 2)
        self.assertEqual(summary["revisit_given_novel_success"]["successes"], 1)
        self.assertEqual(summary["joint"]["successes"], 1)

    def test_json_and_csv_truth_values(self):
        self.assertTrue(truth(True))
        self.assertTrue(truth("1.0"))
        self.assertFalse(truth(False))
        self.assertFalse(truth("0.0"))
        self.assertFalse(truth(None))

    def test_frozen_router_numeric_comparison_is_strict(self):
        compare_value("weights", [1.0, 2.0], [1.0, 2.0 + 1e-13])
        with self.assertRaisesRegex(RuntimeError, "frozen numeric field changed"):
            compare_value("weights", [1.0, 2.0], [1.0, 2.0 + 1e-6])


if __name__ == "__main__":
    unittest.main()
