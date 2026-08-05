import copy
import json
import unittest
from pathlib import Path

from MemNavData.summarize_expanded_3leg_router_eval import arm_summary
from MemNavData.validate_expanded_3leg_router_eval import validate_selection


ROOT = Path(__file__).resolve().parent


class ExpandedThreeLegBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "expanded_3leg_router_eval_20260805.json").read_text()
        )
        cls.base_manifest = json.loads(
            (ROOT / cls.manifest["base_manifest"]["file"]).read_text()
        )

    def test_selection_is_frozen_and_scene_disjoint(self):
        scenes = validate_selection(self.manifest, self.base_manifest)
        self.assertEqual(scenes, self.base_manifest["selection"]["selected_scenes"][:10])
        self.assertFalse(set(scenes) & set(self.base_manifest["training_scenes"]))

    def test_selection_change_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["selection"]["selected_scenes"][-1] = \
            self.base_manifest["selection"]["selected_scenes"][10]
        broken["episodes"][broken["selection"]["selected_scenes"][-1]] = \
            broken["episodes"].pop("gTV8FGcVJC9")
        with self.assertRaisesRegex(RuntimeError, "frozen rule"):
            validate_selection(broken, self.base_manifest)

    def test_summary_uses_sequential_eligibility(self):
        base = {
            "spl_a": 0.7,
            "spl_b": 0.6,
            "spl_c": 0.5,
            "joint_spl": 0.4,
            "final_dist_a": 0.3,
            "final_dist_b": 0.4,
            "final_dist_c": 0.5,
            "router_active_episode_a": False,
            "router_active_episode_b": False,
            "router_active_episode_c": True,
            "geometry_verification_ms": [11.0],
        }
        rows = [
            dict(base, reached_a=True, reached_b=True, reached_c=True, joint=True),
            dict(base, reached_a=True, reached_b=False, reached_c=False, joint=False,
                 router_active_episode_c=False),
            dict(base, reached_a=False, reached_b=False, reached_c=False, joint=False,
                 router_active_episode_c=False),
        ]
        summary = arm_summary(rows)
        self.assertEqual(summary["leg_A_novel"]["successes"], 2)
        self.assertEqual(summary["leg_B_novel_given_A"]["eligible"], 2)
        self.assertEqual(summary["leg_B_novel_given_A"]["successes"], 1)
        self.assertEqual(summary["leg_C_revisit_given_AB"]["eligible"], 1)
        self.assertEqual(summary["leg_C_revisit_given_AB"]["successes"], 1)
        self.assertEqual(summary["joint"]["successes"], 1)
        self.assertEqual(
            summary["router"]["revisit_C_activation_rate_given_AB"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
