from __future__ import annotations

import math
import unittest

from MemNavData.summarize_hm3d_heldout_val10_revisit import (
    RUNTIME_REPAIR_SCENE_SCHEMA,
    SCENE_SCHEMA,
    cluster_interval,
    exact_mcnemar_p,
    paired,
    validate_scene_contract_schema,
)


class Hm3dMinivalSummaryTest(unittest.TestCase):
    def test_scene_contract_schema_lineages_are_explicit(self) -> None:
        self.assertEqual(validate_scene_contract_schema(
            {"schema_version": SCENE_SCHEMA}, "legacy"), SCENE_SCHEMA)
        self.assertEqual(validate_scene_contract_schema({
            "schema_version": RUNTIME_REPAIR_SCENE_SCHEMA,
            "runtime_repair_method_change": False,
        }, "repair"), RUNTIME_REPAIR_SCENE_SCHEMA)
        with self.assertRaisesRegex(RuntimeError, "method-change guard"):
            validate_scene_contract_schema({
                "schema_version": RUNTIME_REPAIR_SCENE_SCHEMA,
                "runtime_repair_method_change": True,
            }, "bad-repair")
        with self.assertRaisesRegex(RuntimeError, "schema_version changed"):
            validate_scene_contract_schema(
                {"schema_version": "unknown"}, "unknown")

    def test_exact_mcnemar_known_values(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(2, 0), 0.5)
        self.assertEqual(exact_mcnemar_p(12, 0), 0.00048828125)
        self.assertEqual(exact_mcnemar_p(6, 1), 0.125)

    def test_paired_joint_and_conditional(self) -> None:
        keys = [("s0", "e0"), ("s0", "e1"), ("s1", "e0")]
        left = {
            keys[0]: {"reached_a": True, "reached_b": False,
                      "joint": False},
            keys[1]: {"reached_a": True, "reached_b": True,
                      "joint": True},
            keys[2]: {"reached_a": False, "reached_b": False,
                      "joint": False},
        }
        right = {
            keys[0]: {"reached_a": True, "reached_b": True,
                      "joint": True},
            keys[1]: {"reached_a": True, "reached_b": True,
                      "joint": True},
            keys[2]: {"reached_a": False, "reached_b": False,
                      "joint": False},
        }
        joint = paired("left", "right", left, right, keys,
                       conditional_b=False)
        conditional = paired("left", "right", left, right, keys,
                             conditional_b=True)
        self.assertEqual(joint["eligible"], 3)
        self.assertEqual(conditional["eligible"], 2)
        self.assertEqual(conditional["right_only_gain"], 1)
        self.assertEqual(conditional["left_only_loss"], 0)
        self.assertTrue(math.isclose(
            conditional["risk_difference_right_minus_left"], 0.5))

    def test_cluster_bootstrap_is_reproducible(self) -> None:
        scenes = ["s0", "s1"]
        episode_ids = {"s0": ["e0", "e1"], "s1": ["e0", "e1"]}
        left = {}
        right = {}
        for scene in scenes:
            for episode in episode_ids[scene]:
                key = (scene, episode)
                left[key] = {"reached_a": True, "reached_b": False,
                             "joint": False}
                right[key] = {"reached_a": True, "reached_b": scene == "s0",
                              "joint": scene == "s0"}
        first = cluster_interval(
            scenes, episode_ids, left, right, conditional_b=True,
            seed=17, resamples=1000)
        second = cluster_interval(
            scenes, episode_ids, left, right, conditional_b=True,
            seed=17, resamples=1000)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[1], 1.0)


if __name__ == "__main__":
    unittest.main()
