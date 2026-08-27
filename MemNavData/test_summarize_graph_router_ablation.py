import unittest
import copy
import tempfile
from pathlib import Path
from unittest import mock

from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    paired_summary,
)
from MemNavData.summarize_graph_router_ablation import summarize_ablation


def row(*, joint=False, active=False):
    return {
        "scene": "scene",
        "episode": "episode_0000",
        "seed": 7,
        "recall_gap": 48,
        "reached_a": True,
        "reached_b": joint,
        "joint": joint,
        "spl_a": 1.0,
        "spl_b": float(joint),
        "geo_a": 2.0,
        "geo_b": 3.0,
        "path_a": 2.0,
        "path_b": 3.0,
        "final_dist_a": 0.5,
        "final_dist_b": 0.5 if joint else 2.0,
        "steps_a": 10,
        "steps_b": 20,
        "router_plans_a": 1,
        "router_plans_b": 2,
        "router_active_plans_a": 0,
        "router_active_plans_b": int(active),
        "router_active_episode_a": False,
        "router_active_episode_b": active,
        "geometry_verification_ms": [5.0],
        "selected_candidate_ranks": [1],
        "deterministic_plan_seeds": True,
    }


class GraphAblationSummaryTest(unittest.TestCase):
    def test_arm_reports_conditional_revisit_and_activation(self):
        report = arm_summary([row(joint=True, active=True)])
        self.assertEqual(report["joint"]["successes"], 1)
        self.assertEqual(
            report["revisit_given_novel_success"]["successes"], 1)
        self.assertEqual(report["router"]["revisit_activation_episodes"], 1)

    def test_pairing_reports_one_recovered_episode(self):
        key = ("scene", "episode_0000")
        report = paired_summary(
            "direct", "graph", {key: row()},
            {key: row(joint=True, active=True)}, {key})
        self.assertEqual(
            report["outcomes"]["right_only_joint_success"], 1)
        self.assertEqual(report["joint_sr_delta_right_minus_left"], 1.0)

    def test_pairing_requires_identical_shared_novel_trace(self):
        key = ("scene", "episode_0000")
        left = row()
        right = copy.deepcopy(left)
        left["leg1_trace_sha256"] = "a" * 64
        right["leg1_trace_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "trace mismatch"):
            paired_summary("direct", "graph", {key: left}, {key: right}, {key})

    def test_partial_mode_uses_only_complete_scene_intersection(self):
        manifest = {
            "selection": {"selected_scenes": ["scene_a", "scene_b"]},
            "training_scenes": [],
            "episodes": {
                "scene_a": [{"episode": "episode_0000"}],
                "scene_b": [{"episode": "episode_0000"}],
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            graph = root / "graph"
            for arm_root in (reference, graph):
                metric = (arm_root / "scenes" / "00_scene_a" /
                          "geometry_router" / "metric.csv")
                metric.parent.mkdir(parents=True)
                metric.touch()
            metric = (reference / "scenes" / "01_scene_b" /
                      "geometry_router" / "metric.csv")
            metric.parent.mkdir(parents=True)
            metric.touch()

            def fake_load(_scene_root, _arm, scene):
                value = row(joint=True, active=True)
                value["scene"] = scene
                return {(scene, "episode_0000"): value}

            with mock.patch(
                    "MemNavData.summarize_graph_router_ablation.load_arm",
                    side_effect=fake_load):
                report = summarize_ablation(
                    manifest, reference, {"graph": graph},
                    complete_scene_intersection=True)
        self.assertEqual(report["audit"]["scenes"], 1)
        self.assertEqual(report["audit"]["episodes"], 1)
        self.assertTrue(report["audit"]["not_a_full_manifest_result"])
        self.assertEqual(
            report["audit"]["excluded_scenes"][0]["scene"], "scene_b")


if __name__ == "__main__":
    unittest.main()
