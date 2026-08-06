import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.arrival_shadow import ArrivalShadowConfig
from MemNavData.summarize_arrival_shadow import plan_paths, summarize


def plan(distance, *, gt_distance, critic_stop=False):
    return {
        "goal_aux_pose": [distance, 0.0],
        "anchor": 4,
        "router_selected_anchor": 4,
        "router_active": True,
        "graph_subgoal_enabled": True,
        "graph_subgoal_complete": True,
        "navdp_critic_max": -0.7,
        "navdp_stop_evidence": critic_stop,
        "evaluation_gt_goal_distance_m": gt_distance,
        "evaluation_gt_arrived": gt_distance < 1.0,
    }


class ArrivalShadowSummaryTest(unittest.TestCase):
    def _write(self, root, episode, plans):
        path = root / "07_scene/geometry_router" / f"{episode}_plans.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"legB": plans}), encoding="utf-8")
        return path

    def test_replay_separates_pose_and_critic_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "episode_0000", [
                plan(0.6, gt_distance=1.2),
                plan(0.5, gt_distance=1.1),
                plan(0.4, gt_distance=0.8, critic_stop=True),
            ])
            self._write(root, "episode_0001", [
                plan(0.6, gt_distance=2.0),
                plan(0.5, gt_distance=1.8),
                plan(0.4, gt_distance=1.6, critic_stop=False),
            ])
            paths = plan_paths(root, "geometry_router")
            report = summarize(
                paths, ArrivalShadowConfig(window_plans=3))
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["gt_arrived_episodes"], 1)
            self.assertEqual(report["pose_consensus"]["triggered_episodes"], 2)
            self.assertEqual(report["pose_consensus"]["precision"], 0.5)
            self.assertEqual(
                report["pose_plus_navdp_critic"]["triggered_episodes"], 1)
            self.assertEqual(
                report["pose_plus_navdp_critic"]["precision"], 1.0)
            self.assertEqual(
                report["pose_plus_navdp_critic"]["recall_given_gt_arrival"],
                1.0,
            )

    def test_missing_traces_rejected(self):
        with self.assertRaisesRegex(ValueError, "no arrival-shadow"):
            summarize([], ArrivalShadowConfig())


if __name__ == "__main__":
    unittest.main()
