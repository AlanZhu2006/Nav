import unittest

from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    paired_summary,
)


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


if __name__ == "__main__":
    unittest.main()
