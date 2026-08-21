import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.analyze_unknown_goal_natural_stream_hard_pilot import (
    build_report,
)


def candidate(anchor, rank, label, matches, inliers, ratio):
    return {
        "anchor": anchor,
        "rank": rank,
        "label": label,
        "matches": matches,
        "inliers": inliers,
        "inlier_ratio": ratio,
    }


class HardPilotAuditTests(unittest.TestCase):
    def teacher(self, root: Path, episode: str) -> Path:
        records = [
            {
                "decision_index": 0, "leg": "legA", "leg_plan_index": 0,
                "step": 0, "topk_support_label": 0,
                "first_positive_rank": None, "candidates": [],
            },
            {
                "decision_index": 1, "leg": "legB", "leg_plan_index": 0,
                "step": 0, "topk_support_label": 0,
                "first_positive_rank": None,
                "candidates": [candidate(2, 1, 0, 30, 20, 0.7)],
            },
            {
                "decision_index": 2, "leg": "legC", "leg_plan_index": 0,
                "step": 0, "topk_support_label": 1,
                "first_positive_rank": 2,
                "candidates": [
                    candidate(3, 1, 0, 30, 20, 0.7),
                    candidate(4, 2, 1, 15, 8, 0.6),
                ],
            },
            {
                "decision_index": 3, "leg": "legC", "leg_plan_index": 1,
                "step": 8, "topk_support_label": 1,
                "first_positive_rank": 1,
                "candidates": [candidate(4, 1, 1, 30, 20, 0.7)],
            },
            {
                "decision_index": 4, "leg": "legC", "leg_plan_index": 2,
                "step": 16, "topk_support_label": -1,
                "first_positive_rank": None,
                "candidates": [candidate(5, 1, -1, 10, 4, 0.4)],
            },
        ]
        path = root / f"{episode}.json"
        path.write_text(json.dumps({
            "status": "complete",
            "inputs": {"episode_root": f"/dataset/scene0/{episode}"},
            "records": records,
        }), encoding="utf-8")
        return path

    def test_reconstructs_geometry_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.teacher(Path(tmp), "episode_0000")
            report = build_report(
                [path], min_matches=20, min_inliers=12,
                min_inlier_ratio=0.5)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["combined"]["plans"], 5)
        self.assertEqual(
            report["combined"]["strict_negative_geometry_false_support_plans"],
            1)
        leg_c = report["episodes"][0]["by_leg"]["legC"]
        self.assertEqual(leg_c["positive_support_plans"], 2)
        self.assertEqual(leg_c["geometry_correct_positive_plans"], 1)
        self.assertEqual(leg_c["geometry_miss_positive_plans"], 1)
        self.assertEqual(leg_c["dino_top1_miss_positive_plans"], 1)
        self.assertEqual(leg_c["max_consecutive_positive_support_plans"], 2)
        self.assertEqual(leg_c["max_positive_anchor_plan_count"], 2)

    def test_rejects_duplicate_episode_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.teacher(Path(tmp), "episode_0000")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_report([path, path], min_matches=20, min_inliers=12,
                             min_inlier_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
