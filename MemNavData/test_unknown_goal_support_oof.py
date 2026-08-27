import json
import unittest

import numpy as np
import pandas as pd

from MemNavData.analyze_unknown_goal_support_oof import (
    choose_risk_matched_threshold,
    exact_mcnemar_p,
    scene_folds,
    session_feature_table,
    summarize_decisions,
)


class UnknownGoalSupportOOFTest(unittest.TestCase):
    def test_risk_threshold_maximizes_positive_coverage(self):
        scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        labels = np.asarray([True, False, True, False, True, False])
        result = choose_risk_matched_threshold(scores, labels, 1.0 / 3.0)
        self.assertEqual(result["positive_active"], 2)
        self.assertEqual(result["strict_false_activations"], 1)
        self.assertAlmostEqual(result["threshold"], 0.7)

    def test_scene_folds_are_disjoint_and_complete(self):
        scenes = [f"scene-{index}" for index in range(11)]
        folds = scene_folds(scenes, 5, 17)
        flattened = [str(item) for fold in folds for item in fold]
        self.assertEqual(set(flattened), set(scenes))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            [list(fold) for fold in folds],
            [list(fold) for fold in scene_folds(scenes, 5, 17)],
        )

    def test_session_features_use_only_deployment_values(self):
        rows = []
        for rank, point, score, matches, inliers in (
            (0, [1.0, 0.0], 0.9, 20, 12),
            (1, [0.0, 1.0], 0.8, 10, 5),
        ):
            rows.append({
                "session_id": "s0",
                "scene": "scene0",
                "candidate_rank": rank,
                "candidate_frame": rank,
                "dino_cosine": score,
                "geometry_matches": matches,
                "geometry_inliers": inliers,
                "geometry_inlier_ratio": inliers / matches,
                "geometry_hard_pass": int(rank == 0),
                "geometry_pose_recovered_rate": 1.0,
                "cloud_overlap_f1_center": 0.2 + rank,
                "anchor_goal_distance_norm_center": 0.5 + rank,
                "goal_refine_translation_norm_median": 0.1 + rank,
                "goal_refine_rotation_deg_median": 1.0 + rank,
                "predicted_relative_xy_m_center_json": json.dumps(point),
                "session_has_positive": True,
                "session_is_strict_no_match": False,
                "label": 1 if rank == 0 else 0,
            })
        table, names = session_feature_table(pd.DataFrame(rows))
        self.assertEqual(len(table), 1)
        self.assertEqual(len(names), 21)
        self.assertAlmostEqual(float(table.iloc[0]["dino_margin"]), 0.1)
        self.assertAlmostEqual(
            float(table.iloc[0]["pointgoal_direction_agreement"]), 0.0)
        self.assertTrue(bool(table.iloc[0]["top2_has_positive"]))

    def test_decision_summary_keeps_wrong_anchor_separate(self):
        records = [
            dict(is_positive=True, is_strict_no_match=False,
                 active=True, selected_positive=True),
            dict(is_positive=True, is_strict_no_match=False,
                 active=True, selected_positive=False),
            dict(is_positive=True, is_strict_no_match=False,
                 active=False, selected_positive=False),
            dict(is_positive=False, is_strict_no_match=True,
                 active=True, selected_positive=False),
            dict(is_positive=False, is_strict_no_match=True,
                 active=False, selected_positive=False),
        ]
        result = summarize_decisions(records)
        self.assertEqual(result["positive_correct_anchor_activated"], 1)
        self.assertEqual(result["positive_wrong_anchor_activated"], 1)
        self.assertEqual(result["strict_false_activations"], 1)
        self.assertEqual(result["correct_support_decisions"], 2)

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(12, 0), 0.00048828125)


if __name__ == "__main__":
    unittest.main()
