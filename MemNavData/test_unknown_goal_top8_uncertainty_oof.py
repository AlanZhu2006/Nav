import unittest

import numpy as np
import pandas as pd

from MemNavData.analyze_unknown_goal_top8_uncertainty_oof import (
    TOP8_FEATURE_NAMES,
    compare_with_f2,
    top8_summary_table,
)


def geometry_rows(count=8):
    rows = []
    for rank in range(count):
        rows.append({
            "session_id": "s0",
            "candidate_rank": rank,
            "dino_cosine": 0.95 - 0.01 * rank,
            "geometry_matches": 30 - rank,
            "geometry_inliers": 20 - rank,
            "geometry_inlier_ratio": 0.8 - 0.05 * rank,
            "geometry_hard_pass": int(rank == 2),
            "geometry_essential_available_rate": 1.0,
            "geometry_pose_recovered_rate": 0.5,
            "geometry_pass_rate": 0.25,
            # Forbidden teacher fields may exist but must not affect features.
            "label": int(rank == 7),
            "covisibility": float(rank),
        })
    return pd.DataFrame(rows)


class Top8SummaryTests(unittest.TestCase):
    def test_fixed_feature_contract_and_first_pass(self):
        result = top8_summary_table(geometry_rows().sample(frac=1, random_state=7))
        self.assertEqual(result.shape, (1, 1 + len(TOP8_FEATURE_NAMES)))
        self.assertEqual(result.iloc[0]["top8_hard_pass_count"], 1.0)
        self.assertEqual(result.iloc[0]["top8_first_hard_pass_rank_or_8"], 2.0)
        self.assertTrue(np.isfinite(result[list(TOP8_FEATURE_NAMES)]).all().all())

    def test_no_pass_sentinel(self):
        rows = geometry_rows()
        rows["geometry_hard_pass"] = 0
        result = top8_summary_table(rows)
        self.assertEqual(result.iloc[0]["top8_first_hard_pass_rank_or_8"], 8.0)

    def test_requires_exact_top8(self):
        with self.assertRaisesRegex(ValueError, "ranks 0..7"):
            top8_summary_table(geometry_rows(7))

    def test_teacher_fields_do_not_change_features(self):
        original = top8_summary_table(geometry_rows())
        changed = geometry_rows()
        changed["label"] = 1 - changed["label"]
        changed["covisibility"] = -999.0
        revised = top8_summary_table(changed)
        np.testing.assert_allclose(
            original[list(TOP8_FEATURE_NAMES)],
            revised[list(TOP8_FEATURE_NAMES)],
        )

    def test_frozen_gate(self):
        geometry = {
            "strict_false_activations": 9,
            "positive_wrong_anchor_activated": 14,
            "positive_correct_anchor_activated": 93,
            "correct_support_decisions": 365,
        }
        f2 = {
            "strict_false_activations": 5,
            "positive_wrong_anchor_activated": 12,
            "positive_correct_anchor_activated": 89,
            "correct_support_decisions": 365,
        }
        f8 = {
            "strict_false_activations": 8,
            "positive_wrong_anchor_activated": 13,
            "positive_correct_anchor_activated": 94,
            "correct_support_decisions": 367,
        }
        gate = compare_with_f2(
            {"methods": {"factor": f8, "geometry": geometry}},
            {"methods": {"factor": f2, "geometry": geometry}},
        )
        self.assertTrue(gate["top8_pass"])


if __name__ == "__main__":
    unittest.main()
