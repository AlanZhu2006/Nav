import unittest
import json

import numpy as np

from audit_raw_novel_anchor_attribution import (
    angular_error_deg,
    cohort_summary,
    eligible_anchor_indices,
    local_angle_to_point_deg,
    random_anchor_null,
)


class RawNovelAnchorAttributionTest(unittest.TestCase):
    def test_wrapped_angular_error(self):
        self.assertAlmostEqual(angular_error_deg(179.0, -179.0), 2.0)
        self.assertAlmostEqual(angular_error_deg(-170.0, 170.0), 20.0)

    def test_habitat_local_angle(self):
        current = np.asarray([0.0, 0.0])
        self.assertAlmostEqual(
            local_angle_to_point_deg(current, 0.0, np.asarray([0.0, -1.0])),
            0.0,
        )
        self.assertAlmostEqual(
            local_angle_to_point_deg(current, 0.0, np.asarray([-1.0, 0.0])),
            90.0,
        )

    def test_candidate_interval_is_inferred_from_receipt(self):
        plan = {
            "frame_idx": 206,
            "candidate_ceiling": 205,
            "candidate_count": 144,
        }
        indices = eligible_anchor_indices(plan)
        self.assertEqual(indices[0], 31)
        self.assertEqual(indices[-1], 174)
        self.assertEqual(len(indices), 144)

    def test_random_anchor_null_is_deterministic(self):
        records = [
            {
                "dino_anchor_physical_error_deg": 10.0,
                "eligible_anchor_errors_deg": [10.0, 100.0],
            },
            {
                "dino_anchor_physical_error_deg": 20.0,
                "eligible_anchor_errors_deg": [20.0, 120.0],
            },
        ]
        left = random_anchor_null(records, seed=7, resamples=1000)
        right = random_anchor_null(records, seed=7, resamples=1000)
        self.assertEqual(left, right)
        self.assertEqual(left["factual_count_le_30_deg"], 2)

    def test_cohort_summary_is_json_serializable(self):
        records = [
            {
                "scene": "scene-a",
                "dino_anchor_physical_error_deg": 10.0,
                "raw_bearing_error_deg": 5.0,
                "uniform_anchor_error_mean_deg": 55.0,
                "uniform_anchor_probability_le_30_deg": 0.5,
                "constant_uturn_error_deg": 20.0,
                "dino_anchor_to_raw_bearing_error_deg": 5.0,
                "eligible_anchor_errors_deg": [10.0, 100.0],
            }
        ]
        payload = cohort_summary(records, seed=7, resamples=10)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
