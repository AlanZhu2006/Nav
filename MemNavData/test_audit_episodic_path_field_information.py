import math
import unittest

import numpy as np

from MemNavData.audit_episodic_path_field_information import (
    summarize_rows,
    world_delta_from_bearing,
    world_delta_to_forward_left,
)
from MemNavData.bearing_diagnostics import (
    bearing_error_deg_from_world_delta,
)


class EpisodicPathFieldInformationAuditTest(unittest.TestCase):
    def test_world_conversion_matches_bearing_diagnostic(self):
        for bearing in (-2.4, -0.3, 0.0, 1.7, math.pi):
            for yaw in (-1.2, 0.0, 2.1):
                delta = world_delta_from_bearing(bearing)
                prediction = world_delta_to_forward_left(delta, yaw)
                self.assertAlmostEqual(
                    bearing_error_deg_from_world_delta(
                        prediction, delta, yaw),
                    0.0,
                    places=6,
                )

    def test_frozen_gate_uses_all_five_criteria(self):
        rows = []
        for bin_name in ("a", "b", "c"):
            for index in range(10):
                rows.append({
                    "bin_name": bin_name,
                    "direct_error_deg": 50.0,
                    "route_error_deg": 10.0 if bin_name != "c" else 48.0,
                    "gauge_direction_error_deg": 0.0,
                    "progress_finite_monotone": True,
                })
        summary = summarize_rows(rows)
        self.assertTrue(summary["information_gate_pass"])
        self.assertEqual(summary["n"], 30)
        self.assertAlmostEqual(
            summary["within_30_improvement_fraction"], 2.0 / 3.0)

        rows[0] = dict(rows[0], progress_finite_monotone=False)
        self.assertFalse(summarize_rows(rows)["information_gate_pass"])

    def test_direction_vector_is_unit_length(self):
        vector = world_delta_to_forward_left([2.0, -3.0], 0.7)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)


if __name__ == "__main__":
    unittest.main()
