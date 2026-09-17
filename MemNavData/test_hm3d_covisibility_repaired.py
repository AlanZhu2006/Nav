"""CPU tests for support measurement and SR-independent query selection."""
import copy
from pathlib import Path
import unittest

import numpy as np
import build_hm3d_covisibility_repaired as b


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.protocol = b.load(Path(__file__).with_name("hm3d_covisibility_repaired_protocol_20260909.json"))
        self.intrinsic = np.array([[100., 0, 60], [0, 100., 36], [0, 0, 1]])
        self.depth = np.full((72, 120), 2., dtype=np.float32)
        self.identity = np.eye(4)

    def test_exact_bin_boundaries(self):
        expected = {0.: None, .0999999: None, .1: "c10_30", .29999: "c10_30",
                    .3: "c30_50", .5: "c50_70", .7: "c70_90", .9: "c90_100", 1.: "c90_100"}
        for value, label in expected.items():
            self.assertEqual(b.support_bin(value, self.protocol), label)

    def test_invalid_support_is_not_clipped(self):
        for value in (-.01, 1.001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                b.support_bin(value, self.protocol)

    def test_selection_order_independent_and_missing_preserved(self):
        rows = [{"bin": "c10_30", "yaw_index": n, "exact_history_jpeg": False} for n in range(20)]
        a = b.select_candidates(rows, "scene/episode", self.protocol)
        self.assertEqual(a, b.select_candidates(rows[::-1], "scene/episode", self.protocol))
        self.assertIsNone(a["c90_100"])

    def test_exact_jpeg_never_selected(self):
        rows = [{"bin": "c90_100", "yaw_index": 0, "exact_history_jpeg": True}]
        self.assertIsNone(b.select_candidates(rows, "scene/episode", self.protocol)["c90_100"])

    def test_no_model_scores_used_for_selection(self):
        rows = [{"bin": "c10_30", "yaw_index": n, "exact_history_jpeg": False, "SR": n, "DINO": n} for n in range(5)]
        a = b.select_candidates(rows, "s/e", self.protocol)["c10_30"]["yaw_index"]
        for row in rows:
            row.update(SR=-row["SR"], DINO=-row["DINO"])
        self.assertEqual(a, b.select_candidates(rows, "s/e", self.protocol)["c10_30"]["yaw_index"])

    def test_actual_projection_not_legacy_fy(self):
        matrix = np.eye(4)
        matrix[0, 0], matrix[1, 1] = 2*355.81464/480, 2*355.81464/270
        intrinsic = b.projection_intrinsics(matrix, 480, 270)
        self.assertAlmostEqual(intrinsic[0, 0], intrinsic[1, 1])
        self.assertGreater(abs(intrinsic[1, 1]-351.687), 4.)

    def test_same_surface_has_full_visibility(self):
        points = b.surface_points(self.depth, self.identity, self.intrinsic, self.protocol)
        self.assertAlmostEqual(b.visibility(points, self.identity, self.depth, self.intrinsic, .3), 1.)

    def test_occluded_surface_has_zero_visibility(self):
        points = b.surface_points(self.depth, self.identity, self.intrinsic, self.protocol)
        self.assertEqual(b.visibility(points, self.identity, self.depth-1, self.intrinsic, .3), 0.)

    def test_invalid_reference_depth_does_not_support(self):
        points = b.surface_points(self.depth, self.identity, self.intrinsic, self.protocol)
        self.assertEqual(b.visibility(points, self.identity, self.depth*0, self.intrinsic, .3), 0.)

    def test_rear_facing_reference_does_not_support(self):
        points = b.surface_points(self.depth, self.identity, self.intrinsic, self.protocol)
        rotated = np.diag([-1., 1., -1., 1.])
        self.assertEqual(b.visibility(points, rotated, self.depth, self.intrinsic, .3), 0.)

    def test_eligible_and_all_history_are_separate(self):
        transforms = [self.identity]*41
        depths = [self.depth]*39 + [self.depth-1]*2
        a = b.annotate(self.depth, self.identity, transforms, depths, self.intrinsic, self.protocol)
        self.assertEqual(a["q_all"], 1.)
        self.assertEqual(a["q_eligible"], 0.)
        self.assertEqual(a["argmax_eligible"], 39)

    def test_float_depth_above_png_limit_retained(self):
        depth = np.full_like(self.depth, 8.)
        points = b.surface_points(depth, self.identity, self.intrinsic, self.protocol)
        self.assertTrue(np.all(points[:, 2] == -8.))
        self.assertEqual(b.visibility(points, self.identity, depth, self.intrinsic, .3), 1.)


if __name__ == "__main__":
    unittest.main()
