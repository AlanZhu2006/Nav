import unittest

from MemNavData.run_hm3d_episodic_path_field_gate import (
    audit_path_field_plans,
)


def plan(progress=0.0, remaining=8.0):
    return {
        "role_label_visible": False,
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": True,
        "certified_relocalization_guidance_mode": "episodic_path_field",
        "episodic_path_field_status": "active",
        "episodic_path_field_error": None,
        "episodic_path_field_error_type": None,
        "episodic_path_field_evaluator_pose_consumed": False,
        "episodic_path_field_habitat_path_consumed": False,
        "episodic_path_field_scale_receipt_sha256": "a" * 64,
        "episodic_path_field_goal_start_frame": 100,
        "episodic_path_field_target_anchor": 20,
        "path_progress_m": progress,
        "path_remaining_m": remaining,
        "path_cross_track_m": 0.1,
        "path_unit_bearing": [1.0, 0.0],
        "route_coordinate_state_updated": True,
        "route_coordinate_endpoint_fallback_available": False,
        "route_coordinate_native_fallback_available": False,
        "route_coordinate_distance_gate_present": False,
    }


class Hm3dEpisodicPathFieldGateTest(unittest.TestCase):
    def test_accepts_monotone_runtime_only_receipts(self):
        result = audit_path_field_plans([
            plan(0.0, 8.0), plan(0.5, 7.5), plan(1.2, 6.8),
        ])
        self.assertTrue(result["progress_finite_monotone"])
        self.assertEqual(result["active_path_readout_count"], 3)
        self.assertEqual(result["visual_route_update_count"], 3)
        self.assertFalse(result["evaluator_pose_consumed"])

    def test_rejects_progress_regression(self):
        with self.assertRaisesRegex(RuntimeError, "progress regressed"):
            audit_path_field_plans([plan(1.0, 7.0), plan(0.9, 7.1)])

    def test_rejects_geometry_failure(self):
        failed = plan()
        failed["episodic_path_field_status"] = "geometry_failure"
        failed["episodic_path_field_error"] = "bad scale"
        with self.assertRaisesRegex(RuntimeError, "valid path-field state"):
            audit_path_field_plans([failed])

    def test_rejects_any_post_authorization_fallback(self):
        failed = plan()
        failed["route_coordinate_native_fallback_available"] = True
        with self.assertRaisesRegex(RuntimeError, "fallback or distance gate"):
            audit_path_field_plans([failed])


if __name__ == "__main__":
    unittest.main()
