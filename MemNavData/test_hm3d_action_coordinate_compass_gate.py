from __future__ import annotations

import copy
import unittest

from MemNavData.run_hm3d_action_coordinate_compass_gate import (
    audit_action_coordinate_plans,
)


def plan(progress: float, translation: float) -> dict:
    return {
        "role_label_visible": False,
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": True,
        "certified_relocalization_guidance_mode": (
            "action_coordinate_compass"),
        "action_coordinate_compass_status": "active",
        "action_coordinate_compass_error": None,
        "action_coordinate_compass_error_type": None,
        "action_coordinate_schema_version": (
            "action_coordinate_route_compass_v1_20260902"),
        "action_coordinate_evaluator_pose_consumed": False,
        "action_coordinate_habitat_path_consumed": False,
        "action_coordinate_metric_scale_consumed": False,
        "action_coordinate_visual_gate_present_after_initialization": False,
        "action_coordinate_distance_regime_present": False,
        "action_coordinate_endpoint_fallback_available": False,
        "action_coordinate_native_fallback_available": False,
        "action_coordinate_progress_m": progress,
        "action_coordinate_route_progress_fraction": progress / 10.0,
        "action_coordinate_update_translation_m": translation,
        "action_coordinate_update_yaw_rad": 0.1,
        "action_coordinate_route_extent_m": 10.0,
        "action_coordinate_unit_bearing": [1.0, 0.0],
        "action_coordinate_controller_pointgoal": [2.5, 0.0],
        "memory_controller_pointgoal_distance_m": 2.5,
        "action_coordinate_history_receipt_sha256": "a" * 64,
        "action_coordinate_history_receipt_count": 100,
        "action_coordinate_goal_start_frame": 120,
        "action_coordinate_target_anchor": 20,
        "executor_motion_receipt": {
            "contract": "frame_bound_realized_executor_motion_v1",
            "frame_index": 121,
            "executed_translation_m": translation,
            "executed_yaw_rad": 0.1,
        },
    }


class ActionCoordinateGateAuditTest(unittest.TestCase):
    def test_valid_monotone_receipts(self) -> None:
        result = audit_action_coordinate_plans([
            plan(0.0, 0.0), plan(0.7, 0.7), plan(1.4, 0.7),
        ])
        self.assertTrue(result["progress_finite_monotone"])
        self.assertEqual(result["positive_translation_update_count"], 2)
        self.assertEqual(result["fixed_controller_radius_m"], 2.5)

    def test_progress_regression_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "progress regressed"):
            audit_action_coordinate_plans([
                plan(0.0, 0.0), plan(1.0, 1.0), plan(0.5, 0.5),
            ])

    def test_fallback_flag_fails(self) -> None:
        rows = [plan(0.0, 0.0), plan(0.5, 0.5)]
        rows = copy.deepcopy(rows)
        rows[-1]["action_coordinate_native_fallback_available"] = True
        with self.assertRaisesRegex(RuntimeError, "fallback"):
            audit_action_coordinate_plans(rows)


if __name__ == "__main__":
    unittest.main()
