from __future__ import annotations

import copy
import unittest

from MemNavData.run_hm3d_se2_route_compass_gate import (
    audit_se2_route_plans,
)


def plan(progress: float, path_length: float, forward: float) -> dict:
    return {
        "role_label_visible": False,
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": True,
        "certified_relocalization_guidance_mode": "se2_route_compass",
        "se2_route_compass_status": "active",
        "se2_route_compass_error": None,
        "se2_route_compass_error_type": None,
        "se2_route_schema_version": "se2_projected_route_compass_v2_20260902",
        "se2_route_evaluator_pose_consumed_as_odometry_proxy": True,
        "se2_route_world_pose_consumed_by_policy": False,
        "se2_route_executor_odometry_required": True,
        "se2_route_executor_odometry_source": (
            "habitat_pose_difference_odometry_proxy_v1"),
        "se2_route_habitat_path_consumed": False,
        "se2_route_lingbot_translation_consumed": False,
        "se2_route_metric_scale_consumed": False,
        "se2_route_visual_gate_present_after_initialization": False,
        "se2_route_distance_regime_present": False,
        "se2_route_endpoint_fallback_available": False,
        "se2_route_native_fallback_available": False,
        "se2_route_projected_progress_m": progress,
        "se2_route_progress_fraction": progress / 10.0,
        "se2_route_query_path_length_m": path_length,
        "se2_route_cross_track_error_m": 0.2,
        "se2_route_extent_m": 10.0,
        "se2_route_estimated_position": [-progress, 0.2],
        "se2_route_projected_position": [-progress, 0.0],
        "se2_route_reference_position": [-progress - 2.5, 0.0],
        "se2_route_unit_bearing": [1.0, 0.0],
        "se2_route_controller_pointgoal": [2.5, 0.0],
        "memory_controller_pointgoal_distance_m": 2.5,
        "se2_route_history_receipt_sha256": "b" * 64,
        "se2_route_history_receipt_count": 100,
        "se2_route_goal_start_frame": 120,
        "se2_route_target_anchor": 20,
        "executor_motion_receipt": {
            "contract": "frame_bound_realized_executor_motion_v1",
            "frame_index": 121,
            "executed_translation_m": abs(forward),
            "executed_yaw_rad": 0.1,
            "local_se2": {
                "contract": "frame_bound_local_se2_v1",
                "executed_forward_m": forward,
                "executed_left_m": 0.01,
                "executed_yaw_rad": 0.1,
                "source": "habitat_pose_difference_odometry_proxy_v1",
            },
        },
    }


class SE2RouteGateAuditTest(unittest.TestCase):
    def test_valid_local_receipts(self) -> None:
        result = audit_se2_route_plans([
            plan(0.0, 0.0, 0.0),
            plan(0.4, 0.7, 0.7),
            plan(0.8, 1.4, 0.7),
        ])
        self.assertTrue(result["progress_finite_monotone"])
        self.assertEqual(result["frame_bound_local_se2_receipt_count"], 3)
        self.assertEqual(result["fixed_controller_radius_m"], 2.5)

    def test_progress_regression_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "progress regressed"):
            audit_se2_route_plans([
                plan(0.0, 0.0, 0.0),
                plan(1.0, 1.0, 1.0),
                plan(0.5, 2.0, 1.0),
            ])

    def test_missing_local_receipt_fails(self) -> None:
        rows = [plan(0.0, 0.0, 0.0), plan(0.5, 1.0, 1.0)]
        rows = copy.deepcopy(rows)
        rows[-1]["executor_motion_receipt"].pop("local_se2")
        with self.assertRaisesRegex(RuntimeError, r"local SE\(2\)"):
            audit_se2_route_plans(rows)

    def test_forbidden_lingbot_translation_fails(self) -> None:
        rows = [plan(0.0, 0.0, 0.0), plan(0.5, 1.0, 1.0)]
        rows[-1]["se2_route_lingbot_translation_consumed"] = True
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            audit_se2_route_plans(rows)

    def test_odometry_proxy_cannot_be_hidden_or_renamed(self) -> None:
        rows = [plan(0.0, 0.0, 0.0), plan(0.5, 1.0, 1.0)]
        rows = copy.deepcopy(rows)
        rows[-1]["executor_motion_receipt"]["local_se2"]["source"] = (
            "claimed_rgb_only")
        with self.assertRaisesRegex(RuntimeError, r"local SE\(2\)"):
            audit_se2_route_plans(rows)


if __name__ == "__main__":
    unittest.main()
