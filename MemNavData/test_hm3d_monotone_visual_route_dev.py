from __future__ import annotations

import unittest

from MemNavData.analyze_hm3d_monotone_visual_route_dev import (
    contrast,
    mcnemar,
)
from MemNavData.run_hm3d_monotone_visual_route_dev import (
    ARMS,
    first_target_proof,
    guidance_receipt_audit,
    rotated_arm_order,
)


def mono_plan(**updates):
    plan = {
        "navdp_depth_source": "monocular_sidecar",
        "metric_depth_sensor_consumed": False,
        "monocular_depth_receipt": {
            "depth_contract": "raw_lingbot_depth_first40_v1",
            "metric_depth_sensor_consumed": False,
            "frame_index": 80,
            "scale_active": True,
            "scale_receipt_sha256": "scale-sha",
            "scale_receipt": {
                "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
                "whole_episode_ground_cache_consumed": False,
            },
        },
    }
    plan.update(updates)
    return plan


class MonotoneVisualRouteDevelopmentTest(unittest.TestCase):
    def test_exact_mcnemar_and_clustered_contrast(self):
        self.assertAlmostEqual(mcnemar(12, 0), 0.00048828125)
        rows = [
            {"scene": "s0", "route": 1, "endpoint": 0},
            {"scene": "s1", "route": 1, "endpoint": 1},
            {"scene": "s2", "route": 0, "endpoint": 1},
        ]
        result = contrast(
            rows, left="route", right="endpoint", seed=7)
        self.assertEqual(result["paired_gains"], 1)
        self.assertEqual(result["paired_losses"], 1)
        self.assertEqual(result["paired_net_gain"], 0)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.0)

    def test_arm_order_rotates_without_changing_population(self):
        self.assertEqual(rotated_arm_order(0), ARMS)
        self.assertEqual(rotated_arm_order(1), ARMS[1:] + ARMS[:1])
        self.assertEqual(rotated_arm_order(2), ARMS[2:] + ARMS[:2])
        self.assertEqual(rotated_arm_order(3), ARMS)

    def test_target_proof_is_minimal_and_deterministic(self):
        plans = [mono_plan(), mono_plan(
            certified_relocalization_ok=True,
            certified_relocalization_accepted=True,
            certified_relocalization_reason="strict_geometry_certificate_pass",
            anchor=64,
            certified_relocalization_certificate={"inlier_count": 21},
            certified_relocalization_proposal_order="geometry_first",
            certified_relocalization_pointgoal_units=(
                "lingbot_raw_direction_only"),
        )]
        self.assertEqual(first_target_proof(plans), {
            "ok": True,
            "accepted": True,
            "reason": "strict_geometry_certificate_pass",
            "selected_anchor": 64,
            "certificate": {"inlier_count": 21},
            "proposal_order": "geometry_first",
            "pointgoal_units": "lingbot_raw_direction_only",
        })

    def test_visual_route_accept_requires_monotone_no_fallback_receipt(self):
        plan = mono_plan(
            certified_relocalization_ok=True,
            certified_relocalization_accepted=True,
            certified_relocalization_reason="strict_geometry_certificate_pass",
            certified_relocalization_guidance_mode="episodic_path_field",
            certified_relocalization_certificate={"inlier_count": 21},
            certified_relocalization_proposal_order="geometry_first",
            certified_relocalization_pointgoal_units=(
                "lingbot_raw_direction_only"),
            anchor=64,
            role_label_visible=False,
            episodic_path_field_status="active",
            episodic_path_field_error=None,
            episodic_path_field_error_type=None,
            episodic_path_field_evaluator_pose_consumed=False,
            episodic_path_field_habitat_path_consumed=False,
            episodic_path_field_scale_receipt_sha256="scale-sha",
            episodic_path_field_goal_start_frame=128,
            episodic_path_field_target_anchor=64,
            route_coordinate_state_updated=True,
            route_coordinate_endpoint_fallback_available=False,
            route_coordinate_native_fallback_available=False,
            route_coordinate_distance_gate_present=False,
            path_progress_m=2.0,
            path_remaining_m=18.0,
            path_cross_track_m=0.1,
            path_unit_bearing=[0.8, 0.6],
        )
        audit = guidance_receipt_audit(
            "mono_cec_visual_route", "revisit", [plan])
        self.assertFalse(audit["fully_rejected"])
        self.assertTrue(audit["route"]["progress_finite_monotone"])
        self.assertFalse(audit["route"]["endpoint_fallback_available"])

    def test_target_rejection_is_not_a_route_failure(self):
        plan = mono_plan(
            certified_relocalization_ok=True,
            certified_relocalization_accepted=False,
            certified_relocalization_reason="insufficient_geometry_support",
            certified_relocalization_certificate=None,
            certified_relocalization_proposal_order="geometry_first",
            certified_relocalization_pointgoal_units=None,
            anchor=None,
        )
        audit = guidance_receipt_audit(
            "mono_cec_visual_route", "novel", [plan])
        self.assertTrue(audit["fully_rejected"])
        self.assertIsNone(audit["route"])

    def test_endpoint_accept_cannot_expose_path_field(self):
        plan = mono_plan(
            certified_relocalization_ok=True,
            certified_relocalization_accepted=True,
            certified_relocalization_reason="strict_geometry_certificate_pass",
            certified_relocalization_guidance_mode="endpoint_bearing",
            certified_relocalization_certificate={"inlier_count": 21},
            certified_relocalization_proposal_order="geometry_first",
            certified_relocalization_pointgoal_units=(
                "lingbot_raw_direction_only"),
            anchor=64,
            episodic_path_field_requested=True,
        )
        with self.assertRaisesRegex(RuntimeError, "entered route tracking"):
            guidance_receipt_audit(
                "mono_cec_endpoint", "revisit", [plan])


if __name__ == "__main__":
    unittest.main()
