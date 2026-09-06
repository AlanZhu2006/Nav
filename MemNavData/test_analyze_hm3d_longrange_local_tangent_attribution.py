import unittest

from MemNavData.analyze_hm3d_longrange_local_tangent_attribution import (
    fixed_motion_panel,
    fixed_plan_panel,
)


class LocalTangentAnalysisTest(unittest.TestCase):
    def test_fixed_plan_panel(self):
        plans = [
            {
                "local_tangent_oracle_arm": "x",
                "local_tangent_oracle_current_geodesic_m": value,
                "local_tangent_oracle_signed_heading_deg": heading,
                "local_tangent_oracle_controller": controller,
                "local_tangent_oracle_pointgoal": pointgoal,
                "navdp_critic_max": critic,
            }
            for value, heading, controller, critic, pointgoal in (
                (10.0, -100.0, "mixed_image_pointgoal", -0.6,
                 [-0.434120, -2.462019]),
                (8.0, 10.0, "native_imagegoal", 0.2,
                 [2.462019, 0.434120]),
                (9.0, 170.0, "native_imagegoal", -0.4,
                 [-2.462019, 0.434120]),
            )
        ]
        panel = fixed_plan_panel(plans)
        self.assertEqual(panel["plan_count"], 3)
        self.assertEqual(panel["minimum_planned_geodesic_m"], 8.0)
        self.assertEqual(panel["maximum_geodesic_reduction_m"], 2.0)
        self.assertEqual(panel["critic_below_minus_0p5_count"], 1)
        self.assertEqual(panel["behind_pointgoal_count"], 2)
        self.assertEqual(panel["rear_dead_zone_165deg_count"], 1)
        self.assertAlmostEqual(
            panel["post_navdp_clip_pointgoal_norm_min_m"], 0.434120)
        self.assertEqual(panel["mixed_plan_count"], 1)
        self.assertEqual(panel["native_plan_count"], 2)

    def test_fixed_motion_panel(self):
        payload = {
            "rollout_traces": {"query": [
                {"x": 0.0, "y": 1.0, "z": 0.0},
                {"x": 0.0, "y": 1.0, "z": 0.0},
                {"x": 0.3, "y": 0.8, "z": 0.4},
            ]},
            "query_result": {
                "end_position": [0.3, 0.8, 0.4],
                "path_len_m": 0.5,
                "blocked_step_count": 2,
                "termination_reason": "max_steps",
            },
        }
        panel = fixed_motion_panel(payload)
        self.assertEqual(panel["stationary_transition_count"], 1)
        self.assertAlmostEqual(panel["trace_realized_planar_motion_m"], 0.5)
        self.assertAlmostEqual(
            panel["recomputed_total_realized_planar_motion_m"], 0.5)
        self.assertAlmostEqual(panel["vertical_span_m"], 0.2)
        self.assertEqual(panel["blocked_step_count"], 2)


if __name__ == "__main__":
    unittest.main()
