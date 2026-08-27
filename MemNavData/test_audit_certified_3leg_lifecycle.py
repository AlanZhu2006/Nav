import unittest

from MemNavData.audit_certified_3leg_lifecycle import (
    angular_error_deg,
    multigoal_topology_audit,
    percentile,
    summarize_leg,
    world_to_local_forward_left,
)


def plan(*, cached, accepted=True, anchor=7, rank=2, bearing=(1.0, 0.0)):
    return {
        "certified_relocalization_cached": cached,
        "certified_relocalization_accepted": accepted,
        "certified_relocalization_uncached_ms": 10.0 if not cached else 10.0,
        "certified_relocalization_pnp": {
            "inliers": 20,
            "reprojection_rmse_px": 1.0,
            "query_inlier_coverage": 0.2,
            "reference_inlier_coverage": 0.3,
        },
        "router_selected_anchor": anchor,
        "router_selected_candidate_dino_rank": rank,
        "router_candidate_order_dino": [4, 7],
        "memory_bearing_unit": list(bearing),
    }


def record(plans, *, success=True):
    return {
        "selection_index": 0,
        "scene": "scene",
        "episode": "episode",
        "plans": {"legB": plans, "legC": []},
        "rollout_traces": {
            "legA": [
                {"step": value, "x": float(value), "z": 0.0, "yaw": 0.0}
                for value in range(5)
            ],
            "legB": [
                {"step": value, "x": 0.0, "z": 0.0, "yaw": 0.0}
                for value in range(9)
            ],
            "legC": [],
        },
        "goal_xz": {"legB": (0.0, -1.0), "legC": (0.0, 0.0)},
        "success": {"legB": success, "legC": False},
        "termination": {"legB": "success" if success else "stuck",
                        "legC": "censored"},
        "steps": {"legB": 8, "legC": 0},
        "final_distance_m": {"legB": 0.9 if success else 2.0,
                             "legC": 3.0},
    }


class CertifiedThreeLegLifecycleTest(unittest.TestCase):
    def test_world_bearing_and_angular_error(self):
        current = {"x": 0.0, "z": 0.0, "yaw": 0.0}
        self.assertEqual(
            world_to_local_forward_left(current, (0.0, -2.0)),
            (2.0, 0.0),
        )
        self.assertAlmostEqual(angular_error_deg((1.0, 0.0), (1.0, 1.0)), 45.0)

    def test_percentile_linear_interpolation(self):
        self.assertEqual(percentile([], 0.5), None)
        self.assertEqual(percentile([1.0], 0.5), 1.0)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)

    def test_uncached_localization_is_not_counted_per_plan(self):
        rows = [record([
            plan(cached=False, bearing=(1.0, 0.0)),
            plan(cached=True, bearing=(0.0, 1.0)),
            plan(cached=True, bearing=(-1.0, 0.0)),
        ])]
        result = summarize_leg(rows, "legB")
        self.assertEqual(result["planning_requests"], 3)
        self.assertEqual(result["independent_uncached_localizations"], 1)
        self.assertEqual(result["cached_reuses"], 2)
        self.assertEqual(result["episodes_uncached_then_cached_only"], 1)
        self.assertEqual(result["episodes_bearing_changed_after_motion"], 1)

    def test_navigation_failure_remains_distinct_from_certificate_reject(self):
        rows = [record([plan(cached=False)], success=False)]
        result = summarize_leg(rows, "legB")
        self.assertEqual(result["accepted_independent_localizations"], 1)
        self.assertEqual(result["rejected_independent_localizations"], 0)
        self.assertEqual(result["navigation_success"], 0)
        self.assertEqual(
            len(result["navigation_failures_after_accepted_certificate"]), 1)

    def test_goal_switch_topology_is_bidirectional_and_shorter(self):
        value = record([], success=True)
        value["plans"] = {
            "legB": [plan(cached=False, anchor=1)],
            "legC": [plan(cached=False, anchor=3)],
        }
        value["rollout_traces"]["legB"] = [
            {"step": index, "x": float(index), "z": 0.0, "yaw": 0.0}
            for index in range(9)
        ]
        # Old routing reverses all of B (8 m) and then A 4->3 (1 m), whereas
        # the previous-anchor route follows A 1->3 (2 m).
        result = multigoal_topology_audit([value])
        self.assertEqual(result["temporal_direction_counts"], {"forward": 1})
        self.assertEqual(result["legacy_over_anchor_route_ratio_median"], 4.5)


if __name__ == "__main__":
    unittest.main()
