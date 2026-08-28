import math
import unittest

from MemNavData.analyze_native_revisit_failure_20260828 import (
    bearing_to_goal_deg,
    classify_failure,
    direction_stratum,
    relative_direction_deg,
    trace_behavior,
    wrap_angle,
)


class AngleConventionTest(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(math.pi + 0.1), -math.pi + 0.1)
        self.assertAlmostEqual(wrap_angle(-math.pi - 0.1), math.pi - 0.1)

    def test_relative_direction_matches_sealed_novel_field(self):
        # Real values from 58NLZxWBSpk/episode_0001 pair_00_novel:
        # stored initial_path_direction_relative_to_a_end_deg = -165.0747782711247
        got = relative_direction_deg(-1.7492694217672964, 1.1318289710634222)
        self.assertAlmostEqual(got, -165.0747782711247, places=9)

    def test_direction_stratum_thresholds(self):
        self.assertEqual(direction_stratum(0.0), "front")
        self.assertEqual(direction_stratum(-45.0), "front")
        self.assertEqual(direction_stratum(46.0), "lateral")
        self.assertEqual(direction_stratum(-134.9), "lateral")
        self.assertEqual(direction_stratum(135.0), "back")
        self.assertEqual(direction_stratum(-165.1), "back")

    def test_bearing_to_goal_habitat_frame(self):
        # yaw 0 faces -Z: a goal straight ahead is bearing 0.
        self.assertAlmostEqual(
            bearing_to_goal_deg(0, 0, 0.0, 0.0, -1.0), 0.0)
        # Goal on the agent's left (-X) is +90.
        self.assertAlmostEqual(
            bearing_to_goal_deg(0, 0, 0.0, -1.0, 0.0), 90.0)
        # Goal directly behind is 180.
        self.assertAlmostEqual(
            abs(bearing_to_goal_deg(0, 0, 0.0, 0.0, 1.0)), 180.0)


class TraceBehaviorTest(unittest.TestCase):
    def _row(self, x, z, yaw):
        return {"x": x, "z": z, "yaw": yaw}

    def test_forward_walk_toward_goal(self):
        trace = [self._row(0.0, -i * 0.5, 0.0) for i in range(5)]
        stats = trace_behavior(trace, (0.0, -4.0))
        self.assertAlmostEqual(stats["initial_goal_distance_m"], 4.0)
        self.assertAlmostEqual(stats["min_goal_distance_m"], 2.0)
        self.assertAlmostEqual(stats["path_length_m"], 2.0)
        self.assertTrue(stats["ever_aligned_within_45deg"])
        self.assertEqual(stats["first_alignment_step"], 0)
        self.assertAlmostEqual(stats["max_abs_cumulative_turn_deg"], 0.0)

    def test_goal_behind_never_aligned(self):
        # Agent walks -Z while the goal sits behind it at +Z.
        trace = [self._row(0.0, -i * 0.5, 0.0) for i in range(4)]
        stats = trace_behavior(trace, (0.0, 3.0))
        self.assertFalse(stats["ever_aligned_within_45deg"])
        self.assertIsNone(stats["first_alignment_step"])
        self.assertGreater(stats["final_goal_distance_m"],
                           stats["initial_goal_distance_m"])

    def test_u_turn_registers_cumulative_heading_change(self):
        trace = [self._row(0.0, 0.0, yaw)
                 for yaw in (0.0, math.pi / 2, math.pi)]
        stats = trace_behavior(trace, (0.0, 5.0))
        self.assertAlmostEqual(
            stats["max_abs_cumulative_turn_deg"], 180.0, places=6)
        self.assertTrue(stats["ever_aligned_within_45deg"])


class FailureTaxonomyTest(unittest.TestCase):
    def test_near_miss(self):
        self.assertEqual(
            classify_failure({"min_goal_distance_m": 1.2,
                              "initial_goal_distance_m": 5.0}),
            "near_miss_no_arrival")

    def test_never_approached(self):
        self.assertEqual(
            classify_failure({"min_goal_distance_m": 4.8,
                              "initial_goal_distance_m": 5.0}),
            "never_approached")

    def test_partial_approach_stall(self):
        self.assertEqual(
            classify_failure({"min_goal_distance_m": 3.0,
                              "initial_goal_distance_m": 5.0}),
            "partial_approach_stall")


if __name__ == "__main__":
    unittest.main()
