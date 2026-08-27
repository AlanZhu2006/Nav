import unittest

from MemNavData.audit_online_memory_oracle import (
    adaptive_keyframes,
    positive_retained,
    summarize_episode,
    trace_path_length,
    uniform_keyframes,
    wrap_angle,
)


class OnlineMemoryOracleTest(unittest.TestCase):
    def setUp(self):
        self.trace = [
            {"frame_idx": 10 + i, "x": float(i), "z": 0.0, "yaw": 0.0}
            for i in range(6)
        ]

    def test_trace_path_length(self):
        self.assertAlmostEqual(trace_path_length(self.trace, 1, 4), 3.0)
        self.assertEqual(trace_path_length(self.trace, 2, 2), 0.0)

    def test_uniform_keyframes_preserve_endpoints(self):
        selected = uniform_keyframes(10, 4)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 9)
        self.assertLessEqual(len(selected), 4)

    def test_adaptive_keyframes_preserve_motion_coverage(self):
        selected = adaptive_keyframes(self.trace, distance_m=2.0, yaw_deg=20.0)
        self.assertEqual(selected, [0, 2, 4, 5])

    def test_adaptive_keyframes_retain_turn(self):
        trace = [dict(row) for row in self.trace]
        trace[1]["yaw"] = 0.5
        selected = adaptive_keyframes(trace, distance_m=10.0, yaw_deg=20.0)
        self.assertIn(1, selected)

    def test_positive_retained_uses_trace_index(self):
        rows = [
            {"trace_index": 1, "teacher_covis": 0.7},
            {"trace_index": 4, "teacher_covis": 0.2},
        ]
        self.assertTrue(positive_retained(rows, [0, 1], 0.5))
        self.assertFalse(positive_retained(rows, [0, 4], 0.5))

    def test_episode_summary(self):
        rows = [
            {"trace_index": 0, "candidate_frame": 10,
             "teacher_covis": 0.1, "goal_distance_m": 3.0,
             "goal_yaw_error_deg": 30.0},
            {"trace_index": 2, "candidate_frame": 12,
             "teacher_covis": 0.8, "goal_distance_m": 1.0,
             "goal_yaw_error_deg": 5.0},
        ]
        result = summarize_episode(
            rows, self.trace, positive_threshold=0.5, uniform_cap=6,
            adaptive_distance_m=2.0, adaptive_yaw_deg=20.0)
        self.assertTrue(result["online_memory_has_positive"])
        self.assertEqual(result["best_frame"], 12)
        self.assertEqual(result["positive_frames"], 1)
        self.assertTrue(result["uniform_retains_positive"])
        self.assertTrue(result["adaptive_retains_positive"])

    def test_wrap_angle_boundary(self):
        self.assertAlmostEqual(wrap_angle(2.0 * 3.141592653589793), 0.0)


if __name__ == "__main__":
    unittest.main()
