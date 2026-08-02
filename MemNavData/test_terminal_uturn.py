import math
import unittest

import numpy as np

from MemNavData.terminal_uturn import (
    TerminalManeuverExecutor,
    plan_staged_terminal_maneuver,
    plan_terminal_maneuver,
    relative_xy_to_world,
    wrap_angle,
)
from MemNavData.summarize_terminal_uturn import summarize
from MemNavData.diag_oracle_retrieval_firsthop import (
    angle_error_deg,
    lookahead_point,
    world_delta_to_local,
)
from MemNavData.visual_yaw_refinement import (
    VisualYawEstimate,
    visual_yaw_action_decision,
    yaw_correction_from_rotation,
)
from NavDP.baselines.memnav.pose_alignment import lingbot_relative_yaw


class _OpenPathfinder:
    def snap_point(self, point):
        return np.asarray(point, dtype=np.float64)

    def is_navigable(self, _point):
        return True

    def distance_to_closest_obstacle(self, _point):
        return 10.0


class _BlockedPathfinder(_OpenPathfinder):
    def is_navigable(self, _point):
        return False


class _NarrowAtGoalPathfinder(_OpenPathfinder):
    """Straight centreline is open; lateral motion is blocked near the goal."""

    def is_navigable(self, point):
        x, _y, z = np.asarray(point)
        return not (abs(x) > 0.20 and z < 1.20)


class TerminalUTurnTest(unittest.TestCase):
    def test_firsthop_direction_metric_uses_navdp_axes(self):
        np.testing.assert_allclose(
            world_delta_to_local([0.0, -1.0], 0.0), [1.0, 0.0])
        np.testing.assert_allclose(
            world_delta_to_local([-1.0, 0.0], 0.0), [0.0, 1.0])
        hop = lookahead_point([[0.2, 0.0], [0.8, 0.0]], 0.7)
        np.testing.assert_allclose(hop, [0.8, 0.0])
        self.assertAlmostEqual(angle_error_deg([1.0, 0.0], [0.0, 1.0]), 90.0)

    @staticmethod
    def _visual_estimate(correction_deg, reliable=True, reason="ok"):
        return VisualYawEstimate(
            yaw_correction_rad=math.radians(correction_deg),
            bearing_correction_rad=math.radians(correction_deg),
            matches=40,
            inliers=30,
            inlier_ratio=0.75,
            off_axis_deg=1.0,
            bearing_mad_deg=1.0,
            consensus_error_deg=0.5,
            reliable=reliable,
            reason=reason,
        )

    def test_visual_yaw_control_gate_fails_closed(self):
        apply, reason = visual_yaw_action_decision(
            self._visual_estimate(30.0))
        self.assertTrue(apply)
        self.assertEqual(reason, "ok")

        apply, reason = visual_yaw_action_decision(
            self._visual_estimate(7.9))
        self.assertFalse(apply)
        self.assertIn("deadband", reason)

        apply, reason = visual_yaw_action_decision(
            self._visual_estimate(46.0))
        self.assertFalse(apply)
        self.assertIn("bound", reason)

        apply, reason = visual_yaw_action_decision(
            self._visual_estimate(20.0, reliable=False, reason="disagreement"))
        self.assertFalse(apply)
        self.assertIn("rejected", reason)

    def test_visual_rotation_yaw_sign(self):
        for expected in (-1.1, -0.2, 0.0, 0.4, 1.3):
            c, s = math.cos(expected), math.sin(expected)
            rotation = np.array([
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ])
            self.assertLess(abs(wrap_angle(
                yaw_correction_from_rotation(rotation) - expected)), 1e-12)

    def test_lingbot_rotation_basis_recovers_habitat_yaw(self):
        basis = np.diag([-1.0, -1.0, 1.0])
        for expected in (-math.pi + 0.01, -1.2, 0.0, 0.7, math.pi - 0.01):
            cosine, sine = math.cos(expected), math.sin(expected)
            corrected = np.array([
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ])
            raw_lingbot = basis.T @ corrected @ basis
            self.assertLess(
                abs(wrap_angle(lingbot_relative_yaw(raw_lingbot) - expected)),
                1e-9,
            )

    def test_navdp_local_xy_to_habitat_world(self):
        np.testing.assert_allclose(
            relative_xy_to_world([1.0, 0.0], [2.0, 3.0], 0.0),
            [2.0, 2.0],
        )
        np.testing.assert_allclose(
            relative_xy_to_world([0.0, 1.0], [2.0, 3.0], 0.0),
            [1.0, 3.0],
        )

    def test_same_position_reversal_is_forward_only_teardrop(self):
        path = plan_terminal_maneuver(
            current_xz=[0.0, 0.0],
            current_yaw=math.pi,
            goal_xz=[0.0, 0.0],
            goal_yaw=0.0,
            radius=0.40,
            pathfinder=_OpenPathfinder(),
        )
        self.assertIsNotNone(path)
        self.assertIn(path.mode, ("RLR", "LRL"))
        self.assertAlmostEqual(path.length_m, 0.4 * 7.0 * math.pi / 3.0, places=6)
        np.testing.assert_allclose(path.points_xz[0], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(path.points_xz[-1], [0.0, 0.0], atol=1e-9)
        self.assertLess(abs(wrap_angle(path.yaws[-1])), 1e-9)

        translation = np.linalg.norm(np.diff(path.points_xz, axis=0), axis=1)
        yaw_change = np.abs([wrap_angle(x) for x in np.diff(path.yaws)])
        self.assertTrue(np.all(translation[yaw_change > 1e-7] > 1e-7))
        self.assertLessEqual(yaw_change.max(), math.radians(4.5) + 1e-9)

    def test_summary_uses_attempted_terminal_denominator(self):
        common = {
            "spl_B": "0.0", "spl_B_with_terminal": "0.0",
            "gate_B_mean": "0.2", "terminal_path_m": "",
            "pre_turn_yaw_err_deg": "", "post_turn_yaw_err_deg": "",
            "pre_turn_goal_cos": "", "post_turn_goal_cos": "",
        }
        reached = dict(
            common,
            reached_B="1.0", spl_B="0.8", spl_B_with_terminal="0.4",
            terminal_attempted="True", terminal_completed="True",
            terminal_success="True", terminal_path_type="RLR",
            terminal_path_m="2.8", pre_turn_yaw_err_deg="170",
            post_turn_yaw_err_deg="3", pre_turn_goal_cos="0.8",
            post_turn_goal_cos="0.95", terminal_failure="",
        )
        navigation_failure = dict(
            common,
            reached_B="0.0", terminal_attempted="False",
            terminal_completed="False", terminal_success="False",
            terminal_path_type="", terminal_failure="",
        )
        summary = summarize([reached, navigation_failure])
        self.assertEqual(summary["navigation_sr"], 0.5)
        self.assertEqual(summary["terminal_completion_given_attempt"], 1.0)
        self.assertEqual(summary["terminal_pose_success_given_attempt"], 1.0)
        self.assertEqual(summary["terminal_pose_success_overall"], 0.5)

    def test_summary_final_pose_does_not_require_completed_path(self):
        already_aligned = {
            "reached_B": "1.0", "spl_B": "0.8",
            "spl_B_with_terminal": "0.8", "gate_B_mean": "0.2",
            "terminal_attempted": "True", "terminal_completed": "False",
            "terminal_success": "False", "terminal_path_type": "",
            "terminal_failure": "no collision-free forward-only path",
            "terminal_final_goal_dist_m": "0.95",
            "post_turn_yaw_err_deg": "6.0",
            "terminal_path_m": "", "pre_turn_yaw_err_deg": "6.0",
            "pre_turn_goal_cos": "0.9", "post_turn_goal_cos": "",
        }
        summary = summarize([already_aligned])
        self.assertEqual(summary["terminal_completion_given_attempt"], 0.0)
        self.assertEqual(summary["terminal_pose_success_given_attempt"], 1.0)

    def test_general_pose_connections_end_at_requested_pose(self):
        pathfinder = _OpenPathfinder()
        cases = [
            ([0.0, 0.0], 0.0, [1.2, -0.4], 0.7),
            ([2.0, -1.0], -1.4, [-0.3, 0.8], 2.2),
            ([-0.5, 0.2], math.pi - 0.01, [0.7, 1.4], -math.pi + 0.03),
        ]
        for current_xz, current_yaw, goal_xz, goal_yaw in cases:
            with self.subTest(current_xz=current_xz, goal_xz=goal_xz):
                path = plan_terminal_maneuver(
                    current_xz, current_yaw, goal_xz, goal_yaw,
                    pathfinder=pathfinder,
                )
                self.assertIsNotNone(path)
                np.testing.assert_allclose(path.points_xz[0], current_xz, atol=1e-8)
                np.testing.assert_allclose(path.points_xz[-1], goal_xz, atol=1e-8)
                self.assertLess(abs(wrap_angle(path.yaws[0] - current_yaw)), 1e-8)
                self.assertLess(abs(wrap_angle(path.yaws[-1] - goal_yaw)), 1e-8)

                translation = np.linalg.norm(np.diff(path.points_xz, axis=0), axis=1)
                yaw_change = np.abs([wrap_angle(x) for x in np.diff(path.yaws)])
                self.assertTrue(np.all(translation[yaw_change > 1e-7] > 1e-7))

    def test_executor_reaches_goal_without_in_place_rotation(self):
        pathfinder = _OpenPathfinder()
        path = plan_terminal_maneuver(
            [0.0, 0.0], math.pi, [0.0, 0.0], 0.0,
            pathfinder=pathfinder,
        )
        executor = TerminalManeuverExecutor(path)
        position = np.array([0.0, 0.0, 0.0])
        yaw = math.pi
        travelled = 0.0
        while not executor.done and not executor.failed:
            previous_yaw = yaw
            position, yaw, step = executor.step(position, yaw, pathfinder, 0.0)
            if abs(wrap_angle(yaw - previous_yaw)) > 1e-7:
                self.assertGreater(step, 1e-7)
            travelled += step

        self.assertFalse(executor.failed)
        self.assertTrue(executor.done)
        np.testing.assert_allclose(position[[0, 2]], [0.0, 0.0], atol=1e-8)
        self.assertLess(abs(wrap_angle(yaw)), 1e-8)
        # Executor sums straight chords between arc samples, so it is slightly
        # shorter than the analytic arc length.
        self.assertLess(abs(travelled - path.length_m), 5e-4)

    def test_no_feasible_path_fails_closed(self):
        path = plan_terminal_maneuver(
            [0.0, 0.0], math.pi, [0.0, 0.0], 0.0,
            pathfinder=_BlockedPathfinder(),
        )
        self.assertIsNone(path)

    def test_staged_turn_uses_free_space_then_returns_to_current_pose(self):
        pathfinder = _NarrowAtGoalPathfinder()
        direct = plan_terminal_maneuver(
            [0.0, 0.0], math.pi, [0.0, 0.0], 0.0,
            pathfinder=pathfinder,
        )
        self.assertIsNone(direct)
        staged = plan_staged_terminal_maneuver(
            [0.0, 0.0], math.pi, 0.0,
            stage_max_m=2.5,
            pathfinder=pathfinder,
        )
        self.assertIsNotNone(staged)
        self.assertTrue(staged.mode.startswith("STAGED-"))
        np.testing.assert_allclose(staged.points_xz[0], [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(staged.points_xz[-1], [0.0, 0.0], atol=1e-8)
        self.assertLess(abs(wrap_angle(staged.yaws[-1])), 1e-8)

        translation = np.linalg.norm(np.diff(staged.points_xz, axis=0), axis=1)
        yaw_change = np.abs([wrap_angle(x) for x in np.diff(staged.yaws)])
        self.assertTrue(np.all(translation[yaw_change > 1e-7] > 1e-7))


if __name__ == "__main__":
    unittest.main()
