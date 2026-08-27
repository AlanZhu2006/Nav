import unittest

import numpy as np

from MemNavData.goat_navdp_discrete_adapter import (
    DiscreteAdapterConfig,
    GoatNavAction,
    NavDPAdapterDisposition,
    best_scored_motion_candidate,
    navdp_waypoints_to_goat_decision,
    navdp_waypoints_to_goat_actions,
)


class GoatNavDPDiscreteAdapterTest(unittest.TestCase):
    def test_short_endpoint_is_only_an_arrival_proposal(self):
        decision = navdp_waypoints_to_goat_decision(
            np.array([[0.05, 0.00, 0.0], [0.10, 0.00, 0.0]])
        )
        self.assertEqual(
            decision.disposition, NavDPAdapterDisposition.ARRIVAL_PROPOSAL)
        self.assertEqual(decision.actions, ())
        self.assertTrue(decision.requires_arrival_certificate)
        self.assertLessEqual(decision.max_radius_m, 0.20)
        self.assertNotIn(GoatNavAction.SUBTASK_STOP, decision.actions)

    def test_loop_endpoint_near_origin_is_motion_not_arrival(self):
        decision = navdp_waypoints_to_goat_decision(np.array([
            [0.30, 0.00, 0.0],
            [0.60, 0.00, 0.0],
            [0.30, 0.00, 0.0],
            [0.05, 0.00, 0.0],
        ]))
        self.assertEqual(
            decision.disposition, NavDPAdapterDisposition.MOTION)
        self.assertEqual(
            decision.reason, "motion_chunk_loop_endpoint_truncated")
        self.assertAlmostEqual(decision.endpoint_norm_m, 0.05)
        self.assertAlmostEqual(decision.max_radius_m, 0.60)
        self.assertIn(GoatNavAction.MOVE_FORWARD, decision.actions)

    def test_motion_only_compatibility_wrapper_never_stops(self):
        actions = navdp_waypoints_to_goat_actions(
            np.array([[0.05, 0.00, 0.0], [0.10, 0.00, 0.0]])
        )
        self.assertEqual(actions, ())

    def test_straight_metric_path_moves_forward(self):
        actions = navdp_waypoints_to_goat_actions(
            np.array([[0.25, 0.00, 0.0], [0.50, 0.00, 0.0]])
        )
        self.assertEqual(
            actions,
            (GoatNavAction.MOVE_FORWARD, GoatNavAction.MOVE_FORWARD),
        )

    def test_left_path_uses_goat_thirty_degree_turns(self):
        actions = navdp_waypoints_to_goat_actions(
            np.array([[0.00, 0.25, 0.0], [0.00, 0.50, 0.0]])
        )
        self.assertEqual(
            actions[:4],
            (
                GoatNavAction.TURN_LEFT,
                GoatNavAction.TURN_LEFT,
                GoatNavAction.TURN_LEFT,
                GoatNavAction.MOVE_FORWARD,
            ),
        )

    def test_metric_lookahead_preserves_released_controller_scale(self):
        trajectory = np.array([
            [0.10, 0.00, 0.0],
            [0.20, 0.00, 0.0],
            [0.30, 0.00, 0.0],
            [0.40, 0.00, 0.0],
            [0.50, 0.20, 0.0],
            [0.60, 0.40, 0.0],
            [0.70, 0.60, 0.0],
        ])
        point_lookahead = navdp_waypoints_to_goat_decision(
            trajectory, DiscreteAdapterConfig(execution_horizon=1))
        metric_lookahead = navdp_waypoints_to_goat_decision(
            trajectory,
            DiscreteAdapterConfig(
                lookahead_distance_m=0.70, execution_horizon=1),
        )
        self.assertEqual(
            point_lookahead.actions, (GoatNavAction.MOVE_FORWARD,))
        self.assertEqual(
            metric_lookahead.actions, (GoatNavAction.TURN_LEFT,))

    def test_action_chunk_is_bounded(self):
        config = DiscreteAdapterConfig(execution_horizon=3)
        actions = navdp_waypoints_to_goat_actions(
            np.array([[0.25 * index, 0.0, 0.0] for index in range(1, 20)]),
            config,
        )
        self.assertEqual(len(actions), 3)

    def test_nonfinite_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            navdp_waypoints_to_goat_actions(
                np.array([[0.0, np.nan, 0.0]])
            )

    def test_same_batch_fallback_uses_highest_scored_motion(self):
        zero = np.zeros((2, 3), dtype=float)
        lower_motion = np.array(
            [[0.25, 0.0, 0.0], [0.50, 0.0, 0.0]], dtype=float)
        higher_motion = np.array(
            [[0.30, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=float)
        decision = best_scored_motion_candidate(
            np.stack([zero, lower_motion, higher_motion]),
            np.array([0.9, 0.1, 0.5]),
        )
        self.assertTrue(decision.is_motion)
        self.assertEqual(decision.candidate_index, 2)
        self.assertNotIn(GoatNavAction.SUBTASK_STOP, decision.actions)

    def test_same_batch_fallback_fails_closed_when_all_are_zero(self):
        decision = best_scored_motion_candidate(
            np.zeros((3, 2, 3), dtype=float),
            np.array([0.9, 0.8, 0.7]),
        )
        self.assertEqual(
            decision.disposition, NavDPAdapterDisposition.CONVERSION_STALLED)
        self.assertEqual(decision.actions, ())


if __name__ == "__main__":
    unittest.main()
