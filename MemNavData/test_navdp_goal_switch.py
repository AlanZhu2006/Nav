import unittest

from MemNavData.navdp_goal_switch import (
    navdp_server_base,
    normalize_navdp_candidate_scores,
    normalize_navdp_trajectory_candidates,
    reset_navdp_short_memory,
    should_reset_before_leg,
    trajectory_selector_for_leg,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_checked = False

    def raise_for_status(self):
        self.status_checked = True

    def json(self):
        return self.payload


class NavDPGoalSwitchTest(unittest.TestCase):
    def test_before_b_is_a_single_transition_ablation(self):
        self.assertFalse(should_reset_before_leg("before_b", 0))
        self.assertTrue(should_reset_before_leg("before_b", 1))
        self.assertFalse(should_reset_before_leg("before_b", 2))
        self.assertTrue(should_reset_before_leg("every_goal", 1))
        self.assertTrue(should_reset_before_leg("every_goal", 2))
        self.assertFalse(should_reset_before_leg("carry", 1))

    def test_oracle_selector_can_be_isolated_to_novel_b(self):
        self.assertEqual(
            trajectory_selector_for_leg("oracle_geodesic", "leg_b", 0),
            "server",
        )
        self.assertEqual(
            trajectory_selector_for_leg("oracle_geodesic", "leg_b", 1),
            "oracle_geodesic",
        )
        self.assertEqual(
            trajectory_selector_for_leg("oracle_geodesic", "leg_b", 2),
            "server",
        )
        self.assertEqual(
            trajectory_selector_for_leg("oracle_geodesic", "all", 0),
            "oracle_geodesic",
        )
        self.assertEqual(
            trajectory_selector_for_leg("server", "leg_b", None),
            "server",
        )

    def test_leg_scoped_selector_requires_leg_identity(self):
        with self.assertRaisesRegex(ValueError, "requires leg_index"):
            trajectory_selector_for_leg("oracle_geodesic", "leg_b", None)

    def test_navdp_singleton_candidate_batch_is_normalized(self):
        candidates = normalize_navdp_trajectory_candidates(
            [[[[0.0, 0.0, 0.0]] * 24] * 16]
        )
        self.assertEqual(candidates.shape, (16, 24, 3))
        scores = normalize_navdp_candidate_scores([[0.0] * 16], 16)
        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape, (16,))

    def test_non_singleton_candidate_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "all_trajectory shape"):
            normalize_navdp_trajectory_candidates(
                [[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]]
            )
        self.assertIsNone(normalize_navdp_candidate_scores([[0.0] * 8], 16))

    def test_hybrid_targets_navdp_not_memnav(self):
        self.assertEqual(
            navdp_server_base("hybrid_pose", "http://memnav", "http://navdp"),
            "http://navdp",
        )
        calls = []
        response = FakeResponse({"algo": "navdp"})

        def post(url, json):
            calls.append((url, json))
            return response

        payload = reset_navdp_short_memory(
            post,
            "hybrid_pose",
            "http://memnav",
            "http://navdp",
        )
        self.assertEqual(payload, {"algo": "navdp"})
        self.assertEqual(
            calls,
            [("http://navdp/navigator_reset_env", {"env_id": 0})],
        )
        self.assertTrue(response.status_checked)

    def test_standalone_memnav_reset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "standalone MemNav"):
            navdp_server_base("memnav", "http://memnav", None)

    def test_wrong_server_response_fails_closed(self):
        def post(_url, json):
            self.assertEqual(json, {"env_id": 3})
            return FakeResponse({"algo": "memnav"})

        with self.assertRaisesRegex(RuntimeError, "non-NavDP"):
            reset_navdp_short_memory(
                post,
                "navdp",
                "http://navdp",
                None,
                env_id=3,
            )


if __name__ == "__main__":
    unittest.main()
