import unittest

from MemNavData.navdp_goal_switch import (
    navdp_server_base,
    reset_navdp_short_memory,
    should_reset_before_leg,
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
