import unittest

from analyze_revisit_xnavdp_capability import analyze_payloads


def _r2_episode(scene, episode, bearing, mixed, base, official, negative=0):
    selection = {
        "scene": scene,
        "episodes": [{
            "episode": episode,
            "r0_first_router_state": (
                {"bearing_deg": bearing} if bearing is not None else None),
        }],
    }
    report = {
        "scene": scene,
        "episode": episode,
        "r0_reached_b": mixed,
        "r0_path_b_m": 2.0,
        "base_point": {"reached_b": base},
        "official_mpc": {
            "reached_b": official,
            "path_b_m": 1.5,
            "official_safety": {
                "control_count": 10 if bearing is not None else 0,
                "negative_velocity_controls": negative,
            },
        },
    }
    return selection, report


class RevisitXNavDPCapabilityAuditTest(unittest.TestCase):
    def test_bins_and_discordances_are_descriptive(self):
        s1, r1 = _r2_episode(
            "scene_a", "episode_0000", -160.0, False, False, True, 10)
        s2, r2 = _r2_episode(
            "scene_b", "episode_0000", -110.0, True, True, False, 9)
        s3, r3 = _r2_episode(
            "scene_c", "episode_0000", None, False, False, False)
        result = analyze_payloads(
            {"scene_groups": [s1, s2, s3]},
            {
                "conditional_b_denominator": 3,
                "episodes": [r1, r2, r3],
            },
        )
        self.assertEqual(
            result["by_first_active_bearing"]["deep_rear"]["official_x_successes"],
            1,
        )
        self.assertEqual(
            result["by_first_active_bearing"]["side_or_side_rear"]["mixed_successes"],
            1,
        )
        self.assertEqual(result["discordant_episode_count"], 2)
        self.assertFalse(
            result["deployment_authorization"]["authorize_bearing_threshold_router"])

    def test_missing_selection_episode_fails(self):
        _, report = _r2_episode(
            "scene_a", "episode_0000", 170.0, True, True, True, 10)
        with self.assertRaises(ValueError):
            analyze_payloads(
                {"scene_groups": []},
                {"conditional_b_denominator": 1, "episodes": [report]},
            )


if __name__ == "__main__":
    unittest.main()
