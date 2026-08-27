import unittest

from raw_novel_forced_anchor_replay import (
    angular_error_deg,
    bearing_from_aux_pose,
    deterministic_counterfactual_anchors,
    eligible_anchor_indices,
    raw_buffer_episode,
)


class RawNovelForcedAnchorReplayTest(unittest.TestCase):
    def test_angle_contract(self):
        self.assertAlmostEqual(bearing_from_aux_pose([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(bearing_from_aux_pose([0.0, 1.0]), 90.0)
        self.assertAlmostEqual(angular_error_deg(179.0, -179.0), 2.0)

    def test_eligible_interval(self):
        self.assertEqual(
            eligible_anchor_indices(
                {"frame_idx": 100, "candidate_ceiling": 99, "candidate_count": 20}
            ),
            list(range(49, 69)),
        )

    def test_sampling_is_identity_bound_and_excludes_factual(self):
        eligible = list(range(25))
        left = deterministic_counterfactual_anchors(
            eligible, 7, count=12, seed=3, identity="scene/episode"
        )
        right = deterministic_counterfactual_anchors(
            eligible, 7, count=12, seed=3, identity="scene/episode"
        )
        self.assertEqual(left, right)
        self.assertEqual(len(left), 12)
        self.assertEqual(len(set(left)), 12)
        self.assertNotIn(7, left)

    def test_raw_buffer_episode_skips_native(self):
        self.assertEqual(
            raw_buffer_episode(
                {
                    "arm_order": [
                        "native",
                        "raw_direct",
                        "raw_fixed_bearing",
                        "geometry_fixed",
                        "certified",
                    ]
                }
            ),
            3,
        )
        self.assertEqual(
            raw_buffer_episode(
                {
                    "arm_order": [
                        "raw_fixed_bearing",
                        "geometry_fixed",
                        "certified",
                        "native",
                        "raw_direct",
                    ]
                }
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
