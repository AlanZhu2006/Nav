import copy
import unittest

from MemNavData.multigoal_benchmark_contract import (
    DOUBLE_REVISIT_PROTOCOL,
    DoubleRevisitObservation,
    ROLE_SYMMETRIC_PROTOCOL,
    RoleSymmetryObservation,
    validate_double_revisit_contract,
    validate_role_symmetric_contract,
)


def valid_metadata():
    return {
        "gen_protocol": ROLE_SYMMETRIC_PROTOCOL,
        "n_legs": 3,
        "role_sequence": ["initial_imagegoal", "novel", "revisit"],
        "initial_yaw_mode": "uniform",
        "initial_goal_pose_source": "expert_arrival_frame_exact",
        "initial_start_pose_source": "first_stored_expert_frame_exact",
        "novel_b_goal_yaw": "expert_arrival_heading",
        "novel_b_goal_image_source": "expert_arrival_frame_exact",
        "role_pairing": "same_episode_geodesic",
        "role_distance_match_tolerance_m": 0.5,
        "role_distance_error_m": 0.3,
        "initial_distance_band_m": [3.0, 9.0],
        "novel_distance_band_m": [3.0, 9.0],
        "geo_startA": 5.0,
        "geo_AB": 5.3,
        "n_frames": 12,
        "switches": [5, 8],
        "anchor_margin": 2,
        "novel_covis": 0.1,
        "covis_band": [0.2, 1.0],
        "covis_pos_lo": 0.1,
        "goals": [
            {
                "name": "B",
                "kind": "novel",
                "covis_argmax": -1,
                "covis_curve": [0.0, 0.01, 0.0, 0.02, 0.0],
            },
            {
                "name": "C",
                "kind": "revisit",
                "covis": 0.6,
                "covis_argmax": 2,
                "covis_curve": [0.0, 0.1, 0.6, 0.3, 0.2, 0.05, 0.04, 0.03],
                "anchor_frame_limit": 5,
                "non_anchor_max_covis": 0.05,
            },
        ],
    }


def valid_observation():
    return RoleSymmetryObservation(
        geo_a_m=5.0,
        geo_b_m=5.3,
        initial_pose_error_m=1e-5,
        a_terminal_pose_error_m=1e-5,
        b_terminal_pose_error_m=1e-5,
        b_terminal_yaw_error_deg=1e-4,
        goal_b_matches_terminal_rgb=True,
    )


def valid_double_revisit_metadata():
    return {
        "gen_protocol": DOUBLE_REVISIT_PROTOCOL,
        "n_legs": 3,
        "role_sequence": ["initial_imagegoal", "revisit", "revisit"],
        "initial_yaw_mode": "uniform",
        "initial_goal_pose_source": "expert_arrival_frame_exact",
        "initial_start_pose_source": "first_stored_expert_frame_exact",
        "double_revisit_goal_image_source": "metadata_pose_render",
        "double_revisit_distance_min_m": {"B": 2.0, "C": 2.0},
        "double_revisit_min_anchor_gap": 2,
        "double_revisit_anchor_gap": 2,
        "initial_distance_band_m": [3.0, 9.0],
        "geo_startA": 5.0,
        "geo_AB": 3.0,
        "geo_BC": 3.5,
        "n_frames": 12,
        "switches": [5, 8],
        "anchor_margin": 2,
        "covis_band": [0.2, 1.0],
        "covis_pos_lo": 0.1,
        "goals": [
            {
                "name": "B", "kind": "revisit", "covis": 0.6,
                "covis_argmax": 2,
                "covis_curve": [0.0, 0.1, 0.6, 0.3, 0.2],
                "anchor_frame_limit": 5, "non_anchor_max_covis": 0.0,
            },
            {
                "name": "C", "kind": "revisit", "covis": 0.7,
                "covis_argmax": 4,
                "covis_curve": [0.0, 0.1, 0.2, 0.3, 0.7, 0.05, 0.04, 0.03],
                "anchor_frame_limit": 5, "non_anchor_max_covis": 0.05,
            },
        ],
    }


def valid_double_revisit_observation():
    return DoubleRevisitObservation(
        geo_a_m=5.0,
        geo_b_m=3.0,
        geo_c_m=3.5,
        initial_pose_error_m=1e-5,
        a_terminal_pose_error_m=1e-5,
        goal_b_matches_render=True,
        goal_c_matches_render=True,
    )


class MultigoalBenchmarkContractTest(unittest.TestCase):
    def test_double_revisit_episode_passes(self):
        report = validate_double_revisit_contract(
            valid_double_revisit_metadata(),
            valid_double_revisit_observation(),
        )
        self.assertTrue(report["ok"], report["issues"])

    def test_double_revisit_anchors_must_be_distinct(self):
        metadata = valid_double_revisit_metadata()
        metadata["goals"][1]["covis_argmax"] = 3
        metadata["goals"][1]["covis"] = 0.3
        metadata["double_revisit_anchor_gap"] = 1
        report = validate_double_revisit_contract(
            metadata, valid_double_revisit_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any(
            "sufficiently distinct" in issue for issue in report["issues"]))

    def test_double_revisit_second_goal_rejects_recent_leg_b_support(self):
        metadata = valid_double_revisit_metadata()
        metadata["goals"][1]["covis_curve"][6] = 0.4
        metadata["goals"][1]["non_anchor_max_covis"] = 0.4
        report = validate_double_revisit_contract(
            metadata, valid_double_revisit_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any(
            "hard negative" in issue for issue in report["issues"]))

    def test_role_symmetric_episode_passes(self):
        report = validate_role_symmetric_contract(
            valid_metadata(), valid_observation())
        self.assertTrue(report["ok"], report["issues"])

    def test_legacy_protocol_and_path_aligned_start_fail(self):
        metadata = valid_metadata()
        metadata["gen_protocol"] = "multileg_v2_symmetric_20260807"
        metadata["initial_yaw_mode"] = "path_aligned"
        report = validate_role_symmetric_contract(metadata, valid_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any("gen_protocol" in issue for issue in report["issues"]))
        self.assertTrue(any("uniformly" in issue for issue in report["issues"]))

    def test_unbounded_second_novel_fails(self):
        metadata = valid_metadata()
        metadata["novel_distance_band_m"] = [3.0, 100.0]
        observation = copy.copy(valid_observation())
        report = validate_role_symmetric_contract(metadata, observation)
        self.assertFalse(report["ok"])
        self.assertTrue(any("bands differ" in issue for issue in report["issues"]))

    def test_goal_pose_and_pixels_must_match_terminal_frame(self):
        observation = RoleSymmetryObservation(
            geo_a_m=5.0,
            geo_b_m=5.3,
            initial_pose_error_m=1e-5,
            a_terminal_pose_error_m=0.10,
            b_terminal_pose_error_m=0.20,
            b_terminal_yaw_error_deg=15.0,
            goal_b_matches_terminal_rgb=False,
        )
        report = validate_role_symmetric_contract(
            valid_metadata(), observation)
        self.assertFalse(report["ok"])
        self.assertGreaterEqual(len(report["issues"]), 4)

    def test_initial_pose_must_match_first_stored_frame_contract(self):
        metadata = valid_metadata()
        metadata["initial_start_pose_source"] = "legacy_unrecorded_pre_step"
        report = validate_role_symmetric_contract(metadata, valid_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any("initial pose" in issue for issue in report["issues"]))

    def test_non_finite_values_fail_closed(self):
        metadata = valid_metadata()
        metadata["geo_AB"] = float("nan")
        observation = RoleSymmetryObservation(
            geo_a_m=5.0,
            geo_b_m=float("nan"),
            initial_pose_error_m=1e-5,
            a_terminal_pose_error_m=float("nan"),
            b_terminal_pose_error_m=1e-5,
            b_terminal_yaw_error_deg=1e-4,
            goal_b_matches_terminal_rgb=True,
        )
        report = validate_role_symmetric_contract(metadata, observation)
        self.assertFalse(report["ok"])
        self.assertTrue(any("outside" in issue for issue in report["issues"]))
        self.assertTrue(any("non-finite" in issue for issue in report["issues"]))

    def test_non_finite_distance_band_fails_closed(self):
        metadata = valid_metadata()
        metadata["novel_distance_band_m"] = [3.0, float("inf")]
        report = validate_role_symmetric_contract(metadata, valid_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any("non-finite" in issue for issue in report["issues"]))

    def test_within_episode_role_distance_must_match(self):
        metadata = valid_metadata()
        metadata["geo_AB"] = 6.0
        metadata["role_distance_error_m"] = 1.0
        observation = copy.copy(valid_observation())
        object.__setattr__(observation, "geo_b_m", 6.0)
        report = validate_role_symmetric_contract(metadata, observation)
        self.assertFalse(report["ok"])
        self.assertTrue(any(
            "paired-role tolerance" in issue for issue in report["issues"]))

    def test_revisit_c_must_anchor_in_leg_a(self):
        metadata = valid_metadata()
        metadata["goals"][1]["covis_argmax"] = 6
        report = validate_role_symmetric_contract(metadata, valid_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any(
            "valid leg-A" in issue for issue in report["issues"]))

    def test_leg_b_must_be_hard_negative_for_revisit_c(self):
        metadata = valid_metadata()
        metadata["goals"][1]["covis_curve"][6] = 0.4
        metadata["goals"][1]["non_anchor_max_covis"] = 0.4
        report = validate_role_symmetric_contract(metadata, valid_observation())
        self.assertFalse(report["ok"])
        self.assertTrue(any(
            "hard-negative" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
