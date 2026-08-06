from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from MemNavData.diag_lingbot_goal_loop_closure import (
    CandidateSeed,
    _HABITAT_TO_DATA_ROTATION,
    lingbot_relative_prediction,
    navdp_ground_truth_relative,
    raw_rgb_dir,
    relative_pose_errors,
    select_deployment_seeds,
    validate_scene_role,
)


class LingBotGoalLoopClosureTest(unittest.TestCase):
    @staticmethod
    def teacher_row(session, frame, dino, covis):
        scene, episode, _suffix = session.split("/", 2)
        root = Path("/dataset") / scene / episode
        return {
            "session_id": session,
            "scene": scene,
            "episode": episode,
            "kind": "cross_episode_train",
            "query_path": str(root.parent / "episode_query" / "videos"
                              / "chunk-000" / "observation.images.rgb"
                              / "16.jpg"),
            "candidate_path": str(root / "videos" / "chunk-000"
                                  / "observation.images.rgb"
                                  / f"{frame}.jpg"),
            "candidate_frame": frame,
            "dino_cosine": dino,
            "teacher_covis": covis,
        }

    def test_deployment_selection_preserves_set_labels(self):
        matched = "scene_a/episode_0000/matched"
        no_match = "scene_b/episode_0000/no_match"
        ambiguous = "scene_c/episode_0000/ambiguous"
        frame = pd.DataFrame([
            self.teacher_row(matched, 8, 0.9, 0.8),
            self.teacher_row(matched, 20, 0.8, 0.05),
            self.teacher_row(no_match, 8, 0.9, 0.05),
            self.teacher_row(no_match, 20, 0.8, 0.10),
            self.teacher_row(ambiguous, 8, 0.9, 0.30),
            self.teacher_row(ambiguous, 20, 0.8, 0.15),
        ])
        seeds = select_deployment_seeds(
            frame, kind="cross_episode_train", sessions=(), max_sessions=0,
            top_k=2, minimum_gap=4, positive_threshold=0.5,
            negative_threshold=0.1, minimum_anchor=8)
        self.assertEqual(len(seeds), 6)
        by_session = {}
        for seed in seeds:
            by_session.setdefault(seed.session_id, []).append(seed)
            self.assertEqual(seed.candidate_path.name,
                             f"{seed.candidate_frame}.jpg")
        self.assertTrue(by_session[matched][0].session_has_positive)
        self.assertFalse(by_session[matched][0].session_is_strict_no_match)
        self.assertFalse(by_session[no_match][0].session_has_positive)
        self.assertTrue(by_session[no_match][0].session_is_strict_no_match)
        self.assertFalse(by_session[ambiguous][0].session_has_positive)
        self.assertFalse(
            by_session[ambiguous][0].session_is_strict_no_match)
        self.assertIn(-1, {seed.label for seed in by_session[ambiguous]})
        validate_scene_role(
            seeds, {"development": ["scene_a", "scene_b", "scene_c"]},
            "development")
        with self.assertRaisesRegex(RuntimeError, "outside development"):
            validate_scene_role(
                seeds, {"development": ["scene_a", "scene_b"]},
                "development")

    def test_cross_episode_replay_uses_candidate_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query" / "videos" / "chunk-000" / "rgb"
            candidate = (root / "candidate" / "videos" / "chunk-000"
                         / "observation.images.rgb")
            query.mkdir(parents=True)
            candidate.mkdir(parents=True)
            seed = CandidateSeed(
                session_id="scene/episode/cross", scene="scene",
                episode="episode", kind="cross_episode_train",
                query_path=query / "16.jpg",
                candidate_path=candidate / "42.jpg",
                candidate_frame=42, dino_cosine=0.8,
                teacher_covis=0.7, label=1,
                session_has_positive=True,
                session_is_strict_no_match=False,
                session_max_covis=0.7)
            self.assertEqual(raw_rgb_dir(seed), candidate.resolve())

    def test_navdp_mount_fix_preserves_forward_translation(self):
        candidate = np.eye(4)
        candidate[:3, :3] = _HABITAT_TO_DATA_ROTATION
        query = candidate.copy()
        query[:3, 3] = [0.0, 1.0, 0.0]
        mount = np.eye(4)
        mount[:3, :3] = _HABITAT_TO_DATA_ROTATION
        target_xy, target_rotation = navdp_ground_truth_relative(
            candidate, query, mount)
        np.testing.assert_allclose(target_xy, [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(target_rotation, np.eye(3), atol=1e-12)

    def test_zero_pose_error_under_correct_axis_conversion(self):
        anchor_pose9 = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        goal_pose9 = anchor_pose9.copy()
        goal_pose9[2] = 1.0
        predicted_xy, predicted_rotation = lingbot_relative_prediction(
            anchor_pose9, goal_pose9, metric_scale=1.0)
        errors = relative_pose_errors(
            predicted_xy, np.array([1.0, 0.0]),
            predicted_rotation, np.eye(3))
        np.testing.assert_allclose(predicted_xy, [1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(errors["relative_position_error_m"], 0.0)
        self.assertAlmostEqual(
            errors["relative_position_direction_error_deg"], 0.0)
        self.assertAlmostEqual(errors["relative_rotation_error_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
