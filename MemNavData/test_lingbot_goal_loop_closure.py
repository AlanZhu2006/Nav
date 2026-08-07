from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from MemNavData.diag_lingbot_goal_loop_closure import (
    BoundedEpisodeCache,
    CandidateSeed,
    CollectionCheckpoint,
    _HABITAT_TO_DATA_ROTATION,
    lingbot_relative_prediction,
    navdp_ground_truth_relative,
    raw_rgb_dir,
    relative_pose_errors,
    seed_manifest_sha256,
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

    def test_selection_and_resume_hash_preserve_explicit_causal_sample(self):
        session = "scene_a/episode_0000/matched"
        first = self.teacher_row(session, 8, 0.9, 0.8)
        first["causal_manifest_sample_id"] = "train/scene_a/b_t0/factual"
        frame = pd.DataFrame([first])
        seeds = select_deployment_seeds(
            frame, kind="cross_episode_train", sessions=(), max_sessions=0,
            top_k=1, minimum_gap=4, positive_threshold=0.5,
            negative_threshold=0.1, minimum_anchor=8)
        self.assertEqual(
            seeds[0].causal_manifest_sample_id,
            "train/scene_a/b_t0/factual")
        original = seed_manifest_sha256(seeds)
        changed = [CandidateSeed(
            **{**seeds[0].__dict__,
               "causal_manifest_sample_id": "train/scene_a/b_mid/factual"})]
        self.assertNotEqual(original, seed_manifest_sha256(changed))

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

    def test_episode_cache_is_bounded_and_evicts_lru(self):
        evicted = []
        cache = BoundedEpisodeCache(
            2, on_evict=lambda key, value: evicted.append((key, value)))
        first = cache.get_or_load(("scene", "ep0"), lambda: {"value": 0})
        cache.get_or_load(("scene", "ep1"), lambda: {"value": 1})
        self.assertIs(
            cache.get_or_load(
                ("scene", "ep0"), lambda: self.fail("unexpected reload")),
            first,
        )
        cache.get_or_load(("scene", "ep2"), lambda: {"value": 2})
        self.assertEqual(evicted, [(('scene', 'ep1'), {"value": 1})])
        self.assertEqual(len(cache), 2)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(
            [item[0] for item in evicted],
            [("scene", "ep1"), ("scene", "ep0"), ("scene", "ep2")],
        )

    def test_collection_checkpoint_is_session_atomic_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.sqlite3"
            signature = {"seed_manifest_sha256": "fixed", "total": 3}
            checkpoint = CollectionCheckpoint(
                path, signature, resume=False)
            checkpoint.save_session(
                "session_a",
                [1, 2],
                [(1, {"session_id": "session_a", "value": 10})],
            )
            checkpoint.close()

            with self.assertRaises(FileExistsError):
                CollectionCheckpoint(path, signature, resume=False)
            with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                CollectionCheckpoint(
                    path, {"seed_manifest_sha256": "different"},
                    resume=True)

            resumed = CollectionCheckpoint(path, signature, resume=True)
            self.assertEqual(resumed.completed_sessions(), {"session_a"})
            self.assertEqual(resumed.last_completed_session(), "session_a")
            self.assertEqual(
                resumed.rows(),
                [{"session_id": "session_a", "value": 10}],
            )
            with self.assertRaisesRegex(
                    RuntimeError, "already checkpointed"):
                resumed.save_session(
                    "session_corrupt", [1],
                    [(1, {"session_id": "session_corrupt", "value": 99})],
                )
            self.assertNotIn(
                "session_corrupt", resumed.completed_sessions())
            self.assertEqual(len(resumed.rows()), 1)
            # A complete session may legitimately have no row when all of its
            # anchors fall outside the valid cache range.
            resumed.save_session("session_b", [3], [])
            progress = resumed.progress(
                total_sessions=2, total_seeds=3,
                status="collecting", last_session="session_b")
            self.assertEqual(progress["completed_sessions"], 2)
            self.assertEqual(progress["completed_seeds"], 3)
            self.assertEqual(progress["saved_rows"], 1)
            self.assertEqual(progress["signature_sha256"],
                             "2971febdfceed9566549f83aa0600351"
                             "ec0b8f2c3db7b9da128b643254fe4aea")
            resumed.close()

    def test_slurm_collection_exposes_bounded_cache_and_resume(self):
        script = (Path(__file__).resolve().parent
                  / "slurm_lingbot_native_localizer.sbatch").read_text()
        self.assertIn("--max-cached-episodes", script)
        self.assertIn("RESUME_RUN_ROOT", script)
        self.assertIn("--resume", script)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", script)


if __name__ == "__main__":
    unittest.main()
