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
    feature_episode_root,
    lingbot_relative_prediction,
    navdp_ground_truth_relative,
    raw_rgb_dir,
    relative_pose_errors,
    resolve_routed_feature_cache_pairs,
    seed_manifest_sha256,
    select_deployment_seeds,
    select_train_augmented_seeds,
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

    def test_train_augmentation_exposes_missed_positive_without_replacing_topk(self):
        matched = "scene_a/episode_0000/matched"
        no_match = "scene_b/episode_0000/no_match"
        frame = pd.DataFrame([
            self.teacher_row(matched, 8, 0.99, 0.05),
            self.teacher_row(matched, 20, 0.98, 0.30),
            self.teacher_row(matched, 32, 0.70, 0.80),
            self.teacher_row(no_match, 8, 0.99, 0.05),
            self.teacher_row(no_match, 20, 0.90, 0.10),
            self.teacher_row(no_match, 32, 0.80, 0.15),
        ])
        seeds = select_train_augmented_seeds(
            frame, kind="cross_episode_train", sessions=(), max_sessions=0,
            top_k=2, minimum_gap=4, positive_threshold=0.5,
            negative_threshold=0.2, minimum_anchor=8)
        by_session = {}
        for seed in seeds:
            by_session.setdefault(seed.session_id, []).append(seed)
        self.assertEqual(
            [seed.candidate_frame for seed in by_session[matched]],
            [8, 20, 32])
        self.assertEqual(
            {seed.label for seed in by_session[matched]}, {0, -1, 1})
        self.assertEqual(
            [seed.selection_origin for seed in by_session[matched]],
            ["deployment_topk", "deployment_topk",
             "teacher_forced_positive"])
        self.assertEqual(
            [seed.candidate_frame for seed in by_session[no_match]], [8, 20])
        self.assertTrue(all(
            seed.session_is_strict_no_match for seed in by_session[no_match]))

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

    def test_routed_cache_pair_supersedes_single_feature_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk = root / "patch" / "scene" / "episode_0000" / "chunk-000"
            chunk.mkdir(parents=True)
            aggregator = chunk / "lingbot_cache.npz"
            camera = chunk / "lingbot_cam_cache.npz"
            aggregator.write_bytes(b"aggregator")
            camera.write_bytes(b"camera")

            class Registry:
                def resolve_manifest_pair(self, record, scene, episode):
                    self.last = record, scene, episode
                    return aggregator, camera

            registry = Registry()
            seed = CandidateSeed(
                session_id="scene/episode_0000/goal_b_t0",
                scene="scene", episode="episode_0000", kind="revisit_b",
                query_path=root / "goal.jpg",
                candidate_path=root / "8.jpg", candidate_frame=8,
                dino_cosine=0.8, teacher_covis=0.7, label=1,
                session_has_positive=True,
                session_is_strict_no_match=False,
                session_max_covis=0.7,
                causal_manifest_sample_id="train/scene/episode_0000/b")
            episode_record = {"episode": "episode_0000", "n_frames": 32}
            manifest = {
                "flow_cache_routing": {"mode": "pinned"},
                "scenes": [{
                    "scene": "scene",
                    "selected_episodes": [episode_record],
                }],
            }
            routed = resolve_routed_feature_cache_pairs(
                manifest, [seed], route_registry=registry)
            self.assertIsNotNone(routed)
            pairs, provenance = routed
            self.assertEqual(
                pairs[("scene", "episode_0000")],
                (aggregator.resolve(), camera.resolve()))
            self.assertEqual(provenance, {"mode": "pinned"})
            self.assertEqual(
                registry.last, (episode_record, "scene", "episode_0000"))

    def test_routed_cache_rejects_missing_episode_and_bad_layout(self):
        root = Path("/tmp/routed-cache-test")
        seed = CandidateSeed(
            session_id="scene/episode_0000/goal", scene="scene",
            episode="episode_0000", kind="revisit_b",
            query_path=root / "goal.jpg", candidate_path=root / "8.jpg",
            candidate_frame=8, dino_cosine=0.8, teacher_covis=0.7,
            label=1, session_has_positive=True,
            session_is_strict_no_match=False, session_max_covis=0.7)
        manifest = {
            "flow_cache_routing": {"mode": "pinned"},
            "scenes": [{"scene": "other", "selected_episodes": [{
                "episode": "episode_0000"}]}],
        }

        class UnusedRegistry:
            def resolve_manifest_pair(self, _record, _scene, _episode):
                self.fail("route resolution must follow membership validation")

        with self.assertRaisesRegex(RuntimeError, "absent from causal"):
            resolve_routed_feature_cache_pairs(
                manifest, [seed], route_registry=UnusedRegistry())

        class BadRegistry:
            def resolve_manifest_pair(self, _record, _scene, _episode):
                return root / "wrong.npz", root / "also_wrong.npz"

        manifest["scenes"][0]["scene"] = "scene"
        with self.assertRaisesRegex(RuntimeError, "invalid layout"):
            resolve_routed_feature_cache_pairs(
                manifest, [seed], route_registry=BadRegistry())

    def test_legacy_feature_root_accepts_explicit_3leg_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "mp3d_3leg" / "scene" / "episode_0000"
            episode.mkdir(parents=True)
            seed = CandidateSeed(
                session_id="scene/episode_0000/goal", scene="scene",
                episode="episode_0000", kind="revisit_b",
                query_path=root / "goal.jpg", candidate_path=root / "8.jpg",
                candidate_frame=8, dino_cosine=0.8, teacher_covis=0.7,
                label=1, session_has_positive=True,
                session_is_strict_no_match=False, session_max_covis=0.7)
            self.assertEqual(feature_episode_root(root, seed), episode.resolve())

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
