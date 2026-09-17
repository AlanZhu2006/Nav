"""Causality, transform convention, and state ownership for GEM reactivation."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from NavDP.baselines.memnav.gem.reactivation import (
    PacketRequest, fit_similarity, isolated_stream, retrieve_packet)


class ReactivationTests(unittest.TestCase):
    def test_future_descriptors_cannot_change_request(self):
        rng = np.random.default_rng(31)
        keys = rng.normal(size=(200, 16))
        expected = retrieve_packet(keys[:129], 128)
        keys[129:] = np.nan
        self.assertEqual(retrieve_packet(keys, 128), expected)
        self.assertLessEqual(max(expected.historical_frames), 128 - 64)
        self.assertEqual(expected.recent_frames[-1], 128)

    def test_ties_and_no_history(self):
        keys = np.ones((100, 4))
        self.assertIsNone(retrieve_packet(keys, 64))
        self.assertEqual(retrieve_packet(keys, 99).reference, 0)
        self.assertEqual(retrieve_packet(keys, 99).historical_frames, tuple(range(8)))

    def test_disjoint_identity_is_required(self):
        with self.assertRaises(ValueError):
            PacketRequest(0, 8, (0, 1), (1, 8), .9)

    def test_similarity_known_transform_and_reverse(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            points = rng.normal(size=(100, 3))
            rotation = Rotation.random(random_state=rng).as_matrix()
            scale = np.exp(rng.normal())
            translation = rng.normal(size=3)
            target = scale * points @ rotation.T + translation
            actual, _ = fit_similarity(points, target)
            reverse, _ = fit_similarity(target, points)
            np.testing.assert_allclose(actual[:3, :3], scale * rotation, atol=1e-12)
            np.testing.assert_allclose(actual[:3, 3], translation, atol=1e-12)
            np.testing.assert_allclose(reverse @ actual, np.eye(4), atol=1e-12)
        with self.assertRaises(ValueError):
            fit_similarity(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_state_restored_on_inference_exception(self):
        original_kv = {'k_0': np.arange(6)}
        original_cam = [{'k_0': np.arange(3)}]
        agg = SimpleNamespace(use_sdpa=True, kv_cache_manager=None,
            kv_cache=original_kv, total_frames_processed=31, _cached_pos3d=object())
        camera = SimpleNamespace(kv_cache=original_cam, frame_idx=180, pos_cache=object())

        def clean():
            for key in agg.kv_cache:
                agg.kv_cache[key] = None
            agg.total_frames_processed = 0
            agg._cached_pos3d = None
            camera.kv_cache = None
            camera.frame_idx = 0

        model = SimpleNamespace(aggregator=agg, camera_head=camera, clean_kv_cache=clean)
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            with isolated_stream(model):
                self.assertIn('k_0', agg.kv_cache)
                self.assertIsNone(agg.kv_cache['k_0'])
                agg.kv_cache['new'] = np.ones(2)
                camera.kv_cache = [{'changed': True}]
                raise RuntimeError('inference failed')
        self.assertIs(agg.kv_cache, original_kv)
        self.assertIs(camera.kv_cache, original_cam)
        self.assertEqual(agg.total_frames_processed, 31)
        self.assertEqual(camera.frame_idx, 180)
        np.testing.assert_array_equal(agg.kv_cache['k_0'], np.arange(6))


if __name__ == '__main__':
    unittest.main()
