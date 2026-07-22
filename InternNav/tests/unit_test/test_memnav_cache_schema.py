import unittest
from pathlib import Path
import tempfile

import numpy as np

from internnav.model.basemodel.memnav.cache_schema import (
    CACHE_SCHEMA_VERSION,
    KEYFRAME_POLICY,
    auto_keyframe_interval,
    camera_keyframe_indices,
    post_scale_keyframe_indices,
    validate_cache_pair,
    validate_cache_files,
)


def _pair(num_frames=1329, scale=8, interval=5, window=32):
    anchor_indices = post_scale_keyframe_indices(num_frames, scale, interval)
    cam_indices = camera_keyframe_indices(num_frames, scale, interval)
    shared = {
        'cache_schema_version': np.array([CACHE_SCHEMA_VERSION]),
        'keyframe_policy': np.array([KEYFRAME_POLICY]),
        'num_frames': np.array([num_frames]),
        'num_scale_frames': np.array([scale]),
        'keyframe_interval': np.array([interval]),
        'precompute_signature': np.array(['unit-test']),
        'kv_cache_sliding_window': np.array([window]),
    }
    aggregator = {
        **shared,
        'meta': np.array([scale, 6, 1, 1, 1]),
        'dino_cls': np.zeros((num_frames, 2)),
        'anchor_k': np.zeros((len(anchor_indices), 1)),
        'anchor_v': np.zeros((len(anchor_indices), 1)),
        'anchor_frame_indices': anchor_indices,
    }
    camera = {
        **shared,
        'cam_pose_enc': np.zeros((num_frames, 9)),
        'cam_k': np.zeros((len(cam_indices), 1)),
        'cam_v': np.zeros((len(cam_indices), 1)),
        'cam_frame_indices': cam_indices,
    }
    return aggregator, camera


class MemNavCacheSchemaTest(unittest.TestCase):
    def test_official_auto_interval_and_indices(self):
        self.assertEqual(auto_keyframe_interval(320), 1)
        self.assertEqual(auto_keyframe_interval(321), 2)
        self.assertEqual(auto_keyframe_interval(1329), 5)
        np.testing.assert_array_equal(
            post_scale_keyframe_indices(20, 8, 5), [8, 13, 18]
        )
        np.testing.assert_array_equal(
            camera_keyframe_indices(20, 8, 5),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 18],
        )

    def test_versioned_sparse_pair_validates(self):
        aggregator, camera = _pair()
        layout = validate_cache_pair(
            aggregator,
            camera,
            expected_num_frames=1329,
            expected_num_scale_frames=8,
            expected_sliding_window=32,
            require_versioned=True,
        )
        self.assertFalse(layout.legacy_dense)
        self.assertEqual(layout.keyframe_interval, 5)
        self.assertEqual(layout.cam_frame_indices[-1], 1328)

    def test_mixed_or_shifted_cache_fails_closed(self):
        aggregator, camera = _pair()
        camera['precompute_signature'] = np.array(['different-run'])
        with self.assertRaisesRegex(ValueError, 'precompute_signature mismatch'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        aggregator, camera = _pair()
        aggregator['anchor_frame_indices'] = aggregator['anchor_frame_indices'] + 1
        with self.assertRaisesRegex(ValueError, 'anchor_frame_indices'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        aggregator, camera = _pair()
        aggregator['anchor_frame_indices'] = aggregator[
            'anchor_frame_indices'
        ].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'integer dtype'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_invalid_layout_scalars_fail_closed(self):
        for field, value, message in (
            ('num_frames', 0, 'num_frames must be positive'),
            ('num_scale_frames', 2000, 'num_scale_frames must be in'),
            ('keyframe_interval', 0, 'keyframe_interval must be positive'),
            ('precompute_signature', '', 'precompute_signature must be non-empty'),
        ):
            aggregator, camera = _pair()
            aggregator[field] = np.array([value])
            camera[field] = np.array([value])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_legacy_dense_requires_explicit_opt_in(self):
        num_frames, scale = 12, 8
        aggregator = {
            'meta': np.array([scale, 6, 1, 1, 1]),
            'dino_cls': np.zeros((num_frames, 2)),
            'anchor_k': np.zeros((num_frames - scale, 1)),
            'anchor_v': np.zeros((num_frames - scale, 1)),
        }
        camera = {
            'cam_pose_enc': np.zeros((num_frames, 9)),
            'cam_k': np.zeros((num_frames, 1)),
            'cam_v': np.zeros((num_frames, 1)),
        }
        layout = validate_cache_pair(aggregator, camera)
        self.assertTrue(layout.legacy_dense)
        with self.assertRaisesRegex(ValueError, 'versioned LingBot cache required'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_file_preflight_reads_shapes_and_rejects_truncated_payload(self):
        aggregator, camera = _pair(num_frames=20, interval=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregator_path = root / 'lingbot_cache.npz'
            camera_path = root / 'lingbot_cam_cache.npz'
            np.savez(aggregator_path, **aggregator)
            np.savez(camera_path, **camera)
            layout = validate_cache_files(
                aggregator_path,
                camera_path,
                expected_num_frames=20,
                expected_num_scale_frames=8,
                expected_sliding_window=32,
            )
            self.assertEqual(layout.keyframe_interval, 2)

            camera['cam_v'] = camera['cam_v'][:-1]
            np.savez(camera_path, **camera)
            with self.assertRaisesRegex(ValueError, 'payload/index length mismatch'):
                validate_cache_files(aggregator_path, camera_path)


if __name__ == '__main__':
    unittest.main()
