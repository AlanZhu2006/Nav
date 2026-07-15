import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from internnav.dataset.memnav_pose_conventions import GENERATED_ZUP_FRAME_CONVENTION


def _source_episode(root: Path, convention=GENERATED_ZUP_FRAME_CONVENTION,
                    scene='scene', episode_name='episode_0000'):
    episode = root / 'vln_n1' / scene / episode_name
    (episode / 'data/chunk-000').mkdir(parents=True)
    (episode / 'videos/chunk-000/observation.images.rgb').mkdir(parents=True)
    (episode / 'meta').mkdir()
    (episode / 'data/chunk-000/episode_000000.parquet').touch()
    (episode / 'meta/gen_meta.json').write_text(json.dumps({
        'n_frames': 130,
        'switches': [100, 130],
        'anchor_margin': 39,
        'frame_convention': convention,
        'goals': [{
            'covis_curve': [0.0] * 40 + [0.8] + [0.0] * 59,
            'yaw_habitat': 0.0,
        }],
    }))
    (episode / 'goal_1.jpg').touch()
    # Dynamic E(k) requires k >= anchor_margin + exclude_recent = 122.
    (episode / 'videos/chunk-000/observation.images.rgb/122.jpg').touch()
    return episode


class MemNavInputCoverageTest(unittest.TestCase):
    def test_aux_goal_translation_uses_the_actual_endpoint(self):
        dataset = object.__new__(MemNav_Dataset)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
        ])
        dataset.process_actions = lambda *args, **kwargs: (
            points, None, None, None, np.array([0, 1])
        )
        # Like NavDP.xyz_to_xyt, the last row stores the penultimate point.
        dataset.xyz_to_xyt = lambda *args, **kwargs: np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
        ])
        actions, goal = dataset._build_actions(
            np.repeat(np.eye(4)[None], 3, axis=0), np.eye(4), pred_digit=1
        )
        np.testing.assert_allclose(goal, [2.0, 0.5, 0.25])
        self.assertEqual(actions.shape, (1, 3))

    def test_strict_coverage_requires_both_lingbot_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            features = Path(tmp) / 'features'
            episode = _source_episode(root)
            feature_dir = features / episode.relative_to(root) / 'videos/chunk-000'
            feature_dir.mkdir(parents=True)
            (feature_dir / 'lingbot_cache.npz').touch()

            with self.assertRaisesRegex(RuntimeError, 'Incomplete MemNav feature coverage'):
                MemNav_Dataset(
                    root,
                    feature_root=features,
                    strict_feature_coverage=True,
                )

    def test_generated_pose_marker_is_required_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            features = Path(tmp) / 'features'
            episode = _source_episode(root, convention='stored=Zup(M_W); render=habitat')
            feature_dir = features / episode.relative_to(root) / 'videos/chunk-000'
            feature_dir.mkdir(parents=True)
            (feature_dir / 'lingbot_cache.npz').touch()
            (feature_dir / 'lingbot_cam_cache.npz').touch()

            with self.assertRaisesRegex(RuntimeError, 'Invalid generated pose convention'):
                MemNav_Dataset(
                    root,
                    feature_root=features,
                    strict_feature_coverage=True,
                    require_generated_pose_convention=True,
                )

    def test_scene_split_has_no_leakage_and_fixed_k_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            features = Path(tmp) / 'features'
            for index in range(20):
                episode = _source_episode(root, scene=f'scene_{index:02d}')
                feature_dir = features / episode.relative_to(root) / 'videos/chunk-000'
                feature_dir.mkdir(parents=True)
                (feature_dir / 'lingbot_cache.npz').touch()
                (feature_dir / 'lingbot_cam_cache.npz').touch()

            common = dict(
                feature_root=features,
                strict_feature_coverage=True,
                validation_fraction=0.3,
                split_seed=7,
                sampling_mode='fixed_leg',
                sampling_seed=11,
            )
            train = MemNav_Dataset(root, data_split='train', **common)
            val = MemNav_Dataset(root, data_split='val', **common)
            train_scenes = {Path(path).parent.name for path in train.trajectory_dirs}
            val_scenes = {Path(path).parent.name for path in val.trajectory_dirs}
            self.assertFalse(train_scenes & val_scenes)
            self.assertEqual(len(train_scenes | val_scenes), 20)

            sample = train.samples[0]
            first = train._sample_k_and_digit(sample, sample['k_lo'], sample['k_hi'])
            second = train._sample_k_and_digit(sample, sample['k_lo'], sample['k_hi'])
            self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
