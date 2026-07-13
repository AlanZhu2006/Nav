import json
import sys
import types

import pytest

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from internnav.dataset.memnav_scene_splits import (
    R2R_TEST_SCENES,
    R2R_TRAIN_SCENES,
    R2R_VAL_UNSEEN_SCENES,
    normalize_scene_split,
)
from scripts.dataset_converters.audit_memnav_sampling import audit


def _write_episode(root, features, scene, episode_name='episode_0000'):
    episode = root / 'mp3d_2leg' / scene / episode_name
    rgb_dir = episode / 'videos/chunk-000/observation.images.rgb'
    data_dir = episode / 'data/chunk-000'
    meta_dir = episode / 'meta'
    rgb_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    (data_dir / 'episode_000000.parquet').touch()
    (data_dir / 'path.ply').touch()
    (rgb_dir / '50.jpg').touch()
    (rgb_dir / '129.jpg').touch()
    (episode / 'goal_1.jpg').touch()
    curve = [0.0] * 50
    curve[45] = 0.8
    (meta_dir / 'gen_meta.json').write_text(json.dumps({
        'n_frames': 130,
        'switches': [130],
        'anchor_margin': 39,
        'goals': [{'kind': 'revisit', 'covis_curve': curve}],
    }))

    feature_dir = (
        features / 'mp3d_2leg' / scene / episode_name / 'videos/chunk-000'
    )
    feature_dir.mkdir(parents=True)
    (feature_dir / 'lingbot_cache.npz').touch()
    (feature_dir / 'lingbot_cam_cache.npz').touch()


def _install_fake_lingbot(monkeypatch):
    lingbot_map = types.ModuleType('lingbot_map')
    utils = types.ModuleType('lingbot_map.utils')
    load_fn = types.ModuleType('lingbot_map.utils.load_fn')
    load_fn.load_and_preprocess_images = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'lingbot_map', lingbot_map)
    monkeypatch.setitem(sys.modules, 'lingbot_map.utils', utils)
    monkeypatch.setitem(sys.modules, 'lingbot_map.utils.load_fn', load_fn)


def test_official_r2r_building_splits_are_disjoint_and_complete():
    assert len(R2R_TRAIN_SCENES) == 61
    assert len(R2R_VAL_UNSEEN_SCENES) == 11
    assert len(R2R_TEST_SCENES) == 18
    assert R2R_TRAIN_SCENES.isdisjoint(R2R_VAL_UNSEEN_SCENES)
    assert R2R_TRAIN_SCENES.isdisjoint(R2R_TEST_SCENES)
    assert R2R_VAL_UNSEEN_SCENES.isdisjoint(R2R_TEST_SCENES)


def test_scene_split_aliases_and_typos():
    assert normalize_scene_split('train') == 'r2r_train'
    assert normalize_scene_split('val-unseen') == 'r2r_val_unseen'
    assert normalize_scene_split(None) == 'all'
    with pytest.raises(ValueError, match='unknown scene_split'):
        normalize_scene_split('r2r_validation')


def test_dataset_and_audit_filter_by_building(monkeypatch, tmp_path):
    root = tmp_path / 'raw'
    features = tmp_path / 'features'
    train_scene = '17DRP5sb8fy'
    val_scene = '2azQ1b91cZZ'
    _write_episode(root, features, train_scene)
    _write_episode(root, features, val_scene)
    _install_fake_lingbot(monkeypatch)

    dataset = MemNav_Dataset(
        root,
        feature_root=features,
        scene_split='r2r_train',
        add_goalA=False,
        window_size=32,
        num_scale=8,
        lingbot_repo=str(tmp_path),
    )
    train_stats = audit(str(root), str(features), scene_split='r2r_train')
    val_stats = audit(str(root), str(features), scene_split='r2r_val_unseen')

    assert dataset.scene_ids == (train_scene,)
    assert len(dataset) == 1
    assert train_stats['scenes'] == 1
    assert train_stats['semantic_total_samples'] == 2
    assert val_stats['scenes'] == 1
    assert val_stats['semantic_total_samples'] == 2
