from collections import Counter
import json

import numpy as np
import pytest

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from scripts.dataset_converters.audit_memnav_sampling import audit


def _dataset(sampling_mode='random_leg', add_goalA=False):
    dataset = object.__new__(MemNav_Dataset)
    dataset.num_scale = 8
    dataset.window_size = 32
    dataset.covis_pos_hi = 0.5
    dataset.covis_pos_lo = 0.1
    dataset.anchor_margin_default = 39
    dataset.glimpse_pos = 14
    dataset.glimpse_neg = 83
    dataset.goal_slack = 4
    dataset.add_goalA = add_goalA
    dataset.sampling_mode = sampling_mode
    dataset.label_stats = Counter()
    return dataset


def _write_meta(tmp_path, meta, goal_frames=()):
    traj_dir = tmp_path / 'episode'
    rgb_dir = traj_dir / 'videos/chunk-000/observation.images.rgb'
    rgb_dir.mkdir(parents=True)
    meta_path = traj_dir / 'meta/gen_meta.json'
    meta_path.parent.mkdir()
    meta_path.write_text(json.dumps(meta))
    for frame in goal_frames:
        (rgb_dir / f'{frame}.jpg').touch()
    return meta_path, rgb_dir, traj_dir


def _revisit_meta():
    curve = [0.0] * 50
    curve[45] = 0.8
    return {
        'n_frames': 120,
        'switches': [50],
        'anchor_margin': 39,
        'goals': [{'kind': 'revisit', 'covis_curve': curve}],
    }


def _li_guo_label(sample, k, glimpse_pos=14, glimpse_neg=83):
    """Reference formula from 3af2c8d, kept local to make drift visible."""
    pos = np.zeros(k + 1, dtype=bool)
    neg = np.zeros(k + 1, dtype=bool)
    if sample['has_covis']:
        length = min(int(sample['leg_start']), k + 1)
        pos[:length] = sample['pos_pre'][:length]
        neg[:length] = sample['neg_pre'][:length]
        null_pos = bool(sample['null_pos'])
    else:
        idx = np.arange(k + 1)
        target = int(sample['T_A'])
        margin = int(sample['amargin'])
        pos = idx >= (target - glimpse_pos)
        neg = (idx <= (target - glimpse_neg)) & (idx >= margin)
        neg &= ~pos
        null_pos = not bool(pos.any())
    return pos, neg, null_pos


def test_random_leg_sampling_matches_li_guo_ranges_and_preserves_semantics(tmp_path):
    dataset = _dataset()
    meta_path, rgb_dir, traj_dir = _write_meta(tmp_path, _revisit_meta(), goal_frames=(50,))
    (traj_dir / 'goal_1.jpg').touch()

    samples = dataset._parse_meta(str(meta_path), str(rgb_dir), str(traj_dir))

    assert len(samples) == 1
    sample = samples[0]
    assert sample['k_lo'] == 50
    assert sample['k_hi'] == 115
    assert sample['goal_step'] == 119
    assert sample['goal_kind'] == 'revisit'
    assert sample['null_pos'] is False

    np.random.seed(0)
    sampled_steps = {dataset._sample_current_step(sample, 120) for _ in range(20)}
    assert len(sampled_steps) > 1
    assert min(sampled_steps) >= 50
    assert max(sampled_steps) <= 115


def test_random_sequence_and_labels_match_li_guo_reference(tmp_path):
    dataset = _dataset(add_goalA=True)
    curve = [0.0] * 130
    curve[100] = 0.8
    meta = {
        'n_frames': 200,
        'switches': [130],
        'anchor_margin': 39,
        'goals': [{'kind': 'revisit', 'covis_curve': curve}],
    }
    meta_path, rgb_dir, traj_dir = _write_meta(
        tmp_path, meta, goal_frames=(129, 130)
    )
    (traj_dir / 'goal_1.jpg').touch()
    covis_sample, goal_a_sample = dataset._parse_meta(
        str(meta_path), str(rgb_dir), str(traj_dir)
    )

    np.random.seed(123)
    expected_steps = [np.random.randint(130, 196) for _ in range(32)]
    np.random.seed(123)
    actual_steps = [dataset._sample_current_step(covis_sample, 200) for _ in range(32)]
    assert actual_steps == expected_steps

    for sample in (covis_sample, goal_a_sample):
        for k in range(sample['k_lo'], sample['k_hi'] + 1):
            expected_pos, expected_neg, expected_null = _li_guo_label(sample, k)
            pos, neg, _, null_pos, _ = dataset._build_label(sample, k)
            assert np.array_equal(pos, expected_pos)
            assert np.array_equal(neg, expected_neg)
            assert null_pos is expected_null


def test_random_leg_label_ignores_own_leg_but_masks_unreconstructable_frames(tmp_path):
    dataset = _dataset()
    meta_path, rgb_dir, traj_dir = _write_meta(tmp_path, _revisit_meta(), goal_frames=(50,))
    (traj_dir / 'goal_1.jpg').touch()
    sample = dataset._parse_meta(str(meta_path), str(rgb_dir), str(traj_dir))[0]

    pos, neg, candidate, null_pos, is_revisit = dataset._build_label(sample, k=80)

    assert pos[45]
    assert not candidate[:39].any()
    assert candidate[39:].all()
    assert not pos[50:].any()
    assert not neg[50:].any()
    assert null_pos is False
    assert is_revisit is True


def test_goal_a_uses_li_guo_dynamic_glimpse_thresholds(tmp_path):
    dataset = _dataset(add_goalA=True)
    meta = {'n_frames': 130, 'switches': [130], 'anchor_margin': 39, 'goals': []}
    meta_path, rgb_dir, traj_dir = _write_meta(tmp_path, meta, goal_frames=(129,))
    sample = dataset._parse_meta(str(meta_path), str(rgb_dir), str(traj_dir))[0]

    assert sample['has_covis'] is False
    assert sample['k_lo'] == 39
    assert sample['k_hi'] == 125

    pos, neg, candidate, null_pos, is_revisit = dataset._build_label(sample, k=40)
    assert not pos.any()
    assert neg[39:41].all()
    assert not candidate[:39].any()
    assert null_pos is True
    assert is_revisit is False

    pos, neg, _, null_pos, is_revisit = dataset._build_label(sample, k=120)
    assert pos[115:121].all()
    assert not (pos & neg).any()
    assert null_pos is False
    assert is_revisit is True


def test_fixed_switch_mode_reproduces_previous_checkpoint_sampling(tmp_path):
    dataset = _dataset(sampling_mode='fixed_switch')
    meta_path, rgb_dir, traj_dir = _write_meta(tmp_path, _revisit_meta(), goal_frames=(49,))
    (traj_dir / 'goal_1.jpg').touch()
    sample = dataset._parse_meta(str(meta_path), str(rgb_dir), str(traj_dir))[0]

    assert sample['k_lo'] == 49
    assert sample['k_hi'] == 49
    assert dataset._sample_current_step(sample, 120) == 49


def test_anchor_margin_boundary_is_a_valid_full_window():
    dataset = _dataset()
    sample = {'goal_step': 49, 'k_lo': 39, 'k_hi': 45}

    assert dataset._sample_current_step(sample, 50) >= 39


def test_sampling_rejects_cache_shorter_than_metadata(tmp_path):
    dataset = _dataset()
    meta_path, rgb_dir, traj_dir = _write_meta(tmp_path, _revisit_meta(), goal_frames=(50,))
    (traj_dir / 'goal_1.jpg').touch()
    sample = dataset._parse_meta(str(meta_path), str(rgb_dir), str(traj_dir))[0]

    with pytest.raises(RuntimeError, match='does not cover goal_step=119'):
        dataset._sample_current_step(sample, n_frames=100)


def test_sampling_audit_separates_li_guo_and_semantic_counts(tmp_path):
    root = tmp_path / 'raw'
    features = tmp_path / 'features'
    for episode_name, peak in (('strong', 0.8), ('weak', 0.4)):
        episode = root / 'vln_n1' / 'scene' / episode_name
        rgb_dir = episode / 'videos/chunk-000/observation.images.rgb'
        rgb_dir.mkdir(parents=True)
        (rgb_dir / '50.jpg').touch()
        (rgb_dir / '129.jpg').touch()
        (episode / 'goal_1.jpg').touch()
        (episode / 'meta').mkdir()
        curve = [0.0] * 50
        curve[45] = peak
        (episode / 'meta/gen_meta.json').write_text(json.dumps({
            'n_frames': 130,
            'switches': [130],
            'anchor_margin': 39,
            'goals': [{'kind': 'revisit', 'covis_curve': curve}],
        }))
        feature_dir = features / 'vln_n1' / 'scene' / episode_name / 'videos/chunk-000'
        feature_dir.mkdir(parents=True)
        (feature_dir / 'lingbot_cache.npz').touch()
        (feature_dir / 'lingbot_cam_cache.npz').touch()

    stats = audit(str(root), str(features))

    assert stats['li_guo_covis_samples'] == 2
    assert stats['semantic_covis_samples'] == 1
    assert stats['skip_weak_revisit'] == 1
    assert stats['goalA_samples'] == 2
    assert stats['li_guo_total_samples'] == 4
    assert stats['semantic_total_samples'] == 3
    assert stats['semantic_total_samples_cached'] == 3
