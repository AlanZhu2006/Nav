#!/usr/bin/env python3
"""Audit Li Guo-style MemNav sample counts without loading images or models."""

import argparse
from collections import Counter
import json
import os

from internnav.dataset.memnav_scene_splits import (
    normalize_scene_split,
    scene_ids_for_split,
)


def _semantic_reason(curve, kind, pos_hi, anchor_margin):
    valid = range(anchor_margin, len(curve))
    if anchor_margin < 0 or not valid:
        return 'no_valid_candidates'
    has_positive = any(curve[i] >= pos_hi for i in valid)
    if kind == 'revisit':
        return None if has_positive else 'weak_revisit'
    if kind == 'novel':
        return 'novel_has_positive' if has_positive else None
    return 'unknown_goal_kind'


def audit(root_dirs, feature_root=None, goal_slack=4, glimpse_neg=83, scene_split='all'):
    scene_split = normalize_scene_split(scene_split)
    allowed_scene_ids = scene_ids_for_split(scene_split)
    stats = Counter()
    selected_scenes = set()
    for group in sorted(os.listdir(root_dirs)):
        group_path = os.path.join(root_dirs, group)
        if not os.path.isdir(group_path):
            continue
        for scene in sorted(os.listdir(group_path)):
            if allowed_scene_ids is not None and scene not in allowed_scene_ids:
                continue
            scene_path = os.path.join(group_path, scene)
            if not os.path.isdir(scene_path):
                continue
            selected_scenes.add(scene)
            for episode in sorted(os.listdir(scene_path)):
                episode_path = os.path.join(scene_path, episode)
                meta_path = os.path.join(episode_path, 'meta/gen_meta.json')
                if not os.path.isfile(meta_path):
                    continue
                with open(meta_path) as handle:
                    meta = json.load(handle)
                goals = meta.get('goals') or []
                switches = meta.get('switches') or []
                n_frames = int(meta.get('n_frames', 0))
                anchor_margin = int(meta.get('anchor_margin', 39))
                pos_hi = float(meta.get('covis_pos_hi', 0.5))
                rgb_dir = os.path.join(
                    episode_path, 'videos/chunk-000/observation.images.rgb'
                )

                cached = True
                if feature_root:
                    rel = os.path.relpath(
                        os.path.join(episode_path, 'videos/chunk-000'), root_dirs
                    )
                    feature_dir = os.path.join(feature_root, rel)
                    cached = all(
                        os.path.isfile(os.path.join(feature_dir, name))
                        for name in ('lingbot_cache.npz', 'lingbot_cam_cache.npz')
                    )
                stats['episodes'] += 1
                stats['cached_episodes'] += int(cached)

                for j, goal in enumerate(goals):
                    curve = goal.get('covis_curve') or []
                    if not curve:
                        stats['skip_missing_curve'] += 1
                        continue
                    leg_start = len(curve)
                    leg_end = (
                        int(switches[j + 1]) if j + 1 < len(switches) else n_frames
                    )
                    goal_step = leg_end - 1
                    k_lo = max(leg_start, anchor_margin)
                    k_hi = goal_step - goal_slack
                    goal_image = os.path.join(episode_path, f'goal_{j + 1}.jpg')
                    if k_hi < k_lo:
                        stats['skip_short_leg'] += 1
                        continue
                    if not (
                        os.path.isfile(goal_image)
                        and os.path.isfile(os.path.join(rgb_dir, f'{k_lo}.jpg'))
                    ):
                        stats['skip_missing_goal_input'] += 1
                        continue

                    stats['li_guo_covis_samples'] += 1
                    stats['li_guo_covis_samples_cached'] += int(cached)
                    reason = _semantic_reason(
                        curve, goal.get('kind'), pos_hi, anchor_margin
                    )
                    if reason:
                        stats[f'skip_{reason}'] += 1
                        continue
                    kind = goal.get('kind')
                    stats['semantic_covis_samples'] += 1
                    stats[f'semantic_{kind}_samples'] += 1
                    if cached:
                        stats['semantic_covis_samples_cached'] += 1
                        stats[f'semantic_{kind}_samples_cached'] += 1

                if switches:
                    a_frame = int(switches[0]) - 1
                    k_lo = anchor_margin
                    k_hi = a_frame - goal_slack
                    if (
                        k_hi >= k_lo
                        and a_frame - glimpse_neg >= anchor_margin
                        and os.path.isfile(os.path.join(rgb_dir, f'{a_frame}.jpg'))
                    ):
                        stats['goalA_samples'] += 1
                        stats['goalA_samples_cached'] += int(cached)

    stats['li_guo_total_samples'] = (
        stats['li_guo_covis_samples'] + stats['goalA_samples']
    )
    stats['li_guo_total_samples_cached'] = (
        stats['li_guo_covis_samples_cached'] + stats['goalA_samples_cached']
    )
    stats['semantic_total_samples'] = (
        stats['semantic_covis_samples'] + stats['goalA_samples']
    )
    stats['semantic_total_samples_cached'] = (
        stats['semantic_covis_samples_cached'] + stats['goalA_samples_cached']
    )
    stats['scenes'] = len(selected_scenes)
    return {'scene_split': scene_split, **dict(sorted(stats.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dirs', required=True)
    parser.add_argument('--feature_root')
    parser.add_argument('--goal_slack', type=int, default=4)
    parser.add_argument('--glimpse_neg', type=int, default=83)
    parser.add_argument('--scene_split', default='all')
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
