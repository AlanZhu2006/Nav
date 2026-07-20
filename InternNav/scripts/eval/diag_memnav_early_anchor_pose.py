#!/usr/bin/env python3
"""Verify that LingBot can relocalize a goal from anchors before frame 39.

The current MemNav loader starts retrieval at ``num_scale + window - 1``.  The
new deep-warm goal insertion, however, can replay a shorter prefix for an early
anchor.  This diagnostic measures the resulting goal-pose error after one fixed
per-episode Sim(3), so lowering the candidate floor is tested geometrically
rather than assumed safe from retrieval accuracy alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parents[3]
if os.fspath(SCRIPT.parents[2]) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT.parents[2]))

from internnav.model.basemodel.memnav.cache_schema import validate_cache_files  # noqa: E402
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream  # noqa: E402


def umeyama(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    variance = np.square(source_centered).sum() / len(source)
    scale = np.trace(np.diag(singular) @ correction) / variance
    translation = target_mean - scale * (rotation @ source_mean)
    return float(scale), rotation, translation


def transform(scale, rotation, translation, points):
    points = np.asarray(points, dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def load_gt(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    extrinsics = np.asarray(
        [np.stack(value) for value in frame['action']], dtype=np.float64
    ).reshape(-1, 4, 4)
    return extrinsics[:, :3, 3]


def load_cache(stream, aggregate_path: Path, camera_path: Path, rgb_dir: Path):
    layout = validate_cache_files(
        aggregate_path,
        camera_path,
        expected_num_scale_frames=stream.num_scale,
        expected_sliding_window=stream.window,
        require_versioned=True,
    )
    with np.load(aggregate_path, allow_pickle=False) as aggregate:
        if 'scale_k' in aggregate.files:
            scale_k, scale_v, anchor_k, anchor_v = stream._cache_to_layered(
                aggregate['scale_k'], aggregate['scale_v'],
                aggregate['anchor_k'], aggregate['anchor_v'], stream.device,
            )
        else:
            scale_k, scale_v = stream.get_scale_kv(os.fspath(rgb_dir))
            anchor_k = torch.as_tensor(
                aggregate['anchor_k'], device=stream.device, dtype=torch.bfloat16
            ).permute(1, 2, 0, 3, 4).contiguous()
            anchor_v = torch.as_tensor(
                aggregate['anchor_v'], device=stream.device, dtype=torch.bfloat16
            ).permute(1, 2, 0, 3, 4).contiguous()
    with np.load(camera_path, allow_pickle=False) as camera:
        cam_k, cam_v = stream._cam_to_device(
            camera['cam_k'], camera['cam_v'], stream.device
        )
        trajectory = camera['cam_pose_enc'].astype(np.float64)
    return {
        'scale_k': scale_k,
        'scale_v': scale_v,
        'anchor_k': anchor_k,
        'anchor_v': anchor_v,
        'anchor_frame_indices': torch.as_tensor(
            layout.anchor_frame_indices, dtype=torch.long
        ),
        'cam_k': cam_k,
        'cam_v': cam_v,
        'cam_frame_indices': torch.as_tensor(
            layout.cam_frame_indices, dtype=torch.long
        ),
        'cam_pose_enc': trajectory,
    }


def parse_case(value: str):
    episode, goal_text, anchors_text = value.rsplit(':', 2)
    anchors = [int(item) for item in anchors_text.split(',')]
    if not anchors:
        raise ValueError(f'case has no anchors: {value}')
    return episode, int(goal_text), anchors


@torch.inference_mode()
def evaluate_case(args, stream, case):
    episode, goal_index, anchors = case
    episode_dir = Path(args.root_dir) / episode
    cache_dir = Path(args.feature_root) / episode / 'videos/chunk-000'
    rgb_dir = episode_dir / 'videos/chunk-000/observation.images.rgb'
    aggregate_path = cache_dir / 'lingbot_cache.npz'
    camera_path = cache_dir / 'lingbot_cam_cache.npz'
    cache = load_cache(stream, aggregate_path, camera_path, rgb_dir)

    gt = load_gt(episode_dir / 'data/chunk-000/episode_000000.parquet')
    raw_trajectory = cache['cam_pose_enc'][:, :3]
    count = min(len(gt), len(raw_trajectory))
    scale, rotation, translation = umeyama(
        raw_trajectory[:count], gt[:count]
    )
    aligned = transform(scale, rotation, translation, raw_trajectory[:count])
    trajectory_ate = float(
        np.sqrt(np.square(aligned - gt[:count]).sum(-1).mean())
    )

    meta = json.loads((episode_dir / 'meta/gen_meta.json').read_text())
    goals = meta['goals']
    switches = meta['switches']
    leg_end = switches[goal_index + 1] if goal_index + 1 < len(switches) else len(gt)
    goal_step = int(leg_end) - 1
    goal_path = episode_dir / f'goal_{goal_index + 1}.jpg'
    goal_image = stream.load_images([os.fspath(goal_path)])[0].to(stream.device)
    curve = np.asarray(goals[goal_index]['covis_curve'], dtype=np.float64)
    pos_hi = float(meta.get('covis_pos_hi', 0.5))
    pos_lo = float(meta.get('covis_pos_lo', 0.1))

    rows = []
    for anchor in anchors:
        if not stream.num_scale <= anchor < len(curve):
            raise ValueError(
                f'anchor {anchor} must be in [{stream.num_scale}, {len(curve) - 1}]'
            )
        _, goal_aggregate = stream.goal_append_warm(
            goal_image,
            cache,
            anchor,
            os.fspath(rgb_dir),
            args.warm,
            return_agg=True,
        )
        predicted = stream.camera_pose(
            cache['cam_k'],
            cache['cam_v'],
            anchor + 1,
            goal_aggregate,
            cache['cam_frame_indices'],
        )[-1, :3].float().cpu().numpy()
        aligned_goal = transform(
            scale, rotation, translation, predicted[None]
        )[0]
        overlap = float(curve[anchor])
        outcome = (
            'positive' if overlap >= pos_hi
            else 'negative' if overlap <= pos_lo
            else 'gray'
        )
        rows.append({
            'anchor': anchor,
            'warm_replay_frames': anchor - max(stream.num_scale, anchor - args.warm + 1) + 1,
            'covis': overlap,
            'outcome': outcome,
            'aligned_goal_xyz': aligned_goal.tolist(),
            'gt_goal_xyz': gt[goal_step].tolist(),
            'goal_translation_error_m': float(
                np.linalg.norm(aligned_goal - gt[goal_step])
            ),
        })
    return {
        'episode': episode,
        'goal_index': goal_index,
        'goal_step': goal_step,
        'sim3_scale': scale,
        'trajectory_ate_m': trajectory_ate,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--lingbot-repo', required=True)
    parser.add_argument('--lingbot-weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--num-scale', type=int, default=8)
    parser.add_argument('--max-frame-num', type=int, default=4096)
    parser.add_argument('--warm', type=int, default=64)
    parser.add_argument('--case', action='append', required=True)
    args = parser.parse_args()
    if args.warm < 0:
        parser.error('--warm must be non-negative')
    output = Path(args.output).resolve()
    try:
        output.relative_to(WORKTREE.resolve())
    except ValueError as error:
        raise RuntimeError(f'output must stay inside {WORKTREE}') from error

    stream = LingBotStream(
        lingbot_repo=args.lingbot_repo,
        weights=args.lingbot_weights,
        window=args.window,
        num_scale=args.num_scale,
        max_frame_num=args.max_frame_num,
        device='cuda',
    )
    result = {
        'metadata': {
            'warm': args.warm,
            'window': args.window,
            'num_scale': args.num_scale,
            'sim3_fit': 'one full-trajectory transform per episode',
        },
        'cases': [
            evaluate_case(args, stream, parse_case(value))
            for value in args.case
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f'REPORT_SAVED {output}')


if __name__ == '__main__':
    main()
