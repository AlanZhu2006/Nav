#!/usr/bin/env python3
"""Upper-bound test: substitute an oracle local-route bearing into revisit pose.

The current checkpoint encodes the final endpoint bearing in its revisit tokens.
This diagnostic keeps the visual condition, endpoint range, retrieval gate, model
weights, and DDPM random streams fixed, but replaces only that bearing with the
direction obtained by summing future expert action labels over a chosen horizon.

Future actions are unavailable at inference.  A gain is therefore only evidence
that a learned route predictor may be useful; it is not a deployable method or a
closed-loop navigation result.  A loss is also informative: the frozen decoder
may have learned endpoint-token semantics that cannot be repurposed without
training a separate residual route token.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parents[3]
if os.fspath(SCRIPT.parents[2]) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT.parents[2]))

from internnav.dataset.memnav_dataset_lerobot import (  # noqa: E402
    MemNav_Dataset,
    memnav_collate_fn,
)
from scripts.eval.eval_memnav_offline import (  # noqa: E402
    _cuda_generator,
    _dataset_cache_contract,
    load_checkpoint,
)
from scripts.train.configs.memnav import memnav_exp_cfg  # noqa: E402


def route_direction_from_actions(actions, horizon, fallback):
    """Return unit robot-frame direction from the first ``horizon`` actions."""
    if actions.ndim != 3 or actions.shape[-1] < 2:
        raise ValueError('actions must have shape [B,T,D>=2]')
    if fallback.shape != actions.shape[:1] + (2,):
        raise ValueError('fallback must have shape [B,2]')
    horizon = min(max(int(horizon), 1), actions.shape[1])
    displacement = actions[:, :horizon, :2].sum(dim=1)
    radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    direction = displacement / radius.clamp_min(1e-6)
    return torch.where(radius > 1e-6, direction, fallback)


def circular_blend_direction(endpoint, route, alpha):
    """Interpolate unit 2-D directions along the shortest signed arc."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must lie in [0,1]')
    endpoint = F.normalize(endpoint, dim=-1)
    route = F.normalize(route, dim=-1)
    endpoint_angle = torch.atan2(endpoint[..., 1], endpoint[..., 0])
    route_angle = torch.atan2(route[..., 1], route[..., 0])
    delta = torch.atan2(
        torch.sin(route_angle - endpoint_angle),
        torch.cos(route_angle - endpoint_angle),
    )
    angle = endpoint_angle + alpha * delta
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def rebuild_revisit_tokens(model, condition, direction):
    """Rebuild current checkpoint tokens after changing only pose-code bearing."""
    core = getattr(model, 'core', model)
    if not hasattr(core, 'revisit_merge'):
        raise TypeError('model must be MemNavNet or a MemNavPolicy wrapper')
    merge = core.revisit_merge
    encoder = merge.pose_encoder
    range_code = torch.asinh(
        condition['pose_range_steps'] / encoder.distance_unit_steps
    ).clamp(max=5.0)
    reliability = (
        condition['pose_reliability']
        if encoder.condition_on_reliability
        else torch.ones_like(condition['pose_reliability'])
    )
    pose_code = torch.cat(
        (direction, range_code.unsqueeze(-1), reliability.unsqueeze(-1)), dim=-1
    )
    adapted = pose_code + merge.rel_adapter(pose_code)
    return merge.revisit_head(adapted).view(-1, merge.n_out, merge.dim)


def _bootstrap_delta(values, baseline):
    delta = np.asarray(values, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    rng = np.random.default_rng(0)
    draws = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(-1)
    return {
        'paired_delta_vs_baseline': float(delta.mean()),
        'paired_delta_ci95': np.quantile(draws, [0.025, 0.975]).tolist(),
        'fraction_better': float(np.mean(delta < 0.0)),
    }


def summarize(records, strategies):
    groups = {
        'all': records,
        'revisit': [row for row in records if row['is_revisit']],
        'novel': [row for row in records if not row['is_revisit']],
        '3leg_goal_c': [
            row for row in records
            if row['is_3leg'] and row['goal_label'] == 'C'
        ],
        'route_angle_ge_45': [
            row for row in records if row['route_angle_deg'] >= 45.0
        ],
        'remaining_span_ge_256': [
            row for row in records if row['remaining_span'] >= 256
        ],
    }
    result = {}
    for name, rows in groups.items():
        group = {'rows': len(rows), 'strategies': {}}
        if not rows:
            result[name] = group
            continue
        baseline = [row['strategies']['baseline'] for row in rows]
        for strategy in strategies:
            values = [row['strategies'][strategy] for row in rows]
            stats = _bootstrap_delta(values, baseline)
            stats['action_mse'] = float(np.mean(values))
            group['strategies'][strategy] = stats
        result[name] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--lingbot-repo', required=True)
    parser.add_argument('--lingbot-weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection-indices', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampling-seed', type=int, default=0)
    parser.add_argument('--diffusion-seed', type=int, default=104729)
    parser.add_argument('--horizons', type=int, nargs='+', default=(1, 2, 4, 8, 24))
    parser.add_argument('--blend-horizon', type=int, default=4)
    parser.add_argument('--blend-alphas', type=float, nargs='+', default=(0.25, 0.5, 0.75))
    args = parser.parse_args()
    if any(value < 1 for value in args.horizons) or args.blend_horizon < 1:
        parser.error('route horizons must be positive')
    if any(not 0.0 <= value <= 1.0 for value in args.blend_alphas):
        parser.error('blend alphas must lie in [0,1]')
    output = Path(args.output).resolve()
    try:
        output.relative_to(WORKTREE.resolve())
    except ValueError as error:
        raise RuntimeError(f'output must stay inside {WORKTREE}') from error

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    config = copy.deepcopy(memnav_exp_cfg)
    config.il.lingbot_repo = args.lingbot_repo
    config.il.lingbot_weights = args.lingbot_weights
    dataset = MemNav_Dataset(
        args.root_dir,
        predict_size=config.il.predict_size,
        image_size=config.il.image_size,
        lingbot_repo=config.il.lingbot_repo,
        feature_root=args.feature_root,
        window_size=config.il.window_size,
        num_scale=config.il.num_scale,
        **_dataset_cache_contract(config),
        data_split='all',
        validation_fraction=config.il.validation_fraction,
        split_seed=config.il.split_seed,
        sampling_mode='fixed_leg',
        sampling_seed=args.sampling_seed,
    )
    indices = [int(value) for value in args.selection_indices.split(',') if value]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= len(dataset)
    ):
        raise ValueError('invalid --selection-indices')
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=memnav_collate_fn,
    )
    model, checkpoint = load_checkpoint(config, args.checkpoint)
    model.eval()

    strategies = ['baseline']
    strategies.extend(f'route_h{value}' for value in args.horizons)
    strategies.extend(
        f'blend_h{args.blend_horizon}_a{int(round(100 * value)):03d}'
        for value in args.blend_alphas
    )
    records = []
    reconstruction_max_abs = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            condition = model.prepare_condition(batch)
            device = next(model.parameters()).device
            labels = batch['batch_labels'].to(device)
            endpoint = condition['raw_pose_direction']

            reconstructed = rebuild_revisit_tokens(model, condition, endpoint)
            reconstruction_max_abs = max(
                reconstruction_max_abs,
                float((reconstructed - condition['revisit']).abs().max().cpu()),
            )
            if reconstruction_max_abs > 2e-5:
                raise RuntimeError(
                    'revisit-token reconstruction changed the baseline: '
                    f'{reconstruction_max_abs}'
                )

            directions = {
                horizon: route_direction_from_actions(labels, horizon, endpoint)
                for horizon in set(list(args.horizons) + [args.blend_horizon])
            }
            changed_conditions = {'baseline': condition}
            for horizon in args.horizons:
                changed = dict(condition)
                changed['revisit'] = rebuild_revisit_tokens(
                    model, condition, directions[horizon]
                )
                changed_conditions[f'route_h{horizon}'] = changed
            for alpha in args.blend_alphas:
                name = f'blend_h{args.blend_horizon}_a{int(round(100 * alpha)):03d}'
                changed = dict(condition)
                direction = circular_blend_direction(
                    endpoint, directions[args.blend_horizon], alpha
                )
                changed['revisit'] = rebuild_revisit_tokens(model, condition, direction)
                changed_conditions[name] = changed

            paired_seed = args.diffusion_seed + 3 * batch_index
            initial_noise = torch.randn(
                labels.shape,
                device=device,
                generator=_cuda_generator(device, paired_seed),
            )
            mse = {}
            for strategy in strategies:
                actions = model.sample_actions_from_condition(
                    changed_conditions[strategy],
                    initial_noise=initial_noise,
                    generator=_cuda_generator(device, paired_seed + 1),
                )
                mse[strategy] = (
                    (actions - labels).square().mean(dim=(1, 2)).cpu()
                )
            for row, identity in enumerate(batch['sample_identities']):
                records.append({
                    'sample_identity': identity,
                    'goal_label': batch['goal_labels'][row],
                    'is_revisit': bool(batch['batch_is_revisit'][row] > 0.5),
                    'is_3leg': 'mp3d_3leg/' in identity,
                    'route_angle_deg': float(
                        batch['batch_decision_route_angle_deg'][row]
                    ),
                    'remaining_span': int(
                        batch['goal_steps'][row] - batch['cur_steps'][row]
                    ),
                    'strategies': {
                        strategy: float(mse[strategy][row])
                        for strategy in strategies
                    },
                })
            print(
                f'batch={batch_index + 1}/{len(loader)} records={len(records)}',
                flush=True,
            )

    result = {
        'metadata': {
            'checkpoint': os.fspath(Path(checkpoint).resolve()),
            'dataset_fingerprint': dataset.dataset_fingerprint,
            'selection_indices': indices,
            'diffusion_seed': args.diffusion_seed,
            'horizons': args.horizons,
            'blend_horizon': args.blend_horizon,
            'blend_alphas': args.blend_alphas,
            'strategies': strategies,
            'revisit_reconstruction_max_abs': reconstruction_max_abs,
            'future_action_labels_used_for_pose_bearing_only': True,
            'endpoint_range_and_gate_held_fixed': True,
            'encoded_visual_condition_held_fixed': True,
            'paired_ddpm_randomness': True,
            'closed_loop_navigation': False,
        },
        'summary': summarize(records, strategies),
        'records': records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result['summary'], indent=2, sort_keys=True))
    print(f'REPORT_SAVED {output}')


if __name__ == '__main__':
    main()
