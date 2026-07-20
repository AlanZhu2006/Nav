#!/usr/bin/env python3
"""Upper-bound test: can route-aware revisit gating improve DDPM actions?

The diagnostic computes the expensive visual condition once, then resamples the
same DDPM initial noise while changing only the effective revisit gate.  Future
route angle is deliberately used as an oracle selector.  Therefore a gain says
an inference-safe route-compatibility predictor is worth building; no gain says
that merely suppressing endpoint-pose conditioning cannot solve long planning.
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


def apply_gate_strategy(gate, route_angle_deg, strategy):
    """Return a modified gate; route labels never enter the encoded condition."""
    if strategy == 'baseline':
        return gate
    if strategy == 'zero_all':
        return torch.zeros_like(gate)
    if strategy == 'cosine_half':
        factor = 0.5 * (1.0 + torch.cos(torch.deg2rad(route_angle_deg)))
        return gate * factor.clamp(0.0, 1.0)
    if strategy == 'cosine_positive':
        factor = torch.cos(torch.deg2rad(route_angle_deg)).clamp(0.0, 1.0)
        return gate * factor
    if strategy.startswith('zero_ge_'):
        threshold = float(strategy.removeprefix('zero_ge_'))
        return torch.where(route_angle_deg >= threshold, 0.0, gate)
    raise ValueError(f'unknown gate strategy {strategy!r}')


def _summarize(records, strategies):
    groups = {
        'all': records,
        'revisit': [row for row in records if row['is_revisit']],
        '3leg_goal_c': [
            row for row in records
            if row['is_3leg'] and row['goal_label'] == 'C'
        ],
        'route_angle_ge_45': [
            row for row in records if row['route_angle_deg'] >= 45.0
        ],
    }
    result = {}
    for group_name, rows in groups.items():
        group = {'rows': len(rows), 'strategies': {}}
        if not rows:
            result[group_name] = group
            continue
        baseline = np.asarray(
            [row['strategies']['baseline']['action_mse'] for row in rows]
        )
        for strategy in strategies:
            values = np.asarray(
                [row['strategies'][strategy]['action_mse'] for row in rows]
            )
            delta = values - baseline
            rng = np.random.default_rng(0)
            bootstrap = delta[
                rng.integers(0, len(delta), size=(20000, len(delta)))
            ].mean(-1)
            group['strategies'][strategy] = {
                'action_mse': float(values.mean()),
                'paired_delta_vs_baseline': float(delta.mean()),
                'paired_delta_ci95': np.quantile(
                    bootstrap, [0.025, 0.975]
                ).tolist(),
                'fraction_better': float(np.mean(delta < 0)),
            }
        result[group_name] = group
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
    parser.add_argument(
        '--thresholds', type=int, nargs='+', default=(30, 45, 60, 90)
    )
    args = parser.parse_args()
    if any(not 0 <= value <= 180 for value in args.thresholds):
        parser.error('--thresholds must lie in [0, 180]')
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
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
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
    strategies = ['baseline', 'zero_all', 'cosine_half', 'cosine_positive']
    strategies.extend(f'zero_ge_{value}' for value in args.thresholds)
    records = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            condition = model.prepare_condition(batch)
            device = next(model.parameters()).device
            labels = batch['batch_labels'].to(device)
            angles = batch['batch_decision_route_angle_deg'].to(device)
            paired_seed = args.diffusion_seed + 3 * batch_index
            initial_noise = torch.randn(
                labels.shape,
                device=device,
                generator=_cuda_generator(device, paired_seed),
            )
            outputs = {}
            for strategy in strategies:
                changed = dict(condition)
                changed['effective_revisit_gate'] = apply_gate_strategy(
                    condition['effective_revisit_gate'], angles, strategy
                )
                actions = model.sample_actions_from_condition(
                    changed,
                    initial_noise=initial_noise,
                    generator=_cuda_generator(device, paired_seed + 1),
                )
                outputs[strategy] = {
                    'mse': (actions - labels).square().mean(dim=(1, 2)).cpu(),
                    'gate': changed['effective_revisit_gate'].cpu(),
                }
            for row, identity in enumerate(batch['sample_identities']):
                records.append({
                    'sample_identity': identity,
                    'goal_label': batch['goal_labels'][row],
                    'is_revisit': bool(batch['batch_is_revisit'][row] > 0.5),
                    'is_3leg': 'mp3d_3leg/' in identity,
                    'route_angle_deg': float(angles[row].cpu()),
                    'strategies': {
                        strategy: {
                            'action_mse': float(outputs[strategy]['mse'][row]),
                            'effective_gate': float(outputs[strategy]['gate'][row]),
                        }
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
            'strategies': strategies,
            'oracle_future_route_labels_used_for_gate_only': True,
            'encoded_visual_condition_held_fixed': True,
            'paired_ddpm_randomness': True,
        },
        'summary': _summarize(records, strategies),
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
