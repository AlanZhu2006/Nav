"""Offline upper bound for NavDP-style stochastic candidate selection.

The policy receives exactly the normal MemNav inputs and generates ``N`` complete
DDPM action samples.  Ground truth is used only after generation to report the
best-of-N action MSE.  This is therefore a diagnostic upper bound for a future
candidate ranker, not an inference algorithm and not closed-loop navigation.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from internnav.dataset.memnav_dataset_lerobot import (
    MemNav_Dataset,
    build_fixed_memnav_eval_subset,
    memnav_collate_fn,
)
from internnav.model.basemodel.memnav.metrics import compute_memnav_batch_records
from scripts.eval.eval_memnav_offline import (
    _dataset_cache_contract,
    _git_commit,
    load_checkpoint,
)
from scripts.train.configs.memnav import memnav_exp_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root-dir', default=None)
    parser.add_argument('--feature-root', default=None)
    parser.add_argument('--data-split', choices=('train', 'val', 'all'), default='val')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--sampling-seed', type=int, default=0)
    parser.add_argument('--candidate-count', type=int, default=8)
    parser.add_argument('--candidate-chunk', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=28)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--selection-seed', type=int, default=0)
    parser.add_argument('--diffusion-seed', type=int, default=104729)
    parser.add_argument('--expected-dataset-fingerprint', default='')
    parser.add_argument('--expected-eval-fingerprint', default='')
    parser.add_argument('--save-per-sample', action='store_true')
    return parser.parse_args()


def _cuda_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _repeat_sampling_condition(condition, repeats):
    """Repeat only the tensors consumed by ``sample_actions_from_condition``."""
    keys = ('current_state', 'revisit', 'novel', 'effective_revisit_gate')
    return {
        key: condition[key].repeat_interleave(int(repeats), dim=0)
        for key in keys
    }


def sample_action_candidates(model, condition, count, chunk, seed):
    """Generate ``[B, N, T, 3]`` actions without recomputing LingBot features."""
    if int(count) < 1 or int(chunk) < 1:
        raise ValueError('candidate count and chunk must be positive')
    batch_size = condition['current_state'].shape[0]
    device = condition['current_state'].device
    predict_size = int(model.core.predict_size)
    pieces = []
    offset = 0
    while offset < count:
        width = min(int(chunk), int(count) - offset)
        repeated = _repeat_sampling_condition(condition, width)
        initial = torch.randn(
            batch_size * width,
            predict_size,
            3,
            device=device,
            generator=_cuda_generator(device, seed + 2 * offset),
        )
        flat = model.sample_actions_from_condition(
            repeated,
            initial_noise=initial,
            generator=_cuda_generator(device, seed + 2 * offset + 1),
        )
        pieces.append(flat.reshape(batch_size, width, predict_size, 3))
        offset += width
    return torch.cat(pieces, dim=1)


def attach_candidate_metrics(records, actions, target):
    """Attach per-row random-candidate and oracle-best scalar diagnostics."""
    target = target.to(actions.device)[:, None]
    if actions.ndim != 4 or actions.shape[0] != target.shape[0]:
        raise ValueError('candidate actions must have shape [B,N,T,3]')
    if tuple(actions.shape[2:]) != tuple(target.shape[2:]):
        raise ValueError('candidate action horizon must match target')
    if len(records) != actions.shape[0]:
        raise ValueError('record count must match candidate batch size')

    square = (actions - target).square()
    mse = square.mean(dim=(2, 3))
    axis_mse = square.mean(dim=2)
    mean_action = actions.mean(dim=1, keepdim=True)
    diversity = (actions - mean_action).square().mean(dim=(1, 2, 3)).sqrt()
    candidate_count = actions.shape[1]
    cutoffs = [n for n in (1, 2, 4, 8, 16, 32) if n <= candidate_count]

    for row, record in enumerate(records):
        values = mse[row]
        best_value, best_index = values.min(dim=0)
        record.update({
            'candidate_count': int(candidate_count),
            'candidate_first_mse': float(values[0].item()),
            'candidate_mean_mse': float(values.mean().item()),
            'candidate_median_mse': float(values.median().item()),
            'candidate_best_mse': float(best_value.item()),
            'candidate_best_index': int(best_index.item()),
            'candidate_worst_mse': float(values.max().item()),
            'candidate_diversity_rmse': float(diversity[row].item()),
            'candidate_oracle_reduction_vs_mean': float(
                (1.0 - best_value / values.mean().clamp_min(1e-12)).item()
            ),
        })
        for cutoff in cutoffs:
            record[f'candidate_best_of_{cutoff}_mse'] = float(
                values[:cutoff].min().item()
            )
        for axis, name in enumerate(('x', 'y', 'theta')):
            record[f'candidate_mean_mse_{name}'] = float(
                axis_mse[row, :, axis].mean().item()
            )
            record[f'candidate_best_mse_{name}'] = float(
                axis_mse[row, best_index, axis].item()
            )
    return records


def _mean(records, key):
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize_candidate_group(records):
    if not records:
        return {'count': 0}
    candidate_count = int(records[0]['candidate_count'])
    result = {
        'count': len(records),
        'candidate_first_mse': _mean(records, 'candidate_first_mse'),
        'candidate_mean_mse': _mean(records, 'candidate_mean_mse'),
        'candidate_best_mse': _mean(records, 'candidate_best_mse'),
        'candidate_diversity_rmse': _mean(records, 'candidate_diversity_rmse'),
        'oracle_reduction_vs_group_mean': None,
        'rows_oracle_better_than_half_mean': sum(
            row['candidate_best_mse'] <= 0.5 * row['candidate_mean_mse']
            for row in records
        ),
    }
    if result['candidate_mean_mse']:
        result['oracle_reduction_vs_group_mean'] = (
            1.0 - result['candidate_best_mse'] / result['candidate_mean_mse']
        )
    for cutoff in (1, 2, 4, 8, 16, 32):
        key = f'candidate_best_of_{cutoff}_mse'
        if cutoff <= candidate_count:
            result[key] = _mean(records, key)
    for axis in ('x', 'y', 'theta'):
        result[f'candidate_mean_mse_{axis}'] = _mean(
            records, f'candidate_mean_mse_{axis}'
        )
        result[f'candidate_best_mse_{axis}'] = _mean(
            records, f'candidate_best_mse_{axis}'
        )
    return result


def _build_groups(records):
    return {
        'all': records,
        'hard_turn': [r for r in records if r.get('decision_curriculum_hard')],
        'span_ge_128': [r for r in records if r['remaining_path_span'] >= 128],
        'span_ge_256': [r for r in records if r['remaining_path_span'] >= 256],
        'three_leg_goal_c': [r for r in records if r.get('goal_label') == 'C'],
        'three_leg_goal_c_revisit': [
            r for r in records
            if r.get('goal_label') == 'C' and r.get('is_revisit')
        ],
        'two_leg': [
            r for r in records if 'mp3d_2leg' in (r.get('cache_path') or '')
        ],
        'revisit': [r for r in records if r.get('is_revisit')],
        'novel': [r for r in records if not r.get('is_revisit')],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('MemNav candidate evaluation requires CUDA')
    if args.max_samples < 1:
        raise ValueError('--max-samples must be positive')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = copy.deepcopy(memnav_exp_cfg)
    root_dir = args.root_dir or config.il.root_dir
    feature_root = args.feature_root or getattr(config.il, 'feature_root', None)
    cache_contract = _dataset_cache_contract(config)
    dataset = MemNav_Dataset(
        root_dir,
        predict_size=config.il.predict_size,
        image_size=config.il.image_size,
        lingbot_repo=config.il.lingbot_repo,
        feature_root=feature_root,
        window_size=config.il.window_size,
        num_scale=config.il.num_scale,
        **cache_contract,
        data_split=args.data_split,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        sampling_mode='fixed_leg',
        sampling_seed=args.sampling_seed,
    )
    if (
        args.expected_dataset_fingerprint
        and dataset.dataset_fingerprint != args.expected_dataset_fingerprint
    ):
        raise RuntimeError(
            'dataset fingerprint mismatch: '
            f'expected={args.expected_dataset_fingerprint} '
            f'actual={dataset.dataset_fingerprint}'
        )
    eval_dataset = build_fixed_memnav_eval_subset(
        dataset, args.max_samples, selection_seed=args.selection_seed
    )
    if (
        args.expected_eval_fingerprint
        and eval_dataset.dataset_fingerprint != args.expected_eval_fingerprint
    ):
        raise RuntimeError(
            'evaluation fingerprint mismatch: '
            f'expected={args.expected_eval_fingerprint} '
            f'actual={eval_dataset.dataset_fingerprint}'
        )
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=memnav_collate_fn,
    )
    model, checkpoint = load_checkpoint(config, args.checkpoint)
    model.eval()
    records = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            condition = model.prepare_condition(batch)
            outputs = model.forward_with_condition(batch, condition)
            batch_records = compute_memnav_batch_records(outputs, batch)
            actions = sample_action_candidates(
                model,
                condition,
                args.candidate_count,
                args.candidate_chunk,
                args.diffusion_seed + 1009 * batch_index,
            )
            attach_candidate_metrics(
                batch_records, actions, batch['batch_labels']
            )
            for record in batch_records:
                record['sample_index'] = len(records)
                records.append(record)
            print(
                f'[candidate-oracle] batch={batch_index + 1}/{len(loader)} '
                f'samples={len(records)}',
                flush=True,
            )

    groups = _build_groups(records)
    elapsed = time.time() - started
    result = {
        'evaluation_type': 'offline-ddpm-candidate-oracle',
        'closed_loop_navigation': False,
        'oracle_used_at_inference': False,
        'purpose': (
            'Upper-bound screen for whether candidate ranking can help; ground '
            'truth selects the best candidate only after normal policy sampling.'
        ),
        'checkpoint': str(Path(checkpoint).resolve()),
        'git_commit': _git_commit(),
        'root_dir': root_dir,
        'feature_root': feature_root,
        'cache_contract': cache_contract,
        'data_split': args.data_split,
        'validation_fraction': args.validation_fraction,
        'split_seed': args.split_seed,
        'sampling_mode': 'fixed_leg',
        'sampling_seed': args.sampling_seed,
        'dataset_fingerprint': dataset.dataset_fingerprint,
        'eval_dataset_fingerprint': eval_dataset.dataset_fingerprint,
        'selection_indices': eval_dataset.memnav_selection_indices,
        'candidate_count': args.candidate_count,
        'candidate_chunk': args.candidate_chunk,
        'random_seed': args.seed,
        'selection_seed': args.selection_seed,
        'diffusion_seed': args.diffusion_seed,
        'evaluated_samples': len(records),
        'elapsed_seconds': elapsed,
        'peak_cuda_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
        'summary': {
            name: summarize_candidate_group(rows)
            for name, rows in groups.items()
        },
    }
    if args.save_per_sample:
        result['per_sample'] = records
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(result['summary'], indent=2, sort_keys=True))
    print(f'[candidate-oracle] wrote {output}')


if __name__ == '__main__':
    main()
