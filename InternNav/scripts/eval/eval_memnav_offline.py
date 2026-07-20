"""Fixed-split offline checkpoint diagnostics for the current MemNav model.

This measures the training objective and failure decomposition; it is not a
closed-loop Habitat navigation benchmark.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from internnav.dataset.memnav_dataset_lerobot import (
    MemNav_Dataset,
    build_fixed_memnav_eval_subset,
    memnav_collate_fn,
)
from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy
from internnav.model.basemodel.memnav.route_sketch import route_direction_targets
from internnav.model.basemodel.memnav.metrics import (
    attach_full_diffusion_records,
    compute_memnav_batch_records,
    summarize_memnav_records,
)
from scripts.train.configs.memnav import memnav_exp_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='memnav.ckpt or checkpoint directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--root-dir', default=None)
    parser.add_argument('--feature-root', default=None)
    parser.add_argument('--lingbot-repo', default=None)
    parser.add_argument('--lingbot-weights', default=None)
    parser.add_argument('--data-split', choices=('train', 'val', 'all'), default='val')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--sampling-seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument(
        '--selection-indices',
        help='comma-separated explicit dataset indices; overrides --max-samples',
    )
    parser.add_argument(
        '--subset-mode',
        choices=('balanced-fixed', 'random'),
        default='balanced-fixed',
        help='selection rule when --max-samples is smaller than the dataset',
    )
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--oracle-positive', action='store_true')
    parser.add_argument(
        '--retrieval-anchor-mode', choices=('projected', 'raw'),
        default='projected',
        help='evaluation-only anchor selector; rank-loss logits remain unchanged',
    )
    parser.add_argument(
        '--anchor-margin-override', type=int,
        help=(
            'evaluation-only earliest candidate/goal-insertion frame; leaves '
            'the initial scale block intact'
        ),
    )
    parser.add_argument(
        '--full-diffusion-goal-shuffle',
        action='store_true',
        help=(
            'run complete paired DDPM sampling for correct and cyclically shuffled '
            'goal images; substantially slower than the training-noise diagnostic'
        ),
    )
    parser.add_argument('--diffusion-seed', type=int, default=104729)
    parser.add_argument('--save-per-sample', action='store_true')
    return parser.parse_args()


def _git_commit():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def _checkpoint_file(path):
    path = Path(path)
    return path / 'memnav.ckpt' if path.is_dir() else path


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location='cpu')


def _validate_route_checkpoint_metadata(model, checkpoint, state):
    """Reject version/horizon ambiguity for checkpoints that contain route state."""
    route_prefix = 'core.route_sketch.'
    has_route_state = any(key.startswith(route_prefix) for key in state)
    if not has_route_state:
        return None
    metadata_path = checkpoint.parent / 'memnav_metadata.json'
    if not metadata_path.is_file():
        raise ValueError(
            'Route checkpoint is missing memnav_metadata.json; cannot verify code version'
        )
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    route_module = model.core.route_sketch
    if route_module is None:
        raise ValueError(
            'Checkpoint contains route-sketch state but the evaluator model has '
            'route sketch disabled'
        )
    expected_code = getattr(route_module, 'CODE_VERSION', None)
    if metadata.get('route_sketch_code') != expected_code:
        raise ValueError(
            'Route-sketch version mismatch: '
            f"{metadata.get('route_sketch_code')!r} != {expected_code!r}"
        )
    saved_horizons = (
        metadata.get('training_objective', {}).get('route_horizons')
    )
    expected_horizons = list(route_module.horizons)
    if saved_horizons != expected_horizons:
        raise ValueError(
            f'Route horizon mismatch: {saved_horizons!r} != {expected_horizons!r}'
        )
    return metadata


def load_checkpoint(config, checkpoint):
    checkpoint = _checkpoint_file(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = MemNavPolicy(MemNavModelConfig(model_cfg=config.model_dump()))
    state = _torch_load(checkpoint)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    _validate_route_checkpoint_metadata(model, checkpoint, state)
    state = model.upgrade_checkpoint_state_dict(state)
    current = model.state_dict()
    unexpected = [key for key in state if key not in current]
    mismatched = [
        key for key, value in state.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    ]
    missing = [key for key in current if 'lingbot.' not in key and key not in state]
    if unexpected or mismatched or missing:
        raise RuntimeError(
            f'checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]} '
            f'mismatched={mismatched[:8]}'
        )
    model.load_state_dict(state, strict=False)
    print(f'[checkpoint] loaded {checkpoint} ({len(state)} non-LingBot tensors)')
    return model, checkpoint


def _goal_derangement(batch_size):
    if batch_size < 2:
        raise ValueError(
            'full-diffusion goal shuffle needs at least two samples per batch; '
            'increase --batch-size or evaluate more samples'
        )
    return torch.roll(torch.arange(batch_size), shifts=-1)


def _shuffle_goal_condition(batch, permutation):
    """Change only goal inputs; current observation, memory, and GT stay fixed."""
    shuffled = dict(batch)
    for key in ('batch_goal_image', 'batch_goal_cls'):
        value = batch.get(key)
        if value is not None:
            shuffled[key] = value.index_select(0, permutation.to(value.device))
    return shuffled


def _cuda_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _dataset_cache_contract(config):
    """Return the audited cache contract instead of silently weakening it.

    Offline evaluation used to hard-code strict file coverage while dropping the
    version/signature fields from ``memnav_exp_cfg``.  A run could therefore read
    a complete but stale cache generation even when the Slurm environment asked
    for a specific versioned cache.  Keep this helper small and independently
    testable because evaluator provenance is part of the experiment result.
    """
    return {
        'strict_feature_coverage': bool(
            getattr(config.il, 'strict_feature_coverage', True)
        ),
        'require_versioned_cache': bool(
            getattr(config.il, 'require_versioned_cache', False)
        ),
        'expected_cache_signature': str(
            getattr(config.il, 'expected_cache_signature', '') or ''
        ),
        'require_generated_pose_convention': bool(
            getattr(config.il, 'require_generated_pose_convention', False)
        ),
    }


def _attach_route_sketch_records(records, outputs, batch, horizons):
    """Attach label-side route diagnostics without changing policy inputs."""
    prediction = outputs.get('route_direction')
    if prediction is None:
        return False
    target, valid = route_direction_targets(
        batch['batch_labels'].to(prediction.device), horizons
    )
    if prediction.shape != target.shape or len(records) != prediction.shape[0]:
        raise ValueError('route prediction, target and record shapes must agree')
    cosine = (prediction * target).sum(dim=-1).clamp(-1.0, 1.0)
    error_deg = torch.rad2deg(torch.arccos(cosine))
    raw_norm = outputs['route_raw_direction_norm']
    residual_scale = outputs['route_residual_scale']
    curvature_gate = outputs['route_curvature_gate']
    for row_index, record in enumerate(records):
        for horizon_index, horizon in enumerate(horizons):
            prefix = f'route_h{int(horizon)}'
            is_valid = bool(valid[row_index, horizon_index].item())
            record[f'{prefix}_valid'] = is_valid
            record[f'{prefix}_pred_x'] = float(
                prediction[row_index, horizon_index, 0].item()
            )
            record[f'{prefix}_pred_y'] = float(
                prediction[row_index, horizon_index, 1].item()
            )
            record[f'{prefix}_target_x'] = float(
                target[row_index, horizon_index, 0].item()
            )
            record[f'{prefix}_target_y'] = float(
                target[row_index, horizon_index, 1].item()
            )
            record[f'{prefix}_error_deg'] = (
                float(error_deg[row_index, horizon_index].item())
                if is_valid else None
            )
            record[f'{prefix}_raw_norm'] = float(
                raw_norm[row_index, horizon_index].item()
            )
            record[f'{prefix}_residual_scale'] = float(
                residual_scale[horizon_index].item()
            )
        record['route_curvature_gate'] = float(curvature_gate[row_index].item())
    return True


def _summarize_route_sketch(records, horizons, enabled):
    summary = {'enabled': bool(enabled), 'horizons': list(horizons)}
    if not enabled:
        return summary

    groups = {
        'all': lambda record: True,
        'hard_turn': lambda record: bool(record.get('decision_curriculum_hard')),
        'goal_c': lambda record: record.get('goal_label') == 'C',
    }
    summary['groups'] = {}
    for group_name, predicate in groups.items():
        group = {}
        for horizon in horizons:
            key = f'route_h{int(horizon)}_error_deg'
            values = [
                float(record[key]) for record in records
                if predicate(record) and record.get(key) is not None
            ]
            group[f'h{int(horizon)}_error_deg'] = (
                sum(values) / len(values) if values else None
            )
            group[f'h{int(horizon)}_valid_count'] = len(values)
        summary['groups'][group_name] = group
        gate_values = [
            float(record['route_curvature_gate'])
            for record in records if predicate(record)
        ]
        group['curvature_gate'] = (
            sum(gate_values) / len(gate_values) if gate_values else None
        )
    summary['residual_scale'] = {
        f'h{int(horizon)}': records[0][f'route_h{int(horizon)}_residual_scale']
        for horizon in horizons
    }
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('MemNav evaluation requires CUDA for the LingBot front-end')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = copy.deepcopy(memnav_exp_cfg)
    if args.lingbot_repo:
        config.il.lingbot_repo = args.lingbot_repo
    if args.lingbot_weights:
        config.il.lingbot_weights = args.lingbot_weights
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
    original_anchor_margins = sorted({
        int(sample['amargin']) for sample in dataset.samples
    })
    if args.anchor_margin_override is not None:
        if args.anchor_margin_override < config.il.num_scale:
            raise ValueError(
                '--anchor-margin-override must be at least num_scale='
                f'{config.il.num_scale}'
            )
        for sample in dataset.samples:
            sample['amargin'] = int(args.anchor_margin_override)
        dataset.dataset_fingerprint = hashlib.sha256(
            (
                f'{dataset.dataset_fingerprint}\n'
                f'anchor_margin_override={args.anchor_margin_override}'
            ).encode('utf-8')
        ).hexdigest()
    dataset_size = len(dataset)
    selection_indices = list(range(dataset_size))
    selection_mode = 'full'
    if args.selection_indices:
        selection_indices = [
            int(value) for value in args.selection_indices.split(',') if value
        ]
        if (
            not selection_indices
            or len(set(selection_indices)) != len(selection_indices)
            or min(selection_indices) < 0
            or max(selection_indices) >= dataset_size
        ):
            raise ValueError(
                '--selection-indices must be unique valid dataset indices'
            )
        eval_dataset = Subset(dataset, selection_indices)
        subset_manifest = (
            f'{dataset.dataset_fingerprint}\n'
            + ','.join(str(index) for index in selection_indices)
        )
        eval_dataset.dataset_fingerprint = hashlib.sha256(
            subset_manifest.encode('utf-8')
        ).hexdigest()
        selection_mode = 'explicit'
        print(
            f'[subset] explicit: {len(eval_dataset)}/{dataset_size}; '
            f'fingerprint={eval_dataset.dataset_fingerprint}'
        )
    elif 0 < args.max_samples < dataset_size:
        if args.subset_mode == 'balanced-fixed':
            eval_dataset = build_fixed_memnav_eval_subset(
                dataset, args.max_samples, selection_seed=args.seed
            )
            selection_indices = eval_dataset.memnav_selection_indices
            print(
                f'[subset] balanced fixed: {len(eval_dataset)}/{dataset_size}; '
                f'revisit={eval_dataset.memnav_num_revisit}, '
                f'novel={eval_dataset.memnav_num_novel}; '
                f'fingerprint={eval_dataset.dataset_fingerprint}'
            )
            selection_mode = 'balanced-fixed'
        else:
            rng = np.random.default_rng(args.seed)
            selection_indices = sorted(
                rng.choice(dataset_size, args.max_samples, replace=False).tolist()
            )
            eval_dataset = Subset(dataset, selection_indices)
            subset_manifest = (
                f'{dataset.dataset_fingerprint}\n'
                + ','.join(str(index) for index in selection_indices)
            )
            eval_dataset.dataset_fingerprint = hashlib.sha256(
                subset_manifest.encode('utf-8')
            ).hexdigest()
            selection_mode = 'random'
    else:
        eval_dataset = dataset
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
    route_module = model.core.route_sketch
    route_horizons = tuple(route_module.horizons) if route_module is not None else ()
    route_enabled = False
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            eval_batch = dict(batch)
            eval_batch['diagnostic_retrieval_anchor_mode'] = (
                args.retrieval_anchor_mode
            )
            if args.anchor_margin_override is not None:
                eval_batch['diagnostic_anchor_min_frame'] = int(
                    args.anchor_margin_override
                )
            condition = None
            if args.full_diffusion_goal_shuffle:
                condition = model.prepare_condition(eval_batch)
                outputs = model.forward_with_condition(eval_batch, condition)
            else:
                outputs = model(eval_batch)
            oracle_outputs = None
            if args.oracle_positive:
                oracle_batch = dict(eval_batch)
                oracle_batch['diagnostic_oracle_positive'] = True
                oracle_batch['diagnostic_noise'] = outputs['noise']
                oracle_batch['diagnostic_timesteps'] = outputs['timesteps']
                oracle_outputs = model(oracle_batch)
            batch_records = compute_memnav_batch_records(outputs, batch, oracle_outputs)
            route_enabled = (
                _attach_route_sketch_records(
                    batch_records, outputs, batch, route_horizons
                ) or route_enabled
            )
            if args.full_diffusion_goal_shuffle:
                batch_size = batch['batch_labels'].shape[0]
                permutation = _goal_derangement(batch_size)
                shuffled_batch = _shuffle_goal_condition(eval_batch, permutation)
                shuffled_condition = model.prepare_condition(shuffled_batch)
                device = next(model.parameters()).device
                paired_seed = args.diffusion_seed + 3 * batch_index
                initial_noise = torch.randn(
                    batch['batch_labels'].shape,
                    device=device,
                    generator=_cuda_generator(device, paired_seed),
                )
                sampled_actions = model.sample_actions_from_condition(
                    condition,
                    initial_noise=initial_noise,
                    generator=_cuda_generator(device, paired_seed + 1),
                )
                shuffled_actions = model.sample_actions_from_condition(
                    shuffled_condition,
                    initial_noise=initial_noise,
                    generator=_cuda_generator(device, paired_seed + 1),
                )
                attach_full_diffusion_records(
                    batch_records,
                    sampled_actions,
                    shuffled_actions,
                    batch,
                    permutation,
                )
            for record in batch_records:
                record['sample_index'] = len(records)
                records.append(record)
            if batch_index % args.log_every == 0 or batch_index + 1 == len(loader):
                print(f'[eval] batch={batch_index + 1}/{len(loader)} samples={len(records)}')

    elapsed = time.time() - started
    result = {
        'evaluation_type': 'fixed-offline-training-diagnostic',
        'closed_loop_navigation': False,
        'checkpoint': str(checkpoint.resolve()),
        'git_commit': _git_commit(),
        'root_dir': root_dir,
        'feature_root': feature_root,
        'cache_contract': cache_contract,
        'data_split': args.data_split,
        'validation_fraction': args.validation_fraction,
        'split_seed': args.split_seed,
        'sampling_mode': 'fixed_leg',
        'sampling_seed': args.sampling_seed,
        'retrieval_anchor_mode': args.retrieval_anchor_mode,
        'original_anchor_margins': original_anchor_margins,
        'anchor_margin_override': args.anchor_margin_override,
        'random_seed': args.seed,
        'dataset_fingerprint': dataset.dataset_fingerprint,
        'dataset_size': dataset_size,
        'subset_mode': selection_mode,
        'selection_indices': selection_indices,
        'eval_dataset_fingerprint': getattr(
            eval_dataset, 'dataset_fingerprint', dataset.dataset_fingerprint
        ),
        'evaluated_samples': len(records),
        'oracle_positive': args.oracle_positive,
        'full_diffusion_goal_shuffle': args.full_diffusion_goal_shuffle,
        'diffusion_seed': args.diffusion_seed,
        'goal_shuffle_scope': (
            'within_batch_cyclic_derangement'
            if args.full_diffusion_goal_shuffle else None
        ),
        'paired_diffusion_randomness': bool(args.full_diffusion_goal_shuffle),
        'elapsed_seconds': elapsed,
        'samples_per_second': len(records) / elapsed,
        'peak_cuda_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
        'metrics': summarize_memnav_records(records),
        'route_sketch': _summarize_route_sketch(
            records, route_horizons, route_enabled
        ),
    }
    if args.save_per_sample:
        result['per_sample'] = records
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f'[eval] wrote {output}')


if __name__ == '__main__':
    main()
