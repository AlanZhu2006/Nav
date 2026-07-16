"""Fixed-split offline checkpoint diagnostics for the current MemNav model.

This measures the training objective and failure decomposition; it is not a
closed-loop Habitat navigation benchmark.
"""

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn
from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy
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
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--oracle-positive', action='store_true')
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


def load_checkpoint(config, checkpoint):
    checkpoint = _checkpoint_file(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = MemNavPolicy(MemNavModelConfig(model_cfg=config.model_dump()))
    state = _torch_load(checkpoint)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
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
    dataset = MemNav_Dataset(
        root_dir,
        predict_size=config.il.predict_size,
        image_size=config.il.image_size,
        lingbot_repo=config.il.lingbot_repo,
        feature_root=feature_root,
        window_size=config.il.window_size,
        num_scale=config.il.num_scale,
        strict_feature_coverage=True,
        require_generated_pose_convention=getattr(
            config.il, 'require_generated_pose_convention', False
        ),
        data_split=args.data_split,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        sampling_mode='fixed_leg',
        sampling_seed=args.sampling_seed,
    )
    dataset_size = len(dataset)
    if 0 < args.max_samples < dataset_size:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(dataset_size, args.max_samples, replace=False).tolist())
        eval_dataset = Subset(dataset, indices)
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
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            condition = None
            if args.full_diffusion_goal_shuffle:
                condition = model.prepare_condition(batch)
                outputs = model.forward_with_condition(batch, condition)
            else:
                outputs = model(batch)
            oracle_outputs = None
            if args.oracle_positive:
                oracle_batch = dict(batch)
                oracle_batch['diagnostic_oracle_positive'] = True
                oracle_batch['diagnostic_noise'] = outputs['noise']
                oracle_batch['diagnostic_timesteps'] = outputs['timesteps']
                oracle_outputs = model(oracle_batch)
            batch_records = compute_memnav_batch_records(outputs, batch, oracle_outputs)
            if args.full_diffusion_goal_shuffle:
                batch_size = batch['batch_labels'].shape[0]
                permutation = _goal_derangement(batch_size)
                shuffled_batch = _shuffle_goal_condition(batch, permutation)
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
        'data_split': args.data_split,
        'validation_fraction': args.validation_fraction,
        'split_seed': args.split_seed,
        'sampling_mode': 'fixed_leg',
        'sampling_seed': args.sampling_seed,
        'random_seed': args.seed,
        'dataset_fingerprint': dataset.dataset_fingerprint,
        'dataset_size': dataset_size,
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
