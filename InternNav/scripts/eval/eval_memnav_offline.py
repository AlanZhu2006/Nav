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
from internnav.model.basemodel.memnav.metrics import (
    compute_memnav_batch_totals,
    finalize_memnav_metrics,
    merge_memnav_totals,
)
from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy
from scripts.train.configs.memnav import memnav_exp_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Offline diagnostic evaluation for a MemNav checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root-dir', default=None)
    parser.add_argument('--feature-root', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=0, help='0 evaluates the full dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--dataset-role', default='train-diagnostic')
    return parser.parse_args()


def load_checkpoint(config, checkpoint):
    model_cfg = MemNavModelConfig(model_cfg=config.model_dump())
    model = MemNavPolicy(model_cfg)
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith('core.lingbot.')]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f'checkpoint mismatch: missing={bad_missing[:10]} '
            f'unexpected={incompatible.unexpected_keys[:10]}'
        )
    print(
        f'[checkpoint] loaded {checkpoint}; '
        f'expected frozen-LingBot omissions={len(incompatible.missing_keys)}'
    )
    return model


def git_commit():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('MemNav offline evaluation requires CUDA')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = copy.deepcopy(memnav_exp_cfg)
    root_dir = args.root_dir or config.il.root_dir
    feature_root = args.feature_root or getattr(config.il, 'feature_root', None)
    config.il.root_dir = root_dir
    config.il.feature_root = feature_root

    dataset = MemNav_Dataset(
        root_dir,
        predict_size=config.il.predict_size,
        image_size=config.il.image_size,
        lingbot_repo=config.il.lingbot_repo,
        feature_root=feature_root,
        window_size=config.il.window_size,
        num_scale=config.il.num_scale,
    )
    dataset_size = len(dataset)
    if args.max_samples > 0 and args.max_samples < dataset_size:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(dataset_size, size=args.max_samples, replace=False)).tolist()
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
    model = load_checkpoint(config, args.checkpoint)
    model.eval()

    totals = {}
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            outputs = model(batch)
            merge_memnav_totals(totals, compute_memnav_batch_totals(outputs, batch))
            if batch_index % args.log_every == 0 or batch_index + 1 == len(loader):
                print(
                    f'[eval] batch={batch_index + 1}/{len(loader)} '
                    f'samples={totals["num_samples"]}'
                )

    elapsed = time.time() - start
    metrics = finalize_memnav_metrics(
        totals,
        w_retrieval=config.il.w_retrieval,
        w_aux_pose=config.il.w_aux_pose,
    )
    result = {
        'evaluation_type': 'offline-checkpoint-diagnostic',
        'dataset_role': args.dataset_role,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'root_dir': root_dir,
        'feature_root': feature_root,
        'git_commit': git_commit(),
        'seed': args.seed,
        'dataset_size': dataset_size,
        'evaluated_samples': len(eval_dataset),
        'batch_size': args.batch_size,
        'elapsed_seconds': elapsed,
        'samples_per_second': len(eval_dataset) / elapsed,
        'peak_cuda_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
        'metrics': metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'[eval] wrote {output}')


if __name__ == '__main__':
    main()
