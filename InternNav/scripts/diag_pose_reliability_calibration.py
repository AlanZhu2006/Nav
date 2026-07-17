#!/usr/bin/env python3
"""Collect real MemNav reliability cues/targets for offline calibration.

The frozen LingBot path is evaluated exactly as training sees it, including an
oracle-positive anchor by default.  Only observable reliability features and a
training-only bearing-quality target are saved; no repository cache is changed.
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
from torch.utils.data import DataLoader


SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parents[2]
MOTHER = Path('/home/asus/Research/Nav')
if os.fspath(SCRIPT.parents[1]) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT.parents[1]))

from internnav.dataset.memnav_dataset_lerobot import (  # noqa: E402
    MemNav_Dataset,
    memnav_collate_fn,
)
from internnav.model.basemodel.memnav.memnav_policy import (  # noqa: E402
    MemNavModelConfig,
    MemNavPolicy,
)
from internnav.model.basemodel.memnav.revisit_pose import (  # noqa: E402
    GaugeInvariantRevisitPose,
)
from scripts.train.configs.memnav import memnav_exp_cfg  # noqa: E402


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _load_model(args) -> MemNavPolicy:
    config = copy.deepcopy(memnav_exp_cfg)
    config.il.root_dir = args.root_dir
    config.il.feature_root = args.feature_root
    config.il.lingbot_repo = args.lingbot_repo
    config.il.lingbot_weights = args.lingbot_weights
    config.il.novel_backbone_weights = args.dino_weights
    config.il.window_size = args.window
    config.il.num_scale = args.num_scale
    config.il.max_frame_num = args.max_frame_num
    config.il.goal_warm = args.goal_warm
    config.il.require_versioned_cache = args.require_versioned_cache
    model = MemNavPolicy(MemNavModelConfig(model_cfg=config.model_dump()))
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    state = model.upgrade_checkpoint_state_dict(state)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        name for name in incompatible.missing_keys
        if not name.startswith('core.lingbot.')
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f'checkpoint mismatch: missing={missing} '
            f'unexpected={incompatible.unexpected_keys}'
        )
    return model.eval()


def _revisit_rows(batch):
    """Drop dynamic novel rows before the expensive frozen LingBot forward."""
    mask = batch['batch_is_revisit'] > 0.5
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return None
    size = int(mask.numel())
    selected = {}
    for name, value in batch.items():
        if torch.is_tensor(value) and value.ndim and len(value) == size:
            selected[name] = value[indices]
        elif isinstance(value, (list, tuple)) and len(value) == size:
            rows = [value[index] for index in indices]
            selected[name] = tuple(rows) if isinstance(value, tuple) else rows
        else:
            selected[name] = value
    return selected


def collect(args) -> dict:
    dataset = MemNav_Dataset(
        args.root_dir,
        predict_size=24,
        image_size=518,
        lingbot_repo=args.lingbot_repo,
        feature_root=args.feature_root,
        window_size=args.window,
        num_scale=args.num_scale,
        strict_feature_coverage=args.strict_feature_coverage,
        require_versioned_cache=args.require_versioned_cache,
        expected_cache_signature=args.expected_cache_signature,
        require_generated_pose_convention=True,
        add_goalA=args.add_goal_a,
        data_split='all',
        sampling_mode='fixed_leg',
        sampling_seed=0,
    )
    model = _load_model(args)
    core = model.core
    records = []
    for seed in range(args.seeds):
        dataset.sampling_seed = seed
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=memnav_collate_fn,
        )
        for batch_index, batch in enumerate(loader):
            batch = _revisit_rows(batch)
            if batch is None:
                print(
                    f'seed={seed} batch={batch_index + 1}/{len(loader)} '
                    f'revisit_records={len(records)} (no revisit rows)',
                    flush=True,
                )
                continue
            if args.oracle_positive:
                batch['diagnostic_oracle_positive'] = True
            with torch.inference_mode():
                encoded = core.encode_memory(batch)
                (
                    _revisit,
                    _aux_pose,
                    _rotation,
                    raw_direction,
                    reliability,
                    features,
                    _range_steps,
                ) = core.build_revisit(
                    encoded['cur_pose'],
                    encoded['goal_pose'],
                    encoded['pose_context'],
                )
                gt_xy = batch['batch_goal_rel_pose'][..., :2].to(raw_direction.device)
                gt_unit = gt_xy / torch.linalg.vector_norm(
                    gt_xy, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                raw_alignment = (raw_direction * gt_unit).sum(-1).clamp(-1.0, 1.0)
                quality = raw_alignment.clamp(0.0, 1.0)
                direction_error_deg = torch.rad2deg(torch.arccos(raw_alignment))
            for row in range(len(batch['cache_paths'])):
                if not bool(batch['batch_is_revisit'][row] > 0.5):
                    continue
                records.append({
                    'seed': seed,
                    'sample_identity': batch['sample_identities'][row],
                    'goal_label': batch['goal_labels'][row],
                    'cur_step': int(batch['cur_steps'][row]),
                    'anchor_step': int(encoded['anchor_idx'][row].item()),
                    'episode': os.path.relpath(
                        Path(batch['cache_paths'][row]).parents[2], args.feature_root
                    ),
                    'raw_alignment': float(raw_alignment[row].cpu()),
                    'direction_error_deg': float(direction_error_deg[row].cpu()),
                    'quality': float(quality[row].cpu()),
                    'reliability': float(reliability[row].cpu()),
                    'features': features[row].float().cpu().tolist(),
                })
            print(
                f'seed={seed} batch={batch_index + 1}/{len(loader)} '
                f'revisit_records={len(records)}',
                flush=True,
            )
            if args.max_records and len(records) >= args.max_records:
                records = records[:args.max_records]
                break
        if args.max_records and len(records) >= args.max_records:
            break
    if not records:
        raise RuntimeError('no revisit records were collected')
    quality = np.asarray([row['quality'] for row in records], dtype=np.float64)
    predicted = np.asarray([row['reliability'] for row in records], dtype=np.float64)
    direction_error = np.asarray(
        [row['direction_error_deg'] for row in records], dtype=np.float64
    )
    return {
        'metadata': {
            'checkpoint': os.path.abspath(args.checkpoint),
            'root_dir': os.path.abspath(args.root_dir),
            'feature_root': os.path.abspath(args.feature_root),
            'feature_names': list(GaugeInvariantRevisitPose.RELIABILITY_FEATURES),
            'oracle_positive': bool(args.oracle_positive),
            'goal_warm': args.goal_warm,
            'add_goal_a': bool(args.add_goal_a),
            'seeds': args.seeds,
            'records': len(records),
        },
        'summary': {
            'quality_mean': float(quality.mean()),
            'quality_std': float(quality.std()),
            'quality_median': float(np.median(quality)),
            'direction_error_median_deg': float(np.median(direction_error)),
            'direction_error_p90_deg': float(np.percentile(direction_error, 90)),
            'reliability_mean': float(predicted.mean()),
            'reliability_std': float(predicted.std()),
            'brier': float(np.mean((predicted - quality) ** 2)),
            'correlation': (
                float(np.corrcoef(predicted, quality)[0, 1])
                if predicted.std() > 0 and quality.std() > 0 else None
            ),
        },
        'records': records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lingbot-repo', required=True)
    parser.add_argument('--lingbot-weights', required=True)
    parser.add_argument('--dino-weights', required=True)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--num-scale', type=int, default=8)
    parser.add_argument('--max-frame-num', type=int, default=4096)
    parser.add_argument('--goal-warm', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--max-records', type=int, default=0)
    parser.add_argument('--add-goal-a', action='store_true')
    parser.add_argument('--strict-feature-coverage', action='store_true')
    parser.add_argument('--require-versioned-cache', action='store_true')
    parser.add_argument('--expected-cache-signature', default='')
    parser.add_argument(
        '--oracle-positive', action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    output = Path(args.output)
    if _inside(output, MOTHER) or not _inside(output, WORKTREE):
        raise RuntimeError(
            f'output must be inside personal worktree {WORKTREE}, not mother Nav: {output}'
        )
    result = collect(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result['summary'], indent=2, sort_keys=True))
    print(f'REPORT_SAVED {output}')


if __name__ == '__main__':
    main()
