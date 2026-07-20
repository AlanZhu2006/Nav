#!/usr/bin/env python3
"""Compare inference-only MemNav retrieval strategies on fixed labelled rows.

This diagnostic deliberately skips LingBot's expensive GCT pose path.  It keeps
the checkpoint, frozen DINO features, candidate masks, and retrieval labels fixed,
then changes only how a historical anchor is selected.  The result is an offline
retrieval diagnostic, not a navigation score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parents[3]
if os.fspath(SCRIPT.parents[2]) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT.parents[2]))

from internnav.dataset.memnav_dataset_lerobot import (  # noqa: E402
    MemNav_Dataset,
    memnav_collate_fn,
)
from internnav.model.basemodel.memnav.memnav_policy import (  # noqa: E402
    MemNavModelConfig,
    MemNavPolicy,
)
from scripts.train.configs.memnav import memnav_exp_cfg  # noqa: E402


def masked_argmax(scores: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Argmax over candidate entries only."""
    if scores.shape != candidate.shape:
        raise ValueError('scores and candidate must have the same shape')
    if not bool(candidate.any(-1).all()):
        raise ValueError('every row must contain at least one candidate')
    floor = torch.finfo(scores.dtype).min
    return scores.masked_fill(~candidate, floor).argmax(-1)


def temporal_mass_anchor(
    logits: torch.Tensor,
    candidate: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Select the centre of the highest-probability temporal neighbourhood.

    Unlike single-frame argmax, this rewards a contiguous band of moderately
    similar frames.  It is fully inference-safe: only retrieval logits and the
    structural candidate mask are consumed.
    """
    if radius < 1:
        raise ValueError('radius must be positive')
    if logits.shape != candidate.shape:
        raise ValueError('logits and candidate must have the same shape')
    if not bool(candidate.any(-1).all()):
        raise ValueError('every row must contain at least one candidate')
    floor = torch.finfo(logits.dtype).min
    probability = torch.softmax(logits.masked_fill(~candidate, floor).float(), -1)
    kernel = probability.new_ones(1, 1, 2 * radius + 1)
    support = F.conv1d(
        probability[:, None], kernel, padding=radius
    ).squeeze(1)
    support = support.masked_fill(~candidate, -1.0)
    return support.argmax(-1)


def candidate_zscore(
    scores: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Standardize scores per row over candidates without leaking labels."""
    if scores.shape != candidate.shape:
        raise ValueError('scores and candidate must have the same shape')
    count = candidate.sum(-1, keepdim=True)
    if not bool((count > 0).all()):
        raise ValueError('every row must contain at least one candidate')
    safe = scores.masked_fill(~candidate, 0.0)
    mean = safe.sum(-1, keepdim=True) / count
    variance = (
        (scores - mean).square().masked_fill(~candidate, 0.0).sum(-1, keepdim=True)
        / count
    )
    standardized = (scores - mean) / variance.sqrt().clamp_min(1e-6)
    return standardized.masked_fill(~candidate, torch.finfo(scores.dtype).min)


def topk_rerank_anchor(
    shortlist_scores: torch.Tensor,
    rerank_scores: torch.Tensor,
    candidate: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Shortlist with one representation, rerank with the other."""
    if k < 1:
        raise ValueError('k must be positive')
    if not (
        shortlist_scores.shape == rerank_scores.shape == candidate.shape
    ):
        raise ValueError('scores and candidate must have the same shape')
    if not bool(candidate.any(-1).all()):
        raise ValueError('every row must contain at least one candidate')
    width = shortlist_scores.shape[-1]
    k = min(int(k), width)
    floor = torch.finfo(shortlist_scores.dtype).min
    shortlist = shortlist_scores.masked_fill(~candidate, floor).topk(k, -1).indices
    allowed = torch.zeros_like(candidate).scatter(1, shortlist, True) & candidate
    return masked_argmax(rerank_scores, allowed)


def classify_anchor(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> list[str]:
    """Return positive/negative/gray for each selected anchor."""
    selected_positive = positive.gather(1, anchor[:, None]).squeeze(1)
    selected_negative = negative.gather(1, anchor[:, None]).squeeze(1)
    result = []
    for is_positive, is_negative in zip(
        selected_positive.cpu().tolist(), selected_negative.cpu().tolist()
    ):
        if is_positive:
            result.append('positive')
        elif is_negative:
            result.append('negative')
        else:
            result.append('gray')
    return result


def nearest_positive_distance(
    anchor: torch.Tensor,
    positive: torch.Tensor,
) -> torch.Tensor:
    """Temporal distance to the closest positive, or -1 on novel rows."""
    timeline = torch.arange(positive.shape[1], device=positive.device)[None]
    distance = (timeline - anchor[:, None]).abs().masked_fill(
        ~positive, positive.shape[1] + 1
    )
    nearest = distance.min(-1).values
    return torch.where(positive.any(-1), nearest, nearest.new_full((), -1))


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
    config.il.require_versioned_cache = args.require_versioned_cache
    config.il.expected_cache_signature = args.expected_cache_signature
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


def _topk_positive(logits, candidate, positive, k):
    k = min(int(k), logits.shape[1])
    floor = torch.finfo(logits.dtype).min
    indices = logits.masked_fill(~candidate, floor).topk(k, dim=-1).indices
    return positive.gather(1, indices).any(-1)


def _summarize(records, strategy_names, dilation_radii):
    revisit = [record for record in records if record['is_revisit']]
    novel = [record for record in records if not record['is_revisit']]
    strategy_summary = {}
    for name in strategy_names:
        selected = [record['strategies'][name] for record in revisit]
        counts = {
            outcome: sum(row['outcome'] == outcome for row in selected)
            for outcome in ('positive', 'gray', 'negative')
        }
        distances = [row['nearest_positive_distance'] for row in selected]
        strategy_summary[name] = {
            'rows': len(selected),
            'positive': counts['positive'],
            'gray': counts['gray'],
            'negative': counts['negative'],
            'positive_rate': counts['positive'] / max(1, len(selected)),
            'nonnegative_rate': (
                counts['positive'] + counts['gray']
            ) / max(1, len(selected)),
            'mean_nearest_positive_distance': float(np.mean(distances)),
            'median_nearest_positive_distance': float(np.median(distances)),
            'within_positive_radius': {
                str(radius): float(np.mean(np.asarray(distances) <= radius))
                for radius in dilation_radii
            },
        }
    return {
        'rows': len(records),
        'revisit_rows': len(revisit),
        'novel_rows': len(novel),
        'gate_accuracy_at_0_5': float(np.mean([
            (record['gate_probability'] >= 0.5) == record['is_revisit']
            for record in records
        ])),
        'projected_recall_at_k': {
            str(k): float(np.mean([
                record['projected_recall_at_k'][str(k)] for record in revisit
            ]))
            for k in (1, 5, 10)
        },
        'raw_recall_at_k': {
            str(k): float(np.mean([
                record['raw_recall_at_k'][str(k)] for record in revisit
            ]))
            for k in (1, 5, 10)
        },
        'strategies': strategy_summary,
    }


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
        add_goalA=True,
        data_split=args.data_split,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        sampling_mode='fixed_leg',
        sampling_seed=0,
    )
    original_anchor_margins = sorted({
        int(sample['amargin']) for sample in dataset.samples
    })
    if args.anchor_margin_override is not None:
        if args.anchor_margin_override < args.num_scale:
            raise ValueError(
                'anchor margin must leave the initial scale block intact: '
                f'{args.anchor_margin_override} < num_scale={args.num_scale}'
            )
        for sample in dataset.samples:
            sample['amargin'] = int(args.anchor_margin_override)
        dataset.dataset_fingerprint = hashlib.sha256(
            (
                f'{dataset.dataset_fingerprint}\n'
                f'anchor_margin_override={args.anchor_margin_override}'
            ).encode('utf-8')
        ).hexdigest()
    # Retrieval consumes memory CLS and the goal image, not current-window RGB.
    dataset._load_images = lambda _rgb, indices: torch.empty(  # noqa: SLF001
        len(indices), 3, 1, 1
    )
    model = _load_model(args)
    core = model.core
    goal_cls_cache: dict[str, torch.Tensor] = {}
    records = []
    strategy_names = ['projected_top1', 'raw_top1']
    strategy_names.extend(
        f'blend_projected_{weight:g}' for weight in args.blend_projected_weights
    )
    for topk in args.cross_rerank_topk:
        strategy_names.extend((
            f'projected_top{topk}_raw_rerank',
            f'raw_top{topk}_projected_rerank',
        ))
    for radius in args.temporal_radii:
        strategy_names.extend((
            f'projected_mass_r{radius}',
            f'raw_mass_r{radius}',
        ))

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
            identities = batch['sample_identities']
            missing_rows = [
                index for index, identity in enumerate(identities)
                if identity not in goal_cls_cache
            ]
            if missing_rows:
                with torch.inference_mode():
                    values = core.lingbot.dino(
                        batch['batch_goal_image'][missing_rows].to(core.device)
                    )['cls'].cpu()
                for index, value in zip(missing_rows, values):
                    goal_cls_cache[identities[index]] = value
            goal_cls = torch.stack(
                [goal_cls_cache[identity] for identity in identities]
            ).to(core.device)
            mem_cls = batch['batch_mem_cls'].to(core.device)
            candidate = batch['batch_cand_mask'].to(core.device).bool()
            positive = batch['batch_pos_mask'].to(core.device).bool()
            negative = batch['batch_neg_mask'].to(core.device).bool()

            with torch.inference_mode():
                _, gate_logit, projected_logits, _ = core.retrieval(
                    goal_cls, mem_cls, candidate
                )
                raw_goal = F.normalize(goal_cls.float(), dim=-1)
                raw_mem = F.normalize(mem_cls.float(), dim=-1)
                raw_cos = (raw_goal[:, None] * raw_mem).sum(-1)
                raw_logits = raw_cos / float(args.raw_temperature)
                projected_z = candidate_zscore(projected_logits, candidate)
                raw_z = candidate_zscore(raw_logits, candidate)

                anchors = {
                    'projected_top1': masked_argmax(projected_logits, candidate),
                    'raw_top1': masked_argmax(raw_logits, candidate),
                }
                for weight in args.blend_projected_weights:
                    blended = float(weight) * projected_z + (1.0 - float(weight)) * raw_z
                    anchors[f'blend_projected_{weight:g}'] = masked_argmax(
                        blended, candidate
                    )
                for topk in args.cross_rerank_topk:
                    anchors[f'projected_top{topk}_raw_rerank'] = topk_rerank_anchor(
                        projected_logits, raw_logits, candidate, topk
                    )
                    anchors[f'raw_top{topk}_projected_rerank'] = topk_rerank_anchor(
                        raw_logits, projected_logits, candidate, topk
                    )
                for radius in args.temporal_radii:
                    anchors[f'projected_mass_r{radius}'] = temporal_mass_anchor(
                        projected_logits, candidate, radius
                    )
                    anchors[f'raw_mass_r{radius}'] = temporal_mass_anchor(
                        raw_logits, candidate, radius
                    )
                outcomes = {
                    name: classify_anchor(anchor, positive, negative)
                    for name, anchor in anchors.items()
                }
                distances = {
                    name: nearest_positive_distance(anchor, positive)
                    for name, anchor in anchors.items()
                }
                projected_recall = {
                    k: _topk_positive(
                        projected_logits, candidate, positive, k
                    ) for k in (1, 5, 10)
                }
                raw_recall = {
                    k: _topk_positive(raw_logits, candidate, positive, k)
                    for k in (1, 5, 10)
                }

            for row, identity in enumerate(identities):
                row_strategies = {}
                for name in strategy_names:
                    row_strategies[name] = {
                        'anchor': int(anchors[name][row].cpu()),
                        'outcome': outcomes[name][row],
                        'nearest_positive_distance': int(distances[name][row].cpu()),
                    }
                records.append({
                    'seed': seed,
                    'sample_identity': identity,
                    'goal_label': batch['goal_labels'][row],
                    'cur_step': int(batch['cur_steps'][row]),
                    'is_revisit': bool(batch['batch_is_revisit'][row] > 0.5),
                    'candidate_count': int(candidate[row].sum().cpu()),
                    'positive_count': int(positive[row].sum().cpu()),
                    'negative_count': int(negative[row].sum().cpu()),
                    'gray_count': int((
                        candidate[row] & ~(positive[row] | negative[row])
                    ).sum().cpu()),
                    'gate_probability': float(torch.sigmoid(gate_logit[row]).cpu()),
                    'projected_recall_at_k': {
                        str(k): bool(projected_recall[k][row].cpu())
                        for k in (1, 5, 10)
                    },
                    'raw_recall_at_k': {
                        str(k): bool(raw_recall[k][row].cpu())
                        for k in (1, 5, 10)
                    },
                    'strategies': row_strategies,
                })
            print(
                f'seed={seed} batch={batch_index + 1}/{len(loader)} '
                f'records={len(records)}',
                flush=True,
            )

    return {
        'metadata': {
            'checkpoint': os.path.abspath(args.checkpoint),
            'root_dir': os.path.abspath(args.root_dir),
            'feature_root': os.path.abspath(args.feature_root),
            'dataset_fingerprint': dataset.dataset_fingerprint,
            'data_split': args.data_split,
            'validation_fraction': args.validation_fraction,
            'split_seed': args.split_seed,
            'seeds': args.seeds,
            'raw_temperature': args.raw_temperature,
            'original_anchor_margins': original_anchor_margins,
            'anchor_margin_override': args.anchor_margin_override,
            'temporal_radii': args.temporal_radii,
            'records': len(records),
        },
        'summary': _summarize(records, strategy_names, args.dilation_radii),
        'records': records,
    }


def parse_args():
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
    parser.add_argument('--anchor-margin-override', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--raw-temperature', type=float, default=0.04)
    parser.add_argument(
        '--blend-projected-weights', type=float, nargs='+',
        default=(0.25, 0.5, 0.75),
    )
    parser.add_argument('--cross-rerank-topk', type=int, nargs='+', default=(3, 5))
    parser.add_argument('--temporal-radii', type=int, nargs='+', default=(2, 4, 8, 16))
    parser.add_argument('--dilation-radii', type=int, nargs='+', default=(4, 8, 16, 32))
    parser.add_argument('--data-split', choices=('all', 'train', 'val'), default='all')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--strict-feature-coverage', action='store_true')
    parser.add_argument('--require-versioned-cache', action='store_true')
    parser.add_argument('--expected-cache-signature', default='')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    if args.raw_temperature <= 0 or not math.isfinite(args.raw_temperature):
        parser.error('--raw-temperature must be finite and positive')
    if any(
        not math.isfinite(weight) or not 0.0 <= weight <= 1.0
        for weight in args.blend_projected_weights
    ):
        parser.error('--blend-projected-weights must all be finite and in [0, 1]')
    if any(topk < 1 for topk in args.cross_rerank_topk):
        parser.error('--cross-rerank-topk must all be positive')
    if any(radius < 1 for radius in args.temporal_radii):
        parser.error('--temporal-radii must all be positive')
    if any(radius < 0 for radius in args.dilation_radii):
        parser.error('--dilation-radii must all be non-negative')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('retrieval diagnostic requires CUDA for goal DINO')
    output = Path(args.output)
    if not _inside(output, WORKTREE):
        raise RuntimeError(f'output must stay inside personal worktree {WORKTREE}')
    torch.manual_seed(0)
    np.random.seed(0)
    result = collect(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result['summary'], indent=2, sort_keys=True))
    print(f'REPORT_SAVED {output}')


if __name__ == '__main__':
    main()
