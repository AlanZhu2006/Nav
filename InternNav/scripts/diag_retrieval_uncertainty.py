#!/usr/bin/env python3
"""Probe whether observable retrieval uncertainty predicts a wrong MemNav anchor.

This diagnostic never runs the expensive GCT pose path.  It samples many current
steps, applies the checkpoint's real retrieval head, and records margin, entropy,
and temporal-support cues for later calibration.  Ground-truth positive masks are
used only to label the selected anchor as hit/miss.
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


def _auc(scores: np.ndarray, labels: np.ndarray):
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


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
        data_split='all',
        sampling_mode='fixed_leg',
        sampling_seed=0,
    )
    # Retrieval does not consume current-window pixels.  Avoid decoding and
    # collating 32 unused 518x518 images per sample while retaining real goals.
    dataset._load_images = lambda _rgb, indices: torch.empty(  # noqa: SLF001
        len(indices), 3, 1, 1
    )
    model = _load_model(args)
    core = model.core
    goal_cls_cache: dict[str, torch.Tensor] = {}
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
            with torch.inference_mode():
                match, gate_logit, logits, max_raw_cos = core.retrieval(
                    goal_cls, mem_cls, candidate
                )
                probability = torch.softmax(logits.float(), dim=-1)
                top_values, top_indices = logits.float().topk(2, dim=-1)
                margin = top_values[:, 0] - top_values[:, 1]
                top_probability = probability.gather(1, match[:, None]).squeeze(1)
                entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
                candidate_count = candidate.sum(-1)
                normalized_entropy = entropy / candidate_count.float().log().clamp_min(1.0)
                timeline = torch.arange(logits.shape[1], device=logits.device)[None]
                local = candidate & ((timeline - match[:, None]).abs() <= args.support_radius)
                local_mass = (probability * local).sum(-1)
                local_neighbor_mass = local_mass - top_probability

                raw_goal = F.normalize(goal_cls.float(), dim=-1)
                raw_mem = F.normalize(mem_cls.float(), dim=-1)
                raw_cos = (raw_goal[:, None] * raw_mem).sum(-1)
                raw_ranked = raw_cos.masked_fill(~candidate, -1.0).topk(2, dim=-1)
                raw_margin = raw_ranked.values[:, 0] - raw_ranked.values[:, 1]
                hit = positive.gather(1, match[:, None]).squeeze(1)

            for row, identity in enumerate(identities):
                records.append({
                    'seed': seed,
                    'sample_identity': identity,
                    'goal_label': batch['goal_labels'][row],
                    'cur_step': int(batch['cur_steps'][row]),
                    'match_step': int(match[row].cpu()),
                    'is_revisit': bool(batch['batch_is_revisit'][row] > 0.5),
                    'hit': bool(hit[row].cpu()),
                    'candidate_count': int(candidate_count[row].cpu()),
                    'rank_margin': float(margin[row].cpu()),
                    'raw_margin': float(raw_margin[row].cpu()),
                    'top_probability': float(top_probability[row].cpu()),
                    'normalized_entropy': float(normalized_entropy[row].cpu()),
                    'local_mass': float(local_mass[row].cpu()),
                    'local_neighbor_mass': float(local_neighbor_mass[row].cpu()),
                    'max_raw_cos': float(max_raw_cos[row].cpu()),
                    'gate_probability': float(torch.sigmoid(gate_logit[row]).cpu()),
                    'top2_steps': top_indices[row].cpu().tolist(),
                })
            print(
                f'seed={seed} batch={batch_index + 1}/{len(loader)} '
                f'records={len(records)}',
                flush=True,
            )

    revisit = [row for row in records if row['is_revisit']]
    labels = np.asarray([row['hit'] for row in revisit], dtype=bool)
    feature_names = (
        'rank_margin', 'raw_margin', 'top_probability',
        'normalized_entropy', 'local_mass', 'local_neighbor_mass', 'max_raw_cos',
    )
    features = {}
    for name in feature_names:
        values = np.asarray([row[name] for row in revisit], dtype=np.float64)
        auc = _auc(-values if name == 'normalized_entropy' else values, labels)
        features[name] = {
            'hit_mean': float(values[labels].mean()) if labels.any() else None,
            'miss_mean': float(values[~labels].mean()) if (~labels).any() else None,
            'auc_for_hit': auc,
        }
    gate_labels = np.asarray([row['is_revisit'] for row in records], dtype=bool)
    gate_prediction = np.asarray(
        [row['gate_probability'] > 0.5 for row in records], dtype=bool
    )
    return {
        'metadata': {
            'checkpoint': os.path.abspath(args.checkpoint),
            'root_dir': os.path.abspath(args.root_dir),
            'feature_root': os.path.abspath(args.feature_root),
            'seeds': args.seeds,
            'support_radius': args.support_radius,
            'records': len(records),
        },
        'summary': {
            'revisit_records': len(revisit),
            'retrieval_hits': int(labels.sum()),
            'retrieval_hit_rate': float(labels.mean()),
            'gate_accuracy': float(np.mean(gate_prediction == gate_labels)),
            'uncertainty_features': features,
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
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--support-radius', type=int, default=16)
    parser.add_argument('--strict-feature-coverage', action='store_true')
    parser.add_argument('--require-versioned-cache', action='store_true')
    parser.add_argument('--expected-cache-signature', default='')
    args = parser.parse_args()
    if args.support_radius < 1:
        parser.error('--support-radius must be positive')
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
