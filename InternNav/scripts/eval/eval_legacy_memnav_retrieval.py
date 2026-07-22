#!/usr/bin/env python3
"""Audit an old MemNav retrieval checkpoint under deterministic objectives.

The July-15 checkpoint used a projected DINO head with a learned null key and a
single joint softmax.  Its W&B run logged only random training minibatches, so
those values cannot distinguish minibatch noise from poor retrieval.  This
script extracts only that legacy head and evaluates it without changing the
checkpoint, dataset, or policy code.

Three panels are supported:

* ``legacy`` reproduces the old label/candidate semantics, but replaces random
  per-access ``k`` with deterministic seeded draws;
* ``current8`` uses the present retrieval labels and the validated early anchor
  floor of frame 8;
* ``current39`` keeps the present labels but restores the old/default frame-39
  candidate floor, isolating the anchor-floor change.

Every panel also evaluates raw frozen DINO cosine similarity.  The report keeps
the old null-slot objective separate from the current frame-only listwise loss.
This is an offline retrieval diagnostic, not a navigation score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT = Path(__file__).resolve()
INTERNNAV_ROOT = SCRIPT.parents[2]
if os.fspath(INTERNNAV_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(INTERNNAV_ROOT))

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset  # noqa: E402
from internnav.model.basemodel.memnav.lingbot_stream import (  # noqa: E402
    LingBotStream,
)


class LegacyRetrievalHead(nn.Module):
    """Exact retrieval parameterization used by commit ed15f61."""

    def __init__(self, dino_dim=1024, proj_dim=256):
        super().__init__()
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        self.null_key = nn.Parameter(torch.empty(proj_dim))
        self.log_temp = nn.Parameter(torch.empty(()))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.01, 1.0)

    def components(self, goal_cls, mem_cls):
        goal = F.normalize(self.proj_goal(goal_cls), dim=-1)
        memory = F.normalize(self.proj_mem(mem_cls), dim=-1)
        null = F.normalize(self.null_key, dim=-1)
        cosine = (goal[:, None] * memory).sum(-1)
        null_cosine = (goal * null).sum(-1)
        return cosine, null_cosine


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_legacy_head(checkpoint, device):
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location='cpu')
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict):
        raise TypeError('checkpoint does not contain a state dictionary')

    names = (
        'proj_goal.weight', 'proj_goal.bias',
        'proj_mem.weight', 'proj_mem.bias',
        'null_key', 'log_temp',
    )
    extracted = {}
    source_keys = {}
    for name in names:
        suffix = f'retrieval.{name}'
        matches = [key for key in state if key.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f'expected exactly one checkpoint tensor ending in {suffix!r}; '
                f'found {matches}'
            )
        extracted[name] = state[matches[0]].detach().cpu()
        source_keys[name] = matches[0]
    del payload, state

    head = LegacyRetrievalHead(
        dino_dim=int(extracted['proj_goal.weight'].shape[1]),
        proj_dim=int(extracted['proj_goal.weight'].shape[0]),
    )
    head.load_state_dict(extracted, strict=True)
    return head.to(device).eval(), source_keys


def _legacy_static_label(curve, kind, pos_hi, pos_lo, anchor_margin):
    """Reproduce ``memnav_labels.build_retrieval_label`` at ed15f61."""
    curve = np.asarray(curve, dtype=np.float32)
    if curve.ndim != 1 or curve.size == 0:
        return None, 'invalid_curve'
    if not np.isfinite(curve).all() or np.any((curve < 0.0) | (curve > 1.0)):
        return None, 'invalid_curve'
    if not 0.0 <= float(pos_lo) < float(pos_hi) <= 1.0:
        return None, 'invalid_thresholds'
    if int(anchor_margin) < 0:
        return None, 'invalid_anchor_margin'
    candidate = np.arange(curve.size) >= int(anchor_margin)
    if not candidate.any():
        return None, 'no_valid_candidates'
    positive = candidate & (curve >= float(pos_hi))
    negative = candidate & (curve <= float(pos_lo))
    if kind == 'revisit':
        if not positive.any():
            return None, 'weak_revisit'
        null_positive = False
    elif kind == 'novel':
        if positive.any():
            return None, 'novel_has_positive'
        null_positive = True
    else:
        return None, 'unknown_goal_kind'
    return {
        'positive': positive,
        'negative': negative,
        'candidate': candidate,
        'null_positive': null_positive,
    }, None


def _dataset(args, *, anchor_min, exclude_recent):
    return MemNav_Dataset(
        args.root_dir,
        predict_size=24,
        image_size=518,
        lingbot_repo=args.lingbot_repo,
        feature_root=args.feature_root,
        window_size=args.window,
        num_scale=args.num_scale,
        retrieval_anchor_min_frame=anchor_min,
        exclude_recent=exclude_recent,
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
        retrieval_only=True,
    )


def _current_descriptors(dataset):
    result = []
    for index, sample in enumerate(dataset.samples):
        trajectory = int(sample['traj_idx'])
        result.append({
            'identity': str(sample['sample_identity']),
            'goal_path': os.fspath(sample['goal_img_path']),
            'feature_path': os.fspath(
                dataset.trajectory_feature_path[trajectory]
            ),
            'camera_feature_path': os.fspath(
                dataset.trajectory_camera_feature_path[trajectory]
            ),
            'sample': sample,
            'dataset_index': index,
            'label_contract': 'current',
        })
    return result


def _legacy_descriptors(dataset, args):
    """Rebuild the old samples from raw meta on the selected scene slice."""
    result = []
    skipped = defaultdict(int)
    for trajectory, trajectory_dir in enumerate(dataset.trajectory_dirs):
        meta_path = os.path.join(trajectory_dir, args.meta_filename)
        try:
            with open(meta_path, encoding='utf-8') as handle:
                meta = json.load(handle)
        except Exception:
            skipped['invalid_meta'] += 1
            continue
        goals = meta.get('goals') or []
        switches = meta.get('switches') or []
        frame_count = int(meta.get('n_frames', 0))
        pos_hi = float(meta.get('covis_pos_hi', 0.5))
        pos_lo = float(meta.get('covis_pos_lo', 0.1))
        anchor_margin = int(meta.get(
            'anchor_margin', args.num_scale + args.window - 1
        ))
        relative_trajectory = os.path.relpath(trajectory_dir, args.root_dir)
        feature_path = os.fspath(dataset.trajectory_feature_path[trajectory])
        camera_feature_path = os.fspath(
            dataset.trajectory_camera_feature_path[trajectory]
        )
        rgb_dir = os.fspath(dataset.trajectory_rgb_dir[trajectory])

        for goal_index, goal in enumerate(goals):
            curve = goal.get('covis_curve')
            if not curve:
                continue
            curve = np.asarray(curve, dtype=np.float32)
            leg_start = int(curve.shape[0])
            leg_end = (
                int(switches[goal_index + 1])
                if goal_index + 1 < len(switches) else frame_count
            )
            goal_step = leg_end - 1
            k_lo = max(leg_start, anchor_margin)
            k_hi = goal_step - 4
            if k_lo < anchor_margin or k_hi < k_lo:
                skipped['invalid_k_range'] += 1
                continue
            goal_path = os.path.join(
                trajectory_dir, f'goal_{goal_index + 1}.jpg'
            )
            if not os.path.isfile(goal_path):
                skipped['missing_goal_image'] += 1
                continue
            label, reason = _legacy_static_label(
                curve, goal.get('kind'), pos_hi, pos_lo, anchor_margin
            )
            if label is None:
                skipped[reason] += 1
                continue
            result.append({
                'identity': f'{relative_trajectory}:legacy-goal-{goal_index}',
                'goal_path': goal_path,
                'feature_path': feature_path,
                'camera_feature_path': camera_feature_path,
                'k_lo': int(k_lo),
                'k_hi': int(k_hi),
                'goal_step': int(goal_step),
                'leg_start': int(leg_start),
                'positive_pre': label['positive'],
                'negative_pre': label['negative'],
                'semantic_null_positive': bool(label['null_positive']),
                'anchor_margin': int(anchor_margin),
                'label_contract': 'legacy_covis',
            })

        if switches:
            arrival = int(switches[0]) - 1
            k_lo = anchor_margin
            k_hi = arrival - 4
            clean_negative = arrival - 83 >= anchor_margin
            goal_path = os.path.join(rgb_dir, f'{arrival}.jpg')
            if k_hi >= k_lo and clean_negative and os.path.isfile(goal_path):
                result.append({
                    'identity': f'{relative_trajectory}:legacy-goal-A',
                    'goal_path': goal_path,
                    'feature_path': feature_path,
                    'camera_feature_path': camera_feature_path,
                    'k_lo': int(k_lo),
                    'k_hi': int(k_hi),
                    'goal_step': int(arrival),
                    'arrival': int(arrival),
                    'anchor_margin': int(anchor_margin),
                    'label_contract': 'legacy_goal_a',
                })
    return result, dict(sorted(skipped.items()))


def _fixed_k(descriptor, seed):
    sample = descriptor.get('sample')
    if sample is not None:
        k_lo = int(sample['k_lo'])
        k_hi = int(sample['k_hi'])
    else:
        k_lo = int(descriptor['k_lo'])
        k_hi = int(descriptor['k_hi'])
    identity = f"{int(seed)}:{descriptor['identity']}"
    sample_seed = int.from_bytes(
        hashlib.sha256(identity.encode('utf-8')).digest()[:8], 'big'
    )
    return int(np.random.default_rng(sample_seed).integers(k_lo, k_hi + 1))


def _labels(descriptor, k, current_dataset=None):
    contract = descriptor['label_contract']
    if contract == 'current':
        positive, negative, candidate, null_positive = (
            current_dataset._build_label(descriptor['sample'], k)  # noqa: SLF001
        )
    else:
        indices = np.arange(k + 1)
        candidate = indices >= int(descriptor['anchor_margin'])
        positive = np.zeros(k + 1, dtype=bool)
        negative = np.zeros(k + 1, dtype=bool)
        if contract == 'legacy_covis':
            length = min(int(descriptor['leg_start']), k + 1)
            positive[:length] = descriptor['positive_pre'][:length]
            negative[:length] = descriptor['negative_pre'][:length]
            null_positive = bool(descriptor['semantic_null_positive'])
        else:
            arrival = int(descriptor['arrival'])
            positive = indices >= arrival - 14
            negative = (indices <= arrival - 83) & candidate
            null_positive = not bool(positive.any())
        positive &= candidate
        negative &= candidate & ~positive
    return (
        np.ascontiguousarray(positive, dtype=bool),
        np.ascontiguousarray(negative, dtype=bool),
        np.ascontiguousarray(candidate, dtype=bool),
        bool(null_positive),
    )


def _row_signature(identity, seed, k, positive, negative, candidate):
    digest = hashlib.sha256()
    digest.update(f'{identity}\n{seed}\n{k}\n'.encode('utf-8'))
    digest.update(np.packbits(positive).tobytes())
    digest.update(np.packbits(negative).tobytes())
    digest.update(np.packbits(candidate).tobytes())
    return digest.hexdigest()


def _prepare_rows(descriptors, seed, current_dataset, load_dino):
    rows = []
    for descriptor in descriptors:
        dino = load_dino(descriptor['feature_path'])
        k_hi = min(
            _fixed_k(descriptor, seed),
            int(descriptor.get('goal_step', len(dino) - 1)) - 1,
            len(dino) - 2,
        )
        k_lo = int(
            descriptor.get('sample', descriptor).get('k_lo', 0)
        )
        k = max(min(k_hi, len(dino) - 2), min(k_lo, k_hi))
        positive, negative, candidate, null_positive = _labels(
            descriptor, k, current_dataset
        )
        if not candidate.any():
            raise RuntimeError(
                f"row {descriptor['identity']} seed={seed} has no candidate"
            )
        rows.append({
            'descriptor': descriptor,
            'identity': descriptor['identity'],
            'goal_path': descriptor['goal_path'],
            'k': int(k),
            'memory': np.asarray(dino[:k + 1], dtype=np.float32),
            'positive': positive,
            'negative': negative,
            'candidate': candidate,
            'null_positive': bool(null_positive),
            'is_revisit': bool(not null_positive),
            'signature': _row_signature(
                descriptor['identity'], seed, k,
                positive, negative, candidate,
            ),
        })
    return rows


def _select_descriptors(
    descriptors, max_samples, selection_seed, current_dataset, load_dino
):
    if max_samples <= 0 or max_samples >= len(descriptors):
        return descriptors
    rows = _prepare_rows(descriptors, 0, current_dataset, load_dino)
    revisit = [row for row in rows if row['is_revisit']]
    novel = [row for row in rows if not row['is_revisit']]
    rng = np.random.default_rng(int(selection_seed))
    target_revisit = min(max_samples // 2, len(revisit))
    target_novel = min(max_samples - target_revisit, len(novel))
    if target_revisit + target_novel < max_samples:
        target_revisit = min(len(revisit), max_samples - target_novel)
    selected = []
    if target_revisit:
        selected.extend(rng.choice(revisit, target_revisit, replace=False))
    if target_novel:
        selected.extend(rng.choice(novel, target_novel, replace=False))
    identities = {row['identity'] for row in selected}
    return [row for row in descriptors if row['identity'] in identities]


def _pad(rows, device):
    batch_size = len(rows)
    width = max(len(row['memory']) for row in rows)
    dimension = int(rows[0]['memory'].shape[-1])
    memory = torch.zeros(batch_size, width, dimension, dtype=torch.float32)
    observed = torch.zeros(batch_size, width, dtype=torch.bool)
    positive = torch.zeros_like(observed)
    negative = torch.zeros_like(observed)
    candidate = torch.zeros_like(observed)
    for index, row in enumerate(rows):
        length = len(row['memory'])
        memory[index, :length] = torch.from_numpy(row['memory'])
        observed[index, :length] = True
        positive[index, :length] = torch.from_numpy(row['positive'])
        negative[index, :length] = torch.from_numpy(row['negative'])
        candidate[index, :length] = torch.from_numpy(row['candidate'])
    return (
        memory.to(device), observed.to(device), positive.to(device),
        negative.to(device), candidate.to(device),
    )


def _masked_lse(scores, mask):
    floor = torch.finfo(scores.dtype).min
    return scores.masked_fill(~mask, floor).logsumexp(-1)


def _topk_positive(scores, candidate, positive, k):
    count = min(int(k), scores.shape[-1])
    floor = torch.finfo(scores.dtype).min
    indices = scores.masked_fill(~candidate, floor).topk(count, -1).indices
    return positive.gather(1, indices).any(-1)


def _score_rows(rows, goal_cls, memory, positive, negative, candidate, head, raw_temp):
    projected_cosine, null_cosine = head.components(goal_cls, memory)
    temperature = head.temperature.to(projected_cosine)
    projected = projected_cosine / temperature
    raw = (
        F.normalize(goal_cls.float(), dim=-1)[:, None]
        * F.normalize(memory.float(), dim=-1)
    ).sum(-1) / float(raw_temp)
    floor = torch.finfo(projected.dtype).min

    null_positive = torch.tensor(
        [row['null_positive'] for row in rows],
        device=projected.device, dtype=torch.bool,
    )
    is_revisit = ~null_positive
    nonpositive = candidate & ~positive
    rankable = positive.any(-1) & nonpositive.any(-1)
    pn_valid = positive.any(-1) & negative.any(-1)

    output = []
    ranking = {}
    for name, scores in (('projected', projected), ('raw', raw)):
        match = scores.masked_fill(~candidate, floor).argmax(-1)
        selected_positive = positive.gather(1, match[:, None]).squeeze(1)
        selected_negative = negative.gather(1, match[:, None]).squeeze(1)
        selected_gray = (
            candidate.gather(1, match[:, None]).squeeze(1)
            & ~selected_positive & ~selected_negative
        )
        all_loss = _masked_lse(scores, candidate) - _masked_lse(scores, positive)
        pn_mask = positive | negative
        pn_loss = _masked_lse(scores, pn_mask) - _masked_lse(scores, positive)
        ranking[name] = {
            'match': match,
            'positive': selected_positive,
            'negative': selected_negative,
            'gray': selected_gray,
            'all_loss': all_loss,
            'pn_loss': pn_loss,
            'r5': _topk_positive(scores, candidate, positive, 5),
            'r10': _topk_positive(scores, candidate, positive, 10),
        }

    legacy_real = projected.masked_fill(~candidate, floor)
    legacy_null = null_cosine[:, None] / temperature
    legacy_logits = torch.cat([legacy_real, legacy_null], -1)
    pos_full = torch.cat([positive, null_positive[:, None]], -1)
    neg_full = torch.cat([negative, is_revisit[:, None]], -1)
    valid_full = pos_full | neg_full
    legacy_loss = _masked_lse(legacy_logits, valid_full) - _masked_lse(
        legacy_logits, pos_full
    )
    legacy_probability = legacy_logits.softmax(-1)
    legacy_prediction = legacy_logits.argmax(-1)
    legacy_correct = pos_full.gather(
        1, legacy_prediction[:, None]
    ).squeeze(1)
    gate = 1.0 - legacy_probability[:, -1]

    for index, row in enumerate(rows):
        item = {
            'identity': row['identity'],
            'k': row['k'],
            'signature': row['signature'],
            'is_revisit': bool(is_revisit[index].cpu()),
            'candidate_count': int(candidate[index].sum().cpu()),
            'positive_count': int(positive[index].sum().cpu()),
            'negative_count': int(negative[index].sum().cpu()),
            'gray_count': int((candidate[index] & ~(positive[index] | negative[index])).sum().cpu()),
            'rankable': bool(rankable[index].cpu()),
            'pn_valid': bool(pn_valid[index].cpu()),
            'legacy_loss': legacy_loss[index].detach().cpu().item(),
            'legacy_correct': bool(legacy_correct[index].cpu()),
            'legacy_predicted_null': bool(
                legacy_prediction[index].item() == projected.shape[-1]
            ),
            'legacy_gate': gate[index].detach().cpu().item(),
        }
        for name in ('projected', 'raw'):
            values = ranking[name]
            outcome = (
                'positive' if bool(values['positive'][index].cpu())
                else 'negative' if bool(values['negative'][index].cpu())
                else 'gray'
            )
            item[name] = {
                'match': int(values['match'][index].cpu()),
                'outcome': outcome,
                'all_candidate_loss': values['all_loss'][index].detach().cpu().item(),
                'positive_negative_loss': values['pn_loss'][index].detach().cpu().item(),
                'recall_at_5': bool(values['r5'][index].cpu()),
                'recall_at_10': bool(values['r10'][index].cpu()),
            }
        output.append(item)
    return output


def _mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank_summary(records, name):
    rows = [record for record in records if record['rankable']]
    pn_rows = [record for record in records if record['pn_valid']]
    outcomes = {
        value: sum(row[name]['outcome'] == value for row in rows)
        for value in ('positive', 'gray', 'negative')
    }
    return {
        'rankable_rows': len(rows),
        'all_candidate_listwise_loss': _mean([
            row[name]['all_candidate_loss'] for row in rows
        ]),
        'positive_negative_loss': _mean([
            row[name]['positive_negative_loss'] for row in pn_rows
        ]),
        'strict_top1': outcomes['positive'] / max(1, len(rows)),
        'recall_at_5': _mean([row[name]['recall_at_5'] for row in rows]),
        'recall_at_10': _mean([row[name]['recall_at_10'] for row in rows]),
        'selected_positive': outcomes['positive'],
        'selected_gray': outcomes['gray'],
        'selected_negative': outcomes['negative'],
        'loss_correlation_candidate_count': _correlation(
            [row[name]['all_candidate_loss'] for row in rows],
            [row['candidate_count'] for row in rows],
        ),
        'loss_correlation_positive_fraction': _correlation(
            [row[name]['all_candidate_loss'] for row in rows],
            [row['positive_count'] / row['candidate_count'] for row in rows],
        ),
    }


def _batch_noise(values, batch_size, seed=104729, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, batch_size))].mean(-1)
    return {
        'batch_size': int(batch_size),
        'bootstrap_draws': int(draws),
        'mean': float(means.mean()),
        'std': float(means.std()),
        'p05': float(np.quantile(means, 0.05)),
        'p50': float(np.quantile(means, 0.50)),
        'p95': float(np.quantile(means, 0.95)),
        'min': float(means.min()),
        'max': float(means.max()),
    }


def _seed_summary(records):
    revisit = [record for record in records if record['is_revisit']]
    novel = [record for record in records if not record['is_revisit']]
    legacy_correct = [record for record in records if record['legacy_correct']]
    return {
        'rows': len(records),
        'revisit_rows': len(revisit),
        'novel_rows': len(novel),
        'candidate_count_mean': _mean([row['candidate_count'] for row in records]),
        'positive_count_mean_revisit': _mean([row['positive_count'] for row in revisit]),
        'legacy_null_objective': {
            'loss': _mean([row['legacy_loss'] for row in records]),
            'loss_revisit': _mean([row['legacy_loss'] for row in revisit]),
            'loss_novel': _mean([row['legacy_loss'] for row in novel]),
            'retrieval_accuracy': len(legacy_correct) / max(1, len(records)),
            'seen_match_accuracy': _mean([row['legacy_correct'] for row in revisit]),
            'unseen_null_accuracy': _mean([
                row['legacy_predicted_null'] for row in novel
            ]),
            'gate_mean_revisit': _mean([row['legacy_gate'] for row in revisit]),
            'gate_mean_novel': _mean([row['legacy_gate'] for row in novel]),
            'gate_accuracy_at_0_5': _mean([
                (row['legacy_gate'] >= 0.5) == row['is_revisit']
                for row in records
            ]),
        },
        'projected_frame_ranking': _rank_summary(records, 'projected'),
        'raw_frame_ranking': _rank_summary(records, 'raw'),
        'fixed_checkpoint_minibatch_noise': {
            'legacy_loss_batch4': _batch_noise(
                [row['legacy_loss'] for row in records], 4
            ),
            'legacy_loss_batch32': _batch_noise(
                [row['legacy_loss'] for row in records], 32
            ),
        },
    }


def _path_value(mapping, path):
    value = mapping
    for key in path.split('.'):
        value = value[key]
    return value


def _across_seed(seed_summaries, records):
    paths = (
        'legacy_null_objective.loss',
        'legacy_null_objective.retrieval_accuracy',
        'legacy_null_objective.seen_match_accuracy',
        'legacy_null_objective.unseen_null_accuracy',
        'projected_frame_ranking.all_candidate_listwise_loss',
        'projected_frame_ranking.strict_top1',
        'projected_frame_ranking.recall_at_5',
        'raw_frame_ranking.all_candidate_listwise_loss',
        'raw_frame_ranking.strict_top1',
        'raw_frame_ranking.recall_at_5',
    )
    metrics = {}
    for path in paths:
        values = [
            _path_value(summary, path) for summary in seed_summaries.values()
        ]
        values = np.asarray([value for value in values if value is not None], dtype=np.float64)
        metrics[path] = {
            'mean': float(values.mean()) if len(values) else None,
            'std': float(values.std()) if len(values) else None,
            'min': float(values.min()) if len(values) else None,
            'max': float(values.max()) if len(values) else None,
        }

    grouped = defaultdict(list)
    for record in records:
        if record['rankable']:
            grouped[record['identity']].append(
                record['projected']['all_candidate_loss']
            )
    within = [np.std(values) for values in grouped.values() if len(values) > 1]
    return {
        'seed_aggregate_metrics': metrics,
        'rankable_identities_seen_in_multiple_seeds': len(within),
        'mean_within_identity_projected_loss_std': (
            float(np.mean(within)) if within else None
        ),
        'median_within_identity_projected_loss_std': (
            float(np.median(within)) if within else None
        ),
    }


def _evaluate_panel(
    name, descriptors, current_dataset, dataset_metadata, args,
    lingbot, head, goal_cache, load_dino,
):
    descriptors = _select_descriptors(
        descriptors, args.max_samples, args.selection_seed,
        current_dataset, load_dino,
    )
    all_records = []
    seed_summaries = {}
    started = time.time()
    for seed in args.sampling_seeds:
        rows = _prepare_rows(descriptors, seed, current_dataset, load_dino)
        records = []
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset:offset + args.batch_size]
            missing_paths = []
            for row in batch_rows:
                if row['goal_path'] not in goal_cache and row['goal_path'] not in missing_paths:
                    missing_paths.append(row['goal_path'])
            if missing_paths:
                images = current_dataset._load_and_preprocess(  # noqa: SLF001
                    missing_paths,
                    mode=current_dataset.preprocess_mode,
                    image_size=current_dataset.image_size,
                    patch_size=current_dataset.patch_size,
                )
                with torch.inference_mode():
                    cls = lingbot.dino(images.to(lingbot.device))['cls'].cpu()
                for path, value in zip(missing_paths, cls):
                    goal_cache[path] = value
            goal_cls = torch.stack([
                goal_cache[row['goal_path']] for row in batch_rows
            ]).to(lingbot.device)
            memory, _observed, positive, negative, candidate = _pad(
                batch_rows, lingbot.device
            )
            with torch.inference_mode():
                batch_records = _score_rows(
                    batch_rows, goal_cls, memory, positive, negative,
                    candidate, head, args.raw_temperature,
                )
            for record in batch_records:
                record['seed'] = int(seed)
                record['panel'] = name
            records.extend(batch_records)
            print(
                f'panel={name} seed={seed} '
                f'rows={min(offset + len(batch_rows), len(rows))}/{len(rows)}',
                flush=True,
            )
        seed_summaries[str(seed)] = _seed_summary(records)
        all_records.extend(records)

    fingerprint = hashlib.sha256()
    for record in all_records:
        fingerprint.update(record['signature'].encode('ascii'))
    return {
        'metadata': {
            **dataset_metadata,
            'panel': name,
            'descriptors': len(descriptors),
            'sampling_seeds': list(args.sampling_seeds),
            'evaluated_rows': len(all_records),
            'evaluation_fingerprint': fingerprint.hexdigest(),
            'elapsed_seconds': time.time() - started,
        },
        'per_seed': seed_summaries,
        'across_seed': _across_seed(seed_summaries, all_records),
        'records': all_records if args.save_records else [],
    }


def _eval_subset_fingerprint(dataset):
    manifest = (
        f'{dataset.dataset_fingerprint}\n'
        + ','.join(str(index) for index in range(len(dataset)))
    )
    return hashlib.sha256(manifest.encode('utf-8')).hexdigest()


def _verify_current_getitem(dataset, descriptors, seed, load_dino):
    """Compare the lightweight audit path with the production dataset path."""
    dataset.sampling_seed = int(seed)
    expected_rows = _prepare_rows(descriptors, seed, dataset, load_dino)
    mismatches = []
    actual_revisit = 0
    actual_rankable = 0
    actual_pn_valid = 0
    for index, expected in enumerate(expected_rows):
        actual = dataset[index]
        actual_k = int(actual['mem_cls'].shape[0] - 1)
        actual_positive = actual['pos_mask'].cpu().numpy().astype(bool)
        actual_negative = actual['neg_mask'].cpu().numpy().astype(bool)
        actual_candidate = actual['cand_mask'].cpu().numpy().astype(bool)
        actual_null = bool(actual['null_pos'])
        actual_revisit += int(not actual_null)
        actual_rankable += int(
            actual_positive.any()
            and (actual_candidate & ~actual_positive).any()
        )
        actual_pn_valid += int(
            actual_positive.any() and actual_negative.any()
        )
        differences = []
        if actual_k != expected['k']:
            differences.append(f"k={actual_k}!={expected['k']}")
        for name, actual_mask in (
            ('positive', actual_positive),
            ('negative', actual_negative),
            ('candidate', actual_candidate),
        ):
            expected_mask = expected[name]
            if actual_mask.shape != expected_mask.shape:
                differences.append(
                    f'{name}_shape={actual_mask.shape}!={expected_mask.shape}'
                )
            elif not np.array_equal(actual_mask, expected_mask):
                differences.append(
                    f'{name}_different='
                    f'{int(np.count_nonzero(actual_mask != expected_mask))}'
                )
        if actual_null != expected['null_positive']:
            differences.append(
                f"null={actual_null}!={expected['null_positive']}"
            )
        if differences and len(mismatches) < 20:
            mismatches.append({
                'index': index,
                'identity': expected['identity'],
                'differences': differences,
            })
    if mismatches:
        raise RuntimeError(
            'manual fixed retrieval rows disagree with dataset __getitem__: '
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        'seed': int(seed),
        'rows_checked': len(expected_rows),
        'mismatches': 0,
        'actual_revisit_rows': actual_revisit,
        'actual_rankable_rows': actual_rankable,
        'actual_positive_negative_rows': actual_pn_valid,
    }


def collect(args):
    device = torch.device('cuda')
    head, checkpoint_keys = _extract_legacy_head(args.checkpoint, device)
    print(
        f'legacy_temperature={head.temperature.detach().cpu().item():.9f} '
        f'checkpoint={args.checkpoint}',
        flush=True,
    )
    lingbot = LingBotStream(
        lingbot_repo=args.lingbot_repo,
        weights=args.lingbot_weights,
        window=args.window,
        num_scale=args.num_scale,
        max_frame_num=args.max_frame_num,
        device='cuda',
    ).eval()

    @lru_cache(maxsize=4)
    def load_dino(path):
        with np.load(path) as payload:
            return payload['dino_cls'].astype(np.float32)

    goal_cache = {}
    panels = {}
    if 'legacy' in args.panels:
        reference = _dataset(args, anchor_min=None, exclude_recent=0)
        descriptors, skipped = _legacy_descriptors(reference, args)
        panels['legacy'] = _evaluate_panel(
            'legacy', descriptors, reference,
            {
                'label_contract': 'ed15f61 legacy candidate/label/null objective',
                'reference_dataset_fingerprint': reference.dataset_fingerprint,
                'legacy_descriptor_skips': skipped,
                'legacy_anchor_margins': sorted({
                    int(descriptor['anchor_margin'])
                    for descriptor in descriptors
                }),
                'legacy_run_scene_overlap_expected': True,
            },
            args, lingbot, head, goal_cache, load_dino,
        )
        del reference

    for panel_name, anchor in (('current8', 8), ('current39', 39)):
        if panel_name not in args.panels:
            continue
        dataset = _dataset(args, anchor_min=anchor, exclude_recent=83)
        descriptors = _current_descriptors(dataset)
        verification = (
            _verify_current_getitem(
                dataset, descriptors, args.sampling_seeds[0], load_dino
            )
            if args.verify_current_getitem else None
        )
        panels[panel_name] = _evaluate_panel(
            panel_name, descriptors, dataset,
            {
                'label_contract': 'current unified exclude-recent objective',
                'retrieval_anchor_min_frame': anchor,
                'dataset_fingerprint': dataset.dataset_fingerprint,
                'eval_dataset_fingerprint': _eval_subset_fingerprint(dataset),
                'production_getitem_verification': verification,
            },
            args, lingbot, head, goal_cache, load_dino,
        )
        del dataset

    return {
        'metadata': {
            'evaluation_type': 'fixed legacy checkpoint retrieval audit',
            'checkpoint': os.path.abspath(args.checkpoint),
            'checkpoint_sha256': _sha256(args.checkpoint),
            'checkpoint_tensor_keys': checkpoint_keys,
            'legacy_temperature': head.temperature.detach().cpu().item(),
            'legacy_source_commit': 'ed15f6148dc645a28e200f64e7f93ff02e1c7fa5',
            'root_dir': args.root_dir,
            'feature_root': args.feature_root,
            'data_split': args.data_split,
            'validation_fraction': args.validation_fraction,
            'split_seed': args.split_seed,
            'raw_temperature': args.raw_temperature,
            'max_samples': args.max_samples,
            'cache_contract': {
                'strict_feature_coverage': args.strict_feature_coverage,
                'require_versioned_cache': args.require_versioned_cache,
                'expected_cache_signature': args.expected_cache_signature,
            },
            'caveat': (
                'The legacy run trained with scene_split=all; a present-day val '
                'slice is deterministic but was not held out from that checkpoint.'
            ),
        },
        'panels': panels,
    }


def collect_contract_verification(args):
    """Verify manual fixed rows against production ``__getitem__`` on CPU."""
    if 'legacy' in args.panels:
        raise ValueError('--contract-verification-only supports current panels only')

    @lru_cache(maxsize=4)
    def load_dino(path):
        with np.load(path) as payload:
            return payload['dino_cls'].astype(np.float32)

    panels = {}
    for panel_name, anchor in (('current8', 8), ('current39', 39)):
        if panel_name not in args.panels:
            continue
        dataset = _dataset(args, anchor_min=anchor, exclude_recent=83)
        descriptors = _current_descriptors(dataset)
        panels[panel_name] = {
            'dataset_fingerprint': dataset.dataset_fingerprint,
            'eval_dataset_fingerprint': _eval_subset_fingerprint(dataset),
            'retrieval_anchor_min_frame': anchor,
            'verification': _verify_current_getitem(
                dataset, descriptors, args.sampling_seeds[0], load_dino
            ),
        }
    return {
        'metadata': {
            'evaluation_type': 'fixed retrieval dataset contract verification',
            'root_dir': args.root_dir,
            'feature_root': args.feature_root,
            'data_split': args.data_split,
            'validation_fraction': args.validation_fraction,
            'split_seed': args.split_seed,
        },
        'panels': panels,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lingbot-repo', required=True)
    parser.add_argument('--lingbot-weights', required=True)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--num-scale', type=int, default=8)
    parser.add_argument('--max-frame-num', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sampling-seeds', type=int, nargs='+', default=(0, 1, 2, 3, 4))
    parser.add_argument('--selection-seed', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--raw-temperature', type=float, default=0.01)
    parser.add_argument('--panels', nargs='+', choices=('legacy', 'current8', 'current39'), default=('legacy', 'current8', 'current39'))
    parser.add_argument('--data-split', choices=('all', 'train', 'val'), default='val')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--meta-filename', default='meta/gen_meta.json')
    parser.add_argument('--strict-feature-coverage', action='store_true')
    parser.add_argument('--require-versioned-cache', action='store_true')
    parser.add_argument('--expected-cache-signature', default='')
    parser.add_argument('--save-records', action='store_true')
    parser.add_argument('--verify-current-getitem', action='store_true')
    parser.add_argument('--contract-verification-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if not args.sampling_seeds:
        parser.error('--sampling-seeds cannot be empty')
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds):
        parser.error('--sampling-seeds must be unique')
    if args.max_samples < 0:
        parser.error('--max-samples cannot be negative')
    if args.raw_temperature <= 0 or not math.isfinite(args.raw_temperature):
        parser.error('--raw-temperature must be finite and positive')
    if args.require_versioned_cache and not args.expected_cache_signature:
        parser.error('--require-versioned-cache needs --expected-cache-signature')
    return args


def main():
    args = parse_args()
    if not args.contract_verification_only and not torch.cuda.is_available():
        raise RuntimeError('legacy retrieval evaluation requires CUDA for goal DINO')
    torch.manual_seed(0)
    np.random.seed(0)
    result = (
        collect_contract_verification(args)
        if args.contract_verification_only else collect(args)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    concise = result['panels'] if args.contract_verification_only else {
        name: {
            'metadata': panel['metadata'],
            'per_seed': panel['per_seed'],
            'across_seed': panel['across_seed'],
        }
        for name, panel in result['panels'].items()
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    print(f'REPORT_SAVED {output}', flush=True)


if __name__ == '__main__':
    main()
