"""Deterministic offline diagnostics for the current MemNav architecture."""

import math
import statistics

import torch
import torch.nn.functional as F


def _rotation_error_degrees(prediction, target):
    cosine = (((prediction * target).sum(dim=(-2, -1)) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cosine))


def _action_mse(outputs):
    dims = tuple(range(1, outputs['noise'].ndim))
    return (outputs['noise_pred'] - outputs['noise']).square().mean(dims)


def _action_mse_by_axis(outputs):
    square_error = (outputs['noise_pred'] - outputs['noise']).square()
    return square_error.mean(dim=1)


def _aux_vectors(outputs, batch):
    prediction = outputs['aux_pose']
    target = batch['batch_goal_rel_pose'].to(prediction.device)[..., :2]
    pred_norm = torch.linalg.norm(prediction, dim=-1)
    target_norm = torch.linalg.norm(target, dim=-1)
    pred_unit = prediction / pred_norm.clamp(min=1e-6).unsqueeze(-1)
    target_unit = target / target_norm.clamp(min=1e-6).unsqueeze(-1)
    cosine = (pred_unit * target_unit).sum(-1).clamp(-1, 1)
    return prediction, target, (prediction - target).square(), torch.rad2deg(torch.arccos(cosine))


def attach_full_diffusion_records(
    records,
    sampled_actions,
    shuffled_goal_actions,
    batch,
    shuffle_indices,
):
    """Attach paired complete-DDPM metrics to existing per-sample records.

    Both trajectories must have been generated from identical initial and
    intermediate DDPM randomness.  A positive shuffled-goal penalty means the
    correct goal produces an action closer to the target.
    """
    target = batch['batch_labels'].to(sampled_actions.device)
    shuffled_goal_actions = shuffled_goal_actions.to(sampled_actions.device)
    if sampled_actions.shape != target.shape:
        raise ValueError(
            f'sampled action shape {tuple(sampled_actions.shape)} does not match '
            f'target {tuple(target.shape)}'
        )
    if shuffled_goal_actions.shape != target.shape:
        raise ValueError('shuffled-goal action shape must match the target')
    if len(records) != target.shape[0]:
        raise ValueError('record count must match the diffusion batch size')

    correct_sq = (sampled_actions - target).square()
    shuffled_sq = (shuffled_goal_actions - target).square()
    sensitivity_sq = (sampled_actions - shuffled_goal_actions).square()
    correct_mse = correct_sq.mean(dim=(1, 2))
    shuffled_mse = shuffled_sq.mean(dim=(1, 2))
    sensitivity_mse = sensitivity_sq.mean(dim=(1, 2))
    correct_axis = correct_sq.mean(dim=1)
    shuffled_axis = shuffled_sq.mean(dim=1)
    sensitivity_ratio = torch.sqrt(sensitivity_mse) / torch.sqrt(correct_mse).clamp(min=1e-8)

    if torch.is_tensor(shuffle_indices):
        shuffle_indices = shuffle_indices.detach().cpu().tolist()
    shuffle_indices = [int(index) for index in shuffle_indices]
    if sorted(shuffle_indices) != list(range(len(records))):
        raise ValueError('shuffle_indices must be a batch permutation')
    if any(index == source for index, source in enumerate(shuffle_indices)):
        raise ValueError('goal shuffle must be a derangement (no unchanged rows)')
    identities = batch.get('sample_identities') or [None] * len(records)

    for index, record in enumerate(records):
        record.update({
            'full_diffusion_action_mse': float(correct_mse[index].item()),
            'full_diffusion_shuffled_goal_action_mse': float(
                shuffled_mse[index].item()
            ),
            'full_diffusion_shuffled_goal_penalty': float(
                (shuffled_mse[index] - correct_mse[index]).item()
            ),
            'full_diffusion_goal_sensitivity_mse': float(
                sensitivity_mse[index].item()
            ),
            'full_diffusion_goal_sensitivity_ratio': float(
                sensitivity_ratio[index].item()
            ),
            'shuffled_goal_source_batch_index': shuffle_indices[index],
            'shuffled_goal_source_identity': identities[shuffle_indices[index]],
        })
        for axis_index, axis in enumerate(('x', 'y', 'theta')):
            record[f'full_diffusion_action_mse_{axis}'] = float(
                correct_axis[index, axis_index].item()
            )
            record[f'full_diffusion_shuffled_goal_action_mse_{axis}'] = float(
                shuffled_axis[index, axis_index].item()
            )
    return records


def compute_memnav_batch_records(outputs, batch, oracle_outputs=None):
    """Return JSON-serializable, per-sample records for one fixed batch."""
    logits = outputs['ret_logits']
    device = logits.device
    revisit = batch['batch_is_revisit'].to(device).bool()
    null_pos = batch['batch_null_pos'].to(device).bool()
    if not torch.equal(null_pos, ~revisit):
        raise ValueError('null_pos must be exactly the inverse of is_revisit')

    pos = batch['batch_pos_mask'].to(device).bool()
    neg = batch['batch_neg_mask'].to(device).bool()
    candidate = batch['batch_cand_mask'].to(device).bool()
    if pos.shape != neg.shape or pos.shape != candidate.shape or pos.shape != logits.shape:
        raise ValueError('current retrieval logits and label masks must have identical [B,L] shape')
    if (pos & neg).any() or (pos & ~candidate).any() or (neg & ~candidate).any():
        raise ValueError('retrieval labels must be disjoint subsets of candidate mask')
    if pos[~revisit].any() or (pos.any(-1) != revisit).any():
        raise ValueError('dynamic revisit labels disagree with positive masks')

    prediction = outputs.get('match_idx', logits.argmax(-1))
    selected_candidate = candidate.gather(1, prediction[:, None]).squeeze(1)
    if not selected_candidate.all():
        raise ValueError('retrieval selected a structurally masked frame')
    selected_positive = pos.gather(1, prediction[:, None]).squeeze(1)
    selected_negative = neg.gather(1, prediction[:, None]).squeeze(1)
    selected_ignored = selected_candidate & ~selected_positive & ~selected_negative

    neg_inf = torch.finfo(logits.dtype).min
    rank_rows = pos.any(-1) & neg.any(-1)
    lse_pn = logits.masked_fill(~(pos | neg), neg_inf).logsumexp(-1)
    lse_p = logits.masked_fill(~pos, neg_inf).logsumexp(-1)
    rank_loss = logits.new_zeros(logits.shape[0])
    rank_loss[rank_rows] = lse_pn[rank_rows] - lse_p[rank_rows]
    action = _action_mse(outputs)
    action_by_axis = _action_mse_by_axis(outputs)
    aux_pred, aux_gt, aux_sq, aux_direction_error = _aux_vectors(outputs, batch)
    rotation_error = _rotation_error_degrees(
        outputs['R_rel'], batch['batch_goal_rel_rotation'].to(device)
    )
    basis = outputs['R_rel'].new_tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    converted_rotation = basis @ outputs['R_rel'] @ basis.transpose(-1, -2)
    converted_rotation_error = _rotation_error_degrees(
        converted_rotation, batch['batch_goal_rel_rotation'].to(device)
    )
    gate = outputs['revisit_gate']
    gate_feature = outputs['gate_feature']
    gate_bce = F.binary_cross_entropy(gate, revisit.float(), reduction='none')

    oracle_action = oracle_aux_pred = oracle_aux_sq = oracle_direction_error = None
    oracle_anchor_positive = None
    if oracle_outputs is not None:
        oracle_action = _action_mse(oracle_outputs)
        oracle_aux_pred, _, oracle_aux_sq, oracle_direction_error = _aux_vectors(
            oracle_outputs, batch
        )
        oracle_anchor = oracle_outputs['anchor_idx']
        oracle_anchor_positive = pos.gather(1, oracle_anchor[:, None]).squeeze(1)
        if not (oracle_anchor_positive | ~revisit).all():
            raise ValueError('oracle-positive run did not anchor every revisit row on a positive')

    cur_steps = batch.get('cur_steps') or [logits.shape[1] - 1] * logits.shape[0]
    goal_steps = batch.get('goal_steps') or [None] * logits.shape[0]
    cache_paths = batch.get('cache_paths') or [None] * logits.shape[0]
    goal_labels = batch.get('goal_labels') or [None] * logits.shape[0]
    sample_identities = batch.get('sample_identities') or [None] * logits.shape[0]
    leg_starts = batch.get('leg_starts') or [None] * logits.shape[0]
    goal_js = batch.get('batch_goal_j')
    has_covis = batch.get('batch_has_covis')
    records = []
    for index in range(logits.shape[0]):
        if bool(selected_positive[index]):
            outcome = 'positive'
        elif bool(selected_negative[index]):
            outcome = 'negative'
        elif bool(selected_ignored[index]):
            outcome = 'ignored'
        else:
            raise ValueError('unclassified retrieval result')
        record = {
            'cache_path': cache_paths[index],
            'sample_identity': sample_identities[index],
            'cur_step': int(cur_steps[index]),
            'goal_step': int(goal_steps[index]) if goal_steps[index] is not None else None,
            'goal_j': int(goal_js[index]) if goal_js is not None else None,
            'goal_label': goal_labels[index],
            'has_covis': bool(has_covis[index]) if has_covis is not None else None,
            'leg_start': int(leg_starts[index]) if leg_starts[index] is not None else None,
            'memory_length': int(cur_steps[index]) + 1,
            'remaining_path_span': (
                int(goal_steps[index]) - int(cur_steps[index])
                if goal_steps[index] is not None else None
            ),
            'is_revisit': bool(revisit[index]),
            'num_candidates': int(candidate[index].sum().item()),
            'num_positive': int(pos[index].sum().item()),
            'num_negative': int(neg[index].sum().item()),
            'rank_loss': float(rank_loss[index].item()) if bool(rank_rows[index]) else None,
            'match_index': int(prediction[index].item()),
            'retrieval_temporal_gap': (
                int(cur_steps[index]) - int(prediction[index].item())
            ),
            'match_outcome': outcome,
            'match_correct': bool(selected_positive[index]) if bool(revisit[index]) else None,
            'gate': float(gate[index].item()),
            'gate_feature': float(gate_feature[index].item()),
            'gate_bce': float(gate_bce[index].item()),
            'action_mse': float(action[index].item()),
            'action_noise_mse_x': float(action_by_axis[index, 0].item()),
            'action_noise_mse_y': float(action_by_axis[index, 1].item()),
            'action_noise_mse_theta': float(action_by_axis[index, 2].item()),
            'aux_pred_x': float(aux_pred[index, 0].item()),
            'aux_pred_y': float(aux_pred[index, 1].item()),
            'aux_gt_x': float(aux_gt[index, 0].item()),
            'aux_gt_y': float(aux_gt[index, 1].item()),
            'goal_distance': float(torch.linalg.norm(aux_gt[index]).item()),
            'aux_mse_x': float(aux_sq[index, 0].item()),
            'aux_mse_y': float(aux_sq[index, 1].item()),
            'aux_direction_error_deg': float(aux_direction_error[index].item()),
            'rotation_error_raw_deg': float(rotation_error[index].item()),
            'rotation_error_converted_deg': float(
                converted_rotation_error[index].item()
            ),
        }
        if oracle_outputs is not None:
            record.update({
                'oracle_anchor_positive': bool(oracle_anchor_positive[index]),
                'oracle_action_mse': float(oracle_action[index].item()),
                'oracle_aux_pred_x': float(oracle_aux_pred[index, 0].item()),
                'oracle_aux_pred_y': float(oracle_aux_pred[index, 1].item()),
                'oracle_aux_mse_x': float(oracle_aux_sq[index, 0].item()),
                'oracle_aux_mse_y': float(oracle_aux_sq[index, 1].item()),
                'oracle_aux_direction_error_deg': float(
                    oracle_direction_error[index].item()
                ),
            })
        records.append(record)
    return records


def _mean(records, key, predicate=lambda _: True):
    values = [record[key] for record in records if predicate(record) and record.get(key) is not None]
    return sum(values) / len(values) if values else None


def _std(records, key, predicate=lambda _: True):
    values = [record[key] for record in records if predicate(record) and record.get(key) is not None]
    return statistics.pstdev(values) if values else None


def _binary_metrics(records, threshold):
    labels = [record['is_revisit'] for record in records]
    predictions = [record['gate'] >= threshold for record in records]
    positives = sum(labels)
    negatives = len(labels) - positives
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
    tpr = tp / positives if positives else None
    tnr = tn / negatives if negatives else None
    balanced = 0.5 * (tpr + tnr) if tpr is not None and tnr is not None else None
    return {
        'threshold': float(threshold),
        'accuracy': (tp + tn) / len(records),
        'balanced_accuracy': balanced,
        'revisit_recall': tpr,
        'novel_recall': tnr,
    }


def compute_gate_threshold_sweep(records):
    """Return reference 0.5 performance and the best balanced threshold."""
    if not records:
        raise ValueError('cannot sweep empty records')
    scores = sorted({float(record['gate']) for record in records})
    if not all(math.isfinite(score) for score in scores):
        raise ValueError('gate contains non-finite scores')
    epsilon = max(1e-6, (scores[-1] - scores[0]) * 1e-6)
    thresholds = [scores[0] - epsilon]
    thresholds.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    thresholds.append(scores[-1] + epsilon)
    points = [_binary_metrics(records, threshold) for threshold in thresholds]
    valid = [point for point in points if point['balanced_accuracy'] is not None]
    best = max(
        valid,
        key=lambda point: (
            point['balanced_accuracy'], point['accuracy'],
            -abs(point['threshold'] - 0.5),
        ),
    ) if valid else None
    return {'reference': _binary_metrics(records, 0.5), 'best': best}


_GROUP_METRIC_KEYS = (
    'action_mse',
    'action_noise_mse_x',
    'action_noise_mse_y',
    'action_noise_mse_theta',
    'gate',
    'gate_feature',
    'aux_mse_x',
    'aux_mse_y',
    'aux_direction_error_deg',
    'rotation_error_converted_deg',
    'full_diffusion_action_mse',
    'full_diffusion_shuffled_goal_action_mse',
    'full_diffusion_shuffled_goal_penalty',
    'full_diffusion_goal_sensitivity_mse',
    'full_diffusion_goal_sensitivity_ratio',
)


def _summarize_group(records):
    revisit_records = [record for record in records if record['is_revisit']]
    result = {
        'num_samples': len(records),
        'num_revisit': len(revisit_records),
        'num_novel': len(records) - len(revisit_records),
    }
    for key in _GROUP_METRIC_KEYS:
        # Aux pose and camera-rotation values are intentionally unsupported on
        # novel rows; the tensors exist only because the decoder has a fixed
        # interface.  Never let them contaminate B/C group diagnostics.
        source = (
            revisit_records
            if key.startswith('aux_') or key.startswith('rotation_')
            else records
        )
        value = _mean(source, key)
        if value is not None:
            result[key] = value
    return result


def _group_records(records, key_fn):
    groups = {}
    for record in records:
        key = key_fn(record)
        if key is not None:
            groups.setdefault(str(key), []).append(record)
    return {
        key: _summarize_group(selected)
        for key, selected in sorted(groups.items())
    }


def _gap_bin(value):
    if value is None:
        return None
    if value < 256:
        return '000-255'
    if value < 512:
        return '256-511'
    return '512+'


def _span_bin(value):
    if value is None:
        return None
    if value < 128:
        return '000-127'
    if value < 256:
        return '128-255'
    return '256+'


def summarize_memnav_records(records):
    if not records:
        raise ValueError('cannot summarize empty records')
    revisit = lambda record: record['is_revisit']
    novel = lambda record: not record['is_revisit']
    num_revisit = sum(record['is_revisit'] for record in records)
    num_novel = len(records) - num_revisit

    def fraction(outcome, predicate):
        selected = [record for record in records if predicate(record)]
        return (sum(record['match_outcome'] == outcome for record in selected) / len(selected)
                if selected else None)

    metrics = {
        'num_samples': len(records),
        'num_revisit': num_revisit,
        'num_novel': num_novel,
        'revisit_fraction': num_revisit / len(records),
        'action_mse': _mean(records, 'action_mse'),
        'action_mse_revisit': _mean(records, 'action_mse', revisit),
        'action_mse_novel': _mean(records, 'action_mse', novel),
        'action_noise_mse_x': _mean(records, 'action_noise_mse_x'),
        'action_noise_mse_y': _mean(records, 'action_noise_mse_y'),
        'action_noise_mse_theta': _mean(records, 'action_noise_mse_theta'),
        'rank_loss_revisit': _mean(records, 'rank_loss', revisit),
        'revisit_match_accuracy': _mean(records, 'match_correct', revisit),
        'revisit_match_negative_fraction': fraction('negative', revisit),
        'revisit_match_ignored_fraction': fraction('ignored', revisit),
        'novel_match_negative_fraction': fraction('negative', novel),
        'novel_match_ignored_fraction': fraction('ignored', novel),
        'gate_bce': _mean(records, 'gate_bce'),
        'gate_revisit': _mean(records, 'gate', revisit),
        'gate_novel': _mean(records, 'gate', novel),
        'gate_feature_revisit': _mean(records, 'gate_feature', revisit),
        'gate_feature_novel': _mean(records, 'gate_feature', novel),
        'aux_mse_x_revisit': _mean(records, 'aux_mse_x', revisit),
        'aux_mse_y_revisit': _mean(records, 'aux_mse_y', revisit),
        'aux_direction_error_deg_revisit': _mean(
            records, 'aux_direction_error_deg', revisit
        ),
        'aux_pred_x_mean_revisit': _mean(records, 'aux_pred_x', revisit),
        'aux_pred_x_std_revisit': _std(records, 'aux_pred_x', revisit),
        'aux_pred_y_mean_revisit': _mean(records, 'aux_pred_y', revisit),
        'aux_pred_y_std_revisit': _std(records, 'aux_pred_y', revisit),
        'aux_gt_x_std_revisit': _std(records, 'aux_gt_x', revisit),
        'aux_gt_y_std_revisit': _std(records, 'aux_gt_y', revisit),
        'rotation_error_raw_deg_revisit': _mean(
            records, 'rotation_error_raw_deg', revisit
        ),
        'rotation_error_converted_deg_revisit': _mean(
            records, 'rotation_error_converted_deg', revisit
        ),
        'gate_threshold_sweep': compute_gate_threshold_sweep(records),
        'by_goal_label': _group_records(records, lambda record: record.get('goal_label')),
        'revisit_by_retrieval_gap': _group_records(
            [record for record in records if record['is_revisit']],
            lambda record: _gap_bin(record.get('retrieval_temporal_gap')),
        ),
        'revisit_by_remaining_path_span': _group_records(
            [record for record in records if record['is_revisit']],
            lambda record: _span_bin(record.get('remaining_path_span')),
        ),
    }
    for key in (
        'full_diffusion_action_mse',
        'full_diffusion_shuffled_goal_action_mse',
        'full_diffusion_shuffled_goal_penalty',
        'full_diffusion_goal_sensitivity_mse',
        'full_diffusion_goal_sensitivity_ratio',
        'full_diffusion_action_mse_x',
        'full_diffusion_action_mse_y',
        'full_diffusion_action_mse_theta',
        'full_diffusion_shuffled_goal_action_mse_x',
        'full_diffusion_shuffled_goal_action_mse_y',
        'full_diffusion_shuffled_goal_action_mse_theta',
    ):
        if any(record.get(key) is not None for record in records):
            metrics[key] = _mean(records, key)
            metrics[f'{key}_revisit'] = _mean(records, key, revisit)
            metrics[f'{key}_novel'] = _mean(records, key, novel)
    if metrics['gate_revisit'] is not None and metrics['gate_novel'] is not None:
        metrics['gate_separation'] = metrics['gate_revisit'] - metrics['gate_novel']
    if any('oracle_action_mse' in record for record in records):
        metrics.update({
            'oracle_action_mse': _mean(records, 'oracle_action_mse'),
            'oracle_action_mse_revisit': _mean(records, 'oracle_action_mse', revisit),
            'oracle_aux_mse_x_revisit': _mean(records, 'oracle_aux_mse_x', revisit),
            'oracle_aux_mse_y_revisit': _mean(records, 'oracle_aux_mse_y', revisit),
            'oracle_aux_direction_error_deg_revisit': _mean(
                records, 'oracle_aux_direction_error_deg', revisit
            ),
        })
        if (metrics['oracle_action_mse_revisit'] is not None
                and metrics['action_mse_revisit'] is not None):
            metrics['oracle_action_delta_revisit'] = (
                metrics['oracle_action_mse_revisit'] - metrics['action_mse_revisit']
            )
        else:
            metrics['oracle_action_delta_revisit'] = None
        if (metrics['oracle_aux_mse_y_revisit'] is not None
                and metrics['aux_mse_y_revisit'] is not None):
            metrics['oracle_aux_mse_y_delta_revisit'] = (
                metrics['oracle_aux_mse_y_revisit'] - metrics['aux_mse_y_revisit']
            )
        else:
            metrics['oracle_aux_mse_y_delta_revisit'] = None
    return metrics
