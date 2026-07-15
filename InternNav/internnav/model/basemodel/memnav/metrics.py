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
            'cur_step': int(cur_steps[index]),
            'goal_step': int(goal_steps[index]) if goal_steps[index] is not None else None,
            'memory_length': int(cur_steps[index]) + 1,
            'is_revisit': bool(revisit[index]),
            'num_candidates': int(candidate[index].sum().item()),
            'num_positive': int(pos[index].sum().item()),
            'num_negative': int(neg[index].sum().item()),
            'rank_loss': float(rank_loss[index].item()) if bool(rank_rows[index]) else None,
            'match_index': int(prediction[index].item()),
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
    }
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
