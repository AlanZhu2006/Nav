import math

import torch


def _compute_memnav_batch_vectors(outputs, batch):
    """Validate a batch and return the per-sample vectors used by diagnostics."""
    logits = outputs['ret_logits']
    device = logits.device
    revisit = batch['batch_is_revisit'].to(device).bool()
    null_pos = batch['batch_null_pos'].to(device).bool()
    if not torch.equal(null_pos, ~revisit):
        raise ValueError('null targets must be the inverse of metadata revisit labels')

    pos_real = batch['batch_pos_mask'].to(device).bool()
    neg_real = batch['batch_neg_mask'].to(device).bool()
    candidate_real = batch['batch_mem_mask'].to(device).bool()
    if pos_real.shape != neg_real.shape or pos_real.shape != candidate_real.shape:
        raise ValueError('retrieval masks must have identical shapes')
    if pos_real.shape[1] + 1 != logits.shape[1]:
        raise ValueError('retrieval masks must align with logits before the null slot')
    if (pos_real & neg_real).any() or ((pos_real | neg_real) & ~candidate_real).any():
        raise ValueError('retrieval labels must be disjoint structural candidates')
    if pos_real[~revisit].any():
        raise ValueError('novel samples cannot contain real positive labels')

    pos_full = torch.cat([pos_real, null_pos[:, None]], dim=1)
    neg_full = torch.cat([neg_real, revisit[:, None]], dim=1)
    valid = pos_full | neg_full
    if not pos_full.any(-1).all() or not valid.any(-1).all():
        raise ValueError('every retrieval sample needs a valid positive')

    neg_inf = torch.finfo(logits.dtype).min
    retrieval = (
        logits.masked_fill(~valid, neg_inf).logsumexp(-1)
        - logits.masked_fill(~pos_full, neg_inf).logsumexp(-1)
    )
    prediction = logits.argmax(-1)
    correct = pos_full.gather(1, prediction[:, None]).squeeze(1).float()

    null_index = logits.shape[1] - 1
    pred_null = prediction == null_index
    real_prediction = prediction.clamp(max=null_index - 1)
    pred_candidate = candidate_real.gather(1, real_prediction[:, None]).squeeze(1)
    pred_positive = (
        ~pred_null & pos_real.gather(1, real_prediction[:, None]).squeeze(1)
    )
    pred_negative = (
        ~pred_null & neg_real.gather(1, real_prediction[:, None]).squeeze(1)
    )
    pred_ignored = ~pred_null & pred_candidate & ~pred_positive & ~pred_negative
    if (~pred_null & ~pred_candidate).any():
        raise ValueError('retrieval selected a structurally masked frame')

    # The policy executes the best real frame regardless of the null logit; the
    # null probability is used separately as a soft revisit/novel gate.
    top_real_prediction = logits[:, :-1].argmax(-1)
    top_real_candidate = candidate_real.gather(
        1, top_real_prediction[:, None]
    ).squeeze(1)
    top_real_positive = pos_real.gather(
        1, top_real_prediction[:, None]
    ).squeeze(1)
    top_real_negative = neg_real.gather(
        1, top_real_prediction[:, None]
    ).squeeze(1)
    top_real_ignored = top_real_candidate & ~top_real_positive & ~top_real_negative
    if (~top_real_candidate).any():
        raise ValueError('retrieval selected a structurally masked real frame')

    noise = outputs['noise']
    reduce_dims = tuple(range(1, noise.ndim))
    ng = (outputs['noise_ng'] - noise).square().mean(reduce_dims)
    mg = (outputs['noise_mg'] - noise).square().mean(reduce_dims)
    action = 0.5 * (ng + mg)
    oracle_gate_action = None
    if outputs.get('noise_oracle_gate') is not None:
        oracle_gate_action = (
            outputs['noise_oracle_gate'] - noise
        ).square().mean(reduce_dims)

    goal_pose = batch['batch_goal_rel_pose'].to(device)
    aux = (outputs['aux_pose'] - goal_pose).square().mean(-1)
    gate = outputs['revisit_gate']
    gate_predicts_revisit = gate >= 0.5
    return {
        'logits': logits,
        'revisit': revisit,
        'pos_real': pos_real,
        'neg_real': neg_real,
        'candidate_real': candidate_real,
        'retrieval': retrieval,
        'prediction': prediction,
        'correct': correct,
        'pred_null': pred_null,
        'pred_positive': pred_positive,
        'pred_negative': pred_negative,
        'pred_ignored': pred_ignored,
        'top_real_prediction': top_real_prediction,
        'top_real_positive': top_real_positive,
        'top_real_negative': top_real_negative,
        'top_real_ignored': top_real_ignored,
        'ng': ng,
        'mg': mg,
        'action': action,
        'oracle_gate_action': oracle_gate_action,
        'aux': aux,
        'gate': gate,
        'gate_predicts_revisit': gate_predicts_revisit,
    }


def compute_memnav_batch_totals(outputs, batch):
    """Return additive offline metrics for one MemNav batch."""
    vectors = _compute_memnav_batch_vectors(outputs, batch)
    logits = vectors['logits']
    revisit = vectors['revisit']
    revisit_f = revisit.float()
    novel_f = (~revisit).float()
    ng = vectors['ng']
    mg = vectors['mg']
    action = vectors['action']
    retrieval = vectors['retrieval']
    correct = vectors['correct']
    pred_positive = vectors['pred_positive']
    pred_negative = vectors['pred_negative']
    pred_ignored = vectors['pred_ignored']
    pred_null = vectors['pred_null']
    top_real_positive = vectors['top_real_positive']
    top_real_negative = vectors['top_real_negative']
    top_real_ignored = vectors['top_real_ignored']
    gate = vectors['gate']
    gate_predicts_revisit = vectors['gate_predicts_revisit']
    aux = vectors['aux']

    totals = {
        'num_samples': int(logits.shape[0]),
        'num_revisit': int(revisit.sum().item()),
        'num_novel': int((~revisit).sum().item()),
        'sum_ng_loss': float(ng.sum().item()),
        'sum_mg_loss': float(mg.sum().item()),
        'sum_action_loss': float(action.sum().item()),
        'sum_action_revisit': float((action * revisit_f).sum().item()),
        'sum_action_novel': float((action * novel_f).sum().item()),
        'sum_retrieval_loss': float(retrieval.sum().item()),
        'sum_retrieval_correct': float(correct.sum().item()),
        'sum_revisit_correct': float((correct * revisit_f).sum().item()),
        'sum_novel_correct': float((correct * novel_f).sum().item()),
        'sum_revisit_pred_positive': int((pred_positive & revisit).sum().item()),
        'sum_revisit_pred_negative': int((pred_negative & revisit).sum().item()),
        'sum_revisit_pred_ignored': int((pred_ignored & revisit).sum().item()),
        'sum_revisit_pred_null': int((pred_null & revisit).sum().item()),
        'sum_novel_pred_positive': int((pred_positive & ~revisit).sum().item()),
        'sum_novel_pred_negative': int((pred_negative & ~revisit).sum().item()),
        'sum_novel_pred_ignored': int((pred_ignored & ~revisit).sum().item()),
        'sum_novel_pred_null': int((pred_null & ~revisit).sum().item()),
        'sum_revisit_top_real_positive': int(
            (top_real_positive & revisit).sum().item()
        ),
        'sum_revisit_top_real_negative': int(
            (top_real_negative & revisit).sum().item()
        ),
        'sum_revisit_top_real_ignored': int(
            (top_real_ignored & revisit).sum().item()
        ),
        'sum_gate_binary_correct': int(
            (gate_predicts_revisit == revisit).sum().item()
        ),
        'sum_gate_revisit_correct': int(
            (gate_predicts_revisit & revisit).sum().item()
        ),
        'sum_gate_novel_correct': int(
            (~gate_predicts_revisit & ~revisit).sum().item()
        ),
        'sum_gate_revisit': float((gate * revisit_f).sum().item()),
        'sum_gate_novel': float((gate * novel_f).sum().item()),
        'sum_aux_revisit': float((aux * revisit_f).sum().item()),
    }
    oracle_gate_action = vectors['oracle_gate_action']
    if oracle_gate_action is not None:
        totals.update({
            'sum_oracle_gate_action_loss': float(oracle_gate_action.sum().item()),
            'sum_oracle_gate_action_revisit': float(
                (oracle_gate_action * revisit_f).sum().item()
            ),
            'sum_oracle_gate_action_novel': float(
                (oracle_gate_action * novel_f).sum().item()
            ),
        })
    return totals


def merge_memnav_totals(totals, batch_totals):
    for key, value in batch_totals.items():
        totals[key] = totals.get(key, 0) + value
    return totals


def finalize_memnav_metrics(totals, w_retrieval=1.0, w_aux_pose=0.5):
    count = int(totals.get('num_samples', 0))
    revisit = int(totals.get('num_revisit', 0))
    novel = int(totals.get('num_novel', 0))
    if count == 0:
        raise ValueError('cannot finalize empty MemNav metrics')

    def average(name, denominator):
        if denominator == 0:
            return None
        return float(totals.get(name, 0.0) / denominator)

    action = average('sum_action_loss', count)
    retrieval = average('sum_retrieval_loss', count)
    aux = average('sum_aux_revisit', revisit)
    total = action + w_retrieval * retrieval
    if aux is not None:
        total += w_aux_pose * aux

    gate_revisit = average('sum_gate_revisit', revisit)
    gate_novel = average('sum_gate_novel', novel)
    oracle_gate_action = (
        average('sum_oracle_gate_action_loss', count)
        if 'sum_oracle_gate_action_loss' in totals else None
    )
    return {
        'num_samples': count,
        'num_revisit': revisit,
        'num_novel': novel,
        'revisit_fraction': revisit / count,
        'loss': total,
        'action_loss': action,
        'ng_loss': average('sum_ng_loss', count),
        'mg_loss': average('sum_mg_loss', count),
        'action_loss_revisit': average('sum_action_revisit', revisit),
        'action_loss_novel': average('sum_action_novel', novel),
        'retrieval_loss': retrieval,
        'retrieval_accuracy': average('sum_retrieval_correct', count),
        'revisit_match_accuracy': average('sum_revisit_correct', revisit),
        'novel_null_accuracy': average('sum_novel_correct', novel),
        'revisit_pred_positive_fraction': average('sum_revisit_pred_positive', revisit),
        'revisit_pred_negative_fraction': average('sum_revisit_pred_negative', revisit),
        'revisit_pred_ignored_fraction': average('sum_revisit_pred_ignored', revisit),
        'revisit_pred_null_fraction': average('sum_revisit_pred_null', revisit),
        'novel_pred_positive_fraction': average('sum_novel_pred_positive', novel),
        'novel_pred_negative_fraction': average('sum_novel_pred_negative', novel),
        'novel_pred_ignored_fraction': average('sum_novel_pred_ignored', novel),
        'novel_pred_null_fraction': average('sum_novel_pred_null', novel),
        'revisit_top_real_match_accuracy': average(
            'sum_revisit_top_real_positive', revisit
        ),
        'revisit_top_real_negative_fraction': average(
            'sum_revisit_top_real_negative', revisit
        ),
        'revisit_top_real_ignored_fraction': average(
            'sum_revisit_top_real_ignored', revisit
        ),
        'gate_accuracy_at_0_5': average('sum_gate_binary_correct', count),
        'gate_revisit_accuracy_at_0_5': average('sum_gate_revisit_correct', revisit),
        'gate_novel_accuracy_at_0_5': average('sum_gate_novel_correct', novel),
        'gate_revisit': gate_revisit,
        'gate_novel': gate_novel,
        'gate_separation': (
            gate_revisit - gate_novel
            if gate_revisit is not None and gate_novel is not None else None
        ),
        'aux_pose_mse_revisit': aux,
        'oracle_gate_action_loss': oracle_gate_action,
        'oracle_gate_action_loss_revisit': (
            average('sum_oracle_gate_action_revisit', revisit)
            if oracle_gate_action is not None else None
        ),
        'oracle_gate_action_loss_novel': (
            average('sum_oracle_gate_action_novel', novel)
            if oracle_gate_action is not None else None
        ),
        'oracle_gate_delta_vs_mg': (
            oracle_gate_action - average('sum_mg_loss', count)
            if oracle_gate_action is not None else None
        ),
    }


def compute_memnav_batch_records(outputs, batch):
    """Return JSON-serializable per-sample diagnostics for one batch."""
    vectors = _compute_memnav_batch_vectors(outputs, batch)
    logits = vectors['logits']
    probabilities = logits.softmax(-1)
    revisit = vectors['revisit']
    pos_real = vectors['pos_real']
    neg_real = vectors['neg_real']
    candidate_real = vectors['candidate_real']
    ignored_real = candidate_real & ~pos_real & ~neg_real
    top_real_prediction = vectors['top_real_prediction']
    top_real_logit = logits[:, :-1].gather(
        1, top_real_prediction[:, None]
    ).squeeze(1)
    null_logit = logits[:, -1]

    count = int(logits.shape[0])
    cur_steps = batch.get('cur_steps')
    goal_steps = batch.get('goal_steps')
    cache_paths = batch.get('cache_paths')
    if cur_steps is None:
        cur_steps = [int(candidate_real.shape[1] - 1)] * count
    if goal_steps is None:
        goal_steps = [None] * count
    if cache_paths is None:
        cache_paths = [None] * count
    if not (len(cur_steps) == len(goal_steps) == len(cache_paths) == count):
        raise ValueError('per-sample metadata must align with the batch size')

    def outcome(i, prefix):
        if prefix == 'joint':
            if bool(vectors['pred_null'][i]):
                return 'null'
            positive = vectors['pred_positive'][i]
            negative = vectors['pred_negative'][i]
            ignored = vectors['pred_ignored'][i]
        else:
            positive = vectors['top_real_positive'][i]
            negative = vectors['top_real_negative'][i]
            ignored = vectors['top_real_ignored'][i]
        if bool(positive):
            return 'positive'
        if bool(negative):
            return 'negative'
        if bool(ignored):
            return 'ignored'
        raise ValueError(f'unclassified {prefix} retrieval outcome')

    records = []
    for i in range(count):
        record = {
            'cache_path': cache_paths[i],
            'cur_step': int(cur_steps[i]),
            'memory_length': int(cur_steps[i]) + 1,
            'goal_step': (
                int(goal_steps[i]) if goal_steps[i] is not None else None
            ),
            'is_revisit': bool(revisit[i]),
            'num_candidates': int(candidate_real[i].sum().item()),
            'num_positive': int(pos_real[i].sum().item()),
            'num_negative': int(neg_real[i].sum().item()),
            'num_ignored': int(ignored_real[i].sum().item()),
            'retrieval_loss': float(vectors['retrieval'][i].item()),
            'retrieval_correct': bool(vectors['correct'][i]),
            'joint_prediction': int(vectors['prediction'][i].item()),
            'joint_outcome': outcome(i, 'joint'),
            'top_real_prediction': int(top_real_prediction[i].item()),
            'top_real_outcome': outcome(i, 'top_real'),
            'gate': float(vectors['gate'][i].item()),
            'null_probability': float(probabilities[i, -1].item()),
            'top_real_logit': float(top_real_logit[i].item()),
            'null_logit': float(null_logit[i].item()),
            'max_real_null_margin': float(
                (top_real_logit[i] - null_logit[i]).item()
            ),
            'action_loss': float(vectors['action'][i].item()),
            'ng_action_loss': float(vectors['ng'][i].item()),
            'mg_action_loss': float(vectors['mg'][i].item()),
            'aux_pose_mse': float(vectors['aux'][i].item()),
        }
        if vectors['oracle_gate_action'] is not None:
            record['oracle_gate_action_loss'] = float(
                vectors['oracle_gate_action'][i].item()
            )
        records.append(record)
    return records


def _binary_score_metrics(records, score_key, threshold):
    labels = [bool(record['is_revisit']) for record in records]
    predictions = [
        float(record[score_key]) >= float(threshold) for record in records
    ]
    num_revisit = sum(labels)
    num_novel = len(labels) - num_revisit
    true_positive = sum(pred and label for pred, label in zip(predictions, labels))
    true_negative = sum(
        (not pred) and (not label) for pred, label in zip(predictions, labels)
    )
    false_positive = num_novel - true_negative
    false_negative = num_revisit - true_positive

    revisit_recall = true_positive / num_revisit if num_revisit else None
    novel_recall = true_negative / num_novel if num_novel else None
    balanced = (
        0.5 * (revisit_recall + novel_recall)
        if revisit_recall is not None and novel_recall is not None else None
    )
    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator if precision_denominator else None
    )
    f1 = (
        2.0 * precision * revisit_recall / (precision + revisit_recall)
        if precision is not None and revisit_recall is not None
        and precision + revisit_recall > 0 else None
    )
    return {
        'threshold': float(threshold),
        'accuracy': (true_positive + true_negative) / len(labels),
        'balanced_accuracy': balanced,
        'revisit_recall': revisit_recall,
        'novel_recall': novel_recall,
        'revisit_precision': precision,
        'revisit_f1': f1,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative,
    }


def compute_gate_threshold_sweep(
    records,
    score_key='gate',
    reference_threshold=0.5,
    include_points=True,
):
    """Find the exact balanced-accuracy optimum over all observed score gaps."""
    if not records:
        raise ValueError('cannot sweep an empty record set')
    scores = sorted({float(record[score_key]) for record in records})
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(f'{score_key} contains non-finite scores')
    span = max(scores) - min(scores)
    epsilon = max(1e-6, span * 1e-6)
    thresholds = [scores[0] - epsilon]
    thresholds.extend(
        0.5 * (left + right) for left, right in zip(scores, scores[1:])
    )
    thresholds.append(scores[-1] + epsilon)

    points = [
        _binary_score_metrics(records, score_key, threshold)
        for threshold in thresholds
    ]
    valid = [point for point in points if point['balanced_accuracy'] is not None]
    if not valid:
        raise ValueError('threshold sweep requires both revisit and novel samples')
    best = max(
        valid,
        key=lambda point: (
            point['balanced_accuracy'],
            point['accuracy'],
            -abs(point['threshold'] - reference_threshold),
        ),
    )
    result = {
        'score_key': score_key,
        'num_samples': len(records),
        'reference': _binary_score_metrics(
            records, score_key, reference_threshold
        ),
        'best': best,
    }
    if include_points:
        result['points'] = points
    return result


def _mean(records, key, predicate=None):
    values = [
        float(record[key]) for record in records
        if key in record and (predicate is None or predicate(record))
    ]
    return math.fsum(values) / len(values) if values else None


def _fraction(records, predicate):
    if not records:
        return None
    return sum(bool(predicate(record)) for record in records) / len(records)


def compute_memory_length_diagnostics(records):
    """Summarize retrieval, gate, and action behavior by observed history length."""
    definitions = (
        ('up_to_320', None, 320),
        ('321_to_1024', 320, 1024),
        ('1025_to_2048', 1024, 2048),
        ('over_2048', 2048, None),
    )
    buckets = {}
    for name, lower, upper in definitions:
        selected = [
            record for record in records
            if (lower is None or record['memory_length'] > lower)
            and (upper is None or record['memory_length'] <= upper)
        ]
        if not selected:
            buckets[name] = {'num_samples': 0}
            continue
        revisit = [record for record in selected if record['is_revisit']]
        novel = [record for record in selected if not record['is_revisit']]
        gate_sweep = compute_gate_threshold_sweep(
            selected, 'gate', 0.5, include_points=False
        ) if revisit and novel else None
        margin_sweep = compute_gate_threshold_sweep(
            selected, 'max_real_null_margin', 0.0, include_points=False
        ) if revisit and novel else None
        oracle_gate = _mean(selected, 'oracle_gate_action_loss')
        mg_action = _mean(selected, 'mg_action_loss')
        buckets[name] = {
            'num_samples': len(selected),
            'num_revisit': len(revisit),
            'num_novel': len(novel),
            'mean_memory_length': _mean(selected, 'memory_length'),
            'action_loss': _mean(selected, 'action_loss'),
            'ng_action_loss': _mean(selected, 'ng_action_loss'),
            'mg_action_loss': mg_action,
            'oracle_gate_action_loss': oracle_gate,
            'oracle_gate_delta_vs_mg': (
                oracle_gate - mg_action
                if oracle_gate is not None and mg_action is not None else None
            ),
            'retrieval_loss': _mean(selected, 'retrieval_loss'),
            'retrieval_accuracy': _mean(
                selected, 'retrieval_correct'
            ),
            'revisit_top_real_match_accuracy': _fraction(
                revisit, lambda record: record['top_real_outcome'] == 'positive',
            ),
            'revisit_top_real_negative_fraction': _fraction(
                revisit, lambda record: record['top_real_outcome'] == 'negative',
            ),
            'novel_null_accuracy': _fraction(
                novel, lambda record: record['joint_outcome'] == 'null',
            ),
            'gate_revisit': _mean(revisit, 'gate'),
            'gate_novel': _mean(novel, 'gate'),
            'gate_threshold': gate_sweep,
            'max_real_null_margin_threshold': margin_sweep,
        }
    return buckets
