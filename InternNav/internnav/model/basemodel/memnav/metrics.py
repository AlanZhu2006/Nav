import torch


def compute_memnav_batch_totals(outputs, batch):
    """Return additive offline metrics for one MemNav batch."""
    logits = outputs['ret_logits']
    device = logits.device
    revisit = batch['batch_is_revisit'].to(device).bool()
    null_pos = batch['batch_null_pos'].to(device).bool()
    if not torch.equal(null_pos, ~revisit):
        raise ValueError('null targets must be the inverse of metadata revisit labels')

    pos_real = batch['batch_pos_mask'].to(device).bool()
    neg_real = batch['batch_neg_mask'].to(device).bool()
    if pos_real.shape[1] + 1 != logits.shape[1]:
        raise ValueError('retrieval masks must align with logits before the null slot')

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

    noise = outputs['noise']
    reduce_dims = tuple(range(1, noise.ndim))
    ng = (outputs['noise_ng'] - noise).square().mean(reduce_dims)
    mg = (outputs['noise_mg'] - noise).square().mean(reduce_dims)
    action = 0.5 * (ng + mg)

    goal_pose = batch['batch_goal_rel_pose'].to(device)
    aux = (outputs['aux_pose'] - goal_pose).square().mean(-1)
    gate = outputs['revisit_gate']
    revisit_f = revisit.float()
    novel_f = (~revisit).float()

    return {
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
        'sum_gate_revisit': float((gate * revisit_f).sum().item()),
        'sum_gate_novel': float((gate * novel_f).sum().item()),
        'sum_aux_revisit': float((aux * revisit_f).sum().item()),
    }


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
        'gate_revisit': gate_revisit,
        'gate_novel': gate_novel,
        'gate_separation': (
            gate_revisit - gate_novel
            if gate_revisit is not None and gate_novel is not None else None
        ),
        'aux_pose_mse_revisit': aux,
    }
