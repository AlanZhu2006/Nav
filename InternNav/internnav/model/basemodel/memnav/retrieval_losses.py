"""Retrieval objectives shared by training and dependency-light tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_retrieval_losses(
    logits: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    top1_margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return set-InfoNCE, top-1 margin, and the valid-row mask.

    Set-InfoNCE rewards total probability assigned to every valid positive, but
    live inference uses one ``argmax`` anchor.  The optional hinge term directly
    encodes that deployment condition by requiring the best positive logit to
    exceed the best negative by ``top1_margin``.

    Invalid/novel rows are indexed out *before* subtraction so finite dtype
    floors cannot overflow under mixed precision.  Empty batches return
    differentiable zeros.
    """
    if logits.shape != positive.shape or logits.shape != negative.shape:
        raise ValueError(
            f"logits/positive/negative shapes must match, got "
            f"{tuple(logits.shape)}, {tuple(positive.shape)}, {tuple(negative.shape)}"
        )
    positive = positive.bool()
    negative = negative.bool()
    valid = positive.any(-1) & negative.any(-1)
    floor = torch.finfo(logits.dtype).min

    lse_all = logits.masked_fill(~(positive | negative), floor).logsumexp(-1)
    lse_pos = logits.masked_fill(~positive, floor).logsumexp(-1)
    denom = valid.sum().clamp(min=1)
    set_loss = (lse_all[valid] - lse_pos[valid]).sum() / denom

    best_pos = logits.masked_fill(~positive, floor).max(-1).values
    best_neg = logits.masked_fill(~negative, floor).max(-1).values
    top1_loss = F.relu(best_neg[valid] - best_pos[valid] + float(top1_margin)).sum() / denom
    return set_loss, top1_loss, valid
