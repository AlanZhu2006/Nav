"""Small, dependency-light helpers for the MemNav revisit-gate curriculum.

The retrieval head predicts whether the goal is already in memory.  That scalar is
also used as a decoder attention bias, so a poor gate early in training can hide the
teacher-forced revisit token from the action loss.  These helpers let the trainer
start from the ground-truth revisit label and linearly hand control back to the
predicted gate.  They intentionally contain no model imports so the schedule and its
gradient behaviour can be unit-tested without constructing LingBot.
"""

from __future__ import annotations

import torch


def linear_teacher_ratio(step: int, start: float, end: float, decay_steps: int) -> float:
    """Return a clamped linear teacher-forcing ratio for ``step``.

    ``decay_steps == 0`` means the transition is disabled and ``end`` is used
    immediately.  This makes ``start=end=0`` a clean backward-compatible off switch.
    """
    if not 0.0 <= start <= 1.0:
        raise ValueError(f"teacher start must be in [0, 1], got {start}")
    if not 0.0 <= end <= 1.0:
        raise ValueError(f"teacher end must be in [0, 1], got {end}")
    if decay_steps < 0:
        raise ValueError(f"teacher decay_steps must be >= 0, got {decay_steps}")
    if decay_steps == 0:
        return float(end)
    progress = min(max(float(step), 0.0) / float(decay_steps), 1.0)
    return float(start + (end - start) * progress)


def blend_decoder_gate(
    predicted_gate: torch.Tensor,
    is_revisit: torch.Tensor,
    teacher_ratio: float,
    *,
    training: bool,
) -> torch.Tensor:
    """Blend predicted and ground-truth gates while preserving useful gradients.

    At ratio 1 the decoder sees the exact revisit label; at ratio 0 this is exactly
    the original inference path.  Between them, the action-loss gradient reaching the
    predicted gate is scaled by ``1 - teacher_ratio``.  Gate BCE is independent and
    continues to train the retrieval head even at ratio 1.
    """
    if not training:
        return predicted_gate
    ratio = float(teacher_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"teacher_ratio must be in [0, 1], got {ratio}")
    if ratio == 0.0:
        return predicted_gate
    target = is_revisit.to(device=predicted_gate.device, dtype=predicted_gate.dtype)
    return predicted_gate + ratio * (target - predicted_gate)
