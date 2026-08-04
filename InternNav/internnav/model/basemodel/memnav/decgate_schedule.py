"""Decoder-gate routing helpers: fusion-mode math + logit-space teacher curriculum.

Why these exist (diag_retrieval/diag_decgate_zsweep.py on ckpt-5570): the action
loss is LOCALLY OPTIMAL in the decoder gate logit z — moving z by ±2 is flat and
±4 hurts on both revisit and novel rows — so dec_gate_a/b random-walk at init
instead of training.  The decoder co-adapted to the closed-gate routing it saw
from step 0 (z ≈ 10·max_cos − 8 ≈ −5 with fresh projections) and never learned
to read the revisit tokens, which removes the very gradient that would open the
gate.  Same deadlock the probability-space gate curriculum broke on the old
shared-gate architecture; these helpers break it for the decoder-owned gate.

Pure functions (no model imports) so the schedule, blend gradients, and fusion
bias math are unit-testable without constructing LingBot.
"""

from __future__ import annotations

import torch

# symmetric  : revisit += z/2, novel -= z/2 (zero common mode vs obstacle columns)
# residual   : revisit += z, novel += 0 (visual-goal branch is the always-on base
#              policy; memory is strictly additive — the §10.5 lesson that the
#              visual branch has value even on true revisits)
# value_scale: no attention bias; revisit token VALUES *= sigmoid(z) (and novel
#              *= 1-sigmoid(z) when scale_novel) — gradient to z scales with the
#              readout magnitude instead of the attention weight on a suppressed
#              column, so a closed gate still passes usable gradient
DECGATE_FUSIONS = ("symmetric", "residual", "value_scale")
# same tilt ceiling the original g∈[1e-4, 1-1e-4] clamp imposed on log-odds
Z_CLAMP = 9.2


def decgate_fusion_code(mode: str) -> float:
    """Persistent numeric code stored in checkpoints (buffer)."""
    if mode not in DECGATE_FUSIONS:
        raise ValueError(f"dec_gate_fusion must be one of {DECGATE_FUSIONS}, got {mode!r}")
    return float(DECGATE_FUSIONS.index(mode))


def branch_bias_values(dec_gate_logit: torch.Tensor, fusion: str):
    """Per-sample (revisit_bias, novel_bias) attention-logit offsets, both [B].

    Only for the bias fusions; ``value_scale`` never builds an attention bias.
    """
    z = dec_gate_logit.clamp(-Z_CLAMP, Z_CLAMP)
    if fusion == "symmetric":
        return 0.5 * z, -0.5 * z
    if fusion == "residual":
        return z, torch.zeros_like(z)
    raise ValueError(f"no attention bias for fusion {fusion!r}")


def linear_teacher_ratio(step: int, start: float, end: float, decay_steps: int) -> float:
    """Clamped linear teacher ratio; ``decay_steps == 0`` disables (returns ``end``)."""
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


def blend_decoder_gate_logit(
    z_pred: torch.Tensor,
    is_revisit: torch.Tensor,
    teacher_ratio: float,
    teacher_z: float,
) -> torch.Tensor:
    """Logit-space port of the probability-space gate curriculum.

    ``z_used = z_pred + r·(±teacher_z − z_pred)``: at r=1 the decoder routes by
    the GT revisit label (magnitude ``teacher_z``, NOT the ±9.2 rail — the fixed
    eval showed the visual branch has value even on GT revisits, so the teacher
    opens the gate without annihilating it); at r=0 this is exactly the
    inference path.  d(z_used)/d(z_pred) = 1−r, so the action-loss gradient
    hands over linearly.  Caller guards on ``training`` — eval never blends.
    """
    ratio = float(teacher_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"teacher_ratio must be in [0, 1], got {ratio}")
    if teacher_z <= 0.0:
        raise ValueError(f"teacher_z must be > 0, got {teacher_z}")
    if ratio == 0.0:
        return z_pred
    target = is_revisit.to(device=z_pred.device, dtype=z_pred.dtype)
    z_gt = (2.0 * target - 1.0) * float(teacher_z)          # ±teacher_z by GT label
    return z_pred + ratio * (z_gt - z_pred)
