"""Lightweight helpers for decoder revisit/visual branch fusion."""

from __future__ import annotations

import torch


GATE_FUSIONS = ("complementary", "residual")


def gate_fusion_code(mode: str) -> float:
    """Return the persistent numeric code used in MemNav checkpoints."""
    if mode not in GATE_FUSIONS:
        raise ValueError(f"gate_fusion must be one of {GATE_FUSIONS}, got {mode!r}")
    return float(mode == "residual")


def branch_log_weights(
    gate: torch.Tensor,
    residual_code: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log attention weights for revisit and visual-goal columns.

    ``complementary`` (code 0) reproduces the original routing exactly:
    ``[log(g), log(1-g)]``.  ``residual`` (code 1) treats the visual goal as
    the always-available base policy and adds memory only when confident:
    ``[log(g), 0]``.  Keeping the code tensor-valued avoids a GPU ``.item()``
    synchronization in every diffusion iteration and lets checkpoint loading
    switch semantics without rebuilding the model.
    """
    g = gate.clamp(1e-4, 1 - 1e-4)
    code = torch.as_tensor(residual_code, device=g.device, dtype=g.dtype)
    if code.numel() != 1:
        raise ValueError(f"residual_code must be scalar, got shape {tuple(code.shape)}")
    revisit = torch.log(g)
    visual = (1.0 - code) * torch.log1p(-g)
    return revisit, visual
