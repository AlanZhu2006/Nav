"""Shared neural architecture for Phase-B training and runtime ranking.

The checkpoint state-dict keys are part of the deployment ABI.  Keeping the
module in a small dependency-light file lets the live MemNav server load the
ranker without importing the training/evaluation stack (pandas, sklearn, and
teacher-facing helpers).
"""

from __future__ import annotations

import torch
from torch import nn


class LingBotNativeLocalizer(nn.Module):
    """Permutation-invariant set localizer plus metric residual covariance."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.no_match_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pose_mean_head = nn.Linear(hidden_dim, 2)
        self.pose_log_variance_head = nn.Linear(hidden_dim, 2)
        # A new checkpoint is exactly the raw LingBot pose. Learned geometry is
        # a residual improvement, never an arbitrary replacement at step zero.
        nn.init.zeros_(self.pose_mean_head.weight)
        nn.init.zeros_(self.pose_mean_head.bias)

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError("features/mask must have [sessions,candidates,...]")
        encoded = self.encoder(features)
        candidate = self.rank_head(encoded).squeeze(-1)
        candidate = candidate.masked_fill(~mask, -1e4)
        weight = mask.unsqueeze(-1).to(encoded.dtype)
        pooled_mean = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled_max = encoded.masked_fill(
            ~mask.unsqueeze(-1), -1e4).max(1).values
        no_match_logit = self.no_match_head(
            torch.cat([pooled_mean, pooled_max], dim=-1)).squeeze(-1)
        residual = self.pose_mean_head(encoded)
        log_variance = self.pose_log_variance_head(encoded).clamp(-6.0, 6.0)
        return candidate, no_match_logit, residual, log_variance


__all__ = ["LingBotNativeLocalizer"]
