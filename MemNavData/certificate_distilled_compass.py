"""Small structured student for certificate-distilled episodic localization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CDECLoss:
    total: torch.Tensor
    task: torch.Tensor
    certificate_pass: torch.Tensor
    certificate_rank: torch.Tensor


class CertificateDistilledCompass(nn.Module):
    """Permutation-equivariant posterior over memory anchors plus NULL.

    Candidate features are frozen patch-correspondence summaries.  A learned
    NULL token participates in the same set transformer and softmax as all
    anchors, so ranking and abstention cannot drift as unrelated heads.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 heads: int = 4, layers: int = 2, dropout: float = 0.10):
        super().__init__()
        if input_dim < 1 or hidden_dim < 4 or heads < 1 or layers < 1:
            raise ValueError("invalid CDEC dimensions")
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by heads")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=2 * hidden_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.null_token, std=0.02)
        self.task_head = nn.Linear(hidden_dim, 1)
        # These heads exist only to distill privileged certificate structure
        # into the shared representation.  Their outputs are not runtime inputs.
        self.certificate_pass_head = nn.Linear(hidden_dim, 1)
        self.certificate_rank_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have [sessions,candidates,input_dim]")
        encoded = self.encoder(features)
        null = self.null_token.expand(len(features), -1, -1)
        contextual = self.set_encoder(torch.cat([encoded, null], dim=1))
        candidate_count = features.shape[1]
        candidate = contextual[:, :candidate_count]
        return {
            "task_logits": self.task_head(contextual).squeeze(-1),
            "certificate_pass_logits": (
                self.certificate_pass_head(candidate).squeeze(-1)),
            "certificate_rank_logits": (
                self.certificate_rank_head(candidate).squeeze(-1)),
        }


def set_valued_task_loss(
    logits: torch.Tensor,
    positive_candidates: torch.Tensor,
    task_mask: torch.Tensor,
    session_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """NLL where any valid anchor is correct and NULL handles no-match."""
    if logits.ndim != 2 or positive_candidates.shape != (
            logits.shape[0], logits.shape[1] - 1):
        raise ValueError("task logits/positive mask are misaligned")
    if task_mask.shape != (logits.shape[0],):
        raise ValueError("task mask is misaligned")
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[:, :-1] = positive_candidates.bool()
    no_anchor = ~positive_candidates.bool().any(dim=1)
    allowed[:, -1] = no_anchor
    log_numerator = torch.logsumexp(
        logits.masked_fill(~allowed, -torch.inf), dim=1)
    per_session = torch.logsumexp(logits, dim=1) - log_numerator
    selected = task_mask.bool()
    if not selected.any():
        return logits.sum() * 0.0
    if session_weight is None:
        return per_session[selected].mean()
    if session_weight.shape != (logits.shape[0],):
        raise ValueError("session weight is misaligned")
    weight = session_weight[selected]
    return (per_session[selected] * weight).sum() / weight.sum().clamp_min(1e-8)


def cdec_loss(
    outputs: dict[str, torch.Tensor],
    *,
    positive_candidates: torch.Tensor,
    task_mask: torch.Tensor,
    session_weight: torch.Tensor,
    certificate_pass: torch.Tensor,
    teacher_top_index: torch.Tensor,
    pass_positive_weight: torch.Tensor,
    lambda_pass: float,
    lambda_rank: float,
) -> CDECLoss:
    task = set_valued_task_loss(
        outputs["task_logits"], positive_candidates, task_mask,
        session_weight=session_weight)
    pass_loss = F.binary_cross_entropy_with_logits(
        outputs["certificate_pass_logits"], certificate_pass.float(),
        pos_weight=pass_positive_weight)
    rank_loss = F.cross_entropy(
        outputs["certificate_rank_logits"], teacher_top_index.long())
    total = task + float(lambda_pass) * pass_loss + float(lambda_rank) * rank_loss
    return CDECLoss(total, task, pass_loss, rank_loss)


__all__ = [
    "CDECLoss",
    "CertificateDistilledCompass",
    "cdec_loss",
    "set_valued_task_loss",
]
