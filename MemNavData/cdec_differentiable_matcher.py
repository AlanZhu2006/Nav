"""Differentiable patch correspondence expert for CDEC.

The module consumes frozen LingBot DINO patch tokens, learns only a small
low-rank metric, and summarizes soft bidirectional correspondences with cycle
consistency.  It predicts task match and privileged certificate support as
separate heads; neither head predicts navigation actions.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DifferentiablePatchMatcher(nn.Module):
    def __init__(self, token_dim: int = 1024, projection_dim: int = 48,
                 hidden_dim: int = 64, grid_size: int = 8,
                 dropout: float = 0.10):
        super().__init__()
        if token_dim < 1 or projection_dim < 2 or hidden_dim < 4 or grid_size < 2:
            raise ValueError("invalid patch matcher dimensions")
        self.token_dim = int(token_dim)
        self.projection_dim = int(projection_dim)
        self.grid_size = int(grid_size)
        self.patch_count = grid_size * grid_size
        self.projection = nn.Linear(token_dim, projection_dim, bias=False)
        # A deterministic JL-style initialization approximately preserves the
        # frozen DINO metric before any certificate/task supervision.
        generator = torch.Generator().manual_seed(20260813)
        with torch.no_grad():
            initial = torch.randn(
                projection_dim, token_dim, generator=generator)
            initial /= math.sqrt(projection_dim)
            self.projection.weight.copy_(initial)
        axis = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "coordinates", torch.stack([xx, yy], dim=-1).reshape(-1, 2),
            persistent=True)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.10)))
        self.patch_encoder = nn.Sequential(
            nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU())
        # query mean/max, memory mean/max, and seven global statistics.
        self.relation_encoder = nn.Sequential(
            nn.Linear(4 * 32 + 7, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.task_match_head = nn.Linear(hidden_dim, 1)
        self.certificate_pass_head = nn.Linear(hidden_dim, 1)

    def project_bank(self, token_bank: torch.Tensor) -> torch.Tensor:
        if (token_bank.ndim != 3 or token_bank.shape[1] != self.patch_count
                or token_bank.shape[2] != self.token_dim):
            raise ValueError("token bank shape violates matcher contract")
        return F.normalize(self.projection(token_bank.float()), dim=-1)

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        count = probability.shape[-1]
        return -(probability * probability.clamp_min(1e-8).log()).sum(-1) / math.log(count)

    def relation(self, query: torch.Tensor, memory: torch.Tensor,
                 dino_cosine: torch.Tensor) -> torch.Tensor:
        if query.shape != memory.shape or query.ndim != 3:
            raise ValueError("query/memory projections must be aligned")
        if dino_cosine.shape != (len(query),):
            raise ValueError("DINO cosine is misaligned")
        similarity = torch.einsum("bqd,bmd->bqm", query, memory).float()
        temperature = self.log_temperature.exp().clamp(0.03, 0.50)
        query_to_memory = torch.softmax(similarity / temperature, dim=-1)
        memory_to_query = torch.softmax(similarity / temperature, dim=-2)
        coordinates = self.coordinates.float()
        expected_memory = torch.einsum(
            "bqm,md->bqd", query_to_memory, coordinates)
        expected_query = torch.einsum(
            "bqm,qd->bmd", memory_to_query, coordinates)
        cycle_query = torch.einsum(
            "bqm,bmd->bqd", query_to_memory, expected_query)
        cycle_memory = torch.einsum(
            "bqm,bqd->bmd", memory_to_query, expected_memory)
        coordinate_batch = coordinates.unsqueeze(0).expand(len(query), -1, -1)
        query_feature = torch.cat([
            similarity.max(dim=-1).values.unsqueeze(-1),
            self._entropy(query_to_memory).unsqueeze(-1),
            expected_memory - coordinate_batch,
            cycle_query - coordinate_batch,
            coordinate_batch,
        ], dim=-1)
        memory_probability = memory_to_query.transpose(1, 2)
        memory_feature = torch.cat([
            similarity.max(dim=-2).values.unsqueeze(-1),
            self._entropy(memory_probability).unsqueeze(-1),
            expected_query - coordinate_batch,
            cycle_memory - coordinate_batch,
            coordinate_batch,
        ], dim=-1)
        query_encoded = self.patch_encoder(query_feature)
        memory_encoded = self.patch_encoder(memory_feature)
        query_max = similarity.max(dim=-1).values
        memory_max = similarity.max(dim=-2).values
        global_feature = torch.stack([
            dino_cosine.float(),
            similarity.mean(dim=(1, 2)),
            query_max.mean(dim=1), query_max.std(dim=1),
            memory_max.mean(dim=1), memory_max.std(dim=1),
            0.5 * (self._entropy(query_to_memory).mean(dim=1)
                   + self._entropy(memory_probability).mean(dim=1)),
        ], dim=-1)
        pooled = torch.cat([
            query_encoded.mean(dim=1), query_encoded.max(dim=1).values,
            memory_encoded.mean(dim=1), memory_encoded.max(dim=1).values,
            global_feature,
        ], dim=-1)
        return self.relation_encoder(pooled)

    def forward(self, token_bank: torch.Tensor, query_index: torch.Tensor,
                candidate_index: torch.Tensor,
                dino_cosine: torch.Tensor) -> dict[str, torch.Tensor]:
        if query_index.shape != candidate_index.shape or query_index.ndim != 1:
            raise ValueError("pair indices must be aligned vectors")
        projected = self.project_bank(token_bank)
        relation = self.relation(
            projected[query_index], projected[candidate_index], dino_cosine)
        return {
            "task_match_logits": self.task_match_head(relation).squeeze(-1),
            "certificate_pass_logits": (
                self.certificate_pass_head(relation).squeeze(-1)),
        }


def listwise_positive_loss(logits: torch.Tensor,
                           positive: torch.Tensor) -> torch.Tensor:
    """Set-valued listwise loss over sessions known to contain a positive."""
    if logits.ndim != 2 or positive.shape != logits.shape:
        raise ValueError("listwise logits and labels must align")
    valid = positive.bool().any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    numerator = torch.logsumexp(
        logits.masked_fill(~positive.bool(), -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator[valid] - numerator[valid]).mean()


__all__ = ["DifferentiablePatchMatcher", "listwise_positive_loss"]
