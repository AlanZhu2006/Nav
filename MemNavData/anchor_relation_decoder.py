"""Train-only local goal relation decoder; never imported by a controller.

The visual-only probe and geometry-conditioned model share a small readout.
Neither model estimates the current streaming pose or an activation decision.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def transport_goal(
    anchor_c2w: torch.Tensor,
    goal_in_anchor: torch.Tensor,
    current_c2w: torch.Tensor,
) -> torch.Tensor:
    """Express an anchor-local 3-D goal in the current camera coordinate frame.

    Translations and local offsets must have the same units. This operation
    does not estimate or correct drift or determine a collision-free route.
    """
    world_goal = (
        anchor_c2w[..., :3, :3] @ goal_in_anchor.unsqueeze(-1)
    ).squeeze(-1) + anchor_c2w[..., :3, 3]
    return (
        current_c2w[..., :3, :3].transpose(-1, -2)
        @ (world_goal - current_c2w[..., :3, 3]).unsqueeze(-1)
    ).squeeze(-1)


class RelationBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * width, width),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                memory_valid: torch.Tensor | None = None) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        update, _ = self.attention(
            self.query_norm(query), normalized_memory, normalized_memory,
            key_padding_mask=None if memory_valid is None else ~memory_valid,
            need_weights=False,
        )
        query = query + update
        return query + self.ff(self.ff_norm(query))


class AnchorRelationDecoder(nn.Module):
    """Read local goal position from frozen patch features and optional geometry.

    ``requires_geometry=False`` is an explicitly named input ablation, not a
    substitute for real depth. No role, similarity score, proof statistic,
    current pose, scene identity, or ground-truth coordinate is an input.
    """

    def __init__(
        self, feature_dim: int = 1024, width: int = 128,
        layers: int = 2, heads: int = 4, dropout: float = 0.05,
        output_dim: int = 2, requires_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.requires_geometry = requires_geometry
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width)
        )
        self.pixel_encoding = nn.Linear(2, width, bias=False)
        self.geometry_encoding = (
            nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
            if requires_geometry else None
        )
        self.blocks = nn.ModuleList([
            RelationBlock(width, heads, dropout) for _ in range(layers)
        ])
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * width), nn.Linear(3 * width, width),
            nn.GELU(), nn.Linear(width, output_dim),
        )

    @staticmethod
    def _pixels(tokens: torch.Tensor) -> torch.Tensor:
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("patch tokens must form a square raster")
        coordinates = torch.linspace(-1, 1, side, device=tokens.device,
                                     dtype=tokens.dtype)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, side * side, 2)

    def forward(
        self, goal_patches: torch.Tensor, anchor_patches: torch.Tensor,
        anchor_xyz: torch.Tensor | None = None,
        *, goal_valid: torch.Tensor | None = None,
        anchor_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (goal_patches.ndim != 3
                or goal_patches.shape != anchor_patches.shape):
            raise ValueError("aligned B x patches x channels inputs required")
        if self.requires_geometry != (anchor_xyz is not None):
            raise ValueError("geometry input does not match the declared model")
        for mask in (goal_valid, anchor_valid):
            if mask is not None and (mask.shape != goal_patches.shape[:2]
                                     or mask.dtype != torch.bool
                                     or not mask.any(1).all()):
                raise ValueError("each sample needs a nonempty boolean patch mask")
        position = self.pixel_encoding(self._pixels(goal_patches))
        query = self.projection(goal_patches) + position
        memory = self.projection(anchor_patches) + position
        if anchor_xyz is not None:
            if anchor_xyz.shape != (*anchor_patches.shape[:2], 3):
                raise ValueError("geometry must be aligned with anchor patches")
            valid_xyz = anchor_xyz if anchor_valid is None else anchor_xyz[anchor_valid]
            if not torch.isfinite(valid_xyz).all():
                raise ValueError("geometry must be finite, not a missing-depth fill")
            # Masked padding is not measured geometry. Neutralize it before the
            # MLP, and exclude it from attention AND pooling below.
            xyz = (anchor_xyz if anchor_valid is None else
                   torch.where(anchor_valid[..., None], anchor_xyz, 0.))
            memory = memory + self.geometry_encoding(xyz)
        for block in self.blocks:
            query = block(query, memory, anchor_valid)
        def mean(value, mask):
            if mask is None:
                return value.mean(1)
            return (value * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = (query.amax(1) if goal_valid is None else
                   query.masked_fill(~goal_valid[..., None], -torch.inf).amax(1))
        summary = torch.cat((mean(query, goal_valid), maximum,
                             mean(memory, anchor_valid)), dim=-1)
        return self.readout(summary)


def position_direction_loss(
    prediction: torch.Tensor, target: torch.Tensor,
    angle_weight: float = 0.1, minimum_direction_range: float = 0.25,
) -> torch.Tensor:
    """Planar position supervision plus angle loss away from a zero baseline.

    Units are metres in the cached-only diagnostic. A geometry-conditioned
    version must explicitly normalize BOTH its coordinates and its labels.
    """
    position = F.smooth_l1_loss(prediction, target, beta=0.2)
    valid = torch.linalg.vector_norm(target, dim=-1) >= minimum_direction_range
    direction = 1 - (
        F.normalize(prediction, dim=-1, eps=1e-4)
        * F.normalize(target, dim=-1, eps=1e-4)
    ).sum(-1)
    angle = direction[valid].mean() if valid.any() else prediction.sum() * 0
    return position + angle_weight * angle
