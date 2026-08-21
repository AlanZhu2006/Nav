"""Minimal deployable network definition for Pi3X spatial reliability."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Pi3XSpatialReliabilityHead(nn.Module):
    """Encode coarse point geometry per view, then reason across causal views."""

    def __init__(self, descriptor_dim: int, *, model_dim: int = 64,
                 layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.spatial_projection = nn.Linear(128, model_dim)
        self.descriptor_projection = nn.Sequential(
            nn.LayerNorm(descriptor_dim), nn.Linear(descriptor_dim, model_dim)
        )
        self.pose_projection = nn.Sequential(
            nn.Linear(12, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.role_embedding = nn.Embedding(5, model_dim)
        self.age_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.action_head = nn.Linear(model_dim, 1)
        self.support_head = nn.Linear(model_dim, 1)

    @staticmethod
    def _signed_log(value: torch.Tensor) -> torch.Tensor:
        return torch.sign(value) * torch.log1p(torch.abs(value))

    def forward(
        self,
        descriptors: torch.Tensor,
        roles: torch.Tensor,
        relative_age: torch.Tensor,
        valid: torch.Tensor,
        world_points: torch.Tensor,
        local_points: torch.Tensor,
        confidence: torch.Tensor,
        poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or world_points.ndim != 5:
            raise ValueError("invalid global/spatial view tensors")
        batch, views, patch_h, patch_w, channels = world_points.shape
        if channels != 3 or local_points.shape != world_points.shape:
            raise ValueError("invalid spatial point channels")
        y = torch.linspace(-1.0, 1.0, patch_h, device=world_points.device)
        x = torch.linspace(-1.0, 1.0, patch_w, device=world_points.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)[None, None].expand(
            batch, views, -1, -1, -1
        )
        spatial = torch.cat([
            self._signed_log(world_points),
            self._signed_log(local_points),
            confidence,
            grid,
        ], dim=-1)
        spatial = spatial.reshape(batch * views, patch_h, patch_w, 9).permute(
            0, 3, 1, 2
        )
        encoded_spatial = self.spatial_encoder(spatial)
        average = F.adaptive_avg_pool2d(encoded_spatial, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(encoded_spatial, 1).flatten(1)
        encoded_spatial = self.spatial_projection(
            torch.cat([average, maximum], dim=1)
        ).reshape(batch, views, self.model_dim)
        encoded_pose = poses.clone()
        encoded_pose[..., :3, 3] = self._signed_log(encoded_pose[..., :3, 3])
        encoded_pose = self.pose_projection(encoded_pose.reshape(batch, views, 12))
        role_ids = (roles + 1).clamp(min=0, max=4)
        encoded = (
            encoded_spatial
            + self.descriptor_projection(descriptors)
            + encoded_pose
            + self.role_embedding(role_ids)
            + self.age_projection(relative_age.unsqueeze(-1))
        )
        cls = self.cls.expand(batch, -1, -1)
        encoded = torch.cat([cls, encoded], dim=1)
        padding = torch.cat([
            torch.zeros((batch, 1), dtype=torch.bool, device=valid.device),
            ~valid,
        ], dim=1)
        pooled = self.output_norm(self.view_encoder(
            encoded, src_key_padding_mask=padding
        )[:, 0])
        return (
            self.action_head(pooled).squeeze(-1),
            self.support_head(pooled).squeeze(-1),
        )
