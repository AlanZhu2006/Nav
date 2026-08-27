"""Geometry-token bridge from frozen LingBot-Map to frozen NavDP.

The adapter is deliberately independent of both heavyweight backbones.  Its
input contract is the causal output of LingBot and its output contract is the
``[B, 128, 384]`` observation latent consumed by the official NavDP decoder.
This keeps unit tests and checkpoint receipts small and makes it impossible for
an adapter-only checkpoint to silently contain policy or perception weights.

This module does *not* assert that the bridge is effective.  Effectiveness is a
prospective question governed by
``MONOCULAR_DUAL_TIMESCALE_EXPERT_PROTOCOL_20260818.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryAdapterConfig:
    """Shape-complete deployment contract for :class:`GeometryTokenAdapter`."""

    lingbot_dim: int = 2048
    depth_feature_dim: int = 256
    navdp_dim: int = 384
    navdp_tokens: int = 128
    recent_frames: int = 8
    special_tokens_per_frame: int = 6
    pooled_grid_side: int = 16
    scale_feature_dim: int = 6
    heads: int = 8
    layers: int = 2
    feedforward_multiplier: int = 4
    dropout: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _pool_square_tokens(tokens: Tensor, target_side: int, name: str) -> Tensor:
    """Adaptive-average-pool ``[B,N,D]`` square-grid tokens to ``target_side``."""

    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [B,N,D], got {tuple(tokens.shape)}")
    count = int(tokens.shape[1])
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError(f"{name} token count {count} is not a square grid")
    batch, _, dim = tokens.shape
    image = tokens.transpose(1, 2).reshape(batch, dim, side, side)
    pooled = F.adaptive_avg_pool2d(image, (target_side, target_side))
    return pooled.flatten(2).transpose(1, 2)


class GeometryTokenAdapter(nn.Module):
    """Resample LingBot short-term geometry into NavDP observation tokens.

    Parameters are intentionally limited to modality projections, positional
    embeddings, 128 resampling queries, and a small Transformer decoder.  The
    caller owns LingBot and NavDP and must keep them frozen.

    Inputs
    ------
    window_tokens:
        ``[B,T,6+P,2048]`` post-GCA LingBot tokens for the most recent causal
        frames.  ``T`` may be smaller than eight during bootstrap.
    depth_features:
        ``[B,Pd,256]`` current-frame frozen LingBot DPT features.
    scale_features:
        ``[B,6]`` explicit camera-height/scale/quality fields.  A missing scale
        must be represented by its validity field, never by an oracle constant.
    frame_valid_mask:
        Optional ``[B,T]`` mask.  Invalid history slots are excluded from
        cross-attention; current/depth/scale evidence always remains available.
    """

    TEACHER_QUERY_SUFFIX = "rgbd_encoder.former_query.position_embedding.weight"

    def __init__(self, config: GeometryAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeometryAdapterConfig()
        cfg = self.config
        if cfg.navdp_dim % cfg.heads:
            raise ValueError("navdp_dim must be divisible by heads")
        if cfg.recent_frames < 1 or cfg.special_tokens_per_frame < 1:
            raise ValueError("recent frame and special-token counts must be positive")
        if cfg.pooled_grid_side < 1 or cfg.navdp_tokens < 1:
            raise ValueError("pooled grid and output token counts must be positive")

        self.lingbot_projection = nn.Linear(cfg.lingbot_dim, cfg.navdp_dim)
        self.depth_projection = nn.Linear(cfg.depth_feature_dim, cfg.navdp_dim)
        self.scale_projection = nn.Sequential(
            nn.Linear(cfg.scale_feature_dim, cfg.navdp_dim),
            nn.GELU(),
            nn.Linear(cfg.navdp_dim, cfg.navdp_dim),
        )

        self.modality_embedding = nn.Embedding(4, cfg.navdp_dim)
        self.frame_embedding = nn.Embedding(cfg.recent_frames, cfg.navdp_dim)
        self.special_embedding = nn.Embedding(
            cfg.special_tokens_per_frame, cfg.navdp_dim
        )
        spatial_tokens = cfg.pooled_grid_side * cfg.pooled_grid_side
        self.current_spatial_embedding = nn.Embedding(spatial_tokens, cfg.navdp_dim)
        self.depth_spatial_embedding = nn.Embedding(spatial_tokens, cfg.navdp_dim)

        self.source_norm = nn.LayerNorm(cfg.navdp_dim)
        self.queries = nn.Parameter(torch.zeros(cfg.navdp_tokens, cfg.navdp_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.navdp_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.feedforward_multiplier * cfg.navdp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.resampler = nn.TransformerDecoder(layer, num_layers=cfg.layers)
        self.output_norm = nn.LayerNorm(cfg.navdp_dim)

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.config.navdp_tokens, self.config.navdp_dim

    @torch.no_grad()
    def initialize_queries_from_navdp(
        self, state_dict: Mapping[str, Tensor]
    ) -> str:
        """Copy the official NavDP former-query slots and return the matched key.

        Prefixes such as ``module.`` are accepted, but ambiguous matches fail.
        No other teacher tensor is copied into the adapter.
        """

        matches = [
            key for key in state_dict if key.endswith(self.TEACHER_QUERY_SUFFIX)
        ]
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one NavDP former-query tensor, found "
                f"{matches}"
            )
        value = state_dict[matches[0]]
        if tuple(value.shape) != tuple(self.queries.shape):
            raise ValueError(
                f"teacher query shape {tuple(value.shape)} != adapter shape "
                f"{tuple(self.queries.shape)}"
            )
        self.queries.copy_(value.to(device=self.queries.device, dtype=self.queries.dtype))
        return matches[0]

    def _source_from_compact(
        self,
        recent_specials: Tensor,
        current_patches: Tensor,
        depth_features: Tensor,
        scale_features: Tensor,
        frame_valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if recent_specials.ndim != 4:
            raise ValueError(
                "recent_specials must have shape [B,T,S,D], got "
                f"{tuple(recent_specials.shape)}"
            )
        batch, frames, special_count, lingbot_dim = recent_specials.shape
        if not 1 <= frames <= cfg.recent_frames:
            raise ValueError(
                f"window contains {frames} frames; expected 1..{cfg.recent_frames}"
            )
        if special_count != cfg.special_tokens_per_frame:
            raise ValueError(
                f"special token count {special_count} != configured "
                f"{cfg.special_tokens_per_frame}"
            )
        if lingbot_dim != cfg.lingbot_dim:
            raise ValueError(
                f"LingBot dim {lingbot_dim} != configured {cfg.lingbot_dim}"
            )
        if current_patches.ndim != 3 or current_patches.shape[0] != batch:
            raise ValueError("current_patches must have shape [B,N,lingbot_dim]")
        if current_patches.shape[-1] != cfg.lingbot_dim:
            raise ValueError(
                f"current patch dim {current_patches.shape[-1]} != "
                f"{cfg.lingbot_dim}"
            )
        if depth_features.ndim != 3 or tuple(depth_features.shape[:1]) != (batch,):
            raise ValueError("depth_features must have shape [B,N,depth_dim]")
        if depth_features.shape[-1] != cfg.depth_feature_dim:
            raise ValueError(
                f"depth feature dim {depth_features.shape[-1]} != "
                f"{cfg.depth_feature_dim}"
            )
        if tuple(scale_features.shape) != (batch, cfg.scale_feature_dim):
            raise ValueError(
                f"scale_features must have shape {(batch, cfg.scale_feature_dim)}, "
                f"got {tuple(scale_features.shape)}"
            )
        _require_finite("recent_specials", recent_specials)
        _require_finite("current_patches", current_patches)
        _require_finite("depth_features", depth_features)
        _require_finite("scale_features", scale_features)

        if frame_valid_mask is None:
            frame_valid_mask = torch.ones(
                batch, frames, dtype=torch.bool, device=recent_specials.device
            )
        elif tuple(frame_valid_mask.shape) != (batch, frames):
            raise ValueError(
                f"frame_valid_mask must have shape {(batch, frames)}, got "
                f"{tuple(frame_valid_mask.shape)}"
            )
        else:
            frame_valid_mask = frame_valid_mask.to(
                device=recent_specials.device, dtype=torch.bool
            )
        if not frame_valid_mask[:, -1].all():
            raise ValueError("the current (last) frame must always be valid")

        special = self.lingbot_projection(recent_specials)
        # Short prefixes are right-aligned so the current frame always occupies
        # the same temporal slot as a mature eight-frame FIFO.
        frame_positions = torch.arange(
            cfg.recent_frames - frames,
            cfg.recent_frames,
            device=recent_specials.device,
        )
        special_positions = torch.arange(
            cfg.special_tokens_per_frame, device=recent_specials.device
        )
        special = (
            special
            + self.frame_embedding(frame_positions)[None, :, None]
            + self.special_embedding(special_positions)[None, None, :]
            + self.modality_embedding.weight[0][None, None, None]
        )
        special = special.reshape(
            batch, frames * cfg.special_tokens_per_frame, cfg.navdp_dim
        )

        current_patches = _pool_square_tokens(
            current_patches, cfg.pooled_grid_side, "current LingBot patches"
        )
        current_patches = self.lingbot_projection(current_patches)
        current_patches = (
            current_patches
            + self.current_spatial_embedding.weight[None]
            + self.modality_embedding.weight[1][None, None]
        )

        depth = _pool_square_tokens(
            depth_features, cfg.pooled_grid_side, "LingBot depth features"
        )
        depth = self.depth_projection(depth)
        depth = (
            depth
            + self.depth_spatial_embedding.weight[None]
            + self.modality_embedding.weight[2][None, None]
        )

        scale = self.scale_projection(scale_features).unsqueeze(1)
        scale = scale + self.modality_embedding.weight[3][None, None]

        source = self.source_norm(torch.cat([special, current_patches, depth, scale], dim=1))
        special_padding = (~frame_valid_mask).unsqueeze(-1).expand(
            -1, -1, cfg.special_tokens_per_frame
        ).reshape(batch, -1)
        always_valid = torch.zeros(
            batch,
            current_patches.shape[1] + depth.shape[1] + 1,
            dtype=torch.bool,
            device=recent_specials.device,
        )
        padding = torch.cat([special_padding, always_valid], dim=1)
        return source, padding

    def _source_from_pooled_compact(
        self,
        recent_specials: Tensor,
        pooled_current_patches: Tensor,
        pooled_depth_features: Tensor,
        scale_features: Tensor,
        frame_valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Build a source sequence from deterministically pooled disk shards.

        Spatial pooling has no learned parameters.  Caching its 16x16 output
        therefore reduces PT1 storage by roughly five times while leaving all
        learned projections and the resampler untouched.
        """

        cfg = self.config
        expected = cfg.pooled_grid_side * cfg.pooled_grid_side
        if pooled_current_patches.ndim != 3 or tuple(
            pooled_current_patches.shape[1:]
        ) != (expected, cfg.lingbot_dim):
            raise ValueError(
                "pooled_current_patches must have shape "
                f"[B,{expected},{cfg.lingbot_dim}], got "
                f"{tuple(pooled_current_patches.shape)}"
            )
        if pooled_depth_features.ndim != 3 or tuple(
            pooled_depth_features.shape[1:]
        ) != (expected, cfg.depth_feature_dim):
            raise ValueError(
                "pooled_depth_features must have shape "
                f"[B,{expected},{cfg.depth_feature_dim}], got "
                f"{tuple(pooled_depth_features.shape)}"
            )
        # When the input grid already has the target side length, the compact
        # path's adaptive average pooling is an exact identity.
        return self._source_from_compact(
            recent_specials,
            pooled_current_patches,
            pooled_depth_features,
            scale_features,
            frame_valid_mask,
        )

    def _source_tokens(
        self,
        window_tokens: Tensor,
        depth_features: Tensor,
        scale_features: Tensor,
        frame_valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if window_tokens.ndim != 4:
            raise ValueError(
                "window_tokens must have shape [B,T,N,D], got "
                f"{tuple(window_tokens.shape)}"
            )
        if window_tokens.shape[2] <= cfg.special_tokens_per_frame:
            raise ValueError("window has no spatial patch tokens")
        recent_specials = window_tokens[:, :, : cfg.special_tokens_per_frame]
        current_patches = window_tokens[:, -1, cfg.special_tokens_per_frame :]
        return self._source_from_compact(
            recent_specials,
            current_patches,
            depth_features,
            scale_features,
            frame_valid_mask,
        )

    def forward(
        self,
        window_tokens: Tensor,
        depth_features: Tensor,
        scale_features: Tensor,
        frame_valid_mask: Tensor | None = None,
    ) -> Tensor:
        source, padding = self._source_tokens(
            window_tokens, depth_features, scale_features, frame_valid_mask
        )
        queries = self.queries.unsqueeze(0).expand(window_tokens.shape[0], -1, -1)
        output = self.resampler(
            tgt=queries, memory=source, memory_key_padding_mask=padding
        )
        output = self.output_norm(output)
        _require_finite("adapter output", output)
        return output

    def forward_compact(
        self,
        recent_specials: Tensor,
        current_patches: Tensor,
        depth_features: Tensor,
        scale_features: Tensor,
        frame_valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Memory-efficient equivalent of :meth:`forward` for cached training.

        Only six special tokens are used from older recent frames; spatial
        patches are consumed from the current frame.  Persisting this compact
        representation is about eight times smaller than writing all eight
        dense LingBot frames and is exactly decision-equivalent to ``forward``.
        """

        source, padding = self._source_from_compact(
            recent_specials,
            current_patches,
            depth_features,
            scale_features,
            frame_valid_mask,
        )
        queries = self.queries.unsqueeze(0).expand(recent_specials.shape[0], -1, -1)
        output = self.resampler(
            tgt=queries, memory=source, memory_key_padding_mask=padding
        )
        output = self.output_norm(output)
        _require_finite("adapter output", output)
        return output

    def forward_pooled_compact(
        self,
        recent_specials: Tensor,
        pooled_current_patches: Tensor,
        pooled_depth_features: Tensor,
        scale_features: Tensor,
        frame_valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward an offline-pooled source shard into NavDP latent space."""

        source, padding = self._source_from_pooled_compact(
            recent_specials,
            pooled_current_patches,
            pooled_depth_features,
            scale_features,
            frame_valid_mask,
        )
        queries = self.queries.unsqueeze(0).expand(recent_specials.shape[0], -1, -1)
        output = self.resampler(
            tgt=queries, memory=source, memory_key_padding_mask=padding
        )
        output = self.output_norm(output)
        _require_finite("adapter output", output)
        return output


@dataclass(frozen=True)
class DistillationWeights:
    token: float = 1.0
    denoise: float = 1.0
    critic: float = 0.25
    rank: float = 0.25


def _pairwise_rank_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Teacher-weighted pairwise logistic loss for critic candidate rankings."""

    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("critic scores must share shape [B,K]")
    teacher_delta = teacher[:, :, None] - teacher[:, None, :]
    student_delta = student[:, :, None] - student[:, None, :]
    upper = torch.triu(
        torch.ones(
            teacher.shape[1], teacher.shape[1],
            dtype=torch.bool, device=teacher.device
        ),
        diagonal=1,
    )
    if not upper.any():
        return student.sum() * 0.0
    target = (teacher_delta > 0).to(student)
    confidence = teacher_delta.abs().detach()
    loss = F.binary_cross_entropy_with_logits(
        student_delta, target, reduction="none"
    )
    weighted = loss * confidence
    denominator = confidence[:, upper].sum().clamp_min(1e-8)
    return weighted[:, upper].sum() / denominator


def geometry_distillation_losses(
    student_tokens: Tensor,
    teacher_tokens: Tensor,
    *,
    student_epsilon: Tensor | None = None,
    teacher_epsilon: Tensor | None = None,
    student_critic: Tensor | None = None,
    teacher_critic: Tensor | None = None,
    weights: DistillationWeights | None = None,
) -> dict[str, Tensor]:
    """Compute the frozen protocol's token and optional functional losses."""

    weights = weights or DistillationWeights()
    if student_tokens.shape != teacher_tokens.shape or student_tokens.ndim != 3:
        raise ValueError("student/teacher tokens must share shape [B,N,D]")
    teacher_tokens = teacher_tokens.detach()
    student_norm = F.layer_norm(student_tokens, (student_tokens.shape[-1],))
    teacher_norm = F.layer_norm(teacher_tokens, (teacher_tokens.shape[-1],))
    token_smooth_l1 = F.smooth_l1_loss(student_norm, teacher_norm)
    token_cosine = 1.0 - F.cosine_similarity(
        student_norm, teacher_norm, dim=-1
    ).mean()
    token = token_smooth_l1 + token_cosine

    zero = student_tokens.sum() * 0.0
    denoise = zero
    if (student_epsilon is None) != (teacher_epsilon is None):
        raise ValueError("student and teacher epsilon must be supplied together")
    if student_epsilon is not None:
        if student_epsilon.shape != teacher_epsilon.shape:
            raise ValueError("student/teacher epsilon shapes differ")
        denoise = F.smooth_l1_loss(student_epsilon, teacher_epsilon.detach())

    critic = zero
    rank = zero
    if (student_critic is None) != (teacher_critic is None):
        raise ValueError("student and teacher critic scores must be supplied together")
    if student_critic is not None:
        if student_critic.shape != teacher_critic.shape:
            raise ValueError("student/teacher critic shapes differ")
        critic = F.smooth_l1_loss(student_critic, teacher_critic.detach())
        rank = _pairwise_rank_loss(student_critic, teacher_critic.detach())

    total = (
        weights.token * token
        + weights.denoise * denoise
        + weights.critic * critic
        + weights.rank * rank
    )
    return {
        "loss": total,
        "token": token,
        "token_smooth_l1": token_smooth_l1,
        "token_cosine": token_cosine,
        "denoise": denoise,
        "critic": critic,
        "rank": rank,
    }


def adapter_parameter_receipt(adapter: GeometryTokenAdapter) -> dict[str, object]:
    """Small serializable audit used before constructing an optimizer."""

    trainable = {
        name: int(parameter.numel())
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    return {
        "architecture": "geometry_token_adapter_v1",
        "config": adapter.config.to_dict(),
        "trainable_parameter_count": int(sum(trainable.values())),
        "trainable_tensors": trainable,
        "output_shape_without_batch": list(adapter.output_shape),
    }
