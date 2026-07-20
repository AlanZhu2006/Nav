"""Inference-safe hierarchical route sketch for MemNav.

The endpoint/revisit tokens answer where the final image goal is.  They do not
explicitly represent which locally feasible direction starts a route around
obstacles.  This module predicts several robot-frame route directions from the
already available current and goal-conditioned memory, then injects them as a
zero-initialized residual into existing current-state slots.  No future pose or
expert action is consumed by the forward path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def route_direction_targets(actions, horizons):
    """Build multi-horizon unit directions from robot-frame action deltas.

    This is deliberately a label-side utility: the policy forward path never
    receives ``actions``.  Keeping the target next to the route representation
    lets training and offline diagnostics share the exact same convention.
    """
    if actions.ndim != 3 or actions.shape[-1] < 2:
        raise ValueError('route targets require actions with shape [B,T,D>=2]')
    horizons = tuple(int(value) for value in horizons)
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError('route target horizons must be non-empty and positive')
    if any(value > actions.shape[1] for value in horizons):
        raise ValueError(
            'route target horizon exceeds the action prediction length'
        )
    displacement = torch.stack(
        [actions[:, :value, :2].sum(dim=1) for value in horizons], dim=1
    )
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    valid = torch.isfinite(radius) & (radius > 1e-6)
    safe = torch.where(
        valid.unsqueeze(-1), displacement, torch.zeros_like(displacement)
    )
    direction = safe / radius.clamp_min(1e-6).unsqueeze(-1)
    return direction, valid


def route_curvature_gate(direction):
    """Continuous turn gate from predicted short/long route separation."""
    if direction.ndim != 3 or direction.shape[1] < 2 or direction.shape[-1] != 2:
        raise ValueError('route curvature needs directions with shape [B,K>=2,2]')
    cosine = (direction[:, 0] * direction[:, -1]).sum(dim=-1).clamp(-1.0, 1.0)
    return (0.5 * (1.0 - cosine)).clamp(0.0, 1.0)


def build_residual_route_sketch(dim, horizons):
    """Construct the optional adapter without advancing the training RNG.

    A route-off/on experiment must see the same diffusion noise after model
    construction.  ``nn.Linear`` initialization otherwise consumes the global
    CPU generator before the first training step, so equal command-line seeds
    do not define a paired optimization run.
    """
    with torch.random.fork_rng(devices=[]):
        return ResidualRouteSketch(dim, horizons)


class ResidualRouteSketch(nn.Module):
    """Predict multi-horizon directions and form residual decoder tokens.

    Keeping the memory length unchanged avoids changing the decoder positional
    embedding or attention mask.  At initialization every residual scale is
    exactly zero, so enabling the module while loading a legacy checkpoint is
    functionally identical to the legacy policy before any update.
    """

    CODE_VERSION = 'residual_route_direction_v2_curvature_gate'

    def __init__(self, dim: int, horizons=(2, 8, 24)):
        super().__init__()
        horizons = tuple(int(value) for value in horizons)
        if dim < 1:
            raise ValueError('route sketch dim must be positive')
        if len(horizons) < 2 or any(value < 1 for value in horizons):
            raise ValueError('route horizons need at least two positive values')
        if len(set(horizons)) != len(horizons):
            raise ValueError('route horizons must be unique')
        if tuple(sorted(horizons)) != horizons:
            raise ValueError('route horizons must be strictly increasing')

        self.dim = int(dim)
        self.horizons = horizons
        self.num_horizons = len(horizons)
        input_dim = 3 * self.dim + 1
        self.input_norm = nn.LayerNorm(input_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
        )
        self.direction_head = nn.Linear(
            self.dim, 2 * self.num_horizons
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.residual_scale = nn.Parameter(
            torch.zeros(self.num_horizons)
        )
        horizon_code = torch.tensor(
            [math.log1p(value) / math.log1p(horizons[-1]) for value in horizons],
            dtype=torch.float32,
        )
        self.register_buffer('horizon_code', horizon_code, persistent=True)

    def forward(self, current_state, revisit, novel, revisit_gate):
        """Return modified current tokens and route diagnostics.

        Args:
            current_state: ``[B,Nc,D]`` current visual/geometric tokens.
            revisit: ``[B,Nr,D]`` endpoint pose tokens.
            novel: ``[B,Nn,D]`` current/goal image tokens.
            revisit_gate: ``[B]`` semantic revisit probability.
        """
        tensors = (current_state, revisit, novel)
        if any(value.ndim != 3 for value in tensors):
            raise ValueError('route token inputs must have shape [B,N,D]')
        batch = current_state.shape[0]
        if any(value.shape[0] != batch for value in tensors):
            raise ValueError('route token inputs must share a batch dimension')
        if any(value.shape[-1] != self.dim for value in tensors):
            raise ValueError('route token input width does not match configured dim')
        if revisit_gate.shape != (batch,):
            raise ValueError('revisit_gate must have shape [B]')
        if current_state.shape[1] < self.num_horizons:
            raise ValueError(
                'current_state needs at least one slot per route horizon'
            )

        # Treat the existing policy representation as the adapter's input, not
        # as an additional optimization target.  In particular, the auxiliary
        # route loss must not perturb the legacy current/revisit/novel encoders
        # while the zero-initialized residual is still closed.  The original
        # ``current_state`` below remains on the normal diffusion gradient path.
        pooled = torch.cat(
            (
                current_state.detach().mean(dim=1),
                revisit.detach().mean(dim=1),
                novel.detach().mean(dim=1),
                revisit_gate.detach().to(current_state).unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = self.trunk(self.input_norm(pooled))
        raw_direction = self.direction_head(hidden).view(
            batch, self.num_horizons, 2
        )
        direction = F.normalize(raw_direction, dim=-1, eps=1e-6)
        curvature_gate = route_curvature_gate(direction)
        horizon = self.horizon_code.to(current_state).view(
            1, self.num_horizons, 1
        ).expand(batch, -1, -1)
        route_tokens = self.token_encoder(
            torch.cat((direction, horizon), dim=-1)
        )
        scale = torch.tanh(self.residual_scale).to(current_state)
        modified = current_state.clone()
        modified[:, :self.num_horizons] = (
            modified[:, :self.num_horizons]
            + curvature_gate.view(batch, 1, 1)
            * scale.view(1, self.num_horizons, 1)
            * route_tokens
        )
        return {
            'current_state': modified,
            'direction': direction,
            'raw_direction_norm': torch.linalg.vector_norm(
                raw_direction, dim=-1
            ),
            'curvature_gate': curvature_gate,
            'residual_scale': scale,
        }
