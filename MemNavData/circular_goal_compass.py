"""Cyclic Goal Compass primitives and small C8-equivariant readouts.

The compass consumes a ring of eight *monocular* views acquired at one
physical state.  Index ``j`` is both a camera view and the candidate local
heading represented by that view.  No absolute-direction positional encoding
is permitted.  Consequently, cyclically rotating the scan must rotate the
output by exactly the same amount.

The frozen NavDP ImageGoal encoder supplies one feature per ``(goal, view)``
pair.  This module owns only the small readout and pure audit utilities; it
does not import Habitat or NavDP.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


NUM_DIRECTIONS = 8
DIRECTION_STEP_DEG = 360.0 / NUM_DIRECTIONS
TEACHER_TEMPERATURE_M = 0.25


def wrap_angle_rad(value: float | np.ndarray) -> float | np.ndarray:
    """Wrap finite radians to ``[-pi, pi)``."""
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("angle contains NaN or infinity")
    result = (array + np.pi) % (2.0 * np.pi) - np.pi
    return float(result) if result.ndim == 0 else result


def deterministic_gauge_bin(group_id: str, *, salt: str) -> int:
    """Choose a content-stable cyclic origin without consulting a label."""
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    digest = hashlib.sha256(f"{salt}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % NUM_DIRECTIONS


def scan_yaws(base_yaw_rad: float, gauge_bin: int) -> np.ndarray:
    """Return the eight world yaw values of one active scan."""
    if not 0 <= int(gauge_bin) < NUM_DIRECTIONS:
        raise ValueError("gauge_bin must lie in [0, 7]")
    start = float(base_yaw_rad) + int(gauge_bin) * 2.0 * np.pi / NUM_DIRECTIONS
    offsets = np.arange(NUM_DIRECTIONS, dtype=np.float64)
    return np.asarray(wrap_angle_rad(
        start + offsets * 2.0 * np.pi / NUM_DIRECTIONS), dtype=np.float64)


def native_scan_index(gauge_bin: int) -> int:
    """Index whose world heading equals the original expert/native yaw."""
    if not 0 <= int(gauge_bin) < NUM_DIRECTIONS:
        raise ValueError("gauge_bin must lie in [0, 7]")
    return (-int(gauge_bin)) % NUM_DIRECTIONS


def world_forward_xz(yaw_rad: float) -> np.ndarray:
    """Habitat camera-forward unit vector in the world x-z plane."""
    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    return np.asarray([-math.sin(yaw), -math.cos(yaw)], dtype=np.float64)


def circular_bin_error(predicted: int, target: int,
                       *, bins: int = NUM_DIRECTIONS) -> int:
    """Unsigned shortest distance between two discrete circular bins."""
    predicted = int(predicted)
    target = int(target)
    bins = int(bins)
    if bins <= 1 or not 0 <= predicted < bins or not 0 <= target < bins:
        raise ValueError("invalid circular bin")
    delta = abs(predicted - target)
    return min(delta, bins - delta)


def teacher_distribution(
    advantages_m: Sequence[float],
    valid_mask: Sequence[bool],
    *,
    temperature_m: float = TEACHER_TEMPERATURE_M,
) -> np.ndarray:
    """Masked listwise teacher distribution from metric progress values."""
    advantages = np.asarray(advantages_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if advantages.shape != (NUM_DIRECTIONS,) or valid.shape != advantages.shape:
        raise ValueError("teacher field must contain exactly eight directions")
    if not math.isfinite(float(temperature_m)) or temperature_m <= 0.0:
        raise ValueError("temperature_m must be finite and positive")
    if not valid.any():
        raise ValueError("teacher field has no valid direction")
    if not np.isfinite(advantages[valid]).all():
        raise ValueError("valid teacher advantages must be finite")
    scaled = np.full_like(advantages, -np.inf)
    scaled[valid] = advantages[valid] / float(temperature_m)
    maximum = float(np.max(scaled[valid]))
    weights = np.zeros_like(advantages)
    weights[valid] = np.exp(scaled[valid] - maximum)
    weights /= float(weights.sum())
    return weights


def masked_argmax(values: Sequence[float], valid_mask: Sequence[bool]) -> int:
    values_array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values_array.shape != (NUM_DIRECTIONS,) or valid.shape != values_array.shape:
        raise ValueError("field must contain exactly eight directions")
    if not valid.any() or not np.isfinite(values_array[valid]).all():
        raise ValueError("field has no finite valid direction")
    masked = np.where(valid, values_array, -np.inf)
    return int(np.argmax(masked))


def scene_macro_mean(values: Sequence[float], scenes: Sequence[str]) -> float:
    """Mean of per-scene means; frames never masquerade as independent N."""
    array = np.asarray(values, dtype=np.float64)
    scene_array = np.asarray(scenes, dtype=object)
    if array.ndim != 1 or scene_array.shape != array.shape or not len(array):
        raise ValueError("values and scenes must be aligned non-empty vectors")
    if not np.isfinite(array).all():
        raise ValueError("values contain NaN or infinity")
    unique = sorted({str(scene) for scene in scene_array})
    return float(np.mean([
        array[scene_array == scene].mean() for scene in unique
    ]))


def scene_cluster_bootstrap(
    values: Sequence[float],
    scenes: Sequence[str],
    *,
    seed: int,
    resamples: int = 5000,
) -> Mapping[str, float | int]:
    """Percentile interval by resampling scene clusters with replacement."""
    array = np.asarray(values, dtype=np.float64)
    scene_array = np.asarray(scenes, dtype=object)
    if array.ndim != 1 or scene_array.shape != array.shape or not len(array):
        raise ValueError("values and scenes must be aligned non-empty vectors")
    if not np.isfinite(array).all() or int(resamples) < 100:
        raise ValueError("bootstrap input is invalid")
    unique = np.asarray(sorted({str(scene) for scene in scene_array}), dtype=object)
    if len(unique) < 2:
        raise ValueError("at least two scene clusters are required")
    cluster_means = np.asarray([
        array[scene_array == scene].mean() for scene in unique
    ], dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(unique), size=(int(resamples), len(unique)))
    samples = cluster_means[indices].mean(axis=1)
    lower, median, upper = np.percentile(samples, [2.5, 50.0, 97.5])
    return {
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "scene_clusters": int(len(unique)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def deterministic_scene_folds(
    scenes: Iterable[str], *, folds: int, salt: str,
) -> tuple[tuple[str, ...], ...]:
    """Balanced, content-stable scene folds independent of any label."""
    unique = sorted({str(scene) for scene in scenes})
    folds = int(folds)
    if folds < 2 or len(unique) < folds:
        raise ValueError("not enough scenes for requested folds")
    ordered = sorted(
        unique,
        key=lambda scene: hashlib.sha256(
            f"{salt}:{scene}".encode("utf-8")).hexdigest(),
    )
    result = [[] for _ in range(folds)]
    for index, scene in enumerate(ordered):
        result[index % folds].append(scene)
    return tuple(tuple(group) for group in result)


try:  # Keep geometry/audit helpers importable in the Habitat-only environment.
    import torch
    from torch import nn
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - exercised by Habitat production stage
    torch = None
    nn = None
    torch_functional = None


if nn is not None:
    class CyclicLinearCompass(nn.Module):
        """Shared linear evidence at every view; exactly C8 equivariant."""

        def __init__(self, feature_dim: int = 384) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(
                int(feature_dim), elementwise_affine=False)
            self.readout = nn.Linear(int(feature_dim), 1)

        def forward(self, features):
            if features.ndim != 3 or features.shape[1] != NUM_DIRECTIONS:
                raise ValueError("features must have shape [batch, 8, dim]")
            return self.readout(self.normalization(features)).squeeze(-1)


    class CyclicGoalCompass(nn.Module):
        """Small circular-convolution head with no absolute direction token."""

        def __init__(self, feature_dim: int = 384, hidden_dim: int = 128) -> None:
            super().__init__()
            feature_dim = int(feature_dim)
            hidden_dim = int(hidden_dim)
            if feature_dim < 1 or hidden_dim < 4:
                raise ValueError("invalid compass dimensions")
            self.normalization = nn.LayerNorm(
                feature_dim, elementwise_affine=False)
            self.project = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
            )
            self.circular_context = nn.Conv1d(
                hidden_dim, hidden_dim // 2, kernel_size=3, padding=1,
                padding_mode="circular")
            self.output = nn.Conv1d(
                hidden_dim // 2, 1, kernel_size=3, padding=1,
                padding_mode="circular")

        def forward(self, features):
            if features.ndim != 3 or features.shape[1] != NUM_DIRECTIONS:
                raise ValueError("features must have shape [batch, 8, dim]")
            hidden = self.project(self.normalization(features)).transpose(1, 2)
            hidden = torch_functional.gelu(self.circular_context(hidden))
            return self.output(hidden).squeeze(1)


    def masked_listwise_loss(
        logits,
        advantages_m,
        valid_mask,
        *,
        temperature_m: float = TEACHER_TEMPERATURE_M,
    ):
        """Cross entropy against a metric, same-state soft ranking target."""
        if logits.ndim != 2 or logits.shape[1] != NUM_DIRECTIONS:
            raise ValueError("logits must have shape [batch, 8]")
        if advantages_m.shape != logits.shape or valid_mask.shape != logits.shape:
            raise ValueError("teacher tensors must match logits")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every state must have a valid direction")
        teacher_logits = advantages_m / float(temperature_m)
        teacher_logits = teacher_logits.masked_fill(~valid_mask, -torch.inf)
        target = torch.softmax(teacher_logits, dim=1)
        prediction = logits.masked_fill(~valid_mask, -torch.inf)
        log_prediction = torch.log_softmax(prediction, dim=1)
        terms = torch.where(
            valid_mask, target * log_prediction,
            torch.zeros_like(log_prediction))
        return -terms.sum(dim=1).mean()


__all__ = [
    "DIRECTION_STEP_DEG",
    "NUM_DIRECTIONS",
    "TEACHER_TEMPERATURE_M",
    "circular_bin_error",
    "deterministic_gauge_bin",
    "deterministic_scene_folds",
    "masked_argmax",
    "native_scan_index",
    "scan_yaws",
    "scene_cluster_bootstrap",
    "scene_macro_mean",
    "teacher_distribution",
    "world_forward_xz",
    "wrap_angle_rad",
]

if nn is not None:
    __all__.extend([
        "CyclicGoalCompass",
        "CyclicLinearCompass",
        "masked_listwise_loss",
    ])
