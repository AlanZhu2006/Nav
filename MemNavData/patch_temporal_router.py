"""Dependency-light features for a learned patch/temporal memory router.

The live retrieval path already computes a DINO global descriptor.  A global
cosine is useful for candidate generation, but it cannot tell a genuinely
revisited place from a semantically similar corridor.  This module builds two
small feature families for a *selective* reliability head:

* symmetric correspondence statistics from frozen, spatial DINO patch tokens;
* temporal support around a candidate in the existing retrieval score curve.

Neither feature family uses navigation success, simulator pose, episode phase,
or geometric-teacher labels at inference time.  SIFT/essential geometry remains
the fail-closed teacher/fallback until scene-disjoint evaluation demonstrates a
safe confidence region.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


PATCH_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
TEMPORAL_WINDOWS = (1, 2, 4, 8, 16, 32)


def _as_finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _unit_rows(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name} must have shape [patches, channels]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError(f"{name} contains a zero-norm patch")
    return values / norm


def patch_feature_names() -> Tuple[str, ...]:
    names = ["dino_global_cosine"]
    for statistic in ("mean", "median", "q75", "q90", "q95"):
        names.extend([
            f"best_match_{statistic}_side_mean",
            f"best_match_{statistic}_side_absdiff",
        ])
    names.extend([
        "mutual_match_fraction",
        "mutual_similarity_mean",
        "mutual_similarity_q25",
    ])
    for threshold in PATCH_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        names.extend([
            f"match_fraction_gt_{suffix}_side_mean",
            f"match_fraction_gt_{suffix}_side_absdiff",
        ])
    names.extend([
        "affine_residual_median_direction_mean",
        "affine_residual_median_direction_absdiff",
        "affine_residual_q90_direction_mean",
        "affine_residual_q90_direction_absdiff",
        "mutual_displacement_x_std",
        "mutual_displacement_y_std",
    ])
    return tuple(names)


def temporal_feature_names() -> Tuple[str, ...]:
    names = [
        "temporal_candidate_cosine",
        "temporal_score_minus_session_max",
        "temporal_rank_fraction",
    ]
    for window in TEMPORAL_WINDOWS:
        names.extend([
            f"temporal_w{window}_mean",
            f"temporal_w{window}_max",
            f"temporal_w{window}_std",
            f"temporal_w{window}_near_peak_fraction",
        ])
    names.extend([
        "temporal_far_peak_prominence",
        "temporal_far_context_available",
        "temporal_frame_fraction",
    ])
    return tuple(names)


def _side_statistics(best: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.mean(best),
        np.median(best),
        *np.quantile(best, (0.75, 0.90, 0.95)),
    ], dtype=np.float64)


def _affine_residual(source: np.ndarray,
                     target: np.ndarray) -> Tuple[float, float]:
    """Least-squares spatial-consistency summary without RANSAC or GT pose."""
    if len(source) < 3:
        # Normalized coordinates lie in [-1, 1], so 3 is an explicit invalid
        # sentinel outside the attainable residual range.
        return 3.0, 3.0
    design = np.concatenate(
        [source, np.ones((len(source), 1), dtype=np.float64)], axis=1)
    transform = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = np.linalg.norm(design @ transform - target, axis=1)
    return float(np.median(residual)), float(np.quantile(residual, 0.90))


def symmetric_patch_relation_features(query_tokens: np.ndarray,
                                      memory_tokens: np.ndarray,
                                      global_cosine: float) -> np.ndarray:
    """Summarize frozen DINO patch correspondences for one image pair.

    Tokens must form an equal square grid (for example an adaptively pooled
    8x8 grid).  The output is invariant to swapping query and memory: per-side
    values are represented by their mean and absolute difference, and affine
    residuals are evaluated in both directions.
    """
    query = _unit_rows(query_tokens, "query_tokens")
    memory = _unit_rows(memory_tokens, "memory_tokens")
    if query.shape != memory.shape:
        raise ValueError(
            f"query/memory tokens must have equal shape, got "
            f"{query.shape} and {memory.shape}")
    patch_count = query.shape[0]
    grid_size = int(round(math.sqrt(patch_count)))
    if grid_size * grid_size != patch_count:
        raise ValueError("patch count must form a square spatial grid")
    global_cosine = float(global_cosine)
    if not math.isfinite(global_cosine):
        raise ValueError("global_cosine must be finite")

    similarity = query @ memory.T
    query_best_index = np.argmax(similarity, axis=1)
    memory_best_index = np.argmax(similarity, axis=0)
    query_best = similarity[np.arange(patch_count), query_best_index]
    memory_best = similarity[memory_best_index, np.arange(patch_count)]

    output = [global_cosine]
    query_stats = _side_statistics(query_best)
    memory_stats = _side_statistics(memory_best)
    for query_value, memory_value in zip(query_stats, memory_stats):
        output.extend([
            0.5 * (query_value + memory_value),
            abs(query_value - memory_value),
        ])

    mutual = memory_best_index[query_best_index] == np.arange(patch_count)
    mutual_similarity = query_best[mutual]
    output.extend([
        float(np.mean(mutual)),
        float(np.mean(mutual_similarity)) if mutual_similarity.size else 0.0,
        (float(np.quantile(mutual_similarity, 0.25))
         if mutual_similarity.size else 0.0),
    ])
    for threshold in PATCH_THRESHOLDS:
        query_fraction = float(np.mean(query_best > threshold))
        memory_fraction = float(np.mean(memory_best > threshold))
        output.extend([
            0.5 * (query_fraction + memory_fraction),
            abs(query_fraction - memory_fraction),
        ])

    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.stack([xx, yy], axis=-1).reshape(patch_count, 2)
    query_points = coordinates[mutual]
    memory_points = coordinates[query_best_index[mutual]]
    forward_median, forward_q90 = _affine_residual(
        query_points, memory_points)
    reverse_median, reverse_q90 = _affine_residual(
        memory_points, query_points)
    output.extend([
        0.5 * (forward_median + reverse_median),
        abs(forward_median - reverse_median),
        0.5 * (forward_q90 + reverse_q90),
        abs(forward_q90 - reverse_q90),
    ])
    if mutual_similarity.size:
        displacement = memory_points - query_points
        output.extend([
            float(np.std(displacement[:, 0])),
            float(np.std(displacement[:, 1])),
        ])
    else:
        output.extend([3.0, 3.0])

    result = np.asarray(output, dtype=np.float64)
    expected = len(patch_feature_names())
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise RuntimeError(
            f"invalid patch feature vector {result.shape}, expected {expected}")
    return result


def temporal_score_features(candidate_frame: int,
                            frames: Iterable[int],
                            scores: Iterable[float]) -> np.ndarray:
    """Describe local support in a retrieval score curve around one candidate."""
    frame_values = np.asarray(list(frames), dtype=np.int64).reshape(-1)
    score_values = _as_finite_vector(scores, "scores")
    if frame_values.shape != score_values.shape:
        raise ValueError("frames and scores must be aligned")
    if len(np.unique(frame_values)) != len(frame_values):
        raise ValueError("frames must be unique within one retrieval session")
    candidate_frame = int(candidate_frame)
    selected = np.flatnonzero(frame_values == candidate_frame)
    if selected.size != 1:
        raise ValueError("candidate_frame must occur exactly once")
    score = float(score_values[selected[0]])
    session_max = float(np.max(score_values))
    rank_fraction = (
        float(np.sum(score_values > score)) + 0.5) / len(score_values)
    output = [score, score - session_max, rank_fraction]

    for window in TEMPORAL_WINDOWS:
        local = score_values[np.abs(frame_values - candidate_frame) <= window]
        if not local.size:
            raise RuntimeError("candidate must belong to every local window")
        output.extend([
            float(np.mean(local)),
            float(np.max(local)),
            float(np.std(local)),
            float(np.mean(local >= session_max - 0.02)),
        ])

    far = score_values[np.abs(frame_values - candidate_frame) > 16]
    output.extend([
        session_max - float(np.max(far)) if far.size else 0.0,
        float(bool(far.size)),
        candidate_frame / max(float(np.max(frame_values)), 1.0),
    ])
    result = np.asarray(output, dtype=np.float64)
    expected = len(temporal_feature_names())
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise RuntimeError(
            f"invalid temporal feature vector {result.shape}, expected {expected}")
    return result


def combined_feature_names() -> Tuple[str, ...]:
    # The candidate cosine appears in both families.  Keep the patch copy and
    # drop the first temporal column to avoid an exact duplicate feature.
    return patch_feature_names() + temporal_feature_names()[1:]


def combine_patch_temporal(patch: np.ndarray,
                           temporal: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float64)
    temporal = np.asarray(temporal, dtype=np.float64)
    if patch.shape[:-1] != temporal.shape[:-1]:
        raise ValueError("patch and temporal batch dimensions must match")
    if patch.shape[-1] != len(patch_feature_names()):
        raise ValueError("unexpected patch feature dimension")
    if temporal.shape[-1] != len(temporal_feature_names()):
        raise ValueError("unexpected temporal feature dimension")
    result = np.concatenate([patch, temporal[..., 1:]], axis=-1)
    if result.shape[-1] != len(combined_feature_names()):
        raise RuntimeError("combined feature dimension mismatch")
    return result
