#!/usr/bin/env python3
"""Train Phase-B LingBot-native localization and pose-uncertainty heads.

This trainer consumes only an artifact that passed
``audit_lingbot_native_localizer_artifact.py``.  It deliberately separates:

* absolute candidate verification plus ranking inside a DINO top-K set;
* true global Novel/no-match, distinct from a shortlist miss; and
* a residual metric-translation mean and diagonal covariance.

Inputs are an explicit allow-list of deployment-time DINO/LingBot quantities.
Ground-truth pose, pose error and co-visibility appear only as targets.  Model
selection and early stopping use a deterministic scene split *within* the 40
training scenes.  The ten development scenes are evaluated only after the
configuration and stopping epoch have been frozen.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from MemNavData.audit_lingbot_native_localizer_artifact import (
        CSV_NAME,
        corrected_teacher_alignment,
        load_json,
        parse_bool,
        parse_xy,
        sha256,
    )
    from MemNavData.train_neural_set_localizer import (
        localization_metrics,
        select_match_threshold,
    )
except ModuleNotFoundError:  # direct script invocation
    from audit_lingbot_native_localizer_artifact import (  # type: ignore
        CSV_NAME,
        corrected_teacher_alignment,
        load_json,
        parse_bool,
        parse_xy,
        sha256,
    )
    from train_neural_set_localizer import (  # type: ignore
        localization_metrics,
        select_match_threshold,
    )


SCALAR_INPUT_COLUMNS = (
    "dino_cosine",
    "metric_scale_m_per_raw",
    "depth_scale_raw",
    "cloud_overlap_f1_center",
    "anchor_goal_distance_norm_center",
    "goal_refine_translation_norm_median",
    "goal_refine_rotation_deg_median",
    "goal_depth_confidence_mean",
    "candidate_depth_confidence_mean",
)
METRIC_SCALE_SOURCES = (
    "cached_ground_anchored",
    "runtime_ground_anchored",
    "pooled_fallback",
)
PREDICTED_XY_COLUMN = "predicted_relative_xy_m_center_json"
TARGET_XY_COLUMN = "target_relative_xy_m_center_json"
FORBIDDEN_INPUT_FRAGMENTS = (
    "target_",
    "relative_position_error",
    "relative_distance_error",
    "relative_rotation_error",
    "relative_position_direction_error",
    "teacher_covis",
    "label",
)


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_torch_save(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_feature_matrix(frame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Construct the sole deployment-input allow-list and pose targets."""
    missing = set(SCALAR_INPUT_COLUMNS) | {PREDICTED_XY_COLUMN, TARGET_XY_COLUMN,
                                           "metric_scale_source"}
    if absent := missing - set(frame.columns):
        raise ValueError(f"exact rows missing training columns: {sorted(absent)}")
    features = []
    names = []
    for column in SCALAR_INPUT_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite deployment input: {column}")
        features.append(values[:, None])
        names.append(column)
    predicted_xy = np.stack([
        parse_xy(value, PREDICTED_XY_COLUMN)
        for value in frame[PREDICTED_XY_COLUMN]
    ])
    target_xy = np.stack([
        parse_xy(value, TARGET_XY_COLUMN)
        for value in frame[TARGET_XY_COLUMN]
    ])
    features.extend([
        predicted_xy[:, :1],
        predicted_xy[:, 1:2],
        np.linalg.norm(predicted_xy, axis=1, keepdims=True),
    ])
    names.extend([
        "lingbot_predicted_forward_m",
        "lingbot_predicted_lateral_m",
        "lingbot_predicted_distance_m",
    ])
    source = frame["metric_scale_source"].astype(str).to_numpy()
    known = np.zeros(len(frame), dtype=bool)
    for category in METRIC_SCALE_SOURCES:
        features.append((source == category).astype(np.float64)[:, None])
        names.append(f"metric_scale_source={category}")
        known |= source == category
    features.append((~known).astype(np.float64)[:, None])
    names.append("metric_scale_source=other")
    for name in names:
        if any(fragment in name for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise RuntimeError(f"forbidden target leaked into input: {name}")
    matrix = np.concatenate(features, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("constructed feature matrix is non-finite")
    return matrix, names, predicted_xy.astype(np.float32), target_xy.astype(np.float32)


@dataclass
class PackedExactSessions:
    features: torch.Tensor
    mask: torch.Tensor
    rank_target: torch.Tensor
    candidate_target: torch.Tensor
    candidate_supervision_mask: torch.Tensor
    no_match_target: torch.Tensor
    no_match_supervision_mask: torch.Tensor
    selected_match_target: torch.Tensor
    predicted_xy: torch.Tensor
    target_xy: torch.Tensor
    pose_mask: torch.Tensor
    covisibility: np.ndarray
    session_ids: tuple[str, ...]
    scenes: tuple[str, ...]

    @property
    def target(self) -> torch.Tensor:
        # Compatibility with localization_metrics/select_match_threshold.
        return torch.cat([
            self.rank_target,
            (1.0 - self.selected_match_target)[:, None],
        ], dim=-1)

    def to(self, device: torch.device) -> "PackedExactSessions":
        return PackedExactSessions(
            self.features.to(device),
            self.mask.to(device),
            self.rank_target.to(device),
            self.candidate_target.to(device),
            self.candidate_supervision_mask.to(device),
            self.no_match_target.to(device),
            self.no_match_supervision_mask.to(device),
            self.selected_match_target.to(device),
            self.predicted_xy.to(device),
            self.target_xy.to(device),
            self.pose_mask.to(device),
            self.covisibility,
            self.session_ids,
            self.scenes,
        )


def pack_exact_sessions(
    features: np.ndarray,
    groups: np.ndarray,
    scenes: np.ndarray,
    covisibility: np.ndarray,
    predicted_xy: np.ndarray,
    target_xy: np.ndarray,
    session_has_positive: np.ndarray,
    session_is_strict_no_match: np.ndarray,
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> PackedExactSessions:
    features = np.asarray(features, dtype=np.float32)
    groups = np.asarray(groups, dtype=str).reshape(-1)
    scenes = np.asarray(scenes, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float32).reshape(-1)
    predicted_xy = np.asarray(predicted_xy, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    session_has_positive = np.asarray(
        session_has_positive, dtype=bool).reshape(-1)
    session_is_strict_no_match = np.asarray(
        session_is_strict_no_match, dtype=bool).reshape(-1)
    if (features.ndim != 2 or not len(features)
            or predicted_xy.shape != (len(features), 2)
            or target_xy.shape != (len(features), 2)
            or not (len(features) == len(groups) == len(scenes)
                    == len(covisibility) == len(session_has_positive)
                    == len(session_is_strict_no_match))):
        raise ValueError("exact session inputs must be non-empty and aligned")
    if not 0.0 <= negative_threshold < positive_threshold <= 1.0:
        raise ValueError("invalid positive/negative thresholds")
    if (not np.isfinite(features).all()
            or not np.isfinite(covisibility).all()
            or not np.isfinite(predicted_xy).all()
            or not np.isfinite(target_xy).all()):
        raise ValueError("exact session inputs must be finite")
    order: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        order.setdefault(str(group), []).append(index)
    indices = [np.asarray(value, dtype=np.int64) for value in order.values()]
    width = max(map(len, indices))
    batch = np.zeros((len(indices), width, features.shape[1]), np.float32)
    mask = np.zeros((len(indices), width), bool)
    rank_target = np.zeros((len(indices), width), np.float32)
    candidate_target = np.zeros((len(indices), width), np.float32)
    candidate_supervision_mask = np.zeros((len(indices), width), bool)
    no_match_target = np.zeros(len(indices), np.float32)
    no_match_supervision_mask = np.zeros(len(indices), bool)
    selected_match_target = np.zeros(len(indices), np.float32)
    predicted = np.zeros((len(indices), width, 2), np.float32)
    target = np.zeros((len(indices), width, 2), np.float32)
    pose_mask = np.zeros((len(indices), width), bool)
    teacher = np.full((len(indices), width), np.nan, np.float32)
    session_scenes = []
    for row, index in enumerate(indices):
        if len(set(scenes[index])) != 1:
            raise ValueError("one exact localization session crosses scenes")
        count = len(index)
        batch[row, :count] = features[index]
        mask[row, :count] = True
        predicted[row, :count] = predicted_xy[index]
        target[row, :count] = target_xy[index]
        teacher[row, :count] = covisibility[index]
        positive = covisibility[index] >= positive_threshold
        negative = covisibility[index] <= negative_threshold
        candidate_target[row, :count] = positive.astype(np.float32)
        candidate_supervision_mask[row, :count] = positive | negative
        pose_mask[row, :count] = positive
        if positive.any():
            weights = covisibility[index][positive]
            rank_target[row, :count][positive] = weights / weights.sum()
            selected_match_target[row] = 1.0
        has_positive_values = np.unique(session_has_positive[index])
        strict_no_match_values = np.unique(
            session_is_strict_no_match[index])
        if len(has_positive_values) != 1 or len(strict_no_match_values) != 1:
            raise ValueError("session-level match flags differ within a session")
        has_positive = bool(has_positive_values[0])
        strict_no_match = bool(strict_no_match_values[0])
        if has_positive and strict_no_match:
            raise ValueError("session cannot be positive and strict no-match")
        if positive.any() and not has_positive:
            raise ValueError("selected positive contradicts session_has_positive")
        # The no-match head means true global Novel, not merely a shortlist
        # miss.  Ambiguous sessions supervise neither class.
        no_match_target[row] = float(strict_no_match)
        no_match_supervision_mask[row] = has_positive or strict_no_match
        session_scenes.append(str(scenes[index[0]]))
    return PackedExactSessions(
        torch.from_numpy(batch),
        torch.from_numpy(mask),
        torch.from_numpy(rank_target),
        torch.from_numpy(candidate_target),
        torch.from_numpy(candidate_supervision_mask),
        torch.from_numpy(no_match_target),
        torch.from_numpy(no_match_supervision_mask),
        torch.from_numpy(selected_match_target),
        torch.from_numpy(predicted),
        torch.from_numpy(target),
        torch.from_numpy(pose_mask),
        teacher,
        tuple(order),
        tuple(session_scenes),
    )


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
        # A new checkpoint is exactly the raw LingBot pose.  Learned geometry
        # is a residual improvement, never an arbitrary replacement at step 0.
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


def model_loss(
    model: nn.Module,
    data: PackedExactSessions,
    index: torch.Tensor,
    *,
    pose_weight: float,
    pose_tail_weight: float = 0.5,
    pose_tail_fraction: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    candidate_logits, no_match_logit, residual, log_variance = model(
        data.features[index], data.mask[index])
    match_session = data.selected_match_target[index] > 0.5
    if match_session.any():
        rank_loss = -(
            data.rank_target[index][match_session]
            * torch.log_softmax(candidate_logits[match_session], dim=-1)
        ).sum(-1).mean()
    else:
        rank_loss = candidate_logits.sum() * 0.0

    def balanced_binary_loss(
            logits: torch.Tensor, target: torch.Tensor,
            supervision_mask: torch.Tensor) -> torch.Tensor:
        logits = logits[supervision_mask]
        target = target[supervision_mask]
        if not logits.numel():
            return logits.sum() * 0.0
        per_example = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none")
        positive = target > 0.5
        class_means = []
        if positive.any():
            class_means.append(per_example[positive].mean())
        if (~positive).any():
            class_means.append(per_example[~positive].mean())
        return torch.stack(class_means).mean()

    candidate_validity_loss = balanced_binary_loss(
        candidate_logits,
        data.candidate_target[index],
        data.candidate_supervision_mask[index],
    )
    no_match_loss = balanced_binary_loss(
        no_match_logit,
        data.no_match_target[index],
        data.no_match_supervision_mask[index],
    )
    set_loss = rank_loss + candidate_validity_loss + no_match_loss
    positive = data.pose_mask[index]
    if positive.any():
        residual_target = (
            data.target_xy[index] - data.predicted_xy[index])
        error = residual[positive] - residual_target[positive]
        pose_nll = 0.5 * (
            torch.exp(-log_variance[positive]) * error.square()
            + log_variance[positive]).sum(-1).mean()
        pose_huber = F.smooth_l1_loss(
            residual[positive], residual_target[positive])
        corrected_error = torch.linalg.vector_norm(
            data.predicted_xy[index][positive] + residual[positive]
            - data.target_xy[index][positive], dim=-1)
        tail_count = max(
            1, int(math.ceil(float(pose_tail_fraction)
                             * corrected_error.numel())))
        pose_cvar = torch.topk(
            corrected_error, k=tail_count, largest=True).values.mean()
        pose_loss = (pose_nll + 0.25 * pose_huber
                     + float(pose_tail_weight) * pose_cvar)
    else:
        pose_nll = residual.sum() * 0.0
        pose_huber = residual.sum() * 0.0
        pose_cvar = residual.sum() * 0.0
        pose_loss = pose_nll
    total = set_loss + float(pose_weight) * pose_loss
    return total, {
        "total_loss": float(total.detach()),
        "set_loss": float(set_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "candidate_validity_loss": float(candidate_validity_loss.detach()),
        "no_match_loss": float(no_match_loss.detach()),
        "pose_nll": float(pose_nll.detach()),
        "pose_huber": float(pose_huber.detach()),
        "pose_cvar": float(pose_cvar.detach()),
    }


@dataclass
class Prediction:
    probability: np.ndarray
    candidate_validity: np.ndarray
    no_match_probability: np.ndarray
    corrected_xy: np.ndarray
    variance_xy: np.ndarray


def predict(model: nn.Module, packed: PackedExactSessions,
            device: torch.device) -> Prediction:
    model.eval()
    with torch.no_grad():
        data = packed.to(device)
        candidate_logits, no_match_logit, residual, log_variance = model(
            data.features, data.mask)
        rank_probability = torch.softmax(candidate_logits, dim=-1)
        candidate_validity = torch.sigmoid(candidate_logits) * data.mask
        no_match_probability = torch.sigmoid(no_match_logit)
        shortlist_validity = candidate_validity.max(dim=-1).values
        usable_match_probability = (
            (1.0 - no_match_probability) * shortlist_validity)
        candidate_probability = (
            usable_match_probability[:, None] * rank_probability)
        set_probability = torch.cat(
            [candidate_probability,
             (1.0 - usable_match_probability)[:, None]], dim=-1)
        return Prediction(
            set_probability.cpu().numpy(),
            candidate_validity.cpu().numpy(),
            no_match_probability.cpu().numpy(),
            (data.predicted_xy + residual).cpu().numpy(),
            torch.exp(log_variance).cpu().numpy(),
        )


def _direction_error(predicted: np.ndarray, target: np.ndarray,
                     minimum_target_distance: float = 0.25) -> np.ndarray:
    predicted_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    valid = ((predicted_norm > 1e-9)
             & (target_norm >= minimum_target_distance))
    result = np.full(len(predicted), np.nan, dtype=np.float64)
    cosine = np.sum(predicted[valid] * target[valid], axis=1) / (
        predicted_norm[valid] * target_norm[valid])
    result[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return result


def pose_metrics(packed: PackedExactSessions,
                 prediction: Prediction) -> dict:
    positive = packed.pose_mask.numpy()
    raw = packed.predicted_xy.numpy()[positive]
    corrected = prediction.corrected_xy[positive]
    target = packed.target_xy.numpy()[positive]
    variance = prediction.variance_xy[positive]
    if not len(target):
        return {"positive_candidates": 0}
    raw_error = np.linalg.norm(raw - target, axis=1)
    corrected_error = np.linalg.norm(corrected - target, axis=1)
    uncertainty = np.sqrt(np.maximum(variance.sum(axis=1), 0.0))
    raw_direction = _direction_error(raw, target)
    corrected_direction = _direction_error(corrected, target)
    try:
        from scipy.stats import spearmanr
        correlation = float(spearmanr(
            uncertainty, corrected_error).statistic)
        if not np.isfinite(correlation):
            correlation = None
    except (ImportError, ValueError):
        correlation = None

    def percentile(values: np.ndarray, quantile: float) -> Optional[float]:
        values = values[np.isfinite(values)]
        return float(np.quantile(values, quantile)) if len(values) else None

    order = np.argsort(uncertainty)
    coverage = {}
    for fraction in (0.50, 0.80):
        count = max(1, int(math.floor(fraction * len(order))))
        coverage[str(fraction)] = {
            "count": count,
            "translation_error_mean_m": float(
                corrected_error[order[:count]].mean()),
        }
    return {
        "positive_candidates": int(len(target)),
        "raw_translation_error_median_m": percentile(raw_error, 0.5),
        "raw_translation_error_p90_m": percentile(raw_error, 0.9),
        "corrected_translation_error_median_m": percentile(
            corrected_error, 0.5),
        "corrected_translation_error_p90_m": percentile(
            corrected_error, 0.9),
        "raw_direction_error_median_deg": percentile(raw_direction, 0.5),
        "raw_direction_error_p90_deg": percentile(raw_direction, 0.9),
        "corrected_direction_error_median_deg": percentile(
            corrected_direction, 0.5),
        "corrected_direction_error_p90_deg": percentile(
            corrected_direction, 0.9),
        "uncertainty_error_spearman": correlation,
        "risk_at_low_uncertainty_coverage": coverage,
    }


def localization_factor_metrics(
        packed: PackedExactSessions, prediction: Prediction,
        *, match_threshold: float) -> dict:
    """Report the three distinct localization decisions without conflation."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    def binary_report(target: np.ndarray, score: np.ndarray) -> dict:
        target = np.asarray(target, dtype=np.int64)
        score = np.asarray(score, dtype=np.float64)
        if not len(target):
            return {"count": 0}
        return {
            "count": int(len(target)),
            "positives": int(target.sum()),
            "accuracy_at_0_5": float(np.mean((score >= 0.5) == target)),
            "roc_auc": (
                float(roc_auc_score(target, score))
                if len(np.unique(target)) == 2 else None),
            "average_precision": (
                float(average_precision_score(target, score))
                if target.any() else None),
            "brier": float(np.mean(np.square(score - target))),
        }

    candidate_mask = packed.candidate_supervision_mask.numpy()
    candidate = binary_report(
        packed.candidate_target.numpy()[candidate_mask],
        prediction.candidate_validity[candidate_mask],
    )
    no_match_mask = packed.no_match_supervision_mask.numpy()
    global_match_target = (
        1.0 - packed.no_match_target.numpy()[no_match_mask])
    global_match = binary_report(
        global_match_target,
        1.0 - prediction.no_match_probability[no_match_mask],
    )
    final_match_score = 1.0 - prediction.probability[:, -1]
    selected_match = packed.selected_match_target.numpy() > 0.5
    known_global_match = (
        no_match_mask & (packed.no_match_target.numpy() < 0.5))
    candidate_miss = known_global_match & ~selected_match
    strict_no_match = (
        no_match_mask & (packed.no_match_target.numpy() > 0.5))
    return {
        "candidate_absolute_verification": candidate,
        "global_memory_match": global_match,
        "ambiguous_sessions_ignored_by_no_match_bce": int(
            np.sum(~no_match_mask)),
        "selected_positive_sessions": int(np.sum(selected_match)),
        "known_match_shortlist_miss_sessions": int(np.sum(candidate_miss)),
        "known_match_shortlist_miss_safe_reject_rate": (
            float(np.mean(
                final_match_score[candidate_miss] < match_threshold))
            if candidate_miss.any() else None),
        "strict_no_match_false_activation_rate": (
            float(np.mean(
                final_match_score[strict_no_match] >= match_threshold))
            if strict_no_match.any() else None),
    }


def apply_pose_gain(
        packed: PackedExactSessions, prediction: Prediction,
        gain: float) -> Prediction:
    if not 0.0 <= gain <= 1.0:
        raise ValueError("pose residual gain must lie in [0, 1]")
    raw = packed.predicted_xy.numpy()
    residual = prediction.corrected_xy - raw
    corrected = raw + float(gain) * residual
    # The variance head describes the full residual distribution.  If the
    # controller only accepts a fraction of its mean, the rejected mean is a
    # remaining bias term and must contribute to predicted risk.
    variance = (prediction.variance_xy
                + np.square((1.0 - float(gain)) * residual))
    return Prediction(
        probability=prediction.probability,
        candidate_validity=prediction.candidate_validity,
        no_match_probability=prediction.no_match_probability,
        corrected_xy=corrected,
        variance_xy=variance,
    )


def select_pose_gain(
        packed: PackedExactSessions,
        prediction: Prediction) -> tuple[float, dict]:
    """Select a conservative residual gain on train-internal validation.

    Translation p90 is primary because the observed failure is a long tail.
    Median, direction p90, and then the smaller gain break ties.  Including
    gain=0 makes the raw LingBot pose an explicit safe baseline.
    """
    candidates = []
    for gain in np.linspace(0.0, 1.0, 11):
        metrics = pose_metrics(packed, apply_pose_gain(
            packed, prediction, float(gain)))
        translation_p90 = metrics.get("corrected_translation_error_p90_m")
        translation_median = metrics.get(
            "corrected_translation_error_median_m")
        direction_p90 = metrics.get("corrected_direction_error_p90_deg")
        key = (
            -float(translation_p90 if translation_p90 is not None else np.inf),
            -float(translation_median if translation_median is not None else np.inf),
            -float(direction_p90 if direction_p90 is not None else np.inf),
            -float(gain),
        )
        candidates.append((key, float(gain), metrics))
    _key, gain, metrics = max(candidates, key=lambda item: item[0])
    return gain, metrics


def combined_metrics(
    packed: PackedExactSessions,
    prediction: Prediction,
    *,
    positive_threshold: float,
    match_threshold: float = 0.5,
) -> dict:
    return {
        "localization": localization_metrics(
            packed, prediction.probability,
            positive_threshold=positive_threshold,
            match_threshold=match_threshold),
        "localization_factors": localization_factor_metrics(
            packed, prediction, match_threshold=match_threshold),
        "pose": pose_metrics(packed, prediction),
    }


def selection_key(metrics: dict) -> tuple[float, float, float, float]:
    localization = metrics["localization"]
    pose = metrics["pose"]
    pose_p90 = pose.get("corrected_translation_error_p90_m")
    return (
        float(localization["joint_localization_accuracy"]),
        float(localization["conditional_candidate_recall_at_1"] or 0.0),
        -float(localization["match_brier"]),
        -float(pose_p90 if pose_p90 is not None else np.inf),
    )


def stratified_scene_split(
        scenes: np.ndarray, covisibility: np.ndarray,
        strict_no_match: np.ndarray, *, positive_threshold: float,
        validation_count: int,
        salt: str = "lingbot-native-phase-b") -> tuple[set[str], set[str]]:
    """Make a deterministic split that keeps both learnable classes on both sides."""
    scenes = np.asarray(scenes, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float64).reshape(-1)
    strict_no_match = np.asarray(strict_no_match, dtype=bool).reshape(-1)
    if not (len(scenes) == len(covisibility) == len(strict_no_match)):
        raise ValueError("scene split inputs are not aligned")
    unique = set(scenes)
    if not 1 <= validation_count < len(unique):
        raise ValueError("invalid scene validation count")
    flags = {
        scene: {
            "selected_positive": bool(np.any(
                covisibility[scenes == scene] >= positive_threshold)),
            "strict_no_match": bool(np.any(strict_no_match[scenes == scene])),
        }
        for scene in unique
    }
    for category in ("selected_positive", "strict_no_match"):
        if sum(value[category] for value in flags.values()) < 2:
            raise RuntimeError(
                f"need at least two scenes containing {category} sessions")
    ranked = sorted(
        unique,
        key=lambda scene: hashlib.sha256(
            f"{salt}:{scene}".encode()).hexdigest(),
    )
    tune: set[str] = set()

    def preserves_core(candidate: str) -> bool:
        after = tune | {candidate}
        return all(any(
            flags[scene][category] for scene in unique - after)
            for category in ("selected_positive", "strict_no_match"))

    for category in ("selected_positive", "strict_no_match"):
        if any(flags[scene][category] for scene in tune):
            continue
        candidate = next((
            scene for scene in ranked
            if scene not in tune and flags[scene][category]
            and preserves_core(scene)
        ), None)
        if candidate is None:
            raise RuntimeError(
                f"cannot place {category} in scene validation split")
        tune.add(candidate)
    for scene in ranked:
        if len(tune) >= validation_count:
            break
        if scene not in tune and preserves_core(scene):
            tune.add(scene)
    if len(tune) != validation_count:
        raise RuntimeError("cannot fill a class-preserving validation split")
    core = unique - tune
    for name, chosen in (("core", core), ("validation", tune)):
        for category in ("selected_positive", "strict_no_match"):
            if not any(flags[scene][category] for scene in chosen):
                raise RuntimeError(
                    f"{name} scene split has no {category} sessions")
    return core, tune


def train_model(
    train: PackedExactSessions,
    validation: Optional[PackedExactSessions],
    *,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    pose_weight: float,
    pose_tail_weight: float,
    pose_tail_fraction: float,
    seed: int,
    device: torch.device,
    positive_threshold: float,
    log_callback=None,
) -> tuple[LingBotNativeLocalizer, int, Optional[dict]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = LingBotNativeLocalizer(
        input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    data = train.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = epochs
    best_metrics = None
    best_key = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(data.session_ids), generator=generator)
        loss_sum = {name: 0.0 for name in (
            "total_loss", "set_loss", "rank_loss", "no_match_loss",
            "candidate_validity_loss", "pose_nll", "pose_huber",
            "pose_cvar")}
        batches = 0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            loss, losses = model_loss(
                model, data, index,
                pose_weight=pose_weight,
                pose_tail_weight=pose_tail_weight,
                pose_tail_fraction=pose_tail_fraction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in losses.items():
                loss_sum[name] += value
            batches += 1
        epoch_log = {
            f"train/{name}": value / max(batches, 1)
            for name, value in loss_sum.items()
        }
        epoch_log["epoch"] = epoch
        if validation is not None and (epoch % 5 == 0 or epoch == epochs):
            prediction = predict(model, validation, device)
            threshold, localization = select_match_threshold(
                validation, prediction.probability,
                positive_threshold=positive_threshold)
            localization_factors = localization_factor_metrics(
                validation, prediction, match_threshold=threshold)
            pose_gain, pose_report = select_pose_gain(
                validation, prediction)
            metrics = {
                "localization": localization,
                "localization_factors": localization_factors,
                "pose": pose_report,
                "match_threshold": threshold,
                "pose_gain": pose_gain,
            }
            key = selection_key(metrics)
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = metrics
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
            epoch_log.update({
                "validation/joint_localization_accuracy": localization[
                    "joint_localization_accuracy"],
                "validation/conditional_candidate_recall_at_1": (
                    localization["conditional_candidate_recall_at_1"] or 0.0),
                "validation/match_brier": localization["match_brier"],
                "validation/candidate_verification_auc": (
                    localization_factors[
                        "candidate_absolute_verification"].get(
                            "roc_auc") or 0.0),
                "validation/global_memory_match_auc": (
                    localization_factors["global_memory_match"].get(
                        "roc_auc") or 0.0),
                "validation/pose_translation_p90_m": (
                    metrics["pose"].get(
                        "corrected_translation_error_p90_m") or 0.0),
            })
        if log_callback is not None:
            log_callback(epoch_log)
        if validation is not None and stale >= patience:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_metrics


class WandbLogger:
    def __init__(self, *, mode: str, project: str, name: str,
                 config: dict, output_dir: Path):
        self.run = None
        if mode == "disabled":
            return
        import wandb
        self.run = wandb.init(
            project=project,
            name=name,
            mode=mode,
            config=config,
            dir=str(output_dir),
        )

    def log(self, values: dict) -> None:
        if self.run is not None:
            self.run.log(values)

    def finish(self, summary: Optional[dict] = None) -> Optional[str]:
        if self.run is None:
            return None
        if summary:
            for key, value in summary.items():
                if value is not None:
                    self.run.summary[key] = value
        url = self.run.url
        self.run.finish()
        return url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run-dir", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--development-csv", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--pose-tail-weight", type=float, default=0.5)
    parser.add_argument("--pose-tail-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"),
        default="disabled")
    parser.add_argument("--wandb-project", default="memnav")
    parser.add_argument("--wandb-name", default="lingbot-native-phase-b")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help=("load the exact artifacts and run one all-head backward pass "
              "without evaluating development metrics or writing a model"))
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if (args.epochs < 1 or args.patience < 1 or args.batch_size < 1
            or args.learning_rate <= 0.0 or args.weight_decay < 0.0
            or args.pose_weight < 0.0 or args.pose_tail_weight < 0.0
            or not 0.0 < args.pose_tail_fraction <= 1.0
            or args.hidden_dim < 1
            or not 0.0 <= args.dropout < 1.0
            or not 0.0 < args.positive_threshold <= 1.0
            or not 0.0 <= args.negative_threshold < args.positive_threshold):
        raise ValueError("invalid Phase-B training configuration")
    required = (
        args.train_run_dir / CSV_NAME,
        args.train_audit,
        args.development_csv,
        args.teacher_csv,
        args.split_manifest,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    audit = load_json(args.train_audit)
    if not audit.get("training_artifact_approved"):
        raise RuntimeError("train artifact was not approved by strict audit")
    train_csv = (args.train_run_dir / CSV_NAME).resolve()
    if audit.get("provenance", {}).get("rows_csv_sha256") != sha256(train_csv):
        raise RuntimeError("train CSV changed after artifact audit")

    started = time.time()
    split = load_json(args.split_manifest)
    train_scenes = set(split.get("train", []))
    development_scenes = set(split.get("development", []))
    if not train_scenes or not development_scenes or train_scenes & development_scenes:
        raise RuntimeError("train/development scene split is invalid")
    train_frame = pd.read_csv(train_csv)
    development_frame = pd.read_csv(args.development_csv)
    teacher = pd.read_csv(args.teacher_csv)
    corrected_teacher_alignment(
        train_frame, teacher,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold)
    corrected_teacher_alignment(
        development_frame, teacher,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold)
    if set(train_frame["scene"].astype(str)) != train_scenes:
        raise RuntimeError("train artifact does not cover the frozen train scenes")
    if set(development_frame["scene"].astype(str)) != development_scenes:
        raise RuntimeError(
            "development artifact does not cover the frozen development scenes")

    train_features, feature_names, train_predicted, train_target = (
        build_feature_matrix(train_frame))
    dev_features, dev_names, dev_predicted, dev_target = (
        build_feature_matrix(development_frame))
    if feature_names != dev_names:
        raise RuntimeError("train/development feature schema differs")
    train_groups = train_frame["session_id"].to_numpy(dtype=str)
    train_scene_rows = train_frame["scene"].to_numpy(dtype=str)
    train_covis = train_frame["teacher_covis"].to_numpy(dtype=np.float64)
    train_has_positive = np.asarray([
        parse_bool(value) for value in train_frame["session_has_positive"]
    ])
    train_strict_no_match = np.asarray([
        parse_bool(value)
        for value in train_frame["session_is_strict_no_match"]
    ])
    dev_groups = development_frame["session_id"].to_numpy(dtype=str)
    dev_scene_rows = development_frame["scene"].to_numpy(dtype=str)
    dev_covis = development_frame["teacher_covis"].to_numpy(dtype=np.float64)
    dev_has_positive = np.asarray([
        parse_bool(value)
        for value in development_frame["session_has_positive"]
    ])
    dev_strict_no_match = np.asarray([
        parse_bool(value)
        for value in development_frame["session_is_strict_no_match"]
    ])

    validation_count = max(2, int(round(0.2 * len(train_scenes))))
    core_scenes, tune_scenes = stratified_scene_split(
        train_scene_rows,
        train_covis,
        train_strict_no_match,
        positive_threshold=args.positive_threshold,
        validation_count=validation_count,
    )

    def scene_mask(rows: np.ndarray, chosen: set[str]) -> np.ndarray:
        return np.asarray([scene in chosen for scene in rows], dtype=bool)

    core_mask = scene_mask(train_scene_rows, core_scenes)
    tune_mask = scene_mask(train_scene_rows, tune_scenes)
    mean = train_features[core_mask].mean(axis=0)
    scale = train_features[core_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0

    def pack_train(mask: np.ndarray, normalized: np.ndarray) -> PackedExactSessions:
        return pack_exact_sessions(
            normalized[mask], train_groups[mask], train_scene_rows[mask],
            train_covis[mask], train_predicted[mask], train_target[mask],
            train_has_positive[mask], train_strict_no_match[mask],
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold)

    normalized_train = (train_features - mean) / scale
    core = pack_train(core_mask, normalized_train)
    tune = pack_train(tune_mask, normalized_train)
    device = torch.device(args.device)
    if args.preflight_only:
        torch.manual_seed(0)
        model = LingBotNativeLocalizer(
            train_features.shape[1], hidden_dim=args.hidden_dim,
            dropout=args.dropout).to(device)
        core_device = core.to(device)
        loss, components = model_loss(
            model, core_device,
            torch.arange(len(core_device.session_ids), device=device),
            pose_weight=args.pose_weight,
            pose_tail_weight=args.pose_tail_weight,
            pose_tail_fraction=args.pose_tail_fraction)
        loss.backward()
        gradient_norms = {}
        for group, module in (
                ("encoder", model.encoder),
                ("rank_head", model.rank_head),
                ("no_match_head", model.no_match_head),
                ("pose_mean_head", model.pose_mean_head),
                ("pose_log_variance_head", model.pose_log_variance_head)):
            gradients = [
                parameter.grad.detach().norm()
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            norm = float(torch.stack(gradients).norm()) if gradients else 0.0
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(f"missing/non-finite gradient for {group}")
            gradient_norms[group] = norm
        print(json.dumps({
            "status": "preflight_passed",
            "train_rows": int(len(train_frame)),
            "train_sessions": int(train_frame["session_id"].nunique()),
            "train_scenes": len(train_scenes),
            "development_rows_schema_checked": int(len(development_frame)),
            "input_dim": train_features.shape[1],
            "loss": components,
            "gradient_norms": gradient_norms,
            "train_artifact_identity_sha256": audit.get(
                "artifact_identity_sha256"),
        }, indent=2, sort_keys=True), flush=True)
        return
    args.out_dir.mkdir(parents=True)
    logger = WandbLogger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        name=args.wandb_name,
        output_dir=args.out_dir,
        config={
            "phase": "lingbot_native_phase_b",
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size_sessions": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pose_weight": args.pose_weight,
            "pose_tail_weight": args.pose_tail_weight,
            "pose_tail_fraction": args.pose_tail_fraction,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "train_artifact_identity": audit.get("artifact_identity_sha256"),
        },
    )
    selector, selected_epoch, selector_metrics = train_model(
        core, tune,
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        pose_weight=args.pose_weight,
        pose_tail_weight=args.pose_tail_weight,
        pose_tail_fraction=args.pose_tail_fraction,
        seed=7,
        device=device,
        positive_threshold=args.positive_threshold,
        log_callback=logger.log,
    )
    del selector
    match_threshold = float(selector_metrics["match_threshold"])
    pose_gain = float(selector_metrics["pose_gain"])

    # Freeze epoch/threshold, then refit three seeds on every training scene.
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_train = (train_features - mean) / scale
    normalized_dev = (dev_features - mean) / scale
    all_train = pack_exact_sessions(
        normalized_train, train_groups, train_scene_rows, train_covis,
        train_predicted, train_target, train_has_positive,
        train_strict_no_match,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold)
    development = pack_exact_sessions(
        normalized_dev, dev_groups, dev_scene_rows, dev_covis,
        dev_predicted, dev_target, dev_has_positive, dev_strict_no_match,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold)
    states = []
    probabilities = []
    candidate_validities = []
    no_match_probabilities = []
    corrected = []
    variances = []
    seed_reports = []
    for seed in (17, 29, 43):
        model, _, _ = train_model(
            all_train, None,
            input_dim=train_features.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=selected_epoch,
            patience=args.patience,
            pose_weight=args.pose_weight,
            pose_tail_weight=args.pose_tail_weight,
            pose_tail_fraction=args.pose_tail_fraction,
            seed=seed,
            device=device,
            positive_threshold=args.positive_threshold,
        )
        prediction = predict(model, development, device)
        probabilities.append(prediction.probability)
        candidate_validities.append(prediction.candidate_validity)
        no_match_probabilities.append(prediction.no_match_probability)
        corrected.append(prediction.corrected_xy)
        variances.append(prediction.variance_xy)
        seed_reports.append(combined_metrics(
            development, apply_pose_gain(development, prediction, pose_gain),
            positive_threshold=args.positive_threshold,
            match_threshold=match_threshold))
        states.append({
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        })
    ensemble_mean = np.mean(corrected, axis=0)
    ensemble_variance = np.mean([
        variance + np.square(mean_value)
        for variance, mean_value in zip(variances, corrected)
    ], axis=0) - np.square(ensemble_mean)
    ensemble = Prediction(
        probability=np.mean(probabilities, axis=0),
        candidate_validity=np.mean(candidate_validities, axis=0),
        no_match_probability=np.mean(no_match_probabilities, axis=0),
        corrected_xy=ensemble_mean,
        variance_xy=np.maximum(ensemble_variance, 1e-8),
    )
    ensemble = apply_pose_gain(development, ensemble, pose_gain)
    development_metrics = combined_metrics(
        development, ensemble,
        positive_threshold=args.positive_threshold,
        match_threshold=match_threshold)
    dino_probability = np.zeros_like(ensemble.probability)
    for row in range(len(development.session_ids)):
        count = int(development.mask[row].sum())
        raw_indices = np.flatnonzero(dev_groups == development.session_ids[row])
        score = dev_features[raw_indices, feature_names.index("dino_cosine")]
        dino_probability[row, :count] = np.exp(score - score.max())
        dino_probability[row, :count] /= dino_probability[row, :count].sum()
    dino_metrics = localization_metrics(
        development, dino_probability,
        positive_threshold=args.positive_threshold,
        match_threshold=0.5)

    checkpoint_path = args.out_dir / "lingbot_native_phase_b.pt"
    artifact = {
        "deployment_approved": False,
        "model_kind": (
            "lingbot_native_verify_rank_true_nomatch_translation_uncertainty"),
        "input_dim": train_features.shape[1],
        "feature_names": feature_names,
        "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "match_threshold": match_threshold,
            "pose_gain": pose_gain,
            "positive_threshold": args.positive_threshold,
        },
        "states": states,
        "train_artifact_identity_sha256": audit.get(
            "artifact_identity_sha256"),
    }
    atomic_torch_save(checkpoint_path, artifact)
    report = {
        "training_complete": True,
        "deployment_approved": False,
        "reason": (
            "Phase-B development evaluation; closed-loop and untouched final "
            "scenes remain required"),
        "objective": (
            "factor candidate verification, candidate rank, and true global "
            "Novel/no-match; learn an optional LingBot metric translation "
            "residual covariance without GT input leakage"),
        "protocol": {
            "train_scenes": sorted(train_scenes),
            "tuning_train_scenes": sorted(core_scenes),
            "tuning_validation_scenes": sorted(tune_scenes),
            "development_scenes": sorted(development_scenes),
            "development_evaluated_after_freeze": True,
            "selected_epoch": selected_epoch,
            "match_threshold": match_threshold,
            "pose_gain": pose_gain,
            "seeds": [17, 29, 43],
            "input_feature_allow_list": feature_names,
            "forbidden_target_fragments": list(FORBIDDEN_INPUT_FRAGMENTS),
        },
        "selector_validation": selector_metrics,
        "development_seed_metrics": seed_reports,
        "development_ensemble": development_metrics,
        "development_dino_candidate_baseline": dino_metrics,
        "inputs": {
            "train_csv": str(train_csv),
            "train_csv_sha256": sha256(train_csv),
            "train_audit": str(args.train_audit.resolve()),
            "train_artifact_identity_sha256": audit.get(
                "artifact_identity_sha256"),
            "development_csv": str(args.development_csv.resolve()),
            "development_csv_sha256": sha256(args.development_csv),
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_csv_sha256": sha256(args.teacher_csv),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256(args.split_manifest),
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "elapsed_seconds": time.time() - started,
    }
    summary = {
        "development/joint_localization_accuracy": development_metrics[
            "localization"]["joint_localization_accuracy"],
        "development/candidate_recall_at_1": development_metrics[
            "localization"]["conditional_candidate_recall_at_1"],
        "development/match_brier": development_metrics[
            "localization"]["match_brier"],
        "development/pose_translation_median_m": development_metrics[
            "pose"].get("corrected_translation_error_median_m"),
        "development/pose_translation_p90_m": development_metrics[
            "pose"].get("corrected_translation_error_p90_m"),
        "development/uncertainty_error_spearman": development_metrics[
            "pose"].get("uncertainty_error_spearman"),
        "development/candidate_verification_auc": development_metrics[
            "localization_factors"]["candidate_absolute_verification"].get(
                "roc_auc"),
        "development/global_memory_match_auc": development_metrics[
            "localization_factors"]["global_memory_match"].get("roc_auc"),
        "development/strict_no_match_false_activation_rate": (
            development_metrics["localization_factors"].get(
                "strict_no_match_false_activation_rate")),
        "development/pose_gain": pose_gain,
    }
    report["wandb_url"] = logger.finish(summary)
    atomic_write_json(args.out_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
