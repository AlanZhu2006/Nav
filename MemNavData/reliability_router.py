"""Small, dependency-light pieces for a selective learned memory router.

The learned score is deliberately *not* allowed to replace geometric
verification everywhere.  It can accept or reject only examples outside a
calibrated uncertainty interval; examples inside the interval defer to the
existing SIFT/essential-matrix teacher.

This module contains no scikit-learn dependency so an exported linear head can
be evaluated by the live server without changing its runtime environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Union

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("embeddings must have shape [N, D] with D > 0")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings must have non-zero norm")
    return values / norms


def symmetric_relation_features(goal: np.ndarray,
                                memory: np.ndarray) -> np.ndarray:
    """Return cheap symmetric pair features from frozen DINO CLS tokens.

    For D-dimensional inputs the result has ``2 * D + 1`` columns:
    ``abs(g-m)``, ``g*m`` and cosine similarity.  The representation is
    symmetric because place identity should not depend on query ordering.
    Both embeddings already exist in the live retrieval path, so evaluating a
    linear head over these features does not require another visual backbone.
    """
    goal = _unit_rows(goal)
    memory = _unit_rows(memory)
    if goal.shape != memory.shape:
        raise ValueError(
            f"goal/memory embeddings must have equal shape, got "
            f"{goal.shape} and {memory.shape}")
    product = goal * memory
    cosine = product.sum(axis=1, keepdims=True)
    return np.concatenate([np.abs(goal - memory), product, cosine], axis=1)


@dataclass(frozen=True)
class SelectiveThresholds:
    """Probability interval for a fail-closed cascade.

    ``p <= reject_max`` bypasses memory, ``p >= accept_min`` enables memory,
    and values between the two thresholds defer to geometric verification.
    A disabled side is represented by ``-inf`` or ``+inf`` respectively.
    """

    reject_max: float
    accept_min: float
    reject_calibration_count: int
    accept_calibration_count: int
    min_samples: int

    def __post_init__(self) -> None:
        if math.isnan(self.reject_max) or math.isnan(self.accept_min):
            raise ValueError("selective thresholds cannot be NaN")
        if self.reject_max >= self.accept_min:
            raise ValueError("reject_max must be strictly below accept_min")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")


def calibrate_zero_error_thresholds(labels: ArrayLike,
                                    probabilities: ArrayLike,
                                    min_samples: int = 20
                                    ) -> SelectiveThresholds:
    """Calibrate conservative accept/reject regions on a training split.

    The accepted region lies strictly above every observed negative and the
    rejected region strictly below every observed positive.  A side is
    disabled unless it contains at least ``min_samples`` calibration examples.
    This is an empirical zero-error rule, not a statistical guarantee; callers
    must still validate it on scene-disjoint data.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]

    if positive.size:
        reject_max = float(np.nextafter(positive.min(), -np.inf))
        reject_count = int(np.sum(probabilities <= reject_max))
    else:
        reject_max = 1.0
        reject_count = labels.size
    if reject_count < min_samples:
        reject_max = -math.inf
        reject_count = 0

    if negative.size:
        accept_min = float(np.nextafter(negative.max(), np.inf))
        accept_count = int(np.sum(probabilities >= accept_min))
    else:
        accept_min = 0.0
        accept_count = labels.size
    if accept_count < min_samples:
        accept_min = math.inf
        accept_count = 0

    # Perfectly separated calibration scores leave an empty interval between
    # max(negative) and min(positive).  Split that gap at its midpoint while
    # retaining zero empirical errors.  Overlapping class distributions already
    # produce reject_max < accept_min and therefore a genuine defer interval.
    if (reject_count and accept_count and reject_max >= accept_min
            and positive.size and negative.size):
        midpoint = float((positive.min() + negative.max()) / 2.0)
        reject_max = midpoint
        accept_min = float(np.nextafter(midpoint, np.inf))

    return SelectiveThresholds(
        reject_max=reject_max,
        accept_min=accept_min,
        reject_calibration_count=reject_count,
        accept_calibration_count=accept_count,
        min_samples=int(min_samples),
    )


def selective_decisions(probabilities: ArrayLike,
                        thresholds: SelectiveThresholds) -> np.ndarray:
    """Return -1 (reject), 0 (defer), or +1 (accept) for each score."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    decisions = np.zeros(probabilities.shape, dtype=np.int8)
    decisions[probabilities <= thresholds.reject_max] = -1
    decisions[probabilities >= thresholds.accept_min] = 1
    return decisions


@dataclass
class LinearReliabilityRouter:
    """Portable standardized logistic-regression inference head."""

    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    thresholds: SelectiveThresholds
    feature_version: str = "dino_cls_symmetric_v1"

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        self.coefficient = np.asarray(
            self.coefficient, dtype=np.float64).reshape(-1)
        if not (self.mean.shape == self.scale.shape == self.coefficient.shape):
            raise ValueError("mean, scale, and coefficient shapes must match")
        if self.mean.size == 0 or not np.isfinite(self.mean).all():
            raise ValueError("linear router parameters must be finite and non-empty")
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0.0):
            raise ValueError("feature scales must be finite and positive")
        if not np.isfinite(self.coefficient).all() or not np.isfinite(self.intercept):
            raise ValueError("linear router parameters must be finite")

    @property
    def feature_dim(self) -> int:
        return int(self.mean.size)

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features[None]
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected feature shape [N, {self.feature_dim}], got "
                f"{features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("features must be finite")
        logits = ((features - self.mean) / self.scale) @ self.coefficient
        logits = logits + float(self.intercept)
        # Stable sigmoid without a scipy dependency.
        out = np.empty_like(logits)
        nonnegative = logits >= 0.0
        out[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exp_logits = np.exp(logits[~nonnegative])
        out[~nonnegative] = exp_logits / (1.0 + exp_logits)
        return out

    def predict_proba(self, goal: np.ndarray, memory: np.ndarray) -> np.ndarray:
        return self.predict_proba_from_features(
            symmetric_relation_features(goal, memory))

    def decisions(self, goal: np.ndarray, memory: np.ndarray) -> np.ndarray:
        return selective_decisions(
            self.predict_proba(goal, memory), self.thresholds)

    def to_dict(self) -> dict:
        return {
            "feature_version": self.feature_version,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": float(self.intercept),
            "thresholds": asdict(self.thresholds),
        }

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearReliabilityRouter":
        return cls(
            mean=payload["mean"],
            scale=payload["scale"],
            coefficient=payload["coefficient"],
            intercept=float(payload["intercept"]),
            thresholds=SelectiveThresholds(**payload["thresholds"]),
            feature_version=payload.get(
                "feature_version", "dino_cls_symmetric_v1"),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearReliabilityRouter":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
