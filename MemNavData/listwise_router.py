"""Small scene-disjoint listwise ranker for task-aligned memory retrieval.

The ranker is intentionally separate from the Novel/Revisit verifier.  It is
trained only on sessions that contain both a co-visible and a non-co-visible
candidate, and learns which candidate should be checked first.  Sessions with
no usable memory frame must still be rejected by the reliability cascade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ListwiseLinearModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    l2: float
    positive_threshold: float
    training_sessions: int
    converged: bool
    iterations: int
    objective: float

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.coefficient):
            raise ValueError("features do not match listwise model")
        standardized = (features - self.mean) / self.scale
        return standardized @ self.coefficient

    def portable(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "l2": self.l2,
            "positive_threshold": self.positive_threshold,
            "training_sessions": self.training_sessions,
            "converged": self.converged,
            "iterations": self.iterations,
            "objective": self.objective,
        }


def _validate(features: np.ndarray, groups: Sequence[str],
              covisibility: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    groups = np.asarray(groups, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or not len(features):
        raise ValueError("features must be a non-empty matrix")
    if not (len(features) == len(groups) == len(covisibility)):
        raise ValueError("features, groups, and co-visibility must align")
    if (not np.isfinite(features).all()
            or not np.isfinite(covisibility).all()):
        raise ValueError("listwise inputs must be finite")
    if np.any((covisibility < 0.0) | (covisibility > 1.0)):
        raise ValueError("co-visibility must lie in [0, 1]")
    return features, groups, covisibility


def ranking_session_indices(groups: Sequence[str], covisibility: np.ndarray,
                            positive_threshold: float) -> list[np.ndarray]:
    groups = np.asarray(groups, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float64).reshape(-1)
    sessions = []
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        positive = covisibility[index] >= positive_threshold
        if positive.any() and (~positive).any():
            sessions.append(index)
    return sessions


def fit_listwise_linear(features: np.ndarray, groups: Sequence[str],
                        covisibility: np.ndarray, *, l2: float = 0.01,
                        positive_threshold: float = 0.5,
                        max_iterations: int = 250) -> ListwiseLinearModel:
    """Fit a convex ListNet-style linear scorer with per-session targets."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    features, groups, covisibility = _validate(
        features, groups, covisibility)
    if not np.isfinite(l2) or l2 <= 0.0:
        raise ValueError("l2 must be finite and positive")
    if not 0.0 < positive_threshold <= 1.0:
        raise ValueError("positive threshold must lie in (0, 1]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    sessions = ranking_session_indices(
        groups, covisibility, positive_threshold)
    if not sessions:
        raise ValueError(
            "listwise training needs a session with positive and non-positive candidates")

    targets = []
    for index in sessions:
        target = np.zeros(len(index), dtype=np.float64)
        positive = covisibility[index] >= positive_threshold
        weights = covisibility[index][positive]
        target[positive] = weights / weights.sum()
        targets.append(target)

    def objective(coefficient: np.ndarray):
        loss = 0.0
        gradient = np.zeros_like(coefficient)
        for index, target in zip(sessions, targets):
            matrix = standardized[index]
            logits = matrix @ coefficient
            probability = np.exp(logits - logsumexp(logits))
            loss += float(logsumexp(logits) - target @ logits)
            gradient += matrix.T @ (probability - target)
        inverse_count = 1.0 / len(sessions)
        loss = loss * inverse_count + 0.5 * l2 * float(
            coefficient @ coefficient)
        gradient = gradient * inverse_count + l2 * coefficient
        return loss, gradient

    result = minimize(
        objective, np.zeros(features.shape[1], dtype=np.float64),
        method="L-BFGS-B", jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-10})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("listwise optimizer produced non-finite parameters")
    return ListwiseLinearModel(
        mean=mean,
        scale=scale,
        coefficient=np.asarray(result.x, dtype=np.float64),
        l2=float(l2),
        positive_threshold=float(positive_threshold),
        training_sessions=len(sessions),
        converged=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
    )


def scene_group_oof_scores(features: np.ndarray, groups: Sequence[str],
                           scenes: Sequence[str], covisibility: np.ndarray, *,
                           l2: float, positive_threshold: float = 0.5,
                           folds: int = 5,
                           max_iterations: int = 250) -> np.ndarray:
    """Produce scene-disjoint scores for regularization selection."""
    from sklearn.model_selection import GroupKFold

    features, groups, covisibility = _validate(
        features, groups, covisibility)
    scenes = np.asarray(scenes, dtype=str).reshape(-1)
    if len(scenes) != len(features):
        raise ValueError("scenes must align with features")
    unique_scenes = np.unique(scenes)
    fold_count = min(int(folds), len(unique_scenes))
    if fold_count < 2:
        raise ValueError("scene OOF needs at least two scenes")
    for group in np.unique(groups):
        if len(np.unique(scenes[groups == group])) != 1:
            raise ValueError("a retrieval session crosses scene roles")

    scores = np.full(len(features), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=fold_count)
    for train_index, test_index in splitter.split(
            features, groups=scenes):
        model = fit_listwise_linear(
            features[train_index], groups[train_index],
            covisibility[train_index], l2=l2,
            positive_threshold=positive_threshold,
            max_iterations=max_iterations)
        scores[test_index] = model.score(features[test_index])
    if not np.isfinite(scores).all():
        raise RuntimeError("scene OOF listwise scores are incomplete")
    return scores
