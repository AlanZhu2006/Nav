"""Small K+1 probabilistic baseline for memory localization.

Candidate frames are scored jointly with an explicit no-match (dustbin) state.
Unlike the old scalar gate, the model receives direct supervision for both the
selected memory node and operational abstention.  This linear implementation
is only an objective/label diagnostic; it is not the final dense geometric
matcher and is never exported as deployment-approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _validate(features: np.ndarray, groups: Sequence[str],
              covisibility: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    groups = np.asarray(groups, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or not len(features):
        raise ValueError("features must be a non-empty matrix")
    if not (len(features) == len(groups) == len(covisibility)):
        raise ValueError("features, groups, and co-visibility must align")
    if not np.isfinite(features).all() or not np.isfinite(covisibility).all():
        raise ValueError("probabilistic localizer inputs must be finite")
    if np.any((covisibility < 0.0) | (covisibility > 1.0)):
        raise ValueError("co-visibility must lie in [0, 1]")
    return features, groups, covisibility


def session_indices(groups: Sequence[str]) -> list[np.ndarray]:
    groups = np.asarray(groups, dtype=str).reshape(-1)
    order: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        order.setdefault(str(group), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in order.values()]


@dataclass(frozen=True)
class ProbabilisticSetModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    dustbin_bias: float
    l2: float
    positive_threshold: float
    training_sessions: int
    positive_sessions: int
    converged: bool
    iterations: int
    objective: float

    def predict(self, features: np.ndarray, groups: Sequence[str]) -> tuple[
            np.ndarray, np.ndarray]:
        """Return per-candidate probability and repeated per-row dustbin mass."""
        features = np.asarray(features, dtype=np.float64)
        groups = np.asarray(groups, dtype=str).reshape(-1)
        if (features.ndim != 2 or features.shape[1] != len(self.coefficient)
                or len(features) != len(groups)):
            raise ValueError("features/groups do not match probabilistic model")
        standardized = (features - self.mean) / self.scale
        candidate_probability = np.zeros(len(features), dtype=np.float64)
        dustbin_probability = np.zeros(len(features), dtype=np.float64)
        for index in session_indices(groups):
            logits = np.concatenate([
                standardized[index] @ self.coefficient,
                np.asarray([self.dustbin_bias]),
            ])
            logits -= logits.max()
            probability = np.exp(logits)
            probability /= probability.sum()
            candidate_probability[index] = probability[:-1]
            dustbin_probability[index] = probability[-1]
        return candidate_probability, dustbin_probability

    def portable(self) -> dict:
        return {
            "deployment_approved": False,
            "model_kind": "linear_k_plus_one_objective_diagnostic",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "dustbin_bias": self.dustbin_bias,
            "l2": self.l2,
            "positive_threshold": self.positive_threshold,
            "training_sessions": self.training_sessions,
            "positive_sessions": self.positive_sessions,
            "converged": self.converged,
            "iterations": self.iterations,
            "objective": self.objective,
        }


def fit_probabilistic_set(features: np.ndarray, groups: Sequence[str],
                          covisibility: np.ndarray, *, l2: float = 0.01,
                          positive_threshold: float = 0.5,
                          max_iterations: int = 250) -> ProbabilisticSetModel:
    """Fit a convex listwise softmax with one explicit no-match state."""
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
    sessions = session_indices(groups)
    targets = []
    positive_sessions = 0
    for index in sessions:
        target = np.zeros(len(index) + 1, dtype=np.float64)
        positive = covisibility[index] >= positive_threshold
        if positive.any():
            weights = covisibility[index][positive]
            target[:-1][positive] = weights / weights.sum()
            positive_sessions += 1
        else:
            target[-1] = 1.0
        targets.append(target)

    dimension = features.shape[1]

    def objective(parameter: np.ndarray):
        coefficient, dustbin = parameter[:dimension], parameter[-1]
        loss = 0.0
        gradient_w = np.zeros(dimension, dtype=np.float64)
        gradient_dustbin = 0.0
        for index, target in zip(sessions, targets):
            matrix = standardized[index]
            logits = np.concatenate([
                matrix @ coefficient, np.asarray([dustbin])])
            probability = np.exp(logits - logsumexp(logits))
            loss += float(logsumexp(logits) - target @ logits)
            residual = probability - target
            gradient_w += matrix.T @ residual[:-1]
            gradient_dustbin += float(residual[-1])
        inverse_count = 1.0 / len(sessions)
        loss = (loss * inverse_count
                + 0.5 * l2 * float(coefficient @ coefficient))
        gradient = np.concatenate([
            gradient_w * inverse_count + l2 * coefficient,
            np.asarray([gradient_dustbin * inverse_count]),
        ])
        return loss, gradient

    result = minimize(
        objective, np.zeros(dimension + 1, dtype=np.float64),
        method="L-BFGS-B", jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-10})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("probabilistic optimizer produced non-finite parameters")
    return ProbabilisticSetModel(
        mean=mean,
        scale=scale,
        coefficient=np.asarray(result.x[:dimension], dtype=np.float64),
        dustbin_bias=float(result.x[-1]),
        l2=float(l2),
        positive_threshold=float(positive_threshold),
        training_sessions=len(sessions),
        positive_sessions=positive_sessions,
        converged=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
    )


def scene_group_oof_probabilities(
        features: np.ndarray, groups: Sequence[str], scenes: Sequence[str],
        covisibility: np.ndarray, *, l2: float,
        positive_threshold: float = 0.5, folds: int = 5,
        max_iterations: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Return scene-disjoint out-of-fold candidate and dustbin probabilities."""
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
            raise ValueError("a localization session crosses scene roles")

    candidate = np.full(len(features), np.nan, dtype=np.float64)
    dustbin = np.full(len(features), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=fold_count)
    for train_index, test_index in splitter.split(features, groups=scenes):
        model = fit_probabilistic_set(
            features[train_index], groups[train_index],
            covisibility[train_index], l2=l2,
            positive_threshold=positive_threshold,
            max_iterations=max_iterations)
        candidate[test_index], dustbin[test_index] = model.predict(
            features[test_index], groups[test_index])
    if not np.isfinite(candidate).all() or not np.isfinite(dustbin).all():
        raise RuntimeError("scene OOF probabilities are incomplete")
    return candidate, dustbin


def evaluate_probabilistic_set(
        groups: Sequence[str], covisibility: np.ndarray,
        candidate_probability: np.ndarray, dustbin_probability: np.ndarray,
        *, positive_threshold: float = 0.5) -> dict:
    """Session-level ranking, abstention, calibration, and joint correctness."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    groups = np.asarray(groups, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float64).reshape(-1)
    candidate_probability = np.asarray(
        candidate_probability, dtype=np.float64).reshape(-1)
    dustbin_probability = np.asarray(
        dustbin_probability, dtype=np.float64).reshape(-1)
    if not (groups.shape == covisibility.shape == candidate_probability.shape
            == dustbin_probability.shape):
        raise ValueError("probabilistic evaluation inputs must align")
    if (not np.isfinite(covisibility).all()
            or not np.isfinite(candidate_probability).all()
            or not np.isfinite(dustbin_probability).all()):
        raise ValueError("probabilistic evaluation inputs must be finite")

    match_target = []
    match_score = []
    predicted_match = []
    selected_positive = []
    reciprocal_ranks = []
    joint_correct = []
    selected_overlap = []
    for index in session_indices(groups):
        positive = covisibility[index] >= positive_threshold
        has_match = bool(positive.any())
        no_match = float(dustbin_probability[index[0]])
        if not np.allclose(dustbin_probability[index], no_match):
            raise ValueError("dustbin probability differs inside a session")
        predicts_match = no_match < 0.5
        order = np.argsort(-candidate_probability[index], kind="stable")
        pick = int(order[0])
        pick_positive = bool(positive[pick])
        match_target.append(int(has_match))
        match_score.append(1.0 - no_match)
        predicted_match.append(int(predicts_match))
        selected_positive.append(int(pick_positive))
        selected_overlap.append(float(covisibility[index[pick]]))
        joint_correct.append(int(
            (has_match and predicts_match and pick_positive)
            or (not has_match and not predicts_match)))
        if has_match:
            first = int(np.flatnonzero(positive[order])[0]) + 1
            reciprocal_ranks.append(1.0 / first)

    target = np.asarray(match_target, dtype=np.int64)
    score = np.asarray(match_score, dtype=np.float64)
    prediction = np.asarray(predicted_match, dtype=np.int64)
    selected = np.asarray(selected_positive, dtype=np.int64)
    true_positive = int(np.sum((target == 1) & (prediction == 1)))
    false_positive = int(np.sum((target == 0) & (prediction == 1)))
    false_negative = int(np.sum((target == 1) & (prediction == 0)))
    positive_count = int(target.sum())
    return {
        "sessions": int(len(target)),
        "sessions_with_candidate_positive": positive_count,
        "match_roc_auc": (
            float(roc_auc_score(target, score))
            if len(np.unique(target)) == 2 else None),
        "match_average_precision": (
            float(average_precision_score(target, score))
            if positive_count else None),
        "match_brier": float(np.mean((score - target) ** 2)),
        "match_accuracy_at_0_5": float(np.mean(prediction == target)),
        "match_precision_at_0_5": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else None),
        "match_recall_at_0_5": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else None),
        "conditional_candidate_recall_at_1": (
            float(selected[target == 1].mean()) if positive_count else None),
        "mean_reciprocal_positive_rank": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None),
        "selected_overlap_mean": float(np.mean(selected_overlap)),
        "joint_localization_accuracy": float(np.mean(joint_correct)),
    }
