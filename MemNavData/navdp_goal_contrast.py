"""Pure helpers for the NavDP goal-contrast direction diagnostic.

The score consumed here is a paired denoising-error contrast emitted by the
frozen NavDP server.  It is deliberately named an advantage rather than a
likelihood: larger values mean the ImageGoal-conditioned denoiser reconstructed
the injected noise better than the zero-goal denoiser on the same trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from MemNavData.novel_a_bearing_gate import wrap_deg


def poisson_binomial_upper_tail(successes: int,
                                probabilities: Sequence[float]) -> float:
    """Exact P(X >= successes) for independent unequal Bernoulli nulls."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or np.any(~np.isfinite(probs)):
        raise ValueError("probabilities must be a finite vector")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not 0 <= successes <= len(probs):
        raise ValueError("success count is out of range")
    distribution = np.asarray([1.0], dtype=np.float64)
    for probability in probs:
        updated = np.zeros(len(distribution) + 1, dtype=np.float64)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return float(distribution[successes:].sum())


def trajectory_heading_deg(trajectory: Any) -> float | None:
    """Return NavDP's endpoint heading using the existing 0.3 m convention."""
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"unexpected trajectory shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("trajectory contains non-finite values")
    planar = value[:, :2]
    distances = np.linalg.norm(planar, axis=1)
    eligible = np.flatnonzero(distances >= 0.3)
    endpoint = planar[eligible[-1]] if eligible.size else planar[-1]
    if float(np.linalg.norm(endpoint)) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(endpoint[1], endpoint[0])))


def _single_batch_vector(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite batch of {count} scores")
    return array


def goal_contrast_diagnostics(response: dict,
                              requested_heading_deg: float | None = None
                              ) -> dict[str, Any]:
    """Select one trajectory by goal contrast and summarize its controls."""
    candidates = np.asarray(response.get("all_trajectory"), dtype=np.float64)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if (candidates.ndim != 3 or candidates.shape[-1] != 3
            or not np.isfinite(candidates).all()):
        raise ValueError("all_trajectory is not a finite single-batch tensor")
    count = int(len(candidates))
    payload = response.get("goal_contrast")
    if not isinstance(payload, dict):
        raise ValueError("response omitted goal_contrast")
    if payload.get("score_semantics") != "nogoal_mse_minus_goal_mse":
        raise ValueError("unexpected goal-contrast score semantics")
    if payload.get("is_calibrated_likelihood") is not False:
        raise ValueError("goal contrast must not claim calibrated likelihood")

    scores = _single_batch_vector(
        payload.get("goal_advantage"), count, "goal_advantage")
    normalized_scores = _single_batch_vector(
        payload.get("normalized_goal_advantage"), count,
        "normalized_goal_advantage")
    order = np.argsort(-scores, kind="stable")
    chosen_index = int(order[0])
    chosen_heading = trajectory_heading_deg(candidates[chosen_index])
    headings = [trajectory_heading_deg(candidate) for candidate in candidates]
    result: dict[str, Any] = {
        "candidate_count": count,
        "goal_candidate_index": chosen_index,
        "goal_selected_heading_deg": chosen_heading,
        "goal_score": float(scores[chosen_index]),
        "goal_score_margin": float(scores[order[0]] - scores[order[1]])
        if count > 1 else math.inf,
        "goal_score_std": float(np.std(scores)),
        "normalized_goal_score": float(normalized_scores[chosen_index]),
        "goal_selected_request_error_deg": None,
        "control_candidate_index": None,
        "control_selected_heading_deg": None,
        "control_score": None,
        "control_score_margin": None,
        "control_selected_request_error_deg": None,
        "goal_vs_control_at_goal_choice": None,
    }
    if requested_heading_deg is not None and chosen_heading is not None:
        result["goal_selected_request_error_deg"] = abs(wrap_deg(
            chosen_heading - float(requested_heading_deg)))

    if "control_goal_advantage" in payload:
        control_scores = _single_batch_vector(
            payload["control_goal_advantage"], count,
            "control_goal_advantage")
        control_order = np.argsort(-control_scores, kind="stable")
        control_index = int(control_order[0])
        control_heading = headings[control_index]
        result.update({
            "control_candidate_index": control_index,
            "control_selected_heading_deg": control_heading,
            "control_score": float(control_scores[control_index]),
            "control_score_margin": float(
                control_scores[control_order[0]]
                - control_scores[control_order[1]])
            if count > 1 else math.inf,
            "goal_vs_control_at_goal_choice": float(
                scores[chosen_index] - control_scores[chosen_index]),
        })
        if requested_heading_deg is not None and control_heading is not None:
            result["control_selected_request_error_deg"] = abs(wrap_deg(
                control_heading - float(requested_heading_deg)))
    return result


def exact_mcnemar_p(gains: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired outcomes."""
    gains = int(gains)
    losses = int(losses)
    if gains < 0 or losses < 0:
        raise ValueError("paired counts must be non-negative")
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    lower = min(gains, losses)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * tail / (2.0 ** discordant))


def summarize_goal_contrast(rows: list[dict], threshold_deg: float) -> dict:
    """Summarize correct-goal selection against shuffled-goal and chance."""
    if not rows:
        return {"states": 0}

    def hit(row: dict, key: str) -> bool:
        value = row.get(key)
        return value is not None and float(value) <= threshold_deg

    goal_request = [hit(row, "goal_request_error_deg") for row in rows]
    goal_executed = [hit(row, "goal_executed_error_deg") for row in rows]
    control_request = [hit(row, "control_request_error_deg") for row in rows]
    control_executed = [hit(row, "control_executed_error_deg") for row in rows]
    request_gains = sum(goal and not control for goal, control in zip(
        goal_request, control_request))
    request_losses = sum(control and not goal for goal, control in zip(
        goal_request, control_request))
    executed_gains = sum(goal and not control for goal, control in zip(
        goal_executed, control_executed))
    executed_losses = sum(control and not goal for goal, control in zip(
        goal_executed, control_executed))
    probabilities = [float(row["random_request_hit_probability"])
                     for row in rows]
    specificity = [float(row["goal_vs_control_at_goal_choice"])
                   for row in rows]
    request_pairs = [(
        float(row["goal_request_error_deg"]),
        float(row["control_request_error_deg"]),
    ) for row in rows]
    goal_ranks = [int(row["goal_oracle_request_rank"]) for row in rows]
    control_ranks = [int(row["control_oracle_request_rank"]) for row in rows]
    return {
        "states": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "threshold_deg": float(threshold_deg),
        "goal_request_hits": int(sum(goal_request)),
        "goal_request_rate": float(np.mean(goal_request)),
        "goal_executed_hits": int(sum(goal_executed)),
        "goal_executed_rate": float(np.mean(goal_executed)),
        "control_request_hits": int(sum(control_request)),
        "control_request_rate": float(np.mean(control_request)),
        "control_executed_hits": int(sum(control_executed)),
        "control_executed_rate": float(np.mean(control_executed)),
        "goal_control_same_request_states": int(sum(
            row["goal_chosen_direction_deg"]
            == row["control_chosen_direction_deg"] for row in rows)),
        "goal_lower_request_error_states": int(sum(
            goal < control for goal, control in request_pairs)),
        "goal_higher_request_error_states": int(sum(
            goal > control for goal, control in request_pairs)),
        "goal_equal_request_error_states": int(sum(
            goal == control for goal, control in request_pairs)),
        "mean_goal_oracle_request_rank": float(np.mean(goal_ranks)),
        "median_goal_oracle_request_rank": float(np.median(goal_ranks)),
        "mean_control_oracle_request_rank": float(np.mean(control_ranks)),
        "random_expected_request_rank": float(
            (int(rows[0]["direction_count"]) + 1) / 2.0),
        "goal_vs_control_request_gains": int(request_gains),
        "goal_vs_control_request_losses": int(request_losses),
        "goal_vs_control_request_mcnemar_p": exact_mcnemar_p(
            request_gains, request_losses),
        "goal_vs_control_executed_gains": int(executed_gains),
        "goal_vs_control_executed_losses": int(executed_losses),
        "goal_vs_control_executed_mcnemar_p": exact_mcnemar_p(
            executed_gains, executed_losses),
        "random_request_expected_hits": float(sum(probabilities)),
        "goal_request_random_null_upper_tail_p": (
            poisson_binomial_upper_tail(sum(goal_request), probabilities)),
        "median_goal_request_error_deg": float(np.median([
            row["goal_request_error_deg"] for row in rows])),
        "median_control_request_error_deg": float(np.median([
            row["control_request_error_deg"] for row in rows])),
        "mean_goal_score_margin": float(np.mean([
            row["goal_score_margin"] for row in rows])),
        "goal_specificity_positive_states": int(sum(
            value > 0.0 for value in specificity)),
        "mean_goal_vs_control_at_goal_choice": float(np.mean(specificity)),
    }
