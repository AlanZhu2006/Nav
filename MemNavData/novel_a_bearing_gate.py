"""Pure protocol helpers for the frozen Novel-A oracle-bearing gate."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


ARMS = ("native", "ideal_periodic_yaw", "oracle_token_periodic")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wrap_deg(value: float) -> float:
    """Wrap degrees into [-180, 180)."""
    return (float(value) + 180.0) % 360.0 - 180.0


def rotated_arm_order(scene_index: int, episode_index: int) -> tuple[str, ...]:
    if scene_index < 0 or episode_index < 0:
        raise ValueError("scene and episode indices must be non-negative")
    offset = (int(scene_index) + int(episode_index)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def token_request_deg(residual_deg: float, clip_deg: float = 100.0) -> float:
    if not math.isfinite(residual_deg):
        raise ValueError("bearing residual must be finite")
    if not math.isfinite(clip_deg) or clip_deg <= 0.0:
        raise ValueError("clip_deg must be finite and positive")
    magnitude = min(abs(float(residual_deg)), float(clip_deg))
    return math.copysign(magnitude, float(residual_deg)) if magnitude else 0.0


def normalize_selected_trajectory(value: Any) -> np.ndarray:
    trajectory = np.asarray(value, dtype=float)
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"unexpected selected trajectory shape {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError("selected trajectory contains non-finite values")
    return trajectory


def _normalize_candidates(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    candidates = np.asarray(value, dtype=float)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        return None
    return candidates if np.isfinite(candidates).all() else None


def _normalize_values(value: Any, count: int) -> np.ndarray | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float).reshape(-1)
    if len(values) != count or not np.isfinite(values).all():
        return None
    return values


def _trajectory_heading_deg(trajectory: np.ndarray) -> float | None:
    planar = np.asarray(trajectory, dtype=float)[:, :2]
    distances = np.linalg.norm(planar, axis=1)
    eligible = np.flatnonzero(distances >= 0.3)
    endpoint = planar[eligible[-1]] if eligible.size else planar[-1]
    if float(np.linalg.norm(endpoint)) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(endpoint[1], endpoint[0])))


def critic_shadow_diagnostics(
    response: dict,
    *,
    requested_heading_deg: float | None = None,
) -> dict[str, Any]:
    """Summarize the goal-blind critic without using it as a direction router."""
    selected = normalize_selected_trajectory(response["trajectory"])
    selected_heading = _trajectory_heading_deg(selected)
    output: dict[str, Any] = {
        "candidate_count": None,
        "critic_max": None,
        "critic_min": None,
        "critic_unique_4dp": None,
        "selected_candidate_index": None,
        "selected_heading_deg": selected_heading,
        "heading_resultant_r": None,
        "selected_request_error_deg": None,
        "best_direction_candidate_index": None,
        "best_direction_critic_rank": None,
        "best_direction_error_deg": None,
    }
    candidates = _normalize_candidates(response.get("all_trajectory"))
    if candidates is None:
        return output
    count = int(len(candidates))
    output["candidate_count"] = count
    headings = [_trajectory_heading_deg(candidate) for candidate in candidates]
    finite_headings = np.asarray(
        [heading for heading in headings if heading is not None], dtype=float)
    if finite_headings.size:
        radians = np.radians(finite_headings)
        output["heading_resultant_r"] = float(np.hypot(
            np.mean(np.cos(radians)), np.mean(np.sin(radians))))

    if candidates.shape[1:] == selected.shape:
        errors = np.max(np.abs(candidates - selected[None]), axis=(1, 2))
        output["selected_candidate_index"] = int(np.argmin(errors))

    values = _normalize_values(response.get("all_values"), count)
    if values is not None:
        output["critic_max"] = float(np.max(values))
        output["critic_min"] = float(np.min(values))
        output["critic_unique_4dp"] = int(len(np.unique(np.round(values, 4))))

    if requested_heading_deg is None or not finite_headings.size:
        return output
    requested = float(requested_heading_deg)
    directional_errors = np.asarray([
        float("inf") if heading is None else abs(wrap_deg(heading - requested))
        for heading in headings
    ])
    best = int(np.argmin(directional_errors))
    output["best_direction_candidate_index"] = best
    output["best_direction_error_deg"] = float(directional_errors[best])
    if selected_heading is not None:
        output["selected_request_error_deg"] = abs(
            wrap_deg(selected_heading - requested))
    if values is not None:
        # Rank one is the critic's preferred candidate. Stable sorting makes
        # ties deterministic and does not turn this shadow into a selector.
        order = np.argsort(-values, kind="stable")
        output["best_direction_critic_rank"] = int(
            np.flatnonzero(order == best)[0] + 1)
    return output
