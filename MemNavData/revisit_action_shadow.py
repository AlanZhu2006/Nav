"""Pure diagnostics for paired Revisit memory/native NavDP rollouts.

The factual memory-conditioned request appends the current observation to
NavDP's short FIFO.  A native ImageGoal counterfactual is then sampled from
that exact FIFO through the server's read-only resample endpoint.  This module
only validates and summarizes the two responses; it never makes an action
decision and deliberately exposes no deployment threshold.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from MemNavData.navdp_goal_switch import (
        navdp_candidate_diversity,
        normalize_navdp_candidate_scores,
        normalize_navdp_trajectory_candidates,
    )
except ModuleNotFoundError:  # Direct execution from MemNavData/.
    from navdp_goal_switch import (  # type: ignore
        navdp_candidate_diversity,
        normalize_navdp_candidate_scores,
        normalize_navdp_trajectory_candidates,
    )


ACTION_SHADOW_MODE = "native_counterfactual"
ACTION_SHADOW_KEYS = (
    "revisit_action_shadow_mode",
    "revisit_action_shadow_available",
    "revisit_action_shadow_reason",
    "revisit_action_shadow_seed",
    "revisit_action_shadow_memory_mutated",
    "revisit_action_shadow_queue_hash_match",
    "revisit_action_shadow_pointgoal_distance_m",
    "revisit_action_shadow_memory_selected_endpoint_m",
    "revisit_action_shadow_native_selected_endpoint_m",
    "revisit_action_shadow_memory_selected_heading_deg",
    "revisit_action_shadow_native_selected_heading_deg",
    "revisit_action_shadow_memory_zero_candidate_fraction",
    "revisit_action_shadow_native_zero_candidate_fraction",
    "revisit_action_shadow_memory_critic_max",
    "revisit_action_shadow_native_critic_max",
    "revisit_action_shadow_memory_stop_evidence",
    "revisit_action_shadow_native_stop_evidence",
    "revisit_action_shadow_endpoint_mean_ratio_memory_over_native",
    "revisit_action_shadow_selected_endpoint_ratio_memory_over_native",
    "revisit_action_shadow_endpoint_to_pointgoal_ratio",
    "revisit_action_shadow_memory_trajectory_candidate_count",
    "revisit_action_shadow_memory_candidate_endpoint_length_mean",
    "revisit_action_shadow_memory_candidate_endpoint_length_std",
    "revisit_action_shadow_memory_candidate_heading_resultant",
    "revisit_action_shadow_memory_candidate_heading_max_separation_deg",
    "revisit_action_shadow_memory_candidate_endpoint_pairwise_mean",
    "revisit_action_shadow_memory_candidate_path_pairwise_rms_mean",
    "revisit_action_shadow_native_trajectory_candidate_count",
    "revisit_action_shadow_native_candidate_endpoint_length_mean",
    "revisit_action_shadow_native_candidate_endpoint_length_std",
    "revisit_action_shadow_native_candidate_heading_resultant",
    "revisit_action_shadow_native_candidate_heading_max_separation_deg",
    "revisit_action_shadow_native_candidate_endpoint_pairwise_mean",
    "revisit_action_shadow_native_candidate_path_pairwise_rms_mean",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _selected_trajectory(response: dict[str, Any]) -> np.ndarray:
    selected = np.asarray(response.get("trajectory"), dtype=np.float64)
    if selected.ndim == 3 and selected.shape[0] == 1:
        selected = selected[0]
    _require(
        selected.ndim == 2 and selected.shape[-1] == 3,
        f"unexpected selected trajectory shape {selected.shape}",
    )
    _require(np.isfinite(selected).all(), "selected trajectory is non-finite")
    return selected


def _trajectory_heading_deg(trajectory: np.ndarray) -> float | None:
    endpoint = np.asarray(trajectory[-1, :2], dtype=np.float64)
    if float(np.linalg.norm(endpoint)) <= 1e-8:
        return None
    return float(np.degrees(np.arctan2(endpoint[1], endpoint[0])))


def rollout_summary(
    response: dict[str, Any],
    *,
    stop_threshold: float,
) -> dict[str, Any]:
    """Summarize one frozen NavDP response without privileged geometry."""

    candidates = normalize_navdp_trajectory_candidates(
        response.get("all_trajectory")
    )
    _require(np.isfinite(candidates).all(), "candidate trajectories are non-finite")
    selected = _selected_trajectory(response)
    diversity = navdp_candidate_diversity(candidates)
    endpoint_lengths = np.linalg.norm(candidates[:, -1, :2], axis=1)
    scores = normalize_navdp_candidate_scores(
        response.get("all_values"), len(candidates)
    )
    critic_max = None
    if scores is not None:
        finite = scores[np.isfinite(scores)]
        if finite.size:
            critic_max = float(np.max(finite))
    selected_endpoint = float(np.linalg.norm(selected[-1, :2]))
    return {
        **diversity,
        "selected_endpoint_m": selected_endpoint,
        "selected_heading_deg": _trajectory_heading_deg(selected),
        "zero_candidate_fraction": float(np.mean(endpoint_lengths <= 1e-8)),
        "critic_max": critic_max,
        "stop_evidence": (
            bool(critic_max < float(stop_threshold))
            if critic_max is not None
            else None
        ),
    }


def unavailable_action_shadow(reason: str) -> dict[str, Any]:
    """Return an explicit diagnostic receipt when no paired rollout exists."""

    result = {key: None for key in ACTION_SHADOW_KEYS}
    result.update(
        revisit_action_shadow_mode=ACTION_SHADOW_MODE,
        revisit_action_shadow_available=False,
        revisit_action_shadow_reason=str(reason),
    )
    return result


def paired_action_shadow_diagnostics(
    memory_response: dict[str, Any],
    native_response: dict[str, Any],
    *,
    expected_seed: int,
    pointgoal_distance_m: float,
    stop_threshold: float,
) -> dict[str, Any]:
    """Validate and summarize same-FIFO, same-seed memory/native rollouts."""

    seed = int(expected_seed)
    _require(
        int(memory_response.get("diffusion_seed", -1)) == seed,
        "memory rollout seed mismatch",
    )
    _require(
        int(native_response.get("diffusion_seed", -1)) == seed,
        "native shadow seed mismatch",
    )
    _require(
        native_response.get("memory_mutated") is False,
        "native shadow did not assert read-only FIFO semantics",
    )
    before = native_response.get("queue_hashes_before")
    after = native_response.get("queue_hashes_after")
    _require(
        isinstance(before, list) and bool(before),
        "native shadow omitted FIFO fingerprints",
    )
    _require(before == after, "native shadow changed FIFO content")
    pointgoal_distance = _finite_float(
        pointgoal_distance_m, "pointgoal_distance_m"
    )
    _require(pointgoal_distance >= 0.0, "pointgoal distance must be non-negative")

    memory = rollout_summary(memory_response, stop_threshold=stop_threshold)
    native = rollout_summary(native_response, stop_threshold=stop_threshold)
    memory_mean = float(memory["candidate_endpoint_length_mean"])
    native_mean = float(native["candidate_endpoint_length_mean"])
    memory_selected = float(memory["selected_endpoint_m"])
    native_selected = float(native["selected_endpoint_m"])

    result = {key: None for key in ACTION_SHADOW_KEYS}
    result.update(
        revisit_action_shadow_mode=ACTION_SHADOW_MODE,
        revisit_action_shadow_available=True,
        revisit_action_shadow_reason="paired_same_fifo_same_seed",
        revisit_action_shadow_seed=seed,
        revisit_action_shadow_memory_mutated=False,
        revisit_action_shadow_queue_hash_match=True,
        revisit_action_shadow_pointgoal_distance_m=pointgoal_distance,
        revisit_action_shadow_memory_selected_endpoint_m=memory_selected,
        revisit_action_shadow_native_selected_endpoint_m=native_selected,
        revisit_action_shadow_memory_selected_heading_deg=memory[
            "selected_heading_deg"
        ],
        revisit_action_shadow_native_selected_heading_deg=native[
            "selected_heading_deg"
        ],
        revisit_action_shadow_memory_zero_candidate_fraction=memory[
            "zero_candidate_fraction"
        ],
        revisit_action_shadow_native_zero_candidate_fraction=native[
            "zero_candidate_fraction"
        ],
        revisit_action_shadow_memory_critic_max=memory["critic_max"],
        revisit_action_shadow_native_critic_max=native["critic_max"],
        revisit_action_shadow_memory_stop_evidence=memory["stop_evidence"],
        revisit_action_shadow_native_stop_evidence=native["stop_evidence"],
        revisit_action_shadow_endpoint_mean_ratio_memory_over_native=(
            memory_mean / native_mean if native_mean > 1e-8 else None
        ),
        revisit_action_shadow_selected_endpoint_ratio_memory_over_native=(
            memory_selected / native_selected if native_selected > 1e-8 else None
        ),
        revisit_action_shadow_endpoint_to_pointgoal_ratio=(
            memory_mean / pointgoal_distance
            if pointgoal_distance > 1e-8
            else None
        ),
    )
    for prefix, summary in (("memory", memory), ("native", native)):
        for field in (
            "trajectory_candidate_count",
            "candidate_endpoint_length_mean",
            "candidate_endpoint_length_std",
            "candidate_heading_resultant",
            "candidate_heading_max_separation_deg",
            "candidate_endpoint_pairwise_mean",
            "candidate_path_pairwise_rms_mean",
        ):
            result[f"revisit_action_shadow_{prefix}_{field}"] = summary[field]
    return result
