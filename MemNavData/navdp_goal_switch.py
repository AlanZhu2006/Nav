"""Causal controls for NavDP state at an ImageGoal switch.

LingBot/MemNav owns the long-term episode memory.  NavDP separately keeps a
bounded FIFO of recent decision observations.  These helpers reset only that
NavDP FIFO, so a goal-switch ablation cannot accidentally erase the long-term
memory that the revisit leg is intended to test.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


RESET_MODES = ("carry", "before_b", "before_c", "every_goal")
TRAJECTORY_SELECTOR_SCOPES = ("all", "leg_a", "leg_b", "leg_c")


def should_reset_before_leg(mode: str, leg_index: int) -> bool:
    """Return whether the NavDP FIFO is reset before the requested leg.

    Leg indices are zero based: A=0, B=1, C=2.  ``before_b`` deliberately
    isolates the Novel A->B transition; ``every_goal`` is the broader follow-up
    ablation and resets before both B and C.  ``before_c`` is the minimal
    double-Revisit intervention: B remains natural, while C cannot consume
    the intervening B rollout through NavDP's bounded local FIFO.
    """
    if mode not in RESET_MODES:
        raise ValueError(f"unknown NavDP goal-switch mode: {mode!r}")
    if leg_index < 1:
        return False
    return (
        mode == "every_goal"
        or (mode == "before_b" and leg_index == 1)
        or (mode == "before_c" and leg_index == 2)
    )


def trajectory_selector_for_leg(
    selector: str,
    scope: str,
    leg_index: int | None,
) -> str:
    """Limit a privileged trajectory-selector intervention to one leg.

    This keeps all non-target legs on the server-selected trajectory, which is
    necessary for a causal multi-goal diagnostic.  It is intentionally an
    evaluation helper and never changes diffusion samples.
    """
    if selector not in ("server", "oracle_geodesic"):
        raise ValueError(f"unknown trajectory selector: {selector!r}")
    if scope not in TRAJECTORY_SELECTOR_SCOPES:
        raise ValueError(f"unknown trajectory selector scope: {scope!r}")
    if selector == "server" or scope == "all":
        return selector
    if leg_index is None:
        raise ValueError("a leg-scoped trajectory selector requires leg_index")
    target_leg = {"leg_a": 0, "leg_b": 1, "leg_c": 2}[scope]
    return selector if int(leg_index) == target_leg else "server"


def normalize_navdp_trajectory_candidates(value) -> np.ndarray:
    """Normalize NavDP's single-env candidate batch to ``[N,T,3]``."""
    candidates = np.asarray(value, dtype=float)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if (candidates.ndim != 3 or candidates.shape[-1] != 3
            or len(candidates) == 0):
        raise ValueError(f"unexpected all_trajectory shape {candidates.shape}")
    return candidates


def normalize_navdp_candidate_scores(value, candidate_count: int):
    """Return a single-env score vector, or ``None`` for an unknown layout."""
    scores = np.asarray(value, dtype=float)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    return scores if scores.shape == (int(candidate_count),) else None


def pool_navdp_candidate_sets(responses: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate candidate paths/scores returned for independent seeds."""
    if not responses:
        raise ValueError("at least one candidate response is required")
    path_sets = []
    score_sets = []
    expected_shape = None
    for response in responses:
        paths = normalize_navdp_trajectory_candidates(
            response.get("all_trajectory"))
        if expected_shape is None:
            expected_shape = paths.shape[1:]
        elif paths.shape[1:] != expected_shape:
            raise ValueError("candidate trajectory shapes differ across seeds")
        scores = normalize_navdp_candidate_scores(
            response.get("all_values"), len(paths))
        if scores is None:
            raise ValueError("candidate score shape differs from trajectories")
        path_sets.append(paths)
        score_sets.append(scores)
    return np.concatenate(path_sets, axis=0), np.concatenate(score_sets, axis=0)


def navdp_candidate_diversity(candidates) -> dict:
    """Summarize directional and geometric diversity without saving paths."""
    paths = normalize_navdp_trajectory_candidates(candidates)
    endpoints = paths[:, -1, :2]
    lengths = np.linalg.norm(endpoints, axis=1)
    valid = lengths > 1e-8
    if np.any(valid):
        angles = np.arctan2(endpoints[valid, 1], endpoints[valid, 0])
        resultant = float(np.abs(np.mean(np.exp(1j * angles))))
        angle_delta = np.angle(
            np.exp(1j * (angles[:, None] - angles[None, :])))
        max_heading_separation_deg = float(
            np.degrees(np.max(np.abs(angle_delta))))
    else:
        resultant = None
        max_heading_separation_deg = None
    if len(paths) > 1:
        upper = np.triu_indices(len(paths), k=1)
        endpoint_pairwise = np.linalg.norm(
            endpoints[:, None] - endpoints[None, :], axis=-1)[upper]
        path_pairwise = np.sqrt(np.mean(
            np.square(paths[:, None, :, :2] - paths[None, :, :, :2]),
            axis=(2, 3),
        ))[upper]
        endpoint_pairwise_mean = float(np.mean(endpoint_pairwise))
        path_pairwise_rms_mean = float(np.mean(path_pairwise))
    else:
        endpoint_pairwise_mean = 0.0
        path_pairwise_rms_mean = 0.0
    return {
        "trajectory_candidate_count": int(len(paths)),
        "candidate_endpoint_length_mean": float(np.mean(lengths)),
        "candidate_endpoint_length_std": float(np.std(lengths)),
        "candidate_heading_resultant": resultant,
        "candidate_heading_max_separation_deg": max_heading_separation_deg,
        "candidate_endpoint_pairwise_mean": endpoint_pairwise_mean,
        "candidate_path_pairwise_rms_mean": path_pairwise_rms_mean,
    }


def navdp_server_base(
    server_backend: str,
    base_url: str,
    novel_base_url: str | None,
) -> str:
    """Select the server that owns the frozen NavDP local controller."""
    if server_backend in ("navdp", "cec_portability"):
        return base_url
    if server_backend in ("hybrid_oracle", "hybrid_pose"):
        if novel_base_url is None:
            raise ValueError(f"{server_backend} requires a NavDP novel server")
        return novel_base_url
    raise ValueError(
        "short-memory-only reset is unavailable for a standalone MemNav server"
    )


def reset_navdp_short_memory(
    post: Callable,
    server_backend: str,
    base_url: str,
    novel_base_url: str | None,
    env_id: int = 0,
) -> dict:
    """Clear only NavDP's recent-observation FIFO through its HTTP endpoint."""
    navdp_base = navdp_server_base(server_backend, base_url, novel_base_url)
    response = post(
        f"{navdp_base}/navigator_reset_env",
        json={"env_id": int(env_id)},
    )
    response.raise_for_status()
    payload = response.json()
    expected_algo = (
        "cec_controller_portability"
        if server_backend == "cec_portability" else "navdp")
    if payload.get("algo") != expected_algo:
        raise RuntimeError(
            "short-memory reset reached a non-NavDP/wrong controller server: "
            f"{payload.get('algo')!r}"
        )
    return payload
