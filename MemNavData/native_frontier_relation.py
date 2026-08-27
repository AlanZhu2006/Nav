"""Goal-conditioned relations between frozen NavDP plans and metric frontiers.

The observed frontier is geometric and therefore goal-blind.  NavDP's native
ImageGoal trajectories carry the missing goal-conditioned signal, but their
short local horizon is not a global plan.  This module only summarizes how a
metric frontier candidate relates to that *already sampled* native proposal
set.  It never reads Habitat pose, geodesic distance, success, or target
coordinates.

The resulting fixed-length vector is intended for
``candidate.features.native_proposal_relation`` in
``novel_candidate_set_schema_v2``.  A deterministic union shortlist retains
native-aligned, patch-related, topological, and angularly diverse candidates;
it is a coverage heuristic, not a learned decision or a utility label.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np


NATIVE_RELATION_SCHEMA_VERSION = "nlsr_native_frontier_relation_v1"

NATIVE_RELATION_FEATURE_NAMES = (
    "selected_endpoint_distance_m",
    "selected_path_distance_m",
    "selected_direction_cosine",
    "selected_radial_gap_m",
    "best_endpoint_distance_m",
    "best_path_distance_m",
    "best_direction_cosine",
    "best_radial_gap_m",
    "endpoint_aligned_fraction",
    "path_aligned_fraction",
    "native_endpoint_length_mean_m",
    "native_endpoint_length_std_m",
    "native_heading_resultant",
    "native_heading_max_separation_rad",
    "native_endpoint_pairwise_mean_m",
    "native_moving_fraction",
    "native_value_present",
    "value_weighted_endpoint_distance_m",
    "value_weighted_direction_cosine",
    "native_value_std",
    "selected_value_zscore",
)


class NativeFrontierRelationError(ValueError):
    """A deployment proposal or candidate violates the relation contract."""


@dataclass(frozen=True)
class NativeRelationConfig:
    endpoint_alignment_m: float = 0.75
    path_alignment_m: float = 0.75
    moving_epsilon_m: float = 1e-5
    softmax_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "endpoint_alignment_m",
            "path_alignment_m",
            "moving_epsilon_m",
            "softmax_temperature",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or float(value) <= 0.0):
                raise NativeFrontierRelationError(
                    f"{name} must be finite and positive")


def _finite_trajectories(value: object) -> np.ndarray:
    trajectories = np.asarray(value, dtype=np.float64)
    if (trajectories.ndim != 3 or trajectories.shape[0] < 1
            or trajectories.shape[1] < 1 or trajectories.shape[2] not in (2, 3)
            or not np.isfinite(trajectories).all()):
        raise NativeFrontierRelationError(
            "native trajectories must be finite [K,T,2|3]")
    return trajectories[..., :2].copy()


def _finite_candidate(value: object) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64)
    if candidate.shape != (2,) or not np.isfinite(candidate).all():
        raise NativeFrontierRelationError(
            "frontier candidate must be finite [forward,left]")
    if float(np.linalg.norm(candidate)) <= 0.0:
        raise NativeFrontierRelationError(
            "frontier candidate must not coincide with the current pose")
    return candidate.copy()


def _selected_index(value: object, count: int) -> int:
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or not 0 <= int(value) < count):
        raise NativeFrontierRelationError(
            "selected trajectory index is outside the native proposal set")
    return int(value)


def _finite_values(value: object, count: int) -> np.ndarray | None:
    if value is None:
        return None
    scores = np.asarray(value, dtype=np.float64).reshape(-1)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise NativeFrontierRelationError(
            "native values must be finite and have one value per trajectory")
    return scores.copy()


def _direction_cosines(endpoints: np.ndarray, candidate: np.ndarray,
                       epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(endpoints, axis=1)
    moving = lengths > float(epsilon)
    cosine = np.zeros(len(endpoints), dtype=np.float64)
    if np.any(moving):
        candidate_length = float(np.linalg.norm(candidate))
        cosine[moving] = (
            endpoints[moving] @ candidate
            / (lengths[moving] * candidate_length)
        )
        cosine = np.clip(cosine, -1.0, 1.0)
    return cosine, moving


def _heading_max_separation(endpoints: np.ndarray,
                            moving: np.ndarray) -> tuple[float, float]:
    if not np.any(moving):
        return 0.0, 0.0
    headings = np.arctan2(endpoints[moving, 1], endpoints[moving, 0])
    resultant = abs(np.mean(np.exp(1j * headings)))
    if len(headings) == 1:
        maximum = 0.0
    else:
        delta = headings[:, None] - headings[None, :]
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        maximum = float(np.max(np.abs(delta)))
    return float(resultant), maximum


def _pairwise_endpoint_mean(endpoints: np.ndarray) -> float:
    if len(endpoints) < 2:
        return 0.0
    distances = np.linalg.norm(
        endpoints[:, None, :] - endpoints[None, :, :], axis=-1)
    upper = distances[np.triu_indices(len(endpoints), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _softmax(value: np.ndarray, temperature: float) -> np.ndarray:
    centered = value / float(temperature)
    centered -= float(np.max(centered))
    weights = np.exp(centered)
    denominator = float(np.sum(weights))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise NativeFrontierRelationError("native value softmax is invalid")
    return weights / denominator


def native_frontier_relation(
    native_trajectories: object,
    candidate_forward_left_m: object,
    *,
    selected_index: int,
    native_values: object = None,
    config: NativeRelationConfig = NativeRelationConfig(),
) -> np.ndarray:
    """Return a fixed, finite deployment-only relation feature vector."""
    if not isinstance(config, NativeRelationConfig):
        raise NativeFrontierRelationError("config has the wrong type")
    trajectories = _finite_trajectories(native_trajectories)
    candidate = _finite_candidate(candidate_forward_left_m)
    selected = _selected_index(selected_index, len(trajectories))
    values = _finite_values(native_values, len(trajectories))

    endpoints = trajectories[:, -1, :]
    endpoint_distance = np.linalg.norm(endpoints - candidate[None, :], axis=1)
    path_distance = np.min(
        np.linalg.norm(
            trajectories - candidate[None, None, :], axis=-1),
        axis=1,
    )
    endpoint_lengths = np.linalg.norm(endpoints, axis=1)
    candidate_length = float(np.linalg.norm(candidate))
    radial_gap = np.abs(endpoint_lengths - candidate_length)
    direction_cosine, moving = _direction_cosines(
        endpoints, candidate, config.moving_epsilon_m)
    resultant, max_separation = _heading_max_separation(endpoints, moving)

    if values is None:
        value_present = 0.0
        weighted_endpoint_distance = 0.0
        weighted_direction_cosine = 0.0
        value_std = 0.0
        selected_zscore = 0.0
    else:
        value_present = 1.0
        weights = _softmax(values, config.softmax_temperature)
        weighted_endpoint_distance = float(weights @ endpoint_distance)
        weighted_direction_cosine = float(weights @ direction_cosine)
        value_std = float(np.std(values))
        selected_zscore = (
            float((values[selected] - np.mean(values)) / value_std)
            if value_std > 1e-12 else 0.0
        )

    features = np.asarray([
        endpoint_distance[selected],
        path_distance[selected],
        direction_cosine[selected],
        radial_gap[selected],
        np.min(endpoint_distance),
        np.min(path_distance),
        np.max(direction_cosine),
        np.min(radial_gap),
        np.mean(endpoint_distance <= config.endpoint_alignment_m),
        np.mean(path_distance <= config.path_alignment_m),
        np.mean(endpoint_lengths),
        np.std(endpoint_lengths),
        resultant,
        max_separation,
        _pairwise_endpoint_mean(endpoints),
        np.mean(moving),
        value_present,
        weighted_endpoint_distance,
        weighted_direction_cosine,
        value_std,
        selected_zscore,
    ], dtype=np.float64)
    if (features.shape != (len(NATIVE_RELATION_FEATURE_NAMES),)
            or not np.isfinite(features).all()):
        raise NativeFrontierRelationError(
            "native/frontier relation produced invalid features")
    return features


def _candidate_id(candidate: Mapping[str, object]) -> str:
    value = candidate.get("candidate_id")
    if not isinstance(value, str) or not value or value != value.strip():
        raise NativeFrontierRelationError(
            "frontier candidate_id must be a trimmed non-empty string")
    return value


def _candidate_scalar(candidate: Mapping[str, object], key: str) -> float:
    value = candidate.get(key)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise NativeFrontierRelationError(
            f"frontier candidate {key} must be finite")
    return float(value)


def _bearing(candidate: Mapping[str, object]) -> float:
    return math.atan2(
        _candidate_scalar(candidate, "subgoal_left_m"),
        _candidate_scalar(candidate, "subgoal_forward_m"),
    )


def _angular_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + math.pi)
               % (2.0 * math.pi) - math.pi)


def native_conditioned_union_shortlist(
    candidates: Sequence[Mapping[str, object]],
    native_trajectories: object,
    *,
    selected_index: int,
    native_values: object = None,
    max_candidates: int = 8,
    native_slots: int = 2,
    patch_slots: int = 2,
    topology_slots: int = 2,
    config: NativeRelationConfig = NativeRelationConfig(),
) -> list[dict[str, object]]:
    """Build a deterministic, source-balanced deployment shortlist.

    Candidate generation remains independent of privileged utility.  The
    returned records are copies with ``native_proposal_relation`` and explicit
    ``native_relation_selection_sources`` fields added.
    """
    counts = (max_candidates, native_slots, patch_slots, topology_slots)
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           or int(value) < 0 for value in counts):
        raise NativeFrontierRelationError(
            "shortlist counts must be non-negative integers")
    if int(max_candidates) < 1:
        raise NativeFrontierRelationError("max_candidates must be positive")
    if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes, bytearray)):
        raise NativeFrontierRelationError("candidates must be a sequence")

    enriched: list[dict[str, object]] = []
    ids: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise NativeFrontierRelationError(
                "every frontier candidate must be a mapping")
        candidate_id = _candidate_id(raw)
        if candidate_id in ids:
            raise NativeFrontierRelationError(
                "frontier candidate ids must be unique")
        ids.add(candidate_id)
        local = (
            _candidate_scalar(raw, "subgoal_forward_m"),
            _candidate_scalar(raw, "subgoal_left_m"),
        )
        relation = native_frontier_relation(
            native_trajectories,
            local,
            selected_index=selected_index,
            native_values=native_values,
            config=config,
        )
        row = dict(raw)
        row["native_proposal_relation_schema"] = (
            NATIVE_RELATION_SCHEMA_VERSION)
        row["native_proposal_relation"] = relation.tolist()
        row["native_relation_selection_sources"] = []
        enriched.append(row)

    by_id = {str(row["candidate_id"]): row for row in enriched}
    chosen: list[str] = []

    def add(rows: Sequence[Mapping[str, object]], source: str,
            limit: int) -> None:
        accepted = 0
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id in chosen:
                sources = by_id[candidate_id][
                    "native_relation_selection_sources"]
                assert isinstance(sources, list)
                if source not in sources:
                    sources.append(source)
                continue
            if len(chosen) >= int(max_candidates) or accepted >= int(limit):
                break
            chosen.append(candidate_id)
            sources = by_id[candidate_id][
                "native_relation_selection_sources"]
            assert isinstance(sources, list)
            sources.append(source)
            accepted += 1

    native_order = sorted(enriched, key=lambda row: (
        # First preserve the policy's actually selected ImageGoal direction.
        # Best-of-K agreement remains in the feature vector for the learned
        # ranker, but cannot displace the selected native intent in this
        # deterministic coverage slot.
        float(row["native_proposal_relation"][1]),  # selected path distance
        float(row["native_proposal_relation"][0]),  # selected endpoint distance
        float(row["native_proposal_relation"][5]),  # best path distance
        str(row["candidate_id"]),
    ))
    add(native_order, "native_aligned", int(native_slots))

    patch_order = sorted(
        (row for row in enriched
         if bool(row.get("goal_patch_relation_present", False))),
        key=lambda row: (
            -_candidate_scalar(row, "goal_patch_relation_score"),
            str(row["candidate_id"]),
        ),
    )
    add(patch_order, "goal_patch", int(patch_slots))

    topology_order = sorted(enriched, key=lambda row: (
        -_candidate_scalar(row, "topology_score"),
        str(row["candidate_id"]),
    ))
    add(topology_order, "topology", int(topology_slots))

    # Fill remaining slots by farthest bearing from the already selected set.
    while len(chosen) < min(int(max_candidates), len(enriched)):
        selected_bearings = [_bearing(by_id[value]) for value in chosen]
        remaining = [row for row in enriched
                     if str(row["candidate_id"]) not in chosen]
        if not remaining:
            break
        if selected_bearings:
            next_row = sorted(remaining, key=lambda row: (
                -min(_angular_distance(_bearing(row), value)
                     for value in selected_bearings),
                -_candidate_scalar(row, "topology_score"),
                str(row["candidate_id"]),
            ))[0]
        else:
            next_row = topology_order[0]
        add([next_row], "angular_diverse", 1)

    return [by_id[candidate_id] for candidate_id in chosen]


__all__ = [
    "NATIVE_RELATION_FEATURE_NAMES",
    "NATIVE_RELATION_SCHEMA_VERSION",
    "NativeFrontierRelationError",
    "NativeRelationConfig",
    "native_conditioned_union_shortlist",
    "native_frontier_relation",
]
