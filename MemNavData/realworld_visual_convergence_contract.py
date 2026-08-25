"""Fail-closed contract for calibrating scale-free visual arrival evidence.

This module deliberately owns no matcher, policy, ROS publisher, or actuator.
It consumes the measurement rows produced by the real-world LightGlue audit
and separates three questions that were previously conflated:

1. does the current/goal pair have enough two-view support to be meaningful;
2. is the supported pair visually close to the goal view in image coordinates;
3. has that evidence persisted while the terminal controller is holding still.

No monocular translation norm is read.  A rule is supplied explicitly after a
location-disjoint physical calibration; this file never tunes thresholds from
evaluation rows.  Until a frozen rule passes a held-out confirmation protocol,
the returned STOP decision is shadow evidence only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "realworld_scale_free_arrival_contract_v1_20260825"
CALIBRATION_SCHEMA_VERSION = (
    "realworld_scale_free_arrival_population_v1_20260825"
)
VISUAL_AUDIT_SCHEMA_VERSION = "realworld_visual_convergence_audit_v1_20260825"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPLITS = frozenset({"calibration", "confirmation"})


class ArrivalContractError(ValueError):
    """An input would make the arrival result ambiguous or unauditable."""


def _finite_number(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ArrivalContractError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArrivalContractError(f"{name} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ArrivalContractError(f"{name} must be a finite number")
    return parsed


def _rule_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ArrivalContractError(f"{name} must be a finite JSON number")
    return _finite_number(value, name)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ArrivalContractError(f"{name} must be a non-negative integer")
    parsed = value
    if parsed < 0:
        raise ArrivalContractError(f"{name} must be a non-negative integer")
    return parsed


def _sha256(value: object, name: str) -> str:
    parsed = str(value)
    if not _SHA256.fullmatch(parsed):
        raise ArrivalContractError(f"{name} must be a lowercase SHA-256")
    return parsed


@dataclass(frozen=True)
class ScaleFreeConvergenceRule:
    """A pre-frozen image-space gate; lower residuals mean closer views."""

    min_fundamental_inliers: int
    min_reference_hull_coverage: float
    min_query_hull_coverage: float
    max_identity_flow_median_diag: float
    max_affine_corner_identity_diag: float
    max_abs_affine_rotation_deg: float
    min_affine_scale: float
    max_affine_scale: float
    consecutive_hold_observations: int
    maximum_frame_gap: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.min_fundamental_inliers) is not int
            or self.min_fundamental_inliers < 8
        ):
            raise ArrivalContractError(
                "min_fundamental_inliers must be an integer >= 8"
            )
        for name in (
            "min_reference_hull_coverage",
            "min_query_hull_coverage",
        ):
            value = _rule_number(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ArrivalContractError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "max_identity_flow_median_diag",
            "max_affine_corner_identity_diag",
            "max_abs_affine_rotation_deg",
        ):
            value = _rule_number(getattr(self, name), name)
            if value < 0.0:
                raise ArrivalContractError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        minimum_scale = _rule_number(self.min_affine_scale, "min_affine_scale")
        maximum_scale = _rule_number(self.max_affine_scale, "max_affine_scale")
        if minimum_scale <= 0.0 or maximum_scale < minimum_scale:
            raise ArrivalContractError(
                "affine scale interval must be finite, positive, and ordered"
            )
        object.__setattr__(self, "min_affine_scale", minimum_scale)
        object.__setattr__(self, "max_affine_scale", maximum_scale)
        if (
            type(self.consecutive_hold_observations) is not int
            or self.consecutive_hold_observations < 2
        ):
            raise ArrivalContractError(
                "consecutive_hold_observations must be an integer >= 2"
            )
        if type(self.maximum_frame_gap) is not int or self.maximum_frame_gap < 1:
            raise ArrivalContractError(
                "maximum_frame_gap must be a positive integer"
            )

    def receipt(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class ObservationDecision:
    passed: bool
    reason: str
    convergence_residual: float | None
    metric_translation_consumed: bool = False

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_scale_free_observation(
    row: Mapping[str, object],
    rule: ScaleFreeConvergenceRule,
) -> ObservationDecision:
    """Apply one frozen rule without reading metric distance or semantic role."""

    if not isinstance(row, Mapping):
        raise ArrivalContractError("visual convergence row must be a mapping")
    if row.get("schema_version") != VISUAL_AUDIT_SCHEMA_VERSION:
        return ObservationDecision(False, "audit_schema_mismatch", None)
    if row.get("certificate_precheck_passed") is not True:
        return ObservationDecision(False, "two_view_precheck_rejected", None)

    try:
        inliers = _nonnegative_int(
            row.get("fundamental_inliers"), "fundamental_inliers"
        )
        reference_hull = _finite_number(
            row.get("fundamental_reference_hull_coverage"),
            "fundamental_reference_hull_coverage",
        )
        query_hull = _finite_number(
            row.get("fundamental_query_hull_coverage"),
            "fundamental_query_hull_coverage",
        )
        identity_flow = _finite_number(
            row.get("identity_flow_median_diag"),
            "identity_flow_median_diag",
        )
        affine_identity = _finite_number(
            row.get("affine_corner_identity_max_diag"),
            "affine_corner_identity_max_diag",
        )
        affine_rotation = abs(_finite_number(
            row.get("affine_rotation_deg"), "affine_rotation_deg"
        ))
        affine_scale = _finite_number(row.get("affine_scale"), "affine_scale")
    except ArrivalContractError as error:
        return ObservationDecision(False, f"invalid_measurement:{error}", None)

    checks = (
        (inliers >= rule.min_fundamental_inliers, "insufficient_inliers"),
        (
            reference_hull >= rule.min_reference_hull_coverage,
            "insufficient_reference_coverage",
        ),
        (
            query_hull >= rule.min_query_hull_coverage,
            "insufficient_query_coverage",
        ),
        (
            row.get("affine_valid") is True,
            "affine_model_unavailable",
        ),
        (
            identity_flow <= rule.max_identity_flow_median_diag,
            "identity_flow_too_large",
        ),
        (
            affine_identity <= rule.max_affine_corner_identity_diag,
            "affine_identity_error_too_large",
        ),
        (
            affine_rotation <= rule.max_abs_affine_rotation_deg,
            "affine_rotation_too_large",
        ),
        (
            rule.min_affine_scale <= affine_scale <= rule.max_affine_scale,
            "affine_scale_outside_range",
        ),
    )
    residual = max(identity_flow, affine_identity)
    for passed, reason in checks:
        if not passed:
            return ObservationDecision(False, reason, residual)
    return ObservationDecision(True, "scale_free_visual_convergence", residual)


@dataclass(frozen=True)
class LatchDecision:
    disposition: str
    reason: str
    streak: int
    observation_passed: bool
    shadow_stop_authorized: bool
    runtime_stop_authorized: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
            "authority": "shadow_only_until_physical_confirmation",
        }


class ScaleFreeConvergenceLatch:
    """Two-stage terminal state: request hold, then accumulate still evidence.

    ``shadow_stop_authorized`` means that the supplied frozen rule and temporal
    contract passed.  ``runtime_stop_authorized`` intentionally remains false:
    promotion to an actuator contract is a separate, physically confirmed
    release step.
    """

    def __init__(self, rule: ScaleFreeConvergenceRule) -> None:
        self.rule = rule
        self._goal_sha256: str | None = None
        self._last_frame_index: int | None = None
        self._streak = 0
        self._hold_requested = False

    def reset(self) -> None:
        self._goal_sha256 = None
        self._last_frame_index = None
        self._streak = 0
        self._hold_requested = False

    def update(
        self,
        *,
        goal_sha256: str,
        frame_index: int,
        observation: Mapping[str, object],
        terminal_hold_active: bool,
    ) -> LatchDecision:
        goal_sha256 = _sha256(goal_sha256, "goal_sha256")
        frame_index = _nonnegative_int(frame_index, "frame_index")
        if type(terminal_hold_active) is not bool:
            raise ArrivalContractError("terminal_hold_active must be bool")

        goal_changed = self._goal_sha256 not in (None, goal_sha256)
        noncausal = (
            self._last_frame_index is not None
            and (
                frame_index <= self._last_frame_index
                or frame_index - self._last_frame_index
                > self.rule.maximum_frame_gap
            )
        )
        if goal_changed or noncausal:
            self._streak = 0
            self._hold_requested = False
        self._goal_sha256 = goal_sha256
        self._last_frame_index = frame_index

        decision = evaluate_scale_free_observation(observation, self.rule)
        if not decision.passed:
            self._streak = 0
            self._hold_requested = False
            return LatchDecision(
                disposition="replan",
                reason=decision.reason,
                streak=0,
                observation_passed=False,
                shadow_stop_authorized=False,
            )
        if not terminal_hold_active:
            self._streak = 0
            self._hold_requested = True
            return LatchDecision(
                disposition="request_hold",
                reason="visual_candidate_requires_stationary_confirmation",
                streak=0,
                observation_passed=True,
                shadow_stop_authorized=False,
            )
        if not self._hold_requested:
            self._streak = 0
            self._hold_requested = True
            return LatchDecision(
                disposition="request_hold",
                reason="terminal_hold_requires_explicit_causal_arm",
                streak=0,
                observation_passed=True,
                shadow_stop_authorized=False,
            )

        self._streak += 1
        authorized = self._streak >= self.rule.consecutive_hold_observations
        return LatchDecision(
            disposition="shadow_stop" if authorized else "hold",
            reason=(
                "persistent_scale_free_visual_convergence"
                if authorized
                else "accumulating_stationary_visual_evidence"
            ),
            streak=self._streak,
            observation_passed=True,
            shadow_stop_authorized=authorized,
        )


def validate_labeled_population(payload: Mapping[str, object]) -> dict[str, Any]:
    """Validate physical labels and enforce location-disjoint confirmation.

    This function validates provenance only.  It does not inspect visual scores
    and therefore cannot leak confirmation metrics into threshold selection.
    """

    if not isinstance(payload, Mapping):
        raise ArrivalContractError("calibration population must be a mapping")
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ArrivalContractError("calibration population schema mismatch")
    radius = _finite_number(payload.get("success_radius_m"), "success_radius_m")
    if radius <= 0.0:
        raise ArrivalContractError("success_radius_m must be positive")
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ArrivalContractError("samples must be a sequence")
    if not samples:
        raise ArrivalContractError("calibration population is empty")

    seen_ids: set[str] = set()
    split_locations = {split: set() for split in _SPLITS}
    counts = {
        split: {"positive": 0, "negative": 0, "samples": 0}
        for split in _SPLITS
    }
    normalized = []
    for ordinal, raw in enumerate(samples):
        if not isinstance(raw, Mapping):
            raise ArrivalContractError(f"sample {ordinal} must be a mapping")
        sample_id = str(raw.get("sample_id", "")).strip()
        location_id = str(raw.get("location_id", "")).strip()
        split = str(raw.get("split", ""))
        if not sample_id or sample_id in seen_ids:
            raise ArrivalContractError("sample ids must be non-empty and unique")
        if not location_id:
            raise ArrivalContractError("location_id must be non-empty")
        if split not in _SPLITS:
            raise ArrivalContractError(
                "split must be calibration or confirmation"
            )
        seen_ids.add(sample_id)
        split_locations[split].add(location_id)
        distance = _finite_number(raw.get("distance_m"), "distance_m")
        yaw = _finite_number(raw.get("yaw_deg"), "yaw_deg")
        if distance < 0.0:
            raise ArrivalContractError("distance_m must be non-negative")
        goal_digest = _sha256(raw.get("goal_sha256"), "goal_sha256")
        frame_digest = _sha256(raw.get("frame_sha256"), "frame_sha256")
        arrived = distance <= radius
        counts[split]["samples"] += 1
        counts[split]["positive" if arrived else "negative"] += 1
        normalized.append({
            "sample_id": sample_id,
            "location_id": location_id,
            "split": split,
            "distance_m": distance,
            "yaw_deg": yaw,
            "goal_sha256": goal_digest,
            "frame_sha256": frame_digest,
            "arrived": arrived,
        })

    overlap = split_locations["calibration"].intersection(
        split_locations["confirmation"]
    )
    if overlap:
        raise ArrivalContractError(
            "calibration/confirmation location leakage: "
            + ",".join(sorted(overlap))
        )
    for split in sorted(_SPLITS):
        if not split_locations[split]:
            raise ArrivalContractError(f"{split} split has no locations")
        if counts[split]["positive"] == 0 or counts[split]["negative"] == 0:
            raise ArrivalContractError(
                f"{split} split requires positive and negative physical labels"
            )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "success_radius_m": radius,
        "counts": counts,
        "locations": {
            split: sorted(split_locations[split]) for split in sorted(_SPLITS)
        },
        "samples": normalized,
        "location_disjoint": True,
        "threshold_selection_reads_confirmation": False,
    }


__all__ = [
    "ArrivalContractError",
    "CALIBRATION_SCHEMA_VERSION",
    "LatchDecision",
    "ObservationDecision",
    "SCHEMA_VERSION",
    "ScaleFreeConvergenceLatch",
    "ScaleFreeConvergenceRule",
    "VISUAL_AUDIT_SCHEMA_VERSION",
    "evaluate_scale_free_observation",
    "validate_labeled_population",
]
