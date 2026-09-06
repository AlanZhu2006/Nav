"""Fail-closed controller interface for verified Revisit memory evidence.

The memory system is allowed to decide *whether* a revisit match is verified
and to estimate its camera-relative direction.  It is deliberately not
allowed to choose a local controller or expose an uncalibrated metric distance
to the canonical policy.  The canonical ``verified_bearing_v1`` interface
therefore projects every non-zero verified PointGoal onto one frozen radius.

``verified_bounded_metric_v1`` is a deliberately separate experimental arm.
It reuses the already-frozen first-40 monocular scale receipt, but can only
*shrink* that radius.  It never expands memory authority beyond the canonical
2.5 m residual and is not the default paper method.

``verified_metric_v1`` is the corresponding full-distance challenger.  It
uses the same receipt to metricize the verified LingBot translation norm and
exposes that distance up to frozen NavDP's native 10 m PointGoal support.  It
exists only for paired attribution and is not a certified metric claim.

``verified_navdp_support_projection_v1`` is a long-range interface
attribution arm.  NavDP erases a negative forward PointGoal component before
the decoder sees it.  The adapter therefore maps a verified rear-half-plane
bearing to the nearest unit-bearing boundary of NavDP's supported front
half-plane, preserving the fixed 2.5 m token norm.  It never rejects, changes
the certificate, reads navigation outcomes, or executes actions itself.

This module is intentionally independent of Habitat, Torch, Flask, and NavDP.
It is the auditable boundary between a direction source and a controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


REVISIT_ADAPTER_SCHEMA_VERSION = 2
REVISIT_ADAPTER_MODES = (
    "legacy_metric",
    "navdp_front_support_v1",
    "raw_fixed_bearing_v1",
    "verified_bearing_v1",
    "verified_navdp_support_projection_v1",
    "verified_bounded_metric_v1",
    "verified_metric_v1",
)
FIXED_BEARING_MODES = frozenset({
    "raw_fixed_bearing_v1",
    "verified_bearing_v1",
    "verified_navdp_support_projection_v1",
})
BOUNDED_METRIC_MODES = frozenset({"verified_bounded_metric_v1"})
FULL_METRIC_MODES = frozenset({"verified_metric_v1"})
SCALED_METRIC_MODES = BOUNDED_METRIC_MODES | FULL_METRIC_MODES
POINTGOAL_UNITS = (
    "metric_m",
    "lingbot_raw_direction_only",
    "pi3x_current_camera_direction_only",
)

# B0 froze this value before its paired rollout: the episode-balanced median
# first-active memory radius was 2.513 m and was rounded to 2.5 m.  It is a
# semantic constant, not an evaluation-time hyperparameter.
VERIFIED_BEARING_RADIUS_M = 2.5
# NavDP's frozen point-goal preprocessor clips each supported coordinate to
# ten metres.  A radial cap at that same native envelope preserves bearing
# instead of relying on a later component-wise clip.  The current HM3D audit
# lies below this value, so saturation must remain zero in the formal probe.
NAVDP_POINTGOAL_RADIUS_MAX_M = 10.0
ZERO_BEARING_EPS = 1e-12


@dataclass(frozen=True)
class RevisitAdapterDecision:
    """One source-to-controller decision with complete audit provenance."""

    mode: str
    source: str
    takeover: bool
    reason: str
    controller_contract: str
    raw_pointgoal: tuple[float, float] | None
    raw_pointgoal_units: str
    unit_bearing: tuple[float, float] | None
    controller_pointgoal: tuple[float, float] | None
    raw_pointgoal_norm: float | None
    raw_distance_m: float | None
    controller_distance_m: float | None
    metric_scale_m_per_raw: float | None = None
    controller_radius_cap_m: float | None = None
    support_projection_applied: bool | None = None
    controller_unit_bearing: tuple[float, float] | None = None

    def audit_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible fields for every planning step."""

        audit = {
            "revisit_adapter_schema_version": (
                REVISIT_ADAPTER_SCHEMA_VERSION),
            "revisit_adapter_mode": self.mode,
            "revisit_adapter_source": self.source,
            "revisit_adapter_takeover": self.takeover,
            "revisit_adapter_reason": self.reason,
            "revisit_adapter_controller_contract": (
                self.controller_contract),
            "memory_unbounded_pointgoal": (
                list(self.raw_pointgoal)
                if self.raw_pointgoal is not None else None),
            "memory_unbounded_pointgoal_units": self.raw_pointgoal_units,
            "memory_unbounded_pointgoal_norm": self.raw_pointgoal_norm,
            "memory_bearing_unit": (
                list(self.unit_bearing)
                if self.unit_bearing is not None else None),
            "memory_controller_pointgoal": (
                list(self.controller_pointgoal)
                if self.controller_pointgoal is not None else None),
            "memory_unbounded_pointgoal_distance_m": self.raw_distance_m,
            "memory_controller_pointgoal_distance_m": (
                self.controller_distance_m),
            "memory_pointgoal_fixed_radius_m": (
                VERIFIED_BEARING_RADIUS_M
                if self.mode in FIXED_BEARING_MODES else None),
        }
        if self.mode in SCALED_METRIC_MODES:
            # Keep the frozen canonical adapter schema unchanged.  This
            # additive, arm-specific receipt makes the challenger auditable
            # without silently rewriting old verified_bearing_v1 artifacts.
            audit.update({
                "memory_metric_scale_m_per_raw": (
                    self.metric_scale_m_per_raw),
                "memory_pointgoal_radius_cap_m": (
                    self.controller_radius_cap_m),
            })
            if self.mode in BOUNDED_METRIC_MODES:
                audit["bounded_metric_adapter_schema_version"] = 1
            else:
                audit["full_metric_adapter_schema_version"] = 1
        if self.mode == "verified_navdp_support_projection_v1":
            audit.update({
                "navdp_support_projection_schema_version": 1,
                "memory_navdp_support_projection_applied": bool(
                    self.support_projection_applied),
                "memory_controller_bearing_unit": (
                    None if self.controller_unit_bearing is None
                    else list(self.controller_unit_bearing)),
                "memory_source_bearing_heading_deg": (
                    None if self.unit_bearing is None else math.degrees(
                        math.atan2(
                            self.unit_bearing[1], self.unit_bearing[0]))),
                "memory_controller_bearing_heading_deg": (
                    None if self.controller_unit_bearing is None
                    else math.degrees(math.atan2(
                        self.controller_unit_bearing[1],
                        self.controller_unit_bearing[0]))),
            })
        return audit


def _finite_pointgoal(
    pointgoal: Sequence[float] | None,
) -> tuple[float, float] | None:
    if pointgoal is None:
        return None
    try:
        if len(pointgoal) != 2:
            return None
        if isinstance(pointgoal[0], bool) or isinstance(pointgoal[1], bool):
            return None
        point = (float(pointgoal[0]), float(pointgoal[1]))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    return point if all(math.isfinite(value) for value in point) else None


def project_unit_bearing_to_navdp_support(
    unit_bearing: Sequence[float],
) -> tuple[tuple[float, float], bool]:
    """Project a unit bearing onto NavDP's representable front half-plane.

    NavDP's frozen preprocessing clips the forward coordinate to ``[0, 10]``.
    Leaving a rear bearing unchanged therefore makes its token norm collapse
    continuously to zero near 180 degrees.  The nearest unit vector in the
    supported closed half-plane is the original vector when forward is
    non-negative, otherwise the signed lateral boundary.  Exact 180 degrees
    has two equivalent projections; positive-left is the deterministic tie.
    """

    point = _finite_pointgoal(unit_bearing)
    if point is None:
        raise ValueError("unit_bearing must contain two finite values")
    norm = math.hypot(*point)
    if norm <= ZERO_BEARING_EPS:
        raise ValueError("unit_bearing must be non-zero")
    forward, left = point[0] / norm, point[1] / norm
    if forward >= 0.0:
        return (forward, left), False
    lateral = -1.0 if left < 0.0 else 1.0
    return (0.0, lateral), True


def adapt_revisit_pointgoal(
    *,
    mode: str,
    router_active: bool,
    pointgoal: Sequence[float] | None,
    source: str = "geometry_memory",
    pointgoal_units: str = "metric_m",
    metric_scale_m_per_raw: float | None = None,
) -> RevisitAdapterDecision:
    """Convert verified memory evidence into a controller request.

    ``legacy_metric`` exactly preserves the old metric PointGoal interface and
    exists only for paired attribution/backward compatibility.

    ``navdp_front_support_v1`` preserves the legacy metric vector only when
    its forward component is inside frozen NavDP's PointGoal preprocessing
    support.  NavDP clips a negative forward component to zero, so sending a
    behind-agent target would silently destroy source information; that case
    fails closed to native ImageGoal control.

    ``verified_bearing_v1`` is the canonical source-agnostic interface: a
    verified non-zero vector contributes only its unit bearing, projected onto
    the frozen 2.5 m local radius.  Missing, malformed, inactive, and zero
    evidence abstains to native ImageGoal control.

    A certified relocalizer may pass ``lingbot_raw_direction_only`` vectors to
    ``verified_bearing_v1``.  Their arbitrary norm is audited but never called
    metres and never reaches the controller; only the normalized direction
    survives the fixed-radius projection.

    ``raw_fixed_bearing_v1`` is an ablation, not a verified method.  It applies
    the identical fixed-radius projection to an always-on raw metric proposal,
    so experiments can isolate controller input scale from certificate and
    abstention effects.

    ``verified_bounded_metric_v1`` is a full-mono challenger.  It accepts only
    a verified ``lingbot_raw_direction_only`` vector and one positive frozen
    first-40 scale.  The vector norm is metricized, then capped at the same
    2.5 m radius used by ``verified_bearing_v1``.  Missing or invalid scale
    fails closed to native ImageGoal navigation.

    ``verified_metric_v1`` shares the same evidence and failure contract but
    exposes the complete metricized norm while it remains inside NavDP's
    frozen 10 m PointGoal support.  The support cap is an interface bound, not
    a tuned radius, and is recorded on every decision.
    """

    if mode not in REVISIT_ADAPTER_MODES:
        raise ValueError(f"unsupported revisit adapter mode {mode!r}")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if pointgoal_units not in POINTGOAL_UNITS:
        raise ValueError(f"unsupported PointGoal units {pointgoal_units!r}")

    raw = _finite_pointgoal(pointgoal)
    raw_norm = math.hypot(*raw) if raw is not None else None
    raw_distance = raw_norm if pointgoal_units == "metric_m" else None
    unit = None
    if raw_norm is not None and raw_norm > ZERO_BEARING_EPS:
        unit = (raw[0] / raw_norm, raw[1] / raw_norm)

    if not router_active:
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=False,
            reason="router_inactive",
            controller_contract="native_imagegoal",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=unit,
            controller_pointgoal=None,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=raw_distance,
            controller_distance_m=None,
        )
    if raw is None:
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=False,
            reason=("missing_pointgoal" if pointgoal is None
                    else "invalid_pointgoal"),
            controller_contract="native_imagegoal",
            raw_pointgoal=None,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=None,
            controller_pointgoal=None,
            raw_pointgoal_norm=None,
            raw_distance_m=None,
            controller_distance_m=None,
        )

    if mode == "legacy_metric":
        if pointgoal_units != "metric_m":
            return RevisitAdapterDecision(
                mode=mode, source=source, takeover=False,
                reason="metric_units_required",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=unit,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=None,
                controller_distance_m=None,
            )
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=True,
            reason="legacy_metric_pointgoal",
            controller_contract="configured_revisit_controller",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=unit,
            controller_pointgoal=raw,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=raw_distance,
            controller_distance_m=raw_distance,
        )

    if mode == "navdp_front_support_v1":
        if pointgoal_units != "metric_m":
            return RevisitAdapterDecision(
                mode=mode, source=source, takeover=False,
                reason="metric_units_required",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=unit,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=None,
                controller_distance_m=None,
            )
        if unit is None:
            return RevisitAdapterDecision(
                mode=mode,
                source=source,
                takeover=False,
                reason="zero_pointgoal",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=None,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=raw_distance,
                controller_distance_m=None,
            )
        if raw[0] < 0.0:
            return RevisitAdapterDecision(
                mode=mode,
                source=source,
                takeover=False,
                reason="pointgoal_behind_navdp_support",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=unit,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=raw_distance,
                controller_distance_m=None,
            )
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=True,
            reason="pointgoal_inside_navdp_support",
            controller_contract="mixed_imagegoal_pointgoal",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=unit,
            controller_pointgoal=raw,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=raw_distance,
            controller_distance_m=raw_distance,
        )

    if mode == "raw_fixed_bearing_v1" and pointgoal_units != "metric_m":
        return RevisitAdapterDecision(
            mode=mode, source=source, takeover=False,
            reason="metric_units_required",
            controller_contract="native_imagegoal",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=unit,
            controller_pointgoal=None,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=None,
            controller_distance_m=None,
        )

    if mode in SCALED_METRIC_MODES:
        if pointgoal_units != "lingbot_raw_direction_only":
            return RevisitAdapterDecision(
                mode=mode, source=source, takeover=False,
                reason="lingbot_scale_free_units_required",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=unit,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=None,
                controller_distance_m=None,
            )
        if metric_scale_m_per_raw is None:
            return RevisitAdapterDecision(
                mode=mode, source=source, takeover=False,
                reason="metric_scale_unavailable",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=unit,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=None,
                controller_distance_m=None,
            )
        try:
            if isinstance(metric_scale_m_per_raw, bool):
                raise ValueError("boolean scale")
            metric_scale = float(metric_scale_m_per_raw)
        except (TypeError, ValueError, OverflowError):
            metric_scale = float("nan")
        if not math.isfinite(metric_scale) or metric_scale <= 0.0:
            return RevisitAdapterDecision(
                mode=mode, source=source, takeover=False,
                reason="invalid_metric_scale",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=unit,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=None,
                controller_distance_m=None,
            )
        if unit is None:
            return RevisitAdapterDecision(
                mode=mode, source=source, takeover=False,
                reason="zero_bearing",
                controller_contract="native_imagegoal",
                raw_pointgoal=raw,
                raw_pointgoal_units=pointgoal_units,
                unit_bearing=None,
                controller_pointgoal=None,
                raw_pointgoal_norm=raw_norm,
                raw_distance_m=None,
                controller_distance_m=None,
                metric_scale_m_per_raw=metric_scale,
                controller_radius_cap_m=(
                    VERIFIED_BEARING_RADIUS_M
                    if mode in BOUNDED_METRIC_MODES
                    else NAVDP_POINTGOAL_RADIUS_MAX_M),
            )
        metric_distance = metric_scale * raw_norm
        controller_radius_cap = (
            VERIFIED_BEARING_RADIUS_M
            if mode in BOUNDED_METRIC_MODES
            else NAVDP_POINTGOAL_RADIUS_MAX_M
        )
        controller_distance = min(metric_distance, controller_radius_cap)
        controller_pointgoal = (
            unit[0] * controller_distance,
            unit[1] * controller_distance,
        )
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=True,
            reason=(
                "verified_bounded_metric_bearing"
                if mode in BOUNDED_METRIC_MODES
                else "verified_metric_distance"),
            controller_contract="mixed_imagegoal_pointgoal",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=unit,
            controller_pointgoal=controller_pointgoal,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=metric_distance,
            controller_distance_m=controller_distance,
            metric_scale_m_per_raw=metric_scale,
            controller_radius_cap_m=controller_radius_cap,
        )

    if unit is None:
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=False,
            reason="zero_bearing",
            controller_contract="native_imagegoal",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=None,
            controller_pointgoal=None,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=raw_distance,
            controller_distance_m=None,
        )

    if mode == "verified_navdp_support_projection_v1":
        controller_unit, projected = project_unit_bearing_to_navdp_support(
            unit)
        controller_pointgoal = (
            controller_unit[0] * VERIFIED_BEARING_RADIUS_M,
            controller_unit[1] * VERIFIED_BEARING_RADIUS_M,
        )
        return RevisitAdapterDecision(
            mode=mode,
            source=source,
            takeover=True,
            reason=(
                "verified_bearing_projected_to_navdp_support"
                if projected else "verified_bearing_inside_navdp_support"),
            controller_contract="mixed_imagegoal_pointgoal",
            raw_pointgoal=raw,
            raw_pointgoal_units=pointgoal_units,
            unit_bearing=unit,
            controller_pointgoal=controller_pointgoal,
            raw_pointgoal_norm=raw_norm,
            raw_distance_m=raw_distance,
            controller_distance_m=VERIFIED_BEARING_RADIUS_M,
            support_projection_applied=projected,
            controller_unit_bearing=controller_unit,
        )

    controller_pointgoal = (
        unit[0] * VERIFIED_BEARING_RADIUS_M,
        unit[1] * VERIFIED_BEARING_RADIUS_M,
    )
    if mode == "raw_fixed_bearing_v1":
        reason = "raw_uncertified_fixed_bearing"
    elif pointgoal_units in (
            "lingbot_raw_direction_only",
            "pi3x_current_camera_direction_only"):
        reason = "verified_scale_free_bearing"
    else:
        reason = "verified_geometry_bearing"
    return RevisitAdapterDecision(
        mode=mode,
        source=source,
        takeover=True,
        reason=reason,
        controller_contract="mixed_imagegoal_pointgoal",
        raw_pointgoal=raw,
        raw_pointgoal_units=pointgoal_units,
        unit_bearing=unit,
        controller_pointgoal=controller_pointgoal,
        raw_pointgoal_norm=raw_norm,
        raw_distance_m=raw_distance,
        controller_distance_m=VERIFIED_BEARING_RADIUS_M,
    )


def validate_revisit_adapter_configuration(
    *,
    mode: str,
    server_backend: str,
    revisit_controller: str,
    router_is_automatic_geometry: bool,
    router_is_certified_relocalization: bool = False,
) -> None:
    """Reject configurations that would corrupt the canonical method claim."""

    if mode not in REVISIT_ADAPTER_MODES:
        raise ValueError(f"unsupported revisit adapter mode {mode!r}")
    if mode == "legacy_metric":
        return
    if server_backend != "hybrid_pose" or revisit_controller != "navdp_mixed":
        raise ValueError(
            f"{mode} requires hybrid_pose with the existing navdp_mixed "
            "controller")
    if mode in ("navdp_front_support_v1", "raw_fixed_bearing_v1"):
        return
    if not router_is_automatic_geometry:
        raise ValueError(
            f"{mode} requires an automatic, geometry-verified router; "
            "the phase-oracle route is not deployable")


__all__ = [
    "REVISIT_ADAPTER_MODES",
    "REVISIT_ADAPTER_SCHEMA_VERSION",
    "FIXED_BEARING_MODES",
    "BOUNDED_METRIC_MODES",
    "FULL_METRIC_MODES",
    "SCALED_METRIC_MODES",
    "POINTGOAL_UNITS",
    "RevisitAdapterDecision",
    "VERIFIED_BEARING_RADIUS_M",
    "NAVDP_POINTGOAL_RADIUS_MAX_M",
    "adapt_revisit_pointgoal",
    "project_unit_bearing_to_navdp_support",
    "validate_revisit_adapter_configuration",
]
