"""Continuous path-conditioned bearings for long-range Revisit control.

Canonical CEC sends the direction of one certified historical goal to NavDP.
That endpoint chord is sufficient for short, mostly convex returns, but it can
point through walls when the traversed history bends.  This module preserves
the same controller authority -- one fixed-radius PointGoal bearing -- while
changing only *where that bearing points*: a local point on the already
traversed, temporally ordered history curve.

The implementation deliberately contains no Habitat state, map search,
waypoint arrival gate, or alternative controller. One frozen path is built at
the goal switch from LingBot translations between the causal history tail and
the certified anchor. A local visual witness supplies a monotone temporal
route address; pose refines the state in that address's local geometry. The
reference is always one canonical NavDP horizon farther along the curve (or
the terminal endpoint when less remains), so accumulated global pose scale
never limits localization and the same control law applies to short and long
returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from MemNavData.revisit_bearing_adapter import VERIFIED_BEARING_RADIUS_M


PATH_FIELD_SCHEMA_VERSION = "episodic_path_field_v3_20260902"
NUMERICAL_EPS_M = 1e-6


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _pose9(value: Sequence[float], name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (9,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite pose9 vector")
    # The bearing conversion validates the quaternion more thoroughly.  Do it
    # once here as well, including on a completed path with no output bearing.
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    if quaternion_norm <= 1e-12:
        raise ValueError(f"{name} must contain a non-zero XYZW quaternion")
    return pose


def _scale_free_relative_xy(
    current_pose9: np.ndarray,
    goal_translation: np.ndarray,
) -> np.ndarray:
    """Express a world translation as NavDP ``[forward, left]``.

    Keep this small transform local so the path-field core stays renderer,
    Torch, and OpenCV independent.  It is algebraically identical to CEC's
    audited pose9 conversion: pose9 stores an XYZW camera-to-world quaternion
    and MemNav uses ``[camera_z, -camera_x]``.
    """

    x, y, z, w = current_pose9[3:7]
    norm = float(np.linalg.norm(current_pose9[3:7]))
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    camera_to_world = np.asarray([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)
    relative = camera_to_world.T @ (goal_translation - current_pose9[:3])
    return np.asarray([relative[2], -relative[0]], dtype=np.float64)


@dataclass(frozen=True)
class PathProjection:
    """One monotone projection of the current camera onto a frozen path."""

    progress_m: float
    cross_track_m: float
    segment: int
    point_raw_xz: tuple[float, float]


@dataclass(frozen=True)
class PathFieldGuidance:
    """Auditable output of one path-field readout."""

    schema_version: str
    progress_m: float
    remaining_m: float
    cross_track_m: float
    projection_segment: int
    reference_arc_m: float
    reference_raw_xz: tuple[float, float]
    terminal_within_horizon: bool
    complete: bool
    unit_bearing: tuple[float, float] | None
    controller_pointgoal: tuple[float, float] | None

    def audit_dict(self) -> dict:
        return {
            "episodic_path_field_schema_version": self.schema_version,
            "path_progress_m": self.progress_m,
            "path_remaining_m": self.remaining_m,
            "path_cross_track_m": self.cross_track_m,
            "path_projection_segment": self.projection_segment,
            "path_reference_arc_m": self.reference_arc_m,
            "path_reference_raw_xz": list(self.reference_raw_xz),
            "path_terminal_within_horizon": self.terminal_within_horizon,
            "path_complete": self.complete,
            "path_unit_bearing": (
                None if self.unit_bearing is None
                else list(self.unit_bearing)),
            "path_controller_pointgoal": (
                None if self.controller_pointgoal is None
                else list(self.controller_pointgoal)),
        }


@dataclass(frozen=True)
class FrozenEpisodicPath:
    """A temporally ordered current-to-goal curve in LingBot coordinates.

    ``raw_xz`` starts at the causal history tail and ends at the certified goal
    hypothesis.  ``metric_scale_m_per_raw`` is the immutable first-40 receipt
    already used by the dense monocular stream.  Scale determines *arc-length
    lookahead only*; translation magnitude never crosses the CEC authority
    boundary and NavDP still receives the canonical 2.5 m bearing.
    """

    raw_xz: np.ndarray
    cumulative_m: np.ndarray
    source_indices: tuple[int | None, ...]
    metric_scale_m_per_raw: float
    controller_horizon_m: float = VERIFIED_BEARING_RADIUS_M

    @classmethod
    def from_history(
        cls,
        translations: np.ndarray,
        *,
        start_index: int,
        anchor_index: int,
        metric_scale_m_per_raw: float,
        terminal_translation: Sequence[float] | None = None,
        controller_horizon_m: float = VERIFIED_BEARING_RADIUS_M,
    ) -> "FrozenEpisodicPath":
        """Freeze a path from the causal tail to one certified old anchor.

        ``translations`` is an ordered causal LingBot stream with shape
        ``[frames, >=3]``.  Both temporal directions are supported because a
        later multi-goal query may target either side of its route-start
        anchor.  If supplied, the PnP terminal translation is appended after
        the historical anchor, making final alignment part of the same curve.
        """

        translations = np.asarray(translations, dtype=np.float64)
        if translations.ndim != 2 or translations.shape[1] < 3:
            raise ValueError("translations must have shape [frames, >=3]")
        if not np.isfinite(translations).all():
            raise ValueError("translations must be finite")
        count = len(translations)
        if not 0 <= int(start_index) < count:
            raise ValueError("start_index is outside translations")
        if not 0 <= int(anchor_index) < count:
            raise ValueError("anchor_index is outside translations")
        scale = _finite_positive(
            metric_scale_m_per_raw, "metric_scale_m_per_raw")
        horizon = _finite_positive(
            controller_horizon_m, "controller_horizon_m")

        direction = 1 if anchor_index >= start_index else -1
        indices: list[int | None] = list(range(
            int(start_index), int(anchor_index) + direction, direction))
        points = [
            np.asarray(translations[index, (0, 2)], dtype=np.float64)
            for index in indices
        ]
        if terminal_translation is not None:
            terminal = np.asarray(terminal_translation, dtype=np.float64)
            if terminal.shape not in ((3,), (9,)) or not np.isfinite(
                    terminal).all():
                raise ValueError(
                    "terminal_translation must be a finite 3D or pose9 vector")
            points.append(np.asarray(terminal[(0, 2),], dtype=np.float64))
            indices.append(None)

        # Remove only numerical duplicates.  When a duplicate is the certified
        # terminal, retain its terminal identity without manufacturing an edge.
        clean_points: list[np.ndarray] = []
        clean_indices: list[int | None] = []
        for point, source in zip(points, indices):
            if clean_points and scale * float(np.linalg.norm(
                    point - clean_points[-1])) <= NUMERICAL_EPS_M:
                clean_points[-1] = point
                clean_indices[-1] = source
            else:
                clean_points.append(point)
                clean_indices.append(source)
        if len(clean_points) < 2:
            raise ValueError("episodic path has no non-zero spatial extent")

        raw_xz = np.stack(clean_points, axis=0)
        edge_m = scale * np.linalg.norm(np.diff(raw_xz, axis=0), axis=1)
        if not np.isfinite(edge_m).all() or np.any(edge_m <= 0.0):
            raise ValueError("episodic path contains an invalid edge")
        cumulative_m = np.concatenate([
            np.zeros(1, dtype=np.float64), np.cumsum(edge_m),
        ])
        return cls(
            raw_xz=raw_xz,
            cumulative_m=cumulative_m,
            source_indices=tuple(clean_indices),
            metric_scale_m_per_raw=scale,
            controller_horizon_m=horizon,
        )

    @property
    def total_length_m(self) -> float:
        return float(self.cumulative_m[-1])

    def _validate_progress(self, progress_m: float) -> float:
        result = float(progress_m)
        if not math.isfinite(result):
            raise ValueError("minimum progress must be finite")
        return float(np.clip(result, 0.0, self.total_length_m))

    def sample_raw_xz(self, arc_m: float) -> np.ndarray:
        """Linearly interpolate one point on the frozen path."""

        arc = float(arc_m)
        if not math.isfinite(arc):
            raise ValueError("arc_m must be finite")
        arc = float(np.clip(arc, 0.0, self.total_length_m))
        if arc >= self.total_length_m:
            return self.raw_xz[-1].copy()
        segment = int(np.searchsorted(
            self.cumulative_m, arc, side="right") - 1)
        segment = min(max(segment, 0), len(self.raw_xz) - 2)
        lo = float(self.cumulative_m[segment])
        hi = float(self.cumulative_m[segment + 1])
        alpha = (arc - lo) / (hi - lo)
        return ((1.0 - alpha) * self.raw_xz[segment]
                + alpha * self.raw_xz[segment + 1])

    def source_indices_in_control_window(
        self,
        *,
        minimum_progress_m: float,
    ) -> tuple[int, ...]:
        """Return taught-frame addresses in the next controller horizon.

        The window is a consequence of the frozen replanning contract: NavDP
        cannot execute farther than one 2.5 m residual before the next route
        observation.  It is not a short/long classifier or a memory-activation
        gate.  The same interval defines both visual localization and control.
        """

        lower = self._validate_progress(minimum_progress_m)
        upper = min(
            self.total_length_m, lower + self.controller_horizon_m)
        result = []
        for source, progress in zip(self.source_indices, self.cumulative_m):
            if source is None:
                continue
            value = float(progress)
            if (value + NUMERICAL_EPS_M >= lower
                    and value <= upper + NUMERICAL_EPS_M):
                result.append(int(source))
        return tuple(result)

    def source_indices_at_or_after_progress(
        self,
        *,
        minimum_progress_m: float,
    ) -> tuple[int, ...]:
        """Return the still-unvisited visual addresses on the taught route.

        Route localization and route control deliberately have different
        horizons. Appearance may identify any later address on the one
        already-authorized temporal route; the controller still receives only
        a 2.5 m local lookahead. This prevents accumulated monocular path scale
        from becoming an artificial localization gate.
        """

        lower = self._validate_progress(minimum_progress_m)
        return tuple(
            int(source)
            for source, progress in zip(
                self.source_indices, self.cumulative_m)
            if (source is not None
                and float(progress) + NUMERICAL_EPS_M >= lower)
        )

    def progress_for_source_index(self, source_index: int) -> float:
        """Return the immutable route coordinate of one taught RGB frame."""

        target = int(source_index)
        matches = [
            float(progress)
            for source, progress in zip(
                self.source_indices, self.cumulative_m)
            if source is not None and int(source) == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"source index {target} is not a unique route address")
        return matches[0]

    def project_monotone(
        self,
        current_translation: Sequence[float],
        *,
        minimum_progress_m: float,
    ) -> PathProjection:
        """Project locally without a discrete waypoint/arrival decision.

        The admissible interval advances by at most the same 2.5 m horizon
        already granted to NavDP.  This prevents a self-intersecting history
        from jumping to a spatially nearby but temporally remote corridor.  It
        is a continuous progress constraint, not a success or distance gate.
        """

        current = np.asarray(current_translation, dtype=np.float64)
        if current.shape not in ((3,), (9,)) or not np.isfinite(current).all():
            raise ValueError("current_translation must be finite 3D or pose9")
        current_metric = current[(0, 2),] * self.metric_scale_m_per_raw
        route_metric = self.raw_xz * self.metric_scale_m_per_raw
        lower = self._validate_progress(minimum_progress_m)
        upper = min(
            self.total_length_m, lower + self.controller_horizon_m)

        best: tuple[float, float, int, np.ndarray] | None = None
        for segment in range(len(route_metric) - 1):
            segment_lo = float(self.cumulative_m[segment])
            segment_hi = float(self.cumulative_m[segment + 1])
            allowed_lo = max(lower, segment_lo)
            allowed_hi = min(upper, segment_hi)
            if allowed_hi + NUMERICAL_EPS_M < allowed_lo:
                continue
            edge = route_metric[segment + 1] - route_metric[segment]
            edge_sq = float(edge @ edge)
            if edge_sq <= 0.0:
                continue
            t_lo = (allowed_lo - segment_lo) / (segment_hi - segment_lo)
            t_hi = (allowed_hi - segment_lo) / (segment_hi - segment_lo)
            unconstrained = float(
                (current_metric - route_metric[segment]) @ edge / edge_sq)
            fraction = float(np.clip(unconstrained, t_lo, t_hi))
            point_metric = route_metric[segment] + fraction * edge
            distance = float(np.linalg.norm(current_metric - point_metric))
            progress = segment_lo + fraction * (segment_hi - segment_lo)
            # Prefer the earlier temporal point on an exact distance tie.  This
            # makes self-intersections conservative and deterministic.
            candidate = (distance, progress, segment, point_metric)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("no segment overlaps the monotone progress interval")
        distance, progress, segment, point_metric = best
        return PathProjection(
            progress_m=float(progress),
            cross_track_m=float(distance),
            segment=int(segment),
            point_raw_xz=tuple(
                (point_metric / self.metric_scale_m_per_raw).tolist()),
        )

    def guidance(
        self,
        current_pose9: Sequence[float],
        *,
        minimum_progress_m: float,
    ) -> PathFieldGuidance:
        """Return one fixed-authority local bearing and its full receipt."""

        current = _pose9(current_pose9, "current_pose9")
        projection = self.project_monotone(
            current, minimum_progress_m=minimum_progress_m)
        remaining = max(0.0, self.total_length_m - projection.progress_m)
        complete = remaining <= NUMERICAL_EPS_M
        reference_arc = min(
            self.total_length_m,
            projection.progress_m + self.controller_horizon_m,
        )
        reference = self.sample_raw_xz(reference_arc)
        unit: tuple[float, float] | None = None
        controller: tuple[float, float] | None = None
        if not complete:
            reference_translation = current[:3].copy()
            reference_translation[0] = reference[0]
            reference_translation[2] = reference[1]
            vector = _scale_free_relative_xy(
                current, reference_translation)
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise RuntimeError(
                    "non-complete path produced a zero local bearing")
            vector /= norm
            unit = (float(vector[0]), float(vector[1]))
            controller = tuple(
                float(value * self.controller_horizon_m) for value in vector)
        return PathFieldGuidance(
            schema_version=PATH_FIELD_SCHEMA_VERSION,
            progress_m=projection.progress_m,
            remaining_m=remaining,
            cross_track_m=projection.cross_track_m,
            projection_segment=projection.segment,
            reference_arc_m=reference_arc,
            reference_raw_xz=(float(reference[0]), float(reference[1])),
            terminal_within_horizon=(
                remaining <= self.controller_horizon_m),
            complete=complete,
            unit_bearing=unit,
            controller_pointgoal=controller,
        )


__all__ = [
    "FrozenEpisodicPath",
    "NUMERICAL_EPS_M",
    "PATH_FIELD_SCHEMA_VERSION",
    "PathFieldGuidance",
    "PathProjection",
]
