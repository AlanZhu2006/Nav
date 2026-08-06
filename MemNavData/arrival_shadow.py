"""Privileged-input-free arrival diagnostics for virtual loop closure.

The detector deliberately does not control the rollout.  It consumes only
signals already produced by MemNav/NavDP and reports two increasingly strict
events:

``pose_ready``
    A stable, completed memory-graph route whose raw goal pose remains nearby.
``strict_ready``
    ``pose_ready`` plus NavDP's own critic-based stop evidence.

Habitat goal coordinates never enter this module.  The evaluator may compare
the shadow events with privileged distance afterwards, strictly for scoring.
"""

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ArrivalShadowConfig:
    """Conservative provisional thresholds for diagnostic collection only."""

    window_plans: int = 3
    distance_m: float = 0.75
    max_distance_mad_m: float = 0.20
    max_distance_growth_m: float = 0.15

    def __post_init__(self):
        if self.window_plans < 2:
            raise ValueError("arrival shadow window must contain at least 2 plans")
        for name in (
                "distance_m", "max_distance_mad_m",
                "max_distance_growth_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _finite_distance(raw_pose) -> Optional[float]:
    if raw_pose is None:
        return None
    pose = np.asarray(raw_pose, dtype=np.float64)
    if pose.shape != (2,) or not np.isfinite(pose).all():
        return None
    return float(np.linalg.norm(pose))


def _finite_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


class ArrivalShadowDetector:
    """Stateful, fail-closed arrival observer that never receives GT pose."""

    def __init__(self, config: ArrivalShadowConfig):
        self.config = config
        self._window = deque(maxlen=config.window_plans)
        self._plan_count = 0
        self._pose_trigger_count = 0
        self._strict_trigger_count = 0
        self._first_pose_step = None
        self._first_strict_step = None

    @staticmethod
    def _anchor(signal: Mapping):
        value = signal.get("router_selected_anchor")
        if value is None:
            value = signal.get("anchor")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def update(self, signal: Mapping, *, step: Optional[int] = None) -> dict:
        """Observe one planning response and return serializable diagnostics."""
        self._plan_count += 1
        sample = {
            "distance_m": _finite_distance(signal.get("goal_aux_pose")),
            "anchor": self._anchor(signal),
            "router_active": signal.get("router_active") is True,
            "graph_enabled": signal.get("graph_subgoal_enabled") is True,
            "route_complete": signal.get("graph_subgoal_complete") is True,
            "critic_max": _finite_float(signal.get("navdp_critic_max")),
            "critic_stop": signal.get("navdp_stop_evidence") is True,
        }
        self._window.append(sample)

        full = len(self._window) == self.config.window_plans
        distances = [item["distance_m"] for item in self._window]
        distances_valid = full and all(value is not None for value in distances)
        median = mad = growth = None
        if distances_valid:
            values = np.asarray(distances, dtype=np.float64)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            growth = float(values[-1] - values[0])

        anchors = [item["anchor"] for item in self._window]
        anchor_stable = bool(
            full and anchors[0] is not None
            and all(anchor == anchors[0] for anchor in anchors)
        )
        router_stable = bool(
            full and all(item["router_active"] for item in self._window)
        )
        # Route completion is allowed to transition on the current plan.  The
        # temporal evidence applies to localization/distance stability; asking
        # the route to have been complete for the whole window would delay a
        # valid stop after the final graph node has just been consumed.
        graph_enabled = bool(full and sample["graph_enabled"])
        route_complete = bool(full and sample["route_complete"])
        distance_stable = bool(
            distances_valid
            and median <= self.config.distance_m
            and mad <= self.config.max_distance_mad_m
            and growth <= self.config.max_distance_growth_m
        )
        pose_ready = bool(
            anchor_stable and router_stable and graph_enabled
            and route_complete and distance_stable
        )
        critic_available = sample["critic_max"] is not None
        strict_ready = bool(pose_ready and sample["critic_stop"])

        reasons = []
        if not full:
            reasons.append("warming_up")
        else:
            if not router_stable:
                reasons.append("router_unstable")
            if not anchor_stable:
                reasons.append("anchor_unstable")
            if not graph_enabled:
                reasons.append("graph_disabled")
            if not route_complete:
                reasons.append("route_incomplete")
            if not distances_valid:
                reasons.append("goal_pose_missing")
            elif not distance_stable:
                reasons.append("goal_distance_unstable_or_far")
        if pose_ready and not sample["critic_stop"]:
            reasons.append(
                "critic_not_stopped" if critic_available
                else "critic_unavailable")

        if pose_ready:
            self._pose_trigger_count += 1
            if self._first_pose_step is None:
                self._first_pose_step = step
        if strict_ready:
            self._strict_trigger_count += 1
            if self._first_strict_step is None:
                self._first_strict_step = step

        return {
            "arrival_shadow_goal_distance_m": sample["distance_m"],
            "arrival_shadow_distance_median_m": median,
            "arrival_shadow_distance_mad_m": mad,
            "arrival_shadow_distance_growth_m": growth,
            "arrival_shadow_anchor": sample["anchor"],
            "arrival_shadow_anchor_stable": anchor_stable,
            "arrival_shadow_router_stable": router_stable,
            "arrival_shadow_graph_enabled": graph_enabled,
            "arrival_shadow_route_complete": route_complete,
            "arrival_shadow_pose_ready": pose_ready,
            "arrival_shadow_critic_available": critic_available,
            "arrival_shadow_critic_max": sample["critic_max"],
            "arrival_shadow_critic_stop": sample["critic_stop"],
            "arrival_shadow_strict_ready": strict_ready,
            "arrival_shadow_reason": "ready" if not reasons else ",".join(reasons),
            "arrival_shadow_window_count": len(self._window),
        }

    def summary(self) -> dict:
        return {
            "arrival_shadow_plan_count": self._plan_count,
            "arrival_shadow_pose_trigger_count": self._pose_trigger_count,
            "arrival_shadow_strict_trigger_count": self._strict_trigger_count,
            "arrival_shadow_pose_triggered": self._pose_trigger_count > 0,
            "arrival_shadow_strict_triggered": self._strict_trigger_count > 0,
            "arrival_shadow_first_pose_step": self._first_pose_step,
            "arrival_shadow_first_strict_step": self._first_strict_step,
        }
