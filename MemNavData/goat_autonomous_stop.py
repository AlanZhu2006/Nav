#!/usr/bin/env python3
"""Observable terminal-view search for autonomous GOAT ImageGoal stopping.

The navigation policy is allowed to *propose* that it may have arrived, but a
zero/short trajectory is never interpreted as ``SUBTASK_STOP``.  Instead this
module builds a bounded, closed-loop camera-view search.  Every visited view is
checked by the independent current-RGB-to-ImageGoal arrival certificate.  A
STOP is emitted only through the already frozen
``goat_certified_arrival_contract``.

There are two deployment paths:

* a certified Revisit may provide a PnP-derived target-view yaw/pitch.  Search
  first moves toward that view, then falls back to a complete yaw sweep;
* an unsupported/Novel goal has no defensible global view direction, so it
  performs only the complete yaw sweep.

When no view verifies arrival, the schedule returns to its original camera
orientation before handing control back to NavDP.  The module never consumes
simulator pose, depth, distance, a Novel/Revisit label, or an official GOAT
success metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import enum
from typing import Mapping, Optional, Sequence, Tuple

from MemNavData.goat_certified_arrival_contract import (
    ArrivalEvidence,
    decide_subtask_stop,
)
from MemNavData.goat_terminal_alignment import (
    TerminalAlignmentConfig,
    TerminalAlignmentDisposition,
    plan_terminal_alignment,
)


SCHEMA_VERSION = "goat_autonomous_visual_stop_v1_20260818"


class SearchDisposition(str, enum.Enum):
    MOTION = "motion"
    STOP = "stop"
    REPLAN = "replan"


@dataclass(frozen=True)
class TerminalSearchConfig:
    """Fixed action budget and discrete camera-search geometry."""

    yaw_quantum_deg: float = 30.0
    full_yaw_steps: int = 12
    maximum_directed_actions: int = 8
    include_pitch_alignment: bool = True

    def validate(self) -> None:
        if self.yaw_quantum_deg <= 0.0:
            raise ValueError("yaw quantum must be positive")
        if self.full_yaw_steps < 1:
            raise ValueError("full yaw sweep must contain at least one step")
        swept = self.yaw_quantum_deg * self.full_yaw_steps
        if abs(swept - 360.0) > 1e-9:
            raise ValueError("full yaw sweep must close exactly at 360 degrees")
        if self.maximum_directed_actions < 1:
            raise ValueError("directed action budget must be positive")


@dataclass(frozen=True)
class ScheduledViewAction:
    action: str
    phase: str
    probe_after_action: bool = True


@dataclass(frozen=True)
class SearchDecision:
    disposition: SearchDisposition
    action: Optional[str]
    phase: str
    reason: str
    stop_decision: Optional[Mapping[str, object]]


def _inverse_action(action: str) -> str:
    inverse = {
        "turn_left": "turn_right",
        "turn_right": "turn_left",
        "look_up": "look_down",
        "look_down": "look_up",
    }
    if action not in inverse:
        raise ValueError(f"terminal search cannot invert action {action!r}")
    return inverse[action]


def _directed_actions(
    yaw_right_deg: Optional[float],
    pitch_up_deg: Optional[float],
    config: TerminalSearchConfig,
) -> Tuple[str, ...]:
    if yaw_right_deg is None:
        return ()
    pitch = float(pitch_up_deg or 0.0) if config.include_pitch_alignment else 0.0
    alignment = plan_terminal_alignment(
        float(yaw_right_deg),
        pitch,
        certificate_accepted=True,
        near_goal=True,
        config=TerminalAlignmentConfig(
            yaw_quantum_deg=config.yaw_quantum_deg,
            pitch_quantum_deg=30.0,
            yaw_tolerance_deg=config.yaw_quantum_deg / 2.0,
            pitch_tolerance_deg=15.0,
            max_abs_yaw_deg=180.0,
            max_abs_pitch_deg=60.0,
            max_actions=config.maximum_directed_actions,
        ),
    )
    if alignment.disposition in (
        TerminalAlignmentDisposition.MOTION,
        TerminalAlignmentDisposition.ALREADY_ALIGNED,
    ):
        return tuple(alignment.actions)
    return ()


def build_terminal_view_schedule(
    *,
    revisit_yaw_right_deg: Optional[float] = None,
    revisit_pitch_up_deg: Optional[float] = None,
    config: TerminalSearchConfig = TerminalSearchConfig(),
) -> Tuple[ScheduledViewAction, ...]:
    """Build an outcome-independent search that restores pose on rejection.

    The initial camera view is always probed before executing this schedule.
    A complete rightward yaw sweep returns to the post-directed orientation.
    The inverse directed suffix then restores the pre-search orientation.
    """

    config.validate()
    directed = _directed_actions(
        revisit_yaw_right_deg, revisit_pitch_up_deg, config)
    schedule = [
        ScheduledViewAction(action=action, phase="directed_alignment")
        for action in directed
    ]
    schedule.extend(
        ScheduledViewAction(action="turn_right", phase="full_yaw_sweep")
        for _ in range(config.full_yaw_steps)
    )
    schedule.extend(
        ScheduledViewAction(
            action=_inverse_action(action), phase="restore_orientation")
        for action in reversed(directed)
    )
    return tuple(schedule)


class AutonomousVisualStopSearch:
    """Small deterministic state machine driven only by visual evidence."""

    def __init__(
        self,
        *,
        revisit_yaw_right_deg: Optional[float] = None,
        revisit_pitch_up_deg: Optional[float] = None,
        config: TerminalSearchConfig = TerminalSearchConfig(),
    ) -> None:
        self.config = config
        self.schedule = build_terminal_view_schedule(
            revisit_yaw_right_deg=revisit_yaw_right_deg,
            revisit_pitch_up_deg=revisit_pitch_up_deg,
            config=config,
        )
        self.cursor = 0
        self.probe_count = 0
        self.motion_count = 0
        self.finished = False
        self.authorized_stop = False

    def observe(self, evidence: ArrivalEvidence) -> SearchDecision:
        """Consume one current-view certificate and choose STOP/motion/replan."""

        if self.finished:
            raise RuntimeError("terminal visual search is already finished")
        self.probe_count += 1
        stop = decide_subtask_stop(evidence)
        if bool(stop["authorized_subtask_stop"]):
            self.finished = True
            self.authorized_stop = True
            return SearchDecision(
                disposition=SearchDisposition.STOP,
                action=None,
                phase="verified_arrival",
                reason=str(stop["reason"]),
                stop_decision=stop,
            )
        if self.cursor >= len(self.schedule):
            self.finished = True
            return SearchDecision(
                disposition=SearchDisposition.REPLAN,
                action=None,
                phase="search_exhausted",
                reason="no_camera_view_certified_arrival",
                stop_decision=stop,
            )
        item = self.schedule[self.cursor]
        self.cursor += 1
        self.motion_count += 1
        return SearchDecision(
            disposition=SearchDisposition.MOTION,
            action=item.action,
            phase=item.phase,
            reason="continue_bounded_terminal_view_search",
            stop_decision=stop,
        )

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(self.config),
            "schedule": [asdict(item) for item in self.schedule],
            "cursor": int(self.cursor),
            "probe_count": int(self.probe_count),
            "motion_count": int(self.motion_count),
            "finished": bool(self.finished),
            "authorized_stop": bool(self.authorized_stop),
        }


def arrival_evidence_from_payload(payload: Mapping[str, object]) -> ArrivalEvidence:
    """Project an arrival-service payload onto the frozen STOP contract."""

    return ArrivalEvidence(
        native_zero_proposal=True,
        stream_frame_count=int(payload.get("frame_count", 0)),
        certificate_accepted=bool(payload.get("certificate_accepted", False)),
        predicted_distance_m=payload.get("predicted_distance_m"),
        metric_scale_available=bool(payload.get("metric_scale_available", False)),
    )


def schedule_net_discrete_rotation(
    schedule: Sequence[ScheduledViewAction],
) -> Tuple[int, int]:
    """Return net yaw and pitch quanta for static closure audits."""

    yaw = 0
    pitch = 0
    for item in schedule:
        if item.action == "turn_right":
            yaw += 1
        elif item.action == "turn_left":
            yaw -= 1
        elif item.action == "look_up":
            pitch += 1
        elif item.action == "look_down":
            pitch -= 1
        else:
            raise ValueError(f"unexpected terminal action {item.action!r}")
    return yaw, pitch
