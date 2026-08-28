"""Pure consumed-development adapter for a certified initial bearing turn."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from MemNavData.cec_handoff_contract import verify_handoff_packet_envelope


@dataclass(frozen=True)
class CertifiedAlignmentTurn:
    forward: float
    left: float
    turn_rad: float
    packet_sha256: str


def bounded_turn_delta(
    remaining_rad: float,
    *,
    max_step_deg: float = 30.0,
    atol_rad: float = 1e-9,
) -> float:
    """Return one signed, zero-translation turn action.

    The caller must acquire a fresh observation after applying the returned
    delta and must replan after the remaining angle reaches zero.  Keeping
    that I/O contract in the evaluator makes this helper pure and testable.
    """

    try:
        remaining = float(remaining_rad)
        maximum = math.radians(float(max_step_deg))
        tolerance = float(atol_rad)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bounded turn parameters are not numeric") from exc
    if (not math.isfinite(remaining) or not math.isfinite(maximum)
            or not math.isfinite(tolerance)):
        raise ValueError("bounded turn parameters must be finite")
    if maximum <= 0.0 or maximum > math.pi or tolerance < 0.0:
        raise ValueError("bounded turn limits are invalid")
    if abs(remaining) <= tolerance:
        return 0.0
    return math.copysign(min(abs(remaining), maximum), remaining)


def validate_bounded_turn_trace(
    trace: Any,
    *,
    expected_turn_rad: float,
    max_step_deg: float = 30.0,
) -> dict[str, Any]:
    """Validate the deployable observation-turn receipt without Habitat."""

    if not isinstance(trace, list):
        raise ValueError("bounded turn trace must be a list")
    expected = float(expected_turn_rad)
    maximum = float(max_step_deg)
    if not math.isfinite(expected) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("bounded turn expectation is invalid")
    if abs(expected) > 1e-9 and not trace:
        raise ValueError("nonzero bounded turn has no actions")
    total = 0.0
    previous_after = None
    previous_frame = None
    packet_sha256 = None
    for index, row in enumerate(trace):
        if not isinstance(row, Mapping) or row.get("action_index") != index:
            raise ValueError("bounded turn action order changed")
        try:
            before = float(row["yaw_before_rad"])
            after = float(row["yaw_after_rad"])
            delta_deg = float(row["turn_delta_deg"])
            translation = float(row["translation_m"])
            frame_idx = int(row["memory_frame_idx"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("bounded turn receipt is incomplete") from exc
        if not all(map(math.isfinite, (before, after, delta_deg, translation))):
            raise ValueError("bounded turn receipt is nonfinite")
        if abs(delta_deg) > maximum + 1e-9 or abs(translation) > 1e-12:
            raise ValueError("bounded turn action limit changed")
        delta = math.radians(delta_deg)
        wrapped = math.atan2(math.sin(before + delta), math.cos(before + delta))
        if abs(math.atan2(math.sin(after - wrapped), math.cos(after - wrapped))) > 1e-8:
            raise ValueError("bounded turn yaw transition changed")
        if previous_after is not None:
            continuity = math.atan2(
                math.sin(before - previous_after), math.cos(before - previous_after))
            if abs(continuity) > 1e-8 or frame_idx <= previous_frame:
                raise ValueError("bounded turn observations are not sequential")
        image_sha = row.get("observation_jpg_sha256")
        packet = row.get("packet_sha256")
        if (not isinstance(image_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", image_sha) is None
                or not isinstance(packet, str)
                or re.fullmatch(r"[0-9a-f]{64}", packet) is None):
            raise ValueError("bounded turn hashes are invalid")
        if packet_sha256 is None:
            packet_sha256 = packet
        elif packet != packet_sha256:
            raise ValueError("bounded turn packet changed mid-action")
        if row.get("fresh_observation_required_before_next_action") is not True:
            raise ValueError("fresh-observation contract changed")
        total += delta
        previous_after = after
        previous_frame = frame_idx
    if abs(total - expected) > 1e-8:
        raise ValueError("bounded turn total differs from certified bearing")
    if trace and abs(float(trace[-1]["remaining_after_deg"])) > 1e-8:
        raise ValueError("bounded turn trace did not finish")
    return {
        "action_count": len(trace),
        "total_turn_deg": math.degrees(total),
        "max_abs_action_deg": max(
            (abs(float(row["turn_delta_deg"])) for row in trace), default=0.0),
        "zero_translation": True,
        "fresh_observation_receipts": len(trace),
        "packet_sha256": packet_sha256,
    }


def certified_alignment_turn(
    response: Mapping[str, Any],
) -> CertifiedAlignmentTurn | None:
    """Return the sealed robot-local turn authorized by one CEC response.

    A forced-reject arm may carry a shadow-accepted packet while preserving the
    controller's native ImageGoal action.  That is intentionally eligible in
    the mechanism test: the proof supplies direction, while authority still
    determines which goal image generated the unchanged local trajectory.
    """

    if not isinstance(response, Mapping):
        raise ValueError("controller response must be a mapping")
    if (response.get("cec_takeover") is not True
            and response.get("cec_shadow_takeover") is not True):
        return None
    packet = response.get("cec_handoff_packet")
    public = verify_handoff_packet_envelope(packet)
    if public.get("accepted") is not True:
        raise ValueError("bearing alignment requires an accepted CEC proof")
    direction = public.get("direction_vector")
    try:
        if (not isinstance(direction, list) or len(direction) != 2
                or any(isinstance(value, bool) for value in direction)):
            raise ValueError
        forward, left = float(direction[0]), float(direction[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bearing alignment direction is not numeric") from exc
    norm = math.hypot(forward, left)
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("bearing alignment direction is invalid")
    return CertifiedAlignmentTurn(
        forward=forward,
        left=left,
        turn_rad=math.atan2(left, forward),
        packet_sha256=str(packet["packet_sha256"]),
    )


__all__ = [
    "CertifiedAlignmentTurn",
    "bounded_turn_delta",
    "certified_alignment_turn",
    "validate_bounded_turn_trace",
]
