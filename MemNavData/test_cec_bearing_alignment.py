import hashlib
import math

import pytest

from MemNavData.cec_bearing_alignment import (
    bounded_turn_delta,
    certified_alignment_turn,
    validate_bounded_turn_trace,
)
from MemNavData.cec_handoff_contract import build_handoff_packet


def packet(direction=(0.0, 1.0)) -> dict:
    anchor = b"anchor"
    proof = {
        "certified_relocalization_schema_version": 3,
        "frame_idx": 12,
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 4,
        "selected_anchor_image_sha256": hashlib.sha256(anchor).hexdigest(),
        "direction_vector": list(direction),
        "pointgoal_units": "lingbot_raw_direction_only",
        "certificate": {"accepted": True},
    }
    return build_handoff_packet(
        proof,
        current_rgb=b"current",
        goal_rgb=b"goal",
        anchor_jpeg=anchor,
        causal_history_sha256="0" * 64,
    )


def test_no_authority_returns_none() -> None:
    assert certified_alignment_turn({
        "cec_takeover": False,
        "cec_shadow_takeover": False,
    }) is None


def test_shadow_packet_authorizes_left_turn() -> None:
    result = certified_alignment_turn({
        "cec_takeover": False,
        "cec_shadow_takeover": True,
        "cec_handoff_packet": packet(),
    })
    assert result is not None
    assert math.isclose(result.turn_rad, math.pi / 2.0)
    assert result.forward == 0.0
    assert result.left == 1.0


def test_tampered_direction_fails_closed() -> None:
    value = packet()
    value["public_proof"]["direction_vector"] = [-1.0, 0.0]
    with pytest.raises(ValueError, match="digest"):
        certified_alignment_turn({
            "cec_takeover": True,
            "cec_handoff_packet": value,
        })


def test_bounded_turn_delta_clips_both_signs() -> None:
    assert math.isclose(
        bounded_turn_delta(math.radians(95.0)), math.radians(30.0))
    assert math.isclose(
        bounded_turn_delta(math.radians(-47.0)), math.radians(-30.0))
    assert math.isclose(
        bounded_turn_delta(math.radians(12.0)), math.radians(12.0))


def test_bounded_turn_delta_stops_at_tolerance() -> None:
    assert bounded_turn_delta(1e-12) == 0.0
    with pytest.raises(ValueError, match="limits"):
        bounded_turn_delta(1.0, max_step_deg=0.0)


def bounded_trace() -> list[dict]:
    packet_sha = "a" * 64
    image_shas = ["b" * 64, "c" * 64, "d" * 64]
    values = []
    before = 0.1
    for index, delta in enumerate((30.0, 30.0, 15.0)):
        after = math.atan2(
            math.sin(before + math.radians(delta)),
            math.cos(before + math.radians(delta)),
        )
        values.append({
            "action_index": index,
            "packet_sha256": packet_sha,
            "observation_jpg_sha256": image_shas[index],
            "memory_frame_idx": 40 + index,
            "yaw_before_rad": before,
            "yaw_after_rad": after,
            "turn_delta_deg": delta,
            "remaining_after_deg": 45.0 - 30.0 * index if index < 2 else 0.0,
            "translation_m": 0.0,
            "fresh_observation_required_before_next_action": True,
        })
        before = after
    return values


def test_bounded_turn_trace_binds_total_and_observations() -> None:
    result = validate_bounded_turn_trace(
        bounded_trace(), expected_turn_rad=math.radians(75.0))
    assert result["action_count"] == 3
    assert result["fresh_observation_receipts"] == 3
    assert math.isclose(result["max_abs_action_deg"], 30.0)


def test_bounded_turn_trace_rejects_missing_observation_progress() -> None:
    trace = bounded_trace()
    trace[1]["memory_frame_idx"] = trace[0]["memory_frame_idx"]
    with pytest.raises(ValueError, match="sequential"):
        validate_bounded_turn_trace(
            trace, expected_turn_rad=math.radians(75.0))
