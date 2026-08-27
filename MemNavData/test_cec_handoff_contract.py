import copy
import hashlib

import pytest

from MemNavData.cec_handoff_contract import (
    HANDOFF_PACKET_SCHEMA,
    build_handoff_packet,
    project_handoff_packet,
    verify_handoff_packet,
    verify_handoff_packet_envelope,
)
from MemNavData.controller_portability_contract import CEC_POINTGOAL_UNITS


CURRENT = b"current-query-rgb"
GOAL = b"target-imagegoal-rgb"
ANCHOR = b"certified-history-anchor"
HISTORY_SHA = hashlib.sha256(b"ordered-causal-history").hexdigest()


def accepted_proof(**updates):
    proof = {
        "certified_relocalization_schema_version": 3,
        "frame_idx": 212,
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 135,
        "selected_anchor_image_sha256": hashlib.sha256(ANCHOR).hexdigest(),
        "direction_vector": [-1.245, 0.128],
        "pointgoal_units": CEC_POINTGOAL_UNITS,
        "certificate": {
            "schema_version": 3,
            "accepted": True,
            "checks": {"minimum_inliers": True},
        },
    }
    proof.update(updates)
    return proof


def packet():
    return build_handoff_packet(
        accepted_proof(), current_rgb=CURRENT, goal_rgb=GOAL,
        anchor_jpeg=ANCHOR, causal_history_sha256=HISTORY_SHA)


def verify(value):
    return verify_handoff_packet(
        value, current_rgb=CURRENT, goal_rgb=GOAL,
        anchor_jpeg=ANCHOR, causal_history_sha256=HISTORY_SHA)


def test_packet_binds_proof_history_and_three_images():
    value = packet()
    assert value["schema_version"] == HANDOFF_PACKET_SCHEMA
    assert value["single_use"] is True
    assert verify_handoff_packet_envelope(value)["selected_anchor"] == 135
    assert verify(value)["selected_anchor"] == 135


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("current_rgb_sha256", "0" * 64, "packet digest"),
        ("goal_rgb_sha256", "0" * 64, "packet digest"),
        ("anchor_jpeg_sha256", "0" * 64, "packet digest"),
        ("causal_history_sha256", "0" * 64, "packet digest"),
        ("proof_sha256", "0" * 64, "packet digest"),
        ("single_use", False, "single-use"),
    ],
)
def test_packet_tampering_fails_closed(field, replacement, message):
    value = copy.deepcopy(packet())
    value[field] = replacement
    with pytest.raises(ValueError, match=message):
        verify(value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"current_rgb": b"other"}, "current RGB"),
        ({"goal_rgb": b"other"}, "goal RGB"),
        ({"anchor_jpeg": b"other"}, "anchor JPEG"),
        ({"causal_history_sha256": "1" * 64}, "causal-history"),
    ],
)
def test_replay_against_different_inputs_fails_closed(kwargs, message):
    inputs = dict(
        current_rgb=CURRENT, goal_rgb=GOAL, anchor_jpeg=ANCHOR,
        causal_history_sha256=HISTORY_SHA)
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        verify_handoff_packet(packet(), **inputs)


def test_rejection_cannot_be_mislabeled_as_a_handoff():
    with pytest.raises(ValueError, match="only an accepted"):
        build_handoff_packet(
            accepted_proof(accepted=False), current_rgb=CURRENT,
            goal_rgb=GOAL, anchor_jpeg=ANCHOR,
            causal_history_sha256=HISTORY_SHA)


@pytest.mark.parametrize("leak", [
    {"query_role": "revisit"},
    {"certificate": {"accepted": True, "gt_pose": [0, 0, 0]}},
])
def test_privileged_fields_are_rejected_recursively(leak):
    proof = accepted_proof()
    proof.update(leak)
    with pytest.raises(ValueError, match="privileged"):
        build_handoff_packet(
            proof, current_rgb=CURRENT, goal_rgb=GOAL,
            anchor_jpeg=ANCHOR, causal_history_sha256=HISTORY_SHA)


@pytest.mark.parametrize(
    ("controller", "adapter"),
    [
        ("navdp", "bearing_mixedgoal"),
        ("iplanner", "bearing_pointgoal"),
        ("viplanner", "bearing_pointgoal"),
        ("vint", "verified_anchor_imagegoal"),
        ("gnm", "verified_anchor_imagegoal"),
        ("nomad", "verified_anchor_imagegoal"),
    ],
)
def test_one_packet_projects_to_declared_controller_interfaces(
        controller, adapter):
    projection = project_handoff_packet(
        controller, packet(), current_rgb=CURRENT, goal_rgb=GOAL,
        anchor_jpeg=ANCHOR, causal_history_sha256=HISTORY_SHA)
    assert projection.takeover is True
    assert projection.adapter == adapter
    if adapter.startswith("bearing_"):
        assert projection.payload["goal_x"]
        assert projection.payload["goal_y"]
    else:
        assert projection.payload["cec_anchor_sha256"] == hashlib.sha256(
            ANCHOR).hexdigest()
