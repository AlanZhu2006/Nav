"""Pure contract for a controller-independent CEC handoff packet.

CEC is an authorization layer, not a controller.  A portability experiment
therefore needs a controller-independent unit of treatment: one accepted CEC
proof bound to the exact causal history, current observation, target image and
certified history anchor that produced it.  This module creates and verifies
that single-use artifact without importing Torch, Habitat, Flask or ROS.

The packet is intentionally valid for one high-level controller decision only.
After an executor changes the physical state, reusing the old robot-centric
bearing would no longer be causal.  A later decision requires a new proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from MemNavData.controller_portability_contract import (
    CEC_FIXED_RADIUS_M,
    CEC_POINTGOAL_UNITS,
    CecProjection,
    cec_proof_sha256,
    project_cec_proof,
)


HANDOFF_PACKET_SCHEMA = "cec_certified_handoff_packet_v1_20260827"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PUBLIC_PROOF_FIELDS = (
    "certified_relocalization_schema_version",
    "frame_idx",
    "ok",
    "accepted",
    "reason",
    "selected_anchor",
    "selected_anchor_image_sha256",
    "direction_vector",
    "pointgoal_units",
    "certificate",
)
FORBIDDEN_KEYS = frozenset({
    "analysis_role",
    "role",
    "goal_role",
    "query_role",
    "is_revisit",
    "is_novel",
    "oracle_pose",
    "gt_pose",
    "ground_truth_pose",
    "habitat_pose",
    "evaluation_gt_arrived",
    "evaluation_gt_goal_distance_m",
    "evaluation_gt_bearing_error_deg",
})


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("handoff payload must be finite JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes) or not value:
        raise ValueError("handoff image payloads must be non-empty bytes")
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _reject_privileged_recursive(value: Any, path: str = "packet") -> None:
    if isinstance(value, Mapping):
        leaked = sorted(str(key) for key in value if key in FORBIDDEN_KEYS)
        if leaked:
            raise ValueError(
                f"{path} contains privileged fields: {', '.join(leaked)}")
        for key, child in value.items():
            _reject_privileged_recursive(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_privileged_recursive(child, f"{path}[{index}]")


def public_cec_proof(proof: Mapping[str, Any]) -> dict[str, Any]:
    """Select the deployment-visible CEC fields used by every executor."""

    if not isinstance(proof, Mapping):
        raise ValueError("CEC proof must be a mapping")
    _reject_privileged_recursive(proof, "proof")
    public = {field: proof.get(field) for field in PUBLIC_PROOF_FIELDS}
    # Round-trip through canonical JSON to detach mutable nested structures and
    # reject NaN/Infinity before the proof is sealed.
    return json.loads(_canonical_json(public).decode("utf-8"))


def _packet_body(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items()
            if key != "packet_sha256"}


def build_handoff_packet(
    proof: Mapping[str, Any],
    *,
    current_rgb: bytes,
    goal_rgb: bytes,
    anchor_jpeg: bytes,
    causal_history_sha256: str,
) -> dict[str, Any]:
    """Seal one accepted CEC decision as a single-use handoff packet."""

    public = public_cec_proof(proof)
    if public.get("accepted") is not True or public.get("ok") is not True:
        raise ValueError("only an accepted, valid CEC proof can issue a handoff")
    certificate = public.get("certificate")
    if (not isinstance(certificate, Mapping)
            or certificate.get("accepted") is not True):
        raise ValueError("handoff requires an accepted atomic certificate")
    if public.get("pointgoal_units") != CEC_POINTGOAL_UNITS:
        raise ValueError("handoff proof has the wrong direction units")
    anchor_sha256 = _sha256_bytes(anchor_jpeg)
    if public.get("selected_anchor_image_sha256") != anchor_sha256:
        raise ValueError("certified anchor bytes do not match the proof")
    _sha256_text(causal_history_sha256, "causal_history_sha256")

    packet = {
        "schema_version": HANDOFF_PACKET_SCHEMA,
        "single_use": True,
        "decision_scope": "one_high_level_controller_decision",
        "public_proof": public,
        "proof_sha256": cec_proof_sha256(public),
        "current_rgb_sha256": _sha256_bytes(current_rgb),
        "goal_rgb_sha256": _sha256_bytes(goal_rgb),
        "anchor_jpeg_sha256": anchor_sha256,
        "causal_history_sha256": causal_history_sha256,
        "fixed_radius_m": CEC_FIXED_RADIUS_M,
        "pointgoal_units": CEC_POINTGOAL_UNITS,
        "role_label_visible": False,
        "metric_depth_sensor_consumed": False,
    }
    _reject_privileged_recursive(packet)
    packet["packet_sha256"] = hashlib.sha256(
        _canonical_json(packet)).hexdigest()
    return packet


def verify_handoff_packet(
    packet: Mapping[str, Any],
    *,
    current_rgb: bytes,
    goal_rgb: bytes,
    anchor_jpeg: bytes,
    causal_history_sha256: str,
) -> dict[str, Any]:
    """Verify all causal/input bindings and return the public CEC proof."""

    public = verify_handoff_packet_envelope(packet)
    if packet.get("current_rgb_sha256") != _sha256_bytes(current_rgb):
        raise ValueError("handoff current RGB binding mismatch")
    if packet.get("goal_rgb_sha256") != _sha256_bytes(goal_rgb):
        raise ValueError("handoff goal RGB binding mismatch")
    if packet.get("anchor_jpeg_sha256") != _sha256_bytes(anchor_jpeg):
        raise ValueError("handoff anchor JPEG binding mismatch")
    if (packet.get("causal_history_sha256")
            != _sha256_text(causal_history_sha256,
                            "causal_history_sha256")):
        raise ValueError("handoff causal-history binding mismatch")
    if public.get("selected_anchor_image_sha256") != _sha256_bytes(anchor_jpeg):
        raise ValueError("handoff public proof lost its anchor binding")
    return public


def verify_handoff_packet_envelope(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a packet's self-contained envelope without image bytes.

    This is the independent-verifier entry point.  It validates the sealed
    packet, proof and digest relationships from receipts alone.  The online
    router additionally calls :func:`verify_handoff_packet` with the actual
    RGB/goal/anchor bytes before granting control.
    """

    if not isinstance(packet, Mapping):
        raise ValueError("handoff packet must be a mapping")
    _reject_privileged_recursive(packet)
    if packet.get("schema_version") != HANDOFF_PACKET_SCHEMA:
        raise ValueError("handoff packet schema changed")
    if packet.get("single_use") is not True:
        raise ValueError("handoff packet must be single-use")
    if packet.get("decision_scope") != "one_high_level_controller_decision":
        raise ValueError("handoff packet decision scope changed")
    if packet.get("role_label_visible") is not False:
        raise ValueError("handoff packet exposed a runtime role label")
    if packet.get("metric_depth_sensor_consumed") is not False:
        raise ValueError("handoff packet sensor contract changed")
    try:
        radius = float(packet.get("fixed_radius_m"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("handoff radius is invalid") from exc
    if (not math.isfinite(radius)
            or not math.isclose(radius, CEC_FIXED_RADIUS_M,
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("handoff radius changed")
    if packet.get("pointgoal_units") != CEC_POINTGOAL_UNITS:
        raise ValueError("handoff pointgoal units changed")

    expected_packet_sha = hashlib.sha256(
        _canonical_json(_packet_body(packet))).hexdigest()
    if packet.get("packet_sha256") != expected_packet_sha:
        raise ValueError("handoff packet digest mismatch")

    public = public_cec_proof(packet.get("public_proof"))
    if packet.get("proof_sha256") != cec_proof_sha256(public):
        raise ValueError("handoff proof digest mismatch")
    _sha256_text(packet.get("current_rgb_sha256"), "current_rgb_sha256")
    _sha256_text(packet.get("goal_rgb_sha256"), "goal_rgb_sha256")
    anchor_sha256 = _sha256_text(
        packet.get("anchor_jpeg_sha256"), "anchor_jpeg_sha256")
    _sha256_text(
        packet.get("causal_history_sha256"), "causal_history_sha256")
    if public.get("selected_anchor_image_sha256") != anchor_sha256:
        raise ValueError("handoff public proof lost its anchor binding")
    return public


def project_handoff_packet(
    controller: str,
    packet: Mapping[str, Any],
    *,
    current_rgb: bytes,
    goal_rgb: bytes,
    anchor_jpeg: bytes,
    causal_history_sha256: str,
) -> CecProjection:
    """Verify one packet and project it into a controller-native interface."""

    proof = verify_handoff_packet(
        packet,
        current_rgb=current_rgb,
        goal_rgb=goal_rgb,
        anchor_jpeg=anchor_jpeg,
        causal_history_sha256=causal_history_sha256,
    )
    projection = project_cec_proof(
        controller, proof, anchor_jpeg=anchor_jpeg)
    if projection.proof_sha256 != packet["proof_sha256"]:
        raise ValueError("controller projection changed the sealed proof")
    return projection


__all__ = [
    "HANDOFF_PACKET_SCHEMA",
    "build_handoff_packet",
    "project_handoff_packet",
    "public_cec_proof",
    "verify_handoff_packet",
    "verify_handoff_packet_envelope",
]
