"""Pure proof and executor contract for an initial route-tangent alignment.

This module intentionally imports only the Python standard library so the
Habitat evaluator can validate the control handoff without importing Torch or
the LingBot runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


ROUTE_ALIGNMENT_PACKET_SCHEMA_VERSION = (
    "proof_bound_route_alignment_packet_v1_20260904"
)
ROUTE_ALIGNMENT_EXECUTOR_SOURCE_PREFIX = (
    "proof_bound_atomic_route_alignment_yaw_v1:"
)
FIRST_CERTIFIABLE_ANCHOR = 8
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ROUTE_ALIGNMENT_FORBIDDEN_KEYS = frozenset({
    "analysis_role", "role", "goal_role", "query_role", "is_revisit",
    "is_novel", "oracle_pose", "gt_pose", "ground_truth_pose",
    "habitat_pose", "evaluation_gt_arrived",
    "evaluation_gt_goal_distance_m", "evaluation_gt_bearing_error_deg",
})


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("route-alignment payload must be finite JSON") from exc


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_privileged_fields(value: Any, path: str = "packet") -> None:
    if isinstance(value, Mapping):
        leaked = sorted(str(key) for key in value
                        if key in ROUTE_ALIGNMENT_FORBIDDEN_KEYS)
        if leaked:
            raise ValueError(
                f"{path} contains privileged fields: {', '.join(leaked)}")
        for key, child in value.items():
            _reject_privileged_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_privileged_fields(child, f"{path}[{index}]")


def _packet_body(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items()
            if key != "packet_sha256"}


def build_route_alignment_packet(
    *,
    authority_proof: Mapping[str, Any],
    current_frame: int,
    goal_start_frame: int,
    target_anchor: int,
    current_rgb_sha256: str,
    goal_image_sha256: str,
    anchor_image_sha256: str,
    history_edge_receipt_sha256: str,
    unit_bearing: Iterable[float],
    tangent_baseline_m: float,
    controller_radius_m: float,
) -> dict[str, Any]:
    """Seal the one route tangent allowed to trigger an atomic alignment."""

    if not isinstance(authority_proof, Mapping):
        raise ValueError("route alignment requires an authority proof")
    _reject_privileged_fields(authority_proof, "authority_proof")
    proof = json.loads(_canonical_json(dict(authority_proof)).decode("utf-8"))
    target = _strict_int(target_anchor, "target_anchor")
    current = _strict_int(current_frame, "current_frame")
    goal_start = _strict_int(goal_start_frame, "goal_start_frame")
    if target < FIRST_CERTIFIABLE_ANCHOR or not target < goal_start <= current:
        raise ValueError("route alignment violates the causal frame order")
    if (proof.get("ok") is not True or proof.get("accepted") is not True
            or int(proof.get("selected_anchor", -1)) != target):
        raise ValueError("route alignment lacks accepted target authority")
    certificate = proof.get("certificate")
    if not isinstance(certificate, Mapping) or certificate.get(
            "accepted") is not True:
        raise ValueError("route alignment lacks an accepted certificate")
    anchor_sha = _sha256_text(anchor_image_sha256, "anchor_image_sha256")
    if proof.get("selected_anchor_image_sha256") != anchor_sha:
        raise ValueError("route alignment anchor binding changed")
    try:
        direction = tuple(float(value) for value in unit_bearing)
        baseline = float(tangent_baseline_m)
        radius = float(controller_radius_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("route alignment geometry is not numeric") from exc
    if (len(direction) != 2 or not all(math.isfinite(value)
                                       for value in direction)
            or not math.isclose(math.hypot(*direction), 1.0,
                                abs_tol=1e-6)
            or not math.isfinite(baseline) or baseline <= 0.0
            or not math.isfinite(radius) or radius <= 0.0):
        raise ValueError("route alignment geometry is invalid")
    turn_rad = math.atan2(direction[1], direction[0])
    packet = {
        "schema_version": ROUTE_ALIGNMENT_PACKET_SCHEMA_VERSION,
        "action_scope": "one_initial_atomic_bounded_yaw_alignment",
        "guidance_mode": "monocular_route_tangent",
        "authority_proof": proof,
        "authority_proof_sha256": hashlib.sha256(
            _canonical_json(proof)).hexdigest(),
        "current_frame": current,
        "goal_start_frame": goal_start,
        "target_anchor": target,
        "current_rgb_sha256": _sha256_text(
            current_rgb_sha256, "current_rgb_sha256"),
        "goal_image_sha256": _sha256_text(
            goal_image_sha256, "goal_image_sha256"),
        "anchor_image_sha256": anchor_sha,
        "history_edge_receipt_sha256": _sha256_text(
            history_edge_receipt_sha256,
            "history_edge_receipt_sha256"),
        "unit_bearing": [float(value) for value in direction],
        "required_turn_rad": float(turn_rad),
        "tangent_baseline_m": baseline,
        "controller_radius_m": radius,
        "role_label_visible": False,
        "simulator_pose_consumed": False,
        "metric_depth_sensor_consumed": False,
    }
    _reject_privileged_fields(packet)
    packet["packet_sha256"] = hashlib.sha256(
        _canonical_json(packet)).hexdigest()
    return packet


def verify_route_alignment_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the self-contained route-alignment authorization envelope."""

    if not isinstance(packet, Mapping):
        raise ValueError("route alignment packet must be a mapping")
    _reject_privileged_fields(packet)
    if packet.get("schema_version") != ROUTE_ALIGNMENT_PACKET_SCHEMA_VERSION:
        raise ValueError("route alignment packet schema changed")
    if packet.get("action_scope") != (
            "one_initial_atomic_bounded_yaw_alignment"):
        raise ValueError("route alignment action scope changed")
    if packet.get("guidance_mode") != "monocular_route_tangent":
        raise ValueError("route alignment guidance mode changed")
    if (packet.get("role_label_visible") is not False
            or packet.get("simulator_pose_consumed") is not False
            or packet.get("metric_depth_sensor_consumed") is not False):
        raise ValueError("route alignment sensor/role boundary changed")
    expected_packet_sha = hashlib.sha256(
        _canonical_json(_packet_body(packet))).hexdigest()
    if packet.get("packet_sha256") != expected_packet_sha:
        raise ValueError("route alignment packet digest mismatch")
    proof = packet.get("authority_proof")
    if not isinstance(proof, Mapping):
        raise ValueError("route alignment authority proof is missing")
    if packet.get("authority_proof_sha256") != hashlib.sha256(
            _canonical_json(proof)).hexdigest():
        raise ValueError("route alignment authority proof digest mismatch")
    current = _strict_int(packet.get("current_frame"), "current_frame")
    goal_start = _strict_int(
        packet.get("goal_start_frame"), "goal_start_frame")
    target = _strict_int(packet.get("target_anchor"), "target_anchor")
    if target < FIRST_CERTIFIABLE_ANCHOR or not target < goal_start <= current:
        raise ValueError("route alignment causal frame order changed")
    anchor_sha = _sha256_text(
        packet.get("anchor_image_sha256"), "anchor_image_sha256")
    for key in ("current_rgb_sha256", "goal_image_sha256",
                "history_edge_receipt_sha256"):
        _sha256_text(packet.get(key), key)
    if (proof.get("ok") is not True or proof.get("accepted") is not True
            or int(proof.get("selected_anchor", -1)) != target
            or proof.get("selected_anchor_image_sha256") != anchor_sha):
        raise ValueError("route alignment authority binding changed")
    certificate = proof.get("certificate")
    if not isinstance(certificate, Mapping) or certificate.get(
            "accepted") is not True:
        raise ValueError("route alignment certificate changed")
    try:
        direction = tuple(float(value) for value in packet["unit_bearing"])
        turn = float(packet["required_turn_rad"])
        baseline = float(packet["tangent_baseline_m"])
        radius = float(packet["controller_radius_m"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("route alignment geometry is incomplete") from exc
    direction_turn = math.atan2(direction[1], direction[0])
    turn_error = math.atan2(
        math.sin(turn - direction_turn), math.cos(turn - direction_turn))
    if (len(direction) != 2 or not all(math.isfinite(value)
                                       for value in direction)
            or not math.isclose(math.hypot(*direction), 1.0,
                                abs_tol=1e-6)
            or not math.isfinite(turn) or abs(turn_error) > 1e-9
            or not math.isfinite(baseline) or baseline <= 0.0
            or not math.isfinite(radius) or radius <= 0.0):
        raise ValueError("route alignment geometry changed")
    return dict(packet)


def route_alignment_executor_source(packet_sha256: str) -> str:
    return ROUTE_ALIGNMENT_EXECUTOR_SOURCE_PREFIX + _sha256_text(
        packet_sha256, "route_alignment_packet_sha256")


__all__ = [
    "ROUTE_ALIGNMENT_EXECUTOR_SOURCE_PREFIX",
    "ROUTE_ALIGNMENT_PACKET_SCHEMA_VERSION",
    "build_route_alignment_packet",
    "route_alignment_executor_source",
    "verify_route_alignment_packet",
]
