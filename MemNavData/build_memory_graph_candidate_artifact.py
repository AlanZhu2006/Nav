#!/usr/bin/env python3
"""Assemble deployment Phase-B inference into memory-graph candidates.

This is a pure artifact transformer.  It never imports a model, reads an
image, opens a LingBot cache, or computes a teacher label.  The mandatory
input is a fully validated ``nlsr_phase_b_deployment_inference_v1`` artifact.
It produces one direct point-goal candidate for every accepted localization.

An independently produced reverse-route artifact is optional.  When supplied,
its first short subgoal is appended as a second candidate.  It can never
replace or reorder the direct candidate:

* direct is always candidate 0, priority 0, and ``graph_hops == 0``;
* reverse, if available, is candidate 1 with one or more graph hops; and
* a rejected/no-match state has no memory candidate.

The output is intentionally a small intermediate schema.  A later unified
precollection join can map these fields into ``novel_candidate_set_v2`` while
keeping rollout/teacher labels neutral.  Keeping this boundary separate makes
it impossible for optional graph routing to change the localization decision
or for an absent route to delete a valid direct point goal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

try:
    from MemNavData.phase_b_deployment_inference_contract import (
        InferencePins,
        METRIC_SCALE_SOURCE,
        POSE_CONVENTION,
        PRIVILEGED_INPUT_POLICY,
        ValidatedPhaseBInference,
        canonical_json_bytes,
        load_pinned_canonical_json,
        sha256_bytes,
        sha256_file,
        validate_phase_b_deployment_inference,
    )
except ModuleNotFoundError:  # direct script invocation
    from phase_b_deployment_inference_contract import (  # type: ignore
        InferencePins,
        METRIC_SCALE_SOURCE,
        POSE_CONVENTION,
        PRIVILEGED_INPUT_POLICY,
        ValidatedPhaseBInference,
        canonical_json_bytes,
        load_pinned_canonical_json,
        sha256_bytes,
        sha256_file,
        validate_phase_b_deployment_inference,
    )


SCHEMA_VERSION = "nlsr_memory_graph_candidate_artifact_v1"
PURPOSE = (
    "deployment-only direct-first memory candidates from pinned Phase-B "
    "inference; optional reverse graph is append-only and carries no labels"
)
REVERSE_ROUTE_SCHEMA_VERSION = "nlsr_phase_b_reverse_route_artifact_v1"
REVERSE_ROUTE_PURPOSE = (
    "deployment-only reverse routes from routed causal LingBot x-z poses and "
    "external causal metric scale; no teacher, GT pose, navmesh, or Pathfinder"
)
REVERSE_ROUTE_SEMANTICS = "reverse_recorded_pose_chain_resampled_v1"
FIRST_SUBGOAL_SEMANTICS = "current_lingbot_xz_pose_to_first_reverse_node_v1"
DIRECT_POLICY = "always_emit_selected_direct_first_priority0_hops0_v1"
REVERSE_POLICY = "append_only_priority1_never_replace_direct_v1"
POSE_P90_SEMANTICS = (
    "direct_endpoint_diagonal_gaussian_chi2_2d_p90_"
    "reverse_route_uncertainty_unavailable_v1"
)
DEPTH_CONFIDENCE_SEMANTICS = "lingbot_vggt_nonnegative_model_score_v1"
CHI_SQUARE_2D_P90 = 4.605170185988092
ZERO_SHA = "0" * 64

REVERSE_TOP_KEYS = frozenset(
    {"schema_version", "purpose", "provenance", "configuration", "records", "summary"}
)
REVERSE_PROVENANCE_KEYS = frozenset(
    {
        "input_manifest_sha256",
        "input_inference_artifact_sha256",
        "flow_route_artifact_sha256",
        "causal_scale_artifact_sha256",
        "producer_source_sha256",
        "configuration_sha256",
        "pose_convention",
        "privileged_input_policy",
    }
)
REVERSE_CONFIGURATION_KEYS = frozenset(
    {"spacing_m", "route_semantics", "first_subgoal_semantics"}
)
REVERSE_RECORD_KEYS = frozenset(
    {
        "sample_id",
        "causal_prefix_sha256",
        "decision_frame",
        "cam_pose_prefix_sha256",
        "anchor_frame_index",
        "start_frame_index",
        "metric_scale_m_per_raw",
        "metric_scale_source",
        "current_native_position_xz_raw",
        "current_native_yaw_rad",
        "route_nodes",
        "first_subgoal_forward_left_m",
        "graph_path_m",
    }
)
REVERSE_NODE_KEYS = frozenset(
    {"frame_index", "native_position_xz_raw", "segment_path_m"}
)
REVERSE_SUMMARY_KEYS = frozenset(
    {
        "record_count",
        "route_node_count",
        "future_frames_consumed",
        "privileged_inputs_consumed",
    }
)

OUTPUT_TOP_KEYS = frozenset(
    {"schema_version", "purpose", "provenance", "configuration", "records", "summary"}
)
OUTPUT_PROVENANCE_KEYS = frozenset(
    {
        "input_manifest_sha256",
        "input_inference_artifact_sha256",
        "input_reverse_route_artifact_sha256",
        "phase_b_checkpoint_sha256",
        "flow_route_artifact_sha256",
        "causal_scale_artifact_sha256",
        "inference_producer_source_sha256",
        "reverse_route_producer_source_sha256",
        "assembler_source_sha256",
        "inference_contract_source_sha256",
        "configuration_sha256",
        "pose_convention",
        "privileged_input_policy",
    }
)
OUTPUT_CONFIGURATION_KEYS = frozenset(
    {
        "direct_candidate_policy",
        "reverse_candidate_policy",
        "reverse_routes_enabled",
        "pose_translation_p90_semantics",
        "depth_confidence_semantics",
        "maximum_memory_candidates_per_state",
    }
)
OUTPUT_RECORD_KEYS = frozenset(
    {
        "sample_id",
        "scene",
        "source_episode_id",
        "goal_role",
        "decision_frame",
        "causal_prefix_sha256",
        "navdp_fifo_sha256",
        "goal_sha256",
        "activation_status",
        "usable_match_probability",
        "dustbin_probability",
        "selected_anchor_frame_index",
        "memory_candidates",
    }
)
OUTPUT_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_type",
        "candidate_mode",
        "priority",
        "anchor_frame_index",
        "route_frame_indices",
        "subgoal_forward_m",
        "subgoal_left_m",
        "graph_path_m",
        "graph_hops",
        "localization_probability",
        "candidate_validity_probability",
        "pose_translation_p90_m",
        "pose_translation_uncertainty_present",
        "pose_yaw_p90_deg",
        "pose_yaw_uncertainty_present",
        "depth_confidence_mean",
        "depth_confidence_present",
    }
)
OUTPUT_SUMMARY_KEYS = frozenset(
    {
        "record_count",
        "activated_match_record_count",
        "rejected_no_match_record_count",
        "direct_candidate_count",
        "reverse_candidate_count",
        "states_without_reverse_route",
        "future_frames_consumed",
        "privileged_inputs_consumed",
    }
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY_FRAGMENTS = (
    "teacher",
    "oracle",
    "ground_truth",
    "gt_",
    "covis",
    "pathfinder",
    "navmesh",
    "habitat_pose",
    "target_",
    "label",
)
_TOLERANCE = 1e-6


class MemoryGraphCandidateBuildError(ValueError):
    """An inference, route, or output artifact failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryGraphCandidateBuildError(message)


def _mapping(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = frozenset(map(str, value.keys()))
    _require(
        actual == expected,
        f"{label} fields changed: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
    )
    return value


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA256",
    )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def _finite(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite numeric",
    )
    result = float(value)
    _require(minimum is None or result >= minimum, f"{label} is below {minimum}")
    _require(maximum is None or result <= maximum, f"{label} exceeds {maximum}")
    return result


def _vector(value: object, length: int, label: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == length,
        f"{label} must be a length-{length} list",
    )
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _close(left: float, right: float, label: str) -> None:
    _require(
        math.isclose(left, right, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE),
        f"{label} differs: {left} != {right}",
    )


def _scan_forbidden_keys(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            _require(
                not any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS),
                f"{location} contains forbidden key {raw_key!r}",
            )
            _scan_forbidden_keys(child, f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{location}[{index}]")


@dataclass(frozen=True)
class ReverseRoutePins:
    producer_source_sha256: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        _sha(self.producer_source_sha256, "reverse pins producer SHA")
        _sha(self.configuration_sha256, "reverse pins configuration SHA")


@dataclass(frozen=True)
class ValidatedReverseRoutes:
    artifact: Mapping[str, Any]
    records_by_sample: Mapping[str, Mapping[str, Any]]


def validate_reverse_route_artifact(
    *,
    artifact: Mapping[str, Any],
    artifact_sha256: str,
    manifest_sha256: str,
    inference_artifact_sha256: str,
    inference: ValidatedPhaseBInference,
    inference_pins: InferencePins,
    pins: ReverseRoutePins,
) -> ValidatedReverseRoutes:
    """Validate optional routes without treating them as localization output."""

    _sha(artifact_sha256, "reverse-route artifact SHA")
    _require(
        sha256_bytes(canonical_json_bytes(artifact)) == artifact_sha256,
        "reverse-route artifact differs from its canonical SHA pin",
    )
    top = _mapping(artifact, REVERSE_TOP_KEYS, "reverse-route artifact")
    _require(
        top["schema_version"] == REVERSE_ROUTE_SCHEMA_VERSION,
        "reverse-route schema changed",
    )
    _require(top["purpose"] == REVERSE_ROUTE_PURPOSE, "reverse-route purpose changed")
    _scan_forbidden_keys(top, "reverse-route artifact")
    provenance = _mapping(
        top["provenance"], REVERSE_PROVENANCE_KEYS, "reverse-route provenance"
    )
    expected = {
        "input_manifest_sha256": manifest_sha256,
        "input_inference_artifact_sha256": inference_artifact_sha256,
        "flow_route_artifact_sha256": inference_pins.flow_route_artifact_sha256,
        "causal_scale_artifact_sha256": inference_pins.causal_scale_artifact_sha256,
        "producer_source_sha256": pins.producer_source_sha256,
        "configuration_sha256": pins.configuration_sha256,
        "pose_convention": POSE_CONVENTION,
        "privileged_input_policy": PRIVILEGED_INPUT_POLICY,
    }
    for field, value in expected.items():
        _require(
            provenance[field] == value,
            f"reverse-route provenance {field} differs from its external pin",
        )
    configuration = _mapping(
        top["configuration"], REVERSE_CONFIGURATION_KEYS, "reverse-route configuration"
    )
    _require(
        sha256_bytes(canonical_json_bytes(configuration)) == pins.configuration_sha256,
        "reverse-route configuration differs from its external pin",
    )
    _finite(configuration["spacing_m"], "reverse spacing", minimum=1e-12)
    _require(
        configuration["route_semantics"] == REVERSE_ROUTE_SEMANTICS,
        "reverse route semantics changed",
    )
    _require(
        configuration["first_subgoal_semantics"] == FIRST_SUBGOAL_SEMANTICS,
        "reverse first-subgoal semantics changed",
    )

    raw_records = top["records"]
    _require(isinstance(raw_records, list), "reverse-route records must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    node_count = 0
    for record_index, raw in enumerate(raw_records):
        record = _mapping(raw, REVERSE_RECORD_KEYS, f"reverse record {record_index}")
        sample_id = record["sample_id"]
        _require(
            isinstance(sample_id, str) and sample_id in inference.records_by_sample,
            "reverse route references an unknown inference sample",
        )
        _require(sample_id not in records, f"duplicate reverse route for {sample_id}")
        source = inference.records_by_sample[sample_id]
        selected = source["selected_anchor_frame_index"]
        _require(selected is not None, f"reverse route exists for rejected sample {sample_id}")
        expected_record = {
            "causal_prefix_sha256": source["causal_prefix_sha256"],
            "decision_frame": source["decision_frame"],
            "cam_pose_prefix_sha256": source["cam_pose_prefix_sha256"],
            "anchor_frame_index": selected,
            "start_frame_index": int(source["decision_frame"]) - 1,
            "metric_scale_m_per_raw": source["metric_scale_m_per_raw"],
            "metric_scale_source": METRIC_SCALE_SOURCE,
        }
        for field, value in expected_record.items():
            if field == "metric_scale_m_per_raw":
                _close(float(record[field]), float(value), f"{sample_id} route scale")
            else:
                _require(
                    record[field] == value,
                    f"reverse route {sample_id} changed inference field {field}",
                )
        _sha(record["causal_prefix_sha256"], f"{sample_id} reverse prefix SHA")
        _sha(record["cam_pose_prefix_sha256"], f"{sample_id} reverse pose SHA")
        decision = _integer(record["decision_frame"], f"{sample_id} decision", minimum=1)
        start = _integer(record["start_frame_index"], f"{sample_id} start")
        anchor = _integer(record["anchor_frame_index"], f"{sample_id} anchor")
        _require(
            0 <= anchor < start < decision and start == decision - 1,
            f"{sample_id} reverse indices are not a causal prefix",
        )
        scale = _finite(
            record["metric_scale_m_per_raw"], f"{sample_id} route scale", minimum=1e-12
        )
        current = _vector(
            record["current_native_position_xz_raw"], 2, f"{sample_id} current x-z"
        )
        yaw = _finite(record["current_native_yaw_rad"], f"{sample_id} current yaw")
        nodes_raw = record["route_nodes"]
        _require(
            isinstance(nodes_raw, list) and bool(nodes_raw),
            f"{sample_id} reverse route has no nodes",
        )
        nodes = []
        previous_frame = start
        previous_position = current
        path_sum = 0.0
        for node_index, raw_node in enumerate(nodes_raw):
            node = _mapping(
                raw_node, REVERSE_NODE_KEYS, f"{sample_id}.route_nodes[{node_index}]"
            )
            frame = _integer(node["frame_index"], f"{sample_id} route node frame")
            _require(
                anchor <= frame < previous_frame,
                f"{sample_id} reverse node frames are not strictly descending",
            )
            previous_frame = frame
            position = _vector(
                node["native_position_xz_raw"], 2, f"{sample_id} route node x-z"
            )
            segment = _finite(
                node["segment_path_m"],
                f"{sample_id} route segment path",
                minimum=1e-12,
            )
            # ``segment_path_m`` may exceed the endpoint chord when the
            # recorded trajectory bends between resampled nodes, but it can
            # never be shorter than that chord.  Recomputing this lower bound
            # from the pinned native x-z coordinates and the same per-episode
            # scale catches both raw/metric unit mixups and nodes copied from
            # a different pose stream.
            chord_m = scale * math.hypot(
                position[0] - previous_position[0],
                position[1] - previous_position[1],
            )
            tolerance = _TOLERANCE * max(1.0, abs(segment), abs(chord_m))
            _require(
                segment + tolerance >= chord_m,
                f"{sample_id} route segment {node_index} is shorter than its "
                f"scaled x-z chord: {segment} < {chord_m}",
            )
            path_sum += segment
            nodes.append(node)
            previous_position = position
        _require(
            int(nodes[-1]["frame_index"]) == anchor,
            f"{sample_id} reverse route does not end at the selected anchor",
        )
        graph_path = _finite(
            record["graph_path_m"], f"{sample_id} graph path", minimum=1e-12
        )
        _close(graph_path, path_sum, f"{sample_id} graph path sum")
        first_position = [float(value) for value in nodes[0]["native_position_xz_raw"]]
        delta_x = scale * (first_position[0] - current[0])
        delta_z = scale * (first_position[1] - current[1])
        expected_forward = math.cos(yaw) * delta_x + math.sin(yaw) * delta_z
        expected_left = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_z
        first_subgoal = _vector(
            record["first_subgoal_forward_left_m"],
            2,
            f"{sample_id} first reverse subgoal",
        )
        _close(first_subgoal[0], expected_forward, f"{sample_id} reverse forward")
        _close(first_subgoal[1], expected_left, f"{sample_id} reverse left")
        _require(
            math.hypot(*first_subgoal) > 1e-9,
            f"{sample_id} reverse first subgoal is degenerate",
        )
        records[str(sample_id)] = record
        node_count += len(nodes)
    summary = _mapping(top["summary"], REVERSE_SUMMARY_KEYS, "reverse-route summary")
    expected_summary = {
        "record_count": len(records),
        "route_node_count": node_count,
        "future_frames_consumed": 0,
        "privileged_inputs_consumed": 0,
    }
    _require(summary == expected_summary, "reverse-route summary disagrees with records")
    return ValidatedReverseRoutes(artifact=artifact, records_by_sample=records)


def _translation_p90(variance: Sequence[object]) -> float:
    values = [_finite(item, "translation variance", minimum=1e-12) for item in variance]
    _require(len(values) == 2, "translation variance must have two axes")
    return math.sqrt(CHI_SQUARE_2D_P90 * max(values))


def _direct_candidate(
    sample_id: str, inference_record: Mapping[str, Any]
) -> dict[str, Any]:
    anchor = int(inference_record["selected_anchor_frame_index"])
    selected = next(
        candidate
        for candidate in inference_record["candidates"]
        if int(candidate["anchor_frame_index"]) == anchor
    )
    subgoal = [float(value) for value in selected["corrected_relative_forward_left_m"]]
    confidence = min(
        float(selected["goal_depth_confidence_mean"]),
        float(selected["candidate_depth_confidence_mean"]),
    )
    return {
        "candidate_id": f"memory-direct-anchor-{anchor:06d}",
        "candidate_type": "memory_graph",
        "candidate_mode": "direct",
        "priority": 0,
        "anchor_frame_index": anchor,
        "route_frame_indices": [],
        "subgoal_forward_m": subgoal[0],
        "subgoal_left_m": subgoal[1],
        # Direct is a point-goal chord, not a graph path.  Its coordinates
        # already carry distance; zero keeps graph-path semantics unambiguous.
        "graph_path_m": 0.0,
        "graph_hops": 0,
        "localization_probability": float(selected["set_probability"]),
        "candidate_validity_probability": float(
            selected["candidate_validity_probability"]
        ),
        "pose_translation_p90_m": _translation_p90(
            selected["translation_variance_forward_left_m2"]
        ),
        "pose_translation_uncertainty_present": True,
        # The Phase-B checkpoint has no yaw residual/covariance head.  Zero is
        # neutral and the separate mask prevents it from claiming perfect yaw.
        "pose_yaw_p90_deg": 0.0,
        "pose_yaw_uncertainty_present": False,
        "depth_confidence_mean": confidence,
        "depth_confidence_present": True,
    }


def _reverse_candidate(
    inference_record: Mapping[str, Any], route: Mapping[str, Any]
) -> dict[str, Any]:
    direct = _direct_candidate(str(inference_record["sample_id"]), inference_record)
    anchor = int(inference_record["selected_anchor_frame_index"])
    route_frames = [int(node["frame_index"]) for node in route["route_nodes"]]
    subgoal = [float(value) for value in route["first_subgoal_forward_left_m"]]
    result = dict(direct)
    result.update(
        {
            "candidate_id": f"memory-reverse-anchor-{anchor:06d}",
            "candidate_mode": "reverse",
            "priority": 1,
            "route_frame_indices": route_frames,
            "subgoal_forward_m": subgoal[0],
            "subgoal_left_m": subgoal[1],
            "graph_path_m": float(route["graph_path_m"]),
            "graph_hops": len(route_frames),
            # The Phase-B covariance describes the goal localized relative to
            # the selected anchor.  A reverse candidate's immediate waypoint
            # is instead a node on the recorded pose chain.  Reusing the goal
            # covariance here would silently attach endpoint uncertainty to a
            # different geometric quantity.  The reverse-route v1 schema has
            # no route-pose covariance, so represent it honestly as absent.
            "pose_translation_p90_m": 0.0,
            "pose_translation_uncertainty_present": False,
        }
    )
    return result


def _validate_output_candidate(candidate: Mapping[str, Any], label: str) -> None:
    _mapping(candidate, OUTPUT_CANDIDATE_KEYS, label)
    _require(candidate["candidate_type"] == "memory_graph", f"{label} type changed")
    mode = candidate["candidate_mode"]
    _require(mode in ("direct", "reverse"), f"{label} mode is invalid")
    priority = _integer(candidate["priority"], f"{label}.priority")
    hops = _integer(candidate["graph_hops"], f"{label}.graph_hops")
    route = candidate["route_frame_indices"]
    _require(isinstance(route, list), f"{label}.route_frame_indices must be a list")
    _finite(candidate["subgoal_forward_m"], f"{label}.forward")
    _finite(candidate["subgoal_left_m"], f"{label}.left")
    graph_path = _finite(candidate["graph_path_m"], f"{label}.graph_path", minimum=0.0)
    if mode == "direct":
        _require(
            priority == 0 and hops == 0 and route == [] and graph_path == 0.0,
            f"{label} violates direct priority/hops/path invariants",
        )
    else:
        _require(
            priority == 1 and hops == len(route) and hops >= 1 and graph_path > 0.0,
            f"{label} violates append-only reverse invariants",
        )
    for field in ("localization_probability", "candidate_validity_probability"):
        _finite(candidate[field], f"{label}.{field}", minimum=0.0, maximum=1.0)
    translation_p90 = _finite(
        candidate["pose_translation_p90_m"],
        f"{label}.translation p90",
        minimum=0.0,
    )
    yaw_p90 = _finite(candidate["pose_yaw_p90_deg"], f"{label}.yaw p90", minimum=0.0)
    # LingBot/VGGT confidence is a non-negative model score, not a calibrated
    # probability, and valid deployment values commonly exceed one.
    _finite(candidate["depth_confidence_mean"], f"{label}.depth", minimum=0.0)
    for field in (
        "pose_translation_uncertainty_present",
        "pose_yaw_uncertainty_present",
        "depth_confidence_present",
    ):
        _require(type(candidate[field]) is bool, f"{label}.{field} must be boolean")
    if candidate["pose_translation_uncertainty_present"]:
        _require(translation_p90 > 0.0, f"{label}.translation p90 must be positive")
    else:
        _require(translation_p90 == 0.0, f"{label}.absent translation p90 must be zero")
    if candidate["pose_yaw_uncertainty_present"]:
        _require(yaw_p90 > 0.0, f"{label}.yaw p90 must be positive")
    else:
        _require(yaw_p90 == 0.0, f"{label}.absent yaw p90 must be zero")
    _require(
        candidate["pose_translation_uncertainty_present"] is (mode == "direct"),
        f"{label} translation uncertainty mask disagrees with candidate mode",
    )
    _require(
        candidate["pose_yaw_uncertainty_present"] is False,
        f"{label} cannot claim unavailable yaw uncertainty",
    )
    _require(
        candidate["depth_confidence_present"] is True,
        f"{label} must retain the selected localization depth score",
    )


def build_memory_graph_candidate_artifact(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    inference_artifact: Mapping[str, Any],
    inference_artifact_sha256: str,
    inference_pins: InferencePins,
    reverse_route_artifact: Mapping[str, Any] | None = None,
    reverse_route_artifact_sha256: str | None = None,
    reverse_route_pins: ReverseRoutePins | None = None,
) -> dict[str, Any]:
    """Build deterministic direct-first memory candidates from frozen inputs."""

    _require(
        (reverse_route_artifact is None)
        == (reverse_route_artifact_sha256 is None)
        == (reverse_route_pins is None),
        "reverse artifact, SHA, and pins must be supplied together",
    )
    inference = validate_phase_b_deployment_inference(
        artifact=inference_artifact,
        artifact_sha256=inference_artifact_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        pins=inference_pins,
    )
    reverse = None
    if reverse_route_artifact is not None:
        assert reverse_route_artifact_sha256 is not None
        assert reverse_route_pins is not None
        reverse = validate_reverse_route_artifact(
            artifact=reverse_route_artifact,
            artifact_sha256=reverse_route_artifact_sha256,
            manifest_sha256=manifest_sha256,
            inference_artifact_sha256=inference_artifact_sha256,
            inference=inference,
            inference_pins=inference_pins,
            pins=reverse_route_pins,
        )
    configuration = {
        "direct_candidate_policy": DIRECT_POLICY,
        "reverse_candidate_policy": REVERSE_POLICY,
        "reverse_routes_enabled": reverse is not None,
        "pose_translation_p90_semantics": POSE_P90_SEMANTICS,
        "depth_confidence_semantics": DEPTH_CONFIDENCE_SEMANTICS,
        "maximum_memory_candidates_per_state": 2,
    }
    assembler_path = Path(__file__)
    contract_path = assembler_path.with_name("phase_b_deployment_inference_contract.py")
    provenance = inference_artifact["provenance"]
    output_provenance = {
        "input_manifest_sha256": manifest_sha256,
        "input_inference_artifact_sha256": inference_artifact_sha256,
        "input_reverse_route_artifact_sha256": (
            reverse_route_artifact_sha256 if reverse is not None else ZERO_SHA
        ),
        "phase_b_checkpoint_sha256": inference_pins.phase_b_checkpoint_sha256,
        "flow_route_artifact_sha256": inference_pins.flow_route_artifact_sha256,
        "causal_scale_artifact_sha256": inference_pins.causal_scale_artifact_sha256,
        "inference_producer_source_sha256": (
            inference_pins.inference_producer_source_sha256
        ),
        "reverse_route_producer_source_sha256": (
            reverse_route_pins.producer_source_sha256
            if reverse_route_pins is not None
            else ZERO_SHA
        ),
        "assembler_source_sha256": sha256_file(assembler_path),
        "inference_contract_source_sha256": sha256_file(contract_path),
        "configuration_sha256": sha256_bytes(canonical_json_bytes(configuration)),
        "pose_convention": provenance["pose_convention"],
        "privileged_input_policy": provenance["privileged_input_policy"],
    }
    records = []
    matched = direct_count = reverse_count = missing_reverse = 0
    reverse_by_sample = {} if reverse is None else reverse.records_by_sample
    for sample_id in sorted(inference.records_by_sample):
        source = inference.records_by_sample[sample_id]
        selected = source["selected_anchor_frame_index"]
        candidates = []
        if selected is not None:
            matched += 1
            candidates.append(_direct_candidate(sample_id, source))
            direct_count += 1
            route = reverse_by_sample.get(sample_id)
            if route is not None:
                candidates.append(_reverse_candidate(source, route))
                reverse_count += 1
            else:
                missing_reverse += 1
        for index, candidate in enumerate(candidates):
            _validate_output_candidate(candidate, f"{sample_id}.memory_candidates[{index}]")
        if candidates:
            _require(
                candidates[0]["candidate_mode"] == "direct"
                and candidates[0]["priority"] == 0
                and candidates[0]["graph_hops"] == 0,
                f"{sample_id} direct candidate lost first priority",
            )
            _require(
                len(candidates) == 1 or candidates[1]["candidate_mode"] == "reverse",
                f"{sample_id} reverse candidate replaced direct",
            )
        record = {
            "sample_id": sample_id,
            "scene": source["scene"],
            "source_episode_id": source["source_episode_id"],
            "goal_role": source["goal_role"],
            "decision_frame": source["decision_frame"],
            "causal_prefix_sha256": source["causal_prefix_sha256"],
            "navdp_fifo_sha256": source["navdp_fifo_sha256"],
            "goal_sha256": source["goal_sha256"],
            "activation_status": "matched" if selected is not None else "rejected_no_match",
            "usable_match_probability": source["usable_match_probability"],
            "dustbin_probability": source["dustbin_probability"],
            "selected_anchor_frame_index": selected,
            "memory_candidates": candidates,
        }
        _mapping(record, OUTPUT_RECORD_KEYS, f"output record {sample_id}")
        records.append(record)
    summary = {
        "record_count": len(records),
        "activated_match_record_count": matched,
        "rejected_no_match_record_count": len(records) - matched,
        "direct_candidate_count": direct_count,
        "reverse_candidate_count": reverse_count,
        "states_without_reverse_route": missing_reverse,
        "future_frames_consumed": 0,
        "privileged_inputs_consumed": 0,
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "provenance": output_provenance,
        "configuration": configuration,
        "records": records,
        "summary": summary,
    }
    _mapping(artifact, OUTPUT_TOP_KEYS, "output artifact")
    _mapping(output_provenance, OUTPUT_PROVENANCE_KEYS, "output provenance")
    _mapping(configuration, OUTPUT_CONFIGURATION_KEYS, "output configuration")
    _mapping(summary, OUTPUT_SUMMARY_KEYS, "output summary")
    _scan_forbidden_keys(artifact, "output artifact")
    canonical_json_bytes(artifact)
    return artifact


def write_artifact(
    artifact: Mapping[str, Any],
    output: Path,
    sha_output: Path | None = None,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Atomically write canonical JSON plus an exact sidecar."""

    _require(not (resume and overwrite), "resume and overwrite are mutually exclusive")
    sidecar = output.with_suffix(output.suffix + ".sha256") if sha_output is None else sha_output
    _require(output.resolve() != sidecar.resolve(), "output and sidecar paths must differ")
    payload = canonical_json_bytes(artifact)
    digest = sha256_bytes(payload)
    sidecar_payload = f"{digest}  {output.name}\n".encode("ascii")
    exists = output.exists(), sidecar.exists()
    if resume:
        _require(exists == (True, True), "resume requires output and sidecar")
        _require(
            output.read_bytes() == payload and sidecar.read_bytes() == sidecar_payload,
            "resume output differs from deterministic rebuilt content",
        )
        return "resumed", digest
    _require(overwrite or not any(exists), "output exists without --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    for destination, content in ((output, payload), (sidecar, sidecar_payload)):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return "written", digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--inference-artifact", type=Path, required=True)
    parser.add_argument("--expected-inference-sha", required=True)
    parser.add_argument("--expected-flow-route-sha", required=True)
    parser.add_argument("--causal-scale-artifact", type=Path, required=True)
    parser.add_argument("--expected-causal-scale-sha", required=True)
    parser.add_argument("--phase-b-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-phase-b-checkpoint-sha", required=True)
    parser.add_argument("--deployment-approval-artifact", type=Path, required=True)
    parser.add_argument("--expected-deployment-approval-sha", required=True)
    parser.add_argument("--expected-inference-producer-sha", required=True)
    parser.add_argument("--expected-inference-configuration-sha", required=True)
    parser.add_argument("--expected-dino-checkpoint-sha", required=True)
    parser.add_argument("--expected-dino-feature-producer-sha", required=True)
    parser.add_argument("--expected-lingbot-commit", required=True)
    parser.add_argument("--expected-lingbot-weights-sha", required=True)
    parser.add_argument("--expected-lingbot-stream-sha", required=True)
    parser.add_argument("--reverse-route-artifact", type=Path)
    parser.add_argument("--expected-reverse-route-sha")
    parser.add_argument("--expected-reverse-producer-sha")
    parser.add_argument("--expected-reverse-configuration-sha")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha-out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reverse_arguments = (
        args.reverse_route_artifact,
        args.expected_reverse_route_sha,
        args.expected_reverse_producer_sha,
        args.expected_reverse_configuration_sha,
    )
    _require(
        not any(value is not None for value in reverse_arguments)
        or all(value is not None for value in reverse_arguments),
        "all reverse-route arguments must be supplied together",
    )
    manifest = load_pinned_canonical_json(args.manifest, args.expected_manifest_sha)
    inference = load_pinned_canonical_json(
        args.inference_artifact, args.expected_inference_sha
    )
    inference_pins = InferencePins(
        flow_route_artifact_sha256=args.expected_flow_route_sha,
        causal_scale_artifact_path=args.causal_scale_artifact,
        causal_scale_artifact_sha256=args.expected_causal_scale_sha,
        phase_b_checkpoint_path=args.phase_b_checkpoint,
        phase_b_checkpoint_sha256=args.expected_phase_b_checkpoint_sha,
        deployment_approval_artifact_path=args.deployment_approval_artifact,
        deployment_approval_artifact_sha256=args.expected_deployment_approval_sha,
        inference_producer_source_sha256=args.expected_inference_producer_sha,
        inference_configuration_sha256=args.expected_inference_configuration_sha,
        dino_encoder_checkpoint_sha256=args.expected_dino_checkpoint_sha,
        dino_feature_producer_sha256=args.expected_dino_feature_producer_sha,
        lingbot_commit=args.expected_lingbot_commit,
        lingbot_weights_sha256=args.expected_lingbot_weights_sha,
        lingbot_stream_source_sha256=args.expected_lingbot_stream_sha,
    )
    reverse_artifact = (
        load_pinned_canonical_json(
            args.reverse_route_artifact, args.expected_reverse_route_sha
        )
        if args.reverse_route_artifact is not None
        else None
    )
    reverse_pins = (
        ReverseRoutePins(
            producer_source_sha256=args.expected_reverse_producer_sha,
            configuration_sha256=args.expected_reverse_configuration_sha,
        )
        if args.reverse_route_artifact is not None
        else None
    )
    artifact = build_memory_graph_candidate_artifact(
        manifest=manifest,
        manifest_sha256=args.expected_manifest_sha,
        inference_artifact=inference,
        inference_artifact_sha256=args.expected_inference_sha,
        inference_pins=inference_pins,
        reverse_route_artifact=reverse_artifact,
        reverse_route_artifact_sha256=args.expected_reverse_route_sha,
        reverse_route_pins=reverse_pins,
    )
    status, digest = write_artifact(
        artifact,
        args.out,
        args.sha_out,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "records": len(artifact["records"]),
                "direct_candidates": artifact["summary"]["direct_candidate_count"],
                "reverse_candidates": artifact["summary"]["reverse_candidate_count"],
                "output_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
