#!/usr/bin/env python3
"""Strict contract for deployment Phase-B localization inference.

This module is deliberately *not* an image/model inference runner.  A real GPU
producer still has to compute a causal DINO shortlist, run the LingBot
goal-append measurements for every anchor, and serialize the result described
here.  The validator does physically load every frozen Phase-B ensemble state
to prove checkpoint compatibility, then proves that the serialized result:

* covers the exact pinned Goal-B/Goal-C manifest;
* uses only anchors and goal-append prefixes available at the decision time;
* binds a deployment-approved Phase-B checkpoint and every upstream model;
* uses the external causal metric scale and LingBot's native x-z plane; and
* contains no teacher co-visibility, GT pose, oracle, Pathfinder, or navmesh
  input.

The explicit raw/native coordinate witnesses are intentional.  They make the
axis convention testable instead of trusting a prose field: for native
``[x,z]`` translation, NavDP point-goal ``[forward,left]`` must be
``[z,-x]`` after multiplication by the causal metric scale.

The current repository does not yet contain the GPU producer or an approved
``lingbot_native_phase_b.pt``.  Consequently this contract must fail closed
for the current unfinished training artifact; callers must never synthesize
an inference artifact from teacher columns to get past it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    from MemNavData.external_causal_scale_contract import (
        CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
    )
    from MemNavData.phase_b_feature_schema import (
        CHECKPOINT_MODEL_KIND,
        CHECKPOINT_SCHEMA_VERSION as PHASE_B_CHECKPOINT_SCHEMA_VERSION,
        EXTERNAL_CAUSAL_SCALE_SOURCE as METRIC_SCALE_SOURCE,
        FEATURE_DIMENSION,
        FEATURE_NAMES as PHASE_B_FEATURE_NAMES,
        FEATURE_NAMES_SHA256,
        FEATURE_SCHEMA_VERSION as PHASE_B_FEATURE_SCHEMA_VERSION,
        validate_checkpoint_metadata,
    )
except ModuleNotFoundError:  # direct script invocation
    from external_causal_scale_contract import (  # type: ignore
        CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
    )
    from phase_b_feature_schema import (  # type: ignore
        CHECKPOINT_MODEL_KIND,
        CHECKPOINT_SCHEMA_VERSION as PHASE_B_CHECKPOINT_SCHEMA_VERSION,
        EXTERNAL_CAUSAL_SCALE_SOURCE as METRIC_SCALE_SOURCE,
        FEATURE_DIMENSION,
        FEATURE_NAMES as PHASE_B_FEATURE_NAMES,
        FEATURE_NAMES_SHA256,
        FEATURE_SCHEMA_VERSION as PHASE_B_FEATURE_SCHEMA_VERSION,
        validate_checkpoint_metadata,
    )


SCHEMA_VERSION = "nlsr_phase_b_deployment_inference_v1"
PURPOSE = (
    "deployment-only Phase-B candidate localization over an exact causal "
    "Goal-B/Goal-C manifest; no teacher, GT pose, oracle, navmesh, or "
    "Pathfinder input"
)
SUPPORTED_MANIFEST_SCHEMAS = frozenset(
    {
        "nlsr_v2_expert_candidate_manifest_v2",
        "nlsr_v2_multistage_expert_candidate_manifest_v1",
    }
)
POSE_CONVENTION = "lingbot_native_xz_to_navdp_forward_left_v1"
CANDIDATE_SELECTION_SEMANTICS = "causal_dino_topk_temporal_diverse_v1"
ENSEMBLE_SEMANTICS = "mean_member_probabilities_total_predictive_variance_v1"
PRIVILEGED_INPUT_POLICY = "strict_deployment_allowlist_no_teacher_gt_or_navmesh_v1"

DEPLOYMENT_APPROVAL_SCHEMA_VERSION = "lingbot_native_phase_b_approval_v1"
DEPLOYMENT_APPROVAL_PURPOSE = (
    "detached approval for an immutable Phase-B checkpoint and frozen "
    "deployment inference configuration"
)
DEPLOYMENT_APPROVAL_KEYS = frozenset({
    "schema_version",
    "purpose",
    "deployment_approved",
    "checkpoint_sha256",
    "checkpoint_schema_version",
    "model_kind",
    "feature_schema_version",
    "input_dim",
    "feature_names_sha256",
    "train_artifact_identity_sha256",
    "development_artifact_identity_sha256",
    "train_audit_sha256",
    "development_audit_sha256",
    "deployment_input_contract_approved",
    "train_exact_coverage_approved",
    "development_exact_coverage_approved",
    "inference_configuration_sha256",
    "closed_loop_evidence_artifact_sha256",
})

TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "purpose", "provenance", "configuration", "records", "summary"}
)
PROVENANCE_KEYS = frozenset(
    {
        "input_manifest_sha256",
        "input_manifest_schema_version",
        "flow_route_artifact_sha256",
        "causal_scale_artifact_sha256",
        "phase_b_checkpoint_sha256",
        "phase_b_checkpoint_model_kind",
        "phase_b_checkpoint_schema_version",
        "phase_b_checkpoint_deployment_approved",
        "phase_b_checkpoint_deployment_input_contract_approved",
        "phase_b_external_causal_scale_coverage_approved",
        "phase_b_feature_schema_version",
        "phase_b_feature_names",
        "phase_b_feature_names_sha256",
        "deployment_approval_artifact_sha256",
        "inference_producer_source_sha256",
        "inference_configuration_sha256",
        "dino_encoder_checkpoint_sha256",
        "dino_feature_producer_sha256",
        "lingbot_commit",
        "lingbot_weights_sha256",
        "lingbot_stream_source_sha256",
        "pose_convention",
        "privileged_input_policy",
    }
)
CONFIGURATION_KEYS = frozenset(
    {
        "candidate_shortlist_size",
        "temporal_minimum_gap_frames",
        "neighbor_offsets",
        "match_threshold",
        "ensemble_member_count",
        "candidate_selection_semantics",
        "ensemble_semantics",
    }
)
RECORD_KEYS = frozenset(
    {
        "sample_id",
        "scene",
        "source_episode_id",
        "goal_role",
        "decision_frame",
        "causal_prefix_sha256",
        "navdp_fifo_sha256",
        "goal_sha256",
        "cam_pose_prefix_sha256",
        "cam_pose_prefix_frame_count",
        "metric_scale_m_per_raw",
        "metric_scale_source",
        "external_scale_record_sha256",
        "external_scale_prefix_end_frame_exclusive",
        "external_scale_cam_pose_prefix_sha256",
        "external_scale_rgb_prefix_content_sequence_sha256",
        "global_no_match_probability",
        "usable_match_probability",
        "dustbin_probability",
        "selected_anchor_frame_index",
        "candidates",
    }
)
CANDIDATE_KEYS = frozenset(
    {
        "candidate_rank",
        "anchor_frame_index",
        "goal_append_prefix_end_frame_exclusive",
        "goal_append_input_sha256",
        "dino_cosine",
        "phase_b_input_features",
        "rank_probability",
        "candidate_validity_probability",
        "set_probability",
        "native_relative_translation_xz_raw",
        "scaled_relative_translation_xz_m",
        "raw_relative_forward_left_m",
        "corrected_relative_forward_left_m",
        "translation_variance_forward_left_m2",
        "goal_depth_confidence_mean",
        "candidate_depth_confidence_mean",
    }
)
SUMMARY_KEYS = frozenset(
    {
        "record_count",
        "goal_b_record_count",
        "goal_c_record_count",
        "activated_match_record_count",
        "rejected_no_match_record_count",
        "candidate_count",
        "future_frames_consumed",
        "privileged_inputs_consumed",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
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
_PROBABILITY_TOLERANCE = 2e-6
_NUMERIC_TOLERANCE = 1e-6


class PhaseBInferenceContractError(ValueError):
    """A deployment inference artifact failed its immutable contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseBInferenceContractError(message)


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite canonical JSON with one trailing newline."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PhaseBInferenceContractError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_pinned_canonical_json(
    path: Path | str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    """Load an exact physical JSON artifact under an external SHA pin."""

    source = Path(path)
    _sha(expected_sha256, f"{source.name} expected SHA256")
    _require(source.is_file(), f"pinned JSON is missing: {source}")
    raw = source.read_bytes()
    _require(
        sha256_bytes(raw) == expected_sha256,
        f"pinned JSON SHA256 changed: {source}",
    )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PhaseBInferenceContractError(
                    f"{source} contains non-finite constant {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBInferenceContractError(
            f"pinned JSON is invalid: {source}"
        ) from error
    _require(isinstance(value, Mapping), f"pinned JSON must be an object: {source}")
    _require(
        raw == canonical_json_bytes(value),
        f"pinned JSON is not canonical: {source}",
    )
    return value


def _mapping(
    value: object,
    expected_keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = frozenset(map(str, value.keys()))
    _require(
        actual == expected_keys,
        f"{label} fields changed: missing={sorted(expected_keys - actual)} "
        f"extra={sorted(actual - expected_keys)}",
    )
    return value


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA256",
    )
    return value


def _commit(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None,
        f"{label} must be a 40- or 64-character lowercase git digest",
    )
    return value


def _string(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    result = int(value)
    _require(minimum is None or result >= minimum, f"{label} is below {minimum}")
    return result


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


def _probability(value: object, label: str) -> float:
    return _finite(value, label, minimum=0.0, maximum=1.0)


def _vector(value: object, length: int, label: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == length,
        f"{label} must be a length-{length} list",
    )
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _close(left: float, right: float, label: str, tolerance: float = _NUMERIC_TOLERANCE) -> None:
    _require(
        math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance),
        f"{label} differs: {left} != {right}",
    )


def _scan_forbidden_keys(value: object, location: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            _require(
                not any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS),
                f"{location} contains forbidden deployment key {raw_key!r}",
            )
            _scan_forbidden_keys(child, f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{location}[{index}]")


@dataclass(frozen=True)
class InferencePins:
    """External authorities that must not be trusted from the artifact itself."""

    flow_route_artifact_sha256: str
    causal_scale_artifact_path: Path | str
    causal_scale_artifact_sha256: str
    phase_b_checkpoint_path: Path | str
    phase_b_checkpoint_sha256: str
    deployment_approval_artifact_path: Path | str
    deployment_approval_artifact_sha256: str
    inference_producer_source_sha256: str
    inference_configuration_sha256: str
    dino_encoder_checkpoint_sha256: str
    dino_feature_producer_sha256: str
    lingbot_commit: str
    lingbot_weights_sha256: str
    lingbot_stream_source_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "flow_route_artifact_sha256",
            "causal_scale_artifact_sha256",
            "phase_b_checkpoint_sha256",
            "deployment_approval_artifact_sha256",
            "inference_producer_source_sha256",
            "inference_configuration_sha256",
            "dino_encoder_checkpoint_sha256",
            "dino_feature_producer_sha256",
            "lingbot_weights_sha256",
            "lingbot_stream_source_sha256",
        ):
            _sha(getattr(self, field), f"pins.{field}")
        _commit(self.lingbot_commit, "pins.lingbot_commit")


@dataclass(frozen=True)
class ValidatedPhaseBInference:
    artifact: Mapping[str, Any]
    records_by_sample: Mapping[str, Mapping[str, Any]]
    manifest_samples: Mapping[str, Mapping[str, Any]]
    match_threshold: float


def _load_physical_checkpoint(pins: InferencePins) -> Mapping[str, Any]:
    """Load the immutable checkpoint bytes; provenance booleans are not authority."""

    checkpoint_path = Path(pins.phase_b_checkpoint_path)
    _require(checkpoint_path.is_file(),
             f"pinned Phase-B checkpoint is missing: {checkpoint_path}")
    _require(sha256_file(checkpoint_path) == pins.phase_b_checkpoint_sha256,
             "physical Phase-B checkpoint SHA256 changed")
    try:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PhaseBInferenceContractError(
            f"physical Phase-B checkpoint cannot be loaded safely: {error}"
        ) from error
    _require(isinstance(checkpoint, Mapping),
             "physical Phase-B checkpoint is not an object")
    try:
        validate_checkpoint_metadata(
            checkpoint, require_deployment_input_contract=True)
    except RuntimeError as error:
        raise PhaseBInferenceContractError(str(error)) from error
    return checkpoint


def _validate_checkpoint_states(
    checkpoint: Mapping[str, Any], ensemble_member_count: int,
) -> None:
    """Prove that every serialized member can actually construct the model."""

    try:
        import torch
        try:
            from MemNavData.train_lingbot_native_localizer import (
                LingBotNativeLocalizer,
            )
        except ModuleNotFoundError:
            from train_lingbot_native_localizer import (  # type: ignore
                LingBotNativeLocalizer,
            )
        configuration = checkpoint.get("config")
        _require(isinstance(configuration, Mapping),
                 "Phase-B checkpoint config is malformed")
        hidden_dim = _integer(
            configuration.get("hidden_dim"), "checkpoint hidden dimension",
            minimum=1)
        dropout = _finite(
            configuration.get("dropout"), "checkpoint dropout",
            minimum=0.0, maximum=1.0)
        states = checkpoint.get("states")
        _require(isinstance(states, list)
                 and len(states) == ensemble_member_count,
                 "checkpoint ensemble count differs from inference configuration")
        for index, state in enumerate(states):
            _require(isinstance(state, Mapping),
                     f"checkpoint state {index} is malformed")
            for name, value in state.items():
                _require(isinstance(name, str) and torch.is_tensor(value),
                         f"checkpoint state {index} contains a non-tensor")
                _require(bool(torch.isfinite(value).all()),
                         f"checkpoint state {index}.{name} is non-finite")
            model = LingBotNativeLocalizer(
                FEATURE_DIMENSION, hidden_dim=hidden_dim, dropout=dropout)
            model.load_state_dict(state, strict=True)
    except PhaseBInferenceContractError:
        raise
    except Exception as error:
        raise PhaseBInferenceContractError(
            f"Phase-B checkpoint state cannot be loaded strictly: {error}"
        ) from error


def _load_and_validate_approval(
    pins: InferencePins,
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    approval = load_pinned_canonical_json(
        pins.deployment_approval_artifact_path,
        pins.deployment_approval_artifact_sha256,
    )
    approval = _mapping(
        approval, DEPLOYMENT_APPROVAL_KEYS, "deployment approval")
    _require(approval["schema_version"] == DEPLOYMENT_APPROVAL_SCHEMA_VERSION,
             "deployment approval schema changed")
    _require(approval["purpose"] == DEPLOYMENT_APPROVAL_PURPOSE,
             "deployment approval purpose changed")
    expected = {
        "deployment_approved": True,
        "checkpoint_sha256": pins.phase_b_checkpoint_sha256,
        "checkpoint_schema_version": PHASE_B_CHECKPOINT_SCHEMA_VERSION,
        "model_kind": CHECKPOINT_MODEL_KIND,
        "feature_schema_version": PHASE_B_FEATURE_SCHEMA_VERSION,
        "input_dim": FEATURE_DIMENSION,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "train_artifact_identity_sha256": checkpoint.get(
            "train_artifact_identity_sha256"),
        "development_artifact_identity_sha256": checkpoint.get(
            "development_artifact_identity_sha256"),
        "train_audit_sha256": checkpoint.get("train_audit_sha256"),
        "development_audit_sha256": checkpoint.get(
            "development_audit_sha256"),
        "deployment_input_contract_approved": True,
        "train_exact_coverage_approved": True,
        "development_exact_coverage_approved": True,
        "inference_configuration_sha256": (
            pins.inference_configuration_sha256),
    }
    for field, value in expected.items():
        _require(approval[field] == value,
                 f"deployment approval {field} differs from its subject")
    _sha(approval["closed_loop_evidence_artifact_sha256"],
         "deployment approval closed-loop evidence SHA")
    for field in ("train_artifact_identity_sha256",
                  "development_artifact_identity_sha256",
                  "train_audit_sha256",
                  "development_audit_sha256"):
        _sha(approval[field], f"deployment approval {field}")
    return approval


def _physical_scale_record_index(
    pins: InferencePins,
    *,
    checkpoint: Mapping[str, Any],
    samples: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    scale_artifact = load_pinned_canonical_json(
        pins.causal_scale_artifact_path, pins.causal_scale_artifact_sha256)
    _require(scale_artifact.get("schema_version")
             == CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
             "physical causal-scale artifact schema changed")
    provenance = scale_artifact.get("provenance")
    _require(isinstance(provenance, Mapping),
             "physical causal-scale provenance is malformed")
    _require(provenance.get("input_manifest_sha256") == manifest_sha256,
             "physical causal-scale artifact binds another manifest")
    estimator = provenance.get("estimator")
    _require(isinstance(estimator, Mapping),
             "physical causal-scale estimator provenance is malformed")
    expected_estimator = {
        "lingbot_commit": pins.lingbot_commit,
        "weights_sha256": pins.lingbot_weights_sha256,
        "lingbot_stream_source_sha256": pins.lingbot_stream_source_sha256,
    }
    for field, value in expected_estimator.items():
        _require(estimator.get(field) == value,
                 f"physical causal-scale estimator {field} changed")
    _require(estimator.get("kind")
             == "frozen_lingbot_compute_metric_scale_prefix",
             "physical causal-scale estimator kind changed")
    _sha(provenance.get("producer_source_sha256"),
         "physical causal-scale producer SHA")
    configuration_object = scale_artifact.get("configuration")
    _require(isinstance(configuration_object, Mapping),
             "physical causal-scale configuration is malformed")
    _require(provenance.get("configuration_sha256")
             == sha256_bytes(canonical_json_bytes(configuration_object)),
             "physical causal-scale configuration digest changed")

    # The runtime scale is not merely valid in isolation: it must use the same
    # estimator/configuration/model convention seen by *both* audited training
    # roles.  The manifest and scale artifact SHA are intentionally allowed to
    # differ at deployment because held-out episodes require newly generated
    # artifacts; their physical bytes are independently pinned above.
    coverage = checkpoint.get("external_causal_scale_coverage")
    _require(isinstance(coverage, Mapping),
             "Phase-B checkpoint external-scale coverage is malformed")
    role_expected = {
        "producer_source_sha256": provenance.get("producer_source_sha256"),
        "configuration_sha256": provenance.get("configuration_sha256"),
        "lingbot_commit": pins.lingbot_commit,
        "weights_sha256": pins.lingbot_weights_sha256,
        "stream_source_sha256": pins.lingbot_stream_source_sha256,
        "manifest_schema_version": provenance.get(
            "input_manifest_schema_version"),
        "source": METRIC_SCALE_SOURCE,
    }
    for role in ("train", "development"):
        role_coverage = coverage.get(role)
        _require(isinstance(role_coverage, Mapping),
                 f"Phase-B checkpoint lacks {role} external-scale binding")
        _require(role_coverage.get("approved") is True
                 and role_coverage.get(
                     "exact_manifest_sample_coverage_approved") is True,
                 f"Phase-B checkpoint {role} scale coverage is not exact")
        _require(role_coverage.get("split_roles") == [role],
                 f"Phase-B checkpoint {role} scale role changed")
        for field in ("manifest_sha256", "scale_artifact_sha256"):
            historical_pins = role_coverage.get(field)
            _require(isinstance(historical_pins, list)
                     and len(historical_pins) == 1,
                     f"Phase-B checkpoint {role} {field} is malformed")
            _sha(historical_pins[0],
                 f"Phase-B checkpoint {role} historical {field}")
        for field, expected_value in role_expected.items():
            _require(role_coverage.get(field) == expected_value,
                     f"Phase-B checkpoint {role} external-scale {field} "
                     "differs from physical inference scale")
    expected: dict[str, dict[str, list[Any]]] = {}
    for sample_id, sample in samples.items():
        source_id = str(sample["source_episode_id"])
        group = expected.setdefault(
            source_id, {"sample_ids": [], "decision_frames": []})
        group["sample_ids"].append(sample_id)
        group["decision_frames"].append(int(sample["decision_frame"]))
    raw_records = scale_artifact.get("records")
    _require(isinstance(raw_records, list),
             "physical causal-scale records are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for raw_record in raw_records:
        _require(isinstance(raw_record, Mapping),
                 "physical causal-scale record is malformed")
        scene = _string(raw_record.get("scene"), "scale record scene")
        episode = _string(raw_record.get("episode"), "scale record episode")
        source_id = f"{scene}/{episode}"
        _require(source_id in expected and source_id not in result,
                 "physical causal-scale record episode is extra or duplicated")
        expected_group = expected[source_id]
        _require(raw_record.get("sample_ids")
                 == sorted(expected_group["sample_ids"]),
                 f"physical causal-scale {source_id} sample binding changed")
        _require(raw_record.get("decision_frames")
                 == sorted(set(expected_group["decision_frames"])),
                 f"physical causal-scale {source_id} decision binding changed")
        _require(raw_record.get("valid") is True,
                 f"physical causal-scale {source_id} is invalid")
        scale = _finite(raw_record.get("metric_scale_m_per_raw"),
                        f"physical causal-scale {source_id}", minimum=1e-12)
        prefix_end = _integer(
            raw_record.get("prefix_end_frame_exclusive"),
            f"physical causal-scale {source_id} prefix", minimum=1)
        _require(prefix_end <= min(expected_group["decision_frames"]),
                 f"physical causal-scale {source_id} prefix is non-causal")
        cam_pose_sha = _sha(raw_record.get("cam_pose_prefix_sha256"),
                            f"physical causal-scale {source_id} pose prefix")
        rgb = raw_record.get("rgb_prefix")
        _require(isinstance(rgb, Mapping),
                 f"physical causal-scale {source_id} RGB prefix is malformed")
        rgb_sha = _sha(rgb.get("content_sequence_sha256"),
                       f"physical causal-scale {source_id} RGB prefix")
        debug = raw_record.get("debug")
        _require(isinstance(debug, Mapping),
                 f"physical causal-scale {source_id} quality is missing")
        n_frames = _integer(debug.get("n_frames"),
                            f"physical causal-scale {source_id} n_frames",
                            minimum=1)
        n_valid = _integer(debug.get("n_valid"),
                           f"physical causal-scale {source_id} n_valid",
                           minimum=1)
        _require(n_valid <= n_frames
                 and n_valid >= max(3, n_frames // 8),
                 f"physical causal-scale {source_id} support is invalid")
        h_est = _finite(debug.get("h_est"),
                        f"physical causal-scale {source_id} h_est",
                        minimum=1e-12)
        h_iqr = _finite(debug.get("h_iqr"),
                        f"physical causal-scale {source_id} h_iqr",
                        minimum=0.0)
        configuration = configuration_object
        bias = _finite(configuration.get("bias_correction"),
                       "physical causal-scale bias", minimum=1e-12)
        height = _finite(raw_record.get("camera_height_m"),
                         f"physical causal-scale {source_id} camera height",
                         minimum=1e-12)
        ground_h = _finite(raw_record.get("ground_h_est_raw"),
                           f"physical causal-scale {source_id} ground height",
                           minimum=1e-12)
        _close(h_est, ground_h,
               f"physical causal-scale {source_id} debug ground height")
        unclamped = bias * height / h_est
        scale_min = _finite(configuration.get("scale_min"),
                            "physical causal-scale minimum", minimum=1e-12)
        scale_max = _finite(configuration.get("scale_max"),
                            "physical causal-scale maximum", minimum=1e-12)
        _require(scale_min < scale_max,
                 "physical causal-scale range is invalid")
        _close(scale, min(max(unclamped, scale_min), scale_max),
               f"physical causal-scale {source_id} pinned formula")
        result[source_id] = {
            "record": raw_record,
            "record_sha256": sha256_bytes(canonical_json_bytes(raw_record)),
            "metric_scale_m_per_raw": scale,
            "prefix_end_frame_exclusive": prefix_end,
            "cam_pose_prefix_sha256": cam_pose_sha,
            "rgb_prefix_content_sequence_sha256": rgb_sha,
            "valid_frame_ratio": n_valid / n_frames,
            "relative_h_iqr": h_iqr / h_est,
            "clamped": int(not math.isclose(
                scale, unclamped, rel_tol=1e-6, abs_tol=1e-6)),
        }
    _require(set(result) == set(expected),
             "physical causal-scale records do not exactly cover manifest episodes")
    return result


def _manifest_sample_index(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    schema = manifest.get("schema_version")
    _require(schema in SUPPORTED_MANIFEST_SCHEMAS, f"unsupported manifest {schema!r}")
    routing = manifest.get("flow_cache_routing")
    _require(isinstance(routing, Mapping), "manifest lacks routed-cache provenance")
    samples = manifest.get("samples")
    _require(isinstance(samples, list) and bool(samples), "manifest samples are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for index, sample in enumerate(samples):
        _require(isinstance(sample, Mapping), f"manifest sample {index} is malformed")
        sample_id = _string(sample.get("sample_id"), f"manifest sample {index}.id")
        _require(sample_id not in result, f"duplicate manifest sample {sample_id}")
        decision = _integer(
            sample.get("decision_frame"), f"manifest {sample_id}.decision", minimum=1
        )
        role = sample.get("goal_role", "B")
        _require(role in ("B", "C"), f"manifest {sample_id} has invalid goal role")
        if role == "C":
            _require(
                schema == "nlsr_v2_multistage_expert_candidate_manifest_v1",
                f"Goal-C sample {sample_id} requires the multistage manifest",
            )
        for field in ("causal_prefix", "navdp_fifo", "goal"):
            _require(
                isinstance(sample.get(field), Mapping),
                f"manifest {sample_id}.{field} is malformed",
            )
        prefix = sample["causal_prefix"]
        if "frame_count" in prefix:
            _require(
                prefix["frame_count"] == decision,
                f"manifest {sample_id} prefix is not decision-exclusive",
            )
        _sha(prefix.get("causal_prefix_sha256"), f"manifest {sample_id} prefix SHA")
        _sha(sample["navdp_fifo"].get("fifo_sha256"), f"manifest {sample_id} FIFO SHA")
        _sha(sample["goal"].get("content_sha256"), f"manifest {sample_id} goal SHA")
        _string(sample.get("scene"), f"manifest {sample_id}.scene")
        _string(
            sample.get("source_episode_id"),
            f"manifest {sample_id}.source_episode_id",
        )
        _require(
            sample.get("split_role") in ("train", "development"),
            f"manifest {sample_id} split role is not train/development",
        )
        result[sample_id] = sample
    return result


def _validate_candidate(
    raw: object,
    *,
    record_label: str,
    rank: int,
    decision_frame: int,
    metric_scale: float,
    neighbor_offsets: Sequence[int],
    scale_quality: Sequence[float],
) -> Mapping[str, Any]:
    candidate = _mapping(raw, CANDIDATE_KEYS, f"{record_label}.candidates[{rank}]")
    _require(
        _integer(candidate["candidate_rank"], f"{record_label}.candidate_rank", minimum=0)
        == rank,
        f"{record_label} candidate ranks must be contiguous and ordered",
    )
    anchor = _integer(
        candidate["anchor_frame_index"], f"{record_label}.anchor", minimum=0
    )
    _require(anchor < decision_frame, f"{record_label} candidate uses a future anchor")
    neighbor_frames = [anchor + int(offset) for offset in neighbor_offsets]
    _require(min(neighbor_frames) >= 0 and max(neighbor_frames) < decision_frame,
             f"{record_label} candidate neighbor offsets are not decision-exclusive")
    append_end = _integer(
        candidate["goal_append_prefix_end_frame_exclusive"],
        f"{record_label}.goal_append_prefix_end",
        minimum=1,
    )
    _require(
        append_end == anchor + 1 and append_end <= decision_frame,
        f"{record_label} goal append is not bound to the causal anchor prefix",
    )
    _sha(candidate["goal_append_input_sha256"], f"{record_label}.goal append SHA")
    dino = _finite(candidate["dino_cosine"], f"{record_label}.dino", minimum=-1.0, maximum=1.0)
    features = _vector(
        candidate["phase_b_input_features"],
        len(PHASE_B_FEATURE_NAMES),
        f"{record_label}.phase_b_input_features",
    )
    raw_xz = _vector(
        candidate["native_relative_translation_xz_raw"], 2, f"{record_label}.raw_xz"
    )
    scaled_xz = _vector(
        candidate["scaled_relative_translation_xz_m"], 2, f"{record_label}.scaled_xz"
    )
    raw_forward_left = _vector(
        candidate["raw_relative_forward_left_m"], 2, f"{record_label}.raw_forward_left"
    )
    _vector(
        candidate["corrected_relative_forward_left_m"],
        2,
        f"{record_label}.corrected_forward_left",
    )
    variance = _vector(
        candidate["translation_variance_forward_left_m2"],
        2,
        f"{record_label}.translation_variance",
    )
    _require(
        all(value > 0.0 for value in variance),
        f"{record_label} predictive variance must be strictly positive",
    )
    for axis in range(2):
        _close(
            scaled_xz[axis],
            metric_scale * raw_xz[axis],
            f"{record_label} x-z metric scaling axis {axis}",
        )
    # LingBot x-z -> NavDP [forward,left] is [z,-x].
    _close(raw_forward_left[0], scaled_xz[1], f"{record_label} forward axis")
    _close(raw_forward_left[1], -scaled_xz[0], f"{record_label} left axis")

    # LingBot/VGGT confidence is a positive model score, not a calibrated
    # probability; real collector rows commonly exceed 1.0.
    goal_confidence = _finite(
        candidate["goal_depth_confidence_mean"],
        f"{record_label}.goal depth confidence",
        minimum=0.0,
    )
    candidate_confidence = _finite(
        candidate["candidate_depth_confidence_mean"],
        f"{record_label}.candidate depth confidence",
        minimum=0.0,
    )
    _close(features[0], dino, f"{record_label} DINO feature")
    _close(features[1], metric_scale, f"{record_label} scale feature")
    _require(features[2] > 0.0, f"{record_label} depth scale must be positive")
    _require(0.0 <= features[3] <= 1.0, f"{record_label} cloud overlap is invalid")
    _require(
        all(value >= 0.0 for value in features[4:7]),
        f"{record_label} distance/refinement features must be non-negative",
    )
    _close(features[7], goal_confidence, f"{record_label} goal confidence feature")
    _close(
        features[8], candidate_confidence, f"{record_label} candidate confidence feature"
    )
    _close(features[9], raw_forward_left[0], f"{record_label} raw forward feature")
    _close(features[10], raw_forward_left[1], f"{record_label} raw left feature")
    _close(
        features[11],
        math.hypot(*raw_forward_left),
        f"{record_label} raw distance feature",
    )
    # External causal scale has its own trained category.  It must never be
    # silently represented as `other`, which would recreate the train/runtime
    # distribution shift that feature schema v2 removes.
    _require(
        features[12:17] == [0.0, 0.0, 0.0, 1.0, 0.0],
        f"{record_label} metric-scale one-hot does not encode external causal scale",
    )
    _close(features[17], float(scale_quality[0]),
           f"{record_label} external scale valid-frame ratio")
    _close(features[18], float(scale_quality[1]),
           f"{record_label} external scale relative h_iqr")
    _close(features[19], float(scale_quality[2]),
           f"{record_label} external scale clamp flag")
    _probability(candidate["rank_probability"], f"{record_label}.rank_probability")
    _probability(
        candidate["candidate_validity_probability"],
        f"{record_label}.candidate_validity_probability",
    )
    _probability(candidate["set_probability"], f"{record_label}.set_probability")
    return candidate


def validate_phase_b_deployment_inference(
    *,
    artifact: Mapping[str, Any],
    artifact_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    pins: InferencePins,
) -> ValidatedPhaseBInference:
    """Validate a complete, deployment-only Phase-B inference artifact."""

    _sha(artifact_sha256, "inference artifact SHA256")
    _sha(manifest_sha256, "manifest SHA256")
    _require(
        sha256_bytes(canonical_json_bytes(artifact)) == artifact_sha256,
        "in-memory inference artifact differs from its canonical SHA pin",
    )
    _require(
        sha256_bytes(canonical_json_bytes(manifest)) == manifest_sha256,
        "in-memory manifest differs from its canonical SHA pin",
    )
    top = _mapping(artifact, TOP_LEVEL_KEYS, "inference artifact")
    _require(top["schema_version"] == SCHEMA_VERSION, "inference schema changed")
    _require(top["purpose"] == PURPOSE, "inference purpose/boundary changed")
    _scan_forbidden_keys(top)

    samples = _manifest_sample_index(manifest)
    checkpoint = _load_physical_checkpoint(pins)
    _load_and_validate_approval(pins, checkpoint)
    physical_scale_records = _physical_scale_record_index(
        pins,
        checkpoint=checkpoint,
        samples=samples,
        manifest_sha256=manifest_sha256,
    )
    provenance = _mapping(top["provenance"], PROVENANCE_KEYS, "inference provenance")
    expected_provenance = {
        "input_manifest_sha256": manifest_sha256,
        "input_manifest_schema_version": manifest["schema_version"],
        "flow_route_artifact_sha256": pins.flow_route_artifact_sha256,
        "causal_scale_artifact_sha256": pins.causal_scale_artifact_sha256,
        "phase_b_checkpoint_sha256": pins.phase_b_checkpoint_sha256,
        "deployment_approval_artifact_sha256": (
            pins.deployment_approval_artifact_sha256
        ),
        "inference_producer_source_sha256": pins.inference_producer_source_sha256,
        "inference_configuration_sha256": pins.inference_configuration_sha256,
        "dino_encoder_checkpoint_sha256": pins.dino_encoder_checkpoint_sha256,
        "dino_feature_producer_sha256": pins.dino_feature_producer_sha256,
        "lingbot_commit": pins.lingbot_commit,
        "lingbot_weights_sha256": pins.lingbot_weights_sha256,
        "lingbot_stream_source_sha256": pins.lingbot_stream_source_sha256,
        "phase_b_checkpoint_model_kind": CHECKPOINT_MODEL_KIND,
        "phase_b_checkpoint_schema_version": PHASE_B_CHECKPOINT_SCHEMA_VERSION,
        "phase_b_feature_schema_version": PHASE_B_FEATURE_SCHEMA_VERSION,
        "pose_convention": POSE_CONVENTION,
        "privileged_input_policy": PRIVILEGED_INPUT_POLICY,
    }
    for field, expected in expected_provenance.items():
        _require(
            provenance[field] == expected,
            f"inference provenance {field} differs from its external pin",
        )
    _require(
        provenance["phase_b_checkpoint_deployment_approved"] is True,
        "Phase-B checkpoint is not deployment-approved",
    )
    _require(
        provenance["phase_b_checkpoint_deployment_input_contract_approved"] is True
        and provenance["phase_b_external_causal_scale_coverage_approved"] is True,
        "Phase-B checkpoint lacks train/development external-scale coverage",
    )
    _require(
        provenance["phase_b_feature_names"] == list(PHASE_B_FEATURE_NAMES),
        "Phase-B feature order differs from the checkpoint contract",
    )
    _require(
        provenance["phase_b_feature_names_sha256"] == FEATURE_NAMES_SHA256,
        "Phase-B feature-name digest changed",
    )
    routing = manifest.get("flow_cache_routing")
    assert isinstance(routing, Mapping)
    _require(
        routing.get("artifact_sha256") == pins.flow_route_artifact_sha256,
        "manifest routed-cache artifact differs from the inference pin",
    )

    configuration = _mapping(
        top["configuration"], CONFIGURATION_KEYS, "inference configuration"
    )
    _require(
        sha256_bytes(canonical_json_bytes(configuration))
        == pins.inference_configuration_sha256,
        "inference configuration content differs from its external pin",
    )
    shortlist_size = _integer(
        configuration["candidate_shortlist_size"], "candidate shortlist size", minimum=1
    )
    temporal_minimum_gap = _integer(
        configuration["temporal_minimum_gap_frames"],
        "temporal minimum gap",
        minimum=0,
    )
    offsets = configuration["neighbor_offsets"]
    _require(
        isinstance(offsets, list)
        and bool(offsets)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        and len(offsets) == len(set(offsets))
        and 0 in offsets,
        "neighbor offsets must be unique integers containing zero",
    )
    match_threshold = _probability(configuration["match_threshold"], "match threshold")
    ensemble_member_count = _integer(
        configuration["ensemble_member_count"], "ensemble member count", minimum=1)
    _validate_checkpoint_states(checkpoint, ensemble_member_count)
    checkpoint_configuration = checkpoint.get("config")
    assert isinstance(checkpoint_configuration, Mapping)
    _close(
        _finite(checkpoint_configuration.get("match_threshold"),
                "checkpoint match threshold", minimum=0.0, maximum=1.0),
        match_threshold,
        "checkpoint/inference match threshold",
    )
    _require(
        configuration["candidate_selection_semantics"]
        == CANDIDATE_SELECTION_SEMANTICS,
        "candidate selection semantics changed",
    )
    _require(
        configuration["ensemble_semantics"] == ENSEMBLE_SEMANTICS,
        "ensemble semantics changed",
    )

    raw_records = top["records"]
    _require(isinstance(raw_records, list) and bool(raw_records), "inference records missing")
    records: dict[str, Mapping[str, Any]] = {}
    candidate_count = 0
    matched_count = 0
    for record_index, raw in enumerate(raw_records):
        record = _mapping(raw, RECORD_KEYS, f"inference record {record_index}")
        sample_id = _string(record["sample_id"], f"record {record_index}.sample_id")
        _require(sample_id in samples, f"inference references unknown sample {sample_id}")
        _require(sample_id not in records, f"duplicate inference record {sample_id}")
        sample = samples[sample_id]
        expected_role = sample.get("goal_role", "B")
        expected = {
            "scene": sample["scene"],
            "source_episode_id": sample["source_episode_id"],
            "goal_role": expected_role,
            "decision_frame": sample["decision_frame"],
            "causal_prefix_sha256": sample["causal_prefix"]["causal_prefix_sha256"],
            "navdp_fifo_sha256": sample["navdp_fifo"]["fifo_sha256"],
            "goal_sha256": sample["goal"]["content_sha256"],
        }
        for field, value in expected.items():
            _require(
                record[field] == value,
                f"inference record {sample_id} changed manifest field {field}",
            )
        decision = int(record["decision_frame"])
        _sha(record["cam_pose_prefix_sha256"], f"{sample_id} camera-pose prefix SHA")
        _require(
            _integer(
                record["cam_pose_prefix_frame_count"],
                f"{sample_id} camera-pose prefix count",
                minimum=1,
            )
            == decision,
            f"{sample_id} camera-pose prefix is not decision-exclusive",
        )
        scale = _finite(
            record["metric_scale_m_per_raw"],
            f"{sample_id} metric scale",
            minimum=1e-12,
        )
        _require(
            record["metric_scale_source"] == METRIC_SCALE_SOURCE,
            f"{sample_id} does not use the external causal metric scale",
        )
        scale_record = physical_scale_records[str(sample["source_episode_id"])]
        _close(scale, float(scale_record["metric_scale_m_per_raw"]),
               f"{sample_id} physical external scale")
        scale_witness = {
            "external_scale_record_sha256": scale_record["record_sha256"],
            "external_scale_prefix_end_frame_exclusive": scale_record[
                "prefix_end_frame_exclusive"],
            "external_scale_cam_pose_prefix_sha256": scale_record[
                "cam_pose_prefix_sha256"],
            "external_scale_rgb_prefix_content_sequence_sha256": scale_record[
                "rgb_prefix_content_sequence_sha256"],
        }
        for field, value in scale_witness.items():
            _require(record[field] == value,
                     f"{sample_id} {field} differs from physical scale record")
        _require(int(scale_record["prefix_end_frame_exclusive"]) <= decision,
                 f"{sample_id} external scale prefix crosses decision")
        raw_candidates = record["candidates"]
        _require(
            isinstance(raw_candidates, list)
            and 1 <= len(raw_candidates) <= shortlist_size,
            f"{sample_id} candidate count violates the frozen shortlist",
        )
        candidates = [
            _validate_candidate(
                candidate,
                record_label=sample_id,
                rank=rank,
                decision_frame=decision,
                metric_scale=scale,
                neighbor_offsets=offsets,
                scale_quality=(
                    float(scale_record["valid_frame_ratio"]),
                    float(scale_record["relative_h_iqr"]),
                    float(scale_record["clamped"]),
                ),
            )
            for rank, candidate in enumerate(raw_candidates)
        ]
        anchors = [int(candidate["anchor_frame_index"]) for candidate in candidates]
        _require(len(anchors) == len(set(anchors)), f"{sample_id} anchors are duplicated")
        for left_index, left in enumerate(anchors):
            for right in anchors[left_index + 1:]:
                _require(abs(left - right) >= temporal_minimum_gap,
                         f"{sample_id} anchors violate temporal minimum gap")
        rank_probabilities = [float(candidate["rank_probability"]) for candidate in candidates]
        for left, right in zip(rank_probabilities, rank_probabilities[1:]):
            _require(
                left + _PROBABILITY_TOLERANCE >= right,
                f"{sample_id} candidates are not ordered by rank probability",
            )
        _close(
            sum(rank_probabilities),
            1.0,
            f"{sample_id} rank probabilities",
            _PROBABILITY_TOLERANCE,
        )
        global_no_match = _probability(
            record["global_no_match_probability"], f"{sample_id} global no-match"
        )
        usable = _probability(record["usable_match_probability"], f"{sample_id} usable match")
        dustbin = _probability(record["dustbin_probability"], f"{sample_id} dustbin")
        maximum_validity = max(
            float(candidate["candidate_validity_probability"])
            for candidate in candidates
        )
        _close(
            usable,
            (1.0 - global_no_match) * maximum_validity,
            f"{sample_id} usable-match factorization",
            _PROBABILITY_TOLERANCE,
        )
        _close(dustbin, 1.0 - usable, f"{sample_id} dustbin probability")
        set_probabilities = [float(candidate["set_probability"]) for candidate in candidates]
        for rank, (set_probability, rank_probability) in enumerate(
            zip(set_probabilities, rank_probabilities)
        ):
            _close(
                set_probability,
                usable * rank_probability,
                f"{sample_id} candidate {rank} set probability",
                _PROBABILITY_TOLERANCE,
            )
        _close(
            sum(set_probabilities) + dustbin,
            1.0,
            f"{sample_id} complete set probability",
            _PROBABILITY_TOLERANCE,
        )
        selected = record["selected_anchor_frame_index"]
        if usable >= match_threshold:
            _require(
                selected == anchors[0],
                f"{sample_id} selected anchor is not the top causal candidate",
            )
            matched_count += 1
        else:
            _require(selected is None, f"{sample_id} rejected match retains an anchor")
        records[sample_id] = record
        candidate_count += len(candidates)
    _require(
        set(records) == set(samples),
        "inference records do not exactly cover the pinned manifest",
    )

    summary = _mapping(top["summary"], SUMMARY_KEYS, "inference summary")
    role_b = sum(sample.get("goal_role", "B") == "B" for sample in samples.values())
    role_c = sum(sample.get("goal_role", "B") == "C" for sample in samples.values())
    expected_summary = {
        "record_count": len(records),
        "goal_b_record_count": role_b,
        "goal_c_record_count": role_c,
        "activated_match_record_count": matched_count,
        "rejected_no_match_record_count": len(records) - matched_count,
        "candidate_count": candidate_count,
        "future_frames_consumed": 0,
        "privileged_inputs_consumed": 0,
    }
    _require(summary == expected_summary, "inference summary disagrees with records")
    return ValidatedPhaseBInference(
        artifact=artifact,
        records_by_sample=records,
        manifest_samples=samples,
        match_threshold=match_threshold,
    )


__all__ = [
    "CANDIDATE_KEYS",
    "CANDIDATE_SELECTION_SEMANTICS",
    "CHECKPOINT_MODEL_KIND",
    "DEPLOYMENT_APPROVAL_KEYS",
    "DEPLOYMENT_APPROVAL_PURPOSE",
    "DEPLOYMENT_APPROVAL_SCHEMA_VERSION",
    "ENSEMBLE_SEMANTICS",
    "InferencePins",
    "METRIC_SCALE_SOURCE",
    "PHASE_B_FEATURE_NAMES",
    "POSE_CONVENTION",
    "PRIVILEGED_INPUT_POLICY",
    "PURPOSE",
    "PhaseBInferenceContractError",
    "SCHEMA_VERSION",
    "ValidatedPhaseBInference",
    "canonical_json_bytes",
    "load_pinned_canonical_json",
    "sha256_bytes",
    "sha256_file",
    "validate_phase_b_deployment_inference",
]
