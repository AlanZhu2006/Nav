#!/usr/bin/env python3
"""Single versioned authority for Phase-B deployment features.

The feature order and its digest are part of the checkpoint ABI.  Keeping the
definition in one small module prevents the trainer and deployment validator
from independently (and subtly differently) serializing the same list.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping, Sequence


CHECKPOINT_SCHEMA_VERSION = 3
FEATURE_SCHEMA_VERSION = (
    "lingbot_native_phase_b_features_v3_external_causal_scale_quality"
)
CHECKPOINT_MODEL_KIND = (
    "lingbot_native_verify_rank_true_nomatch_translation_uncertainty"
)
EXTERNAL_CAUSAL_SCALE_SOURCE = "external_causal_first_prefix_v1"

SCALAR_INPUT_COLUMNS = (
    "dino_cosine",
    "metric_scale_m_per_raw",
    "depth_scale_raw",
    "cloud_overlap_f1_center",
    "anchor_goal_distance_norm_center",
    "goal_refine_translation_norm_median",
    "goal_refine_rotation_deg_median",
    "goal_depth_confidence_mean",
    "candidate_depth_confidence_mean",
)
PREDICTED_POSE_FEATURE_NAMES = (
    "lingbot_predicted_forward_m",
    "lingbot_predicted_lateral_m",
    "lingbot_predicted_distance_m",
)
METRIC_SCALE_SOURCES = (
    "cached_ground_anchored",
    "runtime_ground_anchored",
    "pooled_fallback",
    EXTERNAL_CAUSAL_SCALE_SOURCE,
)
EXTERNAL_SCALE_QUALITY_COLUMNS = (
    "external_scale_valid_frame_ratio",
    "external_scale_relative_h_iqr",
    "external_scale_clamped",
)
FEATURE_NAMES = (
    *SCALAR_INPUT_COLUMNS,
    *PREDICTED_POSE_FEATURE_NAMES,
    *(f"metric_scale_source={source}" for source in METRIC_SCALE_SOURCES),
    "metric_scale_source=other",
    *EXTERNAL_SCALE_QUALITY_COLUMNS,
)
FEATURE_DIMENSION = len(FEATURE_NAMES)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical bytes used by every Phase-B ABI digest."""

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


FEATURE_NAMES_SHA256 = hashlib.sha256(
    canonical_json_bytes(list(FEATURE_NAMES))
).hexdigest()


def validate_checkpoint_metadata(
    artifact: Mapping[str, object],
    *,
    require_deployment_input_contract: bool,
) -> None:
    """Fail closed for legacy 16/17-D or otherwise shifted checkpoints."""

    if artifact.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Phase-B checkpoint schema is obsolete")
    if artifact.get("model_kind") != CHECKPOINT_MODEL_KIND:
        raise RuntimeError("Phase-B checkpoint model kind is incompatible")
    if artifact.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError("Phase-B feature schema is obsolete")
    if artifact.get("input_dim") != FEATURE_DIMENSION:
        raise RuntimeError(
            "Phase-B checkpoint input dimension is incompatible; legacy "
            "16/17-D checkpoints are forbidden"
        )
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("Phase-B checkpoint feature order is incompatible")
    if artifact.get("feature_names_sha256") != FEATURE_NAMES_SHA256:
        raise RuntimeError("Phase-B checkpoint feature-name digest is incompatible")
    if artifact.get("metric_scale_source_categories") != list(
        METRIC_SCALE_SOURCES
    ):
        raise RuntimeError("Phase-B checkpoint scale-source schema is incompatible")
    for field in (
        "train_artifact_identity_sha256",
        "development_artifact_identity_sha256",
        "train_audit_sha256",
        "development_audit_sha256",
    ):
        value = artifact.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(
                f"Phase-B checkpoint {field} is missing or incompatible"
            )
    for field in ("normalization_mean", "normalization_scale"):
        values = artifact.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RuntimeError(f"Phase-B checkpoint {field} is incompatible")
        if len(values) != FEATURE_DIMENSION:
            raise RuntimeError(f"Phase-B checkpoint {field} is incompatible")
        if not all(isinstance(value, (int, float))
                   and not isinstance(value, bool) for value in values):
            raise RuntimeError(f"Phase-B checkpoint {field} is incompatible")
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Phase-B checkpoint {field} is incompatible"
            ) from error
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"Phase-B checkpoint {field} is incompatible")
        if field == "normalization_scale" and not all(value > 0.0 for value in numeric):
            raise RuntimeError("Phase-B checkpoint normalization_scale must be positive")
    coverage = artifact.get("external_causal_scale_coverage")
    if require_deployment_input_contract:
        if (
            artifact.get("deployment_input_contract_approved") is not True
            or not isinstance(coverage, Mapping)
            or coverage.get("approved") is not True
            or coverage.get("train_exact_coverage_approved") is not True
            or coverage.get("development_exact_coverage_approved") is not True
        ):
            raise RuntimeError(
                "Phase-B checkpoint lacks exact train/development external-scale coverage"
            )


__all__ = [
    "CHECKPOINT_MODEL_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "EXTERNAL_CAUSAL_SCALE_SOURCE",
    "EXTERNAL_SCALE_QUALITY_COLUMNS",
    "FEATURE_DIMENSION",
    "FEATURE_NAMES",
    "FEATURE_NAMES_SHA256",
    "FEATURE_SCHEMA_VERSION",
    "METRIC_SCALE_SOURCES",
    "PREDICTED_POSE_FEATURE_NAMES",
    "SCALAR_INPUT_COLUMNS",
    "canonical_json_bytes",
    "validate_checkpoint_metadata",
]
