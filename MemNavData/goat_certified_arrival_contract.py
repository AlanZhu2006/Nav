#!/usr/bin/env python3
"""Frozen decision boundary for certified GOAT ImageGoal arrival.

The geometry service measures the current-to-goal translation but does not
know why it was queried.  The GOAT runner owns the native NavDP zero proposal
and combines the two here.  This pure module is deliberately independent of
Habitat, Torch, and model code so the semantic STOP boundary is easy to test
and impossible to retune through a runtime argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


SCHEMA_VERSION = "goat_certified_arrival_contract_v1_20260815"
ARRIVAL_DISTANCE_THRESHOLD_M = 0.075
MINIMUM_CAUSAL_STREAM_FRAMES = 64
CAUSAL_SCALE_CONFIG = {
    "confidence_quantile": 0.5,
    "pixel_stride": 4,
    "histogram_bins": 60,
    "peak_threshold": 0.3,
    "bias_correction": 1.15,
    "minimum_scale": 0.8,
    "maximum_scale": 6.0,
}
TRAIN_REPORT_SHA256 = (
    "13f265b200f02c877557bdc18a846688274961ddc451ead463dbcb319d528373"
)
TRAIN_VERIFICATION_SHA256 = (
    "ffb2576ef25f1a0ff571d66640ec7cddd611417858a35d4e15de1c6ef2ea7dfd"
)


@dataclass(frozen=True)
class ArrivalEvidence:
    native_zero_proposal: bool
    stream_frame_count: int
    certificate_accepted: bool
    predicted_distance_m: float | None
    metric_scale_available: bool


def contract_receipt() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "distance_threshold_m": ARRIVAL_DISTANCE_THRESHOLD_M,
        "minimum_causal_stream_frames": MINIMUM_CAUSAL_STREAM_FRAMES,
        "causal_scale_config": dict(CAUSAL_SCALE_CONFIG),
        "train_report_sha256": TRAIN_REPORT_SHA256,
        "train_verification_sha256": TRAIN_VERIFICATION_SHA256,
        "zero_without_certificate_semantics": "abstain_never_stop",
    }


def decide_subtask_stop(evidence: ArrivalEvidence) -> dict[str, object]:
    """Return the frozen semantic STOP decision and an auditable reason."""

    if type(evidence.native_zero_proposal) is not bool:
        raise TypeError("native_zero_proposal must be bool")
    if type(evidence.certificate_accepted) is not bool:
        raise TypeError("certificate_accepted must be bool")
    if type(evidence.metric_scale_available) is not bool:
        raise TypeError("metric_scale_available must be bool")
    if (type(evidence.stream_frame_count) is not int
            or evidence.stream_frame_count < 0):
        raise ValueError("stream_frame_count must be a non-negative integer")

    distance = evidence.predicted_distance_m
    if distance is not None:
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise TypeError("predicted_distance_m must be numeric or None")
        distance = float(distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(
                "predicted_distance_m must be finite and non-negative")

    if not evidence.native_zero_proposal:
        reason = "native_zero_trigger_absent"
    elif evidence.stream_frame_count < MINIMUM_CAUSAL_STREAM_FRAMES:
        reason = "causal_scale_prefix_incomplete"
    elif not evidence.certificate_accepted:
        reason = "geometry_certificate_rejected"
    elif not evidence.metric_scale_available:
        reason = "causal_metric_scale_unavailable"
    elif distance is None:
        reason = "predicted_distance_unavailable"
    elif distance > ARRIVAL_DISTANCE_THRESHOLD_M:
        reason = "predicted_distance_above_frozen_threshold"
    else:
        reason = "certified_arrival"

    authorized = reason == "certified_arrival"
    return {
        "schema_version": SCHEMA_VERSION,
        "authorized_subtask_stop": authorized,
        "reason": reason,
        "evidence": asdict(evidence),
        "contract": contract_receipt(),
    }
