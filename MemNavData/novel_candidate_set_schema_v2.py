"""Strict data contract for Novel graph/frontier candidate sets.

This module validates JSON-like records only.  It neither imports Habitat nor
collects features.  Its purpose is to reject explicit privileged columns and
malformed set semantics before a collector artifact reaches a trainer.  An
allow-list cannot prove that an otherwise legal tensor was built causally;
builder-code and prefix-content audits remain mandatory.

Deployment features use an exact allow-list.  Privileged geodesic, GT pose,
oracle, and rollout outcome fields are permitted only under label dictionaries.
Every set contains native candidate 0 and one final dustbin candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "novel_candidate_set_v2"
NATIVE_CANDIDATE_ID = "native"
DUSTBIN_CANDIDATE_ID = "dustbin"
CANDIDATE_TYPES = ("native", "memory_graph", "frontier", "dustbin")
TRAINABLE_SPLIT_ROLES = ("train", "development")
MAX_RESIDUAL_CANDIDATES = 32
FEATURE_PRESENCE_MASK_ORDER = (
    "goal_patch_relation",
    "goal_temporal_relation",
    "local_map_relation",
    "native_proposal_relation",
    "pose_uncertainty",
    "depth_confidence",
    "clearance",
)
FEATURE_PRESENCE_MASK_SIZE = len(FEATURE_PRESENCE_MASK_ORDER)
FEATURE_PRESENCE_MASK_FIELDS = (
    ("goal_patch_relation",),
    ("goal_temporal_relation",),
    ("local_map_relation",),
    ("native_proposal_relation",),
    ("pose_translation_p90_m", "pose_yaw_p90_deg"),
    ("depth_confidence_mean",),
    ("clearance_lower_m",),
)
SET_FEATURE_PRESENCE_MASK_ORDER = (
    "native_stagnation_plans",
    "graph_node_count",
    "graph_edge_count",
    "graph_age_frames",
    "memory_candidate_count",
    "frontier_candidate_count",
)
SET_FEATURE_PRESENCE_MASK_SIZE = len(SET_FEATURE_PRESENCE_MASK_ORDER)
USEFUL_ADVANTAGE_MARGIN_M = 0.25
REGRESSION_ADVANTAGE_MARGIN_M = 0.25
PROPOSAL_PROXY_POSITIVE_MARGIN_M = 0.0

TOP_LEVEL_KEYS = frozenset({
    "schema_version",
    "provenance",
    "set_features",
    "candidates",
    "set_labels",
})

PROVENANCE_KEYS = frozenset({
    "dataset_id",
    "scene_id",
    "episode_id",
    "session_id",
    "group_id",
    "goal_epoch",
    "state_id",
    "state_source",
    "goal_source_episode_id",
    "plan_index",
    "prefix_frames",
    "prefix_sha256",
    "goal_sha256",
    "navdp_fifo_sha256",
    "split_role",
    "split_sha256",
    "source_policy_sha256",
    "candidate_generator_sha256",
    "feature_builder_sha256",
    "rollout_labeler_sha256",
    "environment_id",
    "navmesh_sha256",
})

# These are the only set-level values a trainer may consume as inputs.
SET_FEATURE_KEYS = frozenset({
    "feature_presence_mask",
    "native_stagnation_plans",
    "graph_node_count",
    "graph_edge_count",
    "graph_age_frames",
    "memory_candidate_count",
    "frontier_candidate_count",
})

# Vector-valued relation fields preserve patch/temporal/local-map information
# without granting arbitrary collector columns access to the model.
CANDIDATE_FEATURE_KEYS = frozenset({
    "candidate_type_onehot",
    "goal_patch_relation",
    "goal_temporal_relation",
    "local_map_relation",
    "native_proposal_relation",
    "feature_presence_mask",
    "subgoal_forward_m",
    "subgoal_left_m",
    "graph_path_m",
    "graph_hops",
    "frontier_boundary_m",
    "frontier_novelty_m",
    "pose_translation_p90_m",
    "pose_yaw_p90_deg",
    "depth_confidence_mean",
    "clearance_lower_m",
})

CANDIDATE_LABEL_KEYS = frozenset({
    "geodesic_progress_h8_m",
    "geodesic_progress_h24_m",
    "advantage_h24_m",
    "harm",
    "useful",
    "reachable",
    "collision_h8",
    "regression_h24",
    "proposal_proxy_progress_m",
    "proposal_proxy_reachable",
    "proposal_proxy_positive",
    "proposal_proxy_label_valid",
    "rollout_label_valid",
    "teacher_covisibility",
    "covisibility_label_valid",
    "pose_residual_forward_m",
    "pose_residual_left_m",
    "pose_residual_yaw_rad",
    "pose_label_valid",
})

SET_LABEL_KEYS = frozenset({
    "global_match",
    "strict_no_match",
    "ambiguous",
    "candidate_set_has_positive",
    "candidate_universe_has_positive",
    "candidate_coverage_miss",
    "coverage_label_valid",
    "proposal_proxy_set_has_positive",
    "proposal_proxy_universe_has_positive",
    "proposal_proxy_coverage_miss",
    "proposal_proxy_coverage_label_valid",
    "oracle_best_candidate_id",
})

PRIVILEGED_LABEL_DENY_LIST = frozenset(
    set(CANDIDATE_LABEL_KEYS)
    | set(SET_LABEL_KEYS)
    | {
        "current_goal_geodesic_m",
        "goal_world_x",
        "goal_world_y",
        "goal_world_z",
        "habitat_pose",
        "pathfinder_reachable",
        "target_relative_pose",
    }
)

PRIVILEGED_FEATURE_DENY_FRAGMENTS = (
    "geodesic",
    "pathfinder",
    "goal_world",
    "habitat",
    "oracle",
    "target_",
    "gt_",
    "label",
    "success",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_KEYS = frozenset({
    "candidate_id", "candidate_type", "features", "labels"})
_CANDIDATE_BOOL_LABELS = frozenset({
    "harm",
    "useful",
    "reachable",
    "collision_h8",
    "regression_h24",
    "proposal_proxy_reachable",
    "proposal_proxy_positive",
    "proposal_proxy_label_valid",
    "rollout_label_valid",
    "covisibility_label_valid",
    "pose_label_valid",
})
_SET_BOOL_LABELS = frozenset({
    "global_match",
    "strict_no_match",
    "ambiguous",
    "candidate_set_has_positive",
    "candidate_universe_has_positive",
    "candidate_coverage_miss",
    "coverage_label_valid",
    "proposal_proxy_set_has_positive",
    "proposal_proxy_universe_has_positive",
    "proposal_proxy_coverage_miss",
    "proposal_proxy_coverage_label_valid",
})

_NONNEGATIVE_SET_FEATURES = frozenset(
    SET_FEATURE_KEYS - {"feature_presence_mask"})
_NONNEGATIVE_CANDIDATE_FEATURES = frozenset({
    "graph_path_m",
    "graph_hops",
    "frontier_boundary_m",
    "frontier_novelty_m",
    "pose_translation_p90_m",
    "pose_yaw_p90_deg",
    "depth_confidence_mean",
    "clearance_lower_m",
})


class CandidateSetValidationError(ValueError):
    """Raised when a candidate-set artifact must fail closed."""


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    location: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CandidateSetValidationError(f"{location} must be a mapping")
    actual = frozenset(map(str, value.keys()))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise CandidateSetValidationError(
            f"{location} keys differ: missing={missing} extra={extra}")
    return value


def _require_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateSetValidationError(
            f"{location} must be a non-empty string")
    return value


def _require_sha256(value: object, location: str) -> str:
    value = _require_string(value, location)
    if not _SHA256_RE.fullmatch(value):
        raise CandidateSetValidationError(
            f"{location} must be a lowercase SHA256")
    return value


def _numeric_shape(value: object, location: str) -> tuple[int, ...]:
    """Validate a finite rectangular numeric scalar/vector and return shape."""
    if isinstance(value, bool):
        raise CandidateSetValidationError(
            f"{location} boolean is not a numeric feature")
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise CandidateSetValidationError(
                f"{location} must be finite")
        return ()
    if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)):
        raise CandidateSetValidationError(
            f"{location} must be a numeric scalar or sequence")
    if not value:
        raise CandidateSetValidationError(
            f"{location} sequence cannot be empty")
    child_shapes = [
        _numeric_shape(child, f"{location}[{index}]")
        for index, child in enumerate(value)
    ]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise CandidateSetValidationError(
            f"{location} must be rectangular")
    return (len(value),) + child_shapes[0]


def _numeric_all_zero(value: object) -> bool:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value) == 0.0
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(_numeric_all_zero(child) for child in value)
    )


def _require_nonnegative_integer(value: object, location: str) -> int:
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or int(value) < 0):
        raise CandidateSetValidationError(
            f"{location} must be a non-negative integer")
    return int(value)


def _require_binary_mask(
    value: object,
    expected_length: int,
    location: str,
) -> None:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) != expected_length):
        raise CandidateSetValidationError(
            f"{location} must have length {expected_length}")
    for index, item in enumerate(value):
        if (isinstance(item, bool) or not isinstance(item, Real)
                or float(item) not in (0.0, 1.0)):
            raise CandidateSetValidationError(
                f"{location}[{index}] must be numeric 0 or 1")


def _validate_provenance(value: object) -> Mapping[str, object]:
    provenance = _require_exact_keys(
        value, PROVENANCE_KEYS, "provenance")
    for key in (
        "dataset_id",
        "scene_id",
        "episode_id",
        "session_id",
        "group_id",
        "goal_epoch",
        "state_id",
        "goal_source_episode_id",
        "environment_id",
    ):
        _require_string(provenance[key], f"provenance.{key}")
    for key in (
        "prefix_sha256",
        "goal_sha256",
        "navdp_fifo_sha256",
        "split_sha256",
        "source_policy_sha256",
        "candidate_generator_sha256",
        "feature_builder_sha256",
        "rollout_labeler_sha256",
        "navmesh_sha256",
    ):
        _require_sha256(provenance[key], f"provenance.{key}")
    state_source = provenance["state_source"]
    if state_source not in ("expert", "on_policy"):
        raise CandidateSetValidationError(
            "provenance.state_source must be expert or on_policy")
    for key in ("plan_index", "prefix_frames"):
        _require_nonnegative_integer(
            provenance[key], f"provenance.{key}")
    if int(provenance["prefix_frames"]) < 1:
        raise CandidateSetValidationError(
            "provenance.prefix_frames must be positive")
    split_role = provenance["split_role"]
    if split_role not in TRAINABLE_SPLIT_ROLES:
        raise CandidateSetValidationError(
            f"split_role {split_role!r} is not train/development")
    return provenance


def _validate_feature_keys(
    features: object,
    expected: frozenset[str],
    location: str,
) -> dict[str, tuple[int, ...]]:
    features = _require_exact_keys(features, expected, location)
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in features.items():
        lowered = str(key).lower()
        if (str(key) in PRIVILEGED_LABEL_DENY_LIST
                or any(fragment in lowered
                       for fragment in PRIVILEGED_FEATURE_DENY_FRAGMENTS)):
            raise CandidateSetValidationError(
                f"privileged field leaked into {location}: {key}")
        shapes[str(key)] = _numeric_shape(
            value, f"{location}.{key}")
    return shapes


def _validate_candidate(
    value: object,
    index: int,
) -> tuple[str, str, dict[str, tuple[int, ...]], Mapping[str, object]]:
    location = f"candidates[{index}]"
    candidate = _require_exact_keys(value, _CANDIDATE_KEYS, location)
    candidate_id = _require_string(
        candidate["candidate_id"], f"{location}.candidate_id")
    candidate_type = _require_string(
        candidate["candidate_type"], f"{location}.candidate_type")
    if candidate_type not in CANDIDATE_TYPES:
        raise CandidateSetValidationError(
            f"{location}.candidate_type is unknown: {candidate_type}")
    shapes = _validate_feature_keys(
        candidate["features"], CANDIDATE_FEATURE_KEYS,
        f"{location}.features")
    features = candidate["features"]
    onehot = features["candidate_type_onehot"]
    if (not isinstance(onehot, Sequence) or isinstance(onehot, str)
            or len(onehot) != len(CANDIDATE_TYPES)):
        raise CandidateSetValidationError(
            f"{location}.features.candidate_type_onehot must have length 4")
    expected_onehot = [
        float(kind == candidate_type) for kind in CANDIDATE_TYPES]
    if [float(item) for item in onehot] != expected_onehot:
        raise CandidateSetValidationError(
            f"{location}.candidate_type disagrees with onehot")
    _require_binary_mask(
        features["feature_presence_mask"],
        FEATURE_PRESENCE_MASK_SIZE,
        f"{location}.features.feature_presence_mask",
    )
    for present, feature_keys in zip(
            features["feature_presence_mask"],
            FEATURE_PRESENCE_MASK_FIELDS):
        if float(present) == 0.0:
            for feature_key in feature_keys:
                if not _numeric_all_zero(features[feature_key]):
                    raise CandidateSetValidationError(
                        f"{location}.features.{feature_key} must be zero "
                        "when absent")
    for key in _NONNEGATIVE_CANDIDATE_FEATURES:
        feature = features[key]
        if not isinstance(feature, Real) or isinstance(feature, bool):
            raise CandidateSetValidationError(
                f"{location}.features.{key} must be a numeric scalar")
        if float(feature) < 0.0:
            raise CandidateSetValidationError(
                f"{location}.features.{key} must be non-negative")
    if not float(features["graph_hops"]).is_integer():
        raise CandidateSetValidationError(
            f"{location}.features.graph_hops must be integer-valued")
    depth_confidence = float(features["depth_confidence_mean"])
    if not 0.0 <= depth_confidence <= 1.0:
        raise CandidateSetValidationError(
            f"{location}.features.depth_confidence_mean outside [0, 1]")

    labels = _require_exact_keys(
        candidate["labels"], CANDIDATE_LABEL_KEYS,
        f"{location}.labels")
    for key, label in labels.items():
        if key in _CANDIDATE_BOOL_LABELS:
            if not isinstance(label, bool):
                raise CandidateSetValidationError(
                    f"{location}.labels.{key} must be boolean")
        else:
            shape = _numeric_shape(label, f"{location}.labels.{key}")
            if shape:
                raise CandidateSetValidationError(
                    f"{location}.labels.{key} must be scalar")
    covisibility = float(labels["teacher_covisibility"])
    if not 0.0 <= covisibility <= 1.0:
        raise CandidateSetValidationError(
            f"{location}.labels.teacher_covisibility outside [0, 1]")
    proxy_progress = float(labels["proposal_proxy_progress_m"])
    if not labels["proposal_proxy_label_valid"]:
        if (proxy_progress != 0.0
                or labels["proposal_proxy_reachable"]
                or labels["proposal_proxy_positive"]):
            raise CandidateSetValidationError(
                f"{location} invalid proposal-proxy labels must be neutral")
    else:
        if (not labels["proposal_proxy_reachable"]
                and proxy_progress != 0.0):
            raise CandidateSetValidationError(
                f"{location} unreachable proposal proxy progress must be zero")
        expected_proxy_positive = bool(
            labels["proposal_proxy_reachable"]
            and proxy_progress > PROPOSAL_PROXY_POSITIVE_MARGIN_M
        )
        if (bool(labels["proposal_proxy_positive"])
                != expected_proxy_positive):
            raise CandidateSetValidationError(
                f"{location} proposal proxy positive disagrees with "
                "reachability/progress")
    if labels["useful"] and (
            not labels["rollout_label_valid"]
            or not labels["reachable"]
            or labels["harm"]
            or labels["collision_h8"]):
        raise CandidateSetValidationError(
            f"{location} useful label contradicts safety/validity")
    if not labels["rollout_label_valid"]:
        rollout_values = (
            labels["geodesic_progress_h8_m"],
            labels["geodesic_progress_h24_m"],
            labels["advantage_h24_m"],
        )
        if (any(float(item) != 0.0 for item in rollout_values)
                or labels["harm"] or labels["useful"]
                or labels["collision_h8"] or labels["regression_h24"]
                or labels["reachable"]):
            raise CandidateSetValidationError(
                f"{location} invalid rollout labels must be neutral")
    elif not labels["reachable"]:
        raise CandidateSetValidationError(
            f"{location} valid rollout must be reachable")
    expected_harm = bool(
        labels["collision_h8"] or labels["regression_h24"])
    if bool(labels["harm"]) != expected_harm:
        raise CandidateSetValidationError(
            f"{location} harm must equal collision-or-regression")
    expected_useful = bool(
        labels["rollout_label_valid"]
        and labels["reachable"]
        and not labels["harm"]
        and not labels["collision_h8"]
        and float(labels["advantage_h24_m"])
        >= USEFUL_ADVANTAGE_MARGIN_M
    )
    if bool(labels["useful"]) != expected_useful:
        raise CandidateSetValidationError(
            f"{location} useful label disagrees with advantage/safety")
    if (not labels["covisibility_label_valid"]
            and covisibility != 0.0):
        raise CandidateSetValidationError(
            f"{location} invalid co-visibility label must be zero")
    if not labels["pose_label_valid"]:
        pose_values = (
            labels["pose_residual_forward_m"],
            labels["pose_residual_left_m"],
            labels["pose_residual_yaw_rad"],
        )
        if any(float(item) != 0.0 for item in pose_values):
            raise CandidateSetValidationError(
                f"{location} invalid pose labels must be zero")
    if candidate_type in ("native", "dustbin"):
        if float(labels["advantage_h24_m"]) != 0.0:
            raise CandidateSetValidationError(
                f"{location} native/dustbin advantage must be zero")
        if labels["useful"]:
            raise CandidateSetValidationError(
                f"{location} native/dustbin cannot be residual-positive")
    if candidate_type == "native" and labels["regression_h24"]:
        raise CandidateSetValidationError(
            f"{location} native candidate cannot regress relative to itself")
    if candidate_type == "native" and labels["proposal_proxy_label_valid"]:
        raise CandidateSetValidationError(
            f"{location} native candidate cannot carry proposal-proxy labels")
    if candidate_type == "dustbin":
        for key, feature in features.items():
            if key == "candidate_type_onehot":
                continue
            if not _numeric_all_zero(feature):
                raise CandidateSetValidationError(
                    f"{location} dustbin feature {key} must be zero")
        for key, label in labels.items():
            if isinstance(label, bool):
                if label:
                    raise CandidateSetValidationError(
                        f"{location} dustbin label {key} must be false")
            elif float(label) != 0.0:
                raise CandidateSetValidationError(
                    f"{location} dustbin label {key} must be zero")
    return candidate_id, candidate_type, shapes, labels


def validate_candidate_set(record: object) -> dict[str, tuple[int, ...]]:
    """Validate one candidate set and return its model feature shapes."""
    record = _require_exact_keys(record, TOP_LEVEL_KEYS, "record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise CandidateSetValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}")
    _validate_provenance(record["provenance"])
    set_shapes = _validate_feature_keys(
        record["set_features"], SET_FEATURE_KEYS, "set_features")
    set_features = record["set_features"]
    for key in _NONNEGATIVE_SET_FEATURES:
        _require_nonnegative_integer(
            set_features[key], f"set_features.{key}")
    _require_binary_mask(
        set_features["feature_presence_mask"],
        SET_FEATURE_PRESENCE_MASK_SIZE,
        "set_features.feature_presence_mask",
    )
    for present, feature_key in zip(
            set_features["feature_presence_mask"],
            SET_FEATURE_PRESENCE_MASK_ORDER):
        if float(present) == 0.0 and int(set_features[feature_key]) != 0:
            raise CandidateSetValidationError(
                f"set_features.{feature_key} must be zero when absent")

    candidates = record["candidates"]
    if (not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes, bytearray))
            or len(candidates) < 2):
        raise CandidateSetValidationError(
            "candidates must contain native candidate0 and final dustbin")
    if len(candidates) - 2 > MAX_RESIDUAL_CANDIDATES:
        raise CandidateSetValidationError(
            f"candidate set exceeds residual K={MAX_RESIDUAL_CANDIDATES}")
    candidate_rows = [
        _validate_candidate(candidate, index)
        for index, candidate in enumerate(candidates)
    ]
    candidate_ids = [row[0] for row in candidate_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CandidateSetValidationError(
            "candidate_id values must be unique within a set")
    if candidate_rows[0][0:2] != (NATIVE_CANDIDATE_ID, "native"):
        raise CandidateSetValidationError(
            "candidate 0 must be id='native', type='native'")
    if candidate_rows[-1][0:2] != (DUSTBIN_CANDIDATE_ID, "dustbin"):
        raise CandidateSetValidationError(
            "final candidate must be id='dustbin', type='dustbin'")
    if sum(row[1] == "native" for row in candidate_rows) != 1:
        raise CandidateSetValidationError("set must contain exactly one native")
    if sum(row[1] == "dustbin" for row in candidate_rows) != 1:
        raise CandidateSetValidationError("set must contain exactly one dustbin")
    memory_count = sum(
        row[1] == "memory_graph" for row in candidate_rows[1:-1])
    frontier_count = sum(
        row[1] == "frontier" for row in candidate_rows[1:-1])
    if int(set_features["memory_candidate_count"]) != memory_count:
        raise CandidateSetValidationError(
            "set_features.memory_candidate_count disagrees with candidates")
    if int(set_features["frontier_candidate_count"]) != frontier_count:
        raise CandidateSetValidationError(
            "set_features.frontier_candidate_count disagrees with candidates")

    reference_shapes = candidate_rows[0][2]
    for index, row in enumerate(candidate_rows[1:], start=1):
        if row[2] != reference_shapes:
            raise CandidateSetValidationError(
                f"candidate feature shapes differ at index {index}")

    set_labels = _require_exact_keys(
        record["set_labels"], SET_LABEL_KEYS, "set_labels")
    for key in _SET_BOOL_LABELS:
        if not isinstance(set_labels[key], bool):
            raise CandidateSetValidationError(
                f"set_labels.{key} must be boolean")
    if sum(bool(set_labels[key]) for key in (
            "global_match", "strict_no_match", "ambiguous")) != 1:
        raise CandidateSetValidationError(
            "exactly one match/no-match/ambiguous set label must be true")
    oracle_id = _require_string(
        set_labels["oracle_best_candidate_id"],
        "set_labels.oracle_best_candidate_id")
    if oracle_id not in candidate_ids:
        raise CandidateSetValidationError(
            "oracle_best_candidate_id is absent from candidates")
    native_labels = candidate_rows[0][3]
    if (not native_labels["rollout_label_valid"]
            or not native_labels["reachable"]):
        raise CandidateSetValidationError(
            "native candidate requires a valid reachable rollout label")
    native_progress_h24 = float(
        native_labels["geodesic_progress_h24_m"])
    for candidate_id, _candidate_type, _shapes, labels in candidate_rows[1:-1]:
        if labels["rollout_label_valid"]:
            expected_advantage = (
                float(labels["geodesic_progress_h24_m"])
                - native_progress_h24
            )
            actual_advantage = float(labels["advantage_h24_m"])
            if not math.isclose(
                    actual_advantage, expected_advantage,
                    rel_tol=1e-6, abs_tol=1e-6):
                raise CandidateSetValidationError(
                    f"candidate {candidate_id!r} advantage disagrees with "
                    "residual-minus-native progress")
            expected_regression = (
                actual_advantage <= -REGRESSION_ADVANTAGE_MARGIN_M
            )
            if bool(labels["regression_h24"]) != expected_regression:
                raise CandidateSetValidationError(
                    f"candidate {candidate_id!r} regression_h24 disagrees "
                    "with the frozen negative-advantage margin")
    positive_rows = [
        row for row in candidate_rows[1:-1] if bool(row[3]["useful"])
    ]
    positive_ids = {row[0] for row in positive_rows}
    has_positive = bool(set_labels["candidate_set_has_positive"])
    if has_positive != bool(positive_ids):
        raise CandidateSetValidationError(
            "candidate_set_has_positive disagrees with residual labels")
    if has_positive:
        best_row = sorted(
            positive_rows,
            key=lambda row: (
                -float(row[3]["advantage_h24_m"]), row[0]),
        )[0]
        if oracle_id != best_row[0]:
            raise CandidateSetValidationError(
                "positive set oracle must identify the maximum valid useful "
                "advantage")
    if not has_positive and oracle_id != DUSTBIN_CANDIDATE_ID:
        raise CandidateSetValidationError(
            "no-positive set oracle must be dustbin")
    coverage_valid = bool(set_labels["coverage_label_valid"])
    universe_has_positive = bool(
        set_labels["candidate_universe_has_positive"])
    coverage_miss = bool(set_labels["candidate_coverage_miss"])
    if coverage_valid:
        if has_positive and not universe_has_positive:
            raise CandidateSetValidationError(
                "candidate-set positive requires universe positive")
        expected_miss = universe_has_positive and not has_positive
        if coverage_miss != expected_miss:
            raise CandidateSetValidationError(
                "candidate_coverage_miss disagrees with universe/set labels")
    elif universe_has_positive or coverage_miss:
        raise CandidateSetValidationError(
            "invalid coverage labels must be neutral")

    proxy_positive_ids = {
        row[0] for row in candidate_rows[1:-1]
        if bool(row[3]["proposal_proxy_positive"])
    }
    proxy_set_has_positive = bool(
        set_labels["proposal_proxy_set_has_positive"])
    if proxy_set_has_positive != bool(proxy_positive_ids):
        raise CandidateSetValidationError(
            "proposal_proxy_set_has_positive disagrees with proxy labels")
    proxy_coverage_valid = bool(
        set_labels["proposal_proxy_coverage_label_valid"])
    proxy_universe_has_positive = bool(
        set_labels["proposal_proxy_universe_has_positive"])
    proxy_coverage_miss = bool(
        set_labels["proposal_proxy_coverage_miss"])
    if proxy_coverage_valid:
        if proxy_set_has_positive and not proxy_universe_has_positive:
            raise CandidateSetValidationError(
                "proposal-proxy set positive requires proxy-universe positive")
        expected_proxy_miss = (
            proxy_universe_has_positive and not proxy_set_has_positive)
        if proxy_coverage_miss != expected_proxy_miss:
            raise CandidateSetValidationError(
                "proposal_proxy_coverage_miss disagrees with proxy "
                "universe/set labels")
    elif proxy_universe_has_positive or proxy_coverage_miss:
        raise CandidateSetValidationError(
            "invalid proposal-proxy coverage labels must be neutral")

    out = {f"set.{key}": shape for key, shape in set_shapes.items()}
    out.update({
        f"candidate.{key}": shape
        for key, shape in reference_shapes.items()
    })
    return out


def validate_candidate_dataset(records: Iterable[object]) -> dict:
    """Validate a collection, group isolation, and cross-record shapes."""
    records = list(records)
    if not records:
        raise CandidateSetValidationError("candidate dataset cannot be empty")
    decision_keys: set[tuple[object, ...]] = set()
    state_provenance: dict[str, tuple[object, ...]] = {}
    goal_epoch_provenance: dict[
        tuple[str, str, str], tuple[str, str]] = {}
    group_roles: dict[str, tuple[str, str]] = {}
    scene_roles: dict[str, str] = {}
    scene_environments: dict[str, tuple[str, str]] = {}
    episode_origins: dict[str, tuple[str, str, str]] = {}
    goal_source_references: list[
        tuple[int, str, tuple[str, str, str]]] = []
    session_provenance: dict[str, tuple[str, str, str, str]] = {}
    prefix_progression: dict[
        tuple[str, str, str], list[tuple[int, int]]] = {}
    artifact_signature: tuple[str, ...] | None = None
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for index, record in enumerate(records):
        shapes = validate_candidate_set(record)
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise CandidateSetValidationError(
                f"record {index} model feature shapes differ")
        provenance = record["provenance"]
        decision_key = (
            provenance["state_id"],
            provenance["goal_epoch"],
        )
        if decision_key in decision_keys:
            raise CandidateSetValidationError(
                f"duplicate decision record: {decision_key}")
        decision_keys.add(decision_key)
        group_id = str(provenance["group_id"])
        group_value = (
            str(provenance["scene_id"]), str(provenance["split_role"]))
        previous_group = group_roles.setdefault(group_id, group_value)
        if previous_group != group_value:
            raise CandidateSetValidationError(
                f"group {group_id!r} crosses scene or split role")
        scene_id = str(provenance["scene_id"])
        split_role = str(provenance["split_role"])
        previous_role = scene_roles.setdefault(scene_id, split_role)
        if previous_role != split_role:
            raise CandidateSetValidationError(
                f"scene {scene_id!r} crosses split roles")
        environment_value = (
            str(provenance["environment_id"]),
            str(provenance["navmesh_sha256"]),
        )
        previous_environment = scene_environments.setdefault(
            scene_id, environment_value)
        if previous_environment != environment_value:
            raise CandidateSetValidationError(
                f"scene {scene_id!r} has inconsistent environment/navmesh")
        episode_id = str(provenance["episode_id"])
        episode_origin = (scene_id,) + environment_value
        previous_episode_origin = episode_origins.setdefault(
            episode_id, episode_origin)
        if previous_episode_origin != episode_origin:
            raise CandidateSetValidationError(
                f"episode {episode_id!r} crosses scenes/environments/navmeshes")
        goal_source_references.append((
            index,
            str(provenance["goal_source_episode_id"]),
            episode_origin,
        ))
        session_id = str(provenance["session_id"])
        session_value = (
            scene_id,
            str(provenance["episode_id"]),
            group_id,
            split_role,
        )
        previous_session = session_provenance.setdefault(
            session_id, session_value)
        if previous_session != session_value:
            raise CandidateSetValidationError(
                f"session {session_id!r} has inconsistent provenance")
        state_id = str(provenance["state_id"])
        state_value = (
            scene_id,
            str(provenance["episode_id"]),
            session_id,
            int(provenance["plan_index"]),
            int(provenance["prefix_frames"]),
            str(provenance["prefix_sha256"]),
            str(provenance["navdp_fifo_sha256"]),
            str(provenance["state_source"]),
            str(provenance["environment_id"]),
            str(provenance["navmesh_sha256"]),
        )
        previous_state = state_provenance.setdefault(state_id, state_value)
        if previous_state != state_value:
            raise CandidateSetValidationError(
                f"state {state_id!r} has inconsistent causal provenance")
        goal_epoch_key = (
            scene_id, session_id, str(provenance["goal_epoch"]))
        goal_epoch_value = (
            str(provenance["goal_source_episode_id"]),
            str(provenance["goal_sha256"]),
        )
        previous_goal_epoch = goal_epoch_provenance.setdefault(
            goal_epoch_key, goal_epoch_value)
        if previous_goal_epoch != goal_epoch_value:
            raise CandidateSetValidationError(
                f"goal epoch {goal_epoch_key!r} changes source or content")
        prefix_key = (
            scene_id, session_id, str(provenance["goal_epoch"]))
        prefix_progression.setdefault(prefix_key, []).append((
            int(provenance["plan_index"]),
            int(provenance["prefix_frames"]),
        ))
        signature = tuple(str(provenance[key]) for key in (
            "dataset_id",
            "split_sha256",
            "source_policy_sha256",
            "candidate_generator_sha256",
            "feature_builder_sha256",
            "rollout_labeler_sha256",
        ))
        if artifact_signature is None:
            artifact_signature = signature
        elif signature != artifact_signature:
            raise CandidateSetValidationError(
                "records disagree on dataset/model/builder signature")
    for prefix_key, progression in prefix_progression.items():
        ordered = sorted(progression)
        if any(current[1] < previous[1]
               for previous, current in zip(ordered, ordered[1:])):
            raise CandidateSetValidationError(
                f"causal prefix length decreases in {prefix_key}")
    for index, goal_source_episode_id, episode_origin in goal_source_references:
        goal_origin = episode_origins.get(goal_source_episode_id)
        if goal_origin is None:
            raise CandidateSetValidationError(
                f"record {index} goal source episode is absent from artifact")
        if goal_origin != episode_origin:
            raise CandidateSetValidationError(
                f"record {index} uses a cross-scene/environment goal source")
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(records),
        "scenes": len(scene_roles),
        "groups": len(group_roles),
        "feature_shapes": reference_shapes,
        "artifact_signature": artifact_signature,
    }


def canonical_candidate_set_sha256(record: object) -> str:
    """Validate and hash one canonical JSON representation."""
    validate_candidate_set(record)
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
