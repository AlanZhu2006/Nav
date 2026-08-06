"""Lightweight set-wise utility and risk ranker for NLSR proposals.

The module deliberately consumes only records accepted by
``novel_candidate_set_schema_v2``.  Feature extraction uses a frozen field
order, includes every declared presence mask, and never reads provenance or
label values as model inputs.  Native ImageGoal is the explicit zero-advantage
baseline; the final dustbin is a no-match state and is not a controllable
proposal.

The network is a deterministic, dropout-free masked DeepSets model.  It emits
per-candidate advantage location/scale, harm, and rank scores plus a set-level
coverage-miss logit.  The training losses fail closed around label-validity
masks and keep native in the listwise objective, so an unsafe or unhelpful
residual must beat a real baseline rather than an artificial all-zero row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from MemNavData.novel_candidate_set_schema_v2 import (
    CANDIDATE_FEATURE_KEYS,
    CANDIDATE_TYPES,
    FEATURE_PRESENCE_MASK_ORDER,
    PROVENANCE_KEYS,
    SCHEMA_VERSION,
    SET_FEATURE_KEYS,
    SET_FEATURE_PRESENCE_MASK_ORDER,
    validate_candidate_dataset,
    validate_candidate_set,
)


CHECKPOINT_FORMAT_VERSION = "nlsr_set_ranker_checkpoint_v2"
LIGHTWEIGHT_PARAMETER_MIN = 50_000
LIGHTWEIGHT_PARAMETER_MAX = 200_000

# These orders, rather than mapping iteration order, are the complete input
# allow-list.  A schema extension therefore requires an intentional ranker
# version change instead of silently becoming a training feature.
CANDIDATE_VECTOR_FIELD_ORDER = (
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
)
SET_VECTOR_FIELD_ORDER = (
    "feature_presence_mask",
    "native_stagnation_plans",
    "graph_node_count",
    "graph_edge_count",
    "graph_age_frames",
    "memory_candidate_count",
    "frontier_candidate_count",
)
RELATION_FIELD_ORDER = (
    "goal_patch_relation",
    "goal_temporal_relation",
    "local_map_relation",
    "native_proposal_relation",
)
CANDIDATE_SCALAR_FIELD_ORDER = (
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
)
SET_SCALAR_FIELD_ORDER = SET_FEATURE_PRESENCE_MASK_ORDER
BATCH_ARTIFACT_SIGNATURE_FIELDS = (
    "dataset_id",
    "split_sha256",
    "source_policy_sha256",
    "candidate_generator_sha256",
    "feature_builder_sha256",
    "rollout_labeler_sha256",
)

if frozenset(CANDIDATE_VECTOR_FIELD_ORDER) != CANDIDATE_FEATURE_KEYS:
    raise RuntimeError("candidate ranker allow-list disagrees with schema v2")
if frozenset(SET_VECTOR_FIELD_ORDER) != SET_FEATURE_KEYS:
    raise RuntimeError("set ranker allow-list disagrees with schema v2")


class NLSRRankerError(ValueError):
    """Raised when vectorization, masks, or checkpoint metadata fail closed."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NLSRRankerError("value is not canonical JSON") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def state_dict_sha256(value: object) -> str:
    """Hash exact dense tensor names, dtypes, shapes, and CPU bytes."""
    if not isinstance(value, Mapping) or not value:
        raise NLSRRankerError("state_dict must be a nonempty mapping")
    if any(not isinstance(key, str) for key in value):
        raise NLSRRankerError("state_dict keys must be strings")
    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = value[key]
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise NLSRRankerError("state_dict entries must be named tensors")
        if tensor.layout != torch.strided:
            raise NLSRRankerError("state_dict tensors must use strided layout")
        detached = tensor.detach()
        if not torch.isfinite(detached).all():
            raise NLSRRankerError(
                f"state_dict tensor {key!r} contains non-finite values")
        canonical = detached.to(device="cpu").contiguous()
        descriptor = _canonical_json_bytes({
            "key": key,
            "dtype": str(canonical.dtype),
            "shape": list(canonical.shape),
        })
        raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _require_plain_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NLSRRankerError(f"{location} must be an integer")
    return int(value)


def _require_sha256_text(value: object, location: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise NLSRRankerError(f"{location} must be a lowercase SHA256")
    return value


def _shape_size(shape: Sequence[int]) -> int:
    size = 1
    for dimension in shape:
        if int(dimension) < 1:
            raise NLSRRankerError(f"invalid feature shape {tuple(shape)!r}")
        size *= int(dimension)
    return size


def _flatten_numeric(value: object, location: str) -> list[float]:
    if isinstance(value, bool):
        raise NLSRRankerError(f"{location} contains a boolean")
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise NLSRRankerError(f"{location} is not finite")
        return [result]
    if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)):
        raise NLSRRankerError(f"{location} is not numeric")
    flattened: list[float] = []
    for index, item in enumerate(value):
        flattened.extend(_flatten_numeric(item, f"{location}[{index}]"))
    return flattened


@dataclass(frozen=True)
class RankerFeatureSpec:
    """Frozen feature shapes and their canonical schema-contract hash."""

    schema_version: str
    candidate_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    set_shapes: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise NLSRRankerError(
                f"feature spec requires schema {SCHEMA_VERSION!r}")
        if tuple(key for key, _ in self.candidate_shapes) != (
                CANDIDATE_VECTOR_FIELD_ORDER):
            raise NLSRRankerError("candidate feature order is not canonical")
        if tuple(key for key, _ in self.set_shapes) != SET_VECTOR_FIELD_ORDER:
            raise NLSRRankerError("set feature order is not canonical")
        for key, shape in self.candidate_shapes + self.set_shapes:
            if not isinstance(shape, tuple):
                raise NLSRRankerError(f"shape for {key!r} must be a tuple")
            if shape:
                _shape_size(shape)

    @property
    def candidate_dim(self) -> int:
        return sum(_shape_size(shape) if shape else 1
                   for _key, shape in self.candidate_shapes)

    @property
    def set_dim(self) -> int:
        return sum(_shape_size(shape) if shape else 1
                   for _key, shape in self.set_shapes)

    def field_slice(self, field: str, *, candidate: bool) -> slice:
        fields = self.candidate_shapes if candidate else self.set_shapes
        offset = 0
        for key, shape in fields:
            width = _shape_size(shape) if shape else 1
            if key == field:
                return slice(offset, offset + width)
            offset += width
        raise NLSRRankerError(f"field {field!r} is absent from feature spec")

    def to_dict(self) -> dict[str, object]:
        def encode(fields: tuple[tuple[str, tuple[int, ...]], ...]) -> list[dict]:
            return [
                {"field": field, "shape": list(shape)}
                for field, shape in fields
            ]

        return {
            "schema_version": self.schema_version,
            "candidate_shapes": encode(self.candidate_shapes),
            "set_shapes": encode(self.set_shapes),
            "candidate_dim": self.candidate_dim,
            "set_dim": self.set_dim,
            "schema_contract_sha256": self.schema_contract_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RankerFeatureSpec":
        if not isinstance(value, Mapping):
            raise NLSRRankerError("feature spec must be a mapping")
        expected = frozenset({
            "schema_version", "candidate_shapes", "set_shapes",
            "candidate_dim", "set_dim", "schema_contract_sha256",
        })
        if frozenset(value) != expected:
            raise NLSRRankerError("feature spec keys are not exact")

        def decode(name: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
            rows = value[name]
            if not isinstance(rows, Sequence) or isinstance(rows, str):
                raise NLSRRankerError(f"{name} must be a sequence")
            decoded = []
            for row in rows:
                if (not isinstance(row, Mapping)
                        or frozenset(row) != {"field", "shape"}
                        or not isinstance(row["field"], str)
                        or not isinstance(row["shape"], Sequence)):
                    raise NLSRRankerError(f"malformed {name} row")
                shape: list[int] = []
                for dimension in row["shape"]:
                    if (isinstance(dimension, bool)
                            or not isinstance(dimension, int)):
                        raise NLSRRankerError(f"malformed {name} shape")
                    shape.append(int(dimension))
                decoded.append((row["field"], tuple(shape)))
            return tuple(decoded)

        result = cls(
            schema_version=str(value["schema_version"]),
            candidate_shapes=decode("candidate_shapes"),
            set_shapes=decode("set_shapes"),
        )
        if (_require_plain_integer(value["candidate_dim"], "candidate_dim")
                != result.candidate_dim):
            raise NLSRRankerError("candidate feature dimension mismatch")
        if (_require_plain_integer(value["set_dim"], "set_dim")
                != result.set_dim):
            raise NLSRRankerError("set feature dimension mismatch")
        if value["schema_contract_sha256"] != result.schema_contract_sha256:
            raise NLSRRankerError("feature schema-contract hash mismatch")
        return result

    @property
    def schema_contract_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "candidate_types": list(CANDIDATE_TYPES),
            "candidate_feature_allow_list": sorted(CANDIDATE_FEATURE_KEYS),
            "set_feature_allow_list": sorted(SET_FEATURE_KEYS),
            "candidate_vector_order": list(CANDIDATE_VECTOR_FIELD_ORDER),
            "set_vector_order": list(SET_VECTOR_FIELD_ORDER),
            "candidate_presence_order": list(FEATURE_PRESENCE_MASK_ORDER),
            "set_presence_order": list(SET_FEATURE_PRESENCE_MASK_ORDER),
            "candidate_shapes": [
                [key, list(shape)] for key, shape in self.candidate_shapes],
            "set_shapes": [
                [key, list(shape)] for key, shape in self.set_shapes],
        }
        return _sha256_json(payload)


def feature_spec_from_record(record: object) -> RankerFeatureSpec:
    """Infer a strict feature spec from one already valid schema-v2 set."""
    shapes = validate_candidate_set(record)
    return RankerFeatureSpec(
        schema_version=SCHEMA_VERSION,
        candidate_shapes=tuple(
            (key, tuple(shapes[f"candidate.{key}"]))
            for key in CANDIDATE_VECTOR_FIELD_ORDER
        ),
        set_shapes=tuple(
            (key, tuple(shapes[f"set.{key}"]))
            for key in SET_VECTOR_FIELD_ORDER
        ),
    )


def feature_spec_from_dataset(records: Iterable[object]) -> RankerFeatureSpec:
    """Validate a complete artifact and infer its one cross-record spec."""
    rows = list(records)
    validate_candidate_dataset(rows)
    return feature_spec_from_record(rows[0])


@dataclass(frozen=True)
class CandidateSetBatch:
    """Padded candidate tensors, masks, labels, and non-model identifiers."""

    candidate_features: Tensor
    set_features: Tensor
    valid_mask: Tensor
    native_mask: Tensor
    dustbin_mask: Tensor
    residual_mask: Tensor
    selectable_mask: Tensor
    advantage_target: Tensor
    harm_target: Tensor
    rollout_label_valid: Tensor
    coverage_miss_target: Tensor
    coverage_label_valid: Tensor
    candidate_ids: tuple[tuple[str, ...], ...]
    scene_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.candidate_features.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.candidate_features.shape[1])

    def to(self, device: torch.device | str) -> "CandidateSetBatch":
        tensor_fields = {
            field: getattr(self, field).to(device)
            for field in (
                "candidate_features", "set_features", "valid_mask",
                "native_mask", "dustbin_mask", "residual_mask",
                "selectable_mask", "advantage_target", "harm_target",
                "rollout_label_valid", "coverage_miss_target",
                "coverage_label_valid",
            )
        }
        return CandidateSetBatch(
            **tensor_fields,
            candidate_ids=self.candidate_ids,
            scene_ids=self.scene_ids,
            group_ids=self.group_ids,
        )


def _vectorize_fields(
    features: Mapping[str, object],
    fields: tuple[tuple[str, tuple[int, ...]], ...],
    location: str,
) -> list[float]:
    values: list[float] = []
    for key, shape in fields:
        flattened = _flatten_numeric(features[key], f"{location}.{key}")
        expected = _shape_size(shape) if shape else 1
        if len(flattened) != expected:
            raise NLSRRankerError(
                f"{location}.{key} width {len(flattened)} != {expected}")
        values.extend(flattened)
    return values


def vectorize_candidate_sets(
    records: Iterable[object],
    *,
    feature_spec: RankerFeatureSpec | None = None,
    device: torch.device | str | None = None,
) -> CandidateSetBatch:
    """Validate and pad candidate sets without exposing labels as features."""
    rows = list(records)
    if not rows:
        raise NLSRRankerError("cannot vectorize an empty candidate batch")
    if feature_spec is None:
        feature_spec = feature_spec_from_record(rows[0])
    batch_signature: tuple[str, ...] | None = None
    split_role: str | None = None
    decisions: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        row_spec = feature_spec_from_record(row)
        if row_spec != feature_spec:
            raise NLSRRankerError(
                f"record {index} feature spec differs from the frozen spec")
        provenance = row["provenance"]
        signature = tuple(
            str(provenance[field])
            for field in BATCH_ARTIFACT_SIGNATURE_FIELDS)
        if batch_signature is None:
            batch_signature = signature
            split_role = str(provenance["split_role"])
        elif signature != batch_signature:
            raise NLSRRankerError(
                "candidate batch mixes artifact provenance signatures")
        elif str(provenance["split_role"]) != split_role:
            raise NLSRRankerError(
                "candidate batch mixes train/development split roles")
        decision = (
            str(provenance["state_id"]), str(provenance["goal_epoch"]))
        if decision in decisions:
            raise NLSRRankerError(
                f"candidate batch repeats decision {decision!r}")
        decisions.add(decision)

    batch_size = len(rows)
    max_candidates = max(len(row["candidates"]) for row in rows)
    candidate_features = torch.zeros(
        batch_size, max_candidates, feature_spec.candidate_dim,
        dtype=torch.float32, device=device)
    set_features = torch.zeros(
        batch_size, feature_spec.set_dim,
        dtype=torch.float32, device=device)
    valid_mask = torch.zeros(
        batch_size, max_candidates, dtype=torch.bool, device=device)
    native_mask = torch.zeros_like(valid_mask)
    dustbin_mask = torch.zeros_like(valid_mask)
    advantage_target = torch.zeros(
        batch_size, max_candidates, dtype=torch.float32, device=device)
    harm_target = torch.zeros_like(advantage_target)
    rollout_label_valid = torch.zeros_like(valid_mask)
    coverage_miss_target = torch.zeros(
        batch_size, dtype=torch.float32, device=device)
    coverage_label_valid = torch.zeros(
        batch_size, dtype=torch.bool, device=device)
    candidate_ids: list[tuple[str, ...]] = []
    scene_ids: list[str] = []
    group_ids: list[str] = []

    for batch_index, row in enumerate(rows):
        vector = _vectorize_fields(
            row["set_features"], feature_spec.set_shapes,
            f"records[{batch_index}].set_features")
        set_features[batch_index] = torch.tensor(
            vector, dtype=torch.float32, device=device)
        ids: list[str] = []
        for candidate_index, candidate in enumerate(row["candidates"]):
            values = _vectorize_fields(
                candidate["features"], feature_spec.candidate_shapes,
                f"records[{batch_index}].candidates[{candidate_index}]")
            candidate_features[batch_index, candidate_index] = torch.tensor(
                values, dtype=torch.float32, device=device)
            valid_mask[batch_index, candidate_index] = True
            kind = str(candidate["candidate_type"])
            native_mask[batch_index, candidate_index] = kind == "native"
            dustbin_mask[batch_index, candidate_index] = kind == "dustbin"
            labels = candidate["labels"]
            advantage_target[batch_index, candidate_index] = float(
                labels["advantage_h24_m"])
            harm_target[batch_index, candidate_index] = float(labels["harm"])
            rollout_label_valid[batch_index, candidate_index] = bool(
                labels["rollout_label_valid"])
            ids.append(str(candidate["candidate_id"]))
        set_labels = row["set_labels"]
        coverage_miss_target[batch_index] = float(
            set_labels["candidate_coverage_miss"])
        coverage_label_valid[batch_index] = bool(
            set_labels["coverage_label_valid"])
        candidate_ids.append(tuple(ids))
        scene_ids.append(str(row["provenance"]["scene_id"]))
        group_ids.append(str(row["provenance"]["group_id"]))

    residual_mask = valid_mask & ~native_mask & ~dustbin_mask
    selectable_mask = native_mask | residual_mask
    if not torch.all(native_mask.sum(dim=1) == 1):
        raise NLSRRankerError("each row must contain exactly one native")
    if not torch.all(dustbin_mask.sum(dim=1) == 1):
        raise NLSRRankerError("each row must contain exactly one dustbin")
    if torch.any(native_mask & dustbin_mask):
        raise NLSRRankerError("native and dustbin masks overlap")
    return CandidateSetBatch(
        candidate_features=candidate_features,
        set_features=set_features,
        valid_mask=valid_mask,
        native_mask=native_mask,
        dustbin_mask=dustbin_mask,
        residual_mask=residual_mask,
        selectable_mask=selectable_mask,
        advantage_target=advantage_target,
        harm_target=harm_target,
        rollout_label_valid=rollout_label_valid,
        coverage_miss_target=coverage_miss_target,
        coverage_label_valid=coverage_label_valid,
        candidate_ids=tuple(candidate_ids),
        scene_ids=tuple(scene_ids),
        group_ids=tuple(group_ids),
    )


@dataclass(frozen=True)
class NLSRRankerConfig:
    relation_projection_dim: int = 16
    candidate_scalar_dim: int = 32
    type_projection_dim: int = 16
    presence_projection_dim: int = 16
    candidate_hidden_dim: int = 96
    set_hidden_dim: int = 64
    head_hidden_dim: int = 96
    advantage_log_scale_min: float = -4.0
    advantage_log_scale_max: float = 2.0
    init_seed: int = 0

    def __post_init__(self) -> None:
        for field in (
            "relation_projection_dim", "candidate_scalar_dim",
            "type_projection_dim", "presence_projection_dim",
            "candidate_hidden_dim", "set_hidden_dim", "head_hidden_dim",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise NLSRRankerError(f"config.{field} must be positive")
        for field in (
            "advantage_log_scale_min", "advantage_log_scale_max",
        ):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value))):
                raise NLSRRankerError(
                    f"config.{field} must be a finite real")
        if self.advantage_log_scale_min >= self.advantage_log_scale_max:
            raise NLSRRankerError("invalid advantage log-scale bounds")
        if isinstance(self.init_seed, bool) or not isinstance(self.init_seed, int):
            raise NLSRRankerError("config.init_seed must be an integer")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "NLSRRankerConfig":
        if not isinstance(value, Mapping):
            raise NLSRRankerError("model config must be a mapping")
        expected = frozenset(cls.__dataclass_fields__)
        if frozenset(value) != expected:
            raise NLSRRankerError("model config keys are not exact")
        return cls(**dict(value))


@dataclass(frozen=True)
class NLSRRankerOutput:
    advantage_mean: Tensor
    advantage_log_scale: Tensor
    harm_logit: Tensor
    rank_score: Tensor
    coverage_logit: Tensor
    valid_mask: Tensor
    native_mask: Tensor
    dustbin_mask: Tensor
    residual_mask: Tensor
    selectable_mask: Tensor

    def masked_rank_score(self) -> Tensor:
        """Return native+residual scores with padding/dustbin at ``-inf``."""
        return self.rank_score.masked_fill(~self.selectable_mask, -torch.inf)


class NLSRSetRanker(nn.Module):
    """A small deterministic masked DeepSets utility/risk model."""

    def __init__(
        self,
        feature_spec: RankerFeatureSpec,
        config: NLSRRankerConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_spec = feature_spec
        self.config = config or NLSRRankerConfig()
        relation_widths = {
            field: self._slice_width(feature_spec.field_slice(
                field, candidate=True))
            for field in RELATION_FIELD_ORDER
        }
        type_width = self._slice_width(feature_spec.field_slice(
            "candidate_type_onehot", candidate=True))
        presence_width = self._slice_width(feature_spec.field_slice(
            "feature_presence_mask", candidate=True))
        if type_width != len(CANDIDATE_TYPES):
            raise NLSRRankerError("candidate type onehot width is not four")
        if presence_width != len(FEATURE_PRESENCE_MASK_ORDER):
            raise NLSRRankerError("candidate presence-mask width mismatch")
        set_presence_width = self._slice_width(feature_spec.field_slice(
            "feature_presence_mask", candidate=False))
        if set_presence_width != len(SET_FEATURE_PRESENCE_MASK_ORDER):
            raise NLSRRankerError("set presence-mask width mismatch")

        config = self.config
        # Forking the RNG makes construction reproducible without perturbing a
        # caller's data-loader/model initialization stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.init_seed)
            self.relation_projections = nn.ModuleDict({
                field: nn.Sequential(
                    nn.Linear(width, config.relation_projection_dim),
                    nn.SiLU(),
                )
                for field, width in relation_widths.items()
            })
            self.type_projection = nn.Linear(
                type_width, config.type_projection_dim, bias=False)
            self.scalar_projection = nn.Sequential(
                nn.Linear(
                    len(CANDIDATE_SCALAR_FIELD_ORDER),
                    config.candidate_scalar_dim),
                nn.SiLU(),
            )
            self.presence_projection = nn.Sequential(
                nn.Linear(
                    presence_width, config.presence_projection_dim,
                    bias=False),
                nn.SiLU(),
            )
            candidate_fusion_dim = (
                len(RELATION_FIELD_ORDER) * config.relation_projection_dim
                + config.type_projection_dim
                + config.candidate_scalar_dim
                + config.presence_projection_dim
            )
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_fusion_dim, config.candidate_hidden_dim),
                nn.LayerNorm(config.candidate_hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    config.candidate_hidden_dim,
                    config.candidate_hidden_dim),
                nn.SiLU(),
            )
            self.set_encoder = nn.Sequential(
                nn.Linear(
                    len(SET_SCALAR_FIELD_ORDER) + set_presence_width,
                    config.set_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.set_hidden_dim, config.set_hidden_dim),
                nn.SiLU(),
            )
            context_dim = (
                3 * config.candidate_hidden_dim + config.set_hidden_dim)
            self.candidate_head = nn.Sequential(
                nn.Linear(
                    config.candidate_hidden_dim + context_dim,
                    config.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.head_hidden_dim, 4),
            )
            self.coverage_head = nn.Sequential(
                nn.Linear(context_dim, config.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.head_hidden_dim, 1),
            )

    @staticmethod
    def _slice_width(value: slice) -> int:
        return int(value.stop) - int(value.start)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def assert_lightweight_parameter_budget(
        self,
        minimum: int = LIGHTWEIGHT_PARAMETER_MIN,
        maximum: int = LIGHTWEIGHT_PARAMETER_MAX,
    ) -> None:
        minimum = _require_plain_integer(minimum, "parameter minimum")
        maximum = _require_plain_integer(maximum, "parameter maximum")
        if minimum < 1 or maximum < minimum:
            raise NLSRRankerError("invalid lightweight parameter budget")
        if not minimum <= self.parameter_count <= maximum:
            raise NLSRRankerError(
                f"model has {self.parameter_count} parameters; expected "
                f"[{minimum}, {maximum}]")

    def _candidate_embeddings(self, features: Tensor) -> Tensor:
        presence = features[..., self.feature_spec.field_slice(
            "feature_presence_mask", candidate=True)]
        blocks: list[Tensor] = []
        for presence_index, field in enumerate(RELATION_FIELD_ORDER):
            values = features[..., self.feature_spec.field_slice(
                field, candidate=True)]
            block = self.relation_projections[field](values)
            # Absent relation values are schema-zero, but this also removes a
            # projection bias so missingness is represented only by the mask.
            block = block * presence[..., presence_index:presence_index + 1]
            blocks.append(block)
        candidate_type = features[..., self.feature_spec.field_slice(
            "candidate_type_onehot", candidate=True)]
        scalars = torch.cat([
            features[..., self.feature_spec.field_slice(field, candidate=True)]
            for field in CANDIDATE_SCALAR_FIELD_ORDER
        ], dim=-1)
        scalars = torch.sign(scalars) * torch.log1p(torch.abs(scalars))
        blocks.extend((
            self.type_projection(candidate_type),
            self.scalar_projection(scalars),
            self.presence_projection(presence),
        ))
        return self.candidate_encoder(torch.cat(blocks, dim=-1))

    def _set_embeddings(self, features: Tensor) -> Tensor:
        presence = features[..., self.feature_spec.field_slice(
            "feature_presence_mask", candidate=False)]
        scalars = torch.cat([
            features[..., self.feature_spec.field_slice(field, candidate=False)]
            for field in SET_SCALAR_FIELD_ORDER
        ], dim=-1)
        scalars = torch.log1p(scalars.clamp_min(0.0))
        return self.set_encoder(torch.cat((scalars, presence), dim=-1))

    @staticmethod
    def _validate_masks(
        candidate_features: Tensor,
        valid_mask: Tensor,
        native_mask: Tensor,
        dustbin_mask: Tensor,
    ) -> None:
        expected = candidate_features.shape[:2]
        for name, mask in (
            ("valid", valid_mask), ("native", native_mask),
            ("dustbin", dustbin_mask),
        ):
            if mask.dtype != torch.bool or mask.shape != expected:
                raise NLSRRankerError(
                    f"{name}_mask must be bool with shape {tuple(expected)}")
        if torch.any((native_mask | dustbin_mask) & ~valid_mask):
            raise NLSRRankerError("native/dustbin must be valid candidates")
        if torch.any(native_mask & dustbin_mask):
            raise NLSRRankerError("native and dustbin masks overlap")
        if not torch.all(native_mask.sum(dim=1) == 1):
            raise NLSRRankerError("each set requires exactly one native")
        if not torch.all(dustbin_mask.sum(dim=1) == 1):
            raise NLSRRankerError("each set requires exactly one dustbin")

    def forward(
        self,
        candidate_features: Tensor,
        set_features: Tensor,
        valid_mask: Tensor,
        native_mask: Tensor,
        dustbin_mask: Tensor,
    ) -> NLSRRankerOutput:
        if (candidate_features.ndim != 3
                or candidate_features.shape[-1]
                != self.feature_spec.candidate_dim):
            raise NLSRRankerError("candidate feature tensor shape mismatch")
        if (set_features.ndim != 2
                or set_features.shape[0] != candidate_features.shape[0]
                or set_features.shape[1] != self.feature_spec.set_dim):
            raise NLSRRankerError("set feature tensor shape mismatch")
        if (not torch.isfinite(candidate_features).all()
                or not torch.isfinite(set_features).all()):
            raise NLSRRankerError("model input contains non-finite values")
        self._validate_masks(
            candidate_features, valid_mask, native_mask, dustbin_mask)
        residual_mask = valid_mask & ~native_mask & ~dustbin_mask
        selectable_mask = native_mask | residual_mask

        candidate_embedding = self._candidate_embeddings(candidate_features)
        candidate_embedding = candidate_embedding * valid_mask.unsqueeze(-1)
        residual_float = residual_mask.unsqueeze(-1).to(
            candidate_embedding.dtype)
        residual_count = residual_float.sum(dim=1).clamp_min(1.0)
        residual_mean = (
            candidate_embedding * residual_float).sum(dim=1) / residual_count
        minus_inf = torch.finfo(candidate_embedding.dtype).min
        residual_max = candidate_embedding.masked_fill(
            ~residual_mask.unsqueeze(-1), minus_inf).max(dim=1).values
        has_residual = residual_mask.any(dim=1, keepdim=True)
        residual_max = torch.where(
            has_residual, residual_max, torch.zeros_like(residual_max))
        native_embedding = (
            candidate_embedding * native_mask.unsqueeze(-1)).sum(dim=1)
        set_embedding = self._set_embeddings(set_features)
        context = torch.cat((
            residual_mean, residual_max, native_embedding, set_embedding,
        ), dim=-1)
        expanded_context = context.unsqueeze(1).expand(
            -1, candidate_features.shape[1], -1)
        raw = self.candidate_head(torch.cat((
            candidate_embedding, expanded_context,
        ), dim=-1))
        coverage_logit = self.coverage_head(context).squeeze(-1)
        if not torch.isfinite(raw).all() or not torch.isfinite(
                coverage_logit).all():
            raise NLSRRankerError(
                "model output became non-finite for finite inputs")
        log_scale_range = (
            self.config.advantage_log_scale_max
            - self.config.advantage_log_scale_min)
        advantage_log_scale = (
            self.config.advantage_log_scale_min
            + log_scale_range * torch.sigmoid(raw[..., 1]))
        return NLSRRankerOutput(
            advantage_mean=raw[..., 0],
            advantage_log_scale=advantage_log_scale,
            harm_logit=raw[..., 2],
            rank_score=raw[..., 3],
            coverage_logit=coverage_logit,
            valid_mask=valid_mask,
            native_mask=native_mask,
            dustbin_mask=dustbin_mask,
            residual_mask=residual_mask,
            selectable_mask=selectable_mask,
        )

    def forward_batch(self, batch: CandidateSetBatch) -> NLSRRankerOutput:
        return self(
            batch.candidate_features,
            batch.set_features,
            batch.valid_mask,
            batch.native_mask,
            batch.dustbin_mask,
        )


@dataclass(frozen=True)
class NLSRLossConfig:
    advantage_weight: float = 1.0
    rank_weight: float = 1.0
    harm_weight: float = 1.0
    coverage_weight: float = 0.5
    student_t_degrees_freedom: float = 3.0
    rank_target_temperature: float = 0.25
    rank_prediction_temperature: float = 1.0
    rank_harm_penalty_m: float = 2.0
    harm_positive_weight: float = 4.0
    coverage_positive_weight: float = 4.0

    def __post_init__(self) -> None:
        for field in (
            "advantage_weight", "rank_weight", "harm_weight",
            "coverage_weight",
        ):
            raw = getattr(self, field)
            if (isinstance(raw, bool) or not isinstance(raw, Real)
                    or not math.isfinite(float(raw)) or float(raw) < 0.0):
                raise NLSRRankerError(
                    f"loss config {field} must be non-negative")
        for field in (
            "student_t_degrees_freedom", "rank_target_temperature",
            "rank_prediction_temperature", "harm_positive_weight",
            "coverage_positive_weight",
        ):
            raw = getattr(self, field)
            if (isinstance(raw, bool) or not isinstance(raw, Real)
                    or not math.isfinite(float(raw)) or float(raw) <= 0.0):
                raise NLSRRankerError(f"loss config {field} must be positive")
        if (isinstance(self.rank_harm_penalty_m, bool)
                or not isinstance(self.rank_harm_penalty_m, Real)
                or not math.isfinite(float(self.rank_harm_penalty_m))
                or float(self.rank_harm_penalty_m) < 0.0):
            raise NLSRRankerError(
                "loss config rank_harm_penalty_m must be non-negative")


@dataclass(frozen=True)
class NLSRLossBreakdown:
    total: Tensor
    advantage: Tensor
    rank: Tensor
    harm: Tensor
    coverage: Tensor
    advantage_count: int
    rank_set_count: int
    harm_count: int
    coverage_count: int


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_log_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    if scores.shape != mask.shape:
        raise NLSRRankerError("masked softmax shape mismatch")
    masked = scores.masked_fill(~mask, -torch.inf)
    return F.log_softmax(masked, dim=-1)


def compute_nlsr_losses(
    output: NLSRRankerOutput,
    batch: CandidateSetBatch,
    config: NLSRLossConfig | None = None,
) -> NLSRLossBreakdown:
    """Compute label-masked robust regression, ranking, and safety losses."""
    config = config or NLSRLossConfig()
    expected = batch.valid_mask.shape
    for name, value in (
        ("advantage_mean", output.advantage_mean),
        ("advantage_log_scale", output.advantage_log_scale),
        ("harm_logit", output.harm_logit),
        ("rank_score", output.rank_score),
    ):
        if value.shape != expected:
            raise NLSRRankerError(f"{name} shape differs from batch")
    if output.coverage_logit.shape != batch.coverage_miss_target.shape:
        raise NLSRRankerError("coverage logit shape differs from batch")
    if any(not torch.isfinite(value).all() for value in (
            output.advantage_mean, output.advantage_log_scale,
            output.harm_logit, output.rank_score, output.coverage_logit)):
        raise NLSRRankerError("ranker output contains non-finite values")
    for name in (
        "valid_mask", "native_mask", "dustbin_mask", "residual_mask",
        "selectable_mask",
    ):
        if not torch.equal(getattr(output, name), getattr(batch, name)):
            raise NLSRRankerError(
                f"output {name} differs from its vectorized batch")
    for name, target in (
        ("advantage_target", batch.advantage_target),
        ("harm_target", batch.harm_target),
    ):
        if target.shape != expected or not torch.isfinite(target).all():
            raise NLSRRankerError(f"{name} is malformed")
    if (batch.rollout_label_valid.dtype != torch.bool
            or batch.rollout_label_valid.shape != expected):
        raise NLSRRankerError("rollout validity mask is malformed")
    if not torch.all((batch.harm_target == 0.0)
                     | (batch.harm_target == 1.0)):
        raise NLSRRankerError("harm targets must be binary")
    if (not torch.isfinite(batch.coverage_miss_target).all()
            or batch.coverage_miss_target.shape
            != batch.coverage_label_valid.shape):
        raise NLSRRankerError("coverage targets are malformed")
    if (batch.coverage_label_valid.dtype != torch.bool
            or not torch.all((batch.coverage_miss_target == 0.0)
                             | (batch.coverage_miss_target == 1.0))):
        raise NLSRRankerError("coverage validity/targets are malformed")

    # Structural native/dustbin rows never dilute residual regression or harm
    # calibration.  Native is introduced explicitly in the listwise loss.
    residual_label_mask = batch.residual_mask & batch.rollout_label_valid
    residual = output.advantage_mean - batch.advantage_target
    degrees = torch.as_tensor(
        config.student_t_degrees_freedom,
        dtype=residual.dtype, device=residual.device)
    normalizer = (
        0.5 * torch.log(degrees * torch.as_tensor(
            math.pi, dtype=residual.dtype, device=residual.device))
        + torch.lgamma(0.5 * degrees)
        - torch.lgamma(0.5 * (degrees + 1.0))
    )
    # ``square(residual / scale)`` overflows for perfectly finite float32
    # residuals above roughly 1e19.  ``hypot`` evaluates the same term without
    # squaring; float64 keeps every finite float32 residual representable after
    # division.  Unlike a log(abs(residual)) formulation, its zero derivative
    # is also finite.
    standardized = (
        residual.to(torch.float64)
        * torch.exp(-output.advantage_log_scale.to(torch.float64))
        / torch.sqrt(degrees.to(torch.float64))
    )
    stable_log1p_square = (
        2.0 * torch.log(torch.hypot(
            torch.ones_like(standardized), standardized))
    ).to(residual.dtype)
    advantage_terms = (
        output.advantage_log_scale + normalizer
        + 0.5 * (degrees + 1.0)
        * stable_log1p_square
    )
    advantage_loss = _masked_mean(advantage_terms, residual_label_mask)

    harm_terms = F.binary_cross_entropy_with_logits(
        output.harm_logit,
        batch.harm_target,
        reduction="none",
        pos_weight=torch.as_tensor(
            config.harm_positive_weight,
            dtype=output.harm_logit.dtype,
            device=output.harm_logit.device),
    )
    harm_loss = _masked_mean(harm_terms, residual_label_mask)

    rank_mask = batch.selectable_mask & batch.rollout_label_valid
    if not torch.all((rank_mask & batch.native_mask).sum(dim=1) == 1):
        raise NLSRRankerError(
            "every listwise set requires its valid native baseline")
    target_utility = (
        batch.advantage_target
        - config.rank_harm_penalty_m * batch.harm_target)
    target_log_probability = _masked_log_softmax(
        target_utility / config.rank_target_temperature, rank_mask)
    target_probability = target_log_probability.exp().masked_fill(
        ~rank_mask, 0.0)
    predicted_log_probability = _masked_log_softmax(
        output.rank_score / config.rank_prediction_temperature, rank_mask)
    rank_per_set = -(
        target_probability
        * predicted_log_probability.masked_fill(~rank_mask, 0.0)
    ).sum(dim=-1)
    # A native-only list has probability one and zero CE by construction.  It
    # carries no pairwise ranking supervision and must not dilute supervised
    # sets merely because batch composition contains coverage misses or only
    # invalid residual labels.
    rank_set_mask = (rank_mask & batch.residual_mask).any(dim=1)
    rank_loss = _masked_mean(rank_per_set, rank_set_mask)

    coverage_terms = F.binary_cross_entropy_with_logits(
        output.coverage_logit,
        batch.coverage_miss_target,
        reduction="none",
        pos_weight=torch.as_tensor(
            config.coverage_positive_weight,
            dtype=output.coverage_logit.dtype,
            device=output.coverage_logit.device),
    )
    coverage_loss = _masked_mean(
        coverage_terms, batch.coverage_label_valid)
    total = (
        config.advantage_weight * advantage_loss
        + config.rank_weight * rank_loss
        + config.harm_weight * harm_loss
        + config.coverage_weight * coverage_loss
    )
    if any(not torch.isfinite(value).all() for value in (
            advantage_loss, rank_loss, harm_loss, coverage_loss, total)):
        raise NLSRRankerError(
            "loss became non-finite under the supplied targets/config")
    return NLSRLossBreakdown(
        total=total,
        advantage=advantage_loss,
        rank=rank_loss,
        harm=harm_loss,
        coverage=coverage_loss,
        advantage_count=int(residual_label_mask.sum().item()),
        rank_set_count=int(rank_set_mask.sum().item()),
        harm_count=int(residual_label_mask.sum().item()),
        coverage_count=int(batch.coverage_label_valid.sum().item()),
    )


@dataclass(frozen=True)
class DeclaredSceneGroupSplit:
    train: tuple[object, ...]
    development: tuple[object, ...]
    train_scenes: tuple[str, ...]
    development_scenes: tuple[str, ...]
    train_groups: tuple[str, ...]
    development_groups: tuple[str, ...]


def assert_scene_group_disjoint(
    train_records: Iterable[object],
    development_records: Iterable[object],
) -> None:
    """Fail closed on scene, group, session, or episode leakage."""
    train = list(train_records)
    development = list(development_records)
    if not train or not development:
        raise NLSRRankerError("train and development partitions must be nonempty")
    for row in train + development:
        validate_candidate_set(row)
    if any(row["provenance"]["split_role"] != "train" for row in train):
        raise NLSRRankerError("train partition contains a non-train split role")
    if any(row["provenance"]["split_role"] != "development"
           for row in development):
        raise NLSRRankerError(
            "development partition contains a non-development split role")

    def identifiers(rows: list[object], key: str) -> set[str]:
        return {str(row["provenance"][key]) for row in rows}

    for key in (
        "scene_id", "environment_id", "group_id",
        "session_id", "episode_id", "state_id",
    ):
        overlap = identifiers(train, key) & identifiers(development, key)
        if overlap:
            raise NLSRRankerError(
                f"train/development {key} overlap: {sorted(overlap)[:3]}")
    train_goal_sources = identifiers(train, "goal_source_episode_id")
    development_episodes = identifiers(development, "episode_id")
    development_goal_sources = identifiers(
        development, "goal_source_episode_id")
    train_episodes = identifiers(train, "episode_id")
    cross_goal = (
        (train_goal_sources & development_episodes)
        | (development_goal_sources & train_episodes))
    if cross_goal:
        raise NLSRRankerError(
            "goal-source episodes cross train/development partitions")


def split_by_declared_role(
    records: Iterable[object],
) -> DeclaredSceneGroupSplit:
    """Honor the immutable schema split and prove scene/group isolation."""
    rows = list(records)
    validate_candidate_dataset(rows)
    train = tuple(
        row for row in rows
        if row["provenance"]["split_role"] == "train")
    development = tuple(
        row for row in rows
        if row["provenance"]["split_role"] == "development")
    assert_scene_group_disjoint(train, development)
    return DeclaredSceneGroupSplit(
        train=train,
        development=development,
        train_scenes=tuple(sorted({
            str(row["provenance"]["scene_id"]) for row in train})),
        development_scenes=tuple(sorted({
            str(row["provenance"]["scene_id"]) for row in development})),
        train_groups=tuple(sorted({
            str(row["provenance"]["group_id"]) for row in train})),
        development_groups=tuple(sorted({
            str(row["provenance"]["group_id"]) for row in development})),
    )


def dataset_provenance_sha256(records: Iterable[object]) -> str:
    """Hash all causal provenance, independent of record/input order."""
    rows = list(records)
    validate_candidate_dataset(rows)
    provenance_rows = [
        {key: row["provenance"][key] for key in sorted(PROVENANCE_KEYS)}
        for row in rows
    ]
    provenance_rows.sort(key=lambda row: (
        str(row["scene_id"]), str(row["episode_id"]),
        str(row["session_id"]), str(row["state_id"]),
        str(row["goal_epoch"]), int(row["plan_index"]),
    ))
    return _sha256_json(provenance_rows)


def dataset_content_sha256(records: Iterable[object]) -> str:
    """Hash every validated feature, label, and provenance field canonically."""
    rows = list(records)
    validate_candidate_dataset(rows)
    ordered = sorted(rows, key=lambda row: (
        str(row["provenance"]["scene_id"]),
        str(row["provenance"]["episode_id"]),
        str(row["provenance"]["session_id"]),
        str(row["provenance"]["state_id"]),
        str(row["provenance"]["goal_epoch"]),
        int(row["provenance"]["plan_index"]),
    ))
    return _sha256_json(ordered)


def _source_provenance(records: Sequence[object]) -> dict[str, object]:
    provenance = records[0]["provenance"]
    keys = (
        "dataset_id", "split_sha256", "source_policy_sha256",
        "candidate_generator_sha256", "feature_builder_sha256",
        "rollout_labeler_sha256",
    )
    return {key: provenance[key] for key in keys}


def build_checkpoint_metadata(
    model: NLSRSetRanker,
    records: Iterable[object],
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build deterministic, JSON-portable model/data provenance metadata."""
    rows = list(records)
    report = validate_candidate_dataset(rows)
    inferred_spec = feature_spec_from_record(rows[0])
    if inferred_spec != model.feature_spec:
        raise NLSRRankerError("model feature spec differs from checkpoint data")
    source = _source_provenance(rows)
    artifact_signature_sha256 = _sha256_json(
        list(report["artifact_signature"]))
    if extra is None:
        extra_value: dict[str, object] = {}
    elif isinstance(extra, Mapping):
        extra_value = dict(extra)
        _canonical_json_bytes(extra_value)
    else:
        raise NLSRRankerError("checkpoint extra metadata must be a mapping")
    metadata = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_contract_sha256": model.feature_spec.schema_contract_sha256,
        "dataset_provenance_sha256": dataset_provenance_sha256(rows),
        "dataset_content_sha256": dataset_content_sha256(rows),
        "artifact_signature_sha256": artifact_signature_sha256,
        "state_dict_sha256": state_dict_sha256(model.state_dict()),
        "source_provenance": source,
        "feature_spec": model.feature_spec.to_dict(),
        "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_count,
        "extra": extra_value,
    }
    metadata["metadata_sha256"] = _sha256_json(metadata)
    return metadata


def make_portable_checkpoint(
    model: NLSRSetRanker,
    records: Iterable[object],
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a CPU-only state-dict checkpoint with strict JSON metadata."""
    metadata = build_checkpoint_metadata(model, records, extra=extra)
    state_dict = {
        key: value.detach().to(device="cpu").contiguous().clone()
        for key, value in model.state_dict().items()
    }
    if state_dict_sha256(state_dict) != metadata["state_dict_sha256"]:
        raise NLSRRankerError("portable state_dict changed during CPU export")
    return {"metadata": metadata, "state_dict": state_dict}


def load_portable_checkpoint(
    payload: object,
    *,
    expected_records: Iterable[object] | None = None,
) -> NLSRSetRanker:
    """Reconstruct a strict CPU model, optionally binding expected data."""
    if (not isinstance(payload, Mapping)
            or frozenset(payload) != {"metadata", "state_dict"}):
        raise NLSRRankerError("checkpoint payload keys are not exact")
    metadata = payload["metadata"]
    expected_metadata_keys = frozenset({
        "checkpoint_format_version", "schema_version",
        "schema_contract_sha256", "dataset_provenance_sha256",
        "dataset_content_sha256", "artifact_signature_sha256",
        "state_dict_sha256", "source_provenance", "feature_spec",
        "model_config", "parameter_count", "extra", "metadata_sha256",
    })
    if (not isinstance(metadata, Mapping)
            or frozenset(metadata) != expected_metadata_keys):
        raise NLSRRankerError("checkpoint metadata keys are not exact")
    if metadata["checkpoint_format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise NLSRRankerError("checkpoint format version mismatch")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise NLSRRankerError("checkpoint schema version mismatch")
    _canonical_json_bytes(dict(metadata))
    _require_sha256_text(
        metadata["metadata_sha256"], "checkpoint metadata_sha256")
    metadata_body = {
        key: value for key, value in metadata.items()
        if key != "metadata_sha256"
    }
    if _sha256_json(metadata_body) != metadata["metadata_sha256"]:
        raise NLSRRankerError("checkpoint metadata hash mismatch")
    spec = RankerFeatureSpec.from_dict(metadata["feature_spec"])
    if metadata["schema_contract_sha256"] != spec.schema_contract_sha256:
        raise NLSRRankerError("checkpoint schema hash mismatch")
    config = NLSRRankerConfig.from_dict(metadata["model_config"])
    model = NLSRSetRanker(spec, config)
    if (_require_plain_integer(
            metadata["parameter_count"], "checkpoint parameter_count")
            != model.parameter_count):
        raise NLSRRankerError("checkpoint parameter count mismatch")
    _require_sha256_text(
        metadata["dataset_provenance_sha256"],
        "checkpoint dataset_provenance_sha256")
    _require_sha256_text(
        metadata["dataset_content_sha256"],
        "checkpoint dataset_content_sha256")
    _require_sha256_text(
        metadata["artifact_signature_sha256"],
        "checkpoint artifact_signature_sha256")
    _require_sha256_text(
        metadata["state_dict_sha256"],
        "checkpoint state_dict_sha256")
    source = metadata["source_provenance"]
    source_keys = frozenset({
        "dataset_id", "split_sha256", "source_policy_sha256",
        "candidate_generator_sha256", "feature_builder_sha256",
        "rollout_labeler_sha256",
    })
    if not isinstance(source, Mapping) or frozenset(source) != source_keys:
        raise NLSRRankerError("checkpoint source provenance keys are not exact")
    if not isinstance(source["dataset_id"], str) or not source["dataset_id"]:
        raise NLSRRankerError("checkpoint dataset_id must be nonempty")
    for key in source_keys - {"dataset_id"}:
        _require_sha256_text(source[key], f"checkpoint source {key}")
    if expected_records is not None:
        rows = list(expected_records)
        if feature_spec_from_dataset(rows) != spec:
            raise NLSRRankerError("checkpoint/data feature spec mismatch")
        if (dataset_provenance_sha256(rows)
                != metadata["dataset_provenance_sha256"]):
            raise NLSRRankerError("checkpoint/data provenance hash mismatch")
        if (dataset_content_sha256(rows)
                != metadata["dataset_content_sha256"]):
            raise NLSRRankerError("checkpoint/data content hash mismatch")
        report = validate_candidate_dataset(rows)
        if (_sha256_json(list(report["artifact_signature"]))
                != metadata["artifact_signature_sha256"]):
            raise NLSRRankerError("checkpoint/data artifact signature mismatch")
        if _source_provenance(rows) != metadata["source_provenance"]:
            raise NLSRRankerError("checkpoint/data source provenance mismatch")
    if not isinstance(payload["state_dict"], Mapping):
        raise NLSRRankerError("checkpoint state_dict must be a mapping")
    if any(not isinstance(key, str) or not isinstance(value, Tensor)
           for key, value in payload["state_dict"].items()):
        raise NLSRRankerError("checkpoint state_dict entries must be tensors")
    if (state_dict_sha256(payload["state_dict"])
            != metadata["state_dict_sha256"]):
        raise NLSRRankerError("checkpoint state_dict hash mismatch")
    try:
        model.load_state_dict(dict(payload["state_dict"]), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise NLSRRankerError("checkpoint state_dict is incompatible") from exc
    return model
