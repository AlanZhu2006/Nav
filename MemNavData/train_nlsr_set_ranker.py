#!/usr/bin/env python3
"""Train and calibrate the lightweight NLSR candidate-set ranker.

The input is a canonical JSON array or canonical JSONL stream of complete
``novel_candidate_set_v2`` records.  The declared train scenes are split into
scene-disjoint core/tune partitions for early stopping, the selected epoch is
then refit on every train scene, and declared development scenes are opened
only for one final calibration pass.  Class weights are derived exclusively
from declared train records.

The resumable state binds full feature/label content, causal provenance,
feature shapes, source-code hashes, partitions, and all optimization settings.
Any contract drift fails closed instead of silently continuing a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import torch

# Keep the advertised shebang/direct-script path equivalent to ``python -m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MemNavData.nlsr_set_ranker as ranker_module
import MemNavData.novel_candidate_set_schema_v2 as schema_module
from MemNavData.nlsr_set_ranker import (
    CandidateSetBatch,
    NLSRLossConfig,
    NLSRRankerConfig,
    NLSRSetRanker,
    RankerFeatureSpec,
    compute_nlsr_losses,
    dataset_content_sha256,
    dataset_provenance_sha256,
    feature_spec_from_dataset,
    load_portable_checkpoint,
    make_portable_checkpoint,
    split_by_declared_role,
    state_dict_sha256,
    vectorize_candidate_sets,
)
from MemNavData.novel_candidate_set_schema_v2 import (
    SCHEMA_VERSION,
    validate_candidate_dataset,
)


TRAINER_FORMAT_VERSION = "nlsr_set_ranker_trainer_v2"
RESUME_FORMAT_VERSION = "nlsr_set_ranker_resume_v2"
OUTPUT_MANIFEST_VERSION = "nlsr_set_ranker_outputs_v2"
FINAL_CHECKPOINT_NAME = "nlsr_set_ranker.pt"
RESUME_NAME = "training_state.pt"
METRICS_NAME = "metrics.json"
CALIBRATION_NAME = "calibration.json"
PROVENANCE_NAME = "provenance.json"
MANIFEST_NAME = "manifest.json"
COVERAGE_POLICY_REQUIRED = "required"
COVERAGE_POLICY_ADVISORY_UNAVAILABLE = "advisory_unavailable"
COVERAGE_POLICIES = frozenset(
    {COVERAGE_POLICY_REQUIRED, COVERAGE_POLICY_ADVISORY_UNAVAILABLE}
)


class NLSRTrainingError(RuntimeError):
    """Raised when the training/calibration protocol must fail closed."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NLSRTrainingError("value is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def deterministic_torch_bytes(value: object) -> bytes:
    """Serialize a safe tensor tree with one deterministic torch format."""
    buffer = io.BytesIO()
    torch.save(
        value,
        buffer,
        pickle_protocol=2,
        _use_new_zipfile_serialization=True,
    )
    return buffer.getvalue()


def atomic_torch_save(path: Path, value: object) -> None:
    """Write a fixed-format torch payload from an in-memory byte string.

    The in-memory zip writer uses fixed archive metadata and deterministic
    storage numbering.  One fixed pickle protocol plus materializing the whole
    payload before the atomic rename makes byte hashes reproducible for
    identical tensor trees.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = deterministic_torch_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _without_one_trailing_newline(raw: bytes) -> bytes:
    return raw[:-1] if raw.endswith(b"\n") else raw


def load_canonical_candidate_rows(path: Path) -> tuple[list[dict], dict]:
    """Read exact canonical JSON/JSONL and validate the complete artifact."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    if not raw:
        raise NLSRTrainingError("candidate artifact is empty")
    suffix = path.suffix.lower()
    records: list[dict]
    if suffix == ".jsonl":
        payload = _without_one_trailing_newline(raw)
        lines = payload.split(b"\n")
        if not lines or any(not line for line in lines):
            raise NLSRTrainingError("canonical JSONL cannot contain blank rows")
        records = []
        for index, line in enumerate(lines):
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NLSRTrainingError(
                    f"invalid JSONL candidate row {index}") from exc
            if canonical_json_bytes(record) != line:
                raise NLSRTrainingError(
                    f"candidate JSONL row {index} is not canonical")
            if not isinstance(record, dict):
                raise NLSRTrainingError(
                    f"candidate JSONL row {index} must be an object")
            records.append(record)
        encoding = "canonical_jsonl_v1"
    elif suffix == ".json":
        payload = _without_one_trailing_newline(raw)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NLSRTrainingError("invalid JSON candidate artifact") from exc
        if canonical_json_bytes(value) != payload:
            raise NLSRTrainingError("candidate JSON artifact is not canonical")
        if not isinstance(value, list) or any(
                not isinstance(record, dict) for record in value):
            raise NLSRTrainingError(
                "canonical JSON artifact must be an array of records")
        records = value
        encoding = "canonical_json_array_v1"
    else:
        raise NLSRTrainingError(
            "candidate artifact suffix must be .json or .jsonl")
    report = validate_candidate_dataset(records)
    return records, {
        "path": str(path),
        "encoding": encoding,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "file_bytes": len(raw),
        "record_count": len(records),
        "dataset_content_sha256": dataset_content_sha256(records),
        "dataset_provenance_sha256": dataset_provenance_sha256(records),
        "schema_report": report,
    }


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    max_epochs: int = 200
    patience: int = 25
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    tune_scene_fraction: float = 0.2
    development_calibration_scene_fraction: float = 0.5
    minimum_improvement: float = 1e-5
    advantage_alpha: float = 0.05
    risk_alpha: float = 0.05
    target_harm_upper: float = 0.05
    target_coverage_miss_upper: float = 0.05
    minimum_advantage_lcb_m: float = 0.25
    minimum_advantage_calibration_scenes: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "seed", "max_epochs", "patience", "batch_size",
            "minimum_advantage_calibration_scenes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise NLSRTrainingError(f"config.{name} must be an integer")
        if self.seed < 0:
            raise NLSRTrainingError("config.seed must be non-negative")
        if min(self.max_epochs, self.patience, self.batch_size) < 1:
            raise NLSRTrainingError(
                "epochs, patience, and batch size must be positive")
        if self.minimum_advantage_calibration_scenes < 1:
            raise NLSRTrainingError(
                "minimum calibration scenes must be positive")
        for name in (
            "learning_rate", "gradient_clip_norm", "minimum_improvement",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise NLSRTrainingError(f"config.{name} must be positive")
        if (not math.isfinite(float(self.weight_decay))
                or self.weight_decay < 0.0):
            raise NLSRTrainingError(
                "config.weight_decay must be non-negative")
        for name in (
            "tune_scene_fraction", "development_calibration_scene_fraction",
            "advantage_alpha", "risk_alpha",
            "target_harm_upper", "target_coverage_miss_upper",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise NLSRTrainingError(f"config.{name} must be in (0, 1)")
        if (not math.isfinite(float(self.minimum_advantage_lcb_m))
                or self.minimum_advantage_lcb_m < 0.0):
            raise NLSRTrainingError(
                "minimum advantage LCB must be non-negative")
        if self.device not in ("cpu", "cuda"):
            raise NLSRTrainingError("config.device must be cpu or cuda")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def deterministic_tune_scenes(
    train_scenes: Sequence[str],
    fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scenes = sorted(set(map(str, train_scenes)))
    if len(scenes) < 2:
        raise NLSRTrainingError(
            "at least two train scenes are required for early stopping")
    ranked = sorted(scenes, key=lambda scene: (
        hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest(), scene))
    tune_count = max(1, int(round(len(scenes) * float(fraction))))
    tune_count = min(tune_count, len(scenes) - 1)
    tune = tuple(sorted(ranked[:tune_count]))
    core = tuple(sorted(set(scenes) - set(tune)))
    return core, tune


def _partition_train_records(
    train_records: Sequence[dict],
    fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], tuple[str, ...], tuple[str, ...]]:
    scenes = sorted({
        str(row["provenance"]["scene_id"]) for row in train_records})
    core_scenes, tune_scenes = deterministic_tune_scenes(
        scenes, fraction, seed)
    core_set, tune_set = set(core_scenes), set(tune_scenes)
    core = [
        row for row in train_records
        if str(row["provenance"]["scene_id"]) in core_set]
    tune = [
        row for row in train_records
        if str(row["provenance"]["scene_id"]) in tune_set]
    if not core or not tune:
        raise NLSRTrainingError("core/tune scene partition is empty")
    return core, tune, core_scenes, tune_scenes


def _partition_development_records(
    development_records: Sequence[dict],
    fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], tuple[str, ...], tuple[str, ...], bool]:
    """Create a deterministic scene-disjoint calibration/audit partition.

    With fewer than two development scenes, supervised calibration and an
    independent audit cannot both exist.  In that case calibration is empty,
    every development row remains available for a fail-closed audit, and the
    caller must keep ``shadow_eligible=false``.
    """
    scenes = sorted({
        str(row["provenance"]["scene_id"])
        for row in development_records
    })
    if len(scenes) < 2:
        return [], list(development_records), (), tuple(scenes), False
    audit_scenes, calibration_scenes = deterministic_tune_scenes(
        scenes, fraction, seed + 2_000_003)
    calibration_set = set(calibration_scenes)
    audit_set = set(audit_scenes)
    calibration = [
        row for row in development_records
        if str(row["provenance"]["scene_id"]) in calibration_set
    ]
    audit = [
        row for row in development_records
        if str(row["provenance"]["scene_id"]) in audit_set
    ]
    if not calibration or not audit or calibration_set & audit_set:
        raise NLSRTrainingError(
            "development calibration/audit partition is invalid")
    return (
        calibration,
        audit,
        tuple(sorted(calibration_set)),
        tuple(sorted(audit_set)),
        True,
    )


def _balanced_positive_weight(
    labels: Sequence[bool],
) -> tuple[float, dict[str, object]]:
    positives = sum(map(bool, labels))
    negatives = len(labels) - positives
    if positives and negatives:
        weight = float(negatives) / float(positives)
    else:
        weight = 1.0
    # Avoid a single rare row producing an arbitrarily large first update.
    weight = min(max(weight, 1.0 / 100.0), 100.0)
    return weight, {
        "valid": len(labels),
        "positive": positives,
        "negative": negatives,
        "positive_weight": weight,
        "both_classes_observed": bool(positives and negatives),
    }


def class_weights_from_train(
    train_records: Sequence[dict],
) -> tuple[NLSRLossConfig, dict[str, object]]:
    harm: list[bool] = []
    coverage: list[bool] = []
    for row in train_records:
        for candidate in row["candidates"]:
            if (candidate["candidate_type"] not in ("native", "dustbin")
                    and candidate["labels"]["rollout_label_valid"]):
                harm.append(bool(candidate["labels"]["harm"]))
        if row["set_labels"]["coverage_label_valid"]:
            coverage.append(bool(
                row["set_labels"]["candidate_coverage_miss"]))
    harm_weight, harm_report = _balanced_positive_weight(harm)
    coverage_weight, coverage_report = _balanced_positive_weight(coverage)
    return NLSRLossConfig(
        harm_positive_weight=harm_weight,
        coverage_positive_weight=coverage_weight,
    ), {
        "source_split_role": "train",
        "harm": harm_report,
        "coverage_miss": coverage_report,
    }


def _module_sha256(module: object) -> str:
    source = getattr(module, "__file__", None)
    if not source:
        raise NLSRTrainingError("module has no source path")
    return sha256_file(Path(source))


def _training_contract(
    records: Sequence[dict],
    artifact_audit: Mapping[str, object],
    feature_spec: RankerFeatureSpec,
    config: TrainingConfig,
    selection_loss_config: NLSRLossConfig,
    refit_loss_config: NLSRLossConfig,
    core_scenes: Sequence[str],
    tune_scenes: Sequence[str],
    development_scenes: Sequence[str],
    development_calibration_scenes: Sequence[str],
    development_audit_scenes: Sequence[str],
    selection_class_balance: Mapping[str, object],
    refit_class_balance: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    def loss_dict(value: NLSRLossConfig) -> dict[str, object]:
        return {
            field: getattr(value, field)
            for field in value.__dataclass_fields__
        }

    payload = {
        "trainer_format_version": TRAINER_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact_file_sha256": artifact_audit["file_sha256"],
        "dataset_content_sha256": dataset_content_sha256(records),
        "dataset_provenance_sha256": dataset_provenance_sha256(records),
        "schema_contract_sha256": feature_spec.schema_contract_sha256,
        "feature_spec": feature_spec.to_dict(),
        "training_config": config.to_dict(),
        "loss_config": {
            "selection_core": loss_dict(selection_loss_config),
            "refit_all_train": loss_dict(refit_loss_config),
        },
        "class_balance": {
            "selection_core": dict(selection_class_balance),
            "refit_all_train": dict(refit_class_balance),
        },
        "partitions": {
            "core_scenes": list(core_scenes),
            "tune_scenes": list(tune_scenes),
            "development_scenes": list(development_scenes),
            "development_calibration_scenes": list(
                development_calibration_scenes),
            "development_audit_scenes": list(development_audit_scenes),
        },
        "source_sha256": {
            "trainer": sha256_file(Path(__file__)),
            "ranker": _module_sha256(ranker_module),
            "schema": _module_sha256(schema_module),
        },
        "runtime": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": str(torch.__version__),
            "device": config.device,
        },
    }
    return payload, canonical_sha256(payload)


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _canonical_tree(value: object) -> object:
    """Convert a safe optimizer/resume tree into canonical hash material."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise NLSRTrainingError("resume tensor contains non-finite values")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, Mapping):
        rows = [
            [_canonical_tree(key), _canonical_tree(item)]
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: canonical_json_bytes(row[0]))
        return {"kind": "mapping", "items": rows}
    if isinstance(value, list):
        return {"kind": "list", "items": [_canonical_tree(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_canonical_tree(item) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise NLSRTrainingError("resume scalar contains non-finite value")
        return {"kind": type(value).__name__, "value": value}
    raise NLSRTrainingError(
        f"unsupported resume value type: {type(value).__name__}")


def resume_payload_sha256(state: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in state.items()
        if key != "resume_payload_sha256"
    }
    return canonical_sha256(_canonical_tree(payload))


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _new_model(
    feature_spec: RankerFeatureSpec,
    seed: int,
    device: torch.device,
) -> NLSRSetRanker:
    model = NLSRSetRanker(
        feature_spec, NLSRRankerConfig(init_seed=seed))
    model.assert_lightweight_parameter_budget()
    return model.to(device)


def _new_optimizer(
    model: NLSRSetRanker,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _batch_indices(count: int, batch_size: int, seed: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(count, generator=generator).tolist()
    return [order[start:start + batch_size]
            for start in range(0, count, batch_size)]


def _slice_batch(
    packed: CandidateSetBatch,
    indices: Sequence[int],
    device: torch.device,
) -> CandidateSetBatch:
    index = torch.as_tensor(indices, dtype=torch.long,
                            device=packed.candidate_features.device)
    tensor_names = (
        "candidate_features", "set_features", "valid_mask", "native_mask",
        "dustbin_mask", "residual_mask", "selectable_mask",
        "advantage_target", "harm_target", "rollout_label_valid",
        "coverage_miss_target", "coverage_label_valid",
    )
    tensors = {
        name: getattr(packed, name).index_select(0, index).to(device)
        for name in tensor_names
    }
    return CandidateSetBatch(
        **tensors,
        candidate_ids=tuple(packed.candidate_ids[item] for item in indices),
        scene_ids=tuple(packed.scene_ids[item] for item in indices),
        group_ids=tuple(packed.group_ids[item] for item in indices),
    )


def train_one_epoch(
    model: NLSRSetRanker,
    optimizer: torch.optim.Optimizer,
    packed: CandidateSetBatch,
    loss_config: NLSRLossConfig,
    config: TrainingConfig,
    epoch_seed: int,
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for indices in _batch_indices(
            packed.batch_size, config.batch_size, epoch_seed):
        batch = _slice_batch(packed, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_nlsr_losses(
            model.forward_batch(batch), batch, loss_config)
        if not torch.isfinite(loss.total):
            raise NLSRTrainingError("training loss became non-finite")
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise NLSRTrainingError("gradient norm became non-finite")
        optimizer.step()
        total += float(loss.total.detach().cpu()) * len(indices)
        count += len(indices)
    return total / max(count, 1)


def evaluate_loss(
    model: NLSRSetRanker,
    packed: CandidateSetBatch,
    loss_config: NLSRLossConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    component_counts = {
        "advantage": 0,
        "rank": 0,
        "harm": 0,
        "coverage": 0,
    }
    component_sums = {key: 0.0 for key in component_counts}
    with torch.no_grad():
        for start in range(0, packed.batch_size, batch_size):
            indices = list(range(
                start, min(start + batch_size, packed.batch_size)))
            batch = _slice_batch(packed, indices, device)
            loss = compute_nlsr_losses(
                model.forward_batch(batch), batch, loss_config)
            counts = {
                "advantage": loss.advantage_count,
                "rank": loss.rank_set_count,
                "harm": loss.harm_count,
                "coverage": loss.coverage_count,
            }
            for key, count in counts.items():
                component_sums[key] += (
                    float(getattr(loss, key).detach().cpu()) * count)
                component_counts[key] += count
    if (packed.batch_size < 1
            or any(not math.isfinite(value)
                   for value in component_sums.values())):
        raise NLSRTrainingError("evaluation loss is invalid")
    components = {
        key: (
            component_sums[key] / component_counts[key]
            if component_counts[key] else 0.0)
        for key in component_sums
    }
    components["total"] = (
        loss_config.advantage_weight * components["advantage"]
        + loss_config.rank_weight * components["rank"]
        + loss_config.harm_weight * components["harm"]
        + loss_config.coverage_weight * components["coverage"]
    )
    if any(not math.isfinite(value) for value in components.values()):
        raise NLSRTrainingError("evaluation aggregate is invalid")
    return {
        key: components[key]
        for key in ("total", "advantage", "rank", "harm", "coverage")
    }


def _state_dict_cpu(model: NLSRSetRanker) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items()
    }


def _make_resume_state(
    contract_sha256: str,
    model: NLSRSetRanker,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    model_state = _state_dict_cpu(model)
    return {
        "resume_format_version": RESUME_FORMAT_VERSION,
        "contract_sha256": contract_sha256,
        "phase": "selection",
        "next_epoch": 0,
        "model_state": model_state,
        "model_state_sha256": state_dict_sha256(model_state),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "best_model_state": None,
        "best_model_state_sha256": None,
        "best_epoch": -1,
        "best_metric": None,
        "stale_epochs": 0,
        "selected_epochs": None,
        "refit_initial_model_state_sha256": None,
        "selection_history": [],
        "refit_history": [],
        "resume_payload_sha256": None,
    }


def _validate_resume_state(
    state: object,
    contract_sha256: str,
    config: TrainingConfig,
) -> dict[str, object]:
    expected = frozenset({
        "resume_format_version", "contract_sha256", "phase", "next_epoch",
        "model_state", "model_state_sha256", "optimizer_state",
        "best_model_state", "best_model_state_sha256", "best_epoch",
        "best_metric", "stale_epochs", "selected_epochs",
        "refit_initial_model_state_sha256",
        "selection_history", "refit_history", "resume_payload_sha256",
    })
    if not isinstance(state, dict) or frozenset(state) != expected:
        raise NLSRTrainingError("resume state keys are not exact")
    if state["resume_format_version"] != RESUME_FORMAT_VERSION:
        raise NLSRTrainingError("resume format version mismatch")
    if state["contract_sha256"] != contract_sha256:
        raise NLSRTrainingError(
            "resume refused: training/data contract drifted")
    if (not isinstance(state["resume_payload_sha256"], str)
            or resume_payload_sha256(state)
            != state["resume_payload_sha256"]):
        raise NLSRTrainingError("resume payload hash mismatch")
    if state["phase"] not in ("selection", "refit", "complete"):
        raise NLSRTrainingError("resume phase is invalid")
    if (not isinstance(state["model_state"], Mapping)
            or state_dict_sha256(state["model_state"])
            != state["model_state_sha256"]):
        raise NLSRTrainingError("resume model state hash mismatch")
    best_state = state["best_model_state"]
    if best_state is None:
        if state["best_model_state_sha256"] is not None:
            raise NLSRTrainingError("resume best-state hash is inconsistent")
    elif (not isinstance(best_state, Mapping)
          or state_dict_sha256(best_state)
          != state["best_model_state_sha256"]):
        raise NLSRTrainingError("resume best model state hash mismatch")
    for key in ("next_epoch", "best_epoch", "stale_epochs"):
        if (isinstance(state[key], bool) or not isinstance(state[key], int)):
            raise NLSRTrainingError(f"resume {key} must be an integer")
        if key != "best_epoch" and int(state[key]) < 0:
            raise NLSRTrainingError(f"resume {key} must be non-negative")
    if not isinstance(state["selection_history"], list) or not isinstance(
            state["refit_history"], list):
        raise NLSRTrainingError("resume history must be a list")
    if not isinstance(state["optimizer_state"], Mapping):
        raise NLSRTrainingError("resume optimizer state must be a mapping")

    metric_keys = frozenset({
        "total", "advantage", "rank", "harm", "coverage",
    })

    def finite_number(value: object, location: str) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise NLSRTrainingError(f"{location} must be finite numeric")
        return float(value)

    best_metric: float | None = None
    best_epoch = -1
    stale_epochs = 0
    for index, row in enumerate(state["selection_history"]):
        if (not isinstance(row, Mapping) or frozenset(row) != {
                "epoch", "core_train_total", "tune", "improved",
                "model_state_sha256"}):
            raise NLSRTrainingError(
                "selection history row keys are not exact")
        if (isinstance(row["epoch"], bool)
                or row["epoch"] != index + 1):
            raise NLSRTrainingError(
                "selection history epochs are not consecutive")
        finite_number(
            row["core_train_total"], "selection core train total")
        tune = row["tune"]
        if not isinstance(tune, Mapping) or frozenset(tune) != metric_keys:
            raise NLSRTrainingError("selection tune metrics are not exact")
        for key in metric_keys:
            finite_number(tune[key], f"selection tune {key}")
        if not isinstance(row["improved"], bool):
            raise NLSRTrainingError(
                "selection improved flag must be boolean")
        model_hash = row["model_state_sha256"]
        if (not isinstance(model_hash, str) or len(model_hash) != 64
                or any(character not in "0123456789abcdef"
                       for character in model_hash)):
            raise NLSRTrainingError(
                "selection history model hash is invalid")
        metric = float(tune["total"])
        expected_improved = (
            best_metric is None
            or metric < best_metric - config.minimum_improvement)
        if row["improved"] != expected_improved:
            raise NLSRTrainingError(
                "selection improved flag violates early-stop recurrence")
        if expected_improved:
            best_metric = metric
            best_epoch = index
            stale_epochs = 0
        else:
            stale_epochs += 1

    history_count = len(state["selection_history"])
    if history_count > config.max_epochs:
        raise NLSRTrainingError("selection history exceeds max epochs")
    if int(state["best_epoch"]) != best_epoch:
        raise NLSRTrainingError("resume best epoch disagrees with history")
    if state["best_metric"] is None:
        if best_metric is not None:
            raise NLSRTrainingError("resume best metric is missing")
    elif (best_metric is None
          or finite_number(state["best_metric"], "resume best metric")
          != best_metric):
        raise NLSRTrainingError("resume best metric disagrees with history")
    if int(state["stale_epochs"]) != stale_epochs:
        raise NLSRTrainingError("resume stale epochs disagree with history")
    if history_count == 0:
        if best_state is not None or best_epoch != -1:
            raise NLSRTrainingError(
                "empty selection history cannot have a best state")
    elif best_state is None:
        raise NLSRTrainingError(
            "nonempty selection history requires a best state")
    elif (state["best_model_state_sha256"]
          != state["selection_history"][best_epoch]["model_state_sha256"]):
        raise NLSRTrainingError(
            "resume best state disagrees with its selection epoch")

    for index, row in enumerate(state["refit_history"]):
        if (not isinstance(row, Mapping) or frozenset(row) != {
                "epoch", "full_train_total", "model_state_sha256"}):
            raise NLSRTrainingError("refit history row keys are not exact")
        if (isinstance(row["epoch"], bool)
                or row["epoch"] != index + 1):
            raise NLSRTrainingError(
                "refit history epochs are not consecutive")
        finite_number(row["full_train_total"], "refit train total")
        model_hash = row["model_state_sha256"]
        if (not isinstance(model_hash, str) or len(model_hash) != 64
                or any(character not in "0123456789abcdef"
                       for character in model_hash)):
            raise NLSRTrainingError("refit history model hash is invalid")

    phase = str(state["phase"])
    next_epoch = int(state["next_epoch"])
    selected_epochs = state["selected_epochs"]
    if phase == "selection":
        if next_epoch != history_count:
            raise NLSRTrainingError(
                "selection next_epoch disagrees with history")
        if selected_epochs is not None or state["refit_history"]:
            raise NLSRTrainingError(
                "selection phase cannot contain refit state")
        if state["refit_initial_model_state_sha256"] is not None:
            raise NLSRTrainingError(
                "selection phase cannot contain a refit initial state")
        if (history_count
                and state["model_state_sha256"]
                != state["selection_history"][-1]["model_state_sha256"]):
            raise NLSRTrainingError(
                "selection current state disagrees with its history")
        if (history_count >= config.max_epochs
                or stale_epochs >= config.patience):
            raise NLSRTrainingError(
                "selection phase persisted past its stopping boundary")
    else:
        initial_refit_hash = state["refit_initial_model_state_sha256"]
        if (not isinstance(initial_refit_hash, str)
                or len(initial_refit_hash) != 64
                or any(character not in "0123456789abcdef"
                       for character in initial_refit_hash)):
            raise NLSRTrainingError("refit initial state hash is invalid")
        if (isinstance(selected_epochs, bool)
                or not isinstance(selected_epochs, int)
                or selected_epochs < 1):
            raise NLSRTrainingError(
                "refit/complete phase requires selected epochs")
        if selected_epochs != best_epoch + 1:
            raise NLSRTrainingError(
                "selected epochs disagree with the best selection epoch")
        if not (history_count >= config.max_epochs
                or stale_epochs >= config.patience):
            raise NLSRTrainingError(
                "refit began before the selection stopping boundary")
        if next_epoch != len(state["refit_history"]):
            raise NLSRTrainingError(
                "refit next_epoch disagrees with history")
        expected_current_hash = (
            state["refit_history"][-1]["model_state_sha256"]
            if state["refit_history"] else initial_refit_hash)
        if state["model_state_sha256"] != expected_current_hash:
            raise NLSRTrainingError(
                "refit current state disagrees with its history")
        if phase == "refit" and next_epoch >= selected_epochs:
            raise NLSRTrainingError(
                "refit phase persisted after its selected epoch count")
        if phase == "complete" and next_epoch != selected_epochs:
            raise NLSRTrainingError(
                "complete phase has an incomplete refit history")
    return state


def _load_resume(
    path: Path,
    contract_sha256: str,
    config: TrainingConfig,
) -> dict[str, object]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise NLSRTrainingError("cannot load safe resume state") from exc
    return _validate_resume_state(state, contract_sha256, config)


def _restore_model_optimizer(
    state: Mapping[str, object],
    feature_spec: RankerFeatureSpec,
    model_seed: int,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[NLSRSetRanker, torch.optim.Optimizer]:
    model = _new_model(feature_spec, model_seed, device)
    try:
        model.load_state_dict(state["model_state"], strict=True)
    except RuntimeError as exc:
        raise NLSRTrainingError("resume model shape mismatch") from exc
    optimizer = _new_optimizer(model, config)
    try:
        optimizer.load_state_dict(state["optimizer_state"])
    except (ValueError, RuntimeError) as exc:
        raise NLSRTrainingError("resume optimizer state mismatch") from exc
    _optimizer_to_device(optimizer, device)
    return model, optimizer


def _save_resume(path: Path, state: dict[str, object]) -> None:
    state["model_state_sha256"] = state_dict_sha256(state["model_state"])
    if state["best_model_state"] is not None:
        state["best_model_state_sha256"] = state_dict_sha256(
            state["best_model_state"])
    state["resume_payload_sha256"] = resume_payload_sha256(state)
    atomic_torch_save(path, state)


@dataclass(frozen=True)
class FitResult:
    complete: bool
    model: NLSRSetRanker | None
    selected_epochs: int | None
    state: dict[str, object]
    epochs_run_this_call: int


def fit_with_resume(
    *,
    core_records: Sequence[dict],
    tune_records: Sequence[dict],
    train_records: Sequence[dict],
    feature_spec: RankerFeatureSpec,
    selection_loss_config: NLSRLossConfig,
    refit_loss_config: NLSRLossConfig,
    config: TrainingConfig,
    contract_sha256: str,
    resume_path: Path,
    resume: bool,
    epoch_budget: int = 0,
) -> FitResult:
    if isinstance(epoch_budget, bool) or not isinstance(epoch_budget, int):
        raise NLSRTrainingError("epoch budget must be an integer")
    if epoch_budget < 0:
        raise NLSRTrainingError("epoch budget must be non-negative")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise NLSRTrainingError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
    core_batch = vectorize_candidate_sets(
        core_records, feature_spec=feature_spec)
    tune_batch = vectorize_candidate_sets(
        tune_records, feature_spec=feature_spec)
    train_batch = vectorize_candidate_sets(
        train_records, feature_spec=feature_spec)
    if resume:
        if not resume_path.is_file():
            raise NLSRTrainingError("--resume requested but state is absent")
        state = _load_resume(resume_path, contract_sha256, config)
    else:
        if resume_path.exists():
            raise NLSRTrainingError(
                "training state already exists; use --resume")
        model = _new_model(feature_spec, config.seed, device)
        optimizer = _new_optimizer(model, config)
        state = _make_resume_state(contract_sha256, model, optimizer)
        _save_resume(resume_path, state)
    epochs_run = 0

    while state["phase"] != "complete":
        if epoch_budget and epochs_run >= epoch_budget:
            return FitResult(
                complete=False, model=None,
                selected_epochs=state["selected_epochs"], state=state,
                epochs_run_this_call=epochs_run)
        if state["phase"] == "selection":
            model, optimizer = _restore_model_optimizer(
                state, feature_spec, config.seed, config, device)
            epoch = int(state["next_epoch"])
            train_loss = train_one_epoch(
                model, optimizer, core_batch, selection_loss_config,
                config, config.seed * 1_000_003 + epoch, device)
            tune_metrics = evaluate_loss(
                model, tune_batch, selection_loss_config,
                config.batch_size, device)
            metric = float(tune_metrics["total"])
            epoch_model_state = _state_dict_cpu(model)
            epoch_model_state_sha256 = state_dict_sha256(epoch_model_state)
            best_metric = state["best_metric"]
            improved = (
                best_metric is None
                or metric < float(best_metric) - config.minimum_improvement)
            if improved:
                best_state = _cpu_tree(epoch_model_state)
                state["best_model_state"] = best_state
                state["best_model_state_sha256"] = state_dict_sha256(best_state)
                state["best_epoch"] = epoch
                state["best_metric"] = metric
                state["stale_epochs"] = 0
            else:
                state["stale_epochs"] = int(state["stale_epochs"]) + 1
            state["selection_history"].append({
                "epoch": epoch + 1,
                "core_train_total": train_loss,
                "tune": tune_metrics,
                "improved": improved,
                "model_state_sha256": epoch_model_state_sha256,
            })
            state["next_epoch"] = epoch + 1
            state["model_state"] = epoch_model_state
            state["optimizer_state"] = _cpu_tree(optimizer.state_dict())
            epochs_run += 1
            stop = (
                int(state["stale_epochs"]) >= config.patience
                or int(state["next_epoch"]) >= config.max_epochs)
            if stop:
                if int(state["best_epoch"]) < 0:
                    raise NLSRTrainingError("early stopping never selected an epoch")
                selected = int(state["best_epoch"]) + 1
                refit_model = _new_model(
                    feature_spec, config.seed + 1_000_003, device)
                refit_optimizer = _new_optimizer(refit_model, config)
                state.update({
                    "phase": "refit",
                    "next_epoch": 0,
                    "selected_epochs": selected,
                    "model_state": _state_dict_cpu(refit_model),
                    "optimizer_state": _cpu_tree(refit_optimizer.state_dict()),
                })
                state["refit_initial_model_state_sha256"] = (
                    state_dict_sha256(state["model_state"]))
            # Persist only a phase-valid state.  In particular, never expose a
            # selection checkpoint that has already crossed its stop boundary.
            _save_resume(resume_path, state)
        elif state["phase"] == "refit":
            model, optimizer = _restore_model_optimizer(
                state, feature_spec, config.seed + 1_000_003,
                config, device)
            epoch = int(state["next_epoch"])
            selected = int(state["selected_epochs"])
            if epoch >= selected:
                state["phase"] = "complete"
                _save_resume(resume_path, state)
                continue
            train_loss = train_one_epoch(
                model, optimizer, train_batch, refit_loss_config,
                config, (config.seed + 1_000_003) * 1_000_003 + epoch,
                device)
            epoch_model_state = _state_dict_cpu(model)
            epoch_model_state_sha256 = state_dict_sha256(epoch_model_state)
            state["refit_history"].append({
                "epoch": epoch + 1,
                "full_train_total": train_loss,
                "model_state_sha256": epoch_model_state_sha256,
            })
            state["next_epoch"] = epoch + 1
            state["model_state"] = epoch_model_state
            state["optimizer_state"] = _cpu_tree(optimizer.state_dict())
            if int(state["next_epoch"]) >= selected:
                state["phase"] = "complete"
            _save_resume(resume_path, state)
            epochs_run += 1

    final_model = _new_model(
        feature_spec, config.seed + 1_000_003, torch.device(config.device))
    final_model.load_state_dict(state["model_state"], strict=True)
    final_model.eval()
    return FitResult(
        complete=True,
        model=final_model,
        selected_epochs=int(state["selected_epochs"]),
        state=state,
        epochs_run_this_call=epochs_run,
    )


def finite_sample_lcb_quantile(
    normalized_overprediction: Sequence[float],
    alpha: float,
) -> tuple[float, int]:
    values = sorted(float(value) for value in normalized_overprediction)
    if not values or any(not math.isfinite(value) for value in values):
        raise NLSRTrainingError(
            "advantage calibration residuals must be finite and nonempty")
    if not 0.0 < alpha < 1.0:
        raise NLSRTrainingError("conformal alpha must be in (0, 1)")
    rank = min(len(values), int(math.ceil((len(values) + 1) * (1.0 - alpha))))
    quantile = max(0.0, values[rank - 1])
    return quantile, rank


def zero_failure_upper_bound(count: int, alpha: float) -> float:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise NLSRTrainingError("independent count must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise NLSRTrainingError("risk alpha must be in (0, 1)")
    return 1.0 if count == 0 else 1.0 - alpha ** (1.0 / count)


def calibrate_zero_bad_threshold(
    probabilities: Sequence[float],
    bad_labels: Sequence[bool],
    valid: Sequence[bool],
    unit_ids: Sequence[str],
    *,
    alpha: float,
    target_upper: float,
) -> dict[str, object]:
    if not (len(probabilities) == len(bad_labels) == len(valid)
            == len(unit_ids)):
        raise NLSRTrainingError("threshold calibration arrays differ in length")
    rows = []
    for probability, bad, is_valid, unit_id in zip(
            probabilities, bad_labels, valid, unit_ids):
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise NLSRTrainingError("calibration probability outside [0, 1]")
        if is_valid:
            rows.append((probability, bool(bad), str(unit_id)))
    # -1 is a genuine fail-closed sentinel because calibrated probabilities
    # are in [0, 1]; using 0 could accidentally activate an underflowed sigmoid.
    thresholds = [-1.0] + sorted({row[0] for row in rows})
    zero_bad_options = []
    passing_options = []
    for threshold in thresholds:
        activated = [row for row in rows if row[0] <= threshold]
        unit_bad: dict[str, bool] = {}
        for _probability, bad, unit_id in activated:
            unit_bad[unit_id] = unit_bad.get(unit_id, False) or bad
        bad_units = sum(unit_bad.values())
        independent_units = len(unit_bad)
        upper = (
            zero_failure_upper_bound(independent_units, alpha)
            if bad_units == 0 else 1.0)
        result = {
            "threshold": threshold,
            "valid_rows": len(rows),
            "activated_rows": len(activated),
            "independent_units": independent_units,
            "bad_units": bad_units,
            "one_sided_upper": upper,
            "target_upper": float(target_upper),
            "bound_passed": bool(bad_units == 0 and upper <= target_upper),
        }
        if bad_units == 0:
            zero_bad_options.append(result)
            if result["bound_passed"]:
                passing_options.append(result)
    pool = passing_options or zero_bad_options
    if not pool:
        raise NLSRTrainingError("threshold calibration has no safe sentinel")
    selected = sorted(
        pool,
        key=lambda item: (
            -int(item["activated_rows"]), float(item["threshold"])),
    )[0]
    selected = dict(selected)
    selected["statistically_supported"] = bool(passing_options)
    return selected


def _sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(value).detach().cpu()


def _fail_closed_calibration(
    config: TrainingConfig,
    *,
    reason: str,
) -> dict[str, object]:
    threshold = {
        "threshold": -1.0,
        "valid_rows": 0,
        "activated_rows": 0,
        "independent_units": 0,
        "bad_units": 0,
        "one_sided_upper": 1.0,
        "target_upper": 0.0,
        "bound_passed": False,
        "statistically_supported": False,
        "independence_unit": "scene_id",
    }
    return {
        "calibration_format_version": "nlsr_set_ranker_calibration_v3",
        "development_only": True,
        "scene_disjoint_audit": False,
        "calibration_scenes": [],
        "audit_scenes": [],
        "advantage": {
            "alpha": config.advantage_alpha,
            "calibration_rows": 0,
            "calibration_scenes": 0,
            "minimum_scenes": config.minimum_advantage_calibration_scenes,
            "score_aggregation": (
                "maximum_normalized_overprediction_per_scene"),
            "supported": False,
            "one_sided_normalized_quantile": 0.0,
            "finite_sample_rank": 0,
            "minimum_lcb_m": config.minimum_advantage_lcb_m,
        },
        "harm": dict(threshold),
        "coverage_miss_abstain": dict(threshold),
        "coverage_policy": COVERAGE_POLICY_REQUIRED,
        "rank": {"minimum_residual_minus_native_score": 0.0},
        "decision_protocol": {
            "inputs": "model_outputs_and_structural_candidate_masks_only",
            "ground_truth_validity_used_for_decision": False,
            "selected_invalid_label": "retain_decision_mark_unevaluable",
        },
        "fail_closed": {
            "reason": str(reason),
            "risk_threshold_sentinel": -1.0,
            "no_eligible_residual": "native",
        },
        "shadow_eligible": False,
        "deployment_approved": False,
    }


def fit_development_calibration(
    model: NLSRSetRanker,
    calibration_records: Sequence[dict],
    feature_spec: RankerFeatureSpec,
    config: TrainingConfig,
) -> dict[str, object]:
    """Fit thresholds on development-calibration scenes only."""
    if not calibration_records:
        return _fail_closed_calibration(
            config, reason="insufficient_scene_disjoint_development_scenes")
    device = next(model.parameters()).device
    batch = vectorize_candidate_sets(
        calibration_records, feature_spec=feature_spec).to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward_batch(batch)
    mean = output.advantage_mean.detach().cpu()
    scale = output.advantage_log_scale.detach().cpu().exp()
    harm_probability = _sigmoid(output.harm_logit)
    coverage_probability = _sigmoid(output.coverage_logit)
    cpu_batch = batch.to("cpu")
    residual_valid = cpu_batch.residual_mask & cpu_batch.rollout_label_valid
    normalized_rows = (
        (mean - cpu_batch.advantage_target) / scale
    )[residual_valid].tolist()
    normalized_by_scene: dict[str, list[float]] = {}
    for row_index, record in enumerate(calibration_records):
        scene_id = str(record["provenance"]["scene_id"])
        for candidate_index, candidate in enumerate(record["candidates"]):
            if (candidate["candidate_type"] not in ("native", "dustbin")
                    and candidate["labels"]["rollout_label_valid"]):
                normalized_by_scene.setdefault(scene_id, []).append(float(
                    (mean[row_index, candidate_index]
                     - cpu_batch.advantage_target[row_index, candidate_index])
                    / scale[row_index, candidate_index]))
    normalized_scene_max = [
        max(values) for _scene, values in sorted(normalized_by_scene.items())]
    if normalized_scene_max:
        q_lcb, conformal_rank = finite_sample_lcb_quantile(
            normalized_scene_max, config.advantage_alpha)
    else:
        q_lcb, conformal_rank = 0.0, 0

    harm_probabilities: list[float] = []
    harm_labels: list[bool] = []
    harm_valid: list[bool] = []
    harm_units: list[str] = []
    for row_index, record in enumerate(calibration_records):
        for candidate_index, candidate in enumerate(record["candidates"]):
            is_residual = candidate["candidate_type"] not in (
                "native", "dustbin")
            is_valid = bool(
                is_residual and candidate["labels"]["rollout_label_valid"])
            harm_probabilities.append(float(
                harm_probability[row_index, candidate_index]))
            harm_labels.append(bool(candidate["labels"]["harm"]))
            harm_valid.append(is_valid)
            harm_units.append(str(record["provenance"]["scene_id"]))
    harm_calibration = calibrate_zero_bad_threshold(
        harm_probabilities, harm_labels, harm_valid, harm_units,
        alpha=config.risk_alpha, target_upper=config.target_harm_upper)
    harm_calibration["independence_unit"] = "scene_id"
    coverage_calibration = calibrate_zero_bad_threshold(
        coverage_probability.tolist(),
        [bool(row["set_labels"]["candidate_coverage_miss"])
         for row in calibration_records],
        [bool(row["set_labels"]["coverage_label_valid"])
         for row in calibration_records],
        [str(row["provenance"]["scene_id"])
         for row in calibration_records],
        alpha=config.risk_alpha,
        target_upper=config.target_coverage_miss_upper,
    )
    coverage_calibration["independence_unit"] = "scene_id"
    coverage_policy = (
        COVERAGE_POLICY_REQUIRED
        if int(coverage_calibration["valid_rows"]) > 0
        else COVERAGE_POLICY_ADVISORY_UNAVAILABLE
    )

    advantage_supported = (
        len(normalized_scene_max)
        >= config.minimum_advantage_calibration_scenes)
    calibration = {
        "calibration_format_version": "nlsr_set_ranker_calibration_v3",
        "development_only": True,
        "scene_disjoint_audit": True,
        "calibration_scenes": sorted({
            str(row["provenance"]["scene_id"])
            for row in calibration_records
        }),
        "audit_scenes": [],
        "advantage": {
            "alpha": config.advantage_alpha,
            "calibration_rows": len(normalized_rows),
            "calibration_scenes": len(normalized_scene_max),
            "minimum_scenes": config.minimum_advantage_calibration_scenes,
            "score_aggregation": "maximum_normalized_overprediction_per_scene",
            "supported": advantage_supported,
            "one_sided_normalized_quantile": q_lcb,
            "finite_sample_rank": conformal_rank,
            "minimum_lcb_m": config.minimum_advantage_lcb_m,
        },
        "harm": harm_calibration,
        "coverage_miss_abstain": coverage_calibration,
        "coverage_policy": coverage_policy,
        "rank": {
            "minimum_residual_minus_native_score": 0.0,
        },
        "decision_protocol": {
            "inputs": "model_outputs_and_structural_candidate_masks_only",
            "ground_truth_validity_used_for_decision": False,
            "selected_invalid_label": "retain_decision_mark_unevaluable",
        },
        "fail_closed": {
            "no_eligible_residual": "native",
        },
        "shadow_eligible": bool(
            advantage_supported
            and harm_calibration["statistically_supported"]
            and (
                coverage_policy == COVERAGE_POLICY_ADVISORY_UNAVAILABLE
                or coverage_calibration["statistically_supported"]
            )),
        "deployment_approved": False,
    }
    return calibration


def structural_policy_decisions(
    output: object,
    batch: CandidateSetBatch,
    calibration: Mapping[str, object],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    """Select candidates without consulting any rollout/coverage GT field."""
    mean = output.advantage_mean.detach().cpu()
    scale = output.advantage_log_scale.detach().cpu().exp()
    rank = output.rank_score.detach().cpu()
    harm_probability = _sigmoid(output.harm_logit)
    coverage_probability = _sigmoid(output.coverage_logit)
    native_mask = output.native_mask.detach().cpu()
    residual_mask = output.residual_mask.detach().cpu()
    selectable_mask = output.selectable_mask.detach().cpu()
    for label, tensor in (
        ("advantage mean", mean),
        ("advantage scale", scale),
        ("rank score", rank),
        ("harm probability", harm_probability),
        ("coverage probability", coverage_probability),
    ):
        if not torch.isfinite(tensor).all():
            raise NLSRTrainingError(
                f"decision model output contains non-finite {label}"
            )
    q_lcb = float(calibration["advantage"][
        "one_sided_normalized_quantile"])
    harm_threshold = float(calibration["harm"]["threshold"])
    coverage_threshold = float(
        calibration["coverage_miss_abstain"]["threshold"])
    coverage_policy = calibration.get("coverage_policy")
    if coverage_policy not in COVERAGE_POLICIES:
        raise NLSRTrainingError(
            "calibration coverage_policy is absent or unsupported"
        )
    if any(not math.isfinite(value) for value in (
            q_lcb, harm_threshold, coverage_threshold)):
        raise NLSRTrainingError("calibration contains non-finite thresholds")
    lcb = mean - q_lcb * scale
    decisions: list[dict[str, object]] = []
    for row_index, candidate_ids in enumerate(batch.candidate_ids):
        count = len(candidate_ids)
        native_indices = torch.nonzero(
            native_mask[row_index, :count], as_tuple=False).flatten().tolist()
        if len(native_indices) != 1:
            raise NLSRTrainingError("decision row has no unique native")
        native_index = native_indices[0]
        selectable = torch.nonzero(
            selectable_mask[row_index, :count],
            as_tuple=False,
        ).flatten().tolist()
        top_rank_index = max(
            selectable, key=lambda index: (float(rank[row_index, index]), -index))
        selected = native_index
        coverage_passed = bool(
            coverage_policy == COVERAGE_POLICY_ADVISORY_UNAVAILABLE
            or float(coverage_probability[row_index]) <= coverage_threshold
        )
        reason = "coverage_risk_abstain"
        if coverage_passed:
            eligible = []
            for index in torch.nonzero(
                    residual_mask[row_index, :count],
                    as_tuple=False).flatten().tolist():
                if (float(rank[row_index, index])
                        <= float(rank[row_index, native_index])):
                    continue
                if float(lcb[row_index, index]) < config.minimum_advantage_lcb_m:
                    continue
                if float(harm_probability[row_index, index]) > harm_threshold:
                    continue
                eligible.append(index)
            if eligible:
                selected = max(
                    eligible,
                    key=lambda index: (float(rank[row_index, index]), -index),
                )
                reason = "residual_passed_model_bounds"
            else:
                reason = "no_structurally_eligible_residual"
        decisions.append({
            "selected_index": selected,
            "selected_candidate_id": str(candidate_ids[selected]),
            "activated": selected != native_index,
            "reason": reason,
            "top_rank_candidate_id": str(candidate_ids[top_rank_index]),
            "coverage_miss_probability": float(
                coverage_probability[row_index]),
            "coverage_policy": str(coverage_policy),
            "selected_advantage_lcb_m": float(lcb[row_index, selected]),
            "selected_harm_probability": float(
                harm_probability[row_index, selected]),
        })
    return decisions


def evaluate_calibrated_audit(
    model: NLSRSetRanker,
    audit_records: Sequence[dict],
    feature_spec: RankerFeatureSpec,
    config: TrainingConfig,
    calibration: Mapping[str, object],
) -> dict[str, object]:
    """Audit frozen decisions; GT may annotate but can never rewrite them."""
    if not audit_records:
        raise NLSRTrainingError("development audit partition is empty")
    device = next(model.parameters()).device
    batch = vectorize_candidate_sets(
        audit_records, feature_spec=feature_spec).to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward_batch(batch)
    decisions = structural_policy_decisions(output, batch, calibration, config)
    cpu_batch = batch.to("cpu")
    mean = output.advantage_mean.detach().cpu()
    scale = output.advantage_log_scale.detach().cpu().exp()
    harm_probability = _sigmoid(output.harm_logit)
    coverage_probability = _sigmoid(output.coverage_logit)
    residual_valid = cpu_batch.residual_mask & cpu_batch.rollout_label_valid
    target = cpu_batch.advantage_target[residual_valid]
    prediction = mean[residual_valid]
    q_lcb = float(calibration["advantage"][
        "one_sided_normalized_quantile"])
    interval_coverage = (
        float((target >= (prediction - q_lcb * scale[residual_valid]))
              .float().mean())
        if target.numel() else 0.0)
    harm_targets = cpu_batch.harm_target[residual_valid]
    harm_predictions = harm_probability[residual_valid]
    coverage_valid = cpu_batch.coverage_label_valid
    coverage_targets = cpu_batch.coverage_miss_target[coverage_valid]
    coverage_predictions = coverage_probability[coverage_valid]

    rank_correct = 0
    evaluable_activations: list[dict[str, object]] = []
    decision_trace: list[dict[str, object]] = []
    for record, decision in zip(audit_records, decisions):
        oracle_id = str(record["set_labels"]["oracle_best_candidate_id"])
        expected_id = "native" if oracle_id == "dustbin" else oracle_id
        rank_correct += decision["top_rank_candidate_id"] == expected_id
        trace = {
            "state_id": str(record["provenance"]["state_id"]),
            "group_id": str(record["provenance"]["group_id"]),
            "scene_id": str(record["provenance"]["scene_id"]),
            **decision,
            "label_evaluable": True,
        }
        if bool(decision["activated"]):
            candidate = record["candidates"][int(decision["selected_index"])]
            label_valid = bool(candidate["labels"]["rollout_label_valid"])
            trace["label_evaluable"] = label_valid
            if label_valid:
                labelled = {
                    **trace,
                    "advantage_h24_m": float(
                        candidate["labels"]["advantage_h24_m"]),
                    "harm": bool(candidate["labels"]["harm"]),
                    "useful": bool(candidate["labels"]["useful"]),
                }
                evaluable_activations.append(labelled)
        decision_trace.append(trace)

    activated_scenes: dict[str, bool] = {}
    for row in evaluable_activations:
        scene_id = str(row["scene_id"])
        activated_scenes[scene_id] = (
            activated_scenes.get(scene_id, False) or bool(row["harm"]))
    harmed_scenes = sum(activated_scenes.values())
    activation_risk_upper = (
        zero_failure_upper_bound(len(activated_scenes), config.risk_alpha)
        if harmed_scenes == 0 else 1.0)
    activation_count = sum(bool(row["activated"]) for row in decision_trace)
    unevaluable_count = sum(
        bool(row["activated"]) and not bool(row["label_evaluable"])
        for row in decision_trace)
    metrics = {
        "audit_sets": len(audit_records),
        "audit_scenes": sorted({
            str(row["provenance"]["scene_id"]) for row in audit_records}),
        "rank_or_native_accuracy": float(rank_correct) / len(audit_records),
        "advantage": {
            "valid_rows": int(target.numel()),
            "mae_m": float((prediction - target).abs().mean())
            if target.numel() else 0.0,
            "rmse_m": float((prediction - target).square().mean().sqrt())
            if target.numel() else 0.0,
            "lcb_coverage": interval_coverage,
        },
        "harm": {
            "valid_rows": int(harm_targets.numel()),
            "positive_rows": int(harm_targets.sum()),
            "brier": float((harm_predictions - harm_targets).square().mean())
            if harm_targets.numel() else 0.0,
        },
        "coverage_miss": {
            "valid_sets": int(coverage_targets.numel()),
            "positive_sets": int(coverage_targets.sum()),
            "brier": float(
                (coverage_predictions - coverage_targets).square().mean())
            if coverage_targets.numel() else 0.0,
        },
        "calibrated_policy": {
            "activations": activation_count,
            "evaluable_activations": len(evaluable_activations),
            "unevaluable_activations": unevaluable_count,
            "activation_scenes": len(activated_scenes),
            "useful_activations": sum(
                bool(row["useful"]) for row in evaluable_activations),
            "harmful_activations": sum(
                bool(row["harm"]) for row in evaluable_activations),
            "mean_advantage_h24_m": (
                sum(float(row["advantage_h24_m"])
                    for row in evaluable_activations)
                / len(evaluable_activations)
                if evaluable_activations else 0.0),
            "harmed_scenes": harmed_scenes,
            "harm_one_sided_upper": activation_risk_upper,
            "independence_unit": "scene_id",
            "decision_trace": decision_trace,
        },
    }
    return metrics


def calibrate_development(
    model: NLSRSetRanker,
    calibration_records: Sequence[dict],
    audit_records: Sequence[dict],
    feature_spec: RankerFeatureSpec,
    config: TrainingConfig,
    *,
    scene_disjoint_split_available: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    calibration = fit_development_calibration(
        model, calibration_records, feature_spec, config)
    audit_scenes = sorted({
        str(row["provenance"]["scene_id"]) for row in audit_records})
    calibration["audit_scenes"] = audit_scenes
    calibration["scene_disjoint_audit"] = bool(
        scene_disjoint_split_available)
    if set(calibration["calibration_scenes"]) & set(audit_scenes):
        raise NLSRTrainingError(
            "development calibration and audit scenes overlap")
    calibration["shadow_eligible"] = bool(
        scene_disjoint_split_available
        and calibration["advantage"]["supported"]
        and calibration["harm"]["statistically_supported"]
        and (
            calibration["coverage_policy"]
            == COVERAGE_POLICY_ADVISORY_UNAVAILABLE
            or calibration["coverage_miss_abstain"]["statistically_supported"]
        ))
    calibration["deployment_approved"] = False
    metrics = evaluate_calibrated_audit(
        model, audit_records, feature_spec, config, calibration)
    return calibration, metrics


def _read_canonical_json_file(path: Path) -> object:
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise NLSRTrainingError(
            f"final JSON {Path(path).name} has noncanonical framing")
    payload = raw[:-1]
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NLSRTrainingError(
            f"cannot parse final JSON {Path(path).name}") from exc
    if canonical_json_bytes(value) != payload:
        raise NLSRTrainingError(
            f"final JSON {Path(path).name} is not canonical")
    return value


def verify_final_bundle(
    output_dir: Path,
    records: Sequence[dict],
    contract_sha256: str,
    config: TrainingConfig,
) -> dict[str, str]:
    """Read every final artifact back and verify all cross-file bindings."""
    output_dir = Path(output_dir)
    expected_hashed_files = frozenset({
        FINAL_CHECKPOINT_NAME,
        RESUME_NAME,
        METRICS_NAME,
        CALIBRATION_NAME,
        PROVENANCE_NAME,
    })
    manifest = _read_canonical_json_file(output_dir / MANIFEST_NAME)
    expected_manifest_keys = frozenset({
        "output_manifest_version", "training_contract_sha256",
        "dataset_content_sha256", "checkpoint_metadata_sha256",
        "checkpoint_state_dict_sha256", "files", "deployment_approved",
    })
    if (not isinstance(manifest, Mapping)
            or frozenset(manifest) != expected_manifest_keys):
        raise NLSRTrainingError("final manifest keys are not exact")
    if manifest["output_manifest_version"] != OUTPUT_MANIFEST_VERSION:
        raise NLSRTrainingError("final manifest version mismatch")
    if (manifest["training_contract_sha256"] != contract_sha256
            or manifest["dataset_content_sha256"]
            != dataset_content_sha256(records)):
        raise NLSRTrainingError("final manifest contract/data mismatch")
    if manifest["deployment_approved"] is not False:
        raise NLSRTrainingError("final manifest attempted deployment approval")
    files = manifest["files"]
    if (not isinstance(files, Mapping)
            or frozenset(files) != expected_hashed_files):
        raise NLSRTrainingError("final manifest file set is not exact")
    verified_hashes: dict[str, str] = {}
    for name in sorted(expected_hashed_files):
        descriptor = files[name]
        if (not isinstance(descriptor, Mapping)
                or frozenset(descriptor) != {"sha256"}
                or not isinstance(descriptor["sha256"], str)):
            raise NLSRTrainingError(
                f"final manifest descriptor for {name} is invalid")
        actual = sha256_file(output_dir / name)
        if descriptor["sha256"] != actual:
            raise NLSRTrainingError(f"final file hash mismatch for {name}")
        verified_hashes[name] = actual

    calibration = _read_canonical_json_file(output_dir / CALIBRATION_NAME)
    metrics = _read_canonical_json_file(output_dir / METRICS_NAME)
    provenance = _read_canonical_json_file(output_dir / PROVENANCE_NAME)
    for name, payload in (
            ("calibration", calibration),
            ("metrics", metrics),
            ("provenance", provenance)):
        if not isinstance(payload, Mapping):
            raise NLSRTrainingError(f"final {name} must be an object")
        if payload.get("deployment_approved") is not False:
            raise NLSRTrainingError(
                f"final {name} attempted deployment approval")
    if (calibration.get("training_contract_sha256") != contract_sha256
            or metrics.get("training_contract_sha256") != contract_sha256
            or provenance.get("training_contract_sha256") != contract_sha256):
        raise NLSRTrainingError("final JSON contract binding mismatch")
    contract = provenance.get("training_contract")
    if (not isinstance(contract, Mapping)
            or canonical_sha256(contract) != contract_sha256):
        raise NLSRTrainingError("final provenance contract hash mismatch")

    try:
        portable = torch.load(
            output_dir / FINAL_CHECKPOINT_NAME,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise NLSRTrainingError(
            "cannot weights-only load final checkpoint") from exc
    restored = load_portable_checkpoint(
        portable, expected_records=records)
    metadata = portable["metadata"]
    if (metadata["metadata_sha256"]
            != manifest["checkpoint_metadata_sha256"]
            or state_dict_sha256(restored.state_dict())
            != manifest["checkpoint_state_dict_sha256"]):
        raise NLSRTrainingError("final checkpoint metadata/state mismatch")
    extra = metadata["extra"]
    if (not isinstance(extra, Mapping)
            or extra.get("training_contract_sha256") != contract_sha256
            or extra.get("calibration_sha256")
            != verified_hashes[CALIBRATION_NAME]
            or extra.get("metrics_sha256") != verified_hashes[METRICS_NAME]
            or extra.get("provenance_sha256")
            != verified_hashes[PROVENANCE_NAME]
            or extra.get("deployment_approved") is not False):
        raise NLSRTrainingError("final checkpoint extra bindings mismatch")

    try:
        resume_state = torch.load(
            output_dir / RESUME_NAME,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise NLSRTrainingError(
            "cannot weights-only load final resume state") from exc
    resume_state = _validate_resume_state(
        resume_state, contract_sha256, config)
    if (resume_state["phase"] != "complete"
            or resume_state["model_state_sha256"]
            != manifest["checkpoint_state_dict_sha256"]):
        raise NLSRTrainingError("final resume/checkpoint state mismatch")
    return verified_hashes


@dataclass(frozen=True)
class RunResult:
    status: str
    report: dict[str, object]


def run_training(
    artifact_path: Path,
    output_dir: Path,
    config: TrainingConfig,
    *,
    resume: bool = False,
    epoch_budget: int = 0,
    preflight_only: bool = False,
) -> RunResult:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise NLSRTrainingError("CUDA was requested but is unavailable")
    records, artifact_audit = load_canonical_candidate_rows(artifact_path)
    declared = split_by_declared_role(records)
    train_records = list(declared.train)
    development_records = list(declared.development)
    core, tune, core_scenes, tune_scenes = _partition_train_records(
        train_records, config.tune_scene_fraction, config.seed)
    development_scenes = tuple(declared.development_scenes)
    (
        development_calibration_records,
        development_audit_records,
        development_calibration_scenes,
        development_audit_scenes,
        development_split_available,
    ) = _partition_development_records(
        development_records,
        config.development_calibration_scene_fraction,
        config.seed,
    )
    feature_spec = feature_spec_from_dataset(records)
    selection_loss_config, selection_class_balance = class_weights_from_train(
        core)
    selection_class_balance = {
        **selection_class_balance,
        "source_partition": "train_core",
        "source_scenes": list(core_scenes),
    }
    refit_loss_config, refit_class_balance = class_weights_from_train(
        train_records)
    refit_class_balance = {
        **refit_class_balance,
        "source_partition": "train_all",
        "source_scenes": sorted({
            str(row["provenance"]["scene_id"]) for row in train_records}),
    }
    contract, contract_sha256 = _training_contract(
        records, artifact_audit, feature_spec, config,
        selection_loss_config, refit_loss_config,
        core_scenes, tune_scenes, development_scenes,
        development_calibration_scenes, development_audit_scenes,
        selection_class_balance, refit_class_balance)
    preflight = {
        "status": "preflight_passed",
        "trainer_format_version": TRAINER_FORMAT_VERSION,
        "artifact": artifact_audit,
        "records": {
            "train": len(train_records),
            "core": len(core),
            "tune": len(tune),
            "development": len(development_records),
            "development_calibration": len(
                development_calibration_records),
            "development_audit": len(development_audit_records),
        },
        "scenes": contract["partitions"],
        "development_scene_disjoint_split_available": (
            development_split_available),
        "class_balance": {
            "selection_core": selection_class_balance,
            "refit_all_train": refit_class_balance,
        },
        "feature_spec": feature_spec.to_dict(),
        "parameter_count": _new_model(
            feature_spec, config.seed, torch.device("cpu")).parameter_count,
        "training_contract_sha256": contract_sha256,
    }
    if preflight_only:
        return RunResult(status="preflight_passed", report=preflight)

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NLSRTrainingError("output path exists and is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [
        output_dir / name for name in (
            FINAL_CHECKPOINT_NAME, METRICS_NAME, CALIBRATION_NAME,
            PROVENANCE_NAME, MANIFEST_NAME)
    ]
    if not resume and any(path.exists() for path in final_paths):
        raise NLSRTrainingError(
            "final outputs already exist; use a new directory or --resume")
    fit = fit_with_resume(
        core_records=core,
        tune_records=tune,
        train_records=train_records,
        feature_spec=feature_spec,
        selection_loss_config=selection_loss_config,
        refit_loss_config=refit_loss_config,
        config=config,
        contract_sha256=contract_sha256,
        resume_path=output_dir / RESUME_NAME,
        resume=resume,
        epoch_budget=epoch_budget,
    )
    if not fit.complete:
        report = dict(preflight)
        report.update({
            "status": "training_incomplete",
            "phase": fit.state["phase"],
            "next_epoch": fit.state["next_epoch"],
            "selected_epochs": fit.selected_epochs,
            "epochs_run_this_call": fit.epochs_run_this_call,
            "resume_path": str(output_dir / RESUME_NAME),
            "resume_sha256": sha256_file(output_dir / RESUME_NAME),
        })
        return RunResult(status="training_incomplete", report=report)

    if fit.model is None or fit.selected_epochs is None:
        raise NLSRTrainingError("complete fit did not return its final model")
    development_batch = vectorize_candidate_sets(
        development_records, feature_spec=feature_spec)
    development_loss = evaluate_loss(
        fit.model, development_batch, refit_loss_config,
        config.batch_size, torch.device(config.device))
    calibration, development_metrics = calibrate_development(
        fit.model,
        development_calibration_records,
        development_audit_records,
        feature_spec,
        config,
        scene_disjoint_split_available=development_split_available,
    )
    calibration["training_contract_sha256"] = contract_sha256
    metrics = {
        "metrics_format_version": "nlsr_set_ranker_metrics_v2",
        "training_contract_sha256": contract_sha256,
        "deployment_approved": False,
        "selected_epochs": fit.selected_epochs,
        "selection_best_epoch": int(fit.state["best_epoch"]) + 1,
        "selection_best_total": float(fit.state["best_metric"]),
        "selection_history": fit.state["selection_history"],
        "refit_history": fit.state["refit_history"],
        "development_loss": development_loss,
        "development": development_metrics,
        "class_balance_train_only": {
            "selection_core": selection_class_balance,
            "refit_all_train": refit_class_balance,
        },
    }
    provenance = {
        "provenance_format_version": "nlsr_set_ranker_provenance_v2",
        "training_contract": contract,
        "training_contract_sha256": contract_sha256,
        "artifact": artifact_audit,
        "feature_spec": feature_spec.to_dict(),
        "partitions": contract["partitions"],
        "development_used_only_after_epoch_freeze": True,
        "class_weights_derived_only_from_train": True,
        "selection_class_weights_derived_only_from_core_train": True,
        "refit_class_weights_derived_only_from_all_train": True,
        "deployment_approved": False,
    }
    calibration_path = output_dir / CALIBRATION_NAME
    metrics_path = output_dir / METRICS_NAME
    provenance_path = output_dir / PROVENANCE_NAME
    atomic_write_json(calibration_path, calibration)
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(provenance_path, provenance)
    json_hashes = {
        CALIBRATION_NAME: sha256_file(calibration_path),
        METRICS_NAME: sha256_file(metrics_path),
        PROVENANCE_NAME: sha256_file(provenance_path),
    }
    portable = make_portable_checkpoint(
        fit.model, records,
        extra={
            "trainer_format_version": TRAINER_FORMAT_VERSION,
            "training_contract_sha256": contract_sha256,
            "selected_epochs": fit.selected_epochs,
            "calibration_sha256": json_hashes[CALIBRATION_NAME],
            "metrics_sha256": json_hashes[METRICS_NAME],
            "provenance_sha256": json_hashes[PROVENANCE_NAME],
            "deployment_approved": False,
        },
    )
    checkpoint_path = output_dir / FINAL_CHECKPOINT_NAME
    atomic_torch_save(checkpoint_path, portable)
    file_hashes = dict(json_hashes)
    file_hashes[FINAL_CHECKPOINT_NAME] = sha256_file(checkpoint_path)
    file_hashes[RESUME_NAME] = sha256_file(output_dir / RESUME_NAME)
    manifest = {
        "output_manifest_version": OUTPUT_MANIFEST_VERSION,
        "training_contract_sha256": contract_sha256,
        "dataset_content_sha256": dataset_content_sha256(records),
        "checkpoint_metadata_sha256": portable["metadata"][
            "metadata_sha256"],
        "checkpoint_state_dict_sha256": portable["metadata"][
            "state_dict_sha256"],
        "files": {
            name: {"sha256": digest}
            for name, digest in sorted(file_hashes.items())
        },
        "deployment_approved": False,
    }
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    verified_hashes = verify_final_bundle(
        output_dir, records, contract_sha256, config)
    report = {
        "status": "training_complete",
        "output_dir": str(output_dir),
        "training_contract_sha256": contract_sha256,
        "selected_epochs": fit.selected_epochs,
        "checkpoint_sha256": verified_hashes[FINAL_CHECKPOINT_NAME],
        "metrics_sha256": verified_hashes[METRICS_NAME],
        "manifest_sha256": sha256_file(output_dir / MANIFEST_NAME),
        "checkpoint_state_dict_sha256": portable["metadata"][
            "state_dict_sha256"],
        "shadow_eligible": calibration["shadow_eligible"],
        "bundle_readback_verified": True,
        "deployment_approved": False,
    }
    return RunResult(status="training_complete", report=report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--tune-scene-fraction", type=float, default=0.2)
    parser.add_argument(
        "--development-calibration-scene-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument("--advantage-alpha", type=float, default=0.05)
    parser.add_argument("--risk-alpha", type=float, default=0.05)
    parser.add_argument("--target-harm-upper", type=float, default=0.05)
    parser.add_argument(
        "--target-coverage-miss-upper", type=float, default=0.05)
    parser.add_argument("--minimum-advantage-lcb-m", type=float, default=0.25)
    parser.add_argument(
        "--minimum-advantage-calibration-scenes", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--epoch-budget", type=int, default=0,
        help="Stop cleanly after this many epochs in this invocation; 0=all")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        tune_scene_fraction=args.tune_scene_fraction,
        development_calibration_scene_fraction=(
            args.development_calibration_scene_fraction),
        minimum_improvement=args.minimum_improvement,
        advantage_alpha=args.advantage_alpha,
        risk_alpha=args.risk_alpha,
        target_harm_upper=args.target_harm_upper,
        target_coverage_miss_upper=args.target_coverage_miss_upper,
        minimum_advantage_lcb_m=args.minimum_advantage_lcb_m,
        minimum_advantage_calibration_scenes=(
            args.minimum_advantage_calibration_scenes),
        device=args.device,
    )
    result = run_training(
        args.artifact,
        args.out_dir,
        config,
        resume=args.resume,
        epoch_budget=args.epoch_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result.report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
