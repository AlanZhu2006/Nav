"""Fail-closed runtime for the factorized CDEC anchor ranker.

The learned component in this module has deliberately narrow authority: it
orders an already frozen causal DINO shortlist.  Its scores are pairwise
utilities, not an open-set match probability and not an activation decision.
Only the independent PnP certificate may turn a proposed anchor into a bearing
that reaches the navigation controller.

The optional circular summary is useful for interpreting the learned episodic
direction posterior.  It is diagnostic until a selected hypothesis carries a
valid geometry certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from MemNavData.patch_temporal_router import (
        directional_patch_feature_names,
        directional_patch_relation_features,
    )
except ModuleNotFoundError:  # direct script invocation
    from patch_temporal_router import (  # type: ignore
        directional_patch_feature_names,
        directional_patch_relation_features,
    )


CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION = (
    "cdec_factorized_pairwise_runtime_v3_fixed_batch_float32_20260813")
CDEC_PATCH_GRID_SIZE = 8
CDEC_RAW_PATCH_GRID_SIZE = 37
CDEC_RAW_PATCH_COUNT = CDEC_RAW_PATCH_GRID_SIZE ** 2
CDEC_DINO_INFERENCE_BATCH_SIZE = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    shifted = scores / temperature
    shifted -= float(np.max(shifted))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def pool_dino_patch_tokens(patch_tokens: Any,
                           *, grid_size: int = CDEC_PATCH_GRID_SIZE) -> np.ndarray:
    """Exactly reproduce the label-blind 37x37 -> 8x8 cache transform.

    LingBot's context-free DINO trunk emits ``[N,1369,1024]`` float patch
    tokens.  The training cache used PyTorch adaptive average pooling,
    L2-normalization, then fp16 storage.  Runtime intentionally repeats that
    quantization before computing relation features, preventing an unnoticed
    train/deploy feature mismatch.
    """

    import torch
    import torch.nn.functional as functional

    tensor = (patch_tokens.detach() if isinstance(patch_tokens, torch.Tensor)
              else torch.as_tensor(patch_tokens))
    squeeze = tensor.ndim == 2
    if squeeze:
        tensor = tensor.unsqueeze(0)
    if (tensor.ndim != 3 or tensor.shape[1] != CDEC_RAW_PATCH_COUNT
            or tensor.shape[2] < 1):
        raise ValueError(
            "DINO patch tokens must have shape [batch,1369,channels]")
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("pooled grid size must be at least two")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("DINO patch tokens must be finite")
    batch, _patches, channels = tensor.shape
    pooled = tensor.float().reshape(
        batch, CDEC_RAW_PATCH_GRID_SIZE, CDEC_RAW_PATCH_GRID_SIZE,
        channels).permute(0, 3, 1, 2)
    pooled = functional.adaptive_avg_pool2d(pooled, (grid_size, grid_size))
    pooled = functional.normalize(pooled.flatten(2).transpose(1, 2), dim=-1)
    # This cast is part of the frozen training feature contract.
    result = pooled.to(dtype=torch.float16).cpu().numpy()
    expected = (batch, grid_size * grid_size, channels)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(
            f"invalid pooled patch tensor {result.shape}, expected {expected}")
    return result[0] if squeeze else result


def pad_dino_image_batch(images: Any, *,
                         batch_size: int = CDEC_DINO_INFERENCE_BATCH_SIZE):
    """Pad a shortlist to the frozen training GEMM shape with its last image.

    BF16 transformer kernels are mathematically batch-independent but not
    bitwise invariant to batch shape.  The label-blind cache was extracted at
    batch 16, so deployment must use that same shape.  Duplicate padding is
    removed immediately after the context-free DINO forward and cannot alter
    another image's attention because DINO has no cross-image operation.
    """

    import torch

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a torch tensor [batch,channels,H,W]")
    count = len(images)
    batch_size = int(batch_size)
    if count < 1 or batch_size < 1 or count > batch_size:
        raise ValueError("real DINO batch must lie in [1, frozen batch size]")
    if count == batch_size:
        return images, count
    padding = images[-1:].expand(
        batch_size - count, *images.shape[1:])
    return torch.cat((images, padding), dim=0), count


def relation_feature_matrix(query_tokens: np.ndarray,
                            memory_tokens: np.ndarray,
                            global_cosines: Sequence[float]) -> np.ndarray:
    """Build the exact directional relation feature matrix for one shortlist."""

    query = np.asarray(query_tokens)
    memory = np.asarray(memory_tokens)
    cosine = np.asarray(global_cosines, dtype=np.float64)
    if query.ndim != 2:
        raise ValueError("query tokens must have shape [patches,channels]")
    if (memory.ndim != 3 or memory.shape[1:] != query.shape
            or cosine.shape != (len(memory),)):
        raise ValueError("memory tokens and cosine shortlist are misaligned")
    if not len(memory):
        raise ValueError("shortlist must contain at least one candidate")
    # The training cache stores relation features as float32.  Preserve that
    # quantization boundary before the ranker promotes them to float64 for its
    # standardized dot product; otherwise exact pooled tokens still produce a
    # small train/deploy score drift.
    features = np.asarray([
        directional_patch_relation_features(query, candidate, score)
        for candidate, score in zip(memory, cosine)
    ], dtype=np.float32)
    expected = (len(memory), len(directional_patch_feature_names()))
    if features.shape != expected or not np.isfinite(features).all():
        raise RuntimeError(
            f"invalid CDEC features {features.shape}, expected {expected}")
    return features


def circular_direction_summary(probability: Sequence[float],
                               bearing_vectors: Sequence[Sequence[float]]) -> dict:
    """Map shortlist mass to a scale-free circular bearing diagnostic."""

    mass = np.asarray(probability, dtype=np.float64)
    vectors = np.asarray(bearing_vectors, dtype=np.float64)
    if (mass.ndim != 1 or not len(mass) or vectors.shape != (len(mass), 2)
            or not np.isfinite(mass).all() or not np.isfinite(vectors).all()
            or np.any(mass < 0.0)):
        raise ValueError("probability and bearing vectors must be finite/aligned")
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("probability mass must be positive")
    mass = mass / total
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("every candidate bearing must be non-zero")
    unit = vectors / norms[:, None]
    mean = np.sum(mass[:, None] * unit, axis=0)
    resultant = float(np.linalg.norm(mean))
    positive = mass > 0.0
    entropy = float(-np.sum(mass[positive] * np.log(mass[positive])))
    normalized_entropy = (
        entropy / math.log(len(mass)) if len(mass) > 1 else 0.0)
    direction = (mean / resultant).tolist() if resultant > 1e-12 else None
    angle = (float(math.atan2(mean[1], mean[0]))
             if resultant > 1e-12 else None)
    return {
        "mean_unit_bearing_forward_left": direction,
        "mean_angle_rad_left_positive": angle,
        "resultant_length": resultant,
        "entropy_nats": entropy,
        "normalized_entropy": normalized_entropy,
        "execution_authorized": False,
        "reason": "diagnostic_until_geometry_certificate",
    }


class CDECPairwiseRanker:
    """Strict loader and scorer for one immutable all-train fit."""

    def __init__(self, artifact: str | Path, *, allow_unapproved: bool = False):
        source_path = Path(artifact)
        if source_path.is_symlink():
            raise ValueError("CDEC artifact may not be a symbolic link")
        path = source_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CDEC artifact is not a physical file: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid CDEC JSON artifact: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("CDEC artifact root must be an object")
        if payload.get("schema_version") != CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported CDEC runtime artifact schema")
        approved = payload.get("deployment_approved")
        if not isinstance(approved, bool):
            raise ValueError("deployment_approved must be boolean")
        if not approved and not allow_unapproved:
            raise PermissionError(
                "CDEC artifact is not deployment-approved; pass the explicit "
                "research-only override to load it")
        semantics = payload.get("runtime_semantics")
        if not isinstance(semantics, dict):
            raise ValueError("runtime_semantics is missing")
        expected_semantics = {
            "authority": "rank_frozen_causal_shortlist_only",
            "activation_authority": "independent_atomic_pnp_certificate",
            "fallback": "native_imagegoal",
            "score_calibration": "uncalibrated_pairwise_utility",
        }
        for key, value in expected_semantics.items():
            if semantics.get(key) != value:
                raise ValueError(f"unsafe CDEC runtime semantic for {key}")
        model = payload.get("model")
        if not isinstance(model, dict):
            raise ValueError("CDEC artifact model is missing")
        feature_names = tuple(map(str, model.get("feature_names", ())))
        expected_names = directional_patch_feature_names()
        if feature_names != expected_names:
            raise ValueError("CDEC feature names/order changed")
        length = len(expected_names)
        self.coefficient = _finite_vector(
            model.get("coefficient"), name="coefficient", length=length)
        self.mean = _finite_vector(model.get("mean"), name="mean", length=length)
        self.scale = _finite_vector(model.get("scale"), name="scale", length=length)
        if np.any(self.scale <= 0.0):
            raise ValueError("CDEC feature scale must be positive")
        intercept = model.get("intercept")
        try:
            valid_intercept = (
                not isinstance(intercept, bool)
                and math.isfinite(float(intercept)))
        except (TypeError, ValueError, OverflowError):
            valid_intercept = False
        if not valid_intercept:
            raise ValueError("CDEC intercept must be finite")
        if float(intercept) != 0.0:
            raise ValueError("pairwise runtime requires the frozen zero intercept")
        grid_size = model.get("patch_grid_size")
        try:
            valid_grid = (
                not isinstance(grid_size, bool)
                and float(grid_size).is_integer()
                and int(grid_size) == CDEC_PATCH_GRID_SIZE)
        except (TypeError, ValueError, OverflowError):
            valid_grid = False
        if not valid_grid:
            raise ValueError("CDEC patch grid contract changed")
        dino_batch_size = model.get("dino_inference_batch_size")
        try:
            valid_dino_batch = (
                not isinstance(dino_batch_size, bool)
                and float(dino_batch_size).is_integer()
                and int(dino_batch_size) == CDEC_DINO_INFERENCE_BATCH_SIZE)
        except (TypeError, ValueError, OverflowError):
            valid_dino_batch = False
        if not valid_dino_batch:
            raise ValueError("CDEC DINO inference batch contract changed")
        if model.get("relation_storage_dtype") != "float32":
            raise ValueError("CDEC relation quantization contract changed")
        self.path = path
        self.artifact_sha256 = sha256(path)
        self.payload = payload
        self.deployment_approved = approved
        self.feature_names = feature_names

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "artifact": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "schema_version": CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION,
            "deployment_approved": self.deployment_approved,
            "authority": "rank_frozen_causal_shortlist_only",
            "activation_authority": "independent_atomic_pnp_certificate",
            "posterior_semantics": "uncalibrated_within_shortlist_softmax",
        }

    def score_features(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if (matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names)
                or not len(matrix) or not np.isfinite(matrix).all()):
            raise ValueError("CDEC features must be a finite non-empty matrix")
        scores = ((matrix - self.mean) / self.scale) @ self.coefficient
        if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
            raise RuntimeError("CDEC ranker emitted invalid scores")
        return scores

    def rank_features(self, features: np.ndarray, anchors: Sequence[int],
                      *, temperature: float = 1.0,
                      bearing_vectors: Sequence[Sequence[float]] | None = None,
                      include_features: bool = False) -> dict[str, Any]:
        anchor = np.asarray(anchors)
        if (anchor.ndim != 1 or not len(anchor)
                or any(isinstance(value, (bool, np.bool_)) for value in anchor)
                or not np.issubdtype(anchor.dtype, np.number)
                or not np.isfinite(anchor.astype(np.float64)).all()
                or not np.equal(
                    anchor.astype(np.float64),
                    np.floor(anchor.astype(np.float64))).all()
                or len({int(value) for value in anchor}) != len(anchor)):
            raise ValueError("anchors must be a non-empty unique integer vector")
        try:
            anchor = anchor.astype(np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("anchors must be integers") from error
        matrix = np.asarray(features, dtype=np.float64)
        if len(matrix) != len(anchor):
            raise ValueError("features and anchors are misaligned")
        scores = self.score_features(matrix)
        probability = _stable_softmax(scores, temperature)
        order = np.argsort(-scores, kind="stable")
        selected = int(order[0])
        ordered_scores = scores[order]
        margin = (float(ordered_scores[0] - ordered_scores[1])
                  if len(order) > 1 else None)
        result: dict[str, Any] = {
            "selected_anchor": int(anchor[selected]),
            "selected_index": selected,
            "ranked_anchors": [int(value) for value in anchor[order]],
            "scores": scores.tolist(),
            "within_shortlist_mass": probability.tolist(),
            "top1_margin": margin,
            "score_calibration": "uncalibrated_pairwise_utility",
            "activation_authorized": False,
            "activation_authority": "independent_atomic_pnp_certificate",
        }
        if bearing_vectors is not None:
            result["direction_posterior"] = circular_direction_summary(
                probability, bearing_vectors)
        if include_features:
            result["features"] = matrix.tolist()
            result["feature_names"] = list(self.feature_names)
        return result

    def rank_pooled_tokens(
            self, query_tokens: np.ndarray, memory_tokens: np.ndarray,
            global_cosines: Sequence[float], anchors: Sequence[int],
            *, temperature: float = 1.0,
            bearing_vectors: Sequence[Sequence[float]] | None = None,
            include_features: bool = False) -> dict[str, Any]:
        features = relation_feature_matrix(
            query_tokens, memory_tokens, global_cosines)
        return self.rank_features(
            features, anchors, temperature=temperature,
            bearing_vectors=bearing_vectors,
            include_features=include_features)


__all__ = [
    "CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION",
    "CDEC_DINO_INFERENCE_BATCH_SIZE",
    "CDEC_PATCH_GRID_SIZE",
    "CDEC_RAW_PATCH_COUNT",
    "CDECPairwiseRanker",
    "circular_direction_summary",
    "pad_dino_image_batch",
    "pool_dino_patch_tokens",
    "relation_feature_matrix",
    "sha256",
]
