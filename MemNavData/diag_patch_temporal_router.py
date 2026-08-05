#!/usr/bin/env python3
"""Train and audit a selective DINO patch/temporal reliability router.

This experiment starts from the exact pair table produced by
``diag_distill_geometry_router.py``.  It chooses hard candidates by frozen
DINO cosine and keeps the requested held-out scenes completely outside model
selection and threshold calibration.  The preferred label source is the
task-aligned goal-surface co-visibility produced by
``relabel_router_covisibility.py``; legacy teacher CSVs remain readable for
controlled comparisons.

The exported head is deliberately marked not-for-deployment.  A learned score
may replace geometry only after its confidence tails survive scene-disjoint
evaluation and a closed-loop Novel/Revisit test.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from MemNavData.patch_temporal_router import (
        combine_patch_temporal,
        combined_feature_names,
        directional_combined_feature_names,
        directional_patch_feature_names,
        directional_patch_relation_features,
        patch_feature_names,
        symmetric_from_directional_patch_features,
        symmetric_patch_relation_features,
        temporal_feature_names,
        temporal_score_features,
    )
    from MemNavData.reliability_router import (
        calibrate_zero_error_thresholds,
        selective_decisions,
    )
except ModuleNotFoundError:  # direct script invocation
    from patch_temporal_router import (  # type: ignore
        combine_patch_temporal,
        combined_feature_names,
        directional_combined_feature_names,
        directional_patch_feature_names,
        directional_patch_relation_features,
        patch_feature_names,
        symmetric_from_directional_patch_features,
        symmetric_patch_relation_features,
        temporal_feature_names,
        temporal_score_features,
    )
    from reliability_router import (  # type: ignore
        calibrate_zero_error_thresholds,
        selective_decisions,
    )


DEFAULT_WEIGHT_SHA = (
    "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409")
DEFAULT_LINGBOT_COMMIT = "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
FEATURE_VERSION = "exact_lingbot_dinov2l_patch_temporal_v3"
REQUIRED_COLUMNS = {
    "session_id", "scene", "episode", "kind", "query_path",
    "candidate_path", "candidate_frame", "teacher_pass", "dino_cosine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    resolved = root.resolve()
    try:
        return subprocess.check_output(
            [
                "git", "-c", f"safe.directory={resolved}",
                "-C", str(resolved), *args,
            ], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def parse_path_maps(specifications: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    mappings = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"path map must be OLD=NEW, got {specification!r}")
        old, new = specification.split("=", 1)
        old = old.rstrip("/")
        new = new.rstrip("/")
        if not old or not new:
            raise ValueError(f"path map must be non-empty, got {specification!r}")
        mappings.append((old, new))
    return tuple(sorted(mappings, key=lambda item: len(item[0]), reverse=True))


def remap_path(path: str, mappings: Sequence[Tuple[str, str]]) -> Path:
    for old, new in mappings:
        if path == old or path.startswith(old + "/"):
            return Path(new + path[len(old):])
    return Path(path)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or not labels.size:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    predicted = probabilities >= 0.5
    both_classes = len(np.unique(labels)) == 2
    return {
        "examples": int(labels.size),
        "positive": int(labels.sum()),
        "roc_auc": (float(roc_auc_score(labels, probabilities))
                    if both_classes else None),
        "average_precision": (float(average_precision_score(
            labels, probabilities)) if np.any(labels == 1) else None),
        "brier": float(brier_score_loss(labels, probabilities)),
        "accuracy_at_0.5": float(accuracy_score(labels, predicted)),
        "balanced_accuracy_at_0.5": (
            float(balanced_accuracy_score(labels, predicted))
            if both_classes else None),
    }


def cascade_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    decisions = np.asarray(decisions, dtype=np.int8).reshape(-1)
    if labels.shape != decisions.shape:
        raise ValueError("labels and decisions must be aligned")
    accepted = decisions == 1
    rejected = decisions == -1
    deferred = decisions == 0
    return {
        "examples": int(labels.size),
        "automatic_accept": int(accepted.sum()),
        "automatic_reject": int(rejected.sum()),
        "automatic_coverage": float(np.mean(~deferred)),
        "geometry_fallback": int(deferred.sum()),
        "geometry_fallback_rate": float(np.mean(deferred)),
        "false_accept": int(np.sum(accepted & (labels == 0))),
        "false_reject": int(np.sum(rejected & (labels == 1))),
    }


def grouped_top1_metrics(groups: np.ndarray, labels: np.ndarray,
                         probabilities: np.ndarray,
                         decisions: np.ndarray) -> dict:
    groups = np.asarray(groups, dtype=str).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    decisions = np.asarray(decisions, dtype=np.int8).reshape(-1)
    if not (groups.shape == labels.shape == probabilities.shape
            == decisions.shape):
        raise ValueError("grouped top1 inputs must be aligned")
    result = {}
    for group in sorted(np.unique(groups)):
        selected = groups == group
        result[group] = {
            "classification": binary_metrics(
                labels[selected], probabilities[selected]),
            "selective_cascade": cascade_metrics(
                labels[selected], decisions[selected]),
        }
    return result


def threshold_dict(thresholds) -> dict:
    return {
        "reject_max": (float(thresholds.reject_max)
                       if np.isfinite(thresholds.reject_max) else None),
        "accept_min": (float(thresholds.accept_min)
                       if np.isfinite(thresholds.accept_min) else None),
        "reject_calibration_count": thresholds.reject_calibration_count,
        "accept_calibration_count": thresholds.accept_calibration_count,
        "min_samples": thresholds.min_samples,
    }


def load_cls_cache(path: Path) -> Tuple[Dict[str, np.ndarray], str]:
    cache = np.load(path, allow_pickle=False)
    paths = cache["paths"].astype(str)
    embeddings = cache["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(paths):
        raise ValueError("CLS cache paths and embeddings are not aligned")
    if len(set(paths.tolist())) != len(paths):
        raise ValueError("CLS cache contains duplicate paths")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(embeddings).all():
        raise ValueError("CLS cache contains invalid embeddings")
    embeddings = embeddings / norms
    weight_sha = str(cache["weight_sha"].item())
    return dict(zip(paths.tolist(), embeddings)), weight_sha


def select_hard_candidates(frame, top_k: int):
    ordered = frame.sort_values(
        ["session_id", "dino_cosine", "candidate_frame"],
        ascending=[True, False, True], kind="mergesort")
    selected = ordered.groupby("session_id", sort=False).head(top_k).copy()
    selected["candidate_rank"] = (
        selected.groupby("session_id", sort=False).cumcount() + 1)
    selected.reset_index(drop=True, inplace=True)
    return selected


def selection_digest(selected_frame, full_frame, top_k: int, grid_size: int,
                     weight_sha: str, patch_relation: str) -> str:
    digest = hashlib.sha256()
    # Feature extraction depends on selected image pairs, not their labels.
    # Keeping teacher SHA out allows a corrected offline teacher to reuse the
    # exact same expensive frozen-DINO features.
    digest.update(
        (f"features_v3|top_k={top_k}|grid={grid_size}|weight={weight_sha}"
         f"|relation={patch_relation}").encode())
    for row in selected_frame.itertuples():
        digest.update(
            (f"\nselected\t{row.session_id}\t{row.query_path}"
             f"\t{row.candidate_path}"
             f"\t{row.candidate_frame}\t{row.dino_cosine:.12g}").encode())
    # Temporal features use every score/frame in each selected session, not
    # only its top-K candidates. Include that complete curve while excluding
    # labels, so corrected teachers can share features but changed retrieval
    # inputs cannot silently hit a stale cache.
    sessions = set(selected_frame["session_id"].astype(str))
    temporal_rows = full_frame[
        full_frame["session_id"].astype(str).isin(sessions)].sort_values(
            ["session_id", "candidate_frame", "candidate_path"],
            kind="mergesort")
    for row in temporal_rows.itertuples():
        digest.update(
            (f"\ntemporal\t{row.session_id}\t{row.candidate_path}"
             f"\t{row.candidate_frame}\t{row.dino_cosine:.12g}").encode())
    return digest.hexdigest()


def load_exact_patch_tokens(paths: Sequence[Path], lingbot_repo: Path,
                            weights: Path, device: str, batch_size: int,
                            grid_size: int):
    import torch

    sys.path.insert(0, str(lingbot_repo))
    from lingbot_map.layers.vision_transformer import vit_large
    import lingbot_map.utils.load_fn as load_fn

    # The upstream helper emits one tqdm bar per tiny batch.  Suppressing it is
    # important for auditable Slurm logs; it does not change preprocessing.
    load_fn.tqdm = lambda iterable, **_kwargs: iterable

    raw_state = torch.load(weights, map_location="cpu", weights_only=False)
    if (isinstance(raw_state, dict) and "model" in raw_state
            and isinstance(raw_state["model"], dict)):
        state = raw_state["model"]
    else:
        state = raw_state
    prefix = "aggregator.patch_embed."
    patch_state = {
        key[len(prefix):]: value for key, value in state.items()
        if key.startswith(prefix)}
    if len(patch_state) != 344:
        raise RuntimeError(
            f"expected 344 DINO tensors, found {len(patch_state)}")
    del raw_state, state
    gc.collect()

    model = vit_large(
        img_size=518,
        patch_size=14,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    )
    model.load_state_dict(patch_state, strict=True)
    del patch_state
    gc.collect()
    model = model.to(device).eval()
    torch_device = torch.device(device)
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=torch_device).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=torch_device).view(1, 3, 1, 1)
    use_cuda = torch_device.type == "cuda"

    outputs = []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        images = load_fn.load_and_preprocess_images(
            [str(path) for path in paths[start:start + batch_size]],
            mode="pad", image_size=518, patch_size=14)
        images = images.to(torch_device, non_blocking=use_cuda)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_cuda else torch.autocast("cpu", enabled=False))
        with torch.inference_mode(), autocast:
            encoded = model.forward_features((images - mean) / std)
            patches = encoded["x_norm_patchtokens"].float()
            if patches.shape[1:] != (1369, 1024):
                raise RuntimeError(
                    f"unexpected DINO patch shape {tuple(patches.shape)}")
            patches = patches.reshape(-1, 37, 37, 1024).permute(0, 3, 1, 2)
            patches = torch.nn.functional.adaptive_avg_pool2d(
                patches, (grid_size, grid_size))
            patches = patches.flatten(2).transpose(1, 2)
            patches = torch.nn.functional.normalize(patches, dim=-1)
        outputs.append(patches.to(dtype=torch.float16).cpu().numpy())
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(f"[patch] {min(start + batch_size, len(paths))}/{len(paths)}",
                  flush=True)
        del images, encoded, patches
    elapsed = time.perf_counter() - started
    output = np.concatenate(outputs, axis=0)
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    expected = (len(paths), grid_size * grid_size, 1024)
    if output.shape != expected or not np.isfinite(output).all():
        raise RuntimeError(
            f"invalid pooled patch cache {output.shape}, expected {expected}")
    return output, elapsed


def build_temporal_features(full_frame, selected_frame) -> np.ndarray:
    sessions = {}
    for session_id, group in full_frame.groupby("session_id", sort=False):
        sessions[session_id] = (
            group["candidate_frame"].to_numpy(dtype=np.int64),
            group["dino_cosine"].to_numpy(dtype=np.float64),
        )
    features = []
    for row in selected_frame.itertuples():
        frames, scores = sessions[row.session_id]
        features.append(temporal_score_features(
            row.candidate_frame, frames, scores))
    return np.asarray(features, dtype=np.float32)


def build_or_load_features(selected_frame, full_frame, cls_by_path,
                           readable_path_by_raw: Dict[str, Path],
                           lingbot_repo: Path, weights: Path, device: str,
                           batch_size: int, grid_size: int,
                           cache_path: Path, identity: str,
                           patch_relation: str):
    if patch_relation == "directional":
        relation_function = directional_patch_relation_features
        relation_names = directional_patch_feature_names()
    elif patch_relation == "symmetric":
        relation_function = symmetric_patch_relation_features
        relation_names = patch_feature_names()
    else:
        raise ValueError(f"unsupported patch relation {patch_relation!r}")
    if cache_path.is_file():
        cache = np.load(cache_path, allow_pickle=False)
        cached_identity = str(cache["identity"].item())
        if cached_identity != identity:
            raise RuntimeError(
                f"feature cache identity mismatch: remove {cache_path} explicitly")
        patch = cache["patch"].astype(np.float32)
        temporal = cache["temporal"].astype(np.float32)
        cached_names = tuple(cache["patch_names"].astype(str).tolist())
        if (patch.shape != (len(selected_frame), len(relation_names))
                or temporal.shape != (
                    len(selected_frame), len(temporal_feature_names()))
                or cached_names != relation_names):
            raise RuntimeError("feature cache has an unexpected shape")
        return patch, temporal, {"cache_hit": True, "seconds": 0.0}

    raw_paths = sorted(set(selected_frame["query_path"]).union(
        selected_frame["candidate_path"]))
    readable_paths = [readable_path_by_raw[path] for path in raw_paths]
    missing = [str(path) for path in readable_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected images are unavailable; first={missing[0]}")
    tokens, extraction_seconds = load_exact_patch_tokens(
        readable_paths, lingbot_repo, weights, device, batch_size, grid_size)
    index = {path: position for position, path in enumerate(raw_paths)}
    patch_rows = []
    started = time.perf_counter()
    for row_index, row in enumerate(selected_frame.itertuples()):
        patch_rows.append(relation_function(
            tokens[index[row.query_path]],
            tokens[index[row.candidate_path]],
            row.dino_cosine))
        if row_index == 0 or (row_index + 1) % 500 == 0:
            print(f"[relation] {row_index + 1}/{len(selected_frame)}", flush=True)
    relation_seconds = time.perf_counter() - started
    patch = np.asarray(patch_rows, dtype=np.float32)
    temporal = build_temporal_features(full_frame, selected_frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        identity=np.asarray(identity),
        patch=patch.astype(np.float32),
        temporal=temporal.astype(np.float32),
        patch_names=np.asarray(relation_names),
        temporal_names=np.asarray(temporal_feature_names()),
    )
    del tokens
    return patch, temporal, {
        "cache_hit": False,
        "patch_extraction_seconds": extraction_seconds,
        "relation_seconds": relation_seconds,
        "unique_images": len(raw_paths),
    }


def logistic_pipeline(c_value: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value), class_weight="balanced", max_iter=5000,
            solver="liblinear", random_state=seed),
    )


def scene_oof(features: np.ndarray, labels: np.ndarray,
              scenes: np.ndarray, c_value: float, seed: int) -> np.ndarray:
    from sklearn.model_selection import LeaveOneGroupOut

    probabilities = np.full(labels.shape, np.nan, dtype=np.float64)
    splitter = LeaveOneGroupOut()
    for train_index, test_index in splitter.split(
            features, labels, groups=scenes):
        if len(np.unique(labels[train_index])) != 2:
            raise ValueError("an inner training fold contains only one class")
        model = logistic_pipeline(c_value, seed)
        model.fit(features[train_index], labels[train_index])
        probabilities[test_index] = model.predict_proba(
            features[test_index])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("scene-disjoint OOF probabilities are incomplete")
    return probabilities


def metric_value(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    return -math.inf if value is None else float(value)


def evaluate_family(name: str, features: np.ndarray, labels: np.ndarray,
                    scenes: np.ndarray, kinds: np.ndarray,
                    train_mask: np.ndarray,
                    test_mask: np.ndarray, top1_mask: np.ndarray,
                    c_values: Sequence[float], seed: int,
                    min_top1_calibration: int):
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    train_scenes = scenes[train_mask]
    train_top1 = top1_mask[train_mask]
    if len(np.unique(train_scenes)) < 3:
        raise ValueError("nested training needs at least three scenes")

    candidates = []
    for c_value in c_values:
        probability = scene_oof(
            train_features, train_labels, train_scenes, c_value, seed)
        pair_metrics = binary_metrics(train_labels, probability)
        top1_metrics = binary_metrics(
            train_labels[train_top1], probability[train_top1])
        candidates.append((
            (metric_value(top1_metrics, "average_precision"),
             metric_value(top1_metrics, "roc_auc"),
             metric_value(pair_metrics, "average_precision"),
             -float(c_value)),
            float(c_value), probability, pair_metrics, top1_metrics,
        ))
    _key, selected_c, train_oof_probability, train_pair_metrics, train_top1_metrics = max(
        candidates, key=lambda item: item[0])

    model = logistic_pipeline(selected_c, seed)
    model.fit(train_features, train_labels)
    all_probability = model.predict_proba(features)[:, 1]
    heldout_probability = all_probability[test_mask]
    heldout_top1 = top1_mask[test_mask]
    heldout_labels = labels[test_mask]

    calibration_labels = train_labels[train_top1]
    calibration_probability = train_oof_probability[train_top1]
    thresholds = calibrate_zero_error_thresholds(
        calibration_labels, calibration_probability,
        min_samples=min_top1_calibration)
    min1_thresholds = calibrate_zero_error_thresholds(
        calibration_labels, calibration_probability, min_samples=1)
    heldout_top1_probability = heldout_probability[heldout_top1]
    heldout_top1_labels = heldout_labels[heldout_top1]
    heldout_top1_scenes = scenes[test_mask][heldout_top1]
    heldout_top1_kinds = kinds[test_mask][heldout_top1]
    heldout_decisions = selective_decisions(
        heldout_top1_probability, thresholds)

    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    result = {
        "name": name,
        "feature_dim": int(features.shape[1]),
        "selected_C_from_train_oof": selected_c,
        "train_scene_oof": {
            "pair": train_pair_metrics,
            "top1": train_top1_metrics,
        },
        "heldout": {
            "pair": binary_metrics(heldout_labels, heldout_probability),
            "top1": binary_metrics(
                heldout_top1_labels, heldout_top1_probability),
            "top1_selective_thresholds": threshold_dict(thresholds),
            "top1_selective_cascade": cascade_metrics(
                heldout_top1_labels, heldout_decisions),
            "top1_by_kind": grouped_top1_metrics(
                heldout_top1_kinds, heldout_top1_labels,
                heldout_top1_probability, heldout_decisions),
            "top1_by_scene": grouped_top1_metrics(
                heldout_top1_scenes, heldout_top1_labels,
                heldout_top1_probability, heldout_decisions),
            "top1_min1_tail_capacity": {
                "thresholds": threshold_dict(min1_thresholds),
                "cascade": cascade_metrics(
                    heldout_top1_labels,
                    selective_decisions(
                        heldout_top1_probability, min1_thresholds)),
            },
        },
    }
    portable = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": logistic.coef_[0].tolist(),
        "intercept": float(logistic.intercept_[0]),
        "thresholds": threshold_dict(thresholds),
    }
    return result, portable, all_probability


def covisibility_ranking_metrics(groups: np.ndarray, ranks: np.ndarray,
                                 covisibility: np.ndarray,
                                 scores: np.ndarray,
                                 positive_threshold: float = 0.5) -> dict:
    """Evaluate top-K reranking against continuous task overlap."""
    groups = np.asarray(groups, dtype=str).reshape(-1)
    ranks = np.asarray(ranks, dtype=np.int64).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not (groups.shape == ranks.shape == covisibility.shape == scores.shape):
        raise ValueError("ranking inputs must be aligned")
    if (not len(groups) or not np.isfinite(covisibility).all()
            or not np.isfinite(scores).all()):
        raise ValueError("ranking inputs must be non-empty and finite")

    selected_overlap = []
    positive_ranks = []
    sessions_with_positive = 0
    selected_positive = 0
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        chosen = index[np.argmax(scores[index])]
        overlap = float(covisibility[chosen])
        selected_overlap.append(overlap)
        positive = index[covisibility[index] >= positive_threshold]
        if positive.size:
            sessions_with_positive += 1
            order = index[np.argsort(-scores[index], kind="stable")]
            first = next(
                position for position, item in enumerate(order, 1)
                if covisibility[item] >= positive_threshold)
            positive_ranks.append(first)
            selected_positive += int(overlap >= positive_threshold)
    return {
        "sessions": int(len(selected_overlap)),
        "sessions_with_positive": sessions_with_positive,
        "selected_positive": selected_positive,
        "conditional_recall_at_1": (
            selected_positive / sessions_with_positive
            if sessions_with_positive else None),
        "selected_overlap_mean": float(np.mean(selected_overlap)),
        "selected_overlap_median": float(np.median(selected_overlap)),
        "first_positive_rank_mean": (
            float(np.mean(positive_ranks)) if positive_ranks else None),
        "first_positive_rank_median": (
            float(np.median(positive_ranks)) if positive_ranks else None),
        "mean_reciprocal_positive_rank": (
            float(np.mean(1.0 / np.asarray(positive_ranks)))
            if positive_ranks else None),
    }


def write_selected_csv(path: Path, frame, probabilities: Dict[str, np.ndarray],
                       test_mask: np.ndarray) -> None:
    rows = frame.loc[test_mask].copy()
    for name, values in probabilities.items():
        rows[f"{name}_probability"] = values
    rows.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--cls-cache", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--heldout-scene", action="append", required=True)
    parser.add_argument(
        "--path-map", action="append", default=[],
        help="optional OLD=NEW prefix replacement for moved episode images")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--patch-relation", choices=("symmetric", "directional"),
        default="symmetric",
        help="preserve query/memory direction for directional overlap labels")
    parser.add_argument("--min-top1-calibration", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--c-value", type=float, action="append", default=[])
    parser.add_argument("--expected-weight-sha", default=DEFAULT_WEIGHT_SHA)
    parser.add_argument("--expected-lingbot-commit", default=DEFAULT_LINGBOT_COMMIT)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if args.batch_size < 1 or args.top_k < 1 or args.grid_size < 2:
        raise ValueError("batch size/top-k must be positive and grid size >= 2")
    if args.min_top1_calibration < 1:
        raise ValueError("min top1 calibration must be positive")
    c_values = tuple(args.c_value or (0.001, 0.01, 0.1, 1.0))
    if any(not np.isfinite(value) or value <= 0.0 for value in c_values):
        raise ValueError("all C values must be finite and positive")
    for required in (
            args.teacher_csv, args.cls_cache, args.lingbot_repo, args.weights):
        if not required.exists():
            raise FileNotFoundError(required)
    weight_sha = sha256(args.weights)
    if args.expected_weight_sha and weight_sha != args.expected_weight_sha:
        raise RuntimeError(
            f"LingBot weight SHA mismatch: {weight_sha}")
    lingbot_commit = git_value(args.lingbot_repo, "rev-parse", "HEAD")
    if (args.expected_lingbot_commit
            and lingbot_commit != args.expected_lingbot_commit):
        raise RuntimeError(
            f"LingBot commit mismatch: {lingbot_commit}")

    teacher_sha = sha256(args.teacher_csv)
    frame = pd.read_csv(args.teacher_csv)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing_columns)}")
    if frame.duplicated(["session_id", "candidate_path"]).any():
        raise ValueError("teacher CSV contains duplicate session/candidate pairs")
    if not frame["teacher_pass"].isin([-1, 0, 1]).all():
        raise ValueError("teacher labels must be -1 (unevaluated), 0, or 1")
    cls_by_path, cls_weight_sha = load_cls_cache(args.cls_cache)
    if cls_weight_sha != weight_sha:
        raise RuntimeError("CLS cache and requested weights have different SHA")
    absent_cls = sorted(
        (set(frame["query_path"]) | set(frame["candidate_path"]))
        - set(cls_by_path))
    if absent_cls:
        raise ValueError(
            f"CLS cache lacks {len(absent_cls)} teacher images; first={absent_cls[0]}")

    # Recompute a deterministic sample of global cosines to fail fast on a
    # mismatched CSV/cache combination.
    sample = frame.iloc[::max(1, len(frame) // 256)]
    reconstructed = np.asarray([
        float(cls_by_path[row.query_path] @ cls_by_path[row.candidate_path])
        for row in sample.itertuples()])
    maximum_cosine_error = float(np.max(np.abs(
        reconstructed - sample["dino_cosine"].to_numpy())))
    if maximum_cosine_error > 5e-3:
        raise RuntimeError(
            f"teacher/CLS cosine mismatch: max error {maximum_cosine_error}")

    heldout_set = set(args.heldout_scene)
    all_scenes = set(frame["scene"].unique().tolist())
    missing_heldout = heldout_set - all_scenes
    if missing_heldout:
        raise ValueError(f"heldout scenes absent: {sorted(missing_heldout)}")
    train_scenes = sorted(all_scenes - heldout_set)
    if len(train_scenes) < 3:
        raise ValueError("at least three non-heldout training scenes are required")
    selected = select_hard_candidates(frame, args.top_k)
    if not selected["teacher_pass"].isin([-1, 0, 1]).all():
        raise ValueError("selected teacher labels must be -1, 0, or 1")
    supervised = selected["teacher_pass"].isin([0, 1]).to_numpy()
    all_train_mask = selected["scene"].isin(train_scenes).to_numpy()
    all_test_mask = selected["scene"].isin(sorted(heldout_set)).to_numpy()
    train_mask = all_train_mask & supervised
    test_mask = all_test_mask & supervised
    if (np.any(all_train_mask & all_test_mask) or not train_mask.any()
            or not test_mask.any()):
        raise RuntimeError("train/test scene split is invalid")
    top1_mask = selected["candidate_rank"].eq(1).to_numpy()
    labels = selected["teacher_pass"].to_numpy(dtype=np.int64)
    scenes = selected["scene"].to_numpy(dtype=str)
    kinds = selected["kind"].to_numpy(dtype=str)

    mappings = parse_path_maps(args.path_map)
    selected_raw_paths = set(selected["query_path"]).union(
        selected["candidate_path"])
    readable_path_by_raw = {
        raw: remap_path(raw, mappings) for raw in selected_raw_paths}
    identity = selection_digest(
        selected, frame, args.top_k, args.grid_size, weight_sha,
        args.patch_relation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_path = args.out_dir / "patch_temporal_features.npz"
    patch, temporal, feature_timing = build_or_load_features(
        selected, frame, cls_by_path, readable_path_by_raw,
        args.lingbot_repo.resolve(), args.weights.resolve(), args.device,
        args.batch_size, args.grid_size, feature_cache_path, identity,
        args.patch_relation)
    combined = combine_patch_temporal(patch, temporal).astype(np.float32)
    feature_families = {
        "cosine": patch[:, :1],
        "temporal": temporal,
        "patch": patch,
        "patch_temporal": combined,
    }
    patch_names = (directional_patch_feature_names()
                   if args.patch_relation == "directional"
                   else patch_feature_names())
    combined_names = (directional_combined_feature_names()
                      if args.patch_relation == "directional"
                      else combined_feature_names())
    family_names = {
        "cosine": ("dino_global_cosine",),
        "temporal": temporal_feature_names(),
        "patch": patch_names,
        "patch_temporal": combined_names,
    }
    if args.patch_relation == "directional":
        # Directional features are a strict superset of the old symmetric
        # summaries, so derive the control without another DINO forward pass.
        symmetric_patch = symmetric_from_directional_patch_features(
            patch).astype(np.float32)
        feature_families["patch_symmetric_control"] = symmetric_patch
        feature_families["patch_temporal_symmetric_control"] = (
            combine_patch_temporal(symmetric_patch, temporal).astype(np.float32))
        family_names["patch_symmetric_control"] = patch_feature_names()
        family_names["patch_temporal_symmetric_control"] = (
            combined_feature_names())

    evaluations = {}
    portable = {}
    heldout_probabilities = {}
    for name, features in feature_families.items():
        result, model_payload, all_probability = evaluate_family(
            name, features, labels, scenes, kinds, train_mask, test_mask,
            top1_mask, c_values, args.seed, args.min_top1_calibration)
        if "teacher_covis" in selected.columns:
            heldout_covis = selected.loc[
                all_test_mask, "teacher_covis"].to_numpy(dtype=np.float64)
            if np.isfinite(heldout_covis).all():
                result["heldout"]["covisibility_ranking"] = (
                    covisibility_ranking_metrics(
                        selected.loc[all_test_mask, "session_id"],
                        selected.loc[all_test_mask, "candidate_rank"],
                        heldout_covis, all_probability[all_test_mask]))
        evaluations[name] = result
        portable[name] = dict(model_payload, feature_names=family_names[name])
        heldout_probabilities[name] = all_probability[all_test_mask]
        print(json.dumps({name: result}, indent=2), flush=True)

    primary = "patch_temporal"
    teacher_kind = (
        "task_aligned_covisibility"
        if "teacher_covis" in selected.columns else "legacy_binary")
    export = {
        "deployment_approved": False,
        "reason": (
            "scene-disjoint offline teacher diagnostic; requires "
            "zero-error heldout confidence tails and closed-loop A/B"),
        "feature_version": FEATURE_VERSION,
        "patch_relation": args.patch_relation,
        "teacher_kind": teacher_kind,
        "feature_names": list(family_names[primary]),
        **portable[primary],
        "teacher_csv_sha256": teacher_sha,
        "lingbot_weight_sha256": weight_sha,
        "lingbot_commit": lingbot_commit,
        "train_scenes": train_scenes,
        "heldout_scenes": sorted(heldout_set),
        "top_k": args.top_k,
        "grid_size": args.grid_size,
    }
    with open(args.out_dir / "diagnostic_patch_temporal_router_not_for_deployment.json",
              "w", encoding="utf-8") as handle:
        json.dump(export, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_selected_csv(
        args.out_dir / "heldout_selected_pairs.csv", selected,
        heldout_probabilities, all_test_mask)

    report = {
        "deployment_approved": False,
        "feature_version": FEATURE_VERSION,
        "patch_relation": args.patch_relation,
        "teacher_kind": teacher_kind,
        "created_at_unix": time.time(),
        "repo_root": str(Path(__file__).resolve().parents[1]),
        "repo_commit": git_value(
            Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "teacher_csv": str(args.teacher_csv.resolve()),
        "teacher_csv_sha256": teacher_sha,
        "cls_cache": str(args.cls_cache.resolve()),
        "lingbot_repo": str(args.lingbot_repo.resolve()),
        "lingbot_commit": lingbot_commit,
        "lingbot_weights": str(args.weights.resolve()),
        "lingbot_weight_sha256": weight_sha,
        "maximum_reconstructed_cosine_error": maximum_cosine_error,
        "selection_identity": identity,
        "top_k": args.top_k,
        "grid_size": args.grid_size,
        "train_scenes": train_scenes,
        "heldout_scenes": sorted(heldout_set),
        "train_pairs": int(train_mask.sum()),
        "heldout_pairs": int(test_mask.sum()),
        "train_selected_pairs": int(all_train_mask.sum()),
        "heldout_selected_pairs": int(all_test_mask.sum()),
        "train_ignored_pairs": int(np.sum(all_train_mask & ~supervised)),
        "heldout_ignored_pairs": int(np.sum(all_test_mask & ~supervised)),
        "train_top1_sessions": int(np.sum(train_mask & top1_mask)),
        "heldout_top1_sessions": int(np.sum(test_mask & top1_mask)),
        "heldout_selected_sessions": int(
            selected.loc[all_test_mask, "session_id"].nunique()),
        "unevaluated_teacher_pairs": int(
            frame["teacher_pass"].eq(-1).sum()),
        "feature_timing": feature_timing,
        "candidate_C": list(c_values),
        "primary_family": primary,
        "evaluations": evaluations,
    }
    with open(args.out_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "primary": evaluations[primary],
        "cosine": evaluations["cosine"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
