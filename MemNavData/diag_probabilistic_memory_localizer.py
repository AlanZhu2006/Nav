#!/usr/bin/env python3
"""Audit separate ranking and no-match objectives on cached DINO features.

This is a CPU-only objective diagnostic.  It reuses an exact frozen-DINO
feature cache, selects regularization only with scene-grouped OOF predictions
on training scenes, and evaluates held-out scenes once.  The resulting linear
heads are deliberately marked not for deployment: their purpose is to decide
whether the next dense model should have separate candidate-ranking and
no-match heads instead of another scalar Novel/Revisit gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

try:
    from MemNavData.diag_patch_temporal_router import (
        covisibility_ranking_metrics,
        select_hard_candidates,
    )
    from MemNavData.listwise_router import (
        fit_listwise_linear,
        scene_group_oof_scores,
    )
    from MemNavData.patch_temporal_router import combine_patch_temporal
    from MemNavData.probabilistic_memory_localizer import (
        evaluate_probabilistic_set,
        fit_probabilistic_set,
        scene_group_oof_probabilities,
    )
except ModuleNotFoundError:  # direct script invocation
    from diag_patch_temporal_router import (  # type: ignore
        covisibility_ranking_metrics,
        select_hard_candidates,
    )
    from listwise_router import (  # type: ignore
        fit_listwise_linear,
        scene_group_oof_scores,
    )
    from patch_temporal_router import combine_patch_temporal  # type: ignore
    from probabilistic_memory_localizer import (  # type: ignore
        evaluate_probabilistic_set,
        fit_probabilistic_set,
        scene_group_oof_probabilities,
    )


REQUIRED_COLUMNS = {
    "session_id", "scene", "kind", "candidate_path", "candidate_frame",
    "dino_cosine", "teacher_covis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(metrics: dict, name: str, default: float = -np.inf) -> float:
    value = metrics.get(name)
    return float(value) if value is not None and np.isfinite(value) else default


def grouped_ranking_metrics(
        partitions: np.ndarray, groups: np.ndarray, ranks: np.ndarray,
        covisibility: np.ndarray, scores: np.ndarray, *,
        positive_threshold: float) -> dict:
    """Apply the configured positive threshold to every report partition."""
    partitions = np.asarray(partitions, dtype=str).reshape(-1)
    return {
        partition: covisibility_ranking_metrics(
            groups[partitions == partition], ranks[partitions == partition],
            covisibility[partitions == partition],
            scores[partitions == partition],
            positive_threshold=positive_threshold)
        for partition in sorted(np.unique(partitions))
    }


def select_probabilistic_l2(
        features: np.ndarray, groups: np.ndarray, scenes: np.ndarray,
        covisibility: np.ndarray, l2_values: Sequence[float], *, folds: int,
        positive_threshold: float,
        max_iterations: int) -> tuple[float, dict]:
    """Choose K+1 regularization without observing held-out scenes."""
    candidates = []
    reports = {}
    for l2 in l2_values:
        candidate, dustbin = scene_group_oof_probabilities(
            features, groups, scenes, covisibility, l2=float(l2),
            positive_threshold=positive_threshold, folds=folds,
            max_iterations=max_iterations)
        metrics = evaluate_probabilistic_set(
            groups, covisibility, candidate, dustbin,
            positive_threshold=positive_threshold)
        reports[str(float(l2))] = metrics
        # Joint correctness is primary.  Calibration, ranking, then the
        # stronger regularizer break ties, all using training-scene OOF only.
        key = (
            _metric(metrics, "joint_localization_accuracy"),
            -_metric(metrics, "match_brier", default=np.inf),
            _metric(metrics, "conditional_candidate_recall_at_1"),
            -float(l2),
        )
        candidates.append((key, float(l2)))
    return max(candidates, key=lambda item: item[0])[1], reports


def select_listwise_l2(
        features: np.ndarray, groups: np.ndarray, ranks: np.ndarray,
        scenes: np.ndarray, covisibility: np.ndarray,
        l2_values: Sequence[float], *, folds: int,
        positive_threshold: float,
        max_iterations: int) -> tuple[float, dict]:
    """Choose rank-head regularization with scene-grouped OOF scores."""
    candidates = []
    reports = {}
    for l2 in l2_values:
        scores = scene_group_oof_scores(
            features, groups, scenes, covisibility, l2=float(l2),
            positive_threshold=positive_threshold, folds=folds,
            max_iterations=max_iterations)
        metrics = covisibility_ranking_metrics(
            groups, ranks, covisibility, scores,
            positive_threshold=positive_threshold)
        reports[str(float(l2))] = metrics
        key = (
            _metric(metrics, "conditional_recall_at_1"),
            _metric(metrics, "selected_overlap_mean"),
            _metric(metrics, "mean_reciprocal_positive_rank"),
            -float(l2),
        )
        candidates.append((key, float(l2)))
    return max(candidates, key=lambda item: item[0])[1], reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--heldout-scene", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--l2", type=float, action="append", default=[])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if args.top_k < 1 or args.folds < 2 or args.max_iterations < 1:
        raise ValueError("top-k/iterations must be positive and folds >= 2")
    if not 0.0 < args.positive_threshold <= 1.0:
        raise ValueError("positive threshold must lie in (0, 1]")
    l2_values = tuple(args.l2 or (0.001, 0.01, 0.1))
    if any(not np.isfinite(value) or value <= 0.0 for value in l2_values):
        raise ValueError("all L2 values must be finite and positive")
    for path in (args.teacher_csv, args.feature_cache):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.time()
    frame = pd.read_csv(args.teacher_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    if frame.duplicated(["session_id", "candidate_path"]).any():
        raise ValueError("teacher CSV contains duplicate session/candidate pairs")
    if not np.isfinite(frame["teacher_covis"].to_numpy(dtype=np.float64)).all():
        raise ValueError("teacher co-visibility must be complete and finite")

    selected = select_hard_candidates(frame, args.top_k, "raw", 4)
    cache = np.load(args.feature_cache, allow_pickle=False)
    required_cache = {"patch", "temporal", "patch_names", "temporal_names"}
    if missing_cache := required_cache - set(cache.files):
        raise ValueError(f"feature cache missing arrays: {sorted(missing_cache)}")
    patch = np.asarray(cache["patch"], dtype=np.float64)
    temporal = np.asarray(cache["temporal"], dtype=np.float64)
    if len(selected) != len(patch) or len(selected) != len(temporal):
        raise RuntimeError(
            "selected pair count does not match frozen feature cache")
    if patch.ndim != 2 or temporal.ndim != 2:
        raise ValueError("cached patch/temporal features must be matrices")
    maximum_cosine_error = float(np.max(np.abs(
        patch[:, 0] - selected["dino_cosine"].to_numpy(dtype=np.float64))))
    if maximum_cosine_error > 5e-5:
        raise RuntimeError(
            f"teacher/feature row alignment failed: {maximum_cosine_error}")

    heldout = set(args.heldout_scene)
    all_scenes = set(selected["scene"].astype(str).unique())
    if missing_scenes := heldout - all_scenes:
        raise ValueError(f"held-out scenes absent: {sorted(missing_scenes)}")
    train_scenes = sorted(all_scenes - heldout)
    if len(train_scenes) < 2:
        raise ValueError("at least two training scenes are required")
    train_mask = selected["scene"].isin(train_scenes).to_numpy()
    test_mask = selected["scene"].isin(sorted(heldout)).to_numpy()
    if np.any(train_mask & test_mask) or not train_mask.any() or not test_mask.any():
        raise RuntimeError("scene split is invalid")

    combined = combine_patch_temporal(patch, temporal)
    families = {
        "cosine": patch[:, :1],
        "patch": patch,
        "patch_temporal": combined,
    }
    groups = selected["session_id"].to_numpy(dtype=str)
    scenes = selected["scene"].to_numpy(dtype=str)
    kinds = selected["kind"].to_numpy(dtype=str)
    ranks = selected["candidate_rank"].to_numpy(dtype=np.int64)
    covisibility = selected["teacher_covis"].to_numpy(dtype=np.float64)

    family_reports = {}
    portable_models = {}
    for name, features in families.items():
        selected_l2, oof = select_probabilistic_l2(
            features[train_mask], groups[train_mask], scenes[train_mask],
            covisibility[train_mask], l2_values, folds=args.folds,
            positive_threshold=args.positive_threshold,
            max_iterations=args.max_iterations)
        model = fit_probabilistic_set(
            features[train_mask], groups[train_mask],
            covisibility[train_mask], l2=selected_l2,
            positive_threshold=args.positive_threshold,
            max_iterations=args.max_iterations)
        candidate, dustbin = model.predict(
            features[test_mask], groups[test_mask])
        heldout_metrics = evaluate_probabilistic_set(
            groups[test_mask], covisibility[test_mask], candidate, dustbin,
            positive_threshold=args.positive_threshold)
        family_reports[name] = {
            "selected_l2_from_train_scene_oof": selected_l2,
            "train_scene_oof_by_l2": oof,
            "heldout": heldout_metrics,
        }
        portable_models[name] = model.portable()

    rank_features = combined
    listwise_l2, listwise_oof = select_listwise_l2(
        rank_features[train_mask], groups[train_mask], ranks[train_mask],
        scenes[train_mask], covisibility[train_mask], l2_values,
        folds=args.folds, positive_threshold=args.positive_threshold,
        max_iterations=args.max_iterations)
    listwise_model = fit_listwise_linear(
        rank_features[train_mask], groups[train_mask],
        covisibility[train_mask], l2=listwise_l2,
        positive_threshold=args.positive_threshold,
        max_iterations=args.max_iterations)
    listwise_scores = listwise_model.score(rank_features[test_mask])
    heldout_rank = {
        "dino_cosine": covisibility_ranking_metrics(
            groups[test_mask], ranks[test_mask], covisibility[test_mask],
            selected.loc[test_mask, "dino_cosine"].to_numpy(dtype=np.float64),
            positive_threshold=args.positive_threshold),
        "listwise_patch_temporal": covisibility_ranking_metrics(
            groups[test_mask], ranks[test_mask], covisibility[test_mask],
            listwise_scores, positive_threshold=args.positive_threshold),
        "listwise_by_scene": grouped_ranking_metrics(
            scenes[test_mask], groups[test_mask], ranks[test_mask],
            covisibility[test_mask], listwise_scores,
            positive_threshold=args.positive_threshold),
        "listwise_by_kind": grouped_ranking_metrics(
            kinds[test_mask], groups[test_mask], ranks[test_mask],
            covisibility[test_mask], listwise_scores,
            positive_threshold=args.positive_threshold),
    }

    report = {
        "deployment_approved": False,
        "reason": (
            "CPU linear objective diagnostic on reused frozen features; "
            "requires untouched-scene and closed-loop validation"),
        "purpose": "separate candidate ranking from no-match localization",
        "created_at_unix": time.time(),
        "seconds": time.time() - started,
        "inputs": {
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_csv_sha256": sha256(args.teacher_csv),
            "feature_cache": str(args.feature_cache.resolve()),
            "feature_cache_sha256": sha256(args.feature_cache),
            "feature_cache_identity": (
                str(cache["identity"].item()) if "identity" in cache else None),
            "maximum_cosine_alignment_error": maximum_cosine_error,
        },
        "protocol": {
            "candidate_selection": "raw_dino_top_k",
            "top_k": args.top_k,
            "positive_threshold": args.positive_threshold,
            "l2_candidates": [float(value) for value in l2_values],
            "scene_grouped_oof_folds": min(args.folds, len(train_scenes)),
            "train_scenes": train_scenes,
            "heldout_scenes": sorted(heldout),
            "train_sessions": int(selected.loc[train_mask, "session_id"].nunique()),
            "heldout_sessions": int(selected.loc[test_mask, "session_id"].nunique()),
            "heldout_positive_sessions": int(sum(
                group["teacher_covis"].max() >= args.positive_threshold
                for _, group in selected.loc[test_mask].groupby("session_id"))),
        },
        "probabilistic_k_plus_one": family_reports,
        "candidate_ranking": {
            "selected_l2_from_train_scene_oof": listwise_l2,
            "train_scene_oof_by_l2": listwise_oof,
            "heldout": heldout_rank,
        },
        "diagnostic_models_not_for_deployment": {
            "probabilistic": portable_models,
            "listwise_patch_temporal": listwise_model.portable(),
        },
        "decision": {
            "single_shared_scalar_head": "no_go",
            "separate_rank_and_no_match_heads": "go_for_dense_offline_prototype",
            "old_gate_or_end_to_end_long_train": "no_go",
        },
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "out_report": str(args.out_report),
        "seconds": report["seconds"],
        "heldout_ranking": heldout_rank,
        "heldout_probabilistic": {
            name: result["heldout"] for name, result in family_reports.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
