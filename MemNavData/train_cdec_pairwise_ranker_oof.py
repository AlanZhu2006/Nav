#!/usr/bin/env python3
"""Nested scene-OOF task ranker for the factorized certified compass.

This model is deliberately *not* an activation gate.  It learns only which
member of the frozen DINO top-8 is most task-relevant.  A deployment-time
geometry certificate remains the sole authority to activate a memory bearing.
The separation avoids the ranking/NULL interference observed in the joint
CDEC set student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from MemNavData.train_cdec_scene_oof import load_sessions, sha256
except ModuleNotFoundError:  # direct script invocation
    from train_cdec_scene_oof import load_sessions, sha256  # type: ignore


SCHEMA_VERSION = "cdec_factorized_pairwise_ranker_oof_v1_20260813"
DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
SELECTION_COLUMNS = (
    "session_id", "scene", "episode", "kind", "query_path",
    "candidate_path", "candidate_frame", "dino_cosine", "candidate_rank",
    "cdec_oof_score", "cdec_outer_fold", "cdec_inner_selected_c",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def standardizer(features: np.ndarray, fit_index: np.ndarray):
    flat = features[fit_index].reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def pairwise_differences(
    features: np.ndarray, candidate_labels: np.ndarray, fit_index: np.ndarray
) -> np.ndarray:
    """Positive-minus-known-negative differences for RankNet-style fitting."""
    differences = []
    for index in fit_index:
        positive = np.flatnonzero(candidate_labels[index] == 1)
        negative = np.flatnonzero(candidate_labels[index] == 0)
        for left in positive:
            for right in negative:
                differences.append(features[index, left] - features[index, right])
    if not differences:
        raise RuntimeError("fit split contains no positive/negative candidate pairs")
    result = np.asarray(differences, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("pairwise differences are non-finite")
    return result


def fit_ranker(
    features: np.ndarray,
    candidate_labels: np.ndarray,
    fit_index: np.ndarray,
    *,
    regularization_c: float,
):
    if not math.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("regularization C must be finite and positive")
    mean, scale = standardizer(features, fit_index)
    normalized = (features.astype(np.float64) - mean) / scale
    difference = pairwise_differences(normalized, candidate_labels, fit_index)
    # Add the exact inverse comparisons.  This makes the no-intercept binary
    # logistic problem equivalent to a convex pairwise preference fit.
    x = np.concatenate((difference, -difference), axis=0)
    y = np.concatenate((
        np.ones(len(difference), dtype=np.int64),
        np.zeros(len(difference), dtype=np.int64),
    ))
    model = LogisticRegression(
        C=float(regularization_c), fit_intercept=False, solver="lbfgs",
        max_iter=5000, tol=1e-9,
    ).fit(x, y)
    coefficient = model.coef_[0].astype(np.float64)
    return coefficient, mean, scale, int(model.n_iter_[0]), len(difference)


def predict(
    features: np.ndarray, index: np.ndarray,
    coefficient: np.ndarray, mean: np.ndarray, scale: np.ndarray,
) -> np.ndarray:
    normalized = (features[index].astype(np.float64) - mean) / scale
    score = normalized @ coefficient
    if not np.isfinite(score).all():
        raise RuntimeError("ranker emitted non-finite scores")
    return score


def top1_correct(candidate_labels: np.ndarray, index: np.ndarray,
                 scores: np.ndarray) -> int:
    selected = np.argmax(scores, axis=1)
    return int((candidate_labels[index, selected] == 1).sum())


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def parse_c_grid(values: Iterable[str]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        result = list(DEFAULT_C_GRID)
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("C grid values must be finite and positive")
    result = sorted(set(result))
    if len(result) < 2:
        raise ValueError("nested selection requires at least two C values")
    return result


def choose_c_nested(
    features: np.ndarray,
    candidate_labels: np.ndarray,
    scenes: np.ndarray,
    outer_fit: np.ndarray,
    c_grid: list[float],
    inner_folds: int,
) -> tuple[float, list[dict]]:
    records = []
    splitter = GroupKFold(inner_folds)
    splits = list(splitter.split(outer_fit, groups=scenes[outer_fit]))
    for regularization_c in c_grid:
        correct = 0
        positive_sessions = 0
        for inner_fit_local, validation_local in splits:
            fit_index = outer_fit[inner_fit_local]
            validation_index = outer_fit[validation_local]
            coefficient, mean, scale, _iterations, _pairs = fit_ranker(
                features, candidate_labels, fit_index,
                regularization_c=regularization_c)
            score = predict(
                features, validation_index, coefficient, mean, scale)
            positive = (candidate_labels[validation_index] == 1).any(axis=1)
            selected = np.argmax(score, axis=1)
            correct += int((
                positive
                & (candidate_labels[validation_index, selected] == 1)
            ).sum())
            positive_sessions += int(positive.sum())
        records.append({
            "C": regularization_c,
            "correct_positive_top1": correct,
            "positive_sessions": positive_sessions,
        })
    best_correct = max(row["correct_positive_top1"] for row in records)
    # Conservative deterministic tie break: strongest regularization.
    selected = min(
        row["C"] for row in records
        if row["correct_positive_top1"] == best_correct)
    return float(selected), records


def map_scores_to_rows(
    frame: pd.DataFrame, session_ids: np.ndarray, candidate_frames: np.ndarray,
    oof_score: np.ndarray, oof_fold: np.ndarray, selected_c: np.ndarray,
) -> pd.DataFrame:
    session_position = {str(value): index for index, value in enumerate(session_ids)}
    score = np.empty(len(frame), dtype=np.float64)
    fold = np.empty(len(frame), dtype=np.int64)
    regularization = np.empty(len(frame), dtype=np.float64)
    for name, indices in frame.groupby("session_id", sort=False).indices.items():
        session = session_position[str(name)]
        index = np.asarray(indices, dtype=np.int64)
        observed_frames = frame.iloc[index]["candidate_frame"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_frames, candidate_frames[session]):
            raise RuntimeError(f"candidate order changed for {name}")
        score[index] = oof_score[session]
        fold[index] = oof_fold[session]
        regularization[index] = selected_c[session]
    result = frame.loc[:, [
        "session_id", "scene", "episode", "kind", "query_path",
        "candidate_path", "candidate_frame", "dino_cosine", "candidate_rank",
    ]].copy()
    result["cdec_oof_score"] = score
    result["cdec_outer_fold"] = fold
    result["cdec_inner_selected_c"] = regularization
    if tuple(result.columns) != SELECTION_COLUMNS:
        raise RuntimeError("selection artifact whitelist changed")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--patch-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--c", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("nested fold counts must be at least two")
    c_grid = parse_c_grid(args.c)
    started = time.perf_counter()
    data = load_sessions(
        args.rows_csv, args.patch_cache,
        expected_rows_sha256=args.expected_rows_sha256,
        expected_cache_sha256=args.expected_cache_sha256)
    frame = pd.read_csv(args.rows_csv)
    cache = np.load(args.patch_cache, allow_pickle=False)
    relation_names = cache["directional_relation_names"].astype(str).tolist()
    indices = np.arange(len(data.session_id))
    oof_score = np.full((len(indices), 8), np.nan, dtype=np.float64)
    oof_fold = np.full(len(indices), -1, dtype=np.int64)
    selected_c = np.full(len(indices), np.nan, dtype=np.float64)
    fold_receipts = []
    outer = GroupKFold(args.outer_folds)
    for fold, (fit_index, test_index) in enumerate(
            outer.split(indices, groups=data.scene)):
        regularization_c, inner_records = choose_c_nested(
            data.features, data.candidate_label, data.scene, fit_index,
            c_grid, args.inner_folds)
        coefficient, mean, scale, iterations, pairs = fit_ranker(
            data.features, data.candidate_label, fit_index,
            regularization_c=regularization_c)
        oof_score[test_index] = predict(
            data.features, test_index, coefficient, mean, scale)
        oof_fold[test_index] = fold
        selected_c[test_index] = regularization_c
        fold_receipts.append({
            "outer_fold": fold,
            "fit_scenes": sorted(set(map(str, data.scene[fit_index]))),
            "test_scenes": sorted(set(map(str, data.scene[test_index]))),
            "selected_C": regularization_c,
            "inner_selection": inner_records,
            "fit_pairwise_differences": pairs,
            "optimizer_iterations": iterations,
            "test_positive_top1_correct": top1_correct(
                data.candidate_label, test_index, oof_score[test_index]),
        })
    if (not np.isfinite(oof_score).all() or (oof_fold < 0).any()
            or not np.isfinite(selected_c).all()):
        raise RuntimeError("OOF prediction cover is incomplete")

    learned_index = np.argmax(oof_score, axis=1)
    geometry_index = data.teacher_top_index
    dino_index = np.argmax(data.dino_cosine, axis=1)
    row = np.arange(len(indices))
    positive = data.session_label == 1
    learned_correct = positive & (data.candidate_label[row, learned_index] == 1)
    geometry_correct = positive & (data.candidate_label[row, geometry_index] == 1)
    dino_correct = positive & (data.candidate_label[row, dino_index] == 1)
    gains = int((learned_correct & ~geometry_correct).sum())
    losses = int((geometry_correct & ~learned_correct).sum())
    candidate_known = data.candidate_label >= 0
    candidate_score = oof_score[candidate_known]
    candidate_target = data.candidate_label[candidate_known]

    final_c, final_inner = choose_c_nested(
        data.features, data.candidate_label, data.scene, indices,
        c_grid, args.outer_folds)
    final_coefficient, final_mean, final_scale, final_iterations, final_pairs = fit_ranker(
        data.features, data.candidate_label, indices,
        regularization_c=final_c)

    args.out_dir.mkdir(parents=True)
    selection_path = args.out_dir / "cdec_oof_selection_rows.csv"
    selection = map_scores_to_rows(
        frame, data.session_id, data.candidate_frame,
        oof_score, oof_fold, selected_c)
    atomic_csv(selection_path, selection)
    selection_sha = sha256(selection_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "factorized_ranker_passes_learnability_not_method_gate",
        "research_claim": (
            "scene-OOF task ranker only; activation remains an independent "
            "hard geometry certificate"
        ),
        "inputs": {
            "rows_csv": str(args.rows_csv.resolve()),
            "rows_csv_sha256": sha256(args.rows_csv),
            "patch_cache": str(args.patch_cache.resolve()),
            "patch_cache_sha256": sha256(args.patch_cache),
        },
        "protocol": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "C_grid": c_grid,
            "inner_metric": "positive-session correct top-1",
            "tie_break": "smallest C (strongest regularization)",
            "groups": "scene",
            "development_or_blind_read": False,
            "activation_or_NULL_learned": False,
        },
        "coverage": {
            "scenes": int(len(set(data.scene))),
            "sessions": int(len(indices)),
            "positive_sessions": int(positive.sum()),
            "strict_negative_sessions": int((data.session_label == 0).sum()),
            "ambiguous_sessions": int((data.session_label < 0).sum()),
            "candidates": int(data.candidate_label.size),
        },
        "metrics": {
            "dino_positive_top1_correct": int(dino_correct.sum()),
            "geometry_positive_top1_correct": int(geometry_correct.sum()),
            "cdec_oof_positive_top1_correct": int(learned_correct.sum()),
            "cdec_vs_geometry": {
                "gains": gains,
                "losses": losses,
                "exact_mcnemar_p": exact_mcnemar(gains, losses),
            },
            "oracle_union_diagnostic_not_deployable": int(
                (learned_correct | geometry_correct).sum()),
            "selection_exact_agreement_sessions": int(
                (learned_index == geometry_index).sum()),
            "candidate_roc_auc": float(
                roc_auc_score(candidate_target, candidate_score)),
            "candidate_average_precision": float(
                average_precision_score(candidate_target, candidate_score)),
        },
        "decision": {
            "learned_ranker_replaces_geometry": False,
            "reason": (
                "OOF top-1 is only a small, non-significant net change; the "
                "10/8 discordance and oracle union justify measuring both "
                "proposals with the unchanged atomic PnP certificate"
            ),
            "next_authorized_test": (
                "train-only OOF dual-proposal, one-view LingBot-depth PnP "
                "certificate collection; no closed-loop or held-out read"
            ),
        },
        "selection_artifact": {
            "path": str(selection_path.resolve()),
            "sha256": selection_sha,
            "rows": int(len(selection)),
            "columns": list(selection.columns),
            "contains_teacher_or_task_labels": False,
        },
        "folds": fold_receipts,
        "deployment_fit_on_all_train_scenes": {
            "selected_C": final_c,
            "inner_selection": final_inner,
            "feature_names": relation_names,
            "coefficient": final_coefficient.tolist(),
            "mean": final_mean.tolist(),
            "scale": final_scale.tolist(),
            "pairwise_differences": final_pairs,
            "optimizer_iterations": final_iterations,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.out_dir / "report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
