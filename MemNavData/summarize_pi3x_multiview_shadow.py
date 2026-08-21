#!/usr/bin/env python3
"""Scene-grouped audit of Pi3X multi-view relocalization shadow outputs.

The learned logistic support head here is a feasibility probe over frozen Pi3X
evidence.  It is not the final end-to-end relocalizer and cannot authorize a
closed-loop run.  Every threshold is selected using training scenes only; the
reported prediction for each scene is from an outer model that never saw it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OBSERVABLE_FEATURES = (
    "dino_cosine",
    "best_view_f1_05cm",
    "best_view_f1_10cm",
    "best_view_f1_20cm",
    "best_view_f1_50cm",
    "union_goal_overlap_05cm",
    "union_goal_overlap_10cm",
    "union_goal_overlap_20cm",
    "union_goal_overlap_50cm",
    "union_history_overlap_05cm",
    "union_history_overlap_10cm",
    "union_history_overlap_20cm",
    "union_history_overlap_50cm",
    "goal_to_history_q10_m",
    "goal_to_history_q25_m",
    "goal_to_history_q50_m",
    "history_to_goal_q25_m",
    "goal_confidence_median",
    "current_confidence_median",
    "history_confidence_median",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _matrix(rows: Sequence[dict[str, Any]], indices: Iterable[int]) -> np.ndarray:
    return np.asarray([
        [float(rows[index][feature]) for feature in OBSERVABLE_FEATURES]
        for index in indices
    ], dtype=np.float64)


def _new_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        )),
    ])


def _fit(rows: Sequence[dict[str, Any]], scenes: set[str],
         target_key: str) -> Pipeline:
    indices = [
        index for index, row in enumerate(rows)
        if row["scene"] in scenes and row[target_key] in (0, 1)
    ]
    labels = np.asarray([rows[index][target_key] for index in indices])
    if len(set(labels.tolist())) != 2:
        raise ValueError("candidate support training split lacks both classes")
    model = _new_model()
    model.fit(_matrix(rows, indices), labels)
    return model


def _predict(model: Pipeline, rows: Sequence[dict[str, Any]],
             scenes: set[str]) -> dict[int, float]:
    indices = [index for index, row in enumerate(rows) if row["scene"] in scenes]
    scores = model.predict_proba(_matrix(rows, indices))[:, 1]
    return {index: float(score) for index, score in zip(indices, scores)}


def _session_picks(
    rows: Sequence[dict[str, Any]],
    scores: dict[int, float],
    scenes: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, score in scores.items():
        if scenes is None or rows[index]["scene"] in scenes:
            grouped[rows[index]["session_id"]].append(index)
    picks = []
    for session_id, indices in grouped.items():
        selected = max(
            indices,
            key=lambda index: (scores[index], -rows[index]["candidate_rank"]),
        )
        row = rows[selected]
        picks.append({
            "session_id": session_id,
            "scene": row["scene"],
            "session_label": row["session_label"],
            "selected_row_index": row["row_index"],
            "selected_candidate_rank": row["candidate_rank"],
            "selected_candidate_label": row["candidate_label"],
            "selected_navigation_action_label": row["navigation_action_label"],
            "score": scores[selected],
            "bearing_error_deg": row["goal_bearing_error_deg_reporting_only"],
        })
    return picks


def evaluate_picks(picks: Sequence[dict[str, Any]], threshold: float,
                   correctness_key: str = "selected_candidate_label") -> dict[str, Any]:
    known = [pick for pick in picks if pick["session_label"] in (0, 1)]
    positive = [pick for pick in known if pick["session_label"] == 1]
    negative = [pick for pick in known if pick["session_label"] == 0]
    ambiguous = [pick for pick in picks if pick["session_label"] == -1]
    accepted = [pick for pick in known if pick["score"] >= threshold]
    correct = [
        pick for pick in accepted
        if pick["session_label"] == 1 and pick[correctness_key] == 1
    ]
    wrong_positive = [
        pick for pick in accepted
        if pick["session_label"] == 1 and pick[correctness_key] != 1
    ]
    false_negative_accepts = [
        pick for pick in accepted if pick["session_label"] == 0
    ]
    bearing = np.asarray([
        pick["bearing_error_deg"] for pick in accepted
        if math.isfinite(float(pick["bearing_error_deg"]))
    ])
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "sessions": len(picks),
        "positive_sessions": len(positive),
        "strict_negative_sessions": len(negative),
        "ambiguous_sessions_excluded": len(ambiguous),
        "accepted_known_sessions": len(accepted),
        "correct_positive_accepts": len(correct),
        "wrong_candidate_accepts_in_positive_sessions": len(wrong_positive),
        "strict_negative_false_accepts": len(false_negative_accepts),
        "accepted_precision": len(correct) / len(accepted) if accepted else 1.0,
        "positive_session_recall": len(correct) / len(positive) if positive else math.nan,
        "strict_negative_fpr": (
            len(false_negative_accepts) / len(negative) if negative else math.nan
        ),
    }
    if len(bearing):
        result["accepted_bearing_median_deg"] = float(np.median(bearing))
        for angle in (15, 30, 45, 90):
            result[f"accepted_bearing_within_{angle}deg"] = int((bearing <= angle).sum())
        result["accepted_bearing_catastrophic_gt90deg"] = int((bearing > 90).sum())
    return result


def choose_threshold(
    picks: Sequence[dict[str, Any]],
    *,
    minimum_precision: float,
    maximum_fpr: float,
    correctness_key: str = "selected_candidate_label",
) -> tuple[float, dict[str, Any]]:
    candidates = [math.inf, *sorted({float(pick["score"]) for pick in picks}, reverse=True)]
    feasible = []
    for threshold in candidates:
        metrics = evaluate_picks(picks, threshold, correctness_key)
        if (metrics["accepted_precision"] >= minimum_precision
                and metrics["strict_negative_fpr"] <= maximum_fpr):
            feasible.append((
                metrics["correct_positive_accepts"],
                -metrics["strict_negative_false_accepts"],
                -metrics["wrong_candidate_accepts_in_positive_sessions"],
                -float(threshold),
                threshold,
                metrics,
            ))
    if not feasible:
        metrics = evaluate_picks(picks, math.inf, correctness_key)
        return math.inf, metrics
    best = max(feasible)
    return float(best[-2]), best[-1]


def _nested_scene_oof(
    rows: Sequence[dict[str, Any]],
    *,
    outer_splits: int,
    inner_splits: int,
    minimum_precision: float,
    maximum_fpr: float,
    target_key: str,
    correctness_key: str,
) -> tuple[dict[int, float], list[dict[str, Any]], list[dict[str, Any]]]:
    scenes = np.asarray(sorted({row["scene"] for row in rows}))
    if len(scenes) < 4:
        raise ValueError("nested scene OOF requires at least four scenes")
    outer = KFold(n_splits=min(outer_splits, len(scenes)), shuffle=True, random_state=0)
    oof_scores: dict[int, float] = {}
    oof_picks: list[dict[str, Any]] = []
    folds = []
    for fold_index, (train_scene_indices, test_scene_indices) in enumerate(outer.split(scenes)):
        train_scenes = set(scenes[train_scene_indices].tolist())
        test_scenes = set(scenes[test_scene_indices].tolist())
        inner_scene_array = np.asarray(sorted(train_scenes))
        inner = KFold(
            n_splits=min(inner_splits, len(inner_scene_array)),
            shuffle=True,
            random_state=fold_index + 101,
        )
        inner_scores: dict[int, float] = {}
        for inner_train_indices, inner_validation_indices in inner.split(inner_scene_array):
            inner_train_scenes = set(inner_scene_array[inner_train_indices].tolist())
            inner_validation_scenes = set(inner_scene_array[inner_validation_indices].tolist())
            inner_model = _fit(rows, inner_train_scenes, target_key)
            inner_scores.update(_predict(inner_model, rows, inner_validation_scenes))
        inner_picks = _session_picks(rows, inner_scores, train_scenes)
        threshold, calibration = choose_threshold(
            inner_picks,
            minimum_precision=minimum_precision,
            maximum_fpr=maximum_fpr,
            correctness_key=correctness_key,
        )
        outer_model = _fit(rows, train_scenes, target_key)
        test_scores = _predict(outer_model, rows, test_scenes)
        oof_scores.update(test_scores)
        test_picks = _session_picks(rows, test_scores, test_scenes)
        for pick in test_picks:
            pick["outer_fold"] = fold_index
            pick["fold_threshold"] = threshold
            pick["accepted"] = pick["score"] >= threshold
        oof_picks.extend(test_picks)
        folds.append({
            "fold": fold_index,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "threshold_from_inner_scene_oof": threshold,
            "inner_calibration": calibration,
            "test": evaluate_picks(test_picks, threshold, correctness_key),
        })
    return oof_scores, oof_picks, folds


def _aggregate_folded_picks(picks: Sequence[dict[str, Any]],
                            correctness_key: str) -> dict[str, Any]:
    # Each pick carries its own fold-specific threshold.  Re-express acceptance
    # through a binary score so the common evaluator can aggregate exactly.
    normalized = []
    for pick in picks:
        copied = dict(pick)
        copied["score"] = 1.0 if pick["accepted"] else 0.0
        normalized.append(copied)
    return evaluate_picks(normalized, 0.5, correctness_key)


def _raw_top1(rows: Sequence[dict[str, Any]], feature: str,
              direction: float = 1.0,
              correctness_key: str = "selected_candidate_label") -> dict[str, Any]:
    scores = {index: direction * float(row[feature]) for index, row in enumerate(rows)}
    picks = _session_picks(rows, scores)
    positives = [pick for pick in picks if pick["session_label"] == 1]
    return {
        "feature": feature,
        "positive_session_top1": sum(
            pick[correctness_key] == 1 for pick in positives
        ),
        "positive_sessions": len(positives),
    }


def _navigation_ceiling(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    positive = [
        group for group in grouped.values() if group[0]["session_label"] == 1
    ]
    return {
        "positive_sessions": len(positive),
        "any_top8_bearing_within_30deg": sum(
            any(row["navigation_action_label"] == 1 for row in group)
            for group in positive
        ),
        "all_top8_bearings_wrong": sum(
            all(row["navigation_action_label"] != 1 for row in group)
            for group in positive
        ),
    }


def _load(args: argparse.Namespace) -> list[dict[str, Any]]:
    with args.rows_csv.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    if args.expected_rows_sha256 and _sha256(args.rows_csv) != args.expected_rows_sha256:
        raise ValueError("source rows SHA mismatch")
    shadow = [json.loads(line) for line in args.shadow_jsonl.read_text().splitlines() if line]
    if args.expected_rows is not None and len(shadow) != args.expected_rows:
        raise ValueError(f"shadow has {len(shadow)} rows, expected {args.expected_rows}")
    seen = set()
    rows = []
    for prediction in shadow:
        row_index = int(prediction["row_index"])
        if row_index in seen or not 0 <= row_index < len(source):
            raise ValueError(f"invalid or duplicate row_index {row_index}")
        seen.add(row_index)
        original = source[row_index]
        if original["scene"] != prediction["scene"]:
            raise ValueError(f"scene mismatch at row {row_index}")
        row = dict(prediction)
        row.update({
            "row_index": row_index,
            "session_id": original["session_id"],
            "scene": original["scene"],
            "candidate_rank": int(original["candidate_rank"]),
            "candidate_label": int(original["candidate_label"]),
            "session_label": int(original["session_label"]),
            "dino_cosine": float(original["dino_cosine"]),
        })
        bearing_error = float(row["goal_bearing_error_deg_reporting_only"])
        if row["session_label"] == 1:
            row["navigation_action_label"] = int(
                math.isfinite(bearing_error) and bearing_error <= 30.0
            )
        elif row["session_label"] == 0:
            # Novel/strict-no-match remains a reject target even when one
            # zero-shot bearing happens to be correct on the training scene.
            row["navigation_action_label"] = 0
        else:
            row["navigation_action_label"] = -1
        for feature in OBSERVABLE_FEATURES:
            if feature not in row or not math.isfinite(float(row[feature])):
                raise ValueError(f"non-finite observable {feature} at row {row_index}")
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load(args)
    support_scores, support_picks, support_folds = _nested_scene_oof(
        rows,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        minimum_precision=args.minimum_precision,
        maximum_fpr=args.maximum_fpr,
        target_key="candidate_label",
        correctness_key="selected_candidate_label",
    )
    navigation_scores, navigation_picks, navigation_folds = _nested_scene_oof(
        rows,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        minimum_precision=args.minimum_precision,
        maximum_fpr=args.maximum_fpr,
        target_key="navigation_action_label",
        correctness_key="selected_navigation_action_label",
    )
    known_indices = [
        index for index, row in enumerate(rows) if row["candidate_label"] in (0, 1)
    ]
    labels = np.asarray([rows[index]["candidate_label"] for index in known_indices])
    learned_support = np.asarray([support_scores[index] for index in known_indices])
    navigation_indices = [
        index for index, row in enumerate(rows)
        if row["navigation_action_label"] in (0, 1)
    ]
    navigation_labels = np.asarray([
        rows[index]["navigation_action_label"] for index in navigation_indices
    ])
    learned_navigation = np.asarray([
        navigation_scores[index] for index in navigation_indices
    ])
    raw_overlap = np.asarray([
        rows[index]["best_view_f1_20cm"] for index in known_indices
    ])
    dino = np.asarray([rows[index]["dino_cosine"] for index in known_indices])
    summary = {
        "schema_version": 1,
        "status": "offline_shadow_only_not_closed_loop_authority",
        "rows": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "sessions": len({row["session_id"] for row in rows}),
        "feature_contract": list(OBSERVABLE_FEATURES),
        "candidate_metrics": {
            "learned_support_nested_scene_oof_roc_auc": float(
                roc_auc_score(labels, learned_support)
            ),
            "learned_support_nested_scene_oof_average_precision": float(
                average_precision_score(labels, learned_support)
            ),
            "learned_navigation_nested_scene_oof_roc_auc": float(
                roc_auc_score(navigation_labels, learned_navigation)
            ),
            "learned_navigation_nested_scene_oof_average_precision": float(
                average_precision_score(navigation_labels, learned_navigation)
            ),
            "raw_pi3x_overlap_roc_auc": float(roc_auc_score(labels, raw_overlap)),
            "raw_dino_roc_auc": float(roc_auc_score(labels, dino)),
        },
        "session_top1": {
            "learned_support_nested_scene_oof": _raw_top1(
                [dict(row, learned_support_score=support_scores[index]) for index, row in enumerate(rows)],
                "learned_support_score",
            ),
            "learned_navigation_reliability_nested_scene_oof": _raw_top1(
                [dict(row, learned_navigation_score=navigation_scores[index]) for index, row in enumerate(rows)],
                "learned_navigation_score",
                correctness_key="selected_navigation_action_label",
            ),
            "raw_pi3x_overlap": _raw_top1(rows, "best_view_f1_20cm"),
            "raw_dino": _raw_top1(rows, "dino_cosine"),
            "raw_pi3x_overlap_navigation": _raw_top1(
                rows,
                "best_view_f1_20cm",
                correctness_key="selected_navigation_action_label",
            ),
            "raw_dino_navigation": _raw_top1(
                rows,
                "dino_cosine",
                correctness_key="selected_navigation_action_label",
            ),
        },
        "navigation_top8_ceiling": _navigation_ceiling(rows),
        "support_nested_scene_oof_activation": _aggregate_folded_picks(
            support_picks, "selected_candidate_label"
        ),
        "navigation_nested_scene_oof_activation": _aggregate_folded_picks(
            navigation_picks, "selected_navigation_action_label"
        ),
        "support_outer_folds": support_folds,
        "navigation_outer_folds": navigation_folds,
        "frozen_targets": {
            "minimum_precision": args.minimum_precision,
            "maximum_strict_negative_fpr": args.maximum_fpr,
            "current_certificate_precision_reference": 0.9313,
            "current_certificate_recall_reference": 0.7974,
        },
        "inputs": {
            "shadow_jsonl": str(args.shadow_jsonl),
            "shadow_jsonl_sha256": _sha256(args.shadow_jsonl),
            "rows_csv": str(args.rows_csv),
            "rows_csv_sha256": _sha256(args.rows_csv),
        },
    }
    prediction_rows = []
    support_pick_by_row = {
        pick["selected_row_index"]: pick for pick in support_picks
    }
    navigation_pick_by_row = {
        pick["selected_row_index"]: pick for pick in navigation_picks
    }
    for index, row in enumerate(rows):
        support_pick = support_pick_by_row.get(row["row_index"])
        navigation_pick = navigation_pick_by_row.get(row["row_index"])
        prediction_rows.append({
            "row_index": row["row_index"],
            "scene": row["scene"],
            "session_id": row["session_id"],
            "candidate_rank": row["candidate_rank"],
            "candidate_label_reporting_only": row["candidate_label"],
            "session_label_reporting_only": row["session_label"],
            "navigation_action_label_reporting_only": row["navigation_action_label"],
            "learned_support_oof_score": support_scores[index],
            "learned_navigation_oof_score": navigation_scores[index],
            "support_selected": support_pick is not None,
            "support_accepted": bool(support_pick and support_pick["accepted"]),
            "navigation_selected": navigation_pick is not None,
            "navigation_accepted": bool(
                navigation_pick and navigation_pick["accepted"]
            ),
            "outer_fold": (
                navigation_pick["outer_fold"] if navigation_pick else ""
            ),
            "navigation_fold_threshold": (
                navigation_pick["fold_threshold"] if navigation_pick else ""
            ),
            "bearing_error_deg_reporting_only": row["goal_bearing_error_deg_reporting_only"],
        })
    _atomic_json(args.output_summary, summary)
    _atomic_csv(args.output_predictions, prediction_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-jsonl", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-rows-sha256")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--maximum-fpr", type=float, default=0.0275)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "summary": result["status"],
        "rows": result["rows"],
        "scenes": result["scenes"],
        "activation": result["navigation_nested_scene_oof_activation"],
    }, sort_keys=True))
