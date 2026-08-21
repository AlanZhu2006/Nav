#!/usr/bin/env python3
"""Summarize frozen MicKey shadow predictions without making an SR claim."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from MemNavData.run_mickey_shadow import atomic_json, sha256


def load_predictions(path: Path) -> dict[int, dict]:
    predictions: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            index = int(record["pair_index"])
            if index in predictions:
                raise ValueError(f"duplicate pair index {index} at line {line_number}")
            predictions[index] = record
    if not predictions:
        raise ValueError("prediction file is empty")
    return predictions


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def joined_rows(rows: Sequence[dict[str, str]], predictions: dict[int, dict]) -> list[dict]:
    output = []
    for index, prediction in sorted(predictions.items()):
        if index < 0 or index >= len(rows):
            raise IndexError(f"prediction pair index {index} is out of range")
        row = rows[index]
        for field in ("session_id", "query_relative_path",
                      "candidate_relative_path"):
            if prediction[field] != row[field]:
                raise ValueError(f"pair {index} identity mismatch in {field}")
        learned = prediction["prediction"]
        score = learned.get("solver_support")
        score = float(score) if score is not None else -math.inf
        output.append({
            "pair_index": index,
            "scene": row["scene"],
            "session_id": row["session_id"],
            "candidate_rank": int(row["candidate_rank"]),
            "candidate_label": int(row["candidate_label"]),
            "session_label": int(row["session_label"]),
            "dino_cosine": float(row["dino_cosine"]),
            "fundamental_inliers": int(row["fundamental_inliers"]),
            "score": score,
            "pose_valid": learned["status"] == "ok",
            "latency_ms": float(learned["latency_ms"]),
        })
    return output


def group_sessions(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["session_id"], []).append(row)
    sessions = []
    for session_id, candidates in sorted(groups.items()):
        scene_values = {row["scene"] for row in candidates}
        label_values = {row["session_label"] for row in candidates}
        if len(scene_values) != 1 or len(label_values) != 1:
            raise ValueError(f"session metadata is inconsistent: {session_id}")
        selected = max(
            candidates,
            key=lambda row: (row["score"], row["dino_cosine"],
                             -row["candidate_rank"]))
        sessions.append({
            "session_id": session_id,
            "scene": next(iter(scene_values)),
            "session_label": next(iter(label_values)),
            "max_score": float(selected["score"]),
            "selected_candidate_label": selected["candidate_label"],
            "selected_pair_index": selected["pair_index"],
            "selected_pose_valid": selected["pose_valid"],
        })
    return sessions


def threshold_metrics(sessions: Sequence[dict], threshold: float) -> dict:
    known = [row for row in sessions if row["session_label"] in (0, 1)]
    positives = [row for row in known if row["session_label"] == 1]
    negatives = [row for row in known if row["session_label"] == 0]
    accepted = [row for row in known
                if row["selected_pose_valid"] and row["max_score"] >= threshold]
    correct = [row for row in accepted
               if row["session_label"] == 1
               and row["selected_candidate_label"] == 1]
    false = [row for row in accepted if row not in correct]
    false_negative_accepts = [row for row in accepted
                              if row["session_label"] == 0]
    return {
        "threshold": threshold,
        "known_sessions": len(known),
        "positive_sessions": len(positives),
        "strict_negative_sessions": len(negatives),
        "accepted": len(accepted),
        "correct_accepted": len(correct),
        "incorrect_accepted": len(false),
        "accepted_precision": len(correct) / max(len(accepted), 1),
        "positive_recall": len(correct) / max(len(positives), 1),
        "strict_negative_false_accepts": len(false_negative_accepts),
        "strict_negative_fpr": len(false_negative_accepts) / max(len(negatives), 1),
    }


def choose_threshold(
    sessions: Sequence[dict],
    *,
    minimum_precision: float,
    maximum_negative_fpr: float,
) -> tuple[float, dict]:
    scores = sorted({float(row["max_score"]) for row in sessions
                     if math.isfinite(float(row["max_score"]))}, reverse=True)
    candidates = [math.inf] + scores
    feasible = []
    for threshold in candidates:
        metric = threshold_metrics(sessions, threshold)
        if (metric["accepted_precision"] >= minimum_precision
                and metric["strict_negative_fpr"] <= maximum_negative_fpr):
            feasible.append(metric)
    best = max(
        feasible,
        key=lambda value: (value["correct_accepted"],
                           value["accepted_precision"],
                           value["threshold"]))
    return float(best["threshold"]), best


def scene_oof_threshold(
    sessions: Sequence[dict],
    folds: int,
    *,
    minimum_precision: float,
    maximum_negative_fpr: float,
) -> dict:
    scenes = sorted({row["scene"] for row in sessions})
    if folds < 2 or len(scenes) < folds:
        raise ValueError("scene-OOF requires at least one scene per fold")
    scene_fold = {scene: index % folds for index, scene in enumerate(scenes)}
    heldout_rows = []
    receipts = []
    for fold in range(folds):
        train = [row for row in sessions if scene_fold[row["scene"]] != fold]
        test = [row for row in sessions if scene_fold[row["scene"]] == fold]
        threshold, train_metric = choose_threshold(
            train, minimum_precision=minimum_precision,
            maximum_negative_fpr=maximum_negative_fpr)
        for row in test:
            copied = dict(row)
            copied["oof_threshold"] = threshold
            heldout_rows.append(copied)
        receipts.append({
            "fold": fold,
            "train_scenes": len({row["scene"] for row in train}),
            "test_scenes": len({row["scene"] for row in test}),
            "threshold": threshold,
            "train_metrics": train_metric,
        })
    # OOF thresholds differ by fold, so evaluate decisions explicitly.
    decisions = []
    for row in heldout_rows:
        copied = dict(row)
        copied["accepted"] = (
            row["selected_pose_valid"]
            and row["max_score"] >= row["oof_threshold"])
        copied["correct"] = (
            copied["accepted"] and row["session_label"] == 1
            and row["selected_candidate_label"] == 1)
        decisions.append(copied)
    known = [row for row in decisions if row["session_label"] in (0, 1)]
    accepted = [row for row in known if row["accepted"]]
    correct = [row for row in known if row["correct"]]
    negatives = [row for row in known if row["session_label"] == 0]
    false_negative = [row for row in accepted if row["session_label"] == 0]
    positives = [row for row in known if row["session_label"] == 1]
    return {
        "folds": receipts,
        "aggregate": {
            "known_sessions": len(known),
            "accepted": len(accepted),
            "correct_accepted": len(correct),
            "accepted_precision": len(correct) / max(len(accepted), 1),
            "positive_recall": len(correct) / max(len(positives), 1),
            "strict_negative_false_accepts": len(false_negative),
            "strict_negative_fpr": len(false_negative) / max(len(negatives), 1),
        },
    }


def summarize(rows_csv: Path, predictions_jsonl: Path, folds: int) -> dict:
    source_rows = load_rows(rows_csv)
    joined = joined_rows(source_rows, load_predictions(predictions_jsonl))
    if len(joined) != len(source_rows):
        raise ValueError(
            f"full shadow expected {len(source_rows)} pairs, got {len(joined)}")
    labelled = [row for row in joined if row["candidate_label"] in (0, 1)]
    labels = np.asarray([row["candidate_label"] for row in labelled])
    scores = np.asarray([row["score"] for row in labelled])
    sessions = group_sessions(joined)
    positive_sessions = [row for row in sessions if row["session_label"] == 1]
    all_latencies = np.asarray([row["latency_ms"] for row in joined])
    report = {
        "schema_version": "mickey_shadow_summary_v1",
        "scope": "train40 zero-shot shadow; offline gate only, not navigation SR",
        "inputs": {
            "rows_csv": str(rows_csv.resolve()),
            "rows_csv_sha256": sha256(rows_csv),
            "predictions_jsonl": str(predictions_jsonl.resolve()),
            "predictions_jsonl_sha256": sha256(predictions_jsonl),
        },
        "coverage": {
            "pairs": len(joined),
            "sessions": len(sessions),
            "scenes": len({row["scene"] for row in sessions}),
            "valid_pose_pairs": sum(row["pose_valid"] for row in joined),
        },
        "candidate_support": {
            "labelled_pairs": len(labelled),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        },
        "ranking_without_abstention": {
            "positive_sessions": len(positive_sessions),
            "correct_top1": sum(
                row["selected_candidate_label"] == 1
                for row in positive_sessions),
        },
        "oracle_train_threshold_diagnostic": choose_threshold(
            sessions, minimum_precision=0.9313,
            maximum_negative_fpr=0.0275)[1],
        "scene_grouped_oof_threshold": scene_oof_threshold(
            sessions, folds, minimum_precision=0.9313,
            maximum_negative_fpr=0.0275),
        "latency_ms_per_pair": {
            "median": float(np.median(all_latencies)),
            "p95": float(np.quantile(all_latencies, 0.95)),
            "maximum": float(np.max(all_latencies)),
        },
        "limitations": [
            "candidate labels and thresholds are train40-only diagnostics",
            "pair support and anchor top-1 are not bearing accuracy",
            "offline gates cannot establish closed-loop SR non-inferiority",
        ],
    }
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = summarize(args.rows_csv, args.predictions_jsonl, args.folds)
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
