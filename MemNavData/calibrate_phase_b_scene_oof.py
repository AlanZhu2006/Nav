#!/usr/bin/env python3
"""Scene-grouped out-of-fold calibration for the Phase-B localizer.

Stage-2 (SR_ROADMAP_20260808.md §5.2) showed the model ranks better than
max-DINO per candidate (dev AUC 0.9535 vs 0.9103) but loses the deployment
decision, because its probability scale does not transfer across scenes:

    train-optimal threshold 0.397  ->  dev-optimal 0.807
    transfer loss 10.9 pp          (max-DINO loses only 1.8 pp)
    strict-no-match sessions score 0.044 on train, 0.204 on dev

The cause is that the deployed threshold was picked on *in-sample* train
predictions, where the model has already seen every scene.  This script
re-derives the operating point honestly: it retrains the same architecture
under a scene-grouped K-fold, so every prediction it calibrates on comes from
a scene the fold model never saw, then fits an isotonic map and a threshold on
those out-of-fold scores only.

The development split is read exactly once, at the end, to measure transfer
loss.  It is never used to fit anything: development was already spent by the
trainer's own evaluation and by the Stage-2 decision, so it can only serve as
a read-out here, and the report says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MemNavData.train_lingbot_native_localizer import (  # noqa: E402
    build_feature_matrix,
    pack_exact_sessions,
    predict,
    train_model,
)

POSITIVE, NEGATIVE = 0.5, 0.2
HIDDEN_DIM, DROPOUT = 64, 0.10
LEARNING_RATE, WEIGHT_DECAY = 3e-4, 1e-4
BATCH_SIZE, EPOCHS, PATIENCE = 32, 200, 25
POSE_WEIGHT, POSE_TAIL_WEIGHT, POSE_TAIL_FRACTION = 1.0, 0.5, 0.2


def load_packed(rows: pd.DataFrame, mean=None, scale=None):
    """Feature matrix plus the session-level arrays pack_exact_sessions wants."""
    frame = rows.sort_values(
        ["session_id", "dino_cosine", "candidate_frame"],
        ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    raw, names, predicted_xy, target_xy = build_feature_matrix(frame)
    if mean is None:
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = ((raw - mean) / scale).astype(np.float32)
    return frame, normalized, names, predicted_xy, target_xy, mean, scale


def pack(frame, normalized, predicted_xy, target_xy, index=None):
    if index is None:
        index = np.ones(len(frame), dtype=bool)
    return pack_exact_sessions(
        normalized[index],
        frame.loc[index, "session_id"].to_numpy(dtype=str),
        frame.loc[index, "scene"].to_numpy(dtype=str),
        frame.loc[index, "teacher_covis"].to_numpy(dtype=np.float64),
        predicted_xy[index], target_xy[index],
        frame.loc[index, "session_has_positive"].to_numpy(dtype=bool),
        frame.loc[index, "session_is_strict_no_match"].to_numpy(dtype=bool),
        positive_threshold=POSITIVE, negative_threshold=NEGATIVE)


def session_scores(model, packed, device):
    """Deployment score per session: (1 - P(no match)) * max candidate validity."""
    prediction = predict(model, packed, device)
    validity = np.asarray(prediction.candidate_validity)
    no_match = np.asarray(prediction.no_match_probability).reshape(-1)
    return (1.0 - no_match) * validity.max(axis=1)


def isotonic_fit(scores: np.ndarray, labels: np.ndarray):
    """Pool-adjacent-violators isotonic regression (no sklearn dependency)."""
    order = np.argsort(scores, kind="mergesort")
    x, y = scores[order].astype(float), labels[order].astype(float)
    values, weights = list(y), [1.0] * len(y)
    knots = list(x)
    index = 0
    while index < len(values) - 1:
        if values[index] <= values[index + 1]:
            index += 1
            continue
        total = weights[index] + weights[index + 1]
        merged = (values[index] * weights[index]
                  + values[index + 1] * weights[index + 1]) / total
        values[index:index + 2] = [merged]
        weights[index:index + 2] = [total]
        knots[index:index + 2] = [knots[index + 1]]
        index = max(index - 1, 0)
    return np.asarray(knots), np.asarray(values)


def isotonic_apply(knots, values, scores):
    return np.interp(np.asarray(scores, dtype=float), knots, values,
                     left=values[0], right=values[-1])


def best_threshold(scores, labels):
    unique = sorted(set(float(v) for v in scores))
    grid = ([-np.inf]
            + [0.5 * (a + b) for a, b in zip(unique, unique[1:])]
            + [np.inf])
    scored = [(float(np.mean((np.asarray(scores) >= t) == labels)), t)
              for t in grid]
    accuracy, threshold = max(scored)
    return float(threshold), float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    device = torch.device("cpu")

    train_rows = pd.read_csv(args.train_rows)
    frame, normalized, names, predicted_xy, target_xy, mean, scale = load_packed(
        train_rows)
    scenes = np.array(sorted(frame["scene"].unique()))
    rng = np.random.default_rng(args.seed)
    shuffled = scenes[rng.permutation(len(scenes))]
    folds = np.array_split(shuffled, args.folds)

    oof_score, oof_label, oof_scene = [], [], []
    for number, held_out in enumerate(folds):
        test_mask = frame["scene"].isin(held_out).to_numpy()
        fit = pack(frame, normalized, predicted_xy, target_xy, ~test_mask)
        held = pack(frame, normalized, predicted_xy, target_xy, test_mask)
        model, _epoch, _metrics = train_model(
            fit, None, input_dim=normalized.shape[1], hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
            epochs=EPOCHS, patience=PATIENCE, pose_weight=POSE_WEIGHT,
            pose_tail_weight=POSE_TAIL_WEIGHT,
            pose_tail_fraction=POSE_TAIL_FRACTION,
            seed=args.seed + number, device=device,
            positive_threshold=POSITIVE)
        scores = session_scores(model, held, device)
        # PackedExactSessions carries the match label as selected_match_target
        # and marks non-ambiguous sessions with no_match_supervision_mask.
        labels = held.selected_match_target.numpy() > 0.5
        strict = held.no_match_supervision_mask.numpy().astype(bool)
        oof_score.extend(scores[strict])
        oof_label.extend(labels[strict])
        oof_scene.extend(np.asarray(held.scenes)[strict].tolist())
        print(f"[fold {number}] scenes={len(held_out)} "
              f"sessions={int(strict.sum())}", flush=True)

    oof_score = np.asarray(oof_score, dtype=float)
    oof_label = np.asarray(oof_label, dtype=bool)

    # In-sample reference: the full-train model scoring its own training scenes.
    full = pack(frame, normalized, predicted_xy, target_xy)
    full_model, _e, _m = train_model(
        full, None, input_dim=normalized.shape[1], hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS,
        patience=PATIENCE, pose_weight=POSE_WEIGHT,
        pose_tail_weight=POSE_TAIL_WEIGHT,
        pose_tail_fraction=POSE_TAIL_FRACTION, seed=args.seed,
        device=device, positive_threshold=POSITIVE)
    in_sample = session_scores(full_model, full, device)
    full_label = full.selected_match_target.numpy() > 0.5
    full_strict = full.no_match_supervision_mask.numpy().astype(bool)
    in_threshold, in_accuracy = best_threshold(
        in_sample[full_strict], full_label[full_strict])
    oof_threshold, oof_accuracy = best_threshold(oof_score, oof_label)
    knots, values = isotonic_fit(oof_score, oof_label)
    calibrated_oof = isotonic_apply(knots, values, oof_score)
    calibrated_threshold, calibrated_accuracy = best_threshold(
        calibrated_oof, oof_label)

    # Development is read once, only to measure transfer.
    dev_rows = pd.read_csv(args.development_rows)
    dev_frame, dev_normalized, _n, dev_pred, dev_target, _m, _s = load_packed(
        dev_rows, mean, scale)
    dev_packed = pack(dev_frame, dev_normalized, dev_pred, dev_target)
    dev_scores = session_scores(full_model, dev_packed, device)
    dev_label = dev_packed.selected_match_target.numpy() > 0.5
    dev_strict = dev_packed.no_match_supervision_mask.numpy().astype(bool)
    dev_scores, dev_label = dev_scores[dev_strict], dev_label[dev_strict]
    dev_oracle_threshold, dev_oracle_accuracy = best_threshold(
        dev_scores, dev_label)
    dev_calibrated = isotonic_apply(knots, values, dev_scores)

    def accuracy_at(scores, threshold):
        return float(np.mean((np.asarray(scores) >= threshold) == dev_label))

    report = {
        "scope": ("scene-grouped OOF calibration of the Phase-B operating "
                  "point; development is a read-out only and was already "
                  "spent, so this is diagnostic, not model selection"),
        "folds": args.folds,
        "train_scenes": int(len(scenes)),
        "oof_sessions": int(len(oof_score)),
        "development_sessions": int(len(dev_scores)),
        "score_distribution": {
            "oof_positive_mean": float(oof_score[oof_label].mean()),
            "oof_negative_mean": float(oof_score[~oof_label].mean()),
            "in_sample_positive_mean": float(
                in_sample[full_strict][full_label[full_strict]].mean()),
            "in_sample_negative_mean": float(
                in_sample[full_strict][~full_label[full_strict]].mean()),
            "development_positive_mean": float(dev_scores[dev_label].mean()),
            "development_negative_mean": float(dev_scores[~dev_label].mean()),
        },
        "operating_points": {
            "in_sample_threshold": in_threshold,
            "in_sample_accuracy_on_train": in_accuracy,
            "oof_threshold": oof_threshold,
            "oof_accuracy": oof_accuracy,
            "isotonic_threshold": calibrated_threshold,
            "isotonic_accuracy_on_oof": calibrated_accuracy,
            "development_oracle_threshold": dev_oracle_threshold,
        },
        "development_transfer": {
            "oracle_ceiling": dev_oracle_accuracy,
            "in_sample_threshold": accuracy_at(dev_scores, in_threshold),
            "oof_threshold": accuracy_at(dev_scores, oof_threshold),
            "isotonic_calibrated": accuracy_at(
                dev_calibrated, calibrated_threshold),
        },
    }
    transfer = report["development_transfer"]
    for name in ("in_sample_threshold", "oof_threshold", "isotonic_calibrated"):
        transfer[f"{name}_loss_pp"] = round(
            100.0 * (transfer["oracle_ceiling"] - transfer[name]), 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
