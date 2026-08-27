#!/usr/bin/env python3
"""GLP Stage-2 decision: does the trained Phase-B likelihood beat max-DINO?

Stage 1 (diag_goal_posterior_teacher_replay.py) showed that a calibrated
posterior over DINO-only evidence merely matches the max-DINO + train-threshold
baseline: the framework was harmless but the evidence carried nothing extra to
aggregate.  The gate for Stage 2 was therefore moved onto the Phase-B features.

This replays the same protocol on the audited development split, with three
arms sharing one label authority (the teacher) and one metric definition:

  max_dino       session score = max candidate DINO cosine, threshold picked
                 on the audited TRAIN split only            [Stage-1 baseline]
  phase_b_model  the three-seed ensemble's own set probability
  glp_posterior  the Phase-B per-candidate validity fed into the GLP posterior
                 as a calibrated log-likelihood-ratio, with an explicit
                 unmodelled hypothesis

Reported metrics are identical to Stage 1: joint localization (match decision
AND correct top-1 when a match exists), match accuracy, conditional
candidate recall@1, and match AUC.  Diagnostic only; the development split is
already spent by the trainer's own evaluation, so this cannot become a
model-selection signal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MemNavData.goal_posterior import GoalPosterior  # noqa: E402
from MemNavData.phase_b_feature_schema import (  # noqa: E402
    validate_checkpoint_metadata,
)
from MemNavData.train_lingbot_native_localizer import (  # noqa: E402
    LingBotNativeLocalizer,
    build_feature_matrix,
)

POSITIVE, NEGATIVE = 0.5, 0.2


def session_class(maximum: float) -> str:
    if maximum >= POSITIVE:
        return "pos"
    if maximum <= NEGATIVE:
        return "neg"
    return "amb"


def auc(scores, labels) -> float:
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    y = np.asarray(labels, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[y].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def score_arm(rows, predicts, top1) -> dict:
    joint, match, recall = [], [], []
    for row, predicted, correct in zip(rows, predicts, top1):
        match.append(bool(predicted) == row["has_match"])
        joint.append((row["has_match"] and predicted and correct)
                     or ((not row["has_match"]) and (not predicted)))
        if row["has_match"]:
            recall.append(bool(correct))
    return {
        "sessions": len(rows),
        "joint_localization_accuracy": float(np.mean(joint)),
        "match_accuracy": float(np.mean(match)),
        "conditional_candidate_recall_at_1": (
            float(np.mean(recall)) if recall else float("nan")),
    }


def load_sessions(rows: pd.DataFrame, feature_names, mean, scale):
    """One packed record per session, ordered exactly as the trainer
    orders candidates (descending DINO, then frame)."""
    sessions = []
    for session_id, group in rows.groupby("session_id", sort=True):
        group = group.sort_values(
            ["dino_cosine", "candidate_frame"],
            ascending=[False, True], kind="mergesort")
        # Reuse the trainer's own feature builder: pose channels and the
        # metric-scale one-hots are derived, not raw CSV columns, and a
        # re-implementation here could silently drift from the checkpoint ABI.
        raw, names, _predicted_xy, _target_xy = build_feature_matrix(group)
        if list(names) != list(feature_names):
            raise RuntimeError("feature order differs from the checkpoint ABI")
        if not np.isfinite(raw).all():
            raise RuntimeError(f"non-finite features in {session_id}")
        sessions.append({
            "session_id": session_id,
            "features": ((raw - mean) / scale).astype(np.float32),
            "covis": group["teacher_covis"].to_numpy(dtype=float),
            "dino": group["dino_cosine"].to_numpy(dtype=float),
            "frames": group["candidate_frame"].to_numpy(dtype=int),
            "session_max": float(group["session_max_covis"].iloc[0]),
        })
    return sessions


def ensemble_probabilities(checkpoint, sessions):
    """Mean over seeds of (per-candidate validity, no-match probability)."""
    width = max(len(s["features"]) for s in sessions)
    features = np.zeros((len(sessions), width, checkpoint["input_dim"]),
                        dtype=np.float32)
    mask = np.zeros((len(sessions), width), dtype=bool)
    for index, session in enumerate(sessions):
        count = len(session["features"])
        features[index, :count] = session["features"]
        mask[index, :count] = True
    feature_tensor = torch.from_numpy(features)
    mask_tensor = torch.from_numpy(mask)
    validity_sum = np.zeros((len(sessions), width), dtype=np.float64)
    no_match_sum = np.zeros(len(sessions), dtype=np.float64)
    rank_sum = np.zeros((len(sessions), width), dtype=np.float64)
    states = checkpoint["states"]
    for state in states:
        model = LingBotNativeLocalizer(
            checkpoint["input_dim"],
            hidden_dim=int(checkpoint["config"].get("hidden_dim", 64)),
            dropout=0.0)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits, no_match_logit, _residual, _log_var = model(
                feature_tensor, mask_tensor)
            validity_sum += (torch.sigmoid(logits).numpy()
                             * mask.astype(np.float64))
            rank_sum += torch.softmax(logits, dim=-1).numpy() * mask
            no_match_sum += torch.sigmoid(no_match_logit).numpy().reshape(-1)
    return (validity_sum / len(states), rank_sum / len(states),
            no_match_sum / len(states), mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unmodelled-weight", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
    validate_checkpoint_metadata(checkpoint,
                                 require_deployment_input_contract=True)
    feature_names = list(checkpoint["feature_names"])
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float64)

    train_rows = pd.read_csv(args.train_rows)
    dev_rows = pd.read_csv(args.development_rows)
    dev = [s for s in load_sessions(dev_rows, feature_names, mean, scale)
           if session_class(s["session_max"]) in ("pos", "neg")]
    records = [{"session_id": s["session_id"],
                "has_match": session_class(s["session_max"]) == "pos"}
               for s in dev]

    # --- arm 1: max-DINO with a threshold selected on TRAIN only ------------
    train_sessions = train_rows.groupby("session_id").agg(
        max_dino=("dino_cosine", "max"),
        session_max=("session_max_covis", "first")).reset_index()
    train_sessions["has_match"] = train_sessions["session_max"].map(
        lambda m: session_class(m) == "pos")
    train_sessions = train_sessions[
        train_sessions["session_max"].map(session_class) != "amb"]
    candidates = sorted(train_sessions["max_dino"].unique())
    grid = ([-np.inf]
            + [0.5 * (a + b) for a, b in zip(candidates, candidates[1:])]
            + [np.inf])
    best = max(grid, key=lambda t: float(
        ((train_sessions["max_dino"] >= t) == train_sessions["has_match"]).mean()))
    dino_predicts = [float(s["dino"].max()) >= best for s in dev]
    dino_top1 = [bool(s["covis"][int(np.argmax(s["dino"]))] >= POSITIVE)
                 for s in dev]
    dino_arm = score_arm(records, dino_predicts, dino_top1)
    dino_arm["threshold_selected_on_train"] = float(best)
    dino_arm["match_auc"] = auc([float(s["dino"].max()) for s in dev],
                                [r["has_match"] for r in records])

    # --- arm 2: the trained Phase-B ensemble --------------------------------
    validity, rank, no_match, mask = ensemble_probabilities(checkpoint, dev)
    usable = (1.0 - no_match) * validity.max(axis=1)
    train_all = load_sessions(train_rows, feature_names, mean, scale)
    train_strict = [t for t in train_all
                    if session_class(t["session_max"]) in ("pos", "neg")]
    train_truth = [session_class(t["session_max"]) == "pos"
                   for t in train_strict]
    tv, _tr, tn, _tm = ensemble_probabilities(checkpoint, train_strict)
    train_usable = (1.0 - tn) * tv.max(axis=1)

    def tuned_threshold(scores, truth):
        values = sorted(set(float(v) for v in scores))
        grid = ([-np.inf]
                + [0.5 * (a + b) for a, b in zip(values, values[1:])]
                + [np.inf])
        return max(grid, key=lambda t: float(
            np.mean([(s >= t) == y for s, y in zip(scores, truth)])))

    model_threshold = tuned_threshold(train_usable, train_truth)
    model_predicts = [bool(u >= model_threshold) for u in usable]
    model_top1 = []
    for index, session in enumerate(dev):
        count = len(session["covis"])
        choice = int(np.argmax(rank[index, :count]))
        model_top1.append(bool(session["covis"][choice] >= POSITIVE))
    model_arm = score_arm(records, model_predicts, model_top1)
    model_arm["match_auc"] = auc(usable, [r["has_match"] for r in records])
    model_arm["threshold_selected_on_train"] = float(model_threshold)

    # --- arm 3: GLP posterior over the Phase-B evidence ----------------------
    posterior_scores, posterior_top1 = [], []
    for index, session in enumerate(dev):
        count = len(session["covis"])
        posterior = GoalPosterior(
            unmodeled_log_weight=math.log(args.unmodelled_weight),
            cluster_gap=16)
        for candidate in range(count):
            probability = float(np.clip(validity[index, candidate], 1e-6,
                                        1 - 1e-6))
            posterior.add_node(f"k{candidate:03d}",
                               frame_index=int(session["frames"][candidate]),
                               log_ratio=math.log(probability
                                                  / (1.0 - probability)))
        summary = posterior.summary()
        posterior_scores.append(summary.p_match)
        anchor = summary.best_region_anchor
        posterior_top1.append(
            anchor is not None
            and session["covis"][int(anchor[1:])] >= POSITIVE)
    train_posterior = []
    for index, session in enumerate(train_strict):
        count = len(session["covis"])
        posterior = GoalPosterior(
            unmodeled_log_weight=math.log(args.unmodelled_weight),
            cluster_gap=16)
        for candidate in range(count):
            probability = float(np.clip(tv[index, candidate], 1e-6, 1 - 1e-6))
            posterior.add_node(f"k{candidate:03d}",
                               frame_index=int(session["frames"][candidate]),
                               log_ratio=math.log(probability
                                                  / (1.0 - probability)))
        train_posterior.append(posterior.summary().p_match)
    posterior_threshold = tuned_threshold(train_posterior, train_truth)
    posterior_predicts = [score >= posterior_threshold
                          for score in posterior_scores]
    posterior_arm = score_arm(records, posterior_predicts, posterior_top1)
    posterior_arm["match_auc"] = auc(posterior_scores,
                                     [r["has_match"] for r in records])
    posterior_arm["threshold_selected_on_train"] = float(posterior_threshold)

    report = {
        "scope": ("GLP Stage-2 decision on the audited development split; "
                  "diagnostic only, deployment_approved=false"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
        "feature_schema_version": checkpoint["feature_schema_version"],
        "seeds": len(checkpoint["states"]),
        "development_sessions_scored": len(dev),
        "arms": {"max_dino": dino_arm,
                 "phase_b_model": model_arm,
                 "glp_posterior": posterior_arm},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
