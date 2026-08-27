#!/usr/bin/env python3
"""Stage-1 falsification replay for the GLP posterior decision layer.

Question (GOAL_POSTERIOR_DECISION_LAYER_20260807.md §7 Stage 1): given the
SAME per-candidate DINO evidence, does a calibrated goal-location posterior
match or beat the max-DINO + train-only-threshold baseline on the audited
600-session causal teacher?  If not, the framework adds no information and
Stage 2 must not start.

Protocol mirrors /tmp/nlsr_set_objective_smoke.py exactly:
  - strict sessions: session max teacher_covis >= 0.5 (positive) or <= 0.1
    (negative); ambiguous sessions excluded;
  - every scalar fit or selected (Platt calibration, unmodeled weight,
    cluster gap) uses train sessions only; development is scored once;
  - identical joint / match / conditional-recall@1 definitions.

Read-only over the teacher CSV; writes a JSON report only.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MemNavData.goal_posterior import GoalPosterior  # noqa: E402

TEACHER = "/tmp/nlsr_causal_teacher_20260807.csv"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".diagnostics", "goal_posterior_teacher_replay_20260807")
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = 0.1
UNMODELED_GRID = [math.log(v) for v in
                  (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                   128.0, 256.0, 512.0, 1024.0)]
CLUSTER_GAPS = (4, 8, 16, 32)


def strict_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    maximum = frame.groupby("session_id", sort=False)["teacher_covis"].max()
    keep = maximum[(maximum >= POSITIVE_THRESHOLD)
                   | (maximum <= NEGATIVE_THRESHOLD)]
    return frame[frame["session_id"].isin(keep.index)].copy()


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Newton fit of P(pos | s) = sigmoid(a * s + b) on train candidates."""
    a, b = 1.0, 0.0
    for _ in range(50):
        z = np.clip(a * scores + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - labels
        grad = np.array([np.dot(g, scores), g.sum()])
        w = p * (1.0 - p) + 1e-9
        h11 = np.dot(w * scores, scores)
        h12 = np.dot(w, scores)
        h22 = w.sum()
        hessian = np.array([[h11, h12], [h12, h22]])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(2), grad)
        a, b = a - step[0], b - step[1]
        if float(np.abs(step).max()) < 1e-10:
            break
    if not (math.isfinite(a) and math.isfinite(b)):
        raise RuntimeError("Platt calibration diverged")
    return float(a), float(b)


def auc(scores: list[float], labels: list[bool]) -> float:
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def session_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for session_id, group in frame.groupby("session_id", sort=False):
        group = group.sort_values(
            ["dino_cosine", "candidate_frame", "candidate_path"],
            ascending=[False, True, True], kind="mergesort")
        records.append({
            "session_id": session_id,
            "split": group.iloc[0]["split_role"],
            "has_match": float(group["teacher_covis"].max())
            >= POSITIVE_THRESHOLD,
            "max_dino": float(group.iloc[0]["dino_cosine"]),
            "dino_top1_positive": float(group.iloc[0]["teacher_covis"])
            >= POSITIVE_THRESHOLD,
            "frames": group["candidate_frame"].astype(int).tolist(),
            "scores": group["dino_cosine"].astype(float).tolist(),
            "covis": group["teacher_covis"].astype(float).tolist(),
        })
    return records


def score_arm(rows: list[dict], predicts: list[bool],
              top1_flags: list[bool]) -> dict:
    joint, match, recall = [], [], []
    for row, predicted, top1 in zip(rows, predicts, top1_flags):
        match.append(predicted == row["has_match"])
        joint.append((row["has_match"] and predicted and top1)
                     or ((not row["has_match"]) and (not predicted)))
        if row["has_match"]:
            recall.append(top1)
    return {
        "sessions": len(rows),
        "joint_localization_accuracy": float(np.mean(joint)),
        "match_accuracy": float(np.mean(match)),
        "conditional_candidate_recall_at_1": float(np.mean(recall)),
    }


def dino_baseline(train_rows: list[dict], dev_rows: list[dict]) -> dict:
    values = sorted({r["max_dino"] for r in train_rows})
    thresholds = ([-np.inf]
                  + [0.5 * (u + v) for u, v in zip(values, values[1:])]
                  + [np.inf])

    def arm(rows, threshold):
        predicts = [r["max_dino"] >= threshold for r in rows]
        top1 = [r["dino_top1_positive"] for r in rows]
        return score_arm(rows, predicts, top1)

    threshold = max(
        thresholds,
        key=lambda t: (arm(train_rows, t)["joint_localization_accuracy"],
                       arm(train_rows, t)["match_accuracy"],
                       -abs(t) if np.isfinite(t) else -1e9))
    result = {"threshold_selected_on_train": float(threshold),
              "train": arm(train_rows, threshold),
              "development": arm(dev_rows, threshold)}
    result["development"]["match_auc"] = auc(
        [r["max_dino"] for r in dev_rows],
        [r["has_match"] for r in dev_rows])
    return result


def posterior_predictions(rows: list[dict], a: float, b: float,
                          base_logit: float, unmodeled_log: float,
                          cluster_gap: int) -> tuple[list[bool], list[bool],
                                                     list[float]]:
    predicts, top1_flags, p_matches = [], [], []
    for row in rows:
        posterior = GoalPosterior(
            unmodeled_log_weight=unmodeled_log, cluster_gap=cluster_gap)
        covis_by_id = {}
        for index, (frame, score, covis) in enumerate(
                zip(row["frames"], row["scores"], row["covis"])):
            node_id = f"k{index:03d}"
            log_ratio = (a * score + b) - base_logit
            posterior.add_node(node_id, frame_index=frame,
                               log_ratio=log_ratio)
            covis_by_id[node_id] = covis
        summary = posterior.summary()
        p_matches.append(summary.p_match)
        predicts.append(summary.p_match >= 0.5)
        anchor = summary.best_region_anchor
        top1_flags.append(
            anchor is not None
            and covis_by_id[anchor] >= POSITIVE_THRESHOLD)
    return predicts, top1_flags, p_matches


def main() -> None:
    frame = pd.read_csv(TEACHER)
    strict = strict_sessions(frame)
    train = strict[strict["split_role"].eq("train")]
    development = strict[strict["split_role"].eq("development")]
    train_rows = session_records(train)
    dev_rows = session_records(development)

    baseline = dino_baseline(train_rows, dev_rows)

    # Platt calibration on train candidates only (ambiguous candidates
    # excluded from the fit, present at inference).
    candidate_pos = train["teacher_covis"].to_numpy() >= POSITIVE_THRESHOLD
    candidate_neg = train["teacher_covis"].to_numpy() <= NEGATIVE_THRESHOLD
    fit_mask = candidate_pos | candidate_neg
    a, b = fit_platt(
        train["dino_cosine"].to_numpy(dtype=float)[fit_mask],
        candidate_pos[fit_mask].astype(float))
    base_rate = float(candidate_pos[fit_mask].mean())
    base_logit = math.log(base_rate / (1.0 - base_rate))

    # unmodeled weight + cluster gap selected on train joint accuracy only
    best = None
    for unmodeled_log in UNMODELED_GRID:
        for gap in CLUSTER_GAPS:
            predicts, top1, _ = posterior_predictions(
                train_rows, a, b, base_logit, unmodeled_log, gap)
            scored = score_arm(train_rows, predicts, top1)
            key = (scored["joint_localization_accuracy"],
                   scored["match_accuracy"], -abs(unmodeled_log))
            if best is None or key > best[0]:
                best = (key, unmodeled_log, gap, scored)
    _, unmodeled_log, gap, train_scored = best

    dev_predicts, dev_top1, dev_p = posterior_predictions(
        dev_rows, a, b, base_logit, unmodeled_log, gap)
    dev_scored = score_arm(dev_rows, dev_predicts, dev_top1)
    dev_scored["match_auc"] = auc(dev_p, [r["has_match"] for r in dev_rows])

    # paired transitions vs the baseline on development joint correctness
    threshold = baseline["threshold_selected_on_train"]
    transitions = {"glp_only_correct": [], "baseline_only_correct": []}
    for row, predicted, top1 in zip(dev_rows, dev_predicts, dev_top1):
        base_pred = row["max_dino"] >= threshold
        base_ok = ((row["has_match"] and base_pred
                    and row["dino_top1_positive"])
                   or ((not row["has_match"]) and (not base_pred)))
        glp_ok = ((row["has_match"] and predicted and top1)
                  or ((not row["has_match"]) and (not predicted)))
        if glp_ok and not base_ok:
            transitions["glp_only_correct"].append(row["session_id"])
        if base_ok and not glp_ok:
            transitions["baseline_only_correct"].append(row["session_id"])

    report = {
        "scope": ("Stage-1 falsification replay: calibrated DINO-only "
                  "posterior vs max-DINO threshold baseline; NOT the "
                  "Phase-B feature model"),
        "teacher": TEACHER,
        "strict_session_counts": {
            "train": len(train_rows), "development": len(dev_rows)},
        "platt": {"a": a, "b": b, "train_base_rate": base_rate},
        "selected_on_train": {
            "unmodeled_log_weight": unmodeled_log,
            "unmodeled_weight": math.exp(unmodeled_log),
            "cluster_gap": gap},
        "dino_threshold_baseline": baseline,
        "glp_posterior": {"train": train_scored, "development": dev_scored},
        "development_paired_transitions": {
            "glp_only_correct": transitions["glp_only_correct"],
            "baseline_only_correct": transitions["baseline_only_correct"],
            "n_gain": len(transitions["glp_only_correct"]),
            "n_loss": len(transitions["baseline_only_correct"])},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "report.json")
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwritten: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
