#!/usr/bin/env python3
"""Audit Phase-B at the decision units used by deployment.

The trainer reports a pooled candidate AUC, while deployment first ranks the
candidates *inside one session* and then executes the top-ranked candidate.
Those are different statistical questions.  This diagnostic reports all three
levels without using development outcomes for model selection:

1. pooled strict-candidate AUC;
2. within-session positive/negative pair AUC;
3. paired top-1 correctness, both with and without shortlist misses.

Scene-cluster bootstrap intervals are descriptive.  The development split is
already consumed and this report must never approve a checkpoint or tune an
operating point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from MemNavData.diag_glp_stage2_phase_b import (
    NEGATIVE,
    POSITIVE,
    ensemble_probabilities,
    load_sessions,
)
from MemNavData.phase_b_feature_schema import validate_checkpoint_metadata


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def pairwise_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Probability that a positive score exceeds a negative score.

    Ties contribute one half.  This direct implementation avoids treating rows
    from different sessions as if they competed with each other.
    """

    pos = np.asarray(positive, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(negative, dtype=np.float64).reshape(1, -1)
    if not pos.size or not neg.size:
        return float("nan")
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def pooled_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=bool)
    if score.shape != label.shape:
        raise ValueError("scores and labels must have the same shape")
    return pairwise_auc(score[label], score[~label])


def _cluster_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    value_column: str,
    weight_column: str | None,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if frame.empty or frame["scene"].nunique() < 2:
        raise ValueError("scene bootstrap needs at least two scene clusters")
    scenes = frame["scene"].drop_duplicates().to_numpy(dtype=str)
    by_scene = {scene: frame[frame["scene"] == scene] for scene in scenes}
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        sampled = pd.concat([by_scene[scene] for scene in selected],
                            ignore_index=True)
        if weight_column is None:
            values[index] = float(sampled[value_column].mean())
        else:
            values[index] = float(np.average(
                sampled[value_column], weights=sampled[weight_column]))
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "scene_clusters": int(len(scenes)),
        "resamples": int(resamples),
        "seed": int(seed),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def _scene_from_session(session_id: str) -> str:
    fields = str(session_id).split("/")
    if len(fields) < 2 or not fields[1]:
        raise ValueError(f"cannot recover scene from session id: {session_id!r}")
    return fields[1]


def _strict_top1_records(sessions, validity, rank) -> pd.DataFrame:
    records = []
    for index, session in enumerate(sessions):
        if float(session["session_max"]) < POSITIVE:
            continue
        count = len(session["covis"])
        positive = np.asarray(session["covis"]) >= POSITIVE
        dino_choice = int(np.argmax(session["dino"]))
        model_choice = int(np.argmax(rank[index, :count]))
        records.append({
            "session_id": session["session_id"],
            "scene": _scene_from_session(session["session_id"]),
            "positive_in_shortlist": bool(positive.any()),
            "dino_correct": bool(positive[dino_choice]),
            "phase_b_correct": bool(positive[model_choice]),
            "dino_choice_frame": int(session["frames"][dino_choice]),
            "phase_b_choice_frame": int(session["frames"][model_choice]),
            "dino_choice_covis": float(session["covis"][dino_choice]),
            "phase_b_choice_covis": float(session["covis"][model_choice]),
            "dino_choice_score": float(session["dino"][dino_choice]),
            "phase_b_choice_validity": float(validity[index, model_choice]),
        })
    return pd.DataFrame.from_records(records)


def audit(rows: pd.DataFrame, checkpoint: dict, *,
          bootstrap_resamples: int, bootstrap_seed: int) -> dict:
    feature_names = list(checkpoint["feature_names"])
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float64)
    sessions = load_sessions(rows, feature_names, mean, scale)
    validity, rank, _no_match, _mask = ensemble_probabilities(
        checkpoint, sessions)

    strict_labels: list[bool] = []
    pooled_scores: dict[str, list[float]] = {
        "dino": [],
        "phase_b": [],
    }
    within_records = []
    for index, session in enumerate(sessions):
        count = len(session["covis"])
        covis = np.asarray(session["covis"], dtype=np.float64)
        positive = covis >= POSITIVE
        negative = covis <= NEGATIVE
        strict = positive | negative
        strict_labels.extend(positive[strict].tolist())
        pooled_scores["dino"].extend(
            np.asarray(session["dino"])[strict].tolist())
        pooled_scores["phase_b"].extend(
            validity[index, :count][strict].tolist())
        if positive.any() and negative.any():
            pair_count = int(positive.sum() * negative.sum())
            dino_auc = pairwise_auc(
                np.asarray(session["dino"])[positive],
                np.asarray(session["dino"])[negative])
            model_auc = pairwise_auc(
                validity[index, :count][positive],
                validity[index, :count][negative])
            within_records.append({
                "session_id": session["session_id"],
                "scene": _scene_from_session(session["session_id"]),
                "positive_candidates": int(positive.sum()),
                "negative_candidates": int(negative.sum()),
                "pair_count": pair_count,
                "dino_auc": dino_auc,
                "phase_b_auc": model_auc,
                "delta_phase_b_minus_dino": model_auc - dino_auc,
            })

    within = pd.DataFrame.from_records(within_records)
    if within.empty:
        raise RuntimeError("no session contains both strict positive and negative candidates")
    top1 = _strict_top1_records(sessions, validity, rank)
    if top1.empty:
        raise RuntimeError("no globally positive session was found")
    top1["delta_phase_b_minus_dino"] = (
        top1["phase_b_correct"].astype(int)
        - top1["dino_correct"].astype(int))

    shortlist = top1[top1["positive_in_shortlist"]]
    wins = int((top1["delta_phase_b_minus_dino"] == 1).sum())
    losses = int((top1["delta_phase_b_minus_dino"] == -1).sum())
    labels = np.asarray(strict_labels, dtype=bool)
    pair_weights = within["pair_count"].to_numpy(dtype=np.float64)
    report = {
        "scope": (
            "diagnostic-only Phase-B decision-unit audit on an already-consumed "
            "development split; deployment_approved=false"
        ),
        "thresholds": {
            "positive_covis": POSITIVE,
            "negative_covis": NEGATIVE,
        },
        "sessions": {
            "all": int(len(sessions)),
            "globally_positive": int(len(top1)),
            "positive_in_shortlist": int(len(shortlist)),
            "within_session_auc_eligible": int(len(within)),
            "scene_clusters": int(rows["scene"].nunique()),
        },
        "pooled_strict_candidates": {
            "candidates": int(len(labels)),
            "positives": int(labels.sum()),
            "negatives": int((~labels).sum()),
            "dino_auc": pooled_auc(pooled_scores["dino"], labels),
            "phase_b_auc": pooled_auc(pooled_scores["phase_b"], labels),
        },
        "within_session_pair_auc": {
            "positive_negative_pairs": int(within["pair_count"].sum()),
            "dino_micro": float(np.average(
                within["dino_auc"], weights=pair_weights)),
            "phase_b_micro": float(np.average(
                within["phase_b_auc"], weights=pair_weights)),
            "dino_macro": float(within["dino_auc"].mean()),
            "phase_b_macro": float(within["phase_b_auc"].mean()),
            "delta_micro": float(np.average(
                within["delta_phase_b_minus_dino"], weights=pair_weights)),
            "delta_scene_cluster_bootstrap_95": _cluster_bootstrap_interval(
                within,
                value_column="delta_phase_b_minus_dino",
                weight_column="pair_count",
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
        },
        "top1": {
            "all_global_positive_sessions": {
                "sessions": int(len(top1)),
                "dino_correct": int(top1["dino_correct"].sum()),
                "phase_b_correct": int(top1["phase_b_correct"].sum()),
                "phase_b_wins": wins,
                "phase_b_losses": losses,
                "same": int(len(top1) - wins - losses),
                "delta_scene_cluster_bootstrap_95": (
                    _cluster_bootstrap_interval(
                        top1,
                        value_column="delta_phase_b_minus_dino",
                        weight_column=None,
                        resamples=bootstrap_resamples,
                        seed=bootstrap_seed + 1,
                    )
                ),
            },
            "shortlist_contains_positive": {
                "sessions": int(len(shortlist)),
                "dino_correct": int(shortlist["dino_correct"].sum()),
                "phase_b_correct": int(shortlist["phase_b_correct"].sum()),
            },
            "known_global_match_shortlist_misses": int(
                (~top1["positive_in_shortlist"]).sum()),
        },
        "interpretation_guardrails": [
            "pooled candidate AUC is not a deployment gate",
            "within-session pair AUC does not imply improved top-1",
            "top-1 does not evaluate session activation calibration",
            "the consumed development split cannot select a model or threshold",
        ],
        "top1_discordant_sessions": top1[
            top1["delta_phase_b_minus_dino"] != 0
        ].to_dict(orient="records"),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(
        checkpoint, require_deployment_input_contract=True)
    rows = pd.read_csv(args.development_rows)
    required = {"session_id", "scene", "teacher_covis"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"development rows are missing columns: {missing}")
    report = audit(
        rows,
        checkpoint,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["inputs"] = {
        "development_rows": str(args.development_rows.resolve()),
        "development_rows_sha256": sha256_file(args.development_rows),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
