#!/usr/bin/env python3
"""Nested scene-OOF DINO-preserving actionability residual ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


EXPECTED_SHA = (
    "193c29da7e2904061691361d5285d2211ff61b997619156f8b74262fde18237b")
FEATURES = (
    "depth_scale_raw",
    "cloud_overlap_f1_center",
    "anchor_goal_distance_norm_center",
    "goal_refine_translation_norm_median",
    "goal_refine_rotation_deg_median",
    "goal_depth_confidence_mean",
    "candidate_depth_confidence_mean",
)
C_GRID = (0.01, 0.1, 1.0, 10.0)
ALPHA_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def load_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "session_id", "scene", "candidate_frame", "dino_cosine",
        "relative_position_direction_error_deg_center", "causal_split_role",
        "causal_state_name", "causal_goal_variant", *FEATURES,
    }
    require(not (required - set(table.columns)), "required columns missing")
    table = table.loc[
        table["causal_split_role"].astype(str).eq("train")
        & table["causal_state_name"].astype(str).eq("goal_c_t0")
        & table["causal_goal_variant"].astype(str).eq("factual")].copy()
    table = table.sort_values(
        ["session_id", "dino_cosine", "candidate_frame"],
        ascending=[True, False, True], kind="stable").reset_index(drop=True)
    require(len(table) == 232 and table["session_id"].nunique() == 80
            and table["scene"].nunique() == 40,
            "factual Revisit-C candidate universe changed")
    numeric = table.loc[:, [*FEATURES, "dino_cosine",
                            "relative_position_direction_error_deg_center"]]
    require(bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all()),
            "non-finite model input or target")
    table["actionable"] = table[
        "relative_position_direction_error_deg_center"].le(30.0)
    return table


def session_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table[["session_id", "scene"]].drop_duplicates().reset_index(
        drop=True)
    require(result["session_id"].is_unique, "session maps to multiple scenes")
    return result


def fit_pairwise(table: pd.DataFrame, fit_sessions: set[str], *,
                 regularization_c: float) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, int]:
    fit = table["session_id"].isin(fit_sessions).to_numpy()
    x = table.loc[:, FEATURES].to_numpy(dtype=np.float64)
    mean = x[fit].mean(axis=0)
    scale = x[fit].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    differences = []
    for _session_id, indices in table.loc[fit].groupby(
            "session_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        good = index[table.loc[index, "actionable"].to_numpy(dtype=bool)]
        bad = index[~table.loc[index, "actionable"].to_numpy(dtype=bool)]
        for left in good:
            for right in bad:
                differences.append(normalized[left] - normalized[right])
    require(bool(differences), "training split has no preference pairs")
    positive = np.asarray(differences, dtype=np.float64)
    pair_x = np.concatenate([positive, -positive], axis=0)
    pair_y = np.concatenate([
        np.ones(len(positive), dtype=np.int64),
        np.zeros(len(positive), dtype=np.int64),
    ])
    model = LogisticRegression(
        C=regularization_c, fit_intercept=False, solver="lbfgs",
        max_iter=5000, tol=1e-9).fit(pair_x, pair_y)
    return model.coef_[0].astype(np.float64), mean, scale, len(positive)


def residual_scores(table: pd.DataFrame, coefficient: np.ndarray,
                    mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = table.loc[:, FEATURES].to_numpy(dtype=np.float64)
    result = ((x - mean) / scale) @ coefficient
    require(bool(np.isfinite(result).all()), "non-finite residual score")
    return result


def top1_outcomes(table: pd.DataFrame, scores: np.ndarray,
                  sessions: set[str]) -> pd.DataFrame:
    subset = table.loc[table["session_id"].isin(sessions)].copy()
    score_array = np.asarray(scores, dtype=np.float64)
    require(len(score_array) == len(table), "score/table length mismatch")
    subset["score"] = score_array[subset.index.to_numpy(dtype=np.int64)]
    selected = subset.loc[subset.groupby("session_id", sort=True)[
        "score"].idxmax()].copy()
    return selected[[
        "session_id", "scene", "candidate_frame", "actionable",
        "relative_position_direction_error_deg_center",
    ]].sort_values("session_id").reset_index(drop=True)


def compare_to_dino(table: pd.DataFrame, scores: np.ndarray,
                    sessions: set[str]) -> dict[str, Any]:
    learned = top1_outcomes(table, scores, sessions)
    dino = top1_outcomes(
        table, table["dino_cosine"].to_numpy(dtype=np.float64), sessions)
    joined = learned.merge(
        dino[["session_id", "actionable"]].rename(
            columns={"actionable": "dino_actionable"}),
        on="session_id", validate="one_to_one")
    return {
        "correct": int(joined["actionable"].sum()),
        "gains": int((joined["actionable"]
                      & ~joined["dino_actionable"]).sum()),
        "losses": int((~joined["actionable"]
                       & joined["dino_actionable"]).sum()),
        "sessions": len(joined),
    }


def nested_choice(table: pd.DataFrame, outer_train: pd.DataFrame,
                  *, inner_folds: int) -> tuple[float, float,
                                                list[dict[str, Any]]]:
    splitter = GroupKFold(inner_folds)
    splits = list(splitter.split(
        outer_train, groups=outer_train["scene"].astype(str)))
    records = []
    for regularization_c in C_GRID:
        fold_predictions = []
        for fit_local, validation_local in splits:
            fit_sessions = set(outer_train.iloc[fit_local]["session_id"])
            validation_sessions = set(
                outer_train.iloc[validation_local]["session_id"])
            coefficient, mean, scale, _pairs = fit_pairwise(
                table, fit_sessions, regularization_c=regularization_c)
            residual = residual_scores(table, coefficient, mean, scale)
            fold_predictions.append((validation_sessions, residual))
        for alpha in ALPHA_GRID:
            totals = {"correct": 0, "gains": 0, "losses": 0, "sessions": 0}
            for validation_sessions, residual in fold_predictions:
                score = (table["dino_cosine"].to_numpy(dtype=np.float64)
                         + alpha * residual)
                result = compare_to_dino(table, score, validation_sessions)
                for key in totals:
                    totals[key] += int(result[key])
            records.append({
                "C": regularization_c,
                "alpha": alpha,
                **totals,
            })
    selected = sorted(
        records,
        key=lambda row: (-row["correct"], row["losses"], row["alpha"],
                         row["C"]))[0]
    return float(selected["C"]), float(selected["alpha"]), records


def cluster_bootstrap(outcomes: pd.DataFrame, *, resamples: int,
                      seed: int) -> dict[str, Any]:
    scenes = sorted(outcomes["scene"].astype(str).unique())
    by_scene = {scene: outcomes.loc[outcomes["scene"].eq(scene)]
                for scene in scenes}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        sample = pd.concat([by_scene[str(scene)] for scene in selected],
                           ignore_index=True)
        differences.append(float(
            sample["residual_actionable"].mean()
            - sample["dino_actionable"].mean()))
    return {
        "scenes": len(scenes),
        "resamples": resamples,
        "median": float(np.median(differences)),
        "ci95": [float(np.percentile(differences, 2.5)),
                 float(np.percentile(differences, 97.5))],
    }


def run(table: pd.DataFrame, *, outer_folds: int, inner_folds: int,
        bootstrap_resamples: int, seed: int) -> tuple[dict[str, Any],
                                                      pd.DataFrame]:
    sessions = session_table(table)
    splitter = GroupKFold(outer_folds)
    oof_score = np.full(len(table), np.nan, dtype=np.float64)
    oof_fold = np.full(len(table), -1, dtype=np.int64)
    oof_alpha = np.full(len(table), np.nan, dtype=np.float64)
    fold_reports = []
    for fold, (train_local, test_local) in enumerate(
            splitter.split(sessions, groups=sessions["scene"].astype(str))):
        outer_train = sessions.iloc[train_local].reset_index(drop=True)
        outer_test = sessions.iloc[test_local].reset_index(drop=True)
        regularization_c, alpha, inner = nested_choice(
            table, outer_train, inner_folds=inner_folds)
        train_ids = set(outer_train["session_id"])
        test_ids = set(outer_test["session_id"])
        coefficient, mean, scale, pairs = fit_pairwise(
            table, train_ids, regularization_c=regularization_c)
        residual = residual_scores(table, coefficient, mean, scale)
        score = table["dino_cosine"].to_numpy(dtype=np.float64) + alpha * residual
        test_mask = table["session_id"].isin(test_ids).to_numpy()
        oof_score[test_mask] = score[test_mask]
        oof_fold[test_mask] = fold
        oof_alpha[test_mask] = alpha
        fold_reports.append({
            "fold": fold,
            "train_scenes": sorted(set(outer_train["scene"].astype(str))),
            "test_scenes": sorted(set(outer_test["scene"].astype(str))),
            "selected_C": regularization_c,
            "selected_alpha": alpha,
            "fit_preference_pairs": pairs,
            "test": compare_to_dino(table, score, test_ids),
            "inner_grid": inner,
            "coefficient": coefficient.tolist(),
        })
    require(bool(np.isfinite(oof_score).all()) and bool((oof_fold >= 0).all()),
            "OOF coverage incomplete")

    all_ids = set(sessions["session_id"])
    residual = top1_outcomes(table, oof_score, all_ids).rename(columns={
        "candidate_frame": "residual_candidate_frame",
        "actionable": "residual_actionable",
        "relative_position_direction_error_deg_center": "residual_error_deg",
    })
    dino = top1_outcomes(
        table, table["dino_cosine"].to_numpy(dtype=np.float64), all_ids
    ).rename(columns={
        "candidate_frame": "dino_candidate_frame",
        "actionable": "dino_actionable",
        "relative_position_direction_error_deg_center": "dino_error_deg",
    })
    outcomes = residual.merge(dino.drop(columns="scene"), on="session_id",
                              validate="one_to_one")
    outcomes["gain"] = outcomes["residual_actionable"] & ~outcomes[
        "dino_actionable"]
    outcomes["loss"] = ~outcomes["residual_actionable"] & outcomes[
        "dino_actionable"]
    gains = int(outcomes["gain"].sum())
    losses = int(outcomes["loss"].sum())
    residual_correct = int(outcomes["residual_actionable"].sum())
    dino_correct = int(outcomes["dino_actionable"].sum())
    gain_scenes = int(outcomes.loc[outcomes["gain"], "scene"].nunique())
    positive_alpha_folds = sum(
        report["selected_alpha"] > 0 for report in fold_reports)
    gate_checks = {
        "oof_top1_at_least_77_of_80": residual_correct >= 77,
        "losses_at_most_1": losses <= 1,
        "positive_alpha_in_at_least_4_folds": positive_alpha_folds >= 4,
        "gains_in_at_least_3_scenes": gain_scenes >= 3,
    }
    report = {
        "scope": "nested_scene_oof_train_only_ranking_not_sr",
        "data": {"scenes": 40, "sessions": 80, "candidates": len(table)},
        "features": list(FEATURES),
        "grids": {"C": list(C_GRID), "alpha": list(ALPHA_GRID)},
        "baseline": {
            "dino_top1_actionable": dino_correct,
            "materialized_candidate_oracle": int(table.groupby(
                "session_id")["actionable"].any().sum()),
        },
        "oof": {
            "residual_top1_actionable": residual_correct,
            "gains": gains,
            "losses": losses,
            "risk_difference": (residual_correct - dino_correct) / 80.0,
            "exact_mcnemar_p": exact_mcnemar_p(gains, losses),
            "gain_scenes": gain_scenes,
            "selected_positive_alpha_folds": positive_alpha_folds,
            "scene_cluster_bootstrap": cluster_bootstrap(
                outcomes, resamples=bootstrap_resamples, seed=seed),
        },
        "frozen_gate": {
            **gate_checks,
            "passed": all(gate_checks.values()),
        },
        "folds": fold_reports,
        "authorization": {
            "larger_train_only_selector_study": all(gate_checks.values()),
            "closed_loop": False,
            "long_training": False,
            "development_or_blind_read": False,
        },
    }
    return report, outcomes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", default=EXPECTED_SHA)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    require(args.production_rows.is_file(), "production rows missing")
    require(sha256_file(args.production_rows) == args.expected_sha,
            "production rows SHA mismatch")
    require(args.outer_folds == 5 and args.inner_folds == 4,
            "frozen fold counts changed")
    require(args.bootstrap_resamples >= 1, "invalid bootstrap count")
    return args


def main() -> None:
    args = parse_args()
    report, outcomes = run(
        load_table(args.production_rows),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed)
    report["input_sha256"] = sha256_file(args.production_rows)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    outcomes.to_csv(args.out / "session_outcomes.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
