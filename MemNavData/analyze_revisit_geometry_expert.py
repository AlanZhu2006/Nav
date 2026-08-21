#!/usr/bin/env python3
"""Scene-grouped analysis for the Revisit geometry-expert evidence table.

All learned scores are out-of-fold at the scene level.  The script deliberately
does not select a deployment threshold: it asks whether geometry contributes
candidate information beyond frozen DINO and whether a hard RANSAC rejection
is semantically justified by task-aligned co-visibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "revisit_geometry_expert_analysis_v1"
REQUIRED_COLUMNS = frozenset(
    {
        "session_id",
        "scene",
        "split_role",
        "candidate_rank",
        "candidate_frame",
        "dino_cosine",
        "covisibility",
        "label",
        "geometry_query_keypoints",
        "geometry_candidate_keypoints",
        "geometry_matches",
        "geometry_inliers",
        "geometry_inlier_ratio",
        "geometry_essential_available_rate",
        "geometry_pose_recovered_rate",
        "geometry_pass_rate",
        "geometry_inliers_std",
        "geometry_hard_pass",
        "geometry_state",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2.0 ** discordant))


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"count": int(len(labels)), "roc_auc": None, "average_precision": None}
    return {
        "count": int(len(labels)),
        "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def geometry_feature_matrix(frame) -> np.ndarray:
    matches = frame["geometry_matches"].to_numpy(dtype=np.float64)
    inliers = frame["geometry_inliers"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.log1p(matches),
            np.log1p(inliers),
            frame["geometry_inlier_ratio"].to_numpy(dtype=np.float64),
            frame["geometry_essential_available_rate"].to_numpy(dtype=np.float64),
            frame["geometry_pose_recovered_rate"].to_numpy(dtype=np.float64),
            frame["geometry_pass_rate"].to_numpy(dtype=np.float64),
            np.log1p(frame["geometry_query_keypoints"].to_numpy(dtype=np.float64)),
            np.log1p(frame["geometry_candidate_keypoints"].to_numpy(dtype=np.float64)),
            np.log1p(frame["geometry_inliers_std"].to_numpy(dtype=np.float64)),
        ]
    )


GEOMETRY_FEATURE_NAMES = (
    "log1p_matches",
    "log1p_inliers",
    "inlier_ratio",
    "essential_available_rate",
    "pose_recovered_rate",
    "pass_rate",
    "log1p_query_keypoints",
    "log1p_candidate_keypoints",
    "log1p_inlier_std",
)


def scene_oof_scores(frame, features: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = frame["label"].to_numpy(dtype=np.int64)
    scenes = frame["scene"].astype(str).to_numpy()
    extreme = np.isin(labels, [0, 1])
    unique_scenes = np.unique(scenes)
    if folds < 2 or folds > len(unique_scenes):
        raise ValueError(f"folds must be within [2, {len(unique_scenes)}]")
    if features.shape[0] != len(frame) or not np.isfinite(features).all():
        raise ValueError("feature matrix is non-finite or misaligned")
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    fold_reports = []
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(frame), dtype=np.float64)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(dummy, labels, groups=scenes), start=1
    ):
        fit_index = train_index[extreme[train_index]]
        if len(np.unique(labels[fit_index])) != 2:
            raise RuntimeError(f"fold {fold} training data has one class")
        session_counts = frame.iloc[fit_index].groupby("session_id")["session_id"].transform("size")
        sample_weight = 1.0 / session_counts.to_numpy(dtype=np.float64)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed + fold,
            ),
        )
        model.fit(features[fit_index], labels[fit_index], logisticregression__sample_weight=sample_weight)
        predictions[test_index] = model.predict_proba(features[test_index])[:, 1]
        scaler = model.named_steps["standardscaler"]
        logistic = model.named_steps["logisticregression"]
        fold_reports.append(
            {
                "fold": fold,
                "train_scenes": sorted(set(scenes[train_index])),
                "test_scenes": sorted(set(scenes[test_index])),
                "train_rows": int(len(fit_index)),
                "test_rows": int(len(test_index)),
                "coefficient_standardized": logistic.coef_[0].astype(float).tolist(),
                "intercept": float(logistic.intercept_[0]),
                "feature_mean": scaler.mean_.astype(float).tolist(),
                "feature_scale": scaler.scale_.astype(float).tolist(),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("scene OOF predictions are incomplete")
    return predictions, fold_reports


def within_session_concordance(frame, score_column: str) -> tuple[dict[str, object], dict[str, float]]:
    per_session: dict[str, float] = {}
    total_wins = 0.0
    total_pairs = 0
    for session_id, group in frame.groupby("session_id", sort=False):
        positive = group.loc[group["label"].eq(1), score_column].to_numpy(dtype=np.float64)
        negative = group.loc[group["label"].eq(0), score_column].to_numpy(dtype=np.float64)
        if not len(positive) or not len(negative):
            continue
        comparisons = positive[:, None] - negative[None, :]
        wins = float(np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0))
        pairs = int(comparisons.size)
        per_session[str(session_id)] = wins / pairs
        total_wins += wins
        total_pairs += pairs
    values = np.asarray(list(per_session.values()), dtype=np.float64)
    return (
        {
            "sessions": len(per_session),
            "pairs": total_pairs,
            "pair_weighted_auc": float(total_wins / total_pairs) if total_pairs else None,
            "session_macro_auc": float(values.mean()) if len(values) else None,
        },
        per_session,
    )


def session_selection(frame, score_column: str):
    import pandas as pd

    rows = []
    for session_id, group in frame.groupby("session_id", sort=False):
        group = group.sort_values(
            [score_column, "candidate_rank", "candidate_frame"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        maximum_covis = float(group["covisibility"].max())
        is_positive = maximum_covis >= 0.5
        is_strict_no_match = maximum_covis < 0.2
        selected = group.iloc[0]
        positive_ranks = np.flatnonzero(group["label"].to_numpy(dtype=np.int64) == 1)
        first_positive_rank = int(positive_ranks[0] + 1) if len(positive_ranks) else None
        rows.append(
            {
                "session_id": session_id,
                "scene": str(selected["scene"]),
                "is_positive": is_positive,
                "is_strict_no_match": is_strict_no_match,
                "session_max_covis": maximum_covis,
                "session_score": float(group[score_column].max()),
                "selected_label": int(selected["label"]),
                "selected_positive": int(selected["label"]) == 1,
                "selected_candidate_rank": int(selected["candidate_rank"]),
                "first_positive_rank": first_positive_rank,
            }
        )
    return pd.DataFrame(rows)


def session_metrics(frame, score_column: str) -> tuple[dict[str, object], object]:
    sessions = session_selection(frame, score_column)
    positive = sessions[sessions["is_positive"]]
    strict = sessions[sessions["is_strict_no_match"]]
    existence = sessions[sessions["is_positive"] | sessions["is_strict_no_match"]]
    labels = existence["is_positive"].to_numpy(dtype=np.int64)
    scores = existence["session_score"].to_numpy(dtype=np.float64)
    reciprocal = [
        1.0 / rank for rank in positive["first_positive_rank"] if rank is not None
    ]
    return (
        {
            "sessions": int(len(sessions)),
            "positive_sessions": int(len(positive)),
            "strict_no_match_sessions": int(len(strict)),
            "positive_top1": int(positive["selected_positive"].sum()),
            "positive_top1_rate": float(positive["selected_positive"].mean()) if len(positive) else None,
            "mean_reciprocal_positive_rank": float(np.mean(reciprocal)) if reciprocal else None,
            "existence": binary_metrics(labels, scores),
        },
        sessions,
    )


def hard_gate_metrics(frame) -> dict[str, object]:
    rows = []
    for session_id, group in frame.groupby("session_id", sort=False):
        group = group.sort_values(["candidate_rank", "candidate_frame"], kind="mergesort")
        maximum_covis = float(group["covisibility"].max())
        passed = group[group["geometry_hard_pass"].eq(1)]
        selected = passed.iloc[0] if len(passed) else None
        rows.append(
            {
                "session_id": session_id,
                "scene": str(group.iloc[0]["scene"]),
                "is_positive": maximum_covis >= 0.5,
                "is_strict_no_match": maximum_covis < 0.2,
                "activated": selected is not None,
                "selected_positive": bool(selected is not None and int(selected["label"]) == 1),
            }
        )
    positive = [row for row in rows if row["is_positive"]]
    strict = [row for row in rows if row["is_strict_no_match"]]
    return {
        "sessions": len(rows),
        "positive_sessions": len(positive),
        "strict_no_match_sessions": len(strict),
        "positive_activated": sum(row["activated"] for row in positive),
        "positive_correct_selected": sum(row["selected_positive"] for row in positive),
        "strict_no_match_false_activations": sum(row["activated"] for row in strict),
    }


def bootstrap_scene_values(
    session_values: Mapping[str, float],
    session_scene: Mapping[str, str],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    by_scene: dict[str, list[float]] = {}
    for session_id, value in session_values.items():
        by_scene.setdefault(session_scene[session_id], []).append(float(value))
    scenes = sorted(by_scene)
    if not scenes:
        return []
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(samples):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        values = [value for scene in selected for value in by_scene[str(scene)]]
        result.append(float(np.mean(values)))
    return result


def paired_top1_comparison(left, right, *, bootstrap_samples: int, seed: int) -> dict[str, object]:
    left = left[left["is_positive"]].set_index("session_id")
    right = right[right["is_positive"]].set_index("session_id")
    if set(left.index) != set(right.index):
        raise RuntimeError("positive session identities differ across rankings")
    identities = sorted(left.index)
    delta_by_session = {
        session_id: float(right.at[session_id, "selected_positive"])
        - float(left.at[session_id, "selected_positive"])
        for session_id in identities
    }
    scene_by_session = {session_id: str(left.at[session_id, "scene"]) for session_id in identities}
    wins = sum(value > 0 for value in delta_by_session.values())
    losses = sum(value < 0 for value in delta_by_session.values())
    bootstrap = bootstrap_scene_values(
        delta_by_session,
        scene_by_session,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "sessions": len(identities),
        "right_minus_left_rate": float(np.mean(list(delta_by_session.values()))),
        "wins": wins,
        "losses": losses,
        "ties": len(identities) - wins - losses,
        "exact_mcnemar_p": exact_mcnemar_p(wins, losses),
        "scene_cluster_bootstrap_95ci": (
            [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
            if bootstrap
            else None
        ),
    }


def state_table(frame) -> dict[str, dict[str, object]]:
    result = {}
    for state, group in frame.groupby("geometry_state", sort=True):
        extreme = group[group["label"].isin([0, 1])]
        positive = int(extreme["label"].eq(1).sum())
        result[str(state)] = {
            "rows": int(len(group)),
            "extreme_rows": int(len(extreme)),
            "positive": positive,
            "negative": int(extreme["label"].eq(0).sum()),
            "positive_fraction": float(positive / len(extreme)) if len(extreme) else None,
            "positive_scenes": int(extreme.loc[extreme["label"].eq(1), "scene"].nunique()),
        }
    return result


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--expected-evidence-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if not args.evidence_csv.is_file():
        raise FileNotFoundError(args.evidence_csv)
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap sample count must be at least 100")
    evidence_sha = sha256(args.evidence_csv)
    if args.expected_evidence_sha256 and evidence_sha != args.expected_evidence_sha256:
        raise RuntimeError("evidence CSV SHA mismatch")
    frame = pd.read_csv(args.evidence_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"evidence CSV missing columns: {sorted(missing)}")
    if set(frame["split_role"].astype(str)) != {"train"}:
        raise RuntimeError("analysis is frozen to train-only evidence")
    if frame.duplicated(["session_id", "candidate_frame"]).any():
        raise ValueError("duplicate session/candidate rows")

    geometry_features = geometry_feature_matrix(frame)
    dino = frame["dino_cosine"].to_numpy(dtype=np.float64)[:, None]
    fusion_features = np.column_stack([dino, geometry_features])
    appearance_oof, appearance_folds = scene_oof_scores(frame, dino, args.folds, args.seed)
    geometry_oof, geometry_folds = scene_oof_scores(frame, geometry_features, args.folds, args.seed + 100)
    fusion_oof, fusion_folds = scene_oof_scores(frame, fusion_features, args.folds, args.seed + 200)
    frame = frame.copy()
    frame["score_dino"] = frame["dino_cosine"].astype(float)
    frame["score_appearance_oof"] = appearance_oof
    frame["score_geometry_oof"] = geometry_oof
    frame["score_fusion_oof"] = fusion_oof
    frame["score_geometry_pass_rate"] = frame["geometry_pass_rate"].astype(float)

    labels = frame["label"].to_numpy(dtype=np.int64)
    extreme = np.isin(labels, [0, 1])
    candidate = {}
    within = {}
    per_session_auc = {}
    sessions = {}
    session_frames = {}
    score_columns = {
        "dino": "score_dino",
        "appearance_oof": "score_appearance_oof",
        "geometry_oof": "score_geometry_oof",
        "geometry_pass_rate": "score_geometry_pass_rate",
        "fusion_oof": "score_fusion_oof",
    }
    for name, column in score_columns.items():
        candidate[name] = binary_metrics(labels[extreme], frame.loc[extreme, column].to_numpy(dtype=np.float64))
        within[name], per_session_auc[name] = within_session_concordance(frame, column)
        sessions[name], session_frames[name] = session_metrics(frame, column)

    comparison = paired_top1_comparison(
        session_frames["dino"],
        session_frames["fusion_oof"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    hard_positive = frame[frame["label"].eq(1)]
    positive_reject = hard_positive[hard_positive["geometry_hard_pass"].eq(0)]
    stable_support = frame[(frame["geometry_state"] == "stable_support") & frame["label"].isin([0, 1])]
    stable_support_precision = (
        float(stable_support["label"].eq(1).mean()) if len(stable_support) else None
    )
    hard_recall = float(hard_positive["geometry_hard_pass"].mean()) if len(hard_positive) else None
    geometry_within = within["geometry_oof"]["session_macro_auc"]
    fusion_top1_gate = comparison["wins"] > comparison["losses"] and comparison["exact_mcnemar_p"] < 0.05
    ranking_gate = geometry_within is not None and geometry_within > 0.5 and fusion_top1_gate
    support_unknown_gate = (
        stable_support_precision is not None
        and stable_support_precision >= 0.90
        and len(positive_reject) > 0
        and int(positive_reject["scene"].nunique()) >= 5
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "train_scene_grouped_oof_diagnostic_complete",
        "deployment_approved": False,
        "scope": "train_only; no development; no blind; no closed_loop_claim",
        "input": {
            "evidence_csv": str(args.evidence_csv.resolve()),
            "evidence_csv_sha256": evidence_sha,
            "rows": int(len(frame)),
            "scenes": int(frame["scene"].nunique()),
            "sessions": int(frame["session_id"].nunique()),
            "labels": {str(label): int(frame["label"].eq(label).sum()) for label in (-1, 0, 1)},
        },
        "feature_contract": {
            "appearance": ["dino_cosine"],
            "geometry": list(GEOMETRY_FEATURE_NAMES),
            "fusion": ["dino_cosine", *GEOMETRY_FEATURE_NAMES],
            "model": "standardized logistic regression; balanced classes; equal session mass",
            "folds": args.folds,
        },
        "candidate_extreme_label_metrics": candidate,
        "within_session_concordance": within,
        "session_metrics": sessions,
        "hard_geometry_gate": hard_gate_metrics(frame),
        "geometry_states": state_table(frame),
        "hard_positive_candidate_recall": hard_recall,
        "hard_rejected_positive_candidates": int(len(positive_reject)),
        "hard_rejected_positive_scenes": int(positive_reject["scene"].nunique()),
        "stable_support_precision": stable_support_precision,
        "fusion_vs_dino_positive_session_top1": comparison,
        "folds": {
            "appearance": appearance_folds,
            "geometry": geometry_folds,
            "fusion": fusion_folds,
        },
        "pre_registered_decision": {
            "authorize_ransac_as_second_ranking_expert": bool(ranking_gate),
            "authorize_ransac_support_unknown_semantics": bool(support_unknown_gate),
            "ranking_gate_contract": "geometry OOF within-session macro AUC > 0.5 and fusion top-1 beats DINO with exact McNemar p < 0.05",
            "support_unknown_gate_contract": "stable-support precision >= 0.90 and hard rejection misses positives in >=5 scenes",
            "important_limit": "Neither gate authorizes deployment; a frozen closed-loop paired test is still required.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "scene_oof_scores.csv"
    report_path = args.output_dir / "report.json"
    if scores_path.exists() or report_path.exists():
        raise FileExistsError("analysis output already exists")
    temporary = scores_path.with_suffix(".csv.partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, scores_path)
    report["output"] = {
        "scores_csv": str(scores_path.resolve()),
        "scores_csv_sha256": sha256(scores_path),
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
