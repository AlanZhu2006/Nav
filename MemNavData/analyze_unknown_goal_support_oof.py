#!/usr/bin/env python3
"""Nested scene-OOF audit for an unknown-goal memory-support expert.

This is deliberately an offline, train-only gate.  It never assumes that the
policy is told whether a goal is Novel or Revisit.  It factorizes the decision
into session-level memory existence and conditional anchor ranking, then
compares the resulting support decisions with a hard DINO+RANSAC reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from MemNavData.analyze_revisit_geometry_expert import (
        GEOMETRY_FEATURE_NAMES,
        geometry_feature_matrix,
    )
    from MemNavData.train_lingbot_native_localizer import build_feature_matrix
except ImportError:  # direct execution from MemNavData/
    from analyze_revisit_geometry_expert import (  # type: ignore
        GEOMETRY_FEATURE_NAMES,
        geometry_feature_matrix,
    )
    from train_lingbot_native_localizer import build_feature_matrix  # type: ignore


SCHEMA_VERSION = "unknown_goal_support_nested_scene_oof_v1"
POSITIVE_COVIS = 0.5
GEOMETRY_TOP_K = 8
GEOMETRY_VISUAL_FLOOR = 0.88
DEPLOYMENT_ORIGIN = "deployment_topk"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_xy(value: object) -> np.ndarray:
    parsed = json.loads(str(value))
    array = np.asarray(parsed, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError("predicted PointGoal is malformed")
    return array


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = int(wins) + int(losses)
    if discordant == 0:
        return 1.0
    lower = min(int(wins), int(losses))
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2.0 ** discordant))


def scene_folds(scenes: Sequence[str], folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(map(str, scenes))), dtype=object)
    if folds < 2 or folds > len(unique):
        raise ValueError(f"folds must be in [2, {len(unique)}]")
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    return [np.asarray(part, dtype=object) for part in np.array_split(shuffled, folds)]


def choose_risk_matched_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    false_activation_budget: float,
) -> dict[str, float | int]:
    """Maximize positive coverage under a strict-no-match risk constraint."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if (scores.ndim != 1 or labels.shape != scores.shape
            or not np.isfinite(scores).all() or len(np.unique(labels)) != 2):
        raise ValueError("threshold inputs must be finite binary observations")
    if not 0.0 <= false_activation_budget <= 1.0:
        raise ValueError("false-activation budget must be in [0, 1]")
    unique = np.unique(scores)
    thresholds = np.concatenate([
        [np.nextafter(unique[-1], np.inf)],
        unique[::-1],
        [np.nextafter(unique[0], -np.inf)],
    ])
    negatives = ~labels
    candidates = []
    for threshold in thresholds:
        active = scores >= threshold
        false_rate = float(np.mean(active[negatives]))
        if false_rate <= false_activation_budget + 1e-12:
            candidates.append((
                int(np.sum(active & labels)),
                -int(np.sum(active & negatives)),
                float(threshold),
                false_rate,
            ))
    if not candidates:
        raise RuntimeError("no threshold satisfies the risk budget")
    positive_active, negative_false, threshold, false_rate = max(candidates)
    return {
        "threshold": threshold,
        "positive_active": positive_active,
        "strict_false_activations": -negative_false,
        "strict_false_activation_rate": false_rate,
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "sessions": int(len(labels)),
        "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels.astype(np.int64), scores)),
    }


def session_feature_table(deployment):
    """Build compact, deployment-visible set uncertainty features."""

    import pandas as pd

    rows = []
    feature_names = (
        "dino_top1",
        "dino_top2",
        "dino_margin",
        "dino_mean",
        "dino_std",
        "log1p_matches_top1",
        "log1p_matches_max",
        "log1p_inliers_top1",
        "log1p_inliers_max",
        "inlier_ratio_top1",
        "inlier_ratio_max",
        "hard_pass_top1",
        "hard_pass_count",
        "pose_recovered_rate_max",
        "cloud_overlap_f1_max",
        "anchor_goal_distance_norm_min",
        "goal_refine_translation_norm_min",
        "goal_refine_rotation_deg_min",
        "pointgoal_direction_agreement",
        "pointgoal_distance_log_gap",
        "pointgoal_separation_normalized",
    )
    for session_id, group in deployment.groupby("session_id", sort=False):
        ordered = group.sort_values(
            ["candidate_rank", "candidate_frame"], kind="mergesort")
        if len(ordered) != 2:
            raise ValueError(
                f"deployment session {session_id!r} does not have exactly top-2")
        first, second = ordered.iloc[0], ordered.iloc[1]
        dino = ordered["dino_cosine"].to_numpy(dtype=np.float64)
        matches = ordered["geometry_matches"].to_numpy(dtype=np.float64)
        inliers = ordered["geometry_inliers"].to_numpy(dtype=np.float64)
        ratio = ordered["geometry_inlier_ratio"].to_numpy(dtype=np.float64)
        passes = ordered["geometry_hard_pass"].to_numpy(dtype=np.float64)
        points = np.stack([
            parse_xy(first["predicted_relative_xy_m_center_json"]),
            parse_xy(second["predicted_relative_xy_m_center_json"]),
        ])
        distances = np.linalg.norm(points, axis=1)
        if np.all(distances > 1e-12):
            direction_agreement = float(
                np.clip(np.dot(points[0], points[1]) / np.prod(distances), -1.0, 1.0))
        else:
            direction_agreement = 0.0
        distance_log_gap = float(abs(np.log1p(distances[0]) - np.log1p(distances[1])))
        separation = float(
            np.linalg.norm(points[0] - points[1]) / (distances.sum() + 1e-6))
        values = (
            dino[0],
            dino[1],
            dino[0] - dino[1],
            dino.mean(),
            dino.std(),
            np.log1p(matches[0]),
            np.log1p(matches.max()),
            np.log1p(inliers[0]),
            np.log1p(inliers.max()),
            ratio[0],
            ratio.max(),
            passes[0],
            passes.sum(),
            float(ordered["geometry_pose_recovered_rate"].max()),
            float(ordered["cloud_overlap_f1_center"].max()),
            float(ordered["anchor_goal_distance_norm_center"].min()),
            float(ordered["goal_refine_translation_norm_median"].min()),
            float(ordered["goal_refine_rotation_deg_median"].min()),
            direction_agreement,
            distance_log_gap,
            separation,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"session {session_id!r} has non-finite set features")
        record = {
            "session_id": str(session_id),
            "scene": str(first["scene"]),
            "is_positive": truth(first["session_has_positive"]),
            "is_strict_no_match": truth(first["session_is_strict_no_match"]),
            "dino_selected_positive": int(first["label"]) == 1,
            "top2_has_positive": bool(ordered["label"].eq(1).any()),
        }
        record.update(dict(zip(feature_names, map(float, values))))
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)
    if result["session_id"].duplicated().any():
        raise ValueError("session feature table contains duplicates")
    return result, feature_names


def fit_existence(features: np.ndarray, labels: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        ),
    )
    model.fit(features, labels)
    return model


def inner_oof_existence(
    features: np.ndarray,
    labels: np.ndarray,
    scenes: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    for number, held_scenes in enumerate(scene_folds(scenes, folds, seed), start=1):
        held = np.isin(scenes, held_scenes)
        fit = ~held
        if len(np.unique(labels[fit])) != 2:
            raise RuntimeError("inner existence fold has one training class")
        model = fit_existence(features[fit], labels[fit], seed + number)
        predictions[held] = model.predict_proba(features[held])[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("inner OOF existence predictions are incomplete")
    return predictions


def pairwise_examples(frame, features: np.ndarray, fit_scenes: set[str]):
    examples = []
    labels = []
    weights = []
    scene_mask = frame["scene"].astype(str).isin(fit_scenes).to_numpy()
    subset = frame.loc[scene_mask]
    index_lookup = {index: offset for offset, index in enumerate(frame.index)}
    session_pairs: list[tuple[np.ndarray, int]] = []
    for _session_id, group in subset.groupby("session_id", sort=False):
        positive = group.index[group["label"].eq(1)].tolist()
        negative = group.index[group["label"].eq(0)].tolist()
        if not positive or not negative:
            continue
        pairs = [(p, n) for p in positive for n in negative]
        mass = 1.0 / (2.0 * len(pairs))
        for positive_index, negative_index in pairs:
            difference = (
                features[index_lookup[positive_index]]
                - features[index_lookup[negative_index]])
            examples.extend([difference, -difference])
            labels.extend([1, 0])
            weights.extend([mass, mass])
        session_pairs.append((np.empty(0), len(pairs)))
    if not examples:
        raise RuntimeError("no positive-negative candidate pairs in rank training split")
    return (
        np.asarray(examples, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        int(len(session_pairs)),
        int(sum(count for _, count in session_pairs)),
    )


def fit_pairwise_ranker(
    frame,
    candidate_features: np.ndarray,
    fit_scenes: set[str],
    seed: int,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    examples, labels, weights, sessions, pairs = pairwise_examples(
        frame, candidate_features, fit_scenes)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, max_iter=4000, random_state=seed),
    )
    model.fit(examples, labels, logisticregression__sample_weight=weights)
    return model, {"informative_sessions": sessions, "positive_negative_pairs": pairs}


def hard_geometry_outcomes(geometry, session_meta) -> dict[str, dict[str, object]]:
    eligible = geometry[
        (geometry["candidate_rank"].astype(int) < GEOMETRY_TOP_K)
        & (geometry["dino_cosine"].astype(float) >= GEOMETRY_VISUAL_FLOOR)
    ].copy()
    result = {}
    for session_id, meta in session_meta.set_index("session_id").iterrows():
        candidates = eligible[eligible["session_id"].eq(session_id)].sort_values(
            ["candidate_rank", "candidate_frame"], kind="mergesort")
        passed = candidates[candidates["geometry_hard_pass"].eq(1)]
        selected = passed.iloc[0] if len(passed) else None
        result[str(session_id)] = {
            "active": selected is not None,
            "selected_positive": bool(
                selected is not None
                and float(selected["covisibility"]) >= POSITIVE_COVIS),
            "scene": str(meta["scene"]),
            "is_positive": bool(meta["is_positive"]),
            "is_strict_no_match": bool(meta["is_strict_no_match"]),
        }
    return result


def summarize_decisions(records: Iterable[Mapping[str, object]]) -> dict[str, int | float]:
    rows = list(records)
    positive = [row for row in rows if bool(row["is_positive"])]
    strict = [row for row in rows if bool(row["is_strict_no_match"])]
    positive_activated = sum(bool(row["active"]) for row in positive)
    positive_correct = sum(
        bool(row["active"]) and bool(row["selected_positive"])
        for row in positive)
    positive_wrong = sum(
        bool(row["active"]) and not bool(row["selected_positive"])
        for row in positive)
    strict_false = sum(bool(row["active"]) for row in strict)
    correct_decisions = positive_correct + len(strict) - strict_false
    return {
        "positive_sessions": len(positive),
        "strict_no_match_sessions": len(strict),
        "positive_activated": positive_activated,
        "positive_correct_anchor_activated": positive_correct,
        "positive_wrong_anchor_activated": positive_wrong,
        "strict_false_activations": strict_false,
        "strict_false_activation_rate": (
            float(strict_false / len(strict)) if strict else 0.0),
        "correct_support_decisions": correct_decisions,
        "correct_support_rate": (
            float(correct_decisions / (len(positive) + len(strict)))
            if positive or strict else 0.0),
    }


def bootstrap_difference(
    left: Mapping[str, bool],
    right: Mapping[str, bool],
    scene_by_session: Mapping[str, str],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if set(left) != set(right) or set(left) != set(scene_by_session):
        raise ValueError("bootstrap identities differ")
    by_scene: dict[str, list[float]] = {}
    for session_id in left:
        by_scene.setdefault(scene_by_session[session_id], []).append(
            float(right[session_id]) - float(left[session_id]))
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = rng.choice(scenes, len(scenes), replace=True)
        delta = [item for scene in selected for item in by_scene[str(scene)]]
        values.append(float(np.mean(delta)))
    return values


def run_seed(
    merged,
    deployment,
    session_meta,
    session_features: np.ndarray,
    candidate_features: np.ndarray,
    geometry_outcomes: Mapping[str, Mapping[str, object]],
    *,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    from sklearn.isotonic import IsotonicRegression

    extreme = session_meta[
        session_meta["is_positive"] | session_meta["is_strict_no_match"]
    ].copy().reset_index(drop=True)
    feature_by_session = {
        str(row["session_id"]): session_features[index]
        for index, (_, row) in enumerate(session_meta.iterrows())
    }
    deployment_groups = {
        str(session_id): group.sort_values(
            ["candidate_rank", "candidate_frame"], kind="mergesort")
        for session_id, group in deployment.groupby("session_id", sort=False)
    }
    predictions = []
    fold_reports = []
    for fold_number, held_scenes in enumerate(
        scene_folds(extreme["scene"].astype(str), outer_folds, seed), start=1
    ):
        held_scene_set = set(map(str, held_scenes))
        fit_extreme = extreme[~extreme["scene"].isin(held_scene_set)].copy()
        held_extreme = extreme[extreme["scene"].isin(held_scene_set)].copy()
        fit_labels = fit_extreme["is_positive"].to_numpy(dtype=bool)
        fit_scenes = fit_extreme["scene"].astype(str).to_numpy()
        fit_x = np.stack([
            feature_by_session[str(session_id)]
            for session_id in fit_extreme["session_id"]
        ])
        held_x = np.stack([
            feature_by_session[str(session_id)]
            for session_id in held_extreme["session_id"]
        ])

        inner_predictions = inner_oof_existence(
            fit_x,
            fit_labels,
            fit_scenes,
            folds=inner_folds,
            seed=seed + 1000 * fold_number,
        )
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(inner_predictions, fit_labels.astype(np.int64))
        calibrated_inner = calibrator.predict(inner_predictions)

        fit_ids = fit_extreme["session_id"].astype(str).tolist()
        fit_strict = [
            geometry_outcomes[session_id]
            for session_id in fit_ids
            if bool(geometry_outcomes[session_id]["is_strict_no_match"])
        ]
        geometry_budget = (
            float(np.mean([bool(row["active"]) for row in fit_strict]))
            if fit_strict else 0.0)
        factor_threshold = choose_risk_matched_threshold(
            calibrated_inner, fit_labels, geometry_budget)

        dino_inner = fit_extreme["dino_top1"].to_numpy(dtype=np.float64)
        dino_threshold = choose_risk_matched_threshold(
            dino_inner, fit_labels, geometry_budget)

        existence = fit_existence(fit_x, fit_labels, seed + fold_number)
        held_raw_probability = existence.predict_proba(held_x)[:, 1]
        held_probability = calibrator.predict(held_raw_probability)
        ranker, rank_train = fit_pairwise_ranker(
            merged,
            candidate_features,
            set(fit_extreme["scene"].astype(str)),
            seed + 2000 + fold_number,
        )

        for row_index, (_, meta) in enumerate(held_extreme.iterrows()):
            session_id = str(meta["session_id"])
            candidates = deployment_groups[session_id]
            candidate_offsets = candidates[
                "candidate_feature_index"].to_numpy(dtype=np.int64)
            rank_scores = ranker.decision_function(candidate_features[candidate_offsets])
            factor_pick = int(np.argmax(rank_scores))
            dino_pick = 0  # deployment rows are sorted by candidate_rank
            factor_active = bool(
                held_probability[row_index] >= float(factor_threshold["threshold"]))
            dino_score = float(meta["dino_top1"])
            dino_active = bool(dino_score >= float(dino_threshold["threshold"]))
            predictions.append({
                "seed": seed,
                "outer_fold": fold_number,
                "session_id": session_id,
                "scene": str(meta["scene"]),
                "is_positive": bool(meta["is_positive"]),
                "is_strict_no_match": bool(meta["is_strict_no_match"]),
                "top2_has_positive": bool(meta["top2_has_positive"]),
                "existence_probability": float(held_probability[row_index]),
                "existence_raw_probability": float(held_raw_probability[row_index]),
                "factor_threshold": float(factor_threshold["threshold"]),
                "dino_threshold": float(dino_threshold["threshold"]),
                "geometry_risk_budget": geometry_budget,
                "factor_active": factor_active,
                "factor_selected_positive": bool(
                    int(candidates.iloc[factor_pick]["label"]) == 1),
                "factor_selected_candidate_rank": int(
                    candidates.iloc[factor_pick]["candidate_rank"]),
                "dino_score": dino_score,
                "dino_active": dino_active,
                "dino_selected_positive": bool(
                    int(candidates.iloc[dino_pick]["label"]) == 1),
                "geometry_active": bool(geometry_outcomes[session_id]["active"]),
                "geometry_selected_positive": bool(
                    geometry_outcomes[session_id]["selected_positive"]),
            })
        fold_reports.append({
            "outer_fold": fold_number,
            "held_scenes": sorted(held_scene_set),
            "fit_extreme_sessions": int(len(fit_extreme)),
            "held_extreme_sessions": int(len(held_extreme)),
            "geometry_strict_risk_budget": geometry_budget,
            "factor_threshold": factor_threshold,
            "dino_threshold": dino_threshold,
            "rank_training": rank_train,
        })

    def method_rows(prefix: str) -> list[dict[str, object]]:
        return [
            {
                **row,
                "active": bool(row[f"{prefix}_active"]),
                "selected_positive": bool(row[f"{prefix}_selected_positive"]),
            }
            for row in predictions
        ]

    summaries = {
        name: summarize_decisions(method_rows(name))
        for name in ("geometry", "dino", "factor")
    }
    positive_covered = [
        row for row in predictions
        if bool(row["is_positive"]) and bool(row["top2_has_positive"])
    ]
    wins = sum(
        bool(row["factor_selected_positive"])
        and not bool(row["dino_selected_positive"])
        for row in positive_covered)
    losses = sum(
        bool(row["dino_selected_positive"])
        and not bool(row["factor_selected_positive"])
        for row in positive_covered)

    labels = np.asarray([bool(row["is_positive"]) for row in predictions])
    factor_scores = np.asarray([
        float(row["existence_probability"]) for row in predictions])
    dino_scores = np.asarray([float(row["dino_score"]) for row in predictions])
    # DINO cosine is not a probability; min-max scaling is report-only for Brier.
    dino_min, dino_max = float(dino_scores.min()), float(dino_scores.max())
    dino_probability = (dino_scores - dino_min) / max(dino_max - dino_min, 1e-12)

    identities = [str(row["session_id"]) for row in predictions]
    scene_by_session = {str(row["session_id"]): str(row["scene"]) for row in predictions}
    geometry_correct = {
        str(row["session_id"]): bool(
            (row["is_positive"] and row["geometry_active"]
             and row["geometry_selected_positive"])
            or (row["is_strict_no_match"] and not row["geometry_active"]))
        for row in predictions
    }
    factor_correct = {
        str(row["session_id"]): bool(
            (row["is_positive"] and row["factor_active"]
             and row["factor_selected_positive"])
            or (row["is_strict_no_match"] and not row["factor_active"]))
        for row in predictions
    }
    bootstrap = bootstrap_difference(
        geometry_correct,
        factor_correct,
        scene_by_session,
        samples=bootstrap_samples,
        seed=seed,
    )
    gate = {
        "risk_not_worse": (
            summaries["factor"]["strict_false_activations"]
            <= summaries["geometry"]["strict_false_activations"]),
        "correct_anchor_coverage_better": (
            summaries["factor"]["positive_correct_anchor_activated"]
            > summaries["geometry"]["positive_correct_anchor_activated"]),
        "wrong_anchor_not_worse": (
            summaries["factor"]["positive_wrong_anchor_activated"]
            <= summaries["geometry"]["positive_wrong_anchor_activated"]),
        "conditional_rank_wins": wins > losses,
    }
    gate["stage1_pass"] = all(gate.values())
    report = {
        "seed": seed,
        "methods": summaries,
        "existence": {
            "factor": binary_metrics(labels, factor_scores),
            "dino_minmax_report_only": binary_metrics(labels, dino_probability),
        },
        "conditional_rank_factor_vs_dino": {
            "positive_top2_covered_sessions": len(positive_covered),
            "wins": wins,
            "losses": losses,
            "ties": len(positive_covered) - wins - losses,
            "exact_mcnemar_p": exact_mcnemar_p(wins, losses),
        },
        "factor_minus_geometry_correct_support_rate": {
            "point": float(np.mean([
                float(factor_correct[session_id]) - float(geometry_correct[session_id])
                for session_id in identities
            ])),
            "scene_cluster_bootstrap_95ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        },
        "pre_registered_gate": gate,
        "folds": fold_reports,
    }
    return report, predictions


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-rows", type=Path, required=True)
    parser.add_argument("--geometry-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--seeds", default="20260811,20260812,20260813",
        help="comma-separated scene-fold seeds")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("protocol requires exactly three distinct seeds")
    if not args.phase_rows.is_file() or not args.geometry_evidence.is_file():
        raise FileNotFoundError("input table is missing")
    phase_sha = sha256(args.phase_rows)
    geometry_sha = sha256(args.geometry_evidence)
    phase = pd.read_csv(args.phase_rows).reset_index(drop=True)
    geometry = pd.read_csv(args.geometry_evidence).reset_index(drop=True)
    if set(phase["causal_split_role"].astype(str)) != {"train"}:
        raise RuntimeError("Phase-B input is not train-only")
    if set(geometry["split_role"].astype(str)) != {"train"}:
        raise RuntimeError("geometry input is not train-only")
    keys = ["session_id", "candidate_path"]
    if phase.duplicated(keys).any() or geometry.duplicated(keys).any():
        raise ValueError("input join keys are not unique")
    geometry_columns = [
        *keys,
        "candidate_rank",
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
    ]
    merged = phase.merge(
        geometry[geometry_columns], on=keys, how="left", validate="one_to_one")
    if merged["geometry_matches"].isna().any():
        raise RuntimeError("Phase-B candidates lack geometry evidence")
    if phase["scene"].nunique() != 40 or phase["session_id"].nunique() != 480:
        raise RuntimeError("unexpected train scene/session denominator")

    merged["candidate_feature_index"] = np.arange(len(merged), dtype=np.int64)
    deployment = merged[
        merged["candidate_selection_origin"].eq(DEPLOYMENT_ORIGIN)
    ].copy()
    deployment = deployment.sort_values(
        ["session_id", "candidate_rank", "candidate_frame"],
        kind="mergesort").reset_index(drop=True)
    if len(deployment) != 960 or not deployment.groupby("session_id").size().eq(2).all():
        raise RuntimeError("deployment top-2 contract changed")

    session_meta, session_feature_names = session_feature_table(deployment)
    session_feature_matrix_values = session_meta[
        list(session_feature_names)].to_numpy(dtype=np.float64)
    phase_features, phase_names, _predicted_xy, _target_xy = build_feature_matrix(merged)
    geometry_features = geometry_feature_matrix(merged)
    candidate_features = np.column_stack([phase_features, geometry_features])
    candidate_feature_names = [*phase_names, *GEOMETRY_FEATURE_NAMES]
    if not np.isfinite(candidate_features).all():
        raise RuntimeError("candidate feature matrix is non-finite")

    geometry_outcomes = hard_geometry_outcomes(geometry, session_meta)
    seed_reports = []
    all_predictions = []
    for seed in seeds:
        report, predictions = run_seed(
            merged,
            deployment,
            session_meta,
            session_feature_matrix_values,
            candidate_features,
            geometry_outcomes,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        seed_reports.append(report)
        all_predictions.extend(predictions)
        print(
            f"[seed {seed}] gate={report['pre_registered_gate']['stage1_pass']} "
            f"factor={report['methods']['factor']} ",
            flush=True,
        )

    all_pass = all(
        bool(report["pre_registered_gate"]["stage1_pass"])
        for report in seed_reports)
    existence_all = all(
        bool(report["pre_registered_gate"]["risk_not_worse"])
        and bool(report["pre_registered_gate"]["correct_anchor_coverage_better"])
        and bool(report["pre_registered_gate"]["wrong_anchor_not_worse"])
        for report in seed_reports)
    rank_all = all(
        bool(report["pre_registered_gate"]["conditional_rank_wins"])
        for report in seed_reports)
    if all_pass:
        branch = "advance_to_train_only_counterfactual_action_collection"
    elif existence_all and not rank_all:
        branch = "retain_dino_anchor_advance_existence_only"
    else:
        branch = "stop_before_action_expert_improve_memory_support_observability"

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "train_only_nested_scene_oof_complete",
        "scope": "train-only; unknown goal kind; no development; no consumed pool; no blind; no closed-loop claim",
        "deployment_approved": False,
        "input": {
            "phase_rows": str(args.phase_rows.resolve()),
            "phase_rows_sha256": phase_sha,
            "geometry_evidence": str(args.geometry_evidence.resolve()),
            "geometry_evidence_sha256": geometry_sha,
            "scenes": int(phase["scene"].nunique()),
            "sessions": int(phase["session_id"].nunique()),
            "candidate_rows_all": int(len(phase)),
            "deployment_top2_rows": int(len(deployment)),
            "positive_sessions": int(session_meta["is_positive"].sum()),
            "strict_no_match_sessions": int(session_meta["is_strict_no_match"].sum()),
            "ambiguous_sessions": int((
                ~session_meta["is_positive"]
                & ~session_meta["is_strict_no_match"]).sum()),
            "positive_sessions_with_top2_positive": int(
                (session_meta["is_positive"] & session_meta["top2_has_positive"]).sum()),
        },
        "feature_contract": {
            "existence_feature_names": list(session_feature_names),
            "candidate_feature_names": candidate_feature_names,
            "existence_model": "standardized L2 logistic, C=0.25, balanced classes",
            "anchor_model": "scene-held-out pairwise standardized L2 logistic, C=0.25",
            "geometry_reference": {
                "top_k": GEOMETRY_TOP_K,
                "visual_floor": GEOMETRY_VISUAL_FLOOR,
                "selection": "first DINO-order hard pass; decision-unit reference without temporal confirmation",
            },
        },
        "fold_contract": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seeds": list(seeds),
            "threshold": "maximize positive existence coverage under outer-train hard-geometry strict-no-match false-activation rate",
        },
        "seed_reports": seed_reports,
        "pre_registered_decision": {
            "all_three_seeds_pass": all_pass,
            "existence_coverage_risk_all_three": existence_all,
            "conditional_rank_all_three": rank_all,
            "branch": branch,
        },
        "limits": [
            "Offline co-visibility support is not identical to closed-loop usefulness.",
            "The hard-geometry reference omits online two-plan confirmation and latch.",
            "No action contrast is present; passing only authorizes counterfactual data collection.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    predictions_path = args.output_dir / "session_oof_predictions.csv"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError("output already exists")
    temporary_csv = predictions_path.with_suffix(".csv.partial")
    pd.DataFrame(all_predictions).to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, predictions_path)
    output["output"] = {
        "session_predictions": str(predictions_path.resolve()),
        "session_predictions_sha256": sha256(predictions_path),
    }
    atomic_json(report_path, output)
    print(json.dumps(output["pre_registered_decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
