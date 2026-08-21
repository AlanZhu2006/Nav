#!/usr/bin/env python3
"""Reproduce the local, label-authorized MRC attribution audit.

This script deliberately does not read the formal 24-session contract smoke:
that population was frozen for ABI/timing validation and is not authorized for
feature or threshold selection.  It instead uses the older balanced Revisit
feasibility artifact, the repaired train-only Phase-B artifact, and the two
explicitly exploratory local two-scene runs.

The output is diagnostic only.  In particular, within-scene normalization is
an oracle analysis of nuisance scale, not a deployable calibration recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


FEATURES = {
    "dino_cosine": 1.0,
    "cloud_overlap_f1_median": 1.0,
    "goal_pose_translation_dispersion_norm": -1.0,
    "goal_pose_rotation_dispersion_deg": -1.0,
    "goal_refine_translation_norm_median": -1.0,
    "goal_refine_rotation_deg_median": -1.0,
}


def r_squared(values: np.ndarray, design: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    design = np.column_stack([np.ones(len(values)), design])
    prediction = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((values - prediction) ** 2)) / total


def within_group_z(values: pd.Series, groups: pd.Series) -> pd.Series:
    centered = values - values.groupby(groups).transform("mean")
    scale = values.groupby(groups).transform(
        lambda item: float(item.std(ddof=0)))
    return centered / scale.where(scale > 1e-12, 1.0)


def archived_audit(frame: pd.DataFrame) -> dict:
    exact = frame.loc[
        frame["n_hypotheses"].eq(3)
        & frame["neighbor_offsets"].astype(str).eq("-4;0;4")
        & frame["label"].isin([0, 1])
    ].copy()
    features = {}
    for name, direction in FEATURES.items():
        valid = exact[name].notna()
        subset = exact.loc[valid]
        signed = direction * subset[name].to_numpy(dtype=float)
        labels = subset["label"].to_numpy(dtype=int)
        scene_only = pd.get_dummies(
            subset["scene"], drop_first=True, dtype=float).to_numpy()
        label_only = subset[["label"]].to_numpy(dtype=float)
        features[name] = {
            "n": int(len(subset)),
            "raw_auc": float(roc_auc_score(labels, signed)),
            "within_scene_z_auc": float(roc_auc_score(
                labels,
                direction * within_group_z(
                    subset[name], subset["scene"]).to_numpy(dtype=float),
            )),
            "r2_scene_identity": r_squared(signed, scene_only),
            "r2_candidate_label": r_squared(signed, label_only),
        }

    per_offset = {
        "cloud_overlap_f1": [],
        "goal_refine_translation_norm": [],
    }
    for raw in exact["hypotheses_json"]:
        hypotheses = sorted(json.loads(raw), key=lambda item: item["offset"])
        if [item["offset"] for item in hypotheses] != [-4, 0, 4]:
            continue
        per_offset["cloud_overlap_f1"].append([
            float(item["cloud_overlap_f1"]) for item in hypotheses])
        per_offset["goal_refine_translation_norm"].append([
            float(item["goal_refine_translation_raw"])
            / max(float(item["depth_scale_raw"]), 1e-12)
            for item in hypotheses])
    correlations = {}
    for name, rows in per_offset.items():
        values = np.asarray(rows, dtype=np.float64)
        matrix = np.corrcoef(values, rowvar=False)
        pairwise = [
            float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2])]
        mean = float(np.mean(pairwise))
        correlations[name] = {
            "pairwise_correlations": pairwise,
            "mean_pairwise_correlation": mean,
            "effective_independent_views_exchangeable_approx": (
                3.0 / (1.0 + 2.0 * mean)),
        }
    return {
        "rows": int(len(exact)),
        "sessions": int(exact["session_id"].nunique()),
        "scenes": int(exact["scene"].nunique()),
        "positive": int(exact["label"].eq(1).sum()),
        "negative": int(exact["label"].eq(0).sum()),
        "features": features,
        "nominal_three_view_dependence": correlations,
    }


def phase_b_proposal_audit(frame: pd.DataFrame) -> dict:
    deployment = frame.loc[
        frame["candidate_selection_origin"].eq("deployment_topk")].copy()
    top1 = deployment.sort_values(
        ["session_id", "dino_cosine"], ascending=[True, False]
    ).groupby("session_id", as_index=False).head(1)
    positive = top1.loc[top1["session_has_positive"].astype(bool)]
    strict = top1.loc[top1["session_is_strict_no_match"].astype(bool)]
    ambiguous = top1.loc[
        ~top1["session_has_positive"].astype(bool)
        & ~top1["session_is_strict_no_match"].astype(bool)]
    deployment_positive = deployment.loc[
        deployment["session_id"].isin(positive["session_id"])]
    topk_has_positive = deployment_positive.groupby("session_id")[
        "label"].apply(lambda item: bool(item.eq(1).any()))
    top1_is_positive = positive.set_index("session_id")["label"].eq(1)
    aligned = pd.concat([
        top1_is_positive.rename("top1"),
        topk_has_positive.rename("topk"),
    ], axis=1).fillna(False)
    return {
        "sessions": int(len(top1)),
        "positive_sessions": int(len(positive)),
        "positive_top1_correct": int(positive["label"].eq(1).sum()),
        "positive_top1_wrong_or_ignore": int(positive["label"].ne(1).sum()),
        "positive_proposal_decomposition": {
            "p1_top1_positive": int(aligned["top1"].sum()),
            "p2_top1_wrong_but_topk_positive": int(
                ((~aligned["top1"]) & aligned["topk"]).sum()),
            "p3_deployment_topk_has_no_positive": int(
                (~aligned["topk"]).sum()),
        },
        "strict_no_match_sessions": int(len(strict)),
        "ambiguous_sessions": int(len(ambiguous)),
        "top1_only_certificate_positive_recall_ceiling": float(
            positive["label"].eq(1).mean()),
    }


def sqlite_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    connection = sqlite3.connect(path)
    try:
        return [
            json.loads(payload) for (payload,) in connection.execute(
                "SELECT payload_json FROM rows ORDER BY seed_index")]
    finally:
        connection.close()


def local_rows(csv_path: Path, sqlite_path: Path) -> pd.DataFrame:
    records: list[dict] = []
    if csv_path.is_file():
        records.extend(pd.read_csv(csv_path).to_dict("records"))
    records.extend(sqlite_rows(sqlite_path))
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    return frame.drop_duplicates("session_id", keep="first")


def physical_clip_span(row: pd.Series) -> dict | None:
    candidate = Path(str(row.get("candidate_path", "")))
    if not candidate.is_file():
        return None
    root = next(
        (parent for parent in candidate.parents
         if parent.name.startswith("episode_")), None)
    if root is None:
        return None
    parquet = root / "data/chunk-000/episode_000000.parquet"
    if not parquet.is_file():
        return None
    poses = []
    for value in pd.read_parquet(parquet, columns=["action"])["action"]:
        raw = value.tolist() if hasattr(value, "tolist") else value
        poses.append(np.asarray(raw, dtype=np.float64).reshape(4, 4))
    center = int(row["candidate_frame"])
    offsets = [int(value) for value in str(row["neighbor_offsets"]).split(";")]
    indices = [center + offset for offset in offsets]
    selected = [poses[index] for index in indices]
    distances = [
        float(np.linalg.norm(selected[left][:3, 3] - selected[right][:3, 3]))
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]
    angles = []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            relative = selected[left][:3, :3].T @ selected[right][:3, :3]
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cosine))))
    return {
        "session_id": str(row["session_id"]),
        "label": int(row["label"]),
        "frames": indices,
        "endpoint_span_m": float(np.linalg.norm(
            selected[0][:3, 3] - selected[-1][:3, 3])),
        "minimum_pairwise_baseline_m": min(distances),
        "maximum_pairwise_rotation_deg": max(angles),
    }


def local_audit(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    signed = frame.loc[frame["label"].isin([0, 1])].copy()
    result = {
        "rows": int(len(frame)),
        "signed_rows": int(len(signed)),
        "correct_positive": int(signed["label"].eq(1).sum()),
        "strict_negative": int(
            signed["session_is_strict_no_match"].astype(bool).sum()),
        "ignore_or_ambiguous": int(frame["label"].eq(-1).sum()),
        "feature_auc": {},
        "internal_signal_to_metric_pose_error_spearman": {},
        "physical_clips": [],
    }
    for name, direction in FEATURES.items():
        valid = signed[name].notna()
        subset = signed.loc[valid]
        if subset["label"].nunique() == 2:
            result["feature_auc"][name] = float(roc_auc_score(
                subset["label"], direction * subset[name]))
    pose_errors = [
        "relative_position_error_m_median",
        "relative_position_direction_error_deg_median",
        "relative_rotation_error_deg_median",
    ]
    internal = [
        "goal_pose_translation_dispersion_norm",
        "goal_pose_rotation_dispersion_deg",
        "cloud_overlap_f1_median",
        "goal_refine_translation_norm_median",
    ]
    for signal in internal:
        result["internal_signal_to_metric_pose_error_spearman"][signal] = {}
        for error in pose_errors:
            pair = frame[[signal, error]].dropna()
            result["internal_signal_to_metric_pose_error_spearman"][signal][
                error] = float(pair[signal].corr(pair[error], method="spearman"))
    result["physical_clips"] = [
        item for item in (physical_clip_span(row) for _, row in frame.iterrows())
        if item is not None]
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--archived", type=Path,
        default=Path(".diagnostics/mrc_existing_audit_20260812/"
                     "multiscene100_rows.csv"))
    result.add_argument(
        "--phase-b", type=Path,
        default=Path(".diagnostics/phase_b_train_repaired_20260808/"
                     "lingbot_goal_loop_closure_rows.csv"))
    result.add_argument(
        "--local-csv", type=Path,
        default=Path(".diagnostics/mrc_local_targeted_twoscene_20260812/"
                     "lingbot_goal_loop_closure_rows.csv"))
    result.add_argument(
        "--local-sqlite", type=Path,
        default=Path(".diagnostics/mrc_local_twoscene_top1_20260812/"
                     "lingbot_goal_loop_closure_checkpoint.sqlite3"))
    return result


def main() -> None:
    args = parser().parse_args()
    report = {
        "scope": "diagnostic_only_no_formal_smoke_labels",
        "archived_known_revisit_balanced": archived_audit(
            pd.read_csv(args.archived)),
        "phase_b_deployment_top1": phase_b_proposal_audit(
            pd.read_csv(args.phase_b)),
        "local_two_scene_exploratory": local_audit(
            local_rows(args.local_csv, args.local_sqlite)),
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
