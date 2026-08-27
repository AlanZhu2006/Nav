#!/usr/bin/env python3
"""Train-only nested scene-OOF audit of top-8 set uncertainty.

The conditional anchor ranker and deployment top-2 remain unchanged.  Only the
existence head receives fixed, label-free summaries of the DINO-ordered top-8
geometry evidence.  See UNKNOWN_GOAL_TOP8_UNCERTAINTY_PROTOCOL_20260811.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from MemNavData.analyze_unknown_goal_support_oof import (
    DEPLOYMENT_ORIGIN,
    GEOMETRY_FEATURE_NAMES,
    GEOMETRY_TOP_K,
    GEOMETRY_VISUAL_FLOOR,
    atomic_json,
    geometry_feature_matrix,
    hard_geometry_outcomes,
    run_seed,
    session_feature_table,
    sha256,
)
from MemNavData.train_lingbot_native_localizer import build_feature_matrix


SCHEMA_VERSION = "unknown_goal_top8_set_uncertainty_nested_scene_oof_v1"

TOP8_FEATURE_NAMES = (
    "top8_dino_mean",
    "top8_dino_std",
    "top8_dino_top1_minus_top4",
    "top8_dino_top1_minus_top8",
    "top8_dino_top2_minus_tail_mean",
    "top8_log1p_matches_mean",
    "top8_log1p_matches_std",
    "top8_log1p_matches_max",
    "top8_log1p_inliers_mean",
    "top8_log1p_inliers_std",
    "top8_log1p_inliers_max",
    "top8_inlier_ratio_mean",
    "top8_inlier_ratio_std",
    "top8_inlier_ratio_max",
    "top8_hard_pass_count",
    "top8_first_hard_pass_rank_or_8",
    "top8_essential_available_rate_mean",
    "top8_essential_available_rate_max",
    "top8_pose_recovered_rate_mean",
    "top8_pose_recovered_rate_max",
    "top8_pass_rate_mean",
    "top8_pass_rate_max",
)


def top8_summary_table(geometry):
    """Return fixed label-free summaries for every DINO-ordered top-8 set."""

    import pandas as pd

    required = {
        "session_id",
        "candidate_rank",
        "dino_cosine",
        "geometry_matches",
        "geometry_inliers",
        "geometry_inlier_ratio",
        "geometry_hard_pass",
        "geometry_essential_available_rate",
        "geometry_pose_recovered_rate",
        "geometry_pass_rate",
    }
    missing = sorted(required - set(geometry.columns))
    if missing:
        raise ValueError(f"geometry table lacks top-8 columns: {missing}")
    rows = []
    for session_id, group in geometry.groupby("session_id", sort=False):
        top = group[group["candidate_rank"].astype(int) < GEOMETRY_TOP_K].copy()
        top = top.sort_values("candidate_rank", kind="mergesort")
        ranks = top["candidate_rank"].to_numpy(dtype=np.int64)
        if len(top) != GEOMETRY_TOP_K or not np.array_equal(
            ranks, np.arange(GEOMETRY_TOP_K, dtype=np.int64)
        ):
            raise ValueError(
                f"session {session_id!r} lacks exactly ranks 0..{GEOMETRY_TOP_K - 1}")
        dino = top["dino_cosine"].to_numpy(dtype=np.float64)
        matches = np.log1p(
            top["geometry_matches"].to_numpy(dtype=np.float64))
        inliers = np.log1p(
            top["geometry_inliers"].to_numpy(dtype=np.float64))
        ratios = top["geometry_inlier_ratio"].to_numpy(dtype=np.float64)
        hard_pass = top["geometry_hard_pass"].to_numpy(dtype=np.float64)
        pass_ranks = ranks[hard_pass >= 0.5]
        essential = top[
            "geometry_essential_available_rate"].to_numpy(dtype=np.float64)
        recovered = top[
            "geometry_pose_recovered_rate"].to_numpy(dtype=np.float64)
        pass_rate = top["geometry_pass_rate"].to_numpy(dtype=np.float64)
        values = (
            dino.mean(),
            dino.std(),
            dino[0] - dino[3],
            dino[0] - dino[7],
            dino[1] - dino[2:].mean(),
            matches.mean(),
            matches.std(),
            matches.max(),
            inliers.mean(),
            inliers.std(),
            inliers.max(),
            ratios.mean(),
            ratios.std(),
            ratios.max(),
            hard_pass.sum(),
            float(pass_ranks[0]) if len(pass_ranks) else float(GEOMETRY_TOP_K),
            essential.mean(),
            essential.max(),
            recovered.mean(),
            recovered.max(),
            pass_rate.mean(),
            pass_rate.max(),
        )
        if not np.isfinite(values).all():
            raise ValueError(f"session {session_id!r} has non-finite top-8 summaries")
        record = {"session_id": str(session_id)}
        record.update(dict(zip(TOP8_FEATURE_NAMES, map(float, values))))
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)
    if result["session_id"].duplicated().any():
        raise ValueError("top-8 summary contains duplicate sessions")
    return result


def compare_with_f2(f8_report: dict, f2_report: dict) -> dict[str, bool]:
    """Apply the frozen per-seed F8 decision gate."""

    f8 = f8_report["methods"]["factor"]
    geometry = f8_report["methods"]["geometry"]
    f2 = f2_report["methods"]["factor"]
    if geometry != f2_report["methods"]["geometry"]:
        raise RuntimeError("hard-geometry reference differs from frozen F2 report")
    gate = {
        "risk_not_worse_than_geometry": (
            f8["strict_false_activations"]
            <= geometry["strict_false_activations"]),
        "wrong_anchor_not_worse_than_geometry": (
            f8["positive_wrong_anchor_activated"]
            <= geometry["positive_wrong_anchor_activated"]),
        "correct_anchor_better_than_geometry": (
            f8["positive_correct_anchor_activated"]
            > geometry["positive_correct_anchor_activated"]),
        "correct_anchor_better_than_f2": (
            f8["positive_correct_anchor_activated"]
            > f2["positive_correct_anchor_activated"]),
        "correct_support_better_than_f2": (
            f8["correct_support_decisions"]
            > f2["correct_support_decisions"]),
    }
    gate["top8_pass"] = all(gate.values())
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-rows", type=Path, required=True)
    parser.add_argument("--geometry-evidence", type=Path, required=True)
    parser.add_argument("--f2-report", type=Path, required=True)
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
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("protocol requires exactly three distinct seeds")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    for path in (args.phase_rows, args.geometry_evidence, args.f2_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    phase_sha = sha256(args.phase_rows)
    geometry_sha = sha256(args.geometry_evidence)
    f2_sha = sha256(args.f2_report)
    with args.f2_report.open("r", encoding="utf-8") as handle:
        f2 = json.load(handle)
    if f2.get("schema_version") != "unknown_goal_support_nested_scene_oof_v1":
        raise RuntimeError("F2 report schema changed")
    if f2["input"]["phase_rows_sha256"] != phase_sha:
        raise RuntimeError("F2 report uses different Phase-B rows")
    if f2["input"]["geometry_evidence_sha256"] != geometry_sha:
        raise RuntimeError("F2 report uses different geometry evidence")
    if tuple(f2["fold_contract"]["seeds"]) != seeds:
        raise RuntimeError("F2 report uses different seeds")

    phase = pd.read_csv(args.phase_rows).reset_index(drop=True)
    geometry = pd.read_csv(args.geometry_evidence).reset_index(drop=True)
    if set(phase["causal_split_role"].astype(str)) != {"train"}:
        raise RuntimeError("Phase-B input is not train-only")
    if set(geometry["split_role"].astype(str)) != {"train"}:
        raise RuntimeError("geometry input is not train-only")
    if phase["scene"].nunique() != 40 or phase["session_id"].nunique() != 480:
        raise RuntimeError("unexpected train denominator")
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
    merged["candidate_feature_index"] = np.arange(len(merged), dtype=np.int64)
    deployment = merged[
        merged["candidate_selection_origin"].eq(DEPLOYMENT_ORIGIN)
    ].copy().sort_values(
        ["session_id", "candidate_rank", "candidate_frame"], kind="mergesort"
    ).reset_index(drop=True)
    if len(deployment) != 960 or not deployment.groupby("session_id").size().eq(2).all():
        raise RuntimeError("deployment top-2 contract changed")

    session_meta, f2_feature_names = session_feature_table(deployment)
    top8 = top8_summary_table(geometry)
    session_meta = session_meta.merge(
        top8, on="session_id", how="left", validate="one_to_one")
    if session_meta[list(TOP8_FEATURE_NAMES)].isna().any().any():
        raise RuntimeError("top-8 session join is incomplete")
    existence_feature_names = (*f2_feature_names, *TOP8_FEATURE_NAMES)
    existence_features = session_meta[
        list(existence_feature_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(existence_features).all():
        raise RuntimeError("F8 existence matrix is non-finite")

    phase_features, phase_names, _predicted_xy, _target_xy = build_feature_matrix(
        merged)
    candidate_features = np.column_stack([
        phase_features,
        geometry_feature_matrix(merged),
    ])
    candidate_feature_names = [*phase_names, *GEOMETRY_FEATURE_NAMES]
    geometry_outcomes = hard_geometry_outcomes(geometry, session_meta)

    f2_by_seed = {int(row["seed"]): row for row in f2["seed_reports"]}
    seed_reports = []
    predictions = []
    comparisons = []
    for seed in seeds:
        report, seed_predictions = run_seed(
            merged,
            deployment,
            session_meta,
            existence_features,
            candidate_features,
            geometry_outcomes,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        gate = compare_with_f2(report, f2_by_seed[seed])
        seed_reports.append(report)
        predictions.extend(seed_predictions)
        comparisons.append({
            "seed": seed,
            "f2_factor": f2_by_seed[seed]["methods"]["factor"],
            "f8_factor": report["methods"]["factor"],
            "geometry": report["methods"]["geometry"],
            "pre_registered_gate": gate,
        })
        print(
            f"[seed {seed}] top8_pass={gate['top8_pass']} "
            f"F8={report['methods']['factor']}", flush=True)

    all_pass = all(row["pre_registered_gate"]["top8_pass"] for row in comparisons)
    branch = (
        "carry_top8_uncertainty_into_natural_stream_collection"
        if all_pass
        else "stop_single_state_feature_expansion_collect_natural_stream_evidence"
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "train_only_top8_uncertainty_oof_complete",
        "scope": (
            "train-only model development; unknown goal kind; no development; "
            "no consumed pool; no blind; no closed-loop claim"),
        "deployment_approved": False,
        "input": {
            "phase_rows": str(args.phase_rows.resolve()),
            "phase_rows_sha256": phase_sha,
            "geometry_evidence": str(args.geometry_evidence.resolve()),
            "geometry_evidence_sha256": geometry_sha,
            "f2_report": str(args.f2_report.resolve()),
            "f2_report_sha256": f2_sha,
            "scenes": int(phase["scene"].nunique()),
            "sessions": int(phase["session_id"].nunique()),
        },
        "feature_contract": {
            "f2_existence_feature_names": list(f2_feature_names),
            "added_top8_feature_names": list(TOP8_FEATURE_NAMES),
            "candidate_feature_names": candidate_feature_names,
            "anchor_and_deployment_candidates": "unchanged top-2 pairwise ranker",
            "top8_use": "existence uncertainty summaries only",
            "model": "same standardized L2 logistic, C=0.25, balanced classes",
            "geometry_reference": {
                "top_k": GEOMETRY_TOP_K,
                "visual_floor": GEOMETRY_VISUAL_FLOOR,
            },
        },
        "fold_contract": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seeds": list(seeds),
        },
        "seed_reports": seed_reports,
        "comparisons": comparisons,
        "pre_registered_decision": {
            "all_three_seeds_pass": all_pass,
            "branch": branch,
        },
        "limits": [
            "The same train scenes informed the F8 hypothesis; this is model development, not confirmation.",
            "Top-8 changes existence observation only; it is not a wider action candidate chain.",
            "The hard-geometry reference omits the online two-plan latch.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    prediction_path = args.output_dir / "session_oof_predictions.csv"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError("output already exists")
    temporary = prediction_path.with_suffix(".csv.partial")
    pd.DataFrame(predictions).to_csv(temporary, index=False)
    os.replace(temporary, prediction_path)
    output["output"] = {
        "session_predictions": str(prediction_path.resolve()),
        "session_predictions_sha256": sha256(prediction_path),
    }
    atomic_json(report_path, output)
    print(json.dumps(output["pre_registered_decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
