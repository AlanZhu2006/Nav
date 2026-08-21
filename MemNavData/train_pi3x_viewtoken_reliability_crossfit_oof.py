#!/usr/bin/env python3
"""Scene-OOF calibrated ensemble over frozen Pi3X view tokens.

V1 pooled scores from several inner models to choose a threshold and then
transferred that threshold to a separately refit outer model.  Neural score
scales are not identifiable across those models.  This evaluator keeps every
threshold attached to the exact model that produced its calibration scores.

For each outer test fold, four inner models are trained.  Each member is
calibrated only on scenes excluded from that member's fit set, and then predicts
the untouched outer scenes.  Candidate ranking is aggregated by within-session
Borda rank (hence invariant to member score scale); takeover requires a fixed
number of member-specific calibrated votes.  The same ensemble is directly
deployable and never consumes simulator labels at inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold

from MemNavData.summarize_pi3x_multiview_shadow import (
    choose_threshold,
    evaluate_picks,
)
from MemNavData.train_pi3x_viewtoken_reliability_oof import (
    _atomic_csv,
    _atomic_json,
    _fit,
    _load,
    _predict,
    _sha256,
)


def _within_session_borda(
    rows: Sequence[dict[str, Any]],
    scores: dict[int, float],
    scenes: set[str],
) -> dict[int, float]:
    """Map each member's scores to [0, 1] ranks within each candidate set."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in scores:
        if rows[index]["scene"] in scenes:
            grouped[rows[index]["session_id"]].append(index)
    output: dict[int, float] = {}
    for indices in grouped.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                scores[index], -rows[index]["candidate_rank"]
            ),
        )
        denominator = max(len(ordered) - 1, 1)
        for position, index in enumerate(ordered):
            output[index] = position / denominator
    return output


def _ensemble_picks(
    rows: Sequence[dict[str, Any]],
    member_scores: Sequence[dict[int, float]],
    member_thresholds: Sequence[float],
    scenes: set[str],
    consensus: int,
) -> tuple[list[dict[str, Any]], dict[int, float], dict[int, int]]:
    if not member_scores or len(member_scores) != len(member_thresholds):
        raise ValueError("member scores and thresholds must be non-empty and aligned")
    if not 1 <= consensus <= len(member_scores):
        raise ValueError("consensus outside ensemble size")
    rank_maps = [
        _within_session_borda(rows, scores, scenes) for scores in member_scores
    ]
    universe = set(rank_maps[0])
    if any(set(rank_map) != universe for rank_map in rank_maps[1:]):
        raise ValueError("ensemble members predicted different row universes")
    ensemble_scores = {
        index: float(np.mean([rank_map[index] for rank_map in rank_maps]))
        for index in universe
    }
    votes = {
        index: sum(
            float(scores[index]) >= float(threshold)
            for scores, threshold in zip(member_scores, member_thresholds)
        )
        for index in universe
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in universe:
        grouped[rows[index]["session_id"]].append(index)
    picks = []
    for session_id, indices in sorted(grouped.items()):
        selected = max(
            indices,
            key=lambda index: (
                ensemble_scores[index], -rows[index]["candidate_rank"]
            ),
        )
        row = rows[selected]
        picks.append({
            "session_id": session_id,
            "scene": row["scene"],
            "session_label": row["session_label"],
            "selected_row_index": row["row_index"],
            "selected_candidate_rank": row["candidate_rank"],
            "selected_candidate_label": row["candidate_label"],
            "selected_navigation_action_label": row["navigation_action_label"],
            "bearing_error_deg": row["bearing_error_deg"],
            "score": ensemble_scores[selected],
            "member_votes": votes[selected],
            "accepted": votes[selected] >= consensus,
        })
    return picks, ensemble_scores, votes


def _aggregate(picks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = []
    for pick in picks:
        copied = dict(pick)
        copied["score"] = 1.0 if pick["accepted"] else 0.0
        normalized.append(copied)
    return evaluate_picks(
        normalized, 0.5, correctness_key="selected_navigation_action_label"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    arrays, rows = _load(args)
    scenes = np.asarray(sorted({row["scene"] for row in rows}))
    if args.inner_splits < 2:
        raise ValueError("cross-fit ensemble needs at least two inner splits")
    if not 1 <= args.consensus <= args.inner_splits:
        raise ValueError("consensus must be between one and inner_splits")
    outer = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    oof_rank_scores: dict[int, float] = {}
    oof_votes: dict[int, int] = {}
    oof_picks: list[dict[str, Any]] = []
    fold_reports = []

    for fold, (outer_train_indices, outer_test_indices) in enumerate(outer.split(scenes)):
        train_scenes = set(scenes[outer_train_indices].tolist())
        test_scenes = set(scenes[outer_test_indices].tolist())
        inner_scenes = np.asarray(sorted(train_scenes))
        inner = KFold(
            n_splits=args.inner_splits,
            shuffle=True,
            random_state=fold + 101,
        )
        member_scores: list[dict[int, float]] = []
        member_thresholds: list[float] = []
        member_reports = []
        for member, (fit_indices, calibration_indices) in enumerate(inner.split(inner_scenes)):
            fit_scenes = set(inner_scenes[fit_indices].tolist())
            calibration_scenes = set(inner_scenes[calibration_indices].tolist())
            model = _fit(
                arrays,
                rows,
                fit_scenes,
                args,
                seed=args.seed + 1000 * fold + member,
            )
            calibration_scores = _predict(
                model, arrays, rows, calibration_scenes, args
            )
            calibration_picks = _single_member_picks(
                rows, calibration_scores, calibration_scenes
            )
            threshold, calibration = choose_threshold(
                calibration_picks,
                minimum_precision=args.minimum_precision,
                maximum_fpr=args.maximum_fpr,
                correctness_key="selected_navigation_action_label",
            )
            test_scores = _predict(model, arrays, rows, test_scenes, args)
            member_scores.append(test_scores)
            member_thresholds.append(threshold)
            checkpoint = args.checkpoint_dir / f"outer_{fold}_member_{member}.pt"
            torch.save({
                "schema_version": 2,
                "outer_fold": fold,
                "ensemble_member": member,
                "fit_scenes": sorted(fit_scenes),
                "calibration_scenes": sorted(calibration_scenes),
                "outer_test_scenes": sorted(test_scenes),
                "member_calibration_threshold": threshold,
                "model_config": {
                    "input_dim": int(arrays["view_descriptors"].shape[-1]),
                    "model_dim": args.model_dim,
                    "layers": args.layers,
                    "heads": args.heads,
                },
                "state_dict": model.cpu().state_dict(),
            }, checkpoint)
            member_reports.append({
                "member": member,
                "fit_scenes": sorted(fit_scenes),
                "calibration_scenes": sorted(calibration_scenes),
                "threshold": threshold,
                "calibration": calibration,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        picks, rank_scores, votes = _ensemble_picks(
            rows,
            member_scores,
            member_thresholds,
            test_scenes,
            args.consensus,
        )
        for pick in picks:
            pick["outer_fold"] = fold
        oof_picks.extend(picks)
        oof_rank_scores.update(rank_scores)
        oof_votes.update(votes)
        consensus_ablation = {}
        for required in range(1, len(member_scores) + 1):
            alternate = [dict(pick, accepted=pick["member_votes"] >= required) for pick in picks]
            consensus_ablation[str(required)] = _aggregate(alternate)
        report = {
            "fold": fold,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "members": member_reports,
            "primary_consensus": args.consensus,
            "test": _aggregate(picks),
            "consensus_ablation_reporting_only": consensus_ablation,
        }
        fold_reports.append(report)
        print(json.dumps({
            "fold": fold,
            "test": report["test"],
            "member_thresholds": member_thresholds,
        }, sort_keys=True), flush=True)

    positive_picks = [pick for pick in oof_picks if pick["session_label"] == 1]
    consensus_ablation = {}
    for required in range(1, args.inner_splits + 1):
        alternate = [
            dict(pick, accepted=pick["member_votes"] >= required)
            for pick in oof_picks
        ]
        consensus_ablation[str(required)] = _aggregate(alternate)
    summary = {
        "schema_version": 2,
        "status": "viewtoken_head_crossfit_scene_oof_not_closed_loop_authority",
        "rows": len(rows),
        "scenes": len(scenes),
        "sessions": len({row["session_id"] for row in rows}),
        "positive_session_top1_navigation_correct": sum(
            pick["selected_navigation_action_label"] == 1 for pick in positive_picks
        ),
        "positive_sessions": len(positive_picks),
        "primary_consensus": args.consensus,
        "activation": _aggregate(oof_picks),
        "consensus_ablation_reporting_only": consensus_ablation,
        "outer_folds": fold_reports,
        "model": {
            "name": "pi3x_viewtoken_reliability_crossfit_ensemble_v2",
            "ranking": "within_session_borda_mean",
            "authorization": "member_specific_calibrated_vote",
            "ensemble_members_per_outer_fold": args.inner_splits,
            "model_dim": args.model_dim,
            "layers": args.layers,
            "heads": args.heads,
            "epochs": args.epochs,
            "batch_sessions": args.batch_sessions,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "listwise_weight": args.listwise_weight,
            "support_weight": args.support_weight,
            "pi3x_frozen": True,
            "threshold_and_model_are_bound": True,
            "outer_refit_model": False,
        },
        "targets": {
            "minimum_precision": args.minimum_precision,
            "maximum_strict_negative_fpr": args.maximum_fpr,
            "certificate_recall_reference_not_same_label_semantics": 0.7974,
        },
        "inputs": {
            "rows_csv": str(args.rows_csv),
            "rows_csv_sha256": _sha256(args.rows_csv),
            "shadow_jsonl": str(args.shadow_jsonl),
            "shadow_jsonl_sha256": _sha256(args.shadow_jsonl),
            "descriptors_npz": str(args.descriptors_npz),
            "descriptors_npz_sha256": _sha256(args.descriptors_npz),
        },
    }
    picks_by_row = {pick["selected_row_index"]: pick for pick in oof_picks}
    prediction_rows = []
    for index, row in enumerate(rows):
        pick = picks_by_row.get(index)
        prediction_rows.append({
            "row_index": index,
            "scene": row["scene"],
            "session_id": row["session_id"],
            "candidate_rank": row["candidate_rank"],
            "session_label_reporting_only": row["session_label"],
            "candidate_label_reporting_only": row["candidate_label"],
            "navigation_action_label_reporting_only": row["navigation_action_label"],
            "bearing_error_deg_reporting_only": row["bearing_error_deg"],
            "ensemble_borda_score": oof_rank_scores[index],
            "calibrated_member_votes": oof_votes[index],
            "selected": pick is not None,
            "accepted": bool(pick and pick["accepted"]),
            "outer_fold": pick["outer_fold"] if pick else "",
        })
    _atomic_json(args.output_summary, summary)
    _atomic_csv(args.output_predictions, prediction_rows)
    return summary


def _single_member_picks(
    rows: Sequence[dict[str, Any]],
    scores: dict[int, float],
    scenes: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in scores:
        if rows[index]["scene"] in scenes:
            grouped[rows[index]["session_id"]].append(index)
    picks = []
    for session_id, indices in sorted(grouped.items()):
        selected = max(
            indices,
            key=lambda index: (scores[index], -rows[index]["candidate_rank"]),
        )
        row = rows[selected]
        picks.append({
            "session_id": session_id,
            "scene": row["scene"],
            "session_label": row["session_label"],
            "selected_row_index": row["row_index"],
            "selected_candidate_rank": row["candidate_rank"],
            "selected_candidate_label": row["candidate_label"],
            "selected_navigation_action_label": row["navigation_action_label"],
            "bearing_error_deg": row["bearing_error_deg"],
            "score": scores[selected],
        })
    return picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--shadow-jsonl", type=Path, required=True)
    parser.add_argument("--descriptors-npz", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-rows-sha256")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--consensus", type=int, default=3)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--maximum-fpr", type=float, default=0.0275)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-sessions", type=int, default=24)
    parser.add_argument("--inference-batch-rows", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--listwise-weight", type=float, default=0.5)
    parser.add_argument("--support-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "top1": result["positive_session_top1_navigation_correct"],
        "activation": result["activation"],
    }, sort_keys=True))
