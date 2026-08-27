#!/usr/bin/env python3
"""Cross-fitted learned proof over spatial Pi3X evidence.

The strongest frozen Pi3X overlap proposal is held fixed so this experiment
isolates certificate replacement.  A learned role-aware spatial encoder sees
only Pi3X point grids, confidence, relative cameras, global view tokens and
causal metadata.  Each ensemble member remains bound to its own scene-held-out
calibration threshold.  No learned output is authorized for closed loop here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.nn import functional as F

from MemNavData.pi3x_spatial_reliability_model import (
    Pi3XSpatialReliabilityHead,
)
from MemNavData.summarize_pi3x_multiview_shadow import (
    choose_threshold,
    evaluate_picks,
)
from MemNavData.train_pi3x_viewtoken_reliability_oof import (
    _atomic_csv,
    _atomic_json,
    _batch,
    _load,
    _seed,
    _sha256,
    _session_indices,
)


SPATIAL_FIELDS = {
    "row_indices",
    "view_counts",
    "view_world_points_in_current",
    "view_local_points",
    "view_confidence",
    "view_poses_in_current",
    "view_roles",
    "view_relative_age",
    "view_valid",
    "normalization_scale",
}


def _load_spatial(path: Path, expected_sha256: str | None,
                  expected_rows: int) -> dict[str, np.ndarray]:
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise ValueError("spatial archive SHA mismatch")
    with np.load(path) as archive:
        if set(archive.files) != SPATIAL_FIELDS:
            raise ValueError(f"spatial fields differ: {set(archive.files)}")
        arrays = {name: archive[name] for name in archive.files}
    order = np.argsort(arrays["row_indices"])
    arrays = {name: value[order] for name, value in arrays.items()}
    if arrays["row_indices"].tolist() != list(range(expected_rows)):
        raise ValueError("spatial rows do not match the frozen CSV order")
    if arrays["view_world_points_in_current"].shape[:2] != (
        expected_rows, arrays["view_valid"].shape[1]
    ):
        raise ValueError("spatial archive row/view shapes differ")
    return arrays


def _spatial_batch(
    global_arrays: dict[str, np.ndarray],
    spatial_arrays: dict[str, np.ndarray],
    indices: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    descriptors, roles, age, valid = _batch(global_arrays, indices, device)
    index = np.asarray(indices, dtype=np.int64)
    spatial_roles = torch.as_tensor(
        spatial_arrays["view_roles"][index], device=device, dtype=torch.long
    )
    spatial_age = torch.as_tensor(
        spatial_arrays["view_relative_age"][index], device=device,
        dtype=torch.float32,
    )
    spatial_valid = torch.as_tensor(
        spatial_arrays["view_valid"][index], device=device, dtype=torch.bool
    )
    if not (torch.equal(roles, spatial_roles)
            and torch.equal(valid, spatial_valid)
            and torch.allclose(age, spatial_age)):
        raise ValueError("global and spatial view metadata differ")
    return (
        descriptors,
        roles,
        age,
        valid,
        torch.as_tensor(
            spatial_arrays["view_world_points_in_current"][index],
            device=device, dtype=torch.float32,
        ),
        torch.as_tensor(
            spatial_arrays["view_local_points"][index],
            device=device, dtype=torch.float32,
        ),
        torch.as_tensor(
            spatial_arrays["view_confidence"][index],
            device=device, dtype=torch.float32,
        ),
        torch.as_tensor(
            spatial_arrays["view_poses_in_current"][index],
            device=device, dtype=torch.float32,
        ),
    )


def _fit(
    global_arrays: dict[str, np.ndarray],
    spatial_arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    args: argparse.Namespace,
    *,
    seed: int,
) -> Pi3XSpatialReliabilityHead:
    _seed(seed)
    device = torch.device(args.device)
    model = Pi3XSpatialReliabilityHead(
        int(global_arrays["view_descriptors"].shape[-1]),
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    sessions = _session_indices(rows, scenes)
    training_indices = [index for session in sessions for index in session]
    positive = sum(rows[index]["navigation_action_label"] == 1 for index in training_indices)
    negative = len(training_indices) - positive
    action_pos_weight = torch.tensor(negative / max(positive, 1), device=device)
    support_known = [
        index for index in training_indices if rows[index]["candidate_label"] in (0, 1)
    ]
    support_positive = sum(rows[index]["candidate_label"] == 1 for index in support_known)
    support_negative = len(support_known) - support_positive
    support_pos_weight = torch.tensor(
        support_negative / max(support_positive, 1), device=device
    )
    generator = np.random.default_rng(seed)
    model.train()
    for _epoch in range(args.epochs):
        order = generator.permutation(len(sessions))
        for start in range(0, len(order), args.batch_sessions):
            chosen_sessions = [sessions[i] for i in order[start:start + args.batch_sessions]]
            indices = [index for session in chosen_sessions for index in session]
            action_logits, support_logits = model(*_spatial_batch(
                global_arrays, spatial_arrays, indices, device
            ))
            action_targets = torch.tensor([
                rows[index]["navigation_action_label"] for index in indices
            ], dtype=torch.float32, device=device)
            support_targets = torch.tensor([
                max(rows[index]["candidate_label"], 0) for index in indices
            ], dtype=torch.float32, device=device)
            support_mask = torch.tensor([
                rows[index]["candidate_label"] in (0, 1) for index in indices
            ], dtype=torch.bool, device=device)
            action_loss = F.binary_cross_entropy_with_logits(
                action_logits, action_targets, pos_weight=action_pos_weight
            )
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits[support_mask], support_targets[support_mask],
                pos_weight=support_pos_weight,
            )
            loss = action_loss + args.support_weight * support_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def _predict(
    model: Pi3XSpatialReliabilityHead,
    global_arrays: dict[str, np.ndarray],
    spatial_arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    args: argparse.Namespace,
) -> dict[int, float]:
    device = torch.device(args.device)
    indices = [index for index, row in enumerate(rows) if row["scene"] in scenes]
    scores = {}
    for start in range(0, len(indices), args.inference_batch_rows):
        chosen = indices[start:start + args.inference_batch_rows]
        logits, _support = model(*_spatial_batch(
            global_arrays, spatial_arrays, chosen, device
        ))
        probabilities = torch.sigmoid(logits).cpu().numpy()
        scores.update({index: float(value) for index, value in zip(chosen, probabilities)})
    return scores


def _fixed_proposal_picks(
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
            key=lambda index: (
                rows[index]["raw_pi3x_overlap"],
                -rows[index]["candidate_rank"],
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
            "raw_pi3x_overlap": row["raw_pi3x_overlap"],
            "score": scores[selected],
        })
    return picks


def _aggregate(picks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = [dict(pick, score=1.0 if pick["accepted"] else 0.0) for pick in picks]
    return evaluate_picks(
        normalized, 0.5, correctness_key="selected_navigation_action_label"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    global_arrays, rows = _load(args)
    spatial_arrays = _load_spatial(
        args.spatial_npz, args.expected_spatial_sha256, len(rows)
    )
    scenes = np.asarray(sorted({row["scene"] for row in rows}))
    outer = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    oof_picks: list[dict[str, Any]] = []
    oof_votes: dict[int, int] = {}
    fold_reports = []
    for fold, (train_indices, test_indices) in enumerate(outer.split(scenes)):
        train_scenes = set(scenes[train_indices].tolist())
        test_scenes = set(scenes[test_indices].tolist())
        inner_scenes = np.asarray(sorted(train_scenes))
        inner = KFold(
            n_splits=args.inner_splits, shuffle=True, random_state=fold + 101
        )
        member_scores = []
        member_thresholds = []
        member_reports = []
        for member, (fit_indices, calibration_indices) in enumerate(inner.split(inner_scenes)):
            fit_scenes = set(inner_scenes[fit_indices].tolist())
            calibration_scenes = set(inner_scenes[calibration_indices].tolist())
            model = _fit(
                global_arrays, spatial_arrays, rows, fit_scenes, args,
                seed=args.seed + 1000 * fold + member,
            )
            calibration_scores = _predict(
                model, global_arrays, spatial_arrays, rows, calibration_scenes, args
            )
            calibration_picks = _fixed_proposal_picks(
                rows, calibration_scores, calibration_scenes
            )
            threshold, calibration = choose_threshold(
                calibration_picks,
                minimum_precision=args.minimum_precision,
                maximum_fpr=args.maximum_fpr,
                correctness_key="selected_navigation_action_label",
            )
            test_scores = _predict(
                model, global_arrays, spatial_arrays, rows, test_scenes, args
            )
            member_scores.append(test_scores)
            member_thresholds.append(threshold)
            checkpoint = args.checkpoint_dir / f"outer_{fold}_member_{member}.pt"
            torch.save({
                "schema_version": 1,
                "outer_fold": fold,
                "ensemble_member": member,
                "fit_scenes": sorted(fit_scenes),
                "calibration_scenes": sorted(calibration_scenes),
                "outer_test_scenes": sorted(test_scenes),
                "member_calibration_threshold": threshold,
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
        base_picks = _fixed_proposal_picks(rows, member_scores[0], test_scenes)
        for pick in base_picks:
            index = pick["selected_row_index"]
            votes = sum(
                scores[index] >= threshold
                for scores, threshold in zip(member_scores, member_thresholds)
            )
            pick["member_votes"] = votes
            pick["accepted"] = votes >= args.consensus
            pick["outer_fold"] = fold
            oof_votes[index] = votes
        oof_picks.extend(base_picks)
        consensus_reports = {}
        for required in range(1, args.inner_splits + 1):
            consensus_reports[str(required)] = _aggregate([
                dict(pick, accepted=pick["member_votes"] >= required)
                for pick in base_picks
            ])
        report = {
            "fold": fold,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "members": member_reports,
            "test": _aggregate(base_picks),
            "consensus_ablation_reporting_only": consensus_reports,
        }
        fold_reports.append(report)
        print(json.dumps({
            "fold": fold,
            "member_thresholds": member_thresholds,
            "test": report["test"],
        }, sort_keys=True), flush=True)
    positive = [pick for pick in oof_picks if pick["session_label"] == 1]
    consensus_reports = {}
    for required in range(1, args.inner_splits + 1):
        consensus_reports[str(required)] = _aggregate([
            dict(pick, accepted=pick["member_votes"] >= required)
            for pick in oof_picks
        ])
    summary = {
        "schema_version": 1,
        "status": "pi3x_spatial_learned_proof_crossfit_scene_oof_not_closed_loop_authority",
        "rows": len(rows),
        "scenes": len(scenes),
        "sessions": len(oof_picks),
        "proposal": "frozen_raw_pi3x_overlap_top1",
        "positive_session_top1_navigation_correct": sum(
            pick["selected_navigation_action_label"] == 1 for pick in positive
        ),
        "positive_sessions": len(positive),
        "primary_consensus": args.consensus,
        "activation": _aggregate(oof_picks),
        "consensus_ablation_reporting_only": consensus_reports,
        "outer_folds": fold_reports,
        "model": {
            "name": "pi3x_spatial_reliability_head_v1",
            "pi3x_frozen": True,
            "global_register_tokens": True,
            "scale_free_patch_point_grids": True,
            "relative_camera_poses": True,
            "model_dim": args.model_dim,
            "layers": args.layers,
            "heads": args.heads,
            "epochs": args.epochs,
            "support_weight": args.support_weight,
            "threshold_and_model_are_bound": True,
        },
        "targets": {
            "minimum_precision": args.minimum_precision,
            "maximum_strict_negative_fpr": args.maximum_fpr,
            "certificate_recall_reference_not_same_label_semantics": 0.7974,
            "zero_accepted_catastrophic_gt90deg": True,
        },
        "inputs": {
            "rows_csv_sha256": _sha256(args.rows_csv),
            "shadow_jsonl_sha256": _sha256(args.shadow_jsonl),
            "descriptors_npz_sha256": _sha256(args.descriptors_npz),
            "spatial_npz_sha256": _sha256(args.spatial_npz),
        },
    }
    pick_by_row = {pick["selected_row_index"]: pick for pick in oof_picks}
    predictions = []
    for index, row in enumerate(rows):
        pick = pick_by_row.get(index)
        predictions.append({
            "row_index": index,
            "scene": row["scene"],
            "session_id": row["session_id"],
            "candidate_rank": row["candidate_rank"],
            "session_label_reporting_only": row["session_label"],
            "navigation_action_label_reporting_only": row["navigation_action_label"],
            "bearing_error_deg_reporting_only": row["bearing_error_deg"],
            "raw_pi3x_overlap": row["raw_pi3x_overlap"],
            "selected": pick is not None,
            "calibrated_member_votes": pick["member_votes"] if pick else "",
            "accepted": bool(pick and pick["accepted"]),
            "outer_fold": pick["outer_fold"] if pick else "",
        })
    _atomic_json(args.output_summary, summary)
    _atomic_csv(args.output_predictions, predictions)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--shadow-jsonl", type=Path, required=True)
    parser.add_argument("--descriptors-npz", type=Path, required=True)
    parser.add_argument("--spatial-npz", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=3840)
    parser.add_argument("--expected-rows-sha256")
    parser.add_argument("--expected-spatial-sha256")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--consensus", type=int, default=2)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--maximum-fpr", type=float, default=0.0275)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-sessions", type=int, default=12)
    parser.add_argument("--inference-batch-rows", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
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
