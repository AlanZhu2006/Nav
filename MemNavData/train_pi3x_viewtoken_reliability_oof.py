#!/usr/bin/env python3
"""Scene-nested OOF reliability head over frozen Pi3X view tokens.

This is the first learned replacement for the hand-computed overlap/certificate
decision.  Pi3X and DINO remain frozen.  The head sees only label-blind tokens,
view roles, and causal relative age; simulator labels are loss/evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F

from MemNavData.summarize_pi3x_multiview_shadow import (
    choose_threshold,
    evaluate_picks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class ViewTokenReliabilityHead(nn.Module):
    """Small role-aware set/sequence head; the Pi3X backbone stays frozen."""

    def __init__(self, input_dim: int, model_dim: int = 64,
                 layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, model_dim)
        # Stored roles are -1 padding, 0 current, 1 bridge, 2 anchor, 3 goal.
        self.role_embedding = nn.Embedding(5, model_dim)
        self.age_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.action_head = nn.Linear(model_dim, 1)
        self.support_head = nn.Linear(model_dim, 1)

    def forward(self, descriptors: torch.Tensor, roles: torch.Tensor,
                relative_age: torch.Tensor,
                valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 3 or valid.shape != descriptors.shape[:2]:
            raise ValueError("invalid padded view-token batch")
        role_ids = (roles + 1).clamp(min=0, max=4)
        encoded = self.input_projection(self.input_norm(descriptors))
        encoded = encoded + self.role_embedding(role_ids)
        encoded = encoded + self.age_projection(relative_age.unsqueeze(-1))
        cls = self.cls.expand(len(encoded), -1, -1)
        encoded = torch.cat([cls, encoded], dim=1)
        padding = torch.cat([
            torch.zeros((len(valid), 1), dtype=torch.bool, device=valid.device),
            ~valid,
        ], dim=1)
        pooled = self.output_norm(self.encoder(
            encoded, src_key_padding_mask=padding
        )[:, 0])
        return self.action_head(pooled).squeeze(-1), self.support_head(pooled).squeeze(-1)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if args.expected_rows_sha256 and _sha256(args.rows_csv) != args.expected_rows_sha256:
        raise ValueError("source rows SHA mismatch")
    with args.rows_csv.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    shadow = [json.loads(line) for line in args.shadow_jsonl.read_text().splitlines() if line]
    by_index = {int(row["row_index"]): row for row in shadow}
    if len(by_index) != len(shadow):
        raise ValueError("duplicate shadow row_index")
    with np.load(args.descriptors_npz) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "row_indices", "view_counts", "view_descriptors", "view_roles",
        "view_relative_age", "view_valid",
    }
    if set(arrays) != required:
        raise ValueError(f"descriptor fields differ: {set(arrays)}")
    row_indices = arrays["row_indices"].astype(int).tolist()
    if len(set(row_indices)) != len(source) or set(row_indices) != set(range(len(source))):
        raise ValueError("descriptor row universe differs from source CSV")
    order = np.argsort(arrays["row_indices"])
    arrays = {name: value[order] for name, value in arrays.items()}
    rows = []
    for index, original in enumerate(source):
        prediction = by_index.get(index)
        if prediction is None or prediction["scene"] != original["scene"]:
            raise ValueError(f"missing or mismatched shadow row {index}")
        bearing_error = float(prediction["goal_bearing_error_deg_reporting_only"])
        session_label = int(original["session_label"])
        rows.append({
            "row_index": index,
            "scene": original["scene"],
            "session_id": original["session_id"],
            "candidate_rank": int(original["candidate_rank"]),
            "dino_cosine": float(original["dino_cosine"]),
            "candidate_label": int(original["candidate_label"]),
            "session_label": session_label,
            "bearing_error_deg": bearing_error,
            "raw_pi3x_overlap": float(prediction["best_view_f1_20cm"]),
            "navigation_action_label": (
                int(math.isfinite(bearing_error) and bearing_error <= 30.0)
                if session_label == 1 else (0 if session_label == 0 else -1)
            ),
        })
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"found {len(rows)} rows, expected {args.expected_rows}")
    return arrays, rows


def _session_indices(rows: Sequence[dict[str, Any]], scenes: set[str]) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["scene"] in scenes and row["session_label"] in (0, 1):
            grouped[row["session_id"]].append(index)
    return [grouped[key] for key in sorted(grouped)]


def _batch(arrays: dict[str, np.ndarray], indices: Sequence[int],
           device: torch.device) -> tuple[torch.Tensor, ...]:
    index = np.asarray(indices, dtype=np.int64)
    return (
        torch.as_tensor(arrays["view_descriptors"][index], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["view_roles"][index], device=device, dtype=torch.long),
        torch.as_tensor(arrays["view_relative_age"][index], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["view_valid"][index], device=device, dtype=torch.bool),
    )


def _fit(
    arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    args: argparse.Namespace,
    *,
    seed: int,
) -> ViewTokenReliabilityHead:
    _seed(seed)
    device = torch.device(args.device)
    model = ViewTokenReliabilityHead(
        int(arrays["view_descriptors"].shape[-1]),
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    sessions = _session_indices(rows, scenes)
    if not sessions:
        raise ValueError("empty model-training scene set")
    training_indices = [index for session in sessions for index in session]
    positive = sum(rows[index]["navigation_action_label"] == 1 for index in training_indices)
    negative = sum(rows[index]["navigation_action_label"] == 0 for index in training_indices)
    if not positive or not negative:
        raise ValueError("training split lacks both action classes")
    action_pos_weight = torch.tensor(negative / positive, device=device)
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
            action_logits, support_logits = model(*_batch(arrays, indices, device))
            action_targets = torch.tensor([
                rows[index]["navigation_action_label"] for index in indices
            ], dtype=torch.float32, device=device)
            action_loss = F.binary_cross_entropy_with_logits(
                action_logits, action_targets, pos_weight=action_pos_weight
            )
            support_targets = torch.tensor([
                max(rows[index]["candidate_label"], 0) for index in indices
            ], dtype=torch.float32, device=device)
            support_mask = torch.tensor([
                rows[index]["candidate_label"] in (0, 1) for index in indices
            ], dtype=torch.bool, device=device)
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits[support_mask], support_targets[support_mask],
                pos_weight=support_pos_weight,
            )
            listwise_terms = []
            offset = 0
            for session in chosen_sessions:
                count = len(session)
                labels = action_targets[offset:offset + count]
                logits = action_logits[offset:offset + count]
                if labels.sum() > 0:
                    target = labels / labels.sum()
                    listwise_terms.append(-(target * F.log_softmax(logits, dim=0)).sum())
                offset += count
            listwise_loss = (
                torch.stack(listwise_terms).mean()
                if listwise_terms else action_loss.new_zeros(())
            )
            loss = (
                action_loss
                + args.listwise_weight * listwise_loss
                + args.support_weight * support_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def _predict(
    model: ViewTokenReliabilityHead,
    arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    args: argparse.Namespace,
) -> dict[int, float]:
    device = torch.device(args.device)
    indices = [index for index, row in enumerate(rows) if row["scene"] in scenes]
    scores: dict[int, float] = {}
    for start in range(0, len(indices), args.inference_batch_rows):
        chosen = indices[start:start + args.inference_batch_rows]
        logits, _support = model(*_batch(arrays, chosen, device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
        scores.update({index: float(score) for index, score in zip(chosen, probabilities)})
    return scores


def _picks(rows: Sequence[dict[str, Any]], scores: dict[int, float],
           scenes: set[str] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in scores:
        if scenes is None or rows[index]["scene"] in scenes:
            grouped[rows[index]["session_id"]].append(index)
    output = []
    for session_id, indices in sorted(grouped.items()):
        selected = max(
            indices,
            key=lambda index: (scores[index], -rows[index]["candidate_rank"]),
        )
        row = rows[selected]
        output.append({
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
    return output


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
    outer = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
    oof_scores: dict[int, float] = {}
    oof_picks = []
    fold_reports = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fold, (outer_train_indices, outer_test_indices) in enumerate(outer.split(scenes)):
        train_scenes = set(scenes[outer_train_indices].tolist())
        test_scenes = set(scenes[outer_test_indices].tolist())
        inner_scene_array = np.asarray(sorted(train_scenes))
        inner = KFold(
            n_splits=args.inner_splits, shuffle=True, random_state=fold + 101
        )
        inner_scores: dict[int, float] = {}
        for inner_fold, (fit_indices, validation_indices) in enumerate(inner.split(inner_scene_array)):
            fit_scenes = set(inner_scene_array[fit_indices].tolist())
            validation_scenes = set(inner_scene_array[validation_indices].tolist())
            model = _fit(
                arrays, rows, fit_scenes, args,
                seed=args.seed + 1000 * fold + inner_fold,
            )
            inner_scores.update(_predict(
                model, arrays, rows, validation_scenes, args
            ))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        inner_picks = _picks(rows, inner_scores, train_scenes)
        threshold, calibration = choose_threshold(
            inner_picks,
            minimum_precision=args.minimum_precision,
            maximum_fpr=args.maximum_fpr,
            correctness_key="selected_navigation_action_label",
        )
        model = _fit(
            arrays, rows, train_scenes, args,
            seed=args.seed + 10_000 + fold,
        )
        checkpoint = args.checkpoint_dir / f"outer_fold_{fold}.pt"
        torch.save({
            "schema_version": 1,
            "fold": fold,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "threshold": threshold,
            "model_config": {
                "input_dim": int(arrays["view_descriptors"].shape[-1]),
                "model_dim": args.model_dim,
                "layers": args.layers,
                "heads": args.heads,
            },
            "state_dict": model.cpu().state_dict(),
        }, checkpoint)
        model = model.to(args.device)
        test_scores = _predict(model, arrays, rows, test_scenes, args)
        oof_scores.update(test_scores)
        test_picks = _picks(rows, test_scores, test_scenes)
        for pick in test_picks:
            pick["outer_fold"] = fold
            pick["fold_threshold"] = threshold
            pick["accepted"] = pick["score"] >= threshold
        oof_picks.extend(test_picks)
        fold_reports.append({
            "fold": fold,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "threshold_from_inner_scene_oof": threshold,
            "inner_calibration": calibration,
            "test": evaluate_picks(
                test_picks, threshold,
                correctness_key="selected_navigation_action_label",
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        })
        print(json.dumps({
            "fold": fold,
            "threshold": threshold,
            "test": fold_reports[-1]["test"],
        }, sort_keys=True), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    known = [
        index for index, row in enumerate(rows)
        if row["navigation_action_label"] in (0, 1)
    ]
    labels = np.asarray([rows[index]["navigation_action_label"] for index in known])
    scores = np.asarray([oof_scores[index] for index in known])
    positive_picks = [pick for pick in oof_picks if pick["session_label"] == 1]
    summary = {
        "schema_version": 1,
        "status": "viewtoken_head_nested_scene_oof_not_closed_loop_authority",
        "rows": len(rows),
        "scenes": len(scenes),
        "sessions": len({row["session_id"] for row in rows}),
        "candidate_navigation_roc_auc": float(roc_auc_score(labels, scores)),
        "candidate_navigation_average_precision": float(
            average_precision_score(labels, scores)
        ),
        "positive_session_top1_navigation_correct": sum(
            pick["selected_navigation_action_label"] == 1 for pick in positive_picks
        ),
        "positive_sessions": len(positive_picks),
        "activation": _aggregate(oof_picks),
        "outer_folds": fold_reports,
        "model": {
            "name": "pi3x_viewtoken_reliability_head_v1",
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
    prediction_rows = []
    pick_by_row = {pick["selected_row_index"]: pick for pick in oof_picks}
    for index, row in enumerate(rows):
        pick = pick_by_row.get(index)
        prediction_rows.append({
            "row_index": index,
            "scene": row["scene"],
            "session_id": row["session_id"],
            "candidate_rank": row["candidate_rank"],
            "session_label_reporting_only": row["session_label"],
            "candidate_label_reporting_only": row["candidate_label"],
            "navigation_action_label_reporting_only": row["navigation_action_label"],
            "bearing_error_deg_reporting_only": row["bearing_error_deg"],
            "oof_score": oof_scores[index],
            "selected": pick is not None,
            "accepted": bool(pick and pick["accepted"]),
            "outer_fold": pick["outer_fold"] if pick else "",
            "fold_threshold": pick["fold_threshold"] if pick else "",
        })
    _atomic_json(args.output_summary, summary)
    _atomic_csv(args.output_predictions, prediction_rows)
    return summary


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
