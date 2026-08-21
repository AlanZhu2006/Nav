#!/usr/bin/env python3
"""Unified top-K learned relocalizer with an explicit reject hypothesis.

The frozen Pi3X descriptors encode each DINO-shortlisted candidate's current,
causal-bridge, anchor and ImageGoal views.  This head reasons over all candidates
jointly and scores K historical hypotheses plus one learned REJECT token.  It
therefore selects an actionable bearing or abstains in one relative decision,
without a separately transferred confidence threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F

from MemNavData.summarize_pi3x_multiview_shadow import evaluate_picks
from MemNavData.train_pi3x_viewtoken_reliability_oof import (
    _atomic_csv,
    _atomic_json,
    _batch,
    _load,
    _seed,
    _sha256,
)


class Pi3XSetRelocalizer(nn.Module):
    """Role-aware candidate encoder followed by a top-K + reject transformer."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 64,
        view_layers: int = 2,
        set_layers: int = 2,
        heads: int = 4,
        max_candidates: int = 8,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_candidates = max_candidates
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.role_embedding = nn.Embedding(5, model_dim)
        self.age_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.view_cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.reject_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.view_cls, std=0.02)
        nn.init.normal_(self.reject_token, std=0.02)
        view_block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_encoder = nn.TransformerEncoder(view_block, num_layers=view_layers)
        self.view_norm = nn.LayerNorm(model_dim)
        self.rank_embedding = nn.Embedding(max_candidates, model_dim)
        self.dino_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        set_block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(set_block, num_layers=set_layers)
        self.set_norm = nn.LayerNorm(model_dim)
        self.hypothesis_head = nn.Linear(model_dim, 1)
        self.support_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        descriptors: torch.Tensor,
        roles: torch.Tensor,
        relative_age: torch.Tensor,
        valid: torch.Tensor,
        dino_cosine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 4:
            raise ValueError("expected [batch, candidates, views, descriptor_dim]")
        batch, candidates, views, input_dim = descriptors.shape
        if candidates > self.max_candidates:
            raise ValueError("candidate count exceeds configured maximum")
        flat_descriptors = descriptors.reshape(batch * candidates, views, input_dim)
        flat_roles = roles.reshape(batch * candidates, views)
        flat_age = relative_age.reshape(batch * candidates, views)
        flat_valid = valid.reshape(batch * candidates, views)
        role_ids = (flat_roles + 1).clamp(min=0, max=4)
        encoded = self.input_projection(self.input_norm(flat_descriptors))
        encoded = encoded + self.role_embedding(role_ids)
        encoded = encoded + self.age_projection(flat_age.unsqueeze(-1))
        cls = self.view_cls.expand(batch * candidates, -1, -1)
        encoded = torch.cat([cls, encoded], dim=1)
        padding = torch.cat([
            torch.zeros(
                (batch * candidates, 1), dtype=torch.bool, device=valid.device
            ),
            ~flat_valid,
        ], dim=1)
        candidate = self.view_norm(self.view_encoder(
            encoded, src_key_padding_mask=padding
        )[:, 0]).reshape(batch, candidates, self.model_dim)
        ranks = torch.arange(candidates, device=descriptors.device)
        candidate = candidate + self.rank_embedding(ranks)[None]
        candidate = candidate + self.dino_projection(dino_cosine.unsqueeze(-1))
        reject = self.reject_token.expand(batch, -1, -1)
        hypotheses = self.set_norm(self.set_encoder(torch.cat([candidate, reject], dim=1)))
        logits = self.hypothesis_head(hypotheses).squeeze(-1)
        support_logits = self.support_head(candidate).squeeze(-1)
        return logits, support_logits


def _multi_target_nll(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != target_mask.shape or target_mask.dtype != torch.bool:
        raise ValueError("invalid hypothesis target mask")
    if not target_mask.any(dim=1).all():
        raise ValueError("each session needs at least one target hypothesis")
    target_logits = logits.masked_fill(~target_mask, -torch.inf)
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(
        target_logits, dim=1
    )).mean()


def _sessions(
    rows: Sequence[dict[str, Any]], scenes: set[str], *,
    include_ambiguous: bool = False,
) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        allowed = (0, 1, -1) if include_ambiguous else (0, 1)
        if row["scene"] in scenes and row["session_label"] in allowed:
            grouped[row["session_id"]].append(index)
    sessions = []
    for session_id in sorted(grouped):
        indices = sorted(grouped[session_id], key=lambda i: rows[i]["candidate_rank"])
        ranks = [rows[i]["candidate_rank"] for i in indices]
        # The frozen shortlist retains ranks from the wider retrieval chain
        # (for example 2..9), rather than renumbering its eight rows to 0..7.
        # Their order is observable; only uniqueness is required here.
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"duplicate candidate ranks in {session_id}: {ranks}")
        sessions.append(indices)
    counts = {len(indices) for indices in sessions}
    if len(counts) != 1:
        raise ValueError(f"variable candidate counts are not supported: {counts}")
    return sessions


def _session_batch(
    arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    sessions: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    batch = len(sessions)
    candidates = len(sessions[0])
    flat = [index for session in sessions for index in session]
    descriptors, roles, age, valid = _batch(arrays, flat, device)
    descriptors = descriptors.reshape(batch, candidates, *descriptors.shape[1:])
    roles = roles.reshape(batch, candidates, *roles.shape[1:])
    age = age.reshape(batch, candidates, *age.shape[1:])
    valid = valid.reshape(batch, candidates, *valid.shape[1:])
    dino = torch.tensor([
        [rows[index]["dino_cosine"] for index in session] for session in sessions
    ], dtype=torch.float32, device=device)
    return descriptors, roles, age, valid, dino


def _fit(
    arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    args: argparse.Namespace,
    *,
    seed: int,
) -> Pi3XSetRelocalizer:
    _seed(seed)
    device = torch.device(args.device)
    sessions = _sessions(rows, scenes)
    if not sessions:
        raise ValueError("empty training scene set")
    candidates = len(sessions[0])
    model = Pi3XSetRelocalizer(
        int(arrays["view_descriptors"].shape[-1]),
        model_dim=args.model_dim,
        view_layers=args.view_layers,
        set_layers=args.set_layers,
        heads=args.heads,
        max_candidates=candidates,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(seed)
    model.train()
    for _epoch in range(args.epochs):
        order = generator.permutation(len(sessions))
        for start in range(0, len(order), args.batch_sessions):
            selected_sessions = [
                sessions[index] for index in order[start:start + args.batch_sessions]
            ]
            logits, support_logits = model(*_session_batch(
                arrays, rows, selected_sessions, device
            ))
            batch, candidate_count = support_logits.shape
            target = torch.zeros(
                (batch, candidate_count + 1), dtype=torch.bool, device=device
            )
            support_targets = torch.zeros_like(support_logits)
            support_known = torch.zeros_like(support_logits, dtype=torch.bool)
            for session_index, session in enumerate(selected_sessions):
                actionable = [
                    position for position, row_index in enumerate(session)
                    if rows[row_index]["navigation_action_label"] == 1
                ]
                if actionable:
                    target[session_index, actionable] = True
                else:
                    target[session_index, candidate_count] = True
                for position, row_index in enumerate(session):
                    label = rows[row_index]["candidate_label"]
                    if label in (0, 1):
                        support_known[session_index, position] = True
                        support_targets[session_index, position] = float(label)
            decision_loss = _multi_target_nll(logits, target)
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits[support_known], support_targets[support_known]
            )
            loss = decision_loss + args.support_weight * support_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def _predict_sessions(
    model: Pi3XSetRelocalizer,
    arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    scenes: set[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    sessions = _sessions(rows, scenes, include_ambiguous=True)
    picks = []
    for start in range(0, len(sessions), args.inference_batch_sessions):
        chosen = sessions[start:start + args.inference_batch_sessions]
        logits, _support = model(*_session_batch(arrays, rows, chosen, device))
        logits = logits.cpu().numpy()
        for session, session_logits in zip(chosen, logits):
            candidate_logits = session_logits[:-1]
            selected_position = max(
                range(len(session)),
                key=lambda position: (
                    float(candidate_logits[position]),
                    -rows[session[position]]["candidate_rank"],
                ),
            )
            selected_index = session[selected_position]
            row = rows[selected_index]
            margin = float(candidate_logits[selected_position] - session_logits[-1])
            picks.append({
                "session_id": row["session_id"],
                "scene": row["scene"],
                "session_label": row["session_label"],
                "selected_row_index": row["row_index"],
                "selected_candidate_rank": row["candidate_rank"],
                "selected_candidate_label": row["candidate_label"],
                "selected_navigation_action_label": row["navigation_action_label"],
                "bearing_error_deg": row["bearing_error_deg"],
                "candidate_vs_reject_margin": margin,
                "score": 1.0 if margin > 0.0 else 0.0,
                "accepted": margin > 0.0,
            })
    return picks


def _aggregate(picks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return evaluate_picks(
        picks, 0.5, correctness_key="selected_navigation_action_label"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    arrays, rows = _load(args)
    scenes = np.asarray(sorted({row["scene"] for row in rows}))
    candidate_counts = {
        len(session) for session in _sessions(rows, set(scenes.tolist()))
    }
    if candidate_counts != {args.expected_candidates}:
        raise ValueError(
            f"candidate-count contract differs: {candidate_counts} != "
            f"{{{args.expected_candidates}}}"
        )
    outer = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    oof_picks = []
    fold_reports = []
    for fold, (train_indices, test_indices) in enumerate(outer.split(scenes)):
        train_scenes = set(scenes[train_indices].tolist())
        test_scenes = set(scenes[test_indices].tolist())
        model = _fit(
            arrays, rows, train_scenes, args, seed=args.seed + 1000 * fold
        )
        checkpoint = args.checkpoint_dir / f"outer_fold_{fold}.pt"
        torch.save({
            "schema_version": 1,
            "fold": fold,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "decision_rule": "max_candidate_logit_gt_reject_logit",
            "model_config": {
                "input_dim": int(arrays["view_descriptors"].shape[-1]),
                "model_dim": args.model_dim,
                "view_layers": args.view_layers,
                "set_layers": args.set_layers,
                "heads": args.heads,
                "max_candidates": args.expected_candidates,
            },
            "state_dict": model.cpu().state_dict(),
        }, checkpoint)
        model = model.to(args.device)
        picks = _predict_sessions(model, arrays, rows, test_scenes, args)
        for pick in picks:
            pick["outer_fold"] = fold
        oof_picks.extend(picks)
        report = {
            "fold": fold,
            "train_scenes": sorted(train_scenes),
            "test_scenes": sorted(test_scenes),
            "test": _aggregate(picks),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        }
        fold_reports.append(report)
        print(json.dumps({"fold": fold, "test": report["test"]}, sort_keys=True), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    positive = [pick for pick in oof_picks if pick["session_label"] == 1]
    summary = {
        "schema_version": 1,
        "status": "pi3x_set_relocalizer_scene_oof_not_closed_loop_authority",
        "rows": len(rows),
        "scenes": len(scenes),
        "sessions": len(oof_picks),
        "positive_session_top1_navigation_correct": sum(
            pick["selected_navigation_action_label"] == 1 for pick in positive
        ),
        "positive_sessions": len(positive),
        "activation": _aggregate(oof_picks),
        "outer_folds": fold_reports,
        "model": {
            "name": "pi3x_top8_plus_reject_set_relocalizer_v1",
            "decision": "joint_K_plus_REJECT_argmax",
            "separate_activation_threshold": False,
            "model_dim": args.model_dim,
            "view_layers": args.view_layers,
            "set_layers": args.set_layers,
            "heads": args.heads,
            "epochs": args.epochs,
            "support_weight": args.support_weight,
            "pi3x_frozen": True,
            "dino_shortlist_frozen": True,
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
            "selected": pick is not None,
            "accepted": bool(pick and pick["accepted"]),
            "candidate_vs_reject_margin": (
                pick["candidate_vs_reject_margin"] if pick else ""
            ),
            "outer_fold": pick["outer_fold"] if pick else "",
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
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--maximum-fpr", type=float, default=0.0275)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--view-layers", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-sessions", type=int, default=12)
    parser.add_argument("--inference-batch-sessions", type=int, default=16)
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
