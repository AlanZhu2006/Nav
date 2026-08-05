#!/usr/bin/env python3
"""Train a compact K+1 memory localizer with an explicit no-match state.

The model consumes frozen DINO patch-relation and temporal-support features.
It has separate candidate-ranking and set-level no-match heads, so it learns
``which memory node?`` and ``is there any valid node?`` without a scalar gate
or a hard SIFT threshold.  Hyperparameters and stopping epoch are selected on
scene-disjoint training validation scenes; held-out development scenes are
evaluated once after the configuration is frozen.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

try:
    from MemNavData.diag_patch_temporal_router import select_hard_candidates
    from MemNavData.patch_temporal_router import (
        combine_patch_temporal,
    )
except ModuleNotFoundError:  # direct script invocation
    from diag_patch_temporal_router import select_hard_candidates  # type: ignore
    from patch_temporal_router import (  # type: ignore
        combine_patch_temporal,
    )


REQUIRED_COLUMNS = {
    "session_id", "scene", "candidate_path", "candidate_frame",
    "dino_cosine", "teacher_covis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PackedSessions:
    features: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    covisibility: np.ndarray
    session_ids: tuple[str, ...]
    scenes: tuple[str, ...]

    def to(self, device: torch.device) -> "PackedSessions":
        return PackedSessions(
            self.features.to(device), self.mask.to(device),
            self.target.to(device), self.covisibility,
            self.session_ids, self.scenes)


def pack_sessions(
        features: np.ndarray, groups: np.ndarray, scenes: np.ndarray,
        covisibility: np.ndarray, *, positive_threshold: float) -> PackedSessions:
    features = np.asarray(features, dtype=np.float32)
    groups = np.asarray(groups, dtype=str).reshape(-1)
    scenes = np.asarray(scenes, dtype=str).reshape(-1)
    covisibility = np.asarray(covisibility, dtype=np.float32).reshape(-1)
    if (features.ndim != 2 or not len(features)
            or not (len(features) == len(groups) == len(scenes)
                    == len(covisibility))):
        raise ValueError("session inputs must be non-empty and aligned")
    if not np.isfinite(features).all() or not np.isfinite(covisibility).all():
        raise ValueError("session inputs must be finite")
    order: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        order.setdefault(str(group), []).append(index)
    indices = [np.asarray(value, dtype=np.int64) for value in order.values()]
    width = max(map(len, indices))
    batch = np.zeros((len(indices), width, features.shape[1]), np.float32)
    mask = np.zeros((len(indices), width), bool)
    target = np.zeros((len(indices), width + 1), np.float32)
    teacher = np.full((len(indices), width), np.nan, np.float32)
    session_scenes = []
    for row, index in enumerate(indices):
        if len(set(scenes[index])) != 1:
            raise ValueError("one localization session crosses scenes")
        count = len(index)
        batch[row, :count] = features[index]
        mask[row, :count] = True
        teacher[row, :count] = covisibility[index]
        positive = covisibility[index] >= positive_threshold
        if positive.any():
            weight = covisibility[index][positive]
            target[row, :count][positive] = weight / weight.sum()
        else:
            target[row, -1] = 1.0
        session_scenes.append(str(scenes[index[0]]))
    return PackedSessions(
        torch.from_numpy(batch), torch.from_numpy(mask),
        torch.from_numpy(target), teacher, tuple(order),
        tuple(session_scenes))


class NeuralSetLocalizer(nn.Module):
    """Permutation-invariant candidate scorer plus a separate dustbin head."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.no_match_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError("features/mask must have [sessions,candidates,...]")
        encoded = self.encoder(features)
        candidate = self.rank_head(encoded).squeeze(-1)
        candidate = candidate.masked_fill(~mask, -1e4)
        weight = mask.unsqueeze(-1).to(encoded.dtype)
        pooled_mean = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled_max = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        dustbin = self.no_match_head(
            torch.cat([pooled_mean, pooled_max], dim=-1))
        return torch.cat([candidate, dustbin], dim=-1)


def localization_metrics(
        packed: PackedSessions, probability: np.ndarray,
        *, positive_threshold: float, match_threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != tuple(packed.target.shape):
        raise ValueError("probability shape does not match packed sessions")
    if not 0.0 < match_threshold < 1.0:
        raise ValueError("match_threshold must lie in (0, 1)")
    match_target, match_score, selected_positive = [], [], []
    reciprocal_rank, joint, selected_overlap = [], [], []
    for row in range(len(packed.session_ids)):
        count = int(packed.mask[row].sum().item())
        covis = packed.covisibility[row, :count]
        positive = covis >= positive_threshold
        has_match = bool(positive.any())
        candidate = probability[row, :count]
        dustbin = float(probability[row, -1])
        pick = int(np.argmax(candidate))
        predicts_match = (1.0 - dustbin) >= match_threshold
        pick_positive = bool(positive[pick])
        match_target.append(int(has_match))
        match_score.append(1.0 - dustbin)
        selected_positive.append(int(pick_positive))
        selected_overlap.append(float(covis[pick]))
        joint.append(int(
            (has_match and predicts_match and pick_positive)
            or (not has_match and not predicts_match)))
        if has_match:
            rank = next(
                rank for rank, index in enumerate(
                    np.argsort(-candidate), start=1)
                if positive[index])
            reciprocal_rank.append(1.0 / rank)
    target = np.asarray(match_target)
    score = np.asarray(match_score)
    return {
        "sessions": len(target),
        "positive_sessions": int(target.sum()),
        "joint_localization_accuracy": float(np.mean(joint)),
        "match_accuracy": float(np.mean((score >= 0.5) == target)),
        "match_roc_auc": (
            float(roc_auc_score(target, score))
            if len(np.unique(target)) == 2 else None),
        "match_average_precision": (
            float(average_precision_score(target, score))
            if target.any() else None),
        "match_brier": float(np.mean((score - target) ** 2)),
        "conditional_candidate_recall_at_1": (
            float(np.mean(np.asarray(selected_positive)[target.astype(bool)]))
            if target.any() else None),
        "mean_reciprocal_positive_rank": (
            float(np.mean(reciprocal_rank)) if reciprocal_rank else None),
        "selected_overlap_mean": float(np.mean(selected_overlap)),
        "match_threshold": float(match_threshold),
    }


def select_match_threshold(
        packed: PackedSessions, probability: np.ndarray, *,
        positive_threshold: float) -> tuple[float, dict]:
    """Calibrate abstention on training-validation sessions only."""
    candidates = []
    for threshold in np.linspace(0.10, 0.90, 17):
        metrics = localization_metrics(
            packed, probability, positive_threshold=positive_threshold,
            match_threshold=float(threshold))
        key = (
            float(metrics["joint_localization_accuracy"]),
            float(metrics["match_accuracy"]),
            -abs(float(threshold) - 0.5),
        )
        candidates.append((key, float(threshold), metrics))
    _, threshold, metrics = max(candidates, key=lambda item: item[0])
    return threshold, metrics


def predict(model: nn.Module, packed: PackedSessions,
            device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        data = packed.to(device)
        return torch.softmax(
            model(data.features, data.mask), dim=-1).cpu().numpy()


def metric_key(report: dict) -> tuple[float, float, float, float]:
    return (
        float(report["joint_localization_accuracy"]),
        float(report["conditional_candidate_recall_at_1"] or 0.0),
        -float(report["match_brier"]),
        float(report["match_roc_auc"] or 0.0),
    )


def train_model(
        train: PackedSessions, validation: PackedSessions | None, *,
        input_dim: int, hidden_dim: int, dropout: float,
        weight_decay: float, epochs: int, batch_size: int,
        learning_rate: float, seed: int, device: torch.device,
        positive_threshold: float) -> tuple[NeuralSetLocalizer, int, dict | None]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = NeuralSetLocalizer(input_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    data = train.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = epochs
    best_metrics = None
    best_key = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(data.session_ids), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            logits = model(data.features[index], data.mask[index])
            loss = -(data.target[index]
                     * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if validation is not None and (epoch % 5 == 0 or epoch == epochs):
            metrics = localization_metrics(
                validation, predict(model, validation, device),
                positive_threshold=positive_threshold)
            key = metric_key(metrics)
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = metrics
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
            if stale >= 40:
                break
    if validation is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_metrics
    return model, epochs, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--heldout-scene", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--candidate-selection", choices=["raw", "temporal_nms"],
        default="temporal_nms")
    parser.add_argument("--candidate-min-gap", type=int, default=4)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if (args.top_k < 1 or args.candidate_min_gap < 1 or args.epochs < 1
            or args.batch_size < 1 or args.learning_rate <= 0.0):
        raise ValueError("invalid training configuration")
    for path in (args.teacher_csv, args.feature_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    frame = pd.read_csv(args.teacher_csv)
    if missing := REQUIRED_COLUMNS - set(frame.columns):
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    selected = select_hard_candidates(
        frame, args.top_k, args.candidate_selection,
        args.candidate_min_gap)
    cache = np.load(args.feature_cache, allow_pickle=False)
    patch = np.asarray(cache["patch"], dtype=np.float64)
    temporal = np.asarray(cache["temporal"], dtype=np.float64)
    features = combine_patch_temporal(patch, temporal)
    feature_names = [str(value) for value in cache["patch_names"]]
    feature_names.extend(str(value) for value in cache["temporal_names"][1:])
    if len(feature_names) != features.shape[1]:
        raise RuntimeError("feature names do not match combined feature width")
    if len(selected) != len(features):
        raise RuntimeError("selected candidates do not align with feature cache")
    error = float(np.max(np.abs(
        patch[:, 0]
        - selected["dino_cosine"].to_numpy(dtype=np.float64))))
    if error > 5e-5:
        raise RuntimeError(f"feature alignment error: {error}")

    groups = selected["session_id"].to_numpy(dtype=str)
    scenes = selected["scene"].to_numpy(dtype=str)
    covis = selected["teacher_covis"].to_numpy(dtype=np.float64)
    heldout = set(args.heldout_scene)
    all_scenes = set(scenes)
    if not heldout or not heldout.issubset(all_scenes):
        raise ValueError("held-out scenes are empty or absent")
    training_scenes = sorted(all_scenes - heldout)
    validation_count = max(2, int(round(0.2 * len(training_scenes))))
    tuning_validation = set(sorted(
        training_scenes,
        key=lambda scene: hashlib.sha256(
            f"memnav-set-validation:{scene}".encode()).hexdigest()
    )[:validation_count])
    tuning_train = set(training_scenes) - tuning_validation

    def mask_for(chosen: set[str]) -> np.ndarray:
        return np.asarray([scene in chosen for scene in scenes], dtype=bool)

    core_mask = mask_for(tuning_train)
    tune_mask = mask_for(tuning_validation)
    train_mask = mask_for(set(training_scenes))
    heldout_mask = mask_for(heldout)
    mean = features[core_mask].mean(0)
    scale = features[core_mask].std(0)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale

    def packed(mask: np.ndarray) -> PackedSessions:
        return pack_sessions(
            standardized[mask], groups[mask], scenes[mask], covis[mask],
            positive_threshold=args.positive_threshold)

    core, tune = packed(core_mask), packed(tune_mask)
    device = torch.device(args.device)
    grid = [
        {"hidden_dim": hidden, "dropout": dropout, "weight_decay": decay}
        for hidden in (64, 128)
        for dropout in (0.10, 0.25)
        for decay in (1e-4, 1e-3)
    ]
    trials = []
    best = None
    for trial_index, config in enumerate(grid):
        model, epoch, _ = train_model(
            core, tune, input_dim=features.shape[1], epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=trial_index, device=device,
            positive_threshold=args.positive_threshold, **config)
        threshold, metrics = select_match_threshold(
            tune, predict(model, tune, device),
            positive_threshold=args.positive_threshold)
        record = {
            **config, "selected_epoch": epoch,
            "match_threshold": threshold, "validation": metrics}
        trials.append(record)
        key = (*metric_key(metrics), -config["hidden_dim"], -config["dropout"])
        if best is None or key > best[0]:
            best = (key, record)
    selected_config = best[1]

    # Freeze the architecture/epoch, recompute normalization on every training
    # scene, then fit three seeds.  Only the final ensemble touches held-out
    # development features.
    mean = features[train_mask].mean(0)
    scale = features[train_mask].std(0)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    all_train = packed(train_mask)
    development = packed(heldout_mask)
    args.out_dir.mkdir(parents=True)
    probabilities = []
    seed_reports = []
    states = []
    for seed in (17, 29, 43):
        model, _, _ = train_model(
            all_train, None, input_dim=features.shape[1],
            hidden_dim=selected_config["hidden_dim"],
            dropout=selected_config["dropout"],
            weight_decay=selected_config["weight_decay"],
            epochs=selected_config["selected_epoch"],
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=seed, device=device,
            positive_threshold=args.positive_threshold)
        probability = predict(model, development, device)
        probabilities.append(probability)
        seed_reports.append(localization_metrics(
            development, probability,
            positive_threshold=args.positive_threshold,
            match_threshold=selected_config["match_threshold"]))
        states.append({key: value.detach().cpu() for key, value in
                       model.state_dict().items()})
    ensemble = np.mean(probabilities, axis=0)
    heldout_metrics = localization_metrics(
        development, ensemble,
        positive_threshold=args.positive_threshold,
        match_threshold=selected_config["match_threshold"])

    artifact = {
        "deployment_approved": False,
        "model_kind": "neural_k_plus_one_set_localizer",
        "input_dim": features.shape[1],
        "feature_names": feature_names,
        "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
        "config": {
            key: selected_config[key]
            for key in (
                "hidden_dim", "dropout", "weight_decay", "match_threshold")
        },
        "states": states,
    }
    model_path = args.out_dir / "neural_set_localizer_not_for_deployment.pt"
    torch.save(artifact, model_path)
    report = {
        "deployment_approved": False,
        "reason": (
            "development-scene result; requires frozen blind scenes and "
            "closed-loop comparison before deployment"),
        "objective": (
            "replace hard DINO/SIFT routing with candidate ranking plus an "
            "explicit learned no-match state"),
        "protocol": {
            "top_k": args.top_k,
            "candidate_selection": args.candidate_selection,
            "candidate_min_gap": args.candidate_min_gap,
            "positive_threshold": args.positive_threshold,
            "training_scenes": training_scenes,
            "tuning_train_scenes": sorted(tuning_train),
            "tuning_validation_scenes": sorted(tuning_validation),
            "heldout_development_scenes": sorted(heldout),
            "heldout_evaluated_once_after_freeze": True,
        },
        "hyperparameter_trials": trials,
        "selected": selected_config,
        "heldout_seed_metrics": seed_reports,
        "heldout_ensemble": heldout_metrics,
        "inputs": {
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_sha256": sha256(args.teacher_csv),
            "feature_cache": str(args.feature_cache.resolve()),
            "feature_sha256": sha256(args.feature_cache),
            "maximum_alignment_error": error,
        },
        "artifact": str(model_path.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
