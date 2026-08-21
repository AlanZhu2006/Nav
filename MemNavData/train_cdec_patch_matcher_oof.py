#!/usr/bin/env python3
"""Train the raw-patch CDEC rank/evidence experts with scene-disjoint OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from MemNavData.cdec_differentiable_matcher import (
        DifferentiablePatchMatcher,
        listwise_positive_loss,
    )
    from MemNavData.train_cdec_scene_oof import (
        calibration_scenes,
        decision_metrics,
        load_sessions,
        sha256,
        zero_empirical_fp_threshold,
    )
except ModuleNotFoundError:  # direct invocation
    from cdec_differentiable_matcher import (  # type: ignore
        DifferentiablePatchMatcher,
        listwise_positive_loss,
    )
    from train_cdec_scene_oof import (  # type: ignore
        calibration_scenes,
        decision_metrics,
        load_sessions,
        sha256,
        zero_empirical_fp_threshold,
    )


SCHEMA_VERSION = "cdec_differentiable_patch_matcher_oof_v1"
CONFIGS = {
    "task_only": 0.0,
    "certificate_distilled": 0.25,
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def session_row_matrix(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for _session, index in frame.groupby("session_id", sort=False).indices.items():
        index = np.asarray(index, dtype=np.int64)
        if len(index) != 8:
            raise RuntimeError("raw matcher requires exactly eight rows per session")
        rows.append(index)
    return np.stack(rows)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(
    *, token_bank: torch.Tensor, query_index: torch.Tensor,
    candidate_index: torch.Tensor, dino: torch.Tensor,
    row_matrix: torch.Tensor, candidate_label: torch.Tensor,
    certificate_pass: torch.Tensor, fit_sessions: np.ndarray,
    seed: int, certificate_weight: float, epochs: int,
    projection_dim: int, hidden_dim: int, dropout: float,
    learning_rate: float, weight_decay: float, device: torch.device,
):
    seed_all(seed)
    model = DifferentiablePatchMatcher(
        token_dim=token_bank.shape[-1], projection_dim=projection_dim,
        hidden_dim=hidden_dim, grid_size=int(round(token_bank.shape[1] ** 0.5)),
        dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_session_tensor = torch.as_tensor(
        fit_sessions, dtype=torch.long, device=device)
    fit_rows_matrix = row_matrix[fit_session_tensor]
    fit_rows = fit_rows_matrix.reshape(-1)
    labels = candidate_label[fit_session_tensor]
    known = labels >= 0
    positives = labels == 1
    known_positive_count = int((labels[known] == 1).sum().item())
    known_negative_count = int((labels[known] == 0).sum().item())
    task_positive_weight = torch.tensor(
        known_negative_count / max(1, known_positive_count),
        dtype=torch.float32, device=device)
    certificate = certificate_pass[fit_session_tensor]
    certificate_positive = int(certificate.sum().item())
    certificate_negative = int(certificate.numel() - certificate_positive)
    certificate_positive_weight = torch.tensor(
        certificate_negative / max(1, certificate_positive),
        dtype=torch.float32, device=device)
    final = None
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            token_bank, query_index[fit_rows], candidate_index[fit_rows],
            dino[fit_rows])
        task_logits = output["task_match_logits"].reshape(labels.shape)
        pass_logits = output["certificate_pass_logits"].reshape(labels.shape)
        pointwise = F.binary_cross_entropy_with_logits(
            task_logits[known], labels[known].float(),
            pos_weight=task_positive_weight)
        listwise = listwise_positive_loss(task_logits, positives)
        pass_loss = F.binary_cross_entropy_with_logits(
            pass_logits, certificate.float(),
            pos_weight=certificate_positive_weight)
        total = listwise + 0.25 * pointwise + certificate_weight * pass_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final = (total, listwise, pointwise, pass_loss)
    if final is None or not torch.isfinite(final[0]):
        raise RuntimeError("raw patch matcher training failed")
    return model, {
        "total": float(final[0].detach().cpu()),
        "listwise": float(final[1].detach().cpu()),
        "pointwise": float(final[2].detach().cpu()),
        "certificate_pass": float(final[3].detach().cpu()),
        "temperature": float(model.log_temperature.exp().detach().cpu()),
    }


def infer(model, token_bank, query_index, candidate_index, dino,
          row_matrix, sessions):
    index = torch.as_tensor(sessions, dtype=torch.long, device=token_bank.device)
    rows = row_matrix[index].reshape(-1)
    model.eval()
    with torch.inference_mode():
        output = model(
            token_bank, query_index[rows], candidate_index[rows], dino[rows])
    shape = (len(sessions), row_matrix.shape[1])
    task_logits = output["task_match_logits"].reshape(shape).float().cpu().numpy()
    pass_logits = output["certificate_pass_logits"].reshape(shape).float().cpu().numpy()
    task_probability = 1.0 / (1.0 + np.exp(-np.clip(task_logits, -40, 40)))
    pass_probability = 1.0 / (1.0 + np.exp(-np.clip(pass_logits, -40, 40)))
    selected = np.argmax(task_logits, axis=1)
    row = np.arange(len(sessions))
    return {
        "task_logits": task_logits,
        "pass_logits": pass_logits,
        "task_probability": task_probability,
        "pass_probability": pass_probability,
        "selected": selected,
        "selected_task_evidence": task_probability[row, selected],
        "selected_certificate_evidence": pass_probability[row, selected],
        "selected_product_evidence": np.sqrt(
            task_probability[row, selected] * pass_probability[row, selected]),
    }


def metric_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    valid = labels >= 0
    return {
        "n": int(valid.sum()),
        "roc_auc": float(roc_auc_score(labels[valid], scores[valid])),
        "average_precision": float(
            average_precision_score(labels[valid], scores[valid])),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--patch-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", choices=tuple(CONFIGS))
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--calibration-scenes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--projection-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    configs = args.config or list(CONFIGS)
    seeds = args.seed or [11]
    if len(set(seeds)) != len(seeds) or args.epochs < 1:
        raise ValueError("invalid seed/epoch configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    started = time.perf_counter()
    data = load_sessions(
        args.rows_csv, args.patch_cache,
        expected_rows_sha256=args.expected_rows_sha256,
        expected_cache_sha256=args.expected_cache_sha256)
    frame = pd.read_csv(args.rows_csv)
    cache = np.load(args.patch_cache, allow_pickle=False)
    rows = session_row_matrix(frame)
    token_bank = torch.as_tensor(
        cache["tokens"], dtype=torch.float16, device=device)
    query_index = torch.as_tensor(
        cache["query_index"], dtype=torch.long, device=device)
    candidate_index = torch.as_tensor(
        cache["candidate_index"], dtype=torch.long, device=device)
    dino = torch.as_tensor(
        frame["dino_cosine"].to_numpy(dtype=np.float32), device=device)
    row_matrix = torch.as_tensor(rows, dtype=torch.long, device=device)
    candidate_label = torch.as_tensor(
        data.candidate_label, dtype=torch.long, device=device)
    certificate_pass = torch.as_tensor(
        data.certificate_pass, dtype=torch.bool, device=device)
    outer = list(GroupKFold(args.outer_folds).split(
        np.arange(len(data.session_id)), groups=data.scene))
    prediction_rows = []
    receipts = []
    for config in configs:
        certificate_weight = CONFIGS[config]
        for seed in seeds:
            for fold, (outer_train, outer_test) in enumerate(outer):
                heldout = calibration_scenes(
                    data.scene[outer_train], fold, args.calibration_scenes)
                calibration = outer_train[
                    np.isin(data.scene[outer_train], sorted(heldout))]
                fit = outer_train[
                    ~np.isin(data.scene[outer_train], sorted(heldout))]
                model, final_loss = train_one(
                    token_bank=token_bank, query_index=query_index,
                    candidate_index=candidate_index, dino=dino,
                    row_matrix=row_matrix, candidate_label=candidate_label,
                    certificate_pass=certificate_pass, fit_sessions=fit,
                    seed=seed + 1009 * fold,
                    certificate_weight=certificate_weight, epochs=args.epochs,
                    projection_dim=args.projection_dim,
                    hidden_dim=args.hidden_dim, dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, device=device)
                cal = infer(
                    model, token_bank, query_index, candidate_index, dino,
                    row_matrix, calibration)
                test = infer(
                    model, token_bank, query_index, candidate_index, dino,
                    row_matrix, outer_test)
                thresholds = {
                    name: zero_empirical_fp_threshold(
                        cal[name], data.session_label[calibration])
                    for name in (
                        "selected_task_evidence",
                        "selected_certificate_evidence",
                        "selected_product_evidence",
                    )
                }
                for local, session in enumerate(outer_test):
                    selected = int(test["selected"][local])
                    row = {
                        "config": config, "seed": seed, "outer_fold": fold,
                        "session_index": int(session),
                        "session_id": str(data.session_id[session]),
                        "scene": str(data.scene[session]),
                        "session_label": int(data.session_label[session]),
                        "selected_index": selected,
                        "selected_candidate_label": int(
                            data.candidate_label[session, selected]),
                    }
                    for evidence, threshold in thresholds.items():
                        value = float(test[evidence][local])
                        row[evidence] = value
                        row[f"{evidence}_threshold"] = float(threshold)
                        row[f"{evidence}_active"] = bool(value > threshold)
                    for candidate in range(8):
                        row[f"task_probability_{candidate}"] = float(
                            test["task_probability"][local, candidate])
                        row[f"certificate_probability_{candidate}"] = float(
                            test["pass_probability"][local, candidate])
                    prediction_rows.append(row)
                receipts.append({
                    "config": config, "seed": seed, "outer_fold": fold,
                    "fit_scenes": sorted(set(map(str, data.scene[fit]))),
                    "calibration_scenes": sorted(heldout),
                    "test_scenes": sorted(set(map(str, data.scene[outer_test]))),
                    "thresholds": thresholds,
                    "final_loss": final_loss,
                })
                print(json.dumps({
                    "config": config, "seed": seed, "fold": fold,
                    "loss": final_loss, "thresholds": thresholds,
                }, sort_keys=True), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    predictions = pd.DataFrame(prediction_rows)
    expected = len(configs) * len(seeds) * len(data.session_id)
    if len(predictions) != expected:
        raise RuntimeError("raw matcher OOF coverage is incomplete")
    args.out_dir.mkdir(parents=True)
    predictions_path = args.out_dir / "oof_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics = {}
    for config in configs:
        metrics[config] = {}
        for seed in seeds:
            subset = predictions[
                predictions.config.eq(config) & predictions.seed.eq(seed)
            ].sort_values("session_index")
            index = subset.session_index.to_numpy(dtype=np.int64)
            selected = subset.selected_index.to_numpy(dtype=np.int64)
            candidate_task = np.stack([
                subset[f"task_probability_{candidate}"].to_numpy()
                for candidate in range(8)
            ], axis=1).reshape(-1)
            candidate_certificate = np.stack([
                subset[f"certificate_probability_{candidate}"].to_numpy()
                for candidate in range(8)
            ], axis=1).reshape(-1)
            labels = data.candidate_label[index].reshape(-1)
            pass_labels = data.certificate_pass[index].astype(int).reshape(-1)
            value = {
                "ranking_always": decision_metrics(
                    data, index, selected, np.ones(len(index), dtype=bool)),
                "candidate_task": metric_summary(labels, candidate_task),
                "candidate_certificate": metric_summary(
                    pass_labels, candidate_certificate),
            }
            for evidence in (
                    "selected_task_evidence", "selected_certificate_evidence",
                    "selected_product_evidence"):
                value[evidence] = decision_metrics(
                    data, index, selected,
                    subset[f"{evidence}_active"].to_numpy(dtype=bool))
            metrics[config][str(seed)] = value
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "train40_scene_grouped_raw_patch_oof_not_closed_loop",
        "inputs": {
            "rows_csv": str(args.rows_csv.resolve()),
            "rows_csv_sha256": args.expected_rows_sha256,
            "patch_cache": str(args.patch_cache.resolve()),
            "patch_cache_sha256": args.expected_cache_sha256,
        },
        "configuration": {
            "configs": {name: CONFIGS[name] for name in configs},
            "seeds": seeds, "epochs": args.epochs,
            "projection_dim": args.projection_dim,
            "hidden_dim": args.hidden_dim, "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "outer_folds": args.outer_folds,
            "calibration_scenes": args.calibration_scenes,
            "device": str(device),
        },
        "metrics": metrics,
        "training_receipts": receipts,
        "artifacts": {
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256(predictions_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "train40 only; no development/blind inputs",
            "offline localization metrics are not closed-loop SR",
            "calibration zero-FP is empirical, not a finite-sample guarantee",
        ],
    }
    atomic_json(args.out_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
