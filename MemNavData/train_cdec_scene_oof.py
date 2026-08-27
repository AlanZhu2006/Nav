#!/usr/bin/env python3
"""Scene-disjoint OOF training for Certificate-Distilled Episodic Compass.

The outer test scenes are never used for fitting or calibration.  Within each
outer training split, a deterministic subset of scenes is reserved solely for
the NULL-margin operating point; the student is fit on the remaining scenes.
Three predeclared ablations isolate the value of privileged certificate
distillation from the task-only structured posterior.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

try:
    from MemNavData.certificate_distilled_compass import (
        CertificateDistilledCompass,
        cdec_loss,
    )
except ModuleNotFoundError:  # direct script invocation
    from certificate_distilled_compass import (  # type: ignore
        CertificateDistilledCompass,
        cdec_loss,
    )


SCHEMA_VERSION = "cdec_scene_grouped_oof_v1"
MODEL_CONFIGS = {
    "task_only": (0.0, 0.0),
    "certificate_pass": (0.25, 0.0),
    "certificate_full": (0.25, 0.10),
}
RANK_FIELDS = (
    "fundamental_inliers",
    "fundamental_query_grid_coverage",
    "fundamental_query_hull_coverage",
    "lightglue_score_median",
    "dino_cosine",
)


@dataclass(frozen=True)
class SessionTable:
    session_id: np.ndarray
    scene: np.ndarray
    session_label: np.ndarray
    candidate_label: np.ndarray
    features: np.ndarray
    certificate_pass: np.ndarray
    teacher_top_index: np.ndarray
    dino_cosine: np.ndarray
    candidate_frame: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def teacher_top_index(group: pd.DataFrame) -> int:
    keys = (
        group["candidate_frame"].to_numpy(dtype=np.int64),
        -group["dino_cosine"].to_numpy(dtype=np.float64),
        -group["lightglue_score_median"].to_numpy(dtype=np.float64),
        -group["fundamental_query_hull_coverage"].to_numpy(dtype=np.float64),
        -group["fundamental_query_grid_coverage"].to_numpy(dtype=np.float64),
        -group["fundamental_inliers"].to_numpy(dtype=np.float64),
    )
    return int(np.lexsort(keys)[0])


def load_sessions(rows_csv: Path, cache_path: Path, *,
                  expected_rows_sha256: str,
                  expected_cache_sha256: str) -> SessionTable:
    if sha256(rows_csv) != expected_rows_sha256:
        raise RuntimeError("teacher CSV SHA256 changed")
    if sha256(cache_path) != expected_cache_sha256:
        raise RuntimeError("patch cache SHA256 changed")
    frame = pd.read_csv(rows_csv)
    cache = np.load(cache_path, allow_pickle=False)
    if str(cache["rows_csv_sha256"].item()) != expected_rows_sha256:
        raise RuntimeError("patch cache is bound to another teacher table")
    relation = cache["directional_relation"].astype(np.float32)
    if relation.shape[0] != len(frame) or not np.isfinite(relation).all():
        raise RuntimeError("patch relation table is invalid")
    missing = ({
        "session_id", "scene", "session_label", "candidate_label",
        "candidate_frame", "dino_cosine",
        "fundamental_inliers", "fundamental_query_grid_coverage",
        "fundamental_query_hull_coverage", "lightglue_score_median",
    } - set(frame.columns))
    if missing:
        raise ValueError(f"teacher rows lack columns: {sorted(missing)}")

    session_id, scene, session_label = [], [], []
    candidate_label, features, certificate_pass = [], [], []
    teacher_top, dino, candidate_frame = [], [], []
    for name, indices in frame.groupby("session_id", sort=False).indices.items():
        index = np.asarray(indices, dtype=np.int64)
        group = frame.iloc[index]
        if len(group) != 8 or group["scene"].nunique() != 1:
            raise RuntimeError("each session must contain eight candidates in one scene")
        labels = group["session_label"].to_numpy(dtype=np.int64)
        if not np.all(labels == labels[0]):
            raise RuntimeError("session label changes within a session")
        session_id.append(str(name))
        scene.append(str(group["scene"].iloc[0]))
        session_label.append(int(labels[0]))
        candidate_label.append(group["candidate_label"].to_numpy(dtype=np.int64))
        features.append(relation[index])
        certificate_pass.append((
            group["fundamental_inliers"].to_numpy(dtype=np.float64) >= 32
        ) & (
            group["fundamental_query_grid_coverage"].to_numpy(dtype=np.float64)
            >= 0.75
        ))
        teacher_top.append(teacher_top_index(group))
        dino.append(group["dino_cosine"].to_numpy(dtype=np.float32))
        candidate_frame.append(
            group["candidate_frame"].to_numpy(dtype=np.int64))
    result = SessionTable(
        session_id=np.asarray(session_id),
        scene=np.asarray(scene),
        session_label=np.asarray(session_label, dtype=np.int64),
        candidate_label=np.stack(candidate_label),
        features=np.stack(features),
        certificate_pass=np.stack(certificate_pass),
        teacher_top_index=np.asarray(teacher_top, dtype=np.int64),
        dino_cosine=np.stack(dino),
        candidate_frame=np.stack(candidate_frame),
    )
    if (len(result.session_id) != 480 or len(np.unique(result.scene)) != 40
            or result.features.shape[:2] != (480, 8)):
        raise RuntimeError("CDEC train40 session universe changed")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_session_weights(session_label: np.ndarray,
                             positive: np.ndarray) -> np.ndarray:
    known = session_label >= 0
    actionable = positive.any(axis=1) & known
    null = ~actionable & known
    weights = np.ones(len(session_label), dtype=np.float32)
    if actionable.any() and null.any():
        weights[actionable] = float(known.sum()) / (2.0 * actionable.sum())
        weights[null] = float(known.sum()) / (2.0 * null.sum())
    weights[~known] = 0.0
    return weights


def standardize(features: np.ndarray, fit_index: np.ndarray):
    flattened = features[fit_index].reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = ((features - mean) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise RuntimeError("feature standardization produced non-finite values")
    return normalized, mean, scale


def fit_model(data: SessionTable, fit_index: np.ndarray, *, seed: int,
              lambda_pass: float, lambda_rank: float, epochs: int,
              hidden_dim: int, layers: int, heads: int, dropout: float,
              learning_rate: float, weight_decay: float,
              device: torch.device):
    seed_everything(seed)
    features, mean, scale = standardize(data.features, fit_index)
    model = CertificateDistilledCompass(
        features.shape[-1], hidden_dim=hidden_dim, heads=heads,
        layers=layers, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    index = torch.as_tensor(fit_index, dtype=torch.long, device=device)
    x = torch.as_tensor(features, device=device)
    positive = torch.as_tensor(
        data.candidate_label == 1, dtype=torch.bool, device=device)
    task_mask = torch.as_tensor(
        data.session_label >= 0, dtype=torch.bool, device=device)
    weights = torch.as_tensor(
        balanced_session_weights(data.session_label, data.candidate_label == 1),
        device=device)
    certificate = torch.as_tensor(
        data.certificate_pass, dtype=torch.float32, device=device)
    top = torch.as_tensor(data.teacher_top_index, dtype=torch.long, device=device)
    fit_certificate = data.certificate_pass[fit_index]
    positives = int(fit_certificate.sum())
    negatives = int(fit_certificate.size - positives)
    pass_weight = torch.tensor(
        negatives / max(1, positives), dtype=torch.float32, device=device)
    final_loss = None
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x[index])
        loss = cdec_loss(
            outputs,
            positive_candidates=positive[index],
            task_mask=task_mask[index],
            session_weight=weights[index],
            certificate_pass=certificate[index],
            teacher_top_index=top[index],
            pass_positive_weight=pass_weight,
            lambda_pass=lambda_pass,
            lambda_rank=lambda_rank)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = loss
    if final_loss is None or not torch.isfinite(final_loss.total):
        raise RuntimeError("CDEC training did not produce a finite loss")
    return model, features, mean, scale, {
        "total": float(final_loss.total.detach().cpu()),
        "task": float(final_loss.task.detach().cpu()),
        "certificate_pass": float(final_loss.certificate_pass.detach().cpu()),
        "certificate_rank": float(final_loss.certificate_rank.detach().cpu()),
    }


def infer(model, features: np.ndarray, index: np.ndarray,
          device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        output = model(torch.as_tensor(features[index], device=device))
        task = output["task_logits"].float().cpu().numpy()
        certificate = torch.sigmoid(
            output["certificate_pass_logits"]).float().cpu().numpy()
        rank = output["certificate_rank_logits"].float().cpu().numpy()
    anchor = np.argmax(task[:, :-1], axis=1)
    row = np.arange(len(index))
    margin = task[row, anchor] - task[:, -1]
    probability = np.exp(task - task.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    return {
        "task_logits": task,
        "selected_index": anchor,
        "margin": margin,
        "null_probability": probability[:, -1],
        "selected_probability": probability[row, anchor],
        "certificate_probability": certificate,
        "teacher_rank_logits": rank,
    }


def calibration_scenes(train_scenes: np.ndarray, outer_fold: int,
                       count: int) -> set[str]:
    unique = sorted(set(map(str, train_scenes)))
    if count < 1 or count >= len(unique):
        raise ValueError("calibration scene count is invalid")
    salt = f"cdec-calibration-v1/fold={outer_fold}"
    ordered = sorted(unique, key=lambda scene: (
        hashlib.sha256(f"{salt}\0{scene}".encode()).hexdigest(), scene))
    return set(ordered[:count])


def zero_empirical_fp_threshold(margins: np.ndarray,
                                labels: np.ndarray) -> float:
    negative = np.asarray(margins)[np.asarray(labels) == 0]
    if not len(negative) or not np.isfinite(negative).all():
        raise RuntimeError("calibration split lacks finite strict negatives")
    return float(np.nextafter(np.max(negative), np.inf))


def decision_metrics(table: SessionTable, index: np.ndarray,
                     selected: np.ndarray, active: np.ndarray) -> dict:
    labels = table.session_label[index]
    candidate = table.candidate_label[index]
    chosen_label = candidate[np.arange(len(index)), selected]
    has_positive = (candidate == 1).any(axis=1)
    known = labels >= 0
    positive = labels == 1
    strict_negative = labels == 0
    recoverable = positive & has_positive
    shortlist_miss = positive & ~has_positive
    correct = active & (chosen_label == 1) & positive
    wrong_positive = active & (chosen_label != 1) & positive
    false_activation = active & strict_negative
    safe = ((recoverable & correct)
            | (shortlist_miss & ~active)
            | (strict_negative & ~active))
    return {
        "known_sessions": int(known.sum()),
        "positive_sessions": int(positive.sum()),
        "recoverable_positive_sessions": int(recoverable.sum()),
        "shortlist_miss_sessions": int(shortlist_miss.sum()),
        "strict_negative_sessions": int(strict_negative.sum()),
        "top1_correct_ignoring_abstention": int(
            (positive & (chosen_label == 1)).sum()),
        "active": int((active & known).sum()),
        "correct_anchor": int(correct.sum()),
        "wrong_anchor_on_positive": int(wrong_positive.sum()),
        "strict_negative_false_activation": int(false_activation.sum()),
        "recoverable_positive_abstain": int((recoverable & ~active).sum()),
        "shortlist_miss_false_activation": int((shortlist_miss & active).sum()),
        "exact_safe_action": int((safe & known).sum()),
        "exact_safe_action_rate": float((safe & known).sum() / max(1, known.sum())),
    }


def rows_for_predictions(data: SessionTable, index: np.ndarray, inference: dict,
                         *, config: str, seed: int, fold: int,
                         threshold: float) -> list[dict]:
    selected = inference["selected_index"]
    margin = inference["margin"]
    rows = []
    for local, global_index in enumerate(index):
        chosen = int(selected[local])
        rows.append({
            "config": config,
            "seed": seed,
            "outer_fold": fold,
            "session_id": str(data.session_id[global_index]),
            "scene": str(data.scene[global_index]),
            "session_label": int(data.session_label[global_index]),
            "has_positive_candidate": bool(
                np.any(data.candidate_label[global_index] == 1)),
            "selected_index": chosen,
            "selected_candidate_label": int(
                data.candidate_label[global_index, chosen]),
            "selected_candidate_frame": int(
                data.candidate_frame[global_index, chosen]),
            "margin_anchor_minus_null": float(margin[local]),
            "calibration_threshold": float(threshold),
            "uncalibrated_active": bool(margin[local] > 0.0),
            "calibrated_active": bool(margin[local] > threshold),
            "null_probability": float(inference["null_probability"][local]),
            "selected_probability": float(
                inference["selected_probability"][local]),
            "selected_certificate_probability": float(
                inference["certificate_probability"][local, chosen]),
        })
    return rows


def parse_int_list(values: Iterable[str]) -> list[int]:
    result = [int(value) for value in values]
    if not result or len(set(result)) != len(result):
        raise ValueError("seeds must be non-empty and unique")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--patch-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--config", action="append", choices=tuple(MODEL_CONFIGS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--calibration-scenes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    seeds = parse_int_list(args.seed or ["11", "23", "37", "53", "71"])
    configs = args.config or list(MODEL_CONFIGS)
    if (args.outer_folds < 2 or args.epochs < 1 or args.hidden_dim < 4
            or args.learning_rate <= 0 or args.weight_decay < 0):
        raise ValueError("invalid training configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    started = time.perf_counter()
    data = load_sessions(
        args.rows_csv, args.patch_cache,
        expected_rows_sha256=args.expected_rows_sha256,
        expected_cache_sha256=args.expected_cache_sha256)
    outer = list(GroupKFold(args.outer_folds).split(
        np.arange(len(data.session_id)), groups=data.scene))
    all_rows = []
    training_receipts = []
    for config in configs:
        lambda_pass, lambda_rank = MODEL_CONFIGS[config]
        for seed in seeds:
            for fold, (outer_train, outer_test) in enumerate(outer):
                heldout = calibration_scenes(
                    data.scene[outer_train], fold, args.calibration_scenes)
                calibration = outer_train[
                    np.isin(data.scene[outer_train], sorted(heldout))]
                fit = outer_train[
                    ~np.isin(data.scene[outer_train], sorted(heldout))]
                if set(data.scene[fit]) & set(data.scene[calibration]):
                    raise RuntimeError("fit/calibration scenes overlap")
                if ((set(data.scene[fit]) | set(data.scene[calibration]))
                        & set(data.scene[outer_test])):
                    raise RuntimeError("outer test scene leaked into training")
                model, normalized, mean, scale, final_loss = fit_model(
                    data, fit, seed=seed + 1009 * fold,
                    lambda_pass=lambda_pass, lambda_rank=lambda_rank,
                    epochs=args.epochs, hidden_dim=args.hidden_dim,
                    layers=args.layers, heads=args.heads, dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, device=device)
                calibration_output = infer(model, normalized, calibration, device)
                threshold = zero_empirical_fp_threshold(
                    calibration_output["margin"],
                    data.session_label[calibration])
                test_output = infer(model, normalized, outer_test, device)
                all_rows.extend(rows_for_predictions(
                    data, outer_test, test_output, config=config, seed=seed,
                    fold=fold, threshold=threshold))
                training_receipts.append({
                    "config": config,
                    "seed": seed,
                    "outer_fold": fold,
                    "fit_scenes": sorted(set(map(str, data.scene[fit]))),
                    "calibration_scenes": sorted(heldout),
                    "test_scenes": sorted(set(map(str, data.scene[outer_test]))),
                    "fit_sessions": len(fit),
                    "calibration_sessions": len(calibration),
                    "test_sessions": len(outer_test),
                    "calibration_strict_negatives": int(
                        (data.session_label[calibration] == 0).sum()),
                    "zero_empirical_fp_threshold": threshold,
                    "final_loss": final_loss,
                    "feature_mean_sha256": hashlib.sha256(
                        mean.astype(np.float64).tobytes()).hexdigest(),
                    "feature_scale_sha256": hashlib.sha256(
                        scale.astype(np.float64).tobytes()).hexdigest(),
                })
                print(json.dumps({
                    "config": config, "seed": seed, "fold": fold,
                    "threshold": threshold, "loss": final_loss["total"],
                }, sort_keys=True), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    predictions = pd.DataFrame(all_rows)
    expected = len(configs) * len(seeds) * len(data.session_id)
    if len(predictions) != expected:
        raise RuntimeError(f"OOF prediction cover {len(predictions)} != {expected}")
    args.out_dir.mkdir(parents=True)
    predictions_path = args.out_dir / "oof_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics = {}
    for config in configs:
        metrics[config] = {}
        for seed in seeds:
            subset = predictions[
                predictions["config"].eq(config)
                & predictions["seed"].eq(seed)]
            order = np.asarray([
                int(np.flatnonzero(data.session_id == session)[0])
                for session in subset["session_id"]
            ], dtype=np.int64)
            selected = subset["selected_index"].to_numpy(dtype=np.int64)
            metrics[config][str(seed)] = {
                "ranking_no_abstention": decision_metrics(
                    data, order, selected, np.ones(len(subset), dtype=bool)),
                "joint_argmax": decision_metrics(
                    data, order, selected,
                    subset["uncalibrated_active"].to_numpy(dtype=bool)),
                "zero_calibration_fp": decision_metrics(
                    data, order, selected,
                    subset["calibrated_active"].to_numpy(dtype=bool)),
            }

    # Frozen, training-free reference points on the same 480 sessions.
    teacher_selected = data.teacher_top_index
    teacher_selected_pass = data.certificate_pass[
        np.arange(len(data.session_id)), teacher_selected]
    dino_selected = np.argmax(data.dino_cosine, axis=1)
    baselines = {
        "dino_always": decision_metrics(
            data, np.arange(len(data.session_id)), dino_selected,
            np.ones(len(data.session_id), dtype=bool)),
        "lightglue_rank_always": decision_metrics(
            data, np.arange(len(data.session_id)), teacher_selected,
            np.ones(len(data.session_id), dtype=bool)),
        "ranked_static_pre_certificate": decision_metrics(
            data, np.arange(len(data.session_id)), teacher_selected,
            teacher_selected_pass),
        "static_any_pass_existence_only": {
            "positive_session_covered": int((
                (data.session_label == 1) & data.certificate_pass.any(axis=1)
            ).sum()),
            "strict_negative_false_activation": int((
                (data.session_label == 0) & data.certificate_pass.any(axis=1)
            ).sum()),
            "warning": "existence coverage only; not a selected-anchor metric",
        },
    }
    main_config = "certificate_full" if "certificate_full" in metrics else configs[-1]
    gate_by_seed = {}
    for seed in seeds:
        value = metrics[main_config][str(seed)]["zero_calibration_fp"]
        gate_by_seed[str(seed)] = {
            "strict_negative_fp_le_2": (
                value["strict_negative_false_activation"] <= 2),
            "correct_anchor_gt_ranked_static_86": value["correct_anchor"] > 86,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "train40_scene_grouped_oof_not_closed_loop",
        "inputs": {
            "rows_csv": str(args.rows_csv.resolve()),
            "rows_csv_sha256": args.expected_rows_sha256,
            "patch_cache": str(args.patch_cache.resolve()),
            "patch_cache_sha256": args.expected_cache_sha256,
        },
        "data": {
            "sessions": len(data.session_id),
            "scenes": len(np.unique(data.scene)),
            "positive_sessions": int((data.session_label == 1).sum()),
            "strict_no_match_sessions": int((data.session_label == 0).sum()),
            "ambiguous_sessions": int((data.session_label < 0).sum()),
            "positive_with_top8_anchor": int((
                (data.session_label == 1)
                & (data.candidate_label == 1).any(axis=1)).sum()),
        },
        "model_configs": {
            name: {"lambda_pass": value[0], "lambda_rank": value[1]}
            for name, value in MODEL_CONFIGS.items() if name in configs
        },
        "training": {
            "seeds": seeds,
            "outer_folds": args.outer_folds,
            "calibration_scenes_per_outer_fold": args.calibration_scenes,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "calibration_rule": "nextafter(max strict-negative margin, +inf)",
            "receipts": training_receipts,
        },
        "baselines": baselines,
        "metrics": metrics,
        "pre_registered_gate": {
            "main_config": main_config,
            "by_seed": gate_by_seed,
            "all_seeds_pass": all(
                all(checks.values()) for checks in gate_by_seed.values()),
        },
        "artifacts": {
            "predictions_csv": str(predictions_path.resolve()),
            "predictions_csv_sha256": sha256(predictions_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "train40 only; development and blind are not read",
            "zero empirical calibration FP is not a finite-sample formal risk guarantee",
            "offline anchor/no-match metrics are not a closed-loop SR claim",
        ],
    }
    atomic_json(args.out_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "gate": report["pre_registered_gate"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
