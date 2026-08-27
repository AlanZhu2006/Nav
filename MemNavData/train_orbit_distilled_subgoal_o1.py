#!/usr/bin/env python3
"""Scene-OOF orbit-to-single-view observability gate.

The Habitat producer renders a complete yaw orbit only to balance supervision.
At deployment the tested student consumes one frozen NavDP ImageGoal feature
and predicts the complete camera-relative C8 progress field.  A full-orbit
shared-linear CGC is trained on the same features as an information-acquisition
control, not as the deployable method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

try:
    from MemNavData.circular_goal_compass import (
        DIRECTION_STEP_DEG,
        NUM_DIRECTIONS,
        deterministic_scene_folds,
        masked_listwise_loss,
        scene_cluster_bootstrap,
        scene_macro_mean,
    )
    from MemNavData.train_cgc_scene_oof import (
        TrainingError,
        canonical_json_bytes,
        evaluate_kind,
        extract_features,
        load_dataset,
        load_feature_arrays,
        primary_goal_pair_contrast,
        sha256_file,
    )
except ImportError:  # direct execution with MemNavData on PYTHONPATH
    from circular_goal_compass import (  # type: ignore
        DIRECTION_STEP_DEG,
        NUM_DIRECTIONS,
        deterministic_scene_folds,
        masked_listwise_loss,
        scene_cluster_bootstrap,
        scene_macro_mean,
    )
    from train_cgc_scene_oof import (  # type: ignore
        TrainingError,
        canonical_json_bytes,
        evaluate_kind,
        extract_features,
        load_dataset,
        load_feature_arrays,
        primary_goal_pair_contrast,
        sha256_file,
    )


REPORT_SCHEMA = "orbit_distilled_subgoal_o1_report_v1"
CHECKPOINT_SCHEMA = "orbit_distilled_subgoal_o1_fold_checkpoint_v1"
FOLD_SALT = "orbit-distilled-subgoal-o1-folds-v1-20260810"
FULL_SCENE_COUNT = 31
FULL_PHYSICAL_GROUP_COUNT = 123
FULL_ROW_COUNT = 246
PRIMARY_STATE_NAME = "goal_b_t0"
PRIMARY_VARIANT = "factual"
USEFUL_PROGRESS_MARGIN_M = 0.25
VIEW_DOSES = (1, 2, 4, 8)


class O1Error(RuntimeError):
    """An O1 artifact, schedule, or audit failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise O1Error(message)


class SingleViewCompass(nn.Module):
    """One current-goal feature to a complete camera-relative C8 field."""

    def __init__(self, feature_dim: int = 384) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        require(feature_dim > 0, "feature dimension must be positive")
        self.normalization = nn.LayerNorm(
            feature_dim, elementwise_affine=False)
        self.readout = nn.Linear(feature_dim, NUM_DIRECTIONS)

    def forward(self, features):
        if features.ndim != 2:
            raise ValueError("single-view features must have shape [batch, dim]")
        return self.readout(self.normalization(features))


def camera_relative_rings(values: np.ndarray) -> np.ndarray:
    """Express one world-indexed ring relative to each of its eight views."""
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != NUM_DIRECTIONS:
        raise ValueError("world rings must have shape [rows, 8]")
    return np.stack([
        np.roll(array, -view_index, axis=1)
        for view_index in range(NUM_DIRECTIONS)
    ], axis=1)


def relative_logits_to_world(logits: np.ndarray) -> np.ndarray:
    """Transport every single-view prediction into the common world ring."""
    array = np.asarray(logits)
    if array.ndim != 3 or array.shape[1:] != (
            NUM_DIRECTIONS, NUM_DIRECTIONS):
        raise ValueError("relative logits must have shape [rows, 8, 8]")
    return np.stack([
        np.roll(array[:, view_index], view_index, axis=1)
        for view_index in range(NUM_DIRECTIONS)
    ], axis=1)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float64)
    maximum = np.max(array, axis=-1, keepdims=True)
    shifted = array - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _circular_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    raw = np.abs(np.asarray(predicted, dtype=np.int64)
                 - np.asarray(target, dtype=np.int64))
    return np.minimum(raw, NUM_DIRECTIONS - raw)


def summarize_vector(values: np.ndarray, scenes: np.ndarray, *, seed: int):
    array = np.asarray(values, dtype=np.float64)
    scene_array = np.asarray(scenes, dtype=object)
    require(array.ndim == 1 and scene_array.shape == array.shape and len(array),
            "metric vector is empty or misaligned")
    require(bool(np.isfinite(array).all()), "metric vector is non-finite")
    return {
        "scene_macro_mean": scene_macro_mean(array, scene_array),
        "scene_cluster_bootstrap_95": scene_cluster_bootstrap(
            array, scene_array, seed=seed, resamples=5000),
        "rows": int(len(array)),
        "scene_clusters": int(len(set(map(str, scene_array)))),
    }


def primary_row_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        (arrays["state_name"].astype(str) == PRIMARY_STATE_NAME)
        & (arrays["goal_variant"].astype(str) == PRIMARY_VARIANT),
        dtype=bool,
    )


def train_single_view_model(
        seed: int, train_rows: np.ndarray, arrays: Mapping[str, np.ndarray],
        *, epochs: int, batch_size: int, device: torch.device):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    features_np = arrays["features_correct"][train_rows].astype(np.float32)
    advantages_np = camera_relative_rings(
        arrays["advantages_m"][train_rows]).astype(np.float32)
    valid_np = camera_relative_rings(
        arrays["candidate_valid"][train_rows]).astype(bool)
    feature_dim = int(features_np.shape[-1])
    features = torch.as_tensor(
        features_np.reshape(-1, feature_dim), device=device)
    advantages = torch.as_tensor(
        advantages_np.reshape(-1, NUM_DIRECTIONS), device=device)
    valid = torch.as_tensor(
        valid_np.reshape(-1, NUM_DIRECTIONS), device=device)
    model = SingleViewCompass(feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x0D5A)
    first_gradient_norm = None
    final_loss = None
    model.train()
    for _epoch in range(int(epochs)):
        permutation = torch.randperm(
            len(features), generator=generator).tolist()
        losses = []
        for offset in range(0, len(permutation), int(batch_size)):
            batch = permutation[offset:offset + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[batch])
            loss = masked_listwise_loss(
                logits, advantages[batch], valid[batch])
            require(bool(torch.isfinite(loss)),
                    "single-view training loss became non-finite")
            loss.backward()
            if first_gradient_norm is None:
                squared = sum(
                    float(parameter.grad.detach().square().sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None)
                first_gradient_norm = math.sqrt(squared)
                require(math.isfinite(first_gradient_norm)
                        and first_gradient_norm > 0.0,
                        "single-view first gradient is zero or non-finite")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
    require(final_loss is not None and first_gradient_norm is not None,
            "single-view training loop did not execute")
    return model.eval(), {
        "final_train_loss": final_loss,
        "first_gradient_norm": first_gradient_norm,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_view_examples": int(len(features)),
    }


def predict_single_view(model, features: np.ndarray, device: torch.device,
                        *, batch_size: int = 256) -> np.ndarray:
    array = np.asarray(features)
    require(array.ndim == 3 and array.shape[1] == NUM_DIRECTIONS,
            "single-view feature archive shape changed")
    rows, views, feature_dim = array.shape
    flat = array.reshape(-1, feature_dim)
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(flat), int(batch_size)):
            batch = torch.as_tensor(
                flat[offset:offset + int(batch_size)],
                dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    result = np.concatenate(outputs, axis=0).astype(np.float64)
    return result.reshape(rows, views, NUM_DIRECTIONS)


def evaluate_single_view_oof(
        arrays: Mapping[str, np.ndarray], *, folds,
        seeds: Sequence[int], epochs: int, batch_size: int,
        device: torch.device, checkpoint_root: Path):
    scenes = arrays["scene"].astype(str)
    rows = len(scenes)
    feature_dim = int(arrays["features_correct"].shape[-1])
    oof = np.full(
        (rows, NUM_DIRECTIONS, NUM_DIRECTIONS), np.nan, dtype=np.float64)
    oof_swapped = np.full_like(oof, np.nan)
    fold_records = []
    for fold_index, test_scene_tuple in enumerate(folds):
        test_scenes = set(test_scene_tuple)
        test_rows = np.flatnonzero(np.asarray([
            scene in test_scenes for scene in scenes], dtype=bool))
        train_rows = np.flatnonzero(np.asarray([
            scene not in test_scenes for scene in scenes], dtype=bool))
        require(len(test_rows) and len(train_rows), "single-view outer fold empty")
        correct_predictions = []
        swapped_predictions = []
        seed_records = []
        for seed in seeds:
            model, record = train_single_view_model(
                int(seed), train_rows, arrays, epochs=epochs,
                batch_size=batch_size, device=device)
            correct_predictions.append(predict_single_view(
                model, arrays["features_correct"][test_rows], device))
            swapped_predictions.append(predict_single_view(
                model, arrays["features_goal_swapped"][test_rows], device))
            checkpoint = checkpoint_root / (
                f"single_view_fold{fold_index}_seed{seed}.pt")
            torch.save({
                "schema_version": CHECKPOINT_SCHEMA,
                "feature_dim": feature_dim,
                "fold_index": fold_index,
                "seed": int(seed),
                "test_scenes": sorted(test_scenes),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }, checkpoint)
            record.update({
                "seed": int(seed),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
            })
            seed_records.append(record)
        oof[test_rows] = np.mean(correct_predictions, axis=0)
        oof_swapped[test_rows] = np.mean(swapped_predictions, axis=0)
        fold_records.append({
            "fold_index": fold_index,
            "test_scenes": sorted(test_scenes),
            "train_scene_count": len(set(scenes[train_rows])),
            "test_rows": len(test_rows),
            "seeds": seed_records,
        })
        print(json.dumps({
            "model_kind": "single_view_linear",
            "outer_fold_complete": fold_index,
            "outer_folds": len(folds),
            "test_scenes": sorted(test_scenes),
        }, sort_keys=True), flush=True)
    require(bool(np.isfinite(oof).all() and np.isfinite(oof_swapped).all()),
            "single-view OOF predictions are incomplete")
    return oof, oof_swapped, fold_records


def single_view_metric_report(
        logits: np.ndarray, swapped_logits: np.ndarray,
        arrays: Mapping[str, np.ndarray], *, row_mask: np.ndarray,
        off_axis_only: bool, seed: int) -> Mapping[str, Any]:
    advantages = camera_relative_rings(
        arrays["advantages_m"].astype(np.float64))
    teacher = camera_relative_rings(
        arrays["teacher_distribution"].astype(np.float64))
    oracle = np.argmax(advantages, axis=-1)
    predicted = np.argmax(logits, axis=-1)
    row_indices = np.arange(len(advantages))[:, None]
    view_indices = np.arange(NUM_DIRECTIONS)[None, :]
    selected = advantages[row_indices, view_indices, predicted]
    forward = advantages[..., 0]
    progress_delta = selected - forward
    correct_nll = -np.sum(teacher * _log_softmax(logits), axis=-1)
    swapped_nll = -np.sum(teacher * _log_softmax(swapped_logits), axis=-1)
    nll_increase = swapped_nll - correct_nll
    top1 = (predicted == oracle).astype(np.float64)
    circular_error = _circular_errors(predicted, oracle).astype(np.float64)
    mask = np.broadcast_to(
        np.asarray(row_mask, dtype=bool)[:, None], top1.shape).copy()
    if off_axis_only:
        mask &= oracle != 0
    require(bool(mask.any()), "single-view metric cohort is empty")
    scenes = np.broadcast_to(
        arrays["scene"].astype(object)[:, None], top1.shape)[mask]
    deltas = progress_delta[mask]
    unique_scenes = sorted(set(map(str, scenes)))
    scene_deltas = {
        scene: float(deltas[scenes == scene].mean()) for scene in unique_scenes}
    return {
        "off_axis_only": bool(off_axis_only),
        "top1_exact_bin": summarize_vector(top1[mask], scenes, seed=seed),
        "circular_error_bins": summarize_vector(
            circular_error[mask], scenes, seed=seed + 1),
        "circular_error_degrees": summarize_vector(
            circular_error[mask] * DIRECTION_STEP_DEG,
            scenes, seed=seed + 2),
        "selected_minus_camera_forward_progress_m": summarize_vector(
            deltas, scenes, seed=seed + 3),
        "goal_swap_nll_increase": summarize_vector(
            nll_increase[mask], scenes, seed=seed + 4),
        "paired_progress_counts": {
            "gains_gt_0": int((deltas > 0.0).sum()),
            "ties": int((np.abs(deltas) <= 1e-12).sum()),
            "losses_lt_0": int((deltas < 0.0).sum()),
            "gains_gt_0p25m": int((deltas > USEFUL_PROGRESS_MARGIN_M).sum()),
            "losses_lt_minus_0p25m": int((
                deltas < -USEFUL_PROGRESS_MARGIN_M).sum()),
        },
        "positive_mean_progress_scene_count": sum(
            value > 0.0 for value in scene_deltas.values()),
        "negative_mean_progress_scene_count": sum(
            value < 0.0 for value in scene_deltas.values()),
        "scene_mean_progress_m": scene_deltas,
    }


def global_ring_metric_report(
        logits: np.ndarray, swapped_logits: np.ndarray,
        arrays: Mapping[str, np.ndarray], *, row_mask: np.ndarray,
        seed: int) -> Mapping[str, Any]:
    advantages = arrays["advantages_m"].astype(np.float64)
    teacher = arrays["teacher_distribution"].astype(np.float64)
    oracle = np.argmax(advantages, axis=1)
    predicted = np.argmax(logits, axis=1)
    rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    require(bool(len(rows)), "global-ring metric cohort is empty")
    selected = advantages[np.arange(len(advantages)), predicted]
    correct_nll = -np.sum(teacher * _log_softmax(logits), axis=1)
    swapped_nll = -np.sum(teacher * _log_softmax(swapped_logits), axis=1)
    scenes = arrays["scene"].astype(object)[rows]
    return {
        "top1_exact_bin": summarize_vector(
            (predicted[rows] == oracle[rows]).astype(np.float64),
            scenes, seed=seed),
        "circular_error_degrees": summarize_vector(
            _circular_errors(predicted[rows], oracle[rows]).astype(np.float64)
            * DIRECTION_STEP_DEG, scenes, seed=seed + 1),
        "oracle_regret_m": summarize_vector(
            advantages[rows, oracle[rows]] - selected[rows],
            scenes, seed=seed + 2),
        "goal_swap_nll_increase": summarize_vector(
            (swapped_nll - correct_nll)[rows], scenes, seed=seed + 3),
    }


def view_dose_report(
        relative_logits: np.ndarray, swapped_relative_logits: np.ndarray,
        arrays: Mapping[str, np.ndarray], *, row_mask: np.ndarray,
        seed: int) -> Mapping[str, Any]:
    world = relative_logits_to_world(relative_logits)
    swapped_world = relative_logits_to_world(swapped_relative_logits)
    advantages = arrays["advantages_m"].astype(np.float64)
    teacher = arrays["teacher_distribution"].astype(np.float64)
    oracle = np.argmax(advantages, axis=1)
    selected_rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    result = {}
    for dose in VIEW_DOSES:
        progress_values = []
        nll_values = []
        top1_values = []
        off_axis_values = []
        scenes = []
        for row in selected_rows:
            for start in range(NUM_DIRECTIONS):
                views = [
                    (start + offset) % NUM_DIRECTIONS
                    for offset in range(int(dose))]
                logits = world[row, views].mean(axis=0)
                swapped_logits = swapped_world[row, views].mean(axis=0)
                prediction = int(np.argmax(logits))
                progress_values.append(float(
                    advantages[row, prediction] - advantages[row, start]))
                nll_values.append(float(
                    -np.sum(teacher[row] * _log_softmax(
                        swapped_logits[None])[0])
                    + np.sum(teacher[row] * _log_softmax(logits[None])[0])))
                top1_values.append(float(prediction == int(oracle[row])))
                off_axis_values.append(bool(int(oracle[row]) != start))
                scenes.append(str(arrays["scene"][row]))
        progress = np.asarray(progress_values, dtype=np.float64)
        nll = np.asarray(nll_values, dtype=np.float64)
        top1 = np.asarray(top1_values, dtype=np.float64)
        off_axis = np.asarray(off_axis_values, dtype=bool)
        scene_array = np.asarray(scenes, dtype=object)
        require(bool(off_axis.any()), "view-dose off-axis cohort is empty")
        result[str(dose)] = {
            "views_accumulated": int(dose),
            "all_starts_top1": summarize_vector(
                top1, scene_array, seed=seed + dose * 10),
            "off_axis_selected_minus_start_forward_progress_m": summarize_vector(
                progress[off_axis], scene_array[off_axis],
                seed=seed + dose * 10 + 1),
            "off_axis_goal_swap_nll_increase": summarize_vector(
                nll[off_axis], scene_array[off_axis],
                seed=seed + dose * 10 + 2),
            "off_axis_gains_gt_0p25m": int((
                progress[off_axis] > USEFUL_PROGRESS_MARGIN_M).sum()),
            "off_axis_losses_lt_minus_0p25m": int((
                progress[off_axis] < -USEFUL_PROGRESS_MARGIN_M).sum()),
        }
    return result


def world_prediction_consistency(
        relative_logits: np.ndarray, arrays: Mapping[str, np.ndarray],
        *, row_mask: np.ndarray, seed: int) -> Mapping[str, Any]:
    world_predictions = np.argmax(
        relative_logits_to_world(relative_logits), axis=-1)
    rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    agreement = []
    for row in rows:
        counts = np.bincount(
            world_predictions[row], minlength=NUM_DIRECTIONS)
        agreement.append(float(counts.max() / NUM_DIRECTIONS))
    return summarize_vector(
        np.asarray(agreement), arrays["scene"].astype(object)[rows], seed=seed)


def single_view_gate(metrics: Mapping[str, Any],
                     pair_contrast: Mapping[str, Any]):
    primary = metrics["primary_factual_t0_off_axis"]
    progress = primary["selected_minus_camera_forward_progress_m"][
        "scene_cluster_bootstrap_95"]
    goal = primary["goal_swap_nll_increase"][
        "scene_cluster_bootstrap_95"]
    counts = primary["paired_progress_counts"]
    conditions = {
        "off_axis_progress_lower95_gt_zero": progress["lower_95"] > 0.0,
        "goal_swap_nll_increase_lower95_gt_zero": goal["lower_95"] > 0.0,
        "meaningful_progress_gains_exceed_losses": (
            counts["gains_gt_0p25m"] > counts["losses_lt_minus_0p25m"]),
        "positive_progress_spans_multiple_scenes": (
            primary["positive_mean_progress_scene_count"] >= 2),
        "goal_pair_target_contrast_at_least_25pct": (
            pair_contrast["best_bin_different_rate"] >= 0.25),
    }
    return all(conditions.values()), conditions


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    require(torch.cuda.is_available() or args.device == "cpu",
            "requested CUDA but torch.cuda is unavailable")
    device = torch.device(args.device)
    rows, teacher_report, teacher_dataset = load_dataset(
        args.dataset_root, args.expected_teacher_report_sha256)
    scene_count = len({str(row["scene"]) for row in rows})
    physical_groups = len({str(row["goal_swap_pair_id"]) for row in rows})
    if not args.smoke:
        require(scene_count == FULL_SCENE_COUNT,
                f"formal O1 requires {FULL_SCENE_COUNT} scenes")
        require(physical_groups == FULL_PHYSICAL_GROUP_COUNT,
                f"formal O1 requires {FULL_PHYSICAL_GROUP_COUNT} groups")
        require(len(rows) == FULL_ROW_COUNT,
                f"formal O1 requires {FULL_ROW_COUNT} rows")
        require(args.folds == 5 and args.epochs == 300
                and tuple(args.seeds) == (11, 29, 47),
                "formal O1 schedule is frozen")
        eligibility = teacher_report.get("configuration", {}).get("eligibility")
        require(isinstance(eligibility, Mapping)
                and int(eligibility.get("selected_scene_count", -1))
                == FULL_SCENE_COUNT
                and int(eligibility.get("selected_physical_group_count", -1))
                == FULL_PHYSICAL_GROUP_COUNT,
                "formal teacher eligibility audit is missing")
    require(scene_count >= args.folds, "too few O1 scenes for outer folds")

    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    incomplete = output.with_name(output.name + ".incomplete")
    require(not incomplete.exists(), f"incomplete output exists: {incomplete}")
    incomplete.mkdir(parents=True)
    checkpoints = incomplete / "fold_checkpoints"
    checkpoints.mkdir()
    feature_path = incomplete / "frozen_features.npz"
    episode_roots = (
        args.episode_root.resolve(strict=True),
        args.episode_fallback_root.resolve(strict=True),
    )
    feature_record = extract_features(
        rows,
        dataset_root=args.dataset_root.resolve(),
        episode_roots=episode_roots,
        navdp_root=args.navdp_root.resolve(),
        checkpoint=args.navdp_checkpoint.resolve(),
        expected_checkpoint_sha256=args.expected_navdp_checkpoint_sha256,
        batch_rows=args.feature_batch_rows,
        device=device,
        output_path=feature_path,
    )
    arrays = load_feature_arrays(feature_path)
    folds = deterministic_scene_folds(
        arrays["scene"].astype(str), folds=args.folds, salt=FOLD_SALT)
    pair_contrast = primary_goal_pair_contrast(arrays)

    full_ring = evaluate_kind(
        "linear", arrays, folds=folds, seeds=args.seeds,
        epochs=args.epochs, batch_size=args.batch_size,
        device=device, checkpoint_root=checkpoints)
    full_logits = full_ring.pop("oof_logits")
    full_swapped = full_ring.pop("oof_goal_swapped_logits")

    single_logits, single_swapped, single_folds = evaluate_single_view_oof(
        arrays, folds=folds, seeds=args.seeds, epochs=args.epochs,
        batch_size=args.batch_size, device=device,
        checkpoint_root=checkpoints)
    primary = primary_row_mask(arrays)
    all_factual = arrays["goal_variant"].astype(str) == PRIMARY_VARIANT
    single_metrics = {
        "primary_factual_t0_all_views": single_view_metric_report(
            single_logits, single_swapped, arrays, row_mask=primary,
            off_axis_only=False, seed=2026081001),
        "primary_factual_t0_off_axis": single_view_metric_report(
            single_logits, single_swapped, arrays, row_mask=primary,
            off_axis_only=True, seed=2026081011),
        "all_factual_states_off_axis": single_view_metric_report(
            single_logits, single_swapped, arrays, row_mask=all_factual,
            off_axis_only=True, seed=2026081021),
        "view_dose_primary_factual_t0": view_dose_report(
            single_logits, single_swapped, arrays,
            row_mask=primary, seed=2026081031),
        "world_prediction_consistency_primary_factual_t0": (
            world_prediction_consistency(
                single_logits, arrays, row_mask=primary, seed=2026081041)),
    }
    full_ring_primary = global_ring_metric_report(
        full_logits, full_swapped, arrays, row_mask=primary, seed=2026081051)
    passed, conditions = single_view_gate(single_metrics, pair_contrast)
    decision = (
        "go_single_view_global_to_train_scene_frozen_policy_state_gate"
        if passed else "stop_single_view_goal_bearing_not_observable")

    predictions = incomplete / "oof_predictions.npz"
    np.savez_compressed(
        predictions,
        sample_id=arrays["sample_id"],
        scene=arrays["scene"],
        full_ring_oof_logits=full_logits,
        full_ring_oof_goal_swapped_logits=full_swapped,
        single_view_oof_relative_logits=single_logits,
        single_view_oof_goal_swapped_relative_logits=single_swapped,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "scope": (
            "train-only scene-OOF representation observability; yaw orbit is "
            "training/evaluation supervision, while the tested student consumes "
            "one view; no development, blind, or closed-loop result"),
        "frozen_decision": decision,
        "gate_passed": bool(passed),
        "gate_conditions": conditions,
        "configuration": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "feature_batch_rows": args.feature_batch_rows,
            "fold_salt": FOLD_SALT,
            "folds": [list(fold) for fold in folds],
            "learning_rate": 3e-4,
            "seeds": list(args.seeds),
            "single_view_model": "LayerNorm(no affine)+Linear(384,8)",
            "full_ring_control": "shared LayerNorm+Linear(384,1) per view",
            "teacher_temperature_m": 0.25,
            "useful_progress_margin_m": USEFUL_PROGRESS_MARGIN_M,
            "view_doses": list(VIEW_DOSES),
            "weight_decay": 1e-4,
        },
        "inputs": {
            "teacher_report_sha256": args.expected_teacher_report_sha256,
            "teacher_dataset_sha256": sha256_file(teacher_dataset),
            "navdp_checkpoint_sha256": args.expected_navdp_checkpoint_sha256,
            "feature_archive_sha256": sha256_file(feature_path),
            "oof_predictions_sha256": sha256_file(predictions),
        },
        "dataset_summary": teacher_report["summary"],
        "teacher_eligibility": teacher_report.get(
            "configuration", {}).get("eligibility"),
        "feature_extraction": feature_record,
        "primary_goal_pair_contrast": pair_contrast,
        "single_view": {
            "folds": single_folds,
            "metrics": single_metrics,
        },
        "full_ring_information_control": {
            "training": full_ring,
            "primary_factual_t0": full_ring_primary,
        },
        "limitations": [
            "Habitat geometry is used only to create train-scene supervision",
            "31-scene complete-pair eligibility excludes nine disconnected-goal scenes",
            "expert Novel-B states are not frozen-policy Novel-A failure states",
            "a passing single-view probe authorizes only a train-scene policy-state gate",
            "the full-ring arm is an information control, not a deployable method",
            "no development, final-reserved, or blind scene is read",
        ],
    }
    report_path = incomplete / "report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    (incomplete / "report.json.sha256").write_text(
        f"{sha256_file(report_path)}  report.json\n", encoding="utf-8")
    incomplete.rename(output)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-teacher-report-sha256", required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--episode-fallback-root", type=Path, required=True)
    parser.add_argument("--navdp-root", type=Path, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-navdp-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-rows", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(args.feature_batch_rows > 0 and args.batch_size > 0,
            "batch sizes must be positive")
    require(args.epochs > 0 and args.folds > 1 and bool(args.seeds),
            "training schedule is invalid")
    report = run(args)
    print(json.dumps({
        "frozen_decision": report["frozen_decision"],
        "gate_conditions": report["gate_conditions"],
        "gate_passed": report["gate_passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (O1Error, TrainingError, OSError, ValueError, KeyError) as error:
        print(json.dumps({
            "status": "failed_closed",
            "error": str(error),
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
