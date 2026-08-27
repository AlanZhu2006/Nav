#!/usr/bin/env python3
"""Extract frozen NavDP features and train Cyclic Goal Compass scene-OOF.

This is an observability experiment, not a navigation evaluation.  Every
trainable model sees only the 40 train scenes through outer scene-grouped
folds.  Development/final/blind inputs are not accepted.  Two preregistered
capacities form a ladder rather than a hyperparameter sweep:

``linear``
    one shared linear evidence readout per view;
``ring``
    the same shared projection followed by two circular convolutions.

Both heads are exactly C8-equivariant and have no absolute direction token.
The lower-capacity model wins if it passes the frozen gates.  Goal swapping is
performed within the factual/counterfactual pair at the exact same physical
state and scan, so persistence under swapping is a fail condition.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from MemNavData.circular_goal_compass import (
        NUM_DIRECTIONS,
        CyclicGoalCompass,
        CyclicLinearCompass,
        deterministic_scene_folds,
        masked_listwise_loss,
        scene_cluster_bootstrap,
        scene_macro_mean,
    )
except ImportError:  # direct execution with MemNavData on PYTHONPATH
    from circular_goal_compass import (  # type: ignore
        NUM_DIRECTIONS,
        CyclicGoalCompass,
        CyclicLinearCompass,
        deterministic_scene_folds,
        masked_listwise_loss,
        scene_cluster_bootstrap,
        scene_macro_mean,
    )


FEATURE_SCHEMA = "cgc_frozen_navdp_features_v2"
REPORT_SCHEMA = "cgc_scene_oof_report_v2"
FOLD_SALT = "cgc-scene-oof-v1-20260809"
MODEL_KINDS = ("linear", "ring")
FULL_SCENE_COUNT = 40
FULL_ROW_COUNT = 320
PRIMARY_STATE_NAME = "goal_b_t0"
USEFUL_PROGRESS_MARGIN_M = 0.25
RISK_COVERAGES = (0.25, 0.50, 0.75, 1.00)


class TrainingError(RuntimeError):
    """The CGC artifact or training protocol failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8")


def load_dataset(dataset_root: Path, expected_report_sha256: str):
    report_path = dataset_root / "report.json"
    require(report_path.is_file(), "teacher report is missing")
    require(sha256_file(report_path) == expected_report_sha256,
            "teacher report SHA256 changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("schema_version") == "cgc_multiyaw_teacher_report_v2",
            "teacher report schema changed")
    require(report.get("status") == "complete", "teacher artifact is incomplete")
    require(report.get("configuration", {}).get(
        "deployment_candidate_mask_used") is False,
        "teacher artifact depends on a privileged deployment mask")
    dataset_path = dataset_root / report["dataset"]["relative_path"]
    require(dataset_path.is_file(), "teacher JSONL is missing")
    require(sha256_file(dataset_path) == report["dataset"]["sha256"],
            "teacher JSONL SHA256 changed")
    raw = dataset_path.read_bytes()
    require(raw.endswith(b"\n"), "teacher JSONL is not newline terminated")
    rows = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        value = json.loads(line)
        require(line == canonical_json_bytes(value),
                f"teacher row {line_number} is not canonical")
        require(value.get("schema_version") == "cgc_multiyaw_teacher_v2",
                "teacher row schema changed")
        require(len(value["scan_rgb_relative_paths"]) == NUM_DIRECTIONS,
                "teacher scan is not C8")
        require(len(value["advantages_m"]) == NUM_DIRECTIONS
                and len(value["candidate_valid"]) == NUM_DIRECTIONS,
                "teacher field is not C8")
        require(all(map(bool, value["candidate_valid"])),
                "deployable CGC cannot consume a privileged candidate mask")
        require(len(value["candidate_action_faithful"]) == NUM_DIRECTIONS,
                "counterfactual action diagnostics are not C8")
        rows.append(value)
    require(bool(rows), "teacher dataset is empty")
    return rows, report, dataset_path


def exact_navdp_preprocess(rgb: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Byte-for-byte algorithmic copy of NavDP_Agent.process_image."""
    require(rgb.ndim == 3 and rgb.shape[2] == 3, "RGB input shape is invalid")
    height, width = rgb.shape[:2]
    proportion = image_size / max(height, width)
    resized = cv2.resize(rgb, (-1, -1), fx=proportion, fy=proportion)
    pad_width = max((image_size - resized.shape[1]) // 2, 0)
    pad_height = max((image_size - resized.shape[0]) // 2, 0)
    padded = np.pad(
        resized,
        ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
        mode="constant", constant_values=0)
    resized = cv2.resize(padded, (image_size, image_size))
    return np.asarray(resized, dtype=np.float32) / 255.0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resolve_episode_file(relative: str, roots: Sequence[Path], label: str) -> Path:
    fragment = Path(relative)
    require(not fragment.is_absolute() and ".." not in fragment.parts,
            f"{label} relative path is unsafe")
    matches: list[Path] = []
    for root in roots:
        root_resolved = root.resolve(strict=True)
        candidate = (root_resolved / fragment).resolve(strict=False)
        try:
            candidate.relative_to(root_resolved)
        except ValueError as error:
            raise TrainingError(f"{label} path escapes episode root") from error
        if candidate.is_file():
            matches.append(candidate)
    require(len(matches) == 1,
            f"{label} must resolve in exactly one pinned episode root")
    return matches[0]


def build_swap_index(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["goal_swap_pair_id"])].append(index)
    swap = np.full(len(rows), -1, dtype=np.int64)
    for group_id, indices in groups.items():
        require(len(indices) == 2, f"goal-swap pair is incomplete: {group_id}")
        first, second = indices
        require(rows[first]["goal_variant"] != rows[second]["goal_variant"],
                f"goal-swap variants duplicated: {group_id}")
        swap[first], swap[second] = second, first
    require(bool((swap >= 0).all()), "goal-swap index is incomplete")
    return swap


def load_frozen_imagegoal_encoder(navdp_root: Path, checkpoint: Path,
                                  expected_checkpoint_sha256: str,
                                  device: torch.device):
    require(checkpoint.is_file(), "NavDP checkpoint is missing")
    require(sha256_file(checkpoint) == expected_checkpoint_sha256,
            "NavDP checkpoint SHA256 changed")
    policy_backbone = navdp_root / "policy_backbone.py"
    require(policy_backbone.is_file(), "NavDP policy_backbone.py is missing")
    sys.path.insert(0, str(navdp_root))
    try:
        module = importlib.import_module("policy_backbone")
        model = module.NavDP_ImageGoal_Backbone(
            image_size=224, embed_size=384, device=str(device))
    finally:
        sys.path.pop(0)
    checkpoint_state = torch.load(
        checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint_state, Mapping),
            "NavDP checkpoint is not a state dict")
    prefix = "image_encoder."
    encoder_state = {
        key[len(prefix):]: value for key, value in checkpoint_state.items()
        if key.startswith(prefix)
    }
    require(len(encoder_state) == 177,
            f"unexpected ImageGoal tensor count: {len(encoder_state)}")
    model.load_state_dict(encoder_state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, sha256_file(policy_backbone)


def verify_input_images(rows: Sequence[Mapping[str, Any]], dataset_root: Path,
                        episode_roots: Sequence[Path]) -> None:
    checked_views: dict[str, str] = {}
    checked_goals: dict[str, str] = {}
    for row in rows:
        for relative, expected in zip(
                row["scan_rgb_relative_paths"], row["scan_rgb_sha256"]):
            path = dataset_root / relative
            key = str(path)
            if key not in checked_views:
                require(path.is_file() and sha256_file(path) == expected,
                        f"scan RGB changed: {path}")
                checked_views[key] = expected
            else:
                require(checked_views[key] == expected,
                        "shared scan has conflicting hashes")
        goal = resolve_episode_file(
            str(row["goal_relative_path"]), episode_roots,
            f"{row['sample_id']} goal")
        key = str(goal)
        expected = str(row["goal_content_sha256"])
        if key not in checked_goals:
            require(goal.is_file() and sha256_file(goal) == expected,
                    f"goal image changed: {goal}")
            checked_goals[key] = expected
        else:
            require(checked_goals[key] == expected,
                    "shared goal has conflicting hashes")


def extract_features(rows: Sequence[Mapping[str, Any]], *, dataset_root: Path,
                     episode_roots: Sequence[Path], navdp_root: Path,
                     checkpoint: Path,
                     expected_checkpoint_sha256: str, batch_rows: int,
                     device: torch.device, output_path: Path):
    verify_input_images(rows, dataset_root, episode_roots)
    swap_index = build_swap_index(rows)
    model, backbone_source_sha = load_frozen_imagegoal_encoder(
        navdp_root, checkpoint, expected_checkpoint_sha256, device)
    feature_dim = 384
    correct = np.empty(
        (len(rows), NUM_DIRECTIONS, feature_dim), dtype=np.float16)
    swapped = np.empty_like(correct)

    with torch.inference_mode():
        for start in range(0, len(rows), batch_rows):
            end = min(start + batch_rows, len(rows))
            pair_inputs: list[np.ndarray] = []
            assignments: list[tuple[str, int, int]] = []
            for row_index in range(start, end):
                row = rows[row_index]
                own_goal = exact_navdp_preprocess(load_rgb(resolve_episode_file(
                    str(row["goal_relative_path"]), episode_roots,
                    f"{row['sample_id']} goal")))
                other = rows[int(swap_index[row_index])]
                swapped_goal = exact_navdp_preprocess(load_rgb(
                    resolve_episode_file(
                        str(other["goal_relative_path"]), episode_roots,
                        f"{other['sample_id']} swapped goal")))
                for view_index, relative in enumerate(
                        row["scan_rgb_relative_paths"]):
                    view = exact_navdp_preprocess(load_rgb(
                        dataset_root / relative))
                    pair_inputs.append(np.concatenate((own_goal, view), axis=-1))
                    assignments.append(("correct", row_index, view_index))
                    pair_inputs.append(np.concatenate((swapped_goal, view), axis=-1))
                    assignments.append(("swapped", row_index, view_index))
            inputs = np.stack(pair_inputs, axis=0)
            # Bound peak memory independently of the number of logical rows.
            cursor = 0
            while cursor < len(inputs):
                chunk_end = min(cursor + 64, len(inputs))
                embeddings = model(inputs[cursor:chunk_end]).detach().cpu().numpy()
                require(embeddings.shape == (chunk_end - cursor, feature_dim),
                        "NavDP ImageGoal feature shape changed")
                for local_index, embedding in enumerate(embeddings):
                    kind, row_index, view_index = assignments[cursor + local_index]
                    target = correct if kind == "correct" else swapped
                    target[row_index, view_index] = embedding.astype(np.float16)
                cursor = chunk_end
            print(json.dumps({
                "feature_rows_complete": end,
                "feature_rows_total": len(rows),
            }, sort_keys=True), flush=True)

    advantages = np.zeros((len(rows), NUM_DIRECTIONS), dtype=np.float32)
    valid = np.zeros((len(rows), NUM_DIRECTIONS), dtype=bool)
    teacher = np.zeros((len(rows), NUM_DIRECTIONS), dtype=np.float32)
    for index, row in enumerate(rows):
        valid[index] = np.asarray(row["candidate_valid"], dtype=bool)
        advantages[index, valid[index]] = np.asarray([
            float(value) for value, is_valid in zip(
                row["advantages_m"], valid[index]) if is_valid
        ], dtype=np.float32)
        teacher[index] = np.asarray(row["teacher_distribution"], dtype=np.float32)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([FEATURE_SCHEMA]),
        features_correct=correct,
        features_goal_swapped=swapped,
        advantages_m=advantages,
        candidate_valid=valid,
        teacher_distribution=teacher,
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        goal_swap_pair_id=np.asarray([
            row["goal_swap_pair_id"] for row in rows]),
        scene=np.asarray([row["scene"] for row in rows]),
        state_name=np.asarray([row["state_name"] for row in rows]),
        goal_variant=np.asarray([row["goal_variant"] for row in rows]),
        native_scan_index=np.asarray([
            row["native_scan_index"] for row in rows], dtype=np.int64),
        initial_geodesic_distance_m=np.asarray([
            row["initial_geodesic_distance_m"] for row in rows],
            dtype=np.float32),
    )
    return {
        "schema_version": FEATURE_SCHEMA,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "navdp_policy_backbone_sha256": backbone_source_sha,
        "feature_rows": len(rows),
        "feature_dim": feature_dim,
        "features_sha256": sha256_file(output_path),
        "correct_and_goal_swapped_share_scan": True,
    }


def load_feature_arrays(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(archive["schema_version"].tolist() == [FEATURE_SCHEMA],
                "feature archive schema changed")
        return {key: archive[key].copy() for key in archive.files}


def make_model(kind: str, feature_dim: int):
    if kind == "linear":
        return CyclicLinearCompass(feature_dim)
    if kind == "ring":
        return CyclicGoalCompass(feature_dim, hidden_dim=128)
    raise TrainingError(f"unknown model kind: {kind}")


def cyclic_roll_batch(features, advantages, valid, generator):
    shifts = torch.randint(
        0, NUM_DIRECTIONS, (features.shape[0],), generator=generator,
        device="cpu")
    rolled_features = torch.stack([
        torch.roll(features[index], int(shifts[index]), dims=0)
        for index in range(features.shape[0])
    ])
    rolled_advantages = torch.stack([
        torch.roll(advantages[index], int(shifts[index]), dims=0)
        for index in range(features.shape[0])
    ])
    rolled_valid = torch.stack([
        torch.roll(valid[index], int(shifts[index]), dims=0)
        for index in range(features.shape[0])
    ])
    return rolled_features, rolled_advantages, rolled_valid


def equivariance_error(model, feature_dim: int, device: torch.device) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260809)
    inputs = torch.randn(
        (3, NUM_DIRECTIONS, feature_dim), generator=generator,
        device="cpu").to(device)
    with torch.inference_mode():
        reference = model(inputs)
        errors = []
        for shift in range(NUM_DIRECTIONS):
            shifted = model(torch.roll(inputs, shift, dims=1))
            errors.append(float((
                shifted - torch.roll(reference, shift, dims=1)
            ).abs().max().item()))
    return max(errors)


def train_model(kind: str, seed: int, train_indices: np.ndarray,
                arrays: Mapping[str, np.ndarray], *, epochs: int,
                batch_size: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    feature_dim = int(arrays["features_correct"].shape[-1])
    model = make_model(kind, feature_dim).to(device)
    error = equivariance_error(model, feature_dim, device)
    require(error <= 2e-5, f"{kind} C8 equivariance failed: {error}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4)
    features = torch.as_tensor(
        arrays["features_correct"][train_indices],
        dtype=torch.float32, device=device)
    advantages = torch.as_tensor(
        arrays["advantages_m"][train_indices],
        dtype=torch.float32, device=device)
    valid = torch.as_tensor(
        arrays["candidate_valid"][train_indices],
        dtype=torch.bool, device=device)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed ^ 0x5A17)
    first_gradient_norm = None
    final_loss = None
    model.train()
    for epoch in range(int(epochs)):
        permutation = torch.randperm(
            len(train_indices), generator=cpu_generator).tolist()
        epoch_losses = []
        for offset in range(0, len(permutation), int(batch_size)):
            batch = permutation[offset:offset + int(batch_size)]
            batch_features, batch_advantages, batch_valid = cyclic_roll_batch(
                features[batch], advantages[batch], valid[batch], cpu_generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = masked_listwise_loss(
                logits, batch_advantages, batch_valid)
            require(bool(torch.isfinite(loss)), "training loss became non-finite")
            loss.backward()
            if first_gradient_norm is None:
                squared = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        squared += float(parameter.grad.detach().square().sum().item())
                first_gradient_norm = math.sqrt(squared)
                require(math.isfinite(first_gradient_norm)
                        and first_gradient_norm > 0.0,
                        "first trainable gradient is zero or non-finite")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(epoch_losses))
    require(final_loss is not None and first_gradient_norm is not None,
            "training loop did not execute")
    return model.eval(), {
        "final_train_loss": final_loss,
        "first_gradient_norm": first_gradient_norm,
        "equivariance_max_abs_error": error,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def predict(model, features: np.ndarray, device: torch.device,
            batch_size: int = 128) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[offset:offset + batch_size],
                dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def masked_log_softmax(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, logits, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponent = np.where(valid, np.exp(masked - maximum), 0.0)
    normalizer = exponent.sum(axis=1, keepdims=True)
    require(bool((normalizer > 0.0).all()), "prediction has no valid direction")
    result = np.full_like(masked, -np.inf)
    result[valid] = np.log((exponent / normalizer)[valid])
    return result


def summarize_vector(values: np.ndarray, scenes: np.ndarray, *, seed: int):
    return {
        "scene_macro_mean": scene_macro_mean(values, scenes),
        "scene_cluster_bootstrap_95": scene_cluster_bootstrap(
            values, scenes, seed=seed, resamples=5000),
        "rows": int(len(values)),
        "scene_clusters": int(len(set(map(str, scenes)))),
    }


def risk_coverage(logits: np.ndarray, advantages: np.ndarray,
                  valid: np.ndarray, native_indices: np.ndarray,
                  scenes: np.ndarray, indices: np.ndarray, *, seed: int):
    eligible = []
    for index in indices:
        native = int(native_indices[index])
        if not bool(valid[index, native]):
            continue
        candidates = np.where(valid[index])[0]
        non_native = candidates[candidates != native]
        if not len(non_native):
            continue
        chosen = int(non_native[np.argmax(logits[index, non_native])])
        margin = float(logits[index, chosen] - logits[index, native])
        delta = float(advantages[index, chosen] - advantages[index, native])
        eligible.append((int(index), margin, delta, chosen, native))
    require(bool(eligible), "risk-coverage set contains no eligible state")
    eligible.sort(key=lambda row: (-row[1], row[0]))
    result = {}
    for coverage in RISK_COVERAGES:
        budget_count = max(1, int(round(float(coverage) * len(eligible))))
        budgeted = eligible[:budget_count]
        # A budget is an upper bound, not a command to take over.  If native
        # still has the larger score, deployment abstains even when budget is
        # unused.
        selected = {row[0]: row for row in budgeted if row[1] > 0.0}
        net_values = []
        net_scenes = []
        selected_deltas = []
        for index, _margin, delta, _chosen, _native in eligible:
            value = delta if index in selected else 0.0
            net_values.append(value)
            net_scenes.append(str(scenes[index]))
            if index in selected:
                selected_deltas.append(delta)
        selected_array = np.asarray(selected_deltas, dtype=np.float64)
        result[f"{int(coverage * 100)}"] = {
            "eligible_rows": len(eligible),
            "budget_rows": budget_count,
            "positive_margin_rows": sum(row[1] > 0.0 for row in eligible),
            "intervened_rows": len(selected),
            "actual_coverage": len(selected) / len(eligible),
            "gains_gt_0p25m": int((
                selected_array > USEFUL_PROGRESS_MARGIN_M).sum()),
            "losses_lt_minus_0p25m": int((
                selected_array < -USEFUL_PROGRESS_MARGIN_M).sum()),
            "mean_delta_when_intervened_m": (
                float(selected_array.mean()) if len(selected_array) else None),
            "net_delta_all_states_m": summarize_vector(
                np.asarray(net_values, dtype=np.float64),
                np.asarray(net_scenes, dtype=object), seed=seed),
        }
    return result


def metric_report(logits: np.ndarray, swapped_logits: np.ndarray,
                  arrays: Mapping[str, np.ndarray], *, indices: np.ndarray,
                  seed: int):
    advantages = arrays["advantages_m"].astype(np.float64)
    valid = arrays["candidate_valid"].astype(bool)
    teacher = arrays["teacher_distribution"].astype(np.float64)
    scenes = arrays["scene"].astype(object)
    native = arrays["native_scan_index"].astype(np.int64)
    prediction_masked = np.where(valid, logits, -np.inf)
    swapped_masked = np.where(valid, swapped_logits, -np.inf)
    predictions = np.argmax(prediction_masked, axis=1)
    swapped_predictions = np.argmax(swapped_masked, axis=1)
    oracle = np.argmax(np.where(valid, advantages, -np.inf), axis=1)
    log_prob = masked_log_softmax(logits, valid)
    swapped_log_prob = masked_log_softmax(swapped_logits, valid)
    safe_log_prob = np.where(valid, log_prob, 0.0)
    safe_swapped_log_prob = np.where(valid, swapped_log_prob, 0.0)
    correct_nll = -np.sum(teacher * safe_log_prob, axis=1)
    swapped_nll = -np.sum(teacher * safe_swapped_log_prob, axis=1)
    row_indices = np.arange(len(scenes))
    selected_advantage = advantages[row_indices, predictions]
    oracle_advantage = advantages[row_indices, oracle]
    regret = oracle_advantage - selected_advantage
    top1 = (predictions == oracle).astype(np.float64)
    swapped_top1 = (swapped_predictions == oracle).astype(np.float64)
    goal_nll_delta = swapped_nll - correct_nll
    subset_scenes = scenes[indices]
    return {
        "top1": summarize_vector(top1[indices], subset_scenes, seed=seed),
        "goal_swapped_top1": summarize_vector(
            swapped_top1[indices], subset_scenes, seed=seed + 1),
        "top1_correct_minus_swapped": summarize_vector(
            (top1 - swapped_top1)[indices], subset_scenes, seed=seed + 2),
        "angular_field_regret_m": summarize_vector(
            regret[indices], subset_scenes, seed=seed + 3),
        "goal_swap_nll_increase": summarize_vector(
            goal_nll_delta[indices], subset_scenes, seed=seed + 4),
        "risk_coverage_vs_native_heading_proxy": risk_coverage(
            logits, advantages, valid, native, scenes, indices, seed=seed + 5),
    }


def evaluate_kind(kind: str, arrays: Mapping[str, np.ndarray], *, folds,
                  seeds: Sequence[int], epochs: int, batch_size: int,
                  device: torch.device, checkpoint_root: Path):
    scenes = arrays["scene"].astype(str)
    feature_dim = int(arrays["features_correct"].shape[-1])
    oof = np.full((len(scenes), NUM_DIRECTIONS), np.nan, dtype=np.float64)
    oof_swapped = np.full_like(oof, np.nan)
    fold_records = []
    for fold_index, test_scenes_tuple in enumerate(folds):
        test_scenes = set(test_scenes_tuple)
        test_indices = np.flatnonzero(np.asarray([
            scene in test_scenes for scene in scenes], dtype=bool))
        train_indices = np.flatnonzero(np.asarray([
            scene not in test_scenes for scene in scenes], dtype=bool))
        require(len(test_indices) and len(train_indices), "empty outer fold")
        seed_predictions = []
        seed_swapped_predictions = []
        seed_records = []
        for seed in seeds:
            model, training_record = train_model(
                kind, int(seed), train_indices, arrays,
                epochs=epochs, batch_size=batch_size, device=device)
            seed_predictions.append(predict(
                model, arrays["features_correct"][test_indices], device))
            seed_swapped_predictions.append(predict(
                model, arrays["features_goal_swapped"][test_indices], device))
            checkpoint_path = checkpoint_root / (
                f"{kind}_fold{fold_index}_seed{seed}.pt")
            torch.save({
                "schema_version": "cgc_scene_fold_checkpoint_v1",
                "model_kind": kind,
                "feature_dim": feature_dim,
                "fold_index": fold_index,
                "seed": int(seed),
                "test_scenes": sorted(test_scenes),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }, checkpoint_path)
            training_record.update({
                "seed": int(seed),
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            })
            seed_records.append(training_record)
        oof[test_indices] = np.mean(seed_predictions, axis=0)
        oof_swapped[test_indices] = np.mean(seed_swapped_predictions, axis=0)
        fold_records.append({
            "fold_index": fold_index,
            "test_scenes": sorted(test_scenes),
            "train_scene_count": len(set(scenes[train_indices])),
            "test_rows": len(test_indices),
            "seeds": seed_records,
        })
        print(json.dumps({
            "model_kind": kind,
            "outer_fold_complete": fold_index,
            "outer_folds": len(folds),
            "test_scenes": sorted(test_scenes),
        }, sort_keys=True), flush=True)
    require(bool(np.isfinite(oof).all() and np.isfinite(oof_swapped).all()),
            f"{kind} OOF predictions are incomplete")
    all_indices = np.arange(len(scenes), dtype=np.int64)
    primary_indices = np.flatnonzero(
        arrays["state_name"].astype(str) == PRIMARY_STATE_NAME)
    return {
        "oof_logits": oof,
        "oof_goal_swapped_logits": oof_swapped,
        "folds": fold_records,
        "metrics": {
            "all_novel_b_states": metric_report(
                oof, oof_swapped, arrays, indices=all_indices,
                seed=2026080900 + (0 if kind == "linear" else 100)),
            "primary_goal_b_t0": metric_report(
                oof, oof_swapped, arrays, indices=primary_indices,
                seed=2026080910 + (0 if kind == "linear" else 100)),
        },
    }


def model_passes(metrics: Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
    primary = metrics["primary_goal_b_t0"]
    net = primary["risk_coverage_vs_native_heading_proxy"]["50"][
        "net_delta_all_states_m"]["scene_cluster_bootstrap_95"]
    goal = primary["goal_swap_nll_increase"][
        "scene_cluster_bootstrap_95"]
    conditions = {
        "primary_50pct_net_progress_lower95_gt_zero": net["lower_95"] > 0.0,
        "primary_50pct_budget_realized_coverage_at_least_25pct": (
            primary["risk_coverage_vs_native_heading_proxy"]["50"][
                "actual_coverage"] >= 0.25),
        "primary_goal_swap_nll_increase_lower95_gt_zero": (
            goal["lower_95"] > 0.0),
    }
    return all(conditions.values()), conditions


def primary_goal_pair_contrast(arrays: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    pair_ids = arrays["goal_swap_pair_id"].astype(str)
    state_names = arrays["state_name"].astype(str)
    advantages = arrays["advantages_m"].astype(np.float64)
    valid = arrays["candidate_valid"].astype(bool)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (pair_id, state_name) in enumerate(zip(pair_ids, state_names)):
        if state_name == PRIMARY_STATE_NAME:
            groups[pair_id].append(index)
    differences = []
    for pair_id, indices in groups.items():
        require(len(indices) == 2,
                f"primary goal-swap pair is incomplete: {pair_id}")
        best = [int(np.argmax(np.where(
            valid[index], advantages[index], -np.inf))) for index in indices]
        raw = abs(best[0] - best[1])
        differences.append(min(raw, NUM_DIRECTIONS - raw))
    require(bool(differences), "primary goal-swap contrast set is empty")
    different = sum(value > 0 for value in differences)
    at_least_90 = sum(value >= 2 for value in differences)
    return {
        "pair_count": len(differences),
        "best_bin_different": different,
        "best_bin_different_rate": different / len(differences),
        "best_bin_at_least_90deg": at_least_90,
        "best_bin_at_least_90deg_rate": at_least_90 / len(differences),
    }


def train_final_ensemble(kind: str, arrays: Mapping[str, np.ndarray], *,
                         seeds: Sequence[int], epochs: int, batch_size: int,
                         device: torch.device, checkpoint_root: Path):
    indices = np.arange(len(arrays["scene"]), dtype=np.int64)
    feature_dim = int(arrays["features_correct"].shape[-1])
    records = []
    for seed in seeds:
        model, training_record = train_model(
            kind, int(seed), indices, arrays, epochs=epochs,
            batch_size=batch_size, device=device)
        checkpoint_path = checkpoint_root / f"{kind}_all40_seed{seed}.pt"
        torch.save({
            "schema_version": "cgc_all_train_checkpoint_v1",
            "model_kind": kind,
            "feature_dim": feature_dim,
            "seed": int(seed),
            "train_scene_count": int(len(set(
                arrays["scene"].astype(str)))),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }, checkpoint_path)
        training_record.update({
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        records.append(training_record)
    return records


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    require(torch.cuda.is_available() or args.device == "cpu",
            "requested CUDA but torch.cuda is unavailable")
    device = torch.device(args.device)
    rows, teacher_report, dataset_path = load_dataset(
        args.dataset_root, args.expected_teacher_report_sha256)
    scene_count = len({str(row["scene"]) for row in rows})
    if not args.smoke:
        require(scene_count == FULL_SCENE_COUNT,
                f"formal run requires {FULL_SCENE_COUNT} scenes")
        require(len(rows) == FULL_ROW_COUNT,
                f"formal run requires {FULL_ROW_COUNT} rows")
        require(args.folds == 5 and args.epochs == 300
                and tuple(args.seeds) == (11, 29, 47),
                "formal model/fold schedule is frozen")
    require(scene_count >= args.folds, "too few scenes for outer folds")

    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    incomplete = output.with_name(output.name + ".incomplete")
    require(not incomplete.exists(), f"incomplete output exists: {incomplete}")
    incomplete.mkdir(parents=True)
    checkpoint_root = incomplete / "fold_checkpoints"
    checkpoint_root.mkdir()
    feature_path = incomplete / "frozen_features.npz"
    raw_episode_roots = (args.episode_root, args.episode_fallback_root)
    for root in raw_episode_roots:
        require(root.is_dir() and not root.is_symlink(),
                f"episode root is missing or symbolic: {root}")
    episode_roots = tuple(root.resolve(strict=True)
                          for root in raw_episode_roots)
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
    scenes = arrays["scene"].astype(str)
    folds = deterministic_scene_folds(
        scenes, folds=args.folds, salt=FOLD_SALT)
    primary_pair_contrast = primary_goal_pair_contrast(arrays)
    data_contrast_pass = (
        primary_pair_contrast["best_bin_different_rate"] >= 0.25)
    kind_results = {}
    prediction_payload = {}
    decisions = {}
    selected_kind = None
    for kind in MODEL_KINDS:
        result = evaluate_kind(
            kind, arrays, folds=folds, seeds=args.seeds,
            epochs=args.epochs, batch_size=args.batch_size,
            device=device, checkpoint_root=checkpoint_root)
        prediction_payload[f"{kind}_oof_logits"] = result.pop("oof_logits")
        prediction_payload[f"{kind}_oof_goal_swapped_logits"] = result.pop(
            "oof_goal_swapped_logits")
        kind_results[kind] = result
        passed, conditions = model_passes(result["metrics"])
        passed = bool(passed and data_contrast_pass)
        conditions = dict(conditions)
        conditions["goal_pair_target_contrast_at_least_25pct"] = (
            data_contrast_pass)
        decisions[kind] = {"passed": passed, "conditions": conditions}
        if passed:
            selected_kind = kind
            break

    if selected_kind is not None:
        selected_index = MODEL_KINDS.index(selected_kind)
        for kind in MODEL_KINDS[selected_index + 1:]:
            decisions[kind] = {
                "passed": False,
                "not_evaluated_reason": "lower_capacity_model_passed",
            }

    prediction_path = incomplete / "oof_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_id=arrays["sample_id"],
        scene=arrays["scene"],
        **prediction_payload,
    )
    frozen_decision = (
        f"go_{selected_kind}_to_preregistered_disjoint_policy_state_gate"
        if selected_kind is not None
        else "stop_active_goal_compass_observability_not_established"
    )
    final_ensemble = (
        train_final_ensemble(
            selected_kind, arrays, seeds=args.seeds, epochs=args.epochs,
            batch_size=args.batch_size, device=device,
            checkpoint_root=checkpoint_root)
        if selected_kind is not None else []
    )

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "scope": (
            "train-only five-fold scene OOF observability; no development, "
            "final-reserved, blind, or closed-loop result"),
        "frozen_decision": frozen_decision,
        "selected_model_kind": selected_kind,
        "evaluated_model_kinds": list(kind_results),
        "architecture_selection_order": list(MODEL_KINDS),
        "selection_rule": (
            "choose linear if it passes all gates; otherwise choose ring only "
            "if it passes; otherwise stop"),
        "configuration": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "fold_salt": FOLD_SALT,
            "folds": [list(fold) for fold in folds],
            "learning_rate": 3e-4,
            "model_kinds": list(MODEL_KINDS),
            "risk_coverages": list(RISK_COVERAGES),
            "seeds": list(args.seeds),
            "teacher_temperature_m": 0.25,
            "useful_progress_margin_m": USEFUL_PROGRESS_MARGIN_M,
            "weight_decay": 1e-4,
            "deployment_candidate_mask_used": False,
        },
        "inputs": {
            "teacher_report_sha256": args.expected_teacher_report_sha256,
            "teacher_dataset_sha256": sha256_file(dataset_path),
            "navdp_checkpoint_sha256": args.expected_navdp_checkpoint_sha256,
            "feature_archive_sha256": sha256_file(feature_path),
            "oof_predictions_sha256": sha256_file(prediction_path),
        },
        "feature_extraction": feature_record,
        "dataset_summary": teacher_report["summary"],
        "primary_goal_pair_contrast": primary_pair_contrast,
        "model_results": kind_results,
        "gate_decisions": decisions,
        "selected_all_train_ensemble": final_ensemble,
        "limitations": [
            "teacher progress uses Habitat privileged geometry only as supervision",
            "the active scan cost and scan-induced state changes are not evaluated",
            "expert Novel-B states are not frozen-policy Novel-A failure states",
            "native_scan_index is an expert-heading proxy, not a fresh NavDP proposal",
            "passing authorizes only a preregistered disjoint policy-state gate",
            "development and blind scenes are not authorized for model selection",
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
        "gate_decisions": report["gate_decisions"],
        "selected_model_kind": report["selected_model_kind"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (TrainingError, OSError, ValueError, KeyError) as error:
        print(json.dumps({
            "status": "failed_closed",
            "error": str(error),
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
