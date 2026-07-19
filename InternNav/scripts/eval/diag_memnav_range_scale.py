"""Paired local diagnostic for MemNav's revisit range representation.

This is deliberately an offline training-noise diagnostic, not a closed-loop
navigation benchmark.  It encodes each batch once and keeps retrieval, bearing,
gate, images, diffusion noise, and diffusion timestep fixed while replacing only
coordinate 2 of the robust revisit pose code (the compressed range).

The variants answer separate questions:

* ``zero_range``: does the trained decoder use range at all?
* ``oracle_stream_range``: would a perfect range in the current stream-normalized
  convention help?  It uses GT goal distance divided by the median *past* GT step.
* ``odom_metric_range``: can past executed motion calibrate LingBot's normalized
  range into a canonical metric scale without looking at the future goal?
* ``oracle_metric_range``: upper bound for that metric calibration.

Only revisit rows are changed.  Novel rows retain their original pose code, since
their revisit branch is intentionally gated out and their pose is not meaningful.
"""

import argparse
import copy
import functools
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from internnav.dataset.memnav_dataset_lerobot import (
    MemNav_Dataset,
    build_fixed_memnav_eval_subset,
    memnav_collate_fn,
)
from internnav.model.basemodel.memnav.metrics import compute_memnav_batch_records
from scripts.eval.eval_memnav_offline import load_checkpoint
from scripts.train.configs.memnav import memnav_exp_cfg


VARIANTS = (
    "current",
    "zero_range",
    "oracle_stream_range",
    "odom_metric_range",
    "oracle_metric_range",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--lingbot-repo", default=None)
    parser.add_argument("--lingbot-weights", default=None)
    parser.add_argument("--data-split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--subset-mode", choices=("balanced-fixed", "random"), default="balanced-fixed"
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--oracle-positive-anchor",
        action="store_true",
        help="anchor goal-pose insertion on the best-scoring GT-positive history frame",
    )
    parser.add_argument(
        "--nominal-step-m",
        type=float,
        default=0.0376,
        help="canonical generator cruise distance represented by one range step",
    )
    parser.add_argument("--recent-step-window", type=int, default=64)
    parser.add_argument(
        "--full-diffusion",
        action="store_true",
        help="also compare complete paired DDPM action samples for every range variant",
    )
    parser.add_argument("--diffusion-seed", type=int, default=104729)
    parser.add_argument("--diffusion-repeats", type=int, default=1)
    return parser.parse_args()


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _episode_from_cache(cache_path, feature_root, root_dir):
    cache_path = Path(cache_path).resolve()
    feature_root = Path(feature_root).resolve()
    try:
        relative = cache_path.relative_to(feature_root)
    except ValueError as error:
        raise ValueError(f"cache {cache_path} is not under feature root {feature_root}") from error
    try:
        videos_index = relative.parts.index("videos")
    except ValueError as error:
        raise ValueError(f"cache path has no videos component: {relative}") from error
    episode_relative = Path(*relative.parts[:videos_index])
    episode_dir = Path(root_dir).resolve() / episode_relative
    parquet = episode_dir / "data/chunk-000/episode_000000.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    return episode_relative.as_posix(), parquet


@functools.lru_cache(maxsize=None)
def _load_gt_xy(parquet_path):
    dataframe = pd.read_parquet(parquet_path)
    action = np.asarray(
        [np.stack(item) for item in dataframe["action"]], dtype=np.float64
    ).reshape(-1, 4, 4)
    return action[:, :2, 3]


def _robust_step_m(positions, cur_step, recent_window):
    end = min(int(cur_step) + 1, len(positions))
    if end < 2:
        raise ValueError(f"need at least two past poses, got end={end}")
    step = np.linalg.norm(np.diff(positions[:end], axis=0), axis=1)
    valid = np.isfinite(step) & (step > 1e-6)
    if not np.any(valid):
        raise ValueError("past trajectory contains no non-zero finite motion")
    prefix = float(np.median(step[valid]))
    recent = step[max(0, len(step) - int(recent_window)) :]
    recent_valid = np.isfinite(recent) & (recent > 1e-6)
    recent_value = float(np.median(recent[recent_valid])) if np.any(recent_valid) else prefix
    return prefix, recent_value


def _step_rows(batch, feature_root, root_dir, recent_window):
    rows = []
    for cache_path, cur_step in zip(batch["cache_paths"], batch["cur_steps"]):
        episode, parquet = _episode_from_cache(cache_path, feature_root, root_dir)
        prefix, recent = _robust_step_m(
            _load_gt_xy(str(parquet)), cur_step, recent_window
        )
        rows.append((episode, prefix, recent))
    return rows


def _pose_condition(core, enc, current_state, novel, R_rel, encoded, pose_code):
    merge = core.revisit_merge
    rel_feat = pose_code + merge.rel_adapter(pose_code)
    aux_pose = merge.aux_pose_head(rel_feat)
    revisit = merge.revisit_head(rel_feat).view(-1, merge.n_out, merge.dim)
    reliability = encoded["reliability"]
    effective_gate = core._effective_revisit_gate(enc["revisit_gate"], reliability)
    return {
        "current_state": current_state,
        "revisit": revisit,
        "novel": novel,
        "aux_pose": aux_pose,
        "R_rel": R_rel,
        "raw_pose_direction": encoded["raw_direction"],
        "pose_reliability": reliability,
        "pose_reliability_features": encoded["reliability_features"],
        "pose_range_steps": encoded["range_steps"],
        "aux_range_code": rel_feat[..., 2],
        "ret_logits": enc["ret_logits"],
        "revisit_gate": enc["revisit_gate"],
        "effective_revisit_gate": effective_gate,
        "gate_logit": enc["gate_logit"],
        "gate_feature": enc["gate_feature"],
        "match_idx": enc["match_idx"],
        "anchor_idx": enc["anchor_idx"],
        "anchor_teacher_forced": enc["anchor_teacher_forced"],
    }


def build_paired_conditions(model, batch, gt_prefix_step_m, nominal_step_m):
    """Encode once, then replace only the revisit pose-code range coordinate."""
    core = model.core
    dev = core.device
    enc = core.encode_memory(batch)
    current_state = core.build_current_state(enc["current"], enc["depth_feat"])
    novel = core.novel(
        batch["batch_window_images"][:, -1].to(dev),
        batch["batch_goal_image"].to(dev),
    )
    merge = core.revisit_merge
    t_rel, R_rel = merge._relative_pose(enc["cur_pose"], enc["goal_pose"])
    encoded = merge.pose_encoder(t_rel, R_rel, enc["pose_context"])
    current_code = encoded["pose_code"]

    revisit = batch["batch_is_revisit"].to(dev).bool()
    gt_distance_m = torch.linalg.vector_norm(
        batch["batch_goal_rel_pose"].to(dev)[..., :2], dim=-1
    )
    gt_prefix_step_m = gt_prefix_step_m.to(device=dev, dtype=current_code.dtype)
    nominal = current_code.new_tensor(float(nominal_step_m))
    distance_unit_steps = float(merge.pose_encoder.distance_unit_steps)

    oracle_stream_steps = gt_distance_m / gt_prefix_step_m.clamp_min(1e-6)
    current_metric_m = encoded["range_steps"] * gt_prefix_step_m
    odom_metric_steps = current_metric_m / nominal
    oracle_metric_steps = gt_distance_m / nominal

    replacement_steps = {
        "zero_range": torch.zeros_like(encoded["range_steps"]),
        "oracle_stream_range": oracle_stream_steps,
        "odom_metric_range": odom_metric_steps,
        "oracle_metric_range": oracle_metric_steps,
    }
    pose_codes = {"current": current_code}
    for name, range_steps in replacement_steps.items():
        code = current_code.clone()
        replacement_code = torch.asinh(range_steps / distance_unit_steps).clamp(max=5.0)
        code[:, 2] = torch.where(revisit, replacement_code, current_code[:, 2])
        pose_codes[name] = code

    conditions = {
        name: _pose_condition(
            core, enc, current_state, novel, R_rel, encoded, pose_codes[name]
        )
        for name in VARIANTS
    }
    diagnostics = {
        "gt_distance_m": gt_distance_m,
        "current_range_steps": encoded["range_steps"],
        "oracle_stream_steps": oracle_stream_steps,
        "current_metric_m": current_metric_m,
        "odom_metric_steps": odom_metric_steps,
        "oracle_metric_steps": oracle_metric_steps,
        "range_codes": {name: pose_codes[name][:, 2] for name in VARIANTS},
        # ``range_codes`` are the raw, gauge-invariant LingBot pose codes before
        # RevisitMerge's learned residual adapter.  Record the adapted values as
        # well: these are the coordinates actually consumed by revisit_head and
        # supervised by aux_range_loss.  Keeping both names explicit prevents a
        # trained adapter from being mistaken for an unchanged raw pose stream.
        "adapted_range_codes": {
            name: conditions[name]["aux_range_code"] for name in VARIANTS
        },
    }
    return conditions, diagnostics


def _action_mse(outputs):
    return (outputs["noise_pred"] - outputs["noise"]).square().mean(dim=(1, 2))


def _prediction_sensitivity(outputs, reference):
    return (outputs["noise_pred"] - reference["noise_pred"]).square().mean(dim=(1, 2))


def _cuda_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _sample_full_diffusion(model, conditions, batch_shape, batch_index, args):
    """Return [repeat, batch] paired action MSE and sensitivity per variant."""
    device = next(model.parameters()).device
    target = batch_shape.to(device)
    action_mse = {name: [] for name in VARIANTS}
    sensitivity = {name: [] for name in VARIANTS[1:]}
    for repeat in range(args.diffusion_repeats):
        seed = args.diffusion_seed + 1009 * batch_index + 104729 * repeat
        initial_noise = torch.randn(
            target.shape,
            device=device,
            generator=_cuda_generator(device, seed),
        )
        sampled = {}
        for variant in VARIANTS:
            sampled[variant] = model.sample_actions_from_condition(
                conditions[variant],
                initial_noise=initial_noise,
                generator=_cuda_generator(device, seed + 1),
            )
            action_mse[variant].append(
                (sampled[variant] - target).square().mean(dim=(1, 2))
            )
        for variant in VARIANTS[1:]:
            sensitivity[variant].append(
                (sampled[variant] - sampled["current"]).square().mean(dim=(1, 2))
            )
    return (
        {name: torch.stack(values) for name, values in action_mse.items()},
        {name: torch.stack(values) for name, values in sensitivity.items()},
    )


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _finite_correlation(x, y, rank=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return None
    first, second = x[valid], y[valid]
    if rank:
        first, second = _average_ranks(first), _average_ranks(second)
    return {"statistic": float(np.corrcoef(first, second)[0, 1])}


def _numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _paired_action_summary(records):
    current = np.asarray([row["action_mse_current"] for row in records])
    result = {"num_samples": len(records), "current": _numeric_summary(current)}
    for variant in VARIANTS[1:]:
        values = np.asarray([row[f"action_mse_{variant}"] for row in records])
        delta = values - current
        result[variant] = {
            **_numeric_summary(values),
            "delta_vs_current": _numeric_summary(delta),
            "wins_vs_current": int(np.sum(values < current)),
            "ties_vs_current": int(np.sum(np.isclose(values, current, rtol=0, atol=1e-10))),
            "noise_prediction_sensitivity_mse": _numeric_summary(
                [row[f"noise_prediction_sensitivity_{variant}"] for row in records]
            ),
        }
    return result


def _paired_full_diffusion_summary(records):
    current = np.asarray(
        [row["full_diffusion_action_mse_current"] for row in records]
    )
    result = {"num_samples": len(records), "current": _numeric_summary(current)}
    for variant in VARIANTS[1:]:
        values = np.asarray(
            [row[f"full_diffusion_action_mse_{variant}"] for row in records]
        )
        delta = values - current
        result[variant] = {
            **_numeric_summary(values),
            "delta_vs_current": _numeric_summary(delta),
            "wins_vs_current": int(np.sum(values < current)),
            "ties_vs_current": int(
                np.sum(np.isclose(values, current, rtol=0, atol=1e-10))
            ),
            "sampled_action_sensitivity_mse": _numeric_summary(
                [
                    row[f"full_diffusion_action_sensitivity_{variant}"]
                    for row in records
                ]
            ),
        }
    return result


def _range_summary(records):
    predicted_m = np.asarray([row["current_metric_m"] for row in records])
    target_m = np.asarray([row["goal_distance"] for row in records])
    current_steps = np.asarray([row["current_range_steps"] for row in records])
    target_steps = np.asarray([row["oracle_stream_steps"] for row in records])
    error = predicted_m - target_m
    relative = np.abs(error) / np.maximum(target_m, 1e-6)

    optimistic_alpha = float(np.dot(predicted_m, target_m) / max(np.dot(predicted_m, predicted_m), 1e-12))
    optimistic = optimistic_alpha * predicted_m

    episodes = sorted({row["episode"] for row in records})
    loeo = np.empty_like(predicted_m)
    loeo_alpha = {}
    for episode in episodes:
        held = np.asarray([row["episode"] == episode for row in records])
        train = ~held
        if not np.any(train):
            alpha = 1.0
        else:
            alpha = float(
                np.dot(predicted_m[train], target_m[train])
                / max(np.dot(predicted_m[train], predicted_m[train]), 1e-12)
            )
        loeo[held] = alpha * predicted_m[held]
        loeo_alpha[episode] = alpha

    target_code = np.asarray(
        [row["range_code_oracle_stream_range"] for row in records]
    )
    raw_code = np.asarray([row["range_code_current"] for row in records])
    adapted_code = np.asarray(
        [row["adapted_range_code_current"] for row in records]
    )

    def code_error_summary(prediction):
        code_error = prediction - target_code
        return {
            "mae": float(np.mean(np.abs(code_error))),
            "rmse": float(np.sqrt(np.mean(code_error**2))),
            "bias": float(np.mean(code_error)),
            "prediction": _numeric_summary(prediction),
            "target": _numeric_summary(target_code),
        }

    return {
        "num_samples": len(records),
        "range_code_error": {
            "raw_before_adapter": code_error_summary(raw_code),
            "adapted_consumed_by_policy": code_error_summary(adapted_code),
        },
        "pearson_current_steps_vs_oracle_steps": _finite_correlation(
            current_steps, target_steps
        ),
        "spearman_current_steps_vs_oracle_steps": _finite_correlation(
            current_steps, target_steps, rank=True
        ),
        "pearson_online_metric_m_vs_gt_m": _finite_correlation(
            predicted_m, target_m
        ),
        "spearman_online_metric_m_vs_gt_m": _finite_correlation(
            predicted_m, target_m, rank=True
        ),
        "online_action_calibrated": {
            "mae_m": float(np.mean(np.abs(error))),
            "rmse_m": float(np.sqrt(np.mean(error**2))),
            "median_relative_error": float(np.median(relative)),
            "prediction_m": _numeric_summary(predicted_m),
            "target_m": _numeric_summary(target_m),
        },
        "optimistic_global_scalar_fit": {
            "alpha": optimistic_alpha,
            "mae_m": float(np.mean(np.abs(optimistic - target_m))),
            "rmse_m": float(np.sqrt(np.mean((optimistic - target_m) ** 2))),
        },
        "leave_one_episode_out_scalar": {
            "alpha_by_held_episode": loeo_alpha,
            "mae_m": float(np.mean(np.abs(loeo - target_m))),
            "rmse_m": float(np.sqrt(np.mean((loeo - target_m) ** 2))),
        },
    }


def summarize(records):
    revisit = [row for row in records if row["is_revisit"]]
    result = {
        "all": _paired_action_summary(records),
        "revisit": _paired_action_summary(revisit),
        "revisit_range": _range_summary(revisit),
        "revisit_by_goal_label": {},
    }
    for label in sorted({row["goal_label"] for row in revisit}):
        rows = [row for row in revisit if row["goal_label"] == label]
        result["revisit_by_goal_label"][label] = {
            "action": _paired_action_summary(rows),
            "range": _range_summary(rows),
        }
    if records and "full_diffusion_action_mse_current" in records[0]:
        result["full_diffusion_all"] = _paired_full_diffusion_summary(records)
        result["full_diffusion_revisit"] = _paired_full_diffusion_summary(revisit)
        result["full_diffusion_revisit_by_goal_label"] = {
            label: _paired_full_diffusion_summary(
                [row for row in revisit if row["goal_label"] == label]
            )
            for label in sorted({row["goal_label"] for row in revisit})
        }
    return result


def main():
    args = parse_args()
    if args.nominal_step_m <= 0:
        raise ValueError("--nominal-step-m must be positive")
    if args.diffusion_repeats <= 0:
        raise ValueError("--diffusion-repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("MemNav range diagnostic requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = copy.deepcopy(memnav_exp_cfg)
    if args.lingbot_repo:
        config.il.lingbot_repo = args.lingbot_repo
    if args.lingbot_weights:
        config.il.lingbot_weights = args.lingbot_weights
    dataset = MemNav_Dataset(
        args.root_dir,
        predict_size=config.il.predict_size,
        image_size=config.il.image_size,
        lingbot_repo=config.il.lingbot_repo,
        feature_root=args.feature_root,
        window_size=config.il.window_size,
        num_scale=config.il.num_scale,
        strict_feature_coverage=True,
        require_generated_pose_convention=getattr(
            config.il, "require_generated_pose_convention", False
        ),
        data_split=args.data_split,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        sampling_mode="fixed_leg",
        sampling_seed=args.sampling_seed,
    )
    dataset_size = len(dataset)
    selection_indices = list(range(dataset_size))
    if 0 < args.max_samples < dataset_size:
        if args.subset_mode == "balanced-fixed":
            eval_dataset = build_fixed_memnav_eval_subset(
                dataset, args.max_samples, selection_seed=args.seed
            )
            selection_indices = eval_dataset.memnav_selection_indices
        else:
            rng = np.random.default_rng(args.seed)
            selection_indices = sorted(
                rng.choice(dataset_size, args.max_samples, replace=False).tolist()
            )
            eval_dataset = Subset(dataset, selection_indices)
    else:
        eval_dataset = dataset
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=memnav_collate_fn,
    )

    model, checkpoint = load_checkpoint(config, args.checkpoint)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, original_batch in enumerate(loader):
            batch = dict(original_batch)
            if args.oracle_positive_anchor:
                batch["diagnostic_oracle_positive"] = True
            step_rows = _step_rows(
                batch, args.feature_root, args.root_dir, args.recent_step_window
            )
            gt_prefix = torch.tensor([row[1] for row in step_rows], dtype=torch.float32)
            conditions, diagnostic = build_paired_conditions(
                model, batch, gt_prefix, args.nominal_step_m
            )

            outputs = {"current": model.forward_with_condition(batch, conditions["current"])}
            paired_batch = dict(batch)
            paired_batch["diagnostic_noise"] = outputs["current"]["noise"]
            paired_batch["diagnostic_timesteps"] = outputs["current"]["timesteps"]
            for variant in VARIANTS[1:]:
                outputs[variant] = model.forward_with_condition(
                    paired_batch, conditions[variant]
                )

            batch_records = compute_memnav_batch_records(
                outputs["current"], original_batch
            )
            action_mse = {name: _action_mse(value) for name, value in outputs.items()}
            sensitivity = {
                name: _prediction_sensitivity(value, outputs["current"])
                for name, value in outputs.items()
                if name != "current"
            }
            full_action_mse = full_sensitivity = None
            if args.full_diffusion:
                full_action_mse, full_sensitivity = _sample_full_diffusion(
                    model,
                    conditions,
                    original_batch["batch_labels"],
                    batch_index,
                    args,
                )
            pos_mask = original_batch["batch_pos_mask"].to(model.core.device).bool()
            anchor = conditions["current"]["anchor_idx"]
            anchor_positive = pos_mask.gather(1, anchor[:, None]).squeeze(1)

            for index, record in enumerate(batch_records):
                episode, prefix_step, recent_step = step_rows[index]
                record.update(
                    {
                        "sample_index": len(records),
                        "episode": episode,
                        "anchor_index": int(anchor[index].item()),
                        "anchor_positive": bool(anchor_positive[index].item()),
                        "gt_prefix_step_m": prefix_step,
                        "gt_recent_step_m": recent_step,
                        "current_range_steps": float(
                            diagnostic["current_range_steps"][index].item()
                        ),
                        "oracle_stream_steps": float(
                            diagnostic["oracle_stream_steps"][index].item()
                        ),
                        "current_metric_m": float(
                            diagnostic["current_metric_m"][index].item()
                        ),
                        "odom_metric_steps": float(
                            diagnostic["odom_metric_steps"][index].item()
                        ),
                        "oracle_metric_steps": float(
                            diagnostic["oracle_metric_steps"][index].item()
                        ),
                    }
                )
                for variant in VARIANTS:
                    record[f"range_code_{variant}"] = float(
                        diagnostic["range_codes"][variant][index].item()
                    )
                    record[f"adapted_range_code_{variant}"] = float(
                        diagnostic["adapted_range_codes"][variant][index].item()
                    )
                    record[f"action_mse_{variant}"] = float(
                        action_mse[variant][index].item()
                    )
                for variant in VARIANTS[1:]:
                    record[f"noise_prediction_sensitivity_{variant}"] = float(
                        sensitivity[variant][index].item()
                    )
                if full_action_mse is not None:
                    for variant in VARIANTS:
                        record[f"full_diffusion_action_mse_{variant}"] = float(
                            full_action_mse[variant][:, index].mean().item()
                        )
                        record[f"full_diffusion_action_mse_std_{variant}"] = float(
                            full_action_mse[variant][:, index].std(unbiased=False).item()
                        )
                    for variant in VARIANTS[1:]:
                        record[
                            f"full_diffusion_action_sensitivity_{variant}"
                        ] = float(full_sensitivity[variant][:, index].mean().item())
                records.append(record)

            if batch_index % args.log_every == 0 or batch_index + 1 == len(loader):
                print(
                    f"[range] batch={batch_index + 1}/{len(loader)} "
                    f"samples={len(records)} revisit={sum(r['is_revisit'] for r in records)}"
                )

    elapsed = time.time() - started
    result = {
        "evaluation_type": "paired-offline-range-coordinate-ablation",
        "closed_loop_navigation": False,
        "checkpoint": str(Path(checkpoint).resolve()),
        "git_commit": _git_commit(),
        "root_dir": str(Path(args.root_dir).resolve()),
        "feature_root": str(Path(args.feature_root).resolve()),
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "dataset_size": dataset_size,
        "evaluated_samples": len(records),
        "selection_indices": selection_indices,
        "oracle_positive_anchor": args.oracle_positive_anchor,
        "paired_diffusion_randomness": True,
        "range_intervention_scope": "pose_code_coordinate_2_on_revisit_rows_only",
        "reliability_and_gate_held_fixed": True,
        "nominal_step_m": args.nominal_step_m,
        "recent_step_window": args.recent_step_window,
        "full_diffusion": args.full_diffusion,
        "diffusion_seed": args.diffusion_seed if args.full_diffusion else None,
        "diffusion_repeats": args.diffusion_repeats if args.full_diffusion else None,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(records) / elapsed,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "summary": summarize(records),
        "per_sample": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"[range] wrote {output}")


if __name__ == "__main__":
    main()
