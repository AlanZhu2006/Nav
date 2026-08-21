#!/usr/bin/env python3
"""Cheap temporal-pointer baseline on the audited online V1 Revisit pilot.

This does not claim a method result.  It measures whether context-free DINO CLS
retrieval already selects a history region with visual support for the goal.
The result is the minimum baseline that a learned GCT-special-token temporal
pointer must exceed; it also tells us whether the two candidate-free GCT
failures are fundamentally unfindable or are specifically global-readout
failures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from MemNavData.diag_m2p_s1_gct_query import (
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _git_revision, require)
    from MemNavData.diag_m2p_s1_online_query import (
        DEFAULT_BENCHMARK_ROOT, V1, discover, load_online_episode)
except ModuleNotFoundError:
    from diag_m2p_s1_gct_query import (  # type: ignore
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _git_revision, require)
    from diag_m2p_s1_online_query import (  # type: ignore
        DEFAULT_BENCHMARK_ROOT, V1, discover, load_online_episode)


DEFAULT_OUT = Path(
    ".diagnostics/m2p_s1_online_dino_pointer_all4_20260813")


@torch.inference_mode()
def dino_cls(model: torch.nn.Module, images: torch.Tensor,
             *, device: str, batch_size: int) -> torch.Tensor:
    """Exact context-free DINO CLS used by current MemNav retrieval."""
    outputs = []
    mean = model.aggregator._resnet_mean.reshape(1, 3, 1, 1)
    std = model.aggregator._resnet_std.reshape(1, 3, 1, 1)
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        mean_device = mean.to(device=device, dtype=batch.dtype)
        std_device = std.to(device=device, dtype=batch.dtype)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = model.aggregator.patch_embed.forward_features(
                (batch - mean_device) / std_device)
        outputs.append(encoded["x_norm_clstoken"].float().cpu())
        del batch, encoded
    return F.normalize(torch.cat(outputs), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path,
                        default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--candidate-min-gap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lingbot-repo", type=Path,
                        default=DEFAULT_LINGBOT_REPO)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-iterations", type=int, default=4)
    args = parser.parse_args()
    if args.weights is None:
        args.weights = args.lingbot_repo / "weights/lingbot-map-long.pt"
    require(args.episodes >= 1 and args.batch_size >= 1,
            "episode and batch sizes must be positive")
    require(args.candidate_min_gap >= 1,
            "candidate minimum gap must be positive")
    require(args.benchmark_root.is_dir(), "benchmark root is missing")
    require(args.weights.is_file(), "LingBot weights are missing")
    require(torch.cuda.is_available(), "this diagnostic requires CUDA")
    return args


def selected_row(scores: np.ndarray, curve: np.ndarray,
                 *, lower: int, upper: int) -> dict[str, Any]:
    require(0 <= lower < upper <= len(scores),
            "invalid DINO candidate interval")
    interval = scores[lower:upper]
    order = np.argsort(-interval, kind="stable") + lower
    top1 = int(order[0])
    top5 = order[:min(5, len(order))]
    return {
        "candidate_lower_inclusive": lower,
        "candidate_upper_exclusive": upper,
        "top1_frame": top1,
        "top1_cosine": float(scores[top1]),
        "top1_covis": float(curve[top1]),
        "top5_max_covis": float(np.max(curve[top5])),
        "top5_frames": [int(index) for index in top5],
        "top5_cosines": [float(scores[index]) for index in top5],
        "top5_covis": [float(curve[index]) for index in top5],
    }


def main() -> None:
    args = parse_args()
    paths = discover(args)
    prepared = [load_online_episode(path) for path in paths]
    args.out.mkdir(parents=True, exist_ok=True)
    model = _build_model(args)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    rows = []
    for episode_index, loaded in enumerate(prepared, start=1):
        benchmark = loaded["benchmark"]
        label = f"{benchmark['scene']}/{benchmark['episode']}"
        print(f"[episode {episode_index}/{len(prepared)}] {label}", flush=True)
        history = load_and_preprocess_images(
            [os.fspath(path) for path in loaded["rgb_paths"]], mode="pad",
            image_size=args.image_size, patch_size=args.patch_size)
        history_cls = dino_cls(
            model, history, device=args.device, batch_size=args.batch_size)
        upper = len(history) - args.candidate_min_gap
        require(upper > 0, "history is shorter than candidate minimum gap")
        for goal in loaded["goals"]:
            goal_image = load_and_preprocess_images(
                [os.fspath(goal["path"])], mode="pad",
                image_size=args.image_size, patch_size=args.patch_size)
            goal_cls = dino_cls(
                model, goal_image, device=args.device,
                batch_size=args.batch_size)[0]
            scores = (history_cls @ goal_cls).numpy()
            curve = np.asarray(json.loads(
                (args.benchmark_root / benchmark["scene"]
                 / benchmark["episode"] / "benchmark.json").read_text()
            )["variants"][V1]["goals"][goal["role"]]["covis_curve"],
                dtype=np.float64)
            unrestricted = selected_row(
                scores, curve, lower=0, upper=upper)
            eligible = selected_row(
                scores, curve,
                lower=int(goal["eligible_online_a_frame_floor"]),
                upper=upper)
            row = {
                "episode": label,
                "scene": benchmark["scene"],
                "role": goal["role"],
                "online_frames": len(history),
                "candidate_min_gap": args.candidate_min_gap,
                "teacher_anchor": goal["anchor"],
                "teacher_recall_gap": goal["recall_gap"],
                "teacher_max_eligible_covis": goal["max_covis"],
                **{f"unrestricted_{key}": value
                   for key, value in unrestricted.items()},
                **{f"eligible_{key}": value
                   for key, value in eligible.items()},
            }
            rows.append(row)
            print(f"[result] {label}/{goal['role']} gap={goal['recall_gap']} "
                  f"DINO top1={unrestricted['top1_frame']} "
                  f"covis={unrestricted['top1_covis']:.3f}; "
                  f"top5max={unrestricted['top5_max_covis']:.3f}",
                  flush=True)
        del history, history_cls
        torch.cuda.empty_cache()

    def rate(field: str, threshold: float) -> dict[str, Any]:
        hits = sum(float(row[field]) >= threshold for row in rows)
        return {"hits": hits, "total": len(rows),
                "rate": hits / len(rows) if rows else None}

    summary = {
        "queries": len(rows),
        "unrestricted_top1_covis_ge_0p20": rate(
            "unrestricted_top1_covis", 0.20),
        "unrestricted_top1_covis_ge_0p50": rate(
            "unrestricted_top1_covis", 0.50),
        "unrestricted_top5_max_covis_ge_0p50": rate(
            "unrestricted_top5_max_covis", 0.50),
        "eligible_top1_covis_ge_0p50": rate(
            "eligible_top1_covis", 0.50),
        "scope": "zero_training_temporal_pointer_baseline_not_bearing_or_sr",
    }
    report = {
        "schema": "m2p_s1_online_dino_pointer_v1",
        "created_unix_s": time.time(),
        "configuration": {
            "benchmark_root": os.fspath(args.benchmark_root.resolve()),
            "variant": V1,
            "candidate_min_gap": args.candidate_min_gap,
            "lingbot_revision": _git_revision(args.lingbot_repo),
            "weights": os.fspath(args.weights.resolve()),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "summary": summary,
        "rows": rows,
    }
    strict = _finite_json(report)
    _atomic_json(args.out / "report.json", strict)
    pd.DataFrame(rows).to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(strict["summary"], indent=2, sort_keys=True), flush=True)
    print(f"[written] {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
