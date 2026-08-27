#!/usr/bin/env python3
"""Targeted causal probe: attach a goal query at one explicit history frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

try:
    from MemNavData.diag_m2p_s1_gct_query import (
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _lingbot_relative_direction, _read_only_query, _stream_prefix,
        direction_error_degrees, require, signed_bearing_degrees)
    from MemNavData.diag_m2p_s1_online_query import (
        DEFAULT_BENCHMARK_ROOT, load_online_episode, online_pointgoal)
except ModuleNotFoundError:
    from diag_m2p_s1_gct_query import (  # type: ignore
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _lingbot_relative_direction, _read_only_query, _stream_prefix,
        direction_error_degrees, require, signed_bearing_degrees)
    from diag_m2p_s1_online_query import (  # type: ignore
        DEFAULT_BENCHMARK_ROOT, load_online_episode, online_pointgoal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path,
                        default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--episode", required=True,
                        help="scene/episode_XXXX")
    parser.add_argument("--role", choices=("B", "C"), required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--anchor-source", required=True)
    parser.add_argument("--lingbot-repo", type=Path,
                        default=DEFAULT_LINGBOT_REPO)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
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
    require(args.weights.is_file(), "LingBot weights are missing")
    require(torch.cuda.is_available(), "CUDA is required")
    return args


def main() -> None:
    args = parse_args()
    benchmark_path = args.benchmark_root / args.episode / "benchmark.json"
    require(benchmark_path.is_file(), f"missing benchmark: {benchmark_path}")
    loaded = load_online_episode(benchmark_path)
    goal = next(item for item in loaded["goals"]
                if item["role"] == args.role)
    require(0 <= args.anchor < len(loaded["rgb_paths"]),
            "explicit anchor is outside online history")
    model = _build_model(args)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(
        [os.fspath(path) for path in loaded["rgb_paths"]], mode="pad",
        image_size=args.image_size, patch_size=args.patch_size)
    goal_image = load_and_preprocess_images(
        [os.fspath(goal["path"])], mode="pad",
        image_size=args.image_size, patch_size=args.patch_size)[0]
    current_pose = _stream_prefix(
        model, images, num_scale=args.num_scale, device=args.device,
        progress_label="full-current")
    _stream_prefix(
        model, images[:args.anchor + 1], num_scale=args.num_scale,
        device=args.device, progress_label="explicit-anchor")
    goal_pose, identity, audit = _read_only_query(
        model, goal_image, num_scale=args.num_scale, device=args.device,
        label=f"{args.anchor_source}_{args.role}")
    predicted = _lingbot_relative_direction(current_pose, goal_pose)
    target = online_pointgoal(loaded["poses"][-1], goal["floor_position"])
    curve = json.loads(benchmark_path.read_text(encoding="utf-8"))[
        "variants"]["v1_controlled_pose_perturbation"]["goals"][
            args.role]["covis_curve"]
    report = {
        "schema": "m2p_s1_explicit_anchor_probe_v1",
        "created_unix_s": time.time(),
        "episode": args.episode,
        "role": args.role,
        "anchor": args.anchor,
        "anchor_source": args.anchor_source,
        "anchor_covis": float(curve[args.anchor]),
        "teacher_anchor": int(goal["anchor"]),
        "anchor_minus_teacher": int(args.anchor - int(goal["anchor"])),
        "recall_gap_from_explicit_anchor": int(
            len(images) - 1 - args.anchor),
        "gt_pointgoal": target.tolist(),
        "gt_bearing_deg": signed_bearing_degrees(target),
        "predicted_direction": predicted.tolist(),
        "predicted_bearing_deg": signed_bearing_degrees(predicted),
        "direction_error_deg": direction_error_degrees(predicted, target),
        "predicted_raw_norm": float(np.linalg.norm(predicted)),
        "query_state_identity": identity,
        "query_audit": audit,
        "scope": "targeted_causal_attribution_not_a_method_or_sr_result",
    }
    strict = _finite_json(report)
    _atomic_json(args.out, strict)
    print(json.dumps(strict, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
