#!/usr/bin/env python3
"""Zero-training DINO temporal pointer -> local frozen GCT bearing probe.

Both stages belong to the same LingBot checkpoint: its context-free DINO input
encoder selects one causal history region, then its streaming GCT/camera head
relocalizes the goal at that region.  This script measures bearing only; it does
not yet solve Novel/no-match rejection and is not a closed-loop method result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

try:
    from MemNavData.diag_m2p_s1_gct_query import (
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _lingbot_relative_direction, _read_only_query, _stream_prefix,
        direction_error_degrees, require, signed_bearing_degrees)
    from MemNavData.diag_m2p_s1_online_dino_pointer import dino_cls
    from MemNavData.diag_m2p_s1_online_query import (
        DEFAULT_BENCHMARK_ROOT, discover, load_online_episode,
        online_pointgoal)
except ModuleNotFoundError:
    from diag_m2p_s1_gct_query import (  # type: ignore
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _lingbot_relative_direction, _read_only_query, _stream_prefix,
        direction_error_degrees, require, signed_bearing_degrees)
    from diag_m2p_s1_online_dino_pointer import dino_cls  # type: ignore
    from diag_m2p_s1_online_query import (  # type: ignore
        DEFAULT_BENCHMARK_ROOT, discover, load_online_episode,
        online_pointgoal)


DEFAULT_OUT = Path(
    ".diagnostics/m2p_s1_online_dino_gct_all4_20260813")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path,
                        default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--candidate-min-gap", type=int, default=16)
    parser.add_argument("--dino-batch-size", type=int, default=8)
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
    require(args.episodes >= 1 and args.dino_batch_size >= 1,
            "episode and DINO batch sizes must be positive")
    require(args.candidate_min_gap >= 1,
            "candidate minimum gap must be positive")
    require(args.weights.is_file(), "LingBot weights are missing")
    require(torch.cuda.is_available(), "CUDA is required")
    return args


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
        images = load_and_preprocess_images(
            [os.fspath(path) for path in loaded["rgb_paths"]], mode="pad",
            image_size=args.image_size, patch_size=args.patch_size)
        history_cls = dino_cls(
            model, images, device=args.device,
            batch_size=args.dino_batch_size)
        upper = len(images) - args.candidate_min_gap
        require(upper > 0, "history shorter than candidate minimum gap")
        goal_images = {}
        selections = {}
        for goal in loaded["goals"]:
            role = goal["role"]
            image = load_and_preprocess_images(
                [os.fspath(goal["path"])], mode="pad",
                image_size=args.image_size, patch_size=args.patch_size)[0]
            goal_images[role] = image
            cls = dino_cls(
                model, image[None], device=args.device,
                batch_size=args.dino_batch_size)[0]
            scores = (history_cls[:upper] @ cls).numpy()
            selections[role] = {
                "anchor": int(np.argmax(scores)),
                "cosine": float(np.max(scores)),
            }

        current_pose = _stream_prefix(
            model, images, num_scale=args.num_scale, device=args.device,
            progress_label=f"{label}:full-current")
        local_poses = {}
        audits = {}
        all_identity = True
        for goal in loaded["goals"]:
            role = goal["role"]
            anchor = selections[role]["anchor"]
            _stream_prefix(
                model, images[:anchor + 1], num_scale=args.num_scale,
                device=args.device,
                progress_label=f"{label}:dino-{role}")
            pose, identity, audit = _read_only_query(
                model, goal_images[role], num_scale=args.num_scale,
                device=args.device, label=f"dino_locality_{role}")
            local_poses[role] = pose
            audits[role] = audit
            all_identity = all_identity and identity

        current_gt = loaded["poses"][-1]
        benchmark_payload = json.loads(
            (args.benchmark_root / benchmark["scene"]
             / benchmark["episode"] / "benchmark.json").read_text())
        for goal in loaded["goals"]:
            role = goal["role"]
            anchor = selections[role]["anchor"]
            curve = benchmark_payload["variants"][
                "v1_controlled_pose_perturbation"]["goals"][role][
                    "covis_curve"]
            target = online_pointgoal(current_gt, goal["floor_position"])
            predicted = _lingbot_relative_direction(
                current_pose, local_poses[role])
            row = {
                "episode": label,
                "scene": benchmark["scene"],
                "role": role,
                "history_source": "audited_online_native_navdp_goal_a",
                "online_frames": len(images),
                "candidate_min_gap": args.candidate_min_gap,
                "dino_anchor": anchor,
                "dino_cosine": selections[role]["cosine"],
                "dino_anchor_covis": float(curve[anchor]),
                "teacher_anchor": int(goal["anchor"]),
                "dino_anchor_minus_teacher": int(
                    anchor - int(goal["anchor"])),
                "dino_recall_gap": int(len(images) - 1 - anchor),
                "gt_pointgoal": target.tolist(),
                "gt_bearing_deg": signed_bearing_degrees(target),
                "predicted_direction": predicted.tolist(),
                "predicted_raw_norm": float(np.linalg.norm(predicted)),
                "predicted_bearing_deg": signed_bearing_degrees(predicted),
                "direction_error_deg": direction_error_degrees(
                    predicted, target),
                "query_state_identity": all_identity,
                "query_audit": audits[role],
            }
            rows.append(row)
            print(f"[result] {label}/{role} anchor={anchor} "
                  f"covis={row['dino_anchor_covis']:.3f} "
                  f"error={row['direction_error_deg']:.1f}deg", flush=True)
        _atomic_json(args.out / "partial_rows.json", _finite_json(rows))
        del images, history_cls, goal_images
        model.clean_kv_cache()
        torch.cuda.empty_cache()

    errors = [float(row["direction_error_deg"]) for row in rows]
    summary = {
        "episodes": len(prepared),
        "queries": len(rows),
        "median_direction_error_deg": float(np.nanmedian(errors)),
        **{
            f"cdf_le_{int(threshold)}": {
                "hits": sum(error <= threshold for error in errors),
                "total": len(errors),
                "rate": (sum(error <= threshold for error in errors)
                         / len(errors)),
            }
            for threshold in (15.0, 30.0, 45.0)
        },
        "all_query_state_identity": all(
            row["query_state_identity"] for row in rows),
        "scope": "zero_training_bearing_probe_not_no_match_or_closed_loop",
    }
    report = {
        "schema": "m2p_s1_online_dino_gct_v1",
        "created_unix_s": time.time(),
        "configuration": {
            "benchmark_root": os.fspath(args.benchmark_root.resolve()),
            "candidate_min_gap": args.candidate_min_gap,
            "weights": os.fspath(args.weights.resolve()),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "summary": summary,
        "rows": rows,
    }
    strict = _finite_json(report)
    _atomic_json(args.out / "report.json", strict)
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "query_audit"}
        for row in rows
    ]).to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(strict["summary"], indent=2, sort_keys=True), flush=True)
    print(f"[written] {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
