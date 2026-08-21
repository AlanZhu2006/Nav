#!/usr/bin/env python3
"""Cross-scene no-match smoke for global/local frozen-GCT consistency.

Each audited online-A history is paired with B/C goal images from a different
MP3D scene, guaranteeing that the queried place is absent.  We compare the
candidate-free global GCT bearing with the DINO-pointer local GCT bearing and
record DINO confidence plus raw pose magnitudes.  This is a small mechanism
smoke, not a calibrated Novel/Revisit selector.
"""

from __future__ import annotations

import argparse
import json
import math
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
        require, signed_bearing_degrees)
    from MemNavData.diag_m2p_s1_online_dino_pointer import dino_cls
    from MemNavData.diag_m2p_s1_online_query import (
        DEFAULT_BENCHMARK_ROOT, discover, load_online_episode)
except ModuleNotFoundError:
    from diag_m2p_s1_gct_query import (  # type: ignore
        DEFAULT_LINGBOT_REPO, _atomic_json, _build_model, _finite_json,
        _lingbot_relative_direction, _read_only_query, _stream_prefix,
        require, signed_bearing_degrees)
    from diag_m2p_s1_online_dino_pointer import dino_cls  # type: ignore
    from diag_m2p_s1_online_query import (  # type: ignore
        DEFAULT_BENCHMARK_ROOT, discover, load_online_episode)


DEFAULT_OUT = Path(
    ".diagnostics/m2p_s1_online_negative_consistency_all4_20260813")


def circular_difference_degrees(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


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
    require(args.episodes >= 2,
            "cross-scene cyclic negatives require at least two episodes")
    require(args.candidate_min_gap >= 1 and args.dino_batch_size >= 1,
            "invalid pointer configuration")
    require(args.weights.is_file(), "LingBot weights are missing")
    require(torch.cuda.is_available(), "CUDA is required")
    return args


def main() -> None:
    args = parse_args()
    paths = discover(args)
    prepared = [load_online_episode(path) for path in paths]
    require(len({item["benchmark"]["scene"] for item in prepared})
            == len(prepared), "negative smoke requires distinct source scenes")
    args.out.mkdir(parents=True, exist_ok=True)
    model = _build_model(args)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    rows = []
    for episode_index, loaded in enumerate(prepared, start=1):
        negative = prepared[episode_index % len(prepared)]
        source_label = (f"{loaded['benchmark']['scene']}/"
                        f"{loaded['benchmark']['episode']}")
        negative_label = (f"{negative['benchmark']['scene']}/"
                          f"{negative['benchmark']['episode']}")
        require(loaded["benchmark"]["scene"]
                != negative["benchmark"]["scene"],
                "negative goal came from the same scene")
        print(f"[episode {episode_index}/{len(prepared)}] {source_label} "
              f"<- negative goals {negative_label}", flush=True)
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
        for goal in negative["goals"]:
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
                "margin": float(
                    np.sort(scores)[-1] - np.sort(scores)[-2])
                if len(scores) > 1 else None,
            }

        current_pose = _stream_prefix(
            model, images, num_scale=args.num_scale, device=args.device,
            progress_label=f"{source_label}:full-current")
        global_poses = {}
        audits = {}
        all_identity = True
        for role, image in goal_images.items():
            pose, identity, audit = _read_only_query(
                model, image, num_scale=args.num_scale,
                device=args.device, label=f"negative_global_{role}")
            global_poses[role] = pose
            audits[f"global_{role}"] = audit
            all_identity = all_identity and identity

        for goal in negative["goals"]:
            role = goal["role"]
            anchor = selections[role]["anchor"]
            anchor_pose = _stream_prefix(
                model, images[:anchor + 1], num_scale=args.num_scale,
                device=args.device,
                progress_label=f"{source_label}:negative-local-{role}")
            local_pose, identity, audit = _read_only_query(
                model, goal_images[role], num_scale=args.num_scale,
                device=args.device, label=f"negative_local_{role}")
            audits[f"local_{role}"] = audit
            all_identity = all_identity and identity
            global_direction = _lingbot_relative_direction(
                current_pose, global_poses[role])
            local_direction = _lingbot_relative_direction(
                current_pose, local_pose)
            anchor_residual = _lingbot_relative_direction(
                anchor_pose, local_pose)
            global_bearing = signed_bearing_degrees(global_direction)
            local_bearing = signed_bearing_degrees(local_direction)
            row = {
                "history_episode": source_label,
                "history_scene": loaded["benchmark"]["scene"],
                "negative_goal_episode": negative_label,
                "negative_goal_scene": negative["benchmark"]["scene"],
                "role": role,
                "strict_cross_scene_no_match": True,
                "online_frames": len(images),
                "dino_anchor": anchor,
                "dino_cosine": selections[role]["cosine"],
                "dino_top1_top2_margin": selections[role]["margin"],
                "global_direction": global_direction.tolist(),
                "global_raw_norm": float(np.linalg.norm(global_direction)),
                "global_bearing_deg": global_bearing,
                "local_direction": local_direction.tolist(),
                "local_raw_norm": float(np.linalg.norm(local_direction)),
                "local_bearing_deg": local_bearing,
                "global_local_bearing_disagreement_deg": (
                    circular_difference_degrees(
                        global_bearing, local_bearing)),
                "global_local_goal_translation_l2": float(np.linalg.norm(
                    global_poses[role][:3] - local_pose[:3])),
                "local_goal_from_selected_anchor_raw_norm": float(
                    np.linalg.norm(anchor_residual)),
                "all_query_state_identity": all_identity,
                "query_audits": audits,
            }
            rows.append(row)
            print(f"[result] {source_label} <- {negative_label}/{role} "
                  f"DINO={row['dino_cosine']:.3f} "
                  f"disagree={row['global_local_bearing_disagreement_deg']:.1f}deg "
                  f"global_norm={row['global_raw_norm']:.3f} "
                  f"local_norm={row['local_raw_norm']:.3f}", flush=True)
        _atomic_json(args.out / "partial_rows.json", _finite_json(rows))
        del images, history_cls, goal_images
        model.clean_kv_cache()
        torch.cuda.empty_cache()

    disagreement = [
        float(row["global_local_bearing_disagreement_deg"]) for row in rows]
    cosines = [float(row["dino_cosine"]) for row in rows]
    summary = {
        "histories": len(prepared),
        "strict_cross_scene_no_match_queries": len(rows),
        "bearing_disagreement_median_deg": float(np.median(disagreement)),
        "bearing_disagreement_min_deg": float(np.min(disagreement)),
        "bearing_disagreement_max_deg": float(np.max(disagreement)),
        "dino_cosine_median": float(np.median(cosines)),
        "dino_cosine_min": float(np.min(cosines)),
        "dino_cosine_max": float(np.max(cosines)),
        "all_query_state_identity": all(
            row["all_query_state_identity"] for row in rows),
        "scope": "cross_scene_no_match_smoke_not_calibrated_selector",
    }
    report = {
        "schema": "m2p_s1_online_negative_consistency_v1",
        "created_unix_s": time.time(),
        "configuration": {
            "benchmark_root": os.fspath(args.benchmark_root.resolve()),
            "pairing": "cyclic_next_distinct_scene_same_role",
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
        {key: value for key, value in row.items() if key != "query_audits"}
        for row in rows
    ]).to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(strict["summary"], indent=2, sort_keys=True), flush=True)
    print(f"[written] {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
