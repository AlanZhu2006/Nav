#!/usr/bin/env python3
"""M2P S-1 deployment-form smoke on audited online NavDP Goal-A histories.

Unlike ``diag_m2p_s1_gct_query.py`` (the generated/expert-trajectory mechanism
probe), this script consumes the materialized, hash-verified online-A traces and
the V1 controlled-pose Revisit goals from the shared-online double-Revisit
benchmark.  It evaluates both B and C directly from the same causal A memory.

No anchor is supplied to the candidate-free arm.  The metadata co-visibility
argmax is used only by the oracle-locality attribution arm.  All GCT weights are
frozen and every goal query must pass the exact read-only cache audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from MemNavData.diag_m2p_s1_gct_query import (
        DEFAULT_LINGBOT_REPO,
        _atomic_json,
        _build_model,
        _finite_json,
        _git_revision,
        _lingbot_relative_direction,
        _read_only_query,
        _stream_prefix,
        direction_error_degrees,
        require,
        signed_bearing_degrees,
    )
except ModuleNotFoundError:  # direct invocation from MemNavData/
    from diag_m2p_s1_gct_query import (  # type: ignore
        DEFAULT_LINGBOT_REPO,
        _atomic_json,
        _build_model,
        _finite_json,
        _git_revision,
        _lingbot_relative_direction,
        _read_only_query,
        _stream_prefix,
        direction_error_degrees,
        require,
        signed_bearing_degrees,
    )


DEFAULT_BENCHMARK_ROOT = Path(
    ".diagnostics/shared_online_double_revisit_v0v1_pilot_20260812")
DEFAULT_OUT = Path(
    ".diagnostics/m2p_s1_online_query_v1_smoke_20260813")
V1 = "v1_controlled_pose_perturbation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def online_pointgoal(current: dict[str, Any],
                     target_floor: list[float]) -> np.ndarray:
    """Habitat world target -> NavDP [forward, left] at an online pose."""
    target = np.asarray(target_floor, dtype=np.float64)
    require(target.shape == (3,), "target floor position must have shape (3,)")
    dx = float(target[0] - float(current["x"]))
    dz = float(target[2] - float(current["z"]))
    yaw = float(current["yaw"])
    sine, cosine = math.sin(yaw), math.cos(yaw)
    return np.array([
        -dx * sine - dz * cosine,
        -dx * cosine + dz * sine,
    ], dtype=np.float64)


def discover(args: argparse.Namespace) -> list[Path]:
    rows = []
    for path in args.benchmark_root.glob("*/episode_*/benchmark.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(V1 in payload.get("variants", {}),
                f"{path} lacks {V1}")
        relative = f"{payload['scene']}/{payload['episode']}"
        rows.append((int(payload["online_a_steps"]), relative, path))
    if args.episode:
        wanted = set(args.episode)
        selected = [row for row in rows if row[1] in wanted]
        found = {row[1] for row in selected}
        require(found == wanted, f"unknown episodes: {sorted(wanted - found)}")
        selected.sort(key=lambda row: args.episode.index(row[1]))
    else:
        selected = sorted(rows)[:args.episodes]
    require(bool(selected), "no online benchmark episodes found")
    return [row[2] for row in selected]


def load_online_episode(benchmark_path: Path) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    source = Path(benchmark["source_online_episode"])
    trace_path = source / "online_a_trace.json"
    receipt_path = source / "receipt.json"
    for path in (source / "rgb", trace_path, receipt_path):
        require(path.exists(), f"missing online source asset: {path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(bool(trace.get("reached")), "online A trace did not reach Goal A")
    require(bool(receipt["online_a_control_audit"]["ok"]),
            "online A was not controlled exclusively by frozen native NavDP")
    require(sha256_file(trace_path) == benchmark["source_online_trace_sha256"],
            "online trace hash differs from benchmark binding")
    require(sha256_file(receipt_path)
            == benchmark["source_online_receipt_sha256"],
            "online receipt hash differs from benchmark binding")

    rgb_paths = sorted((source / "rgb").glob("*.jpg"),
                       key=lambda path: int(path.stem))
    poses = trace["poses"]
    require(len(rgb_paths) == len(poses) == int(benchmark["online_a_steps"]),
            "online RGB/pose/benchmark length mismatch")
    for path, pose in zip(rgb_paths, poses):
        require(int(path.stem) == int(pose["step"]),
                "online RGB step order mismatch")
        require(sha256_file(path) == pose["jpg_sha256"],
                f"online RGB hash mismatch: {path}")

    variant = benchmark["variants"][V1]
    goals = []
    for role in ("B", "C"):
        goal = variant["goals"][role]
        asset = variant["assets"][role]
        goal_path = benchmark_path.parent / V1 / asset["rgb"]
        require(goal_path.is_file(), f"missing {role} image: {goal_path}")
        require(sha256_file(goal_path) == asset["rgb_sha256"],
                f"{role} image hash mismatch")
        curve = [float(value) for value in goal["covis_curve"]]
        require(len(curve) == len(rgb_paths),
                f"{role} covisibility curve/history length mismatch")
        # The benchmark deliberately freezes targets only from the eligible
        # online-memory suffix (normally frame >=39); earlier frames still
        # exist in the deployed GCT memory but were not allowed to define B/C.
        eligible_floor = int(goal["eligible_online_a_frame_floor"])
        eligible_argmax = eligible_floor + int(np.argmax(
            curve[eligible_floor:]))
        require(eligible_argmax
                == int(goal["max_online_a_covis_frame"]),
                f"{role} stored eligible covisibility argmax mismatch")
        require(abs(curve[eligible_argmax]
                    - float(goal["max_online_a_covis"])) < 1e-12,
                f"{role} stored eligible max covisibility mismatch")
        global_argmax = int(np.argmax(curve))
        goals.append({
            "role": role,
            "path": goal_path,
            "floor_position": goal["floor_position"],
            # Use the frozen eligible anchor for the oracle-locality control;
            # also report the unrestricted global maximum as an audit field.
            "anchor": eligible_argmax,
            "recall_gap": len(rgb_paths) - 1 - eligible_argmax,
            "max_covis": curve[eligible_argmax],
            "global_max_covis": curve[global_argmax],
            "global_max_covis_frame": global_argmax,
            "eligible_online_a_frame_floor": eligible_floor,
            "source_online_frame": int(goal["source_online_frame"]),
            "translation_from_source_m": float(
                goal["translation_from_source_m"]),
            "yaw_delta_from_source_deg": float(
                goal["yaw_delta_from_source_deg"]),
            "pixel_mae_from_source": float(goal["pixel_mae_from_source"]),
        })
    return {
        "benchmark": benchmark,
        "rgb_paths": rgb_paths,
        "poses": poses,
        "goals": goals,
        "trace_path": trace_path,
    }


def cdf(rows: list[dict[str, Any]], key: str,
        threshold: float) -> dict[str, Any]:
    values = [float(row[key]) for row in rows
              if math.isfinite(float(row[key]))]
    hits = sum(value <= threshold for value in values)
    return {"hits": hits, "total": len(values),
            "rate": hits / len(values) if values else None}


def arm_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "median_direction_error_deg": float(np.nanmedian(values)),
        "cdf_le_15": cdf(rows, key, 15.0),
        "cdf_le_30": cdf(rows, key, 30.0),
        "cdf_le_45": cdf(rows, key, 45.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path,
                        default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=1)
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
    parser.add_argument(
        "--inspect-pose-attribution", action="store_true",
        help=("retain all frozen GCT history poses and report which temporal "
              "frame each queried goal pose collapses toward"),
    )
    args = parser.parse_args()
    if args.weights is None:
        args.weights = args.lingbot_repo / "weights/lingbot-map-long.pt"
    require(args.episodes >= 1, "--episodes must be positive")
    require(args.benchmark_root.is_dir(), "benchmark root is missing")
    require(args.lingbot_repo.is_dir(), "LingBot repository is missing")
    require(args.weights.is_file(), "LingBot weights are missing")
    require(args.device.startswith("cuda") and torch.cuda.is_available(),
            "this diagnostic requires CUDA")
    return args


def main() -> None:
    args = parse_args()
    benchmarks = discover(args)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[selection]", *(path.parent.relative_to(args.benchmark_root)
                           for path in benchmarks), flush=True)
    # Fail all dataset/hash/metadata contracts before paying the 4.4-GB model
    # load cost or touching CUDA.
    prepared = [load_online_episode(path) for path in benchmarks]
    model = _build_model(args)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    rows: list[dict[str, Any]] = []
    for episode_index, (benchmark_path, loaded) in enumerate(
            zip(benchmarks, prepared), start=1):
        benchmark = loaded["benchmark"]
        label = f"{benchmark['scene']}/{benchmark['episode']}"
        print(f"[episode {episode_index}/{len(benchmarks)}] {label} "
              f"online_frames={len(loaded['rgb_paths'])}", flush=True)
        images = load_and_preprocess_images(
            [os.fspath(path) for path in loaded["rgb_paths"]],
            mode="pad", image_size=args.image_size,
            patch_size=args.patch_size)
        goal_images = {
            goal["role"]: load_and_preprocess_images(
                [os.fspath(goal["path"])], mode="pad",
                image_size=args.image_size,
                patch_size=args.patch_size)[0]
            for goal in loaded["goals"]
        }
        stream_output = _stream_prefix(
            model, images, num_scale=args.num_scale,
            device=args.device, progress_label=f"{label}:online-full",
            return_all=args.inspect_pose_attribution)
        if args.inspect_pose_attribution:
            current_pose, history_poses = stream_output
        else:
            current_pose = stream_output
            history_poses = None
        candidate_poses: dict[str, np.ndarray] = {}
        query_audits: dict[str, dict[str, Any]] = {}
        all_identity = True
        for goal in loaded["goals"]:
            role = goal["role"]
            pose, identity, audit = _read_only_query(
                model, goal_images[role], num_scale=args.num_scale,
                device=args.device, label=f"candidate_free_{role}")
            candidate_poses[role] = pose
            query_audits[f"candidate_free_{role}"] = audit
            all_identity = all_identity and identity
        # One exact repeat proves that querying B then C did not affect B.
        repeated_b, identity, audit = _read_only_query(
            model, goal_images["B"], num_scale=args.num_scale,
            device=args.device, label="candidate_free_B_repeat")
        all_identity = all_identity and identity
        query_audits["candidate_free_B_repeat"] = audit
        repeat_l2 = float(np.linalg.norm(repeated_b - candidate_poses["B"]))

        oracle_poses: dict[str, np.ndarray] = {}
        for goal in loaded["goals"]:
            role, anchor = goal["role"], int(goal["anchor"])
            _stream_prefix(
                model, images[:anchor + 1], num_scale=args.num_scale,
                device=args.device,
                progress_label=f"{label}:oracle-{role}")
            pose, identity, audit = _read_only_query(
                model, goal_images[role], num_scale=args.num_scale,
                device=args.device, label=f"oracle_locality_{role}")
            oracle_poses[role] = pose
            all_identity = all_identity and identity
            query_audits[f"oracle_locality_{role}"] = audit

        current_gt = loaded["poses"][-1]
        for goal in loaded["goals"]:
            role = goal["role"]
            target = online_pointgoal(current_gt, goal["floor_position"])
            candidate = _lingbot_relative_direction(
                current_pose, candidate_poses[role])
            oracle = _lingbot_relative_direction(
                current_pose, oracle_poses[role])
            row = {
                "episode": label,
                "scene": benchmark["scene"],
                "role": role,
                "history_source": "audited_online_native_navdp_goal_a",
                "online_frames": len(images),
                "anchor_idx": goal["anchor"],
                "recall_gap": goal["recall_gap"],
                "max_online_a_covis": goal["max_covis"],
                "global_max_online_a_covis": goal["global_max_covis"],
                "global_max_online_a_covis_frame": goal[
                    "global_max_covis_frame"],
                "eligible_online_a_frame_floor": goal[
                    "eligible_online_a_frame_floor"],
                "source_online_frame": goal["source_online_frame"],
                "translation_from_source_m": goal[
                    "translation_from_source_m"],
                "yaw_delta_from_source_deg": goal[
                    "yaw_delta_from_source_deg"],
                "pixel_mae_from_source": goal["pixel_mae_from_source"],
                "gt_pointgoal": target.tolist(),
                "gt_distance_m": float(np.linalg.norm(target)),
                "gt_bearing_deg": signed_bearing_degrees(target),
                "candidate_free_direction": candidate.tolist(),
                "candidate_free_raw_norm": float(np.linalg.norm(candidate)),
                "candidate_free_bearing_deg": signed_bearing_degrees(candidate),
                "candidate_free_direction_error_deg": direction_error_degrees(
                    candidate, target),
                "oracle_locality_direction": oracle.tolist(),
                "oracle_locality_raw_norm": float(np.linalg.norm(oracle)),
                "oracle_locality_bearing_deg": signed_bearing_degrees(oracle),
                "oracle_locality_direction_error_deg": direction_error_degrees(
                    oracle, target),
                "all_query_state_identity": bool(all_identity),
                "episode_repeat_b_pose_l2": repeat_l2,
                "episode_goal_b_c_pose_l2": float(np.linalg.norm(
                    candidate_poses["B"] - candidate_poses["C"])),
                "query_audits": query_audits,
            }
            if history_poses is not None:
                candidate_distances = np.linalg.norm(
                    history_poses[:, :3] - candidate_poses[role][None, :3],
                    axis=1)
                oracle_distances = np.linalg.norm(
                    history_poses[:, :3] - oracle_poses[role][None, :3],
                    axis=1)
                candidate_nearest = int(np.argmin(candidate_distances))
                oracle_nearest = int(np.argmin(oracle_distances))
                row.update({
                    "candidate_free_nearest_gct_history_frame": (
                        candidate_nearest),
                    "candidate_free_nearest_gct_history_distance": float(
                        candidate_distances[candidate_nearest]),
                    "candidate_free_nearest_frame_minus_anchor": int(
                        candidate_nearest - int(goal["anchor"])),
                    "oracle_locality_nearest_gct_history_frame": (
                        oracle_nearest),
                    "oracle_locality_nearest_gct_history_distance": float(
                        oracle_distances[oracle_nearest]),
                    "oracle_locality_nearest_frame_minus_anchor": int(
                        oracle_nearest - int(goal["anchor"])),
                })
            rows.append(row)
            print(f"[result] {label}/{role} gap={goal['recall_gap']} "
                  f"GT={row['gt_bearing_deg']:+.1f}; "
                  f"free={row['candidate_free_bearing_deg']:+.1f} "
                  f"err={row['candidate_free_direction_error_deg']:.1f}; "
                  f"oracle_err={row['oracle_locality_direction_error_deg']:.1f}; "
                  f"identity={all_identity}", flush=True)
        # Long histories can take minutes.  Preserve completed episode rows so
        # an unrelated later failure never erases already-audited evidence.
        _atomic_json(args.out / "partial_rows.json", _finite_json(rows))
        del images, goal_images
        model.clean_kv_cache()
        torch.cuda.empty_cache()

    summary = {
        "episodes": len(benchmarks),
        "queries": len(rows),
        "candidate_free": arm_summary(
            rows, "candidate_free_direction_error_deg"),
        "oracle_locality": arm_summary(
            rows, "oracle_locality_direction_error_deg"),
        "all_query_state_identity": all(
            row["all_query_state_identity"] for row in rows),
        "max_repeat_b_pose_l2": max(
            row["episode_repeat_b_pose_l2"] for row in rows),
        "scope": "mechanism_smoke_on_four_pilot_online_histories_not_formal_sr",
    }
    report = {
        "schema": "m2p_s1_online_query_v1",
        "created_unix_s": time.time(),
        "configuration": {
            "benchmark_root": os.fspath(args.benchmark_root.resolve()),
            "variant": V1,
            "lingbot_repo": os.fspath(args.lingbot_repo.resolve()),
            "lingbot_revision": _git_revision(args.lingbot_repo),
            "weights": os.fspath(args.weights.resolve()),
            "weights_size_bytes": args.weights.stat().st_size,
            "num_scale": args.num_scale,
            "window": args.window,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "summary": summary,
        "rows": rows,
    }
    strict_report = _finite_json(report)
    _atomic_json(args.out / "report.json", strict_report)
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "query_audits"}
        for row in rows
    ]).to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(strict_report["summary"], indent=2,
                     sort_keys=True), flush=True)
    print(f"[written] {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
