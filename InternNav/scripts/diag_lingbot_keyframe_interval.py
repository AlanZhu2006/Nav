#!/usr/bin/env python3
"""Controlled LingBot keyframe-interval diagnostic for MemNav episodes.

This script is intentionally separate from the production cache generator.  It
answers one narrow question: with the same frames, weights, preprocessing, and
sliding-window size, how much does LingBot's official sparse-keyframe policy
change pose quality compared with appending every frame to the KV cache?

The two policies are:

* ``dense``: ``keyframe_interval = 1``;
* ``auto``: ``keyframe_interval = ceil(num_frames / 320)``.

Every input frame is still inferred and receives a pose.  On non-keyframes only
the KV-cache append is suppressed, matching the current official LingBot demo.

Results are pose-only ``.npz`` files, so this diagnostic never overwrites the
training feature caches.  The default output is under the current personal
worktree and an explicit guard rejects writes under the shared/mother Nav tree.

Examples (run in the ``memnav`` conda environment)::

    python InternNav/scripts/diag_lingbot_keyframe_interval.py preflight
    python InternNav/scripts/diag_lingbot_keyframe_interval.py infer
    python InternNav/scripts/diag_lingbot_keyframe_interval.py score

Use ``score --pose-root <existing-cache-root> --variants existing`` to score
the old ``lingbot_cam_cache.npz`` files without running the GPU model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WORKTREE_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SOURCE_ROOT = Path("/home/asus/Research/Nav/memnav_viz/validate_gated")
DEFAULT_EXISTING_ROOT = Path("/home/asus/Research/datasets/memnav_validate_gated_feat")
DEFAULT_LINGBOT_REPO = Path("/tmp/lingbot-map-current")
DEFAULT_WEIGHTS = Path(
    "/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt"
)
DEFAULT_OUTPUT_ROOT = WORKTREE_ROOT / ".diagnostics/lingbot_keyframe_interval"
MOTHER_NAV_ROOT = Path("/home/asus/Research/Nav")

SCALE_FRAMES = 8
ROPE_TRAINED_VIEWS = 320


def is_relative_to(path: Path, parent: Path) -> bool:
    """Python-3.8-compatible ``Path.is_relative_to``."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def discover_episodes(source_root: Path) -> list[tuple[str, Path, list[Path]]]:
    episodes: list[tuple[str, Path, list[Path]]] = []
    pattern = "*/ */episode_*".replace(" ", "")
    for episode_dir in sorted(source_root.glob(pattern)):
        rgb_dir = episode_dir / "videos/chunk-000/observation.images.rgb"
        parquet = episode_dir / "data/chunk-000/episode_000000.parquet"
        metadata = episode_dir / "meta/gen_meta.json"
        if not (rgb_dir.is_dir() and parquet.is_file() and metadata.is_file()):
            continue
        count = len(list(rgb_dir.glob("*.jpg")))
        paths = [rgb_dir / f"{index}.jpg" for index in range(count)]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"non-contiguous RGB frames in {rgb_dir}; first missing={missing[0]}"
            )
        relative = os.fspath(episode_dir.relative_to(source_root))
        episodes.append((relative, episode_dir, paths))
    if not episodes:
        raise RuntimeError(f"no MemNav episodes found under {source_root}")
    return episodes


def select_episodes(
    episodes: list[tuple[str, Path, list[Path]]], requested: str
) -> list[tuple[str, Path, list[Path]]]:
    if not requested:
        return episodes
    names = [item.strip().strip("/") for item in requested.split(",") if item.strip()]
    by_name = {item[0]: item for item in episodes}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown episode(s): {missing}")
    return [by_name[name] for name in names]


def auto_interval(num_frames: int) -> int:
    return max(1, math.ceil(int(num_frames) / ROPE_TRAINED_VIEWS))


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"dense", "auto", "existing"}
    bad = sorted(set(variants) - allowed)
    if bad:
        raise ValueError(f"unknown variant(s) {bad}; allowed={sorted(allowed)}")
    if len(set(variants)) != len(variants):
        raise ValueError(f"duplicate variants: {variants}")
    return variants


def pose_path(pose_root: Path, variant: str, episode: str) -> Path:
    if variant == "existing":
        return (
            pose_root
            / episode
            / "videos/chunk-000/lingbot_cam_cache.npz"
        )
    return pose_root / variant / episode / "pose.npz"


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def preflight(args: argparse.Namespace, require_gpu: bool) -> list[tuple[str, Path, list[Path]]]:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    lingbot_repo = Path(args.lingbot_repo)
    weights = Path(args.weights)

    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if is_relative_to(output_root, MOTHER_NAV_ROOT):
        raise RuntimeError(
            f"refusing to write diagnostics inside mother Nav tree: {output_root}"
        )
    if not is_relative_to(output_root, WORKTREE_ROOT):
        raise RuntimeError(
            f"output must stay inside personal worktree {WORKTREE_ROOT}: {output_root}"
        )
    if not (lingbot_repo / "lingbot_map/models/gct_stream.py").is_file():
        raise FileNotFoundError(f"invalid LingBot repository: {lingbot_repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"LingBot weights not found: {weights}")
    if weights.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"LingBot checkpoint looks truncated: {weights}")

    episodes = select_episodes(discover_episodes(source_root), args.episodes)
    max_frames = max(len(item[2]) for item in episodes)
    if max_frames > args.max_frame_num:
        raise RuntimeError(
            f"max_frame_num={args.max_frame_num} cannot cover longest episode={max_frames}"
        )

    free_bytes = shutil.disk_usage(output_root.parent).free
    # A pose9 float32 file is tiny; use a deliberately conservative 1 MiB/episode/variant.
    required_bytes = len(episodes) * max(len(parse_variants(args.variants)), 1) * 1024**2
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient output disk: free={free_bytes} required~={required_bytes}"
        )

    if require_gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment")
        capability = torch.cuda.get_device_capability()
        if args.dtype == "bf16" and capability[0] < 8:
            raise RuntimeError(f"bf16 requires Ampere or newer GPU, got capability={capability}")
        sys.path.insert(0, os.fspath(lingbot_repo))
        from lingbot_map.models.gct_stream import GCTStream  # noqa: F401
        from lingbot_map.utils.load_fn import load_and_preprocess_images  # noqa: F401

    print("=== dependency preflight ===")
    print(f"worktree={WORKTREE_ROOT}")
    print(f"mother_write_guard={MOTHER_NAV_ROOT}")
    print(f"source_root={source_root}")
    print(f"output_root={output_root}")
    print(f"lingbot_repo={lingbot_repo} commit={git_revision(lingbot_repo)}")
    print(f"weights={weights} bytes={weights.stat().st_size}")
    print(
        f"episodes={len(episodes)} total_frames={sum(len(item[2]) for item in episodes)} "
        f"max_frames={max_frames} variants={parse_variants(args.variants)}"
    )
    for relative, _episode_dir, paths in episodes:
        print(f"  {relative}: N={len(paths)} auto_interval={auto_interval(len(paths))}")
    if require_gpu:
        import torch

        print(
            f"cuda={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
            f"torch={torch.__version__}"
        )
    print("PREFLIGHT_OK", flush=True)
    return episodes


def extend_rope(model, max_frame_num: int) -> int:
    """Extend analytic RoPE tables while proving existing rows are unchanged."""
    import torch
    from lingbot_map.layers.rope import WanRotaryPosEmbed, get_1d_rotary_pos_embed

    count = 0
    for module in model.modules():
        if not isinstance(module, WanRotaryPosEmbed):
            continue
        if int(module.max_seq_len) >= max_frame_num:
            continue
        old = module.freqs
        new = torch.cat(
            [
                get_1d_rotary_pos_embed(
                    dim,
                    max_frame_num,
                    10000.0,
                    use_real=False,
                    repeat_interleave_real=False,
                    freqs_dtype=torch.float64,
                )
                for dim in module.fhw_dim
            ],
            dim=1,
        )
        if not torch.allclose(new[: old.shape[0]].to(old.dtype), old, atol=1e-6):
            raise RuntimeError("RoPE extension changed the existing table overlap")
        module.freqs = new
        module.max_seq_len = max_frame_num
        count += 1
    return count


def build_model(args: argparse.Namespace):
    import torch
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.window,
        kv_cache_scale_frames=SCALE_FRAMES,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=4,
    )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    del checkpoint, state
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    extended = extend_rope(model, args.max_frame_num)
    print(f"model_loaded rope_tables_extended={extended}", flush=True)
    return model.cuda().eval()


def infer_pose(model, images, interval: int, dtype: str) -> np.ndarray:
    import torch

    if not hasattr(model, "_set_skip_append"):
        raise RuntimeError(
            "LingBot checkout lacks _set_skip_append; use the current official keyframe implementation"
        )
    model.clean_kv_cache()
    model._set_skip_append(False)
    poses = []
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            scale = min(SCALE_FRAMES, len(images))
            block = images[:scale][None].cuda(non_blocking=True)
            aggregated, _ = model._aggregate_features(
                block,
                num_frame_for_scale=scale,
                num_frame_per_block=scale,
            )
            output = model.camera_head(
                aggregated,
                causal_inference=True,
                num_frame_per_block=scale,
                num_frame_for_scale=scale,
            )
            poses.append(output[-1][0].float().cpu())

            for index in range(scale, len(images)):
                is_keyframe = ((index - scale) % interval) == 0
                if not is_keyframe:
                    model._set_skip_append(True)
                aggregated, _ = model._aggregate_features(
                    images[index : index + 1][None].cuda(non_blocking=True),
                    num_frame_for_scale=scale,
                    num_frame_per_block=1,
                )
                output = model.camera_head(
                    aggregated,
                    causal_inference=True,
                    num_frame_per_block=1,
                    num_frame_for_scale=scale,
                )
                poses.append(output[-1][0].float().cpu())
                if not is_keyframe:
                    model._set_skip_append(False)
    finally:
        model._set_skip_append(False)
        model.clean_kv_cache()
    pose = torch.cat(poses, dim=0).numpy()
    if pose.shape != (len(images), 9):
        raise RuntimeError(f"unexpected pose shape {pose.shape}, expected {(len(images), 9)}")
    if not np.isfinite(pose).all():
        raise RuntimeError("non-finite LingBot pose output")
    return pose


def run_inference(args: argparse.Namespace) -> None:
    import torch
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    episodes = preflight(args, require_gpu=True)
    variants = parse_variants(args.variants)
    if "existing" in variants:
        raise ValueError("the existing variant is score-only")
    model = build_model(args)
    output_root = Path(args.output_root)
    revision = git_revision(Path(args.lingbot_repo))

    for episode_index, (relative, _episode_dir, paths) in enumerate(episodes, start=1):
        print(
            f"[{episode_index}/{len(episodes)}] load {relative} N={len(paths)} "
            f"mode={args.preprocess_mode}",
            flush=True,
        )
        images = load_and_preprocess_images(
            [os.fspath(path) for path in paths],
            mode=args.preprocess_mode,
            image_size=args.image_size,
            patch_size=args.patch_size,
        )
        for variant in variants:
            interval = 1 if variant == "dense" else auto_interval(len(paths))
            destination = pose_path(output_root, variant, relative)
            expected_meta = {
                "episode": relative,
                "num_frames": len(paths),
                "variant": variant,
                "keyframe_interval": interval,
                "window": args.window,
                "preprocess_mode": args.preprocess_mode,
                "dtype": args.dtype,
                "use_sdpa": bool(args.use_sdpa),
                "lingbot_commit": revision,
                "weights_bytes": Path(args.weights).stat().st_size,
            }
            if destination.is_file() and not args.overwrite:
                with np.load(destination, allow_pickle=False) as saved:
                    saved_meta = json.loads(str(saved["metadata"].item()))
                    saved_pose_shape = tuple(saved["cam_pose_enc"].shape)
                if saved_meta != expected_meta or saved_pose_shape != (len(paths), 9):
                    raise RuntimeError(
                        f"resume metadata mismatch for {destination}; use --overwrite only after review"
                    )
                print(f"  skip complete {variant}: {destination}", flush=True)
                continue

            started = time.monotonic()
            pose = infer_pose(model, images, interval=interval, dtype=args.dtype)
            atomic_savez(
                destination,
                cam_pose_enc=pose,
                metadata=np.array(json.dumps(expected_meta, sort_keys=True)),
            )
            elapsed = time.monotonic() - started
            print(
                f"  saved {variant} interval={interval} elapsed={elapsed:.1f}s {destination}",
                flush=True,
            )
        del images
        torch.cuda.empty_cache()
    print("INFERENCE_COMPLETE", flush=True)


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise RuntimeError("zero-norm quaternion in pose output")
    x, y, z, w = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def fit_sim2(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"Sim(2) expects matching [N,2] arrays, got {source.shape}, {target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    if variance < 1e-12:
        raise RuntimeError("degenerate predicted trajectory for Sim(2) alignment")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_sim2(points: np.ndarray, fit: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * (np.asarray(points) @ rotation.T) + translation


def angular_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(first * second, axis=-1)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def load_ground_truth(episode_dir: Path) -> tuple[np.ndarray, dict]:
    import pandas as pd

    dataframe = pd.read_parquet(episode_dir / "data/chunk-000/episode_000000.parquet")
    action = np.asarray(
        [np.stack(item) for item in dataframe["action"]], dtype=np.float64
    ).reshape(-1, 4, 4)
    with (episode_dir / "meta/gen_meta.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return action, metadata


def orientation_candidates(rotation: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate LingBot horizontal forward axes in its x-z ground plane."""
    candidates = {
        "c2w_col_x": rotation[:, [0, 2], 0],
        "c2w_col_z": rotation[:, [0, 2], 2],
        "w2c_row_x": rotation[:, 0, :][:, [0, 2]],
        "w2c_row_z": rotation[:, 2, :][:, [0, 2]],
    }
    candidates.update({f"neg_{key}": -value for key, value in list(candidates.items())})
    return candidates


def summarize_pose(
    pose: np.ndarray,
    action: np.ndarray,
    metadata: dict,
    rpe_gaps: tuple[int, ...] = (16, 64, 128, 256),
) -> dict:
    length = min(len(pose), len(action))
    if length < 2:
        raise RuntimeError("trajectory is too short to score")
    pose = np.asarray(pose[:length], dtype=np.float64)
    action = np.asarray(action[:length], dtype=np.float64)

    # Correct ground planes for this dataset pair:
    #   LingBot/OpenCV world: x-z with y vertical;
    #   generated NavDP data: x-y with z vertical.
    predicted_xy = pose[:, [0, 2]]
    target_xy = action[:, :2, 3]
    fit = fit_sim2(predicted_xy, target_xy)
    aligned = apply_sim2(predicted_xy, fit)
    residual = np.linalg.norm(aligned - target_xy, axis=1)

    result: dict[str, object] = {
        "n": length,
        "legs": int(metadata.get("n_legs", 0)),
        "sim2_scale": fit[0],
        "ate_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "ate_median_m": float(np.median(residual)),
        "ate_p90_m": float(np.percentile(residual, 90)),
    }

    # RPE uses the one trajectory-level Sim(2), so local nonlinear drift cannot
    # be hidden by independently aligning each interval.
    for gap in rpe_gaps:
        if length <= gap:
            continue
        predicted_delta = aligned[gap:] - aligned[:-gap]
        target_delta = target_xy[gap:] - target_xy[:-gap]
        vector_error = np.linalg.norm(predicted_delta - target_delta, axis=1)
        target_distance = np.linalg.norm(target_delta, axis=1)
        predicted_distance = np.linalg.norm(predicted_delta, axis=1)
        valid = target_distance >= 0.25
        prefix = f"rpe_{gap:03d}"
        result[f"{prefix}_rmse_m"] = float(np.sqrt(np.mean(vector_error**2)))
        result[f"{prefix}_median_m"] = float(np.median(vector_error))
        if np.any(valid):
            result[f"{prefix}_direction_median_deg"] = float(
                np.median(angular_error_deg(predicted_delta[valid], target_delta[valid]))
            )
            result[f"{prefix}_distance_ratio_median"] = float(
                np.median(predicted_distance[valid] / target_distance[valid])
            )

    switches = [int(value) for value in metadata.get("switches", [])]
    bounds = [0] + [value for value in switches if 0 < value < length] + [length]
    leg_rows = []
    for leg, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        if end - start < 2:
            continue
        predicted_delta = aligned[end - 1] - aligned[start]
        target_delta = target_xy[end - 1] - target_xy[start]
        predicted_distance = float(np.linalg.norm(predicted_delta))
        target_distance = float(np.linalg.norm(target_delta))
        leg_rows.append(
            {
                "leg": leg,
                "start": start,
                "end": end,
                "gt_displacement_m": target_distance,
                "pred_displacement_m": predicted_distance,
                "distance_ratio": predicted_distance / max(target_distance, 1e-12),
                "direction_error_deg": float(
                    angular_error_deg(predicted_delta[None], target_delta[None])[0]
                ),
                "vector_error_m": float(np.linalg.norm(predicted_delta - target_delta)),
            }
        )
    result["legs_detail"] = leg_rows

    # Use the same fitted planar rotation to audit the quaternion convention.
    # GT camera forward in this generated data is -camera local z; equivalently
    # it is the robot's local +y after removing the M_W mount.
    predicted_rotation = quat_to_matrix(pose[:, 3:7])
    target_forward = -action[:, :2, 2]
    candidate_errors = {}
    for name, forward in orientation_candidates(predicted_rotation).items():
        aligned_forward = forward @ fit[1].T
        errors = angular_error_deg(aligned_forward, target_forward)
        candidate_errors[name] = {
            "median_deg": float(np.median(errors)),
            "p90_deg": float(np.percentile(errors, 90)),
        }
    result["yaw_axis_audit"] = candidate_errors
    return result


def aggregate_rows(rows: list[dict], variant: str) -> dict:
    subset = [row for row in rows if row["variant"] == variant]
    output: dict[str, object] = {"variant": variant, "episodes": len(subset)}
    for legs in sorted({int(row["legs"]) for row in subset}):
        group = [row for row in subset if int(row["legs"]) == legs]
        ate = np.asarray([row["ate_rmse_m"] for row in group], dtype=np.float64)
        output[f"{legs}leg"] = {
            "episodes": len(group),
            "ate_mean_m": float(np.mean(ate)),
            "ate_median_m": float(np.median(ate)),
            "ate_iqr_m": [float(np.percentile(ate, 25)), float(np.percentile(ate, 75))],
            "ate_range_m": [float(np.min(ate)), float(np.max(ate))],
        }
    return output


def run_score(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    pose_root = Path(args.pose_root or args.output_root)
    episodes = select_episodes(discover_episodes(source_root), args.episodes)
    variants = parse_variants(args.variants)
    rows = []
    for relative, episode_dir, _paths in episodes:
        action, metadata = load_ground_truth(episode_dir)
        for variant in variants:
            path = pose_path(pose_root, variant, relative)
            if not path.is_file():
                if args.allow_missing:
                    print(f"MISSING {variant} {relative}: {path}")
                    continue
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as saved:
                if "cam_pose_enc" not in saved:
                    raise RuntimeError(f"missing cam_pose_enc in {path}")
                pose = np.asarray(saved["cam_pose_enc"], dtype=np.float64)
                pose_metadata = (
                    json.loads(str(saved["metadata"].item()))
                    if "metadata" in saved
                    else None
                )
            expected_interval = (
                1
                if variant == "dense"
                else auto_interval(len(_paths))
                if variant == "auto"
                else None
            )
            if pose_metadata is not None:
                saved_interval = int(pose_metadata["keyframe_interval"])
                if expected_interval is not None and saved_interval != expected_interval:
                    raise RuntimeError(
                        f"interval mismatch in {path}: saved={saved_interval}, "
                        f"expected={expected_interval}"
                    )
                interval = saved_interval
            else:
                interval = expected_interval
            row = summarize_pose(pose, action, metadata)
            row.update(
                episode=relative,
                variant=variant,
                interval=interval,
                pose_path=os.fspath(path),
            )
            rows.append(row)
            leg_text = ", ".join(
                f"L{leg['leg']}={leg['pred_displacement_m']:.2f}/{leg['gt_displacement_m']:.2f}m "
                f"dir={leg['direction_error_deg']:.1f}deg"
                for leg in row["legs_detail"]
            )
            best_axis, best_value = min(
                row["yaw_axis_audit"].items(), key=lambda item: item[1]["median_deg"]
            )
            print(
                f"{variant:8s} {relative:50s} N={row['n']:4d} "
                f"ATE={row['ate_rmse_m']:.3f}m scale={row['sim2_scale']:.3f} "
                f"yaw={best_axis}:{best_value['median_deg']:.2f}deg | {leg_text}"
            )

    summaries = [aggregate_rows(rows, variant) for variant in variants]
    report = {
        "schema_version": 2,
        "coordinate_convention": {
            "lingbot_translation_plane": "x-z (y vertical)",
            "dataset_translation_plane": "x-y (z vertical)",
            "alignment": "one proper Sim(2) per episode; no reflection",
            "angle_wrap": "atan2(cross,dot), absolute shortest circular error",
        },
        "source_root": os.fspath(source_root),
        "pose_root": os.fspath(pose_root),
        "rows": rows,
        "summaries": summaries,
    }
    destination = Path(args.report)
    if is_relative_to(destination, MOTHER_NAV_ROOT):
        raise RuntimeError(f"refusing to write report in mother Nav tree: {destination}")
    if not is_relative_to(destination, WORKTREE_ROOT):
        raise RuntimeError(f"report must stay inside personal worktree: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    print("\n=== aggregate ===")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"REPORT_SAVED {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "infer", "score"])
    parser.add_argument("--source-root", default=os.fspath(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=os.fspath(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pose-root", default="")
    parser.add_argument("--lingbot-repo", default=os.fspath(DEFAULT_LINGBOT_REPO))
    parser.add_argument("--weights", default=os.fspath(DEFAULT_WEIGHTS))
    parser.add_argument("--episodes", default="", help="comma-separated relative episode paths")
    parser.add_argument("--variants", default="dense,auto")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--preprocess-mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--report",
        default=os.fspath(DEFAULT_OUTPUT_ROOT / "keyframe_interval_report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        preflight(args, require_gpu=True)
    elif args.command == "infer":
        # Make imports deterministic before run_inference imports LingBot modules.
        sys.path.insert(0, os.fspath(Path(args.lingbot_repo)))
        run_inference(args)
    elif args.command == "score":
        run_score(args)
    else:  # pragma: no cover - argparse enforces choices
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
