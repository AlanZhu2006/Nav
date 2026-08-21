#!/usr/bin/env python3
"""Small, label-blind Pi3X multi-view relocalization diagnostic.

The script gives Pi3X a causal temporal window around one retrieved history
anchor followed by the ImageGoal.  It then asks whether the predicted goal
point cloud is geometrically supported by any history view in the *joint*
reconstruction.  Labels and simulator goal poses are used only after inference
for reporting; they never enter the score.

This is deliberately a smoke diagnostic, not a model-selection result.  The
local machine currently contains full trajectories for only two PT1 scenes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.spatial import cKDTree


DEFAULT_ROWS = Path(
    ".diagnostics/certificate_distilled_compass_20260813/"
    "static_top8_480_lightglue_open_set_rows.csv"
)
DEFAULT_DATA_ROOT = Path(
    "/home/asus/Research/Nav/memnav_viz/validate_gated/mp3d_3leg"
)
DEFAULT_PI3_ROOT = Path("/home/asus/Research/Pi3")
DEFAULT_SNAPSHOT = Path(
    "/home/asus/.cache/huggingface/hub/models--yyfz233--Pi3X/"
    "snapshots/bb1deea4d7423de5b30691739cb451a3f57dc1d5"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _verify_source_images(
    data_root: Path,
    rows: Iterable[dict[str, str]],
) -> int:
    """Bind query/candidate images to the frozen CSV content identities."""
    cache: dict[Path, str] = {}
    for row in rows:
        episode_root = data_root / row["scene"] / row["episode"]
        pairs = (
            (
                data_root / row["query_relative_path"],
                row.get("query_content_sha256", ""),
            ),
            (
                episode_root / (
                    "videos/chunk-000/observation.images.rgb/"
                    f"{row['candidate_frame']}.jpg"
                ),
                row.get("candidate_rgb_content_sha256", ""),
            ),
        )
        for path, expected in pairs:
            if len(expected) != 64:
                raise ValueError(f"missing frozen image SHA for {path}")
            if path not in cache:
                if not path.is_file():
                    raise FileNotFoundError(path)
                cache[path] = _sha256(path)
            if cache[path] != expected:
                raise ValueError(
                    f"source image content mismatch: {path}: "
                    f"{cache[path]} != {expected}"
                )
    return len(cache)


def _action_poses(episode_dir: Path) -> np.ndarray:
    parquet = episode_dir / "data/chunk-000/episode_000000.parquet"
    values = pq.read_table(parquet, columns=["action"])["action"].to_pylist()
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"unexpected action pose shape {poses.shape} in {parquet}")
    return poses


def _goal_center(query_path: Path, goal_role: str) -> np.ndarray:
    meta = json.loads((query_path.parent / "meta/gen_meta.json").read_text())
    goal = next(item for item in meta["goals"] if item["name"] == goal_role)
    center = np.asarray(goal["pos"], dtype=np.float64)
    # Generator positions are agent feet; recorded RGB poses are camera centers.
    center[2] += 0.5
    return center


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t such that dst ~= s * R @ src + t."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("similarity inputs must both have shape [N,3]")
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_zero, dst_zero = src - src_mean, dst - dst_mean
    variance = float(np.square(src_zero).sum() / len(src))
    if variance < 1e-10:
        raise ValueError("degenerate predicted history trajectory")
    covariance = dst_zero.T @ src_zero / len(src)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[2, 2] = -1
    rotation = left @ sign @ right_t
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def _transform(points: np.ndarray, scale: float, rotation: np.ndarray,
               translation: np.ndarray) -> np.ndarray:
    return scale * points @ rotation.T + translation


def _filtered_points(
    points: np.ndarray,
    confidence: np.ndarray,
    center: np.ndarray,
    *,
    stride: int,
    confidence_quantile: float,
    min_range_m: float = 0.15,
    max_range_m: float = 10.0,
) -> np.ndarray:
    points = points[::stride, ::stride].reshape(-1, 3)
    confidence = confidence[::stride, ::stride].reshape(-1)
    finite = np.isfinite(points).all(1) & np.isfinite(confidence)
    if not finite.any():
        return np.empty((0, 3), dtype=np.float64)
    cutoff = float(np.quantile(confidence[finite], confidence_quantile))
    distance = np.linalg.norm(points - center[None], axis=1)
    mask = finite & (confidence >= cutoff)
    mask &= (distance >= min_range_m) & (distance <= max_range_m)
    return points[mask]


def _directed_nn(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        return np.asarray([math.inf], dtype=np.float64)
    return cKDTree(target).query(source, k=1, workers=-1)[0]


def _overlap_metrics(
    goal_points: np.ndarray,
    history_points: Iterable[np.ndarray],
    thresholds_m: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50),
) -> dict[str, float]:
    history_points = list(history_points)
    union = np.concatenate(history_points, axis=0)
    goal_to_union = _directed_nn(goal_points, union)
    union_to_goal = _directed_nn(union, goal_points)
    result: dict[str, float] = {
        "goal_to_history_q10_m": float(np.quantile(goal_to_union, 0.10)),
        "goal_to_history_q25_m": float(np.quantile(goal_to_union, 0.25)),
        "goal_to_history_q50_m": float(np.quantile(goal_to_union, 0.50)),
        "history_to_goal_q25_m": float(np.quantile(union_to_goal, 0.25)),
    }
    for threshold in thresholds_m:
        suffix = f"{round(100 * threshold):02d}cm"
        forward = float(np.mean(goal_to_union <= threshold))
        reverse = float(np.mean(union_to_goal <= threshold))
        result[f"union_goal_overlap_{suffix}"] = forward
        result[f"union_history_overlap_{suffix}"] = reverse
        best_f1 = 0.0
        for history in history_points:
            g_to_h = _directed_nn(goal_points, history)
            h_to_g = _directed_nn(history, goal_points)
            precision = float(np.mean(g_to_h <= threshold))
            recall = float(np.mean(h_to_g <= threshold))
            if precision + recall > 0:
                best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
        result[f"best_view_f1_{suffix}"] = best_f1
    return result


def _planar_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)[:2]
    second = np.asarray(second, dtype=np.float64)[:2]
    first_norm, second_norm = np.linalg.norm(first), np.linalg.norm(second)
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return math.nan
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _scale_free_bearing_from_c2w(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Return OpenCV-camera [forward, left] from two predicted c2w poses."""
    relative = current[:3, :3].T @ (goal[:3, 3] - current[:3, 3])
    return np.asarray([relative[2], -relative[0]], dtype=np.float64)


def _inference_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    """Resolve autocast dtype; ``None`` means native float32 execution."""
    if requested == "float32":
        return None
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if requested != "auto":
        raise ValueError(f"unsupported inference dtype {requested!r}")
    if device.type != "cuda":
        return None
    return (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )


def _pack_view_descriptors(
    items: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Pad variable-length per-view tokens without losing the row identity."""
    if not items:
        raise ValueError("cannot pack an empty descriptor collection")
    dimensions = {np.asarray(item["descriptors"]).shape[1] for item in items}
    if len(dimensions) != 1:
        raise ValueError("view descriptor dimensions differ")
    dimension = dimensions.pop()
    maximum_views = max(len(item["descriptors"]) for item in items)
    descriptors = np.zeros(
        (len(items), maximum_views, dimension), dtype=np.float16
    )
    roles = np.full((len(items), maximum_views), -1, dtype=np.int8)
    relative_age = np.zeros((len(items), maximum_views), dtype=np.float32)
    valid = np.zeros((len(items), maximum_views), dtype=np.bool_)
    row_indices = np.zeros(len(items), dtype=np.int64)
    view_counts = np.zeros(len(items), dtype=np.int16)
    for output_index, item in enumerate(items):
        values = np.asarray(item["descriptors"], dtype=np.float16)
        item_roles = np.asarray(item["roles"], dtype=np.int8)
        item_ages = np.asarray(item["relative_age"], dtype=np.float32)
        count = len(values)
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError("invalid descriptor matrix")
        if item_roles.shape != (count,) or item_ages.shape != (count,):
            raise ValueError("descriptor metadata has a different view count")
        descriptors[output_index, :count] = values
        roles[output_index, :count] = item_roles
        relative_age[output_index, :count] = item_ages
        valid[output_index, :count] = True
        row_indices[output_index] = int(item["row_index"])
        view_counts[output_index] = count
    return {
        "row_indices": row_indices,
        "view_counts": view_counts,
        "view_descriptors": descriptors,
        "view_roles": roles,
        "view_relative_age": relative_age,
        "view_valid": valid,
    }


def _scale_free_spatial_geometry(
    predicted_poses: np.ndarray,
    predicted_points: np.ndarray,
    local_points: np.ndarray,
    confidence: np.ndarray,
    *,
    patch_size: int,
) -> dict[str, np.ndarray | float]:
    """Preserve Pi3X spatial evidence in the live-current camera gauge.

    The transformation and scale are derived only from Pi3X outputs.  They do
    not use simulator poses, co-visibility labels, or certificate features.
    """
    poses = np.asarray(predicted_poses, dtype=np.float64)
    world = np.asarray(predicted_points, dtype=np.float64)
    local = np.asarray(local_points, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or world.shape != local.shape or world.ndim != 4
            or world.shape[-1] != 3 or conf.shape != world.shape[:-1]):
        raise ValueError("invalid Pi3X spatial output shapes")
    height, width = world.shape[1:3]
    if height % patch_size or width % patch_size:
        raise ValueError("spatial output is not divisible by the patch size")
    center = patch_size // 2
    world = world[:, center::patch_size, center::patch_size]
    local = local[:, center::patch_size, center::patch_size]
    conf = conf[:, center::patch_size, center::patch_size]
    world_from_current = poses[0]
    current_from_world = np.linalg.inv(world_from_current)
    homogeneous = np.concatenate(
        [world, np.ones((*world.shape[:-1], 1), dtype=np.float64)], axis=-1
    )
    points_current = np.einsum(
        "ij,nhwj->nhwi", current_from_world, homogeneous
    )[..., :3]
    poses_current = np.einsum("ij,njk->nik", current_from_world, poses)
    current_depth = local[0, ..., 2]
    finite_positive = np.isfinite(current_depth) & (current_depth > 1e-6)
    high_confidence = finite_positive & (conf[0] >= np.median(conf[0]))
    scale_values = current_depth[high_confidence]
    if not len(scale_values):
        scale_values = current_depth[finite_positive]
    if not len(scale_values):
        raise ValueError("Pi3X current view has no finite positive depth")
    scale = float(np.median(scale_values))
    if not math.isfinite(scale) or scale <= 1e-6:
        raise ValueError("invalid label-blind Pi3X spatial scale")
    points_current /= scale
    local /= scale
    poses_current[:, :3, 3] /= scale
    outputs = {
        "world_points_in_current": points_current.astype(np.float16),
        "local_points": local.astype(np.float16),
        "confidence": conf.astype(np.float16),
        "poses_in_current": poses_current[:, :3].astype(np.float16),
        "normalization_scale": scale,
    }
    for name, value in outputs.items():
        if name != "normalization_scale" and not np.isfinite(value).all():
            raise ValueError(f"non-finite float16 spatial output in {name}")
    return outputs


def _pack_spatial_geometry(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pad variable-view spatial Pi3X outputs while retaining row identity."""
    if not items:
        raise ValueError("cannot pack an empty spatial collection")
    point_shapes = {
        tuple(np.asarray(item["world_points_in_current"]).shape[1:])
        for item in items
    }
    if len(point_shapes) != 1:
        raise ValueError("spatial patch shapes differ")
    patch_h, patch_w, channels = point_shapes.pop()
    if channels != 3:
        raise ValueError("spatial point channel count differs")
    maximum_views = max(len(item["world_points_in_current"]) for item in items)
    rows = len(items)
    shape = (rows, maximum_views, patch_h, patch_w)
    world = np.zeros((*shape, 3), dtype=np.float16)
    local = np.zeros((*shape, 3), dtype=np.float16)
    confidence = np.zeros((*shape, 1), dtype=np.float16)
    poses = np.zeros((rows, maximum_views, 3, 4), dtype=np.float16)
    roles = np.full((rows, maximum_views), -1, dtype=np.int8)
    relative_age = np.zeros((rows, maximum_views), dtype=np.float32)
    valid = np.zeros((rows, maximum_views), dtype=np.bool_)
    row_indices = np.zeros(rows, dtype=np.int64)
    view_counts = np.zeros(rows, dtype=np.int16)
    scales = np.zeros(rows, dtype=np.float32)
    for output_index, item in enumerate(items):
        item_world = np.asarray(item["world_points_in_current"], dtype=np.float16)
        count = len(item_world)
        world[output_index, :count] = item_world
        local[output_index, :count] = np.asarray(item["local_points"], dtype=np.float16)
        confidence[output_index, :count, ..., 0] = np.asarray(
            item["confidence"], dtype=np.float16
        )
        poses[output_index, :count] = np.asarray(
            item["poses_in_current"], dtype=np.float16
        )
        roles[output_index, :count] = np.asarray(item["roles"], dtype=np.int8)
        relative_age[output_index, :count] = np.asarray(
            item["relative_age"], dtype=np.float32
        )
        valid[output_index, :count] = True
        row_indices[output_index] = int(item["row_index"])
        view_counts[output_index] = count
        scales[output_index] = float(item["normalization_scale"])
    return {
        "row_indices": row_indices,
        "view_counts": view_counts,
        "view_world_points_in_current": world,
        "view_local_points": local,
        "view_confidence": confidence,
        "view_poses_in_current": poses,
        "view_roles": roles,
        "view_relative_age": relative_age,
        "view_valid": valid,
        "normalization_scale": scales,
    }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary_name, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _true_bearing_from_generator_pose(current_gl: np.ndarray,
                                      true_goal_center: np.ndarray) -> np.ndarray:
    """Return [forward, left] for generator OpenGL c2w and a world goal.

    Generator ``action`` poses use right/up/back camera axes.  Pi3X documents
    right/down/forward (OpenCV), hence the fixed two-axis sign conversion.
    This function is reporting-only and never feeds the model or score.
    """
    gl_to_cv = np.diag([1.0, -1.0, -1.0])
    current_cv_rotation = current_gl[:3, :3] @ gl_to_cv
    relative = current_cv_rotation.T @ (true_goal_center - current_gl[:3, 3])
    return np.asarray([relative[2], -relative[0]], dtype=np.float64)


def _history_frames(anchor: int, decision_frame: int, n_frames: int,
                    offsets: tuple[int, ...]) -> list[int]:
    upper = min(decision_frame - 1, n_frames - 1)
    frames = sorted({min(max(anchor + offset, 0), upper) for offset in offsets})
    if anchor not in frames:
        frames.append(anchor)
        frames.sort()
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} distinct causal history frames")
    return frames


def _causal_bridge_frames(
    anchor: int,
    decision_frame: int,
    n_frames: int,
    *,
    bridge_count: int,
    anchor_offsets: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    """Return a current-to-anchor visual bridge and its local support frames."""
    upper = min(decision_frame - 1, n_frames - 1)
    if not 0 <= anchor <= upper:
        raise ValueError("anchor must be strictly inside the causal history")
    if bridge_count < 2:
        raise ValueError("bridge_count must be at least two")
    bridge = {
        int(round(value))
        for value in np.linspace(anchor, upper, num=bridge_count)
    }
    support = {
        min(max(anchor + offset, 0), upper) for offset in anchor_offsets
    }
    support.add(anchor)
    bridge.update(support)
    # Current view is input zero.  Reverse temporal order makes every following
    # bridge frame progress from current back toward the retrieved anchor.
    frames = sorted(bridge, reverse=True)
    support_frames = sorted(support)
    return frames, support_frames


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    sys.path.insert(0, str(args.pi3_root))
    from pi3.models.pi3x import Pi3X

    with args.rows_csv.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows_sha256 = _sha256(args.rows_csv)
    if (args.expected_rows_sha256 is not None
            and rows_sha256 != args.expected_rows_sha256):
        raise ValueError(
            f"rows CSV SHA mismatch: {rows_sha256} != {args.expected_rows_sha256}"
        )
    model_path = args.snapshot / "model.safetensors"
    # Hugging Face snapshots normally expose a symlink into a content-addressed
    # blob store.  The explicit SHA pin below is the identity authority.
    if not model_path.is_file():
        raise FileNotFoundError(f"Pi3X weights missing: {model_path}")
    model_sha256 = _sha256(model_path)
    if (args.expected_model_sha256 is not None
            and model_sha256 != args.expected_model_sha256):
        raise ValueError(
            f"Pi3X weight SHA mismatch: {model_sha256} != {args.expected_model_sha256}"
        )
    if args.all_available:
        selected = []
        for index, row in enumerate(all_rows):
            episode_dir = args.data_root / row["scene"] / row["episode"]
            query_path = args.data_root / row["query_relative_path"]
            candidate_path = episode_dir / (
                "videos/chunk-000/observation.images.rgb/"
                f"{row['candidate_frame']}.jpg"
            )
            parquet = episode_dir / "data/chunk-000/episode_000000.parquet"
            if query_path.exists() and candidate_path.exists() and parquet.exists():
                selected.append((index, row))
    else:
        selected = [(index, all_rows[index]) for index in args.row_indices]

    if args.expected_output_rows is not None and len(selected) != args.expected_output_rows:
        raise ValueError(
            f"selected {len(selected)} rows, expected {args.expected_output_rows}"
        )
    verified_source_images = _verify_source_images(
        args.data_root, [row for _, row in selected]
    )

    device = torch.device(args.device)
    model = Pi3X.from_pretrained(str(args.snapshot), local_files_only=True).eval()
    model.disable_multimodal()
    model = model.to(device)
    dtype = _inference_dtype(device, args.inference_dtype)

    outputs: list[dict[str, Any]] = []
    descriptor_items: list[dict[str, Any]] = []
    spatial_items: list[dict[str, Any]] = []
    captured_camera_decoder_input: dict[str, torch.Tensor] = {}
    descriptor_hook = None
    if args.output_descriptors_npz is not None:
        def capture_camera_decoder_input(_module, inputs) -> None:
            if not inputs:
                raise RuntimeError("Pi3X camera decoder received no hidden input")
            captured_camera_decoder_input["hidden"] = inputs[0].detach()

        descriptor_hook = model.camera_decoder.register_forward_pre_hook(
            capture_camera_decoder_input
        )
    for row_index, row in selected:
        episode_dir = args.data_root / row["scene"] / row["episode"]
        poses = _action_poses(episode_dir)
        anchor = int(row["candidate_frame"])
        if args.history_mode == "causal_bridge":
            frames, support_frames = _causal_bridge_frames(
                anchor,
                int(row["decision_frame"]),
                len(poses),
                bridge_count=args.bridge_frames,
                anchor_offsets=tuple(args.anchor_offsets),
            )
        else:
            frames = _history_frames(
                anchor,
                int(row["decision_frame"]),
                len(poses),
                tuple(args.offsets),
            )
            support_frames = list(frames)
        history_paths = [
            episode_dir / f"videos/chunk-000/observation.images.rgb/{frame}.jpg"
            for frame in frames
        ]
        query_path = args.data_root / row["query_relative_path"]
        missing = [str(path) for path in [*history_paths, query_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"row {row_index} missing: {missing}")

        decision_frame = min(int(row["decision_frame"]), len(poses) - 1)
        current_path = episode_dir / (
            "videos/chunk-000/observation.images.rgb/"
            f"{decision_frame}.jpg"
        )
        if not current_path.exists():
            raise FileNotFoundError(f"row {row_index} missing current view: {current_path}")
        # Put the live camera first so Pi3X's gauge is anchored at the frame in
        # which NavDP needs the bearing.  History still supplies revisit proof.
        images = torch.stack([
            _load_rgb(path, args.width, args.height)
            for path in [current_path, *history_paths, query_path]
        ])[None].to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        autocast_context = (
            contextlib.nullcontext()
            if dtype is None
            else torch.autocast(device_type=device.type, dtype=dtype)
        )
        with torch.inference_mode(), autocast_context:
            prediction = model(imgs=images)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started

        predicted_poses = prediction["camera_poses"][0].float().cpu().numpy()
        predicted_points = prediction["points"][0].float().cpu().numpy()
        predicted_local_points = prediction["local_points"][0].float().cpu().numpy()
        confidence = torch.sigmoid(prediction["conf"][0, ..., 0]).float().cpu().numpy()
        denominator = max(decision_frame - anchor, 1)
        roles = [0]
        roles.extend(2 if frame == anchor else 1 for frame in frames)
        roles.append(3)
        relative_age = [0.0]
        relative_age.extend(
            (decision_frame - frame) / denominator for frame in frames
        )
        relative_age.append(-1.0)
        if args.output_descriptors_npz is not None:
            hidden = captured_camera_decoder_input.pop("hidden", None)
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != len(frames) + 2:
                raise RuntimeError("failed to capture one Pi3X hidden sequence")
            register_count = int(model.patch_start_idx)
            if register_count <= 0 or hidden.shape[1] <= register_count:
                raise RuntimeError("Pi3X hidden sequence lacks register/patch tokens")
            view_descriptors = hidden[:, :register_count].mean(dim=1).float().cpu().numpy()
            descriptor_items.append({
                "row_index": row_index,
                "descriptors": view_descriptors,
                "roles": roles,
                "relative_age": relative_age,
            })
        if args.output_spatial_npz is not None:
            spatial = _scale_free_spatial_geometry(
                predicted_poses,
                predicted_points,
                predicted_local_points,
                confidence,
                patch_size=int(model.patch_size),
            )
            spatial_items.append({
                "row_index": row_index,
                "roles": roles,
                "relative_age": relative_age,
                **spatial,
            })
        scale, rotation, translation = _umeyama(
            predicted_poses[1:1 + len(frames), :3, 3], poses[frames, :3, 3]
        )
        aligned_poses = predicted_poses.copy()
        aligned_poses[:, :3, 3] = _transform(
            predicted_poses[:, :3, 3], scale, rotation, translation
        )
        # All authorization features below stay in Pi3X's own joint metric
        # gauge.  The GT-derived Sim(3) above is reporting-only.
        filtered = [
            _filtered_points(
                predicted_points[index],
                confidence[index],
                predicted_poses[index, :3, 3],
                stride=args.point_stride,
                confidence_quantile=args.confidence_quantile,
            )
            for index in range(len(frames) + 2)
        ]
        support_indices = [
            1 + frames.index(frame) for frame in support_frames
        ]
        history_error = np.linalg.norm(
            aligned_poses[1:1 + len(frames), :3, 3] - poses[frames, :3, 3], axis=1
        )
        true_goal_center = _goal_center(query_path, row["goal_role"])
        predicted_goal_center = aligned_poses[-1, :3, 3]
        predicted_bearing = _scale_free_bearing_from_c2w(
            predicted_poses[0], predicted_poses[-1]
        )
        true_bearing = _true_bearing_from_generator_pose(
            poses[decision_frame], true_goal_center
        )
        result: dict[str, Any] = {
            "row_index": row_index,
            "scene": row["scene"],
            "episode": row["episode"],
            "candidate_rank": int(row["candidate_rank"]),
            "anchor_frame": anchor,
            "history_frames": frames,
            "support_frames": support_frames,
            "history_mode": args.history_mode,
            "current_frame": decision_frame,
            "candidate_label_reporting_only": int(row["candidate_label"]),
            "teacher_covis_reporting_only": float(row["teacher_covis"]),
            "elapsed_s": elapsed_s,
            "sim3_scale_reporting_only": scale,
            "history_alignment_rmse_m_reporting_only": float(np.sqrt(np.mean(history_error ** 2))),
            "goal_position_error_m_reporting_only": float(np.linalg.norm(
                predicted_goal_center - true_goal_center
            )),
            "goal_bearing_error_deg_reporting_only": _planar_angle_deg(
                predicted_bearing, true_bearing
            ),
            "predicted_scale_free_bearing": predicted_bearing.tolist(),
            "true_scale_free_bearing_reporting_only": true_bearing.tolist(),
            "predicted_goal_center_reporting_only": predicted_goal_center.tolist(),
            "true_goal_center_reporting_only": true_goal_center.tolist(),
            "history_confidence_median": float(np.median(confidence[1:1 + len(frames)])),
            "current_confidence_median": float(np.median(confidence[0])),
            "goal_confidence_median": float(np.median(confidence[-1])),
            "point_counts": [len(points) for points in filtered],
        }
        result.update(_overlap_metrics(
            filtered[-1], [filtered[index] for index in support_indices]
        ))
        outputs.append(result)
        if not args.quiet:
            print(json.dumps(result, sort_keys=True), flush=True)
        del prediction, images
        torch.cuda.empty_cache()
    if descriptor_hook is not None:
        descriptor_hook.remove()
    descriptor_sha256 = None
    if args.output_descriptors_npz is not None:
        packed = _pack_view_descriptors(descriptor_items)
        _atomic_npz(args.output_descriptors_npz, packed)
        descriptor_sha256 = _sha256(args.output_descriptors_npz)
        descriptor_receipt = {
            "schema_version": 1,
            "method": "pi3x_post_decoder_register_mean_per_view",
            "row_indices_match_shadow_order": True,
            "roles": {
                "-1": "padding",
                "0": "current",
                "1": "causal_bridge_or_local_support",
                "2": "retrieved_anchor",
                "3": "image_goal",
            },
            "contains_reporting_labels": False,
            "output": str(args.output_descriptors_npz),
            "output_sha256": descriptor_sha256,
            "rows": len(descriptor_items),
            "shape": list(packed["view_descriptors"].shape),
            "storage_dtype": str(packed["view_descriptors"].dtype),
            "model_sha256": model_sha256,
            "rows_csv_sha256": rows_sha256,
            "history_mode": args.history_mode,
            "bridge_frames": args.bridge_frames,
        }
        receipt_path = args.output_descriptors_npz.with_suffix(
            args.output_descriptors_npz.suffix + ".receipt.json"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt_path.name}.", dir=receipt_path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(descriptor_receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, receipt_path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    spatial_sha256 = None
    if args.output_spatial_npz is not None:
        packed_spatial = _pack_spatial_geometry(spatial_items)
        _atomic_npz(args.output_spatial_npz, packed_spatial)
        spatial_sha256 = _sha256(args.output_spatial_npz)
        spatial_receipt = {
            "schema_version": 1,
            "method": "pi3x_scale_free_patch_grid_in_live_current_gauge",
            "row_indices_match_shadow_order": True,
            "contains_reporting_labels": False,
            "contains_certificate_features": False,
            "normalization": "median_positive_high_confidence_current_depth",
            "roles": {
                "-1": "padding",
                "0": "current",
                "1": "causal_bridge_or_local_support",
                "2": "retrieved_anchor",
                "3": "image_goal",
            },
            "output": str(args.output_spatial_npz),
            "output_sha256": spatial_sha256,
            "rows": len(spatial_items),
            "world_points_shape": list(
                packed_spatial["view_world_points_in_current"].shape
            ),
            "storage_dtype": str(
                packed_spatial["view_world_points_in_current"].dtype
            ),
            "model_sha256": model_sha256,
            "rows_csv_sha256": rows_sha256,
            "history_mode": args.history_mode,
            "bridge_frames": args.bridge_frames,
            "patch_size": int(model.patch_size),
        }
        receipt_path = args.output_spatial_npz.with_suffix(
            args.output_spatial_npz.suffix + ".receipt.json"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt_path.name}.", dir=receipt_path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(spatial_receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, receipt_path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output_jsonl.name}.",
            dir=args.output_jsonl.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                for output in outputs:
                    handle.write(json.dumps(output, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, args.output_jsonl)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        print(json.dumps({
            "output_jsonl": str(args.output_jsonl),
            "rows": len(outputs),
        }, sort_keys=True), flush=True)
        receipt_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".receipt.json")
        receipt = {
            "schema_version": 1,
            "method": "pi3x_joint_temporal_history_goal_shadow",
            "inference_is_label_blind": True,
            "reporting_labels_never_enter_model_or_score": True,
            "source_query_candidate_hashes_verified": True,
            "unique_source_images_verified": verified_source_images,
            "rows_csv": str(args.rows_csv),
            "rows_csv_sha256": rows_sha256,
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "pi3_root": str(args.pi3_root),
            "data_root": str(args.data_root),
            "output_jsonl": str(args.output_jsonl),
            "output_jsonl_sha256": _sha256(args.output_jsonl),
            "rows": len(outputs),
            "distinct_scenes": len({output["scene"] for output in outputs}),
            "width": args.width,
            "height": args.height,
            "history_offsets": list(args.offsets),
            "history_mode": args.history_mode,
            "bridge_frames": args.bridge_frames,
            "anchor_offsets": list(args.anchor_offsets),
            "point_stride": args.point_stride,
            "confidence_quantile": args.confidence_quantile,
            "device": str(device),
            "inference_dtype_requested": args.inference_dtype,
            "inference_dtype_effective": (
                "float32" if dtype is None else str(dtype).removeprefix("torch.")
            ),
            "torch_version": torch.__version__,
            "view_descriptors_npz": (
                str(args.output_descriptors_npz)
                if args.output_descriptors_npz is not None else None
            ),
            "view_descriptors_npz_sha256": descriptor_sha256,
            "spatial_geometry_npz": (
                str(args.output_spatial_npz)
                if args.output_spatial_npz is not None else None
            ),
            "spatial_geometry_npz_sha256": spatial_sha256,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt_path.name}.", dir=receipt_path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, receipt_path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pi3-root", type=Path, default=DEFAULT_PI3_ROOT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--row-indices", type=int, nargs="+", default=[40, 0, 11, 16])
    parser.add_argument("--all-available", action="store_true")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-descriptors-npz", type=Path)
    parser.add_argument("--output-spatial-npz", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--expected-output-rows", type=int)
    parser.add_argument("--expected-rows-sha256")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--offsets", type=int, nargs="+", default=[-16, -8, 0, 8, 16])
    parser.add_argument(
        "--history-mode",
        choices=("causal_bridge", "local_window"),
        default="causal_bridge",
    )
    parser.add_argument("--bridge-frames", type=int, default=8)
    parser.add_argument("--anchor-offsets", type=int, nargs="+", default=[-8, 0, 8])
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=126)
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--confidence-quantile", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    args = parser.parse_args()
    if args.width % 14 or args.height % 14:
        parser.error("width and height must be divisible by 14")
    return args


if __name__ == "__main__":
    run(parse_args())
