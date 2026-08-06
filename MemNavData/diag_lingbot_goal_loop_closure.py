#!/usr/bin/env python3
"""Zero-training feasibility test for LingBot-native goal loop closure.

The current geometry router uses DINO for coarse retrieval and SIFT/RANSAC for
candidate verification.  This diagnostic asks whether LingBot's *own* streaming
geometry can provide the verification signal instead:

1. Select scene/session-balanced positive and hard-negative candidate anchors
   from an existing task-aligned co-visibility teacher CSV.
2. Append the same goal image after the candidate and nearby temporal anchors.
3. Measure whether the independently inferred goal poses agree in the common
   LingBot map frame (pose consensus).
4. Predict depth for both the anchor and appended goal, transform the two point
   clouds into that map frame, and measure their symmetric 3-D overlap.

No model weights are changed.  Source data, feature caches, and checkpoints are
read-only; only a CSV and JSON report are written below ``--out-dir``.

This is deliberately a small feasibility diagnostic, not a deployment router.
Thresholds must not be chosen from final-reserved scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = {
    "session_id",
    "scene",
    "episode",
    "kind",
    "query_path",
    "candidate_path",
    "candidate_frame",
    "dino_cosine",
    "teacher_covis",
}


@dataclass(frozen=True)
class CandidateSeed:
    session_id: str
    scene: str
    episode: str
    kind: str
    query_path: Path
    candidate_path: Path
    candidate_frame: int
    dino_cosine: float
    teacher_covis: float
    label: int
    session_has_positive: bool
    session_is_strict_no_match: bool
    session_max_covis: float


@dataclass(frozen=True)
class EpisodePoseData:
    """Ground-truth camera trajectory and generator metadata for one episode."""

    actions: np.ndarray
    base_extrinsic: np.ndarray
    metadata: dict


_HABITAT_TO_DATA_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)
_LINGBOT_TO_DATA_ROTATION_BASIS = np.diag([-1.0, -1.0, 1.0])
_DEFAULT_POOLED_METRIC_SCALE = 2.564


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            # The shared LingBot checkout is owned by another project member.
            # Scope Git's ownership exception to this one read-only invocation;
            # do not mutate the user's global safe.directory configuration.
            ["git", "-c", f"safe.directory={root.resolve()}",
             "-C", str(root), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def temporal_diverse(rows: pd.DataFrame, count: int,
                     minimum_gap: int) -> List[pd.Series]:
    """Greedy high-DINO selection with a minimum raw-frame separation."""
    chosen: List[pd.Series] = []
    for _, row in rows.sort_values(
            ["dino_cosine", "candidate_frame"],
            ascending=[False, True]).iterrows():
        frame = int(row["candidate_frame"])
        if all(abs(frame - int(old["candidate_frame"])) >= minimum_gap
               for old in chosen):
            chosen.append(row)
            if len(chosen) == count:
                break
    return chosen


def select_balanced_seeds(frame: pd.DataFrame, *, kind: str,
                          sessions: Sequence[str], max_sessions: int,
                          per_class: int, minimum_gap: int,
                          positive_threshold: float,
                          negative_threshold: float,
                          minimum_anchor: int) -> List[CandidateSeed]:
    data = frame.loc[frame["kind"].eq(kind)].copy()
    if sessions:
        data = data.loc[data["session_id"].isin(sessions)]
    data = data.loc[
        data["teacher_covis"].notna()
        & data["dino_cosine"].notna()
        & data["candidate_frame"].ge(minimum_anchor)]
    session_ids = sorted(data["session_id"].unique().tolist())
    if max_sessions:
        session_ids = session_ids[:max_sessions]

    result: List[CandidateSeed] = []
    for session_id in session_ids:
        group = data.loc[data["session_id"].eq(session_id)]
        positive = group.loc[
            group["teacher_covis"].ge(positive_threshold)]
        negative = group.loc[
            group["teacher_covis"].le(negative_threshold)]
        selected = [
            (1, row) for row in temporal_diverse(
                positive, per_class, minimum_gap)
        ] + [
            (0, row) for row in temporal_diverse(
                negative, per_class, minimum_gap)
        ]
        # A session without both classes cannot measure verification separation.
        if not any(label == 1 for label, _ in selected) or not any(
                label == 0 for label, _ in selected):
            continue
        for label, row in selected:
            result.append(CandidateSeed(
                session_id=str(row["session_id"]),
                scene=str(row["scene"]),
                episode=str(row["episode"]),
                kind=str(row["kind"]),
                query_path=Path(str(row["query_path"])).resolve(),
                candidate_path=Path(str(row["candidate_path"])).resolve(),
                candidate_frame=int(row["candidate_frame"]),
                dino_cosine=float(row["dino_cosine"]),
                teacher_covis=float(row["teacher_covis"]),
                label=label,
                session_has_positive=True,
                session_is_strict_no_match=False,
                session_max_covis=float(group["teacher_covis"].max()),
            ))
    return result


def select_deployment_seeds(
        frame: pd.DataFrame, *, kind: str, sessions: Sequence[str],
        max_sessions: int, top_k: int, minimum_gap: int,
        positive_threshold: float, negative_threshold: float,
        minimum_anchor: int) -> List[CandidateSeed]:
    """Select temporal-diverse top-DINO candidates, including no-match sets.

    Unlike the balanced feasibility sampler, this preserves the deployment
    question at set level. Sessions whose maximum co-visibility is below the
    negative threshold are tagged strict no-match; sessions that have neither a
    strict positive nor a strict no-match remain explicitly ambiguous.
    Candidate rows in the co-visibility ignore band receive ``label=-1`` but
    remain available to a future calibrated set model.
    """
    data = frame.loc[frame["kind"].eq(kind)].copy()
    if sessions:
        data = data.loc[data["session_id"].isin(sessions)]
    data = data.loc[
        data["teacher_covis"].notna()
        & data["dino_cosine"].notna()
        & data["candidate_frame"].ge(minimum_anchor)]
    session_ids = sorted(data["session_id"].unique().tolist())
    if max_sessions:
        session_ids = session_ids[:max_sessions]

    result: List[CandidateSeed] = []
    for session_id in session_ids:
        group = data.loc[data["session_id"].eq(session_id)]
        maximum_covisibility = float(group["teacher_covis"].max())
        has_positive = maximum_covisibility >= positive_threshold
        strict_no_match = maximum_covisibility <= negative_threshold
        for row in temporal_diverse(group, top_k, minimum_gap):
            covisibility = float(row["teacher_covis"])
            label = (1 if covisibility >= positive_threshold else
                     0 if covisibility <= negative_threshold else -1)
            result.append(CandidateSeed(
                session_id=str(row["session_id"]),
                scene=str(row["scene"]),
                episode=str(row["episode"]),
                kind=str(row["kind"]),
                query_path=Path(str(row["query_path"])).resolve(),
                candidate_path=Path(str(row["candidate_path"])).resolve(),
                candidate_frame=int(row["candidate_frame"]),
                dino_cosine=float(row["dino_cosine"]),
                teacher_covis=covisibility,
                label=label,
                session_has_positive=has_positive,
                session_is_strict_no_match=strict_no_match,
                session_max_covis=maximum_covisibility,
            ))
    return result


def validate_scene_role(seeds: Sequence[CandidateSeed], manifest: dict,
                        allowed_role: str) -> None:
    allowed_scenes = set(manifest.get(allowed_role, []))
    if not allowed_scenes:
        raise ValueError(
            f"split manifest has no scenes for role {allowed_role}")
    selected_scenes = {seed.scene for seed in seeds}
    leaked = selected_scenes - allowed_scenes
    if leaked:
        raise RuntimeError(
            f"selected scenes outside {allowed_role}: {sorted(leaked)}")


def feature_episode_root(feature_root: Path, seed: CandidateSeed) -> Path:
    # Feature roots used by the project either point at ``.../mp3d_2leg`` or at
    # its parent.  Resolve both layouts without writing symlinks.
    direct = feature_root / seed.scene / seed.episode
    nested = feature_root / "mp3d_2leg" / seed.scene / seed.episode
    for path in (direct, nested):
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        f"feature episode absent for {seed.scene}/{seed.episode} under "
        f"{feature_root}")


def raw_rgb_dir(seed: CandidateSeed) -> Path:
    # Query and candidate may come from different episodes.  LingBot replays
    # the candidate episode, so derive its RGB stream from candidate_path.
    path = seed.candidate_path.parent
    if path.is_dir():
        return path.resolve()
    raise FileNotFoundError(path)


def episode_root_from_image(image_path: Path) -> Path:
    """Return the episode root for a rendered goal or raw RGB frame."""
    image_path = Path(image_path)
    if image_path.name == "goal_image.jpg" or (
            image_path.stem.startswith("goal_")
            and image_path.suffix.lower() == ".jpg"):
        return image_path.parent
    for parent in image_path.parents:
        if parent.name == "videos":
            return parent.parent
    raise ValueError(f"cannot locate episode root for {image_path}")


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(
        value.tolist() if hasattr(value, "tolist") else value,
        dtype=np.float64)
    if array.size != 16:
        raise ValueError(f"{name} must contain 16 values, got {array.shape}")
    return array.reshape(4, 4)


def _resolve_generated_mount(extrinsic: np.ndarray,
                             frame_convention: str) -> np.ndarray:
    """Mirror the MemNav loader's legacy identity-mount compatibility fix."""
    result = np.asarray(extrinsic, dtype=np.float64).copy()
    if not str(frame_convention or "").startswith(
            "positions+parquet in data(Zup,M_W)"):
        return result
    rotation = result[:3, :3]
    if np.allclose(rotation, _HABITAT_TO_DATA_ROTATION, atol=1e-6):
        return result
    if not np.allclose(rotation, np.eye(3), atol=1e-6):
        raise ValueError(
            "generated Z-up episode has an unsupported camera mount")
    result[:3, :3] = _HABITAT_TO_DATA_ROTATION
    return result


def load_episode_pose_data(root: Path) -> EpisodePoseData:
    """Load the exact camera-to-world labels consumed by the NavDP loader."""
    root = Path(root)
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    metadata_path = root / "meta" / "gen_meta.json"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    frame = pd.read_parquet(parquet, columns=[
        "action", "observation.camera_extrinsic"])
    if frame.empty:
        raise ValueError(f"empty pose parquet: {parquet}")
    actions = np.stack([
        _matrix(value, "action") for value in frame["action"]
    ])
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    mount = _resolve_generated_mount(
        _matrix(frame.iloc[0]["observation.camera_extrinsic"],
                "camera extrinsic"),
        str(metadata.get("frame_convention", "")))
    return EpisodePoseData(
        actions=actions, base_extrinsic=mount, metadata=metadata)


def _yaw_habitat_to_data_rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    habitat = np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)
    return _HABITAT_TO_DATA_ROTATION @ habitat


def query_camera_to_world(
        query_path: Path,
        pose_cache: Dict[Path, EpisodePoseData]) -> np.ndarray:
    """Resolve a raw trajectory frame or rendered goal to data-frame c2w."""
    query_path = Path(query_path)
    root = episode_root_from_image(query_path).resolve()
    if root not in pose_cache:
        pose_cache[root] = load_episode_pose_data(root)
    episode = pose_cache[root]
    if query_path.name == "goal_image.jpg":
        goal_index = 0
    elif (query_path.stem.startswith("goal_")
          and query_path.suffix.lower() == ".jpg"):
        try:
            goal_index = int(query_path.stem.split("_", 1)[1]) - 1
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid goal image name: {query_path}") from error
    else:
        try:
            frame_index = int(query_path.stem)
        except ValueError as error:
            raise ValueError(f"invalid raw RGB frame name: {query_path}") from error
        if not 0 <= frame_index < len(episode.actions):
            raise IndexError(f"query frame outside trajectory: {query_path}")
        return episode.actions[frame_index].copy()

    goals = episode.metadata.get("goals", [])
    if not 0 <= goal_index < len(goals):
        raise IndexError(f"goal {goal_index + 1} absent from {root}")
    goal = goals[goal_index]
    position = np.asarray(goal.get("pos"), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"invalid goal position in {root}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _yaw_habitat_to_data_rotation(
        float(goal.get("yaw_habitat", 0.0)))
    result[:3, 3] = position
    return result


def navdp_ground_truth_relative(
        candidate_camera_to_world: np.ndarray,
        query_camera_to_world_pose: np.ndarray,
        base_extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return NavDP planar point-goal and relative camera rotation.

    This deliberately mirrors ``NavDP_Base_Dataset.relative_pose``: remove the
    fixed camera mount before reading base-frame forward/lateral translation,
    while the orientation diagnostic compares camera-to-camera rotations.
    """
    candidate = np.asarray(candidate_camera_to_world, dtype=np.float64)
    query = np.asarray(query_camera_to_world_pose, dtype=np.float64)
    mount = np.asarray(base_extrinsic, dtype=np.float64)
    for name, value in (("candidate", candidate), ("query", query),
                        ("base extrinsic", mount)):
        if value.shape != (4, 4):
            raise ValueError(f"{name} pose must be 4x4, got {value.shape}")
    base_rotation = candidate[:3, :3] @ np.linalg.inv(mount[:3, :3])
    local = base_rotation.T @ (query[:3, 3] - candidate[:3, 3])
    planar = np.array([local[1], -local[0]], dtype=np.float64)
    relative_rotation = candidate[:3, :3].T @ query[:3, :3]
    return planar, relative_rotation


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion is non-finite or degenerate")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def lingbot_relative_prediction(
        anchor_pose9: np.ndarray, goal_pose9: np.ndarray,
        metric_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decode LingBot relative translation/rotation in NavDP conventions."""
    anchor = np.asarray(anchor_pose9, dtype=np.float64)
    goal = np.asarray(goal_pose9, dtype=np.float64)
    if anchor.shape != (9,) or goal.shape != (9,):
        raise ValueError("LingBot pose encodings must each have shape (9,)")
    if not np.isfinite(metric_scale) or metric_scale <= 0.0:
        raise ValueError("metric scale must be finite and positive")
    anchor_rotation = quaternion_xyzw_to_matrix(anchor[3:7])
    goal_rotation = quaternion_xyzw_to_matrix(goal[3:7])
    translation = anchor_rotation.T @ (goal[:3] - anchor[:3])
    # LingBot's camera ground plane is x-z. NavDP point-goal is
    # [forward, lateral] = [LingBot z, -LingBot x].
    planar = float(metric_scale) * np.array(
        [translation[2], -translation[0]], dtype=np.float64)
    relative_rotation = anchor_rotation.T @ goal_rotation
    converted_rotation = (
        _LINGBOT_TO_DATA_ROTATION_BASIS
        @ relative_rotation
        @ _LINGBOT_TO_DATA_ROTATION_BASIS.T)
    return planar, converted_rotation


def rotation_error_degrees(predicted: np.ndarray,
                           target: np.ndarray) -> float:
    relative = np.asarray(predicted).T @ np.asarray(target)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_pose_errors(
        predicted_xy: np.ndarray, target_xy: np.ndarray,
        predicted_rotation: np.ndarray,
        target_rotation: np.ndarray) -> dict:
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    position_error = float(np.linalg.norm(predicted_xy - target_xy))
    predicted_norm = float(np.linalg.norm(predicted_xy))
    target_norm = float(np.linalg.norm(target_xy))
    if predicted_norm <= 1e-9 or target_norm <= 1e-9:
        direction_error = float("nan")
    else:
        cosine = np.clip(
            float(predicted_xy @ target_xy) / (predicted_norm * target_norm),
            -1.0, 1.0)
        direction_error = float(np.degrees(np.arccos(cosine)))
    return {
        "relative_position_error_m": position_error,
        "relative_position_direction_error_deg": direction_error,
        "relative_distance_error_m": abs(predicted_norm - target_norm),
        "relative_rotation_error_deg": rotation_error_degrees(
            predicted_rotation, target_rotation),
    }


def load_cache(lb, cache_path: Path, rgb_dir: Path, num_scale: int) -> dict:
    """Small standalone equivalent of MemNavNet._load_cache."""
    from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream

    cam_path = cache_path.with_name("lingbot_cam_cache.npz")
    with np.load(cache_path) as source, np.load(cam_path) as camera:
        cached = {name: source[name] for name in source.files}
        cam = {name: camera[name] for name in camera.files}
    layout = validate_cache_pair(
        cached, cam, expected_num_scale_frames=num_scale,
        require_versioned=False)
    if "scale_k" in cached and "scale_v" in cached:
        sk, sv, ak, av = LingBotStream._cache_to_layered(
            cached["scale_k"], cached["scale_v"],
            cached["anchor_k"], cached["anchor_v"], lb.device)
    else:
        sk, sv = lb.get_scale_kv(str(rgb_dir))
        ak = torch.as_tensor(
            cached["anchor_k"], device=lb.device,
            dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
        av = torch.as_tensor(
            cached["anchor_v"], device=lb.device,
            dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
    ck, cv = LingBotStream._cam_to_device(
        cam["cam_k"], cam["cam_v"], lb.device)
    result = {
        "scale_k": sk,
        "scale_v": sv,
        "anchor_k": ak,
        "anchor_v": av,
        "cam_k": ck,
        "cam_v": cv,
        "cam_pose_enc": torch.as_tensor(
            cam["cam_pose_enc"], device=lb.device, dtype=torch.float32),
    }
    if not layout.legacy_dense:
        result["anchor_frame_indices"] = torch.as_tensor(
            layout.anchor_frame_indices, dtype=torch.long)
        result["cam_frame_indices"] = torch.as_tensor(
            layout.cam_frame_indices, dtype=torch.long)
    return result


def quaternion_angle(q1: torch.Tensor, q2: torch.Tensor) -> float:
    q1 = torch.nn.functional.normalize(q1.float(), dim=-1)
    q2 = torch.nn.functional.normalize(q2.float(), dim=-1)
    cosine = torch.sum(q1 * q2).abs().clamp(0.0, 1.0)
    return float(2.0 * torch.acos(cosine))


@torch.no_grad()
def world_cloud(depth: torch.Tensor, confidence: torch.Tensor,
                pose9: torch.Tensor, *, pixel_stride: int,
                confidence_quantile: float, max_points: int) -> Tuple[torch.Tensor, float]:
    """Depth in camera coordinates -> confidence-filtered LingBot-map points."""
    from lingbot_map.utils.rotation import quat_to_mat

    depth = depth.float()
    confidence = confidence.float()
    pose9 = pose9.float()
    height, width = depth.shape
    d = depth[::pixel_stride, ::pixel_stride]
    c = confidence[::pixel_stride, ::pixel_stride]
    ys = torch.arange(
        0, height, pixel_stride, device=depth.device, dtype=torch.float32)
    xs = torch.arange(
        0, width, pixel_stride, device=depth.device, dtype=torch.float32)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    fy = (height / 2.0) / torch.tan(pose9[7] / 2.0)
    fx = (width / 2.0) / torch.tan(pose9[8] / 2.0)
    cam_x = (x - width / 2.0) * d / fx
    cam_y = (y - height / 2.0) * d / fy
    points = torch.stack([cam_x, cam_y, d], dim=-1)
    threshold = torch.quantile(c.reshape(-1), confidence_quantile)
    valid = torch.isfinite(d) & (d > 1e-6) & torch.isfinite(c) & (c >= threshold)
    points = points[valid]
    if points.shape[0] > max_points:
        indices = torch.linspace(
            0, points.shape[0] - 1, max_points,
            device=points.device).round().long()
        points = points[indices]
    rotation = quat_to_mat(torch.nn.functional.normalize(
        pose9[3:7], dim=-1))
    points = points @ rotation.transpose(0, 1) + pose9[:3]
    return points, float(c[valid].mean()) if valid.any() else float("nan")


@torch.no_grad()
def symmetric_cloud_overlap(first: torch.Tensor, second: torch.Tensor,
                            threshold: float) -> Tuple[float, float, float]:
    if not len(first) or not len(second):
        return float("nan"), float("nan"), float("nan")
    distance = torch.cdist(first, second)
    forward = float((distance.min(dim=1).values <= threshold).float().mean())
    backward = float((distance.min(dim=0).values <= threshold).float().mean())
    harmonic = 2.0 * forward * backward / max(forward + backward, 1e-12)
    return forward, backward, harmonic


@torch.no_grad()
def append_goal_at_anchor(lb, cache: dict, rgb_dir: Path,
                          goal_image: torch.Tensor, anchor: int, warm: int,
                          *, pixel_stride: int, confidence_quantile: float,
                          max_points: int, overlap_ratio: float) -> dict:
    """Append one goal and return geometry-native loop-closure measurements."""
    scale = lb.num_scale
    start = max(scale, anchor - warm + 1)
    indices = cache.get("anchor_frame_indices")
    if indices is None:
        lb._inject(
            cache["scale_k"], cache["scale_v"],
            cache["anchor_k"], cache["anchor_v"],
            n_hist=max(0, start - scale), total_frames=start)
    else:
        lb._inject(
            cache["scale_k"], cache["scale_v"],
            cache["anchor_k"], cache["anchor_v"],
            anchor_frame_indices=indices, raw_start=start)

    paths = [rgb_dir / f"{index}.jpg" for index in range(start, anchor + 1)]
    if not paths or not all(path.is_file() for path in paths):
        missing = next((path for path in paths if not path.is_file()), rgb_dir)
        raise FileNotFoundError(missing)
    warm_images = lb.load_images([str(path) for path in paths]).to(lb.device)
    candidate_agg = candidate_psi = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(len(warm_images)):
            candidate_agg, candidate_psi = lb.model._aggregate_features(
                warm_images[index:index + 1][None],
                num_frame_for_scale=scale, num_frame_per_block=1)
        candidate_depth = lb.model._predict_depth(
            candidate_agg, warm_images[-1:][None], candidate_psi)
        goal_agg, goal_psi = lb.model._aggregate_features(
            goal_image[None, None].to(lb.device),
            num_frame_for_scale=scale, num_frame_per_block=1)
        goal_depth = lb.model._predict_depth(
            goal_agg, goal_image[None, None].to(lb.device), goal_psi)

    lb._inject_camera(
        cache["cam_k"], cache["cam_v"], anchor + 1,
        cache.get("cam_frame_indices"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refinement = lb.model.camera_head(
            goal_agg, causal_inference=True,
            num_frame_per_block=1, num_frame_for_scale=scale)
    poses = [item[0, -1].float() for item in refinement]
    goal_pose = poses[-1]
    anchor_pose = cache["cam_pose_enc"][anchor]

    candidate_d = candidate_depth["depth"][0, -1, ..., 0].float()
    candidate_c = candidate_depth["depth_conf"][0, -1].float()
    goal_d = goal_depth["depth"][0, -1, ..., 0].float()
    goal_c = goal_depth["depth_conf"][0, -1].float()
    candidate_cloud, candidate_confidence = world_cloud(
        candidate_d, candidate_c, anchor_pose,
        pixel_stride=pixel_stride,
        confidence_quantile=confidence_quantile,
        max_points=max_points)
    goal_cloud, goal_confidence = world_cloud(
        goal_d, goal_c, goal_pose,
        pixel_stride=pixel_stride,
        confidence_quantile=confidence_quantile,
        max_points=max_points)
    depth_scale = float(torch.median(torch.cat([
        candidate_d[candidate_d > 1e-6].reshape(-1),
        goal_d[goal_d > 1e-6].reshape(-1),
    ])))
    overlap_threshold = max(1e-4, overlap_ratio * depth_scale)
    overlap_forward, overlap_backward, overlap_f1 = symmetric_cloud_overlap(
        candidate_cloud, goal_cloud, overlap_threshold)

    if len(poses) >= 2:
        refine_translation = float((poses[-1][:3] - poses[-2][:3]).norm())
        refine_rotation = quaternion_angle(poses[-1][3:7], poses[-2][3:7])
    else:
        refine_translation = float("nan")
        refine_rotation = float("nan")
    return {
        "anchor": int(anchor),
        "goal_pose": goal_pose.detach().cpu().numpy(),
        "anchor_goal_distance_raw": float((goal_pose[:3] - anchor_pose[:3]).norm()),
        "goal_refine_translation_raw": refine_translation,
        "goal_refine_rotation_deg": math.degrees(refine_rotation),
        "candidate_depth_confidence": candidate_confidence,
        "goal_depth_confidence": goal_confidence,
        "cloud_overlap_candidate_to_goal": overlap_forward,
        "cloud_overlap_goal_to_candidate": overlap_backward,
        "cloud_overlap_f1": overlap_f1,
        "overlap_threshold_raw": overlap_threshold,
        "depth_scale_raw": depth_scale,
    }


def pairwise_pose_dispersion(results: Sequence[dict]) -> Tuple[float, float]:
    if len(results) < 2:
        return float("nan"), float("nan")
    pose = [torch.from_numpy(result["goal_pose"]).float() for result in results]
    translation = []
    rotation = []
    for left in range(len(pose)):
        for right in range(left + 1, len(pose)):
            translation.append(float((pose[left][:3] - pose[right][:3]).norm()))
            rotation.append(math.degrees(quaternion_angle(
                pose[left][3:7], pose[right][3:7])))
    return float(np.median(translation)), float(np.median(rotation))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def jsonable_measurement(measurement: dict) -> dict:
    result = {}
    for key, value in measurement.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def auc_summary(rows: pd.DataFrame) -> Dict[str, dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    definitions = {
        "dino_cosine": ("dino_cosine", 1.0),
        "lingbot_cloud_overlap": ("cloud_overlap_f1_median", 1.0),
        "lingbot_pose_consistency": ("goal_pose_translation_dispersion_norm", -1.0),
        "lingbot_pose_refinement": ("goal_refine_translation_norm_median", -1.0),
    }
    labels = rows["label"].to_numpy(dtype=np.int64)
    result: Dict[str, dict] = {}
    for name, (column, direction) in definitions.items():
        values = rows[column].to_numpy(dtype=np.float64)
        # Ignore-band candidates are retained for a calibrated set model but
        # must not be silently coerced into either binary AUC class.
        valid = np.isfinite(values) & (labels >= 0)
        if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
            result[name] = {"n": int(valid.sum()), "roc_auc": None, "ap": None}
            continue
        score = direction * values[valid]
        result[name] = {
            "n": int(valid.sum()),
            "roc_auc": float(roc_auc_score(labels[valid], score)),
            "ap": float(average_precision_score(labels[valid], score)),
            "expected_direction": "higher" if direction > 0 else "lower",
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internnav-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "InternNav")
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path,
        help="optional scene-role manifest; required with --allowed-role")
    parser.add_argument(
        "--allowed-role", choices=("train", "development", "final_reserved"),
        help="fail if any selected session is outside this frozen scene role")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha", default="")
    parser.add_argument("--expected-lingbot-commit", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kind", default="revisit_b")
    parser.add_argument(
        "--selection-mode", choices=("balanced", "deployment"),
        default="balanced",
        help=("balanced: positive/negative feasibility pairs; deployment: "
              "temporal-diverse top-DINO sets including true no-match"))
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8,
                        help="candidate count per deployment-mode session")
    parser.add_argument("--candidate-min-gap", type=int, default=4)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.2)
    parser.add_argument("--neighbor-offset", type=int, action="append",
                        default=None,
                        help="repeatable; default: -4, 0, +4")
    parser.add_argument("--warm", type=int, default=64)
    parser.add_argument(
        "--full-replay", action="store_true",
        help=("replay every real frame from the scale block through each "
              "candidate, matching the online pose-only controller"))
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=10)
    parser.add_argument("--confidence-quantile", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=768)
    parser.add_argument("--overlap-ratio", type=float, default=0.025)
    parser.add_argument(
        "--pooled-metric-scale", type=float,
        default=_DEFAULT_POOLED_METRIC_SCALE,
        help="fallback LingBot-units-to-meters scale if ground recovery fails")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = tuple(sorted(set(args.neighbor_offset or (-4, 0, 4))))
    if 0 not in offsets:
        raise ValueError("neighbor offsets must include 0")
    if (args.per_class < 1 or args.top_k < 1 or args.max_sessions < 0
            or args.candidate_min_gap < 1 or args.warm < 1
            or args.num_scale < 1 or args.pixel_stride < 1
            or args.max_points < 16 or args.overlap_ratio <= 0.0
            or not np.isfinite(args.pooled_metric_scale)
            or args.pooled_metric_scale <= 0.0):
        raise ValueError("invalid diagnostic configuration")
    if not 0.0 <= args.negative_threshold < args.positive_threshold <= 1.0:
        raise ValueError("invalid co-visibility thresholds")
    for path in (args.internnav_root, args.teacher_csv, args.feature_root,
                 args.lingbot_repo, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    if bool(args.split_manifest) != bool(args.allowed_role):
        raise ValueError(
            "--split-manifest and --allowed-role must be provided together")
    if args.split_manifest and not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    sys.path.insert(0, str(args.internnav_root.resolve()))

    weight_sha = sha256(args.weights)
    lingbot_commit = git_value(args.lingbot_repo, "rev-parse", "HEAD")
    if args.expected_weight_sha and weight_sha != args.expected_weight_sha:
        raise RuntimeError(
            f"LingBot weight SHA mismatch: {weight_sha} != "
            f"{args.expected_weight_sha}")
    if (args.expected_lingbot_commit
            and lingbot_commit != args.expected_lingbot_commit):
        raise RuntimeError(
            f"LingBot commit mismatch: {lingbot_commit} != "
            f"{args.expected_lingbot_commit}")

    teacher = pd.read_csv(args.teacher_csv)
    missing = REQUIRED_COLUMNS - set(teacher.columns)
    if missing:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    selection_arguments = dict(
        kind=args.kind, sessions=args.session,
        max_sessions=args.max_sessions,
        minimum_gap=args.candidate_min_gap,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        minimum_anchor=args.num_scale)
    if args.selection_mode == "balanced":
        seeds = select_balanced_seeds(
            teacher, per_class=args.per_class, **selection_arguments)
    else:
        seeds = select_deployment_seeds(
            teacher, top_k=args.top_k, **selection_arguments)
    if not seeds:
        raise RuntimeError(
            f"no {args.selection_mode} candidate seeds selected")
    split_manifest_sha = None
    if args.split_manifest:
        with args.split_manifest.open(encoding="utf-8") as handle:
            split_manifest = json.load(handle)
        validate_scene_role(seeds, split_manifest, args.allowed_role)
        split_manifest_sha = sha256(args.split_manifest)
    for seed in seeds:
        if not seed.query_path.is_file():
            raise FileNotFoundError(seed.query_path)
        if not seed.candidate_path.is_file():
            raise FileNotFoundError(seed.candidate_path)

    # Validate every selected raw/cache dependency before allocating model
    # weights. This path is also invoked as a standalone Slurm preflight.
    from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair

    checked_episodes = set()
    pose_cache: Dict[Path, EpisodePoseData] = {}
    for seed in seeds:
        key = (seed.scene, seed.episode)
        if key not in checked_episodes:
            checked_episodes.add(key)
            episode_root = feature_episode_root(args.feature_root, seed)
            cache_path = (episode_root / "videos" / "chunk-000"
                          / "lingbot_cache.npz")
            cam_path = cache_path.with_name("lingbot_cam_cache.npz")
            for required in (cache_path, cam_path):
                if not required.exists():
                    raise FileNotFoundError(required)
            with np.load(cache_path) as cached, np.load(cam_path) as camera:
                validate_cache_pair(
                    cached, camera,
                    expected_num_scale_frames=args.num_scale,
                    require_versioned=False)
        candidate_root = episode_root_from_image(
            seed.candidate_path).resolve()
        if candidate_root not in pose_cache:
            pose_cache[candidate_root] = load_episode_pose_data(candidate_root)
        candidate_pose_data = pose_cache[candidate_root]
        if int(seed.candidate_path.stem) != seed.candidate_frame:
            raise ValueError(
                "candidate_frame disagrees with candidate filename: "
                f"{seed.candidate_path}")
        if not 0 <= seed.candidate_frame < len(candidate_pose_data.actions):
            raise IndexError(
                f"candidate frame outside trajectory: {seed.candidate_path}")
        query_camera_to_world(seed.query_path, pose_cache)
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_passed",
            "n_seeds": len(seeds),
            "n_episodes": len(checked_episodes),
            "n_pose_episodes": len(pose_cache),
            "selection_mode": args.selection_mode,
            "allowed_role": args.allowed_role,
            "split_manifest_sha256": split_manifest_sha,
            "lingbot_commit": lingbot_commit,
            "lingbot_weight_sha256": weight_sha,
            "teacher_csv_sha256": sha256(args.teacher_csv),
        }, indent=2, sort_keys=True))
        return

    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream

    started = time.time()
    lb = LingBotStream(
        lingbot_repo=str(args.lingbot_repo.resolve()),
        weights=str(args.weights.resolve()),
        num_scale=args.num_scale,
        window=args.window,
        max_frame_num=args.max_frame_num,
        camera_num_iterations=args.camera_num_iterations,
        device=args.device,
    ).eval()
    rows: List[dict] = []
    cache_by_episode: Dict[Tuple[str, str], dict] = {}
    metric_scale_by_episode: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for seed_index, seed in enumerate(seeds, 1):
        key = (seed.scene, seed.episode)
        episode_root = feature_episode_root(args.feature_root, seed)
        cache_path = episode_root / "videos" / "chunk-000" / "lingbot_cache.npz"
        rgb_dir = raw_rgb_dir(seed)
        if key not in cache_by_episode:
            cache_by_episode[key] = load_cache(
                lb, cache_path, rgb_dir, args.num_scale)
        cache = cache_by_episode[key]
        candidate_root = episode_root_from_image(
            seed.candidate_path).resolve()
        candidate_pose_data = pose_cache[candidate_root]
        query_pose = query_camera_to_world(seed.query_path, pose_cache)
        if key not in metric_scale_by_episode:
            camera_height = float(candidate_pose_data.metadata.get(
                "camera_height_m", 0.5))
            ground_scale = lb.get_metric_scale(
                str(rgb_dir), cache["cam_pose_enc"], camera_height)
            if (ground_scale is not None
                    and np.isfinite(ground_scale) and ground_scale > 0.0):
                metric_scale_by_episode[key] = (
                    float(ground_scale), "ground_anchored")
            else:
                metric_scale_by_episode[key] = (
                    float(args.pooled_metric_scale), "pooled_fallback")
        metric_scale, metric_scale_source = metric_scale_by_episode[key]
        goal = lb.load_images([str(seed.query_path)])[0].to(lb.device)
        maximum_anchor = min(
            len(cache["cam_pose_enc"]) - 2,
            len(candidate_pose_data.actions) - 1,
            max(int(path.stem) for path in rgb_dir.glob("*.jpg")
                if path.stem.isdigit()))
        hypotheses = []
        print(
            f"[{seed_index}/{len(seeds)}] {seed.session_id} "
            f"frame={seed.candidate_frame} label={seed.label} "
            f"covis={seed.teacher_covis:.3f}", flush=True)
        for offset in offsets:
            anchor = seed.candidate_frame + offset
            if not args.num_scale <= anchor <= maximum_anchor:
                continue
            replay_warm = (
                anchor - args.num_scale + 1
                if args.full_replay else args.warm
            )
            measurement = append_goal_at_anchor(
                lb, cache, rgb_dir, goal, anchor, replay_warm,
                pixel_stride=args.pixel_stride,
                confidence_quantile=args.confidence_quantile,
                max_points=args.max_points,
                overlap_ratio=args.overlap_ratio)
            measurement["offset"] = offset
            measurement["replay_frames"] = replay_warm
            anchor_pose9 = cache["cam_pose_enc"][anchor].detach().cpu().numpy()
            predicted_xy, predicted_rotation = lingbot_relative_prediction(
                anchor_pose9, measurement["goal_pose"], metric_scale)
            target_xy, target_rotation = navdp_ground_truth_relative(
                candidate_pose_data.actions[anchor], query_pose,
                candidate_pose_data.base_extrinsic)
            measurement.update(relative_pose_errors(
                predicted_xy, target_xy,
                predicted_rotation, target_rotation))
            measurement.update({
                "metric_scale_m_per_raw": metric_scale,
                "metric_scale_source": metric_scale_source,
                "predicted_relative_xy_m": predicted_xy,
                "target_relative_xy_m": target_xy,
                "target_relative_distance_m": float(np.linalg.norm(target_xy)),
            })
            hypotheses.append(measurement)
        if not hypotheses:
            continue
        translation_dispersion, rotation_dispersion = pairwise_pose_dispersion(
            hypotheses)
        depth_scale = finite_median(
            result["depth_scale_raw"] for result in hypotheses)
        norm = max(depth_scale, 1e-6)
        center = min(hypotheses, key=lambda result: abs(result["offset"]))
        rows.append({
            "session_id": seed.session_id,
            "scene": seed.scene,
            "episode": seed.episode,
            "kind": seed.kind,
            "query_path": str(seed.query_path),
            "candidate_path": str(seed.candidate_path),
            "candidate_frame": seed.candidate_frame,
            "label": seed.label,
            "session_has_positive": seed.session_has_positive,
            "session_is_strict_no_match": seed.session_is_strict_no_match,
            "session_max_covis": seed.session_max_covis,
            "teacher_covis": seed.teacher_covis,
            "dino_cosine": seed.dino_cosine,
            "metric_scale_m_per_raw": metric_scale,
            "metric_scale_source": metric_scale_source,
            "n_hypotheses": len(hypotheses),
            "neighbor_offsets": ";".join(str(item["offset"]) for item in hypotheses),
            "depth_scale_raw": depth_scale,
            "goal_pose_translation_dispersion_raw": translation_dispersion,
            "goal_pose_translation_dispersion_norm": translation_dispersion / norm,
            "goal_pose_rotation_dispersion_deg": rotation_dispersion,
            "cloud_overlap_f1_center": center["cloud_overlap_f1"],
            "cloud_overlap_f1_mean": finite_mean(
                item["cloud_overlap_f1"] for item in hypotheses),
            "cloud_overlap_f1_median": finite_median(
                item["cloud_overlap_f1"] for item in hypotheses),
            "anchor_goal_distance_norm_center": (
                center["anchor_goal_distance_raw"] / norm),
            "goal_refine_translation_norm_median": finite_median(
                item["goal_refine_translation_raw"] / max(
                    item["depth_scale_raw"], 1e-6) for item in hypotheses),
            "goal_refine_rotation_deg_median": finite_median(
                item["goal_refine_rotation_deg"] for item in hypotheses),
            "relative_position_error_m_center": center[
                "relative_position_error_m"],
            "relative_position_error_m_median": finite_median(
                item["relative_position_error_m"] for item in hypotheses),
            "relative_position_direction_error_deg_center": center[
                "relative_position_direction_error_deg"],
            "relative_position_direction_error_deg_median": finite_median(
                item["relative_position_direction_error_deg"]
                for item in hypotheses),
            "relative_distance_error_m_center": center[
                "relative_distance_error_m"],
            "relative_rotation_error_deg_center": center[
                "relative_rotation_error_deg"],
            "relative_rotation_error_deg_median": finite_median(
                item["relative_rotation_error_deg"] for item in hypotheses),
            "predicted_relative_xy_m_center_json": json.dumps(
                center["predicted_relative_xy_m"].tolist()),
            "target_relative_xy_m_center_json": json.dumps(
                center["target_relative_xy_m"].tolist()),
            "goal_pose9_center_json": json.dumps(
                center["goal_pose"].tolist()),
            "goal_depth_confidence_mean": finite_mean(
                item["goal_depth_confidence"] for item in hypotheses),
            "candidate_depth_confidence_mean": finite_mean(
                item["candidate_depth_confidence"] for item in hypotheses),
            "hypotheses_json": json.dumps([
                jsonable_measurement(item) for item in hypotheses
            ], sort_keys=True),
        })
    result_frame = pd.DataFrame(rows)
    if result_frame.empty:
        raise RuntimeError("all selected candidate seeds were skipped")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "lingbot_goal_loop_closure_rows.csv"
    json_path = args.out_dir / "diagnostic_lingbot_goal_loop_closure.json"
    result_frame.to_csv(csv_path, index=False)
    by_label = {}
    for label, name in ((0, "negative"), (1, "positive")):
        subset = result_frame.loc[result_frame["label"].eq(label)]
        by_label[name] = {
            "n": int(len(subset)),
            "dino_cosine_median": finite_median(subset["dino_cosine"]),
            "cloud_overlap_f1_median": finite_median(
                subset["cloud_overlap_f1_median"]),
            "pose_translation_dispersion_norm_median": finite_median(
                subset["goal_pose_translation_dispersion_norm"]),
            "pose_rotation_dispersion_deg_median": finite_median(
                subset["goal_pose_rotation_dispersion_deg"]),
            "goal_refine_translation_norm_median": finite_median(
                subset["goal_refine_translation_norm_median"]),
            "relative_position_error_m_median": finite_median(
                subset["relative_position_error_m_center"]),
            "relative_direction_error_deg_median": finite_median(
                subset["relative_position_direction_error_deg_center"]),
            "relative_rotation_error_deg_median": finite_median(
                subset["relative_rotation_error_deg_center"]),
        }
    session_rows = result_frame.sort_values(
        ["session_id", "dino_cosine"], ascending=[True, False])
    session_first = session_rows.drop_duplicates("session_id")
    positive_sessions = set(session_first.loc[
        session_first["session_has_positive"], "session_id"])
    selected_positive_sessions = set(result_frame.loc[
        result_frame["label"].eq(1), "session_id"])
    strict_no_match_sessions = set(session_first.loc[
        session_first["session_is_strict_no_match"], "session_id"])
    ambiguous_sessions = set(session_first["session_id"]) - (
        positive_sessions | strict_no_match_sessions)
    candidate_recall = (
        len(positive_sessions & selected_positive_sessions)
        / len(positive_sessions) if positive_sessions else float("nan"))
    report = {
        "status": "diagnostic_not_for_deployment",
        "objective": (
            "test whether LingBot-native pose consensus, point-cloud overlap, "
            "metric relative pose, and uncertainty can localize an ImageGoal "
            "without always invoking SIFT/RANSAC"),
        "limitations": ([
            "small deliberately balanced feasibility subset",
        ] if args.selection_mode == "balanced" else [
            "top-DINO deployment-style subset; no learned probability calibration yet",
        ]) + [
            "candidate labels come from task-aligned co-visibility teacher",
            "ground-truth pose errors are evaluation targets, not deployment inputs",
            "no threshold may be selected from final-reserved scenes",
            "closed-loop navigation is not measured here",
        ],
        "n_rows": int(len(result_frame)),
        "n_sessions": int(result_frame["session_id"].nunique()),
        "set_level": {
            "positive_sessions": len(positive_sessions),
            "strict_no_match_sessions": len(strict_no_match_sessions),
            "ambiguous_sessions": len(ambiguous_sessions),
            "positive_session_candidate_recall_at_selected_k": candidate_recall,
        },
        "by_label": by_label,
        "feature_separation": auc_summary(result_frame),
        "config": {
            "kind": args.kind,
            "selection_mode": args.selection_mode,
            "allowed_role": args.allowed_role,
            "sessions": args.session,
            "max_sessions": args.max_sessions,
            "per_class": args.per_class,
            "top_k": args.top_k,
            "candidate_min_gap": args.candidate_min_gap,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "neighbor_offsets": offsets,
            "warm": args.warm,
            "full_replay": args.full_replay,
            "num_scale": args.num_scale,
            "window": args.window,
            "camera_num_iterations": args.camera_num_iterations,
            "pixel_stride": args.pixel_stride,
            "confidence_quantile": args.confidence_quantile,
            "max_points": args.max_points,
            "overlap_ratio": args.overlap_ratio,
            "pooled_metric_scale": args.pooled_metric_scale,
        },
        "provenance": {
            "source_commit": git_value(Path(__file__).resolve().parents[1],
                                       "rev-parse", "HEAD"),
            "lingbot_commit": lingbot_commit,
            "lingbot_weight_sha256": weight_sha,
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_csv_sha256": sha256(args.teacher_csv),
            "split_manifest": (
                str(args.split_manifest.resolve())
                if args.split_manifest else None),
            "split_manifest_sha256": split_manifest_sha,
            "feature_root": str(args.feature_root.resolve()),
            "elapsed_seconds": time.time() - started,
        },
        "rows_csv": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
