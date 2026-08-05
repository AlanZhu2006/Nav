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
    candidate_frame: int
    dino_cosine: float
    teacher_covis: float
    label: int


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
                candidate_frame=int(row["candidate_frame"]),
                dino_cosine=float(row["dino_cosine"]),
                teacher_covis=float(row["teacher_covis"]),
                label=label,
            ))
    return result


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
    # candidate_path is .../observation.images.rgb/<frame>.jpg
    path = seed.query_path.parent / "videos" / "chunk-000" / "observation.images.rgb"
    if path.is_dir():
        return path.resolve()
    # Cross/session CSV layouts may put the query elsewhere; recover from a
    # candidate path in the same episode convention.
    episode = seed.query_path.parent
    raise FileNotFoundError(path if episode.is_dir() else seed.query_path)


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
        valid = np.isfinite(values)
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
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha", default="")
    parser.add_argument("--expected-lingbot-commit", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kind", default="revisit_b")
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--candidate-min-gap", type=int, default=4)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.2)
    parser.add_argument("--neighbor-offset", type=int, action="append",
                        default=None,
                        help="repeatable; default: -4, 0, +4")
    parser.add_argument("--warm", type=int, default=64)
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=10)
    parser.add_argument("--confidence-quantile", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=768)
    parser.add_argument("--overlap-ratio", type=float, default=0.025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = tuple(sorted(set(args.neighbor_offset or (-4, 0, 4))))
    if 0 not in offsets:
        raise ValueError("neighbor offsets must include 0")
    if (args.per_class < 1 or args.max_sessions < 0
            or args.candidate_min_gap < 1 or args.warm < 0
            or args.num_scale < 1 or args.pixel_stride < 1
            or args.max_points < 16 or args.overlap_ratio <= 0.0):
        raise ValueError("invalid diagnostic configuration")
    if not 0.0 <= args.negative_threshold < args.positive_threshold <= 1.0:
        raise ValueError("invalid co-visibility thresholds")
    for path in (args.internnav_root, args.teacher_csv, args.feature_root,
                 args.lingbot_repo, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
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
    seeds = select_balanced_seeds(
        teacher, kind=args.kind, sessions=args.session,
        max_sessions=args.max_sessions, per_class=args.per_class,
        minimum_gap=args.candidate_min_gap,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        minimum_anchor=args.num_scale)
    if not seeds:
        raise RuntimeError("no balanced candidate seeds selected")
    for seed in seeds:
        if not seed.query_path.is_file():
            raise FileNotFoundError(seed.query_path)

    # Validate every selected raw/cache dependency before allocating model
    # weights. This path is also invoked as a standalone Slurm preflight.
    from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair

    checked_episodes = set()
    for seed in seeds:
        key = (seed.scene, seed.episode)
        if key in checked_episodes:
            continue
        checked_episodes.add(key)
        episode_root = feature_episode_root(args.feature_root, seed)
        cache_path = episode_root / "videos" / "chunk-000" / "lingbot_cache.npz"
        cam_path = cache_path.with_name("lingbot_cam_cache.npz")
        rgb_dir = raw_rgb_dir(seed)
        for required in (cache_path, cam_path, rgb_dir / f"{seed.candidate_frame}.jpg"):
            if not required.exists():
                raise FileNotFoundError(required)
        with np.load(cache_path) as cached, np.load(cam_path) as camera:
            validate_cache_pair(
                cached, camera,
                expected_num_scale_frames=args.num_scale,
                require_versioned=False)
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_passed",
            "n_seeds": len(seeds),
            "n_episodes": len(checked_episodes),
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
    for seed_index, seed in enumerate(seeds, 1):
        key = (seed.scene, seed.episode)
        episode_root = feature_episode_root(args.feature_root, seed)
        cache_path = episode_root / "videos" / "chunk-000" / "lingbot_cache.npz"
        rgb_dir = raw_rgb_dir(seed)
        if key not in cache_by_episode:
            cache_by_episode[key] = load_cache(
                lb, cache_path, rgb_dir, args.num_scale)
        cache = cache_by_episode[key]
        goal = lb.load_images([str(seed.query_path)])[0].to(lb.device)
        maximum_anchor = min(
            len(cache["cam_pose_enc"]) - 2,
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
            measurement = append_goal_at_anchor(
                lb, cache, rgb_dir, goal, anchor, args.warm,
                pixel_stride=args.pixel_stride,
                confidence_quantile=args.confidence_quantile,
                max_points=args.max_points,
                overlap_ratio=args.overlap_ratio)
            measurement["offset"] = offset
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
            "candidate_frame": seed.candidate_frame,
            "label": seed.label,
            "teacher_covis": seed.teacher_covis,
            "dino_cosine": seed.dino_cosine,
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
            "goal_depth_confidence_mean": finite_mean(
                item["goal_depth_confidence"] for item in hypotheses),
            "candidate_depth_confidence_mean": finite_mean(
                item["candidate_depth_confidence"] for item in hypotheses),
            "hypotheses_json": json.dumps([
                {key: value for key, value in item.items() if key != "goal_pose"}
                for item in hypotheses], sort_keys=True),
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
        }
    report = {
        "status": "diagnostic_not_for_deployment",
        "objective": (
            "test whether LingBot-native pose consensus and point-cloud overlap "
            "can verify DINO candidates without SIFT/RANSAC"),
        "limitations": [
            "small deliberately balanced feasibility subset",
            "candidate labels come from task-aligned co-visibility teacher",
            "no threshold may be selected from final-reserved scenes",
            "closed-loop navigation is not measured here",
        ],
        "n_rows": int(len(result_frame)),
        "n_sessions": int(result_frame["session_id"].nunique()),
        "by_label": by_label,
        "feature_separation": auc_summary(result_frame),
        "config": {
            "kind": args.kind,
            "sessions": args.session,
            "max_sessions": args.max_sessions,
            "per_class": args.per_class,
            "candidate_min_gap": args.candidate_min_gap,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "neighbor_offsets": offsets,
            "warm": args.warm,
            "num_scale": args.num_scale,
            "window": args.window,
            "camera_num_iterations": args.camera_num_iterations,
            "pixel_stride": args.pixel_stride,
            "confidence_quantile": args.confidence_quantile,
            "max_points": args.max_points,
            "overlap_ratio": args.overlap_ratio,
        },
        "provenance": {
            "source_commit": git_value(Path(__file__).resolve().parents[1],
                                       "rev-parse", "HEAD"),
            "lingbot_commit": lingbot_commit,
            "lingbot_weight_sha256": weight_sha,
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_csv_sha256": sha256(args.teacher_csv),
            "feature_root": str(args.feature_root.resolve()),
            "elapsed_seconds": time.time() - started,
        },
        "rows_csv": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
