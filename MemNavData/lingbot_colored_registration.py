#!/usr/bin/env python3
"""Registration helpers for LingBot goal-to-memory localization.

The existing MRC ``cloud_overlap`` feature evaluates two clouds at the pose
already predicted by LingBot.  This module deliberately answers a different
question: can a local registration solver *refine* that pose?  It keeps the
LingBot map frame, predicted depth, and goal-append pose as the initialization,
then runs a deterministic multi-scale ICP variant on confidence-filtered,
colored point clouds.

The functions are dependency-light at import time.  Open3D is imported only by
``multiscale_registration`` so geometry-free unit tests and existing collectors
remain usable when registration is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class RegistrationSchedule:
    """Depth-relative deterministic coarse-to-fine registration schedule."""

    voxel_ratios: tuple[float, ...] = (0.16, 0.08, 0.04)
    correspondence_ratios: tuple[float, ...] = (0.80, 0.40, 0.20)
    iterations: tuple[int, ...] = (60, 40, 25)
    lambda_geometric: float = 0.90

    def validate(self) -> None:
        lengths = {
            len(self.voxel_ratios),
            len(self.correspondence_ratios),
            len(self.iterations),
        }
        if lengths != {len(self.voxel_ratios)} or not self.voxel_ratios:
            raise ValueError("registration schedule fields must have equal non-zero length")
        if any(not np.isfinite(v) or v <= 0.0 for v in self.voxel_ratios):
            raise ValueError("voxel ratios must be finite and positive")
        if any(not np.isfinite(v) or v <= 0.0
               for v in self.correspondence_ratios):
            raise ValueError("correspondence ratios must be finite and positive")
        if any(int(v) < 1 for v in self.iterations):
            raise ValueError("registration iterations must be positive")
        if not np.isfinite(self.lambda_geometric) or not 0.0 <= self.lambda_geometric <= 1.0:
            raise ValueError("lambda_geometric must lie in [0, 1]")


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite xyzw with shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion is degenerate")
    x, y, z, w = quaternion / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a canonical-sign xyzw quaternion."""
    from scipy.spatial.transform import Rotation

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def pose9_to_matrix(pose9: Sequence[float]) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float64)
    if pose9.shape != (9,) or not np.isfinite(pose9).all():
        raise ValueError("LingBot pose must be finite with shape (9,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_xyzw_to_matrix(pose9[3:7])
    transform[:3, 3] = pose9[:3]
    return transform


def apply_world_delta_to_pose9(
        pose9: Sequence[float], world_delta: np.ndarray) -> np.ndarray:
    """Left-compose a world-frame registration correction with a LingBot pose."""
    pose9 = np.asarray(pose9, dtype=np.float64)
    world_delta = np.asarray(world_delta, dtype=np.float64)
    if world_delta.shape != (4, 4) or not np.isfinite(world_delta).all():
        raise ValueError("world delta must be a finite 4x4 matrix")
    refined = world_delta @ pose9_to_matrix(pose9)
    result = pose9.copy()
    result[:3] = refined[:3, 3]
    result[3:7] = matrix_to_quaternion_xyzw(refined[:3, :3])
    return result


@torch.no_grad()
def colored_world_cloud(
        depth: torch.Tensor,
        confidence: torch.Tensor,
        image: torch.Tensor,
        pose9: torch.Tensor,
        *,
        pixel_stride: int,
        confidence_quantile: float,
        max_points: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Back-project a LingBot RGB/depth prediction into its map frame.

    ``image`` must be the exact [0,1] RGB tensor used by LingBot after its
    resize/crop preprocessing.  Sampling is shared across XYZ and RGB, unlike
    the legacy overlap path which discards color.
    """
    from lingbot_map.utils.rotation import quat_to_mat

    if pixel_stride < 1 or max_points < 16:
        raise ValueError("pixel_stride must be positive and max_points >= 16")
    if not np.isfinite(confidence_quantile) or not 0.0 <= confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must lie in [0, 1]")
    depth = depth.float()
    confidence = confidence.float()
    image = image.float()
    pose9 = pose9.float()
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError("depth and confidence must be aligned 2-D tensors")
    if image.ndim != 3 or image.shape[0] != 3 or tuple(image.shape[1:]) != tuple(depth.shape):
        raise ValueError("image must have shape [3,H,W] aligned with depth")
    if pose9.shape != (9,):
        raise ValueError("pose9 must have shape (9,)")

    height, width = depth.shape
    d = depth[::pixel_stride, ::pixel_stride]
    c = confidence[::pixel_stride, ::pixel_stride]
    rgb = image[:, ::pixel_stride, ::pixel_stride].permute(1, 2, 0)
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
    valid = (
        torch.isfinite(d) & (d > 1e-6) & torch.isfinite(c)
        & (c >= threshold) & torch.isfinite(rgb).all(dim=-1)
    )
    points = points[valid]
    colors = rgb[valid].clamp(0.0, 1.0)
    if len(points) > max_points:
        indices = torch.linspace(
            0, len(points) - 1, max_points,
            device=points.device).round().long()
        points = points[indices]
        colors = colors[indices]
    rotation = quat_to_mat(torch.nn.functional.normalize(
        pose9[3:7], dim=-1))
    points = points @ rotation.transpose(0, 1) + pose9[:3]
    mean_confidence = float(c[valid].mean()) if valid.any() else float("nan")
    return (
        points.detach().cpu().numpy().astype(np.float64, copy=False),
        colors.detach().cpu().numpy().astype(np.float64, copy=False),
        mean_confidence,
    )


def _point_cloud(points: np.ndarray, colors: np.ndarray):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite [N,3]")
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError("colors must be finite and aligned [N,3]")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return cloud


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def multiscale_registration(
        source_points: np.ndarray,
        source_colors: np.ndarray,
        target_points: np.ndarray,
        target_colors: np.ndarray,
        *,
        depth_scale: float,
        method: str,
        schedule: RegistrationSchedule | None = None,
        init: np.ndarray | None = None,
) -> dict:
    """Refine source-to-target alignment with geometric or Colored ICP.

    Both clouds are expected in the LingBot map frame, so the default initial
    correction is identity.  The returned transform is a *world-frame delta*
    to left-compose with the LingBot goal pose.
    """
    import open3d as o3d

    if method not in {"geometric", "colored"}:
        raise ValueError("method must be 'geometric' or 'colored'")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    schedule = schedule or RegistrationSchedule()
    schedule.validate()
    transform = np.eye(4, dtype=np.float64) if init is None else np.asarray(
        init, dtype=np.float64).copy()
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("init must be a finite 4x4 matrix")
    source = _point_cloud(source_points, source_colors)
    target = _point_cloud(target_points, target_colors)
    if len(source.points) < 16 or len(target.points) < 16:
        return {
            "status": "insufficient_points",
            "method": method,
            "transform": transform,
            "fitness": 0.0,
            "inlier_rmse": float("inf"),
            "correspondences": 0,
        }

    level_receipts = []
    status = "ok"
    try:
        for voxel_ratio, correspondence_ratio, iterations in zip(
                schedule.voxel_ratios,
                schedule.correspondence_ratios,
                schedule.iterations):
            voxel = max(1e-5, float(voxel_ratio) * depth_scale)
            max_correspondence = max(
                2e-5, float(correspondence_ratio) * depth_scale)
            source_level = source.voxel_down_sample(voxel)
            target_level = target.voxel_down_sample(voxel)
            normal_radius = max(2.5 * voxel, max_correspondence * 0.5)
            search = o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=40)
            source_level.estimate_normals(search)
            target_level.estimate_normals(search)
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=int(iterations),
            )
            robust_scale = max(max_correspondence * 0.5, 1e-5)
            kernel = o3d.pipelines.registration.TukeyLoss(k=robust_scale)
            if method == "colored":
                estimation = (
                    o3d.pipelines.registration
                    .TransformationEstimationForColoredICP(
                        schedule.lambda_geometric, kernel))
                result = o3d.pipelines.registration.registration_colored_icp(
                    source_level, target_level, max_correspondence,
                    transform, estimation, criteria)
            else:
                estimation = (
                    o3d.pipelines.registration
                    .TransformationEstimationPointToPlane(kernel))
                result = o3d.pipelines.registration.registration_icp(
                    source_level, target_level, max_correspondence,
                    transform, estimation, criteria)
            transform = np.asarray(result.transformation, dtype=np.float64)
            level_receipts.append({
                "voxel": voxel,
                "max_correspondence": max_correspondence,
                "source_points": len(source_level.points),
                "target_points": len(target_level.points),
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "correspondences": len(result.correspondence_set),
            })
    except RuntimeError as exc:
        status = f"open3d_runtime_error:{type(exc).__name__}"

    final_distance = schedule.correspondence_ratios[-1] * depth_scale
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, final_distance, transform)
    delta_rotation = transform[:3, :3]
    result = {
        "status": status,
        "method": method,
        "transform": transform,
        "fitness": float(evaluation.fitness),
        "inlier_rmse": float(evaluation.inlier_rmse),
        "correspondences": len(evaluation.correspondence_set),
        "delta_translation_raw": float(np.linalg.norm(transform[:3, 3])),
        "delta_vertical_raw": float(abs(transform[1, 3])),
        "delta_rotation_deg": rotation_angle_degrees(delta_rotation),
        "levels": level_receipts,
    }
    return result


def jsonable_registration(result: dict) -> dict:
    """Convert NumPy-rich registration output to strict JSON primitives."""
    converted = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, np.generic):
            converted[key] = value.item()
        elif isinstance(value, list):
            converted[key] = [jsonable_registration(item) for item in value]
        else:
            converted[key] = value
    return converted
