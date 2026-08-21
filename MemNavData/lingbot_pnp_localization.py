#!/usr/bin/env python3
"""Camera localization of an image goal in a LingBot history map.

LingBot already predicts a camera-to-world pose and dense depth for every
history frame.  This module turns local image correspondences into metric
2D--3D constraints: reference keypoints are lifted through the predicted
depth into the LingBot map, then PnP estimates the goal camera pose in that
same map.  Unlike dense cloud overlap or ICP, a successful solve must explain
specific image correspondences with one camera projection.

The implementation intentionally starts with OpenCV SIFT so the geometric
contract can be tested without adding a learned matcher or new weights.  A
stronger matcher can later feed the same ``solve_camera_pose_pnp`` function.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from MemNavData.lingbot_colored_registration import (
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )
except ModuleNotFoundError:  # direct script invocation
    from lingbot_colored_registration import (  # type: ignore
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )


@dataclass(frozen=True)
class SiftPnPConfig:
    """Frozen mechanics for local-feature camera relocalization."""

    nfeatures: int = 4000
    ratio_threshold: float = 0.75
    # Match the deployed SIFT verifier's one-way Lowe test.  PnP-RANSAC, not
    # an extra descriptor heuristic, is responsible for geometric rejection.
    mutual_ratio_test: bool = False
    # Sparse salient pixels should not be silently removed by an uncalibrated
    # dense-depth confidence threshold. Invalid/non-positive depth is retained
    # as the only pre-PnP depth veto in the primary configuration.
    depth_confidence_quantile: float = 0.0
    min_correspondences: int = 8
    min_inliers: int = 8
    reprojection_error_px: float = 3.0
    iterations: int = 1000
    confidence: float = 0.999
    rng_seed: int = 0

    def validate(self) -> None:
        if self.nfeatures < 32:
            raise ValueError("nfeatures must be at least 32")
        if not 0.0 < self.ratio_threshold < 1.0:
            raise ValueError("ratio_threshold must lie in (0, 1)")
        if not 0.0 <= self.depth_confidence_quantile <= 1.0:
            raise ValueError("depth confidence quantile must lie in [0, 1]")
        if self.min_correspondences < 6 or self.min_inliers < 6:
            raise ValueError("minimum correspondences/inliers must be at least 6")
        if self.reprojection_error_px <= 0.0 or self.iterations < 1:
            raise ValueError("PnP RANSAC settings must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")


def _as_numpy(value: Any, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def image_to_rgb_u8(image: Any) -> np.ndarray:
    """Convert a LingBot-style RGB tensor/array to contiguous HWC uint8."""
    if hasattr(image, "detach"):
        image = image.detach().float().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("image must have three dimensions")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError("image must be RGB with three channels")
    if not np.isfinite(array).all():
        raise ValueError("image contains non-finite values")
    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        if float(array.max(initial=0.0)) <= 1.5:
            array = array * 255.0
        array = np.clip(np.rint(array), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def intrinsics_from_pose9(pose9: Sequence[float], height: int,
                          width: int) -> np.ndarray:
    """Decode LingBot's [vertical FOV, horizontal FOV] camera intrinsics."""
    pose = _as_numpy(pose9)
    if pose.shape != (9,) or not np.isfinite(pose).all():
        raise ValueError("pose9 must be finite with shape (9,)")
    if height < 2 or width < 2:
        raise ValueError("image dimensions are invalid")
    fov_h, fov_w = float(pose[7]), float(pose[8])
    if not 0.0 < fov_h < math.pi or not 0.0 < fov_w < math.pi:
        raise ValueError("pose field of view must lie in (0, pi)")
    fy = (height / 2.0) / math.tan(fov_h / 2.0)
    fx = (width / 2.0) / math.tan(fov_w / 2.0)
    return np.array([
        [fx, 0.0, width / 2.0],
        [0.0, fy, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def map_raw_points_to_lingbot_pad(
        points_xy: np.ndarray, *, raw_height: int, raw_width: int,
        target_height: int, target_width: int,
        patch_size: int = 14) -> np.ndarray:
    """Map raw-image pixels through LingBot's deterministic ``mode=pad``.

    This mirrors ``load_and_preprocess_images`` without resampling the image,
    allowing feature matching at native resolution while sampling the aligned
    518-square LingBot depth prediction.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2]")
    if min(raw_height, raw_width, target_height, target_width, patch_size) < 1:
        raise ValueError("image dimensions and patch size must be positive")
    if target_height != target_width:
        raise ValueError("LingBot pad target must be square")
    target_size = target_width
    if raw_width >= raw_height:
        resized_width = target_size
        resized_height = round(
            raw_height * (resized_width / raw_width) / patch_size) * patch_size
    else:
        resized_height = target_size
        resized_width = round(
            raw_width * (resized_height / raw_height) / patch_size) * patch_size
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    mapped = points.copy()
    mapped[:, 0] = points[:, 0] * resized_width / raw_width + pad_left
    mapped[:, 1] = points[:, 1] * resized_height / raw_height + pad_top
    return mapped


def map_raw_intrinsic_to_lingbot_pad(
        intrinsic: np.ndarray, *, raw_height: int, raw_width: int,
        target_height: int, target_width: int,
        patch_size: int = 14) -> np.ndarray:
    """Map a raw-image camera matrix through LingBot's ``mode=pad``.

    Feature coordinates are transformed by
    :func:`map_raw_points_to_lingbot_pad` before PnP.  The query camera matrix
    must undergo the identical resize-and-pad transform; reusing the history
    camera matrix is valid only when both images came from the same camera.
    """
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    if min(raw_height, raw_width, target_height, target_width, patch_size) < 1:
        raise ValueError("image dimensions and patch size must be positive")
    if target_height != target_width:
        raise ValueError("LingBot pad target must be square")
    target_size = target_width
    if raw_width >= raw_height:
        resized_width = target_size
        resized_height = round(
            raw_height * (resized_width / raw_width) / patch_size) * patch_size
    else:
        resized_height = target_size
        resized_width = round(
            raw_width * (resized_height / raw_height) / patch_size) * patch_size
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    pixel_transform = np.array([
        [resized_width / raw_width, 0.0, float(pad_left)],
        [0.0, resized_height / raw_height, float(pad_top)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return pixel_transform @ intrinsic


def _coverage(points: np.ndarray, height: int, width: int) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull) / max(float(height * width), 1.0))


def _bilinear_depth(points_xy: np.ndarray, depth: np.ndarray,
                    confidence: np.ndarray,
                    confidence_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample aligned depth/confidence at floating point pixel locations."""
    height, width = depth.shape
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    valid_bounds = (
        (x >= 0.0) & (x <= width - 1.0)
        & (y >= 0.0) & (y <= height - 1.0))
    x0 = np.clip(np.floor(x).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0

    def interpolate(array: np.ndarray) -> np.ndarray:
        return (
            (1.0 - wx) * (1.0 - wy) * array[y0, x0]
            + wx * (1.0 - wy) * array[y0, x1]
            + (1.0 - wx) * wy * array[y1, x0]
            + wx * wy * array[y1, x1])

    sampled_depth = interpolate(depth)
    sampled_confidence = interpolate(confidence)
    valid = (
        valid_bounds & np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
        & np.isfinite(sampled_confidence)
        & (sampled_confidence >= confidence_threshold))
    return sampled_depth, valid


def lift_reference_keypoints(
        points_xy: np.ndarray, depth: Any, depth_confidence: Any,
        camera_to_world_pose9: Sequence[float],
        *, confidence_quantile: float) -> tuple[np.ndarray, np.ndarray]:
    """Lift reference pixels into LingBot world coordinates."""
    if hasattr(depth, "detach"):
        depth = depth.detach().float().cpu().numpy()
    if hasattr(depth_confidence, "detach"):
        depth_confidence = depth_confidence.detach().float().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float64)
    confidence = np.asarray(depth_confidence, dtype=np.float64)
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError("depth and confidence must be aligned 2-D arrays")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2]")
    finite_confidence = confidence[np.isfinite(confidence)]
    threshold = (float(np.quantile(finite_confidence, confidence_quantile))
                 if len(finite_confidence) else float("inf"))
    sampled_depth, valid = _bilinear_depth(
        points_xy, depth, confidence, threshold)
    height, width = depth.shape
    intrinsic = intrinsics_from_pose9(camera_to_world_pose9, height, width)
    x = (points_xy[:, 0] - intrinsic[0, 2]) * sampled_depth / intrinsic[0, 0]
    y = (points_xy[:, 1] - intrinsic[1, 2]) * sampled_depth / intrinsic[1, 1]
    camera_points = np.stack([x, y, sampled_depth], axis=1)
    pose = _as_numpy(camera_to_world_pose9)
    rotation = quaternion_xyzw_to_matrix(pose[3:7])
    world_points = camera_points @ rotation.T + pose[:3]
    valid &= np.isfinite(world_points).all(axis=1)
    return world_points, valid


def solve_camera_pose_pnp(
        world_points: np.ndarray, image_points: np.ndarray,
        intrinsic: np.ndarray, *, config: SiftPnPConfig,
        fov_pose9: Sequence[float]) -> dict:
    """Robustly estimate a camera-to-world LingBot pose from 2D--3D pairs."""
    config.validate()
    world_points = np.asarray(world_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if (world_points.ndim != 2 or world_points.shape[1] != 3
            or image_points.shape != (len(world_points), 2)):
        raise ValueError("world/image correspondences must be [N,3]/[N,2]")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    if len(world_points) < config.min_correspondences:
        return {
            "status": "insufficient_correspondences",
            "correspondences": int(len(world_points)),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }
    cv2.setRNGSeed(int(config.rng_seed))
    try:
        success, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
            world_points, image_points, intrinsic, None,
            iterationsCount=int(config.iterations),
            reprojectionError=float(config.reprojection_error_px),
            confidence=float(config.confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        return {
            "status": f"opencv_error:{type(exc).__name__}",
            "correspondences": int(len(world_points)),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }
    if not success or rvec is None or tvec is None or inlier_indices is None:
        return {
            "status": "ransac_failed",
            "correspondences": int(len(world_points)),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }
    inlier_indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
    if len(inlier_indices) >= 6:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                world_points[inlier_indices], image_points[inlier_indices],
                intrinsic, None, rvec, tvec)
        except cv2.error:
            pass
    world_to_camera, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    camera_to_world = world_to_camera.T
    camera_center = -camera_to_world @ translation
    projected, _ = cv2.projectPoints(
        world_points[inlier_indices], rvec, tvec, intrinsic, None)
    residual = projected.reshape(-1, 2) - image_points[inlier_indices]
    reprojection_rmse = float(np.sqrt(np.mean(np.square(residual))))
    camera_depth = (
        world_points[inlier_indices] @ world_to_camera.T + translation)[:, 2]
    cheirality = float(np.mean(camera_depth > 0.0))
    fov_pose = _as_numpy(fov_pose9)
    pose9 = np.concatenate([
        camera_center,
        matrix_to_quaternion_xyzw(camera_to_world),
        fov_pose[7:9],
    ])
    return {
        "status": ("ok" if len(inlier_indices) >= config.min_inliers
                   else "insufficient_inliers"),
        "correspondences": int(len(world_points)),
        "inliers": int(len(inlier_indices)),
        "inlier_ratio": float(len(inlier_indices) / len(world_points)),
        "reprojection_rmse_px": reprojection_rmse,
        "cheirality_fraction": cheirality,
        "pose9": pose9,
        "inlier_indices": inlier_indices,
    }


def _ratio_matches(first_descriptors: np.ndarray,
                   second_descriptors: np.ndarray,
                   threshold: float) -> list[cv2.DMatch]:
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        first_descriptors, second_descriptors, k=2)
    return [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < threshold * pair[1].distance
    ]


def correspondence_pnp_localize(
        reference_points: np.ndarray, query_points: np.ndarray,
        reference_depth: Any, reference_depth_confidence: Any,
        reference_pose9: Sequence[float], *, config: SiftPnPConfig,
        match_scores: np.ndarray | None = None,
        epipolar_threshold_px: float | None = None,
        query_intrinsic: np.ndarray | None = None) -> dict:
    """Localize from matcher-agnostic 2D correspondences and LingBot depth.

    An optional 2-D epipolar pre-filter is useful for learned matchers with
    many correspondences.  PnP remains the final metric consistency check.
    """
    config.validate()
    reference_points = np.asarray(reference_points, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    if (reference_points.ndim != 2 or reference_points.shape[1] != 2
            or query_points.shape != reference_points.shape):
        raise ValueError("matched points must be aligned [N,2] arrays")
    height, width = _as_numpy(reference_depth).shape[-2:]
    base = {
        "matches": int(len(reference_points)),
        "epipolar_inliers": int(len(reference_points)),
        "epipolar_inlier_ratio": (1.0 if len(reference_points) else 0.0),
        "depth_valid_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    if match_scores is not None:
        match_scores = np.asarray(match_scores, dtype=np.float64)
        if match_scores.shape != (len(reference_points),):
            raise ValueError("match_scores must have shape [N]")
        base.update({
            "match_score_mean": (
                float(match_scores.mean()) if len(match_scores) else 0.0),
            "match_score_median": (
                float(np.median(match_scores)) if len(match_scores) else 0.0),
        })
    if len(reference_points) < config.min_correspondences:
        return {**base, "status": "insufficient_matches"}

    if epipolar_threshold_px is not None:
        if epipolar_threshold_px <= 0.0:
            raise ValueError("epipolar threshold must be positive")
        cv2.setRNGSeed(int(config.rng_seed))
        try:
            _fundamental, mask = cv2.findFundamentalMat(
                reference_points.astype(np.float32),
                query_points.astype(np.float32), cv2.USAC_MAGSAC,
                float(epipolar_threshold_px), 0.999, 10000)
        except cv2.error:
            return {
                **base, "status": "epipolar_degenerate",
                "epipolar_inliers": 0, "epipolar_inlier_ratio": 0.0,
            }
        if mask is None:
            return {
                **base, "status": "epipolar_ransac_failed",
                "epipolar_inliers": 0, "epipolar_inlier_ratio": 0.0,
            }
        keep = np.asarray(mask).reshape(-1).astype(bool)
        base.update({
            "epipolar_inliers": int(keep.sum()),
            "epipolar_inlier_ratio": float(keep.mean()),
        })
        reference_points = reference_points[keep]
        query_points = query_points[keep]
        if match_scores is not None:
            match_scores = match_scores[keep]
        if len(reference_points) < config.min_correspondences:
            return {**base, "status": "insufficient_epipolar_inliers"}

    world_points, depth_valid = lift_reference_keypoints(
        reference_points, reference_depth, reference_depth_confidence,
        reference_pose9,
        confidence_quantile=config.depth_confidence_quantile)
    world_points = world_points[depth_valid]
    reference_points = reference_points[depth_valid]
    query_points = query_points[depth_valid]
    base["depth_valid_matches"] = int(len(world_points))
    base["reference_match_coverage"] = _coverage(
        reference_points, height, width)
    base["query_match_coverage"] = _coverage(
        query_points, height, width)
    if len(world_points) < config.min_correspondences:
        return {**base, "status": "insufficient_correspondences"}
    intrinsic = (
        intrinsics_from_pose9(reference_pose9, height, width)
        if query_intrinsic is None
        else np.asarray(query_intrinsic, dtype=np.float64)
    )
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("query_intrinsic must be a finite 3x3 matrix")
    solution = solve_camera_pose_pnp(
        world_points, query_points, intrinsic,
        config=config, fov_pose9=reference_pose9)
    result = {**base, **solution}
    if "pose9" in result:
        # Keep audit metrics even when a numerically solved pose does not meet
        # the frozen minimum-inlier validity contract.
        inliers = np.asarray(result.pop("inlier_indices"), dtype=np.int64)
        result["reference_inlier_coverage"] = _coverage(
            reference_points[inliers], height, width)
        result["query_inlier_coverage"] = _coverage(
            query_points[inliers], height, width)
        singular_values = np.linalg.svd(
            world_points[inliers] - world_points[inliers].mean(axis=0),
            compute_uv=False)
        result["world_inlier_spread_raw"] = float(singular_values[0])
        result["world_inlier_planarity"] = float(
            singular_values[-1] / max(singular_values[0], 1e-12))
    return result


class LightGluePointMatcher:
    """Lazy, pinned SuperPoint+LightGlue correspondence provider.

    Third-party imports happen only when this optional matcher is requested,
    so the core LingBot collector and unit tests retain their old dependency
    surface.
    """

    def __init__(self, repository: Path, *, dependency_root: Path | None,
                 device: str, max_keypoints: int = 2048,
                 reference_cache_size: int = 64):
        repository = Path(repository).resolve()
        if not repository.is_dir():
            raise FileNotFoundError(repository)
        if dependency_root is not None:
            dependency_root = Path(dependency_root).resolve()
            if not dependency_root.is_dir():
                raise FileNotFoundError(dependency_root)
            sys.path.insert(0, str(dependency_root))
        sys.path.insert(0, str(repository))
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd

        self.torch = torch
        self.rbd = rbd
        self.load_image = load_image
        self.device = torch.device(device)
        self.extractor = SuperPoint(
            max_num_keypoints=int(max_keypoints)).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.max_keypoints = int(max_keypoints)
        if int(reference_cache_size) < 0:
            raise ValueError("reference_cache_size must be non-negative")
        self.reference_cache_size = int(reference_cache_size)
        # Across a lifelong episode, different goals frequently retrieve the
        # same immutable history frame.  Cache only SuperPoint features and raw
        # dimensions, not matching results (which depend on the query).  The
        # file stat is part of the key so a reused path cannot return stale
        # features during development.
        self._reference_feature_cache = OrderedDict()
        self._query_key = None
        self._query_features = None
        self._path_query_key = None
        self._path_query = None

    def _extract(self, image: Any):
        image = image.detach().float().to(self.device)
        with self.torch.inference_mode():
            return self.extractor.extract(image)

    def _reference_features(self, path: Path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._reference_feature_cache.get(key)
        if cached is not None:
            self._reference_feature_cache.move_to_end(key)
            return cached
        image = self.load_image(str(path)).to(self.device)
        value = (
            int(image.shape[-2]), int(image.shape[-1]),
            self._extract(image),
        )
        if self.reference_cache_size:
            self._reference_feature_cache[key] = value
            self._reference_feature_cache.move_to_end(key)
            while (len(self._reference_feature_cache)
                   > self.reference_cache_size):
                self._reference_feature_cache.popitem(last=False)
        return value

    def match(self, reference_image: Any, query_image: Any) -> dict:
        reference_image = reference_image.detach().float().to(self.device)
        query_image = query_image.detach().float().to(self.device)
        query_key = (
            str(query_image.device), int(query_image.data_ptr()),
            tuple(query_image.shape))
        if query_key != self._query_key:
            self._query_features = self._extract(query_image)
            self._query_key = query_key
        reference_features = self._extract(reference_image)
        assert self._query_features is not None
        with self.torch.inference_mode():
            matches = self.matcher({
                "image0": reference_features,
                "image1": self._query_features,
            })
        feature0, feature1, matches = [
            self.rbd(item) for item in
            (reference_features, self._query_features, matches)
        ]
        indices = matches["matches"]
        points0 = feature0["keypoints"][indices[:, 0]]
        points1 = feature1["keypoints"][indices[:, 1]]
        return {
            "reference_points": points0.detach().cpu().numpy(),
            "query_points": points1.detach().cpu().numpy(),
            "scores": matches["scores"].detach().cpu().numpy(),
            "reference_keypoints": int(len(feature0["keypoints"])),
            "query_keypoints": int(len(feature1["keypoints"])),
        }

    def match_paths(self, reference_path: Path, query_path: Path, *,
                    target_height: int, target_width: int,
                    patch_size: int = 14) -> dict:
        """Match native-resolution RGBs and map points to LingBot pad pixels."""
        reference_path = Path(reference_path).resolve()
        query_path = Path(query_path).resolve()
        (reference_height, reference_width,
         reference_features) = self._reference_features(reference_path)
        query_key = str(query_path)
        if query_key != self._path_query_key:
            query_image = self.load_image(query_key).to(self.device)
            self._path_query = (query_image, self._extract(query_image))
            self._path_query_key = query_key
        assert self._path_query is not None
        query_image, query_features = self._path_query
        with self.torch.inference_mode():
            matches = self.matcher({
                "image0": reference_features,
                "image1": query_features,
            })
        feature0, feature1, matches = [
            self.rbd(item) for item in
            (reference_features, query_features, matches)
        ]
        indices = matches["matches"]
        raw_reference = feature0["keypoints"][indices[:, 0]].detach().cpu().numpy()
        raw_query = feature1["keypoints"][indices[:, 1]].detach().cpu().numpy()
        reference_points = map_raw_points_to_lingbot_pad(
            raw_reference,
            raw_height=reference_height,
            raw_width=reference_width,
            target_height=target_height, target_width=target_width,
            patch_size=patch_size)
        query_points = map_raw_points_to_lingbot_pad(
            raw_query,
            raw_height=int(query_image.shape[-2]),
            raw_width=int(query_image.shape[-1]),
            target_height=target_height, target_width=target_width,
            patch_size=patch_size)
        return {
            "reference_points": reference_points,
            "query_points": query_points,
            # The frozen candidate rank is defined in the native image
            # coordinates used by the independent static audit.  PnP uses the
            # mapped LingBot-pad coordinates above; exposing both prevents a
            # runtime implementation from quietly changing either contract.
            "reference_raw_points": raw_reference,
            "query_raw_points": raw_query,
            "scores": matches["scores"].detach().cpu().numpy(),
            "reference_keypoints": int(len(feature0["keypoints"])),
            "query_keypoints": int(len(feature1["keypoints"])),
            "coordinate_source": "native_rgb_to_lingbot_pad",
            "reference_raw_hw": [
                reference_height, reference_width],
            "query_raw_hw": [
                int(query_image.shape[-2]), int(query_image.shape[-1])],
        }


def sift_pnp_localize(
        reference_image: Any, reference_depth: Any,
        reference_depth_confidence: Any,
        reference_pose9: Sequence[float], query_image: Any,
        *, config: SiftPnPConfig | None = None,
        query_intrinsic: np.ndarray | None = None) -> dict:
    """Localize ``query_image`` in a LingBot map using one history view."""
    config = config or SiftPnPConfig()
    config.validate()
    reference_rgb = image_to_rgb_u8(reference_image)
    query_rgb = image_to_rgb_u8(query_image)
    if reference_rgb.shape[:2] != query_rgb.shape[:2]:
        raise ValueError("reference and query images must have equal dimensions")
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    query_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=int(config.nfeatures))
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_gray, None)
    query_keypoints, query_descriptors = sift.detectAndCompute(query_gray, None)
    base = {
        "reference_keypoints": int(len(reference_keypoints)),
        "query_keypoints": int(len(query_keypoints)),
        "ratio_matches": 0,
        "mutual_matches": 0,
        "depth_valid_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    if reference_descriptors is None or query_descriptors is None:
        return {**base, "status": "insufficient_features"}

    forward = _ratio_matches(
        query_descriptors, reference_descriptors, config.ratio_threshold)
    base["ratio_matches"] = int(len(forward))
    if config.mutual_ratio_test:
        reverse = _ratio_matches(
            reference_descriptors, query_descriptors, config.ratio_threshold)
        reverse_pairs = {(match.queryIdx, match.trainIdx) for match in reverse}
        matches = [
            match for match in forward
            if (match.trainIdx, match.queryIdx) in reverse_pairs
        ]
    else:
        matches = forward
    base["mutual_matches"] = int(len(matches))
    if len(matches) < config.min_correspondences:
        return {**base, "status": "insufficient_ratio_matches"}

    reference_points = np.asarray([
        reference_keypoints[match.trainIdx].pt for match in matches
    ], dtype=np.float64)
    query_points = np.asarray([
        query_keypoints[match.queryIdx].pt for match in matches
    ], dtype=np.float64)
    localization = correspondence_pnp_localize(
        reference_points, query_points, reference_depth,
        reference_depth_confidence, reference_pose9, config=config,
        query_intrinsic=query_intrinsic)
    return {**base, **localization}


def jsonable_pnp(result: dict) -> dict:
    """Convert NumPy values in a localization result to JSON primitives."""
    converted = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted
