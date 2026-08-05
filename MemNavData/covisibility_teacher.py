"""Task-aligned, occlusion-aware overlap labels for memory retrieval.

The synthetic MP3D episodes store metric camera poses and depth for every
trajectory frame.  Revisit goal images additionally store their precomputed
``covis_curve`` in ``meta/gen_meta.json``.  These signals let us label whether
the *goal surface* is visible from a memory frame instead of using generic
feature matches between unrelated background regions.

All geometry is evaluated offline.  The live router remains RGB-only.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def _matrix(value, name: str) -> np.ndarray:
    result = np.stack([np.asarray(row, dtype=np.float64) for row in value])
    if result.shape != (3, 3) and result.shape != (4, 4):
        raise ValueError(f"{name} must be 3x3 or 4x4, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def backproject_world(depth: np.ndarray, intrinsic: np.ndarray,
                      camera_to_world: np.ndarray, stride: int = 6,
                      depth_min: float = 0.15,
                      depth_max: float = 10.0) -> np.ndarray:
    """Backproject an OpenCV depth image into stored-world surface points."""
    depth = np.asarray(depth, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("depth must be a finite 2D array")
    if intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError("intrinsic/action shapes must be 3x3 and 4x4")
    if stride < 1 or not 0.0 <= depth_min < depth_max:
        raise ValueError("invalid backprojection parameters")

    height, width = depth.shape
    rows, columns = np.meshgrid(
        np.arange(0, height, stride), np.arange(0, width, stride),
        indexing="ij")
    distance = depth[rows, columns]
    valid = (distance > depth_min) & (distance < depth_max)
    columns = columns[valid].astype(np.float64)
    rows = rows[valid].astype(np.float64)
    distance = distance[valid]
    if not distance.size:
        return np.empty((0, 3), dtype=np.float64)

    # Stored actions retain Habitat/OpenGL camera axes: +X right, +Y up,
    # -Z forward.  Depth images use OpenCV +Y down, +Z forward.
    camera = np.stack([
        (columns - intrinsic[0, 2]) / intrinsic[0, 0] * distance,
        -(rows - intrinsic[1, 2]) / intrinsic[1, 1] * distance,
        -distance,
    ], axis=1)
    return (camera @ camera_to_world[:3, :3].T
            + camera_to_world[:3, 3])


def projected_covisibility(world_points: np.ndarray,
                           candidate_depth: np.ndarray,
                           candidate_intrinsic: np.ndarray,
                           candidate_camera_to_world: np.ndarray,
                           tolerance: float = 0.3) -> float:
    """Fraction of query surface visible and depth-consistent in a candidate."""
    points = np.asarray(world_points, dtype=np.float64)
    depth = np.asarray(candidate_depth, dtype=np.float64)
    intrinsic = np.asarray(candidate_intrinsic, dtype=np.float64)
    camera_to_world = np.asarray(
        candidate_camera_to_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("world_points must have shape [N, 3]")
    if depth.ndim != 2 or intrinsic.shape != (3, 3):
        raise ValueError("candidate depth/intrinsic shapes are invalid")
    if camera_to_world.shape != (4, 4) or tolerance <= 0.0:
        raise ValueError("candidate action/tolerance is invalid")
    if not len(points):
        return 0.0

    camera = ((points - camera_to_world[:3, 3])
              @ camera_to_world[:3, :3])
    x, y, z = camera[:, 0], -camera[:, 1], -camera[:, 2]
    visible = z > 0.05
    safe_z = np.maximum(z, 1e-6)
    columns = intrinsic[0, 0] * x / safe_z + intrinsic[0, 2]
    rows = intrinsic[1, 1] * y / safe_z + intrinsic[1, 2]
    height, width = depth.shape
    visible &= (
        (columns >= 0.0) & (columns < width - 1)
        & (rows >= 0.0) & (rows < height - 1))
    column_index = np.clip(columns.astype(np.int64), 0, width - 1)
    row_index = np.clip(rows.astype(np.int64), 0, height - 1)
    visible &= np.abs(
        z - depth[row_index, column_index]) <= tolerance
    return float(np.mean(visible))


def covisibility_label(score: float, positive_threshold: float = 0.5,
                       negative_threshold: float = 0.1) -> int:
    """Map overlap to 1/0/-1 (positive/negative/ambiguous ignore band)."""
    score = float(score)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("covisibility score must lie in [0, 1]")
    if not 0.0 <= negative_threshold < positive_threshold <= 1.0:
        raise ValueError("covisibility thresholds are invalid")
    if score >= positive_threshold:
        return 1
    if score <= negative_threshold:
        return 0
    return -1


def parse_path_maps(specifications: Sequence[str]
                    ) -> Tuple[Tuple[str, str], ...]:
    mappings = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"path map must be OLD=NEW, got {specification!r}")
        old, new = specification.split("=", 1)
        old, new = old.rstrip("/"), new.rstrip("/")
        if not old or not new:
            raise ValueError("path-map prefixes must be non-empty")
        mappings.append((old, new))
    return tuple(sorted(mappings, key=lambda item: len(item[0]), reverse=True))


def remap_path(raw: str, mappings: Sequence[Tuple[str, str]]) -> Path:
    for old, new in mappings:
        if raw == old or raw.startswith(old + "/"):
            return Path(new + raw[len(old):])
    return Path(raw)


def episode_root(image_path: Path) -> Path:
    image_path = Path(image_path)
    if image_path.name == "goal_image.jpg" or (
            image_path.stem.startswith("goal_")
            and image_path.suffix.lower() == ".jpg"):
        return image_path.parent
    for parent in image_path.parents:
        if parent.name == "videos":
            return parent.parent
    raise ValueError(f"cannot locate episode root for {image_path}")


def depth_path(rgb_path: Path) -> Path:
    parts = list(Path(rgb_path).parts)
    try:
        stream_index = parts.index("observation.images.rgb")
    except ValueError as error:
        raise ValueError(f"not a trajectory RGB path: {rgb_path}") from error
    parts[stream_index] = "observation.images.depth"
    return Path(*parts).with_suffix(".png")


class EpisodeCovisibilityCache:
    """Bounded I/O cache for repeated pairwise overlap evaluation."""

    def __init__(self, path_maps: Iterable[Tuple[str, str]] = (),
                 depth_cache_size: int = 512, stride: int = 6,
                 tolerance: float = 0.3) -> None:
        if depth_cache_size < 1 or stride < 1 or tolerance <= 0.0:
            raise ValueError("invalid covisibility cache configuration")
        self.path_maps = tuple(path_maps)
        self.depth_cache_size = int(depth_cache_size)
        self.stride = int(stride)
        self.tolerance = float(tolerance)
        self._episodes: Dict[str, Tuple[Tuple[np.ndarray, ...], np.ndarray]] = {}
        self._depths: OrderedDict[str, np.ndarray] = OrderedDict()
        self._world_points: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, dict] = {}

    def resolve(self, raw: str) -> Path:
        return remap_path(str(raw), self.path_maps)

    def _episode(self, root: Path) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        import pandas as pd

        key = str(root)
        if key not in self._episodes:
            parquet = (root / "data" / "chunk-000"
                       / "episode_000000.parquet")
            if not parquet.is_file():
                raise FileNotFoundError(parquet)
            frame = pd.read_parquet(parquet, columns=[
                "action", "observation.camera_intrinsic"])
            actions = tuple(_matrix(value, "action") for value in frame["action"])
            intrinsic = _matrix(
                frame.iloc[0]["observation.camera_intrinsic"], "intrinsic")
            self._episodes[key] = actions, intrinsic
        return self._episodes[key]

    def _depth(self, rgb_path: Path) -> np.ndarray:
        import cv2

        key = str(rgb_path)
        if key in self._depths:
            value = self._depths.pop(key)
            self._depths[key] = value
            return value
        path = depth_path(rgb_path)
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None:
            raise FileNotFoundError(path)
        if encoded.ndim != 2:
            raise ValueError(f"depth image is not single-channel: {path}")
        value = encoded.astype(np.float64) / 10000.0
        self._depths[key] = value
        while len(self._depths) > self.depth_cache_size:
            self._depths.popitem(last=False)
        return value

    def _points(self, query_path: Path) -> np.ndarray:
        key = str(query_path)
        if key not in self._world_points:
            root = episode_root(query_path)
            actions, intrinsic = self._episode(root)
            frame = int(query_path.stem)
            self._world_points[key] = backproject_world(
                self._depth(query_path), intrinsic, actions[frame],
                stride=self.stride)
        return self._world_points[key]

    def metadata_covisibility(self, query_path: Path,
                              candidate_path: Path,
                              candidate_frame: int) -> float:
        query_root = episode_root(query_path)
        if episode_root(candidate_path) != query_root:
            raise ValueError("stored goal covis_curve is episode-local")
        key = str(query_root)
        if key not in self._metadata:
            meta_path = query_root / "meta" / "gen_meta.json"
            with open(meta_path, encoding="utf-8") as handle:
                self._metadata[key] = json.load(handle)
        if query_path.name == "goal_image.jpg":
            goal_index = 0
        else:
            try:
                goal_index = int(query_path.stem.split("_", 1)[1]) - 1
            except (IndexError, ValueError) as error:
                raise ValueError(f"invalid goal image name: {query_path}") from error
        goals = self._metadata[key].get("goals", [])
        if not 0 <= goal_index < len(goals):
            raise IndexError(f"goal index absent from {query_root}")
        curve = goals[goal_index].get("covis_curve")
        if curve is None or not 0 <= candidate_frame < len(curve):
            raise IndexError(
                f"candidate frame {candidate_frame} absent from goal curve")
        score = float(curve[candidate_frame])
        if not 0.0 <= score <= 1.0:
            raise ValueError("stored covisibility is outside [0, 1]")
        return score

    def depth_covisibility(self, query_path: Path,
                           candidate_path: Path,
                           candidate_frame: int) -> float:
        candidate_root = episode_root(candidate_path)
        actions, intrinsic = self._episode(candidate_root)
        if int(candidate_path.stem) != int(candidate_frame):
            raise ValueError("candidate_frame disagrees with candidate filename")
        return projected_covisibility(
            self._points(query_path), self._depth(candidate_path), intrinsic,
            actions[int(candidate_frame)], tolerance=self.tolerance)

    def pair_covisibility(self, query_raw: str, candidate_raw: str,
                          candidate_frame: int) -> Tuple[float, str]:
        query_path = self.resolve(query_raw)
        candidate_path = self.resolve(candidate_raw)
        is_goal = query_path.name == "goal_image.jpg" or (
            query_path.stem.startswith("goal_")
            and query_path.suffix.lower() == ".jpg")
        if is_goal:
            return (
                self.metadata_covisibility(
                    query_path, candidate_path, int(candidate_frame)),
                "metadata_covis_curve",
            )
        return (
            self.depth_covisibility(
                query_path, candidate_path, int(candidate_frame)),
            "depth_reprojection",
        )
