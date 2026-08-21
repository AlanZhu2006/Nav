"""Sparse observed-space frontier map for Novel-goal diagnostics.

The planner represented here never receives the evaluation goal coordinate.
It converts only already observed RGB-D camera rays and a pose stream into a
2-D free/occupied grid, then ranks the boundary between observed free space
and unknown space.  Habitat or LingBot pose sources are deliberately kept
outside this module so they can be compared with identical map logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


Cell = tuple[int, int]


@dataclass(frozen=True)
class FrontierCandidate:
    """One deterministic representative of a connected frontier component."""

    world_xz: tuple[float, float]
    cell: Cell
    score: float
    boundary_m: float
    distance_m: float
    novelty_m: float
    component_cells: int

    def to_dict(self) -> dict:
        return {
            "world_xz": list(self.world_xz),
            "cell": list(self.cell),
            "score": float(self.score),
            "boundary_m": float(self.boundary_m),
            "distance_m": float(self.distance_m),
            "novelty_m": float(self.novelty_m),
            "component_cells": int(self.component_cells),
        }


@dataclass
class CoverageResidualTrigger:
    """Confirm repeated native proposals before enabling exploration help."""

    threshold_m: float = 0.60
    confirm_plans: int = 3
    streak: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold_m) or self.threshold_m <= 0:
            raise ValueError("threshold_m must be finite and positive")
        if self.confirm_plans < 1:
            raise ValueError("confirm_plans must be positive")
        if self.streak < 0:
            raise ValueError("streak must be non-negative")

    def observe(self, novelty_m: float | None) -> bool:
        """Update the consecutive-low-novelty count and return confirmation."""
        if novelty_m is None:
            self.streak = 0
        else:
            novelty = float(novelty_m)
            if math.isnan(novelty) or novelty < 0:
                raise ValueError("novelty_m must be non-negative and not NaN")
            if novelty < self.threshold_m:
                self.streak += 1
            else:
                self.streak = 0
        return self.streak >= self.confirm_plans

    def reset(self) -> None:
        self.streak = 0


def depth_endpoints_world(
    depth: np.ndarray,
    floor_position: Sequence[float],
    yaw: float,
    intrinsic: np.ndarray,
    *,
    camera_height_m: float = 0.5,
    pixel_stride: int = 12,
    min_depth_m: float = 0.10,
    max_depth_m: float = 5.0,
    obstacle_height_m: tuple[float, float] = (0.15, 1.50),
) -> tuple[np.ndarray, np.ndarray]:
    """Project sampled Habitat depth rays into world x-z endpoints.

    Returns ``(endpoint_xz, obstacle_mask)``.  Rays longer than
    ``max_depth_m`` are truncated and treated as free-space observations,
    whereas a real endpoint is occupied only when its height lies in the
    robot obstacle band.
    """
    image = np.asarray(depth, dtype=np.float64)
    position = np.asarray(floor_position, dtype=np.float64)
    camera = np.asarray(intrinsic, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("depth must be a non-empty 2-D array")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("floor_position must be a finite 3-vector")
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    if not math.isfinite(float(yaw)):
        raise ValueError("yaw must be finite")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError("depth bounds must satisfy 0 < min < max")
    low, high = map(float, obstacle_height_m)
    if not 0 <= low < high:
        raise ValueError("obstacle height band is invalid")

    height, width = image.shape
    vs, us = np.meshgrid(
        np.arange(0, height, pixel_stride),
        np.arange(0, width, pixel_stride),
        indexing="ij",
    )
    raw = image[vs, us]
    valid = np.isfinite(raw) & (raw > min_depth_m)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=bool)
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    raw = raw[valid]
    ray_depth = np.minimum(raw, max_depth_m)
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("intrinsic focal lengths must be positive")

    right = (u - cx) / fx * ray_depth
    up = -(v - cy) / fy * ray_depth
    sine, cosine = math.sin(float(yaw)), math.cos(float(yaw))
    # Habitat camera forward is (-sin(yaw), -cos(yaw)); camera right is
    # (+cos(yaw), -sin(yaw)).
    world_x = position[0] - ray_depth * sine + right * cosine
    world_z = position[2] - ray_depth * cosine - right * sine
    endpoint_y = position[1] + float(camera_height_m) + up
    real_surface = raw <= max_depth_m
    obstacle = (
        real_surface
        & (endpoint_y >= position[1] + low)
        & (endpoint_y <= position[1] + high)
    )
    return np.stack([world_x, world_z], axis=-1), obstacle


class ObservedFrontierGrid:
    """A small sparse occupancy grid built from observed depth rays only."""

    _NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    _NEIGHBORS_8 = tuple(
        (dx, dz)
        for dx in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dz
    )

    def __init__(
        self,
        *,
        resolution_m: float = 0.20,
        obstacle_clearance_m: float = 0.30,
        min_component_cells: int = 3,
        min_novelty_m: float = 0.60,
    ) -> None:
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive")
        if not math.isfinite(obstacle_clearance_m) or obstacle_clearance_m < 0:
            raise ValueError("obstacle_clearance_m must be finite and non-negative")
        if min_component_cells < 1:
            raise ValueError("min_component_cells must be positive")
        if not math.isfinite(min_novelty_m) or min_novelty_m < 0:
            raise ValueError("min_novelty_m must be finite and non-negative")
        self.resolution_m = float(resolution_m)
        self.obstacle_clearance_m = float(obstacle_clearance_m)
        self.min_component_cells = int(min_component_cells)
        self.min_novelty_m = float(min_novelty_m)
        self.free: set[Cell] = set()
        self.obstacle: set[Cell] = set()
        self.visited: set[Cell] = set()

    def world_to_cell(self, point_xz: Sequence[float]) -> Cell:
        point = np.asarray(point_xz, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("world point must be a finite 2-vector")
        return (
            int(math.floor(float(point[0]) / self.resolution_m)),
            int(math.floor(float(point[1]) / self.resolution_m)),
        )

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        return (
            (int(cell[0]) + 0.5) * self.resolution_m,
            (int(cell[1]) + 0.5) * self.resolution_m,
        )

    def _ray_cells(self, start: np.ndarray, end: np.ndarray) -> list[Cell]:
        distance = float(np.linalg.norm(end - start))
        samples = max(1, int(math.ceil(distance / (self.resolution_m * 0.45))))
        points = start[None] + np.linspace(0.0, 1.0, samples + 1)[:, None] * (
            end - start
        )[None]
        cells: list[Cell] = []
        for point in points:
            cell = self.world_to_cell(point)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells

    def integrate_rays(
        self,
        origin_xz: Sequence[float],
        endpoints_xz: np.ndarray,
        obstacle_mask: Sequence[bool],
    ) -> None:
        origin = np.asarray(origin_xz, dtype=np.float64)
        endpoints = np.asarray(endpoints_xz, dtype=np.float64)
        occupied = np.asarray(obstacle_mask, dtype=bool)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("ray origin must be a finite 2-vector")
        if endpoints.ndim != 2 or endpoints.shape[1] != 2:
            raise ValueError("ray endpoints must have shape [N, 2]")
        if len(endpoints) != len(occupied):
            raise ValueError("obstacle mask length differs from ray endpoints")
        if not np.isfinite(endpoints).all():
            raise ValueError("ray endpoints must be finite")
        origin_cell = self.world_to_cell(origin)
        self.visited.add(origin_cell)
        self.free.add(origin_cell)
        for endpoint, is_obstacle in zip(endpoints, occupied):
            cells = self._ray_cells(origin, endpoint)
            free_cells = cells[:-1] if is_obstacle and len(cells) > 1 else cells
            self.free.update(free_cells)
            if is_obstacle:
                cell = cells[-1]
                self.obstacle.add(cell)
                self.free.discard(cell)

    def integrate_depth(
        self,
        depth: np.ndarray,
        floor_position: Sequence[float],
        yaw: float,
        intrinsic: np.ndarray,
        **projection_kwargs,
    ) -> None:
        endpoints, occupied = depth_endpoints_world(
            depth,
            floor_position,
            yaw,
            intrinsic,
            **projection_kwargs,
        )
        position = np.asarray(floor_position, dtype=np.float64)
        self.integrate_rays(position[[0, 2]], endpoints, occupied)

    def _inflated_obstacles(self) -> set[Cell]:
        radius = int(math.ceil(
            self.obstacle_clearance_m / self.resolution_m))
        if radius == 0:
            return set(self.obstacle)
        inflated: set[Cell] = set()
        for ox, oz in self.obstacle:
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if math.hypot(dx, dz) <= radius + 1e-9:
                        inflated.add((ox + dx, oz + dz))
        return inflated

    def _frontier_cells(self) -> set[Cell]:
        blocked = self._inflated_obstacles()
        frontier: set[Cell] = set()
        known = self.free | self.obstacle
        for cell in self.free - blocked:
            if any(
                (cell[0] + dx, cell[1] + dz) not in known
                for dx, dz in self._NEIGHBORS_4
            ):
                frontier.add(cell)
        return frontier

    def _components(self, cells: set[Cell]) -> list[list[Cell]]:
        remaining = set(cells)
        components: list[list[Cell]] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            component = [seed]
            stack = [seed]
            while stack:
                current = stack.pop()
                for dx, dz in self._NEIGHBORS_8:
                    neighbor = (current[0] + dx, current[1] + dz)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components

    def ranked_frontiers(
        self,
        current_xz: Sequence[float],
        *,
        excluded_world_xz: Iterable[Sequence[float]] = (),
    ) -> list[FrontierCandidate]:
        """Return deterministic goal-blind frontier candidates by utility.

        Utility is expressed in metres: connected frontier boundary length and
        distance from the already visited trajectory are rewards, while travel
        distance is a cost.  This is deliberately a fixed feasibility policy,
        not a parameter sweep fitted to evaluation goals.
        """
        current = np.asarray(current_xz, dtype=np.float64)
        if current.shape != (2,) or not np.isfinite(current).all():
            raise ValueError("current_xz must be a finite 2-vector")
        excluded = [np.asarray(point, dtype=np.float64) for point in excluded_world_xz]
        if any(point.shape != (2,) or not np.isfinite(point).all()
               for point in excluded):
            raise ValueError("excluded points must be finite 2-vectors")
        visited_world = np.asarray(
            [self.cell_to_world(cell) for cell in self.visited],
            dtype=np.float64,
        )
        candidates: list[FrontierCandidate] = []
        for component in self._components(self._frontier_cells()):
            if len(component) < self.min_component_cells:
                continue
            cells_world = np.asarray(
                [self.cell_to_world(cell) for cell in component],
                dtype=np.float64,
            )
            centroid = cells_world.mean(axis=0)
            representative_index = int(np.argmin(
                np.linalg.norm(cells_world - centroid[None], axis=1)))
            point = cells_world[representative_index]
            if excluded and min(
                    float(np.linalg.norm(point - item)) for item in excluded
            ) < self.resolution_m * 2.0:
                continue
            distance_m = float(np.linalg.norm(point - current))
            novelty_m = (
                float(np.min(np.linalg.norm(
                    visited_world - point[None], axis=1)))
                if len(visited_world) else distance_m
            )
            if novelty_m < self.min_novelty_m:
                continue
            boundary_m = len(component) * self.resolution_m
            score = boundary_m + 0.5 * novelty_m - 0.35 * distance_m
            cell = component[representative_index]
            candidates.append(FrontierCandidate(
                world_xz=(float(point[0]), float(point[1])),
                cell=cell,
                score=float(score),
                boundary_m=float(boundary_m),
                distance_m=distance_m,
                novelty_m=novelty_m,
                component_cells=len(component),
            ))
        candidates.sort(key=lambda item: (
            -item.score,
            -item.novelty_m,
            item.distance_m,
            item.cell,
        ))
        return candidates

    def distance_to_visited(self, point_xz: Sequence[float]) -> float | None:
        """Metric distance from a point to the recorded camera trajectory."""
        point = np.asarray(point_xz, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("point_xz must be a finite 2-vector")
        if not self.visited:
            return None
        visited_world = np.asarray(
            [self.cell_to_world(cell) for cell in self.visited],
            dtype=np.float64,
        )
        return float(np.min(np.linalg.norm(
            visited_world - point[None], axis=1)))

    def summary(self) -> dict:
        frontier = self._frontier_cells()
        return {
            "free_cells": len(self.free),
            "obstacle_cells": len(self.obstacle),
            "visited_cells": len(self.visited),
            "frontier_cells": len(frontier),
            "resolution_m": self.resolution_m,
        }
