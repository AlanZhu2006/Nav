"""Causal, source-agnostic frontier proposals for NLSR-V2.

This module is deliberately narrower than a navigation policy.  It converts a
causal sequence of planar poses and depth rays into a sparse observed-space
map, proposes multi-scale free/unknown boundaries, and constructs a small
deployment-only shortlist.  It never imports Habitat and it never consumes a
goal position, geodesic distance, success bit, or rollout outcome.

Privileged proposal-reachability diagnostics live behind
``ProposalProxyLabeler`` and are attached *after* the proposal and shortlist
have been frozen.  This separation is intentional: a proxy label can audit
proposal coverage, but cannot influence candidate generation or ranking.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Protocol, Sequence

import numpy as np


SCHEMA_VERSION = "nlsr_v2_frontier_proposal_v1"
GRID_RESOLUTIONS_M = (0.15, 0.20, 0.30)
MAX_SHORTLIST = 6
SHORTLIST_SLOTS = (
    "goal_patch_top2",
    "topology_top2",
    "angular_diverse_top2",
)


class FrontierProposalError(ValueError):
    """Raised when a proposal input must fail closed."""


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def wrap_radians(angle: float) -> float:
    value = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    # Normalize negative zero for canonical artifacts.
    return 0.0 if value == 0.0 else value


def angular_distance(first: float, second: float) -> float:
    return abs(wrap_radians(float(first) - float(second)))


@dataclass(frozen=True)
class SE2Pose:
    """A planar pose whose yaw points along the local forward axis."""

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        values = (self.x_m, self.y_m, self.yaw_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise FrontierProposalError("SE(2) pose must be finite")

    def local_to_map(self, forward_m: float, left_m: float) -> tuple[float, float]:
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (
            self.x_m + cosine * float(forward_m) - sine * float(left_m),
            self.y_m + sine * float(forward_m) + cosine * float(left_m),
        )

    def map_to_local(self, x_m: float, y_m: float) -> tuple[float, float]:
        dx, dy = float(x_m) - self.x_m, float(y_m) - self.y_m
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return cosine * dx + sine * dy, -sine * dx + cosine * dy


@dataclass(frozen=True)
class PlanarScan:
    """One deployment-time planar depth scan in a map pose.

    ``ranges_m`` and ``bearings_rad`` are in the camera/base ground plane.
    ``hit`` distinguishes an observed surface endpoint from a max-range ray.
    """

    frame_index: int
    pose: SE2Pose
    ranges_m: tuple[float, ...]
    bearings_rad: tuple[float, ...]
    hit: tuple[bool, ...]

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or int(self.frame_index) != self.frame_index:
            raise FrontierProposalError("scan frame_index must be an integer")
        if self.frame_index < 0:
            raise FrontierProposalError("scan frame_index must be nonnegative")
        if not (len(self.ranges_m) == len(self.bearings_rad) == len(self.hit)):
            raise FrontierProposalError("scan ray arrays must have equal length")
        if not self.ranges_m:
            raise FrontierProposalError("scan must contain at least one ray")
        if any(type(value) is not bool for value in self.hit):
            raise FrontierProposalError("scan hit values must be strict booleans")
        for distance, bearing in zip(self.ranges_m, self.bearings_rad):
            if (not math.isfinite(float(distance)) or float(distance) <= 0.0
                    or not math.isfinite(float(bearing))):
                raise FrontierProposalError("scan rays must be finite and positive")


@dataclass(frozen=True)
class FrontierConfig:
    resolutions_m: tuple[float, ...] = GRID_RESOLUTIONS_M
    min_range_m: float = 0.20
    max_range_m: float = 8.0
    ray_step_fraction: float = 0.45
    minimum_component_cells: int = 2
    max_representatives_per_component: int = 3
    representative_spacing_m: float = 0.75
    minimum_candidate_distance_m: float = 0.45
    maximum_candidate_distance_m: float = 8.0
    occupied_clearance_radius_m: float = 2.0
    spatial_nms_m: float = 0.45
    bearing_nms_rad: float = math.radians(12.0)
    radial_nms_m: float = 0.75
    context_frames: int = 5
    context_view_half_angle_rad: float = math.radians(80.0)
    max_shortlist: int = MAX_SHORTLIST

    def __post_init__(self) -> None:
        if tuple(float(value) for value in self.resolutions_m) != GRID_RESOLUTIONS_M:
            raise FrontierProposalError(
                f"NLSR-V2 proposal resolutions must be {GRID_RESOLUTIONS_M}")
        positive = (
            self.min_range_m,
            self.max_range_m,
            self.ray_step_fraction,
            self.representative_spacing_m,
            self.minimum_candidate_distance_m,
            self.maximum_candidate_distance_m,
            self.occupied_clearance_radius_m,
            self.spatial_nms_m,
            self.bearing_nms_rad,
            self.radial_nms_m,
            self.context_view_half_angle_rad,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0
                   for value in positive):
            raise FrontierProposalError("frontier metric thresholds must be positive")
        if self.min_range_m >= self.max_range_m:
            raise FrontierProposalError("min_range_m must be below max_range_m")
        if self.minimum_candidate_distance_m >= self.maximum_candidate_distance_m:
            raise FrontierProposalError(
                "minimum candidate distance must be below maximum")
        integers = (
            self.minimum_component_cells,
            self.max_representatives_per_component,
            self.context_frames,
            self.max_shortlist,
        )
        if any(isinstance(value, bool) or int(value) != value or int(value) < 1
               for value in integers):
            raise FrontierProposalError("frontier count thresholds must be positive integers")
        if self.max_shortlist > MAX_SHORTLIST:
            raise FrontierProposalError(f"shortlist cannot exceed {MAX_SHORTLIST}")


@dataclass(frozen=True)
class ProxyMeasurement:
    reachable: bool
    progress_m: float

    def __post_init__(self) -> None:
        if type(self.reachable) is not bool:
            raise FrontierProposalError("proxy reachable must be a strict boolean")
        if not math.isfinite(float(self.progress_m)):
            raise FrontierProposalError("proxy progress must be finite")
        if not self.reachable and float(self.progress_m) != 0.0:
            raise FrontierProposalError(
                "unreachable proxy measurement must have zero progress")


class ProposalProxyLabeler(Protocol):
    """Privileged, audit-only proposal labeler (for example Pathfinder).

    The labeler receives an already-frozen proposal candidate.  Implementations
    may use GT state internally, but their output is never copied into candidate
    features and never changes the shortlist.
    """

    def provenance(self) -> Mapping[str, object]:
        ...

    def label(self, *, sample_id: str, arm: str,
              candidate: Mapping[str, object]) -> ProxyMeasurement:
        ...


class SparseRayGrid:
    """Sparse free/occupied evidence from planar depth-ray carving."""

    _NEIGHBORS_8 = tuple(
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    )

    def __init__(self, resolution_m: float, ray_step_fraction: float = 0.45):
        self.resolution_m = float(resolution_m)
        self.ray_step_fraction = float(ray_step_fraction)
        if (not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0
                or not math.isfinite(self.ray_step_fraction)
                or not 0.0 < self.ray_step_fraction <= 1.0):
            raise FrontierProposalError("invalid sparse-grid resolution or step")
        self.free_votes: dict[tuple[int, int], int] = {}
        self.occupied_votes: dict[tuple[int, int], int] = {}
        self.trace: list[tuple[float, float, int, float]] = []

    def cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (
            math.floor(float(x_m) / self.resolution_m),
            math.floor(float(y_m) / self.resolution_m),
        )

    def center(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            (cell[0] + 0.5) * self.resolution_m,
            (cell[1] + 0.5) * self.resolution_m,
        )

    @staticmethod
    def _vote(table: dict[tuple[int, int], int], cell: tuple[int, int]) -> None:
        table[cell] = table.get(cell, 0) + 1

    def integrate(self, scan: PlanarScan) -> None:
        self.trace.append((
            float(scan.pose.x_m),
            float(scan.pose.y_m),
            int(scan.frame_index),
            float(scan.pose.yaw_rad),
        ))
        step = self.resolution_m * self.ray_step_fraction
        for distance, bearing, hit in zip(
                scan.ranges_m, scan.bearings_rad, scan.hit):
            ray_distance = float(distance)
            endpoint_margin = 0.55 * self.resolution_m if hit else 0.0
            free_end = max(0.0, ray_distance - endpoint_margin)
            sample_count = max(1, int(math.ceil(free_end / step)))
            for value in np.linspace(0.0, free_end, sample_count + 1):
                forward = float(value) * math.cos(float(bearing))
                left = float(value) * math.sin(float(bearing))
                self._vote(
                    self.free_votes,
                    self.cell(*scan.pose.local_to_map(forward, left)),
                )
            if hit:
                forward = ray_distance * math.cos(float(bearing))
                left = ray_distance * math.sin(float(bearing))
                self._vote(
                    self.occupied_votes,
                    self.cell(*scan.pose.local_to_map(forward, left)),
                )

    def free_cells(self) -> set[tuple[int, int]]:
        # A surface endpoint wins over a free ray only when its evidence is at
        # least as strong.  Repeated later traversals can correctly clear one
        # isolated noisy endpoint.
        return {
            cell for cell, votes in self.free_votes.items()
            if votes > self.occupied_votes.get(cell, 0)
        }

    def occupied_cells(self) -> set[tuple[int, int]]:
        return {
            cell for cell, votes in self.occupied_votes.items()
            if votes >= self.free_votes.get(cell, 0)
        }

    def frontier_cells(self) -> set[tuple[int, int]]:
        free = self.free_cells()
        occupied = self.occupied_cells()
        observed = free | occupied
        return {
            cell for cell in free
            if any((cell[0] + dx, cell[1] + dy) not in observed
                   for dx, dy in self._NEIGHBORS_8)
        }

    def frontier_components(self) -> list[tuple[tuple[int, int], ...]]:
        remaining = set(self.frontier_cells())
        components: list[tuple[tuple[int, int], ...]] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            queue = [seed]
            component = []
            while queue:
                cell = queue.pop()
                component.append(cell)
                for dx, dy in self._NEIGHBORS_8:
                    neighbor = cell[0] + dx, cell[1] + dy
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
            components.append(tuple(sorted(component)))
        return sorted(components, key=lambda value: (value[0], len(value)))

    def clearance_m(self, cell: tuple[int, int], maximum_m: float) -> float:
        occupied = self.occupied_cells()
        if not occupied:
            return float(maximum_m)
        x_m, y_m = self.center(cell)
        minimum = min(
            math.hypot(x_m - self.center(other)[0],
                       y_m - self.center(other)[1])
            for other in occupied
        )
        return min(float(maximum_m), float(minimum))


def depth_to_planar_scan(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    pose: SE2Pose,
    frame_index: int,
    *,
    valid_mask: np.ndarray | None = None,
    truncated_mask: np.ndarray | None = None,
    min_range_m: float = 0.20,
    max_range_m: float = 8.0,
    column_stride: int = 8,
    row_fraction: tuple[float, float] = (0.35, 0.68),
) -> PlanarScan:
    """Reduce a pinhole depth image to robust horizontal planar rays.

    The per-column 40th percentile over a central vertical band suppresses
    isolated floor/ceiling pixels while retaining walls and furniture.  The
    result uses only depth and intrinsics available at deployment.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    camera = np.asarray(intrinsic, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise FrontierProposalError("depth must be an HxW array")
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise FrontierProposalError("camera intrinsic must be finite 3x3")
    fx, cx = float(camera[0, 0]), float(camera[0, 2])
    if fx <= 0.0 or not math.isfinite(fx) or not math.isfinite(cx):
        raise FrontierProposalError("camera fx/cx are invalid")
    if (isinstance(column_stride, bool) or int(column_stride) != column_stride
            or column_stride < 1):
        raise FrontierProposalError("column_stride must be a positive integer")
    low, high = map(float, row_fraction)
    if not 0.0 <= low < high <= 1.0:
        raise FrontierProposalError("row_fraction must lie inside [0,1]")
    height, width = depth.shape
    first = min(height - 1, max(0, int(math.floor(low * height))))
    last = min(height, max(first + 1, int(math.ceil(high * height))))
    columns = np.arange(0, width, int(column_stride), dtype=np.int64)
    if columns[-1] != width - 1:
        columns = np.append(columns, width - 1)
    if valid_mask is None:
        valid_full = np.isfinite(depth) & (depth >= float(min_range_m))
    else:
        valid_full = np.asarray(valid_mask)
        if valid_full.shape != depth.shape or valid_full.dtype.kind != "b":
            raise FrontierProposalError(
                "valid_mask must be a boolean array matching depth")
        valid_full = valid_full & np.isfinite(depth) & (depth >= float(min_range_m))
    if truncated_mask is None:
        truncated_full = np.zeros(depth.shape, dtype=bool)
    else:
        truncated_full = np.asarray(truncated_mask)
        if (truncated_full.shape != depth.shape
                or truncated_full.dtype.kind != "b"):
            raise FrontierProposalError(
                "truncated_mask must be a boolean array matching depth")
        if np.any(truncated_full & ~valid_full):
            raise FrontierProposalError(
                "truncated depth pixels must also be marked valid")
    band = depth[first:last, :][:, columns]
    valid = valid_full[first:last, :][:, columns]
    truncated = truncated_full[first:last, :][:, columns]
    ranges, bearings, hits = [], [], []
    for index, column in enumerate(columns.tolist()):
        column_valid = valid[:, index]
        if not np.any(column_valid):
            # Invalid/zero depth says nothing about free space.  Skipping the
            # ray is conservative; converting it into max-range free space
            # would manufacture observations.
            continue
        values = band[:, index][column_valid]
        z_m = float(np.percentile(values, 40.0))
        column_truncated = truncated[:, index][column_valid]
        if np.any(column_truncated):
            truncation_floor = float(np.min(values[column_truncated]))
            surface = z_m < truncation_floor - max(1e-6, 1e-5 * truncation_floor)
        else:
            surface = True
        surface = bool(surface and z_m < float(max_range_m))
        z_m = min(float(z_m), float(max_range_m))
        # Optical +x points right, while the navigation convention is
        # [forward, left], hence left = -(u-cx)*z/fx.
        left_m = -(float(column) - cx) * z_m / fx
        distance = math.hypot(z_m, left_m)
        distance = min(float(max_range_m), max(float(min_range_m), distance))
        bearing = math.atan2(left_m, z_m)
        ranges.append(distance)
        bearings.append(bearing)
        hits.append(bool(surface and distance < float(max_range_m)))
    if not ranges:
        raise FrontierProposalError(
            "depth image has no valid planar rays in the selected row band")
    return PlanarScan(
        frame_index=int(frame_index),
        pose=pose,
        ranges_m=tuple(ranges),
        bearings_rad=tuple(bearings),
        hit=tuple(hits),
    )


def _component_representatives(
    grid: SparseRayGrid,
    component: Sequence[tuple[int, int]],
    config: FrontierConfig,
) -> list[tuple[int, int]]:
    centers = {cell: grid.center(cell) for cell in component}
    mean = np.mean(np.asarray(list(centers.values()), dtype=np.float64), axis=0)
    first = min(
        component,
        key=lambda cell: (
            math.hypot(centers[cell][0] - float(mean[0]),
                       centers[cell][1] - float(mean[1])),
            cell,
        ),
    )
    selected = [first]
    while len(selected) < config.max_representatives_per_component:
        scored = []
        for cell in component:
            if cell in selected:
                continue
            distance = min(
                math.hypot(centers[cell][0] - centers[chosen][0],
                           centers[cell][1] - centers[chosen][1])
                for chosen in selected
            )
            scored.append((distance, cell))
        if not scored:
            break
        distance, candidate = max(scored, key=lambda row: (row[0], tuple(-v for v in row[1])))
        if distance < config.representative_spacing_m:
            break
        selected.append(candidate)
    return selected


def _nearest_trace_context(
    grid: SparseRayGrid,
    x_m: float,
    y_m: float,
    config: FrontierConfig,
) -> list[int]:
    visible, fallback = [], []
    for trace_x, trace_y, frame, yaw in grid.trace:
        distance = math.hypot(x_m - trace_x, y_m - trace_y)
        direction = math.atan2(y_m - trace_y, x_m - trace_x)
        row = (distance, int(frame))
        fallback.append(row)
        if angular_distance(direction, yaw) <= config.context_view_half_angle_rad:
            visible.append(row)
    ordered = sorted(visible or fallback)
    return [frame for _distance, frame in ordered[:config.context_frames]]


def _candidate_from_cell(
    grid: SparseRayGrid,
    cell: tuple[int, int],
    component: Sequence[tuple[int, int]],
    current_pose: SE2Pose,
    config: FrontierConfig,
) -> dict | None:
    x_m, y_m = grid.center(cell)
    forward_m, left_m = current_pose.map_to_local(x_m, y_m)
    distance_m = math.hypot(forward_m, left_m)
    if not (config.minimum_candidate_distance_m
            <= distance_m <= config.maximum_candidate_distance_m):
        return None
    bearing = math.atan2(left_m, forward_m)
    observed = grid.free_cells() | grid.occupied_cells()
    unknown_offsets = [
        (dx, dy) for dx, dy in grid._NEIGHBORS_8
        if (cell[0] + dx, cell[1] + dy) not in observed
    ]
    if unknown_offsets:
        normal_x = float(np.mean([value[0] for value in unknown_offsets]))
        normal_y = float(np.mean([value[1] for value in unknown_offsets]))
        normal_map = math.atan2(normal_y, normal_x)
        normal_bearing = wrap_radians(normal_map - current_pose.yaw_rad)
    else:  # Defensive; frontier_cells guarantees at least one unknown neighbor.
        normal_bearing = bearing
    trace_distance = min(
        math.hypot(x_m - trace_x, y_m - trace_y)
        for trace_x, trace_y, _frame, _yaw in grid.trace
    )
    boundary_m = len(component) * grid.resolution_m
    clearance_m = grid.clearance_m(
        cell, config.occupied_clearance_radius_m)
    # This score is goal-blind and consists only of observed topology.  It is
    # a shortlist coverage heuristic, not a claim of goal progress.
    topology_score = (
        math.log1p(boundary_m / grid.resolution_m)
        + 0.30 * min(trace_distance, 4.0)
        + 0.15 * min(clearance_m, 2.0)
    )
    candidate_id = (
        f"r{int(round(grid.resolution_m * 100)):02d}_"
        f"{cell[0]:+06d}_{cell[1]:+06d}"
    )
    return {
        "candidate_id": candidate_id,
        "map_xy_m": [float(x_m), float(y_m)],
        "subgoal_forward_m": float(forward_m),
        "subgoal_left_m": float(left_m),
        "distance_m": float(distance_m),
        "bearing_rad": float(bearing),
        "frontier_normal_bearing_rad": float(normal_bearing),
        "resolution_m": float(grid.resolution_m),
        "grid_cell": [int(cell[0]), int(cell[1])],
        "frontier_boundary_m": float(boundary_m),
        "frontier_novelty_m": float(trace_distance),
        "clearance_lower_m": float(clearance_m),
        "topology_score": float(topology_score),
        "context_frame_indices": _nearest_trace_context(
            grid, x_m, y_m, config),
        "goal_patch_relation_score": 0.0,
        "goal_patch_relation_present": False,
        "selection_sources": [],
        "source_scales_m": [float(grid.resolution_m)],
    }


def _validate_patch_scores(
    patch_scores_by_frame: Mapping[int, float] | None,
) -> dict[int, float]:
    if patch_scores_by_frame is None:
        return {}
    scores = {}
    for raw_frame, raw_score in patch_scores_by_frame.items():
        if (isinstance(raw_frame, bool) or int(raw_frame) != raw_frame
                or int(raw_frame) < 0):
            raise FrontierProposalError("patch-score frame keys must be nonnegative integers")
        score = float(raw_score)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise FrontierProposalError("patch relation scores must lie in [-1,1]")
        scores[int(raw_frame)] = score
    return scores


def _apply_patch_scores(candidates: Sequence[dict], scores: Mapping[int, float]) -> None:
    for candidate in candidates:
        values = [
            scores[int(frame)]
            for frame in candidate["context_frame_indices"]
            if int(frame) in scores
        ]
        if values:
            candidate["goal_patch_relation_score"] = float(max(values))
            candidate["goal_patch_relation_present"] = True


def deterministic_spatial_bearing_nms(
    candidates: Sequence[Mapping[str, object]],
    config: FrontierConfig = FrontierConfig(),
) -> tuple[list[dict], list[dict]]:
    """Suppress redundant cross-scale proposals using deployment geometry.

    Spatial duplicates are always removed.  Same-bearing proposals are also
    removed when their radial separation is small, preventing one corridor ray
    from consuming the complete shortlist.  Sorting and all ties are explicit.
    """
    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda row: (
            -float(row["topology_score"]),
            -float(row["frontier_boundary_m"]),
            float(row["distance_m"]),
            str(row["candidate_id"]),
        ),
    )
    kept: list[dict] = []
    suppressed: list[dict] = []
    for candidate in ordered:
        duplicate = None
        for winner in kept:
            spatial = math.dist(
                tuple(map(float, candidate["map_xy_m"])),
                tuple(map(float, winner["map_xy_m"])),
            )
            same_ray = (
                angular_distance(
                    float(candidate["bearing_rad"]),
                    float(winner["bearing_rad"]),
                ) <= config.bearing_nms_rad
                and abs(float(candidate["distance_m"])
                        - float(winner["distance_m"])) <= config.radial_nms_m
            )
            if spatial <= config.spatial_nms_m or same_ray:
                duplicate = winner
                break
        if duplicate is None:
            kept.append(candidate)
        else:
            suppressed.append({
                "candidate_id": str(candidate["candidate_id"]),
                "kept_candidate_id": str(duplicate["candidate_id"]),
                "reason": "spatial" if math.dist(
                    tuple(map(float, candidate["map_xy_m"])),
                    tuple(map(float, duplicate["map_xy_m"])),
                ) <= config.spatial_nms_m else "bearing_radial",
            })
            scales = sorted(set(
                map(float, duplicate["source_scales_m"])
            ) | set(map(float, candidate["source_scales_m"])))
            duplicate["source_scales_m"] = scales
    return kept, suppressed


def _append_slot(
    selected: list[dict],
    selected_ids: set[str],
    rows: Sequence[dict],
    source: str,
    count: int,
) -> None:
    added = 0
    for candidate in rows:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in selected_ids:
            if source not in candidate["selection_sources"]:
                candidate["selection_sources"].append(source)
            continue
        if added >= count:
            break
        candidate["selection_sources"].append(source)
        selected.append(candidate)
        selected_ids.add(candidate_id)
        added += 1


def deployment_shortlist(
    candidates: Sequence[Mapping[str, object]],
    *,
    max_shortlist: int = MAX_SHORTLIST,
) -> list[dict]:
    """Select patch/topology/angular coverage slots without privileged data."""
    if (isinstance(max_shortlist, bool) or int(max_shortlist) != max_shortlist
            or not 1 <= int(max_shortlist) <= MAX_SHORTLIST):
        raise FrontierProposalError(f"max_shortlist must lie in [1,{MAX_SHORTLIST}]")
    mutable = [dict(row) for row in candidates]
    for row in mutable:
        row["selection_sources"] = list(row.get("selection_sources", []))
    selected: list[dict] = []
    selected_ids: set[str] = set()

    patch = sorted(
        (row for row in mutable if bool(row["goal_patch_relation_present"])),
        key=lambda row: (
            -float(row["goal_patch_relation_score"]),
            -float(row["topology_score"]),
            str(row["candidate_id"]),
        ),
    )
    _append_slot(selected, selected_ids, patch, "goal_patch_top2", 2)

    topology = sorted(
        mutable,
        key=lambda row: (
            -float(row["topology_score"]),
            -float(row["frontier_boundary_m"]),
            str(row["candidate_id"]),
        ),
    )
    _append_slot(selected, selected_ids, topology, "topology_top2", 2)

    # Greedy max-min bearing coverage, conditioned only on already selected
    # deployment proposals.  Native forward (bearing 0) is included as a fixed
    # reference so both new slots do not collapse onto the forward corridor.
    for _ in range(2):
        remaining = [
            row for row in mutable
            if str(row["candidate_id"]) not in selected_ids
        ]
        if not remaining:
            break
        reference_bearings = [0.0] + [
            float(row["bearing_rad"]) for row in selected]
        choice = min(
            remaining,
            key=lambda row: (
                -min(angular_distance(
                    float(row["bearing_rad"]), reference)
                     for reference in reference_bearings),
                -float(row["topology_score"]),
                str(row["candidate_id"]),
            ),
        )
        _append_slot(
            selected, selected_ids, [choice], "angular_diverse_top2", 1)

    return selected[:int(max_shortlist)]


def generate_frontier_proposals(
    scans: Sequence[PlanarScan],
    current_pose: SE2Pose,
    *,
    patch_scores_by_frame: Mapping[int, float] | None = None,
    config: FrontierConfig = FrontierConfig(),
) -> dict:
    """Generate one arm's candidate universe and deployment shortlist."""
    if not scans:
        raise FrontierProposalError("at least one causal scan is required")
    frames = [int(scan.frame_index) for scan in scans]
    if frames != sorted(set(frames)):
        raise FrontierProposalError("scan frame indices must be unique and increasing")
    patch_scores = _validate_patch_scores(patch_scores_by_frame)
    if any(frame > frames[-1] for frame in patch_scores):
        raise FrontierProposalError("patch scores cannot reference a future frame")

    raw: list[dict] = []
    scale_summaries = []
    for resolution in config.resolutions_m:
        grid = SparseRayGrid(resolution, config.ray_step_fraction)
        for scan in scans:
            grid.integrate(scan)
        components = [
            component for component in grid.frontier_components()
            if len(component) >= config.minimum_component_cells
        ]
        scale_candidates = []
        for component in components:
            for cell in _component_representatives(grid, component, config):
                candidate = _candidate_from_cell(
                    grid, cell, component, current_pose, config)
                if candidate is not None:
                    scale_candidates.append(candidate)
        _apply_patch_scores(scale_candidates, patch_scores)
        raw.extend(scale_candidates)
        scale_summaries.append({
            "resolution_m": float(resolution),
            "free_cell_count": len(grid.free_cells()),
            "occupied_cell_count": len(grid.occupied_cells()),
            "frontier_cell_count": len(grid.frontier_cells()),
            "component_count": len(components),
            "candidate_count": len(scale_candidates),
        })

    nms, suppressed = deterministic_spatial_bearing_nms(raw, config)
    shortlist = deployment_shortlist(
        nms, max_shortlist=config.max_shortlist)
    patch_relation_present = any(
        bool(candidate["goal_patch_relation_present"])
        for candidate in nms
    )
    proposal = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "invalid_reason": None,
        "pose_frame_index": int(frames[-1]),
        "scan_frame_indices": frames,
        "goal_patch_relation_present": patch_relation_present,
        "goal_patch_relation_mask": 1 if patch_relation_present else 0,
        "shortlist_policy": {
            "slots": list(SHORTLIST_SLOTS),
            "max_candidates": int(config.max_shortlist),
            "missing_patch_behavior": (
                "skip goal_patch_top2; preserve zero score with mask=0; "
                "use topology/angular deployment slots only"
            ),
        },
        "scale_summaries": scale_summaries,
        "raw_candidate_count": len(raw),
        "nms_candidate_count": len(nms),
        "shortlist_count": len(shortlist),
        "candidate_universe": nms,
        "shortlist": shortlist,
        "nms_suppressed": suppressed,
    }
    # Canonical serialization here is also a finite-value assertion.
    canonical_json_bytes(proposal)
    return proposal


def invalid_proposal(reason: str, *, pose_frame_index: int | None = None) -> dict:
    if not isinstance(reason, str) or not reason.strip():
        raise FrontierProposalError("invalid proposal requires a reason")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "invalid_reason": reason,
        "pose_frame_index": pose_frame_index,
        "scan_frame_indices": [],
        "goal_patch_relation_present": False,
        "goal_patch_relation_mask": 0,
        "shortlist_policy": {
            "slots": list(SHORTLIST_SLOTS),
            "max_candidates": MAX_SHORTLIST,
            "missing_patch_behavior": (
                "skip goal_patch_top2; preserve zero score with mask=0; "
                "use topology/angular deployment slots only"
            ),
        },
        "scale_summaries": [],
        "raw_candidate_count": 0,
        "nms_candidate_count": 0,
        "shortlist_count": 0,
        "candidate_universe": [],
        "shortlist": [],
        "nms_suppressed": [],
    }


def attach_proposal_proxy_labels(
    *,
    sample_id: str,
    arm: str,
    proposal: Mapping[str, object],
    labeler: ProposalProxyLabeler,
    positive_margin_m: float = 0.0,
) -> dict:
    """Attach an isolated privileged coverage report after proposal freeze."""
    if not math.isfinite(float(positive_margin_m)):
        raise FrontierProposalError("proxy positive margin must be finite")
    proposal_hash_before = canonical_sha256(proposal)
    provenance = dict(labeler.provenance())
    # Reject nested non-canonical/NaN provenance before invoking the labeler.
    canonical_json_bytes(provenance)
    if not bool(proposal.get("valid", False)):
        return {
            "status": "invalid_proposal",
            "label_valid": False,
            "labeler_provenance": provenance,
            "positive_margin_m": float(positive_margin_m),
            "labels": [],
            "universe_has_positive": False,
            "shortlist_has_positive": False,
            "coverage_miss": False,
            "proposal_sha256": proposal_hash_before,
        }
    universe = proposal.get("candidate_universe")
    shortlist = proposal.get("shortlist")
    if not isinstance(universe, Sequence) or not isinstance(shortlist, Sequence):
        raise FrontierProposalError("proposal candidate tables are malformed")
    labels = []
    for candidate in universe:
        if not isinstance(candidate, Mapping):
            raise FrontierProposalError("proposal candidate is malformed")
        # A labeler is external privileged code.  Give it an isolated copy so
        # even a buggy implementation cannot corrupt the proposal before the
        # postcondition below detects the attempted mutation.
        measurement = labeler.label(
            sample_id=sample_id, arm=arm,
            candidate=copy.deepcopy(dict(candidate)))
        if not isinstance(measurement, ProxyMeasurement):
            raise FrontierProposalError("proxy labeler returned an invalid measurement")
        labels.append({
            "candidate_id": str(candidate["candidate_id"]),
            "reachable": bool(measurement.reachable),
            "progress_m": float(measurement.progress_m),
            "positive": bool(
                measurement.reachable
                and measurement.progress_m > float(positive_margin_m)),
        })
    shortlist_ids = {
        str(candidate["candidate_id"])
        for candidate in shortlist
        if isinstance(candidate, Mapping)
    }
    positive_ids = {
        row["candidate_id"] for row in labels if row["positive"]
    }
    universe_positive = bool(positive_ids)
    shortlist_positive = bool(shortlist_ids & positive_ids)
    if canonical_sha256(proposal) != proposal_hash_before:
        raise FrontierProposalError("proxy labeler mutated the frozen proposal")
    return {
        "status": "labeled",
        "label_valid": True,
        "labeler_provenance": provenance,
        "positive_margin_m": float(positive_margin_m),
        "labels": labels,
        "universe_has_positive": universe_positive,
        "shortlist_has_positive": shortlist_positive,
        "coverage_miss": bool(universe_positive and not shortlist_positive),
        "proposal_sha256": proposal_hash_before,
    }
