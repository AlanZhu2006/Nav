"""Deployable causal-RGB route tape for long-range certified revisits.

CEC first authorizes one historical image anchor.  This module does not
perform that open-set decision.  It only specifies how an authorized causal
RGB interval is converted into a sparse local route and how later query
observations advance along it:

* history is sampled at write-time depth receipts;
* every edge is a metric local SE(2) estimate from adjacent sampled RGBs;
* the live state is updated by the same visual-motion contract;
* guidance is the first feasible local route tangent, normalized before it is
  sent to the unchanged NavDP PointGoal input.

No world pose, simulator path, wheel odometry, distance bin, stuck detector,
endpoint controller, or post-authorization native fallback is represented in
this contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np

from MemNavData.monocular_adjacent_motion import PlanarMotionReceipt
from MemNavData.path_budgeted_route_compass import (
    FirstFeasibleLocalTangentRouteCompass,
    first_feasible_tangent_from_local_route,
)
from MemNavData.se2_projected_route_compass import (
    reconstruct_reverse_route_se2,
)


MONOCULAR_ROUTE_TANGENT_RUNTIME_SCHEMA_VERSION = (
    "monocular_route_tangent_runtime_v1_20260903"
)
FIRST_METRIC_FRAME = 40
FIRST_CERTIFIABLE_ANCHOR = 8


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def historical_route_schedule(
    *,
    target_anchor: int,
    goal_start_frame: int,
    cached_depth_frames: Iterable[int],
    first_metric_frame: int = FIRST_METRIC_FRAME,
    minimum_anchor: int = FIRST_CERTIFIABLE_ANCHOR,
) -> dict[str, Any]:
    """Freeze the sparse causal interval used to construct one reverse route.

    Relative depth is cached causally at a fixed writer stride after LingBot's
    eight-frame initialization, including frames that precede the independent
    frame-40 metric-scale receipt.  At query time that already-frozen receipt
    supplies one scale for all stored relative depths.  The selected target
    anchor itself always has depth through CEC's canonical exact replay, even
    when it lies between sparse writer positions.

    All remaining edges follow chronological order and have a depth-bearing
    reference frame.  The first query frame is the final chronological node,
    so reversing the edge chain expresses the route in the query camera's
    initial body frame.
    """

    target = _strict_int(target_anchor, "target_anchor")
    goal_start = _strict_int(goal_start_frame, "goal_start_frame")
    first_metric = _strict_int(first_metric_frame, "first_metric_frame")
    minimum = _strict_int(minimum_anchor, "minimum_anchor")
    if first_metric < 1:
        raise ValueError("first_metric_frame must be positive")
    if minimum < 0 or minimum >= first_metric:
        raise ValueError("minimum_anchor must precede first_metric_frame")
    if target < minimum:
        raise ValueError("anchor predates the CEC certification boundary")
    if target >= goal_start:
        raise ValueError("target_anchor must precede goal_start_frame")

    cached = sorted({
        _strict_int(frame, "cached_depth_frame")
        for frame in cached_depth_frames if 0 <= int(frame) < goal_start
    })

    nodes = [target]
    edge_specs: list[dict[str, Any]] = []
    anchor_depth_cached = target in cached

    for frame in cached:
        if frame <= nodes[-1]:
            continue
        edge_specs.append({
            "from_frame": nodes[-1],
            "to_frame": frame,
            "depth_reference_frame": nodes[-1],
            "match_query_frame": frame,
            "invert_estimate": False,
            "edge_kind": "historical_adjacent_sample",
        })
        nodes.append(frame)

    if nodes[-1] != goal_start:
        edge_specs.append({
            "from_frame": nodes[-1],
            "to_frame": goal_start,
            "depth_reference_frame": nodes[-1],
            "match_query_frame": goal_start,
            "invert_estimate": False,
            "edge_kind": "history_to_query_bridge",
        })
        nodes.append(goal_start)

    if len(nodes) < 2 or len(edge_specs) != len(nodes) - 1:
        raise RuntimeError("route schedule is not a connected causal chain")
    if nodes[0] != target or nodes[-1] != goal_start:
        raise RuntimeError("route schedule endpoints changed")
    return {
        "schema_version": MONOCULAR_ROUTE_TANGENT_RUNTIME_SCHEMA_VERSION,
        "target_anchor": target,
        "goal_start_frame": goal_start,
        "first_metric_frame": first_metric,
        "minimum_anchor": minimum,
        "nodes_chronological": nodes,
        "edges_chronological": edge_specs,
        "anchor_depth_cached": anchor_depth_cached,
        "anchor_depth_available": True,
        "reference_depth_bridge": False,
        "pre_metric_anchor_bridge": False,
        "world_pose_consumed": False,
        "wheel_odometry_consumed": False,
    }


def query_update_schedule(
    *,
    previous_frame: int,
    current_frame: int,
    cached_depth_frames: Iterable[int],
) -> list[int]:
    """Return ordered visual-motion endpoints for one planning interval."""

    previous = _strict_int(previous_frame, "previous_frame")
    current = _strict_int(current_frame, "current_frame")
    if current < previous:
        raise ValueError("query frame index regressed")
    if current == previous:
        return []
    intermediate = sorted({
        _strict_int(frame, "cached_depth_frame")
        for frame in cached_depth_frames
        if previous < int(frame) < current
    })
    return [*intermediate, current]


def accepted_planar_motion(estimate: Mapping[str, Any]) -> PlanarMotionReceipt:
    """Read one local SE(2) receipt only after its adjacency proof passes."""

    validity = estimate.get("local_motion_validity")
    motion = estimate.get("motion")
    if not isinstance(validity, Mapping) or validity.get("accepted") is not True:
        reason = (
            validity.get("reason") if isinstance(validity, Mapping)
            else "missing_local_motion_validity"
        )
        raise RuntimeError(f"adjacent visual motion rejected: {reason}")
    if not isinstance(motion, Mapping):
        raise RuntimeError("accepted adjacent motion has no pose")
    receipt = PlanarMotionReceipt(
        forward_m=float(motion["forward_m"]),
        left_m=float(motion["left_m"]),
        yaw_rad=float(motion["yaw_rad"]),
        vertical_m=float(motion["vertical_m"]),
    )
    values = np.asarray([
        receipt.forward_m,
        receipt.left_m,
        receipt.yaw_rad,
        receipt.vertical_m,
    ], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("adjacent visual motion is non-finite")
    return receipt


def route_from_chronological_motions(
    motions: Iterable[PlanarMotionReceipt],
) -> np.ndarray:
    """Express an authorized reverse route in the initial query body frame."""

    ordered = list(motions)
    if len(ordered) < 2:
        raise ValueError("a route tape requires at least two visual edges")
    return reconstruct_reverse_route_se2(
        [motion.forward_m for motion in reversed(ordered)],
        [motion.left_m for motion in reversed(ordered)],
        [motion.yaw_rad for motion in reversed(ordered)],
    )[0]


def build_route_compass(
    motions: Iterable[PlanarMotionReceipt],
    *,
    tangent_baseline_m: float = 0.30,
    controller_radius_m: float = 2.5,
) -> FirstFeasibleLocalTangentRouteCompass:
    return first_feasible_tangent_from_local_route(
        route_from_chronological_motions(motions),
        tangent_baseline_m=tangent_baseline_m,
        controller_radius_m=controller_radius_m,
    )


def edge_receipt_sha256(edge_receipts: Iterable[Mapping[str, Any]]) -> str:
    """Bind the exact sparse RGB/depth/motion evidence used by one route."""

    encoded = json.dumps(
        list(edge_receipts), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
