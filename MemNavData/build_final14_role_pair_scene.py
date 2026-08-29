#!/usr/bin/env python3
"""Build one scene fragment for the frozen final14 query populations.

This is deliberately separate from the Attempt-7 ``build_paper_role_pair``
pipeline.  The frozen final14 protocol has one primary population containing a
natural unsupported Novel query and a standard-support Revisit query, plus an
independently constructed hard-support Revisit subset.  A missing hard query
must never remove an otherwise valid standard/Novel history.

The builder reads only a successful, materialized online-A history.  It does
not run or inspect any query policy.  Construction labels and co-visibility
diagnostics remain in evaluator-side metadata and are removed by
``runtime_query`` before policy execution.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import build_shared_online_double_revisit as history_tools
import build_shared_online_role_pairs as pair_tools
from final14_role_pair_contract import (
    DEPTH_TOLERANCE_M,
    NOVEL_ATTEMPTS,
    PROTOCOLS,
    SCENE_BUILD_SCHEMA,
    STRATA,
    assigned_direction_stratum,
    direction_in_stratum,
    goal_yaw_bin,
    goal_yaw_radians,
    relative_direction_degrees,
    role_contract,
    stable_u32,
    support_band,
)
from generate_twoleg import covis_curve, covis_frac, geodesic, make_sim, render
from shared_online_role_pair_contract import SCHEMA_VERSION, validate_manifest


ELIGIBLE_FRAME_FLOOR = 39
END_MARGIN_FRAMES = 16
SOURCE_FRAME_STRIDE = 8
MAX_SOURCE_FRAMES = 6
MAX_HISTORIES_PER_SCENE = 3


class NaturalNovelConstructionError(RuntimeError):
    """No proposal passed a result-blind natural-Novel contract."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "no natural Novel query passed the frozen distance/direction/"
            "support contract: "
            f"{json.dumps(self.diagnostics, sort_keys=True)}"
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def deterministic_pose_grid(
    identity: str,
) -> list[tuple[float, float, float]]:
    """Return deterministic (radius, world direction, yaw offset) proposals."""

    phase = stable_u32("final14_revisit_grid", identity) / float(2**32)
    phase *= 2.0 * math.pi
    radii = (0.22, 0.30, 0.46, 0.65, 0.85, 1.00)
    directions = [phase + index * 2.0 * math.pi / 12.0 for index in range(12)]
    offsets = (12.0, -18.0, 24.0, -30.0, 36.0, -45.0, 54.0, -60.0)
    rows = []
    for radius in radii:
        for direction_index, direction in enumerate(directions):
            for offset_index in range(len(offsets)):
                offset = offsets[(direction_index + offset_index) % len(offsets)]
                rows.append((radius, direction, offset))
    return rows


def online_endpoint(history: dict) -> tuple[np.ndarray, float]:
    """Return the exact endpoint bound by the replayed online-A trace."""

    trace = history["trace"]
    return (
        np.asarray(trace["end_position"], dtype=np.float64),
        float(trace["end_yaw"]),
    )


def source_frame_candidates(simulator, history: dict) -> list[dict[str, Any]]:
    endpoint, _endpoint_yaw = online_endpoint(history)
    rows = []
    for frame in range(
        ELIGIBLE_FRAME_FLOOR,
        len(history["poses"]) - END_MARGIN_FRAMES,
        SOURCE_FRAME_STRIDE,
    ):
        try:
            distance = history_tools.goal_distance(
                simulator.pathfinder,
                endpoint,
                history["floor_positions"][frame],
            )
        except RuntimeError:
            continue
        if 2.0 <= distance <= 9.0:
            rows.append({
                "frame": int(frame),
                "source_geodesic_m": float(distance),
                "target_distance_error_m": abs(float(distance) - 3.0),
            })
    rows.sort(key=lambda row: (row["target_distance_error_m"], row["frame"]))
    return rows[:MAX_SOURCE_FRAMES]


def _candidate_json(candidate: dict[str, Any] | None) -> dict | None:
    if candidate is None:
        return None
    return {
        key: value
        for key, value in candidate.items()
        if not key.startswith("_")
    }


def search_revisit_candidates(
    simulator,
    history: dict,
    *,
    scene: str,
    episode: str,
    camera_height: float,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any]]:
    endpoint, _endpoint_yaw = online_endpoint(history)
    sources = source_frame_candidates(simulator, history)
    selected: dict[str, dict[str, Any] | None] = {
        "standard": None,
        "hard": None,
    }
    diagnostics = {
        "source_frames_considered": sources,
        "grid_attempts": 0,
        "pose_rejects": 0,
        "pixel_mae_rejects": 0,
        "query_distance_rejects": 0,
        "support_rejects": 0,
        "fully_scored": 0,
    }
    for source in sources:
        frame = int(source["frame"])
        source_floor = history["floor_positions"][frame]
        source_yaw = float(history["poses"][frame]["yaw"])
        source_rgb = history["rgbs"][frame]
        source_depth = history["depths"][frame]
        source_transform = history["transforms"][frame]
        seen = set()
        identity = f"{scene}/{episode}/{frame}"
        for attempt, (radius, direction, yaw_offset) in enumerate(
            deterministic_pose_grid(identity), start=1
        ):
            diagnostics["grid_attempts"] += 1
            raw = source_floor + np.asarray([
                radius * math.cos(direction), 0.0,
                radius * math.sin(direction),
            ])
            snapped = np.asarray(
                simulator.pathfinder.snap_point(raw), dtype=np.float64
            )
            if (
                not simulator.pathfinder.is_navigable(snapped)
                or abs(float(snapped[1] - source_floor[1])) > 0.20
                or float(np.linalg.norm(snapped[[0, 2]] - raw[[0, 2]])) > 0.20
            ):
                diagnostics["pose_rejects"] += 1
                continue
            translation = float(
                np.linalg.norm(snapped[[0, 2]] - source_floor[[0, 2]])
            )
            if not 0.20 <= translation <= 1.00:
                diagnostics["pose_rejects"] += 1
                continue
            yaw = history_tools.wrap_radians(
                source_yaw + math.radians(yaw_offset)
            )
            identity_key = (
                round(float(snapped[0]), 4),
                round(float(snapped[1]), 4),
                round(float(snapped[2]), 4),
                round(float(yaw), 4),
            )
            if identity_key in seen:
                continue
            seen.add(identity_key)
            camera_position = snapped + np.asarray(
                [0.0, camera_height, 0.0], dtype=np.float64
            )
            rgb, depth = render(simulator, camera_position, yaw)
            pixel_mae = history_tools.pixel_mae(rgb, source_rgb)
            if pixel_mae < 5.0:
                diagnostics["pixel_mae_rejects"] += 1
                continue
            try:
                query_distance = history_tools.goal_distance(
                    simulator.pathfinder, endpoint, snapped
                )
            except RuntimeError:
                diagnostics["query_distance_rejects"] += 1
                continue
            if not 2.0 <= query_distance <= 9.0:
                diagnostics["query_distance_rejects"] += 1
                continue
            points = history_tools.goal_world_points(
                depth, camera_position, yaw
            )
            anchor_covis = float(
                covis_frac(points, source_transform, source_depth)
            )
            curve = covis_curve(
                points,
                history["transforms"],
                history["depths"],
                tol=DEPTH_TOLERANCE_M,
            )
            eligible = curve[ELIGIBLE_FRAME_FLOOR:]
            require(len(eligible) > 0, "online history has no eligible frames")
            best_frame = ELIGIBLE_FRAME_FLOOR + int(np.argmax(eligible))
            best_covis = float(curve[best_frame])
            gap = abs(best_frame - frame)
            yaw_delta = history_tools.angle_delta_degrees(yaw, source_yaw)
            diagnostics["fully_scored"] += 1
            band = support_band(best_covis, gap)
            if band == "standard" and not (
                0.20 <= translation <= 0.80
                and 12.0 <= yaw_delta <= 45.0
            ):
                band = None
            if band == "hard" and not (
                0.30 <= translation <= 1.00
                and 18.0 <= yaw_delta <= 60.0
            ):
                band = None
            if band is None:
                diagnostics["support_rejects"] += 1
                continue
            geometry = pair_tools.query_geometry(
                simulator.pathfinder, endpoint, snapped
            )
            require(geometry is not None, "accepted query geometry disappeared")
            measured_distance, initial_bearing = geometry
            require(
                abs(measured_distance - query_distance) <= 1e-5,
                "query distance changed between measurements",
            )
            target = 0.72 if band == "standard" else 0.40
            ranking = (
                abs(best_covis - target),
                abs(float(query_distance) - 3.0),
                -(len(history["poses"]) - 1 - frame),
                frame,
                attempt,
            )
            record = {
                "support_band": band,
                "source_frame": frame,
                "render_attempt": int(attempt),
                "translation_m": translation,
                "yaw_delta_deg": yaw_delta,
                "source_anchor_covis": anchor_covis,
                "max_online_a_covis": best_covis,
                "max_online_a_covis_frame": best_frame,
                "argmax_gap_frames": gap,
                "eligible_online_a_frame_floor": ELIGIBLE_FRAME_FLOOR,
                "pixel_mae": pixel_mae,
                "query_geodesic_m": float(query_distance),
                "initial_path_bearing_rad": float(initial_bearing),
                "ranking": list(ranking),
                "_position": snapped,
                "_yaw": float(yaw),
                "_rgb": rgb,
                "_depth": depth,
                "_covis_curve": [float(value) for value in curve],
            }
            current = selected[band]
            if current is None or ranking < tuple(current["ranking"]):
                selected[band] = record
    return selected, diagnostics


def _optional_geodesic(pathfinder, first: np.ndarray, second: np.ndarray):
    ok, distance, _path = geodesic(pathfinder, first, second)
    if not ok or not np.isfinite(distance):
        return None
    return float(distance)


def deterministic_novel_position_grid(
    endpoint: np.ndarray,
    endpoint_yaw: float,
    *,
    scene: str,
    episode: str,
    stratum: str,
) -> list[np.ndarray]:
    """Local polar proposals whose *intended* bearing covers one stratum.

    Final admission still uses the first segment of the Habitat shortest path,
    so snapping or obstacles cannot silently relabel a direction.  The hash
    only rotates proposal order; it does not depend on any rendered image or
    policy result.
    """

    relative_angles = {
        "front": (-60, -45, -30, -15, 0, 15, 30, 45, 60),
        "side": (-120, -105, -90, -75, 75, 90, 105, 120),
        "rear": (-180, -165, -150, -135, -121, 121, 135, 150, 165),
    }[stratum]
    radii = (2.25, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5)
    rows = []
    for radius in radii:
        for relative in relative_angles:
            world_yaw = float(endpoint_yaw) + math.radians(relative)
            rows.append(
                np.asarray(endpoint, dtype=np.float64)
                + np.asarray(
                    [
                        -radius * math.sin(world_yaw),
                        0.0,
                        -radius * math.cos(world_yaw),
                    ],
                    dtype=np.float64,
                )
            )
    offset = stable_u32("final14_novel_grid_order", scene, episode) % len(rows)
    return rows[offset:] + rows[:offset]


def sample_natural_novel(
    simulator,
    history: dict,
    *,
    scene: str,
    episode: str,
    scene_rank: int,
    episode_rank: int,
    paired_revisit_position: np.ndarray,
    camera_height: float,
    minimum_paired_distance_m: float = 1.0,
    maximum_paired_distance_m: float | None = None,
    separated_from_positions: list[np.ndarray] | None = None,
    minimum_candidate_separation_m: float = 0.0,
    direction_stratum: str | None = None,
    sampling_seed_namespace: str = "final14_natural",
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(minimum_paired_distance_m >= 0.0,
            "minimum paired distance must be non-negative")
    require(
        maximum_paired_distance_m is None
        or maximum_paired_distance_m >= minimum_paired_distance_m,
        "maximum paired distance precedes minimum paired distance",
    )
    require(minimum_candidate_separation_m >= 0.0,
            "minimum candidate separation must be non-negative")
    separated_from_positions = list(separated_from_positions or [])
    endpoint, endpoint_yaw = online_endpoint(history)
    if direction_stratum is None:
        stratum = assigned_direction_stratum(scene_rank, episode_rank)
    else:
        require(direction_stratum in STRATA,
                f"unknown direction stratum {direction_stratum!r}")
        stratum = str(direction_stratum)
    yaw_bin = goal_yaw_bin(scene, episode)
    goal_yaw = goal_yaw_radians(scene, episode)
    # Preserve the frozen Final14 seed byte-for-byte in the default path.  A
    # distinct namespace is available only to separately frozen construction
    # protocols; it prevents a new structural probe from silently reusing the
    # legacy random stream after changing its direction contract.
    seed_parts: tuple[object, ...] = (
        sampling_seed_namespace, scene, episode, scene_rank, episode_rank,
    )
    if direction_stratum is not None:
        seed_parts += (stratum,)
    seed = stable_u32(*seed_parts)
    pathfinder = simulator.pathfinder
    require(hasattr(pathfinder, "seed"), "PathFinder has no deterministic seed")
    pathfinder.seed(seed)
    local_proposals = deterministic_novel_position_grid(
        endpoint,
        endpoint_yaw,
        scene=scene,
        episode=episode,
        stratum=stratum,
    )
    seen_positions = set()
    diagnostics = {
        "attempts": 0,
        "deterministic_local_proposals": len(local_proposals),
        "deterministic_local_attempts": 0,
        "uniform_random_attempts": 0,
        "duplicate_position_rejects": 0,
        "floor_or_clearance_rejects": 0,
        "non_navigable_rejects": 0,
        "floor_mismatch_rejects": 0,
        "clearance_rejects": 0,
        "candidate_separation_rejects": 0,
        "geodesic_rejects": 0,
        "unreachable_rejects": 0,
        "a_to_b_outside_band_rejects": 0,
        "direction_stratum_rejects": 0,
        "paired_separation_rejects": 0,
        "paired_unreachable_rejects": 0,
        "paired_below_minimum_rejects": 0,
        "paired_above_maximum_rejects": 0,
        "support_rejects": 0,
    }
    for attempt in range(1, NOVEL_ATTEMPTS + 1):
        diagnostics["attempts"] = attempt
        if attempt <= len(local_proposals):
            diagnostics["deterministic_local_attempts"] += 1
            position = np.asarray(
                pathfinder.snap_point(local_proposals[attempt - 1]),
                dtype=np.float64,
            )
            proposal_mode = "deterministic_local_polar_grid"
        else:
            diagnostics["uniform_random_attempts"] += 1
            position = np.asarray(
                pathfinder.get_random_navigable_point(), dtype=np.float64
            )
            proposal_mode = "deterministic_seeded_uniform_fallback"
        position_key = tuple(round(float(value), 4) for value in position)
        if position_key in seen_positions:
            diagnostics["duplicate_position_rejects"] += 1
            continue
        seen_positions.add(position_key)
        if not pathfinder.is_navigable(position):
            diagnostics["non_navigable_rejects"] += 1
            diagnostics["floor_or_clearance_rejects"] += 1
            continue
        if abs(float(position[1] - endpoint[1])) > 0.20:
            diagnostics["floor_mismatch_rejects"] += 1
            diagnostics["floor_or_clearance_rejects"] += 1
            continue
        if float(pathfinder.distance_to_closest_obstacle(position)) < 0.30:
            diagnostics["clearance_rejects"] += 1
            diagnostics["floor_or_clearance_rejects"] += 1
            continue
        if any(
            float(np.linalg.norm(position[[0, 2]] - prior[[0, 2]]))
            < minimum_candidate_separation_m
            for prior in separated_from_positions
        ):
            diagnostics["candidate_separation_rejects"] += 1
            continue
        geometry = pair_tools.query_geometry(pathfinder, endpoint, position)
        if geometry is None:
            diagnostics["unreachable_rejects"] += 1
            diagnostics["geodesic_rejects"] += 1
            continue
        query_distance, initial_bearing = geometry
        if not 2.0 <= query_distance <= 9.0:
            diagnostics["a_to_b_outside_band_rejects"] += 1
            diagnostics["geodesic_rejects"] += 1
            continue
        relative_degrees = relative_direction_degrees(
            initial_bearing, endpoint_yaw
        )
        if not direction_in_stratum(relative_degrees, stratum):
            diagnostics["direction_stratum_rejects"] += 1
            continue
        paired_distance = _optional_geodesic(
            pathfinder, paired_revisit_position, position
        )
        if paired_distance is None:
            diagnostics["paired_unreachable_rejects"] += 1
            diagnostics["paired_separation_rejects"] += 1
            continue
        if paired_distance < minimum_paired_distance_m:
            diagnostics["paired_below_minimum_rejects"] += 1
            diagnostics["paired_separation_rejects"] += 1
            continue
        if (
            maximum_paired_distance_m is not None
            and paired_distance > maximum_paired_distance_m
        ):
            diagnostics["paired_above_maximum_rejects"] += 1
            diagnostics["paired_separation_rejects"] += 1
            continue
        camera_position = position + np.asarray(
            [0.0, camera_height, 0.0], dtype=np.float64
        )
        rgb, depth = render(simulator, camera_position, goal_yaw)
        points = history_tools.goal_world_points(
            depth, camera_position, goal_yaw
        )
        curve = covis_curve(
            points,
            history["transforms"],
            history["depths"],
            tol=DEPTH_TOLERANCE_M,
        )
        maximum = float(curve.max()) if len(curve) else 0.0
        if maximum >= 0.10:
            diagnostics["support_rejects"] += 1
            continue
        record = {
            "support_band": "unsupported_novel",
            "query_geodesic_m": float(query_distance),
            "initial_path_bearing_rad": float(initial_bearing),
            "initial_path_direction_relative_to_a_end_deg": relative_degrees,
            "assigned_direction_stratum": stratum,
            "max_online_a_covis": maximum,
            "max_online_a_covis_frame": (
                int(np.argmax(curve)) if len(curve) else None
            ),
            "eligible_online_a_frame_floor": 0,
            "paired_revisit_separation_m": float(paired_distance),
            "goal_world_yaw_bin": int(yaw_bin),
            "goal_yaw_contract": "identity_hash_eight_world_yaw_bins",
            "sampling_seed": int(seed),
            "accepted_proposal_mode": proposal_mode,
            "sampling_diagnostics": diagnostics,
            "_position": position,
            "_yaw": float(goal_yaw),
            "_rgb": rgb,
            "_depth": depth,
            "_covis_curve": [float(value) for value in curve],
        }
        return record, diagnostics
    raise NaturalNovelConstructionError(diagnostics)


def _write_goal(
    root: Path, role: str, candidate: dict[str, Any]
) -> tuple[Path, Path, str, str]:
    rgb_path = root / role / "goal.jpg"
    depth_path = root / role / "goal_depth.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_bytes(history_tools.jpeg_bytes(candidate["_rgb"]))
    history_tools.write_depth_png(depth_path, candidate["_depth"])
    return (
        rgb_path,
        depth_path,
        history_tools.sha256_file(rgb_path),
        history_tools.sha256_file(depth_path),
    )


def _query_record(
    *,
    query_id: str,
    role: str,
    candidate: dict[str, Any],
    episode_root: Path,
    rgb_path: Path,
    depth_path: Path,
    rgb_sha: str,
    depth_sha: str,
) -> dict[str, Any]:
    record = {
        "query_id": query_id,
        "analysis_role": role,
        "goal_rgb": str(rgb_path.relative_to(episode_root)),
        "goal_rgb_sha256": rgb_sha,
        "goal_depth": str(depth_path.relative_to(episode_root)),
        "goal_depth_sha256": depth_sha,
        "floor_position": pair_tools.json_vector(candidate["_position"]),
        "yaw_rad": float(candidate["_yaw"]),
        "geodesic_from_a_end_m": float(candidate["query_geodesic_m"]),
        "initial_path_bearing_rad": float(
            candidate["initial_path_bearing_rad"]
        ),
        "max_online_a_covis": float(candidate["max_online_a_covis"]),
        "max_online_a_covis_frame": candidate["max_online_a_covis_frame"],
        "eligible_online_a_frame_floor": int(
            candidate["eligible_online_a_frame_floor"]
        ),
        "covis_curve": list(candidate["_covis_curve"]),
        "construction_support_band": candidate["support_band"],
    }
    if role == "novel":
        record.update({
            "sampling_seed": int(candidate["sampling_seed"]),
            "accepted_proposal_mode": candidate["accepted_proposal_mode"],
            "sampling_diagnostics": candidate["sampling_diagnostics"],
            "assigned_direction_stratum": candidate[
                "assigned_direction_stratum"
            ],
            "initial_path_direction_relative_to_a_end_deg": float(
                candidate["initial_path_direction_relative_to_a_end_deg"]
            ),
            "goal_world_yaw_bin": int(candidate["goal_world_yaw_bin"]),
            "goal_yaw_contract": candidate["goal_yaw_contract"],
            "paired_revisit_separation_m": float(
                candidate["paired_revisit_separation_m"]
            ),
        })
    else:
        record.update({
            "source_online_frame": int(candidate["source_frame"]),
            "translation_from_source_m": float(candidate["translation_m"]),
            "yaw_delta_from_source_deg": float(candidate["yaw_delta_deg"]),
            "pixel_mae_from_source": float(candidate["pixel_mae"]),
            "source_anchor_covis": float(candidate["source_anchor_covis"]),
            "argmax_gap_frames": int(candidate["argmax_gap_frames"]),
            "render_attempt": int(candidate["render_attempt"]),
            "candidate_ranking": list(candidate["ranking"]),
        })
    return record


def write_protocol_episode(
    *,
    destination: Path,
    online_episode: Path,
    receipt: dict,
    history: dict,
    natural: dict[str, Any],
    revisit: dict[str, Any],
    protocol: str,
    scene_rank: int,
    episode_rank: int,
) -> dict[str, Any]:
    scene = str(receipt["scene"])
    episode = str(receipt["episode"])
    pair_id = "pair_00"
    pair_root = destination / pair_id
    novel_assets = _write_goal(pair_root, "novel", natural)
    revisit_assets = _write_goal(pair_root, "revisit", revisit)
    novel_query = _query_record(
        query_id=f"{pair_id}_novel",
        role="novel",
        candidate=natural,
        episode_root=destination,
        rgb_path=novel_assets[0],
        depth_path=novel_assets[1],
        rgb_sha=novel_assets[2],
        depth_sha=novel_assets[3],
    )
    revisit_query = _query_record(
        query_id=f"{pair_id}_revisit",
        role="revisit",
        candidate=revisit,
        episode_root=destination,
        rgb_path=revisit_assets[0],
        depth_path=revisit_assets[1],
        rgb_sha=revisit_assets[2],
        depth_sha=revisit_assets[3],
    )
    distance_error = abs(
        float(novel_query["geodesic_from_a_end_m"])
        - float(revisit_query["geodesic_from_a_end_m"])
    )
    bearing_error = abs(math.degrees(history_tools.wrap_radians(
        float(novel_query["initial_path_bearing_rad"])
        - float(revisit_query["initial_path_bearing_rad"])
    )))
    receipt_path = online_episode / "receipt.json"
    trace_path = online_episode / "online_a_trace.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "episode": episode,
        "online_a_episode": str(online_episode.resolve()),
        "online_a_receipt_sha256": history_tools.sha256_file(receipt_path),
        "online_a_trace_sha256": history_tools.sha256_file(trace_path),
        "online_a_steps": len(history["poses"]),
        "online_a_endpoint": {
            "floor_position": pair_tools.json_vector(online_endpoint(history)[0]),
            "yaw_rad": float(online_endpoint(history)[1]),
        },
        "final14_scene_rank": int(scene_rank),
        "final14_source_episode_rank": int(episode_rank),
        "final14_protocol": protocol,
        "pairs": [{
            "pair_id": pair_id,
            "role_distance_error_m": float(distance_error),
            "role_initial_path_bearing_error_deg": float(bearing_error),
            "queries": [novel_query, revisit_query],
        }],
    }
    destination.mkdir(parents=True, exist_ok=True)
    metadata = destination / "role_pairs.json"
    metadata.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    payload["role_pairs_sha256"] = history_tools.sha256_file(metadata)
    return payload


def construct_history(
    online_episode: Path,
    *,
    scene_rank: int,
    episode_rank: int,
) -> dict[str, Any]:
    receipt = json.loads((online_episode / "receipt.json").read_text())
    history = history_tools.load_online_history(online_episode, receipt)
    camera_height = float(receipt["camera_height_m"])
    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
    try:
        candidates, revisit_diagnostics = search_revisit_candidates(
            simulator,
            history,
            scene=str(receipt["scene"]),
            episode=str(receipt["episode"]),
            camera_height=camera_height,
        )
        natural = None
        natural_diagnostics = None
        natural_error = None
        if candidates["standard"] is not None:
            try:
                natural, natural_diagnostics = sample_natural_novel(
                    simulator,
                    history,
                    scene=str(receipt["scene"]),
                    episode=str(receipt["episode"]),
                    scene_rank=scene_rank,
                    episode_rank=episode_rank,
                    paired_revisit_position=candidates["standard"]["_position"],
                    camera_height=camera_height,
                )
            except RuntimeError as error:
                natural_error = str(error)
    finally:
        simulator.close()
    return {
        "receipt": receipt,
        "history": history,
        "standard": candidates["standard"],
        "hard": candidates["hard"],
        "natural": natural,
        "revisit_diagnostics": revisit_diagnostics,
        "natural_diagnostics": natural_diagnostics,
        "natural_error": natural_error,
    }


def write_manifest(root: Path, payload: dict) -> str:
    root.mkdir(parents=True, exist_ok=True)
    if payload["episodes"]:
        validate_manifest(payload)
    path = root / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    digest = history_tools.sha256_file(path)
    (root / "manifest.json.sha256").write_text(digest + "  manifest.json\n")
    return digest


def build(
    online_root: Path,
    out: Path,
    *,
    scene_rank: int,
    source_episode_order: list[str],
    maximum_histories: int = MAX_HISTORIES_PER_SCENE,
    only_scene: str | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    require(scene_rank >= 0, "scene rank must be non-negative")
    require(maximum_histories > 0, "history cap must be positive")
    require(
        len(source_episode_order) == len(set(source_episode_order)),
        "source episode order contains duplicates",
    )
    online_manifest_path = online_root / "manifest.json"
    online_manifest = json.loads(online_manifest_path.read_text())
    require(
        online_manifest.get("schema_version")
        == "shared_online_a_materialized_v1",
        "unexpected online-A materialization schema",
    )
    source_rank = {
        episode: index for index, episode in enumerate(source_episode_order)
    }
    source_rows = list(online_manifest["episodes"])
    if only_scene is not None:
        source_rows = [
            row for row in source_rows if str(row["scene"]) == only_scene
        ]
    scenes = {str(row["scene"]) for row in source_rows}
    if source_rows:
        require(len(scenes) == 1, "scene builder requires exactly one scene")
        source_scene = next(iter(scenes))
    else:
        # Zero materialized histories is a legitimate, fail-closed Final14
        # outcome: a frozen source scene may have no source episodes, no
        # successful/long-enough online-A trace, or only renderer attrition.
        # Bind the empty fragment to the caller's already validated scene
        # identity instead of inventing a scene from absent episode rows.
        require(
            only_scene is not None,
            "empty scene builder requires an explicit scene identity",
        )
        source_scene = str(only_scene)
    materialized = {str(row["episode"]): row for row in source_rows}
    require(
        set(materialized).issubset(source_rank),
        "materialized history is absent from frozen source order",
    )
    ordered = [
        materialized[episode]
        for episode in source_episode_order
        if episode in materialized
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    rows = {name: [] for name in PROTOCOLS}
    attrition = []
    attempts = []
    retained = 0
    try:
        for source in ordered:
            scene = str(source["scene"])
            episode = str(source["episode"])
            episode_rank = source_rank[episode]
            online_episode = online_root / scene / episode
            result = construct_history(
                online_episode,
                scene_rank=scene_rank,
                episode_rank=episode_rank,
            )
            base_constructible = (
                result["standard"] is not None
                and result["natural"] is not None
            )
            attempt = {
                "scene": scene,
                "episode": episode,
                "scene_rank": int(scene_rank),
                "source_episode_rank": int(episode_rank),
                "standard_constructible": result["standard"] is not None,
                "hard_constructible": result["hard"] is not None,
                "natural_constructible": result["natural"] is not None,
                "base_constructible": base_constructible,
                "retained": False,
                "cap_excluded": False,
                "selected": {
                    "standard": _candidate_json(result["standard"]),
                    "hard": _candidate_json(result["hard"]),
                    "natural": _candidate_json(result["natural"]),
                },
                "revisit_diagnostics": result["revisit_diagnostics"],
                "natural_diagnostics": result["natural_diagnostics"],
                "natural_error": result["natural_error"],
            }
            if not base_constructible:
                missing = []
                if result["standard"] is None:
                    missing.append("standard_revisit")
                if result["natural"] is None:
                    missing.append("natural_novel")
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "standard_natural_constructibility",
                    "reason": "missing:" + ",".join(missing),
                })
                attempts.append(attempt)
                continue
            if retained >= maximum_histories:
                attempt["cap_excluded"] = True
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "per_scene_history_cap",
                    "reason": f"retained_first_{maximum_histories}",
                })
                attempts.append(attempt)
                continue

            attempt["retained"] = True
            retained += 1
            natural_destination = (
                temporary / "natural_direction" / scene / episode
            )
            rows["natural_direction"].append(write_protocol_episode(
                destination=natural_destination,
                online_episode=online_episode,
                receipt=result["receipt"],
                history=result["history"],
                natural=result["natural"],
                revisit=result["standard"],
                protocol="natural_direction",
                scene_rank=scene_rank,
                episode_rank=episode_rank,
            ))
            if result["hard"] is not None:
                hard_destination = temporary / "hard_support" / scene / episode
                rows["hard_support"].append(write_protocol_episode(
                    destination=hard_destination,
                    online_episode=online_episode,
                    receipt=result["receipt"],
                    history=result["history"],
                    natural=result["natural"],
                    revisit=result["hard"],
                    protocol="hard_support",
                    scene_rank=scene_rank,
                    episode_rank=episode_rank,
                ))
            else:
                attrition.append({
                    "scene": scene,
                    "episode": episode,
                    "stage": "hard_support_constructibility",
                    "reason": "no_hard_support_candidate",
                })
            attempts.append(attempt)

        online_sha = history_tools.sha256_file(online_manifest_path)
        for protocol, support in (
            ("natural_direction", "standard"),
            ("hard_support", "hard"),
        ):
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "purpose": (
                    "final14 natural unsupported Novel plus standard-support "
                    "Revisit"
                    if protocol == "natural_direction"
                    else "final14 hard-support Revisit subset; duplicated Novel "
                    "query is instrumentation-only and excluded from hard analysis"
                ),
                "source_online_root": str(online_root.resolve()),
                "source_online_manifest_sha256": online_sha,
                "construction_seed": 20260817,
                "contract": role_contract(support=support),
                "episodes": rows[protocol],
            }
            write_manifest(temporary / protocol, manifest)

        receipt = {
            "schema_version": SCENE_BUILD_SCHEMA,
            "protocol": "FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md",
            "scope": "construction only; no query policy outcome read",
            "scene_rank": int(scene_rank),
            "source_scene": source_scene,
            "source_episode_order": source_episode_order,
            "source_materialized_histories": len(ordered),
            "all_materialized_histories_attempted": len(attempts) == len(ordered),
            "maximum_retained_histories_per_scene": int(maximum_histories),
            "retained_standard_natural_histories": len(
                rows["natural_direction"]
            ),
            "retained_hard_support_histories": len(rows["hard_support"]),
            "attrition_count": len(attrition),
            "attrition": attrition,
            "attempts": attempts,
            "policy_outcomes_read": False,
            "final14_runtime_role_visibility": "none",
        }
        (temporary / "construction_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene-rank", type=int, required=True)
    parser.add_argument(
        "--source-episode-order",
        required=True,
        help="comma-separated frozen source episode order, including failures",
    )
    parser.add_argument(
        "--max-histories-per-scene",
        type=int,
        default=MAX_HISTORIES_PER_SCENE,
    )
    parser.add_argument(
        "--only-scene",
        help="consumed-fixture filter; formal per-scene roots need no filter",
    )
    args = parser.parse_args()
    source_order = [
        value.strip() for value in args.source_episode_order.split(",")
        if value.strip()
    ]
    result = build(
        args.online_root,
        args.out,
        scene_rank=args.scene_rank,
        source_episode_order=source_order,
        maximum_histories=args.max_histories_per_scene,
        only_scene=args.only_scene,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "source_materialized_histories",
            "retained_standard_natural_histories",
            "retained_hard_support_histories",
            "attrition_count",
            "policy_outcomes_read",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
