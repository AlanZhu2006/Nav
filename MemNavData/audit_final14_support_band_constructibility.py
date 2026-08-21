#!/usr/bin/env python3
"""Consumed-scene constructibility smoke for the frozen final14 support bands.

The audit renders no final14 scene and runs no navigation policy.  It searches
already-consumed actual-online histories with the wider controlled-pose grid
specified by ``FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md`` and
reports whether standard and hard Revisit queries are geometrically feasible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import build_shared_online_double_revisit as history_tools
from generate_twoleg import covis_curve, covis_frac, render


SCHEMA = "final14_support_band_constructibility_smoke_v1_20260816"


def classify_support(max_covis: float, argmax_gap: int) -> str | None:
    value = float(max_covis)
    gap = int(argmax_gap)
    if 0.55 <= value <= 0.90 and gap <= 24:
        return "standard"
    if 0.25 <= value < 0.55 and gap <= 32:
        return "hard"
    return None


def deterministic_grid(identity: str) -> list[tuple[float, float, float]]:
    """Return (radius, world direction, yaw offset degrees)."""

    phase = (
        int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
        / float(2**32)
        * 2.0
        * math.pi
    )
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


def source_frames(simulator, history: dict, *, maximum: int) -> list[dict[str, Any]]:
    endpoint = history["floor_positions"][-1]
    rows = []
    rejected = 0
    for frame in range(39, len(history["poses"]) - 16, 8):
        try:
            distance = history_tools.goal_distance(
                simulator.pathfinder, endpoint, history["floor_positions"][frame]
            )
        except RuntimeError:
            rejected += 1
            continue
        if 2.0 <= distance <= 9.0:
            rows.append(
                {
                    "frame": int(frame),
                    "source_geodesic_m": float(distance),
                    "target_distance_error_m": abs(float(distance) - 3.0),
                }
            )
    rows.sort(key=lambda row: (row["target_distance_error_m"], row["frame"]))
    return rows[: int(maximum)]


def audit_episode(episode_root: Path, *, max_source_frames: int) -> dict[str, Any]:
    receipt = json.loads((episode_root / "receipt.json").read_text())
    history = history_tools.load_online_history(episode_root, receipt)
    camera_height = float(receipt["camera_height_m"])
    endpoint = history["floor_positions"][-1]
    simulator = history_tools.make_sim(
        str(receipt["source_asset"]), "", agent_radius=0.30
    )
    candidates = {"standard": None, "hard": None}
    diagnostics = {
        "grid_attempts": 0,
        "pose_rejects": 0,
        "visual_prefilter_rejects": 0,
        "query_distance_rejects": 0,
        "support_rejects": 0,
        "fully_scored": 0,
    }
    sources = []
    started = time.monotonic()
    try:
        sources = source_frames(
            simulator, history, maximum=int(max_source_frames)
        )
        for source in sources:
            frame = int(source["frame"])
            source_floor = history["floor_positions"][frame]
            source_yaw = float(history["poses"][frame]["yaw"])
            source_rgb = history["rgbs"][frame]
            source_depth = history["depths"][frame]
            source_transform = history["transforms"][frame]
            seen = set()
            identity = f"{receipt['scene']}/{receipt['episode']}/{frame}"
            for attempt, (radius, direction, yaw_offset) in enumerate(
                deterministic_grid(identity), start=1
            ):
                diagnostics["grid_attempts"] += 1
                raw = source_floor + np.asarray(
                    [radius * math.cos(direction), 0.0, radius * math.sin(direction)]
                )
                snapped = np.asarray(simulator.pathfinder.snap_point(raw), dtype=float)
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
                    [0.0, camera_height, 0.0]
                )
                rgb, depth = render(simulator, camera_position, yaw)
                points = history_tools.goal_world_points(
                    depth, camera_position, yaw
                )
                anchor_covis = float(
                    covis_frac(points, source_transform, source_depth)
                )
                pixel_mae = history_tools.pixel_mae(rgb, source_rgb)
                if anchor_covis < 0.10 or pixel_mae < 5.0:
                    diagnostics["visual_prefilter_rejects"] += 1
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
                curve = covis_curve(
                    points, history["transforms"], history["depths"]
                )
                eligible = curve[39:]
                best_frame = 39 + int(np.argmax(eligible))
                best_covis = float(curve[best_frame])
                gap = abs(best_frame - frame)
                diagnostics["fully_scored"] += 1
                band = classify_support(best_covis, gap)
                yaw_delta = history_tools.angle_delta_degrees(yaw, source_yaw)
                if band == "standard" and not (
                    0.20 <= translation <= 0.80 and 12.0 <= yaw_delta <= 45.0
                ):
                    band = None
                if band == "hard" and not (
                    0.30 <= translation <= 1.00 and 18.0 <= yaw_delta <= 60.0
                ):
                    band = None
                if band is None:
                    diagnostics["support_rejects"] += 1
                    continue
                record = {
                    "source_frame": frame,
                    "render_attempt": int(attempt),
                    "translation_m": translation,
                    "yaw_delta_deg": yaw_delta,
                    "source_anchor_covis": anchor_covis,
                    "max_online_a_covis": best_covis,
                    "max_online_a_covis_frame": best_frame,
                    "argmax_gap_frames": gap,
                    "pixel_mae": pixel_mae,
                    "query_geodesic_m": float(query_distance),
                }
                target = 0.72 if band == "standard" else 0.40
                ranking = (
                    abs(best_covis - target),
                    abs(float(query_distance) - 3.0),
                    -(len(history["poses"]) - 1 - frame),
                    frame,
                    attempt,
                )
                current = candidates[band]
                if current is None or ranking < tuple(current["ranking"]):
                    candidates[band] = {**record, "ranking": list(ranking)}
                if candidates["standard"] is not None and candidates["hard"] is not None:
                    break
            if candidates["standard"] is not None and candidates["hard"] is not None:
                break
    finally:
        simulator.close()
    return {
        "scene": str(receipt["scene"]),
        "episode": str(receipt["episode"]),
        "online_a_steps": len(history["poses"]),
        "source_frames_considered": sources,
        "standard_constructible": candidates["standard"] is not None,
        "hard_constructible": candidates["hard"] is not None,
        "selected": candidates,
        "diagnostics": diagnostics,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--max-source-frames", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.online_root / "manifest.json").read_text())
    if manifest.get("schema_version") != "shared_online_a_materialized_v1":
        raise ValueError("unexpected online-A manifest")
    rows = []
    for index, source in enumerate(manifest["episodes"]):
        episode_root = args.online_root / source["scene"] / source["episode"]
        print(
            f"[audit {index + 1}/{len(manifest['episodes'])}] "
            f"{source['scene']}/{source['episode']}",
            flush=True,
        )
        row = audit_episode(
            episode_root, max_source_frames=args.max_source_frames
        )
        rows.append(row)
        print(
            f"  standard={row['standard_constructible']} "
            f"hard={row['hard_constructible']} "
            f"scored={row['diagnostics']['fully_scored']} "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    report = {
        "schema_version": SCHEMA,
        "scope": "consumed scenes only; constructibility without policy rollout",
        "online_root": str(args.online_root.resolve()),
        "protocol": "FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md",
        "final14_accessed": False,
        "policy_rollout": False,
        "histories": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "standard_constructible": sum(row["standard_constructible"] for row in rows),
        "hard_constructible": sum(row["hard_constructible"] for row in rows),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "histories", "scenes", "standard_constructible", "hard_constructible"
    )}, indent=2))


if __name__ == "__main__":
    main()
