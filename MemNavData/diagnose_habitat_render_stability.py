#!/usr/bin/env python3
"""Time repeated Habitat RGB-D renders without loading a navigation model.

This is an infrastructure diagnostic.  It deliberately holds scene, pose,
camera, navmesh settings, container, and Habitat environment fixed while the
Slurm launcher changes only the GPU class.  Results are JSONL so a native
abort still leaves every completed render on disk/stdout.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import time

import numpy as np

from generate_twoleg import make_sim, render


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--position", nargs=3, type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--renders", type=int, default=16)
    parser.add_argument("--yaw-step-deg", type=float, default=1.0)
    args = parser.parse_args()

    if args.renders < 1:
        parser.error("--renders must be positive")
    faulthandler.enable(all_threads=True)
    # A stalled native renderer now leaves Python stacks in the Slurm log.
    faulthandler.dump_traceback_later(30.0, repeat=True)

    digest = hashlib.sha256()
    with open(args.scene, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if actual_sha != args.scene_sha256:
        raise SystemExit(
            f"scene hash changed: expected={args.scene_sha256} actual={actual_sha}"
        )

    started = time.perf_counter()
    simulator = make_sim(args.scene, "", agent_radius=0.30)
    emit(
        {
            "event": "simulator_ready",
            "elapsed_s": time.perf_counter() - started,
            "navigable_area_m2": float(simulator.pathfinder.navigable_area),
            "scene_sha256": actual_sha,
        }
    )

    position = np.asarray(args.position, dtype=np.float64)
    durations = []
    try:
        for index in range(args.renders):
            yaw = float(args.yaw + math.radians(args.yaw_step_deg) * index)
            before = time.perf_counter()
            rgb, depth = render(simulator, position, yaw)
            elapsed = time.perf_counter() - before
            durations.append(elapsed)
            depth_array = np.asarray(depth)
            finite = np.isfinite(depth_array)
            emit(
                {
                    "event": "render",
                    "index": index,
                    "elapsed_s": elapsed,
                    "yaw_rad": yaw,
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "depth_sha256": hashlib.sha256(depth_array.tobytes()).hexdigest(),
                    "depth_nan": int(np.isnan(depth_array).sum()),
                    "depth_inf": int(np.isinf(depth_array).sum()),
                    "depth_zero": int((depth_array == 0).sum()),
                    "depth_finite_min": (
                        float(depth_array[finite].min()) if finite.any() else None
                    ),
                    "depth_finite_max": (
                        float(depth_array[finite].max()) if finite.any() else None
                    ),
                }
            )
    finally:
        simulator.close()
        faulthandler.cancel_dump_traceback_later()

    values = np.asarray(durations, dtype=np.float64)
    emit(
        {
            "event": "complete",
            "renders": len(durations),
            "median_s": float(np.median(values)),
            "p95_s": float(np.quantile(values, 0.95)),
            "max_s": float(values.max()),
        }
    )


if __name__ == "__main__":
    main()
