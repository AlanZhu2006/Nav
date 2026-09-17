#!/usr/bin/env python3
"""Pre-navigation Bullet turn / MPC zero-reference check on a flat primitive."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_bullet_alignment import PhysicalAlignment, wrap
from MemNavData.habitat_physics_probe import (
    make_sim, primitive, proxy, pose, drive_substep, SUBSTEPS, DT, save)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = args.out.resolve() / "physics.json"
    save(config, {"physics simulator": "bullet", "timestep": DT,
                  "gravity": [0, -9.81, 0], "friction coefficient": .5,
                  "restitution coefficient": 0.0})
    spec = importlib.util.spec_from_file_location("official_mpc", ROOT / "NavDP/utils_tasks/tracking_utils.py")
    tracking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracking)
    results = []
    for angle in (-165., 175., 0.):
        sim = make_sim(config)
        try:
            primitive(sim, "cubeSolid", "alignment_floor", (5, .1, 5), (0, -.1, 0))
            body = proxy(sim)
            for _ in range(480):
                drive_substep(sim, body, 0, 0)
            start = pose(body)
            controller = PhysicalAlignment(True)
            radians = math.radians(angle)
            controller.consider({"certified_relocalization_accepted": True, "router_active": True,
                                 "memory_controller_pointgoal": [2.5*math.cos(radians), 2.5*math.sin(radians)],
                                 "certified_relocalization_guidance_mode": "endpoint_bearing"}, start["yaw_rad"], 0)
            samples = []
            with (args.out / f"actions_{int(angle)}.jsonl").open("x") as stream:
                while controller.active or (angle == 0 and len(samples) < 20):
                    before = pose(body)
                    if controller.active:
                        v, w = controller.command(before["yaw_rad"])
                    else:
                        point = np.array([before["position"][0], -before["position"][2]])
                        mpc = tracking.MPC_Controller(np.tile(point, (24, 1)), N=15,
                                                      desired_v=.376, v_max=.376, w_max=math.pi/4)
                        controls, _ = mpc.solve(np.r_[point, math.pi/2+before["yaw_rad"]])
                        v, w = map(float, controls[1])
                    substeps = []
                    for _ in range(SUBSTEPS):
                        drive_substep(sim, body, v, w)
                        substeps.append(pose(body))
                    after = substeps[-1]
                    if controller.active:
                        controller.observe_after_action(after["yaw_rad"], len(samples))
                    row = dict(before=before, after=after, v=v, w=w, substeps=substeps)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    samples.append(row)
            final = pose(body)
            positions = np.array([start["position"]] + [s["after"]["position"] for s in samples])
            path = float(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1).sum())
            error = abs(wrap(final["yaw_rad"]-start["yaw_rad"]-radians))
            tilt = max(p["tilt_deg"] for s in samples for p in s["substeps"])
            passed = path < .02 and error < math.radians(1) and tilt < 5
            if angle == 0:
                passed = passed and max(abs(s["v"]) for s in samples) < .005
            row = dict(requested_turn_deg=angle, commands=len(samples), path_m=path,
                       yaw_error_deg=math.degrees(error), max_tilt_deg=tilt,
                       passed=bool(passed), position_setters_during_execution=0,
                       navmesh_motion_queries=0, low_level_feedback="actual Bullet yaw")
            results.append(row)
            print(json.dumps(row), flush=True)
        finally:
            sim.close()
    save(args.out / "summary.json", {"passed": all(r["passed"] for r in results), "results": results})
    return 0 if all(r["passed"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
