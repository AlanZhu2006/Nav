#!/usr/bin/env python3
"""Published NavDP MPC -> dynamic Bullet proxy, fixed reference paths only.

This tests the tracking/physics interface, not image-goal navigation SR. MPC
receives reference positions and ideal odometry, never obstacle or NavMesh data.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import magnum as mn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_physics_probe import (
    DT, SUBSTEPS, make_sim, primitive, proxy, pose, drive_substep, contacts, save, sha,
)


def run_case(name, controller_class, config, out):
    sim = make_sim(config)
    try:
        primitive(sim, "cubeSolid", "mpc_floor", (5, .1, 5), (0, -.1, 0))
        body = proxy(sim)
        obstacles = set()
        if name == "reference_through_wall":
            wall = primitive(sim, "cubeSolid", "mpc_wall", (2, 1, .05), (0, 1, -1.5))
            obstacles.add(wall.object_id)
        for _ in range(240):
            sim.step_physics(DT)
        initial = pose(body)
        origin = np.array([initial["position"][0], -initial["position"][2]])
        if name in ("straight", "reference_through_wall"):
            ref = np.column_stack([np.zeros(24), np.linspace(0, 2 if obstacles else 1, 24)])
        else:
            angle = np.linspace(0, 1, 24)
            ref = np.column_stack([np.cos(angle)-1, np.sin(angle)])
            if name == "right_arc":
                ref[:, 0] *= -1
        ref += origin
        started = time.monotonic()
        mpc = controller_class(ref, N=15, desired_v=.376, v_max=.376, w_max=math.pi/4)
        construction_s = time.monotonic()-started
        rows, all_contacts = [], []
        max_tilt, max_penetration = 0, 0
        for index in range(80 if obstacles else 50):
            state = pose(body)
            # Habitat: -Z forward. MPC: x/y plane, heading from +x.
            x0 = np.array([state["position"][0], -state["position"][2],
                           math.pi/2 + state["yaw_rad"]])
            started = time.monotonic()
            u, prediction = mpc.solve(x0)
            solve_s = time.monotonic()-started
            # Match the published base NavDP evaluator's applied horizon index.
            v, w = map(float, u[1])
            step_contacts = []
            for _ in range(SUBSTEPS):
                drive_substep(sim, body, v, w)
                seen = contacts(sim, body.object_id, obstacles)
                step_contacts.extend(c for c in seen if c["obstacle"])
                max_tilt = max(max_tilt, pose(body)["tilt_deg"])
            if step_contacts:
                max_penetration = max(max_penetration, max(-c["distance_m"] for c in step_contacts))
            all_contacts.extend(step_contacts)
            rows.append({"action": index, "before": state, "after": pose(body),
                         "v_mps": v, "yaw_rps": w, "solve_s": solve_s,
                         "predicted_controls": u.tolist(), "predicted_states": prediction.tolist(),
                         "obstacle_contacts": step_contacts})
        final = pose(body)
        actual = np.array([final["position"][0], -final["position"][2]])
        endpoint_error = float(np.linalg.norm(actual-ref[-1]))
        checks = {
            "finite": bool(np.isfinite(actual).all()),
            "command_limits": all(-1e-6 <= r["v_mps"] <= .376+1e-6 and abs(r["yaw_rps"]) <= math.pi/4+1e-6 for r in rows),
            "upright": max_tilt < 5,
            "penetration_below_2cm": max_penetration < .02,
        }
        if obstacles:
            checks.update(obstacle_contacts_seen=bool(all_contacts),
                          wall_not_crossed=all(r["after"]["position"][2] >= -1.17 for r in rows))
        else:
            checks["endpoint_error_below_15cm"] = endpoint_error < .15
        result = {"case": name, "checks": checks, "passed": all(checks.values()),
                  "initial": initial, "final": final, "reference_mpc_xy": ref.tolist(),
                  "endpoint_error_m": endpoint_error, "max_tilt_deg": max_tilt,
                  "max_penetration_m": max_penetration,
                  "obstacle_contact_records": len(all_contacts),
                  "solver_calls": len(rows), "construction_s": construction_s,
                  "solve_median_ms": float(1000*np.median([r["solve_s"] for r in rows])),
                  "solve_p95_ms": float(1000*np.percentile([r["solve_s"] for r in rows], 95)),
                  "navigation_SR": False}
        save(out / f"{name}_trace.json", rows)
        save(out / f"{name}_summary.json", result)
        print(name, "passed=", result["passed"], "endpoint_error_m=", endpoint_error, flush=True)
        return result
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "NavDP/utils_tasks/tracking_utils.py"
    spec = importlib.util.spec_from_file_location("published_navdp_tracking", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = args.out / "audit.physics_config.json"
    save(config, {"physics simulator": "bullet", "timestep": DT, "gravity": [0,-9.81,0],
                  "friction coefficient": .5, "restitution coefficient": 0.0})
    files = [Path(__file__), ROOT / "MemNavData/habitat_physics_probe.py", source]
    for p in files:
        (args.out / p.name).write_bytes(p.read_bytes())
    save(args.out / "manifest.json", {
        "scope": "MPC -> ideal velocity servo -> Bullet cylinder; no neural policy or navigation SR",
        "sources": {str(p): sha(p) for p in files},
        "mpc": {"N": 15, "T_s": .1, "v_max": .376, "w_max": math.pi/4,
                "applied_control_index": 1, "weights_changed": False},
        "not_official_IsaacSim_Dingo": True, "navmesh_used_for_motion": False,
        "references": "fixed straight / left arc / right arc / line through wall",
        "state_input": "ideal current body x/y/yaw; no obstacle coordinates",
    })
    results = [run_case(name, module.MPC_Controller, config.resolve(), args.out)
               for name in ("straight", "left_arc", "right_arc", "reference_through_wall")]
    save(args.out / "summary.json", {"completed": True, "all_passed": all(r["passed"] for r in results),
                                     "results": results})


if __name__ == "__main__":
    main()
