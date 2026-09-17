#!/usr/bin/env python3
"""Recompute saved physics probe measurements without importing the runners."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    if not np.allclose(a, b, rtol=0, atol=1e-9):
        raise AssertionError(f"measurement mismatch: {a} != {b}")


def primitives(root):
    manifest, report = read(root / "manifest.json"), read(root / "summary.json")
    assert hashlib.sha256((root / "probe_source.py").read_bytes()).hexdigest() == manifest["source_sha256"]
    specs = {v["name"]: v for v in manifest["cases"]}
    for summary in report["cases"]:
        name = summary["case"]
        trace = read(root / f"{name}_trace.json")
        initial, rows = trace["initial"], trace["rows"]
        spec = specs[name]
        dt = manifest["physics_dt_s"]
        assert len(rows) == round(spec["seconds"] / dt)
        close([r["elapsed_s"] for r in rows], np.arange(1, len(rows)+1)*dt)
        positions = np.array([r["position"] for r in rows])
        p0 = np.array(initial["position"])
        close(summary["actual_displacement"], positions[-1]-p0)
        obstacle = [c for r in rows for c in r["contacts"] if c["obstacle"]]
        penetration = max([max(0, -c["distance_m"]) for c in obstacle], default=0)
        tilt = max(r["tilt_deg"] for r in rows)
        close(summary["max_obstacle_penetration_m"], penetration)
        close(summary["max_tilt_deg"], tilt)
        assert summary["obstacle_contact_substeps"] == sum(any(c["obstacle"] for c in r["contacts"]) for r in rows)
        checks = dict(finite=bool(np.isfinite(positions[-1]).all()),
                      no_excess_tilt=tilt < 5, penetration_below_2cm=penetration < .02)
        if name == "free_fall":
            expected = .5*9.81*spec["seconds"]**2
            close(summary["expected_drop_m"], expected)
            checks["gravity_active"] = abs(p0[1]-positions[-1,1]-expected) < .025
        elif name in ("straight", "arc"):
            v, w, t, y = spec["speed"], spec["yaw_rate"], spec["seconds"], initial["yaw_rad"]
            if w:
                delta = np.array([v/w*(math.cos(y+w*t)-math.cos(y)),
                                  -v/w*(math.sin(y+w*t)-math.sin(y))])
            else:
                delta = np.array([-v*t*math.sin(y), -v*t*math.cos(y)])
            error = np.linalg.norm((positions[-1]-p0)[[0,2]]-delta)
            close(summary["endpoint_error_m"], error)
            checks["free_motion_agrees"] = error < max(.02, .05*v*t)
            checks["yaw_agrees"] = abs(rows[-1]["yaw_rad"]-y-w*t) < math.radians(3)
        elif name == "wall":
            checks.update(obstacle_contact_seen=bool(obstacle),
                          wall_not_crossed=positions[:,2].min() >= -1.17,
                          no_spurious_sideways_escape=np.abs(positions[:,0]-p0[0]).max() < .03)
        elif name == "side_post":
            checks["off_axis_body_contact_seen"] = bool(obstacle)
        else:
            displacement = np.linalg.norm((positions[-1]-positions[round(2/dt)-1])[[0,2]])
            close(summary["post_stop_planar_displacement_m"], displacement)
            checks["stop_obeyed"] = displacement < .02
        assert checks == summary["checks"], name
        assert bool(all(checks.values())) == summary["passed"]
    assert report["all_passed"] == all(r["passed"] for r in report["cases"])
    return {"folder": str(root), "cases": len(report["cases"]), "all_passed": report["all_passed"]}


def source_receipts(root):
    manifest = read(root / "manifest.json")
    for source, expected in (manifest.get("sources") or manifest["source_files"]).items():
        snapshot = root / Path(source).name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected


def scenes(root):
    source_receipts(root)
    report = read(root / "summary.json")
    for result in report["results"]:
        folder = root / result["scene"]
        rows = read(folder / "trace.json")
        settle = [r for r in rows if r["phase"] == "settle"]
        forward = [r for r in rows if r["phase"] == "forward"]
        assert len(settle) == 480 and len(forward) == 192
        initial = np.array(result["initial"]["position"])
        p0, p1 = np.array(settle[-1]["position"]), np.array(forward[-1]["position"])
        close(result["settling_displacement_m"], p0-initial)
        close(result["forward_displacement_m"], p1-p0)
        close(result["camera_y_change_m"], p0[1]-.75-result["old_navmesh_floor_y"])
        pictures = [np.asarray(Image.open(folder / name), dtype=float)
                    for name in ("settled_camera_pose.png", "legacy_camera_pose.png")]
        close(result["rgb_mae_due_to_settling"], np.abs(pictures[0]-pictures[1]).mean())
        checks = dict(finite=bool(np.isfinite(p1).all()),
                      settled_upright=settle[-1]["tilt_deg"] < 5,
                      forward_upright=max(r["tilt_deg"] for r in forward) < 5,
                      settled_planar_shift_below_2cm=np.linalg.norm((p0-initial)[[0,2]]) < .02,
                      settled_vertical_velocity_below_1cmps=abs(settle[-1]["linear_velocity"][1]) < .01,
                      settled_on_mesh=bool(settle[-1]["contacts"]))
        assert checks == result["checks"]
        assert all(checks.values()) == result["physical_start_passed"]
    return {"folder": str(root), "scenes": len(report["results"]),
            "physical_starts_passed": sum(r["physical_start_passed"] for r in report["results"])}


def mpc(root):
    source_receipts(root)
    report = read(root / "summary.json")
    for result in report["results"]:
        rows = read(root / (result["case"] + "_trace.json"))
        ref, last = np.array(result["reference_mpc_xy"]), rows[-1]["after"]["position"]
        error = np.linalg.norm(np.array([last[0], -last[2]])-ref[-1])
        close(error, result["endpoint_error_m"])
        assert len(rows) == result["solver_calls"]
        for r in rows:
            close([r["v_mps"], r["yaw_rps"]], r["predicted_controls"][1])
            assert -1e-6 <= r["v_mps"] <= .376+1e-6 and abs(r["yaw_rps"]) <= math.pi/4+1e-6
        seen = [c for r in rows for c in r["obstacle_contacts"]]
        assert len(seen) == result["obstacle_contact_records"]
        close(result["max_penetration_m"], max([max(0,-c["distance_m"]) for c in seen], default=0))
        close(result["solve_median_ms"], 1000*np.median([r["solve_s"] for r in rows]))
        if result["case"] == "reference_through_wall":
            assert seen and all(r["after"]["position"][2] >= -1.17 for r in rows)
        else:
            assert error < .15
    return {"folder": str(root), "cases": len(report["results"]), "all_passed": report["all_passed"],
            "note": "MPC endpoint, command bounds and contacts recomputed; substep tilt not independently stored"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = {"verified": True, "primitive": primitives(args.root / "primitive_v3"),
              "scene_starts": [scenes(args.root / name) for name in ("scene_starts_v1", "scene_starts_v2")],
              "mpc": mpc(args.root / "mpc_tracking_v1"), "navigation_SR": False,
              "scope": "independent arithmetic/source receipt verification; not new physics execution"}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
