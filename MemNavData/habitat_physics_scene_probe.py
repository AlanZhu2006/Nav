#!/usr/bin/env python3
"""Check fixed consumed scene starts with Bullet, before loading any policy.

No NavMesh calls. The old floor coordinate is used only to initialize the body.
Actual subsequent position comes from the dynamic object, not a projection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData import generate_twoleg as gen
from MemNavData.habitat_physics_probe import (
    DT, HEIGHT, contacts, drive_substep, pose, proxy, save, sha,
)


def make_scene(glb, config):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(glb)
    cfg.enable_physics = True
    cfg.physics_config_file = str(config)
    camera = habitat_sim.CameraSensorSpec()
    camera.uuid = "color"
    camera.sensor_type = habitat_sim.SensorType.COLOR
    camera.resolution = [gen.H, gen.W]
    camera.hfov = gen.HFOV_DEG
    camera.position = mn.Vector3(0, 0, 0)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [camera]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    if sim.get_physics_simulation_library() != habitat_sim.physics.PhysicsSimulationLibrary.Bullet:
        sim.close()
        raise RuntimeError("Expected Bullet")
    return sim


def rgb(sim, position, yaw):
    sim.get_agent(0).set_state(gen.agent_state(np.asarray(position), yaw))
    return sim.get_sensor_observations()["color"][..., :3].copy()


def run_history(history, out, config, settling):
    receipt = json.loads((Path(history["online_a_episode"]) / "receipt.json").read_text())
    scene = receipt["source_asset"]
    floor = np.array(history["online_a_endpoint"]["floor_position"])
    yaw = history["online_a_endpoint"]["yaw_rad"]
    out.mkdir()
    sim = make_scene(scene, config)
    rows = []
    try:
        legacy_rgb = rgb(sim, floor + [0, .5, 0], yaw)
        Image.fromarray(legacy_rgb).save(out / "legacy_camera_pose.png")
        body = proxy(sim, floor, visible=False)
        body.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
        initial = pose(body)
        for index in range(480):
            if settling == "zero_command":
                drive_substep(sim, body, 0, 0)
            else:
                sim.step_physics(DT)
            rows.append(dict(phase="settle", index=index, **pose(body),
                             contacts=contacts(sim, body.object_id, set())))
        settled = pose(body)
        camera_position = np.array(settled["position"]) + [0, .5 - HEIGHT / 2, 0]
        current_rgb = rgb(sim, camera_position, settled["yaw_rad"])
        Image.fromarray(current_rgb).save(out / "settled_camera_pose.png")
        for index in range(192):
            drive_substep(sim, body, .376, 0)
            rows.append(dict(phase="forward", index=index, **pose(body),
                             contacts=contacts(sim, body.object_id, set())))
        final = pose(body)
        Image.fromarray(rgb(sim, np.array(final["position"]) + [0, .5 - HEIGHT / 2, 0],
                           final["yaw_rad"])).save(out / "after_08s_forward.png")
        shift = np.array(settled["position"]) - np.array(initial["position"])
        physical_floor = np.array(settled["position"]) - [0, HEIGHT/2, 0]
        result = {
            "scene": history["scene"], "asset": scene,
            "settling_mode": settling,
            "initial": initial, "settled": settled, "final": final,
            "settling_displacement_m": shift.tolist(),
            "old_navmesh_floor_y": float(floor[1]),
            "body_base_y_after_settling": float(physical_floor[1]),
            "camera_y_change_m": float(physical_floor[1]-floor[1]),
            "rgb_mae_due_to_settling": float(np.abs(current_rgb.astype(float)-legacy_rgb).mean()),
            "forward_displacement_m": (np.array(final["position"])-np.array(settled["position"])).tolist(),
            "max_post_settling_tilt_deg": max(r["tilt_deg"] for r in rows if r["phase"] == "forward"),
            "contact_substeps": sum(bool(r["contacts"]) for r in rows),
            "settled_contact_count": len(rows[479]["contacts"]),
            "checks": {
                "finite": bool(np.isfinite(list(final["position"])).all()),
                "settled_upright": settled["tilt_deg"] < 5,
                "forward_upright": max(r["tilt_deg"] for r in rows if r["phase"] == "forward") < 5,
                "settled_planar_shift_below_2cm": bool(np.linalg.norm(shift[[0, 2]]) < .02),
                "settled_vertical_velocity_below_1cmps": abs(settled["linear_velocity"][1]) < .01,
                "settled_on_mesh": bool(rows[479]["contacts"]),
            },
            "same_camera_pose_as_legacy": bool(np.linalg.norm(camera_position-(floor+[0,.5,0])) < .001),
            "navigation_SR": False, "policy_loaded": False, "navmesh_queries": 0,
        }
        result["physical_start_passed"] = all(result["checks"].values())
        save(out / "trace.json", rows)
        save(out / "summary.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settling", choices=("passive", "zero_command"), default="passive")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = ROOT / ".diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814/manifest.json"
    histories = json.loads(source.read_text())["episodes"][:4]
    config = args.out / "audit.physics_config.json"
    save(config, {"physics simulator": "bullet", "timestep": DT, "gravity": [0, -9.81, 0],
                  "friction coefficient": .5, "restitution coefficient": 0.0})
    files = [Path(__file__), ROOT / "MemNavData/habitat_physics_probe.py", ROOT / "MemNavData/generate_twoleg.py"]
    save(args.out / "manifest.json", {
        "scope": "consumed 4-history physical start validation; no models or navigation SR",
        "source_manifest": str(source), "source_sha256": sha(source),
        "source_files": {str(p): sha(p) for p in files},
        "habitat_version": habitat_sim.__version__, "built_with_bullet": habitat_sim.built_with_bullet,
        "navigation_proxy": "dynamic cylinder radius 0.30m / height 1.50m; not Dingo",
        "pose_setters": "object initialization only; camera follows measured body",
        "settling_mode": args.settling,
        "image_height_m": .5, "camera_pose_policy": "body center minus 0.25m; yaw only",
        "navmesh_used_for_motion": False, "frozen_history_starts": histories and [h["scene"] for h in histories],
    })
    for path in files:
        (args.out / path.name).write_bytes(path.read_bytes())
    started = time.monotonic()
    results = [run_history(h, args.out / h["scene"], config.resolve(), args.settling) for h in histories]
    save(args.out / "summary.json", {"completed": True, "results": results,
                                      "all_physical_starts_passed": all(r["physical_start_passed"] for r in results),
                                      "wall_s": time.monotonic()-started})


if __name__ == "__main__":
    main()
