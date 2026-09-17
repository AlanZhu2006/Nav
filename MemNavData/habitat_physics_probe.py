#!/usr/bin/env python3
"""Local Bullet execution checks. No NavMesh filtering, policy, or robot I/O.

The dynamic cylinder matches the old navigation envelope, not Dingo/Go2
actuation. Vertical velocity is preserved; position is set only at reset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import habitat_sim
import magnum as mn
import numpy as np


RADIUS = 0.30
HEIGHT = 1.50
DT = 1.0 / 240.0
COMMAND_DT = 0.10
SUBSTEPS = 24
CASES = (
    dict(name="free_fall", seconds=0.35, speed=0.0, yaw_rate=0.0),
    dict(name="straight", seconds=4.0, speed=0.376, yaw_rate=0.0),
    dict(name="arc", seconds=4.0, speed=0.25, yaw_rate=0.30),
    dict(name="wall", seconds=8.0, speed=0.376, yaw_rate=0.0),
    dict(name="side_post", seconds=8.0, speed=0.376, yaw_rate=0.0),
    dict(name="stop", seconds=4.0, speed=0.376, yaw_rate=0.0),
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_sim(physics_config, *, scene="NONE", render=False):
    if not habitat_sim.built_with_bullet:
        raise RuntimeError("This interpreter does not contain Bullet")
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene)
    cfg.enable_physics = True
    cfg.physics_config_file = str(physics_config)
    cfg.create_renderer = bool(render)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    if render:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [360, 640]
        sensor.hfov = 70
        agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    if sim.get_physics_simulation_library() != habitat_sim.physics.PhysicsSimulationLibrary.Bullet:
        sim.close()
        raise RuntimeError("Physics backend is not Bullet")
    return sim


def primitive(sim, kind, name, scale, position, *, dynamic=False, visible=True):
    manager = sim.get_object_template_manager()
    handles = manager.get_template_handles(kind)
    if not handles:
        raise RuntimeError(f"Missing primitive {kind}")
    attrs = manager.get_template_by_handle(handles[0])
    attrs.scale = mn.Vector3(*scale)
    attrs.mass = 20.0 if dynamic else 1.0
    attrs.margin = 0.001
    attrs.friction_coefficient = 0.5
    attrs.restitution_coefficient = 0.0
    attrs.linear_damping = 0.0
    attrs.angular_damping = 0.0
    attrs.is_collidable = True
    attrs.is_visibile = bool(visible)
    template_id = manager.register_template(attrs, name)
    body = sim.get_rigid_object_manager().add_object_by_template_id(template_id)
    # Habitat STATIC objects ignore pose setters. Place first, freeze second.
    body.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    body.translation = mn.Vector3(*position)
    body.rotation = mn.Quaternion()
    body.motion_type = (habitat_sim.physics.MotionType.DYNAMIC if dynamic
                        else habitat_sim.physics.MotionType.STATIC)
    if not np.allclose(list(body.translation), position, rtol=0, atol=1e-6):
        raise RuntimeError(f"Object initialization did not place {name}: {body.translation}")
    return body


def proxy(sim, floor_position=(0.0, 0.0, 0.0), *, visible=True):
    p = np.asarray(floor_position, dtype=float) + [0, HEIGHT / 2 + 0.005, 0]
    return primitive(sim, "cylinderSolid", "audit_dynamic_cylinder",
                     (RADIUS, HEIGHT / 2, RADIUS), p, dynamic=True, visible=visible)


def pose(body):
    rotation = body.rotation
    forward = rotation.transform_vector(mn.Vector3(0, 0, -1))
    up = rotation.transform_vector(mn.Vector3(0, 1, 0))
    return {
        "position": list(body.translation),
        "quaternion_xyzw": list(rotation.vector) + [float(rotation.scalar)],
        "yaw_rad": float(math.atan2(-forward.x, -forward.z)),
        "tilt_deg": float(math.degrees(math.acos(np.clip(up.y, -1, 1)))),
        "linear_velocity": list(body.linear_velocity),
        "angular_velocity": list(body.angular_velocity),
    }


def drive_substep(sim, body, speed, yaw_rate):
    """Ideal velocity servo on a dynamic body; no position/rotation correction.

    Horizontal speed and yaw rate are imposed. Vertical speed is NOT zeroed:
    gravity and floor contacts remain active. This is not wheel-torque control.
    """
    yaw = pose(body)["yaw_rad"]
    previous_y_speed = float(body.linear_velocity.y)
    body.linear_velocity = mn.Vector3(
        -float(speed) * math.sin(yaw), previous_y_speed,
        -float(speed) * math.cos(yaw))
    body.angular_velocity = mn.Vector3(0.0, float(yaw_rate), 0.0)
    sim.step_physics(DT)


def contacts(sim, robot_id, obstacle_ids):
    result = []
    for c in sim.get_physics_contact_points():
        ids = {c.object_id_a, c.object_id_b}
        if robot_id not in ids:
            continue
        result.append({
            "object_a": int(c.object_id_a), "object_b": int(c.object_id_b),
            "obstacle": bool(ids & obstacle_ids),
            "distance_m": float(c.contact_distance),
            "normal_force_n": float(c.normal_force),
            "point_a": list(c.position_on_a_in_ws),
            "point_b": list(c.position_on_b_in_ws),
        })
    return result


def set_observer(sim):
    eye, target = mn.Vector3(3.5, 4.5, 4), mn.Vector3(0, 0.4, -1)
    transform = mn.Matrix4.look_at(eye, target, mn.Vector3.y_axis())
    state = habitat_sim.AgentState()
    state.position = np.array(eye)
    from habitat_sim.utils.common import quat_from_magnum
    state.rotation = quat_from_magnum(mn.Quaternion.from_matrix(transform.rotation()))
    sim.get_agent(0).set_state(state)


def run_case(case, out, physics_config, video):
    name = case["name"]
    sim = make_sim(physics_config, render=video)
    writer = None
    rows = []
    try:
        floor = primitive(sim, "cubeSolid", "audit_floor", (5, .1, 5), (0, -.1, 0))
        body = proxy(sim, (0, 1.25 if name == "free_fall" else 0, 0))
        obstacle_ids = set()
        if name == "wall":
            obstacle = primitive(sim, "cubeSolid", "audit_wall", (2, 1.0, .05), (0, 1.0, -1.5))
            obstacle_ids.add(obstacle.object_id)
        elif name == "side_post":
            obstacle = primitive(sim, "cubeSolid", "audit_thin_post", (.025, .50, .025), (.24, .50, -1.2))
            obstacle_ids.add(obstacle.object_id)
        dimensions = list(body.aabb.size())
        if not np.allclose(dimensions, [2*RADIUS, HEIGHT, 2*RADIUS], atol=.015):
            raise RuntimeError(f"Unexpected proxy dimensions: {dimensions}")
        if name != "free_fall":
            for _ in range(240):
                sim.step_physics(DT)
            body.linear_velocity = mn.Vector3(0, 0, 0)
            body.angular_velocity = mn.Vector3(0, 0, 0)
        initial = pose(body)
        if video:
            import imageio.v2 as imageio
            set_observer(sim)
            writer = imageio.get_writer(str(out / f"{name}.mp4"), fps=10,
                                        codec="libx264", macro_block_size=2)
            writer.append_data(sim.get_sensor_observations()["rgb"][..., :3])
        count = round(case["seconds"] / DT)
        for index in range(count):
            t = index * DT
            speed = case["speed"] if name != "stop" or t < 2.0 else 0.0
            if name == "free_fall":
                sim.step_physics(DT)
            else:
                drive_substep(sim, body, speed, case["yaw_rate"])
            row = {"step": index + 1, "elapsed_s": (index + 1)*DT,
                   "command_speed_mps": speed, "command_yaw_rps": case["yaw_rate"],
                   **pose(body), "contacts": contacts(sim, body.object_id, obstacle_ids)}
            rows.append(row)
            if writer is not None and (index+1) % SUBSTEPS == 0:
                writer.append_data(sim.get_sensor_observations()["rgb"][..., :3])
        p0, p1 = np.array(initial["position"]), np.array(rows[-1]["position"])
        obstacle_contacts = [c for r in rows for c in r["contacts"] if c["obstacle"]]
        result = {
            "case": name, "initial": initial, "final": rows[-1],
            "proxy_dimensions_m": dimensions, "floor_id": floor.object_id,
            "robot_id": body.object_id, "obstacle_ids": sorted(obstacle_ids),
            "actual_displacement": (p1-p0).tolist(),
            "max_tilt_deg": max(r["tilt_deg"] for r in rows),
            "obstacle_contact_substeps": sum(any(c["obstacle"] for c in r["contacts"]) for r in rows),
            "max_obstacle_penetration_m": max([max(0, -c["distance_m"]) for c in obstacle_contacts], default=0),
        }
        checks = {"finite": bool(np.isfinite(p1).all()),
                  "no_excess_tilt": result["max_tilt_deg"] < 5.0,
                  "penetration_below_2cm": result["max_obstacle_penetration_m"] < .02}
        if name == "free_fall":
            expected = .5 * 9.81 * case["seconds"]**2
            result["expected_drop_m"] = expected
            checks["gravity_active"] = abs(float(p0[1]-p1[1])-expected) < .025
        elif name in ("straight", "arc"):
            v, w, duration = case["speed"], case["yaw_rate"], case["seconds"]
            ideal = np.array([v/w*(math.cos(w*duration)-1), -v/w*math.sin(w*duration)]) if w else np.array([0, -v*duration])
            yaw0 = initial["yaw_rad"]
            rotation0 = np.array([[math.cos(yaw0), math.sin(yaw0)],
                                  [-math.sin(yaw0), math.cos(yaw0)]])
            ideal = rotation0 @ ideal
            error = float(np.linalg.norm((p1-p0)[[0, 2]]-ideal))
            result.update(ideal_planar_displacement=ideal.tolist(), endpoint_error_m=error)
            checks["free_motion_agrees"] = error < max(.02, .05*v*duration)
            checks["yaw_agrees"] = abs(rows[-1]["yaw_rad"]-initial["yaw_rad"]-w*duration) < math.radians(3)
        elif name == "wall":
            checks["obstacle_contact_seen"] = bool(obstacle_contacts)
            checks["wall_not_crossed"] = min(r["position"][2] for r in rows) >= -1.17
            checks["no_spurious_sideways_escape"] = max(abs(r["position"][0]-p0[0]) for r in rows) < .03
        elif name == "side_post":
            checks["off_axis_body_contact_seen"] = bool(obstacle_contacts)
        elif name == "stop":
            before_stop = np.array(rows[round(2.0/DT)-1]["position"])
            result["post_stop_planar_displacement_m"] = float(np.linalg.norm((p1-before_stop)[[0, 2]]))
            checks["stop_obeyed"] = result["post_stop_planar_displacement_m"] < .02
        checks = {key: bool(value) for key, value in checks.items()}
        result["checks"] = checks
        result["passed"] = all(checks.values())
        save(out / f"{name}_trace.json", {"initial": initial, "rows": rows})
        save(out / f"{name}_summary.json", result)
        print(json.dumps({k: result[k] for k in ("case", "passed", "checks", "actual_displacement", "max_tilt_deg", "obstacle_contact_substeps")}), flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    cfg = args.out / "audit.physics_config.json"
    save(cfg, {"physics simulator": "bullet", "timestep": DT, "gravity": [0, -9.81, 0],
               "friction coefficient": .5, "restitution coefficient": 0.0})
    manifest = {
        "schema": "habitat_bullet_execution_probe_v1", "cases": CASES,
        "habitat_version": habitat_sim.__version__, "built_with_bullet": habitat_sim.built_with_bullet,
        "source_sha256": sha(__file__), "physics_dt_s": DT,
        "robot_proxy": {"shape": "dynamic cylinder", "radius_m": RADIUS, "height_m": HEIGHT,
                        "mass_kg": 20, "not_Dingo_or_Go2": True},
        "actuation": "ideal horizontal/yaw velocity servo; gravity velocity retained; no pose correction",
        "navmesh_used_for_motion": False, "model_loaded": False, "navigation_SR": False,
    }
    save(args.out / "manifest.json", manifest)
    started = time.monotonic()
    results = []
    try:
        for case in CASES:
            results.append(run_case(case, args.out, cfg.resolve(), args.video))
        result = {"completed": True, "all_passed": all(r["passed"] for r in results),
                  "wall_s": time.monotonic()-started, "cases": results,
                  "scope": "primitive dynamic execution probe, not navigation or real robot validation"}
        save(args.out / "summary.json", result)
        print("PROBE_COMPLETE all_passed=" + str(result["all_passed"]), flush=True)
    except BaseException as error:
        save(args.out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
