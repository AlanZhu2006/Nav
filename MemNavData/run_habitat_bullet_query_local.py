#!/usr/bin/env python3
"""One consumed query, four arms with actual Bullet movement and private models.

No paper/main executor edits. Not an IsaacSim Dingo reproduction or formal SR.
The old online-A history is shared; physics starts at the query boundary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.run_cec_stream_depth_closed_loop import (
    BENCH, MEM_PY, MEM_CKPT, NAV_CKPT, LINGBOT, hab_env, open_port,
)
from MemNavData.habitat_executor_audit import dump, sha

HERE = Path(__file__).resolve()
BULLET_PY = ROOT / ".diagnostics/habitat_bullet_env_20260908/bin/python"


def evaluate():
    import habitat_sim
    import magnum as mn
    import numpy as np
    import eval_2leg_habitat as base
    from MemNavData import generate_twoleg as gen
    from MemNavData.final14_spl_replay import measurement
    from MemNavData.habitat_executor_audit import commanded_step
    from MemNavData.habitat_physics_probe import DT, HEIGHT, SUBSTEPS, proxy, pose, drive_substep, contacts
    from MemNavData.habitat_bullet_alignment import PhysicalAlignment, install_loop_hook

    if os.environ.get("BULLET_XNAVDP_EXPERIMENT") == "1":
        from MemNavData.habitat_xnavdp_tracking import install_rgb_only_transport
        install_rgb_only_transport(base)

    tracker = os.environ["BULLET_QUERY_TRACKER"]
    if tracker not in ("pure_pursuit", "mpc", "xmpc"):
        raise ValueError(tracker)
    output = Path(base.args.out)
    output.mkdir(parents=True, exist_ok=False)
    # The existing evaluator requires an empty output at entry.
    config = output.with_name(output.name + "_physics_config.json")
    dump(config, {"physics simulator": "bullet", "timestep": DT, "gravity": [0,-9.81,0],
                  "friction coefficient": .5, "restitution coefficient": 0.0})
    original_leg, original_select, original_render = base.run_policy_leg, base.select_plan_trajectory, base.render
    alignment_mode = os.environ.get("BULLET_QUERY_ALIGNMENT", "off")
    if alignment_mode not in ("off", "first_rear_certified"):
        raise ValueError(alignment_mode)
    alignment = PhysicalAlignment(alignment_mode == "first_rear_certified")
    if os.environ.get("BULLET_ALIGNMENT_EXPERIMENT") == "1":
        if tracker not in ("mpc", "xmpc"):
            raise RuntimeError("Physical alignment diagnostics require an MPC")
        original_leg = install_loop_hook(base, original_leg, alignment,
                                        output.with_name(output.name + "_runtime_alignment_leg.py"))
        original_memory = base.srv_memory

        def observed_memory(frame, **kwargs):
            result = original_memory(frame, **kwargs)
            if alignment.active:
                import hashlib
                row = {"next_action": len(actions), "jpg_sha256": hashlib.sha256(frame).hexdigest(),
                       "receipt": result, "executor_motion_fields_sent": sorted(kwargs)}
                with (output / "turn_observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            return result

        base.srv_memory = observed_memory
    body, sim_active, mpc, plan_bytes = None, None, None, None
    actions, all_plans = [], []
    first_rgb = None
    first_rgb_sha = None
    xtracker = None
    if tracker in ("mpc", "xmpc"):
        spec = importlib.util.spec_from_file_location("published_navdp_tracking", ROOT / "NavDP/utils_tasks/tracking_utils.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if tracker == "xmpc":
        from MemNavData.habitat_xnavdp_tracking import XTracker
        xtracker = XTracker()

    def make_sim(glb, navmesh, agent_radius=.30, agent_height=1.5, *, recompute_navmesh=True):
        if not habitat_sim.built_with_bullet:
            raise RuntimeError("Bullet is unavailable")
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = glb
        cfg.enable_physics = True
        cfg.physics_config_file = str(config.resolve())
        specs = []
        for uuid, kind in [("color", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
            sensor = habitat_sim.CameraSensorSpec()
            sensor.uuid, sensor.sensor_type = uuid, kind
            sensor.resolution = [gen.H, gen.W]
            sensor.hfov = gen.HFOV_DEG
            sensor.position = mn.Vector3(0,0,0)
            specs.append(sensor)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = specs
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        if sim.get_physics_simulation_library() != habitat_sim.physics.PhysicsSimulationLibrary.Bullet:
            raise RuntimeError("Physics backend mismatch")
        # This mesh is used only by the existing evaluator's PRE-QUERY distance
        # validation. The step() below does not read it or call any filter.
        if recompute_navmesh:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius, settings.agent_height = agent_radius, agent_height
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError("Evaluation distance mesh could not be constructed")
        elif not sim.pathfinder.load_nav_mesh(navmesh):
            raise RuntimeError("Evaluation distance mesh is missing")
        return sim

    def render(*a, **kw):
        nonlocal first_rgb, first_rgb_sha
        rgb, depth = original_render(*a, **kw)
        if body is not None and first_rgb is None:
            import hashlib
            from PIL import Image
            first_rgb = rgb.copy()
            first_rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            Image.fromarray(rgb).save(output / "first_query_rgb.png")
        return rgb, depth

    def select(*a, **kw):
        trajectory, info = original_select(*a, **kw)
        response, position, yaw = a[:3]
        row = {"plan": len(all_plans), "next_action": len(actions), "position": np.asarray(position).tolist(),
               "yaw": float(yaw), "selected_trajectory": np.asarray(trajectory).tolist(),
               "all_trajectory": np.asarray(response.get("all_trajectory", [])).tolist(),
               "all_values": np.asarray(response.get("all_values", [])).tolist(), "selection": info}
        if os.environ.get("BULLET_ALIGNMENT_EXPERIMENT") == "1":
            row.update({key: response.get(key) for key in (
                "navdp_interface_diagnostic", "memory_controller_pointgoal", "aux_pose", "anchor",
                "certified_relocalization_accepted", "certified_relocalization_certificate",
                "monocular_depth_receipt", "memory_frame_idx", "navdp_stop_evidence",
                "pose_controller", "router_active", "diffusion_seed", "metric_depth_sensor_consumed",
                "xnavdp_input_diagnostic", "rtc_robot_state_used", "xnavdp_rgb_receipt")})
            alignment.consider(response, float(yaw), len(actions))
            dump(output / "alignment_receipt.json", alignment.receipt())
        all_plans.append(row)
        with (output / "plans.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        return trajectory, info

    def step(position, yaw, world_path, _unused_pathfinder):
        nonlocal mpc, plan_bytes
        before = pose(body)
        actual_start = np.array(before["position"]) - [0, HEIGHT/2, 0]
        if not np.allclose(actual_start, position, rtol=0, atol=1e-7):
            raise RuntimeError("Evaluator pose differs from the physical body")
        solve_s = None
        x_receipt = None
        aligning = alignment.active
        if aligning:
            v, w = alignment.command(yaw)
        elif tracker == "pure_pursuit":
            _, commanded_yaw, distance, _ = commanded_step(position, yaw, world_path, base.args)
            v, w = distance/.1, (commanded_yaw-yaw)/.1
        elif (xtracker is not None and all_plans[-1].get("pose_controller") in
              ("navdp_image_point_mix", "xnavdp_point_posttrain")):
            v, w, solve_s, x_receipt = xtracker.command(
                len(all_plans)-1, position, yaw, world_path)
        else:
            ref = np.asarray(world_path, dtype=float) * [1,-1]
            if plan_bytes != ref.tobytes():
                mpc = module.MPC_Controller(ref, N=15, desired_v=.376, v_max=.376, w_max=math.pi/4)
                plan_bytes = ref.tobytes()
            x0 = np.array([position[0], -position[2], math.pi/2+yaw])
            started = time.monotonic()
            controls, _ = mpc.solve(x0)
            solve_s = time.monotonic()-started
            v, w = map(float, controls[1])
        substeps = []
        for _ in range(SUBSTEPS):
            drive_substep(sim_active, body, v, w)
            substeps.append(dict(**pose(body), contacts=contacts(sim_active, body.object_id, set())))
        after = substeps[-1]
        actual_end = np.array(after["position"]) - [0, HEIGHT/2, 0]
        displacement = float(np.linalg.norm((actual_end-np.asarray(position))[[0,2]]))
        row = {"action": len(actions), "plan": len(all_plans)-1, "before": before, "after": after,
               "command_speed_mps": v, "command_yaw_rps": w, "mpc_solve_s": solve_s,
               "actual_translation_m": displacement, "substeps": substeps,
               "navmesh_motion_queries": 0}
        actions.append(row)
        if os.environ.get("BULLET_ALIGNMENT_EXPERIMENT") == "1":
            row["command_kind"] = "physical_alignment" if aligning else "mpc_tracking"
        if xtracker is not None:
            row["xmpc_receipt"] = x_receipt
        if aligning:
            alignment.observe_after_action(after["yaw_rad"], row["action"])
            mpc, plan_bytes = None, None
            dump(output / "alignment_receipt.json", alignment.receipt())
        with (output / "physical_actions.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        if max(p["tilt_deg"] for p in substeps) >= 5:
            raise RuntimeError("Physical proxy exceeded predeclared 5 degree tilt; invalidate rollout")
        if len(actions) % 80 == 0:
            print(f"PHYSICS {tracker}: actions={len(actions)} displacement={displacement:.4f}", flush=True)
        return actual_end, after["yaw_rad"], displacement

    def leg(sim, pf, position, yaw, goal, goal_xz, geo, *a, **kw):
        nonlocal body, sim_active
        sim_active = sim
        body = proxy(sim, position, visible=False)
        body.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
        original_start = pose(body)
        for _ in range(480):
            drive_substep(sim, body, 0, 0)
        start = pose(body)
        displacement = np.array(start["position"])-np.array(original_start["position"])
        if np.linalg.norm(displacement[[0,2]]) >= .02 or start["tilt_deg"] >= 5:
            raise RuntimeError("Physical query initialization failed its predeclared bounds")
        actual_start = np.array(start["position"]) - [0, HEIGHT/2, 0]
        dump(output / "initialization.json", {"original": original_start, "settled": start,
             "history_endpoint": np.asarray(position).tolist(), "actual_query_floor_position": actual_start.tolist(),
             "camera_height_m": .5, "settle_control": "zero velocity, gravity retained"})
        result = original_leg(sim, pf, actual_start, start["yaw_rad"], goal, goal_xz, geo, *a, **kw)
        record = measurement(result, goal_xz, geo)
        record.update(physical_initialization=start, first_query_rgb_sha256=first_rgb_sha,
                      tracker=tracker, execution="Bullet dynamic cylinder", navigation_sample_size=1,
                      scope="query-stage diagnostic; not end-to-end A collection")
        dump(output / "terminal_measurements.json", [record])
        return result

    base.make_sim, base.render, base.select_plan_trajectory = make_sim, render, select
    base.pursuit_step, base.run_policy_leg = step, leg
    try:
        if os.environ.get("BULLET_XNAVDP_EXPERIMENT") == "1":
            from MemNavData.habitat_xnavdp_tracking import run_shared_evaluator
            run_shared_evaluator(output.with_name(output.name + "_runtime_shared_evaluator.py"))
        else:
            runpy.run_path(str(ROOT / "MemNavData/eval_shared_online_role_pairs.py"), run_name="__main__")
    finally:
        dump(output / "physics_summary.json", {"actions": len(actions), "plans": len(all_plans),
             "navmesh_motion_queries": 0, "first_query_rgb_sha256": first_rgb_sha,
             "max_tilt_deg": max([p["tilt_deg"] for r in actions for p in r["substeps"]], default=0)})
        if os.environ.get("BULLET_ALIGNMENT_EXPERIMENT") == "1":
            dump(output / "alignment_receipt.json", alignment.receipt())


def local():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--history-index", type=int, default=0,
                        help="Index in the existing four-history consumed manifest; no resampling")
    parser.add_argument("--trackers", choices=("both", "pure_pursuit", "mpc"), default="both")
    parser.add_argument("--continue-after-invalid-physics", action="store_true",
                        help="Attempt other arms after a recorded tilt violation; never assign that arm an SR")
    parser.add_argument("--alignment-pair", action="store_true",
                        help="Consumed CEC/MPC off vs physical first-rear alignment, with interface receipts")
    parser.add_argument("--xnavdp-pair", action="store_true",
                        help="Base+turn/base MPC vs base/X MPC vs X actor/X MPC, fixed consumed query")
    parser.add_argument("--xnavdp-port", type=int, default=21682)
    parser.add_argument("--xnavdp-smoke", action="store_true",
                        help="Only X arm, eight commands, integration preflight with no SR claim")
    parser.add_argument("--xnavdp-rgb-pair", action="store_true",
                        help="X-only legacy BGR versus corrected RGB; all other contracts fixed")
    parser.add_argument("--memnav-port", type=int, default=21680)
    parser.add_argument("--navdp-port", type=int, default=21681)
    args = parser.parse_args()
    if args.xnavdp_rgb_pair:
        args.xnavdp_pair = True
    histories = json.loads((BENCH / "manifest.json").read_text())["episodes"]
    if not 0 <= args.history_index < len(histories):
        parser.error("history index is outside the existing consumed manifest")
    h = histories[args.history_index]
    out = args.out.resolve()
    ports = [args.memnav_port, args.navdp_port] + ([args.xnavdp_port] if args.xnavdp_pair else [])
    if len(set(ports)) != len(ports) or any(open_port(p) for p in ports):
        raise RuntimeError("Private ports unavailable")
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    source = json.loads((Path(h["online_a_episode"]) / "receipt.json").read_text())
    parquet = Path(source["source_episode"]) / "data/chunk-000/episode_000000.parquet"
    variants = [("native", "pure_pursuit"), ("cec", "pure_pursuit"), ("cec", "mpc"), ("native", "mpc")]
    if args.trackers != "both":
        variants = [(p,t) for p,t in variants if t == args.trackers]
    if args.alignment_pair:
        variants = [("cec", "mpc"), ("cec_aligned", "mpc")]
    if args.xnavdp_pair:
        if args.alignment_pair:
            parser.error("Select exactly one diagnostic pair")
        variants = [("cec_aligned", "mpc"), ("cec", "xmpc"), ("cec_x", "xmpc")]
    if args.xnavdp_rgb_pair:
        variants = [("cec_x_legacy", "xmpc"), ("cec_x", "xmpc")]
        if args.history_index % 2:
            variants.reverse()
    if args.xnavdp_smoke:
        if not args.xnavdp_pair:
            parser.error("X smoke requires --xnavdp-pair")
        variants, args.max_steps = [("cec_x", "xmpc")], 8
    files = [HERE, ROOT / "MemNavData/habitat_physics_probe.py", ROOT / "MemNavData/habitat_executor_audit.py",
             ROOT / "MemNavData/eval_2leg_habitat.py", ROOT / "MemNavData/eval_shared_online_role_pairs.py",
             ROOT / "MemNavData/generate_twoleg.py", ROOT / "NavDP/utils_tasks/tracking_utils.py",
             ROOT / "NavDP/baselines/memnav/policy_agent.py", ROOT / "NavDP/baselines/memnav/memnav_server.py"]
    if args.alignment_pair or args.xnavdp_pair:
        files += [ROOT / "MemNavData/habitat_bullet_alignment.py",
                  ROOT / "MemNavData/navdp_interface_diagnostic_server.py",
                  ROOT / "MemNavData/lingbot_pose_diagnostic_server.py",
                  ROOT / "NavDP/baselines/navdp/policy_network.py",
                  ROOT / "NavDP/baselines/navdp/policy_agent.py",
                  ROOT / "NavDP/baselines/navdp/navdp_server.py"]
    if args.xnavdp_pair:
        from MemNavData.habitat_xnavdp_tracking import OFFICIAL, ACADOS, SOURCE, CONFIG
        files += [ROOT / "MemNavData/habitat_xnavdp_tracking.py",
                  ROOT / "MemNavData/xnavdp_mono_diagnostic_server.py",
                  ROOT / "MemNavData/xnavdp_revisit_server.py", SOURCE,
                  OFFICIAL / "baselines/x-navdp/eval/src/policy_agent.py",
                  OFFICIAL / "baselines/x-navdp/eval/src/policy_network_embodiment.py"]
    evaluator_env = dict(hab_env(), PYTHONPATH=f"{ROOT}:{ROOT}/MemNavData",
                         OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # Do not inherit the old environment's pip-vendored requests workaround.
    # This isolated environment has its own requests and Bullet dependencies.
    with (out / "logs/evaluator_import_preflight.log").open("x") as stream:
        subprocess.run([str(BULLET_PY), "-c",
                        "import habitat_sim, requests, casadi, cv2, pandas as pd, numpy as np; "
                        "assert habitat_sim.built_with_bullet; "
                        "import MemNavData.final14_spl_replay; "
                        "import sys; rows=pd.read_parquet(sys.argv[1]); "
                        "assert np.stack(rows.iloc[0]['observation.camera_intrinsic']).shape==(3,3); "
                        "sys.argv=['preflight','--help']; import eval_2leg_habitat", str(parquet)],
                       cwd=ROOT, env=evaluator_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    receipt = {"scope": "consumed one-history query-stage Bullet tracker/policy diagnostic",
               "history_index": args.history_index,
               "continue_after_invalid_physics": args.continue_after_invalid_physics,
               "scene": h["scene"], "episode": h["episode"], "variants": variants,
               "source_manifest_sha256": sha(BENCH / "manifest.json"),
               "history_sha256": h["online_a_trace_sha256"], "max_steps": args.max_steps,
               "depth_source": "monocular_sidecar", "reference_depth_source": "canonical",
               "frozen_residual_m": 2.5, "execution_horizon": 8, "command_dt_s": .1,
               "motion_navmesh_queries": 0, "navmesh_evaluation_distance_check": True,
               "geometry_proxy": "0.30m radius / 1.50m height cylinder; not Dingo/Go2",
               "initial_state": "same 2s zero-command settling, actual floor + 0.5m camera",
               "not_directly_comparable_to_old_absolute_SR": True,
               "sources": {str(p):sha(p) for p in files},
               "weights": {str(p):sha(p) for p in (MEM_CKPT, NAV_CKPT, LINGBOT / "weights/lingbot-map-long.pt")}}
    if args.alignment_pair:
        receipt.update(alignment_pair=True, alignment_trigger="first certified goal, forward < 0",
                       alignment_max_yaw_rate_rad_s=math.pi/4, alignment_max_commands=80,
                       alignment_tolerance_deg=1.0, alignment_budget_included=True,
                       alignment_observation="LingBot every command; NavDP replay at nominal decision stride",
                       alignment_feedback="physical yaw, same low-level pose source as MPC; no goal GT",
                       original_idealized_alignment_switch="off")
    if args.xnavdp_pair:
        receipt.update(xnavdp_stack_comparison=True, x_mpc_config=CONFIG,
                       x_mpc_execution="published first control, eight-command chunk per neural plan",
                       native_fallback_tracker="same base MPC; original image request",
                       x_actor="official posttrain wheeled; eight samples; goal-only conditioning",
                       x_rtc="official enabled, ideal simulator odometry; no goal/path GT",
                       x_depth="same frame-bound LingBot monocular payload, RGB-only wire",
                       baseline_alignment="first certified rear; real turn; fresh RGB and replan",
                       not_mpc_only_comparison=True)
        x_checkpoint = OFFICIAL.parent / "x-navdp_posttrain.ckpt"
        receipt["weights"][str(x_checkpoint)] = sha(x_checkpoint)
        receipt["xnavdp_rgb_contract"] = "rgb_v1"
    if args.xnavdp_rgb_pair:
        receipt.update(xnavdp_rgb_pair=True,
                       xnavdp_rgb_contracts={"cec_x_legacy": "legacy_bgr", "cec_x": "rgb_v1"},
                       xnavdp_rgb_reference="official raw RGB client-to-actor channel semantics",
                       isolated_change="actor RGB channels, including replayed short-term history")
    dump(out / "manifest.json", receipt)
    (out / "sources").mkdir()
    for p in files:
        target = out / "sources" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    processes, streams, results = [], [], []
    try:
        common = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                      LINGBOT_REPO=str(LINGBOT), LINGBOT_WEIGHTS=str(LINGBOT / "weights/lingbot-map-long.pt"),
                      MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
                      MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
                      MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1",
                      MEMNAV_REPORT_TO="none", NAVDP_DISABLE_VIDEO="1")
        shared = f"{ROOT}:{ROOT}/.diagnostics/dependencies/python:{ROOT}/.diagnostics/dependencies/LightGlue:{ROOT}/InternNav/src/diffusion-policy"
        servers = [
            ("memnav", args.memnav_port, dict(common, PYTHONPATH=f"{ROOT}/NavDP/baselines/memnav:{shared}"),
             [MEM_PY, "-u", str(ROOT / "NavDP/baselines/memnav/memnav_server.py"), "--host", "127.0.0.1",
              "--port", str(args.memnav_port), "--checkpoint", str(MEM_CKPT), "--internnav_root", str(ROOT / "InternNav"),
              "--num_samples", "16", "--exclude_recent", "32", "--retrieval", "raw",
              "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
              "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60", "--flow_gate", "auto",
              "--buffer_root", str(out / "buffer"), "--certified_relocalization", "--certified_reference_depth_source", "canonical",
              "--lightglue_repo", str(ROOT / ".diagnostics/dependencies/LightGlue"),
              "--lightglue_dependency_root", str(ROOT / ".diagnostics/dependencies/python"), "--lightglue_max_keypoints", "2048"]),
            ("navdp", args.navdp_port, dict(common, PYTHONPATH=f"{ROOT}/NavDP/baselines/navdp:{shared}"),
             [MEM_PY, "-u", str(ROOT / "NavDP/baselines/navdp/navdp_server.py"), "--port", str(args.navdp_port),
              "--checkpoint", str(NAV_CKPT), "--depth_source", "monocular_sidecar", "--require_monocular_depth_transaction",
              "--monocular_depth_url", f"http://127.0.0.1:{args.memnav_port}/monocular_depth_query"]),
        ]
        if args.alignment_pair or args.xnavdp_pair:
            servers[0][2]["LINGBOT_POSE_LOG"] = str(out / "lingbot_pose.jsonl")
            servers[0][3][2] = str(ROOT / "MemNavData/lingbot_pose_diagnostic_server.py")
            servers[1][2]["NAVDP_INTERFACE_LOG"] = str(out / "navdp_interface.jsonl")
            servers[1][3][2] = str(ROOT / "MemNavData/navdp_interface_diagnostic_server.py")
        if args.xnavdp_pair:
            servers.append(("xnavdp", args.xnavdp_port,
                dict(common, PYTHONPATH=shared,
                     XNAVDP_MONOCULAR_DEPTH_URL=f"http://127.0.0.1:{args.memnav_port}/monocular_depth_query",
                     XNAVDP_DIAGNOSTIC_LOG=str(out / "xnavdp_requests.jsonl")),
                [MEM_PY, "-u", str(ROOT / "MemNavData/xnavdp_mono_diagnostic_server.py"),
                 "--official-root", str(OFFICIAL), "--checkpoint", str(x_checkpoint),
                 "--host", "127.0.0.1", "--port", str(args.xnavdp_port)]))
        for name, port, env, command in servers:
            stream = (out / f"logs/{name}.log").open("x")
            streams.append(stream)
            work = out / "runtime" / name
            work.mkdir(parents=True)
            process = subprocess.Popen(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            dump(out / "owned_processes.json", [{"pid":p.pid} for p in processes])
            started = time.monotonic()
            while not open_port(port):
                if process.poll() is not None:
                    raise RuntimeError(f"Private {name} failed to start")
                if time.monotonic()-started > 600:
                    raise TimeoutError(f"Private {name} startup timeout")
                time.sleep(1)
            print(f"READY private {name} pid={process.pid} port={port}", flush=True)
        for policy, tracker in variants:
            dest = out / "evaluation" / f"{policy}__{tracker}"
            route, adapter = (("native_sidecar", "legacy_metric") if policy == "native"
                              else ("certified_relocalization", "verified_bearing_v1"))
            command = [str(BULLET_PY), "-u", str(HERE), "eval", "--episode_root", str(BENCH / h["scene"]),
                       "--episode_ids", h["episode"], "--scene", source["source_asset"], "--scene_identity", h["scene"],
                       "--host", "127.0.0.1", "--port", str(args.memnav_port), "--novel_port", str(args.navdp_port),
                       "--out", str(dest), "--server_backend", "hybrid_pose", "--hybrid_route", route,
                       "--revisit_adapter", adapter, "--navdp_depth_source", "monocular_sidecar", "--success_dist", "1.0",
                       "--max_steps", str(args.max_steps), "--exec_horizon", "8", "--trajectory_selector", "server",
                       "--trajectory_selector_scope", "all", "--leg1_mode", "shared_trace", "--leg1_goal_source", "own",
                       "--seed", "0", "--terminal_uturn", "off", "--terminal_visual_refine", "off", "--deterministic_plan_seeds",
                       "--retrieval_override", "off", "--certified_cdec_rescue", "off", "--certified_stagnation_graph", "off",
                       "--revisit_controller", "navdp_mixed", "--role_pair_scope", "consumed_integration", "--role_pair_query_role", "revisit"]
            env = dict(evaluator_env, BULLET_QUERY_TRACKER=tracker)
            if args.alignment_pair or args.xnavdp_pair:
                env.update(BULLET_ALIGNMENT_EXPERIMENT="1",
                           BULLET_QUERY_ALIGNMENT=("first_rear_certified" if policy == "cec_aligned" else "off"))
            if args.xnavdp_pair:
                env.update(BULLET_XNAVDP_EXPERIMENT="1",
                           PYTHONPATH=f"{evaluator_env['PYTHONPATH']}:{ACADOS}/interfaces/acados_template",
                           ACADOS_SOURCE_DIR=str(ACADOS),
                           LD_LIBRARY_PATH=f"{ACADOS}/lib:" + env.get("LD_LIBRARY_PATH", ""),
                           X_NAVDP_MPC_CODEGEN_DIR=str(out / "codegen" / policy))
                if policy.startswith("cec_x"):
                    command[command.index("--revisit_controller")+1] = "xnavdp_point"
                    command += ["--xnavdp_port", str(args.xnavdp_port)]
                    env["XNAVDP_RGB_CONTRACT"] = "legacy_bgr" if policy == "cec_x_legacy" else "rgb_v1"
            # Existing habitat environment helper sets paths for the old binary;
            # this interpreter resolves its own Bullet/NumPy packages first.
            env.pop("PYTHONHOME", None)
            if args.xnavdp_pair:
                # Parse/validate the actual CLI before replaying a single A frame.
                validation = command.copy()
                validation[3] = "validate_x"
                with (out / f"logs/{policy}__{tracker}_cli_preflight.log").open("x") as stream:
                    subprocess.run(validation, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
            print(f"START {policy} / {tracker}", flush=True)
            with (out / f"logs/{policy}__{tracker}.log").open("x") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            if code:
                physics_file = dest / "physics_summary.json"
                physics = json.loads(physics_file.read_text()) if physics_file.exists() else {}
                log = (out / f"logs/{policy}__{tracker}.log").read_text()
                if (args.continue_after_invalid_physics and physics.get("max_tilt_deg", 0) >= 5
                        and "Physical proxy exceeded predeclared 5 degree tilt; invalidate rollout" in log):
                    results.append(dict(policy=policy, tracker=tracker, status="invalid_physics", reached=None,
                                        **physics))
                    dump(out / "partial_results.json", results)
                    print(f"INVALID {policy}/{tracker}: tilt={physics['max_tilt_deg']:.3f}; proceeding to next arm", flush=True)
                    continue
                raise subprocess.CalledProcessError(code, command)
            record = json.loads((dest / "terminal_measurements.json").read_text())[0]
            result = dict(policy=policy, tracker=tracker, **{k:v for k,v in record.items() if k != "tracker"})
            results.append(result)
            dump(out / "partial_results.json", results)
            print(f"DONE {policy}/{tracker} reached={record['reached']} final_distance={record['final_goal_dist_m']:.3f}", flush=True)
        changed = [str(p) for p in files if sha(p) != receipt["sources"][str(p)]]
        rgb_paired = len({r["first_query_rgb_sha256"] for r in results}) == 1
        invalid = [r for r in results if r.get("status") == "invalid_physics"]
        dump(out / "summary.json", {"completed":not invalid, "all_arms_attempted":True,
             "results":results, "changed_sources":changed,
             "first_rgb_exactly_paired":rgb_paired, "independent_histories":1, "inferential_SR_claim":False})
        if changed or not rgb_paired:
            raise RuntimeError("Source or physical initial image pairing failed")
        if invalid:
            dump(out / "failure.json", {"type":"InvalidPhysicalArms", "arms":invalid,
                 "message":"All arms attempted; tilt-invalid arms have no navigation SR label"})
            return 2
    except BaseException as error:
        dump(out / "failure.json", {"type":type(error).__name__, "message":str(error)})
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "eval":
        evaluate()
    elif mode == "local":
        raise SystemExit(local())
    elif mode == "validate_x":
        import eval_2leg_habitat
        from MemNavData.habitat_xnavdp_tracking import run_shared_evaluator
        print(run_shared_evaluator(None, validate_only=True))
    else:
        raise SystemExit(mode)
