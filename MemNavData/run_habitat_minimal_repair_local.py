#!/usr/bin/env python3
"""Local execution/input/PointGoal bridges; no robot/HPC services."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np

from MemNavData.bounded_pursuit import apply_collision, command, limits_from_evaluator
from MemNavData.habitat_executor_audit import compare_step, dump, sha
from MemNavData.run_cec_stream_depth_closed_loop import (
    ROOT, BENCH, MEM_PY, HAB_PY, MEM_CKPT, NAV_CKPT, LINGBOT, hab_env, open_port,
)

HERE = Path(__file__).resolve()
VARIANTS = [
    ("native", "legacy_snap", "legacy_bgr"),
    ("native", "bounded_standard", "legacy_bgr"),
    ("native", "bounded_standard", "rgb_v1"),
    ("cec", "bounded_standard", "rgb_v1"),
    ("cec", "bounded_standard", "legacy_bgr"),
    ("cec", "legacy_snap", "legacy_bgr"),
]
DEPTH_VARIANTS = [
    ("native", "bounded_standard", "rgb_v1", "legacy_square"),
    ("native", "bounded_standard", "rgb_v1", "source_rgb"),
    ("cec", "bounded_standard", "rgb_v1", "source_rgb"),
    ("cec", "bounded_standard", "rgb_v1", "legacy_square"),
]
FRONT_GOAL_VARIANTS = [
    ("raw_fixed", "bounded_standard", "rgb_v1", "source_rgb", "heading_off"),
    ("raw_fixed", "bounded_standard", "rgb_v1", "source_rgb", "heading_on"),
    ("cec", "bounded_standard", "rgb_v1", "source_rgb", "heading_on"),
    ("cec", "bounded_standard", "rgb_v1", "source_rgb", "heading_off"),
]
NATIVE_REQUEST_VARIANTS = [
    ("native", "bounded_standard", "rgb_v1", "source_rgb", "heading_on"),
    ("cec", "bounded_standard", "rgb_v1", "source_rgb", "heading_on"),
]


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def evaluate(entrypoint="role_pair", *, query_main=None):
    if entrypoint not in ("role_pair", "goal_a"):
        raise ValueError("unknown repaired evaluator entrypoint")
    if query_main is not None and (entrypoint != "role_pair" or not callable(query_main)):
        raise ValueError("A standalone query runner must be callable; Goal-A is unchanged")
    import eval_2leg_habitat as base
    from MemNavData.final14_spl_replay import measurement
    if base.args.contract_dry_run:
        if entrypoint == "goal_a":
            base.main()
        elif query_main is not None:
            query_main()
        else:
            runpy.run_path(str(ROOT / "MemNavData/eval_shared_online_role_pairs.py"), run_name="__main__")
        return
    out = Path(base.args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = os.environ["MINIMAL_EXECUTOR"]
    color = os.environ["MINIMAL_ACTOR_COLOR"]
    raster = os.environ.get("MINIMAL_DEPTH_RASTER", "")
    heading = os.environ.get("MINIMAL_FRONT_GOAL", "")
    if mode not in ("legacy_snap", "bounded_standard") or color not in ("legacy_bgr", "rgb_v1"):
        raise ValueError("invalid frozen experiment arm")
    original_step, original_select, original_leg, original_post, original_render = (
        base.pursuit_step, base.select_plan_trajectory, base.run_policy_leg, base.requests.post, base.render)
    actions, terminals, plans = [], [], []
    query_active, first_rgb_digest = False, None
    front_adapter = None
    if heading:
        from MemNavData.navdp_front_goal_adapter import (
            FrontGoalAdapter, install_loop_hook, rgb_only_memory_form,
        )
        if heading not in ("heading_off", "heading_on") or mode != "bounded_standard":
            raise ValueError("invalid frozen PointGoal bridge arm")
        front_adapter = FrontGoalAdapter(enabled=heading == "heading_on",
                                         max_turn_rad=limits_from_evaluator(base.args).max_turn_rad)

    def render(*a, **kw):
        nonlocal first_rgb_digest
        value = original_render(*a, **kw)
        if query_active and first_rgb_digest is None:
            import hashlib
            from PIL import Image
            Image.fromarray(value[0]).save(out / "first_query_rgb.png")
            first_rgb_digest = hashlib.sha256(value[0].tobytes()).hexdigest()
        return value

    def post(url, *a, **kw):
        sent = None
        if heading and url.startswith(base.BASE + "/"):
            before = dict(kw.get("data") or {})
            kw["data"] = rgb_only_memory_form(before)
            sent = {"path": url[len(base.BASE):], "query_active": query_active,
                    "next_action_index": len(actions), "original_fields": sorted(before),
                    "sent_fields": sorted(kw["data"]),
                    "removed_fields": sorted(set(before) - set(kw["data"]))}
        if url == f"{base.NOVEL_BASE}/navigator_reset":
            kw["json"] = dict(kw["json"], audit_actor_rgb_contract=color)
            if raster:
                kw["json"]["audit_depth_raster_contract"] = raster
        response = original_post(url, *a, **kw)
        if sent is not None:
            response.raise_for_status()
            payload = response.json()
            sent.update(frame_idx=payload.get("frame_idx"), image_sha256=payload.get("image_sha256"))
            append(out / "memory_http_boundary.jsonl", sent)
        if url.startswith(base.NOVEL_BASE + "/"):
            response.raise_for_status()
            payload = response.json()
            audit = payload.get("execution_input_audit")
            if audit is None or audit["contract"] != color:
                raise RuntimeError("Private base NavDP did not honor the frozen color contract")
            if raster and payload.get("audit_depth_raster_contract") != raster:
                raise RuntimeError("Private NavDP did not honor the frozen depth-raster contract")
            append(out / "navdp_http_receipts.jsonl", {
                "path": url[len(base.NOVEL_BASE):], "audit": audit,
                "query_active": query_active, "next_action_index": len(actions),
                "heading_active": bool(front_adapter and front_adapter.active),
                "depth_source": payload.get("depth_source"),
                "metric_depth_sensor_consumed": payload.get("metric_depth_sensor_consumed"),
                "monocular_depth_receipt": payload.get("monocular_depth_receipt"),
                "audit_depth_raster_contract": payload.get("audit_depth_raster_contract"),
            })
        return response

    def select(*a, **kw):
        trajectory, details = original_select(*a, **kw)
        response, pos, yaw = a[:3]
        alignment_started = bool(front_adapter and front_adapter.consider(response, float(yaw), len(actions)))
        row = {"plan_index": len(plans), "next_action_index": len(actions),
               "position": np.asarray(pos).tolist(), "yaw": float(yaw),
               "selected_trajectory": np.asarray(trajectory).tolist(),
               "all_trajectory": np.asarray(response.get("all_trajectory", [])).tolist(),
               "all_values": np.asarray(response.get("all_values", [])).tolist(),
               "selection": details,
               "pointgoal_alignment_started": alignment_started,
               "receipt": {k: response.get(k) for k in (
                   "execution_input_audit", "diffusion_seed", "pose_controller", "anchor",
                   "memory_controller_pointgoal", "navdp_stop_evidence", "navdp_critic_max",
                   "monocular_depth_receipt", "metric_depth_sensor_consumed", "memory_frame_idx",
                   "certified_relocalization", "critic_fallback_applied", "revisit_adapter_takeover")}}
        plans.append(row)
        append(out / "full_plan_outputs.jsonl", row)
        return trajectory, details

    def step(pos, yaw, path, pf):
        turning = bool(front_adapter and front_adapter.active)
        if turning:
            shadow = None
            request = front_adapter.command(pos, yaw)
            new = result = apply_collision(pos, request, pf.try_step)
            front_adapter.observe(float(result[1]), len(actions))
        else:
            endings, old_yaw, shadow = compare_step(pos, yaw, path, pf, base.args)
            old = original_step(pos, yaw, path, pf)
            np.testing.assert_allclose(old[0], endings["legacy_snap"], atol=1e-9, rtol=0)
            if abs(old[1] - old_yaw) > 1e-10:
                raise RuntimeError("legacy steering reconstruction mismatch")
            request = command(pos, yaw, path, limits_from_evaluator(base.args))
            new = apply_collision(pos, request, pf.try_step)
            result = old if mode == "legacy_snap" else new
        if not actions:
            pf.save_nav_mesh(str(out / "execution.navmesh"))
        row = {
            "action_index": len(actions), "plan_index": len(plans) - 1,
            "position_before": np.asarray(pos).tolist(), "yaw_before": float(yaw),
            "world_path": None if turning else np.asarray(path).tolist(),
            "action_kind": "heading" if turning else "trajectory_tracking",
            "reference_executed": not turning,
            "heading_target_yaw": front_adapter.target_yaw if turning else None,
            "selected_mode": mode, "actual_position": np.asarray(result[0]).tolist(),
            "actual_yaw": float(result[1]), "actual_translation_m": float(result[2]),
            "bounded_request": {
                "position": request.position.tolist(), "yaw": request.yaw,
                "displacement_m": request.displacement_m, "reason": request.reason,
                "reference_distance_m": request.reference_distance_m,
                "reference_index": request.reference_index,
                "nominal_displacement_m": request.nominal_displacement_m,
            },
            "bounded_collision_position": new[0].tolist(), "legacy_shadow": shadow,
        }
        actions.append(row)
        append(out / "executor_actions.jsonl", row)
        if len(actions) % 80 == 0:
            print(f"EXECUTOR {mode}/{color}: {len(actions)} ticks", flush=True)
        return result

    def leg(*a, **kw):
        nonlocal query_active
        query_active = True
        # The evaluator checks an empty output folder before history replay.
        # Materialize the private loop only after that normal initialization.
        implementation = (install_loop_hook(base, original_leg, front_adapter, out / "executed_loop.py")
                          if front_adapter is not None else original_leg)
        result = implementation(*a, **kw)
        query_active = False
        terminals.append(dict(measurement(result, a[5], a[6]),
                              first_query_rgb_sha256=first_rgb_digest))
        dump(out / "terminal_measurements.json", terminals)
        dump(out / "rollout_evidence.json", {
            "entrypoint": entrypoint, "rollout_trace": result["rollout_trace"],
            "memory_trace": result["memory_trace"], "terminal": terminals[-1],
        })
        return result

    base.requests.post, base.select_plan_trajectory = post, select
    base.pursuit_step, base.run_policy_leg = step, leg
    base.render = render
    try:
        if entrypoint == "goal_a":
            if not base.args.stop_after_leg1 or base.args.leg1_mode != "policy":
                raise ValueError("Goal-A entrypoint requires one actual policy rollout")
            base.main()
        elif query_main is not None:
            query_main()
        else:
            runpy.run_path(str(ROOT / "MemNavData/eval_shared_online_role_pairs.py"), run_name="__main__")
    finally:
        dump(out / "executor_summary.json", {
            "mode": mode, "color": color, "actions": len(actions), "plans": len(plans),
            "zero_reference_ticks": sum(r["bounded_request"]["reason"] == "zero_reference_hold_not_arrival" for r in actions),
            "reference_limited_ticks": sum(r["bounded_request"]["reason"] == "reference_limited" for r in actions),
            "legacy_standard_difference_gt_1mm": sum(r["legacy_shadow"] is not None and r["legacy_shadow"]["legacy_vs_try_step_m"] > .001 for r in actions),
            "heading_ticks": sum(r["action_kind"] == "heading" for r in actions),
        })
        if front_adapter is not None:
            dump(out / "front_goal_receipt.json", front_adapter.receipt())


def local():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--histories", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--study", choices=("execution_color", "depth_raster", "front_goal", "native_request"), default="execution_color")
    parser.add_argument("--memnav-port", type=int, default=21690)
    parser.add_argument("--navdp-port", type=int, default=21691)
    args = parser.parse_args()
    variants = {"execution_color": VARIANTS, "depth_raster": DEPTH_VARIANTS,
                "front_goal": FRONT_GOAL_VARIANTS, "native_request": NATIVE_REQUEST_VARIANTS}[args.study]
    heading_study = args.study in ("front_goal", "native_request")
    raster_study = args.study in ("depth_raster", "front_goal", "native_request")
    query_role = "novel" if args.study == "native_request" else "revisit"
    if args.study == "native_request" and (args.histories != 2 or args.max_steps != 64):
        raise ValueError("Native-request check is fixed to first-two histories / 64 ticks")
    if args.study == "depth_raster" and args.histories > 2:
        raise ValueError("The frozen depth bridge uses only the first two consumed histories")
    if args.max_steps < 1 or args.memnav_port == args.navdp_port or any(
            open_port(p) for p in (args.memnav_port, args.navdp_port)):
        raise ValueError("invalid limits or occupied private ports")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    histories = json.loads((BENCH / "manifest.json").read_text())["episodes"][:args.histories]
    files = [HERE, ROOT / "MemNavData/bounded_pursuit.py", ROOT / "MemNavData/habitat_executor_audit.py",
             ROOT / "MemNavData/navdp_execution_audit_server.py", ROOT / "MemNavData/eval_2leg_habitat.py",
             ROOT / "MemNavData/eval_shared_online_role_pairs.py", ROOT / "MemNavData/generate_twoleg.py",
             ROOT / "NavDP/baselines/memnav/policy_agent.py", ROOT / "NavDP/baselines/memnav/memnav_server.py",
             ROOT / "NavDP/baselines/navdp/policy_agent.py", ROOT / "NavDP/baselines/navdp/policy_network.py",
             ROOT / "NavDP/baselines/navdp/navdp_server.py", ROOT / "MemNavData/verify_habitat_minimal_repair.py",
             ROOT / "MemNavData/render_habitat_minimal_repair.py",
             MEM_CKPT, NAV_CKPT, LINGBOT / "weights/lingbot-map-long.pt"]
    if raster_study:
        files += [ROOT / "MemNavData/navdp_depth_raster_audit_server.py",
                  ROOT / "MemNavData/lingbot_depth_raster.py"]
    if heading_study:
        files += [ROOT / "MemNavData/navdp_front_goal_adapter.py",
                  ROOT / "MemNavData/lingbot_pose_diagnostic_server.py"]
    manifest = {
        "schema": "habitat_minimal_execution_bridge_v1", "scope": "consumed query diagnostic",
        "benchmark": str(BENCH), "benchmark_sha256": sha(BENCH / "manifest.json"),
        "histories": [{"scene": h["scene"], "episode": h["episode"],
                       "trace_sha256": h["online_a_trace_sha256"]} for h in histories],
        "study": args.study, "variants": variants, "max_steps": args.max_steps, "exec_horizon": 8,
        "query_role_evaluator_only": query_role,
        "history_source": "shared actual metric-NavDP-A", "query_depth": "LingBot monocular",
        "reference_depth_source": "canonical", "residual_m": 2.5,
        "collision": "NavMesh in environment, standard sliding; NOT GT-free physics",
        "success": "evaluator planar GT distance < 1m; NOT autonomous stop",
        "runtime_role_visible": False, "code_and_weights": {str(p): sha(p) for p in files},
        "pointgoal_heading_adapter": heading_study,
        "model_executor_odometry_fields_removed": heading_study,
    }
    dump(out / "manifest.json", manifest)
    for path in files:
        if path.suffix == ".py":
            dest = out / "source_snapshot" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
    processes, handles, results = [], [], []
    try:
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   LINGBOT_REPO=str(LINGBOT), LINGBOT_WEIGHTS=str(LINGBOT / "weights/lingbot-map-long.pt"),
                   MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
                   MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
                   MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1",
                   MEMNAV_REPORT_TO="none", NAVDP_DISABLE_VIDEO="1")
        shared = f"{ROOT}:{ROOT}/.diagnostics/dependencies/python:{ROOT}/.diagnostics/dependencies/LightGlue:{ROOT}/InternNav/src/diffusion-policy"
        navdp_settings = dict(env, PYTHONPATH=f"{ROOT}/NavDP/baselines/navdp:{shared}",
                             NAVDP_EXECUTION_INPUT_LOG=str(out / "navdp_input_audit.jsonl"))
        navdp_script = "navdp_execution_audit_server.py"
        if raster_study:
            navdp_script = "navdp_depth_raster_audit_server.py"
            navdp_settings["NAVDP_DEPTH_RASTER_LOG_ROOT"] = str(out / "depth_raster_artifacts")
        memory_settings = dict(env, PYTHONPATH=f"{ROOT}/NavDP/baselines/memnav:{shared}")
        memory_script = ROOT / "NavDP/baselines/memnav/memnav_server.py"
        pose_log = out / "lingbot_pose_readout.jsonl"
        if heading_study:
            memory_script = ROOT / "MemNavData/lingbot_pose_diagnostic_server.py"
            memory_settings["LINGBOT_POSE_LOG"] = str(pose_log)
        servers = [
            ("memnav", args.memnav_port, memory_settings,
             [MEM_PY, "-u", str(memory_script),
              "--host", "127.0.0.1", "--port", str(args.memnav_port), "--checkpoint", str(MEM_CKPT),
              "--internnav_root", str(ROOT / "InternNav"), "--num_samples", "16", "--exclude_recent", "32",
              "--retrieval", "raw", "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
              "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60", "--flow_gate", "auto",
              "--buffer_root", str(out / "buffer"), "--certified_relocalization",
              "--certified_reference_depth_source", "canonical",
              "--lightglue_repo", str(ROOT / ".diagnostics/dependencies/LightGlue"),
              "--lightglue_dependency_root", str(ROOT / ".diagnostics/dependencies/python"),
              "--lightglue_max_keypoints", "2048"]),
            ("navdp", args.navdp_port,
             navdp_settings,
             [MEM_PY, "-u", str(ROOT / "MemNavData" / navdp_script),
              "--port", str(args.navdp_port), "--checkpoint", str(NAV_CKPT),
              "--depth_source", "monocular_sidecar", "--require_monocular_depth_transaction",
              "--monocular_depth_url", f"http://127.0.0.1:{args.memnav_port}/monocular_depth_query"]),
        ]
        for name, port, settings, cmd in servers:
            handle = (out / f"logs/{name}.log").open("x")
            handles.append(handle)
            cwd = out / "runtime" / name
            cwd.mkdir(parents=True)
            child = subprocess.Popen(cmd, cwd=cwd, env=settings, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(child)
            dump(out / "owned_processes.json", [{"pid": p.pid, "args": p.args} for p in processes])
            deadline = time.monotonic() + 600
            while not open_port(port):
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"private {name} startup failed; inspect log")
                time.sleep(1)
            print(f"READY private {name}: {child.pid} / {port}", flush=True)
        for index, h in enumerate(histories):
            source = json.loads((Path(h["online_a_episode"]) / "receipt.json").read_text())
            for variant in variants[::1 if index % 2 == 0 else -1]:
                policy, executor, color = variant[:3]
                raster = variant[3] if len(variant) >= 4 else ""
                heading = variant[4] if len(variant) >= 5 else ""
                name = "__".join(variant)
                dest = out / "evaluation" / h["scene"] / name
                route, adapter = {"native": ("native_sidecar", "legacy_metric"),
                                  "cec": ("certified_relocalization", "verified_bearing_v1"),
                                  "raw_fixed": ("phase", "raw_fixed_bearing_v1")}[policy]
                cmd = [HAB_PY, "-u", str(HERE), "eval", "--episode_root", str(BENCH / h["scene"]),
                       "--episode_ids", h["episode"], "--scene", source["source_asset"], "--scene_identity", h["scene"],
                       "--host", "127.0.0.1", "--port", str(args.memnav_port), "--novel_port", str(args.navdp_port),
                       "--out", str(dest), "--server_backend", "hybrid_pose", "--hybrid_route", route,
                       "--revisit_adapter", adapter, "--navdp_depth_source", "monocular_sidecar",
                       "--success_dist", "1.0", "--max_steps", str(args.max_steps), "--exec_horizon", "8",
                       "--trajectory_selector", "server", "--leg1_mode", "shared_trace", "--leg1_goal_source", "own",
                       "--seed", "0", "--terminal_uturn", "off", "--terminal_visual_refine", "off",
                       "--deterministic_plan_seeds", "--retrieval_override", "off", "--certified_cdec_rescue", "off",
                       "--certified_stagnation_graph", "off", "--cec_initial_bearing_alignment", "off",
                       "--revisit_controller", "navdp_mixed", "--role_pair_scope", "consumed_integration",
                       "--role_pair_query_role", query_role]
                print(f"START {h['scene']} {name}", flush=True)
                dump(out / "progress.json", {"stage":"evaluation", "completed_arms":len(results),
                     "total_arms":len(histories)*len(variants), "scene":h["scene"], "arm":name,
                     "supervisor_pid":os.getpid(), "updated_unix":time.time()})
                settings = dict(hab_env(), MINIMAL_EXECUTOR=executor, MINIMAL_ACTOR_COLOR=color,
                                MINIMAL_DEPTH_RASTER=raster, MINIMAL_FRONT_GOAL=heading)
                pose_offset = pose_log.stat().st_size if pose_log.exists() else 0
                with (out / f"logs/{index}_{name}.log").open("x") as handle:
                    subprocess.run(cmd, cwd=ROOT, env=settings, stdout=handle, stderr=subprocess.STDOUT, check=True)
                if heading_study:
                    with pose_log.open() as stream:
                        stream.seek(pose_offset)
                        dump(dest / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
                terminal = json.loads((dest / "terminal_measurements.json").read_text())
                if len(terminal) != 1:
                    raise RuntimeError("expected one complete query terminal")
                result = dict(scene=h["scene"], policy=policy, executor=executor, color=color,
                              **terminal[0], execution=json.loads((dest / "executor_summary.json").read_text()))
                if raster:
                    result["depth_raster"] = raster
                if heading:
                    result["front_goal"] = heading
                results.append(result)
                dump(out / "partial_results.json", results)
                print(f"DONE {h['scene']} {name}: reached={result['reached']} "
                      f"steps={result['steps']} distance={result['final_goal_dist_m']:.3f}", flush=True)
        changed = [str(p) for p in files if sha(p) != manifest["code_and_weights"][str(p)]]
        dump(out / "summary.json", dict(completed=True, results=results, changed_source_files=changed))
        if changed:
            raise RuntimeError("source changed during the bridge; results not frozen")
    except BaseException as error:
        dump(out / "failure.json", dict(type=type(error).__name__, message=str(error)))
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
        for handle in handles:
            handle.close()
    # Finalization belongs to the same supervised job, after private GPU servers exit.
    for stage, script in [("verification", "verify_habitat_minimal_repair.py"),
                          ("first_person_render", "render_habitat_minimal_repair.py")]:
        dump(out / "progress.json", {"stage":stage, "completed_arms":len(results),
             "total_arms":len(histories)*len(variants), "updated_unix":time.time()})
        with (out / "logs" / (stage + ".log")).open("x") as handle:
            try:
                subprocess.run([HAB_PY, str(ROOT / "MemNavData" / script), str(out)],
                               cwd=ROOT, env=hab_env(), stdout=handle, stderr=subprocess.STDOUT, check=True)
            except Exception as error:
                dump(out / "failure.json", {"stage":stage, "type":type(error).__name__, "message":str(error)})
                raise
    dump(out / "progress.json", {"stage":"verified_and_videos_exported", "completed_arms":len(results),
         "total_arms":len(histories)*len(variants), "updated_unix":time.time()})
    print("COMPLETE: independent verification and first-person videos", flush=True)


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "eval":
        evaluate()
    elif mode == "goal_a":
        evaluate("goal_a")
    elif mode == "local":
        local()
    else:
        raise SystemExit(mode)
