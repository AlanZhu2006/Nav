#!/usr/bin/env python3
"""Consumed local query pairs: change only the simulator collision executor.

Private servers only. No production source edits, robot requests, or HPC jobs.
The stored A history is unchanged; this is query-stage mechanism attribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from MemNavData.habitat_executor_audit import MODES, compare_step, dump, sha, summarize_steps
from MemNavData.run_cec_stream_depth_closed_loop import (
    ROOT, BENCH, MEM_PY, HAB_PY, MEM_CKPT, NAV_CKPT, LINGBOT, hab_env, open_port,
)

HERE = Path(__file__).resolve()


def evaluate():
    import numpy as np
    import eval_2leg_habitat as base
    from MemNavData.final14_spl_replay import measurement
    mode = os.environ["EXECUTOR_AUDIT_MODE"]
    if mode not in MODES:
        raise ValueError(mode)
    output = Path(base.args.out)
    original_step, original_select, original_leg = (
        base.pursuit_step, base.select_plan_trajectory, base.run_policy_leg)
    actions, terminal = [], []
    current_plan_index = -1
    plans = []

    def select(*a, **kw):
        nonlocal current_plan_index
        trajectory, details = original_select(*a, **kw)
        current_plan_index += 1
        response, pos, yaw = a[:3]
        value = {
            "plan_index": current_plan_index, "next_action_index": len(actions),
            "position": np.asarray(pos).tolist(), "yaw": float(yaw),
            "selected_trajectory": np.asarray(trajectory).tolist(),
            "all_trajectory": np.asarray(response.get("all_trajectory", [])).tolist(),
            "all_values": np.asarray(response.get("all_values", [])).tolist(),
            "selection": details,
        }
        plans.append(value)
        with (output / "full_plan_outputs.jsonl").open("a") as f:
            f.write(json.dumps(value, allow_nan=False) + "\n")
        return trajectory, details

    def step(pos, yaw, path, pf):
        endings, next_yaw, record = compare_step(pos, yaw, path, pf, base.args)
        expected, expected_yaw, expected_length = original_step(pos, yaw, path, pf)
        if (not np.allclose(expected, endings["legacy_snap"], rtol=0, atol=1e-9)
                or abs(expected_yaw-next_yaw) > 1e-10):
            raise RuntimeError("Diagnostic command reconstruction differs from original executor")
        # Legacy arm returns exactly the original tuple. No injected rounding.
        if mode == "legacy_snap":
            result = expected, expected_yaw, expected_length
        else:
            p = endings[mode]
            result = p, next_yaw, float(np.linalg.norm((p-np.asarray(pos))[[0, 2]]))
        if not actions:
            pf.save_nav_mesh(str(output / "execution.navmesh"))
        record.update(action_index=len(actions), plan_index=current_plan_index,
                      executed_mode=mode, executed_position=np.asarray(result[0]).tolist(),
                      actual_translation_m=float(result[2]), world_path=np.asarray(path).tolist())
        actions.append(record)
        with (output / "executor_actions.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        if len(actions) % 80 == 0:
            print(f"EXECUTOR {mode}: {len(actions)} actions; "
                  f"same-command disagreements "
                  f"{sum(r['legacy_vs_try_step_m'] > .001 for r in actions)}", flush=True)
        return result

    def leg(*a, **kw):
        value = original_leg(*a, **kw)
        terminal.append(measurement(value, a[5], a[6]))
        dump(output / "terminal_measurements.json", terminal)
        return value

    base.select_plan_trajectory, base.pursuit_step, base.run_policy_leg = select, step, leg
    try:
        runpy.run_path(str(ROOT / "MemNavData/eval_shared_online_role_pairs.py"), run_name="__main__")
    finally:
        if output.exists():
            dump(output / "executor_summary.json", dict(mode=mode, **summarize_steps(actions)))


def local():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--memnav-port", type=int, default=21680)
    parser.add_argument("--navdp-port", type=int, default=21681)
    args = parser.parse_args()
    output = args.out.resolve()
    if args.memnav_port == args.navdp_port or any(open_port(p) for p in (args.memnav_port, args.navdp_port)):
        raise RuntimeError("Requested private ports are occupied")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    histories = json.loads((BENCH / "manifest.json").read_text())["episodes"][:args.histories]
    variants = [("mono_native", "legacy_snap"), ("mono_native", "try_step"),
                ("mono_cec", "try_step"), ("mono_cec", "legacy_snap")]
    files = [HERE, ROOT / "MemNavData/habitat_executor_audit.py",
             ROOT / "MemNavData/eval_2leg_habitat.py", ROOT / "MemNavData/eval_shared_online_role_pairs.py",
             ROOT / "MemNavData/generate_twoleg.py", ROOT / "NavDP/baselines/memnav/policy_agent.py",
             ROOT / "NavDP/baselines/memnav/memnav_server.py", ROOT / "NavDP/baselines/navdp/policy_agent.py",
             ROOT / "NavDP/baselines/navdp/navdp_server.py", MEM_CKPT, NAV_CKPT,
             LINGBOT / "weights/lingbot-map-long.pt"]
    receipt = {
        "schema": "local_habitat_executor_factorial_v1_20260908",
        "scope": "consumed query-stage diagnostic; not formal SR confirmation or robot physics",
        "selection": "first N histories in pre-existing four-history manifest; Revisit query only",
        "benchmark": str(BENCH), "benchmark_sha256": sha(BENCH / "manifest.json"),
        "histories": [{"scene": h["scene"], "episode": h["episode"],
                       "trace_sha256": h["online_a_trace_sha256"]} for h in histories],
        "variants": variants, "max_steps": args.max_steps, "exec_horizon": 8,
        "residual_m": 2.5, "observation_depth": "monocular_sidecar",
        "reference_depth_source": "canonical", "history_source": "fixed actual metric-NavDP-A RGB",
        "runtime_role_visible": False, "GT_used_by_executor": True,
        "standard_comparator": "Habitat PathFinder.try_step, sliding enabled",
        "no_sliding": "shadow only; does not control this first paired experiment",
        "code_and_weights": {str(p): sha(p) for p in files},
    }
    dump(output / "manifest.json", receipt)
    processes, handles, results = [], [], []
    try:
        common = os.environ.copy()
        common.update(PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                      LINGBOT_REPO=str(LINGBOT), LINGBOT_WEIGHTS=str(LINGBOT / "weights/lingbot-map-long.pt"),
                      MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
                      MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
                      MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1",
                      MEMNAV_REPORT_TO="none", NAVDP_DISABLE_VIDEO="1")
        shared = f"{ROOT}:{ROOT}/.diagnostics/dependencies/python:{ROOT}/.diagnostics/dependencies/LightGlue:{ROOT}/InternNav/src/diffusion-policy"
        servers = [
            ("memnav", args.memnav_port, dict(common, PYTHONPATH=f"{ROOT}/NavDP/baselines/memnav:{shared}"),
             [MEM_PY, "-u", str(ROOT / "NavDP/baselines/memnav/memnav_server.py"),
              "--host", "127.0.0.1", "--port", str(args.memnav_port), "--checkpoint", str(MEM_CKPT),
              "--internnav_root", str(ROOT / "InternNav"), "--num_samples", "16", "--exclude_recent", "32",
              "--retrieval", "raw", "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
              "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60", "--flow_gate", "auto",
              "--buffer_root", str(output / "buffer"), "--certified_relocalization",
              "--certified_reference_depth_source", "canonical",
              "--lightglue_repo", str(ROOT / ".diagnostics/dependencies/LightGlue"),
              "--lightglue_dependency_root", str(ROOT / ".diagnostics/dependencies/python"),
              "--lightglue_max_keypoints", "2048"]),
            ("navdp", args.navdp_port, dict(common, PYTHONPATH=f"{ROOT}/NavDP/baselines/navdp:{shared}"),
             [MEM_PY, "-u", str(ROOT / "NavDP/baselines/navdp/navdp_server.py"), "--port", str(args.navdp_port),
              "--checkpoint", str(NAV_CKPT), "--depth_source", "monocular_sidecar",
              "--require_monocular_depth_transaction", "--monocular_depth_url",
              f"http://127.0.0.1:{args.memnav_port}/monocular_depth_query"]),
        ]
        for name, port, env, command in servers:
            log = (output / f"logs/{name}.log").open("x")
            handles.append(log)
            cwd = output / "runtime" / name
            cwd.mkdir(parents=True)
            process = subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            dump(output / "owned_processes.json", [{"pid": p.pid} for p in processes])
            started = time.monotonic()
            while not open_port(port):
                if process.poll() is not None:
                    raise RuntimeError(f"Private {name} startup failed; see its log")
                if time.monotonic()-started > 600:
                    raise TimeoutError(f"Private {name} startup timeout")
                time.sleep(1)
            print(f"READY private {name}: pid={process.pid}, port={port}", flush=True)
        for index, h in enumerate(histories):
            source = json.loads((Path(h["online_a_episode"]) / "receipt.json").read_text())
            for policy, mode in variants[::1 if index % 2 == 0 else -1]:
                name = f"{policy}__{mode}"
                dest = output / "evaluation" / h["scene"] / name
                route, adapter = (("native_sidecar", "legacy_metric") if policy == "mono_native"
                                  else ("certified_relocalization", "verified_bearing_v1"))
                command = [HAB_PY, "-u", str(HERE), "eval", "--episode_root", str(BENCH / h["scene"]),
                           "--episode_ids", h["episode"], "--scene", source["source_asset"], "--scene_identity", h["scene"],
                           "--host", "127.0.0.1", "--port", str(args.memnav_port), "--novel_port", str(args.navdp_port),
                           "--out", str(dest), "--server_backend", "hybrid_pose", "--hybrid_route", route,
                           "--revisit_adapter", adapter, "--navdp_depth_source", "monocular_sidecar",
                           "--success_dist", "1.0", "--max_steps", str(args.max_steps), "--exec_horizon", "8",
                           "--trajectory_selector", "server", "--trajectory_selector_scope", "all",
                           "--leg1_mode", "shared_trace", "--leg1_goal_source", "own", "--seed", "0",
                           "--terminal_uturn", "off", "--terminal_visual_refine", "off", "--deterministic_plan_seeds",
                           "--retrieval_override", "off", "--certified_cdec_rescue", "off", "--certified_stagnation_graph", "off",
                           "--revisit_controller", "navdp_mixed", "--role_pair_scope", "consumed_integration",
                           "--role_pair_query_role", "revisit"]
                env = dict(hab_env(), EXECUTOR_AUDIT_MODE=mode)
                print(f"START {h['scene']} {name}", flush=True)
                with (output / f"logs/{index}_{name}.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                terminal = json.loads((dest / "terminal_measurements.json").read_text())
                assert len(terminal) == 1
                stats = json.loads((dest / "executor_summary.json").read_text())
                result = dict(scene=h["scene"], policy=policy, mode=mode, **terminal[0], executor=stats)
                results.append(result)
                dump(output / "partial_results.json", results)
                print(f"DONE {name}: reached={result['reached']} distance={result['final_goal_dist_m']:.3f} "
                      f"actions={stats['actions']} same-command differences={stats['legacy_vs_try_step_gt_1mm']}", flush=True)
        changes = [str(p) for p in files if sha(p) != receipt["code_and_weights"][str(p)]]
        dump(output / "summary.json", dict(completed=True, results=results, changed_source_files=changes,
                                          inferential_SR_claim=False, scope=receipt["scope"]))
    except BaseException as error:
        dump(output / "failure.json", dict(type=type(error).__name__, message=str(error)))
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


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "eval":
        evaluate()
    elif mode == "local":
        local()
    else:
        raise SystemExit(f"Unknown mode {mode}")
