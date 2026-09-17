"""Run one sealed target, with three paired arms of the frozen repair stack.

No construction, A collection, model changes, or outcome-based target selection.
Role/support remain evaluator annotations and never enter the model requests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ARMS = ("native", "raw_fixed", "cec")
SCHEMA = "hm3d_repaired_covisibility_20260909_v1"
POPULATION_SHA = "182f3a6d2519d5b2c178b88345db4d0bb678088dd88487cfecaad9814c6fdaaf"
SCOPE = "repaired query-stage support spectrum on existing actual-mono A; not newly repaired end-to-end A"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def select_task(population, index):
    require(population["schema"] == SCHEMA, "wrong population schema")
    require(population["phase"] == "sealed_before_query_evaluation", "unsealed population")
    require(population["arms"] == list(ARMS), "arm family changed")
    tasks = population["tasks"]
    require(len(tasks) == population["query_count"] and len(tasks)*3 == population["query_arm_count"],
            "task counts disagree")
    require(0 <= index < len(tasks), "task index outside frozen population")
    task = tasks[index]
    require(task["task_index"] == index, "task order changed")
    order = list(itertools.permutations(ARMS))[index % 6]
    require(tuple(task["arm_order"]) == order, "balanced arm order changed")
    return task


def load_task(population_path, index):
    require(sha(population_path) == POPULATION_SHA, "frozen population changed")
    population = load(population_path)
    task = select_task(population, index)
    path = Path(task["construction"])
    require(sha(path) == task["construction_sha256"], "construction changed")
    payload = load(path)
    require(payload["completed"] and payload["navigation_rollouts"] == 0, "invalid construction phase")
    require(payload["history_index"] == task["history_index"], "wrong historical source")
    require(payload["all_historical_rgb_rerender_hashes_match"], "history image audit failed")
    queries = [q for q in payload["queries"] if q["query_id"] == task["query_id"]]
    require(len(queries) == 1, "target is missing or duplicated")
    query = queries[0]
    for field in ("goal_rgb", "goal_depth"):
        require(sha(path.parent / query[field]) == query[field + "_sha256"], "goal asset changed")
    online = Path(payload["online_a_episode"])
    for name, field in (("receipt.json", "online_a_receipt_sha256"),
                        ("online_a_trace.json", "online_a_trace_sha256")):
        require(sha(online / name) == payload[field], "online history changed")
    trace = load(online / "online_a_trace.json")
    require(trace["reached"] and len(trace["poses"]) == payload["online_a_steps"], "wrong A history")
    require(trace["episode_seed"] == payload["source"]["seed"], "seed changed")
    return task, payload, query, {"benchmark": payload, "source": online,
        "receipt": load(online / "receipt.json"), "trace": trace}


def evaluate_query():
    import numpy as np
    import pandas as pd
    import eval_shared_online_role_pairs as shared
    from shared_online_role_pair_contract import runtime_query
    base, args = shared.base, shared.args
    arm, backend = shared.validate_cli()
    require(args.role_pair_scope == "consumed_integration", "query-only scope must be explicit")
    if args.contract_dry_run:
        print(f"SEALED COVISIBILITY CLI OK: {arm}; role is evaluator-only")
        return
    task, payload, annotated, frozen = load_task(os.environ["COVIS_POPULATION"], int(os.environ["COVIS_TASK_INDEX"]))
    require(args.role_pair_query_role == annotated["analysis_role"], "analysis-role mismatch")
    require(base.SCENE_IDENTITY == payload["scene"] and args.episode_ids == payload["episode"], "source mismatch")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "rollout output must be empty")
    receipt, trace = frozen["receipt"], frozen["trace"]
    require(sha(args.scene) == receipt["source_asset_sha256"], "scene asset changed")
    parquet = Path(receipt["source_episode"]) / "data/chunk-000/episode_000000.parquet"
    require(sha(parquet) == receipt["source_parquet_sha256"], "camera carrier changed")
    intrinsic = np.stack([np.asarray(row, float) for row in
        pd.read_parquet(parquet).iloc[0]["observation.camera_intrinsic"]])
    height = float(receipt["camera_height_m"])
    require(math.isclose(height, base.CAM_H, abs_tol=1e-12), "camera height mismatch")
    query = runtime_query(annotated)
    goal_jpg = (Path(task["construction"]).parent / query["goal_rgb"]).read_bytes()
    goal = np.asarray(query["floor_position"], float)
    # Goal annotation depth/GT are never policy inputs. GT below is scoring only.
    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    try:
        base.srv_reset(camera_height=height, seed=int(trace["episode_seed"]),
            episode_len=payload["online_a_steps"] + args.max_steps, camera_intrinsic=intrinsic,
            causal_history_sha256=payload["online_a_trace_sha256"])
        a, replay = shared.replay_prefix(frozen)
        ok, geo, _ = base.geodesic(sim.pathfinder, a["end_pos"], goal)
        require(ok and abs(geo-annotated["geodesic_from_a_end_m"]) <= .05, "query distance changed")
        result = base.run_policy_leg(sim, sim.pathfinder, a["end_pos"], a["end_psi"],
            goal_jpg, goal[[0, 2]], float(geo), None, terminal_mode="off", goal_yaw=query["yaw_rad"],
            camera_intrinsic=intrinsic, policy_backend=backend, episode_seed=int(trace["episode_seed"]), leg_index=1)
        dump(output / "query_plans.json", {"schema": SCHEMA, "arm": arm,
            "query_runtime_fields": sorted(query), "analysis_role_not_forwarded": True,
            "replay": replay, "query_leg": result["plans"],
            "rollout_traces": {"legA": trace["poses"], "query": result["rollout_trace"]},
            "counts": shared.router_counts(result["plans"]), "depth": shared.depth_counts(result["plans"]),
            "query_result": {"reached": bool(result["reached"]), "end_position": result["end_pos"].tolist()}})
    finally:
        sim.close()


def evaluator_command(source, out, mem, nav, *, arm, role):
    from MemNavData.run_repaired_fullmono_local import evaluator_command as original
    cmd = original(source, out, mem, nav, arm=arm, role=role, benchmark=Path("/unused/sealed_query"))
    return cmd[:2] + [str(HERE), "eval"] + cmd[4:]


def run(out, durable, population_path, index, mem, nav):
    from MemNavData.run_repaired_fullmono_local import HAB_PY, execution_environment, private_servers, run_child
    task, payload, query, _ = load_task(population_path, index)
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    os.environ.update(COVIS_POPULATION=str(population_path), COVIS_TASK_INDEX=str(index),
                      REPAIRED_BUFFER_ROOT=str(out / "buffer"), REPAIRED_RUNTIME_ROOT=str(out / "runtime"))
    manifest = {"schema": SCHEMA, "scope": SCOPE, "population": str(population_path),
        "population_sha256": POPULATION_SHA, "task": task, "original_work_root": str(out),
        "runtime_source_receipt": {"path": os.environ["REPAIRED_SOURCE_RECEIPT"],
                                   "sha256": sha(os.environ["REPAIRED_SOURCE_RECEIPT"])},
        "evaluation_source_receipt": {"path": str(HERE.parent / "SOURCE_BUNDLE.sha256"),
                                      "sha256": sha(HERE.parent / "SOURCE_BUNDLE.sha256")},
        "new_a_rollouts": 0, "scene": payload["scene"], "episode": payload["episode"],
        "query_id": query["query_id"], "bin": query["bin"], "q_eligible": query["q_eligible"],
        "analysis_role": query["analysis_role"]}
    dump(out / "manifest.json", manifest)
    dump(durable / "manifest.json", manifest)
    summary = {"schema": SCHEMA, "scope": SCOPE, "completed": False, "queries": []}
    dump(out / "summary.json", summary)
    pose_log = out / "lingbot_pose_readout.jsonl"
    with private_servers(out, mem, nav):
        for arm in task["arm_order"]:
            folder = out / "evaluation" / arm
            command = evaluator_command(payload["source"], folder, mem, nav, arm=arm, role=query["analysis_role"])
            progress = {"task_index": index, "history_index": task["history_index"], "query_id": query["query_id"],
                "scene": payload["scene"], "arm": arm, "completed_arms": len(summary["queries"]), "command": command}
            dump(durable / "progress.json", progress)
            dump(out / "progress.json", progress)
            print(f"QUERY {index} {payload['scene']} {query['query_id']} {arm}", flush=True)
            offset = pose_log.stat().st_size if pose_log.exists() else 0
            elapsed = run_child(command, out / "logs" / f"{arm}.log", environment=execution_environment())
            with pose_log.open() as stream:
                stream.seek(offset)
                dump(folder / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
            terminal = load(folder / "terminal_measurements.json")
            require(len(terminal) == 1, "one terminal per target/arm required")
            row = dict(terminal[0], scene=payload["scene"], episode=payload["episode"], role=query["analysis_role"],
                history_index=task["history_index"], query_id=query["query_id"], bin=query["bin"],
                q_eligible=query["q_eligible"], arm=arm, directory=str(folder), wall_seconds=elapsed)
            summary["queries"].append(row)
            dump(out / "summary.json", summary)
            dump(durable / "partial_summary.json", summary)
            print(f"DONE task={index} arm={arm} SR={row['reached']} ticks={row['steps']} SPL={row['spl']:.6f}", flush=True)
    summary["completed"] = True
    dump(out / "summary.json", summary)
    run_child([HAB_PY, str(HERE), "verify", "--out", str(out)], out / "logs/verification.log",
              environment=execution_environment())


def verify(out):
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(out / "manifest.json"), load(out / "summary.json")
    for field in ("runtime_source_receipt", "evaluation_source_receipt"):
        require(sha(manifest[field]["path"]) == manifest[field]["sha256"], "source receipt changed")
    task, payload, query, frozen = load_task(manifest["population"], manifest["task"]["task_index"])
    require(task == manifest["task"], "task manifest changed")
    require(summary["completed"] and len(summary["queries"]) == 3, "incomplete paired task")
    require([r["arm"] for r in summary["queries"]] == task["arm_order"], "execution order changed")
    records, indexed = [], {}
    trace = frozen["trace"]
    for row in summary["queries"]:
        require(row["scene"] == payload["scene"] and row["query_id"] == query["query_id"]
                and row["role"] == query["analysis_role"], "wrong target")
        checked, evidence, plans, actions = verify_rollout(row)
        folder = Path(row["directory"])
        data = load(folder / "query_plans.json")
        require(data["rollout_traces"]["legA"] == trace["poses"], "history replay mismatch")
        require(data["rollout_traces"]["query"] == evidence["rollout_trace"], "query trajectory mismatch")
        require(data["replay"]["all_rgb_hashes_verified"]
                and data["replay"]["diffusion_samples_during_replay"] == 0, "history resampled")
        np.testing.assert_array_equal(actions[0]["position_before"], trace["end_position"])
        require(actions[0]["yaw_before"] == trace["end_yaw"], "start yaw mismatch")
        require(data["analysis_role_not_forwarded"] and "analysis_role" not in data["query_runtime_fields"], "role leak")
        forbidden = {"analysis_role", "max_online_a_covis", "q_eligible", "covis_curve", "floor_position",
                     "executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m"}
        for boundary in read_rows(folder / "memory_http_boundary.jsonl"):
            require(not forbidden.intersection(boundary["sent_fields"]), "privileged memory request")
        checked.update(query_id=query["query_id"], history_index=task["history_index"], bin=query["bin"],
            q_eligible=query["q_eligible"], wall_seconds=row["wall_seconds"], counts=data["counts"],
            total_turn_deg=sum(abs(math.degrees(a["actual_yaw"]-a["yaw_before"])) for a in actions))
        records.append(checked)
        indexed[row["arm"]] = (row, data, plans, actions)
    native, cec = indexed["native"], indexed["cec"]
    for row, data, _, _ in indexed.values():
        require(row["first_query_rgb_sha256"] == native[0]["first_query_rgb_sha256"]
                and row["goal_xz_evaluator_only"] == native[0]["goal_xz_evaluator_only"]
                and data["replay"] == native[1]["replay"], "arms are not paired")
    takeover = sum(p["receipt"]["revisit_adapter_takeover"] is True for p in cec[2])
    if not takeover:
        require(len(native[2]) == len(cec[2]) and len(native[3]) == len(cec[3]), "no-takeover length differs")
        for a, b in zip(native[2], cec[2]):
            for field in ("selected_trajectory", "all_trajectory", "all_values", "position", "yaw"):
                np.testing.assert_array_equal(a[field], b[field])
        for a, b in zip(native[3], cec[3]):
            for field in ("actual_position", "actual_yaw", "action_kind"):
                require(a[field] == b[field], "no-takeover action differs")
        depths = []
        for row, *_ in (native, cec):
            http = read_rows(Path(row["directory"]) / "navdp_http_receipts.jsonl")
            depths.append([r["monocular_depth_receipt"] for r in http
                if r["query_active"] and r["path"] != "/memory_replay_step" and r["audit"]["image_calls"]])
        require(len(depths[0]) == len(depths[1]) == len(native[2]), "no-takeover depth count differs")
        for a, b in zip(*depths):
            for field in ("depth_png_sha256", "image_sha256", "scale_receipt_sha256", "frame_index"):
                require(a[field] == b[field], "no-takeover depth differs")
            for field in ("input_tensor_sha256", "output_tensor_sha256"):
                require(a["navdp_depth_raster"][field] == b["navdp_depth_raster"][field], "depth tensor differs")
    dump(out / "independent_verification.json", {"verified": True, "schema": SCHEMA, "scope": SCOPE,
        "population_sha256": POPULATION_SHA, "task": task, "query_arm_count": 3, "new_a_rollouts": 0,
        "records": records, "success": {a: indexed[a][0]["reached"] for a in ARMS},
        "cec_takeover_plans": takeover, "no_takeover_exact_native": takeover == 0,
        "verifier_sha256": sha(HERE)})
    print(f"VERIFIED task={task['task_index']}: three arms, full terminal SPL, repaired execution, RGB-only memory")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        sys.argv.pop(1)
        spec = importlib.util.spec_from_file_location("covis_execution_instrumentation",
            HERE.with_name("run_habitat_minimal_repair_local.py"))
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        wrapper.evaluate(query_main=evaluate_query)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "verify", "dry-run", "preflight-inputs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--durable", type=Path)
    parser.add_argument("--population", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--memnav-port", type=int, default=21750)
    parser.add_argument("--navdp-port", type=int, default=21751)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.out.resolve(), args.durable.resolve(), args.population.resolve(), args.index,
            args.memnav_port, args.navdp_port)
    elif args.mode == "verify":
        verify(args.out.resolve())
    elif args.mode == "preflight-inputs":
        population = load(args.population)
        for index in range(len(population["tasks"])):
            load_task(args.population, index)
        print(f"ALL {len(population['tasks'])} FROZEN TARGETS VERIFIED; no navigation run")
    else:
        import subprocess
        from MemNavData.run_repaired_fullmono_local import execution_environment
        source = {"scene": "SiKqEZx7Ejt", "episode": "episode_0003", "asset": "/unused.glb",
                  "source_episode": "/unused/episode_0003", "seed": 2026084203}
        for role in ("novel", "revisit"):
            for arm in ARMS:
                command = evaluator_command(source, args.out / role / arm, args.memnav_port, args.navdp_port,
                    arm=arm, role=role) + ["--contract_dry_run"]
                subprocess.run(command, env=execution_environment(), check=True)
        require(not args.out.exists(), "dry run created outputs")


if __name__ == "__main__":
    main()
