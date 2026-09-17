"""Fixed-history native/GEM pairs for NavDP, ViNT and NoMaD.

One cell owns one controller and one history, and executes both roles with
both arms in the same model processes. Query roles never cross model HTTP.
"""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_repaired_fullmono_local import (
    HAB_PY, MEM_PY, MEM_CKPT, LINGBOT, execution_environment, evaluator_command,
    private_servers, run_child, open_port)

HERE = Path(__file__).resolve()
CONTROLLERS = ("navdp", "vint", "nomad")
ARMS = ("native", "cec")
POPULATIONS = {
    "hm3d": ("/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914/population/natural_direction",
             "f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01", 28, 21),
    "mp3d": ("/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/population/natural_direction",
             "a33f210fdd0cfa84e82c4d403ac79056dcc7959cd1ce84bf62bec8c5632deb69", 42, 25),
}


def load(path):
    return json.loads(Path(path).read_text())


def history(cell):
    benchmark = Path(cell["benchmark"])
    assert sha(benchmark / "manifest.json") == cell["benchmark_sha256"]
    h = load(benchmark / "manifest.json")["episodes"][cell["history_index"]]
    assert h["scene"] == cell["scene"] and h["episode"] == cell["episode"]
    folder = benchmark / h["scene"] / h["episode"]
    payload = load(folder / "role_pairs.json")
    assert sha(folder / "role_pairs.json") == cell["role_pairs_sha256"]
    source = Path(payload["online_a_episode"])
    assert sha(source / "receipt.json") == payload["online_a_receipt_sha256"]
    assert sha(source / "online_a_trace.json") == payload["online_a_trace_sha256"]
    return payload, dict(source=source, receipt=load(source / "receipt.json"), trace=load(source / "online_a_trace.json")), folder


def freeze(out, local=False):
    if out.exists():
        raise FileExistsError(out)
    populations = POPULATIONS
    if local:
        from MemNavData.run_cec_stream_depth_closed_loop import BENCH
        populations = {"mp3d_consumed_smoke": (str(BENCH), sha(BENCH / "manifest.json"), None, None)}
    cells = []
    for dataset, (root, digest, count, scenes) in populations.items():
        assert sha(Path(root) / "manifest.json") == digest
        rows = load(Path(root) / "manifest.json")["episodes"]
        if not local:
            assert len(rows) == count and len({h["scene"] for h in rows}) == scenes
        else:
            rows = rows[:1]
        for controller in CONTROLLERS:
            for i, h in enumerate(rows):
                path = Path(root) / h["scene"] / h["episode"] / "role_pairs.json"
                queries = [q for pair in load(path)["pairs"] for q in pair["queries"]]
                assert len(queries) == 2 and {q["analysis_role"] for q in queries} == {"novel", "revisit"}
                cell = dict(index=len(cells), dataset=dataset, controller=controller,
                    history_index=i, benchmark=root, benchmark_sha256=digest,
                    scene=h["scene"], episode=h["episode"], role_pairs_sha256=sha(path))
                history(cell)
                cells.append(cell)
    dump(out, dict(schema="table1_repaired_three_controllers_v1", local=local,
        scope="consumed interface pilot" if local else "fixed historical query runtime correction; not fresh A",
        controllers=list(CONTROLLERS), max_steps=600, success_radius_m=1., exec_horizon=8,
        arms=list(ARMS), cells=cells, total_rollouts=4*len(cells),
        authority_policy="strict_certificate", nomad_goal_mask=0, nomad_samples=8,
        image_controller_distance_stop_mask=False, native_is_selected_controller=True,
        frozen_protocol_sha256=sha(ROOT / "MemNavData/TABLE1_REPAIRED_THREE_CONTROLLER_PROTOCOL_20260910.md")))


def command(cell, role, arm, out, mem, controller_port):
    _, frozen, _ = history(cell)
    source = dict(scene=cell["scene"], episode=cell["episode"], seed=frozen["trace"]["episode_seed"],
                  asset=frozen["receipt"]["source_asset"], source_episode=frozen["receipt"]["source_episode"])
    cmd = evaluator_command(source, out, mem, controller_port, arm=arm, role=role, benchmark=cell["benchmark"])
    cmd[2:4] = [str(HERE), "eval"]
    return cmd


def evaluate_query():
    import numpy as np
    import pandas as pd
    import eval_shared_online_role_pairs as shared
    base, args = shared.base, shared.args
    _, backend = shared.validate_cli()
    if args.contract_dry_run:
        print("TABLE1 REPAIRED CLI OK")
        return
    plan = load(os.environ["TABLE1_PLAN"])
    cell = plan["cells"][int(os.environ["TABLE1_INDEX"])]
    payload, frozen, folder = history(cell)
    if cell["controller"] != "navdp":
        from MemNavData.image_controller_goal_adapter import ImageControllerGoalAdapter
        ImageControllerGoalAdapter(base, cell["controller"], args.out).install()
    query = next(q for pair in payload["pairs"] for q in pair["queries"]
                 if q["analysis_role"] == args.role_pair_query_role)
    receipt, trace = frozen["receipt"], frozen["trace"]
    goal_path = folder / query["goal_rgb"]
    assert sha(goal_path) == query["goal_rgb_sha256"]
    carrier = Path(receipt["source_episode"]) / "data/chunk-000/episode_000000.parquet"
    assert sha(carrier) == receipt["source_parquet_sha256"]
    intrinsic = np.stack([np.asarray(r, float) for r in pd.read_parquet(carrier).iloc[0]["observation.camera_intrinsic"]])
    assert float(receipt["camera_height_m"]) == base.CAM_H
    assert sha(receipt["source_asset"]) == receipt["source_asset_sha256"]
    sim = base.make_sim(args.scene, "", agent_radius=.30)
    try:
        base.srv_reset(camera_height=receipt["camera_height_m"], seed=trace["episode_seed"],
            episode_len=len(trace["poses"])+600, camera_intrinsic=intrinsic,
            causal_history_sha256=payload["online_a_trace_sha256"])
        a, replay = shared.replay_prefix(frozen)
        goal = np.asarray(query["floor_position"], float)
        ok, geo, _ = base.geodesic(sim.pathfinder, a["end_pos"], goal)
        assert ok and abs(geo-query["geodesic_from_a_end_m"]) <= .05
        result = base.run_policy_leg(sim, sim.pathfinder, a["end_pos"], a["end_psi"],
            goal_path.read_bytes(), goal[[0,2]], float(geo), None, terminal_mode="off",
            goal_yaw=query["yaw_rad"], camera_intrinsic=intrinsic, policy_backend=backend,
            episode_seed=int(trace["episode_seed"]), leg_index=1)
        dump(Path(args.out) / "query_plans.json", dict(controller=cell["controller"],
            original_goal_sha256=sha(goal_path), query_leg=result["plans"], replay=replay,
            rollout_traces=dict(legA=trace["poses"], query=result["rollout_trace"]),
            counts=shared.router_counts(result["plans"]), depth=shared.depth_counts(result["plans"])))
    finally:
        sim.close()


@contextmanager
def image_servers(out, mem_port, controller_port, controller):
    if mem_port == controller_port or any(open_port(p) for p in (mem_port, controller_port)):
        raise ValueError("Private ports occupied")
    dependency = os.environ.get("REPAIRED_DEPENDENCIES", str(ROOT / ".diagnostics/dependencies/python"))
    glue = os.environ.get("REPAIRED_LIGHTGLUE", str(ROOT / ".diagnostics/dependencies/LightGlue"))
    intern = os.environ.get("REPAIRED_INTERNNAV", str(ROOT / "InternNav"))
    shared = f"{ROOT}:{ROOT}/MemNavData:{dependency}:{glue}:{intern}/src/diffusion-policy"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        LINGBOT_REPO=str(LINGBOT), LINGBOT_WEIGHTS=str(LINGBOT / "weights/lingbot-map-long.pt"),
        MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
        MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
        MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1", MEMNAV_REPORT_TO="none",
        NAVDP_DISABLE_VIDEO="1", LINGBOT_POSE_LOG=str(out / "lingbot_pose_readout.jsonl"))
    model_py = os.environ.get("TABLE1_IMAGE_PY", str(ROOT / ".diagnostics/controller_portability_20260821/envs/vint/bin/python"))
    checkpoints = Path(os.environ.get("TABLE1_CHECKPOINTS", ROOT / ".diagnostics/controller_portability_20260821/checkpoints"))
    settings = [
        ("memnav", mem_port, dict(env, PYTHONPATH=f"{ROOT}/NavDP/baselines/memnav:{shared}"),
         [MEM_PY, "-u", str(ROOT / "MemNavData/lingbot_pose_diagnostic_server.py"),
          "--host", "127.0.0.1", "--port", str(mem_port), "--checkpoint", str(MEM_CKPT),
          "--internnav_root", intern, "--num_samples", "16", "--exclude_recent", "32", "--retrieval", "raw",
          "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
          "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60", "--flow_gate", "auto",
          "--buffer_root", str(out / "buffer"), "--certified_relocalization", "--certified_reference_depth_source", "canonical",
          "--lightglue_repo", glue, "--lightglue_dependency_root", dependency, "--lightglue_max_keypoints", "2048"]),
        (controller, controller_port, dict(env, PYTHONPATH=f"{ROOT}:{intern}/src/diffusion-policy",
          IMAGE_CONTROLLER_AUDIT=str(out / "image_controller_http.jsonl")),
         [model_py, "-u", str(ROOT / "MemNavData/image_controller_repaired_server.py"),
          "--controller", controller, "--checkpoint", str(checkpoints / f"{controller}.pth"), "--port", str(controller_port)])]
    children, handles = [], []
    try:
        for name, port, environment, cmd in settings:
            cwd = out / "runtime" / name
            cwd.mkdir(parents=True)
            handle = (out / "logs" / f"{name}.log").open("x")
            handles.append(handle)
            process = subprocess.Popen(cmd, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            children.append(process)
            dump(out / "owned_processes.json", [dict(pid=p.pid, command=p.args) for p in children])
            deadline = time.monotonic()+600
            while not open_port(port):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"{name} failed to start")
                time.sleep(1)
            print(f"READY {name} {process.pid} {port}", flush=True)
        yield
    finally:
        for p in reversed(children):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for h in handles:
            h.close()


def run(args):
    plan = load(args.plan)
    cell = plan["cells"][args.index]
    assert cell["index"] == args.index and cell["controller"] in CONTROLLERS
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    history(cell)
    env = dict(execution_environment(), TABLE1_PLAN=str(args.plan.resolve()), TABLE1_INDEX=str(args.index))
    if cell["controller"] != "navdp":
        env["MINIMAL_DEPTH_RASTER"] = ""
    schedule = [(role, arm) for j, role in enumerate(("novel", "revisit"))
                for arm in (ARMS if (cell["history_index"]+j)%2 == 0 else ARMS[::-1])]
    paths = [HERE, ROOT / "MemNavData/image_controller_policy.py", ROOT / "MemNavData/image_controller_goal_adapter.py",
        ROOT / "MemNavData/image_controller_repaired_server.py", ROOT / "MemNavData/verify_table1_repaired.py",
        ROOT / "MemNavData/eval_2leg_habitat.py", ROOT / "MemNavData/run_habitat_minimal_repair_local.py",
        ROOT / "MemNavData/certified_relocalization_runtime.py", ROOT / "MemNavData/certified_relocalization_contract.py",
        ROOT / "MemNavData/bounded_pursuit.py", ROOT / "MemNavData/navdp_front_goal_adapter.py"]
    dump(args.out / "manifest.json", dict(cell=cell, plan=str(args.plan.resolve()), plan_sha256=sha(args.plan),
        schedule=schedule, max_steps=600, source_hashes={str(p):sha(p) for p in paths}))
    for role, arm in schedule:
        run_child(command(cell, role, arm, args.out / "dry", args.mem_port, args.nav_port)+["--contract_dry_run"],
                  args.out / "logs" / f"dry_{role}_{arm}.log", environment=env)
    server_context = (private_servers(args.out, args.mem_port, args.nav_port) if cell["controller"] == "navdp"
                      else image_servers(args.out, args.mem_port, args.nav_port, cell["controller"]))
    summary = dict(completed=False, cell=cell, records=[])
    dump(args.out / "summary.json", summary)
    with server_context:
        for role, arm in schedule:
            target = args.out / "evaluation" / role / arm
            pose_log = args.out / "lingbot_pose_readout.jsonl"
            offset = pose_log.stat().st_size if pose_log.exists() else 0
            dump(args.out / "progress.json", dict(controller=cell["controller"], role=role, arm=arm,
                completed=len(summary["records"]), total=4))
            print(f"START {cell['controller']} {role} {arm}", flush=True)
            seconds = run_child(command(cell, role, arm, target, args.mem_port, args.nav_port),
                args.out / "logs" / f"{role}_{arm}.log", environment=env)
            with pose_log.open() as stream:
                stream.seek(offset)
                dump(target / "lingbot_frame_poses.json", [json.loads(l) for l in stream if l.strip()])
            terminal = load(target / "terminal_measurements.json")
            assert len(terminal) == 1
            row = dict(terminal[0], role=role, arm=arm, scene=cell["scene"], controller=cell["controller"],
                       directory=str(target), wall_seconds=seconds)
            summary["records"].append(row)
            dump(args.out / "summary.json", summary)
            print(f"DONE {cell['controller']} {role} {arm} SR={row['reached']} steps={row['steps']}", flush=True)
    summary["completed"] = True
    dump(args.out / "summary.json", summary)
    run_child([HAB_PY, str(ROOT / "MemNavData/verify_table1_repaired.py"), str(args.out)],
              args.out / "logs/verification.log", environment=env)
    print("VERIFIED complete controller/history cell", flush=True)


def main():
    mode = sys.argv.pop(1)
    if mode == "eval":
        from MemNavData.run_habitat_minimal_repair_local import evaluate
        evaluate(query_main=evaluate_query)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--mem-port", type=int, default=21930)
    parser.add_argument("--nav-port", type=int, default=21931)
    args = parser.parse_args()
    if mode == "freeze":
        freeze(args.out, args.local)
    elif mode == "run":
        run(args)
    else:
        parser.error("expected freeze/run/eval")


if __name__ == "__main__":
    main()
