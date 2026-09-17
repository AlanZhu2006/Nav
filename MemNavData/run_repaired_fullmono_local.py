#!/usr/bin/env python3
"""Two-source actual-mono integration, with the already-tested repair profile.

New A is executed once, all queries are constructed before evaluating any arm.
Local or explicitly bound HPC sources; no robot, production-default changes,
or outcome-dependent replenishment.
"""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_cec_stream_depth_closed_loop import (
    ROOT, BENCH, MEM_PY, HAB_PY, MEM_CKPT, NAV_CKPT, LINGBOT, hab_env, open_port,
)

HERE = Path(__file__).resolve()
PROFILE = "bounded_rgb_source_depth_front_goal_v1"
ARMS = ("native", "raw_fixed", "cec")
SCENES = ("gxdoqLR6rwA", "pLe4wQe7qrG")
ROUTES = {"native": ("native_sidecar", "legacy_metric"),
          "raw_fixed": ("phase", "raw_fixed_bearing_v1"),
          "cec": ("certified_relocalization", "verified_bearing_v1")}


def sources():
    manifest = json.loads((BENCH / "manifest.json").read_text())
    selected = manifest["episodes"][:2]
    if tuple(h["scene"] for h in selected) != SCENES:
        raise ValueError("The fixed consumed source order changed")
    rows = []
    for h in selected:
        receipt_path = Path(h["online_a_episode"]) / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        episode = Path(receipt["source_episode"])
        asset = Path(receipt["source_asset"])
        metadata = episode / "meta/gen_meta.json"
        parquet = episode / "data/chunk-000/episode_000000.parquet"
        meta = json.loads(metadata.read_text())
        goal = episode / "videos/chunk-000/observation.images.rgb" / f"{int(meta['switch_idx'])-1}.jpg"
        expected = {asset: receipt["source_asset_sha256"], metadata: receipt["source_metadata_sha256"],
                    parquet: receipt["source_parquet_sha256"], goal: receipt["goal_a_sha256"]}
        if any(sha(p) != digest for p, digest in expected.items()):
            raise ValueError("A source asset or start/goal carrier changed")
        rows.append({"scene": h["scene"], "episode": h["episode"], "asset": str(asset),
                     "source_episode": str(episode), "source_episode_root": str(episode.parent.parent),
                     "seed": 0, "source_files": {str(p): d for p, d in expected.items()},
                     "source_receipt": str(receipt_path), "source_receipt_sha256": sha(receipt_path)})
    return rows


def evaluator_command(source, out, mem_port, nav_port, *, arm="native", role=None, benchmark=None):
    route, adapter = ROUTES[arm]
    goal_a = role is None
    if goal_a and arm != "native":
        raise ValueError("Goal-A must be actual native; memory cannot help collect A")
    if not goal_a and role not in ("novel", "revisit"):
        raise ValueError("Query role is evaluator-only and must be explicit")
    source_root = (Path(source["source_episode"]).parent if goal_a
                   else Path(benchmark) / source["scene"])
    command = [HAB_PY, "-u", str(ROOT / "MemNavData/run_habitat_minimal_repair_local.py"),
               "goal_a" if goal_a else "eval", "--episode_root", str(source_root),
               "--episode_ids", source["episode"], "--scene", source["asset"],
               "--scene_identity", source["scene"], "--out", str(out),
               "--host", "127.0.0.1", "--port", str(mem_port), "--novel_port", str(nav_port),
               "--server_backend", "hybrid_pose", "--hybrid_route", route,
               "--revisit_adapter", adapter, "--navdp_depth_source", "monocular_sidecar",
               "--success_dist", "1.0", "--max_steps", "600", "--exec_horizon", "8",
               "--trajectory_selector", "server", "--trajectory_selector_scope", "all",
               "--leg1_mode", "policy" if goal_a else "shared_trace", "--leg1_goal_source", "own",
               "--seed", str(source["seed"]), "--terminal_uturn", "off", "--terminal_visual_refine", "off",
               "--deterministic_plan_seeds", "--retrieval_override", "off", "--certified_cdec_rescue", "off",
               "--certified_stagnation_graph", "off", "--cec_initial_bearing_alignment", "off",
               "--revisit_controller", "navdp_mixed"]
    if goal_a:
        command += ["--write_leg1_trace", "--stop_after_leg1"]
    else:
        command += ["--role_pair_scope", "consumed_integration", "--role_pair_query_role", role]
    return command


def execution_environment():
    return dict(hab_env(), MINIMAL_EXECUTOR="bounded_standard", MINIMAL_ACTOR_COLOR="rgb_v1",
                MINIMAL_DEPTH_RASTER="source_rgb", MINIMAL_FRONT_GOAL="heading_on")


@contextmanager
def private_servers(out, mem_port, nav_port, *, memory_control=False):
    if mem_port == nav_port or any(open_port(p) for p in (mem_port, nav_port)):
        raise ValueError("Private ports are occupied; do not reset another server")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               LINGBOT_REPO=str(LINGBOT), LINGBOT_WEIGHTS=str(LINGBOT / "weights/lingbot-map-long.pt"),
               MEMNAV_WINDOW="32", MEMNAV_NUM_SCALE="8", MEMNAV_MAX_FRAME_NUM="2048",
               MEMNAV_GROUND_SCALE_MAX="6.0", MEMNAV_GATE_FUSION="complementary",
               MEMNAV_AUX_POSE_CALIBRATION="empirical", MEMNAV_COLLISION_SELECT="1",
               MEMNAV_REPORT_TO="none", NAVDP_DISABLE_VIDEO="1")
    dependency_root = Path(os.environ.get("REPAIRED_DEPENDENCIES", ROOT / ".diagnostics/dependencies/python"))
    lightglue_root = Path(os.environ.get("REPAIRED_LIGHTGLUE", ROOT / ".diagnostics/dependencies/LightGlue"))
    internnav_root = Path(os.environ.get("REPAIRED_INTERNNAV", ROOT / "InternNav"))
    buffer_root = Path(os.environ.get("REPAIRED_BUFFER_ROOT", out / "buffer"))
    shared = f"{ROOT}:{ROOT}/MemNavData:{dependency_root}:{lightglue_root}:{internnav_root}/src/diffusion-policy"
    settings = [
        ("memnav", mem_port, dict(env, PYTHONPATH=f"{ROOT}/NavDP/baselines/memnav:{shared}",
                                 LINGBOT_POSE_LOG=str(out / "lingbot_pose_readout.jsonl")),
         [MEM_PY, "-u", str(ROOT / "MemNavData/lingbot_pose_diagnostic_server.py"),
          "--host", "127.0.0.1", "--port", str(mem_port), "--checkpoint", str(MEM_CKPT),
          "--internnav_root", str(internnav_root), "--num_samples", "16", "--exclude_recent", "32",
          "--retrieval", "raw", "--retrieval_candidate_top_k", "32", "--retrieval_candidate_min_gap", "16",
          "--graph_subgoal_spacing_m", "0.0", "--graph_subgoal_arrival_m", "0.60", "--flow_gate", "auto",
          "--buffer_root", str(buffer_root), "--certified_relocalization",
          "--certified_reference_depth_source", "canonical",
          "--lightglue_repo", str(lightglue_root),
          "--lightglue_dependency_root", str(dependency_root),
          "--lightglue_max_keypoints", "2048"]),
        ("navdp", nav_port, dict(env, PYTHONPATH=f"{ROOT}/NavDP/baselines/navdp:{shared}",
                                NAVDP_EXECUTION_INPUT_LOG=str(out / "navdp_input_audit.jsonl"),
                                NAVDP_DEPTH_RASTER_LOG_ROOT=str(out / "depth_raster_artifacts")),
         [MEM_PY, "-u", str(ROOT / "MemNavData/navdp_depth_raster_audit_server.py"),
          "--port", str(nav_port), "--checkpoint", str(NAV_CKPT), "--depth_source", "monocular_sidecar",
          "--require_monocular_depth_transaction",
          "--monocular_depth_url", f"http://127.0.0.1:{mem_port}/monocular_depth_query"]),
    ]
    children, handles = [], []
    try:
        for name, port, environment, command in settings:
            if memory_control:
                command = [command[0], "-u", str(ROOT / "MemNavData/private_gpu_cache_server.py"),
                           "--entrypoint", command[2], *command[3:]]
            handle = (out / "logs" / f"{name}.log").open("x")
            handles.append(handle)
            cwd = Path(os.environ.get("REPAIRED_RUNTIME_ROOT", out / "runtime")) / name
            cwd.mkdir(parents=True)
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            children.append(process)
            dump(out / "owned_processes.json", [{"pid": p.pid, "args": p.args} for p in children])
            deadline = time.monotonic() + 600
            while not open_port(port):
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"private {name} failed before ready")
                time.sleep(1)
            print(f"READY private {name} pid={process.pid} port={port}", flush=True)
        yield
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


def run_child(command, log, *, environment=None):
    started = time.monotonic()
    with log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=environment or hab_env(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.monotonic() - started


def construct(args):
    from MemNavData.materialize_hm3d_fullmono_online_a import materialize_scene
    from MemNavData.build_final14_role_pair_scene import build
    manifest = json.loads((args.out / "manifest.json").read_text())
    reports = []
    for rank, source in enumerate(manifest["sources"]):
        scene = source["scene"]
        destination = args.out / "construction" / scene
        materialization = materialize_scene(
            trace_root=args.out / "goal_a" / scene, scene=scene, asset=Path(source["asset"]),
            episode_root=Path(source["source_episode_root"]), source_episode_order=[source["episode"]],
            out=destination / "online_a", purpose=manifest["scope"])
        built = build(destination / "online_a", destination / "role_pairs",
                      scene_rank=int(source.get("source_scene_rank", rank)),
                      source_episode_order=[source["episode"]], maximum_histories=1, only_scene=scene)
        reports.append({"scene": scene, "materialization": materialization, "construction": built,
                        "benchmark": str(destination / "role_pairs/natural_direction")})
        print(f"CONSTRUCTED {scene}: {built['retained_standard_natural_histories']} histories", flush=True)
    dump(args.out / "construction_summary.json", {"reports": reports, "query_outcomes_read": False})


def source_files():
    # Include all local navigation Python modules, not only the CLI entrypoints.
    # This is a local audit snapshot; a portable HPC bundle needs its own closure.
    paths = list((ROOT / "MemNavData").glob("*.py"))
    paths += list((ROOT / "NavDP/baselines/navdp").glob("*.py"))
    paths += list((ROOT / "NavDP/baselines/memnav").glob("*.py"))
    paths += [ROOT / "MemNavData/REPAIRED_FULLMONO_LOCAL_PROTOCOL_20260908.md",
              MEM_CKPT, NAV_CKPT, LINGBOT / "weights/lingbot-map-long.pt"]
    return sorted(set(paths))


def local(args):
    out = args.out.resolve()
    source_manifest = getattr(args, "source_manifest", None)
    source_payload = json.loads(source_manifest.read_text()) if source_manifest is not None else None
    source_rows = source_payload["sources"] if source_payload is not None else sources()
    if len(source_rows) != 2 or len({s["scene"] for s in source_rows}) != 2:
        raise ValueError("The integration requires two fixed sources in two distinct scenes")
    for source in source_rows:
        if any(sha(p) != h for p, h in source["source_files"].items()):
            raise ValueError("A bound source changed before model startup")
    if any(open_port(p) for p in (args.memnav_port, args.navdp_port)):
        raise ValueError("Private port already in use")
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    files = source_files()
    manifest = {"schema": "repaired_fullmono_local_v1_20260908", "profile": PROFILE,
                "scope": source_payload["scope"] if source_payload else "consumed two-source MP3D integration, not paper confirmation",
                "sources": source_rows, "arms": list(ARMS), "max_steps": 600, "exec_horizon": 8,
                "history_source": "new actual mono-A", "query_role_runtime_visible": False,
                "reference_depth_source": "canonical", "goal_a_before_query_construction": True,
                "all_construction_before_query_evaluation": True,
                "seeds_by_source": [source["seed"] for source in source_rows],
                "code_and_weights": {str(p): sha(p) for p in files},
                "success": "evaluator planar GT distance <1m; not autonomous STOP",
                "collision": "standard NavMesh try_step, ideal low-level state; not rigid-body physics"}
    if source_manifest is not None:
        manifest["source_manifest"] = {"path": str(source_manifest.resolve()), "sha256": sha(source_manifest)}
    central_receipt = os.environ.get("REPAIRED_SOURCE_RECEIPT")
    if central_receipt:
        manifest["source_snapshot_mode"] = "immutable_bundle_reference"
        manifest["immutable_bundle_receipt"] = {"path": central_receipt, "sha256": sha(central_receipt)}
    else:
        manifest["source_snapshot_mode"] = "local_copy"
    dump(out / "manifest.json", manifest)
    for path in files:
        if not central_receipt and path.suffix in (".py", ".md"):
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    goal_a, queries = [], []
    pose_log = out / "lingbot_pose_readout.jsonl"

    def rollout(source, destination, arm, role=None, benchmark=None):
        offset = pose_log.stat().st_size if pose_log.exists() else 0
        name = f"{source['scene']}_{role or 'goal_a'}_{arm}"
        dump(out / "progress.json", {"stage": "goal_a" if role is None else "query",
             "current": name, "completed_a": len(goal_a), "completed_query_arms": len(queries),
             "supervisor_pid": os.getpid(), "updated_unix": time.time()})
        print(f"START {name}", flush=True)
        wall = run_child(evaluator_command(source, destination, args.memnav_port, args.navdp_port,
                                           arm=arm, role=role, benchmark=benchmark),
                         out / "logs" / f"{name}.log", environment=execution_environment())
        with pose_log.open() as stream:
            stream.seek(offset)
            dump(destination / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
        records = json.loads((destination / "terminal_measurements.json").read_text())
        if len(records) != 1:
            raise RuntimeError("Expected exactly one complete rollout")
        row = dict(records[0], scene=source["scene"], episode=source["episode"], arm=arm,
                   role=role or "goal_a", directory=str(destination), wall_seconds_including_audit=wall)
        print(f"DONE {name}: reached={row['reached']} steps={row['steps']} wall={wall:.1f}s", flush=True)
        return row

    try:
        with private_servers(out, args.memnav_port, args.navdp_port):
            for source in source_rows:
                goal_a.append(rollout(source, out / "goal_a" / source["scene"] / source["episode"], "native"))
                dump(out / "goal_a_results.json", goal_a)
            dump(out / "progress.json", {"stage": "construction", "completed_a": len(goal_a),
                                         "completed_query_arms": 0, "updated_unix": time.time()})
            run_child([HAB_PY, "-u", str(HERE), "construct", "--out", str(out)], out / "logs/construction.log")
            built = json.loads((out / "construction_summary.json").read_text())
            for index, report in enumerate(built["reports"]):
                source = source_rows[index]
                benchmark = Path(report["benchmark"])
                histories = json.loads((benchmark / "manifest.json").read_text())["episodes"]
                if not histories:
                    continue
                if len(histories) != 1 or histories[0]["episode"] != source["episode"]:
                    raise RuntimeError("Unexpected constructed population")
                order_start = int(source.get("source_scene_rank", index)) % len(ARMS)
                order = ARMS[order_start:] + ARMS[:order_start]
                for role in ("novel", "revisit"):
                    for arm in order:
                        dest = out / "evaluation" / source["scene"] / role / arm
                        queries.append(rollout(source, dest, arm, role, benchmark))
                        dump(out / "query_results.json", queries)
            changed = [str(p) for p in files if sha(p) != manifest["code_and_weights"][str(p)]]
            dump(out / "summary.json", {"completed": True, "goal_a": goal_a, "queries": queries,
                                        "changed_source_files": changed})
            if changed:
                raise RuntimeError("Source changed during the integration")
        # This belongs to the same supervisor; no unstarted watcher is needed.
        run_child([HAB_PY, "-u", str(ROOT / "MemNavData/verify_repaired_fullmono_local.py"), str(out)],
                  out / "logs/verification.log")
        run_child([HAB_PY, "-u", str(ROOT / "MemNavData/verify_repaired_fullmono_local.py"), str(out), "--render"],
                  out / "logs/render.log")
        dump(out / "progress.json", {"stage": "verified_and_videos_exported", "completed_a": len(goal_a),
                                     "completed_query_arms": len(queries), "updated_unix": time.time()})
    except BaseException as error:
        dump(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "construct"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memnav-port", type=int, default=21710)
    parser.add_argument("--navdp-port", type=int, default=21711)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    (local if args.mode == "local" else construct)(args)
