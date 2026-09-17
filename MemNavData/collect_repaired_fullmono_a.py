"""Collect every declared source A in one scene with the fixed repair stack.

This stage cannot construct or evaluate a Novel/Revisit query. The original
private servers, Goal-A CLI and independent rollout verifier are reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from audit_repaired_fullmono_design import sha

SCHEMA = "repaired_fullmono_actual_a_collection_20260909_v1"


def dump(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def scene_sources(plan, scene_rank):
    if not plan["new_a_required"] or plan["old_a_results_for_selection"]:
        raise ValueError("Actual new A required")
    if plan["status"] != "prepared_not_submitted_not_query_sealed":
        raise ValueError("Unexpected source plan")
    sources = [s for s in plan["sources"] if s["scene_rank"] == scene_rank]
    if not sources or len({s["scene"] for s in sources}) != 1:
        raise ValueError("Expected one nonempty source scene")
    for s in sources:
        expected = 2026082200 + 100 * s["scene_rank"] + s["episode_rank"]
        if s["seed"] != expected:
            raise ValueError("Source seed/rank changed")
    return sources


def bind_source(source):
    files = {f["path"]: f["sha256"] for f in source["task_files"].values()}
    asset = source["asset"]
    files[asset["glb_path"]] = asset["glb_sha256"]
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Source changed: {path}")
    metadata = json.loads(Path(source["task_files"]["metadata"]["path"]).read_text())
    episode = Path(source["source_episode"])
    goal_a = episode / "videos/chunk-000/observation.images.rgb" / f"{int(metadata['switch_idx'])-1}.jpg"
    files[str(goal_a)] = sha(goal_a)
    return {"scene": source["scene"], "episode": source["episode"], "seed": source["seed"],
            "source_index": source["source_index"], "source_scene_rank": source["scene_rank"],
            "source_episode_rank": source["episode_rank"], "asset": asset["glb_path"],
            "source_episode": str(episode), "source_episode_root": str(episode.parent.parent),
            "source_files": files, "goal_a_rgb_path": str(goal_a), "goal_a_rgb_sha256": files[str(goal_a)]}


def collect(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, PROFILE, evaluator_command, execution_environment, private_servers, run_child,
    )
    plan_sha = sha(args.plan)
    if plan_sha != args.plan_sha256:
        raise ValueError("Source plan SHA mismatch")
    plan = json.loads(args.plan.read_text())
    sources = [bind_source(s) for s in scene_sources(plan, args.scene_rank)]
    out, durable = args.out.resolve(), args.durable.resolve()
    if out.name != "task":
        raise ValueError("Node-local work root must be named task for the shared archiver")
    out.mkdir(parents=True, exist_ok=False)
    durable.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    os.environ.update(REPAIRED_BUFFER_ROOT=str(out / "buffer"), REPAIRED_RUNTIME_ROOT=str(out / "runtime"))
    receipt = os.environ["REPAIRED_SOURCE_RECEIPT"]
    if sha(receipt) != plan["base_runtime_sha256"]:
        raise ValueError("Wrong repaired runtime bundle")
    manifest = {"schema": SCHEMA, "profile": PROFILE, "scope": plan["scope"], "sources": sources,
                "source_plan": str(args.plan.resolve()), "source_plan_sha256": plan_sha,
                "source_scene_rank": args.scene_rank, "queries_allowed": False,
                "runtime_source_receipt": {"path": receipt, "sha256": sha(receipt)},
                "collector_source_sha256": sha(__file__), "original_work_root": str(out)}
    dump(out / "manifest.json", manifest)
    dump(durable / "manifest.json", manifest)
    summary = {"schema": SCHEMA, "completed": False, "goal_a": [], "queries": [],
               "construction_performed": False, "source_count": len(sources)}
    dump(out / "summary.json", summary)
    poses = out / "lingbot_pose_readout.jsonl"
    with private_servers(out, args.memnav_port, args.navdp_port):
        for source in sources:
            name = f"{source['scene']}_{source['episode']}_goal_a"
            folder = out / "goal_a" / source["scene"] / source["episode"]
            offset = poses.stat().st_size if poses.exists() else 0
            progress = {"stage": "actual_mono_a", "source_index": source["source_index"],
                        "name": name, "completed_sources": len(summary["goal_a"]), "unix": time.time()}
            dump(durable / "progress.json", progress)
            command = evaluator_command(source, folder, args.memnav_port, args.navdp_port)
            wall = run_child(command, out / "logs" / f"{name}.log", environment=execution_environment())
            with poses.open() as f:
                f.seek(offset)
                dump(folder / "lingbot_frame_poses.json", [json.loads(line) for line in f if line.strip()])
            terminal = json.loads((folder / "terminal_measurements.json").read_text())
            if len(terminal) != 1:
                raise ValueError("Exactly one A terminal is required")
            row = dict(terminal[0], scene=source["scene"], episode=source["episode"], arm="native", role="goal_a",
                       source_index=source["source_index"], seed=source["seed"], directory=str(folder), wall_seconds=wall)
            summary["goal_a"].append(row)
            dump(out / "summary.json", summary)
            dump(durable / "partial_summary.json", summary)
            print(f"A_DONE {name} reached={row['reached']} steps={row['steps']}", flush=True)
    summary["completed"] = True
    dump(out / "summary.json", summary)
    run_child([HAB_PY, str(Path(__file__).resolve()), "verify", "--out", str(out)],
              out / "logs/verification.log", environment=execution_environment())


def verify(out):
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.materialize_online_a_traces import native_control_audit
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["queries_allowed"] or summary["queries"]:
        raise ValueError("Not an A-only run")
    if not summary["completed"] or len(summary["goal_a"]) != len(manifest["sources"]):
        raise ValueError("Incomplete A collection")
    if sha(manifest["source_plan"]) != manifest["source_plan_sha256"]:
        raise ValueError("Source plan changed")
    receipt = manifest["runtime_source_receipt"]
    if sha(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Runtime source changed")
    checked = []
    for source, row in zip(manifest["sources"], summary["goal_a"]):
        if any(row[k] != source[k] for k in ("scene", "episode", "seed", "source_index")):
            raise ValueError("A/source identity mismatch")
        result, _, _, _ = verify_rollout(row)
        trace_path = Path(row["directory"]) / f"{source['episode']}_leg1_trace.json"
        trace = json.loads(trace_path.read_text())
        if not native_control_audit(trace)["ok"] or trace["episode_seed"] != source["seed"]:
            raise ValueError("A was not the declared native rollout")
        if bool(trace["reached"]) != bool(row["reached"]):
            raise ValueError("A success differs from exact terminal recount")
        result.update(source_index=source["source_index"], trace_sha256=sha(trace_path))
        checked.append(result)
    dump(out / "independent_verification.json", {"verified": True, "schema": SCHEMA,
         "sources": len(checked), "goal_a": checked, "query_rollouts": 0, "construction_performed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("collect")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--scene-rank", type=int, required=True)
    run.add_argument("--durable", type=Path, required=True)
    run.add_argument("--memnav-port", type=int, required=True)
    run.add_argument("--navdp-port", type=int, required=True)
    check = sub.add_parser("verify")
    for command in (run, check):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args) if args.mode == "collect" else verify(args.out)
