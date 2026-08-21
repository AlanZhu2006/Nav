#!/usr/bin/env python3
"""Collect one HM3D scene's four actual-online monocular Goal-A traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from MemNavData.deterministic_eval_protocol import validate_leg1_trace
from MemNavData.hm3d_fullmono_mixed_role import (
    audit_goal_a_plans,
    bind_parent_manifest,
    require,
    resolve_parent_scene,
)
from MemNavData.materialize_online_a_traces import native_control_audit


SCHEMA = "hm3d_fullmono_goal_a_scene_v1_20260820"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(root: Path, episode: str) -> dict[str, str]:
    with (root / "metric.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["episode"] == episode]
    require(len(rows) == 1, f"{episode}: expected one metric row")
    return rows[0]


def run(command: list[str], log: Path) -> None:
    with log.open("x") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    require(result.returncode == 0,
            f"Goal-A evaluator failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    parent, expected_parent_sha = bind_parent_manifest(
        protocol, args.protocol, args.parent_manifest)
    scenes = protocol["dataset"]["scenes"]
    spec, scene = resolve_parent_scene(
        protocol, parent, args.scene_index)
    source_rows = parent["episodes"][scene]
    count = int(protocol["dataset"]["episodes_per_scene"])
    require(len(source_rows) in {0, count},
            f"{scene}: expected zero or {count} frozen source episodes")
    if not source_rows:
        require("parent_index" not in spec and
                args.scene_index not in parent["evaluation_scene_indices"],
                f"{scene}: unauthorized empty source scene")

    asset_row = parent["assets"][scene]
    scene_file = Path(asset_row["glb_path"])
    require(scene_file.is_file(), f"explicit HM3D asset missing: {scene_file}")
    require(sha256(scene_file) == asset_row["glb_sha256"],
            f"{scene}: explicit HM3D asset hash changed")
    episode_root = Path(parent["paths"]["generated_root"]) / scene
    require(episode_root.is_dir(), f"{scene}: generated episode root missing")

    scene_root = (
        args.run_root / "goal_a" / "scenes" /
        f"{args.scene_index:02d}_{scene}"
    )
    require(not scene_root.exists(), f"Goal-A output exists: {scene_root}")
    (scene_root / "logs").mkdir(parents=True)

    base_seed = int(protocol["goal_a"]["base_seed"])
    records: list[dict[str, Any]] = []
    for episode_rank, source in enumerate(source_rows):
        episode = str(source["episode"])
        seed = base_seed + 100 * args.scene_index + episode_rank
        output = scene_root / episode
        command = [
            args.hab_python, "-u",
            str(args.source_root / "MemNavData/eval_2leg_habitat.py"),
            "--episode_root", str(episode_root),
            "--episode_ids", episode,
            "--scene", str(scene_file),
            "--scene_identity", scene,
            "--host", "127.0.0.1",
            "--port", str(args.memnav_port),
            "--novel_port", str(args.navdp_port),
            "--server_backend", "hybrid_pose",
            "--success_dist", str(protocol["goal_a"]["success_radius_m"]),
            "--max_steps", str(protocol["goal_a"]["maximum_steps"]),
            "--exec_horizon", str(protocol["goal_a"]["execution_horizon"]),
            "--trajectory_selector", "server",
            "--trajectory_selector_scope", "all",
            "--leg1_mode", "policy",
            "--leg1_goal_source", "own",
            "--write_leg1_trace",
            "--stop_after_leg1",
            "--seed", str(seed),
            "--terminal_uturn", "off",
            "--terminal_visual_refine", "off",
            "--deterministic_plan_seeds",
            "--retrieval_override", "off",
            "--certified_cdec_rescue", "off",
            "--certified_stagnation_graph", "off",
            "--revisit_controller", "navdp_mixed",
            "--hybrid_route", "native_sidecar",
            "--revisit_adapter", "legacy_metric",
            "--navdp_depth_source", "monocular_sidecar",
            "--out", str(output),
        ]
        run(command, scene_root / "logs" / f"{episode}.log")
        trace_path = output / f"{episode}_leg1_trace.json"
        plans_path = output / f"{episode}_plans.json"
        require(trace_path.is_file() and plans_path.is_file(),
                f"{scene}/{episode}: Goal-A receipts missing")
        trace = json.loads(trace_path.read_text())
        validate_leg1_trace(trace)
        require(trace["source_scene"] == scene,
                f"{scene}/{episode}: stable scene identity changed")
        require(trace["source_hybrid_route"] == "native_sidecar",
                f"{scene}/{episode}: Goal-A route changed")
        control = native_control_audit(trace)
        require(control["ok"], f"{scene}/{episode}: Goal-A was not native")
        depth_audit = audit_goal_a_plans(trace["plans"])
        row = metric_row(output, episode)
        reached = int(float(row["reached_A"]))
        require(reached == int(bool(trace["reached"])),
                f"{scene}/{episode}: trace/metric A outcome changed")
        records.append({
            "scene": scene,
            "scene_index": args.scene_index,
            "episode": episode,
            "episode_rank": episode_rank,
            "seed": seed,
            "reached_a": reached,
            "steps": int(trace["steps"]),
            "final_goal_dist_m": float(trace["final_goal_dist_m"]),
            "trace_sha256": sha256(trace_path),
            "trace_path": str(trace_path),
            "native_control_audit": control,
            "depth_audit": depth_audit,
        })

    completion = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scene": scene,
        "scene_index": args.scene_index,
        "protocol_sha256": sha256(args.protocol),
        "parent_manifest_sha256": expected_parent_sha,
        "source_episode_count": len(records),
        "goal_a_successes": sum(row["reached_a"] for row in records),
        "metric_depth_sensor_reads": 0,
        "target_source_episode_count": count,
        "source_generation_constructible": bool(source_rows),
        "all_sources_retained": len(records) == len(source_rows),
        "query_outcomes_read": False,
        "records": records,
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    path = scene_root / "completion.json"
    path.write_bytes(encoded)
    (scene_root / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )
    print(json.dumps({
        "status": "complete", "scene": scene,
        "sources": len(records),
        "goal_a_successes": completion["goal_a_successes"],
        "output": str(scene_root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
