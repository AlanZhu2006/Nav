#!/usr/bin/env python3
"""Collect four frozen actual-online full-mono MP3D Goal-A traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from MemNavData.deterministic_eval_protocol import validate_leg1_trace
from MemNavData.mdtec_monocular_cec_composition import audit_shared_leg_a, require


SCHEMA = "mp3d_table1_fullmono_goal_a_scene_v1_20260829"
PROTOCOL_SCHEMA = "mp3d_table1_fullmono_source_expansion_protocol_v1_20260829"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    require(result.returncode == 0,
            f"Goal-A evaluator failed ({result.returncode}); see {log}")


def metric_row(root: Path, episode: str) -> dict[str, str]:
    with (root / "metric.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["episode"] == episode]
    require(len(rows) == 1, f"{episode}: expected one metric row")
    return rows[0]


def audit_trace(output: Path, *, scene: str, scene_index: int,
                episode: str, episode_rank: int, seed: int) -> dict[str, Any]:
    trace_path = output / f"{episode}_leg1_trace.json"
    plans_path = output / f"{episode}_plans.json"
    require(trace_path.is_file() and plans_path.is_file(),
            f"{scene}/{episode}: Goal-A receipts missing")
    trace = json.loads(trace_path.read_text())
    validate_leg1_trace(trace)
    require(str(trace["source_scene"]) == scene
            and str(trace["episode"]) == episode,
            f"{scene}/{episode}: trace identity changed")
    plans = json.loads(plans_path.read_text())["legA"]
    outcome = {
        "plans": plans,
        "navdp_depth_source": "monocular_sidecar",
        "metric_depth_sensor_consumed_any": any(
            plan.get("metric_depth_sensor_consumed") for plan in plans
        ),
    }
    audit_shared_leg_a(outcome)
    row = metric_row(output, episode)
    reached = int(float(row["reached_A"]))
    require(reached == int(bool(trace["reached"])),
            f"{scene}/{episode}: trace/metric outcome changed")
    return {
        "scene": scene,
        "scene_index": scene_index,
        "episode": episode,
        "episode_rank": episode_rank,
        "seed": seed,
        "reached_a": reached,
        "steps": int(trace["steps"]),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256_file(trace_path),
        "metric_depth_sensor_reads": 0,
    }


def collect(*, source_root: Path, run_root: Path, protocol_path: Path,
            manifest_path: Path, scene_index: int, hab_python: str,
            memnav_port: int, navdp_port: int, smoke: bool = False,
            smoke_max_steps: int = 80) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "source-expansion protocol changed")
    source = protocol["expansion_source"]
    require(sha256_file(manifest_path) == source["manifest_sha256"],
            "phase-2 source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    scenes = list(manifest["selection"]["selected_scenes"])
    require(len(scenes) == int(source["scenes"])
            and len(scenes) == len(set(scenes)), "source scene set changed")
    declared_scenes = protocol["dataset"]["scenes"]
    require([int(row["rank"]) for row in declared_scenes]
            == list(range(len(scenes))), "protocol scene ranks changed")
    require([str(row["scene_id"]) for row in declared_scenes] == scenes,
            "protocol/manifest scene order changed")
    require(0 <= scene_index < len(scenes), "scene index out of range")
    scene = str(scenes[scene_index])
    frozen_episodes = list(source["episode_ids"])
    rows = list(manifest["episodes"][scene])
    require([str(row["episode"]) for row in rows] == frozen_episodes,
            f"{scene}: frozen episode order changed")

    asset = Path(manifest["paths"]["asset_root"]) / scene / f"{scene}.glb"
    require(asset.is_file() and sha256_file(asset)
            == manifest["assets"][scene]["sha256"],
            f"{scene}: asset changed")
    episode_root = Path(manifest["paths"]["expanded_episode_root"]) / scene
    for row in rows:
        episode = str(row["episode"])
        root = episode_root / episode
        files = row["files"]
        expected = {
            root / "meta/gen_meta.json": files["metadata"]["sha256"],
            root / "data/chunk-000/episode_000000.parquet":
                files["parquet"]["sha256"],
            root / "goal_image.jpg": files["goal"]["sha256"],
        }
        for path, digest in expected.items():
            require(path.is_file() and sha256_file(path) == digest,
                    f"{scene}/{episode}: source artifact changed: {path.name}")

    output_root = (run_root / ("smoke" if smoke else "goal_a") / "scenes" /
                   f"{scene_index:02d}_{scene}")
    require(not output_root.exists(), f"Goal-A output exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "logs").mkdir()
    selected_rows = rows[:1] if smoke else rows
    maximum_steps = (smoke_max_steps if smoke
                     else int(protocol["goal_a"]["maximum_steps"]))
    records = []
    for episode_rank, source_row in enumerate(selected_rows):
        episode = str(source_row["episode"])
        seed = int(protocol["goal_a"]["base_seed"]) + episode_rank
        output = output_root / f"{episode}_leg_a_trace"
        command = [
            hab_python, "-u",
            str(source_root / "MemNavData/eval_2leg_habitat.py"),
            "--episode_root", str(episode_root),
            "--episode_ids", episode,
            "--scene", str(asset),
            "--scene_identity", scene,
            "--host", "127.0.0.1",
            "--port", str(memnav_port),
            "--novel_port", str(navdp_port),
            "--server_backend", "hybrid_pose",
            "--success_dist", str(protocol["goal_a"]["success_radius_m"]),
            "--max_steps", str(maximum_steps),
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
            "--hybrid_route", "phase",
            "--revisit_adapter", "legacy_metric",
            "--navdp_depth_source", "monocular_sidecar",
            "--out", str(output),
        ]
        run(command, output_root / "logs" / f"{episode}.log")
        records.append(audit_trace(
            output, scene=scene, scene_index=scene_index,
            episode=episode, episode_rank=episode_rank, seed=seed,
        ))

    completion = {
        "schema_version": SCHEMA,
        "status": "complete",
        "formal": not smoke,
        "scene": scene,
        "scene_index": scene_index,
        "protocol_sha256": sha256_file(protocol_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_episode_count": len(records),
        "target_source_episode_count": (1 if smoke else len(rows)),
        "goal_a_successes": sum(row["reached_a"] for row in records),
        "metric_depth_sensor_reads": 0,
        "query_outcomes_generated": False,
        "query_outcomes_read": False,
        "records": records,
    }
    path = output_root / "completion.json"
    encoded = json.dumps(completion, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded)
    (output_root / "completion.json.sha256").write_text(
        hashlib.sha256(encoded.encode()).hexdigest() + "  completion.json\n"
    )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--hab-python", required=True)
    parser.add_argument("--memnav-port", type=int, required=True)
    parser.add_argument("--navdp-port", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-steps", type=int, default=80)
    args = parser.parse_args()
    result = collect(
        source_root=args.source_root.resolve(),
        run_root=args.run_root.resolve(),
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        scene_index=args.scene_index,
        hab_python=args.hab_python,
        memnav_port=args.memnav_port,
        navdp_port=args.navdp_port,
        smoke=args.smoke,
        smoke_max_steps=args.smoke_max_steps,
    )
    print(json.dumps({
        "status": result["status"],
        "formal": result["formal"],
        "scene": result["scene"],
        "sources": result["source_episode_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
