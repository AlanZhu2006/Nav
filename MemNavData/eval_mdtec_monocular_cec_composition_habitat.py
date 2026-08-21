#!/usr/bin/env python3
"""Per-scene evaluator for the MDTEC monocular x CEC composition experiment.

Deliberately does NOT reimplement ``eval_2leg_habitat``'s simulation loop.
It shells out to the already-audited ``eval_2leg_habitat.py`` CLI three
times per episode, against the SAME already-running MemNav/NavDP server
pair (started once by ``run_mdtec_monocular_cec_composition_scene.sh``,
mirroring Gate D's one-persistent-process-per-scene contract):

  1. ``leg1_mode=policy --write_leg1_trace --stop_after_leg1`` -- the ONE
     shared causal monocular Goal-A rollout, recorded to a trace file.
  2. ``leg1_mode=shared_trace`` with ``--hybrid_route native_sidecar`` --
     resets the server, replays the recorded Goal-A trace, then runs
     Goal-B under plain native ImageGoal control (``raw_native``).
  3. Same shared-trace replay, ``--hybrid_route certified_relocalization``
     -- Goal-B under the existing CEC certificate, native fallback on
     reject (``raw_cec``).

This mirrors how Gate D drove ``eval_2leg_habitat`` primitives directly
rather than re-deriving them, but at the process boundary instead of the
function boundary, because ``shared_trace`` mode's built-in reset+replay
is the only verified mechanism in this codebase for giving two downstream
arms an identical, non-cross-contaminated causal history reconstructed
from one recorded rollout.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mdtec_monocular_cec_composition import (
    ARMS,
    HYBRID_ROUTE,
    audit_arm_leg_b,
    audit_shared_leg_a,
    require,
    rotated_arm_order,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_eval_2leg(base_args: list[str], extra: dict[str, str],
                  flags: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = list(base_args) + ["--out", str(out_dir)]
    for key, value in extra.items():
        cmd += [f"--{key}", str(value)]
    cmd += flags
    result = subprocess.run(cmd, check=False)
    require(result.returncode == 0,
            f"eval_2leg_habitat sub-invocation failed: {' '.join(cmd)}")


def load_episode_plans(out_dir: Path, episode: str) -> dict[str, Any]:
    path = out_dir / f"{episode}_plans.json"
    require(path.is_file(), f"missing plans file: {path}")
    return json.loads(path.read_text())


def load_metric_row(out_dir: Path, episode: str) -> dict[str, Any]:
    path = out_dir / "metric.csv"
    require(path.is_file(), f"missing metric.csv: {path}")
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["episode"] == episode:
                return row
    raise RuntimeError(f"episode {episode} missing from {path}")


def main() -> None:
    protocol_path = Path(os.environ["MDTEC_CEC_COMPOSITION_PROTOCOL"]).resolve()
    manifest_path = Path(os.environ["MDTEC_CEC_COMPOSITION_MANIFEST"]).resolve()
    scene_index = int(os.environ["MDTEC_CEC_COMPOSITION_SCENE_INDEX"])
    scene_root = Path(os.environ["MDTEC_CEC_COMPOSITION_SCENE_ROOT"]).resolve()
    hab_python = os.environ["MDTEC_CEC_COMPOSITION_HAB_PY"]
    eval_2leg_path = os.environ["MDTEC_CEC_COMPOSITION_EVAL_2LEG_PY"]
    host = os.environ["MDTEC_CEC_COMPOSITION_HOST"]
    memnav_port = os.environ["MDTEC_CEC_COMPOSITION_MEMNAV_PORT"]
    navdp_port = os.environ["MDTEC_CEC_COMPOSITION_NAVDP_PORT"]
    smoke = os.environ.get("MDTEC_CEC_COMPOSITION_SMOKE", "0") == "1"
    max_steps = os.environ.get("MDTEC_CEC_COMPOSITION_MAX_STEPS", "500")

    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    require(protocol.get("protocol_version") == 1,
            "unsupported CEC composition protocol version")
    require(tuple(protocol["arms"]) == ARMS, "CEC composition arm set changed")

    selected_scenes = manifest["scenes"]
    require(0 <= scene_index < len(selected_scenes), "scene index out of range")
    scene = selected_scenes[scene_index]
    episodes_per_scene = int(protocol["episodes_per_scene"])
    require(episodes_per_scene == 2, "population selection rule changed")
    all_episode_rows = manifest["episodes"][scene]
    # frozen selection rule: first N by upstream manifest order, never
    # filtered by constructibility/support/results (protocol section 2)
    episode_names = [row["episode"] for row in all_episode_rows[:episodes_per_scene]]
    require(len(episode_names) == episodes_per_scene, "not enough episodes for scene")

    scene_root.mkdir(parents=True, exist_ok=True)
    base_args = [
        hab_python, eval_2leg_path,
        "--episode_root", str(Path(manifest["paths"]["episode_root"]) / scene),
        "--scene", str(Path(manifest["paths"]["asset_root"]) / scene / f"{scene}.glb"),
        "--host", host, "--port", memnav_port, "--novel_port", navdp_port,
        "--server_backend", "hybrid_pose",
        "--success_dist", "1.0", "--max_steps", str(max_steps),
        "--exec_horizon", "8", "--trajectory_selector", "server",
        "--seed", "20260803",
        "--terminal_uturn", "off", "--terminal_visual_refine", "off",
        "--deterministic_plan_seeds",
        "--navdp_depth_source", "monocular_sidecar",
    ]

    records: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episode_names):
        episode_seed = 20260803 + episode_index
        arm_order = rotated_arm_order(scene_index, episode_index)
        trace_dir = scene_root / f"{episode}_leg_a_trace"

        run_eval_2leg(
            base_args, {"episode_ids": episode, "seed": str(episode_seed)},
            ["--leg1_mode", "policy", "--write_leg1_trace",
             "--stop_after_leg1", "--hybrid_route", "phase"],
            trace_dir,
        )
        leg_a_payload = load_episode_plans(trace_dir, episode)
        leg_a_outcome = {
            "plans": leg_a_payload["legA"],
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed_any": any(
                plan.get("metric_depth_sensor_consumed") for plan in
                leg_a_payload["legA"]),
        }
        audit_shared_leg_a(leg_a_outcome)
        leg_a_row = load_metric_row(trace_dir, episode)
        require(int(float(leg_a_row["reached_A"])) == 1
                or int(float(leg_a_row["reached_A"])) == 0,
                "shared Goal-A row malformed")

        arm_outcomes: dict[str, dict[str, Any]] = {}
        arm_rows: dict[str, dict[str, Any]] = {}
        for arm in arm_order:
            arm_dir = scene_root / f"{episode}_{arm}"
            arm_flags = ["--leg1_mode", "shared_trace",
                         "--shared_leg1_trace_root", str(trace_dir),
                         "--hybrid_route", HYBRID_ROUTE[arm]]
            if HYBRID_ROUTE[arm] == "certified_relocalization":
                arm_flags += ["--revisit_adapter", "verified_bearing_v1"]
            run_eval_2leg(
                base_args, {"episode_ids": episode, "seed": str(episode_seed)},
                arm_flags,
                arm_dir,
            )
            payload = load_episode_plans(arm_dir, episode)
            outcome = {
                "plans": payload["legB"],
                "navdp_depth_source": "monocular_sidecar",
                "metric_depth_sensor_consumed_any": any(
                    plan.get("metric_depth_sensor_consumed") for plan in
                    payload["legB"]),
            }
            arm_outcomes[arm] = outcome
            arm_rows[arm] = load_metric_row(arm_dir, episode)

        for arm in arm_order:
            native_outcome = (arm_outcomes.get("raw_native")
                              if arm == "raw_cec" else None)
            contract = audit_arm_leg_b(
                arm, arm_outcomes[arm], native_outcome=native_outcome)
            row = arm_rows[arm]
            plans_name = f"{episode}_{arm}_legB_plans.json"
            (scene_root / plans_name).write_text(json.dumps({
                "protocol_version": protocol["protocol_version"],
                "formal": not smoke,
                "scene_index": scene_index,
                "scene": scene,
                "episode": episode,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "arm": arm,
                "arm_order": list(arm_order),
                "plans": arm_outcomes[arm]["plans"],
            }, indent=2, sort_keys=True, allow_nan=False) + "\n")
            record = {
                "formal": not smoke,
                "scene_index": scene_index,
                "scene": scene,
                "episode": episode,
                "episode_index": episode_index,
                "seed": episode_seed,
                "arm": arm,
                "arm_order": json.dumps(list(arm_order)),
                "reached_A": int(float(leg_a_row["reached_A"])),
                "spl_A": float(leg_a_row["spl_A"]),
                "final_dist_A": float(leg_a_row["final_dist_A"]),
                "reached_B": int(float(row["reached_B"])),
                "spl_B": float(row["spl_B"]),
                "final_dist_B": float(row["final_dist_B"]),
                "reached": int(float(row["reached_B"])),  # B, conditional on A
                "spl": float(row["spl_B"]),
                "metric_depth_sensor_consumed_any": bool(
                    outcome["metric_depth_sensor_consumed_any"]),
                "certified_request_count": contract["certified_request_count"],
                "certified_accept_count": contract["certified_accept_count"],
                "certified_runtime_failure_count":
                    contract["certified_runtime_failure_count"],
                "protocol_sha256": file_sha256(protocol_path),
                "manifest_sha256": file_sha256(manifest_path),
                "plans_file": plans_name,
            }
            records.append(record)
            print(
                f"[mdtec-cec-composition] {scene}/{episode} {arm} "
                f"reached_A={record['reached_A']} reached_B={record['reached_B']} "
                f"spl_B={record['spl_B']:.3f}",
                flush=True,
            )

    csv_path = scene_root / "depth_arms.csv"
    fields = sorted({key for row in records for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    meta = {
        "status": "complete",
        "formal": not smoke,
        "scene_index": scene_index,
        "scene": scene,
        "episodes": episode_names,
        "arms": list(ARMS),
        "records": len(records),
        "protocol_sha256": file_sha256(protocol_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (scene_root / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
