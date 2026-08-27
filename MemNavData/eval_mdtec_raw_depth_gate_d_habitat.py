#!/usr/bin/env python3
"""Prospective same-process Gate-D evaluator for the MDTEC depth readout.

The evaluator reuses the audited Habitat motion and NavDP HTTP contracts from
``eval_2leg_habitat``.  It changes exactly one variable across paired Novel-A
rollouts: the depth source consumed by the frozen NavDP server.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import eval_2leg_habitat as base
from deterministic_eval_protocol import bytes_sha256
from mdtec_raw_depth_gate_d import (
    ARMS,
    DEPTH_SOURCE,
    audit_arm_contract,
    require,
    rotated_arm_order,
)


args = base.args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(protocol: dict[str, Any], manifest_path: Path,
                    overlay_path: Path, formal: bool) -> None:
    manifest_sha = file_sha256(manifest_path)
    overlay_sha = file_sha256(overlay_path)
    require(protocol.get("protocol_version") == 1,
            "unsupported Gate-D protocol version")
    require(protocol["manifest"]["sha256"] == manifest_sha,
            "Gate-D manifest SHA mismatch")
    require(protocol["goal_input_overlay"]["sha256"] == overlay_sha,
            "Gate-D Goal-A overlay SHA mismatch")
    require(tuple(protocol["arms"]) == ARMS, "Gate-D arm set changed")
    evaluation = protocol["evaluation"]
    if formal:
        require(int(evaluation["scenes"]) == 20, "formal scene count changed")
        require(int(evaluation["episodes_per_scene"]) == 2,
                "formal per-scene population changed")
        require(int(args.max_steps) == int(evaluation["max_steps"]),
                "formal max_steps changed")
        require(int(args.exec_horizon) == int(evaluation["execution_horizon"]),
                "formal execution horizon changed")
        require(abs(float(args.success_dist)
                    - float(evaluation["success_distance_m"])) < 1e-12,
                "formal success radius changed")
        require(args.deterministic_plan_seeds is True,
                "formal Gate D requires deterministic plan seeds")


def main() -> None:
    protocol_path = Path(os.environ["MDTEC_GATE_D_PROTOCOL"]).resolve()
    manifest_path = Path(os.environ["MDTEC_GATE_D_MANIFEST"]).resolve()
    overlay_path = Path(os.environ["MDTEC_GATE_D_INPUTS"]).resolve()
    scene_index = int(os.environ["MDTEC_GATE_D_SCENE_INDEX"])
    formal = os.environ.get("MDTEC_GATE_D_SMOKE", "0") != "1"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    overlay = json.loads(overlay_path.read_text())
    validate_inputs(protocol, manifest_path, overlay_path, formal)
    require(overlay["parent_manifest_sha256"] ==
            protocol["manifest"]["sha256"],
            "Goal-A overlay belongs to another manifest")

    selected_scenes = manifest["selection"]["selected_scenes"]
    require(0 <= scene_index < len(selected_scenes), "scene index out of range")
    scene = selected_scenes[scene_index]
    require(Path(args.scene).stem == scene, "scene asset/manifest mismatch")
    episode_dirs = sorted(
        Path(path) for path in glob.glob(
            os.path.join(args.episode_root, "episode_*"))
        if Path(path, "meta", "gen_meta.json").is_file())
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",")
                  if item.strip()}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    expected = [row["episode"] for row in manifest["episodes"][scene]]
    if formal:
        require([path.name for path in episode_dirs] == expected,
                "formal episode population/order changed")
    require(bool(episode_dirs), "no Gate-D episodes selected")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    records: list[dict[str, Any]] = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            meta = json.loads((episode_dir / "meta/gen_meta.json").read_text())
            require(int(meta.get("n_legs", 2)) == 2,
                    "Gate D requires a two-leg carrier")
            rows = pd.read_parquet(
                episode_dir / "data/chunk-000/episode_000000.parquet")
            require(len(rows) == int(meta["n_frames"]),
                    "episode frame count mismatch")
            intrinsic_raw = rows.iloc[0]["observation.camera_intrinsic"]
            camera_intrinsic = np.stack([
                np.asarray(row, dtype=np.float64) for row in intrinsic_raw])
            switch = int(meta["switch_idx"])
            rgb_root = episode_dir / "videos/chunk-000/observation.images.rgb"
            goal_jpg = (rgb_root / f"{switch - 1}.jpg").read_bytes()
            goal_record = overlay["goal_a_images"][scene][episode_dir.name]
            require(int(goal_record["frame_index"]) == switch - 1,
                    "Goal-A frame index mismatch")
            require(int(goal_record["bytes"]) == len(goal_jpg)
                    and goal_record["sha256"] == bytes_sha256(goal_jpg),
                    "Goal-A bytes differ from frozen overlay")
            start_position, start_yaw = base.parquet_pose_hab(
                rows.iloc[0]["action"])
            goal_hab = base.data_to_hab(meta["A"])
            goal_xz = np.asarray(goal_hab[[0, 2]], dtype=float)
            ok, geo_a, _ = base.geodesic(pathfinder, start_position, goal_hab)
            require(ok and np.isfinite(geo_a), "invalid Goal-A geodesic")
            episode_seed = int(args.seed + episode_index)
            arm_order = rotated_arm_order(scene_index, episode_index)

            for arm_position, arm in enumerate(arm_order):
                base.args.navdp_depth_source = DEPTH_SOURCE[arm]
                base.srv_reset(
                    camera_height=float(meta.get("camera_height_m", base.CAM_H)),
                    seed=episode_seed,
                    episode_len=int(meta["n_frames"]),
                    camera_intrinsic=camera_intrinsic,
                )
                outcome = base.run_policy_leg(
                    sim, pathfinder, start_position.copy(), float(start_yaw),
                    goal_jpg, goal_xz, float(geo_a),
                    terminal_mode="off", policy_backend="navdp",
                    success_dist=float(args.success_dist),
                    episode_seed=episode_seed, leg_index=0,
                )
                contract = audit_arm_contract(arm, outcome)
                plans = outcome.pop("plans")
                plans_name = f"{episode_dir.name}_{arm}_plans.json"
                (out / plans_name).write_text(json.dumps({
                    "protocol_version": protocol["protocol_version"],
                    "formal": formal,
                    "scene_index": scene_index,
                    "scene": scene,
                    "episode": episode_dir.name,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "arm": arm,
                    "arm_position": arm_position,
                    "arm_order": list(arm_order),
                    "plans": plans,
                }, indent=2, sort_keys=True, allow_nan=False) + "\n")
                path_len = float(outcome["path_len"])
                reached = bool(outcome["reached"])
                spl = (float(geo_a) / max(float(geo_a), path_len)
                       if reached else 0.0)
                record = {
                    "formal": formal,
                    "scene_index": scene_index,
                    "scene": scene,
                    "episode": episode_dir.name,
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "arm": arm,
                    "depth_source": DEPTH_SOURCE[arm],
                    "arm_position": arm_position,
                    "arm_order": json.dumps(list(arm_order)),
                    "reached": int(reached),
                    "spl": spl,
                    "geo_A": float(geo_a),
                    "path_len_m": path_len,
                    "steps": int(outcome["steps"]),
                    "diagnostic_steps": int(outcome["diagnostic_steps"]),
                    "termination_reason": outcome.get("termination_reason"),
                    "final_dist_m": float(np.linalg.norm(
                        np.asarray(outcome["end_pos"])[[0, 2]] - goal_xz)),
                    "metric_depth_sensor_consumed_any": bool(
                        outcome["metric_depth_sensor_consumed_any"]),
                    "monocular_frame40_survived": bool(
                        outcome["monocular_frame40_survived"]),
                    "monocular_first40_scale_valid": outcome.get(
                        "monocular_first40_scale_valid"),
                    "monocular_first40_scale_clamped": outcome.get(
                        "monocular_first40_scale_clamped"),
                    "monocular_first40_scale_hat": outcome.get(
                        "monocular_first40_scale_hat"),
                    "monocular_bootstrap_path_m": outcome.get(
                        "monocular_bootstrap_path_m"),
                    "goal_jpg_sha256": bytes_sha256(goal_jpg),
                    "protocol_sha256": file_sha256(protocol_path),
                    "manifest_sha256": file_sha256(manifest_path),
                    "input_overlay_sha256": file_sha256(overlay_path),
                    "plans_file": plans_name,
                    **contract,
                }
                records.append(record)
                print(
                    f"[mdtec-gate-d] {scene}/{episode_dir.name} {arm} "
                    f"reached={int(reached)} final={record['final_dist_m']:.3f} "
                    f"steps={record['steps']} plans={len(plans)}",
                    flush=True,
                )
    finally:
        sim.close()

    csv_path = out / "depth_arms.csv"
    fields = sorted({key for row in records for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    meta = {
        "status": "complete",
        "formal": formal,
        "scene_index": scene_index,
        "scene": scene,
        "episodes": [path.name for path in episode_dirs],
        "arms": list(ARMS),
        "records": len(records),
        "server_process_contract": "one_persistent_memnav_plus_navdp_pair",
        "protocol_sha256": file_sha256(protocol_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_overlay_sha256": file_sha256(overlay_path),
    }
    (out / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
