#!/usr/bin/env python3
"""Same-process three-arm Novel-A oracle-bearing mechanism gate.

This evaluator intentionally imports the audited CLI/controller primitives from
``eval_2leg_habitat``. Additional frozen settings are supplied through:

  NOVEL_A_BEARING_PROTOCOL      protocol JSON (required)
  NOVEL_A_BEARING_MANIFEST      frozen 20-scene manifest (required)
  NOVEL_A_BEARING_INPUTS        frozen Goal-A image overlay (required)
  NOVEL_A_BEARING_SCENE_INDEX   index in manifest selected_scenes (required)
  NOVEL_A_BEARING_SMOKE         1 permits a shorter non-formal transport smoke

Formal semantics are frozen in NOVEL_A_BEARING_GATE_PROTOCOL_20260808.md.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import eval_2leg_habitat as base
from deterministic_eval_protocol import bytes_sha256, diffusion_plan_seed
from novel_a_bearing_gate import (
    ARMS,
    critic_shadow_diagnostics,
    normalize_selected_trajectory,
    require,
    rotated_arm_order,
    token_request_deg,
    wrap_deg,
)


args = base.args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jpg(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def geodesic_bearing(pathfinder, position: np.ndarray,
                     target_floor: np.ndarray,
                     minimum_waypoint_m: float) -> float | None:
    ok, _distance, path = base.geodesic(
        pathfinder, np.asarray(position, dtype=float),
        np.asarray(target_floor, dtype=float))
    if not ok or len(path) < 2:
        return None
    origin = np.asarray(position, dtype=float)[[0, 2]]
    for waypoint in path[1:]:
        delta = np.asarray(waypoint, dtype=float)[[0, 2]] - origin
        if float(np.linalg.norm(delta)) >= minimum_waypoint_m:
            return float(base.yaw_facing(delta))
    delta = np.asarray(path[-1], dtype=float)[[0, 2]] - origin
    if float(np.linalg.norm(delta)) < 1e-9:
        return None
    return float(base.yaw_facing(delta))


def request_mixgoal_read_only(
    *,
    image_jpg: bytes,
    goal_jpg: bytes,
    depth: np.ndarray,
    request_deg: float,
    radius_m: float,
    diffusion_seed: int,
) -> dict:
    theta = math.radians(float(request_deg))
    response = requests.post(
        f"http://{args.host}:{args.port}/mixgoal_resample",
        files={
            "image": ("image.jpg", image_jpg),
            "image_goal": ("goal.jpg", goal_jpg),
            "depth": ("depth.png", base.depth_png_bytes(depth)),
        },
        data={
            "goal_data": json.dumps({
                "goal_x": [radius_m * math.cos(theta)],
                "goal_y": [radius_m * math.sin(theta)],
            }),
            "diffusion_seed": str(int(diffusion_seed)),
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    require(payload.get("memory_mutated") is False,
            "mixgoal_resample did not assert read-only FIFO semantics")
    require(int(payload.get("diffusion_seed")) == int(diffusion_seed),
            "mixgoal_resample seed echo mismatch")
    require(isinstance(payload.get("queue_lengths"), list),
            "mixgoal_resample omitted queue lengths")
    hashes_before = payload.get("queue_hashes_before")
    hashes_after = payload.get("queue_hashes_after")
    require(isinstance(hashes_before, list) and hashes_before,
            "mixgoal_resample omitted FIFO content fingerprints")
    require(hashes_before == hashes_after,
            "mixgoal_resample changed FIFO content")
    return base.normalize_navdp_response(payload)


def run_arm(
    *,
    arm: str,
    sim,
    pathfinder,
    start_position: np.ndarray,
    start_yaw: float,
    goal_jpg: bytes,
    goal_xz: np.ndarray,
    episode_seed: int,
    protocol: dict,
) -> dict[str, Any]:
    require(arm in ARMS, f"unknown bearing arm {arm!r}")
    position = np.asarray(start_position, dtype=float).copy()
    yaw = float(start_yaw)
    path_len = 0.0
    path_len_at_reach = None
    step_at_reach = None
    history: list[np.ndarray] = []
    plans: list[dict[str, Any]] = []
    reached = False
    token_burst = 0
    token_disabled_reason = None
    token_path_m = 0.0
    ideal_turn_abs_deg = 0.0
    camera_height = float(protocol["camera_height_m"])
    token_config = protocol["token"]
    ideal_config = protocol["ideal_periodic_yaw"]

    for step in range(args.max_steps):
        is_plan_step = step % args.exec_horizon == 0
        desired_yaw = None
        residual_before = None
        ideal_turn_deg = 0.0
        if is_plan_step and arm == "ideal_periodic_yaw":
            target_floor = np.asarray(
                [goal_xz[0], position[1], goal_xz[1]], dtype=float)
            desired_yaw = geodesic_bearing(
                pathfinder, position, target_floor,
                float(ideal_config["bearing_waypoint_min_distance_m"]))
            require(desired_yaw is not None,
                    "ideal arm could not compute a valid geodesic bearing")
            residual_before = wrap_deg(math.degrees(desired_yaw - yaw))
            if abs(residual_before) > float(ideal_config["trigger_deg"]):
                ideal_turn_deg = float(residual_before)
                ideal_turn_abs_deg += abs(ideal_turn_deg)
                yaw = float(desired_yaw)

        rgb, depth = base.render(
            sim, position + np.asarray([0.0, camera_height, 0.0]), yaw)
        image_jpg = base.jpg_bytes(rgb)

        if is_plan_step:
            plan_index = len(plans)
            request_seed = diffusion_plan_seed(
                int(episode_seed), 0, int(plan_index))
            native = base.srv_plan(
                image_jpg, goal_jpg, depth=depth,
                diffusion_seed=request_seed)
            require(int(native.get("diffusion_seed")) == request_seed,
                    "native ImageGoal seed echo mismatch")
            native_way = normalize_selected_trajectory(native["trajectory"])
            selected_way = native_way
            trajectory_source = "native"
            token_request = None
            token_shadow = None
            token_diffusion_seed = None
            token_queue_hashes_before = None
            token_queue_hashes_after = None
            if arm == "oracle_token_periodic":
                target_floor = np.asarray(
                    [goal_xz[0], position[1], goal_xz[1]], dtype=float)
                desired_yaw = geodesic_bearing(
                    pathfinder, position, target_floor,
                    float(ideal_config["bearing_waypoint_min_distance_m"]))
                require(desired_yaw is not None,
                        "token arm could not compute a valid geodesic bearing")
                residual_before = wrap_deg(math.degrees(desired_yaw - yaw))
                if abs(residual_before) <= float(token_config["trigger_deg"]):
                    token_burst = 0
                elif token_disabled_reason is None:
                    if token_burst >= int(token_config["max_consecutive_plans"]):
                        token_disabled_reason = "max_burst_exhausted"
                    else:
                        token_request = token_request_deg(
                            residual_before,
                            float(token_config["request_clip_deg"]))
                        mixed = request_mixgoal_read_only(
                            image_jpg=image_jpg,
                            goal_jpg=goal_jpg,
                            depth=depth,
                            request_deg=token_request,
                            radius_m=float(token_config["request_radius_m"]),
                            diffusion_seed=request_seed,
                        )
                        selected_way = normalize_selected_trajectory(
                            mixed["trajectory"])
                        trajectory_source = "oracle_token"
                        token_burst += 1
                        token_diffusion_seed = int(mixed["diffusion_seed"])
                        token_queue_hashes_before = mixed[
                            "queue_hashes_before"]
                        token_queue_hashes_after = mixed[
                            "queue_hashes_after"]
                        token_shadow = critic_shadow_diagnostics(
                            mixed, requested_heading_deg=token_request)

            way_world = base.waypoints_to_world(
                selected_way, [position[0], position[2]], yaw)
            plan = {
                "plan_index": plan_index,
                "step": int(step),
                "arm": arm,
                "observation_sha256": bytes_sha256(image_jpg),
                "requested_diffusion_seed": request_seed,
                "native_diffusion_seed": int(native["diffusion_seed"]),
                "trajectory_source": trajectory_source,
                "bearing_residual_before_deg": residual_before,
                "ideal_turn_deg": ideal_turn_deg,
                "token_request_deg": token_request,
                "token_diffusion_seed": token_diffusion_seed,
                "token_queue_hashes_before": token_queue_hashes_before,
                "token_queue_hashes_after": token_queue_hashes_after,
                "token_burst_count": int(token_burst),
                "token_disabled_reason": token_disabled_reason,
                "native_shadow": critic_shadow_diagnostics(native),
                "token_shadow": token_shadow,
                "path_m": 0.0,
                "executed_steps": 0,
                "bearing_residual_after_deg": None,
            }
            plans.append(plan)

        position, yaw, distance = base.pursuit_step(
            position, yaw, way_world, pathfinder)
        path_len += float(distance)
        plans[-1]["path_m"] += float(distance)
        plans[-1]["executed_steps"] += 1
        if plans[-1]["trajectory_source"] == "oracle_token":
            token_path_m += float(distance)
        history.append(np.asarray([position[0], position[2]], dtype=float))

        target_floor = np.asarray(
            [goal_xz[0], position[1], goal_xz[1]], dtype=float)
        desired_after = geodesic_bearing(
            pathfinder, position, target_floor,
            float(ideal_config["bearing_waypoint_min_distance_m"]))
        if desired_after is not None:
            plans[-1]["bearing_residual_after_deg"] = wrap_deg(
                math.degrees(desired_after - yaw))

        final_dist = float(np.linalg.norm(
            np.asarray([position[0], position[2]]) - goal_xz))
        if final_dist < args.success_dist:
            reached = True
            path_len_at_reach = float(path_len)
            step_at_reach = int(step + 1)
            break
        if (len(history) > args.stuck_window
                and np.linalg.norm(
                    history[-1] - history[-args.stuck_window])
                < args.stuck_dist):
            break

    steps = int(step_at_reach if reached else len(history))
    final_dist = float(np.linalg.norm(
        np.asarray([position[0], position[2]]) - goal_xz))
    return {
        "reached": bool(reached),
        "steps": steps,
        "path_len_m": float(path_len),
        "path_len_at_reach_m": path_len_at_reach,
        "final_dist_m": final_dist,
        "end_position": np.asarray(position, dtype=float).tolist(),
        "end_yaw": float(yaw),
        "plan_count": len(plans),
        "ideal_turn_count": sum(
            abs(float(plan["ideal_turn_deg"])) > 0.0 for plan in plans),
        "ideal_turn_abs_deg": float(ideal_turn_abs_deg),
        "token_plan_count": sum(
            plan["trajectory_source"] == "oracle_token" for plan in plans),
        "token_path_m": float(token_path_m),
        "token_disabled_reason": token_disabled_reason,
        "plans": plans,
    }


def validate_protocol(protocol: dict, manifest_path: Path,
                      inputs_path: Path,
                      formal: bool) -> None:
    require(protocol.get("protocol_version") == 1,
            "unsupported bearing-gate protocol")
    require(tuple(protocol.get("arms", [])) == ARMS,
            "protocol arm set/order changed")
    require(file_sha256(manifest_path) == protocol["manifest"]["sha256"],
            "bearing-gate manifest SHA256 mismatch")
    require(file_sha256(inputs_path) == protocol["input_overlay"]["sha256"],
            "bearing-gate Goal-A input overlay SHA256 mismatch")
    evaluation = protocol["evaluation"]
    require(args.server_backend == "navdp", "bearing gate requires native NavDP")
    require(args.deterministic_plan_seeds,
            "bearing gate requires deterministic plan seeds")
    require(args.trajectory_selector == "server",
            "bearing gate forbids candidate-oracle trajectory selection")
    require(args.terminal_uturn == "off"
            and args.terminal_visual_refine == "off",
            "bearing gate disables terminal interventions")
    require(not args.reset_memory, "bearing arms manage reset explicitly")
    require(args.seed == int(evaluation["base_seed"]), "base seed changed")
    require(args.exec_horizon == int(evaluation["execution_horizon"]),
            "execution horizon changed")
    require(math.isclose(args.success_dist,
                         float(evaluation["success_distance_m"])),
            "success threshold changed")
    if formal:
        require(args.max_steps == int(evaluation["max_steps"]),
                "formal physical-step budget changed")
        require(args.episodes == 0, "formal run cannot cap episodes")


def main() -> None:
    protocol_path = Path(os.environ["NOVEL_A_BEARING_PROTOCOL"]).resolve()
    manifest_path = Path(os.environ["NOVEL_A_BEARING_MANIFEST"]).resolve()
    inputs_path = Path(os.environ["NOVEL_A_BEARING_INPUTS"]).resolve()
    scene_index = int(os.environ["NOVEL_A_BEARING_SCENE_INDEX"])
    formal = os.environ.get("NOVEL_A_BEARING_SMOKE", "0") != "1"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    input_overlay = json.loads(inputs_path.read_text())
    validate_protocol(protocol, manifest_path, inputs_path, formal)
    require(input_overlay["parent_manifest_sha256"]
            == protocol["manifest"]["sha256"],
            "Goal-A overlay belongs to another parent manifest")
    selected_scenes = manifest["selection"]["selected_scenes"]
    require(0 <= scene_index < len(selected_scenes), "scene index out of range")
    scene = selected_scenes[scene_index]
    require(Path(args.scene).stem == scene, "scene asset and manifest disagree")

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
    expected_episodes = [
        record["episode"] for record in manifest["episodes"][scene]]
    if formal:
        require([path.name for path in episode_dirs] == expected_episodes,
                "formal scene episode coverage/order changed")
    require(bool(episode_dirs), "no Novel-A episodes selected")

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    records: list[dict[str, Any]] = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            metadata = json.loads(
                (episode_dir / "meta" / "gen_meta.json").read_text())
            require(int(metadata.get("n_legs", 2)) == 2,
                    "bearing gate requires a 2-leg episode")
            rows = pd.read_parquet(
                episode_dir / "data/chunk-000/episode_000000.parquet")
            require(len(rows) == int(metadata["n_frames"]),
                    "episode frame count mismatch")
            camera_intrinsic = np.stack([
                np.asarray(row, dtype=np.float64)
                for row in rows.iloc[0]["observation.camera_intrinsic"]])
            switch = int(metadata["switch_idx"])
            rgb_root = (episode_dir /
                        "videos/chunk-000/observation.images.rgb")
            goal_jpg = read_jpg(rgb_root / f"{switch - 1}.jpg")
            goal_record = input_overlay["goal_a_images"][scene][episode_dir.name]
            require(int(goal_record["frame_index"]) == switch - 1,
                    "Goal-A overlay frame index mismatch")
            require(int(goal_record["bytes"]) == len(goal_jpg)
                    and goal_record["sha256"] == bytes_sha256(goal_jpg),
                    "Goal-A image differs from frozen input overlay")
            start_position, start_yaw = base.parquet_pose_hab(
                rows.iloc[0]["action"])
            goal_hab = base.data_to_hab(metadata["A"])
            goal_xz = np.asarray(goal_hab[[0, 2]], dtype=float)
            target_floor = np.asarray(
                [goal_xz[0], start_position[1], goal_xz[1]], dtype=float)
            ok, geo_a, _ = base.geodesic(
                pathfinder, start_position, target_floor)
            require(ok and np.isfinite(geo_a), "invalid Goal-A geodesic")
            episode_seed = int(args.seed + episode_index)
            camera_height = float(
                metadata.get("camera_height_m", base.CAM_H))
            episode_protocol = dict(protocol)
            episode_protocol["camera_height_m"] = camera_height
            arm_order = rotated_arm_order(scene_index, episode_index)

            for arm_position, arm in enumerate(arm_order):
                base.srv_reset(
                    camera_height=camera_height,
                    seed=episode_seed,
                    episode_len=int(metadata["n_frames"]),
                    camera_intrinsic=camera_intrinsic)
                outcome = run_arm(
                    arm=arm,
                    sim=sim,
                    pathfinder=pathfinder,
                    start_position=start_position,
                    start_yaw=start_yaw,
                    goal_jpg=goal_jpg,
                    goal_xz=goal_xz,
                    episode_seed=episode_seed,
                    protocol=episode_protocol,
                )
                plans_path = output_root / f"{episode_dir.name}_{arm}_plans.json"
                plans_payload = {
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
                    "plans": outcome.pop("plans"),
                }
                plans_path.write_text(json.dumps(
                    plans_payload, indent=2, sort_keys=True,
                    allow_nan=False) + "\n")
                record = {
                    "formal": formal,
                    "scene_index": scene_index,
                    "scene": scene,
                    "episode": episode_dir.name,
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "arm": arm,
                    "arm_position": arm_position,
                    "arm_order": json.dumps(list(arm_order)),
                    "geo_A": float(geo_a),
                    "goal_jpg_sha256": bytes_sha256(goal_jpg),
                    "protocol_sha256": file_sha256(protocol_path),
                    "manifest_sha256": file_sha256(manifest_path),
                    "input_overlay_sha256": file_sha256(inputs_path),
                    "plans_file": plans_path.name,
                    **outcome,
                }
                # Keep CSV scalar and portable; full end pose lives in plans.
                record["end_position"] = json.dumps(record["end_position"])
                records.append(record)
                print(
                    f"[bearing-gate] {scene}/{episode_dir.name} {arm} "
                    f"reached={int(record['reached'])} "
                    f"final={record['final_dist_m']:.3f} "
                    f"steps={record['steps']} plans={record['plan_count']}",
                    flush=True)
    finally:
        sim.close()

    csv_path = output_root / "bearing_arms.csv"
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    run_meta = {
        "status": "complete",
        "formal": formal,
        "scene_index": scene_index,
        "scene": scene,
        "episodes": [path.name for path in episode_dirs],
        "arms": list(ARMS),
        "records": len(records),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_overlay_path": str(inputs_path),
        "input_overlay_sha256": file_sha256(inputs_path),
    }
    (output_root / "run_meta.json").write_text(json.dumps(
        run_meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_meta, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
