#!/usr/bin/env python3
"""Conditional-B decomposition arms for the 3-leg A/B asymmetry audit.

LATEST_TRAINING_RESULTS_20260807.md §20.6 records four benchmark asymmetries
between Novel A and Novel B.  This evaluator isolates them with paired arms
that all reuse the audited controller in eval_2leg_habitat and the EXISTING
generated episodes (no regeneration):

  b2_rendered   expert-A endpoint start, expert-warmed FIFO, goal_1.jpg
                (the current confounded goal image)          [reference]
  b2_arrival    identical, but the goal image is the expert arrival frame
                {switch_b-1}.jpg                             [isolates goal-image asymmetry]
  b1_matched    B evaluated as a FIRST goal: start on the expert leg-2 frame
                whose remaining geodesic to B best matches geo(start->A)
                within the generator's A band, fresh FIFO, arrival goal image
                                                             [isolates order/history + distance]
  b1_role_matched
                same first-goal counterfactual, but also copies A's relative
                initial heading offset onto B's first geodesic segment.  On v4
                data this matches distance band, goal-view construction, FIFO
                state and initial-bearing difficulty without changing NavDP.
  b2_executed   optional: policy executes leg A first, then B with the
                arrival goal image                           [isolates on-policy start error]

All arms share the same episode seed and leg_index so deterministic
per-request diffusion seeds pair exactly.  Run one arm set per live server
process; cross-process pairing is not guaranteed by the CUDA stack.

Environment knobs (base argparse cannot take new flags):
  EVAL3SYM_ARMS   comma list, default "b2_rendered,b2_arrival,b1_matched"
  EVAL3SYM_BAND   "lo,hi" geodesic band for the b1 start, default "3.0,9.0"
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import requests

import eval_2leg_habitat as base
from multigoal_benchmark_contract import (
    ROLE_SYMMETRIC_PROTOCOL,
    RoleSymmetryObservation,
    validate_role_symmetric_contract,
)

args = base.args

DEFAULT_ARMS = "b2_rendered,b2_arrival,b1_matched"
ARM_CHOICES = ("b2_rendered", "b2_arrival", "b2_turned", "b2_token_steer",
               "b1_matched", "b1_role_matched", "b2_executed")
FIFO_WARM_FRAMES = 8

# Point-token steerability of the frozen policy, measured in
# POINT_TOKEN_STEERABILITY_20260808.md: requests are faithful to ~60 deg,
# peak realized turn at a ~90-105 deg request, and 165-195 deg returns zero
# output.  Requests are therefore clipped into the peak region, which also
# keeps them out of the dead zone.
# Trigger and fidelity are different thresholds: requests up to ~60 deg are
# realized faithfully (fidelity), but the native policy will not take a 50 deg
# correction on its own, so steering must trigger far below that (trigger).
STEER_TRIGGER_DEG = 20.0
STEER_PEAK_REQUEST_DEG = 100.0
STEER_RADIUS_M = 2.0
STEER_MAX_PLANS = int(os.environ.get("EVAL3SYM_STEER_MAX_PLANS", "3"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jpg(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def selected_arms() -> list[str]:
    arms = [item.strip() for item in
            os.environ.get("EVAL3SYM_ARMS", DEFAULT_ARMS).split(",")
            if item.strip()]
    require(bool(arms), "EVAL3SYM_ARMS selected no arms")
    for arm in arms:
        require(arm in ARM_CHOICES, f"unknown arm: {arm}")
    require(len(set(arms)) == len(arms), "duplicate arms selected")
    return arms


def b1_band() -> tuple[float, float]:
    raw = os.environ.get("EVAL3SYM_BAND", "3.0,9.0").split(",")
    require(len(raw) == 2, "EVAL3SYM_BAND must be 'lo,hi'")
    lo, hi = float(raw[0]), float(raw[1])
    require(0.0 < lo < hi, "EVAL3SYM_BAND must satisfy 0 < lo < hi")
    return lo, hi


def warm_fifo_from_expert(rgb_root: str, switch_a: int) -> int:
    """Feed expert leg-A frames into NavDP's FIFO without consuming DDPM
    noise (memory_replay_step asserts no diffusion was sampled).

    EVAL3SYM_WARM_STRIDE spaces the warm frames.  Stride 1 (default) uses the
    last 8 consecutive generator frames; deployment FIFOs instead hold one
    frame per plan (~8 sim frames apart), so stride 8 approximates the
    temporal spacing NavDP actually sees closed-loop."""
    stride = max(1, int(os.environ.get("EVAL3SYM_WARM_STRIDE", "1")))
    indices = [switch_a - 1 - stride * k for k in range(FIFO_WARM_FRAMES)]
    indices = sorted(i for i in indices if i >= 0)
    for index in indices:
        base.srv_navdp_memory_replay(
            read_jpg(os.path.join(rgb_root, f"{index}.jpg")))
    return len(indices)


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def geodesic_bearing(pathfinder, position, target_floor) -> float | None:
    """Heading of the first geodesic segment from `position` toward the goal."""
    ok, _distance, path = base.geodesic(
        pathfinder, np.asarray(position, dtype=float), target_floor)
    if not ok or len(path) < 2:
        return None
    origin = np.asarray(position, dtype=float)[[0, 2]]
    for waypoint in path[1:]:
        delta = np.asarray(waypoint, dtype=float)[[0, 2]] - origin
        if float(np.linalg.norm(delta)) >= 0.3:
            return float(base.yaw_facing(delta))
    return float(base.yaw_facing(
        np.asarray(path[-1], dtype=float)[[0, 2]] - origin))


def token_steering_prefix(sim, pathfinder, position, yaw, goal_jpg,
                          desired_yaw, camera_height, episode_seed,
                          b_floor) -> tuple[np.ndarray, float, list[dict]]:
    """Deployable bearing actuation: iterated mixed image+point steps.

    Each plan is a real decision (the observation is appended to NavDP's FIFO
    exactly once by the stepping endpoint), executed for the same
    ``exec_horizon`` as the rest of the benchmark.  This replaces the
    privileged instantaneous yaw teleport of the ``b2_turned`` arm while
    keeping its (oracle) target direction, so the comparison isolates
    actuation feasibility from direction knowledge."""
    plans: list[dict] = []
    position = np.asarray(position, dtype=float)
    for plan_index in range(STEER_MAX_PLANS):
        residual = wrap_deg(np.degrees(desired_yaw - yaw))
        if abs(residual) <= STEER_TRIGGER_DEG:
            break
        request_deg = float(np.sign(residual) * min(
            abs(residual), STEER_PEAK_REQUEST_DEG))
        theta = np.radians(request_deg)
        camera = position + np.array([0.0, camera_height, 0.0])
        rgb, depth = base.render(sim, camera, yaw)
        response = requests.post(
            f"http://{args.host}:{args.port}/navdp_step_ip_mixgoal",
            files={"image": ("image.jpg", base.jpg_bytes(rgb)),
                   "image_goal": ("goal.jpg", goal_jpg),
                   "depth": ("depth.png", base.depth_png_bytes(depth))},
            data={"goal_data": json.dumps(
                      {"goal_x": [STEER_RADIUS_M * float(np.cos(theta))],
                       "goal_y": [STEER_RADIUS_M * float(np.sin(theta))]}),
                  "diffusion_seed": str(int(episode_seed) * 100 + plan_index)},
            timeout=120)
        response.raise_for_status()
        way = np.asarray(response.json()["trajectory"], dtype=float)
        way = way.reshape(-1, way.shape[-1]) if way.ndim == 2 else way[0]
        way_world = base.waypoints_to_world(way, [position[0], position[2]], yaw)
        travelled = 0.0
        for _ in range(args.exec_horizon):
            position, yaw, step_length = base.pursuit_step(
                position, yaw, way_world, pathfinder)
            travelled += float(step_length)
        ok, remaining, _ = base.geodesic(pathfinder, position, b_floor)
        plans.append({
            "plan": plan_index,
            "residual_before_deg": round(residual, 1),
            "request_deg": round(request_deg, 1),
            "residual_after_deg": round(
                wrap_deg(np.degrees(desired_yaw - yaw)), 1),
            "path_m": round(travelled, 3),
            "geodesic_to_b_m": round(float(remaining), 3) if ok else None,
        })
    return position, yaw, plans


def leg_spl(leg: dict, geodesic_m: float) -> float:
    if not leg["reached"]:
        return 0.0
    path = leg.get("path_len_at_reach")
    path = leg["path_len"] if path is None else path
    return float(geodesic_m / max(geodesic_m, path, 1e-6))


def choose_b1_start(rows: pd.DataFrame, pathfinder, switch_a: int,
                    switch_b: int, b_floor: np.ndarray, geo_a: float,
                    band: tuple[float, float]) -> tuple[int, float] | None:
    """Latest expert leg-2 frame whose remaining geodesic to B lies in the
    generator's A band, preferring the distance closest to this episode's
    geo(start->A); None if no frame qualifies (fail closed, no substitute)."""
    candidates: list[tuple[float, int, float]] = []
    for index in range(switch_a, switch_b):
        position, _yaw = base.parquet_pose_hab(rows.iloc[index]["action"])
        ok, distance, _ = base.geodesic(pathfinder, position, b_floor)
        if ok and np.isfinite(distance) and band[0] <= distance <= band[1]:
            candidates.append((abs(distance - geo_a), -index, distance))
    if not candidates:
        return None
    gap, negative_index, distance = min(candidates)
    del gap
    return -negative_index, float(distance)


def main() -> None:
    require(args.server_backend == "navdp",
            "symmetry arms support native NavDP only")
    require(args.deterministic_plan_seeds,
            "symmetry arms require --deterministic_plan_seeds for pairing")
    require(args.trajectory_selector == "server",
            "symmetry arms forbid trajectory selection overrides")
    require(args.terminal_uturn == "off" and args.terminal_visual_refine == "off",
            "position-SR audit disables terminal refinement")
    require(not args.reset_memory, "symmetry arms manage FIFO state per arm")
    arms = selected_arms()
    band = b1_band()

    os.makedirs(args.out, exist_ok=True)
    episode_dirs = sorted(
        path for path in glob.glob(os.path.join(args.episode_root, "episode_*"))
        if os.path.isfile(os.path.join(path, "meta", "gen_meta.json")))
    if args.episode_ids:
        wanted = {item.strip() for item in args.episode_ids.split(",")
                  if item.strip()}
        episode_dirs = [path for path in episode_dirs
                        if os.path.basename(path) in wanted]
    if args.episodes:
        episode_dirs = episode_dirs[:args.episodes]
    require(bool(episode_dirs), "no 3-leg episodes selected")

    sim = base.make_sim(args.scene, "", agent_radius=args.agent_radius)
    pathfinder = sim.pathfinder
    print(f"[eval3sym] episodes={len(episode_dirs)} arms={arms} band={band}")

    results: list[dict] = []
    try:
        for episode_index, episode_dir in enumerate(episode_dirs):
            episode = os.path.basename(episode_dir)
            with open(os.path.join(episode_dir, "meta", "gen_meta.json")) as handle:
                metadata = json.load(handle)
            require(int(metadata.get("n_legs", -1)) == 3,
                    "selected episode is not 3-leg")
            goal_b = metadata["goals"][0]
            require(goal_b.get("kind") == "novel", "Goal B must be Novel")
            switch_a, switch_b = [int(v) for v in metadata["switches"]]

            rows = pd.read_parquet(os.path.join(
                episode_dir, "data/chunk-000/episode_000000.parquet"))
            require(len(rows) == int(metadata["n_frames"]),
                    "parquet frame count mismatch")
            intrinsic = np.stack([
                np.asarray(row, dtype=np.float64)
                for row in rows.iloc[0]["observation.camera_intrinsic"]])
            rgb_root = os.path.join(
                episode_dir, "videos/chunk-000/observation.images.rgb")

            start_floor, start_yaw = base.parquet_pose_hab(
                rows.iloc[0]["action"])
            a_end_pos, a_end_yaw = base.parquet_pose_hab(
                rows.iloc[switch_a - 1]["action"])
            a_hab = base.data_to_hab(metadata["A"])
            b_hab = base.data_to_hab(goal_b["pos"])
            # Generator poses are exact Habitat floor positions.  Preserve
            # their Y coordinate: replacing it with the episode start height
            # falsely rejects valid episodes on ramps and uneven floors.
            b_floor = b_hab
            a_floor = a_hab
            ok_a, geo_a, _ = base.geodesic(pathfinder, start_floor, a_floor)
            ok_b, geo_b, _ = base.geodesic(pathfinder, a_floor, b_floor)
            require(ok_a and ok_b, "episode geodesics are invalid")

            image_rendered = read_jpg(os.path.join(episode_dir, "goal_1.jpg"))
            image_arrival = read_jpg(
                os.path.join(rgb_root, f"{switch_b - 1}.jpg"))
            image_a = read_jpg(os.path.join(rgb_root, f"{switch_a - 1}.jpg"))

            b_terminal_pos, b_terminal_yaw = base.parquet_pose_hab(
                rows.iloc[switch_b - 1]["action"])
            contract = validate_role_symmetric_contract(
                metadata,
                RoleSymmetryObservation(
                    geo_a_m=float(geo_a),
                    geo_b_m=float(geo_b),
                    initial_pose_error_m=float(np.linalg.norm(
                        start_floor - base.data_to_hab(metadata["start"]))),
                    a_terminal_pose_error_m=float(np.linalg.norm(
                        a_end_pos - a_floor)),
                    b_terminal_pose_error_m=float(np.linalg.norm(
                        b_terminal_pos - b_floor)),
                    b_terminal_yaw_error_deg=abs(float(np.degrees(
                        base.wrap_angle(
                            b_terminal_yaw
                            - float(goal_b["yaw_habitat"]))))),
                    goal_b_matches_terminal_rgb=(
                        image_rendered == image_arrival),
                ),
            )
            require(
                contract["ok"],
                "role-paired data contract failed: "
                + "; ".join(contract["issues"]),
            )

            b1_choice = choose_b1_start(
                rows, pathfinder, switch_a, switch_b, b_floor, geo_a, band)
            episode_seed = args.seed + episode_index

            for arm in arms:
                record = {
                    "episode": episode, "arm": arm, "seed": episode_seed,
                    "geo_start_a": float(geo_a), "geo_a_b": float(geo_b),
                    "b1_start_frame": None, "b1_start_geo": None,
                    "warm_frames": 0, "skip_reason": None,
                }
                if arm in ("b1_matched", "b1_role_matched") and b1_choice is None:
                    record["skip_reason"] = "no_leg2_frame_in_band"
                    results.append(record)
                    continue

                base.srv_reset(
                    camera_height=float(
                        metadata.get("camera_height_m", base.CAM_H)),
                    seed=episode_seed,
                    episode_len=int(metadata["n_frames"]),
                    camera_intrinsic=intrinsic)

                if arm in ("b2_rendered", "b2_arrival", "b2_turned",
                           "b2_token_steer"):
                    record["warm_frames"] = warm_fifo_from_expert(
                        rgb_root, switch_a)
                    start_pos, start_psi = a_end_pos, a_end_yaw
                    goal_jpg = (image_rendered if arm == "b2_rendered"
                                else image_arrival)
                    geo_leg = float(geo_b)
                    if arm == "b2_token_steer":
                        # Deployable counterpart of b2_turned: same oracle
                        # target bearing, but realized by iterated mixed
                        # image+point steps instead of a yaw teleport.
                        desired = geodesic_bearing(
                            pathfinder, a_end_pos, b_floor)
                        require(desired is not None,
                                "token steering needs a valid geodesic bearing")
                        start_pos, start_psi, steer_plans = (
                            token_steering_prefix(
                                sim, pathfinder, a_end_pos, a_end_yaw,
                                image_arrival, desired,
                                float(metadata.get("camera_height_m",
                                                   base.CAM_H)),
                                episode_seed, b_floor))
                        record["steer_plans"] = len(steer_plans)
                        record["steer_path_m"] = round(
                            sum(p["path_m"] for p in steer_plans), 3)
                        record["steer_residual_deg"] = (
                            steer_plans[-1]["residual_after_deg"]
                            if steer_plans else 0.0)
                        record["steer_detail"] = json.dumps(steer_plans)
                        ok_s, geo_leg, _ = base.geodesic(
                            pathfinder, np.asarray(start_pos, dtype=float),
                            b_floor)
                        require(ok_s, "post-steering geodesic invalid")
                        geo_leg = float(geo_leg)
                    if arm == "b2_turned":
                        # identical to b2_arrival except the start yaw faces the
                        # first geodesic segment toward B: isolates the pure
                        # initial-heading effect from visual goal overlap.
                        ok_t, _gd_t, path_t = base.geodesic(
                            pathfinder,
                            np.asarray(a_end_pos, dtype=float), b_floor)
                        require(ok_t and len(path_t) >= 2,
                                "b2_turned geodesic path invalid")
                        origin_xz = np.asarray(a_end_pos, dtype=float)[[0, 2]]
                        target_xz = None
                        for waypoint in path_t[1:]:
                            delta = np.asarray(
                                waypoint, dtype=float)[[0, 2]] - origin_xz
                            if float(np.linalg.norm(delta)) >= 0.3:
                                target_xz = delta
                                break
                        if target_xz is None:
                            target_xz = np.asarray(
                                path_t[-1], dtype=float)[[0, 2]] - origin_xz
                        start_psi = float(base.yaw_facing(target_xz))
                        # Optional bearing degradation: how accurate must a
                        # learned bearing head be?  Quantization rounds to a
                        # compass grid; offset applies a fixed-magnitude bias
                        # whose sign alternates deterministically by episode.
                        quant_deg = float(os.environ.get(
                            "EVAL3SYM_TURN_QUANT_DEG", "0"))
                        offset_deg = float(os.environ.get(
                            "EVAL3SYM_TURN_OFFSET_DEG", "0"))
                        if quant_deg > 0:
                            step_rad = np.deg2rad(quant_deg)
                            start_psi = float(
                                np.round(start_psi / step_rad) * step_rad)
                        if offset_deg != 0.0:
                            sign = 1.0 if episode_index % 2 == 0 else -1.0
                            start_psi = float(
                                start_psi + sign * np.deg2rad(offset_deg))
                            record["turn_offset_applied_deg"] = sign * offset_deg
                        record["turn_quant_deg"] = quant_deg
                elif arm in ("b1_matched", "b1_role_matched"):
                    frame_index, frame_geo = b1_choice
                    record["b1_start_frame"] = int(frame_index)
                    record["b1_start_geo"] = float(frame_geo)
                    start_pos, start_psi = base.parquet_pose_hab(
                        rows.iloc[frame_index]["action"])
                    goal_jpg, geo_leg = image_arrival, float(frame_geo)
                    if arm == "b1_role_matched":
                        require(
                            metadata.get("gen_protocol") ==
                            ROLE_SYMMETRIC_PROTOCOL,
                            "b1_role_matched requires role-paired v4 data")
                        distance_error = abs(float(frame_geo) - float(geo_a))
                        distance_tolerance = float(
                            metadata["role_distance_match_tolerance_m"])
                        require(
                            distance_error <= distance_tolerance + 0.10,
                            "b1_role_matched could not reproduce Goal A's "
                            "geodesic distance from stored leg-B frames",
                        )
                        record["matched_geodesic_error_m"] = distance_error
                        relative_offset_deg = metadata.get(
                            "start_heading_offset_deg")
                        require(relative_offset_deg is not None,
                                "v4 data omitted start heading offset")
                        path_yaw = geodesic_bearing(
                            pathfinder, start_pos, b_floor)
                        require(path_yaw is not None,
                                "b1_role_matched geodesic bearing invalid")
                        start_psi = float(
                            path_yaw + np.radians(float(relative_offset_deg)))
                        record["matched_relative_heading_offset_deg"] = float(
                            relative_offset_deg)
                else:  # b2_executed
                    leg_a = base.run_policy_leg(
                        sim, pathfinder, start_floor, start_yaw, image_a,
                        a_hab[[0, 2]], float(geo_a), None,
                        terminal_mode="off", policy_backend=None,
                        success_dist=args.success_dist,
                        episode_seed=episode_seed, leg_index=0)
                    record["a_reached"] = bool(leg_a["reached"])
                    if not leg_a["reached"]:
                        record["skip_reason"] = "executed_a_failed"
                        results.append(record)
                        continue
                    start_pos, start_psi = leg_a["end_pos"], leg_a["end_psi"]
                    ok_e, geo_leg, path_e = base.geodesic(
                        pathfinder,
                        np.asarray(start_pos, dtype=float), b_floor)
                    require(ok_e, "executed-A endpoint geodesic invalid")
                    goal_jpg = image_arrival
                    if os.environ.get("EVAL3SYM_EXECUTED_BEARING", "0") == "1":
                        # deployment-shaped bearing test: rotate the policy's
                        # own A-endpoint heading toward the first geodesic
                        # segment to B (oracle bearing; upper bound for a
                        # goal-switch bearing head on the executed start).
                        require(len(path_e) >= 2,
                                "executed bearing path invalid")
                        origin_xz = np.asarray(start_pos, dtype=float)[[0, 2]]
                        target_xz = None
                        for waypoint in path_e[1:]:
                            delta = np.asarray(
                                waypoint, dtype=float)[[0, 2]] - origin_xz
                            if float(np.linalg.norm(delta)) >= 0.3:
                                target_xz = delta
                                break
                        if target_xz is None:
                            target_xz = np.asarray(
                                path_e[-1], dtype=float)[[0, 2]] - origin_xz
                        start_psi = float(base.yaw_facing(target_xz))
                        record["executed_bearing"] = True

                leg = base.run_policy_leg(
                    sim, pathfinder, start_pos, start_psi, goal_jpg,
                    b_hab[[0, 2]], float(geo_leg), None,
                    terminal_mode="off", policy_backend=None,
                    success_dist=args.success_dist,
                    episode_seed=episode_seed, leg_index=1)
                record.update({
                    "reached": bool(leg["reached"]),
                    "final_dist_m": float(leg["final_goal_dist_m"]),
                    "path_len_m": float(leg["path_len"]),
                    "steps": int(leg["steps"]),
                    "geo_leg_m": float(geo_leg),
                    "spl": leg_spl(leg, float(geo_leg)),
                })
                results.append(record)
                print(f"[eval3sym] {episode} {arm} reached={record['reached']} "
                      f"final={record.get('final_dist_m'):.3f} "
                      f"geo={geo_leg:.3f}")
    finally:
        sim.close()

    fieldnames = sorted({key for record in results for key in record})
    csv_path = os.path.join(args.out, "symmetry_arms.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary: dict = {"arms": {}, "paired": {}}
    by_arm: dict[str, list[dict]] = {}
    for record in results:
        by_arm.setdefault(record["arm"], []).append(record)
    for arm, records in by_arm.items():
        scored = [r for r in records if r.get("skip_reason") is None]
        summary["arms"][arm] = {
            "episodes": len(records),
            "scored": len(scored),
            "skipped": len(records) - len(scored),
            "success": sum(1 for r in scored if r["reached"]),
            "mean_final_dist_m": (
                float(np.mean([r["final_dist_m"] for r in scored]))
                if scored else None),
            "mean_spl": (float(np.mean([r["spl"] for r in scored]))
                         if scored else None),
        }
    for left, right in (("b2_rendered", "b2_arrival"),
                        ("b2_arrival", "b1_matched"),
                        ("b2_arrival", "b2_executed"),
                        ("b2_executed", "b1_role_matched")):
        if left not in by_arm or right not in by_arm:
            continue
        left_map = {r["episode"]: r for r in by_arm[left]
                    if r.get("skip_reason") is None}
        right_map = {r["episode"]: r for r in by_arm[right]
                     if r.get("skip_reason") is None}
        shared = sorted(set(left_map) & set(right_map))
        summary["paired"][f"{left}_vs_{right}"] = {
            "episodes": len(shared),
            "gain": [e for e in shared
                     if right_map[e]["reached"] and not left_map[e]["reached"]],
            "loss": [e for e in shared
                     if left_map[e]["reached"] and not right_map[e]["reached"]],
        }
    summary_path = os.path.join(args.out, "symmetry_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[eval3sym] written: {csv_path} {summary_path}")


if __name__ == "__main__":
    main()
