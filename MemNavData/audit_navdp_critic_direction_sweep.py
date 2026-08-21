#!/usr/bin/env python3
"""Test frozen NavDP signals as arbiters over counterfactual directions.

For each selected factual Novel-A state, the script first performs the normal
ImageGoal request (the only call that appends to NavDP's FIFO).  It then sends
eight equal-angle mixed image/point-goal requests through the read-only
``mixgoal_resample`` endpoint with identical diffusion noise.  The request
whose maximum critic value is largest is selected.

With ``--goal-contrast``, the same fixed candidates are additionally scored by
the difference between zero-goal and ImageGoal-conditioned denoising MSE.  A
deterministically shuffled goal from another scene is scored with the identical
noise as a target-specificity control.  This contrast is not treated as an
exact or calibrated diffusion likelihood.

This diagnostic deliberately separates three questions:

1. Did the critic choose the discrete request nearest the geodesic bearing?
2. Did the selected point-token trajectory actually execute that direction?
3. Was any probed trajectory capable of doing so (execution ceiling)?

Habitat bearing labels and native failure labels make this a consumed-split,
privileged architecture probe.  It is not a deployable navigation result.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import requests

from MemNavData.audit_observed_frontier_bearing_coverage import (
    circular_error_deg,
    data_to_hab,
    matrix_from_nested,
    parquet_floor_pose,
    path_initial_bearing,
    sha256_file,
)
from MemNavData.deterministic_eval_protocol import diffusion_plan_seed
from MemNavData.novel_a_bearing_gate import (
    critic_shadow_diagnostics,
    wrap_deg,
)
from MemNavData.navdp_goal_contrast import (
    goal_contrast_diagnostics,
    summarize_goal_contrast,
)


DEFAULT_DIRECTIONS_DEG = tuple(float(value) for value in range(-180, 180, 45))


def jpg_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def depth_png_bytes(depth: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    encoded = np.clip(
        np.asarray(depth, dtype=np.float64) * 10000.0, 0, 65535
    ).astype(np.uint16)
    Image.fromarray(encoded).save(buffer, format="PNG")
    return buffer.getvalue()


def poisson_binomial_upper_tail(successes: int,
                                probabilities: Sequence[float]) -> float:
    """Exact P(X >= successes) for independent unequal Bernoulli nulls."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or np.any(~np.isfinite(probs)):
        raise ValueError("probabilities must be a finite vector")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not 0 <= successes <= len(probs):
        raise ValueError("success count is out of range")
    distribution = np.asarray([1.0], dtype=np.float64)
    for probability in probs:
        updated = np.zeros(len(distribution) + 1, dtype=np.float64)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return float(distribution[successes:].sum())


def _post_json(session: requests.Session, url: str, payload: dict,
               timeout_s: float) -> dict:
    response = session.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _post_imagegoal(session: requests.Session, base_url: str, *,
                    image_jpg: bytes, goal_jpg: bytes, depth_png: bytes,
                    diffusion_seed_value: int, timeout_s: float) -> dict:
    response = session.post(
        f"{base_url}/imagegoal_step",
        files={
            "image": ("image.jpg", image_jpg),
            "goal": ("goal.jpg", goal_jpg),
            "depth": ("depth.png", depth_png),
        },
        data={"diffusion_seed": str(diffusion_seed_value)},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("diffusion_seed")) != diffusion_seed_value:
        raise RuntimeError("native request did not echo the diffusion seed")
    return payload


def _post_mixgoal(session: requests.Session, base_url: str, *,
                  image_jpg: bytes, goal_jpg: bytes, depth_png: bytes,
                  direction_deg: float, radius_m: float,
                  diffusion_seed_value: int, timeout_s: float,
                  score_goal_contrast: bool = False,
                  control_goal_jpg: bytes | None = None,
                  score_seed_value: int | None = None,
                  score_timesteps: Sequence[int] | None = None,
                  score_noise_samples: int = 1) -> dict:
    theta = math.radians(float(direction_deg))
    files = {
        "image": ("image.jpg", image_jpg),
        "image_goal": ("goal.jpg", goal_jpg),
        "depth": ("depth.png", depth_png),
    }
    data = {
        "goal_data": json.dumps({
            "goal_x": [radius_m * math.cos(theta)],
            "goal_y": [radius_m * math.sin(theta)],
        }),
        "diffusion_seed": str(diffusion_seed_value),
    }
    if score_goal_contrast:
        if score_seed_value is None or score_timesteps is None:
            raise ValueError("goal contrast requires a seed and timesteps")
        data.update({
            "score_goal_contrast": "true",
            "score_seed": str(score_seed_value),
            "score_timesteps": json.dumps(list(score_timesteps)),
            "score_noise_samples": str(score_noise_samples),
        })
        if control_goal_jpg is not None:
            files["control_goal"] = ("control_goal.jpg", control_goal_jpg)
    response = session.post(
        f"{base_url}/mixgoal_resample",
        files=files,
        data=data,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("memory_mutated") is not False:
        raise RuntimeError("mixed request did not assert read-only FIFO semantics")
    if payload.get("queue_hashes_before") != payload.get("queue_hashes_after"):
        raise RuntimeError("mixed request changed the NavDP FIFO")
    if int(payload.get("diffusion_seed")) != diffusion_seed_value:
        raise RuntimeError("mixed request did not echo the diffusion seed")
    if score_goal_contrast:
        score = payload.get("goal_contrast")
        if not isinstance(score, dict):
            raise RuntimeError("mixed request omitted goal contrast")
        if int(score.get("score_seed")) != score_seed_value:
            raise RuntimeError("mixed request did not echo the score seed")
        if list(score.get("timesteps", [])) != list(score_timesteps):
            raise RuntimeError("mixed request changed the score timesteps")
        if int(score.get("noise_samples")) != score_noise_samples:
            raise RuntimeError("mixed request changed the score sample count")
    return payload


def _read_metric(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["episode"]: row for row in csv.DictReader(handle)}


def _frozen_goal_record(args: argparse.Namespace, manifest: dict,
                        overlay: dict, scene: str, episode: str) -> dict:
    anchor_scenes = set(manifest["selection"]["anchor_scenes"])
    episode_root = (args.legacy_episode_root if scene in anchor_scenes
                    else args.expanded_episode_root)
    frozen = overlay["goal_a_images"][scene][episode]
    path = (episode_root / scene / episode / "videos" / "chunk-000" /
            "observation.images.rgb" / f"{frozen['frame_index']}.jpg")
    return {
        "scene": scene,
        "episode": episode,
        "path": path,
        "bytes": int(frozen["bytes"]),
        "sha256": frozen["sha256"],
    }


def _read_verified_goal(record: dict) -> bytes:
    path = Path(record["path"])
    if (not path.is_file() or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]):
        raise RuntimeError(
            f"frozen Goal-A mismatch: {record['scene']}/{record['episode']}")
    return path.read_bytes()


def _control_goal_map(args: argparse.Namespace, manifest: dict,
                      overlay: dict) -> dict[tuple[str, str], dict]:
    """Deterministically derange goals, always crossing a scene boundary."""
    records = [
        _frozen_goal_record(args, manifest, overlay, scene, item["episode"])
        for scene in manifest["selection"]["selected_scenes"]
        for item in manifest["episodes"][scene]
    ]
    if len({record["scene"] for record in records}) < 2:
        raise RuntimeError("shuffled-goal control requires multiple scenes")
    mapping = {}
    for index, record in enumerate(records):
        control = next(
            records[(index + offset) % len(records)]
            for offset in range(1, len(records))
            if records[(index + offset) % len(records)]["scene"]
            != record["scene"]
        )
        mapping[(record["scene"], record["episode"])] = control
    return mapping


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], threshold_deg: float) -> dict:
    if not rows:
        return {"states": 0}

    def hit(row: dict, key: str) -> bool:
        value = row[key]
        return value is not None and float(value) <= threshold_deg

    request_hits = [hit(row, "critic_request_error_deg") for row in rows]
    executed_hits = [hit(row, "critic_executed_error_deg") for row in rows]
    ceiling_hits = [hit(row, "execution_ceiling_error_deg") for row in rows]
    native_hits = [hit(row, "native_executed_error_deg") for row in rows]
    gains = sum(critic and not native
                for critic, native in zip(executed_hits, native_hits))
    losses = sum(native and not critic
                 for critic, native in zip(executed_hits, native_hits))
    probabilities = [float(row["random_request_hit_probability"])
                     for row in rows]
    return {
        "states": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "threshold_deg": float(threshold_deg),
        "critic_request_hits": int(sum(request_hits)),
        "critic_request_rate": float(np.mean(request_hits)),
        "critic_executed_hits": int(sum(executed_hits)),
        "critic_executed_rate": float(np.mean(executed_hits)),
        "execution_ceiling_hits": int(sum(ceiling_hits)),
        "execution_ceiling_rate": float(np.mean(ceiling_hits)),
        "native_executed_hits": int(sum(native_hits)),
        "native_executed_rate": float(np.mean(native_hits)),
        "critic_executed_vs_native_gains": int(gains),
        "critic_executed_vs_native_losses": int(losses),
        "random_request_expected_hits": float(sum(probabilities)),
        "critic_request_random_null_upper_tail_p": (
            poisson_binomial_upper_tail(sum(request_hits), probabilities)),
        "mean_critic_score_margin": float(np.mean([
            row["critic_score_margin"] for row in rows
        ])),
    }


def format_optional_angle(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.1f}"


def run(args: argparse.Namespace) -> dict:
    import pandas as pd
    from MemNavData.generate_twoleg import geodesic, make_sim, render

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    overlay = json.loads(args.input_overlay.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(args.manifest)
    if overlay["parent_manifest_sha256"] != manifest_sha:
        raise RuntimeError("Goal-A overlay does not pin the supplied manifest")
    if sha256_file(args.checkpoint) != manifest["dependencies"][
            "navdp_checkpoint"]["sha256"]:
        raise RuntimeError("NavDP checkpoint differs from the frozen manifest")
    control_goals = (
        _control_goal_map(args, manifest, overlay)
        if args.goal_contrast else {})

    session = requests.Session()
    # A reset also proves that the expected endpoint is alive before loading
    # any Habitat scene.
    _post_json(session, f"{args.base_url}/navigator_reset", {
        "intrinsic": [[355.81464, 0.0, 240.0],
                      [0.0, 351.687, 135.0],
                      [0.0, 0.0, 1.0]],
        "stop_threshold": args.stop_threshold,
        "batch_size": 1,
        "seed": args.base_seed,
    }, args.timeout_s)

    state_rows: list[dict] = []
    direction_rows: list[dict] = []
    selected_scenes = manifest["selection"]["selected_scenes"]
    anchor_scenes = set(manifest["selection"]["anchor_scenes"])
    for scene_index, scene in enumerate(selected_scenes):
        if (args.max_states is not None
                and len(state_rows) >= args.max_states):
            break
        result_root = args.plans_root / f"{scene_index:02d}_{scene}"
        metrics = _read_metric(result_root / "navdp_native" / "metric.csv")
        selected_episodes = [
            item for item in manifest["episodes"][scene]
            if (not args.failures_only
                or float(metrics[item["episode"]]["reached_A"]) <= 0.5)
        ]
        if not selected_episodes:
            continue
        glb = args.asset_root / scene / f"{scene}.glb"
        navmesh = args.asset_root / scene / f"{scene}.navmesh"
        expected_asset = manifest["assets"][scene]
        if (not glb.is_file() or not navmesh.is_file()
                or glb.stat().st_size != int(expected_asset["bytes"])
                or sha256_file(glb) != expected_asset["sha256"]):
            raise RuntimeError(f"frozen scene asset mismatch: {scene}")
        episode_root = (args.legacy_episode_root if scene in anchor_scenes
                        else args.expanded_episode_root)
        sim = make_sim(str(glb), str(navmesh))
        try:
            for episode_item in selected_episodes:
                if (args.max_states is not None
                        and len(state_rows) >= args.max_states):
                    break
                episode = episode_item["episode"]
                episode_dir = episode_root / scene / episode
                meta_path = episode_dir / "meta" / "gen_meta.json"
                parquet_path = (episode_dir / "data" / "chunk-000" /
                                "episode_000000.parquet")
                for label, path in (("metadata", meta_path),
                                    ("parquet", parquet_path)):
                    expected = episode_item["files"][label]
                    if (not path.is_file()
                            or path.stat().st_size != int(expected["bytes"])
                            or sha256_file(path) != expected["sha256"]):
                        raise RuntimeError(
                            f"frozen {label} mismatch: {scene}/{episode}")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                parquet = pd.read_parquet(parquet_path)
                camera_height = float(meta.get("camera_height_m", 0.5))
                intrinsic = matrix_from_nested(
                    parquet.iloc[0]["observation.camera_intrinsic"])
                initial_floor, _ = parquet_floor_pose(
                    parquet.iloc[0]["action"], camera_height)
                goal_floor = data_to_hab(meta["A"])
                goal_floor[1] = initial_floor[1]
                goal_record = _frozen_goal_record(
                    args, manifest, overlay, scene, episode)
                goal_jpg = _read_verified_goal(goal_record)
                control_record = control_goals.get((scene, episode))
                control_goal_jpg = (
                    _read_verified_goal(control_record)
                    if control_record is not None else None)

                plans_path = (result_root / "geometry_router" /
                              f"{episode}_plans.json")
                plans = json.loads(plans_path.read_text(encoding="utf-8"))
                trace_by_frame = {
                    int(row["frame_idx"]): row
                    for row in plans["legA_memory_trace"]
                }
                plan_frames = [int(row["frame_idx"])
                               for row in plans["legA"][:args.max_plans]]
                if set(plan_frames) - set(trace_by_frame):
                    raise RuntimeError(
                        f"plan/trace mismatch: {scene}/{episode}")
                episode_seed = int(metrics[episode]["seed"])
                _post_json(session, f"{args.base_url}/navigator_reset", {
                    "intrinsic": intrinsic.tolist(),
                    "stop_threshold": args.stop_threshold,
                    "batch_size": 1,
                    "seed": episode_seed,
                }, args.timeout_s)

                for plan_index, frame_index in enumerate(plan_frames):
                    if (args.max_states is not None
                            and len(state_rows) >= args.max_states):
                        break
                    trace = trace_by_frame[frame_index]
                    current = np.asarray([
                        float(trace["x"]), initial_floor[1], float(trace["z"])
                    ], dtype=np.float64)
                    rgb, depth = render(
                        sim,
                        current + np.asarray([0.0, camera_height, 0.0]),
                        float(trace["yaw"]),
                    )
                    image_jpg = jpg_bytes(rgb)
                    depth_png = depth_png_bytes(depth)
                    seed = diffusion_plan_seed(episode_seed, 0, plan_index)
                    native = _post_imagegoal(
                        session, args.base_url,
                        image_jpg=image_jpg,
                        goal_jpg=goal_jpg,
                        depth_png=depth_png,
                        diffusion_seed_value=seed,
                        timeout_s=args.timeout_s,
                    )
                    ok, remaining_m, oracle_path = geodesic(
                        sim.pathfinder, current, goal_floor)
                    oracle_world = (
                        path_initial_bearing(oracle_path, current) if ok else None)
                    if oracle_world is None:
                        raise RuntimeError(
                            f"invalid oracle bearing: {scene}/{episode}/{plan_index}")
                    oracle_relative = wrap_deg(math.degrees(
                        oracle_world - float(trace["yaw"])))
                    native_shadow = critic_shadow_diagnostics(
                        native, requested_heading_deg=oracle_relative)

                    probes = []
                    for direction_index, direction_deg in enumerate(
                            args.directions_deg):
                        mixed = _post_mixgoal(
                            session, args.base_url,
                            image_jpg=image_jpg,
                            goal_jpg=goal_jpg,
                            depth_png=depth_png,
                            direction_deg=direction_deg,
                            radius_m=args.radius_m,
                            diffusion_seed_value=seed,
                            timeout_s=args.timeout_s,
                            score_goal_contrast=args.goal_contrast,
                            control_goal_jpg=control_goal_jpg,
                            score_seed_value=(
                                seed + args.score_seed_offset
                                if args.goal_contrast else None),
                            score_timesteps=(
                                args.score_timesteps
                                if args.goal_contrast else None),
                            score_noise_samples=args.score_noise_samples,
                        )
                        shadow = critic_shadow_diagnostics(
                            mixed, requested_heading_deg=direction_deg)
                        if shadow["critic_max"] is None:
                            raise RuntimeError("mixed request omitted critic values")
                        selected_heading = shadow["selected_heading_deg"]
                        executed_error = (
                            abs(wrap_deg(selected_heading - oracle_relative))
                            if selected_heading is not None else None)
                        request_error = abs(wrap_deg(
                            direction_deg - oracle_relative))
                        row = {
                            "scene": scene,
                            "episode": episode,
                            "plan_index": plan_index,
                            "frame_index": frame_index,
                            "direction_index": direction_index,
                            "request_direction_deg": direction_deg,
                            "oracle_relative_deg": oracle_relative,
                            "request_error_deg": request_error,
                            "selected_heading_deg": selected_heading,
                            "executed_error_deg": executed_error,
                            "critic_max": shadow["critic_max"],
                            "critic_min": shadow["critic_min"],
                            "critic_unique_4dp": shadow["critic_unique_4dp"],
                            "heading_resultant_r": shadow["heading_resultant_r"],
                            "selected_request_error_deg": shadow[
                                "selected_request_error_deg"],
                        }
                        if args.goal_contrast:
                            contrast = goal_contrast_diagnostics(
                                mixed, requested_heading_deg=direction_deg)
                            goal_heading = contrast[
                                "goal_selected_heading_deg"]
                            control_heading = contrast[
                                "control_selected_heading_deg"]
                            row.update({
                                "goal_candidate_index": contrast[
                                    "goal_candidate_index"],
                                "goal_selected_heading_deg": goal_heading,
                                "goal_selected_request_error_deg": contrast[
                                    "goal_selected_request_error_deg"],
                                "goal_executed_error_deg": (
                                    abs(wrap_deg(
                                        goal_heading - oracle_relative))
                                    if goal_heading is not None else None),
                                "goal_score": contrast["goal_score"],
                                "goal_candidate_score_margin": contrast[
                                    "goal_score_margin"],
                                "goal_candidate_score_std": contrast[
                                    "goal_score_std"],
                                "normalized_goal_score": contrast[
                                    "normalized_goal_score"],
                                "control_candidate_index": contrast[
                                    "control_candidate_index"],
                                "control_selected_heading_deg": (
                                    control_heading),
                                "control_selected_request_error_deg": contrast[
                                    "control_selected_request_error_deg"],
                                "control_executed_error_deg": (
                                    abs(wrap_deg(
                                        control_heading - oracle_relative))
                                    if control_heading is not None else None),
                                "control_score": contrast["control_score"],
                                "control_candidate_score_margin": contrast[
                                    "control_score_margin"],
                                "goal_vs_control_at_goal_choice": contrast[
                                    "goal_vs_control_at_goal_choice"],
                                "control_goal_scene": control_record["scene"],
                                "control_goal_episode": control_record[
                                    "episode"],
                                "control_goal_sha256": control_record["sha256"],
                            })
                        probes.append(row)
                        direction_rows.append(row)

                    scores = np.asarray(
                        [row["critic_max"] for row in probes], dtype=np.float64)
                    order = np.argsort(-scores, kind="stable")
                    chosen = probes[int(order[0])]
                    native_heading = native_shadow["selected_heading_deg"]
                    native_error = (
                        abs(wrap_deg(native_heading - oracle_relative))
                        if native_heading is not None else None)
                    executable = [row["executed_error_deg"] for row in probes
                                  if row["executed_error_deg"] is not None]
                    request_errors = [row["request_error_deg"] for row in probes]
                    random_hits = sum(
                        error <= args.threshold_deg for error in request_errors)
                    state = {
                        "scene": scene,
                        "episode": episode,
                        "plan_index": plan_index,
                        "frame_index": frame_index,
                        "direction_count": len(probes),
                        "native_reached_A": (
                            float(metrics[episode]["reached_A"]) > 0.5),
                        "goal_remaining_geodesic_m": float(remaining_m),
                        "oracle_relative_deg": oracle_relative,
                        "native_executed_error_deg": native_error,
                        "critic_chosen_direction_deg": chosen[
                            "request_direction_deg"],
                        "critic_request_error_deg": chosen["request_error_deg"],
                        "critic_executed_heading_deg": chosen[
                            "selected_heading_deg"],
                        "critic_executed_error_deg": chosen[
                            "executed_error_deg"],
                        "execution_ceiling_error_deg": (
                            min(executable) if executable else None),
                        "request_ceiling_error_deg": min(request_errors),
                        "critic_score": float(scores[order[0]]),
                        "critic_score_margin": float(
                            scores[order[0]] - scores[order[1]]),
                        "random_request_hit_probability": (
                            random_hits / len(request_errors)),
                    }
                    if args.goal_contrast:
                        goal_scores = np.asarray(
                            [row["goal_score"] for row in probes],
                            dtype=np.float64)
                        goal_order = np.argsort(-goal_scores, kind="stable")
                        goal_choice = probes[int(goal_order[0])]
                        control_scores = np.asarray(
                            [row["control_score"] for row in probes],
                            dtype=np.float64)
                        control_order = np.argsort(
                            -control_scores, kind="stable")
                        control_choice = probes[int(control_order[0])]
                        nearest_request_index = int(np.argmin(request_errors))
                        goal_oracle_request_rank = int(
                            np.flatnonzero(
                                goal_order == nearest_request_index)[0] + 1)
                        control_oracle_request_rank = int(
                            np.flatnonzero(
                                control_order == nearest_request_index)[0] + 1)
                        state.update({
                            "goal_chosen_direction_deg": goal_choice[
                                "request_direction_deg"],
                            "goal_request_error_deg": goal_choice[
                                "request_error_deg"],
                            "goal_executed_heading_deg": goal_choice[
                                "goal_selected_heading_deg"],
                            "goal_executed_error_deg": goal_choice[
                                "goal_executed_error_deg"],
                            "goal_score": float(goal_scores[goal_order[0]]),
                            "goal_score_margin": float(
                                goal_scores[goal_order[0]]
                                - goal_scores[goal_order[1]]),
                            "goal_candidate_score_margin": goal_choice[
                                "goal_candidate_score_margin"],
                            "goal_oracle_request_rank": (
                                goal_oracle_request_rank),
                            "control_chosen_direction_deg": control_choice[
                                "request_direction_deg"],
                            "control_request_error_deg": control_choice[
                                "request_error_deg"],
                            "control_executed_heading_deg": control_choice[
                                "control_selected_heading_deg"],
                            "control_executed_error_deg": control_choice[
                                "control_executed_error_deg"],
                            "control_score": float(
                                control_scores[control_order[0]]),
                            "control_score_margin": float(
                                control_scores[control_order[0]]
                                - control_scores[control_order[1]]),
                            "control_oracle_request_rank": (
                                control_oracle_request_rank),
                            "goal_vs_control_at_goal_choice": goal_choice[
                                "goal_vs_control_at_goal_choice"],
                            "control_goal_scene": control_record["scene"],
                            "control_goal_episode": control_record["episode"],
                            "control_goal_sha256": control_record["sha256"],
                        })
                    state_rows.append(state)
                    message = (
                        f"[{scene}/{episode}/p{plan_index}] "
                        f"oracle={oracle_relative:+.1f} "
                        f"native={format_optional_angle(native_error)} "
                        f"critic_req={format_optional_angle(state['critic_request_error_deg'])} "
                        f"critic_exec={format_optional_angle(state['critic_executed_error_deg'])} "
                        f"ceiling={format_optional_angle(state['execution_ceiling_error_deg'])}")
                    if args.goal_contrast:
                        message += (
                            f" goal_req={format_optional_angle(state['goal_request_error_deg'])}"
                            f" goal_exec={format_optional_angle(state['goal_executed_error_deg'])}"
                            f" control_req={format_optional_angle(state['control_request_error_deg'])}")
                    print(message, flush=True)
        finally:
            sim.close()

    report = {
        "scope": (
            "privileged, failure-enriched architecture diagnostic on consumed "
            "development episodes; no closed-loop claim; deployment_approved=false"
        ),
        "definitions": {
            "direction_requests_deg": list(args.directions_deg),
            "direction_count": len(args.directions_deg),
            "point_token_radius_m": args.radius_m,
            "identical_diffusion_seed_across_directions": True,
            "critic_direction_score": "maximum all_values within one request",
            "threshold_deg": args.threshold_deg,
            "goal_contrast_enabled": args.goal_contrast,
            "goal_contrast_score": (
                "max over candidates of paired "
                "MSE(zero-goal)-MSE(correct-ImageGoal)"
                if args.goal_contrast else None),
            "goal_contrast_is_calibrated_likelihood": False,
            "goal_contrast_timesteps": (
                list(args.score_timesteps) if args.goal_contrast else None),
            "goal_contrast_noise_samples": (
                args.score_noise_samples if args.goal_contrast else None),
            "identical_score_noise_across_candidates_and_directions": (
                args.goal_contrast),
            "control_goal": (
                "deterministic next manifest goal from a different scene"
                if args.goal_contrast else None),
        },
        "selection": {
            "failures_only": args.failures_only,
            "max_plans_per_episode": args.max_plans,
            "max_states": args.max_states,
        },
        "summary": summarize(state_rows, args.threshold_deg),
        "goal_contrast_summary": (
            summarize_goal_contrast(state_rows, args.threshold_deg)
            if args.goal_contrast else None),
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "input_overlay": str(args.input_overlay.resolve()),
            "input_overlay_sha256": sha256_file(args.input_overlay),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "base_url": args.base_url,
            "plans_root": str(args.plans_root.resolve()),
            "score_seed_offset": (
                args.score_seed_offset if args.goal_contrast else None),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "states.csv", state_rows)
    _write_csv(args.out / "directions.csv", direction_rows)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:21000")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("MemNavData/expanded_navdp_router_eval_20260805.json"))
    parser.add_argument(
        "--input-overlay", type=Path,
        default=Path("MemNavData/novel_a_bearing_inputs_20260808.json"))
    parser.add_argument(
        "--plans-root", type=Path,
        default=Path(".diagnostics/twentyscene_local_20260808"))
    parser.add_argument(
        "--asset-root", type=Path,
        default=Path("/home/asus/Research/datasets/mp3d_20scene/assets"))
    parser.add_argument(
        "--legacy-episode-root", type=Path,
        default=Path(
            "/home/asus/Research/Nav-axis-uturn/.diagnostics/"
            "unseen_scene_eval_20260803/episodes"))
    parser.add_argument(
        "--expanded-episode-root", type=Path,
        default=Path("/home/asus/Research/datasets/mp3d_20scene/episodes"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/"
            "navdp_checkpoint.ckpt"))
    parser.add_argument(
        "--out", type=Path, default=None)
    parser.add_argument("--max-plans", type=int, default=1)
    parser.add_argument(
        "--max-states", type=int, default=None,
        help="optional deterministic prefix limit for smoke tests")
    parser.add_argument("--failures-only", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument(
        "--goal-contrast", action=argparse.BooleanOptionalAction,
        default=False,
        help="also rank candidates by paired ImageGoal/null denoising error")
    parser.add_argument(
        "--score-timesteps", default="0,1,2,3,4,5,6,7,8,9",
        help="comma-separated frozen DDPM training timesteps")
    parser.add_argument("--score-noise-samples", type=int, default=1)
    parser.add_argument("--score-seed-offset", type=int, default=104729)
    parser.add_argument("--radius-m", type=float, default=2.0)
    parser.add_argument("--threshold-deg", type=float, default=30.0)
    parser.add_argument("--stop-threshold", type=float, default=-0.5)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    args.directions_deg = DEFAULT_DIRECTIONS_DEG
    try:
        args.score_timesteps = tuple(
            int(value.strip()) for value in args.score_timesteps.split(",")
            if value.strip())
    except ValueError:
        parser.error("score-timesteps must be comma-separated integers")
    if args.out is None:
        args.out = Path(
            ".diagnostics/navdp_goal_contrast_direction_sweep_20260809"
            if args.goal_contrast else
            ".diagnostics/navdp_critic_direction_sweep_20260809")
    if args.max_plans < 1:
        parser.error("max-plans must be positive")
    if args.max_states is not None and args.max_states < 1:
        parser.error("max-states must be positive")
    if args.radius_m <= 0 or args.threshold_deg <= 0 or args.timeout_s <= 0:
        parser.error("radius, threshold, and timeout must be positive")
    if (not args.score_timesteps
            or len(set(args.score_timesteps)) != len(args.score_timesteps)
            or any(value < 0 or value > 9
                   for value in args.score_timesteps)):
        parser.error("score-timesteps must be unique values in [0, 9]")
    if not 1 <= args.score_noise_samples <= 16:
        parser.error("score-noise-samples must lie in [1, 16]")
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
