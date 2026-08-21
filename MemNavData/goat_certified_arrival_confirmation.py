#!/usr/bin/env python3
"""Disjoint GOAT first-ImageGoal confirmation of certified semantic STOP.

One invocation runs exactly one frozen episode.  Native NavDP remains the
navigation policy.  Its no-motion output is treated as an arrival proposal;
the LingBot/PnP sidecar may authorize official ``SUBTASK_STOP`` under the
frozen 7.5 cm contract.  Rejected proposals fall back only to the highest
critic-scored motion trajectory from the already sampled candidate batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Mapping, Tuple

import numpy as np

try:
    from MemNavData.goat_certified_arrival_contract import (
        ArrivalEvidence,
        contract_receipt,
        decide_subtask_stop,
    )
    from MemNavData.goat_contract_smoke import (
        EXPECTED_GOAT_COMMIT,
        _assert_same_pose,
        _build_config,
        _current_image_parameters,
        _episode_scene_id,
        _jsonable,
        _render_raw_goal,
        _select_episodes,
    )
    from MemNavData.goat_navdp_discrete_adapter import (
        DiscreteAdapterConfig,
        GoatNavAction,
        best_scored_motion_candidate,
        navdp_waypoints_to_goat_decision,
    )
    from MemNavData.goat_navdp_runtime_pilot import (
        ACTION_NAMES,
        _camera_intrinsic,
        _depth_png_bytes,
        _metric_value,
        _navdp_plan,
        _navdp_reset,
        _navdp_wire_jpeg_bytes,
        _plan_seed,
        _rgb_jpeg_bytes,
        _validated_critic_receipt,
    )
except ModuleNotFoundError:
    from goat_certified_arrival_contract import (  # type: ignore[no-redef]
        ArrivalEvidence,
        contract_receipt,
        decide_subtask_stop,
    )
    from goat_contract_smoke import (  # type: ignore[no-redef]
        EXPECTED_GOAT_COMMIT,
        _assert_same_pose,
        _build_config,
        _current_image_parameters,
        _episode_scene_id,
        _jsonable,
        _render_raw_goal,
        _select_episodes,
    )
    from goat_navdp_discrete_adapter import (  # type: ignore[no-redef]
        DiscreteAdapterConfig,
        GoatNavAction,
        best_scored_motion_candidate,
        navdp_waypoints_to_goat_decision,
    )
    from goat_navdp_runtime_pilot import (  # type: ignore[no-redef]
        ACTION_NAMES,
        _camera_intrinsic,
        _depth_png_bytes,
        _metric_value,
        _navdp_plan,
        _navdp_reset,
        _navdp_wire_jpeg_bytes,
        _plan_seed,
        _rgb_jpeg_bytes,
        _validated_critic_receipt,
    )


SCHEMA_VERSION = "goat_certified_arrival_episode_v1_20260815"
MANIFEST_SCHEMA = "goat_certified_arrival_manifest_v1_20260815"
NUMPY_SEED_MODULUS = 2 ** 32


def _service_reset_seed(seed: int) -> int:
    """Map the frozen 63-bit episode hash to every service's common domain.

    NavDP's request-level helper accepts 63-bit Torch seeds and internally
    reduces only NumPy's seed.  MemNav's historical reset endpoint forwards
    the seed directly to NumPy, whose legacy API accepts only uint32.  Reset
    both services with the same explicit uint32 value; diffusion requests keep
    their separate frozen 63-bit per-plan seeds.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("service reset seed must be an integer")
    if seed < 0:
        raise ValueError("service reset seed must be non-negative")
    return int(seed % NUMPY_SEED_MODULUS)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _load_manifest(path: pathlib.Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unexpected certified-arrival manifest schema")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("certified-arrival manifest has no episodes")
    pairs = []
    for item in episodes:
        if not isinstance(item, Mapping):
            raise RuntimeError("episode entry is not an object")
        pair = (str(item.get("scene_id", "")), str(item.get("episode_id", "")))
        if not all(pair):
            raise RuntimeError("episode identity is empty")
        pairs.append(pair)
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate episode identity")
    if len({scene for scene, _ in pairs}) != len(pairs):
        raise RuntimeError("confirmation requires one episode per scene")
    if payload.get("arrival_contract") != contract_receipt():
        raise RuntimeError("arrival contract differs from frozen code")
    if int(payload.get("max_navigation_actions", 0)) <= 0:
        raise RuntimeError("navigation-action guard must be positive")
    if int(payload.get("same_observation_resample_limit", -1)) < 0:
        raise RuntimeError("resample limit must be non-negative")
    return payload


def _arrival_reset(
    session: Any,
    arrival_url: str,
    *,
    camera_height_m: float,
    camera_intrinsic: np.ndarray,
    episode_seed: int,
    episode_len: int,
    timeout_s: float,
) -> Dict[str, Any]:
    response = session.post(
        arrival_url.rstrip("/") + "/navigator_reset",
        json={
            "camera_height": float(camera_height_m),
            "camera_intrinsic": camera_intrinsic.tolist(),
            "seed": int(episode_seed),
            "episode_len": int(episode_len),
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("algo") != "memnav":
        raise RuntimeError("unexpected arrival-sidecar reset receipt")
    if not payload.get("certified_arrival", {}).get("enabled"):
        raise RuntimeError("arrival-sidecar certificate is disabled")
    if payload["certified_arrival"].get("contract") != contract_receipt():
        raise RuntimeError("arrival-sidecar contract differs")
    return payload


def _arrival_add(
    session: Any,
    arrival_url: str,
    rgb: Any,
    *,
    expected_frame_index: int,
    timeout_s: float,
) -> Dict[str, Any]:
    response = session.post(
        arrival_url.rstrip("/") + "/memory_step",
        files={"image": ("image.jpg", _rgb_jpeg_bytes(rgb), "image/jpeg")},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("frame_idx", -1)) != int(expected_frame_index):
        raise RuntimeError("arrival stream frame index changed")
    return payload


def _arrival_query(
    session: Any,
    arrival_url: str,
    goal: Any,
    *,
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    started = time.monotonic()
    response = session.post(
        arrival_url.rstrip("/") + "/arrival_query",
        files={"goal": ("goal.jpg", _rgb_jpeg_bytes(goal), "image/jpeg")},
        timeout=timeout_s,
    )
    latency = time.monotonic() - started
    response.raise_for_status()
    payload = response.json()
    if payload.get("contract") != contract_receipt():
        raise RuntimeError("arrival evidence contract changed")
    if payload.get("simulator_depth_consumed") is not False:
        raise RuntimeError("arrival service consumed forbidden simulator depth")
    return payload, latency


def _navdp_resample(
    session: Any,
    navdp_url: str,
    rgb: Any,
    depth: Any,
    goal: Any,
    diffusion_seed: int,
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    started = time.monotonic()
    response = session.post(
        navdp_url.rstrip("/") + "/imagegoal_resample",
        files={
            "image": (
                "image.jpg", _navdp_wire_jpeg_bytes(rgb), "image/jpeg"),
            "goal": (
                "goal.jpg", _navdp_wire_jpeg_bytes(goal), "image/jpeg"),
            "depth": ("depth.png", _depth_png_bytes(depth), "image/png"),
        },
        data={"diffusion_seed": str(int(diffusion_seed))},
        timeout=timeout_s,
    )
    latency = time.monotonic() - started
    response.raise_for_status()
    payload = response.json()
    if payload.get("memory_mutated") is not False:
        raise RuntimeError("read-only NavDP resample mutated its FIFO")
    if int(payload.get("diffusion_seed", -1)) != int(diffusion_seed):
        raise RuntimeError("NavDP resample did not echo deterministic seed")
    _validated_critic_receipt(payload)
    return payload, latency


def _distance_to_target(metrics: Mapping[str, Any]) -> float | None:
    value = _metric_value(metrics, "distance_to_goal")
    if isinstance(value, Mapping):
        value = value.get("distance_to_target")
    if value is None:
        return None
    distance = float(value)
    if not math.isfinite(distance) or distance < 0.0:
        raise RuntimeError("official distance-to-target metric is invalid")
    return distance


def _run_episode(
    env: Any,
    episode: Any,
    manifest: Mapping[str, Any],
    session: Any,
    navdp_url: str,
    arrival_url: str,
    rgb_hfov_deg: float,
    camera_height_m: float,
    request_timeout_s: float,
) -> Dict[str, Any]:
    scene_id = _episode_scene_id(episode)
    episode_id = str(episode.episode_id)
    max_actions = int(manifest["max_navigation_actions"])
    resample_limit = int(manifest["same_observation_resample_limit"])
    base_seed = int(manifest["base_seed"])
    adapter_cfg = DiscreteAdapterConfig(**manifest["adapter"])

    env.current_episode = episode
    observations = env.reset()
    if env.task.active_subtask_idx != 0:
        raise RuntimeError("GOAT task did not reset to subtask zero")
    if not episode.tasks or episode.tasks[0][1] != "image":
        raise RuntimeError("confirmation episode is not ImageGoal-first")
    if "rgb" not in observations or "depth" not in observations:
        raise RuntimeError("GOAT RGB-D observation is absent")

    rgb = np.asarray(observations["rgb"])
    intrinsic = _camera_intrinsic(rgb.shape[0], rgb.shape[1], rgb_hfov_deg)
    parameters, image_index = _current_image_parameters(episode, 0)
    before = env.sim.get_agent_state()
    sensors_before = set(env.sim._sensors)
    goal = np.asarray(_render_raw_goal(env.sim, parameters))
    _assert_same_pose(before, env.sim.get_agent_state())
    if sensors_before != set(env.sim._sensors):
        raise RuntimeError("temporary ImageGoal rendering sensor leaked")

    episode_seed = _service_reset_seed(
        _plan_seed(base_seed, scene_id, episode_id, -1))
    navdp_reset = _navdp_reset(
        session,
        navdp_url,
        intrinsic,
        episode_seed,
        float(manifest["navdp_stop_threshold"]),
        request_timeout_s,
    )
    arrival_reset = _arrival_reset(
        session,
        arrival_url,
        camera_height_m=camera_height_m,
        camera_intrinsic=intrinsic,
        episode_seed=episode_seed,
        episode_len=max_actions + 1,
        timeout_s=request_timeout_s,
    )
    _arrival_add(
        session,
        arrival_url,
        observations["rgb"],
        expected_frame_index=0,
        timeout_s=request_timeout_s,
    )

    started = time.monotonic()
    navigation_actions = 0
    environment_actions = 0
    plan_records = []
    zero_proposal_count = 0
    same_batch_fallback_count = 0
    extra_resample_count = 0
    certified_stop = False
    safe_stall = False
    forced_guard_stop = False
    stop_metrics_before: Dict[str, Any] = {}
    first_zero_distance_m = None
    first_zero_legacy_success = False

    while navigation_actions < max_actions and not env.episode_over:
        plan_index = len(plan_records)
        request_seed = _plan_seed(base_seed, scene_id, episode_id, plan_index)
        trajectory, server_payload, latency_s = _navdp_plan(
            session,
            navdp_url,
            observations["rgb"],
            observations["depth"],
            goal,
            request_seed,
            request_timeout_s,
        )
        primary = navdp_waypoints_to_goat_decision(trajectory, adapter_cfg)
        executed = primary
        arrival_evidence = None
        stop_decision = None
        resamples = []

        if primary.requires_arrival_certificate:
            zero_proposal_count += 1
            arrival_evidence, arrival_latency = _arrival_query(
                session, arrival_url, goal, timeout_s=request_timeout_s)
            stop_decision = decide_subtask_stop(ArrivalEvidence(
                native_zero_proposal=True,
                stream_frame_count=int(arrival_evidence["frame_count"]),
                certificate_accepted=bool(
                    arrival_evidence["certificate_accepted"]),
                predicted_distance_m=arrival_evidence.get(
                    "predicted_distance_m"),
                metric_scale_available=bool(
                    arrival_evidence["metric_scale_available"]),
            ))
            # GT is read only after inference and the frozen decision.
            current_metrics = _jsonable(env.get_metrics())
            current_distance = _distance_to_target(current_metrics)
            stop_decision["post_decision_official_distance_m"] = current_distance
            stop_decision["arrival_query_latency_s"] = arrival_latency
            if first_zero_distance_m is None:
                first_zero_distance_m = current_distance
                first_zero_legacy_success = bool(
                    current_distance is not None and current_distance < 0.25)
            if stop_decision["authorized_subtask_stop"]:
                stop_metrics_before = current_metrics
                observations = env.step(ACTION_NAMES[GoatNavAction.SUBTASK_STOP])
                environment_actions += 1
                certified_stop = True
            else:
                executed = best_scored_motion_candidate(
                    server_payload.get("all_trajectory"),
                    server_payload.get("all_values"),
                    adapter_cfg,
                )
                if executed.is_motion:
                    same_batch_fallback_count += 1

        elif not primary.is_motion:
            executed = best_scored_motion_candidate(
                server_payload.get("all_trajectory"),
                server_payload.get("all_values"),
                adapter_cfg,
            )
            if executed.is_motion:
                same_batch_fallback_count += 1

        # If the original frozen batch contains no executable motion, draw at
        # most the predeclared number of read-only batches from the same image.
        if not certified_stop and not executed.is_motion:
            for resample_index in range(resample_limit):
                seed = _plan_seed(
                    base_seed,
                    scene_id,
                    episode_id,
                    1_000_000 + plan_index * resample_limit + resample_index,
                )
                payload, resample_latency = _navdp_resample(
                    session,
                    navdp_url,
                    observations["rgb"],
                    observations["depth"],
                    goal,
                    seed,
                    request_timeout_s,
                )
                extra_resample_count += 1
                candidate = best_scored_motion_candidate(
                    payload.get("all_trajectory"),
                    payload.get("all_values"),
                    adapter_cfg,
                )
                resamples.append({
                    "resample_index": resample_index,
                    "diffusion_seed": seed,
                    "request_latency_s": resample_latency,
                    "motion_found": candidate.is_motion,
                    "candidate_index": candidate.candidate_index,
                    "reason": candidate.reason,
                })
                if candidate.is_motion:
                    executed = candidate
                    break

        values = np.asarray(server_payload.get("all_values", []), dtype=float)
        finite_values = values[np.isfinite(values)]
        plan_record = {
            "plan_index": plan_index,
            "diffusion_seed": request_seed,
            "request_latency_s": latency_s,
            "stream_frame_count": navigation_actions + 1,
            "selected_endpoint_xy_m": trajectory[-1, :2].tolist(),
            "selected_endpoint_norm_m": float(np.linalg.norm(
                trajectory[-1, :2])),
            "primary_disposition": primary.disposition.value,
            "executed_disposition": executed.disposition.value,
            "executed_candidate_index": executed.candidate_index,
            "executed_reason": executed.reason,
            "action_ids": [int(action) for action in executed.actions],
            "action_names": [ACTION_NAMES[action]
                             for action in executed.actions],
            "critic_max": (
                float(finite_values.max()) if finite_values.size else None),
            "arrival_evidence": arrival_evidence,
            "stop_decision": stop_decision,
            "resamples": resamples,
        }
        plan_records.append(plan_record)

        if certified_stop:
            break
        if not executed.is_motion:
            safe_stall = True
            break

        for action in executed.actions:
            if action == GoatNavAction.SUBTASK_STOP:
                raise RuntimeError("semantic STOP appeared inside motion chunk")
            if navigation_actions >= max_actions or env.episode_over:
                break
            observations = env.step(ACTION_NAMES[action])
            navigation_actions += 1
            environment_actions += 1
            if not env.episode_over:
                _arrival_add(
                    session,
                    arrival_url,
                    observations["rgb"],
                    expected_frame_index=navigation_actions,
                    timeout_s=request_timeout_s,
                )

    if not certified_stop and not env.episode_over:
        # Metric-finalization only.  This action is never attributed to the
        # controller and cannot count as certified success.
        stop_metrics_before = _jsonable(env.get_metrics())
        observations = env.step(ACTION_NAMES[GoatNavAction.SUBTASK_STOP])
        environment_actions += 1
        forced_guard_stop = True

    metrics_after = _jsonable(env.get_metrics())
    official_success = _metric_value(
        metrics_after, "success", "subtask_success", 0)
    certified_success = bool(certified_stop and official_success)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scene_id": scene_id,
        "episode_id": episode_id,
        "first_task": _jsonable(episode.tasks[0]),
        "goal_image_index": int(image_index),
        "goal_image_shape": list(goal.shape),
        "rgb_shape": list(rgb.shape),
        "rgb_hfov_deg": float(rgb_hfov_deg),
        "camera_height_m": float(camera_height_m),
        "camera_intrinsic": intrinsic.tolist(),
        "service_reset_seed_uint32": int(episode_seed),
        "agent_pose_unchanged_by_goal_render": True,
        "temporary_sensor_removed": True,
        "navdp_reset": navdp_reset,
        "arrival_reset": arrival_reset,
        "navigation_actions": navigation_actions,
        "environment_actions_including_stop": environment_actions,
        "plan_count": len(plan_records),
        "zero_proposal_count": zero_proposal_count,
        "same_batch_fallback_count": same_batch_fallback_count,
        "extra_resample_count": extra_resample_count,
        "certified_stop": certified_stop,
        "certified_success": certified_success,
        "safe_stall": safe_stall,
        "forced_guard_stop": forced_guard_stop,
        "episode_over_before_certified_stop": bool(
            env.episode_over and not certified_stop),
        "first_zero_distance_m": first_zero_distance_m,
        "legacy_first_zero_success_counterfactual": (
            first_zero_legacy_success),
        "official_first_subtask_success_diagnostic": official_success,
        "official_first_subtask_spl_diagnostic": _metric_value(
            metrics_after, "spl", "spl_by_subtask", 0),
        "distance_state_before_final_stop": _distance_to_target(
            stop_metrics_before) if stop_metrics_before else None,
        "metrics_after_transition": metrics_after,
        "wall_time_s": float(time.monotonic() - started),
        "plans": plan_records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import requests
    from habitat import Env
    from habitat.config import read_write
    from habitat.datasets import make_dataset

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    episodes = manifest["episodes"]
    if not 0 <= args.index < len(episodes):
        raise RuntimeError("episode index outside frozen manifest")
    expected = episodes[args.index]
    requested = [(str(expected["scene_id"]), str(expected["episode_id"]))]

    goat_code = args.goat_code.resolve()
    data_root = args.data_root.resolve()
    actual_commit = subprocess.run(
        ["git", "-C", str(goat_code), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    if actual_commit != EXPECTED_GOAT_COMMIT:
        raise RuntimeError("GOAT source commit changed")

    config = _build_config(
        goat_code,
        data_root,
        [requested[0][0]],
        args.gpu_device_id,
    )
    with read_write(config):
        config.habitat.environment.max_episode_steps = max(
            5000, int(manifest["max_navigation_actions"]) + 1)
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    selected = _select_episodes(dataset.episodes, requested)
    dataset.episodes = selected

    base_seed = int(manifest["base_seed"])
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    rgb_hfov = float(sensor.hfov)
    camera_height = float(sensor.position[1])
    if not math.isfinite(camera_height) or camera_height <= 0.0:
        raise RuntimeError("GOAT RGB camera height is invalid")

    with requests.Session() as session:
        with Env(config=config, dataset=dataset) as env:
            env.seed(base_seed)
            record = _run_episode(
                env,
                selected[0],
                manifest,
                session,
                args.navdp_url,
                args.arrival_url,
                rgb_hfov,
                camera_height,
                args.request_timeout_s,
            )

    output = {
        "schema_version": "goat_certified_arrival_task_v1_20260815",
        "complete": True,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "episode_index": int(args.index),
        "arrival_contract": contract_receipt(),
        "goat_commit": actual_commit,
        "ground_truth_used_by_decision": False,
        "is_full_goat_benchmark_score": False,
        "record": record,
    }
    output_path = args.output_dir.resolve() / f"episode_{args.index:02d}.json"
    if output_path.exists():
        raise RuntimeError("refusing to overwrite existing episode output")
    _atomic_json(output_path, output)
    print(json.dumps({
        "output": str(output_path),
        "scene_id": record["scene_id"],
        "episode_id": record["episode_id"],
        "certified_stop": record["certified_stop"],
        "certified_success": record["certified_success"],
        "zero_proposal_count": record["zero_proposal_count"],
        "safe_stall": record["safe_stall"],
    }, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goat-code", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--navdp-url", required=True)
    parser.add_argument("--arrival-url", required=True)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
