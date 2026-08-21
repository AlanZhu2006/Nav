#!/usr/bin/env python3
"""Outcome-blind GOAT-to-NavDP runtime pilot on ten frozen ImageGoal starts.

This runner is deliberately narrower than an official GOAT evaluation.  It
loads one official episode from each of ten scenes, executes only its first
ImageGoal subtask with frozen native NavDP, and records runtime/contract
diagnostics.  It does not provide a controller for ObjectGoal or LanguageGoal
and therefore must never be reported as a GOAT SR/SPL score.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
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
except ModuleNotFoundError:
    # Immutable bundles invoke this file by absolute path and put MemNavData,
    # rather than the bundle root, on sys.path.
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


SCHEMA_VERSION = "goat_navdp_runtime_pilot_v1_20260814"
MANIFEST_SCHEMA = "goat_navdp_runtime_pilot_manifest_v1_20260814"
# Frozen NavDP critic-score fallback threshold from its released client/server
# contract.  This is dimensionless and must never be confused with the
# discrete adapter's metric endpoint radius.
NAVDP_UPSTREAM_CRITIC_THRESHOLD = -0.5
ACTION_NAMES = {
    GoatNavAction.SUBTASK_STOP: "subtask_stop",
    GoatNavAction.MOVE_FORWARD: "move_forward",
    GoatNavAction.TURN_LEFT: "turn_left",
    GoatNavAction.TURN_RIGHT: "turn_right",
}


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan_seed(base_seed: int, scene_id: str, episode_id: str, plan: int) -> int:
    material = "{}|{}|{}|{}|goat-navdp-runtime-v1".format(
        int(base_seed), scene_id, episode_id, int(plan)
    ).encode("utf-8")
    # Torch accepts signed-int64-range seeds.  Hashing makes each request
    # independent of previous episode lengths while preserving reproducibility.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("RGB dimensions must be positive")
    if not 0.0 < float(hfov_deg) < 180.0:
        raise ValueError("horizontal FOV must lie in (0, 180)")
    focal = (float(width) / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.asarray(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _normalize_trajectory(raw: Any) -> np.ndarray:
    trajectory = np.asarray(raw, dtype=np.float64)
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("unexpected NavDP trajectory shape {}".format(
            trajectory.shape
        ))
    if trajectory.shape[0] == 0 or not np.isfinite(trajectory).all():
        raise ValueError("NavDP trajectory must be non-empty and finite")
    return trajectory


def _validated_critic_receipt(payload: Mapping[str, Any]) -> Dict[str, Any]:
    values = np.asarray(payload.get("all_values"), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("NavDP returned invalid critic scores")
    critic_max = float(values.max())
    critic_min = float(values.min())
    try:
        echoed_max = float(payload["critic_max"])
        echoed_min = float(payload["critic_min"])
        threshold = float(payload["critic_threshold"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise RuntimeError("NavDP omitted its critic fallback receipt")
    if (not math.isclose(echoed_max, critic_max, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(
                echoed_min, critic_min, rel_tol=0.0, abs_tol=1e-6)):
        raise RuntimeError("NavDP critic receipt does not match returned scores")
    expected_fallback = bool(critic_max < threshold)
    if payload.get("critic_fallback_applied") is not expected_fallback:
        raise RuntimeError("NavDP critic fallback receipt is inconsistent")
    return {
        "critic_max": critic_max,
        "critic_min": critic_min,
        "critic_threshold": threshold,
        "critic_fallback_applied": expected_fallback,
    }


def _rgb_jpeg_bytes(rgb: Any) -> bytes:
    from PIL import Image

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("unexpected RGB shape {}".format(array.shape))
    array = np.asarray(array[..., :3], dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def _navdp_wire_jpeg_bytes(rgb: Any) -> bytes:
    """Encode RGB exactly as the frozen NavDP client/server contract expects.

    The upstream client passes RGB arrays directly to ``cv2.imencode`` while
    the server decodes with PIL and applies ``RGB2BGR``.  Those two channel
    swaps cancel.  A conventional PIL RGB JPEG does not: the server would feed
    BGR into a policy trained on RGB.  Keep the conventional encoder above for
    MemNav endpoints and use this encoder only for NavDP endpoints.
    """
    import cv2

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("unexpected RGB shape {}".format(array.shape))
    array = np.asarray(array[..., :3], dtype=np.uint8)
    encoded, payload = cv2.imencode(
        ".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not encoded:
        raise RuntimeError("OpenCV failed to encode NavDP RGB input")
    return payload.tobytes()


def _depth_png_bytes(depth: Any) -> bytes:
    from PIL import Image

    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("metric depth must be a finite HxW image")
    encoded = np.clip(array * 10000.0, 0, 65535).astype(np.uint16)
    stream = io.BytesIO()
    Image.fromarray(encoded).save(stream, format="PNG")
    return stream.getvalue()


def _load_manifest(path: pathlib.Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unexpected runtime-pilot manifest schema")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise RuntimeError("runtime pilot requires exactly ten episode entries")
    requested: List[Tuple[str, str]] = []
    for item in episodes:
        if not isinstance(item, Mapping):
            raise RuntimeError("manifest episode entry is not an object")
        scene_id = str(item.get("scene_id", ""))
        episode_id = str(item.get("episode_id", ""))
        if not scene_id or not episode_id:
            raise RuntimeError("manifest episode key is empty")
        requested.append((scene_id, episode_id))
    if len({scene for scene, _ in requested}) != 10:
        raise RuntimeError("runtime pilot must cover ten unique scenes")
    if int(payload.get("base_seed", -1)) < 0:
        raise RuntimeError("manifest base_seed must be non-negative")
    if int(payload.get("max_navigation_actions", 0)) <= 0:
        raise RuntimeError("manifest navigation-action guard must be positive")
    return payload


def _requested_episodes(manifest: Mapping[str, Any]) -> List[Tuple[str, str]]:
    return [
        (str(item["scene_id"]), str(item["episode_id"]))
        for item in manifest["episodes"]
    ]


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _navdp_reset(
    session: Any,
    navdp_url: str,
    intrinsic: np.ndarray,
    seed: int,
    stop_threshold: float,
    timeout_s: float,
) -> Dict[str, Any]:
    response = session.post(
        navdp_url.rstrip("/") + "/navigator_reset",
        json={
            "intrinsic": intrinsic.tolist(),
            "stop_threshold": float(stop_threshold),
            "batch_size": 1,
            "seed": int(seed),
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("algo") != "navdp":
        raise RuntimeError("unexpected NavDP reset receipt: {}".format(payload))
    if payload.get("threshold_semantics") != "critic_score_fallback":
        raise RuntimeError(
            "NavDP server did not confirm critic-threshold semantics")
    try:
        echoed_threshold = float(payload["stop_threshold"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise RuntimeError("NavDP server did not echo its critic threshold")
    if not math.isclose(
            echoed_threshold, float(stop_threshold),
            rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("NavDP server changed the requested critic threshold")
    checkpoint_contract = payload.get("checkpoint_contract", {})
    if (checkpoint_contract.get("exact_state_dict") is not True
            or int(checkpoint_contract.get("temporal_depth", -1)) != 16):
        raise RuntimeError("NavDP checkpoint/model contract is not exact")
    return payload


def _navdp_plan(
    session: Any,
    navdp_url: str,
    rgb: Any,
    depth: Any,
    goal: Any,
    diffusion_seed: int,
    timeout_s: float,
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    started = time.monotonic()
    response = session.post(
        navdp_url.rstrip("/") + "/imagegoal_step",
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
    latency_s = time.monotonic() - started
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("diffusion_seed", -1)) != int(diffusion_seed):
        raise RuntimeError("NavDP did not echo the deterministic request seed")
    _validated_critic_receipt(payload)
    trajectory = _normalize_trajectory(payload.get("trajectory"))
    return trajectory, payload, latency_s


def _metric_value(metrics: Mapping[str, Any], *path: Any) -> Any:
    value: Any = metrics
    for key in path:
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)) or key >= len(value):
                return None
            value = value[key]
        else:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
    return _jsonable(value)


def _run_first_image_subtask(
    env: Any,
    episode: Any,
    manifest: Mapping[str, Any],
    session: Any,
    navdp_url: str,
    rgb_hfov_deg: float,
    request_timeout_s: float,
) -> Dict[str, Any]:
    scene_id = _episode_scene_id(episode)
    episode_id = str(episode.episode_id)
    base_seed = int(manifest["base_seed"])
    max_actions = int(manifest["max_navigation_actions"])
    stop_threshold = float(manifest["navdp_stop_threshold"])
    adapter_cfg = DiscreteAdapterConfig(
        forward_step_m=float(manifest["adapter"]["forward_step_m"]),
        turn_angle_deg=float(manifest["adapter"]["turn_angle_deg"]),
        endpoint_stop_radius_m=float(
            manifest["adapter"]["endpoint_stop_radius_m"]
        ),
        lookahead_points=int(manifest["adapter"]["lookahead_points"]),
        execution_horizon=int(manifest["adapter"]["execution_horizon"]),
    )

    env.current_episode = episode
    observations = env.reset()
    if env.task.active_subtask_idx != 0:
        raise RuntimeError("GOAT task did not reset to subtask zero")
    if not episode.tasks or episode.tasks[0][1] != "image":
        raise RuntimeError("runtime-pilot episode does not start with ImageGoal")
    if "rgb" not in observations or "depth" not in observations:
        raise RuntimeError("NavDP RGB-D observation is absent")

    rgb = np.asarray(observations["rgb"])
    depth = np.asarray(observations["depth"])
    intrinsic = _camera_intrinsic(rgb.shape[0], rgb.shape[1], rgb_hfov_deg)
    parameters, image_index = _current_image_parameters(episode, 0)
    state_before = env.sim.get_agent_state()
    sensors_before = set(env.sim._sensors)
    goal = np.asarray(_render_raw_goal(env.sim, parameters))
    _assert_same_pose(state_before, env.sim.get_agent_state())
    if set(env.sim._sensors) != sensors_before:
        raise RuntimeError("temporary ImageGoal rendering sensor leaked")

    episode_seed = _plan_seed(base_seed, scene_id, episode_id, -1)
    reset_started = time.monotonic()
    _navdp_reset(
        session,
        navdp_url,
        intrinsic,
        episode_seed,
        stop_threshold,
        request_timeout_s,
    )
    reset_latency_s = time.monotonic() - reset_started

    started = time.monotonic()
    navigation_actions = 0
    environment_actions = 0
    plan_records: List[Dict[str, Any]] = []
    autonomous_stop = False
    forced_guard_stop = False
    safe_no_motion_abort = False
    same_batch_fallback_count = 0
    episode_over_before_stop = False
    stop_metrics_before: Dict[str, Any] = {}

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
        primary_decision = navdp_waypoints_to_goat_decision(
            trajectory, adapter_cfg)
        decision = primary_decision
        if not decision.is_motion:
            decision = best_scored_motion_candidate(
                server_payload.get("all_trajectory"),
                server_payload.get("all_values"),
                adapter_cfg,
            )
            if decision.is_motion:
                same_batch_fallback_count += 1
        actions = decision.actions
        values = np.asarray(server_payload.get("all_values", []), dtype=float)
        finite_values = values[np.isfinite(values)]
        record = {
            "plan_index": plan_index,
            "diffusion_seed": request_seed,
            "request_latency_s": latency_s,
            "trajectory_endpoint_xy_m": trajectory[-1, :2].tolist(),
            "trajectory_endpoint_norm_m": float(np.linalg.norm(
                trajectory[-1, :2]
            )),
            "primary_disposition": primary_decision.disposition.value,
            "executed_disposition": decision.disposition.value,
            "executed_candidate_index": decision.candidate_index,
            "adapter_reason": decision.reason,
            "action_ids": [int(action) for action in actions],
            "action_names": [ACTION_NAMES[action] for action in actions],
            "critic_max": (
                float(finite_values.max()) if finite_values.size else None
            ),
        }
        plan_records.append(record)

        if not decision.is_motion:
            # This historical runtime pilot has no arrival verifier.  It must
            # fail closed rather than restoring the old zero-trajectory STOP.
            safe_no_motion_abort = True
            break

        for action in actions:
            if action == GoatNavAction.SUBTASK_STOP:
                raise RuntimeError("SUBTASK_STOP appeared inside a motion chunk")
            if navigation_actions >= max_actions or env.episode_over:
                break
            observations = env.step(ACTION_NAMES[action])
            navigation_actions += 1
            environment_actions += 1

    episode_over_before_stop = bool(env.episode_over and not autonomous_stop)
    if not autonomous_stop and not env.episode_over:
        # This is a runtime guard, not a policy success decision.  The forced
        # transition makes official measurement state auditable while the
        # receipt keeps it distinct from an autonomous NavDP stop.
        stop_metrics_before = _jsonable(env.get_metrics())
        observations = env.step(ACTION_NAMES[GoatNavAction.SUBTASK_STOP])
        environment_actions += 1
        forced_guard_stop = True

    metrics_after = _jsonable(env.get_metrics())
    elapsed_s = time.monotonic() - started
    plan_latencies = [float(item["request_latency_s"]) for item in plan_records]
    return {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "first_task": _jsonable(episode.tasks[0]),
        "goal_image_index": int(image_index),
        "goal_image_shape": list(goal.shape),
        "goal_hfov_deg": float(parameters.hfov),
        "rgb_shape": list(rgb.shape),
        "rgb_hfov_deg": float(rgb_hfov_deg),
        "depth_shape": list(depth.shape),
        "camera_intrinsic": intrinsic.tolist(),
        "agent_pose_unchanged_by_goal_render": True,
        "temporary_sensor_removed": True,
        "navdp_reset_latency_s": reset_latency_s,
        "navigation_actions": navigation_actions,
        "environment_actions_including_stop": environment_actions,
        "plan_count": len(plan_records),
        "plan_latency_mean_s": (
            float(np.mean(plan_latencies)) if plan_latencies else None
        ),
        "plan_latency_p95_s": (
            float(np.percentile(plan_latencies, 95)) if plan_latencies else None
        ),
        "wall_time_s": elapsed_s,
        "autonomous_navdp_stop": autonomous_stop,
        "safe_no_motion_abort": safe_no_motion_abort,
        "same_batch_fallback_count": same_batch_fallback_count,
        "forced_guard_stop": forced_guard_stop,
        "episode_over_before_stop": episode_over_before_stop,
        "active_subtask_after": int(env.task.active_subtask_idx),
        "official_first_subtask_success_diagnostic": _metric_value(
            metrics_after, "success", "subtask_success", 0
        ),
        "official_first_subtask_spl_diagnostic": _metric_value(
            metrics_after, "spl", "spl_by_subtask", 0
        ),
        "distance_state_before_stop": _metric_value(
            stop_metrics_before, "distance_to_goal"
        ),
        "metrics_after_transition": metrics_after,
        "plans": plan_records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import requests
    from habitat import Env
    from habitat.config import read_write
    from habitat.datasets import make_dataset

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    requested = _requested_episodes(manifest)
    goat_code = args.goat_code.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

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
        [scene for scene, _ in requested],
        args.gpu_device_id,
    )
    with read_write(config):
        # Only the frozen first-subtask guard controls this pilot.  Keep the
        # official environment ceiling high enough not to terminate first.
        config.habitat.environment.max_episode_steps = max(
            5000, int(manifest["max_navigation_actions"]) + 1
        )
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    selected = _select_episodes(dataset.episodes, requested)
    dataset.episodes = selected

    base_seed = int(manifest["base_seed"])
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    rgb_hfov = float(
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov
    )
    session = requests.Session()
    records: List[Dict[str, Any]] = []
    total_started = time.monotonic()
    with Env(config=config.habitat, dataset=dataset) as env:
        env.seed(base_seed)
        for episode in selected:
            record = _run_first_image_subtask(
                env,
                episode,
                manifest,
                session,
                args.navdp_url,
                rgb_hfov,
                args.request_timeout_s,
            )
            record_path = records_dir / "{}_{}.json".format(
                record["scene_id"], record["episode_id"]
            )
            _atomic_json(record_path, record)
            records.append(record)
            print(json.dumps({
                "scene_id": record["scene_id"],
                "episode_id": record["episode_id"],
                "plan_count": record["plan_count"],
                "navigation_actions": record["navigation_actions"],
                "autonomous_navdp_stop": record["autonomous_navdp_stop"],
                "forced_guard_stop": record["forced_guard_stop"],
                "wall_time_s": record["wall_time_s"],
            }, sort_keys=True), flush=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": True,
        "is_goat_navigation_score": False,
        "paper_metric_use_prohibited": True,
        "purpose": "ten_scene_first_imagegoal_runtime_and_contract_gate_only",
        "reason_not_a_score": (
            "only the first ImageGoal subtask is executed; ObjectGoal and "
            "LanguageGoal controllers and complete sequential episodes are absent"
        ),
        "method_or_threshold_selection_allowed": False,
        "goat_commit": actual_commit,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "navdp_url": args.navdp_url,
        "scene_count": len({record["scene_id"] for record in records}),
        "episode_count": len(records),
        "total_wall_time_s": time.monotonic() - total_started,
        "all_records_complete": len(records) == 10,
        "all_goal_render_contracts_passed": all(
            record["agent_pose_unchanged_by_goal_render"]
            and record["temporary_sensor_removed"]
            for record in records
        ),
        "autonomous_stop_count_diagnostic": sum(
            bool(record["autonomous_navdp_stop"]) for record in records
        ),
        "forced_guard_stop_count_diagnostic": sum(
            bool(record["forced_guard_stop"]) for record in records
        ),
        "records": records,
    }
    _atomic_json(output_dir / "goat_navdp_runtime_pilot.json", payload)
    print(json.dumps({
        "complete": payload["complete"],
        "is_goat_navigation_score": payload["is_goat_navigation_score"],
        "scene_count": payload["scene_count"],
        "episode_count": payload["episode_count"],
        "total_wall_time_s": payload["total_wall_time_s"],
    }, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goat-code", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--navdp-url", required=True)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
