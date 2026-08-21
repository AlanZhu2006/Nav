#!/usr/bin/env python3
"""Autonomous GOAT multi-goal pilot for frozen NavDP plus CEC.

The released GOAT policy remains responsible for ObjectGoal and LanguageGoal
subtasks and is the exact fallback for an uncertified ImageGoal.  CEC may
authorize a role-free Revisit takeover:

* CEC may retrieve and geometrically certify a causal historical observation;
* an accepted certificate supplies only a scale-free bearing to frozen NavDP;
* rejection falls back to the official GOAT action on that same observation;
* a NavDP no-motion output is only an arrival proposal;
* a bounded terminal-view search queries current-RGB-to-goal geometry at every
  view and is the only path to ``SUBTASK_STOP``.

The optional ``navdp`` fallback mode is retained only as a transfer diagnostic.
The runner never reads the GOAT distance/success metric before selecting an
action.  Metrics are copied only after ``env.step`` for retrospective audit.
It is an engineering pilot, not an official GOAT score or a frozen paper
confirmation.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from MemNavData.goat_autonomous_stop import (
        AutonomousVisualStopSearch,
        SearchDecision,
        SearchDisposition,
        arrival_evidence_from_payload,
    )
    from MemNavData.goat_certified_arrival_confirmation import (
        _navdp_resample,
        _service_reset_seed,
    )
    from MemNavData.goat_contract_smoke import (
        EXPECTED_GOAT_COMMIT,
        _episode_scene_id,
        _jsonable,
    )
    from MemNavData.goat_navdp_discrete_adapter import (
        DiscreteAdapterConfig,
        NavDPAdapterDecision,
        NavDPAdapterDisposition,
        best_scored_motion_candidate,
        navdp_waypoints_to_goat_decision,
    )
    from MemNavData.goat_navdp_camera_adapter import (
        NAVDP_CAMERA_HEIGHT,
        NAVDP_CAMERA_HFOV_DEG,
        NAVDP_CAMERA_WIDTH,
        NAVDP_RGB_SENSOR_TYPE,
        NAVDP_RGB_SENSOR_UUID,
        canonical_navdp_intrinsic,
        reproject_goal_to_navdp_camera,
    )
    from MemNavData.goat_navdp_runtime_pilot import (
        NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        _depth_png_bytes,
        _navdp_plan,
        _navdp_reset,
        _navdp_wire_jpeg_bytes,
        _normalize_trajectory,
        _plan_seed,
        _rgb_jpeg_bytes,
        _validated_critic_receipt,
    )
    from MemNavData.goat_sequential_revisit_pilot import (
        ACTION_IDS,
        ACTION_NAMES,
        _append_memory,
        _build_official_policy,
        _goal_assets,
        _official_action,
        _policy_config,
        _probe_certificate,
        _success_for,
    )
    from MemNavData.revisit_bearing_adapter import adapt_revisit_pointgoal
    from MemNavData.xnavdp_revisit_contract import pointgoal_payload
except ModuleNotFoundError:  # immutable/direct-script invocation
    from goat_autonomous_stop import (  # type: ignore
        AutonomousVisualStopSearch,
        SearchDecision,
        SearchDisposition,
        arrival_evidence_from_payload,
    )
    from goat_certified_arrival_confirmation import (  # type: ignore
        _navdp_resample,
        _service_reset_seed,
    )
    from goat_contract_smoke import (  # type: ignore
        EXPECTED_GOAT_COMMIT,
        _episode_scene_id,
        _jsonable,
    )
    from goat_navdp_discrete_adapter import (  # type: ignore
        DiscreteAdapterConfig,
        NavDPAdapterDecision,
        NavDPAdapterDisposition,
        best_scored_motion_candidate,
        navdp_waypoints_to_goat_decision,
    )
    from goat_navdp_camera_adapter import (  # type: ignore
        NAVDP_CAMERA_HEIGHT,
        NAVDP_CAMERA_HFOV_DEG,
        NAVDP_CAMERA_WIDTH,
        NAVDP_RGB_SENSOR_TYPE,
        NAVDP_RGB_SENSOR_UUID,
        canonical_navdp_intrinsic,
        reproject_goal_to_navdp_camera,
    )
    from goat_navdp_runtime_pilot import (  # type: ignore
        NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        _depth_png_bytes,
        _navdp_plan,
        _navdp_reset,
        _navdp_wire_jpeg_bytes,
        _normalize_trajectory,
        _plan_seed,
        _rgb_jpeg_bytes,
        _validated_critic_receipt,
    )
    from goat_sequential_revisit_pilot import (  # type: ignore
        ACTION_IDS,
        ACTION_NAMES,
        _append_memory,
        _build_official_policy,
        _goal_assets,
        _official_action,
        _policy_config,
        _probe_certificate,
        _success_for,
    )
    from revisit_bearing_adapter import adapt_revisit_pointgoal  # type: ignore
    from xnavdp_revisit_contract import pointgoal_payload  # type: ignore


SCHEMA_VERSION = "goat_autonomous_multigoal_pilot_v2_20260818"
DEFAULT_EPISODES = (("5cdEh9F2hJL", "1"), ("4ok3usBNeis", "9"))
IMAGE_MODALITY = "image"
GOAT_ADAPTER_CONFIG = DiscreteAdapterConfig(
    lookahead_distance_m=0.70,
    execution_horizon=1,
)
NAVDP_COLLISION_TRANSLATION_EPS_M = 0.05
NAVDP_COLLISION_RECOVERY_TURNS = 3


def _install_navdp_camera(config: Any) -> Dict[str, Any]:
    """Add a NavDP-shaped RGB-D pair without altering GOAT's policy RGB."""
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        HabitatSimDepthSensorConfig,
        HabitatSimRGBSensorConfig,
    )
    from habitat.core.registry import registry
    from habitat.sims.habitat_simulator.habitat_simulator import (
        HabitatSimRGBSensor,
    )

    if registry.get_sensor(NAVDP_RGB_SENSOR_TYPE) is None:
        class NavDPCanonicalRGBSensor(HabitatSimRGBSensor):
            def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
                del args, kwargs
                return NAVDP_RGB_SENSOR_UUID

        registry.register_sensor(
            NavDPCanonicalRGBSensor, name=NAVDP_RGB_SENSOR_TYPE)

    with read_write(config):
        agent = config.habitat.simulator.agents.main_agent
        official_rgb = agent.sim_sensors.rgb_sensor
        position = list(official_rgb.position)
        orientation = list(official_rgb.orientation)
        agent.sim_sensors.update({
            "navdp_rgb_sensor": HabitatSimRGBSensorConfig(
                type=NAVDP_RGB_SENSOR_TYPE,
                height=NAVDP_CAMERA_HEIGHT,
                width=NAVDP_CAMERA_WIDTH,
                position=position,
                orientation=orientation,
                hfov=int(round(NAVDP_CAMERA_HFOV_DEG)),
            ),
            "depth_sensor": HabitatSimDepthSensorConfig(
                height=NAVDP_CAMERA_HEIGHT,
                width=NAVDP_CAMERA_WIDTH,
                position=position,
                orientation=orientation,
                hfov=int(round(NAVDP_CAMERA_HFOV_DEG)),
                min_depth=0.0,
                max_depth=10.0,
                normalize_depth=False,
            ),
        })
    return {
        "adapter": "dedicated_navdp_rgbd_sensor",
        "observation_uuid": NAVDP_RGB_SENSOR_UUID,
        "width": NAVDP_CAMERA_WIDTH,
        "height": NAVDP_CAMERA_HEIGHT,
        "hfov_deg": NAVDP_CAMERA_HFOV_DEG,
        "intrinsic": canonical_navdp_intrinsic().tolist(),
        "position": position,
        "orientation": orientation,
        "official_goat_rgb_unchanged": True,
    }


def _official_observation(observation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in observation.items()
        if key != NAVDP_RGB_SENSOR_UUID
    }


def _navdp_observation(
    observation: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if NAVDP_RGB_SENSOR_UUID not in observation:
        raise RuntimeError("dedicated NavDP RGB observation is absent")
    if "depth" not in observation:
        raise RuntimeError("dedicated NavDP depth observation is absent")
    rgb = np.asarray(observation[NAVDP_RGB_SENSOR_UUID])
    depth = np.asarray(observation["depth"])
    expected_hw = (NAVDP_CAMERA_HEIGHT, NAVDP_CAMERA_WIDTH)
    if rgb.shape != expected_hw + (3,):
        raise RuntimeError(
            "unexpected NavDP RGB shape {}".format(rgb.shape))
    if depth.shape not in (expected_hw, expected_hw + (1,)):
        raise RuntimeError(
            "unexpected NavDP depth shape {}".format(depth.shape))
    if not np.isfinite(depth).all():
        raise RuntimeError("NavDP depth contains non-finite values")
    return rgb, depth


def _depth_clearance_recovery_turn(
    depth: Any,
) -> Tuple[int, Dict[str, Any]]:
    """Choose a deterministic 90-degree escape from observable depth only."""
    image = np.asarray(depth, dtype=np.float64)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2 or image.shape[1] < 2:
        raise ValueError("collision recovery depth must be HxW")
    midpoint = image.shape[1] // 2

    def clearance(values: np.ndarray) -> Tuple[float, int]:
        valid = values[np.isfinite(values) & (values >= 0.1)]
        if not len(valid):
            return 0.0, 0
        return float(np.percentile(valid, 75.0)), int(len(valid))

    left_score, left_valid = clearance(image[:, :midpoint])
    right_score, right_valid = clearance(image[:, midpoint:])
    turn = (
        ACTION_IDS["turn_left"]
        if left_score >= right_score else ACTION_IDS["turn_right"])
    return int(turn), {
        "policy": "observed_depth_p75_sticky_90deg",
        "left_clearance_m": left_score,
        "right_clearance_m": right_score,
        "left_valid_pixels": left_valid,
        "right_valid_pixels": right_valid,
        "turn_action": ACTION_NAMES[int(turn)],
        "turn_count": NAVDP_COLLISION_RECOVERY_TURNS,
        "forward_probe_after_turns": True,
    }


def _sticky_collision_recovery(
    depth: Any,
    previous_receipt: Optional[Mapping[str, Any]],
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    if previous_receipt is None:
        turn, schedule = _depth_clearance_recovery_turn(depth)
        return turn, schedule, dict(schedule)
    action_name = str(previous_receipt.get("turn_action"))
    if action_name not in ("turn_left", "turn_right"):
        raise RuntimeError("invalid sticky collision-recovery receipt")
    schedule = dict(previous_receipt)
    schedule["sticky_direction_reused"] = True
    return ACTION_IDS[action_name], schedule, dict(previous_receipt)


def _collision_recovery_action_sequence(turn: int) -> List[int]:
    if int(turn) not in (
            ACTION_IDS["turn_left"], ACTION_IDS["turn_right"]):
        raise ValueError("collision recovery requires a turn action")
    return (
        [int(turn)] * NAVDP_COLLISION_RECOVERY_TURNS
        + [ACTION_IDS["move_forward"]]
    )


def _queues_after_navdp_collision(
    queued_actions: Sequence[int],
    terminal_search: Any,
    terminal_fallback: Sequence[int],
    terminal_origin: Optional[Dict[str, Any]],
) -> Tuple[List[int], Any, List[int], Optional[Dict[str, Any]]]:
    """Drop stale NavDP chunks after a collision; keep an in-flight search.

    Recovery is scheduled on the next step and runs before the search loop.
    Aborting ``terminal_search`` here would discard a still-valid arrival
    proposal and force a new DDPM sample.
    """
    del queued_actions
    return [], terminal_search, list(terminal_fallback), terminal_origin


def _critic_fallback_action_sequence(
    decision: NavDPAdapterDecision,
) -> List[int]:
    """Realize NavDP's lateral fallback before drawing another DDPM sample."""
    if not decision.is_motion or not decision.actions:
        return []
    turn = int(decision.actions[0])
    if turn not in (ACTION_IDS["turn_left"], ACTION_IDS["turn_right"]):
        return []
    return _collision_recovery_action_sequence(turn)


def _is_navdp_motion_source(action_source: str) -> bool:
    return (
        "navdp" in action_source
        or action_source in {
            "terminal_rejected_same_batch_fallback",
            "terminal_rejected_resampled_motion",
        }
    )


def _adapted_goal_assets(
    env: Any,
    episode: Any,
    subtask: int,
) -> Tuple[np.ndarray, np.ndarray, int, float, Dict[str, Any]]:
    raw_goal, source_intrinsic, image_index, source_hfov = _goal_assets(
        env, episode, subtask)
    goal, receipt = reproject_goal_to_navdp_camera(
        raw_goal, source_intrinsic)
    receipt["source_hfov_deg"] = float(source_hfov)
    receipt["image_index"] = int(image_index)
    return (
        goal,
        canonical_navdp_intrinsic(),
        int(image_index),
        float(source_hfov),
        receipt,
    )


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_episode(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("episode must be SCENE:EPISODE")
    scene, episode = value.split(":", 1)
    if not scene or not episode:
        raise argparse.ArgumentTypeError("episode must be SCENE:EPISODE")
    return scene, episode


def _select_episodes(
    episodes: Iterable[Any], requested: Sequence[Tuple[str, str]],
) -> List[Any]:
    lookup = {
        (_episode_scene_id(episode), str(episode.episode_id)): episode
        for episode in episodes
    }
    missing = [identity for identity in requested if identity not in lookup]
    if missing:
        raise RuntimeError("requested GOAT episodes are absent: {}".format(
            missing))
    return [lookup[identity] for identity in requested]


def _is_image_task(task: Sequence[Any]) -> bool:
    return len(task) >= 2 and str(task[1]) == IMAGE_MODALITY


def _finite_optional(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _terminal_hints(certificate: Optional[Mapping[str, Any]]) -> Tuple[
        Optional[float], Optional[float]]:
    """Return view alignment only from an accepted geometric certificate."""

    if not isinstance(certificate, Mapping):
        return None, None
    if certificate.get("ok") is not True or certificate.get("accepted") is not True:
        return None, None
    return (
        _finite_optional(certificate.get("terminal_yaw_right_deg")),
        _finite_optional(certificate.get("terminal_pitch_up_deg")),
    )


def _terminal_action_id(decision: SearchDecision) -> int:
    if decision.disposition is SearchDisposition.STOP:
        return int(ACTION_IDS["subtask_stop"])
    if decision.disposition is not SearchDisposition.MOTION:
        raise ValueError("terminal decision does not contain an action")
    if decision.action not in ACTION_IDS:
        raise ValueError("unknown terminal-view action {}".format(
            decision.action))
    action = int(ACTION_IDS[decision.action])
    if action in (ACTION_IDS["stop"], ACTION_IDS["subtask_stop"]):
        raise AssertionError("terminal motion mapped to a stop action")
    return action


def _search_decision_json(
    decision: Optional[SearchDecision],
) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    return {
        "disposition": decision.disposition.value,
        "action": decision.action,
        "phase": decision.phase,
        "reason": decision.reason,
        "stop_decision": decision.stop_decision,
    }


def _reset_memnav(
    session: Any,
    memnav_url: str,
    intrinsic: np.ndarray,
    camera_height: float,
    seed: int,
    episode_len: int,
    timeout_s: float,
) -> Dict[str, Any]:
    response = session.post(memnav_url.rstrip("/") + "/navigator_reset", json={
        "camera_height": float(camera_height),
        "camera_intrinsic": intrinsic.tolist(),
        "seed": int(seed),
        "episode_len": int(episode_len),
    }, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if payload.get("algo") != "memnav":
        raise RuntimeError("unexpected MemNav reset receipt")
    if not payload.get("certified_relocalization", {}).get("enabled"):
        raise RuntimeError("CEC sidecar is not certificate-enabled")
    if not payload.get("certified_arrival", {}).get("enabled"):
        raise RuntimeError("arrival sidecar is not certificate-enabled")
    return payload


def _navdp_replay(
    session: Any, navdp_url: str, rgb: Any, timeout_s: float,
) -> Dict[str, Any]:
    response = session.post(
        navdp_url.rstrip("/") + "/memory_replay_step",
        files={"image": (
            "image.jpg", _navdp_wire_jpeg_bytes(rgb), "image/jpeg")},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("algo") != "navdp" or payload.get("diffusion_sampled") is not False:
        raise RuntimeError("NavDP replay contract changed")
    return payload


def _arrival_query_with_intrinsic(
    session: Any,
    arrival_url: str,
    goal: Any,
    goal_intrinsic: np.ndarray,
    *,
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    """Query terminal geometry with the GOAT goal camera calibration.

    GOAT goal images are rendered by a temporary sensor whose HFOV need not
    match the live agent RGB sensor.  Sending the distinct intrinsic is part
    of the observable task interface; no simulator pose or depth is exposed.
    """

    started = time.monotonic()
    response = session.post(
        arrival_url.rstrip("/") + "/arrival_query",
        files={"goal": ("goal.jpg", _rgb_jpeg_bytes(goal), "image/jpeg")},
        data={"goal_camera_intrinsic": json.dumps(
            np.asarray(goal_intrinsic, dtype=np.float64).tolist())},
        timeout=timeout_s,
    )
    latency = time.monotonic() - started
    response.raise_for_status()
    payload = response.json()
    if payload.get("goal_camera_calibration") != "explicit_distinct_intrinsic":
        raise RuntimeError("arrival service ignored GOAT goal intrinsic")
    if payload.get("simulator_depth_consumed") is not False:
        raise RuntimeError("arrival service consumed forbidden simulator depth")
    return payload, latency


def _mixed_plan(
    session: Any,
    navdp_url: str,
    rgb: Any,
    depth: Any,
    goal: Any,
    bearing: Sequence[float],
    seed: int,
    adapter: DiscreteAdapterConfig,
    timeout_s: float,
) -> Tuple[NavDPAdapterDecision, Dict[str, Any], Dict[str, Any], float]:
    adapted = adapt_revisit_pointgoal(
        mode="verified_bearing_v1",
        router_active=True,
        pointgoal=bearing,
        source="goat_cec_pnp",
        pointgoal_units="lingbot_raw_direction_only",
    )
    if not adapted.takeover:
        raise RuntimeError("accepted CEC bearing failed its adapter contract")
    started = time.monotonic()
    response = session.post(
        navdp_url.rstrip("/") + "/navdp_step_ip_mixgoal",
        files={
            "image": (
                "image.jpg", _navdp_wire_jpeg_bytes(rgb), "image/jpeg"),
            "image_goal": (
                "goal.jpg", _navdp_wire_jpeg_bytes(goal), "image/jpeg"),
            "depth": ("depth.png", _depth_png_bytes(depth), "image/png"),
        },
        data={
            "goal_data": json.dumps(pointgoal_payload(
                adapted.controller_pointgoal)),
            "diffusion_seed": str(int(seed)),
        },
        timeout=timeout_s,
    )
    latency_s = time.monotonic() - started
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("diffusion_seed", -1)) != int(seed):
        raise RuntimeError("mixed NavDP did not echo deterministic seed")
    _validated_critic_receipt(payload)
    decision = navdp_waypoints_to_goat_decision(
        _normalize_trajectory(payload.get("trajectory")), adapter)
    return decision, payload, adapted.audit_dict(), latency_s


def _fallback_motion(
    payload: Mapping[str, Any], adapter: DiscreteAdapterConfig,
) -> NavDPAdapterDecision:
    try:
        return best_scored_motion_candidate(
            payload.get("all_trajectory"), payload.get("all_values"), adapter)
    except (TypeError, ValueError, OverflowError):
        return NavDPAdapterDecision(
            disposition=NavDPAdapterDisposition.CONVERSION_STALLED,
            actions=(),
            endpoint_norm_m=0.0,
            reason="same_batch_fallback_payload_invalid",
        )


def _nonstop_official_fallback(official_action: int) -> Optional[int]:
    action = int(official_action)
    if action in (
        ACTION_IDS["move_forward"], ACTION_IDS["turn_left"],
        ACTION_IDS["turn_right"], ACTION_IDS["look_up"],
        ACTION_IDS["look_down"],
    ):
        return action
    return None


def _image_stop_is_authorized(action: int, action_source: str) -> bool:
    """Keep exact GOAT fallback distinct from adapter-authorized STOP."""

    if int(action) not in (
            ACTION_IDS["stop"], ACTION_IDS["subtask_stop"]):
        return True
    return action_source in (
        "autonomous_visual_subtask_stop",
        "official_goat_image_exact_fallback",
    )


def _read_only_resample_motion(
    session: Any,
    navdp_url: str,
    rgb: Any,
    depth: Any,
    goal: Any,
    scene: str,
    episode_id: str,
    base_seed: int,
    plan_index: int,
    adapter: DiscreteAdapterConfig,
    limit: int,
    timeout_s: float,
) -> Tuple[NavDPAdapterDecision, List[Dict[str, Any]]]:
    attempts = []
    stalled = NavDPAdapterDecision(
        disposition=NavDPAdapterDisposition.CONVERSION_STALLED,
        actions=(), endpoint_norm_m=0.0,
        reason="terminal_search_resamples_exhausted")
    for index in range(limit):
        seed = _plan_seed(
            base_seed, scene, episode_id,
            2_000_000 + plan_index * max(1, limit) + index)
        payload, latency = _navdp_resample(
            session, navdp_url, rgb, depth, goal, seed, timeout_s)
        primary = navdp_waypoints_to_goat_decision(
            _normalize_trajectory(payload.get("trajectory")), adapter)
        decision = primary if primary.is_motion else _fallback_motion(
            payload, adapter)
        attempts.append({
            "seed": int(seed),
            "latency_s": float(latency),
            "primary_disposition": primary.disposition.value,
            "selected_disposition": decision.disposition.value,
            "selected_reason": decision.reason,
        })
        if decision.is_motion:
            return decision, attempts
        stalled = decision
    return stalled, attempts


def _run_episode(
    env: Any,
    episode: Any,
    policy: Any,
    transforms: Any,
    session: Any,
    memnav_url: str,
    navdp_url: str,
    current_intrinsic: np.ndarray,
    camera_height: float,
    base_seed: int,
    max_steps: int,
    request_timeout_s: float,
    same_observation_resamples: int,
    unsupported_image_controller: str,
    navdp_stop_threshold: float,
) -> Dict[str, Any]:
    import torch

    scene = _episode_scene_id(episode)
    episode_id = str(episode.episode_id)
    service_seed = _service_reset_seed(_plan_seed(
        base_seed, scene, episode_id, -1))
    random.seed(base_seed)
    np.random.seed(base_seed % (2 ** 32))
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    env.current_episode = episode
    observation = env.reset()
    memnav_reset = _reset_memnav(
        session, memnav_url, current_intrinsic, camera_height,
        service_seed, max_steps + 1, request_timeout_s)

    hidden = torch.zeros(
        1, policy.num_recurrent_layers, int(policy.net.output_size),
        dtype=torch.float32, device=next(policy.parameters()).device)
    previous_action = torch.zeros(
        1, 1, dtype=torch.long, device=hidden.device)
    mask = torch.zeros(1, 1, dtype=torch.bool, device=hidden.device)

    # Eight native NavDP frames cover only about 0.30 m / 36 degrees
    # (0.0376 m and 4.5 degrees per frame).  One GOAT action already covers
    # 0.25 m or 30 degrees, so replan after every discrete action.  The metric
    # lookahead remains the released controller's 0.7 m.
    adapter = GOAT_ADAPTER_CONFIG
    goal_cache: Dict[
        int, Tuple[Any, np.ndarray, int, float, Dict[str, Any]]] = {}
    queued_actions: List[int] = []
    terminal_search: Optional[AutonomousVisualStopSearch] = None
    terminal_fallback: List[int] = []
    terminal_origin: Optional[Dict[str, Any]] = None
    collision_recovery_actions: List[int] = []
    collision_recovery_receipt: Optional[Dict[str, Any]] = None
    active_subtask: Optional[int] = None
    plan_count = 0
    records = []
    image_stop_attempts = 0
    image_stop_successes = 0
    official_image_stop_count = 0
    official_image_stop_successes = 0
    image_subtask_entries = 0
    cec_accept_count = 0
    native_plan_count = 0
    mixed_plan_count = 0
    terminal_search_count = 0
    navdp_collision_count = 0
    safe_stall = False
    navdp_reset_receipts = []

    for step in range(max_steps):
        if env.episode_over or env.task.active_subtask_idx >= len(episode.tasks):
            break
        subtask = int(env.task.active_subtask_idx)
        task = episode.tasks[subtask]
        is_image = _is_image_task(task)
        navdp_rgb, navdp_depth = _navdp_observation(observation)
        if subtask != active_subtask:
            active_subtask = subtask
            queued_actions = []
            terminal_search = None
            terminal_fallback = []
            terminal_origin = None
            collision_recovery_actions = []
            collision_recovery_receipt = None
            if is_image:
                image_subtask_entries += 1
                subtask_seed = _service_reset_seed(_plan_seed(
                    base_seed, scene, episode_id, 100_000 + subtask))
                navdp_reset = _navdp_reset(
                    session, navdp_url, current_intrinsic, subtask_seed,
                    navdp_stop_threshold, request_timeout_s)
                navdp_reset_receipts.append({
                    "subtask": int(subtask),
                    "receipt": navdp_reset,
                })

        official_action, hidden = _official_action(
            policy, transforms, _official_observation(observation), hidden,
            previous_action, mask, hidden.device)
        chosen_action = int(official_action)
        action_source = "official_goat_nonimage"
        frame_append_mode = None
        certificate = None
        plan_audit = None
        arrival_evidence = None
        terminal_decision = None
        terminal_receipt = None
        terminal_origin_record = terminal_origin
        resample_audit = []
        goal_camera_adapter = None
        collision_recovery_record = None

        if not is_image:
            memory_receipt = _append_memory(
                session, memnav_url, navdp_rgb, request_timeout_s)
            frame_append_mode = "memnav_append_nonimage"
        else:
            if subtask not in goal_cache:
                goal_cache[subtask] = _adapted_goal_assets(
                    env, episode, subtask)
            (
                goal,
                goal_intrinsic,
                image_index,
                goal_hfov,
                goal_camera_adapter,
            ) = goal_cache[subtask]
            memory_receipt = None

            if collision_recovery_actions:
                memory_receipt = _append_memory(
                    session, memnav_url, navdp_rgb, request_timeout_s)
                _navdp_replay(
                    session, navdp_url, navdp_rgb, request_timeout_s)
                frame_append_mode = "collision_recovery_replay"
                chosen_action = int(collision_recovery_actions.pop(0))
                action_source = "navdp_collision_recovery"
                collision_recovery_record = dict(
                    collision_recovery_receipt or {})
                collision_recovery_record["turns_remaining_after_action"] = (
                    len(collision_recovery_actions))
            elif terminal_search is not None:
                memory_receipt = _append_memory(
                    session, memnav_url, navdp_rgb, request_timeout_s)
                _navdp_replay(
                    session, navdp_url, navdp_rgb, request_timeout_s)
                frame_append_mode = "terminal_search_replay"
                arrival_evidence, arrival_latency = (
                    _arrival_query_with_intrinsic(
                        session, memnav_url, goal, goal_intrinsic,
                        timeout_s=request_timeout_s))
                terminal_decision = terminal_search.observe(
                    arrival_evidence_from_payload(arrival_evidence))
                terminal_receipt = terminal_search.receipt()
                if terminal_decision.disposition is SearchDisposition.REPLAN:
                    if terminal_fallback:
                        chosen_action = int(terminal_fallback.pop(0))
                        action_source = "terminal_rejected_same_batch_fallback"
                    else:
                        decision, resample_audit = _read_only_resample_motion(
                            session, navdp_url, navdp_rgb,
                            navdp_depth, goal, scene, episode_id,
                            base_seed, plan_count, adapter,
                            same_observation_resamples, request_timeout_s)
                        plan_count += len(resample_audit)
                        if decision.is_motion:
                            queued_actions = [
                                int(value) for value in decision.actions]
                            chosen_action = queued_actions.pop(0)
                            action_source = "terminal_rejected_resampled_motion"
                        else:
                            fallback = _nonstop_official_fallback(
                                official_action)
                            if fallback is None:
                                safe_stall = True
                                records.append({
                                    "step": int(step),
                                    "subtask_before": subtask,
                                    "task": list(task),
                                    "action_source": "safe_stall_no_stop",
                                    "terminal_search": terminal_receipt,
                                    "arrival_evidence": arrival_evidence,
                                    "terminal_decision": _search_decision_json(
                                        terminal_decision),
                                    "arrival_query_latency_s": arrival_latency,
                                    "resample_audit": resample_audit,
                                    "ground_truth_read_before_action": False,
                                })
                                break
                            chosen_action = fallback
                            action_source = "terminal_rejected_official_motion"
                    terminal_search = None
                    terminal_fallback = []
                    terminal_origin = None
                else:
                    chosen_action = _terminal_action_id(terminal_decision)
                    action_source = (
                        "autonomous_visual_subtask_stop"
                        if terminal_decision.disposition is SearchDisposition.STOP
                        else "terminal_view_{}".format(
                            terminal_decision.phase))
            elif queued_actions:
                memory_receipt = _append_memory(
                    session, memnav_url, navdp_rgb, request_timeout_s)
                _navdp_replay(
                    session, navdp_url, navdp_rgb, request_timeout_s)
                frame_append_mode = "queued_motion_replay"
                chosen_action = int(queued_actions.pop(0))
                action_source = "navdp_motion_chunk"
            else:
                certificate = _probe_certificate(
                    session, memnav_url, navdp_rgb, goal,
                    goal_intrinsic, request_timeout_s)
                frame_append_mode = "cec_probe"
                accepted = bool(
                    certificate.get("ok") is True
                    and certificate.get("accepted") is True
                    and certificate.get("pointgoal_units")
                    == "lingbot_raw_direction_only"
                    and certificate.get("aux_pose") is not None)
                cec_accept_count += int(accepted)
                seed = _plan_seed(
                    base_seed, scene, episode_id,
                    subtask * 10_000 + plan_count)
                if accepted:
                    decision, payload, adapter_audit, latency = _mixed_plan(
                        session, navdp_url, navdp_rgb,
                        navdp_depth, goal,
                        certificate["aux_pose"], seed, adapter,
                        request_timeout_s)
                    controller = "cec_bearing_plus_navdp"
                    mixed_plan_count += 1
                elif unsupported_image_controller == "navdp":
                    trajectory, payload, latency = _navdp_plan(
                        session, navdp_url, navdp_rgb,
                        navdp_depth, goal, seed,
                        request_timeout_s)
                    decision = navdp_waypoints_to_goat_decision(
                        trajectory, adapter)
                    adapter_audit = None
                    controller = "native_imagegoal_navdp"
                    native_plan_count += 1
                else:
                    if unsupported_image_controller != "official":
                        raise ValueError("unknown unsupported ImageGoal controller")
                    _navdp_replay(
                        session, navdp_url, navdp_rgb, request_timeout_s)
                    chosen_action = int(official_action)
                    action_source = "official_goat_image_exact_fallback"
                    plan_audit = {
                        "controller": "official_goat_image_exact_fallback",
                        "certificate_rejected": True,
                        "image_index": int(image_index),
                        "goal_hfov": float(goal_hfov),
                        "goal_camera_adapter": goal_camera_adapter,
                    }
                    if chosen_action in (
                            ACTION_IDS["stop"], ACTION_IDS["subtask_stop"]):
                        official_image_stop_count += 1
                    decision = None
                    payload = None
                    adapter_audit = None
                    latency = 0.0
                    controller = "official_goat_image_exact_fallback"
                if decision is None:
                    # Exact fallback has already selected the observable GOAT
                    # action.  It deliberately bypasses NavDP planning, not
                    # the role-free CEC probe that authorized this fallback.
                    pass
                else:
                    plan_count += 1
                    fallback = _fallback_motion(payload, adapter)
                    critic_audit = _validated_critic_receipt(payload)
                    critic_fallback_actions = (
                        _critic_fallback_action_sequence(decision)
                        if critic_audit["critic_fallback_applied"] else [])
                    plan_audit = {
                        "controller": controller,
                        "seed": int(seed),
                        "latency_s": float(latency),
                        "primary_disposition": decision.disposition.value,
                        "primary_reason": decision.reason,
                        "primary_endpoint_norm_m": float(
                            decision.endpoint_norm_m),
                        "primary_max_radius_m": float(
                            decision.max_radius_m),
                        "fallback_disposition": fallback.disposition.value,
                        "fallback_reason": fallback.reason,
                        "critic": critic_audit,
                        "critic_fallback_actions": [
                            ACTION_NAMES[int(value)]
                            for value in critic_fallback_actions
                        ],
                        "adapter": adapter_audit,
                        "image_index": int(image_index),
                        "goal_hfov": float(goal_hfov),
                        "goal_camera_adapter": goal_camera_adapter,
                    }
                    if decision.is_motion:
                        queued_actions = (
                            [int(value) for value in critic_fallback_actions]
                            if critic_fallback_actions else
                            [int(value) for value in decision.actions]
                        )
                        chosen_action = queued_actions.pop(0)
                        action_source = (
                            "{}_critic_fallback_search".format(controller)
                            if critic_fallback_actions else
                            "{}_motion".format(controller)
                        )
                    elif decision.requires_arrival_certificate:
                        terminal_search_count += 1
                        yaw, pitch = _terminal_hints(certificate)
                        terminal_search = AutonomousVisualStopSearch(
                            revisit_yaw_right_deg=yaw,
                            revisit_pitch_up_deg=pitch,
                        )
                        terminal_fallback = (
                            [int(value) for value in fallback.actions]
                            if fallback.is_motion else [])
                        terminal_origin = {
                            "controller": controller,
                            "plan_index": int(plan_count - 1),
                            "directed_yaw_right_deg": yaw,
                            "directed_pitch_up_deg": pitch,
                        }
                        arrival_evidence, arrival_latency = (
                            _arrival_query_with_intrinsic(
                                session, memnav_url, goal, goal_intrinsic,
                                timeout_s=request_timeout_s))
                        terminal_decision = terminal_search.observe(
                            arrival_evidence_from_payload(arrival_evidence))
                        terminal_receipt = terminal_search.receipt()
                        terminal_origin_record = terminal_origin
                        chosen_action = _terminal_action_id(terminal_decision)
                        action_source = (
                            "autonomous_visual_subtask_stop"
                            if terminal_decision.disposition is SearchDisposition.STOP
                            else "terminal_view_{}".format(
                                terminal_decision.phase))
                    elif fallback.is_motion:
                        queued_actions = [int(value) for value in fallback.actions]
                        chosen_action = queued_actions.pop(0)
                        action_source = "{}_same_batch_fallback".format(controller)
                    else:
                        fallback_action = _nonstop_official_fallback(
                            official_action)
                        if fallback_action is None:
                            safe_stall = True
                            records.append({
                                "step": int(step),
                                "subtask_before": subtask,
                                "task": list(task),
                                "action_source": "safe_stall_no_stop",
                                "certificate": certificate,
                                "navdp_plan": plan_audit,
                                "ground_truth_read_before_action": False,
                            })
                            break
                        chosen_action = fallback_action
                        action_source = "{}_official_motion_fallback".format(
                            controller)

            if chosen_action == ACTION_IDS["subtask_stop"]:
                if action_source == "autonomous_visual_subtask_stop":
                    image_stop_attempts += 1
                elif action_source != "official_goat_image_exact_fallback":
                    raise AssertionError(
                        "ImageGoal SUBTASK_STOP bypassed visual authorization")

        if is_image and not _image_stop_is_authorized(
                chosen_action, action_source):
            raise AssertionError("unverified ImageGoal stop escaped fail-closed gate")

        before = env.sim.get_agent_state()
        before_array = np.asarray(before.position, dtype=float)
        position_before = before_array.tolist()
        next_observation = env.step(ACTION_NAMES[int(chosen_action)])
        after = env.sim.get_agent_state()
        after_array = np.asarray(after.position, dtype=float)
        executed_translation_m = float(np.linalg.norm(
            after_array[[0, 2]] - before_array[[0, 2]]))
        navdp_collision_detected = bool(
            is_image
            and chosen_action == ACTION_IDS["move_forward"]
            and _is_navdp_motion_source(action_source)
            and executed_translation_m < NAVDP_COLLISION_TRANSLATION_EPS_M)
        collision_schedule = None
        if navdp_collision_detected:
            _, next_depth = _navdp_observation(next_observation)
            (
                recovery_turn,
                collision_schedule,
                collision_recovery_receipt,
            ) = _sticky_collision_recovery(
                next_depth, collision_recovery_receipt)
            collision_recovery_actions = _collision_recovery_action_sequence(
                recovery_turn)
            (
                queued_actions,
                terminal_search,
                terminal_fallback,
                terminal_origin,
            ) = _queues_after_navdp_collision(
                queued_actions, terminal_search, terminal_fallback,
                terminal_origin)
            navdp_collision_count += 1
        elif (
            is_image
            and chosen_action == ACTION_IDS["move_forward"]
            and _is_navdp_motion_source(action_source)
            and executed_translation_m >= NAVDP_COLLISION_TRANSLATION_EPS_M
        ):
            collision_recovery_receipt = None
        after_subtask = int(env.task.active_subtask_idx)
        # Official metrics are deliberately read only after the action has
        # already been selected and executed.
        metrics = _jsonable(env.get_metrics())
        subtask_success_after = _success_for(metrics, subtask)
        if (is_image and chosen_action == ACTION_IDS["subtask_stop"]
                and subtask_success_after > 0.0):
            if action_source == "autonomous_visual_subtask_stop":
                image_stop_successes += 1
            elif action_source == "official_goat_image_exact_fallback":
                official_image_stop_successes += 1
        records.append({
            "step": int(step),
            "subtask_before": subtask,
            "subtask_after": after_subtask,
            "task": list(task),
            "position_before": position_before,
            "position_after": after_array.tolist(),
            "executed_translation_m": executed_translation_m,
            "navdp_collision_detected": navdp_collision_detected,
            "collision_recovery": collision_recovery_record,
            "collision_recovery_scheduled": collision_schedule,
            "official_action_id": int(official_action),
            "official_action": ACTION_NAMES[int(official_action)],
            "executed_action_id": int(chosen_action),
            "executed_action": ACTION_NAMES[int(chosen_action)],
            "action_source": action_source,
            "frame_append_mode": frame_append_mode,
            "memory_receipt": memory_receipt,
            "certificate": certificate,
            "navdp_plan": plan_audit,
            "goal_camera_adapter": goal_camera_adapter,
            "arrival_evidence": arrival_evidence,
            "terminal_decision": _search_decision_json(terminal_decision),
            "terminal_search": (
                terminal_receipt if terminal_receipt is not None else
                terminal_search.receipt()
                if terminal_search is not None else None),
            "terminal_origin": terminal_origin_record,
            "resample_audit": resample_audit,
            "ground_truth_read_before_action": False,
            "subtask_success_after": subtask_success_after,
        })
        previous_action.fill_(int(chosen_action))
        mask.fill_(not env.episode_over)
        observation = next_observation

    final_metrics = _jsonable(env.get_metrics())
    success_values = final_metrics.get("success", {}).get(
        "subtask_success", [])
    image_indices = [
        index for index, task in enumerate(episode.tasks)
        if _is_image_task(task)
    ]
    image_successes = sum(
        int(index < len(success_values) and float(success_values[index]) > 0.0)
        for index in image_indices)
    return {
        "scene_id": scene,
        "episode_id": episode_id,
        "status": "complete",
        "steps": len(records),
        "max_steps": int(max_steps),
        "termination_reason": (
            "evaluation_step_guard" if len(records) >= max_steps else
            "all_subtasks_transitioned" if env.task.active_subtask_idx
            >= len(episode.tasks) else
            "safe_stall" if safe_stall else
            "episode_over" if env.episode_over else "loop_terminated"
        ),
        "active_subtask_at_end": int(env.task.active_subtask_idx),
        "subtask_count": len(episode.tasks),
        "image_subtask_count": len(image_indices),
        "image_subtask_entries": int(image_subtask_entries),
        "image_subtask_successes": int(image_successes),
        "image_stop_attempts": int(image_stop_attempts),
        "image_stop_successes": int(image_stop_successes),
        "official_image_stop_count": int(official_image_stop_count),
        "official_image_stop_successes": int(official_image_stop_successes),
        "cec_accept_count": int(cec_accept_count),
        "native_plan_count": int(native_plan_count),
        "mixed_plan_count": int(mixed_plan_count),
        "terminal_search_count": int(terminal_search_count),
        "navdp_collision_count": int(navdp_collision_count),
        "safe_stall": bool(safe_stall),
        "ground_truth_used_by_decision": False,
        "memnav_reset": memnav_reset,
        "navdp_resets": navdp_reset_receipts,
        "metrics": final_metrics,
        "records": records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import requests
    import torch
    from gym import spaces
    from habitat import Env
    from habitat.datasets import make_dataset

    goat_code = args.goat_code.resolve()
    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("refusing to overwrite output directory")
    actual_commit = subprocess.run(
        ["git", "-C", str(goat_code), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True).stdout.strip()
    if actual_commit != EXPECTED_GOAT_COMMIT:
        raise RuntimeError("GOAT source commit changed")
    requested = args.episode or list(DEFAULT_EPISODES)
    config = _policy_config(
        goat_code, data_root, sorted({scene for scene, _ in requested}),
        args.gpu_device_id, args.max_steps)
    camera_adapter = _install_navdp_camera(config)
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset)
    selected = _select_episodes(dataset.episodes, requested)
    dataset.episodes = selected
    torch.set_num_threads(1)
    if args.policy_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("official CUDA policy requested but unavailable")
        policy_device = torch.device("cuda", int(args.gpu_device_id))
    else:
        policy_device = torch.device("cpu")
    sensor = (
        config.habitat.simulator.agents.main_agent.sim_sensors
        .navdp_rgb_sensor)
    current_intrinsic = canonical_navdp_intrinsic()
    camera_height = float(sensor.position[1])

    started = time.monotonic()
    episode_records = []
    with Env(config=config.habitat, dataset=dataset) as env:
        official_observation_space = spaces.Dict(OrderedDict(
            (key, value)
            for key, value in env.observation_space.spaces.items()
            if key != NAVDP_RGB_SENSOR_UUID))
        policy, transforms = _build_official_policy(
            config, official_observation_space, checkpoint, policy_device)
        with requests.Session() as session:
            for episode in selected:
                record = _run_episode(
                    env, episode, policy, transforms, session,
                    args.memnav_url, args.navdp_url, current_intrinsic,
                    camera_height, args.base_seed, args.max_steps,
                    args.request_timeout_s,
                    args.same_observation_resamples,
                    args.unsupported_image_controller,
                    args.navdp_stop_threshold)
                episode_records.append(record)
                print(json.dumps({
                    "scene": record["scene_id"],
                    "episode": record["episode_id"],
                    "termination": record["termination_reason"],
                    "image_successes": record["image_subtask_successes"],
                    "image_subtasks": record["image_subtask_count"],
                    "visual_stops": record["image_stop_attempts"],
                    "successful_visual_stops": record["image_stop_successes"],
                    "terminal_searches": record["terminal_search_count"],
                }, sort_keys=True), flush=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "complete": True,
        "scope": "consumed_scene_autonomous_goat_engineering_pilot_only",
        "is_official_goat_score": False,
        "paper_claim_authorized": False,
        "method": (
            "official_GOAT_for_nonimage_plus_role_free_CEC_bearing_NavDP_"
            "with_calibrated_camera_adapter_configured_unsupported_image_"
            "fallback_and_visual_stop"),
        "ground_truth_used_by_decision": False,
        "imagegoal_official_stop_head_used": bool(
            args.unsupported_image_controller == "official"),
        "object_language_official_controller_retained": True,
        "causal_memory_persists_across_subtasks": True,
        "navdp_short_fifo_resets_at_each_image_subtask": True,
        "navdp_upstream_critic_threshold": (
            float(args.navdp_stop_threshold)),
        "navdp_upstream_critic_threshold_source": (
            "legacy_goat_protocol_default_-0.5"
            if math.isclose(
                args.navdp_stop_threshold,
                NAVDP_UPSTREAM_CRITIC_THRESHOLD,
                rel_tol=0.0, abs_tol=1e-12)
            else "explicit_command_line_override"),
        "adapter_endpoint_stop_radius_m": (
            GOAT_ADAPTER_CONFIG.endpoint_stop_radius_m),
        "adapter_execution_horizon_actions": (
            GOAT_ADAPTER_CONFIG.execution_horizon),
        "adapter_lookahead_distance_m": (
            GOAT_ADAPTER_CONFIG.lookahead_distance_m),
        "navdp_collision_translation_epsilon_m": (
            NAVDP_COLLISION_TRANSLATION_EPS_M),
        "navdp_collision_recovery_turns": NAVDP_COLLISION_RECOVERY_TURNS,
        "camera_adapter": camera_adapter,
        "goat_commit": actual_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "policy_device": str(policy_device),
        "base_seed": int(args.base_seed),
        "max_steps": int(args.max_steps),
        "same_observation_resamples": int(args.same_observation_resamples),
        "unsupported_image_controller": args.unsupported_image_controller,
        "episode_identities": [list(identity) for identity in requested],
        "wall_time_s": float(time.monotonic() - started),
        "episodes": episode_records,
    }
    _atomic_json(output_dir / "goat_autonomous_multigoal_pilot.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goat-code", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=_parse_episode, action="append")
    parser.add_argument("--memnav-url", default="http://127.0.0.1:21640")
    parser.add_argument("--navdp-url", default="http://127.0.0.1:21641")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--policy-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--same-observation-resamples", type=int, default=2)
    parser.add_argument(
        "--navdp-stop-threshold", type=float,
        default=NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        help=("Critic threshold passed verbatim to NavDP.  The legacy GOAT "
              "pilot used -0.5; released eval_imagegoal_wheeled.py defaults "
              "to -3.0, which should be tested as an explicit paired arm."),
    )
    parser.add_argument(
        "--unsupported-image-controller",
        choices=("official", "navdp"), default="official",
        help=("Controller used when role-free CEC rejects.  'official' is the "
              "GOAT-native exact fallback; 'navdp' is a transfer diagnostic."),
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.same_observation_resamples < 0:
        parser.error("--same-observation-resamples must be non-negative")
    if not math.isfinite(args.navdp_stop_threshold):
        parser.error("--navdp-stop-threshold must be finite")
    return args


if __name__ == "__main__":
    run(parse_args())
