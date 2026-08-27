#!/usr/bin/env python3
"""Paired GOAT sequential-Revisit evaluator for official policy vs CEC.

This is an engineering pilot on explicitly consumed HM3D scenes, not a GOAT
benchmark score.  The official monolithic GOAT policy controls every subtask.
The CEC arm may override motion only when the causal history certificate
accepts an ImageGoal.  Certificate rejection is an exact fallback to the
official GOAT action.  Semantic ``subtask_stop`` authority always remains with
the official GOAT policy.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import datetime
import hashlib
import importlib
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from MemNavData.goat_contract_smoke import (
        EXPECTED_GOAT_COMMIT,
        _assert_same_pose,
        _build_config as _build_task_config,
        _current_image_parameters,
        _episode_scene_id,
        _jsonable,
        _render_raw_goal,
    )
    from MemNavData.goat_navdp_discrete_adapter import (
        DiscreteAdapterConfig,
        NavDPAdapterDisposition,
        navdp_waypoints_to_goat_decision,
    )
    from MemNavData.goat_navdp_runtime_pilot import (
        NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        _camera_intrinsic,
        _depth_png_bytes,
        _navdp_wire_jpeg_bytes,
        _normalize_trajectory,
        _plan_seed,
        _rgb_jpeg_bytes,
        _validated_critic_receipt,
    )
    from MemNavData.revisit_bearing_adapter import adapt_revisit_pointgoal
    from MemNavData.xnavdp_revisit_contract import pointgoal_payload
except ModuleNotFoundError:  # immutable/direct-script invocation
    from goat_contract_smoke import (  # type: ignore
        EXPECTED_GOAT_COMMIT,
        _assert_same_pose,
        _build_config as _build_task_config,
        _current_image_parameters,
        _episode_scene_id,
        _jsonable,
        _render_raw_goal,
    )
    from goat_navdp_discrete_adapter import (  # type: ignore
        DiscreteAdapterConfig,
        NavDPAdapterDisposition,
        navdp_waypoints_to_goat_decision,
    )
    from goat_navdp_runtime_pilot import (  # type: ignore
        NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        _camera_intrinsic,
        _depth_png_bytes,
        _navdp_wire_jpeg_bytes,
        _normalize_trajectory,
        _plan_seed,
        _rgb_jpeg_bytes,
        _validated_critic_receipt,
    )
    from revisit_bearing_adapter import adapt_revisit_pointgoal  # type: ignore
    from xnavdp_revisit_contract import pointgoal_payload  # type: ignore


SCHEMA_VERSION = "goat_sequential_revisit_paired_eval_v5_20260815"
DEFAULT_EPISODES = (("5cdEh9F2hJL", "1"), ("4ok3usBNeis", "9"))
ACTION_NAMES = (
    "stop", "move_forward", "turn_left", "turn_right",
    "look_up", "look_down", "subtask_stop",
)
ACTION_IDS = {name: index for index, name in enumerate(ACTION_NAMES)}
GOAT_DISCRETE_ADAPTER_CONFIG = DiscreteAdapterConfig(
    lookahead_distance_m=0.70,
    execution_horizon=1,
)


def first_repeated_image_subtask(tasks: Sequence[Sequence[Any]]) -> int:
    """Return the first ImageGoal whose instance appeared in any prior task.

    GOAT's task-list recurrence stratum includes an earlier LanguageGoal for
    the same instance as well as an earlier ImageGoal.  ObjectGoal entries do
    not carry an instance id and therefore cannot establish exact recurrence.
    The returned index is evaluator-only; the controller still sees only its
    causal RGB stream and the current goal observation.
    """
    prior_instances = set()
    for index, task in enumerate(tasks):
        if len(task) < 3:
            continue
        instance = task[2]
        if (task[1] == "image" and instance is not None
                and str(instance) in prior_instances):
            return index
        if instance is not None:
            prior_instances.add(str(instance))
    raise ValueError("episode has no repeated ImageGoal instance")


def prior_instance_subtasks(
        tasks: Sequence[Sequence[Any]], target: int) -> List[Dict[str, Any]]:
    """Describe evaluator-only task-list evidence preceding ``target``."""
    if target < 0 or target >= len(tasks) or len(tasks[target]) < 3:
        raise ValueError("target is not a valid task index")
    instance = tasks[target][2]
    if tasks[target][1] != "image" or instance is None:
        raise ValueError("target must be an instance-specific ImageGoal")
    prior = []
    for index, task in enumerate(tasks[:target]):
        if len(task) >= 3 and task[2] == instance:
            prior.append({
                "subtask_index": int(index),
                "modality": str(task[1]),
                "instance_id": str(task[2]),
            })
    if not prior:
        raise ValueError("target has no exact prior instance task")
    return prior


def _parse_episode(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("episode must be SCENE:EPISODE")
    scene, episode = value.split(":", 1)
    if not scene or not episode:
        raise argparse.ArgumentTypeError("episode must be SCENE:EPISODE")
    return scene, episode


def _enforce_official_stop_authority(
        official_action: int, executed_action: int) -> int:
    """Fail if CEC ever delays or replaces GOAT's semantic stop decision."""
    if (int(official_action) == ACTION_IDS["subtask_stop"]
            and int(executed_action) != int(official_action)):
        raise RuntimeError("CEC attempted to replace official SUBTASK_STOP")
    return int(executed_action)


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        _jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_episodes(episodes: Iterable[Any], requested):
    lookup = {
        (_episode_scene_id(episode), str(episode.episode_id)): episode
        for episode in episodes
    }
    missing = [identity for identity in requested if identity not in lookup]
    if missing:
        raise RuntimeError("requested episodes absent: {}".format(missing))
    selected = [lookup[identity] for identity in requested]
    for episode in selected:
        first_repeated_image_subtask(episode.tasks)
    return selected


def _manifest_request(path: pathlib.Path, index: int):
    """Load one frozen evaluator-only target from an outcome-blind manifest."""
    payload = json.loads(path.read_text())
    if payload.get("method_or_threshold_selection_allowed") is not False:
        raise RuntimeError("manifest does not forbid method/threshold selection")
    if payload.get("controller_reads_target_metadata") is not False:
        raise RuntimeError("manifest does not preserve role-free control")
    entries = payload.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest has no episode entries")
    if index < 0 or index >= len(entries):
        raise IndexError("manifest index {} outside [0,{})".format(
            index, len(entries)))
    entry = entries[index]
    if int(entry.get("index", index)) != index:
        raise RuntimeError("manifest entry index mismatch")
    scene = str(entry["scene_id"])
    episode = str(entry["episode_id"])
    return (scene, episode), payload, entry


def _validate_manifest_target(episode: Any, entry: Mapping[str, Any]) -> None:
    target = first_repeated_image_subtask(episode.tasks)
    if target != int(entry["target_subtask_index"]):
        raise RuntimeError("frozen target subtask no longer matches dataset")
    task = episode.tasks[target]
    if str(task[2]) != str(entry["target_instance_id"]):
        raise RuntimeError("frozen target instance no longer matches dataset")
    if prior_instance_subtasks(episode.tasks, target) != entry[
            "prior_instance_subtasks"]:
        raise RuntimeError("frozen prior-instance tasks no longer match dataset")
    arm_order = entry.get("arm_order")
    if arm_order not in (["native", "cec"], ["cec", "native"]):
        raise RuntimeError("frozen arm order is invalid")


def _register_policy_modules(goat_code: pathlib.Path) -> None:
    """Register only the released GOAT policy surface needed by this pilot."""
    # _build_task_config performs lean task registration and avoids optional
    # LAVIS/VC1 imports.  Policy subpackages are exposed the same way here.
    contract = importlib.import_module("MemNavData.goat_contract_smoke")
    for name, relative in (
        ("goat_bench.models", "goat_bench/models"),
        ("goat_bench.obs_transformer", "goat_bench/obs_transformer"),
    ):
        if name not in sys.modules:
            contract._install_lean_source_package(  # pylint: disable=protected-access
                name, goat_code / relative)
    importlib.import_module("goat_bench.models.clip_policy")
    importlib.import_module("goat_bench.obs_transformer.resize")


def _policy_config(goat_code: pathlib.Path, data_root: pathlib.Path,
                   scene_ids: Sequence[str], gpu_device_id: int,
                   max_steps: int):
    # First register the exact task/dataset modules and Hydra plugin using the
    # already audited task-config builder.  The returned config is discarded;
    # its registration side effects are the intended contract here.
    _build_task_config(goat_code, data_root, scene_ids, gpu_device_id)
    _register_policy_modules(goat_code)

    from habitat import get_config
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        HabitatSimDepthSensorConfig,
    )

    options = [
        "habitat_baselines.num_environments=1",
        "habitat_baselines.eval.use_ckpt_config=False",
        "habitat_baselines.load_resume_state_config=False",
        "habitat.dataset.split=val_unseen",
        "habitat.dataset.data_path=" + str(
            data_root
            / "data/datasets/goat_bench/hm3d/v1/val_unseen/val_unseen.json.gz"),
        "habitat.dataset.scenes_dir=" + str(data_root / "data/scene_datasets"),
        "habitat.dataset.content_scenes=[{}]".format(
            ",".join(scene_ids)),
        "habitat.task.lab_sensors.goat_goal_sensor.image_cache=" + str(
            data_root
            / "data/goat-assets/goal_cache/iin/val_unseen_embeddings"),
        "habitat.task.lab_sensors.goat_goal_sensor.language_cache=" + str(
            data_root
            / "data/goat-assets/goal_cache/language_nav/"
              "val_unseen_instruction_clip_embeddings.pkl"),
        "habitat.task.lab_sensors.goat_goal_sensor.object_cache=" + str(
            data_root
            / "data/goat-assets/goal_cache/ovon/"
              "category_name_clip_embeddings.pkl"),
        "habitat.simulator.habitat_sim_v0.gpu_device_id={}".format(
            int(gpu_device_id)),
        "habitat.environment.max_episode_steps={}".format(int(max_steps)),
        "habitat.environment.iterator_options.shuffle=False",
        "habitat_baselines.eval.video_option=[]",
        "habitat_baselines.verbose=False",
    ]
    config = get_config(
        str(goat_code / "config/experiments/ver_goat_monolithic.yaml"),
        options,
    )
    with read_write(config):
        agent = config.habitat.simulator.agents.main_agent
        rgb = agent.sim_sensors.rgb_sensor
        agent.sim_sensors.update({
            "depth_sensor": HabitatSimDepthSensorConfig(
                height=int(rgb.height), width=int(rgb.width),
                position=list(rgb.position), orientation=list(rgb.orientation),
                hfov=int(rgb.hfov), min_depth=0.0, max_depth=10.0,
                normalize_depth=False,
            )
        })
    return config


def _build_official_policy(config: Any, observation_space: Any,
                           checkpoint: pathlib.Path, device: Any):
    import torch
    from gym import spaces
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.common.obs_transformers import (
        apply_obs_transforms_obs_space,
        get_active_obs_transforms,
    )

    policy_space = spaces.Dict(OrderedDict(
        (key, value) for key, value in observation_space.spaces.items()
        if key != "depth"))
    transforms = get_active_obs_transforms(config)
    transformed_space = apply_obs_transforms_obs_space(
        policy_space, transforms)
    policy_class = baseline_registry.get_policy("GOATPolicy")
    policy = policy_class.from_config(
        config, transformed_space, spaces.Discrete(len(ACTION_NAMES)))
    raw = torch.load(str(checkpoint), map_location="cpu")
    state = {
        key[len("actor_critic."):]: value
        for key, value in raw["state_dict"].items()
        if key.startswith("actor_critic.")
    }
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.eval()
    return policy, transforms


def _official_action(policy: Any, transforms: Any, observation: Mapping[str, Any],
                     hidden: Any, previous_action: Any, mask: Any,
                     device: Any):
    import torch
    from habitat_baselines.common.obs_transformers import (
        apply_obs_transforms_batch,
    )
    from habitat_baselines.utils.common import batch_obs

    filtered = {key: value for key, value in observation.items()
                if key != "depth"}
    batch = batch_obs([filtered], device=device)
    batch = apply_obs_transforms_batch(batch, transforms)
    with torch.inference_mode():
        _value, action, _log_prob, next_hidden = policy.act(
            batch, hidden, previous_action, mask, deterministic=False)
    return int(action.item()), next_hidden


def _reset_sidecars(session: Any, memnav_url: str, navdp_url: str,
                    intrinsic: np.ndarray, camera_height: float,
                    seed: int, episode_len: int, timeout_s: float) -> None:
    response = session.post(memnav_url.rstrip("/") + "/navigator_reset", json={
        "camera_height": float(camera_height),
        "camera_intrinsic": intrinsic.tolist(),
        "seed": int(seed),
        "episode_len": int(episode_len),
    }, timeout=timeout_s)
    response.raise_for_status()
    receipt = response.json()
    if not receipt.get("certified_relocalization", {}).get("enabled"):
        raise RuntimeError("CEC sidecar is not certificate-enabled")
    response = session.post(navdp_url.rstrip("/") + "/navigator_reset", json={
        "intrinsic": intrinsic.tolist(),
        "stop_threshold": NAVDP_UPSTREAM_CRITIC_THRESHOLD,
        "batch_size": 1,
        "seed": int(seed),
    }, timeout=timeout_s)
    response.raise_for_status()
    navdp_receipt = response.json()
    if navdp_receipt.get("algo") != "navdp":
        raise RuntimeError("unexpected NavDP reset receipt")
    if (navdp_receipt.get("threshold_semantics")
            != "critic_score_fallback"
            or float(navdp_receipt.get("stop_threshold", float("nan")))
            != NAVDP_UPSTREAM_CRITIC_THRESHOLD):
        raise RuntimeError("NavDP critic-threshold contract changed")
    checkpoint_contract = navdp_receipt.get("checkpoint_contract", {})
    if (checkpoint_contract.get("exact_state_dict") is not True
            or int(checkpoint_contract.get("temporal_depth", -1)) != 16):
        raise RuntimeError("NavDP checkpoint/model contract is not exact")


def _append_memory(session: Any, memnav_url: str, rgb: Any,
                   timeout_s: float) -> Dict[str, Any]:
    response = session.post(
        memnav_url.rstrip("/") + "/memory_step",
        files={"image": ("image.jpg", _rgb_jpeg_bytes(rgb), "image/jpeg")},
        timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _probe_certificate(session: Any, memnav_url: str, rgb: Any, goal: Any,
                       goal_intrinsic: np.ndarray,
                       timeout_s: float) -> Dict[str, Any]:
    goal_bytes = _rgb_jpeg_bytes(goal)
    response = session.post(
        memnav_url.rstrip("/") + "/retrieval_probe_step",
        files={
            "image": ("image.jpg", _rgb_jpeg_bytes(rgb), "image/jpeg"),
            "goal": ("goal.jpg", goal_bytes, "image/jpeg"),
        }, timeout=timeout_s)
    response.raise_for_status()
    probe = response.json()
    candidates = probe.get("certified_visual_candidates")
    if not isinstance(candidates, list):
        candidates = []
    response = session.post(
        memnav_url.rstrip("/") + "/certified_relocalize",
        files={"goal": ("goal.jpg", goal_bytes, "image/jpeg")},
        data={
            "candidates": json.dumps(candidates),
            "proposal_order": "geometry_first",
            "learned_rescue": "0",
            "graph_rescue": "0",
            "goal_camera_intrinsic": json.dumps(goal_intrinsic.tolist()),
        }, timeout=timeout_s)
    response.raise_for_status()
    certificate = response.json()
    certificate["probe_frame_idx"] = probe.get("frame_idx")
    certificate["probe_candidates"] = candidates
    certificate["probe_current_goal_cos"] = probe.get("current_goal_cos")
    return certificate


def _navdp_motion(session: Any, navdp_url: str, rgb: Any, depth: Any,
                  goal: Any, bearing: Sequence[float], seed: int,
                  adapter: DiscreteAdapterConfig,
                  timeout_s: float) -> Tuple[Any, Dict[str, Any]]:
    adapted = adapt_revisit_pointgoal(
        mode="verified_bearing_v1", router_active=True,
        pointgoal=bearing, source="goat_cec_pnp",
        pointgoal_units="lingbot_raw_direction_only")
    if not adapted.takeover:
        return None, {"adapter": adapted.audit_dict()}
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
        }, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("diffusion_seed", -1)) != int(seed):
        raise RuntimeError("NavDP did not echo deterministic seed")
    critic = _validated_critic_receipt(payload)
    decision = navdp_waypoints_to_goat_decision(
        _normalize_trajectory(payload.get("trajectory")), adapter)
    return decision, {
        "adapter": adapted.audit_dict(),
        "navdp_disposition": decision.disposition.value,
        "navdp_endpoint_norm_m": decision.endpoint_norm_m,
        "navdp_max_radius_m": decision.max_radius_m,
        "navdp_action_count": len(decision.actions),
        "critic": critic,
    }


def _goal_assets(env: Any, episode: Any, subtask: int):
    parameters, image_index = _current_image_parameters(episode, subtask)
    before = env.sim.get_agent_state()
    sensors = set(env.sim._sensors)
    goal = np.asarray(_render_raw_goal(env.sim, parameters))
    _assert_same_pose(before, env.sim.get_agent_state())
    if sensors != set(env.sim._sensors):
        raise RuntimeError("temporary goal sensor leaked")
    intrinsic = _camera_intrinsic(
        int(goal.shape[0]), int(goal.shape[1]), float(parameters.hfov))
    return goal, intrinsic, int(image_index), float(parameters.hfov)


def _success_for(metrics: Mapping[str, Any], subtask: int) -> float:
    success = metrics.get("success", {})
    values = success.get("subtask_success", []) if isinstance(
        success, Mapping) else []
    return float(values[subtask]) if subtask < len(values) else 0.0


def _run_arm(env: Any, episode: Any, arm: str, policy: Any, transforms: Any,
             session: Any, memnav_url: str, navdp_url: str,
             current_intrinsic: np.ndarray, camera_height: float,
             base_seed: int, max_steps: int, timeout_s: float) -> Dict[str, Any]:
    import torch

    if arm not in ("native", "cec"):
        raise ValueError("unknown arm")
    scene = _episode_scene_id(episode)
    episode_id = str(episode.episode_id)
    target = first_repeated_image_subtask(episode.tasks)
    prior_tasks = prior_instance_subtasks(episode.tasks, target)
    seed = _plan_seed(base_seed, scene, episode_id, 0 if arm == "native" else 1)
    # MemNav seeds both Torch and NumPy's legacy RandomState.  The latter has
    # a strict uint32 domain even though diffusion request seeds are int64.
    service_seed = int(seed % (2 ** 32))
    random.seed(base_seed)
    np.random.seed(base_seed % (2 ** 32))
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    env.current_episode = episode
    observation = env.reset()
    if arm == "cec":
        _reset_sidecars(
            session, memnav_url, navdp_url, current_intrinsic,
            camera_height, service_seed, max_steps + 1, timeout_s)

    hidden = torch.zeros(
        1, policy.num_recurrent_layers,
        int(policy.net.output_size), dtype=torch.float32,
        device=next(policy.parameters()).device)
    # GOAT's released policy uses hidden_size=512; output_size is the same.
    previous_action = torch.zeros(
        1, 1, dtype=torch.long, device=hidden.device)
    mask = torch.zeros(1, 1, dtype=torch.bool, device=hidden.device)
    goal_cache = {}
    queued_actions: List[int] = []
    queued_subtask = None
    plans = 0
    records = []
    first_override = None
    certificate_accepts = 0

    for step in range(max_steps):
        if env.episode_over or env.task.active_subtask_idx >= len(episode.tasks):
            break
        subtask = int(env.task.active_subtask_idx)
        task = episode.tasks[subtask]
        official_action, hidden = _official_action(
            policy, transforms, observation, hidden, previous_action, mask,
            hidden.device)
        chosen_action = official_action
        action_source = "official_goat"
        certificate = None
        plan = None
        frame_append_mode = None

        goal = goal_intrinsic = None
        if task[1] == "image":
            if subtask not in goal_cache:
                goal_cache[subtask] = _goal_assets(env, episode, subtask)
            goal, goal_intrinsic, _image_index, _goal_hfov = goal_cache[subtask]

        if arm == "cec":
            must_probe = bool(
                task[1] == "image"
                and (not queued_actions or official_action == ACTION_IDS[
                    "subtask_stop"]))
            if must_probe:
                frame_append_mode = "probe"
                try:
                    certificate = _probe_certificate(
                        session, memnav_url, observation["rgb"], goal,
                        goal_intrinsic, timeout_s)
                except Exception as error:  # exact official fallback
                    certificate = {
                        "ok": False, "accepted": False,
                        "reason": "runtime_exception_fail_closed",
                        "error": "{}: {}".format(type(error).__name__, error),
                    }
            else:
                frame_append_mode = "append"
                _append_memory(
                    session, memnav_url, observation["rgb"], timeout_s)

            accepted = bool(
                certificate is not None
                and certificate.get("ok") is True
                and certificate.get("accepted") is True
                and certificate.get("pointgoal_units")
                == "lingbot_raw_direction_only")
            certificate_accepts += int(accepted)

            # GOAT success is position-only. Never delay an official semantic
            # stop with U-turn/alignment motion: stop authority remains exact.
            if official_action == ACTION_IDS["subtask_stop"]:
                queued_actions = []
                queued_subtask = None
                chosen_action = official_action
                action_source = "official_goat_subtask_stop"
            elif queued_actions and queued_subtask == subtask:
                chosen_action = queued_actions.pop(0)
                action_source = "cec_navdp_chunk"
            elif accepted:
                try:
                    plan_seed = _plan_seed(
                        base_seed, scene, episode_id, plans + 1000)
                    decision, plan = _navdp_motion(
                        session, navdp_url, observation["rgb"],
                        observation["depth"], goal,
                        certificate.get("aux_pose"), plan_seed,
                        GOAT_DISCRETE_ADAPTER_CONFIG, timeout_s)
                    plans += 1
                    if (decision is not None
                            and decision.disposition
                            is NavDPAdapterDisposition.MOTION):
                        queued_actions = [int(action) for action in decision.actions]
                        queued_subtask = subtask
                        chosen_action = queued_actions.pop(0)
                        action_source = "cec_navdp_chunk"
                except Exception as error:  # exact official fallback
                    plan = {
                        "error": "{}: {}".format(type(error).__name__, error),
                        "fallback": "official_goat",
                    }
            else:
                queued_actions = []
                queued_subtask = None

        chosen_action = _enforce_official_stop_authority(
            official_action, chosen_action)

        if first_override is None and chosen_action != official_action:
            first_override = step
        before = env.sim.get_agent_state()
        position_before = np.asarray(before.position, dtype=float).tolist()
        next_observation = env.step(ACTION_NAMES[chosen_action])
        after_subtask = int(env.task.active_subtask_idx)
        metrics = _jsonable(env.get_metrics())
        records.append({
            "step": step,
            "subtask_before": subtask,
            "subtask_after": after_subtask,
            "task": list(task),
            "position_before": position_before,
            "official_action_id": official_action,
            "official_action": ACTION_NAMES[official_action],
            "executed_action_id": chosen_action,
            "executed_action": ACTION_NAMES[chosen_action],
            "action_source": action_source,
            "frame_append_mode": frame_append_mode,
            "certificate": certificate,
            "navdp_plan": plan,
            "official_subtask_stop_preserved": bool(
                official_action != ACTION_IDS["subtask_stop"]
                or chosen_action == official_action),
            "target_success_after": _success_for(metrics, target),
        })
        previous_action.fill_(chosen_action)
        mask.fill_(not env.episode_over)
        observation = next_observation
        if after_subtask > target:
            break

    metrics = _jsonable(env.get_metrics())
    active_subtask = int(env.task.active_subtask_idx)
    target_entered = any(
        int(record["subtask_before"]) == target for record in records)
    if active_subtask > target:
        termination_reason = "transitioned_past_target"
    elif env.episode_over:
        termination_reason = "episode_over_before_target_transition"
    elif len(records) >= max_steps:
        termination_reason = "evaluation_step_guard"
    else:
        termination_reason = "loop_terminated"
    return {
        "arm": arm,
        "scene_id": scene,
        "episode_id": episode_id,
        "target_repeated_image_subtask": target,
        "target_task": list(episode.tasks[target]),
        "target_prior_instance_subtasks": prior_tasks,
        "complete_through_target": int(env.task.active_subtask_idx) > target,
        "target_entered": target_entered,
        "target_success": _success_for(metrics, target),
        "steps": len(records),
        "max_steps": int(max_steps),
        "active_subtask_at_end": active_subtask,
        "episode_over_at_end": bool(env.episode_over),
        "termination_reason": termination_reason,
        "first_override_step": first_override,
        "certificate_accept_count": certificate_accepts,
        "navdp_plan_count": plans,
        "metrics": metrics,
        "records": records,
    }


def _paired_prefix(native: Mapping[str, Any], cec: Mapping[str, Any]) -> dict:
    boundary = cec.get("first_override_step")
    if boundary is None:
        boundary = min(len(native["records"]), len(cec["records"]))
    paired = True
    mismatch = None
    for index in range(int(boundary)):
        left = native["records"][index]
        right = cec["records"][index]
        if (left["executed_action_id"] != right["executed_action_id"]
                or not np.allclose(
                    left["position_before"], right["position_before"],
                    atol=1e-6)):
            paired = False
            mismatch = index
            break
    return {
        "prefix_paired_before_first_override": paired,
        "compared_steps": int(boundary),
        "first_mismatch_step": mismatch,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import requests
    import subprocess
    import torch
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
    manifest_payload = None
    manifest_entry = None
    manifest_sha256 = None
    if args.manifest is not None:
        if args.episode:
            raise RuntimeError("--manifest and --episode are mutually exclusive")
        if args.index is None:
            raise RuntimeError("--index is required with --manifest")
        manifest_path = args.manifest.resolve()
        identity, manifest_payload, manifest_entry = _manifest_request(
            manifest_path, args.index)
        manifest_sha256 = _sha256_file(manifest_path)
        requested = [identity]
        analysis_contract = manifest_payload.get("analysis_contract", {})
        if int(analysis_contract.get("maximum_steps_per_arm", -1)) != int(
                args.max_steps):
            raise RuntimeError("--max-steps violates frozen manifest contract")
        if int(analysis_contract.get("base_seed", -1)) != int(args.base_seed):
            raise RuntimeError("--base-seed violates frozen manifest contract")
    else:
        if args.index is not None:
            raise RuntimeError("--index requires --manifest")
        requested = args.episode or list(DEFAULT_EPISODES)
    config = _policy_config(
        goat_code, data_root, sorted({scene for scene, _ in requested}),
        args.gpu_device_id, args.max_steps)
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset)
    selected = _select_episodes(dataset.episodes, requested)
    if manifest_entry is not None:
        _validate_manifest_target(selected[0], manifest_entry)
    dataset.episodes = selected
    torch.set_num_threads(1)
    if args.policy_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("official CUDA policy requested but unavailable")
        policy_device = torch.device("cuda", int(args.gpu_device_id))
    else:
        policy_device = torch.device("cpu")

    sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    current_intrinsic = _camera_intrinsic(
        int(sensor.height), int(sensor.width), float(sensor.hfov))
    camera_height = float(sensor.position[1])
    records = []
    started = time.monotonic()
    with Env(config=config.habitat, dataset=dataset) as env:
        policy, transforms = _build_official_policy(
            config, env.observation_space, checkpoint, policy_device)
        with requests.Session() as session:
            for episode in selected:
                if manifest_entry is None:
                    arm_order = ["native", "cec"]
                else:
                    arm_order = list(manifest_entry["arm_order"])
                arm_results = {}
                for arm_name in arm_order:
                    arm_results[arm_name] = _run_arm(
                        env, episode, arm_name, policy, transforms, session,
                        args.memnav_url, args.navdp_url, current_intrinsic,
                        camera_height, args.base_seed, args.max_steps,
                        args.request_timeout_s)
                native = arm_results["native"]
                cec = arm_results["cec"]
                pair = {
                    "scene_id": native["scene_id"],
                    "episode_id": native["episode_id"],
                    "executed_arm_order": arm_order,
                    "native": native,
                    "cec": cec,
                    "prefix_audit": _paired_prefix(native, cec),
                }
                records.append(pair)
                print(json.dumps({
                    "scene": pair["scene_id"],
                    "episode": pair["episode_id"],
                    "native_target_success": native["target_success"],
                    "cec_target_success": cec["target_success"],
                    "cec_accepts": cec["certificate_accept_count"],
                    "prefix_paired": pair["prefix_audit"][
                        "prefix_paired_before_first_override"],
                }, sort_keys=True), flush=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "complete": True,
        "scope": (
            str(manifest_payload.get("evaluation_stage"))
            if manifest_payload is not None
            else "consumed_scene_local_engineering_pilot_only"),
        "is_official_goat_score": False,
        "paper_claim_authorized": bool(
            manifest_payload is not None
            and manifest_payload.get("paper_claim_authorized") is True),
        "role_label_read_by_controller": False,
        "official_goat_stop_authority_retained": True,
        "official_goat_subtask_stop_preserved_exactly": True,
        "official_policy_uses_released_stochastic_eval_semantics": True,
        "paired_policy_sampling_seed": int(args.base_seed),
        "max_steps": int(args.max_steps),
        "policy_device": str(policy_device),
        "runtime_provenance": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "goat_commit": actual_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "manifest_sha256": manifest_sha256,
        "manifest_entry": manifest_entry,
        "manifest_purpose": (
            None if manifest_payload is None else
            manifest_payload.get("purpose")),
        "episode_identities": [list(identity) for identity in requested],
        "wall_time_s": time.monotonic() - started,
        "pairs": records,
    }
    _atomic_json(output_dir / "goat_sequential_revisit_pilot.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goat-code", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=_parse_episode, action="append")
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--memnav-url", default="http://127.0.0.1:21640")
    parser.add_argument("--navdp-url", default="http://127.0.0.1:21641")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--policy-device", choices=("cuda", "cpu"), default="cuda",
        help="CUDA matches the released GOAT evaluator; CPU is diagnostic only.")
    # Habitat-Baselines' released eval entry point seeds Python/NumPy/Torch
    # from habitat.seed, whose unmodified GOAT config value is 100.
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
