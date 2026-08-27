#!/usr/bin/env python3
"""Two-scene GOAT-Bench simulator/task contract smoke.

This harness intentionally does not evaluate a navigation policy and must not
be reported as a GOAT score.  It validates the official GOAT episode loader,
Stretch simulator contract, raw InstanceImageParameters rendering, subtask
transition, and metrics on two frozen val_unseen episodes.

The released repository imports every policy backend from goat_bench/__init__
(including LAVIS and VC-1).  Those modules are irrelevant to this contract
gate, so the harness registers only the released dataset/task/simulator modules
from the pinned source tree.  None of those source files is modified.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import pathlib
import pickle
import random
import subprocess
import sys
import types
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


EXPECTED_GOAT_COMMIT = "74c41d19d4a4c3608d1575b512087b5a529aee0e"
DEFAULT_EPISODES: Tuple[Tuple[str, str], ...] = (
    ("4ok3usBNeis", "3"),
    ("5cdEh9F2hJL", "4"),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_episode(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("episode must be SCENE_ID:EPISODE_ID")
    scene_id, episode_id = value.split(":", 1)
    if not scene_id or not episode_id:
        raise argparse.ArgumentTypeError("episode must be SCENE_ID:EPISODE_ID")
    return scene_id, episode_id


def _install_lean_source_package(name: str, package_dir: pathlib.Path) -> None:
    """Expose a source package without executing its eager ``__init__``."""

    if not package_dir.is_dir():
        raise FileNotFoundError(f"missing {name} package: {package_dir}")
    if name in sys.modules:
        raise RuntimeError(f"{name} was imported before lean registration")

    module = types.ModuleType(name)
    module.__file__ = str(package_dir / "__init__.py")
    module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    module.__package__ = name
    module.__spec__ = importlib.util.spec_from_loader(
        name, loader=None, is_package=True
    )
    sys.modules[name] = module


def _package_dir_without_import(name: str) -> pathlib.Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError(f"cannot locate source package {name}")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(f"ambiguous {name} package locations: {locations}")
    return pathlib.Path(locations[0]).resolve()


def _install_load_pickle_shim(goat_code: pathlib.Path) -> None:
    """Install the only GOAT utility needed by task/sensor registration.

    The released ``goat_bench.utils.utils`` imports Torch and the full visual
    encoder stack at module import time, although the task and measurements
    use only its four-line ``load_pickle`` helper.  Reproducing that helper
    here avoids importing policy code into a simulator-contract smoke.
    """

    utils_dir = goat_code / "goat_bench" / "utils"
    _install_lean_source_package("goat_bench.utils", utils_dir)
    module = types.ModuleType("goat_bench.utils.utils")
    module.__file__ = str(utils_dir / "utils.py")

    def load_pickle(path: str) -> Any:
        with open(path, "rb") as handle:
            return pickle.load(handle)

    module.load_pickle = load_pickle  # type: ignore[attr-defined]
    sys.modules["goat_bench.utils.utils"] = module


def _register_goat_contract_modules(goat_code: pathlib.Path) -> Any:
    # The released Hydra search plugin uses ``pkg://config/tasks``.  Python
    # does not add the process working directory when this harness is invoked
    # by absolute script path, so expose the pinned repository root explicitly.
    goat_root = str(goat_code.resolve())
    if goat_root not in sys.path:
        sys.path.insert(0, goat_root)

    # habitat_baselines has an eager package initializer that imports Torch
    # trainers.  GOAT config registration needs only its structured-config
    # source file, so expose the installed/source package path directly.
    baseline_dir = _package_dir_without_import("habitat_baselines")
    _install_lean_source_package("habitat_baselines", baseline_dir)
    _install_lean_source_package("goat_bench", goat_code / "goat_bench")
    _install_load_pickle_shim(goat_code)

    # Order matters for cross-module registrations.
    names = (
        "goat_bench.config",
        "goat_bench.dataset.ovon_dataset",
        "goat_bench.dataset.languagenav_dataset",
        "goat_bench.task.goat_task",
        "goat_bench.dataset.goat_dataset",
        "goat_bench.task.actions",
        "goat_bench.task.sensors",
        "goat_bench.task.simulator",
        "goat_bench.measurements.nav",
    )
    loaded = {name: importlib.import_module(name) for name in names}
    return loaded["goat_bench.config"]


def _episode_scene_id(episode: Any) -> str:
    filename = pathlib.Path(episode.scene_id).name
    for suffix in (".glb", ".basis"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
    return filename


def _select_episodes(
    episodes: Iterable[Any], requested: Sequence[Tuple[str, str]]
) -> List[Any]:
    by_key = {
        (_episode_scene_id(episode), str(episode.episode_id)): episode
        for episode in episodes
    }
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise RuntimeError(f"requested GOAT episodes are absent: {missing}")
    selected = [by_key[key] for key in requested]
    if len({_episode_scene_id(ep) for ep in selected}) != len(selected):
        raise RuntimeError("contract smoke requires one episode per scene")
    for episode in selected:
        if not episode.tasks or episode.tasks[0][1] != "image":
            raise RuntimeError(
                "frozen contract episodes must begin with an ImageGoal subtask"
            )
    return selected


def _current_image_parameters(episode: Any, subtask_index: int) -> Tuple[Any, int]:
    from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters

    task = episode.tasks[subtask_index]
    if task[1] != "image" or len(task) < 4:
        raise RuntimeError(f"subtask {subtask_index} is not an ImageGoal: {task}")
    instance_id = task[2]
    image_index = int(task[3])
    candidates = [
        goal
        for goal in episode.goals[subtask_index]
        if goal["object_id"] == instance_id
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one goal for {instance_id}, found {len(candidates)}"
        )
    params = candidates[0]["image_goals"][image_index]
    return InstanceImageParameters(**params), image_index


def _render_raw_goal(sim: Any, parameters: Any) -> Any:
    """Call Habitat's released raw InstanceImageGoal rendering primitive."""

    from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor

    renderer = object.__new__(InstanceImageGoalSensor)
    renderer._sim = sim
    return renderer._get_instance_image_goal(parameters)


def _rotation_coefficients(rotation: Any) -> List[float]:
    import numpy as np
    import quaternion

    return np.asarray(quaternion.as_float_array(rotation), dtype=float).tolist()


def _assert_same_pose(before: Any, after: Any) -> None:
    import numpy as np

    if not np.allclose(before.position, after.position, atol=1e-7):
        raise RuntimeError("raw goal rendering moved the agent position")
    if not np.allclose(
        _rotation_coefficients(before.rotation),
        _rotation_coefficients(after.rotation),
        atol=1e-7,
    ):
        raise RuntimeError("raw goal rendering moved the agent rotation")


def _adapter_contract() -> Dict[str, Any]:
    """Verify the frozen NavDP-to-GOAT action and no-motion stop contract."""

    import numpy as np

    try:
        from MemNavData.goat_navdp_discrete_adapter import (
            GoatNavAction,
            NavDPAdapterDisposition,
            navdp_waypoints_to_goat_decision,
            navdp_waypoints_to_goat_actions,
        )
    except ModuleNotFoundError:
        # Immutable HPC bundles invoke this file by absolute path, placing its
        # MemNavData directory (rather than the bundle root) on sys.path.
        from goat_navdp_discrete_adapter import (  # type: ignore[no-redef]
            GoatNavAction,
            NavDPAdapterDisposition,
            navdp_waypoints_to_goat_decision,
            navdp_waypoints_to_goat_actions,
        )

    zero = navdp_waypoints_to_goat_decision(
        np.asarray([[0.05, 0.0, 0.0], [0.10, 0.0, 0.0]])
    )
    straight = navdp_waypoints_to_goat_actions(
        np.asarray([[0.25, 0.0, 0.0], [0.50, 0.0, 0.0]])
    )
    left = navdp_waypoints_to_goat_actions(
        np.asarray([[0.0, 0.25, 0.0], [0.0, 0.50, 0.0]])
    )
    if (zero.disposition is not NavDPAdapterDisposition.ARRIVAL_PROPOSAL
            or zero.actions):
        raise RuntimeError(
            f"near-zero NavDP path was not isolated as a proposal: {zero}")
    if straight != (GoatNavAction.MOVE_FORWARD, GoatNavAction.MOVE_FORWARD):
        raise RuntimeError(f"straight NavDP path conversion changed: {straight}")
    if left[:4] != (
        GoatNavAction.TURN_LEFT,
        GoatNavAction.TURN_LEFT,
        GoatNavAction.TURN_LEFT,
        GoatNavAction.MOVE_FORWARD,
    ):
        raise RuntimeError(f"30-degree turn conversion changed: {left}")
    return {
        "no_motion_is_only_arrival_proposal": True,
        "adapter_never_emits_subtask_stop": True,
        "stop_action_name": "subtask_stop",
        "stop_action_id": int(GoatNavAction.SUBTASK_STOP),
        "straight_action_ids": [int(action) for action in straight],
        "left_action_ids": [int(action) for action in left],
        "forward_step_m": 0.25,
        "turn_angle_deg": 30.0,
    }


def _run_episode_contract(
    env: Any,
    episode: Any,
    artifact_dir: pathlib.Path,
    subtask_stop_action: str,
) -> Dict[str, Any]:
    import numpy as np
    from PIL import Image

    env.current_episode = episode
    observations = env.reset()
    scene_id = _episode_scene_id(episode)
    if not env.sim.pathfinder.is_loaded:
        raise RuntimeError(f"pathfinder did not load for {scene_id}")
    if env.task.active_subtask_idx != 0:
        raise RuntimeError("GOAT task did not reset to subtask zero")
    if "rgb" not in observations:
        raise RuntimeError("official RGB observation is absent")
    if "depth" not in observations:
        raise RuntimeError("NavDP metric-depth observation is absent")

    rgb = np.asarray(observations["rgb"])
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise RuntimeError(f"unexpected RGB shape: {rgb.shape}")
    depth = np.asarray(observations["depth"])
    if depth.shape != rgb.shape[:2] + (1,):
        raise RuntimeError(f"RGB/depth shape mismatch: {rgb.shape}, {depth.shape}")
    if not np.isfinite(depth).all():
        raise RuntimeError("metric depth contains non-finite values")
    if float(depth.min()) < 0.0 or float(depth.max()) <= 0.1:
        raise RuntimeError("metric depth is empty or outside the expected range")

    parameters, image_index = _current_image_parameters(episode, 0)
    before = env.sim.get_agent_state()
    sensors_before = set(env.sim._sensors)
    goal_image = np.asarray(_render_raw_goal(env.sim, parameters))
    sensors_after = set(env.sim._sensors)
    after = env.sim.get_agent_state()
    _assert_same_pose(before, after)
    if sensors_before != sensors_after:
        raise RuntimeError("temporary goal sensor leaked into the simulator")

    expected_shape = tuple(int(x) for x in parameters.image_dimensions) + (3,)
    if goal_image.shape != expected_shape:
        raise RuntimeError(
            f"raw goal shape {goal_image.shape} != dataset shape {expected_shape}"
        )
    if goal_image.dtype != np.uint8:
        raise RuntimeError(f"raw goal dtype is {goal_image.dtype}, expected uint8")
    if float(goal_image.std()) <= 1.0:
        raise RuntimeError("raw goal rendering is nearly constant")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    image_path = artifact_dir / f"{scene_id}_episode_{episode.episode_id}_goal0.png"
    Image.fromarray(goal_image).save(str(image_path))
    image_hash = _sha256_bytes(image_path.read_bytes())

    metrics_before = _jsonable(env.get_metrics())
    first_task = list(episode.tasks[0])
    next_observations = env.step(subtask_stop_action)
    if env.task.active_subtask_idx != 1:
        raise RuntimeError("SUBTASK_STOP did not advance exactly one subtask")
    metrics_after = _jsonable(env.get_metrics())

    current_subtask = _jsonable(next_observations.get("current_subtask"))
    return {
        "scene_id": scene_id,
        "episode_id": str(episode.episode_id),
        "scene_path": str(pathlib.Path(episode.scene_id).resolve()),
        "scene_file_exists": pathlib.Path(episode.scene_id).is_file(),
        "pathfinder_loaded": True,
        "start_is_navigable": bool(env.sim.pathfinder.is_navigable(before.position)),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "first_task": first_task,
        "goal_image_index": image_index,
        "goal_hfov": float(parameters.hfov),
        "goal_image_dimensions": list(parameters.image_dimensions),
        "raw_goal_shape": list(goal_image.shape),
        "raw_goal_std": float(goal_image.std()),
        "raw_goal_png": str(image_path.resolve()),
        "raw_goal_png_sha256": image_hash,
        "agent_pose_unchanged_by_goal_render": True,
        "temporary_sensor_removed": True,
        "metrics_before_subtask_stop": metrics_before,
        "metrics_after_subtask_stop": metrics_after,
        "active_subtask_after_stop": int(env.task.active_subtask_idx),
        "current_subtask_observation_after_stop": current_subtask,
        "subtask_transition_passed": True,
    }


def _build_config(
    goat_code: pathlib.Path,
    data_root: pathlib.Path,
    scene_ids: Sequence[str],
    gpu_device_id: int,
) -> Any:
    goat_config = _register_goat_contract_modules(goat_code)

    from habitat import get_config
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        HabitatSimDepthSensorConfig,
        register_hydra_plugin,
    )

    register_hydra_plugin(goat_config.HabitatConfigPlugin)
    # This is a simulator/task contract gate, not a policy evaluation.  Compose
    # the released GOAT task config directly so the environment is identical
    # without loading the monolithic policy defaults or its model backends.
    config = get_config(str(goat_code / "config/tasks/goat_stretch_hm3d.yaml"))
    with read_write(config):
        config.habitat.seed = 20260814
        config.habitat.dataset.split = "val_unseen"
        config.habitat.dataset.data_path = str(
            data_root
            / "data/datasets/goat_bench/hm3d/v1/val_unseen/val_unseen.json.gz"
        )
        config.habitat.dataset.scenes_dir = str(data_root / "data/scene_datasets")
        config.habitat.dataset.content_scenes = list(scene_ids)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_device_id
        config.habitat.environment.max_episode_steps = 32
        agent = config.habitat.simulator.agents.main_agent
        rgb_sensor = agent.sim_sensors.rgb_sensor
        agent.sim_sensors.update(
            {
                "depth_sensor": HabitatSimDepthSensorConfig(
                    height=int(rgb_sensor.height),
                    width=int(rgb_sensor.width),
                    position=list(rgb_sensor.position),
                    orientation=list(rgb_sensor.orientation),
                    hfov=int(rgb_sensor.hfov),
                    min_depth=0.0,
                    max_depth=10.0,
                    normalize_depth=False,
                )
            }
        )
        config.habitat.task.lab_sensors.goat_goal_sensor.object_cache = str(
            data_root
            / "data/goat-assets/goal_cache/ovon/category_name_clip_embeddings.pkl"
        )
        config.habitat.task.lab_sensors.goat_goal_sensor.image_cache = str(
            data_root
            / "data/goat-assets/goal_cache/iin/val_unseen_embeddings"
        )
        config.habitat.task.lab_sensors.goat_goal_sensor.language_cache = str(
            data_root
            / "data/goat-assets/goal_cache/language_nav/val_unseen_instruction_clip_embeddings.pkl"
        )
        options = config.habitat.environment.iterator_options
        if "shuffle" in options:
            options.shuffle = False
        if "cycle" in options:
            options.cycle = False
        if "group_by_scene" in options:
            options.group_by_scene = False
    return config


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    from habitat import Env
    from habitat.datasets import make_dataset

    goat_code = args.goat_code.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_commit = subprocess.run(
        ["git", "-C", str(goat_code), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != EXPECTED_GOAT_COMMIT:
        raise RuntimeError(
            f"GOAT commit {actual_commit} != frozen {EXPECTED_GOAT_COMMIT}"
        )

    requested = tuple(args.episode or DEFAULT_EPISODES)
    scene_ids = [key[0] for key in requested]
    config = _build_config(goat_code, data_root, scene_ids, args.gpu_device_id)
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    selected = _select_episodes(dataset.episodes, requested)
    dataset.episodes = selected

    random.seed(20260814)
    np.random.seed(20260814)
    adapter_contract = _adapter_contract()
    records: List[Dict[str, Any]] = []
    with Env(config=config.habitat, dataset=dataset) as env:
        env.seed(20260814)
        for episode in selected:
            records.append(
                _run_episode_contract(
                    env,
                    episode,
                    output_dir / "images",
                    adapter_contract["stop_action_name"],
                )
            )

    payload = {
        "schema_version": "goat_two_scene_contract_smoke_v2_20260814",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": True,
        "is_navigation_score": False,
        "purpose": "dataset_simulator_raw_goal_subtask_metric_contract_only",
        "lean_import_scope": {
            "official_source_modules_unmodified": True,
            "goat_package_initializer_bypassed": True,
            "habitat_baselines_initializer_bypassed": True,
            "goat_utils_surface": ["load_pickle"],
        },
        "navdp_discrete_adapter_contract": adapter_contract,
        "goat_commit": actual_commit,
        "data_root": str(data_root),
        "episode_count": len(records),
        "scene_count": len({record["scene_id"] for record in records}),
        "records": records,
        "all_scene_files_exist": all(record["scene_file_exists"] for record in records),
        "all_pathfinders_loaded": all(record["pathfinder_loaded"] for record in records),
        "all_goal_renders_passed": all(
            record["agent_pose_unchanged_by_goal_render"]
            and record["temporary_sensor_removed"]
            for record in records
        ),
        "all_subtask_transitions_passed": all(
            record["subtask_transition_passed"] for record in records
        ),
    }
    receipt = output_dir / "goat_two_scene_contract_smoke.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goat-code", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--episode",
        action="append",
        type=_parse_episode,
        help="frozen SCENE_ID:EPISODE_ID; repeat exactly twice",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requested = args.episode or list(DEFAULT_EPISODES)
    if len(requested) != 2:
        raise SystemExit("contract smoke requires exactly two episodes")
    run(args)


if __name__ == "__main__":
    main()
