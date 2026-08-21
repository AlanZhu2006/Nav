#!/usr/bin/env python3
"""Build the train-only active-scan teacher table for Cyclic Goal Compass.

For each paired factual/counterfactual Novel-B state, this producer:

1. restores the exact generated state pose from the pinned expert manifest;
2. chooses a label-independent cyclic gauge;
3. renders eight monocular views at 45-degree increments without moving;
4. evaluates a one-metre local counterfactual on the frozen navmesh; and
5. records geodesic progress as a masked circular advantage field.

The two goal variants share exactly the same rendered scan.  Habitat goal
position and geodesic distance are labels only and never become model inputs.
Development and final-reserved scenes are rejected by construction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    from MemNavData.circular_goal_compass import (
        NUM_DIRECTIONS,
        circular_bin_error,
        deterministic_gauge_bin,
        masked_argmax,
        native_scan_index,
        scan_yaws,
        teacher_distribution,
        world_forward_xz,
    )
    from MemNavData.habitat_rollout_primitives import (
        DATA_TO_HABITAT_ROTATION,
        parquet_data_pose_to_habitat,
    )
except ImportError:  # direct execution with MemNavData on PYTHONPATH
    from circular_goal_compass import (  # type: ignore
        NUM_DIRECTIONS,
        circular_bin_error,
        deterministic_gauge_bin,
        masked_argmax,
        native_scan_index,
        scan_yaws,
        teacher_distribution,
        world_forward_xz,
    )
    from habitat_rollout_primitives import (  # type: ignore
        DATA_TO_HABITAT_ROTATION,
        parquet_data_pose_to_habitat,
    )


SCHEMA_VERSION = "cgc_multiyaw_teacher_v2"
REPORT_SCHEMA_VERSION = "cgc_multiyaw_teacher_report_v2"
GAUGE_SALT = "cgc-c8-active-scan-v1-20260809"
CAMERA_HEIGHT_M = 0.5
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 270
HFOV_DEG = 68.0
QUERY_RADIUS_M = 1.0
MIN_QUERY_EXTENT_M = 0.30
MAX_QUERY_HEADING_ERROR_DEG = 30.0
STATE_NAMES = frozenset(("goal_b_t0", "goal_b_midpoint_t1"))
EXPECTED_HABITAT_SIM_VERSION = "0.3.1"
ELIGIBILITY_SCHEMA_VERSION = "orbit_distilled_subgoal_eligibility_v1"


class DatasetBuildError(RuntimeError):
    """An input or generated row violated the frozen CGC contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetBuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8")


def load_json_pinned(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    require(path.is_file(), f"missing JSON input: {path}")
    require(sha256_file(path) == expected_sha256, f"SHA256 mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"JSON root must be an object: {path}")
    return value


def resolve_relative_input(relative: str, roots: Sequence[Path],
                           label: str, *, directory: bool = False) -> Path:
    path_fragment = Path(relative)
    require(not path_fragment.is_absolute() and ".." not in path_fragment.parts,
            f"{label} relative path is unsafe")
    matches: list[Path] = []
    for root in roots:
        root_resolved = root.resolve(strict=True)
        candidate = (root_resolved / path_fragment).resolve(strict=False)
        try:
            candidate.relative_to(root_resolved)
        except ValueError as error:
            raise DatasetBuildError(f"{label} path escapes input root") from error
        exists = candidate.is_dir() if directory else candidate.is_file()
        if exists:
            matches.append(candidate)
    require(len(matches) == 1,
            f"{label} must resolve in exactly one pinned episode root")
    return matches[0]


def verify_manifest_file(record: Mapping[str, Any], roots: Sequence[Path],
                         label: str) -> Path:
    required = {"path", "path_sha256", "bytes", "content_sha256"}
    require(set(record) == required, f"{label} file record schema changed")
    relative = record["path"]
    require(isinstance(relative, str) and relative,
            f"{label} path is invalid")
    require(hashlib.sha256(relative.encode("utf-8")).hexdigest()
            == record["path_sha256"], f"{label} path hash changed")
    path = resolve_relative_input(relative, roots, label)
    require(path.is_file(), f"{label} is not a regular file")
    require(path.stat().st_size == int(record["bytes"]),
            f"{label} byte size changed")
    require(sha256_file(path) == record["content_sha256"],
            f"{label} content changed")
    return path


def make_simulator(glb_path: Path, navmesh_path: Path):
    import habitat_sim
    import magnum as mn

    require(getattr(habitat_sim, "__version__", None)
            == EXPECTED_HABITAT_SIM_VERSION,
            "Habitat-Sim runtime version changed")
    simulator_configuration = habitat_sim.SimulatorConfiguration()
    simulator_configuration.scene_id = str(glb_path)
    simulator_configuration.enable_physics = False

    def camera(uuid: str, sensor_type):
        specification = habitat_sim.CameraSensorSpec()
        specification.uuid = uuid
        specification.sensor_type = sensor_type
        specification.resolution = [IMAGE_HEIGHT, IMAGE_WIDTH]
        specification.hfov = HFOV_DEG
        specification.position = mn.Vector3(0.0, 0.0, 0.0)
        return specification

    agent_configuration = habitat_sim.agent.AgentConfiguration()
    agent_configuration.sensor_specifications = [
        camera("color", habitat_sim.SensorType.COLOR),
        camera("depth", habitat_sim.SensorType.DEPTH),
    ]
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(
        simulator_configuration, [agent_configuration]))
    loaded = simulator.pathfinder.load_nav_mesh(str(navmesh_path))
    require(bool(loaded), f"cannot load frozen navmesh: {navmesh_path}")
    return simulator


def render_rgb(simulator, floor_position: np.ndarray, yaw_rad: float) -> np.ndarray:
    import habitat_sim
    import quaternion

    state = habitat_sim.agent.AgentState()
    state.position = np.asarray(floor_position, dtype=np.float64) + np.asarray(
        [0.0, CAMERA_HEIGHT_M, 0.0], dtype=np.float64)
    state.rotation = quaternion.from_rotation_vector([0.0, float(yaw_rad), 0.0])
    simulator.get_agent(0).set_state(state)
    observations = simulator.get_sensor_observations()
    rgb = np.asarray(observations["color"])[..., :3]
    require(rgb.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "Habitat RGB shape changed")
    return rgb.astype(np.uint8, copy=True)


def shortest_path(simulator, start: np.ndarray, goal: np.ndarray):
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(goal, dtype=np.float32)
    found = simulator.pathfinder.find_path(path)
    if not found or not math.isfinite(float(path.geodesic_distance)):
        return None
    return float(path.geodesic_distance), [
        np.asarray(point, dtype=np.float64) for point in path.points]


def heading_error_deg(actual_xz: np.ndarray, requested_xz: np.ndarray) -> float:
    actual_angle = math.atan2(float(actual_xz[0]), float(actual_xz[1]))
    requested_angle = math.atan2(
        float(requested_xz[0]), float(requested_xz[1]))
    delta = (actual_angle - requested_angle + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def counterfactual_field(simulator, start: np.ndarray, goal: np.ndarray,
                         yaws: Sequence[float]) -> Mapping[str, Any]:
    baseline = shortest_path(simulator, start, goal)
    require(baseline is not None, "state-to-goal path is unreachable")
    initial_distance, path_points = baseline
    advantages: list[float] = []
    valid: list[bool] = []
    action_faithful: list[bool] = []
    extents: list[float] = []
    heading_errors: list[float | None] = []
    endpoint_distances: list[float | None] = []

    for yaw in yaws:
        requested_xz = world_forward_xz(float(yaw))
        requested = np.asarray(start, dtype=np.float64).copy()
        requested[[0, 2]] += QUERY_RADIUS_M * requested_xz
        candidate = np.asarray(
            simulator.pathfinder.try_step(start, requested), dtype=np.float64)
        displacement = candidate[[0, 2]] - np.asarray(start)[[0, 2]]
        extent = float(np.linalg.norm(displacement))
        extents.append(extent)
        error = (heading_error_deg(displacement, requested_xz)
                 if extent > 1e-8 else None)
        heading_errors.append(error)
        require(bool(np.isfinite(candidate).all()),
                "try_step returned a non-finite endpoint")
        require(abs(float(candidate[1] - start[1])) <= 0.5,
                "try_step changed floors")
        candidate_path = shortest_path(simulator, candidate, goal)
        require(candidate_path is not None,
                "try_step endpoint is disconnected from the goal")
        endpoint_distance = float(candidate_path[0])
        endpoint_distances.append(endpoint_distance)
        advantages.append(float(initial_distance - endpoint_distance))
        valid.append(True)
        action_faithful.append(bool(
            extent >= MIN_QUERY_EXTENT_M
            and error is not None
            and error <= MAX_QUERY_HEADING_ERROR_DEG
        ))

    # Every requested heading is an intervention, including a blocked one.
    # A collision/no-motion request naturally receives approximately zero
    # progress; an along-wall slide is scored at its actual endpoint.  Habitat
    # therefore supplies training targets but never a deploy-time validity mask.
    require(all(valid), "the counterfactual ring must define all eight actions")
    finite_advantages = np.asarray(advantages, dtype=np.float64)
    best_index = masked_argmax(finite_advantages, valid)
    teacher = teacher_distribution(finite_advantages, valid)
    first_hop_bearing = None
    if len(path_points) >= 2:
        for point in path_points[1:]:
            delta = point[[0, 2]] - np.asarray(start)[[0, 2]]
            if np.linalg.norm(delta) >= 0.30:
                first_hop_bearing = math.atan2(-float(delta[0]), -float(delta[1]))
                break
    return {
        "advantages_m": advantages,
        "candidate_endpoint_distance_m": endpoint_distances,
        "candidate_extent_m": extents,
        "candidate_heading_error_deg": heading_errors,
        "candidate_action_faithful": action_faithful,
        "candidate_valid": valid,
        "initial_geodesic_distance_m": initial_distance,
        "oracle_best_index": best_index,
        "oracle_first_hop_yaw_rad": first_hop_bearing,
        "teacher_distribution": teacher.tolist(),
    }


def index_episode_records(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    scenes = manifest.get("scenes")
    require(isinstance(scenes, list), "manifest scene records are missing")
    for scene_record in scenes:
        require(isinstance(scene_record, Mapping), "manifest scene record is invalid")
        scene = str(scene_record.get("scene", ""))
        episodes = scene_record.get("selected_episodes")
        require(scene and isinstance(episodes, list),
                f"manifest episode records are invalid: {scene}")
        for episode_record in episodes:
            require(isinstance(episode_record, Mapping),
                    f"manifest episode record is invalid: {scene}")
            episode = str(episode_record.get("episode", ""))
            key = f"{scene}/{episode}"
            require(episode and key not in result,
                    f"manifest episode record is not unique: {key}")
            result[key] = episode_record
    require(bool(result), "manifest episode index is empty")
    return result


def read_state_pose(sample: Mapping[str, Any], episode_roots: Sequence[Path],
                    episode_records: Mapping[str, Mapping[str, Any]]):
    import pandas as pd

    source_id = str(sample["source_episode_id"])
    require(source_id in episode_records,
            f"source episode is absent from manifest records: {source_id}")
    episode_record = episode_records[source_id]
    meta_path = verify_manifest_file(
        episode_record["metadata"], episode_roots,
        f"{sample['sample_id']} source metadata")
    parquet_path = verify_manifest_file(
        episode_record["parquet"], episode_roots,
        f"{sample['sample_id']} source parquet")
    verify_manifest_file(
        sample["state_frame"], episode_roots,
        f"{sample['sample_id']} state frame")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    frame_convention = str(metadata.get("frame_convention", ""))
    camera_height = float(metadata.get("camera_height_m", CAMERA_HEIGHT_M))
    require(abs(camera_height - CAMERA_HEIGHT_M) <= 1e-8,
            "camera height changed")
    frame_index = int(sample["navdp_fifo"]["current_frame_index"])
    require(int(sample["decision_frame"]) == frame_index + 1,
            "decision frame and FIFO current index differ")
    state_frame_path = str(sample["state_frame"]["path"])
    require(Path(state_frame_path).stem == str(frame_index),
            "state-frame path and FIFO index differ")
    table = pd.read_parquet(parquet_path)
    require(len(table) == int(episode_record["n_frames"]),
            "parquet row count differs from manifest episode record")
    require(0 <= frame_index < len(table), "state frame is outside parquet")
    row = table.iloc[frame_index]
    pose = parquet_data_pose_to_habitat(
        row["action"], row["observation.camera_extrinsic"],
        camera_height_m=camera_height,
        frame_convention=frame_convention,
    )
    return pose, metadata


def goal_world_position(sample: Mapping[str, Any],
                        episode_roots: Sequence[Path],
                        episode_records: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    goal_path = verify_manifest_file(
        sample["goal"], episode_roots, f"{sample['sample_id']} goal")
    goal_source_id = str(sample["goal_source_episode_id"])
    require(goal_source_id in episode_records,
            f"goal episode is absent from manifest records: {goal_source_id}")
    goal_episode_record = episode_records[goal_source_id]
    require(goal_episode_record["goal_b"] == sample["goal"],
            "sample goal differs from its manifest episode Goal-B record")
    metadata_path = verify_manifest_file(
        goal_episode_record["metadata"], episode_roots,
        f"{sample['sample_id']} goal metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches = [goal for goal in metadata.get("goals", [])
               if goal.get("name") == sample["goal_role"]]
    require(len(matches) == 1, "goal metadata role is ambiguous")
    data_position = np.asarray(matches[0]["pos"], dtype=np.float64)
    require(data_position.shape == (3,) and np.isfinite(data_position).all(),
            "goal position is invalid")
    return DATA_TO_HABITAT_ROTATION @ data_position


def geometry_paths(scene: str, geometry_map: Mapping[str, Any],
                   geometry_root: Path, scene_root: Path):
    record = geometry_map["scenes"].get(scene)
    require(isinstance(record, Mapping), f"scene absent from geometry map: {scene}")
    identity_relative = Path(str(record["identity_path"]))
    identity_path = geometry_root / identity_relative
    require(identity_path.is_file(), f"missing geometry identity: {scene}")
    require(sha256_file(identity_path) == record["identity_sha256"],
            f"geometry identity SHA changed: {scene}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    require(identity.get("schema_version") == 1,
            f"geometry identity schema changed: {scene}")
    require(identity.get("habitat_sim_version")
            == EXPECTED_HABITAT_SIM_VERSION,
            f"geometry identity Habitat version changed: {scene}")
    geometry_id = identity_relative.stem
    navmesh = geometry_root.parent / (
        "navmesh_bake_habitat031_agent030_f8b275152dd1/scenes") / geometry_id / (
            "scene.navmesh")
    glb = scene_root / scene / f"{scene}.glb"
    require(glb.is_file() and navmesh.is_file(),
            f"missing scene geometry for {scene}")
    require(glb.stat().st_size == int(identity["glb"]["bytes"])
            and sha256_file(glb) == identity["glb"]["content_sha256"],
            f"GLB identity changed: {scene}")
    require(navmesh.stat().st_size == int(identity["navmesh"]["bytes"])
            and sha256_file(navmesh) == identity["navmesh"]["content_sha256"],
            f"navmesh identity changed: {scene}")
    return glb, navmesh, geometry_id


def group_samples(manifest: Mapping[str, Any], train_scenes: set[str],
                  max_scenes: int) -> list[tuple[str, list[Mapping[str, Any]]]]:
    samples = [
        sample for sample in manifest["samples"]
        if sample.get("split_role") == "train"
        and sample.get("goal_role") == "B"
        and sample.get("state_name") in STATE_NAMES
    ]
    present_scenes = sorted({str(sample["scene"]) for sample in samples})
    require(set(present_scenes) == train_scenes,
            "manifest Novel-B train scenes differ from split")
    if max_scenes:
        selected = set(present_scenes[:max_scenes])
        samples = [sample for sample in samples if sample["scene"] in selected]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        sample_id = str(sample["sample_id"])
        variant = str(sample["goal_variant"])
        require(variant in ("factual", "counterfactual"),
                "unexpected goal variant")
        require(sample_id.endswith(f"/{variant}"), "sample id/variant mismatch")
        groups[sample_id.rsplit("/", 1)[0]].append(sample)
    for group_id, members in groups.items():
        require(len(members) == 2, f"goal-swap pair is incomplete: {group_id}")
        require({member["goal_variant"] for member in members}
                == {"factual", "counterfactual"},
                f"goal-swap variants changed: {group_id}")
        identity = {
            (member["scene"], member["source_episode_id"],
             member["state_name"], member["navdp_fifo"]["current_frame_index"])
            for member in members
        }
        require(len(identity) == 1, f"goal pair changed physical state: {group_id}")
    return [(key, sorted(value, key=lambda row: row["goal_variant"]))
            for key, value in sorted(groups.items())]


def apply_group_eligibility(
        groups: Sequence[tuple[str, list[Mapping[str, Any]]]],
        eligibility: Mapping[str, Any]) -> tuple[
            list[tuple[str, list[Mapping[str, Any]]]], Mapping[str, Any]]:
    """Apply a pre-model, physical-state eligibility manifest.

    The manifest may exclude complete scenes whose two episode goals occupy
    disconnected navmesh components and individual physical groups whose C8
    intervention basis is undefined.  It cannot select individual goal
    variants: factual/counterfactual pairing remains exact.
    """
    required = {
        "schema_version", "scope", "input_physical_group_count",
        "input_scene_count", "excluded_scenes", "excluded_group_ids",
        "expected_selected_physical_group_count", "expected_selected_scene_count",
    }
    require(set(eligibility) == required,
            "eligibility manifest schema fields changed")
    require(eligibility.get("schema_version") == ELIGIBILITY_SCHEMA_VERSION,
            "unsupported eligibility manifest schema")
    require(isinstance(eligibility.get("scope"), str)
            and bool(eligibility["scope"]), "eligibility scope is invalid")
    group_map = {group_id: members for group_id, members in groups}
    require(len(group_map) == len(groups), "physical group IDs are not unique")
    input_scenes = {str(members[0]["scene"]) for _, members in groups}
    require(int(eligibility["input_physical_group_count"]) == len(groups),
            "eligibility input physical-group count changed")
    require(int(eligibility["input_scene_count"]) == len(input_scenes),
            "eligibility input scene count changed")
    excluded_scenes_raw = eligibility["excluded_scenes"]
    excluded_groups_raw = eligibility["excluded_group_ids"]
    require(isinstance(excluded_scenes_raw, list)
            and all(isinstance(value, str) and value
                    for value in excluded_scenes_raw),
            "eligibility excluded scenes are invalid")
    require(isinstance(excluded_groups_raw, list)
            and all(isinstance(value, str) and value
                    for value in excluded_groups_raw),
            "eligibility excluded groups are invalid")
    excluded_scenes = set(excluded_scenes_raw)
    excluded_groups = set(excluded_groups_raw)
    require(len(excluded_scenes) == len(excluded_scenes_raw)
            and len(excluded_groups) == len(excluded_groups_raw),
            "eligibility exclusions contain duplicates")
    require(excluded_scenes <= input_scenes,
            "eligibility excludes an absent scene")
    require(excluded_groups <= set(group_map),
            "eligibility excludes an absent physical group")
    require(all(str(group_map[group_id][0]["scene"]) not in excluded_scenes
                for group_id in excluded_groups),
            "individual group exclusion is redundant with a scene exclusion")
    selected = [
        (group_id, members) for group_id, members in groups
        if str(members[0]["scene"]) not in excluded_scenes
        and group_id not in excluded_groups
    ]
    selected_scenes = {str(members[0]["scene"]) for _, members in selected}
    require(len(selected) == int(
        eligibility["expected_selected_physical_group_count"]),
        "eligibility selected physical-group count changed")
    require(len(selected_scenes) == int(
        eligibility["expected_selected_scene_count"]),
        "eligibility selected scene count changed")
    require(bool(selected), "eligibility selected no physical groups")
    audit = {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "input_physical_group_count": len(groups),
        "input_scene_count": len(input_scenes),
        "excluded_scene_count": len(excluded_scenes),
        "excluded_scenes": sorted(excluded_scenes),
        "excluded_individual_group_count": len(excluded_groups),
        "excluded_group_ids": sorted(excluded_groups),
        "selected_physical_group_count": len(selected),
        "selected_scene_count": len(selected_scenes),
        "selected_scenes": sorted(selected_scenes),
    }
    return selected, audit


def build(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = load_json_pinned(args.manifest, args.expected_manifest_sha256)
    split = load_json_pinned(args.split_manifest, args.expected_split_sha256)
    geometry_map = load_json_pinned(
        args.geometry_map, args.expected_geometry_map_sha256)
    require(manifest.get("schema_version")
            == "nlsr_v2_multistage_expert_candidate_manifest_v1",
            "unsupported expert manifest schema")
    require(manifest.get("input_roots", {}).get("episode_root")
            == str(args.episode_root),
            "episode root differs from the pinned manifest routing")
    require(geometry_map.get("schema_version") == "frozen_geometry_map_v1",
            "unsupported geometry-map schema")
    train_scenes = set(map(str, split["train"]))
    require(len(train_scenes) == 40, "frozen split no longer has 40 train scenes")
    groups = group_samples(manifest, train_scenes, args.max_scenes)
    require(bool(groups), "no Novel-B training groups selected")
    eligibility_audit = None
    eligibility_sha256 = None
    if args.eligibility_manifest is not None:
        require(args.max_scenes == 0,
                "eligibility manifest cannot be combined with max-scenes")
        eligibility = load_json_pinned(
            args.eligibility_manifest, args.expected_eligibility_sha256)
        groups, eligibility_audit = apply_group_eligibility(groups, eligibility)
        eligibility_sha256 = args.expected_eligibility_sha256
    episode_records = index_episode_records(manifest)
    episode_roots = (args.episode_root, args.episode_fallback_root)
    for root in episode_roots:
        require(root.is_dir() and not root.is_symlink(),
                f"episode root is missing or symbolic: {root}")

    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    incomplete = output.with_name(output.name + ".incomplete")
    require(not incomplete.exists(), f"incomplete output already exists: {incomplete}")
    incomplete.mkdir(parents=True)
    image_root = incomplete / "scan_rgb"
    image_root.mkdir()
    records: list[Mapping[str, Any]] = []
    current_scene = None
    simulator = None
    geometry_cache: dict[str, tuple[Path, Path, str]] = {}
    try:
        for group_index, (group_id, members) in enumerate(groups):
            scene = str(members[0]["scene"])
            if scene != current_scene:
                if simulator is not None:
                    simulator.close()
                geometry = geometry_paths(
                    scene, geometry_map, args.geometry_map.parent,
                    args.scene_root)
                geometry_cache[scene] = geometry
                simulator = make_simulator(geometry[0], geometry[1])
                current_scene = scene
            require(simulator is not None, "simulator did not initialize")
            pose, _metadata = read_state_pose(
                members[0], episode_roots, episode_records)
            start = pose.position
            snapped_start = np.asarray(
                simulator.pathfinder.snap_point(start), dtype=np.float64)
            require(np.isfinite(snapped_start).all()
                    and np.linalg.norm(snapped_start - start) <= 0.15,
                    f"state pose is off frozen navmesh: {group_id}")
            start = snapped_start
            gauge = deterministic_gauge_bin(group_id, salt=GAUGE_SALT)
            yaws = scan_yaws(pose.yaw_rad, gauge)
            group_slug = hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:20]
            group_image_root = image_root / scene / group_slug
            group_image_root.mkdir(parents=True)
            view_paths: list[str] = []
            view_hashes: list[str] = []
            for view_index, yaw in enumerate(yaws):
                rgb = render_rgb(simulator, start, float(yaw))
                path = group_image_root / f"view_{view_index}.jpg"
                Image.fromarray(rgb).save(path, format="JPEG", quality=95,
                                          subsampling=0)
                view_paths.append(str(path.relative_to(incomplete)))
                view_hashes.append(sha256_file(path))

            for member in members:
                goal = goal_world_position(
                    member, episode_roots, episode_records)
                snapped_goal = np.asarray(
                    simulator.pathfinder.snap_point(goal), dtype=np.float64)
                require(np.isfinite(snapped_goal).all()
                        and np.linalg.norm(snapped_goal - goal) <= 0.30,
                        f"goal is off frozen navmesh: {member['sample_id']}")
                field = dict(counterfactual_field(
                    simulator, start, snapped_goal, yaws))
                native_index = native_scan_index(gauge)
                native_advantage = field["advantages_m"][native_index]
                goal_path = verify_manifest_file(
                    member["goal"], episode_roots,
                    f"{member['sample_id']} goal")
                records.append({
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": member["sample_id"],
                    "goal_swap_pair_id": group_id,
                    "scene": scene,
                    "source_episode_id": member["source_episode_id"],
                    "state_name": member["state_name"],
                    "goal_variant": member["goal_variant"],
                    "goal_relative_path": str(member["goal"]["path"]),
                    "goal_content_sha256": member["goal"]["content_sha256"],
                    "scan_rgb_relative_paths": view_paths,
                    "scan_rgb_sha256": view_hashes,
                    "scan_yaw_rad": yaws.tolist(),
                    "gauge_bin": gauge,
                    "native_scan_index": native_index,
                    "native_advantage_m": native_advantage,
                    "state_floor_position_habitat": start.tolist(),
                    "state_base_yaw_rad": pose.yaw_rad,
                    "goal_floor_position_habitat": snapped_goal.tolist(),
                    "geometry_id": geometry_cache[scene][2],
                    **field,
                })
            if (group_index + 1) % 10 == 0:
                print(json.dumps({
                    "groups_complete": group_index + 1,
                    "groups_total": len(groups),
                    "scene": scene,
                }, sort_keys=True), flush=True)
    finally:
        if simulator is not None:
            simulator.close()

    dataset_path = incomplete / "dataset.jsonl"
    with dataset_path.open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record))

    pair_differences = []
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[str(record["goal_swap_pair_id"])].append(record)
    for pair_id, pair in by_pair.items():
        require(len(pair) == 2, f"output goal pair incomplete: {pair_id}")
        pair_differences.append(circular_bin_error(
            int(pair[0]["oracle_best_index"]),
            int(pair[1]["oracle_best_index"])))
    scenes = sorted({str(record["scene"]) for record in records})
    gauge_counts = Counter(int(record["gauge_bin"]) for record in records[::2])
    best_counts = Counter(int(record["oracle_best_index"]) for record in records)
    valid_counts = [sum(map(bool, record["candidate_valid"])) for record in records]
    faithful_counts = [
        sum(map(bool, record["candidate_action_faithful"]))
        for record in records
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "scope": (
            "train-only active C8 monocular scans; Habitat geometry is label-only; "
            "no development, final-reserved, blind, or closed-loop evaluation"),
        "configuration": {
            "camera_height_m": CAMERA_HEIGHT_M,
            "directions": NUM_DIRECTIONS,
            "gauge_salt": GAUGE_SALT,
            "hfov_deg": HFOV_DEG,
            "max_query_heading_error_deg": MAX_QUERY_HEADING_ERROR_DEG,
            "min_query_extent_m": MIN_QUERY_EXTENT_M,
            "query_radius_m": QUERY_RADIUS_M,
            "state_names": sorted(STATE_NAMES),
            "deployment_candidate_mask_used": False,
            "habitat_sim_version": EXPECTED_HABITAT_SIM_VERSION,
            "eligibility": eligibility_audit,
        },
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": args.expected_manifest_sha256,
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": args.expected_split_sha256,
            "geometry_map": str(args.geometry_map.resolve()),
            "geometry_map_sha256": args.expected_geometry_map_sha256,
            "eligibility_manifest": (
                str(args.eligibility_manifest.resolve())
                if args.eligibility_manifest is not None else None),
            "eligibility_manifest_sha256": eligibility_sha256,
            "episode_roots": [str(root.resolve()) for root in episode_roots],
        },
        "summary": {
            "scene_clusters": len(scenes),
            "scenes": scenes,
            "physical_scan_groups": len(by_pair),
            "goal_conditioned_rows": len(records),
            "rendered_rgb_views": len(by_pair) * NUM_DIRECTIONS,
            "gauge_bin_counts": {str(key): gauge_counts[key]
                                 for key in range(NUM_DIRECTIONS)},
            "teacher_best_bin_counts": {str(key): best_counts[key]
                                        for key in range(NUM_DIRECTIONS)},
            "candidate_valid_min": min(valid_counts),
            "candidate_valid_mean": float(np.mean(valid_counts)),
            "candidate_action_faithful_min": min(faithful_counts),
            "candidate_action_faithful_mean": float(np.mean(faithful_counts)),
            "goal_swap_pairs_best_bin_different": sum(
                difference > 0 for difference in pair_differences),
            "goal_swap_pairs_best_bin_at_least_90deg": sum(
                difference >= 2 for difference in pair_differences),
            "goal_swap_pair_count": len(pair_differences),
        },
        "dataset": {
            "relative_path": "dataset.jsonl",
            "sha256": sha256_file(dataset_path),
        },
    }
    report_path = incomplete / "report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    (incomplete / "report.json.sha256").write_text(
        f"{sha256_file(report_path)}  report.json\n", encoding="utf-8")
    incomplete.rename(output)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--geometry-map", type=Path, required=True)
    parser.add_argument("--expected-geometry-map-sha256", required=True)
    parser.add_argument("--eligibility-manifest", type=Path)
    parser.add_argument("--expected-eligibility-sha256")
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--episode-fallback-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(args.max_scenes >= 0, "max-scenes must be non-negative")
    require((args.eligibility_manifest is None)
            == (args.expected_eligibility_sha256 is None),
            "eligibility manifest and expected SHA256 must be supplied together")
    report = build(args)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (DatasetBuildError, OSError, ValueError, KeyError) as error:
        print(json.dumps({
            "status": "failed_closed",
            "error": str(error),
        }, sort_keys=True))
        raise SystemExit(2)
