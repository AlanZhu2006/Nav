"""Fail-closed real Habitat + audited-NavDP backend for paired H24 labels.

This module is the *impure* adapter underneath :mod:`novel_rollout_protocol_v2`.
The protocol module remains responsible for cross-arm equality and label
derivation; this adapter is responsible for proving that one real arm obeyed
the experiment:

* load, never recompute, a content-addressed navmesh;
* reset NavDP and replay only the manifest's preceding FIFO observations;
* restore the exact expert pose and use the frozen current RGB/depth/goal;
* append that current observation exactly once through ``/navdp_plan_atomic``;
* sample exactly one seeded diffusion plan per commitment;
* keep a residual candidate fixed in Habitat world coordinates and reproject
  it at every planning boundary; and
* execute eight kinematic pure-pursuit steps without touching NavDP memory.

The factual/counterfactual goal world position is privileged label metadata.
It is used only for reachability and geodesic progress after a plan has already
been sampled.  It is never placed in an HTTP request or candidate feature.

The concrete classes deliberately use small protocols around HTTP and Habitat.
Unit tests therefore exercise the exact production orchestration with fake
servers/pathfinders, while the real run uses ``RequestsJsonTransport`` and
``PinnedHabitatRuntime`` without importing any mutable evaluator script.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

try:
    from MemNavData.build_novel_candidate_manifest import (
        PARQUET_PREFIX_COLUMNS,
        ROUTED_SCHEMA_VERSION,
        SCHEMA_VERSION,
        canonical_json_bytes as manifest_canonical_json_bytes,
        load_parquet_rows,
    )
    from MemNavData.habitat_rollout_primitives import (
        DATA_TO_HABITAT_ROTATION,
        FrozenGeometryIdentity,
        HabitatPlanarPose,
        load_pinned_navmesh_for_collector,
        local_forward_left_to_world,
        parquet_data_pose_to_habitat,
        wrap_yaw,
    )
    from MemNavData.novel_rollout_protocol_v2 import (
        CandidateArm,
        CommitmentReceipt,
        FrozenDecisionState,
        PlanReceipt,
        PlanRequest,
        PreparationReceipt,
        RolloutBackend,
        RuntimeGeometrySpec,
        StepReceipt,
        canonical_pose_sha256,
        canonical_runtime_geometry_signature,
        canonical_sha256,
        world_goal_to_local,
    )
except ImportError:  # direct ``python MemNavData/<script>.py`` execution
    from build_novel_candidate_manifest import (  # type: ignore
        PARQUET_PREFIX_COLUMNS,
        ROUTED_SCHEMA_VERSION,
        SCHEMA_VERSION,
        canonical_json_bytes as manifest_canonical_json_bytes,
        load_parquet_rows,
    )
    from habitat_rollout_primitives import (  # type: ignore
        DATA_TO_HABITAT_ROTATION,
        FrozenGeometryIdentity,
        HabitatPlanarPose,
        load_pinned_navmesh_for_collector,
        local_forward_left_to_world,
        parquet_data_pose_to_habitat,
        wrap_yaw,
    )
    from novel_rollout_protocol_v2 import (  # type: ignore
        CandidateArm,
        CommitmentReceipt,
        FrozenDecisionState,
        PlanReceipt,
        PlanRequest,
        PreparationReceipt,
        RolloutBackend,
        RuntimeGeometrySpec,
        StepReceipt,
        canonical_pose_sha256,
        canonical_runtime_geometry_signature,
        canonical_sha256,
        world_goal_to_local,
    )


BACKEND_PROTOCOL = "real_habitat_navdp_h24_v2"
ATOMIC_PLAN_PROTOCOL = "navdp_native_first_atomic_plan_v2"
MEMORY_AUDIT_PROTOCOL = "navdp_native_first_fifo_v1"
MULTISTAGE_MANIFEST_SCHEMA_VERSION = (
    "nlsr_v2_multistage_expert_candidate_manifest_v1")
SUPPORTED_MANIFEST_SCHEMAS = frozenset((
    SCHEMA_VERSION,
    ROUTED_SCHEMA_VERSION,
    MULTISTAGE_MANIFEST_SCHEMA_VERSION,
))
GOAL_ROLE_BINDINGS = {
    "B": ("goal_b", "goal_1.jpg", 0, "novel"),
    "C": ("goal_c", "goal_2.jpg", 1, "revisit"),
}
ATOMIC_RECEIPT_FIELDS = frozenset((
    "protocol",
    "mode",
    "diffusion_seed",
    "diffusion_call_count",
    "goal_sha256",
    "goal_item_sha256",
    "current_sha256",
    "current_item_sha256",
    "fifo_before_sha256",
    "fifo_after_append_sha256",
    "fifo_item_sha256_before",
    "fifo_item_sha256",
    "fifo_lengths_before",
    "fifo_lengths_after",
    "point_goal_sha256",
    "critic_max",
    "stop_threshold",
    "low_critic_fallback_applied",
    "raw_selected_trajectory",
    "executable_trajectory",
    "inference_fifo_unchanged",
    "append_count_per_environment",
    "receipt_sha256",
))


class RealH24BackendError(RuntimeError):
    """An external receipt cannot prove the requested H24 experiment."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RealH24BackendError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_string(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA256",
    )
    return value


def _strict_mapping(
    value: object,
    required: Sequence[str],
    label: str,
    *,
    allow_extra: bool = False,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    missing = set(required) - set(value)
    extra = set(value) - set(required)
    _require(not missing, f"{label} is missing fields {sorted(missing)}")
    if not allow_extra:
        _require(not extra, f"{label} has unexpected fields {sorted(extra)}")
    return value  # type: ignore[return-value]


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RealH24BackendError(f"{label} is not numeric") from error
    _require(array.shape == shape, f"{label} must have shape {shape}")
    _require(bool(np.isfinite(array).all()), f"{label} contains NaN or infinity")
    return array.copy()


def _verify_file_record(
    record: object,
    root: Path,
    label: str,
) -> tuple[Path, bytes]:
    row = _strict_mapping(
        record,
        ("path", "path_sha256", "bytes", "content_sha256"),
        label,
    )
    relative = row["path"]
    _require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute(),
        f"{label}.path must be a non-empty relative path",
    )
    _require(
        sha256_bytes(relative.encode("utf-8")) == row["path_sha256"],
        f"{label}.path hash mismatch",
    )
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise RealH24BackendError(f"{label} escapes or is missing: {path}") from error
    _require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    payload = resolved.read_bytes()
    _require(
        isinstance(row["bytes"], int)
        and not isinstance(row["bytes"], bool)
        and len(payload) == row["bytes"],
        f"{label} byte length changed",
    )
    expected = _sha256_string(row["content_sha256"], f"{label}.content_sha256")
    _require(sha256_bytes(payload) == expected, f"{label} content changed")
    return resolved, payload


def _sequence_sha(value: object) -> str:
    return sha256_bytes(manifest_canonical_json_bytes(value))


@dataclass(frozen=True)
class EncodedObservation:
    """Wire-ready RGB JPEG and uint16/1e4 depth PNG."""

    rgb_jpeg: bytes
    depth_png: bytes

    def __post_init__(self) -> None:
        _require(bool(self.rgb_jpeg), "RGB JPEG cannot be empty")
        _require(bool(self.depth_png), "depth PNG cannot be empty")

    @property
    def rgb_sha256(self) -> str:
        return sha256_bytes(self.rgb_jpeg)

    @property
    def depth_sha256(self) -> str:
        return sha256_bytes(self.depth_png)


@dataclass(frozen=True)
class FrozenStateAssets:
    """Byte-exact inputs plus label-only geometry for one manifest sample."""

    state: FrozenDecisionState
    sample_id: str
    manifest_sha256: str
    camera_intrinsic: tuple[tuple[float, float, float], ...]
    camera_height_m: float
    replay_frame_indices: tuple[int, ...]
    replay_rgb_jpegs: tuple[bytes, ...]
    frozen_current: EncodedObservation
    goal_jpeg: bytes
    start_pose: HabitatPlanarPose
    label_goal_world_xyz_m: tuple[float, float, float]
    geometry_identity: FrozenGeometryIdentity
    glb_path: Path
    navmesh_path: Path

    def __post_init__(self) -> None:
        _sha256_string(self.manifest_sha256, "manifest_sha256")
        _require(self.sample_id == self.state.state_id, "sample/state id mismatch")
        intrinsic = _finite_array(self.camera_intrinsic, (3, 3), "camera intrinsic")
        _require(intrinsic[2, 2] != 0.0, "camera intrinsic is singular")
        _require(
            math.isfinite(float(self.camera_height_m))
            and float(self.camera_height_m) > 0.0,
            "camera height must be finite and positive",
        )
        _require(
            len(self.replay_frame_indices) == len(self.replay_rgb_jpegs),
            "replay indices/bytes length mismatch",
        )
        _require(
            tuple(sorted(set(self.replay_frame_indices)))
            == self.replay_frame_indices,
            "replay frame indices must be unique and increasing",
        )
        _require(
            all(bool(payload) for payload in self.replay_rgb_jpegs),
            "replay RGB payload cannot be empty",
        )
        _require(
            self.frozen_current.rgb_sha256 == self.state.current_rgb_sha256,
            "frozen current RGB hash disagrees with state",
        )
        _require(
            self.frozen_current.depth_sha256 == self.state.current_depth_sha256,
            "frozen current depth hash disagrees with state",
        )
        _require(
            sha256_bytes(self.goal_jpeg) == self.state.goal_sha256,
            "goal bytes disagree with frozen state",
        )
        _require(
            canonical_pose_sha256((
                self.start_pose.x_m,
                self.start_pose.z_m,
                self.start_pose.yaw_rad,
            )) == self.state.start_pose_sha256,
            "start pose disagrees with frozen state",
        )
        _finite_array(self.label_goal_world_xyz_m, (3,), "label goal world point")
        _require(
            self.geometry_identity.glb_sha256 == self.state.environment_sha256,
            "geometry identity GLB differs from state",
        )
        _require(
            self.geometry_identity.navmesh_sha256 == self.state.navmesh_sha256,
            "geometry identity navmesh differs from state",
        )


def load_frozen_manifest(
    path: Path | str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    """Load a canonical v1/v2 or multistage manifest under an external pin."""
    source = Path(path)
    expected = _sha256_string(expected_sha256, "expected manifest SHA256")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealH24BackendError(f"cannot read manifest: {source}") from error
    _require(sha256_bytes(raw) == expected, "manifest SHA256 pin mismatch")
    _require(isinstance(payload, Mapping), "manifest must be an object")
    _require(
        payload.get("schema_version") in SUPPORTED_MANIFEST_SCHEMAS,
        f"unsupported manifest schema {payload.get('schema_version')!r}",
    )
    _require(
        raw == manifest_canonical_json_bytes(payload),
        "manifest is not in canonical byte encoding",
    )
    return payload


def _resolve_roots(
    manifest: Mapping[str, Any],
    overrides: Mapping[str, Path | str] | None,
) -> dict[str, Path]:
    roots = _strict_mapping(
        manifest.get("input_roots"),
        ("episode_root", "environment_root", "navmesh_root"),
        "manifest.input_roots",
        allow_extra=True,
    )
    result = {}
    overrides = {} if overrides is None else dict(overrides)
    _require(
        set(overrides) <= {"episode_root", "environment_root", "navmesh_root"},
        "root overrides contain an unsupported root",
    )
    for name in ("episode_root", "environment_root", "navmesh_root"):
        value = overrides.get(name, roots[name])
        path = Path(value).resolve()
        _require(path.is_dir(), f"{name} is not a directory: {path}")
        result[name] = path
    return result


def _episode_record(scene: Mapping[str, Any], episode_name: str) -> Mapping[str, Any]:
    episodes = scene.get("selected_episodes")
    _require(isinstance(episodes, list), "scene selected_episodes is invalid")
    matches = [row for row in episodes
               if isinstance(row, Mapping) and row.get("episode") == episode_name]
    _require(len(matches) == 1, f"episode record not unique: {episode_name}")
    return matches[0]


def _verify_modality_sequence(
    root: Path,
    source_episode_id: str,
    modality: str,
    suffix: str,
    exclusive_end: int,
    expected: Mapping[str, Any],
) -> None:
    expected = _strict_mapping(
        expected,
        ("path_sequence_sha256", "content_sequence_sha256"),
        f"causal {modality} sequence",
    )
    relative_root = (
        Path(source_episode_id)
        / "videos/chunk-000"
        / f"observation.images.{modality}"
    )
    path_rows = []
    content_rows = []
    for frame in range(exclusive_end):
        relative = (relative_root / f"{frame}{suffix}").as_posix()
        path = root / relative
        _require(path.is_file(), f"causal {modality} frame is missing: {path}")
        payload = path.read_bytes()
        path_rows.append(relative)
        content_rows.append({
            "path": relative,
            "bytes": len(payload),
            "content_sha256": sha256_bytes(payload),
        })
    _require(
        _sequence_sha(path_rows) == expected.get("path_sequence_sha256"),
        f"causal {modality} path sequence changed",
    )
    _require(
        _sequence_sha(content_rows) == expected.get("content_sequence_sha256"),
        f"causal {modality} content sequence changed",
    )


def load_state_assets_from_manifest(
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    sample_id: str,
    geometry_identity_path: Path | str,
    *,
    root_overrides: Mapping[str, Path | str] | None = None,
    legacy_camera_height_m: float | None = None,
) -> FrozenStateAssets:
    """Materialize one audited expert decision without reading future images.

    Only the causal image/depth prefix and pose columns are verified.  The goal
    position is loaded from the selected goal episode's metadata but is stored
    in the explicitly label-only field of :class:`FrozenStateAssets`.
    """
    manifest = load_frozen_manifest(manifest_path, expected_manifest_sha256)
    roots = _resolve_roots(manifest, root_overrides)
    samples = manifest.get("samples")
    _require(isinstance(samples, list), "manifest.samples must be a list")
    matches = [row for row in samples
               if isinstance(row, Mapping) and row.get("sample_id") == sample_id]
    _require(len(matches) == 1, f"sample_id is not unique: {sample_id}")
    sample = matches[0]
    manifest_schema = manifest.get("schema_version")
    goal_role = sample.get("goal_role", "B")
    _require(
        isinstance(goal_role, str) and goal_role in GOAL_ROLE_BINDINGS,
        "sample goal_role must be B or C",
    )
    _require(
        goal_role != "C" or manifest_schema == MULTISTAGE_MANIFEST_SCHEMA_VERSION,
        "Goal C is only valid in a multistage manifest",
    )
    goal_record_key, goal_filename, goal_index, expected_goal_kind = (
        GOAL_ROLE_BINDINGS[goal_role]
    )
    scene_name = sample.get("scene")
    _require(isinstance(scene_name, str) and scene_name, "sample scene is invalid")
    scenes = manifest.get("scenes")
    _require(isinstance(scenes, list), "manifest.scenes must be a list")
    scene_matches = [row for row in scenes
                     if isinstance(row, Mapping) and row.get("scene") == scene_name]
    _require(len(scene_matches) == 1, f"scene record not unique: {scene_name}")
    scene = scene_matches[0]

    glb_path, _ = _verify_file_record(
        scene.get("environment"), roots["environment_root"], "scene.environment")
    navmesh_path, _ = _verify_file_record(
        scene.get("navmesh"), roots["navmesh_root"], "scene.navmesh")
    identity = FrozenGeometryIdentity.load_json(geometry_identity_path)
    _require(identity.glb_sha256 == sha256_file(glb_path), "identity/GLB mismatch")
    _require(
        identity.navmesh_sha256 == sha256_file(navmesh_path),
        "identity/navmesh mismatch",
    )

    source_name = sample.get("source_episode")
    goal_name = sample.get("goal_episode")
    source_id = sample.get("source_episode_id")
    _require(
        isinstance(source_name, str)
        and isinstance(goal_name, str)
        and source_id == f"{scene_name}/{source_name}",
        "sample episode identity is invalid",
    )
    default_goal_variant = (
        "factual" if goal_name == source_name else "counterfactual")
    goal_variant = sample.get("goal_variant", default_goal_variant)
    _require(
        goal_variant in ("factual", "counterfactual")
        and ((goal_variant == "factual") == (goal_name == source_name)),
        "sample factual/counterfactual goal binding is invalid",
    )
    goal_source_id = sample.get(
        "goal_source_episode_id", f"{scene_name}/{goal_name}")
    _require(
        goal_source_id == f"{scene_name}/{goal_name}",
        "sample goal episode identity is invalid",
    )
    source_record = _episode_record(scene, source_name)
    goal_record = _episode_record(scene, goal_name)
    source_meta_path, source_meta_raw = _verify_file_record(
        source_record.get("metadata"), roots["episode_root"], "source metadata")
    parquet_path, _ = _verify_file_record(
        source_record.get("parquet"), roots["episode_root"], "source parquet")
    goal_meta_path, goal_meta_raw = _verify_file_record(
        goal_record.get("metadata"), roots["episode_root"], "goal metadata")
    goal_path, goal_jpeg = _verify_file_record(
        sample.get("goal"), roots["episode_root"], "sample goal")
    expected_goal_path, expected_goal_jpeg = _verify_file_record(
        goal_record.get(goal_record_key),
        roots["episode_root"],
        f"goal episode {goal_record_key}",
    )
    _require(
        goal_path == expected_goal_path and goal_jpeg == expected_goal_jpeg,
        f"sample Goal {goal_role} is not {goal_record_key} of its declared "
        "goal episode",
    )
    _require(
        goal_path.name == goal_filename,
        f"Goal {goal_role} must bind to {goal_filename}",
    )
    try:
        source_meta = json.loads(source_meta_raw)
        goal_meta = json.loads(goal_meta_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealH24BackendError("episode metadata is invalid JSON") from error

    decision_frame = sample.get("decision_frame")
    _require(
        isinstance(decision_frame, int)
        and not isinstance(decision_frame, bool)
        and decision_frame >= 1,
        "sample decision_frame is invalid",
    )
    current_index = decision_frame - 1
    prefix = _strict_mapping(
        sample.get("causal_prefix"),
        (
            "exclusive_end_frame", "frame_count", "modalities",
            "parquet_columns", "parquet_row_count", "parquet_rows_sha256",
            "causal_prefix_sha256",
        ),
        "sample.causal_prefix",
    )
    _require(
        prefix["exclusive_end_frame"] == decision_frame
        and prefix["frame_count"] == decision_frame
        and prefix["parquet_row_count"] == decision_frame,
        "sample causal prefix boundary changed",
    )
    modalities = _strict_mapping(prefix["modalities"], ("rgb", "depth"), "modalities")
    _verify_modality_sequence(
        roots["episode_root"], source_id, "rgb", ".jpg", decision_frame,
        modalities["rgb"],
    )
    _verify_modality_sequence(
        roots["episode_root"], source_id, "depth", ".png", decision_frame,
        modalities["depth"],
    )

    n_frames = source_record.get("n_frames")
    _require(isinstance(n_frames, int), "source n_frames is invalid")
    rows = load_parquet_rows(parquet_path, n_frames)
    _require(
        list(PARQUET_PREFIX_COLUMNS) == prefix["parquet_columns"],
        "parquet causal columns changed",
    )
    _require(
        _sequence_sha(rows[:decision_frame]) == prefix["parquet_rows_sha256"],
        "parquet causal prefix changed",
    )
    causal_body = {
        "frame_count": decision_frame,
        "rgb": modalities["rgb"],
        "depth": modalities["depth"],
        "parquet_rows_sha256": prefix["parquet_rows_sha256"],
    }
    _require(
        sha256_bytes(manifest_canonical_json_bytes(causal_body))
        == prefix["causal_prefix_sha256"],
        "causal prefix aggregate hash changed",
    )

    fifo = _strict_mapping(
        sample.get("navdp_fifo"),
        (
            "memory_size", "exec_horizon", "left_zero_pad_count",
            "replay_frame_indices", "current_frame_index",
            "after_append_frame_indices", "path_sequence_sha256",
            "content_sequence_sha256", "fifo_sha256",
        ),
        "sample.navdp_fifo",
    )
    replay_indices_raw = fifo["replay_frame_indices"]
    after_indices_raw = fifo["after_append_frame_indices"]
    _require(
        isinstance(replay_indices_raw, list)
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in replay_indices_raw),
        "FIFO replay indices are invalid",
    )
    _require(
        isinstance(after_indices_raw, list)
        and after_indices_raw == [*replay_indices_raw, current_index]
        and fifo["current_frame_index"] == current_index,
        "FIFO current frame is duplicated, omitted, or reordered",
    )
    replay_indices = tuple(replay_indices_raw)
    rgb_root = roots["episode_root"] / source_id / (
        "videos/chunk-000/observation.images.rgb")
    depth_root = roots["episode_root"] / source_id / (
        "videos/chunk-000/observation.images.depth")
    rgb_records = []
    replay_payloads = []
    current_rgb = b""
    for frame in after_indices_raw:
        path = rgb_root / f"{frame}.jpg"
        _require(path.is_file(), f"FIFO RGB is missing: {path}")
        raw = path.read_bytes()
        relative = path.relative_to(roots["episode_root"]).as_posix()
        rgb_records.append({
            "path": relative,
            "bytes": len(raw),
            "content_sha256": sha256_bytes(raw),
        })
        if frame == current_index:
            current_rgb = raw
        else:
            replay_payloads.append(raw)
    _require(
        _sequence_sha([row["path"] for row in rgb_records])
        == fifo["path_sequence_sha256"],
        "FIFO path sequence changed",
    )
    _require(
        _sequence_sha(rgb_records) == fifo["content_sequence_sha256"],
        "FIFO content sequence changed",
    )
    fifo_body = {key: fifo[key] for key in (
        "memory_size", "exec_horizon", "left_zero_pad_count",
        "replay_frame_indices", "current_frame_index",
        "after_append_frame_indices", "path_sequence_sha256",
        "content_sequence_sha256",
    )}
    _require(
        sha256_bytes(manifest_canonical_json_bytes(fifo_body)) == fifo["fifo_sha256"],
        "FIFO aggregate hash changed",
    )
    current_depth_path = depth_root / f"{current_index}.png"
    _require(current_depth_path.is_file(), "frozen current depth is missing")
    current_depth = current_depth_path.read_bytes()
    state_frame_path, state_frame_raw = _verify_file_record(
        sample.get("state_frame"), roots["episode_root"], "sample state_frame")
    _require(
        state_frame_path == (rgb_root / f"{current_index}.jpg").resolve()
        and state_frame_raw == current_rgb,
        "sample state_frame is not the frozen current RGB",
    )

    row = rows[current_index]
    explicit_height = legacy_camera_height_m
    if explicit_height is not None:
        _require(
            isinstance(explicit_height, (int, float))
            and not isinstance(explicit_height, bool)
            and math.isfinite(float(explicit_height))
            and float(explicit_height) > 0.0,
            "explicit legacy camera height must be finite and positive",
        )
        explicit_height = float(explicit_height)
    metadata_height = source_meta.get("camera_height_m")
    if metadata_height is None:
        _require(
            explicit_height is not None,
            "episode metadata omits camera_height_m; an explicit pinned "
            "legacy_camera_height_m is required",
        )
        camera_height = explicit_height
    else:
        _require(
            isinstance(metadata_height, (int, float))
            and not isinstance(metadata_height, bool)
            and math.isfinite(float(metadata_height))
            and float(metadata_height) > 0.0,
            "metadata camera_height_m must be finite and positive",
        )
        camera_height = float(metadata_height)
        _require(
            explicit_height is None
            or math.isclose(
                explicit_height, camera_height, rel_tol=0.0, abs_tol=1e-9),
            "explicit legacy camera height conflicts with episode metadata",
        )
    frame_convention = source_meta.get("frame_convention")
    _require(isinstance(frame_convention, str), "frame convention is missing")
    start_pose = parquet_data_pose_to_habitat(
        row["action"],
        row["observation.camera_extrinsic"],
        camera_height_m=camera_height,
        frame_convention=frame_convention,
    )
    intrinsic = _finite_array(
        row["observation.camera_intrinsic"], (3, 3), "camera intrinsic")
    goals = goal_meta.get("goals")
    _require(
        isinstance(goals, list)
        and len(goals) > goal_index
        and isinstance(goals[goal_index], Mapping)
        and goals[goal_index].get("kind") == expected_goal_kind,
        f"goal episode has no {expected_goal_kind.title()} Goal "
        f"{goal_role} label",
    )
    goal_data = _finite_array(
        goals[goal_index].get("pos"),
        (3,),
        f"Goal {goal_role} position",
    )
    goal_habitat = DATA_TO_HABITAT_ROTATION @ goal_data

    settings = identity.navmesh_settings
    runtime_geometry = RuntimeGeometrySpec(
        habitat_sim_version=identity.habitat_sim_version,
        agent_radius_m=identity.agent_radius_m,
        agent_height_m=identity.agent_height_m,
        agent_max_climb_m=float(settings["agent_max_climb"]),
        agent_max_slope_deg=float(settings["agent_max_slope"]),
        navmesh_source="loaded_frozen",
        navmesh_settings_sha256=identity.navmesh_settings_sha256,
    )
    state = FrozenDecisionState(
        state_id=sample_id,
        session_id=str(source_id),
        goal_epoch=f"{goal_role}:{sha256_bytes(goal_jpeg)[:16]}",
        goal_sha256=sha256_bytes(goal_jpeg),
        manifest_fifo_sha256=_sha256_string(fifo["fifo_sha256"], "fifo_sha256"),
        current_rgb_sha256=sha256_bytes(current_rgb),
        current_depth_sha256=sha256_bytes(current_depth),
        start_pose_sha256=canonical_pose_sha256((
            start_pose.x_m, start_pose.z_m, start_pose.yaw_rad)),
        environment_id=scene_name,
        environment_sha256=identity.glb_sha256,
        navmesh_sha256=identity.navmesh_sha256,
        runtime_geometry=runtime_geometry,
    )
    return FrozenStateAssets(
        state=state,
        sample_id=sample_id,
        manifest_sha256=expected_manifest_sha256,
        camera_intrinsic=tuple(tuple(float(value) for value in row)
                               for row in intrinsic),
        camera_height_m=camera_height,
        replay_frame_indices=replay_indices,
        replay_rgb_jpegs=tuple(replay_payloads),
        frozen_current=EncodedObservation(current_rgb, current_depth),
        goal_jpeg=goal_jpeg,
        start_pose=start_pose,
        label_goal_world_xyz_m=tuple(float(value) for value in goal_habitat),
        geometry_identity=identity,
        glb_path=glb_path,
        navmesh_path=navmesh_path,
    )


class JsonTransport(Protocol):
    """Small HTTP surface used by the backend and fakes."""

    def post_json(self, endpoint: str, payload: Mapping[str, object]) -> Mapping[str, Any]: ...

    def post_multipart(
        self,
        endpoint: str,
        *,
        files: Mapping[str, tuple[str, bytes, str]],
        data: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]: ...

    def request_json(self, endpoint: str) -> Mapping[str, Any]: ...


class RequestsJsonTransport:
    """Production HTTP transport with bounded timeouts and strict JSON maps."""

    def __init__(self, base_url: str, *, timeout_s: float = 180.0) -> None:
        _require(base_url.startswith(("http://", "https://")), "bad server URL")
        _require(math.isfinite(timeout_s) and timeout_s > 0.0, "bad timeout")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _decode(self, response: Any, endpoint: str) -> Mapping[str, Any]:
        try:
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            status = getattr(response, "status_code", "unknown")
            body = str(getattr(response, "text", ""))[:500]
            raise RealH24BackendError(
                f"NavDP {endpoint} failed ({status}): {body}") from error
        _require(isinstance(payload, Mapping), f"NavDP {endpoint} returned non-object JSON")
        return payload

    def post_json(self, endpoint: str, payload: Mapping[str, object]) -> Mapping[str, Any]:
        import requests
        response = requests.post(
            f"{self.base_url}{endpoint}", json=dict(payload), timeout=self.timeout_s)
        return self._decode(response, endpoint)

    def post_multipart(
        self,
        endpoint: str,
        *,
        files: Mapping[str, tuple[str, bytes, str]],
        data: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        import requests
        request_files = {
            name: (filename, payload, media_type)
            for name, (filename, payload, media_type) in files.items()
        }
        response = requests.post(
            f"{self.base_url}{endpoint}",
            files=request_files,
            data={} if data is None else dict(data),
            timeout=self.timeout_s,
        )
        return self._decode(response, endpoint)

    def request_json(self, endpoint: str) -> Mapping[str, Any]:
        import requests
        response = requests.get(
            f"{self.base_url}{endpoint}", timeout=self.timeout_s)
        return self._decode(response, endpoint)


class HabitatRuntime(Protocol):
    """Runtime operations used by real and synthetic Habitat adapters."""

    @property
    def pose(self) -> HabitatPlanarPose: ...

    @property
    def geometry_identity(self) -> FrozenGeometryIdentity: ...

    def reset_to(self, pose: HabitatPlanarPose) -> None: ...

    def set_pose(self, pose: HabitatPlanarPose) -> None: ...

    def render_encoded(self) -> EncodedObservation: ...

    def snap_point(self, world_xyz: Sequence[float]) -> np.ndarray: ...

    def is_navigable(self, world_xyz: Sequence[float]) -> bool: ...

    def geodesic_distance(self, goal_world_xyz: Sequence[float]) -> tuple[bool, float | None]: ...


class PinnedHabitatRuntime:
    """Concrete Habitat-Sim runtime that can only load a frozen navmesh.

    ``camera_height_m`` is added to the decoded floor pose because the current
    generated episodes used a zero-height sensor mount and placed the agent
    state at camera height.  This matches their frozen RGB/depth bytes.
    """

    def __init__(
        self,
        simulator: object,
        *,
        identity: FrozenGeometryIdentity,
        glb_path: Path | str,
        navmesh_path: Path | str,
        navmesh_settings: object,
        habitat_sim_version: str,
        agent_radius_m: float,
        agent_height_m: float,
        camera_height_m: float,
        rgb_sensor_key: str = "color",
        depth_sensor_key: str = "depth",
    ) -> None:
        self.simulator = simulator
        self.identity = identity
        self.glb_path = Path(glb_path)
        self.navmesh_path = Path(navmesh_path)
        self.navmesh_settings = navmesh_settings
        self.camera_height_m = float(camera_height_m)
        self.rgb_sensor_key = rgb_sensor_key
        self.depth_sensor_key = depth_sensor_key
        self.pathfinder = load_pinned_navmesh_for_collector(
            simulator,
            identity=identity,
            glb_path=self.glb_path,
            navmesh_path=self.navmesh_path,
            habitat_sim_version=habitat_sim_version,
            agent_radius_m=agent_radius_m,
            agent_height_m=agent_height_m,
            navmesh_settings=navmesh_settings,
        )
        self._pose: HabitatPlanarPose | None = None

    @property
    def geometry_identity(self) -> FrozenGeometryIdentity:
        return self.identity

    @classmethod
    def create_frozen(
        cls,
        *,
        identity: FrozenGeometryIdentity,
        glb_path: Path | str,
        navmesh_path: Path | str,
        camera_intrinsic: Sequence[Sequence[float]],
        camera_height_m: float,
        rgb_sensor_key: str = "color",
        depth_sensor_key: str = "depth",
    ) -> "PinnedHabitatRuntime":
        """Construct the production simulator from the pinned scene itself.

        This path avoids accepting a simulator that may have been created for
        another GLB.  It reproduces the generated episodes' zero-offset sensor
        mount; :meth:`set_pose` raises the agent state by ``camera_height_m``.
        """
        try:
            import habitat_sim
            import magnum as mn
        except ImportError as error:  # pragma: no cover - real env only
            raise RealH24BackendError("habitat-sim and magnum are required") from error
        intrinsic = _finite_array(camera_intrinsic, (3, 3), "camera intrinsic")
        width = int(round(2.0 * float(intrinsic[0, 2])))
        height = int(round(2.0 * float(intrinsic[1, 2])))
        _require(width > 0 and height > 0, "camera resolution is invalid")
        _require(
            math.isclose(float(intrinsic[0, 2]), width / 2.0, abs_tol=1e-6)
            and math.isclose(float(intrinsic[1, 2]), height / 2.0, abs_tol=1e-6),
            "production collector requires the pinned centered camera model",
        )
        hfov_deg = math.degrees(
            2.0 * math.atan(float(intrinsic[0, 2]) / float(intrinsic[0, 0])))
        simulator_configuration = habitat_sim.SimulatorConfiguration()
        simulator_configuration.scene_id = str(Path(glb_path).resolve())
        simulator_configuration.enable_physics = False

        def sensor(uuid: str, sensor_type: object) -> object:
            specification = habitat_sim.CameraSensorSpec()
            specification.uuid = uuid
            specification.sensor_type = sensor_type
            specification.resolution = [height, width]
            specification.hfov = hfov_deg
            specification.position = mn.Vector3(0, 0, 0)
            return specification

        agent_configuration = habitat_sim.agent.AgentConfiguration()
        agent_configuration.sensor_specifications = [
            sensor(rgb_sensor_key, habitat_sim.SensorType.COLOR),
            sensor(depth_sensor_key, habitat_sim.SensorType.DEPTH),
        ]
        simulator = habitat_sim.Simulator(habitat_sim.Configuration(
            simulator_configuration, [agent_configuration]))
        version = getattr(habitat_sim, "__version__", None)
        if not isinstance(version, str) or not version:
            simulator.close()
            raise RealH24BackendError("Habitat-Sim does not expose __version__")
        try:
            return cls(
                simulator,
                identity=identity,
                glb_path=glb_path,
                navmesh_path=navmesh_path,
                navmesh_settings=identity.navmesh_settings,
                habitat_sim_version=version,
                agent_radius_m=identity.agent_radius_m,
                agent_height_m=identity.agent_height_m,
                camera_height_m=camera_height_m,
                rgb_sensor_key=rgb_sensor_key,
                depth_sensor_key=depth_sensor_key,
            )
        except BaseException:
            simulator.close()
            raise

    @property
    def pose(self) -> HabitatPlanarPose:
        _require(self._pose is not None, "Habitat pose has not been initialized")
        return self._pose

    def _agent_state(self, pose: HabitatPlanarPose) -> object:
        try:
            import habitat_sim
            import quaternion
        except ImportError as error:  # pragma: no cover - real env only
            raise RealH24BackendError(
                "habitat-sim and numpy-quaternion are required") from error
        state = habitat_sim.agent.AgentState()
        state.position = np.asarray([
            pose.x_m,
            pose.y_m + self.camera_height_m,
            pose.z_m,
        ], dtype=np.float64)
        state.rotation = quaternion.from_rotation_vector(
            [0.0, pose.yaw_rad, 0.0])
        return state

    def set_pose(self, pose: HabitatPlanarPose) -> None:
        _require(isinstance(pose, HabitatPlanarPose), "bad Habitat pose")
        agent = getattr(self.simulator, "get_agent")(0)
        agent.set_state(self._agent_state(pose))
        self._pose = pose

    def reset_to(self, pose: HabitatPlanarPose) -> None:
        # No Simulator.reset() call is needed: the collector owns a kinematic
        # state and physics is disabled.  Most importantly, no navmesh bake is
        # hidden behind reset.
        self.set_pose(pose)

    def render_encoded(self) -> EncodedObservation:
        from PIL import Image
        observations = self.simulator.get_sensor_observations()
        _require(
            self.rgb_sensor_key in observations
            and self.depth_sensor_key in observations,
            "Habitat sensor keys changed",
        )
        rgb = np.asarray(observations[self.rgb_sensor_key])
        _require(rgb.ndim == 3 and rgb.shape[-1] >= 3, "bad Habitat RGB render")
        rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
        depth = np.asarray(observations[self.depth_sensor_key], dtype=np.float32)
        _require(depth.ndim == 2 and bool(np.isfinite(depth).all()), "bad depth render")
        rgb_buffer = io.BytesIO()
        Image.fromarray(rgb).save(rgb_buffer, format="JPEG", quality=95)
        depth_buffer = io.BytesIO()
        depth_u16 = np.clip(depth * 10000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_u16).save(depth_buffer, format="PNG")
        return EncodedObservation(rgb_buffer.getvalue(), depth_buffer.getvalue())

    def snap_point(self, world_xyz: Sequence[float]) -> np.ndarray:
        return np.asarray(self.pathfinder.snap_point(world_xyz), dtype=np.float64)

    def is_navigable(self, world_xyz: Sequence[float]) -> bool:
        return bool(self.pathfinder.is_navigable(world_xyz))

    def geodesic_distance(
        self, goal_world_xyz: Sequence[float]
    ) -> tuple[bool, float | None]:
        try:
            import habitat_sim
        except ImportError as error:  # pragma: no cover - real env only
            raise RealH24BackendError("habitat-sim is required") from error
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = self.pose.position
        shortest.requested_end = _finite_array(
            goal_world_xyz, (3,), "geodesic goal")
        reachable = bool(self.pathfinder.find_path(shortest))
        if not reachable:
            return False, None
        distance = float(shortest.geodesic_distance)
        _require(math.isfinite(distance) and distance >= 0.0, "bad geodesic distance")
        return True, distance


@dataclass(frozen=True)
class PurePursuitConfig:
    v_max_m: float = 0.0376
    lookahead_m: float = 0.7
    min_radius_m: float = 0.40
    max_turn_deg: float = 4.5
    snap_tolerance_m: float = 0.06
    creep_fraction: float = 0.3

    def __post_init__(self) -> None:
        for name in (
            "v_max_m", "lookahead_m", "min_radius_m",
            "max_turn_deg", "snap_tolerance_m", "creep_fraction",
        ):
            value = float(getattr(self, name))
            _require(math.isfinite(value) and value > 0.0, f"bad controller {name}")
        _require(self.creep_fraction <= 1.0, "creep fraction exceeds one")


@dataclass(frozen=True)
class NavDPPlanDiagnostics:
    """Lossless policy outputs retained outside the protocol receipt.

    ``PlanReceipt`` intentionally contains only the execution/audit identity.
    Candidate feature construction also needs the native policy proposal set,
    so the real backend keeps these exact server outputs rather than silently
    reducing them to the selected path.
    """

    plan_sha256: str
    server_selected_trajectory_index: int | None
    raw_selected_trajectory: tuple[tuple[float, ...], ...]
    executable_trajectory: tuple[tuple[float, ...], ...]
    all_trajectory: object
    all_values: object
    critic_max: float
    stop_threshold: float
    low_critic_fallback_applied: bool
    server_receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_sha256": self.plan_sha256,
            "server_selected_trajectory_index": (
                self.server_selected_trajectory_index),
            "raw_selected_trajectory": [
                list(row) for row in self.raw_selected_trajectory],
            "executable_trajectory": [
                list(row) for row in self.executable_trajectory],
            "all_trajectory": self.all_trajectory,
            "all_values": self.all_values,
            "critic_max": self.critic_max,
            "stop_threshold": self.stop_threshold,
            "low_critic_fallback_applied": (
                self.low_critic_fallback_applied),
            "server_receipt_sha256": self.server_receipt_sha256,
        }


def _normalize_plan_trajectory(value: object) -> np.ndarray:
    try:
        trajectory = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RealH24BackendError("NavDP trajectory is not numeric") from error
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    _require(
        trajectory.ndim == 2
        and trajectory.shape == (24, 3)
        and bool(np.isfinite(trajectory).all()),
        f"unexpected NavDP trajectory shape {trajectory.shape}",
    )
    return trajectory.copy()


def _server_selected_index(
    selected: np.ndarray,
    all_trajectory: np.ndarray,
) -> int | None:
    """Find the selected candidate without guessing a changed tensor layout."""
    candidates = all_trajectory
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if candidates.ndim != 3 or candidates.shape[1:] != selected.shape:
        return None
    errors = np.max(np.abs(candidates - selected[None, ...]), axis=(1, 2))
    matches = np.where(errors <= 1e-10)[0]
    return int(matches[0]) if len(matches) == 1 else None


def _pose_tuple(pose: HabitatPlanarPose) -> tuple[float, float, float]:
    return (pose.x_m, pose.z_m, pose.yaw_rad)


def _audit_items(audit: Mapping[str, Any]) -> tuple[str, ...]:
    rows = audit.get("queue_item_sha256")
    _require(
        isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list),
        "memory audit is not a one-environment FIFO",
    )
    result = tuple(
        _sha256_string(value, "processed FIFO item SHA") for value in rows[0])
    _require(
        audit.get("queue_lengths") == [len(result)],
        "memory audit queue length disagrees with item hashes",
    )
    return result


class RealH24RolloutBackend(RolloutBackend):
    """One arm of a real same-state H24 experiment."""

    def __init__(
        self,
        assets: FrozenStateAssets,
        transport: JsonTransport,
        runtime: HabitatRuntime,
        *,
        expected_server_provenance: Mapping[str, str],
        stop_threshold: float,
        reset_seed: int = 0,
        controller: PurePursuitConfig = PurePursuitConfig(),
    ) -> None:
        self.assets = assets
        self.transport = transport
        self.runtime = runtime
        self.expected_server_provenance = dict(expected_server_provenance)
        _require(bool(self.expected_server_provenance), "server provenance pin is empty")
        for name, digest in self.expected_server_provenance.items():
            _require(isinstance(name, str) and name, "bad provenance key")
            _sha256_string(digest, f"server provenance {name}")
        _require(
            math.isfinite(float(stop_threshold)), "stop threshold must be finite")
        _require(
            isinstance(reset_seed, int) and not isinstance(reset_seed, bool)
            and 0 <= reset_seed < 2**63,
            "reset seed is invalid",
        )
        self.stop_threshold = float(stop_threshold)
        self.reset_seed = reset_seed
        self.controller = controller
        _require(
            runtime.geometry_identity == assets.geometry_identity,
            "Habitat runtime geometry identity differs from frozen assets",
        )
        self._prepared = False
        self._current = assets.frozen_current
        self._last_audit: Mapping[str, Any] | None = None
        self._plans: dict[str, tuple[np.ndarray, PlanReceipt]] = {}
        self._plan_diagnostics: dict[str, NavDPPlanDiagnostics] = {}
        self._diffusion_calls = 0
        self._processed_goal_sha256: str | None = None

    def _verify_provenance(self, payload: Mapping[str, Any], endpoint: str) -> None:
        provenance = payload.get("provenance")
        _require(
            isinstance(provenance, Mapping)
            and dict(provenance) == self.expected_server_provenance,
            f"{endpoint} server provenance mismatch",
        )

    def _memory_audit(self) -> Mapping[str, Any]:
        payload = self.transport.request_json("/memory_audit")
        _require(payload.get("algo") == "navdp", "memory audit is not NavDP")
        _require(
            payload.get("protocol") == MEMORY_AUDIT_PROTOCOL,
            "memory audit protocol changed",
        )
        self._verify_provenance(payload, "/memory_audit")
        _sha256_string(payload.get("fifo_sha256"), "processed FIFO SHA")
        _audit_items(payload)
        return payload

    @staticmethod
    def _same_audit(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        fields = (
            "memory_size", "queue_lengths", "queue_item_sha256",
            "padded_model_tensor_sha256", "fifo_sha256",
        )
        return all(first.get(field) == second.get(field) for field in fields)

    def prepare_arm(self, state: FrozenDecisionState) -> PreparationReceipt:
        _require(state == self.assets.state, "backend received the wrong frozen state")
        reset = self.transport.post_json("/navigator_reset", {
            "intrinsic": [list(row) for row in self.assets.camera_intrinsic],
            "stop_threshold": self.stop_threshold,
            "batch_size": 1,
            "seed": self.reset_seed,
        })
        _require(reset.get("algo") == "navdp", "NavDP reset failed")
        empty = self._memory_audit()
        _require(empty.get("queue_lengths") == [0], "NavDP reset left a non-empty FIFO")
        _require(_audit_items(empty) == (), "NavDP reset left FIFO items")

        for ordinal, (frame_index, payload) in enumerate(zip(
                self.assets.replay_frame_indices,
                self.assets.replay_rgb_jpegs)):
            replay = self.transport.post_multipart(
                "/memory_replay_step",
                files={"image": (f"{frame_index}.jpg", payload, "image/jpeg")},
            )
            _require(
                replay.get("algo") == "navdp"
                and replay.get("diffusion_sampled") is False,
                "FIFO replay sampled diffusion or reached the wrong server",
            )
            expected_length = min(ordinal + 1, 8)
            _require(
                replay.get("queue_lengths") == [expected_length]
                and replay.get("memory_size") == 8,
                "FIFO replay length changed",
            )
            replay_audit = self._memory_audit()
            _require(
                replay_audit.get("queue_lengths") == [expected_length],
                "processed FIFO disagrees after replay",
            )

        prepared_audit = self._memory_audit()
        expected_prepared_length = min(len(self.assets.replay_rgb_jpegs), 8)
        items = _audit_items(prepared_audit)
        _require(
            len(items) == expected_prepared_length,
            "prepared FIFO length differs from manifest replay",
        )
        # The manifest current image is intentionally absent here.  It enters
        # once, transactionally, in the first plan call.
        _require(
            len(self.assets.replay_frame_indices) == expected_prepared_length,
            "manifest replay unexpectedly exceeds NavDP memory size",
        )

        self.runtime.reset_to(self.assets.start_pose)
        _require(
            canonical_pose_sha256(_pose_tuple(self.runtime.pose))
            == state.start_pose_sha256,
            "Habitat did not restore the exact expert pose",
        )
        reachable, distance = self.runtime.geodesic_distance(
            self.assets.label_goal_world_xyz_m)
        if reachable:
            _require(
                distance is not None and math.isfinite(distance) and distance >= 0.0,
                "reachable start has invalid geodesic distance",
            )
        else:
            _require(distance is None, "unreachable start returned a distance")

        self._prepared = True
        self._current = self.assets.frozen_current
        self._last_audit = prepared_audit
        self._plans.clear()
        self._plan_diagnostics.clear()
        self._diffusion_calls = 0
        self._processed_goal_sha256 = None
        return PreparationReceipt(
            state_id=state.state_id,
            manifest_fifo_sha256=state.manifest_fifo_sha256,
            processed_fifo_sha256=str(prepared_audit["fifo_sha256"]),
            processed_fifo_item_sha256=items,
            queue_length=len(items),
            current_rgb_sha256=self._current.rgb_sha256,
            current_depth_sha256=self._current.depth_sha256,
            start_pose_sha256=canonical_pose_sha256(_pose_tuple(self.runtime.pose)),
            environment_sha256=state.environment_sha256,
            navmesh_sha256=state.navmesh_sha256,
            runtime_geometry_signature=canonical_runtime_geometry_signature(
                state.environment_sha256,
                state.navmesh_sha256,
                state.runtime_geometry,
            ),
            world_pose_xz_yaw=_pose_tuple(self.runtime.pose),
            initial_goal_distance_m=(float(distance) if reachable else None),
            goal_reachable=reachable,
            diffusion_calls=0,
        )

    def _verify_atomic_receipt(
        self,
        request: PlanRequest,
        response: Mapping[str, Any],
        before: Mapping[str, Any],
        local_goal: tuple[float, float] | None,
    ) -> tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        bool,
    ]:
        self._verify_provenance(response, "/navdp_plan_atomic")
        receipt = response.get("receipt")
        _require(isinstance(receipt, Mapping), "atomic plan omitted its receipt")
        _require(
            set(receipt) == ATOMIC_RECEIPT_FIELDS,
            "atomic receipt schema/fields changed",
        )
        _require(receipt.get("protocol") == ATOMIC_PLAN_PROTOCOL, "atomic plan protocol changed")
        expected_mode = "native" if request.candidate_type == "native" else "image_point"
        _require(receipt.get("mode") == expected_mode, "atomic plan used the wrong branch")
        _require(
            receipt.get("diffusion_seed") == request.diffusion_seed
            and receipt.get("diffusion_call_count") == 1,
            "atomic plan seed/call count mismatch",
        )
        _require(
            receipt.get("append_count_per_environment") == 1
            and receipt.get("inference_fifo_unchanged") is True,
            "atomic plan did not prove a single append/read-only inference",
        )
        claimed_receipt_sha = _sha256_string(
            receipt.get("receipt_sha256"), "atomic receipt SHA")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        _require(
            canonical_sha256(unsigned) == claimed_receipt_sha,
            "atomic receipt canonical hash mismatch",
        )

        executable = _normalize_plan_trajectory(response.get("trajectory"))
        raw_selected = _normalize_plan_trajectory(
            response.get("raw_selected_trajectory"))
        receipt_executable = _normalize_plan_trajectory(
            receipt.get("executable_trajectory"))
        receipt_raw = _normalize_plan_trajectory(
            receipt.get("raw_selected_trajectory"))
        _require(
            bool(np.array_equal(executable, receipt_executable))
            and bool(np.array_equal(raw_selected, receipt_raw)),
            "atomic response trajectories disagree with its signed receipt",
        )
        try:
            all_trajectory = np.asarray(
                response.get("all_trajectory"), dtype=np.float64)
            all_values = np.asarray(
                response.get("all_values"), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise RealH24BackendError(
                "atomic candidate outputs are not numeric") from error
        _require(
            all_trajectory.ndim == 4
            and all_trajectory.shape[0] == 1
            and all_trajectory.shape[1] >= 1
            and all_trajectory.shape[-2:] == (24, 3)
            and all_values.shape == all_trajectory.shape[:2]
            and bool(np.isfinite(all_trajectory).all())
            and bool(np.isfinite(all_values).all()),
            "atomic candidate outputs changed shape or contain non-finite values",
        )
        try:
            critic_max = float(response.get("critic_max"))
            receipt_critic_max = float(receipt.get("critic_max"))
            response_threshold = float(response.get("stop_threshold"))
            receipt_threshold = float(receipt.get("stop_threshold"))
        except (TypeError, ValueError) as error:
            raise RealH24BackendError(
                "atomic critic/fallback metadata are not numeric") from error
        fallback = response.get("low_critic_fallback_applied")
        receipt_fallback = receipt.get("low_critic_fallback_applied")
        _require(
            not isinstance(response.get("critic_max"), bool)
            and not isinstance(receipt.get("critic_max"), bool)
            and not isinstance(response.get("stop_threshold"), bool)
            and not isinstance(receipt.get("stop_threshold"), bool)
            and math.isfinite(critic_max)
            and math.isfinite(response_threshold)
            and critic_max == receipt_critic_max
            and response_threshold == receipt_threshold == self.stop_threshold
            and critic_max == float(all_values.max())
            and isinstance(fallback, bool)
            and isinstance(receipt_fallback, bool)
            and fallback == receipt_fallback
            and fallback == (critic_max < self.stop_threshold),
            "atomic critic/fallback metadata disagree",
        )
        if fallback:
            _require(
                bool(np.all(executable[:, 0] == 0.0))
                and bool(np.all(executable[:, 1] == executable[0, 1]))
                and float(executable[0, 1]) in (-1.0, 0.0, 1.0)
                and bool(np.array_equal(executable[:, 2], raw_selected[:, 2])),
                "atomic low-critic fallback trajectory is malformed",
            )
        else:
            _require(
                bool(np.array_equal(executable, raw_selected)),
                "atomic non-fallback trajectory differs from raw selection",
            )
        _require(
            receipt.get("fifo_before_sha256") == before.get("fifo_sha256")
            and receipt.get("fifo_item_sha256_before")
            == before.get("queue_item_sha256")
            and receipt.get("fifo_lengths_before") == before.get("queue_lengths"),
            "atomic plan started from an unexpected processed FIFO",
        )
        current_items = receipt.get("current_item_sha256")
        _require(
            isinstance(current_items, list) and len(current_items) == 1,
            "atomic plan current processed hash is invalid",
        )
        current_processed = _sha256_string(
            current_items[0], "processed current image SHA")
        _sha256_string(receipt.get("current_sha256"), "processed current batch SHA")
        goal_items = receipt.get("goal_item_sha256")
        _require(
            isinstance(goal_items, list) and len(goal_items) == 1,
            "atomic plan goal processed hash is invalid",
        )
        processed_goal = _sha256_string(goal_items[0], "processed goal image SHA")
        _sha256_string(receipt.get("goal_sha256"), "processed goal batch SHA")
        if self._processed_goal_sha256 is None:
            self._processed_goal_sha256 = processed_goal
        _require(
            processed_goal == self._processed_goal_sha256,
            "identical raw goal produced a different processed hash",
        )
        if expected_mode == "native":
            _require(receipt.get("point_goal_sha256") is None, "native plan processed a point goal")
        else:
            _sha256_string(receipt.get("point_goal_sha256"), "processed point-goal SHA")
            _require(local_goal is not None, "residual local goal is absent")

        after = self._memory_audit()
        after_items = _audit_items(after)
        before_items = _audit_items(before)
        _require(
            after.get("fifo_sha256") == receipt.get("fifo_after_append_sha256")
            and after.get("queue_item_sha256") == receipt.get("fifo_item_sha256")
            and after.get("queue_lengths") == receipt.get("fifo_lengths_after"),
            "server memory audit disagrees with atomic receipt",
        )
        _require(
            after_items[-1:] == (current_processed,),
            "processed current image is not the FIFO tail",
        )
        expected_items = (
            (*before_items[-7:], current_processed)
            if len(before_items) >= 7 else (*before_items, current_processed)
        )
        _require(after_items == expected_items, "atomic plan duplicated or reordered current input")
        _require(
            len(after_items) == min(8, len(before_items) + 1),
            "atomic plan changed FIFO by more than one item",
        )
        return (
            receipt,
            after,
            raw_selected,
            executable,
            all_trajectory,
            all_values,
            critic_max,
            fallback,
        )

    def plan(self, request: PlanRequest) -> PlanReceipt:
        _require(self._prepared and self._last_audit is not None, "arm is not prepared")
        _require(request.state_id == self.assets.state.state_id, "plan state mismatch")
        _require(request.goal_sha256 == self.assets.state.goal_sha256, "plan goal hash mismatch")
        _require(
            request.current_rgb_sha256 == self._current.rgb_sha256
            and request.current_depth_sha256 == self._current.depth_sha256,
            "plan raw current bytes do not match protocol request",
        )
        _require(
            canonical_pose_sha256(_pose_tuple(self.runtime.pose))
            == request.current_pose_sha256
            and _pose_tuple(self.runtime.pose) == request.current_world_pose_xz_yaw,
            "plan Habitat pose differs from request",
        )
        before = self._memory_audit()
        _require(
            self._same_audit(before, self._last_audit),
            "NavDP FIFO mutated between preparation/commitments and planning",
        )
        local_goal = None
        data = {
            "mode": "native" if request.candidate_type == "native" else "image_point",
            "diffusion_seed": str(request.diffusion_seed),
        }
        if request.candidate_type != "native":
            _require(
                request.fixed_world_subgoal_xz_m is not None,
                "residual has no fixed world goal",
            )
            projected = world_goal_to_local(
                request.fixed_world_subgoal_xz_m,
                request.current_world_pose_xz_yaw,
            )
            local_goal = (float(projected[0]), float(projected[1]))
            data["goal_data"] = json.dumps({
                "goal_x": [local_goal[0]],
                "goal_y": [local_goal[1]],
            }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        response = self.transport.post_multipart(
            "/navdp_plan_atomic",
            files={
                "image": ("current.jpg", self._current.rgb_jpeg, "image/jpeg"),
                "image_goal": ("goal.jpg", self.assets.goal_jpeg, "image/jpeg"),
                "depth": ("current.png", self._current.depth_png, "image/png"),
            },
            data=data,
        )
        (
            receipt,
            after,
            raw_selected,
            executable,
            all_trajectory,
            all_values,
            critic_max,
            fallback_applied,
        ) = self._verify_atomic_receipt(
            request, response, before, local_goal)
        plan_sha = canonical_sha256({
            "backend_protocol": BACKEND_PROTOCOL,
            "server_receipt_sha256": receipt["receipt_sha256"],
            "raw_selected_trajectory": raw_selected.tolist(),
            "executable_trajectory": executable.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
            "critic_max": critic_max,
            "stop_threshold": self.stop_threshold,
            "low_critic_fallback_applied": fallback_applied,
            "raw_current_rgb_sha256": self._current.rgb_sha256,
            "raw_current_depth_sha256": self._current.depth_sha256,
            "raw_goal_sha256": self.assets.state.goal_sha256,
        })
        before_length = len(_audit_items(before))
        after_length = len(_audit_items(after))
        result = PlanReceipt(
            state_id=request.state_id,
            candidate_id=request.candidate_id,
            candidate_type=request.candidate_type,
            goal_sha256=request.goal_sha256,
            commitment_index=request.commitment_index,
            diffusion_seed=request.diffusion_seed,
            current_rgb_sha256=request.current_rgb_sha256,
            current_depth_sha256=request.current_depth_sha256,
            current_pose_sha256=request.current_pose_sha256,
            current_world_pose_xz_yaw=request.current_world_pose_xz_yaw,
            fixed_world_subgoal_xz_m=request.fixed_world_subgoal_xz_m,
            local_subgoal_forward_left_m=local_goal,
            plan_sha256=plan_sha,
            fifo_sha256_before=str(before["fifo_sha256"]),
            fifo_sha256_after=str(after["fifo_sha256"]),
            queue_length_before=before_length,
            queue_length_after=after_length,
            diffusion_calls_delta=1,
        )
        self._plans[plan_sha] = (executable, result)
        diagnostics = NavDPPlanDiagnostics(
            plan_sha256=plan_sha,
            server_selected_trajectory_index=_server_selected_index(
                raw_selected, all_trajectory),
            raw_selected_trajectory=tuple(
                tuple(float(value) for value in row) for row in raw_selected),
            executable_trajectory=tuple(
                tuple(float(value) for value in row) for row in executable),
            all_trajectory=all_trajectory.tolist(),
            all_values=all_values.tolist(),
            critic_max=critic_max,
            stop_threshold=self.stop_threshold,
            low_critic_fallback_applied=fallback_applied,
            server_receipt_sha256=str(receipt["receipt_sha256"]),
        )
        self._plan_diagnostics[plan_sha] = diagnostics
        self._last_audit = after
        self._diffusion_calls += 1
        return result

    def plan_diagnostics(self, plan_sha256: str) -> NavDPPlanDiagnostics:
        """Return lossless outputs for native-proposal feature construction."""
        _sha256_string(plan_sha256, "plan_sha256")
        diagnostics = self._plan_diagnostics.get(plan_sha256)
        _require(diagnostics is not None, "plan diagnostics are unavailable")
        return diagnostics

    def export_plan_diagnostics(self) -> dict[str, dict[str, object]]:
        """Canonical-JSON-compatible diagnostics for a collector sidecar."""
        return {
            plan_sha: self._plan_diagnostics[plan_sha].to_dict()
            for plan_sha in sorted(self._plan_diagnostics)
        }

    def _world_path(self, trajectory: np.ndarray, pose: HabitatPlanarPose) -> np.ndarray:
        points = []
        for waypoint in trajectory:
            points.append(local_forward_left_to_world(waypoint[:2], pose)[[0, 2]])
        result = np.asarray(points, dtype=np.float64)
        _require(result.ndim == 2 and result.shape[1] == 2, "world plan is malformed")
        return result

    def _pursuit_step(
        self, path_xz: np.ndarray
    ) -> tuple[HabitatPlanarPose, float, bool, bool, bool, bool]:
        pose = self.runtime.pose
        current_xz = np.asarray([pose.x_m, pose.z_m], dtype=np.float64)
        distances = np.linalg.norm(path_xz - current_xz[None, :], axis=1)
        ahead = np.where(distances >= self.controller.lookahead_m)[0]
        target = path_xz[ahead[0]] if len(ahead) else path_xz[-1]
        delta = target - current_xz
        desired_yaw = math.atan2(-float(delta[0]), -float(delta[1]))
        alpha = float(wrap_yaw(desired_yaw - pose.yaw_rad))
        velocity = self.controller.v_max_m * (
            0.48 + 0.52 * (1.0 + math.cos(alpha)) / 2.0)
        curvature = float(np.clip(
            2.0 * alpha / self.controller.lookahead_m,
            -1.0 / self.controller.min_radius_m,
            1.0 / self.controller.min_radius_m,
        ))
        max_turn = math.radians(self.controller.max_turn_deg)
        new_yaw = float(wrap_yaw(
            pose.yaw_rad + float(np.clip(curvature * velocity, -max_turn, max_turn))))
        forward = np.asarray([-math.sin(new_yaw), 0.0, -math.cos(new_yaw)])

        proposed = pose.position + velocity * forward
        snapped = self.runtime.snap_point(proposed)
        full_rejected = bool(
            snapped.shape != (3,)
            or not bool(np.isfinite(snapped).all())
            or not self.runtime.is_navigable(snapped)
            or np.linalg.norm(snapped[[0, 2]] - proposed[[0, 2]])
            > self.controller.snap_tolerance_m
        )
        collision = full_rejected
        creep_used = False
        if full_rejected:
            creep_used = True
            proposed_creep = (
                pose.position
                + self.controller.creep_fraction * velocity * forward
            )
            snapped_creep = self.runtime.snap_point(proposed_creep)
            creep_valid = bool(
                snapped_creep.shape == (3,)
                and bool(np.isfinite(snapped_creep).all())
                and self.runtime.is_navigable(snapped_creep)
                and np.linalg.norm(
                    snapped_creep[[0, 2]] - proposed_creep[[0, 2]])
                <= self.controller.snap_tolerance_m
            )
            snapped = snapped_creep if creep_valid else pose.position
        moved = float(np.linalg.norm(snapped[[0, 2]] - current_xz))
        zero_motion = moved <= 1e-12
        new_pose = HabitatPlanarPose(
            float(snapped[0]), float(snapped[1]), float(snapped[2]), new_yaw)
        return new_pose, moved, collision, full_rejected, creep_used, zero_motion

    def pursue(self, plan: PlanReceipt, steps: int) -> CommitmentReceipt:
        _require(self._prepared and self._last_audit is not None, "arm is not prepared")
        _require(
            isinstance(steps, int) and not isinstance(steps, bool) and steps == 8,
            "real backend executes exactly eight pursuit steps",
        )
        stored = self._plans.pop(plan.plan_sha256, None)
        _require(stored is not None and stored[1] == plan, "unknown or already executed plan")
        trajectory = stored[0]
        before = self._memory_audit()
        _require(self._same_audit(before, self._last_audit), "FIFO changed before pursuit")
        path_xz = self._world_path(trajectory, self.runtime.pose)

        receipts = []
        for offset in range(steps):
            new_pose, moved, collision, rejected, creep, zero = self._pursuit_step(path_xz)
            self.runtime.set_pose(new_pose)
            observation = self.runtime.render_encoded()
            reachable, distance = self.runtime.geodesic_distance(
                self.assets.label_goal_world_xyz_m)
            if reachable:
                _require(
                    distance is not None and math.isfinite(distance) and distance >= 0.0,
                    "pursuit returned an invalid geodesic distance",
                )
            else:
                _require(distance is None, "unreachable pursuit step returned distance")
            global_step = plan.commitment_index * steps + offset
            receipts.append(StepReceipt(
                global_step_index=global_step,
                pose_sha256=canonical_pose_sha256(_pose_tuple(new_pose)),
                world_pose_xz_yaw=_pose_tuple(new_pose),
                rgb_sha256=observation.rgb_sha256,
                depth_sha256=observation.depth_sha256,
                goal_distance_m=(float(distance) if reachable else None),
                goal_reachable=reachable,
                moved_m=moved,
                collision_detected=collision,
                full_step_rejected=rejected,
                creep_used=creep,
                zero_motion=zero,
            ))
            self._current = observation

        after = self._memory_audit()
        _require(
            self._same_audit(before, after)
            and self._same_audit(after, self._last_audit),
            "NavDP FIFO mutated during pure-pursuit execution",
        )
        return CommitmentReceipt(
            state_id=plan.state_id,
            candidate_id=plan.candidate_id,
            commitment_index=plan.commitment_index,
            plan_sha256=plan.plan_sha256,
            fifo_mutations=0,
            steps=tuple(receipts),
        )


def candidate_arms_from_feature_record(
    record: Mapping[str, Any],
    start_pose: HabitatPlanarPose,
) -> tuple[CandidateArm, ...]:
    """Convert a validated candidate-set record into immutable world arms.

    The feature record is intentionally not revalidated here; callers should
    first use ``novel_candidate_set_schema_v2.validate_candidate_set``.  This
    function consumes only candidate id/type and the explicit local subgoal.
    Dustbin has no executable arm.
    """
    candidates = record.get("candidates")
    _require(isinstance(candidates, Sequence), "candidate record has no candidates")
    arms = []
    for row in candidates:
        _require(isinstance(row, Mapping), "candidate row is not an object")
        candidate_id = row.get("candidate_id")
        candidate_type = row.get("candidate_type")
        _require(isinstance(candidate_id, str), "candidate id is invalid")
        if candidate_type == "dustbin":
            continue
        if candidate_type == "native":
            arms.append(CandidateArm("native", "native"))
            continue
        _require(candidate_type in ("memory_graph", "frontier"), "candidate type is invalid")
        features = row.get("features")
        _require(isinstance(features, Mapping), "candidate features are invalid")
        forward = features.get("subgoal_forward_m")
        left = features.get("subgoal_left_m")
        _require(
            isinstance(forward, (int, float)) and not isinstance(forward, bool)
            and isinstance(left, (int, float)) and not isinstance(left, bool)
            and math.isfinite(float(forward)) and math.isfinite(float(left)),
            "candidate local subgoal is invalid",
        )
        world = local_forward_left_to_world((forward, left), start_pose)
        arms.append(CandidateArm(
            candidate_id,
            str(candidate_type),
            (float(world[0]), float(world[2])),
        ))
    _require(sum(arm.is_native for arm in arms) == 1, "candidate set needs one native arm")
    return tuple(arms)


__all__ = [
    "BACKEND_PROTOCOL",
    "EncodedObservation",
    "FrozenStateAssets",
    "HabitatRuntime",
    "JsonTransport",
    "NavDPPlanDiagnostics",
    "PinnedHabitatRuntime",
    "PurePursuitConfig",
    "RealH24BackendError",
    "RealH24RolloutBackend",
    "RequestsJsonTransport",
    "candidate_arms_from_feature_record",
    "load_frozen_manifest",
    "load_state_assets_from_manifest",
    "sha256_bytes",
]
