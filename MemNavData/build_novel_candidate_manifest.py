#!/usr/bin/env python3
"""Build the frozen, causal NLSR-V2 expert-state sampling manifest.

This is deliberately a *manifest* builder, not a candidate or label builder.
It freezes which three-leg expert prefixes and same-scene goals may be used by
the later, privileged rollout collector.  In particular it never imports
Habitat, reads future observations into a feature, or admits final-reserved
scenes.

The episode and LingBot feature roots are explicit command-line inputs.  The
output is relocatable at the record level (all sample paths are root-relative),
canonical JSON plus a SHA256 sidecar.  Existing outputs are never silently
reused or replaced: ``--resume`` accepts only byte-identical output and
``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

try:
    from MemNavData.flow_cache_routing import (
        FlowRouteRegistry,
        FlowRoutingError,
        load_route_registry,
    )
except ImportError:  # Direct ``python MemNavData/<script>.py`` execution.
    from flow_cache_routing import (  # type: ignore
        FlowRouteRegistry,
        FlowRoutingError,
        load_route_registry,
    )


SCHEMA_VERSION = "nlsr_v2_expert_candidate_manifest_v1"
ROUTED_SCHEMA_VERSION = "nlsr_v2_expert_candidate_manifest_v2"
ALLOWED_ROLES = ("train", "development")
REQUIRED_FLOW_FILES = ("lingbot_cache.npz", "lingbot_cam_cache.npz")
NAVDP_MEMORY_SIZE = 8
NAVDP_EXEC_HORIZON = 8
RGB_RELATIVE = Path("videos/chunk-000/observation.images.rgb")
DEPTH_RELATIVE = Path("videos/chunk-000/observation.images.depth")
PARQUET_RELATIVE = Path("data/chunk-000/episode_000000.parquet")
METADATA_RELATIVE = Path("meta/gen_meta.json")
PARQUET_PREFIX_COLUMNS = (
    "index",
    "observation.camera_intrinsic",
    "observation.camera_extrinsic",
    "action",
)


class ManifestError(RuntimeError):
    """A fail-closed input or output contract violation."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def path_sha256(relative_path: str) -> str:
    return sha256_bytes(relative_path.encode("utf-8"))


def relative_file_record(path: Path, root: Path) -> dict:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ManifestError(f"input escapes its declared root: {path}") from exc
    if not path.is_file():
        raise ManifestError(f"required file is missing: {path}")
    return {
        "path": relative,
        "path_sha256": path_sha256(relative),
        "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path),
    }


def parse_scene_roles(split: Mapping[str, object]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for role in (*ALLOWED_ROLES, "final_reserved"):
        values = split.get(role)
        if not isinstance(values, list) or not all(
                isinstance(scene, str) and scene for scene in values):
            raise ManifestError(f"split role {role!r} must be a string list")
        for scene in values:
            if scene in roles:
                raise ManifestError(
                    f"scene occurs in multiple split roles: {scene}")
            roles[scene] = role
    if not roles:
        raise ManifestError("split manifest has no scenes")
    return roles


def validate_requested_roles(roles: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(role) for role in roles))
    if not normalized:
        raise ManifestError("at least one split role is required")
    forbidden = set(normalized) - set(ALLOWED_ROLES)
    if forbidden:
        raise ManifestError(
            "trainable manifests permit only train/development; rejected "
            f"roles: {sorted(forbidden)}")
    # The role order is protocol, not caller order.
    return tuple(role for role in ALLOWED_ROLES if role in normalized)


def resolve_scene_file(root: Path, scene: str, suffix: str) -> Path:
    candidates = (
        root / f"{scene}{suffix}",
        root / scene / f"{scene}{suffix}",
    )
    existing = []
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() not in {
                item.resolve() for item in existing}:
            existing.append(candidate)
    if len(existing) != 1:
        raise ManifestError(
            f"expected exactly one {suffix} for scene {scene} below {root}; "
            f"found {[str(path) for path in existing]}")
    return existing[0]


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ManifestError(f"{label} must be an integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{label} must be an integer") from exc
    if result != value:
        raise ManifestError(f"{label} must be an exact integer")
    return result


def aligned_midpoint(switch_a: int, switch_b: int,
                     alignment: int = 8) -> int:
    """Nearest decision time to the B-leg midpoint on the A-relative grid.

    Alignment is relative to ``switch_a`` because NavDP begins a new eight-frame
    commitment/FIFO epoch when Goal B becomes active.  Ties choose the earlier
    causal prefix.  Both endpoints are excluded.
    """
    if alignment < 1:
        raise ManifestError("midpoint alignment must be positive")
    candidates = list(range(switch_a + alignment, switch_b, alignment))
    if not candidates:
        raise ManifestError(
            f"Goal-B leg {switch_a}:{switch_b} has no interior "
            f"{alignment}-frame-aligned midpoint")
    return min(
        candidates,
        key=lambda frame: (abs(2 * frame - switch_a - switch_b), frame),
    )


def navdp_fifo_frame_indices(
    decision_frame: int,
    *,
    memory_size: int = NAVDP_MEMORY_SIZE,
    exec_horizon: int = NAVDP_EXEC_HORIZON,
) -> tuple[int, ...]:
    """Return the expert-prefix RGB frames present after the current append.

    ``decision_frame`` is the causal exclusive end, hence the current image is
    ``decision_frame - 1``.  Expert states do not come from a live NavDP FIFO,
    so v1 freezes an explicit decision cadence backwards from that image.  The
    same cadence is then replayed for every factual/counterfactual/candidate
    arm; on-policy rows will instead record their observed live FIFO.
    """
    if (isinstance(decision_frame, bool)
            or not isinstance(decision_frame, int)
            or decision_frame < 1):
        raise ManifestError("decision_frame must be a positive integer")
    if (isinstance(memory_size, bool) or not isinstance(memory_size, int)
            or memory_size < 1):
        raise ManifestError("NavDP memory_size must be a positive integer")
    if (isinstance(exec_horizon, bool) or not isinstance(exec_horizon, int)
            or exec_horizon < 1):
        raise ManifestError("NavDP exec_horizon must be a positive integer")
    current = decision_frame - 1
    reverse = range(current, -1, -exec_horizon)
    return tuple(reversed(tuple(reverse)[:memory_size]))


def _scene_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError("episode metadata scene is missing")
    return Path(value).stem


def _canonical_numeric(value: object, label: str) -> object:
    """Convert an Arrow scalar tree into stable, finite JSON primitives."""
    if isinstance(value, bool):
        raise ManifestError(f"{label} contains a boolean numeric value")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError(f"{label} contains NaN or infinity")
        # Normalize negative zero so producer-specific sign bits do not change
        # the canonical hash without changing the represented transform.
        return 0.0 if value == 0.0 else value
    if isinstance(value, (list, tuple)):
        return [
            _canonical_numeric(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ManifestError(
        f"{label} contains unsupported value type {type(value).__name__}")


def _matrix(value: object, shape: tuple[int, int], label: str) -> list:
    canonical = _canonical_numeric(value, label)
    if (not isinstance(canonical, list)
            or len(canonical) != shape[0]
            or any(not isinstance(row, list) or len(row) != shape[1]
                   for row in canonical)):
        raise ManifestError(f"{label} must have shape {shape}")
    return canonical


def load_parquet_rows(path: Path, n_frames: int) -> list[dict]:
    """Load exactly the causal pose/action columns once per episode."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ManifestError(
            "pyarrow is required to canonicalize parquet causal prefixes") from exc
    try:
        table = parquet.read_table(path, columns=list(PARQUET_PREFIX_COLUMNS))
    except Exception as exc:
        raise ManifestError(
            f"cannot read required parquet prefix columns from {path}: {exc}") from exc
    if tuple(table.column_names) != PARQUET_PREFIX_COLUMNS:
        raise ManifestError(
            f"parquet columns changed in {path}: {table.column_names}")
    if table.num_rows != n_frames:
        raise ManifestError(
            f"parquet/RGB frame count mismatch in {path}: "
            f"{table.num_rows} != {n_frames}")
    rows = []
    for frame, raw in enumerate(table.to_pylist()):
        index = _integer(raw.get("index"), f"parquet[{frame}].index")
        if index != frame:
            raise ManifestError(
                f"parquet index is not contiguous at row {frame}: {index}")
        rows.append({
            "index": index,
            "observation.camera_intrinsic": _matrix(
                raw.get("observation.camera_intrinsic"), (3, 3),
                f"parquet[{frame}].observation.camera_intrinsic"),
            "observation.camera_extrinsic": _matrix(
                raw.get("observation.camera_extrinsic"), (4, 4),
                f"parquet[{frame}].observation.camera_extrinsic"),
            "action": _matrix(
                raw.get("action"), (4, 4),
                f"parquet[{frame}].action"),
        })
    return rows


def load_valid_episode(episode: Path, scene: str) -> dict:
    """Validate one supported generated three-leg episode.

    The repository's generated format is intentionally handled directly:
    ``n_legs=3``, two ``switches``, two goals, numbered RGB frames, one parquet
    shard, and ``goal_1.jpg``/``goal_2.jpg``.  A two-leg ``switch_idx`` record is
    not guessed into a three-leg record.
    """
    metadata_path = episode / METADATA_RELATIVE
    parquet_path = episode / PARQUET_RELATIVE
    goal_b_path = episode / "goal_1.jpg"
    goal_c_path = episode / "goal_2.jpg"
    for path in (metadata_path, parquet_path, goal_b_path, goal_c_path):
        if not path.is_file():
            raise ManifestError(f"episode input is missing: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid episode metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ManifestError(f"episode metadata is not an object: {metadata_path}")
    if _scene_name(metadata.get("scene")) != scene:
        raise ManifestError(f"episode scene mismatch: {episode}")
    if _integer(metadata.get("n_legs"), "n_legs") != 3:
        raise ManifestError(f"not a three-leg episode: {episode}")
    n_frames = _integer(metadata.get("n_frames"), "n_frames")
    switches_raw = metadata.get("switches")
    if not isinstance(switches_raw, list) or len(switches_raw) != 2:
        raise ManifestError(f"three-leg switches are invalid: {episode}")
    switch_a, switch_b = (
        _integer(value, f"switches[{index}]")
        for index, value in enumerate(switches_raw)
    )
    if not 0 < switch_a < switch_b < n_frames:
        raise ManifestError(f"three-leg switches are out of bounds: {episode}")
    midpoint = aligned_midpoint(switch_a, switch_b)
    goals = metadata.get("goals")
    if not isinstance(goals, list) or len(goals) != 2:
        raise ManifestError(f"three-leg goals are invalid: {episode}")
    if (not isinstance(goals[0], dict)
            or goals[0].get("kind") != "novel"):
        raise ManifestError(f"Goal B must be Novel: {episode}")
    if (not isinstance(goals[1], dict)
            or goals[1].get("kind") != "revisit"):
        raise ManifestError(f"Goal C must be Revisit: {episode}")

    rgb_root = episode / RGB_RELATIVE
    depth_root = episode / DEPTH_RELATIVE
    if not rgb_root.is_dir():
        raise ManifestError(f"RGB directory is missing: {rgb_root}")
    if not depth_root.is_dir():
        raise ManifestError(f"depth directory is missing: {depth_root}")
    actual_rgb = {path.name for path in rgb_root.glob("*.jpg")}
    expected_rgb = {f"{index}.jpg" for index in range(n_frames)}
    if actual_rgb != expected_rgb:
        raise ManifestError(f"RGB frame set is incomplete: {episode}")
    actual_depth = {path.name for path in depth_root.glob("*.png")}
    expected_depth = {f"{index}.png" for index in range(n_frames)}
    if actual_depth != expected_depth:
        raise ManifestError(f"depth frame set is incomplete: {episode}")
    parquet_rows = load_parquet_rows(parquet_path, n_frames)
    return {
        "root": episode,
        "name": episode.name,
        "metadata": metadata_path,
        "parquet": parquet_path,
        "goal_b": goal_b_path,
        "goal_c": goal_c_path,
        "rgb_root": rgb_root,
        "depth_root": depth_root,
        "parquet_rows": parquet_rows,
        "n_frames": n_frames,
        "switch_a": switch_a,
        "switch_b": switch_b,
        "midpoint": midpoint,
    }


def select_scene_episodes(episode_root: Path, scene: str,
                          count: int = 2) -> list[dict]:
    if count != 2:
        raise ManifestError("NLSR-V2 v1 requires exactly two episodes per scene")
    scene_root = episode_root / scene
    if not scene_root.is_dir():
        raise ManifestError(f"episode scene directory is missing: {scene_root}")
    valid = []
    failures = []
    for candidate in sorted(
            (path for path in scene_root.glob("episode_*") if path.is_dir()),
            key=lambda path: path.name):
        try:
            valid.append(load_valid_episode(candidate, scene))
        except ManifestError as exc:
            failures.append(f"{candidate.name}: {exc}")
    if len(valid) < count:
        detail = "; ".join(failures[:3])
        raise ManifestError(
            f"scene {scene} has {len(valid)} valid three-leg episodes, "
            f"requires {count}. {detail}")
    return valid[:count]


def _hash_sequence(items: Iterable[object]) -> str:
    return sha256_bytes(canonical_json_bytes(list(items)))


def prefix_record(episode: Mapping[str, object], episode_root: Path,
                  exclusive_end: int, file_hashes: dict[Path, dict]) -> dict:
    root = episode["root"]
    rgb_root = episode["rgb_root"]
    depth_root = episode["depth_root"]
    parquet_rows = episode["parquet_rows"]
    if (not isinstance(root, Path) or not isinstance(rgb_root, Path)
            or not isinstance(depth_root, Path)
            or not isinstance(parquet_rows, list)):
        raise ManifestError("internal episode path contract is invalid")
    if not 0 < exclusive_end <= int(episode["n_frames"]):
        raise ManifestError("causal prefix end is out of bounds")
    modality_records = {}
    for modality, root_path, suffix in (
            ("rgb", rgb_root, ".jpg"),
            ("depth", depth_root, ".png")):
        records = []
        for frame in range(exclusive_end):
            path = root_path / f"{frame}{suffix}"
            record = file_hashes.get(path)
            if record is None:
                record = relative_file_record(path, episode_root)
                file_hashes[path] = record
            records.append(record)
        modality_records[modality] = {
            "path_sequence_sha256": _hash_sequence(
                record["path"] for record in records),
            "content_sequence_sha256": _hash_sequence({
                "path": record["path"],
                "bytes": record["bytes"],
                "content_sha256": record["content_sha256"],
            } for record in records),
        }
    parquet_prefix = parquet_rows[:exclusive_end]
    parquet_hash = _hash_sequence(parquet_prefix)
    causal_hash = sha256_bytes(canonical_json_bytes({
        "frame_count": exclusive_end,
        "rgb": modality_records["rgb"],
        "depth": modality_records["depth"],
        "parquet_rows_sha256": parquet_hash,
    }))
    return {
        "exclusive_end_frame": exclusive_end,
        "frame_count": exclusive_end,
        "modalities": modality_records,
        "parquet_columns": list(PARQUET_PREFIX_COLUMNS),
        "parquet_row_count": exclusive_end,
        "parquet_rows_sha256": parquet_hash,
        "causal_prefix_sha256": causal_hash,
    }


def navdp_fifo_record(
    episode: Mapping[str, object],
    episode_root: Path,
    decision_frame: int,
    file_hashes: dict[Path, dict],
) -> dict:
    """Freeze the exact raw RGB queue reconstructed for an expert state."""
    rgb_root = episode["rgb_root"]
    if not isinstance(rgb_root, Path):
        raise ManifestError("internal RGB root contract is invalid")
    indices = navdp_fifo_frame_indices(decision_frame)
    records = []
    for frame in indices:
        path = rgb_root / f"{frame}.jpg"
        record = file_hashes.get(path)
        if record is None:
            record = relative_file_record(path, episode_root)
            file_hashes[path] = record
        records.append(record)
    payload = {
        "memory_size": NAVDP_MEMORY_SIZE,
        "exec_horizon": NAVDP_EXEC_HORIZON,
        "left_zero_pad_count": NAVDP_MEMORY_SIZE - len(records),
        "replay_frame_indices": list(indices[:-1]),
        "current_frame_index": indices[-1],
        "after_append_frame_indices": list(indices),
        "path_sequence_sha256": _hash_sequence(
            record["path"] for record in records),
        "content_sequence_sha256": _hash_sequence({
            "path": record["path"],
            "bytes": record["bytes"],
            "content_sha256": record["content_sha256"],
        } for record in records),
    }
    return {
        **payload,
        "fifo_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _flow_record(flow_root: Path, scene: str, episode: str) -> tuple[dict, list[dict]]:
    chunk = flow_root / scene / episode / "videos/chunk-000"
    files = []
    missing = []
    for name in REQUIRED_FLOW_FILES:
        path = chunk / name
        relative = path.relative_to(flow_root).as_posix()
        if path.is_file():
            # Full LingBot caches are hundreds of MiB each.  The sampling
            # manifest records availability and byte size; the subsequent
            # cache-schema preflight owns their version/content validation.
            files.append({
                "path": relative,
                "path_sha256": path_sha256(relative),
                "bytes": path.stat().st_size,
            })
        else:
            missing.append({
                "scene": scene,
                "episode": episode,
                "path": relative,
                "cache_file": name,
            })
    return {
        "complete": not missing,
        "files": files,
        "validation_owner": "downstream_versioned_cache_preflight",
    }, missing


def _routed_flow_record(
    registry: FlowRouteRegistry,
    scene: str,
    episode: str,
) -> tuple[dict, list[dict]]:
    """Freeze one exact multi-root route into the causal manifest.

    A missing route remains visible in the same shape as a missing legacy
    single-root pair.  Any malformed or stale declared route is a hard failure,
    rather than being silently converted to "missing".
    """

    key = f"{scene}/{episode}"
    if key not in registry.files_by_episode:
        missing = [
            {
                "scene": scene,
                "episode": episode,
                "path": f"{key}/videos/chunk-000/{name}",
                "cache_file": name,
            }
            for name in REQUIRED_FLOW_FILES
        ]
        return {
            "complete": False,
            "files": [],
            "validation_owner": "downstream_versioned_cache_preflight",
        }, missing
    try:
        files = registry.episode_file_records(scene, episode)
    except FlowRoutingError as error:
        raise ManifestError(
            f"invalid routed flow cache for {key}: {error}") from error
    return {
        "complete": True,
        "files": files,
        "validation_owner": "downstream_versioned_cache_preflight",
    }, []


def build_manifest(*, split_path: Path, episode_root: Path,
                   flow_cache_root: Path | None, environment_root: Path,
                   navmesh_root: Path,
                   roles: Sequence[str] = ALLOWED_ROLES,
                   flow_route_provenance: Path | None = None,
                   expected_flow_route_sha256: str | None = None) -> dict:
    selected_roles = validate_requested_roles(roles)
    roots = {
        "episode_root": episode_root,
        "environment_root": environment_root,
        "navmesh_root": navmesh_root,
    }
    route_requested = flow_route_provenance is not None
    if route_requested != (expected_flow_route_sha256 is not None):
        raise ManifestError(
            "flow route provenance and its expected SHA256 must be supplied together"
        )
    if route_requested == (flow_cache_root is not None):
        raise ManifestError(
            "choose exactly one flow source: a single root or a pinned route artifact"
        )
    route_registry = None
    if route_requested:
        assert flow_route_provenance is not None
        assert expected_flow_route_sha256 is not None
        try:
            route_registry = load_route_registry(
                flow_route_provenance, expected_flow_route_sha256)
        except FlowRoutingError as error:
            raise ManifestError(f"flow route provenance is invalid: {error}") from error
    else:
        assert flow_cache_root is not None
        roots["flow_cache_root"] = flow_cache_root
    for label, root in roots.items():
        if not root.is_dir():
            raise ManifestError(f"{label} is not a directory: {root}")
    if not split_path.is_file():
        raise ManifestError(f"split manifest is missing: {split_path}")
    split_bytes = split_path.read_bytes()
    try:
        split = json.loads(split_bytes)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid split manifest: {split_path}") from exc
    if not isinstance(split, dict):
        raise ManifestError("split manifest must be an object")
    scene_roles = parse_scene_roles(split)

    scenes = []
    samples = []
    missing_flow_caches = []
    file_hashes: dict[Path, dict] = {}
    for role in selected_roles:
        role_scenes = split[role]
        assert isinstance(role_scenes, list)  # established by parse_scene_roles
        for scene in role_scenes:
            if scene_roles.get(scene) != role:
                raise ManifestError(f"scene role changed during build: {scene}")
            if role == "final_reserved":
                raise ManifestError(f"final-reserved scene leaked: {scene}")
            selected = select_scene_episodes(episode_root, scene)
            environment = resolve_scene_file(environment_root, scene, ".glb")
            navmesh = resolve_scene_file(navmesh_root, scene, ".navmesh")
            episode_records = []
            for episode in selected:
                ep_name = str(episode["name"])
                if route_registry is None:
                    assert flow_cache_root is not None
                    flow, missing = _flow_record(
                        flow_cache_root, scene, ep_name)
                else:
                    flow, missing = _routed_flow_record(
                        route_registry, scene, ep_name)
                missing_flow_caches.extend(missing)
                episode_records.append({
                    "episode": ep_name,
                    "n_frames": int(episode["n_frames"]),
                    "switches": [
                        int(episode["switch_a"]),
                        int(episode["switch_b"]),
                    ],
                    "goal_b_midpoint_frame": int(episode["midpoint"]),
                    "metadata": relative_file_record(
                        episode["metadata"], episode_root),  # type: ignore[arg-type]
                    "parquet": relative_file_record(
                        episode["parquet"], episode_root),  # type: ignore[arg-type]
                    "goal_b": relative_file_record(
                        episode["goal_b"], episode_root),  # type: ignore[arg-type]
                    "goal_c": relative_file_record(
                        episode["goal_c"], episode_root),  # type: ignore[arg-type]
                    "flow_cache": flow,
                })
            scenes.append({
                "scene": scene,
                "split_role": role,
                "environment": relative_file_record(
                    environment, environment_root),
                "navmesh": relative_file_record(navmesh, navmesh_root),
                "selected_episodes": episode_records,
            })

            for source_index, source in enumerate(selected):
                partner = selected[1 - source_index]
                state_specs = (
                    ("goal_b_t0", int(source["switch_a"])),
                    ("goal_b_midpoint_t1", int(source["midpoint"])),
                )
                for state_name, decision_frame in state_specs:
                    prefix = prefix_record(
                        source, episode_root, decision_frame, file_hashes)
                    navdp_fifo = navdp_fifo_record(
                        source, episode_root, decision_frame, file_hashes)
                    state_frame = source["rgb_root"] / f"{decision_frame - 1}.jpg"  # type: ignore[operator]
                    state_frame_record = file_hashes.get(state_frame)
                    if state_frame_record is None:
                        state_frame_record = relative_file_record(
                            state_frame, episode_root)
                        file_hashes[state_frame] = state_frame_record
                    for goal_variant, goal_episode in (
                            ("factual", source),
                            ("counterfactual", partner)):
                        if str(goal_episode["name"]) == str(source["name"]):
                            if goal_variant != "factual":
                                raise ManifestError(
                                    "counterfactual goal must use the paired episode")
                        elif goal_variant != "counterfactual":
                            raise ManifestError(
                                "factual goal must use the source episode")
                        goal_record = relative_file_record(
                            goal_episode["goal_b"],  # type: ignore[arg-type]
                            episode_root,
                        )
                        samples.append({
                            "sample_id": (
                                f"{role}/{scene}/{source['name']}/"
                                f"{state_name}/{goal_variant}"
                            ),
                            "split_role": role,
                            "scene": scene,
                            "state_source": "expert",
                            "source_episode": str(source["name"]),
                            "source_episode_id": (
                                f"{scene}/{source['name']}"),
                            "goal_episode": str(goal_episode["name"]),
                            "goal_source_episode_id": (
                                f"{scene}/{goal_episode['name']}"),
                            "goal_variant": goal_variant,
                            "goal_role": "B",
                            "state_name": state_name,
                            "decision_frame": decision_frame,
                            "state_frame": state_frame_record,
                            "causal_prefix": prefix,
                            "navdp_fifo": navdp_fifo,
                            "goal": goal_record,
                        })

    if any(sample["split_role"] == "final_reserved" for sample in samples):
        raise ManifestError("final-reserved sample leaked into manifest")
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ManifestError("sample ids are not unique")
    missing_flow_caches.sort(
        key=lambda row: (row["scene"], row["episode"], row["cache_file"]))
    manifest = {
        "schema_version": (
            ROUTED_SCHEMA_VERSION if route_registry is not None
            else SCHEMA_VERSION
        ),
        "purpose": (
            "frozen causal expert-state sampling for NLSR-V2 candidate and "
            "counterfactual rollout collection; not a trainable feature artifact"
        ),
        "selection": {
            "split_roles": list(selected_roles),
            "episodes_per_scene": 2,
            "episode_order": "first two lexicographically sorted valid 3-leg episodes",
            "states_per_episode": 2,
            "state_times": [
                "Goal-B t0 = switch_A",
                "Goal-B t1 = nearest interior 8-frame-aligned midpoint relative to switch_A; ties earlier",
            ],
            "goal_variants_per_state": ["factual", "counterfactual"],
            "counterfactual_rule": (
                "goal_1.jpg from the other selected episode in the same scene"
            ),
            "prefix_semantics": (
                "decision_frame is exclusive; state_frame is decision_frame-1"
            ),
            "expert_navdp_fifo": (
                "memory_size=8; current=decision_frame-1; preceding replay "
                "frames step backward by 8 from current; left-pad is native "
                "NavDP zero padding"
            ),
        },
        "split": {
            "path": split_path.name,
            "sha256": sha256_bytes(split_bytes),
            "version": split.get("version"),
        },
        "input_roots": {
            label: str(path.resolve()) for label, path in roots.items()
        },
        "scenes": scenes,
        "samples": samples,
        "missing_flow_caches": missing_flow_caches,
        "summary": {
            "scene_count": len(scenes),
            "episode_count": sum(
                len(scene["selected_episodes"]) for scene in scenes),
            "sample_count": len(samples),
            "missing_flow_cache_file_count": len(missing_flow_caches),
            "all_flow_caches_complete": not missing_flow_caches,
        },
    }
    if route_registry is not None:
        manifest["flow_cache_routing"] = route_registry.manifest_record()
    return manifest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_artifact(manifest: Mapping[str, object], output: Path,
                   sha_output: Path, *, resume: bool = False,
                   overwrite: bool = False) -> tuple[str, str]:
    if resume and overwrite:
        raise ManifestError("--resume and --overwrite are mutually exclusive")
    if output.resolve() == sha_output.resolve():
        raise ManifestError("JSON and SHA outputs must be distinct")
    encoded = canonical_json_bytes(manifest)
    digest = sha256_bytes(encoded)
    sidecar = f"{digest}  {output.name}\n".encode("ascii")
    exists = (output.exists(), sha_output.exists())
    if resume:
        if exists != (True, True):
            raise ManifestError(
                "resume requires an existing JSON and SHA sidecar")
        if output.read_bytes() != encoded or sha_output.read_bytes() != sidecar:
            raise ManifestError(
                "resume output differs from the newly built canonical artifact")
        return "resumed", digest
    if any(exists) and not overwrite:
        raise ManifestError(
            "output already exists; use --resume for byte-identical reuse or "
            "--overwrite for explicit replacement")
    # JSON is the authoritative artifact.  If interrupted between replaces,
    # subsequent default/resume runs fail closed on the incomplete pair.
    _atomic_write(output, encoded)
    _atomic_write(sha_output, sidecar)
    return "written", digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path(__file__).with_name(
            "router_multiscene_split_20260805.json"))
    parser.add_argument("--episode-root", type=Path, required=True)
    flow_source = parser.add_mutually_exclusive_group(required=True)
    flow_source.add_argument("--flow-cache-root", type=Path)
    flow_source.add_argument("--flow-route-provenance", type=Path)
    parser.add_argument(
        "--expected-flow-route-sha",
        help="required SHA256 pin when --flow-route-provenance is used")
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--navmesh-root", type=Path, required=True)
    parser.add_argument(
        "--role", action="append", default=[],
        help="train or development; repeatable (default: both)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sha-out", type=Path,
        help="default: <out>.sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = tuple(args.role) if args.role else ALLOWED_ROLES
    manifest = build_manifest(
        split_path=args.split_manifest,
        episode_root=args.episode_root,
        flow_cache_root=args.flow_cache_root,
        environment_root=args.environment_root,
        navmesh_root=args.navmesh_root,
        roles=roles,
        flow_route_provenance=args.flow_route_provenance,
        expected_flow_route_sha256=args.expected_flow_route_sha,
    )
    sha_output = args.sha_out or Path(f"{args.out}.sha256")
    status, digest = write_artifact(
        manifest,
        args.out,
        sha_output,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": status,
        "output": str(args.out),
        "sha_output": str(sha_output),
        "sha256": digest,
        **manifest["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
