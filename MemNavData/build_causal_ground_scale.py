#!/usr/bin/env python3
"""Build causal first-prefix LingBot metric-scale sidecars for NLSR-V2.

Only RGB frames and dense LingBot poses strictly before the earliest selected
decision of an episode are passed to the frozen depth/pose estimator.  The
default cap is 64 frames, matching ``LingBotStream.compute_metric_scale``.
Later trajectory frames, goals, Habitat poses, and Pathfinder are never inputs.

The output is a standalone immutable JSON artifact.  Shared flow caches are
read-only and are not patched in place.  A downstream proposal builder must
pin the artifact SHA, producer SHA, configuration SHA, LingBot commit/weights,
and re-hash the exact RGB/pose prefix before using a scale.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

try:
    from MemNavData.flow_cache_routing import (
        FlowRoutingError,
        registry_from_manifest,
    )
    from MemNavData.build_novel_frontier_candidates import (
        AGGREGATOR_FLOW_RELATIVE,
        FLOW_RELATIVE,
        INPUT_MANIFEST_SCHEMA_VERSION,
        ROUTED_INPUT_MANIFEST_SCHEMA_VERSIONS,
        SUPPORTED_INPUT_MANIFEST_SCHEMA_VERSIONS,
        METADATA_RELATIVE,
        RGB_RELATIVE,
        _validate_versioned_cache_pair,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        write_artifact,
    )
except ModuleNotFoundError:  # Direct execution from MemNavData/.
    from flow_cache_routing import (  # type: ignore
        FlowRoutingError,
        registry_from_manifest,
    )
    from build_novel_frontier_candidates import (  # type: ignore
        AGGREGATOR_FLOW_RELATIVE,
        FLOW_RELATIVE,
        INPUT_MANIFEST_SCHEMA_VERSION,
        ROUTED_INPUT_MANIFEST_SCHEMA_VERSIONS,
        SUPPORTED_INPUT_MANIFEST_SCHEMA_VERSIONS,
        METADATA_RELATIVE,
        RGB_RELATIVE,
        _validate_versioned_cache_pair,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        write_artifact,
    )


SCHEMA_VERSION = "nlsr_v2_causal_ground_scale_v1"
DEFAULT_PREFIX_FRAME_CAP = 64
DEFAULT_NUM_SCALE_FRAMES = 8
DEFAULT_BIAS_CORRECTION = 1.15
DEFAULT_SCALE_RANGE = (0.8, 6.0)


class CausalScaleError(RuntimeError):
    """Raised when a scale artifact cannot prove causal provenance."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalScaleError(message)


def _hex_digest_is_valid(value: object, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha_is_valid(value: object) -> bool:
    return _hex_digest_is_valid(value, (64,))


def _git_commit_is_valid(value: object) -> bool:
    # SHA-1 repositories use 40 hexadecimal characters.  Accept 64 as well so
    # the contract remains valid if a repository has migrated to SHA-256.
    return _hex_digest_is_valid(value, (40, 64))


def ndarray_sha256(value: object) -> str:
    array = np.asarray(value)
    _require(
        np.issubdtype(array.dtype, np.number),
        "pose prefix must be numeric",
    )
    _require(bool(np.isfinite(array).all()), "pose prefix must be finite")
    contiguous = np.ascontiguousarray(array)
    header = json.dumps({
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def rgb_prefix_record(
    episode_dir: Path,
    episode_root: Path,
    frame_count: int,
) -> dict[str, object]:
    _require(
        isinstance(frame_count, int)
        and not isinstance(frame_count, bool)
        and frame_count >= 1,
        "RGB prefix frame_count must be positive",
    )
    rows = []
    for frame in range(frame_count):
        path = episode_dir / RGB_RELATIVE / f"{frame}.jpg"
        _require(path.is_file(), f"causal RGB frame is missing: {path}")
        try:
            relative = path.resolve().relative_to(
                episode_root.resolve()).as_posix()
        except ValueError as error:
            raise CausalScaleError(
                f"causal RGB escapes episode root: {path}") from error
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "content_sha256": sha256_file(path),
        })
    return {
        "frame_count": frame_count,
        "path_sequence_sha256": sha256_bytes(canonical_json_bytes(
            [row["path"] for row in rows])),
        "content_sequence_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _safe_rooted_path(root: Path, relative: object, label: str) -> Path:
    _require(isinstance(relative, str) and bool(relative),
             f"{label} path is missing")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise CausalScaleError(f"{label} escapes its declared root") from error
    return candidate


def _validate_file_record(record: object, root: Path, label: str) -> Path:
    _require(isinstance(record, Mapping), f"{label} record is missing")
    path = _safe_rooted_path(root, record.get("path"), label)
    expected_bytes = record.get("bytes")
    expected_sha = record.get("content_sha256")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes >= 0
        and _sha_is_valid(expected_sha),
        f"{label} record is malformed",
    )
    _require(path.is_file() and path.stat().st_size == expected_bytes,
             f"{label} size changed: {path}")
    _require(sha256_file(path) == expected_sha,
             f"{label} content changed: {path}")
    return path


@dataclass(frozen=True)
class GroundScaleConfiguration:
    prefix_frame_cap: int = DEFAULT_PREFIX_FRAME_CAP
    num_scale_frames: int = DEFAULT_NUM_SCALE_FRAMES
    window: int = 32
    max_frame_num: int = 4096
    conf_quantile: float = 0.5
    pixel_stride: int = 4
    histogram_bins: int = 60
    peak_threshold: float = 0.3
    bias_correction: float = DEFAULT_BIAS_CORRECTION
    scale_min: float = DEFAULT_SCALE_RANGE[0]
    scale_max: float = DEFAULT_SCALE_RANGE[1]
    use_sdpa: bool = True

    def __post_init__(self) -> None:
        integers = (
            self.prefix_frame_cap,
            self.num_scale_frames,
            self.window,
            self.max_frame_num,
            self.pixel_stride,
            self.histogram_bins,
        )
        _require(all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 1
            for value in integers
        ), "ground-scale integer configuration is invalid")
        _require(
            self.prefix_frame_cap >= self.num_scale_frames,
            "prefix cap must include the complete scale block",
        )
        _require(
            self.max_frame_num >= self.prefix_frame_cap,
            "max_frame_num must cover the causal prefix",
        )
        reals = (
            self.conf_quantile,
            self.peak_threshold,
            self.bias_correction,
            self.scale_min,
            self.scale_max,
        )
        _require(all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in reals
        ), "ground-scale real configuration is invalid")
        _require(0.0 <= self.conf_quantile <= 1.0,
                 "conf_quantile must lie in [0,1]")
        _require(0.0 < self.peak_threshold <= 1.0,
                 "peak_threshold must lie in (0,1]")
        _require(self.bias_correction > 0.0,
                 "bias correction must be positive")
        _require(0.0 < self.scale_min < self.scale_max,
                 "scale range is invalid")
        _require(type(self.use_sdpa) is bool, "use_sdpa must be boolean")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class GroundScaleEstimate:
    ground_h_est_raw: float | None
    metric_scale_m_per_raw: float | None
    debug: Mapping[str, object]


class GroundScaleEstimator(Protocol):
    def provenance(self) -> Mapping[str, object]:
        ...

    def estimate(
        self,
        *,
        rgb_paths: Sequence[Path],
        cam_pose_prefix: np.ndarray,
        camera_height_m: float,
        configuration: GroundScaleConfiguration,
    ) -> GroundScaleEstimate:
        ...


def expected_scale_from_ground(
    ground_h_est_raw: float,
    camera_height_m: float,
    configuration: GroundScaleConfiguration,
) -> float:
    _require(
        math.isfinite(float(ground_h_est_raw))
        and float(ground_h_est_raw) > 0.0,
        "ground height must be finite and positive",
    )
    _require(
        math.isfinite(float(camera_height_m))
        and float(camera_height_m) > 0.0,
        "camera height must be finite and positive",
    )
    raw = (
        float(configuration.bias_correction)
        * float(camera_height_m)
        / float(ground_h_est_raw)
    )
    return min(max(raw, configuration.scale_min), configuration.scale_max)


def _json_safe(value: object, label: str = "value") -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        _require(math.isfinite(result), f"{label} is non-finite")
        return result
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            _json_safe(item, f"{label}[{index}]")
            for index, item in enumerate(list(value))
        ]
    raise CausalScaleError(
        f"{label} has unsupported type {type(value).__name__}")


def _flow_file(
    episode_record: Mapping[str, object],
    flow_root: Path | None,
    scene: str,
    episode: str,
    relative_suffix: Path,
) -> Path:
    _require(flow_root is not None, "legacy flow root is absent")
    flow = episode_record.get("flow_cache")
    _require(isinstance(flow, Mapping), "manifest flow record is missing")
    files = flow.get("files")
    _require(isinstance(files, list), "manifest flow file list is missing")
    expected_relative = (Path(scene) / episode / relative_suffix).as_posix()
    matches = [
        row for row in files
        if isinstance(row, Mapping)
        and row.get("path") == expected_relative
    ]
    _require(
        len(matches) == 1,
        f"manifest does not declare one {expected_relative}",
    )
    path = _safe_rooted_path(
        flow_root, matches[0]["path"], "flow cache")
    expected_bytes = matches[0].get("bytes")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and path.is_file()
        and path.stat().st_size == expected_bytes,
        f"flow file size changed: {path}",
    )
    return path


def _episode_index(manifest: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
    scenes = manifest.get("scenes")
    _require(isinstance(scenes, list), "manifest scenes are missing")
    result = {}
    seen_scenes: set[str] = set()
    for scene_record in scenes:
        _require(isinstance(scene_record, Mapping), "manifest scene is malformed")
        scene = scene_record.get("scene")
        split_role = scene_record.get("split_role")
        episodes = scene_record.get("selected_episodes")
        _require(isinstance(scene, str) and scene not in seen_scenes
                 and split_role in ("train", "development")
                 and isinstance(episodes, list),
                 "manifest scene/episodes are malformed")
        seen_scenes.add(scene)
        for episode_record in episodes:
            _require(isinstance(episode_record, Mapping),
                     "manifest episode is malformed")
            episode = episode_record.get("episode")
            key = scene, episode
            _require(
                isinstance(episode, str) and key not in result,
                "manifest episode identity is invalid or duplicated",
            )
            result[(scene, episode)] = episode_record
    return result


def _selected_episode_decisions(
    manifest: Mapping[str, object],
    selected_scenes: Sequence[str],
) -> dict[tuple[str, str], dict[str, object]]:
    samples = manifest.get("samples")
    _require(isinstance(samples, list), "manifest samples are missing")
    scene_records = manifest.get("scenes")
    _require(isinstance(scene_records, list), "manifest scenes are missing")
    manifest_scenes = {
        row.get("scene") for row in scene_records
        if isinstance(row, Mapping) and isinstance(row.get("scene"), str)
    }
    requested = tuple(dict.fromkeys(map(str, selected_scenes)))
    unknown = set(requested) - manifest_scenes
    _require(not unknown, f"requested scenes are absent: {sorted(unknown)}")
    allowed = set(requested) if requested else None
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    seen_sample_ids: set[str] = set()
    seen_sample_identity: set[tuple[str, str, str]] = set()
    for sample in samples:
        _require(isinstance(sample, Mapping), "manifest sample is malformed")
        scene = sample.get("scene")
        episode = sample.get("source_episode")
        decision = sample.get("decision_frame")
        sample_id = sample.get("sample_id")
        _require(
            isinstance(scene, str)
            and isinstance(episode, str)
            and isinstance(decision, int)
            and not isinstance(decision, bool)
            and decision >= 1
            and isinstance(sample_id, str),
            "manifest sample identity/decision is malformed",
        )
        _require(sample_id not in seen_sample_ids,
                 "manifest sample_id is duplicated")
        seen_sample_ids.add(sample_id)
        split_role = sample.get("split_role")
        _require(split_role in ("train", "development"),
                 "manifest sample role is invalid")
        sample_identity = scene, episode, sample_id
        _require(sample_identity not in seen_sample_identity,
                 "manifest sample identity is duplicated")
        seen_sample_identity.add(sample_identity)
        if allowed is not None and scene not in allowed:
            continue
        row = grouped.setdefault((scene, episode), {
            "decisions": set(),
            "sample_ids": [],
            "split_roles": set(),
        })
        row["decisions"].add(decision)  # type: ignore[union-attr]
        row["sample_ids"].append(sample_id)  # type: ignore[union-attr]
        row["split_roles"].add(str(split_role))  # type: ignore[union-attr]
    _require(bool(grouped), "scene selection produced no scale episodes")
    return grouped


CachePairValidator = Callable[[Path, Path, int], object]


def _default_cache_validator(
    aggregator_path: Path,
    camera_path: Path,
    frame_count: int,
) -> object:
    return _validate_versioned_cache_pair(
        aggregator_path,
        camera_path,
        expected_num_frames=frame_count,
    )


def build_scale_artifact(
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
    expected_manifest_sha256: str,
    estimator: GroundScaleEstimator,
    configuration: GroundScaleConfiguration = GroundScaleConfiguration(),
    selected_scenes: Sequence[str] = (),
    cache_pair_validator: CachePairValidator = _default_cache_validator,
) -> dict[str, object]:
    _require(
        manifest.get("schema_version")
        in SUPPORTED_INPUT_MANIFEST_SCHEMA_VERSIONS,
        "input is not an NLSR-V2 causal manifest",
    )
    _require(_sha_is_valid(expected_manifest_sha256),
             "expected manifest SHA is invalid")
    _require(manifest_path.is_file(), "causal manifest is missing")
    _require(
        sha256_file(manifest_path) == expected_manifest_sha256,
        "causal manifest SHA changed",
    )
    try:
        disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CausalScaleError("causal manifest is invalid JSON") from error
    _require(disk_manifest == manifest,
             "in-memory manifest differs from pinned file")
    roots = manifest.get("input_roots")
    _require(isinstance(roots, Mapping), "manifest input roots are missing")
    episode_root = Path(str(roots.get("episode_root", "")))
    _require(episode_root.is_dir(), "episode root is unavailable")
    try:
        route_registry = registry_from_manifest(manifest)
    except FlowRoutingError as error:
        raise CausalScaleError(
            f"manifest routed-flow provenance is invalid: {error}") from error
    if route_registry is None:
        _require(
            manifest.get("schema_version") == INPUT_MANIFEST_SCHEMA_VERSION,
            "routed manifest lacks its flow-cache routing contract",
        )
        flow_root = Path(str(roots.get("flow_cache_root", "")))
        _require(flow_root.is_dir(), "legacy flow root is unavailable")
    else:
        _require(
            manifest.get("schema_version")
            in ROUTED_INPUT_MANIFEST_SCHEMA_VERSIONS,
            "legacy manifest cannot enable multi-root flow routing",
        )
        flow_root = None
    episode_records = _episode_index(manifest)
    grouped = _selected_episode_decisions(manifest, selected_scenes)
    estimator_provenance = _json_safe(estimator.provenance(), "estimator provenance")
    _require(isinstance(estimator_provenance, Mapping),
             "estimator provenance must be a mapping")
    producer_sources = {
        Path(__file__).name: sha256_file(Path(__file__)),
        "build_novel_frontier_candidates.py": sha256_file(
            Path(__file__).with_name("build_novel_frontier_candidates.py")),
        "flow_cache_routing.py": sha256_file(
            Path(__file__).with_name("flow_cache_routing.py")),
    }
    producer_sha = sha256_bytes(canonical_json_bytes(producer_sources))
    records = []
    for scene, episode in sorted(grouped):
        _require((scene, episode) in episode_records,
                 f"sample episode is absent from scene table: {scene}/{episode}")
        episode_record = episode_records[(scene, episode)]
        decisions = sorted(grouped[(scene, episode)]["decisions"])
        split_roles = sorted(grouped[(scene, episode)]["split_roles"])
        _require(len(split_roles) == 1 and split_roles[0] in ("train", "development"),
                 "scale episode crosses or leaves train/development roles")
        scene_records = manifest.get("scenes")
        assert isinstance(scene_records, list)
        scene_matches = [
            row for row in scene_records
            if isinstance(row, Mapping) and row.get("scene") == scene
        ]
        _require(
            len(scene_matches) == 1
            and scene_matches[0].get("split_role") == split_roles[0],
            "sample split role differs from its scene role",
        )
        earliest_decision = int(decisions[0])
        prefix_count = min(configuration.prefix_frame_cap, earliest_decision)
        _require(
            prefix_count >= configuration.num_scale_frames,
            f"earliest decision precedes complete scale block: {scene}/{episode}",
        )
        episode_dir = episode_root / scene / episode
        metadata_path = _validate_file_record(
            episode_record.get("metadata"), episode_root, "episode metadata")
        _require(
            metadata_path.resolve()
            == (episode_dir / METADATA_RELATIVE).resolve(),
            "manifest metadata path differs from the selected episode",
        )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CausalScaleError(
                f"episode metadata is invalid: {metadata_path}") from error
        frame_count = metadata.get("n_frames")
        camera_height = metadata.get("camera_height_m", 0.5)
        _require(
            isinstance(frame_count, int)
            and not isinstance(frame_count, bool)
            and frame_count >= earliest_decision,
            "metadata frame count does not cover selected decisions",
        )
        _require(
            episode_record.get("n_frames") == frame_count,
            "metadata frame count differs from the manifest episode record",
        )
        _require(
            isinstance(camera_height, (int, float))
            and not isinstance(camera_height, bool)
            and math.isfinite(float(camera_height))
            and float(camera_height) > 0.0,
            "metadata camera height is invalid",
        )
        if route_registry is None:
            aggregator_path = _flow_file(
                episode_record, flow_root, scene, episode,
                AGGREGATOR_FLOW_RELATIVE)
            camera_path = _flow_file(
                episode_record, flow_root, scene, episode, FLOW_RELATIVE)
        else:
            try:
                aggregator_path, camera_path = (
                    route_registry.resolve_manifest_pair(
                        episode_record, scene, episode))
            except FlowRoutingError as error:
                raise CausalScaleError(
                    f"routed flow cache binding failed for "
                    f"{scene}/{episode}: {error}") from error
        camera_signature_before = (
            camera_path.stat().st_size, camera_path.stat().st_mtime_ns)
        cache_pair_validator(aggregator_path, camera_path, int(frame_count))
        try:
            with np.load(camera_path, allow_pickle=False) as camera_cache:
                cam_pose = np.asarray(camera_cache["cam_pose_enc"])
                cache_schema_version = int(np.asarray(
                    camera_cache["cache_schema_version"]).reshape(-1)[0])
                precompute_signature = str(np.asarray(
                    camera_cache["precompute_signature"]).reshape(-1)[0])
        except Exception as error:
            raise CausalScaleError(
                f"cannot read camera pose cache: {camera_path}: {error}") from error
        _require(
            cam_pose.shape == (int(frame_count), 9)
            and np.issubdtype(cam_pose.dtype, np.number)
            and bool(np.isfinite(cam_pose[:prefix_count]).all()),
            "causal camera-pose prefix must be dense finite [prefix,9]",
        )
        pose_prefix = np.asarray(cam_pose[:prefix_count]).copy()
        pose_prefix_sha = ndarray_sha256(pose_prefix)
        rgb_record = rgb_prefix_record(
            episode_dir, episode_root, prefix_count)
        rgb_paths = [
            episode_dir / RGB_RELATIVE / f"{frame}.jpg"
            for frame in range(prefix_count)
        ]
        estimate = estimator.estimate(
            rgb_paths=rgb_paths,
            cam_pose_prefix=pose_prefix,
            camera_height_m=float(camera_height),
            configuration=configuration,
        )
        # The estimator opens the JPEGs lazily.  Re-hash after inference so an
        # in-place data refresh cannot make provenance describe different
        # bytes than the model actually consumed.
        _require(
            rgb_prefix_record(episode_dir, episode_root, prefix_count)
            == rgb_record,
            "causal RGB prefix changed while scale was estimated",
        )
        _require(
            ndarray_sha256(pose_prefix) == pose_prefix_sha,
            "estimator mutated the causal camera-pose prefix",
        )
        _require(
            (camera_path.stat().st_size, camera_path.stat().st_mtime_ns)
            == camera_signature_before,
            "camera cache changed while scale was estimated",
        )
        _require(isinstance(estimate, GroundScaleEstimate),
                 "estimator returned a wrong result type")
        debug = _json_safe(estimate.debug, "scale debug")
        if (estimate.ground_h_est_raw is None
                or estimate.metric_scale_m_per_raw is None):
            _require(
                estimate.ground_h_est_raw is None
                and estimate.metric_scale_m_per_raw is None,
                "invalid estimate must neutralize both height and scale",
            )
            valid = False
            invalid_reason = "floor_height_unavailable_from_causal_prefix"
            ground_h = None
            metric_scale = None
        else:
            ground_h = float(estimate.ground_h_est_raw)
            metric_scale = float(estimate.metric_scale_m_per_raw)
            expected_scale = expected_scale_from_ground(
                ground_h, float(camera_height), configuration)
            _require(
                math.isfinite(metric_scale)
                and math.isclose(
                    metric_scale, expected_scale, rel_tol=1e-6, abs_tol=1e-6),
                "estimator scale disagrees with pinned ground-scale formula",
            )
            valid = True
            invalid_reason = None
        records.append({
            "scene": scene,
            "episode": episode,
            "split_role": split_roles[0],
            "sample_ids": sorted(grouped[(scene, episode)]["sample_ids"]),
            "decision_frames": decisions,
            "earliest_decision_frame": earliest_decision,
            "prefix_end_frame_exclusive": prefix_count,
            "camera_height_m": float(camera_height),
            "episode_frame_count": int(frame_count),
            "rgb_prefix": rgb_record,
            "cam_pose_prefix_sha256": pose_prefix_sha,
            "cam_pose_prefix_dtype": pose_prefix.dtype.str,
            "cache_schema_version": cache_schema_version,
            "precompute_signature": precompute_signature,
            "valid": valid,
            "invalid_reason": invalid_reason,
            "ground_h_est_raw": ground_h,
            "metric_scale_m_per_raw": metric_scale,
            "debug": debug,
        })
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "causal first-prefix LingBot ground scale; no GT pose, goal, "
            "Pathfinder, future frame, or whole-episode floor estimate"
        ),
        "provenance": {
            "input_manifest_path": str(manifest_path.resolve()),
            "input_manifest_sha256": expected_manifest_sha256,
            "input_manifest_schema_version": manifest["schema_version"],
            "flow_cache_routing": (
                route_registry.manifest_record()
                if route_registry is not None else {
                    "mode": "strict_single_root_v1",
                    "flow_cache_root": str(flow_root.resolve()),
                }
            ),
            "producer_source_sha256": producer_sha,
            "producer_source_files": producer_sources,
            "configuration_sha256": configuration.sha256,
            "estimator": estimator_provenance,
        },
        "configuration": asdict(configuration),
        "selection": {
            "selected_scenes": sorted(set(map(str, selected_scenes))),
            "all_manifest_scenes": not bool(selected_scenes),
        },
        "records": records,
        "summary": {
            "scene_count": len({record["scene"] for record in records}),
            "episode_count": len(records),
            "valid_episode_count": sum(int(record["valid"]) for record in records),
            "invalid_episode_count": sum(int(not record["valid"]) for record in records),
            "maximum_prefix_frames": max(
                int(record["prefix_end_frame_exclusive"]) for record in records),
            "future_frames_consumed": 0,
        },
    }
    canonical_json_bytes(artifact)
    return artifact


class LingBotGroundScaleEstimator:
    """Lazy real estimator backed by the frozen LingBot depth/pose stream."""

    def __init__(
        self,
        *,
        lingbot_repo: Path,
        weights: Path,
        expected_lingbot_commit: str,
        expected_weights_sha256: str,
        expected_stream_source_sha256: str,
        configuration: GroundScaleConfiguration,
        device: str = "cuda",
    ) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _require(
            os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"),
            "CUBLAS_WORKSPACE_CONFIG must enable deterministic CUDA",
        )
        import torch
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        _require(lingbot_repo.is_dir(), "LingBot repository is missing")
        _require(weights.is_file(), "LingBot weights are missing")
        _require(
            _git_commit_is_valid(expected_lingbot_commit)
            and _sha_is_valid(expected_weights_sha256)
            and _sha_is_valid(expected_stream_source_sha256),
            "LingBot source/weight pins are invalid",
        )
        _require(
            device == "cuda" or device.startswith("cuda:"),
            "LingBot metric-scale inference currently requires CUDA",
        )
        commit = subprocess.check_output(
            ["git", "-C", str(lingbot_repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        _require(commit == expected_lingbot_commit,
                 "LingBot repository commit differs from its pin")
        tracked_changes = subprocess.check_output(
            ["git", "-C", str(lingbot_repo), "status", "--porcelain",
             "--untracked-files=no"],
            text=True,
        ).strip()
        _require(not tracked_changes,
                 "LingBot repository has uncommitted tracked changes")
        weight_sha = sha256_file(weights)
        _require(weight_sha == expected_weights_sha256,
                 "LingBot weights differ from their pin")
        stream_source = (
            Path(__file__).resolve().parents[1]
            / "InternNav/internnav/model/basemodel/memnav/lingbot_stream.py"
        )
        stream_sha = sha256_file(stream_source)
        _require(stream_sha == expected_stream_source_sha256,
                 "LingBotStream source differs from its pin")
        repository_root = str(Path(__file__).resolve().parents[1])
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
        from InternNav.internnav.model.basemodel.memnav.lingbot_stream import (
            LingBotStream,
        )
        self.model = LingBotStream(
            lingbot_repo=str(lingbot_repo),
            weights=str(weights),
            num_scale=configuration.num_scale_frames,
            window=configuration.window,
            max_frame_num=configuration.max_frame_num,
            use_sdpa=configuration.use_sdpa,
            device=device,
        )
        self._provenance = {
            "kind": "frozen_lingbot_compute_metric_scale_prefix",
            "lingbot_commit": commit,
            "weights_path": str(weights.resolve()),
            "weights_sha256": weight_sha,
            "lingbot_stream_source_sha256": stream_sha,
            "device": device,
        }
        self._configuration_sha256 = configuration.sha256

    def provenance(self) -> Mapping[str, object]:
        return self._provenance

    def estimate(
        self,
        *,
        rgb_paths: Sequence[Path],
        cam_pose_prefix: np.ndarray,
        camera_height_m: float,
        configuration: GroundScaleConfiguration,
    ) -> GroundScaleEstimate:
        _require(
            configuration.sha256 == self._configuration_sha256,
            "runtime ground-scale configuration differs from model setup",
        )
        result = self.model.compute_metric_scale(
            [str(path) for path in rgb_paths],
            cam_pose_prefix,
            camera_height_m=float(camera_height_m),
            conf_quantile=configuration.conf_quantile,
            pixel_stride=configuration.pixel_stride,
            nbins=configuration.histogram_bins,
            n_frames=len(rgb_paths),
            peak_thresh=configuration.peak_threshold,
            bias_correction=configuration.bias_correction,
            scale_range=(configuration.scale_min, configuration.scale_max),
            return_debug=True,
        )
        _require(isinstance(result, tuple) and len(result) == 2,
                 "LingBot metric-scale estimator returned a wrong structure")
        scale, debug = result
        _require(isinstance(debug, Mapping),
                 "LingBot metric-scale debug record is missing")
        h_est = debug.get("h_est")
        if scale is None or h_est is None:
            return GroundScaleEstimate(None, None, dict(debug))
        return GroundScaleEstimate(
            float(h_est), float(scale), dict(debug))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-lingbot-commit", required=True)
    parser.add_argument("--expected-weights-sha", required=True)
    parser.add_argument("--expected-stream-source-sha", required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--prefix-frame-cap", type=int, default=64)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--conf-quantile", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--histogram-bins", type=int, default=60)
    parser.add_argument("--peak-threshold", type=float, default=0.3)
    parser.add_argument("--bias-correction", type=float, default=1.15)
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=6.0)
    parser.add_argument("--disable-sdpa", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha-out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configuration = GroundScaleConfiguration(
        prefix_frame_cap=args.prefix_frame_cap,
        num_scale_frames=args.num_scale_frames,
        window=args.window,
        max_frame_num=args.max_frame_num,
        conf_quantile=args.conf_quantile,
        pixel_stride=args.pixel_stride,
        histogram_bins=args.histogram_bins,
        peak_threshold=args.peak_threshold,
        bias_correction=args.bias_correction,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        use_sdpa=not args.disable_sdpa,
    )
    _require(args.manifest.is_file(), "causal manifest is missing")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CausalScaleError("causal manifest is invalid JSON") from error
    _require(isinstance(manifest, Mapping), "causal manifest must be an object")
    estimator = LingBotGroundScaleEstimator(
        lingbot_repo=args.lingbot_repo,
        weights=args.weights,
        expected_lingbot_commit=args.expected_lingbot_commit,
        expected_weights_sha256=args.expected_weights_sha,
        expected_stream_source_sha256=args.expected_stream_source_sha,
        configuration=configuration,
        device=args.device,
    )
    artifact = build_scale_artifact(
        manifest=manifest,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha,
        estimator=estimator,
        configuration=configuration,
        selected_scenes=args.scene,
    )
    sha_out = args.sha_out or Path(f"{args.out}.sha256")
    status, digest = write_artifact(
        artifact,
        args.out,
        sha_out,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": status,
        "output": str(args.out),
        "sha_output": str(sha_out),
        "sha256": digest,
        **artifact["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
