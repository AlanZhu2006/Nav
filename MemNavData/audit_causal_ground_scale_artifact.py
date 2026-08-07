#!/usr/bin/env python3
"""Independently rebind a causal LingBot scale artifact to its real inputs.

The GPU producer already records the causal RGB and camera-pose prefix used for
each episode.  This CPU-only auditor does not estimate scale again.  Instead it
reopens the exact routed LingBot cache pair and source RGB stream, validates the
versioned cache layout, re-hashes every recorded causal prefix through
``ExternalCausalScaleContract``, and publishes one immutable acceptance receipt.

Only prefix bytes are hashed as scale evidence.  Hashing a complete camera
cache would unnecessarily bind the acceptance decision to future rows that the
causal estimator was forbidden to consume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np

try:
    from MemNavData.build_novel_frontier_candidates import (
        _validate_versioned_cache_pair,
    )
    from MemNavData.external_causal_scale_contract import (
        ExternalCausalScaleContract,
        ExternalCausalScaleError,
        ExternalCausalScalePins,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
    )
    from MemNavData.flow_cache_routing import (
        FlowRouteRegistry,
        FlowRoutingError,
        registry_from_manifest,
    )
except ModuleNotFoundError:  # direct ``python MemNavData/<script>.py``
    from build_novel_frontier_candidates import (  # type: ignore
        _validate_versioned_cache_pair,
    )
    from external_causal_scale_contract import (  # type: ignore
        ExternalCausalScaleContract,
        ExternalCausalScaleError,
        ExternalCausalScalePins,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
    )
    from flow_cache_routing import (  # type: ignore
        FlowRouteRegistry,
        FlowRoutingError,
        registry_from_manifest,
    )


RECEIPT_SCHEMA_VERSION = "nlsr_causal_ground_scale_acceptance_v1"
RECEIPT_STATUS = "causal_prefixes_physically_rebound"
CAUSAL_BOUNDARY = (
    "acceptance re-hashes only each episode prefix ending no later than its "
    "earliest selected decision; no goal pixels, future camera rows, Habitat "
    "pose, Pathfinder result, evaluation metric, or pooled scale are consumed"
)
PRODUCER_SOURCE_PATHS = {
    "build_causal_ground_scale.py": "MemNavData/build_causal_ground_scale.py",
    "build_novel_frontier_candidates.py": (
        "MemNavData/build_novel_frontier_candidates.py"
    ),
    "flow_cache_routing.py": "MemNavData/flow_cache_routing.py",
}


class CausalScaleAuditError(RuntimeError):
    """The scale artifact could not be independently accepted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalScaleAuditError(message)


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def _commit(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a full lowercase Git commit",
    )
    return value


def _positive_int(value: object, label: str) -> int:
    _require(
        type(value) is int and int(value) >= 1, f"{label} must be a positive integer"
    )
    return int(value)


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    byte_count: int
    mtime_ns: int
    ctime_ns: int
    content_sha256: str


def _snapshot(path: Path | str, label: str) -> FileSnapshot:
    source = Path(path)
    _require(
        source.is_file() and not source.is_symlink(),
        f"{label} is absent, non-regular, or symlinked: {source}",
    )
    resolved = source.resolve(strict=True)
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{label} changed while hashing: {resolved}",
    )
    return FileSnapshot(
        path=resolved,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        byte_count=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
        content_sha256=digest,
    )


def _assert_unchanged(snapshot: FileSnapshot, label: str) -> None:
    current = _snapshot(snapshot.path, label)
    _require(current == snapshot, f"{label} changed during audit")


@dataclass(frozen=True)
class PhysicalFileSnapshot:
    """Cheap identity check for large routed caches.

    The acceptance evidence hashes only the causal pose prefix.  Hashing the
    complete cache here would both waste I/O and make future pose/KV rows part
    of the scale acceptance decision.  Device/inode/size/mtime/ctime still
    detect replacement or in-place mutation around the prefix validation.
    """

    path: Path
    device: int
    inode: int
    byte_count: int
    mtime_ns: int
    ctime_ns: int


def _physical_snapshot(path: Path | str, label: str) -> PhysicalFileSnapshot:
    source = Path(path)
    _require(
        source.is_file() and not source.is_symlink(),
        f"{label} is absent, non-regular, or symlinked: {source}",
    )
    resolved = source.resolve(strict=True)
    stat = resolved.stat()
    return PhysicalFileSnapshot(
        path=resolved,
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
        byte_count=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        ctime_ns=int(stat.st_ctime_ns),
    )


def _assert_physical_unchanged(
    snapshot: PhysicalFileSnapshot,
    label: str,
) -> None:
    _require(
        _physical_snapshot(snapshot.path, label) == snapshot,
        f"{label} changed during audit",
    )


def _load_canonical(path: Path, expected_sha: str, label: str) -> tuple[dict, bytes]:
    snapshot = _snapshot(path, label)
    _require(
        snapshot.content_sha256 == _sha(expected_sha, f"{label} pin"),
        f"{label} SHA256 changed",
    )
    try:
        raw = snapshot.path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CausalScaleAuditError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    _require(raw == canonical_json_bytes(value), f"{label} is not canonical JSON")
    _assert_unchanged(snapshot, label)
    return value, raw


def _verify_sidecar(path: Path, digest: str, label: str) -> FileSnapshot:
    sidecar = _snapshot(Path(f"{path}.sha256"), f"{label} sidecar")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    _require(
        sidecar.byte_count == len(expected)
        and sidecar.content_sha256 == sha256_bytes(expected)
        and sidecar.path.read_bytes() == expected,
        f"{label} sidecar is non-canonical or mismatched",
    )
    return sidecar


def _episode_index(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    scenes = manifest.get("scenes")
    _require(isinstance(scenes, list), "manifest scenes are absent")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_scene in scenes:
        _require(isinstance(raw_scene, Mapping), "manifest scene is malformed")
        scene = raw_scene.get("scene")
        episodes = raw_scene.get("selected_episodes")
        _require(
            isinstance(scene, str) and isinstance(episodes, list),
            "manifest scene identity/episodes are malformed",
        )
        for raw_episode in episodes:
            _require(isinstance(raw_episode, Mapping), "manifest episode is malformed")
            episode = raw_episode.get("episode")
            key = scene, episode
            _require(
                isinstance(episode, str) and key not in result,
                "manifest episode is empty or duplicated",
            )
            result[(scene, episode)] = raw_episode
    _require(bool(result), "manifest contains no selected episodes")
    return result


CacheResolver = Callable[[Mapping[str, Any], str, str], tuple[Path, Path]]
CacheValidator = Callable[..., object]


def _routed_resolver(manifest: Mapping[str, Any]) -> CacheResolver:
    try:
        registry = registry_from_manifest(manifest)
    except FlowRoutingError as error:
        raise CausalScaleAuditError(
            f"pinned flow-cache routing is invalid: {error}"
        ) from error
    _require(
        isinstance(registry, FlowRouteRegistry),
        "formal scale acceptance requires provenance-pinned multi-root routing",
    )

    def resolve(
        episode_record: Mapping[str, Any],
        scene: str,
        episode: str,
    ) -> tuple[Path, Path]:
        try:
            return registry.resolve_manifest_pair(episode_record, scene, episode)
        except FlowRoutingError as error:
            raise CausalScaleAuditError(
                f"routed cache rebinding failed for {scene}/{episode}: {error}"
            ) from error

    return resolve


def _load_runtime_camera(
    camera_path: Path,
    *,
    expected_num_frames: int,
    prefix_frame_count: int,
) -> tuple[np.ndarray, int, str]:
    try:
        with np.load(camera_path, allow_pickle=False) as camera_cache:
            _require(
                "cam_pose_enc" in camera_cache.files, "camera cache lacks cam_pose_enc"
            )
            schema_value = _cache_scalar(camera_cache, "cache_schema_version")
            signature_value = _cache_scalar(camera_cache, "precompute_signature")
        with zipfile.ZipFile(camera_path) as archive:
            with archive.open("cam_pose_enc.npy") as handle:
                version = np.lib.format.read_magic(handle)
                shape, fortran_order, dtype = np.lib.format._read_array_header(
                    handle, version
                )
                _require(
                    tuple(shape) == (expected_num_frames, 9)
                    and not fortran_order
                    and np.issubdtype(dtype, np.number)
                    and not dtype.hasobject,
                    "camera cache cam_pose_enc must be C-order numeric [num_frames,9]",
                )
                _require(
                    1 <= prefix_frame_count <= expected_num_frames,
                    "camera-pose prefix frame count is out of range",
                )
                prefix_bytes = prefix_frame_count * 9 * dtype.itemsize
                raw_prefix = handle.read(prefix_bytes)
                _require(
                    len(raw_prefix) == prefix_bytes,
                    "camera cache ends within its causal pose prefix",
                )
                cam_pose_enc = np.frombuffer(raw_prefix, dtype=dtype).reshape(
                    prefix_frame_count, 9
                )
    except CausalScaleAuditError:
        raise
    except Exception as error:
        raise CausalScaleAuditError(
            f"cannot read camera-pose cache: {camera_path}: {error}"
        ) from error
    _require(
        cam_pose_enc.shape == (prefix_frame_count, 9)
        and np.issubdtype(cam_pose_enc.dtype, np.number),
        "camera cache causal cam_pose_enc prefix has a wrong shape or dtype",
    )
    _require(
        isinstance(schema_value, (int, np.integer))
        and not isinstance(schema_value, (bool, np.bool_)),
        "camera cache schema version is not an integer scalar",
    )
    _require(
        isinstance(signature_value, (str, np.str_)) and bool(str(signature_value)),
        "camera cache precompute signature is not a non-empty string scalar",
    )
    return cam_pose_enc, int(schema_value), str(signature_value)


@dataclass(frozen=True)
class AuditContract:
    manifest_path: Path
    artifact_path: Path
    output_path: Path
    repository_root: Path
    auditor_path: Path
    expected_manifest_sha256: str
    expected_artifact_sha256: str
    expected_producer_sha256: str
    expected_configuration_sha256: str
    expected_lingbot_commit: str
    expected_weights_sha256: str
    expected_stream_source_sha256: str
    expected_auditor_sha256: str
    expected_acceptance_commit: str
    expected_scene_count: int
    expected_episode_count: int
    expected_sample_count: int

    def validate(self) -> "AuditContract":
        for value, label in (
            (self.expected_manifest_sha256, "manifest SHA256"),
            (self.expected_artifact_sha256, "scale artifact SHA256"),
            (self.expected_producer_sha256, "scale producer bundle SHA256"),
            (self.expected_configuration_sha256, "scale configuration SHA256"),
            (self.expected_weights_sha256, "LingBot weights SHA256"),
            (self.expected_stream_source_sha256, "LingBot stream SHA256"),
            (self.expected_auditor_sha256, "auditor SHA256"),
        ):
            _sha(value, label)
        _commit(self.expected_lingbot_commit, "LingBot commit")
        _commit(self.expected_acceptance_commit, "acceptance commit")
        _positive_int(self.expected_scene_count, "expected scene count")
        _positive_int(self.expected_episode_count, "expected episode count")
        _positive_int(self.expected_sample_count, "expected sample count")
        return self


def _cache_scalar(cache: Any, name: str) -> object:
    _require(name in cache.files, f"camera cache lacks {name}")
    value = np.asarray(cache[name])
    _require(value.size == 1, f"camera cache {name} is not scalar")
    return value.reshape(-1)[0]


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(f"{path}.sha256")
    digest = sha256_bytes(payload)
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    if path.exists() or sidecar.exists():
        _require(
            path.is_file()
            and sidecar.is_file()
            and not path.is_symlink()
            and not sidecar.is_symlink(),
            "existing acceptance output pair is incomplete",
        )
        _require(path.read_bytes() == payload, "existing acceptance receipt differs")
        _require(
            sidecar.read_bytes() == sidecar_payload,
            "existing acceptance receipt sidecar differs",
        )
        return
    descriptors: list[tuple[str, Path]] = []
    try:
        for name, destination, content in (
            ("receipt", path, payload),
            ("sidecar", sidecar, sidecar_payload),
        ):
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.{name}.",
                dir=destination.parent,
                delete=False,
            )
            temporary = Path(handle.name)
            descriptors.append((name, temporary))
            with handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(descriptors[0][1], path)
        os.replace(descriptors[1][1], sidecar)
    finally:
        for _name, temporary in descriptors:
            temporary.unlink(missing_ok=True)


def audit_causal_ground_scale(
    contract: AuditContract,
    *,
    cache_resolver: CacheResolver | None = None,
    cache_validator: CacheValidator = _validate_versioned_cache_pair,
) -> dict[str, object]:
    """Reopen every physical prefix and publish one idempotent receipt."""

    contract.validate()
    manifest, manifest_raw = _load_canonical(
        contract.manifest_path, contract.expected_manifest_sha256, "manifest"
    )
    artifact, artifact_raw = _load_canonical(
        contract.artifact_path,
        contract.expected_artifact_sha256,
        "causal scale artifact",
    )
    manifest_sidecar = _verify_sidecar(
        contract.manifest_path, sha256_bytes(manifest_raw), "manifest"
    )
    artifact_sidecar = _verify_sidecar(
        contract.artifact_path, sha256_bytes(artifact_raw), "causal scale artifact"
    )

    auditor_snapshot = _snapshot(contract.auditor_path, "scale auditor source")
    _require(
        auditor_snapshot.content_sha256 == contract.expected_auditor_sha256,
        "scale auditor source SHA256 changed",
    )
    repository_root = contract.repository_root.resolve(strict=True)
    _require(repository_root.is_dir(), "repository root is unavailable")

    provenance = artifact.get("provenance")
    _require(isinstance(provenance, Mapping), "scale provenance is malformed")
    source_files = provenance.get("producer_source_files")
    _require(
        isinstance(source_files, Mapping), "scale producer source-file map is absent"
    )
    expected_source_files: dict[str, str] = {}
    producer_source_snapshots: dict[str, FileSnapshot] = {}
    for name, relative in PRODUCER_SOURCE_PATHS.items():
        source = _snapshot(repository_root / relative, f"producer source {name}")
        producer_source_snapshots[name] = source
        expected_source_files[name] = source.content_sha256
    _require(
        dict(source_files) == expected_source_files,
        "scale producer source-file bytes differ from its artifact",
    )
    _require(
        sha256_bytes(canonical_json_bytes(expected_source_files))
        == contract.expected_producer_sha256
        == provenance.get("producer_source_sha256"),
        "scale producer bundle binding changed",
    )
    _require(
        provenance.get("input_manifest_path") == str(contract.manifest_path.resolve()),
        "scale artifact names a different physical manifest",
    )
    manifest_routing = manifest.get("flow_cache_routing")
    _require(
        isinstance(manifest_routing, Mapping)
        and provenance.get("flow_cache_routing") == manifest_routing,
        "scale artifact and manifest do not bind the same flow-cache routing",
    )
    route_artifact_path = Path(str(manifest_routing.get("artifact_path", "")))
    route_artifact_sha = _sha(
        manifest_routing.get("artifact_sha256"), "flow route artifact SHA256"
    )
    route_artifact_snapshot = _snapshot(route_artifact_path, "flow route artifact")
    _require(
        route_artifact_snapshot.content_sha256 == route_artifact_sha,
        "flow route artifact SHA256 changed",
    )
    route_sidecar = _verify_sidecar(
        route_artifact_path, route_artifact_sha, "flow route artifact"
    )

    pins = ExternalCausalScalePins(
        manifest_sha256=contract.expected_manifest_sha256,
        artifact_sha256=contract.expected_artifact_sha256,
        producer_sha256=contract.expected_producer_sha256,
        configuration_sha256=contract.expected_configuration_sha256,
        lingbot_commit=contract.expected_lingbot_commit,
        weights_sha256=contract.expected_weights_sha256,
        stream_source_sha256=contract.expected_stream_source_sha256,
    )
    try:
        scale = ExternalCausalScaleContract(
            manifest_path=contract.manifest_path,
            artifact_path=contract.artifact_path,
            pins=pins,
        )
    except ExternalCausalScaleError as error:
        raise CausalScaleAuditError(str(error)) from error

    episodes = _episode_index(manifest)
    samples = manifest.get("samples")
    _require(isinstance(samples, list), "manifest samples are absent")
    sample_ids = [row.get("sample_id") for row in samples if isinstance(row, Mapping)]
    _require(
        len(sample_ids) == len(samples)
        and all(isinstance(value, str) and value for value in sample_ids)
        and len(set(sample_ids)) == len(sample_ids),
        "manifest sample identities are malformed or duplicated",
    )
    scene_count = len({scene for scene, _episode in episodes})
    _require(
        scene_count == contract.expected_scene_count,
        "manifest scene count differs from acceptance contract",
    )
    _require(
        len(episodes) == contract.expected_episode_count,
        "manifest episode count differs from acceptance contract",
    )
    _require(
        len(samples) == contract.expected_sample_count,
        "manifest sample count differs from acceptance contract",
    )

    manifest_summary = manifest.get("summary")
    _require(isinstance(manifest_summary, Mapping), "manifest summary is absent")
    for name, expected in (
        ("scene_count", contract.expected_scene_count),
        ("episode_count", contract.expected_episode_count),
        ("sample_count", contract.expected_sample_count),
    ):
        _require(
            manifest_summary.get(name) == expected, f"manifest summary {name} changed"
        )

    artifact_records = artifact.get("records")
    _require(isinstance(artifact_records, list), "scale records are absent")
    artifact_summary = artifact.get("summary")
    _require(isinstance(artifact_summary, Mapping), "scale summary is absent")
    recomputed_summary = {
        "scene_count": len(
            {row.get("scene") for row in artifact_records if isinstance(row, Mapping)}
        ),
        "episode_count": len(artifact_records),
        "valid_episode_count": sum(
            int(row.get("valid") is True)
            for row in artifact_records
            if isinstance(row, Mapping)
        ),
        "invalid_episode_count": sum(
            int(row.get("valid") is False)
            for row in artifact_records
            if isinstance(row, Mapping)
        ),
        "maximum_prefix_frames": max(
            int(row.get("prefix_end_frame_exclusive", 0))
            for row in artifact_records
            if isinstance(row, Mapping)
        ),
        "future_frames_consumed": 0,
    }
    _require(
        dict(artifact_summary) == recomputed_summary,
        "scale artifact summary differs from its records",
    )
    _require(
        recomputed_summary["episode_count"] == contract.expected_episode_count
        and recomputed_summary["valid_episode_count"] == contract.expected_episode_count
        and recomputed_summary["invalid_episode_count"] == 0,
        "formal acceptance requires one valid scale per episode",
    )

    resolver = cache_resolver or _routed_resolver(manifest)
    record_by_episode = {
        (row["scene"], row["episode"]): row
        for row in artifact_records
        if isinstance(row, Mapping)
    }
    _require(
        set(record_by_episode) == set(episodes),
        "scale records do not exactly cover manifest episodes",
    )

    episode_witnesses: list[dict[str, object]] = []
    rebound_samples: set[str] = set()
    cache_snapshots: list[tuple[str, PhysicalFileSnapshot]] = []
    rgb_snapshots: list[tuple[str, FileSnapshot]] = []
    runtime_rechecks: list[tuple[str, str, Path, int, int]] = []
    for scene, episode in sorted(episodes):
        episode_record = episodes[(scene, episode)]
        n_frames = _positive_int(episode_record.get("n_frames"), "episode frame count")
        aggregator_path, camera_path = resolver(episode_record, scene, episode)
        aggregator_before = _physical_snapshot(
            aggregator_path, "LingBot aggregator cache"
        )
        camera_before = _physical_snapshot(camera_path, "LingBot camera cache")
        scale_record = record_by_episode[(scene, episode)]
        prefix_end = _positive_int(
            scale_record.get("prefix_end_frame_exclusive"),
            "scale prefix frame count",
        )
        episode_root = scale.dataset_root / scene / episode
        episode_rgb_snapshots = [
            _snapshot(
                episode_root
                / "videos/chunk-000/observation.images.rgb"
                / f"{frame}.jpg",
                f"causal RGB {scene}/{episode}/{frame}",
            )
            for frame in range(prefix_end)
        ]
        try:
            cache_validator(
                aggregator_path,
                camera_path,
                expected_num_frames=n_frames,
            )
            cam_pose_enc, cache_schema_version, precompute_signature = (
                _load_runtime_camera(
                    camera_path,
                    expected_num_frames=n_frames,
                    prefix_frame_count=prefix_end,
                )
            )
            scale.validate_runtime_episode(
                scene=scene,
                episode=episode,
                cam_pose_enc=cam_pose_enc,
                cache_schema_version=cache_schema_version,
                precompute_signature=precompute_signature,
            )
        except ExternalCausalScaleError as error:
            raise CausalScaleAuditError(str(error)) from error
        except CausalScaleAuditError:
            raise
        except Exception as error:
            raise CausalScaleAuditError(
                f"cannot validate cache pair for {scene}/{episode}: {error}"
            ) from error
        _assert_physical_unchanged(aggregator_before, "LingBot aggregator cache")
        _assert_physical_unchanged(camera_before, "LingBot camera cache")
        for frame, snapshot in enumerate(episode_rgb_snapshots):
            _assert_unchanged(snapshot, f"causal RGB {scene}/{episode}/{frame}")
            rgb_snapshots.append((f"causal RGB {scene}/{episode}/{frame}", snapshot))
        cache_snapshots.extend(
            (
                (f"LingBot aggregator cache {scene}/{episode}", aggregator_before),
                (f"LingBot camera cache {scene}/{episode}", camera_before),
            )
        )
        runtime_rechecks.append(
            (scene, episode, camera_before.path, n_frames, prefix_end)
        )

        episode_sample_ids = scale_record.get("sample_ids")
        _require(
            isinstance(episode_sample_ids, list)
            and all(isinstance(value, str) for value in episode_sample_ids),
            "scale record sample_ids are malformed",
        )
        rebound_samples.update(episode_sample_ids)
        episode_witnesses.append(
            {
                "scene": scene,
                "episode": episode,
                "split_role": scale_record["split_role"],
                "sample_count": len(episode_sample_ids),
                "prefix_end_frame_exclusive": (
                    scale_record["prefix_end_frame_exclusive"]
                ),
                "cam_pose_prefix_sha256": scale_record["cam_pose_prefix_sha256"],
                "cam_pose_prefix_dtype": scale_record["cam_pose_prefix_dtype"],
                "rgb_prefix_content_sequence_sha256": (
                    scale_record["rgb_prefix"]["content_sequence_sha256"]
                ),
                "cache_schema_version": cache_schema_version,
                "precompute_signature": precompute_signature,
                "aggregator_cache_path": str(aggregator_before.path),
                "camera_cache_path": str(camera_before.path),
                "aggregator_cache_bytes": aggregator_before.byte_count,
                "camera_cache_bytes": camera_before.byte_count,
            }
        )

    _require(
        rebound_samples == set(sample_ids),
        "runtime-rebound episodes do not exactly cover manifest samples",
    )
    sample_witnesses = [
        scale.expected_row_binding(sample_id) for sample_id in sorted(sample_ids)
    ]
    sample_binding_sha = sha256_bytes(canonical_json_bytes(sample_witnesses))
    episode_witness_sha = sha256_bytes(canonical_json_bytes(episode_witnesses))

    # A second independent open catches replacement after an episode's first
    # validation.  It still hashes only the artifact-declared causal prefix.
    for scene, episode, camera_path, n_frames, prefix_end in runtime_rechecks:
        cam_pose_enc, cache_schema_version, precompute_signature = _load_runtime_camera(
            camera_path,
            expected_num_frames=n_frames,
            prefix_frame_count=prefix_end,
        )
        try:
            scale.validate_runtime_episode(
                scene=scene,
                episode=episode,
                cam_pose_enc=cam_pose_enc,
                cache_schema_version=cache_schema_version,
                precompute_signature=precompute_signature,
            )
        except ExternalCausalScaleError as error:
            raise CausalScaleAuditError(str(error)) from error
    for label, snapshot in cache_snapshots:
        _assert_physical_unchanged(snapshot, label)
    for label, snapshot in rgb_snapshots:
        _assert_unchanged(snapshot, label)
    for name, snapshot in producer_source_snapshots.items():
        _assert_unchanged(snapshot, f"producer source {name}")
    _assert_unchanged(auditor_snapshot, "scale auditor source")
    _load_canonical(
        contract.manifest_path, contract.expected_manifest_sha256, "manifest"
    )
    _load_canonical(
        contract.artifact_path,
        contract.expected_artifact_sha256,
        "causal scale artifact",
    )
    _assert_unchanged(manifest_sidecar, "manifest sidecar")
    _assert_unchanged(artifact_sidecar, "causal scale artifact sidecar")
    _assert_unchanged(route_artifact_snapshot, "flow route artifact")
    _assert_unchanged(route_sidecar, "flow route artifact sidecar")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "acceptance_commit": contract.expected_acceptance_commit,
        "inputs": {
            "manifest_path": str(contract.manifest_path.resolve()),
            "manifest_sha256": contract.expected_manifest_sha256,
            "scale_artifact_path": str(contract.artifact_path.resolve()),
            "scale_artifact_sha256": contract.expected_artifact_sha256,
            "flow_route_artifact_path": str(route_artifact_path.resolve()),
            "flow_route_artifact_sha256": route_artifact_sha,
            "flow_routing_record_sha256": sha256_bytes(
                canonical_json_bytes(dict(manifest_routing))
            ),
        },
        "producer": {
            "source_bundle_sha256": contract.expected_producer_sha256,
            "source_files": expected_source_files,
            "configuration_sha256": contract.expected_configuration_sha256,
            "lingbot_commit": contract.expected_lingbot_commit,
            "weights_sha256": contract.expected_weights_sha256,
            "stream_source_sha256": contract.expected_stream_source_sha256,
        },
        "auditor": {
            "path": "MemNavData/audit_causal_ground_scale_artifact.py",
            "content_sha256": contract.expected_auditor_sha256,
        },
        "coverage": {
            "scene_count": scene_count,
            "episode_count": len(episodes),
            "sample_count": len(samples),
            "future_frames_consumed": 0,
            "all_episode_estimates_valid": True,
        },
        "physical_rebinding": {
            "routed_cache_pairs_reopened": len(episodes),
            "independent_prefix_validation_passes": 2,
            "camera_pose_prefix_hash_checks": 2 * len(episodes),
            "rgb_prefix_hash_checks": 2 * len(episodes),
            "sample_binding_sequence_sha256": sample_binding_sha,
            "episode_witness_sequence_sha256": episode_witness_sha,
        },
        "causal_boundary": CAUSAL_BOUNDARY,
    }
    payload = canonical_json_bytes(receipt)
    _atomic_publish(contract.output_path, payload)
    return {
        "causal_scale_acceptance": "passed",
        "receipt_path": str(contract.output_path.resolve()),
        "receipt_sha256": sha256_bytes(payload),
        **receipt["coverage"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--auditor-path", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--expected-producer-sha256", required=True)
    parser.add_argument("--expected-configuration-sha256", required=True)
    parser.add_argument("--expected-lingbot-commit", required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-stream-source-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-acceptance-commit", required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--expected-episode-count", type=int, required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_causal_ground_scale(
        AuditContract(
            manifest_path=args.manifest,
            artifact_path=args.artifact,
            output_path=args.output,
            repository_root=args.repository_root,
            auditor_path=args.auditor_path,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_artifact_sha256=args.expected_artifact_sha256,
            expected_producer_sha256=args.expected_producer_sha256,
            expected_configuration_sha256=args.expected_configuration_sha256,
            expected_lingbot_commit=args.expected_lingbot_commit,
            expected_weights_sha256=args.expected_weights_sha256,
            expected_stream_source_sha256=args.expected_stream_source_sha256,
            expected_auditor_sha256=args.expected_auditor_sha256,
            expected_acceptance_commit=args.expected_acceptance_commit,
            expected_scene_count=args.expected_scene_count,
            expected_episode_count=args.expected_episode_count,
            expected_sample_count=args.expected_sample_count,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
