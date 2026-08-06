#!/usr/bin/env python3
"""Build an atomic, content-addressed geometry bundle for H24 collection.

The input is a canonical expert manifest under an external SHA256 pin.  Every
scene GLB and pre-baked navmesh is checked against that manifest before a
``FrozenGeometryIdentity`` is staged, then hashed a second time before the
bundle is published.  The final directory is installed with one atomic rename
and contains::

    frozen_geometry_map.json
    frozen_geometry_map.json.sha256
    identities/<sha256(scene-id)>.json

``navmesh_settings`` has a deliberately narrow meaning here: it is the exact
settings contract with which the frozen navmesh will be *evaluated and
loaded*.  Recording it does not prove that an already-existing navmesh was
historically baked with those settings.  Establishing historical bake
provenance requires a separate trusted generation receipt.

This command is CPU-only.  It never imports Habitat-Sim, recomputes a navmesh,
or modifies a geometry input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

try:
    from MemNavData.habitat_rollout_primitives import (
        FrozenGeometryError,
        FrozenGeometryIdentity,
        canonical_navmesh_settings,
    )
except ImportError:  # direct ``python MemNavData/<script>.py`` execution
    from habitat_rollout_primitives import (  # type: ignore
        FrozenGeometryError,
        FrozenGeometryIdentity,
        canonical_navmesh_settings,
    )


GEOMETRY_MAP_SCHEMA = "frozen_geometry_map_v1"
MAP_FILENAME = "frozen_geometry_map.json"
IDENTITY_DIRECTORY = "identities"
SETTINGS_SEMANTICS = (
    "evaluation/load contract only; not proof of historical navmesh bake settings"
)
SUPPORTED_MANIFEST_SCHEMAS = frozenset(
    {
        "nlsr_v2_expert_candidate_manifest_v1",
        "nlsr_v2_expert_candidate_manifest_v2",
        "nlsr_v2_multistage_expert_candidate_manifest_v1",
    }
)


class GeometryMapBuildError(RuntimeError):
    """An input or output violates the frozen-geometry build contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometryMapBuildError(message)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GeometryMapBuildError(
            f"value cannot be encoded as canonical JSON: {error}"
        ) from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def _no_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, label: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GeometryMapBuildError(
                    f"{label} contains non-finite JSON constant {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeometryMapBuildError(f"{label} is invalid JSON: {error}") from error


@dataclass(frozen=True)
class FileSnapshot:
    """One stable content hash plus the filesystem identity that produced it."""

    path: Path
    content_sha256: str
    byte_count: int
    stat_signature: tuple[int, int, int, int, int]


def snapshot_regular_file(path: Path | str, label: str) -> FileSnapshot:
    """Hash a non-symlink regular file without following a final symlink."""

    source = Path(path)
    try:
        before_path = source.lstat()
    except OSError as error:
        raise GeometryMapBuildError(f"{label} is missing: {source}") from error
    _require(not stat.S_ISLNK(before_path.st_mode), f"{label} is a symlink: {source}")
    _require(
        stat.S_ISREG(before_path.st_mode),
        f"{label} is not a regular file: {source}",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise GeometryMapBuildError(f"cannot open {label}: {source}") from error
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before_fd.st_mode),
            f"{label} changed to a non-regular file: {source}",
        )
        _require(
            (before_fd.st_dev, before_fd.st_ino)
            == (before_path.st_dev, before_path.st_ino),
            f"{label} changed while opening: {source}",
        )
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = source.lstat()
    except OSError as error:
        raise GeometryMapBuildError(
            f"{label} disappeared while hashing: {source}"
        ) from error
    before_signature = (
        int(before_fd.st_dev),
        int(before_fd.st_ino),
        int(before_fd.st_size),
        int(before_fd.st_mtime_ns),
        int(before_fd.st_ctime_ns),
    )
    after_signature = (
        int(after_fd.st_dev),
        int(after_fd.st_ino),
        int(after_fd.st_size),
        int(after_fd.st_mtime_ns),
        int(after_fd.st_ctime_ns),
    )
    path_signature = (
        int(after_path.st_dev),
        int(after_path.st_ino),
        int(after_path.st_size),
        int(after_path.st_mtime_ns),
        int(after_path.st_ctime_ns),
    )
    _require(
        before_signature == after_signature == path_signature,
        f"{label} changed while hashing: {source}",
    )
    return FileSnapshot(
        path=source,
        content_sha256=digest.hexdigest(),
        byte_count=int(after_fd.st_size),
        stat_signature=after_signature,
    )


def _read_stable_bytes(path: Path | str, label: str) -> tuple[bytes, FileSnapshot]:
    source = Path(path)
    snapshot = snapshot_regular_file(source, label)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise GeometryMapBuildError(f"cannot read {label}: {source}") from error
    after = snapshot_regular_file(source, label)
    _require(after == snapshot, f"{label} changed while reading: {source}")
    _require(
        len(raw) == snapshot.byte_count
        and sha256_bytes(raw) == snapshot.content_sha256,
        f"{label} bytes changed while reading: {source}",
    )
    return raw, snapshot


def _strict_directory(path: Path | str, label: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as error:
        raise GeometryMapBuildError(f"{label} is missing: {source}") from error
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} is a symlink: {source}")
    _require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory: {source}")
    return source.resolve(strict=True)


def parse_root_overrides(values: Sequence[str]) -> dict[str, Path]:
    """Parse explicit, named relocation roots; no positional guessing."""

    allowed = {"environment_root", "navmesh_root"}
    result: dict[str, Path] = {}
    for raw in values:
        _require("=" in raw, f"root override lacks '=': {raw!r}")
        name, path_text = raw.split("=", 1)
        _require(name in allowed, f"unsupported root override {name!r}")
        _require(name not in result, f"duplicate root override {name!r}")
        _require(bool(path_text), f"root override {name!r} has an empty path")
        result[name] = _strict_directory(Path(path_text), name)
    return result


def _resolve_roots(
    manifest: Mapping[str, Any],
    overrides: Mapping[str, Path | str] | None,
) -> dict[str, Path]:
    roots = manifest.get("input_roots")
    _require(isinstance(roots, Mapping), "manifest.input_roots must be an object")
    required = {"environment_root", "navmesh_root"}
    _require(required <= set(roots), "manifest is missing geometry input roots")
    supplied = {} if overrides is None else dict(overrides)
    _require(set(supplied) <= required, "root overrides contain an unsupported name")
    result = {}
    for name in sorted(required):
        value = supplied.get(name, roots[name])
        _require(
            isinstance(value, (str, os.PathLike)) and str(value),
            f"{name} must be a non-empty path",
        )
        result[name] = _strict_directory(Path(value), name)
    return result


def load_pinned_manifest(
    path: Path | str,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], FileSnapshot]:
    expected = _valid_sha256(expected_sha256, "expected manifest SHA256")
    raw, snapshot = _read_stable_bytes(path, "expert manifest")
    _require(snapshot.content_sha256 == expected, "expert manifest SHA256 mismatch")
    value = _decode_json(raw, "expert manifest")
    _require(isinstance(value, Mapping), "expert manifest must be an object")
    _require(
        value.get("schema_version") in SUPPORTED_MANIFEST_SCHEMAS,
        f"unsupported expert manifest schema {value.get('schema_version')!r}",
    )
    _require(raw == canonical_json_bytes(value), "expert manifest is not canonical")
    return value, snapshot


def load_pinned_settings(
    path: Path | str,
    expected_sha256: str,
) -> tuple[dict[str, Any], FileSnapshot]:
    expected = _valid_sha256(expected_sha256, "expected settings SHA256")
    raw, snapshot = _read_stable_bytes(path, "NavMeshSettings JSON")
    _require(snapshot.content_sha256 == expected, "NavMeshSettings SHA256 mismatch")
    value = _decode_json(raw, "NavMeshSettings JSON")
    try:
        canonical = canonical_navmesh_settings(value)
    except FrozenGeometryError as error:
        raise GeometryMapBuildError(f"invalid NavMeshSettings: {error}") from error
    _require(
        raw == canonical_json_bytes(canonical),
        "NavMeshSettings JSON must use canonical compact encoding",
    )
    return canonical, snapshot


def _canonical_relative_path(value: object, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and value, f"{label}.path is invalid")
    _require("\\" not in value and "\x00" not in value, f"{label}.path is not POSIX")
    relative = PurePosixPath(value)
    _require(not relative.is_absolute(), f"{label}.path must be relative")
    _require(
        relative.as_posix() == value
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label}.path is not canonical",
    )
    return relative


def _source_from_record(
    raw_record: object,
    root: Path,
    label: str,
) -> tuple[Path, FileSnapshot]:
    required = {"path", "path_sha256", "bytes", "content_sha256"}
    _require(
        isinstance(raw_record, Mapping) and set(raw_record) == required,
        f"{label} file record fields changed",
    )
    relative = _canonical_relative_path(raw_record["path"], label)
    _require(
        sha256_bytes(relative.as_posix().encode("utf-8"))
        == _valid_sha256(raw_record["path_sha256"], f"{label}.path_sha256"),
        f"{label} path hash mismatch",
    )
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise GeometryMapBuildError(f"{label} is missing: {current}") from error
        _require(
            not stat.S_ISLNK(metadata.st_mode), f"{label} uses a symlink: {current}"
        )
        if current != root / Path(*relative.parts):
            _require(
                stat.S_ISDIR(metadata.st_mode),
                f"{label} has a non-directory path component: {current}",
            )
    source = current
    snapshot = snapshot_regular_file(source, label)
    expected_bytes = raw_record["bytes"]
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes >= 0,
        f"{label}.bytes must be a non-negative integer",
    )
    _require(snapshot.byte_count == expected_bytes, f"{label} byte length changed")
    _require(
        snapshot.content_sha256
        == _valid_sha256(raw_record["content_sha256"], f"{label}.content_sha256"),
        f"{label} content changed",
    )
    return source.resolve(strict=True), snapshot


def _positive_float(value: object, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GeometryMapBuildError(f"{label} must be numeric") from error
    _require(
        math.isfinite(result) and result > 0.0, f"{label} must be finite and positive"
    )
    return result


def _identity_filename(scene_id: str) -> str:
    return f"{sha256_bytes(scene_id.encode('utf-8'))}.json"


@dataclass(frozen=True)
class ExpectedBundle:
    files: Mapping[str, bytes]
    map_sha256: str
    source_snapshots: Mapping[Path, FileSnapshot]


def build_expected_bundle(
    *,
    manifest: Mapping[str, Any],
    roots: Mapping[str, Path],
    habitat_sim_version: str,
    agent_radius_m: float,
    agent_height_m: float,
    navmesh_settings: Mapping[str, Any],
) -> ExpectedBundle:
    version = str(habitat_sim_version)
    _require(version and version.strip() == version, "habitat_sim_version is invalid")
    radius = _positive_float(agent_radius_m, "agent_radius_m")
    height = _positive_float(agent_height_m, "agent_height_m")
    _require(
        math.isclose(
            float(navmesh_settings["agent_radius"]),
            radius,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "agent_radius_m disagrees with NavMeshSettings.agent_radius",
    )
    _require(
        math.isclose(
            float(navmesh_settings["agent_height"]),
            height,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "agent_height_m disagrees with NavMeshSettings.agent_height",
    )
    scenes = manifest.get("scenes")
    _require(isinstance(scenes, list) and scenes, "manifest.scenes must be non-empty")
    summary = manifest.get("summary")
    if isinstance(summary, Mapping) and "scene_count" in summary:
        _require(
            isinstance(summary["scene_count"], int)
            and not isinstance(summary["scene_count"], bool)
            and summary["scene_count"] == len(scenes),
            "manifest summary scene_count disagrees with scenes",
        )

    map_scenes: dict[str, dict[str, str]] = {}
    identity_files: dict[str, bytes] = {}
    snapshots: dict[Path, FileSnapshot] = {}
    seen_paths: dict[Path, str] = {}
    for index, raw_scene in enumerate(scenes):
        label = f"manifest.scenes[{index}]"
        _require(isinstance(raw_scene, Mapping), f"{label} must be an object")
        scene_id = raw_scene.get("scene")
        _require(isinstance(scene_id, str) and scene_id, f"{label}.scene is invalid")
        _require(scene_id not in map_scenes, f"duplicate scene id {scene_id!r}")
        geometry = []
        for record_name, root_name in (
            ("environment", "environment_root"),
            ("navmesh", "navmesh_root"),
        ):
            source, snapshot = _source_from_record(
                raw_scene.get(record_name),
                roots[root_name],
                f"{scene_id}.{record_name}",
            )
            _require(
                source not in seen_paths,
                f"duplicate geometry path for {scene_id}.{record_name}; already used by "
                f"{seen_paths.get(source)}: {source}",
            )
            seen_paths[source] = f"{scene_id}.{record_name}"
            snapshots[source] = snapshot
            geometry.append((source, snapshot))
        glb_path, glb_snapshot = geometry[0]
        navmesh_path, navmesh_snapshot = geometry[1]
        try:
            identity = FrozenGeometryIdentity.capture(
                glb_path=glb_path,
                navmesh_path=navmesh_path,
                habitat_sim_version=version,
                agent_radius_m=radius,
                agent_height_m=height,
                navmesh_settings=navmesh_settings,
            )
        except (FrozenGeometryError, OSError) as error:
            raise GeometryMapBuildError(
                f"cannot capture frozen geometry for {scene_id}: {error}"
            ) from error
        _require(
            identity.glb_sha256 == glb_snapshot.content_sha256
            and identity.glb_bytes == glb_snapshot.byte_count,
            f"{scene_id} GLB drifted during identity capture",
        )
        _require(
            identity.navmesh_sha256 == navmesh_snapshot.content_sha256
            and identity.navmesh_bytes == navmesh_snapshot.byte_count,
            f"{scene_id} navmesh drifted during identity capture",
        )
        relative_identity = f"{IDENTITY_DIRECTORY}/{_identity_filename(scene_id)}"
        _require(
            relative_identity not in identity_files,
            f"identity output path collision for scene {scene_id!r}",
        )
        identity_bytes = identity.canonical_json_bytes()
        _require(
            sha256_bytes(identity_bytes) == identity.identity_sha256,
            f"identity self-hash failed for {scene_id}",
        )
        identity_files[relative_identity] = identity_bytes
        map_scenes[scene_id] = {
            "identity_path": relative_identity,
            "identity_sha256": identity.identity_sha256,
        }

    geometry_map = {
        "schema_version": GEOMETRY_MAP_SCHEMA,
        "scenes": map_scenes,
    }
    map_bytes = canonical_json_bytes(geometry_map)
    map_sha = sha256_bytes(map_bytes)
    files = dict(identity_files)
    files[MAP_FILENAME] = map_bytes
    files[f"{MAP_FILENAME}.sha256"] = f"{map_sha}  {MAP_FILENAME}\n".encode("ascii")
    return ExpectedBundle(files, map_sha, snapshots)


def _expected_directories(files: Mapping[str, bytes]) -> set[str]:
    result = {"."}
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def _verify_bundle_tree(root: Path, files: Mapping[str, bytes], label: str) -> None:
    root = _strict_directory(root, label)
    actual_files: set[str] = set()
    actual_directories = {"."}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root).as_posix()
        for directory in list(directories):
            child = current_path / directory
            metadata = child.lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                f"{label} contains a non-directory or symlink: {child}",
            )
            relative = child.relative_to(root).as_posix()
            actual_directories.add(relative)
        for filename in filenames:
            child = current_path / filename
            relative = child.relative_to(root).as_posix()
            snapshot = snapshot_regular_file(child, f"{label}/{relative}")
            actual_files.add(relative)
            expected = files.get(relative)
            _require(
                expected is not None, f"{label} contains unexpected file {relative}"
            )
            _require(
                snapshot.byte_count == len(expected)
                and snapshot.content_sha256 == sha256_bytes(expected),
                f"{label} file differs: {relative}",
            )
        if relative_current != ".":
            actual_directories.add(relative_current)
    _require(actual_files == set(files), f"{label} file set is incomplete")
    _require(
        actual_directories == _expected_directories(files),
        f"{label} directory set differs",
    )


def _write_file_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging(path: Path | None) -> None:
    if path is not None and os.path.lexists(path):
        shutil.rmtree(path)


def publish_bundle(
    expected: ExpectedBundle,
    output_directory: Path | str,
    *,
    resume: bool = False,
) -> str:
    output = Path(output_directory).absolute()
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    _strict_directory(parent, "output parent")
    _require(output.name not in ("", ".", ".."), "output directory name is invalid")
    exists = os.path.lexists(output)
    if resume:
        _require(exists, "--resume requires an existing complete bundle")
    else:
        _require(not exists, f"output already exists: {output}")
    if exists:
        _require(not output.is_symlink(), f"output is a symlink: {output}")
        _require(output.is_dir(), f"output is not a directory: {output}")

    lock = parent / f".{output.name}.lock"
    try:
        lock_descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError as error:
        raise GeometryMapBuildError(f"another builder owns lock {lock}") from error
    os.close(lock_descriptor)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=parent,
            )
        )
        for relative, payload in sorted(expected.files.items()):
            _write_file_fsync(staging / Path(relative), payload)
        for directory in sorted(
            (
                staging / value
                for value in _expected_directories(expected.files)
                if value != "."
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        _verify_bundle_tree(staging, expected.files, "staged bundle")

        for source, before in expected.source_snapshots.items():
            after = snapshot_regular_file(source, f"geometry recheck {source}")
            _require(after == before, f"geometry source drifted during build: {source}")

        if resume:
            _verify_bundle_tree(output, expected.files, "resume bundle")
            return "resumed"
        _require(not os.path.lexists(output), f"output appeared during build: {output}")
        os.rename(staging, output)
        staging = None
        _fsync_directory(parent)
        _verify_bundle_tree(output, expected.files, "published bundle")
        return "written"
    finally:
        _remove_staging(staging)
        lock.unlink(missing_ok=True)


def build_geometry_map_bundle(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    navmesh_settings_path: Path | str,
    expected_navmesh_settings_sha256: str,
    habitat_sim_version: str,
    agent_radius_m: float,
    agent_height_m: float,
    output_directory: Path | str,
    root_overrides: Mapping[str, Path | str] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    manifest, manifest_snapshot = load_pinned_manifest(
        manifest_path, expected_manifest_sha256
    )
    settings, settings_snapshot = load_pinned_settings(
        navmesh_settings_path, expected_navmesh_settings_sha256
    )
    roots = _resolve_roots(manifest, root_overrides)
    expected = build_expected_bundle(
        manifest=manifest,
        roots=roots,
        habitat_sim_version=habitat_sim_version,
        agent_radius_m=agent_radius_m,
        agent_height_m=agent_height_m,
        navmesh_settings=settings,
    )
    all_snapshots = dict(expected.source_snapshots)
    all_snapshots[manifest_snapshot.path] = manifest_snapshot
    all_snapshots[settings_snapshot.path] = settings_snapshot
    expected = ExpectedBundle(
        files=expected.files,
        map_sha256=expected.map_sha256,
        source_snapshots=all_snapshots,
    )
    status = publish_bundle(expected, output_directory, resume=resume)
    return {
        "status": status,
        "output_directory": str(Path(output_directory).absolute()),
        "geometry_map": str(Path(output_directory).absolute() / MAP_FILENAME),
        "geometry_map_sha256": expected.map_sha256,
        "scene_count": len(manifest["scenes"]),
        "manifest_sha256": manifest_snapshot.content_sha256,
        "navmesh_settings_sha256": settings_snapshot.content_sha256,
        "navmesh_settings_semantics": SETTINGS_SEMANTICS,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen_geometry_map_v1 bundle without Habitat/GPU use.",
        epilog=(
            "The supplied NavMeshSettings are the evaluation/load contract; "
            "they are not proof of the historical bake settings."
        ),
    )
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--navmesh-settings-json", type=Path, required=True)
    parser.add_argument("--expected-navmesh-settings-sha256", required=True)
    parser.add_argument("--habitat-sim-version", required=True)
    parser.add_argument("--agent-radius-m", type=float, required=True)
    parser.add_argument("--agent-height-m", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--root-override",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=("explicitly relocate environment_root or navmesh_root; repeatable"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_geometry_map_bundle(
        manifest_path=args.expert_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        navmesh_settings_path=args.navmesh_settings_json,
        expected_navmesh_settings_sha256=(args.expected_navmesh_settings_sha256),
        habitat_sim_version=args.habitat_sim_version,
        agent_radius_m=args.agent_radius_m,
        agent_height_m=args.agent_height_m,
        output_directory=args.output_directory,
        root_overrides=parse_root_overrides(args.root_override),
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
