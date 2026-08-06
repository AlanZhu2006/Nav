#!/usr/bin/env python3
"""Strict, zero-copy routing for audited LingBot flow-cache pairs.

The NLSR data set may reuse immutable cache pairs owned by another account and
add a small number of locally generated patches.  Copying those multi-GiB
files is wasteful, while a symlink farm makes a nominal single-root manifest
silently escape its declared root.  This module keeps the physical roots
explicit: a small, canonical route artifact maps every episode to exactly one
audited source root and relative chunk.

The causal manifest pins the route artifact by SHA256.  Every consumer reloads
that pin, checks path containment and the frozen file record, and then opens the
physical file directly.  No broad symlink exception, bind-mount recipe, or
payload copy is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping


ROUTE_SCHEMA_VERSION = "nlsr_flow_route_provenance_v1"
ROUTE_STATUS = "flow_routes_audited"
MANIFEST_ROUTING_SCHEMA_VERSION = "nlsr_manifest_flow_routing_v1"
MANIFEST_ROUTING_MODE = "provenance_pinned_multi_root"
FLOW_FILE_NAMES = ("lingbot_cache.npz", "lingbot_cam_cache.npz")


class FlowRoutingError(RuntimeError):
    """A route artifact or a routed cache file violated its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlowRoutingError(message)


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative_posix(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is missing")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == value,
        f"{label} is not a canonical root-relative POSIX path: {value!r}",
    )
    return value


def _episode_key(value: object) -> str:
    key = _relative_posix(value, "route episode")
    parts = PurePosixPath(key).parts
    _require(
        len(parts) == 2
        and bool(parts[0])
        and parts[1].startswith("episode_")
        and "/".join(parts) == key,
        f"route episode must be <scene>/episode_*: {key!r}",
    )
    return key


def _rooted_physical_file(root: Path, relative: str, label: str) -> Path:
    root_resolved = root.resolve()
    _require(root_resolved.is_dir(), f"{label} source root is unavailable: {root}")
    candidate = root / Path(relative)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise FlowRoutingError(
            f"{label} escapes or is absent below its declared source root: "
            f"{candidate}") from error
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} must be a physical regular file: {candidate}",
    )
    return resolved


@dataclass(frozen=True)
class RoutedFile:
    episode: str
    source_id: str
    logical_path: str
    source_relative_path: str
    bytes: int
    content_sha256: str | None
    path: Path

    def manifest_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "path": self.logical_path,
            "path_sha256": sha256_bytes(self.logical_path.encode("utf-8")),
            "bytes": self.bytes,
            "routing": {
                "source_id": self.source_id,
                "source_relative_path": self.source_relative_path,
                "source_relative_path_sha256": sha256_bytes(
                    self.source_relative_path.encode("utf-8")),
            },
        }
        if self.content_sha256 is not None:
            record["content_sha256"] = self.content_sha256
        return record


@dataclass(frozen=True)
class FlowRouteRegistry:
    artifact_path: Path
    artifact_sha256: str
    split_sha256: str
    source_roots: Mapping[str, Path]
    files_by_episode: Mapping[str, Mapping[str, RoutedFile]]
    raw_record: Mapping[str, object]

    def manifest_record(self) -> dict[str, object]:
        return {
            "schema_version": MANIFEST_ROUTING_SCHEMA_VERSION,
            "mode": MANIFEST_ROUTING_MODE,
            "route_schema_version": ROUTE_SCHEMA_VERSION,
            "route_status": ROUTE_STATUS,
            "artifact_path": str(self.artifact_path.resolve()),
            "artifact_sha256": self.artifact_sha256,
            "split_sha256": self.split_sha256,
            "source_roots": {
                source_id: str(root.resolve())
                for source_id, root in sorted(self.source_roots.items())
            },
            "episode_count": len(self.files_by_episode),
        }

    def episode_file_records(self, scene: str, episode: str) -> list[dict[str, object]]:
        key = _episode_key(f"{scene}/{episode}")
        rows = self.files_by_episode.get(key)
        _require(rows is not None, f"route artifact lacks episode {key}")
        _require(
            set(rows) == set(FLOW_FILE_NAMES),
            f"route episode does not contain the exact cache pair: {key}",
        )
        return [rows[name].manifest_record() for name in FLOW_FILE_NAMES]

    def resolve_manifest_file(
        self,
        episode_record: Mapping[str, object],
        scene: str,
        episode: str,
        file_name: str,
    ) -> Path:
        _require(file_name in FLOW_FILE_NAMES, f"unsupported flow file: {file_name}")
        key = _episode_key(f"{scene}/{episode}")
        routed = self.files_by_episode.get(key, {}).get(file_name)
        _require(routed is not None, f"route artifact lacks {key}/{file_name}")
        flow = episode_record.get("flow_cache")
        _require(isinstance(flow, Mapping), "manifest flow-cache record is missing")
        files = flow.get("files")
        _require(isinstance(files, list), "manifest flow-cache files are missing")
        matches = [
            record for record in files
            if isinstance(record, Mapping)
            and record.get("path") == routed.logical_path
        ]
        _require(
            len(matches) == 1,
            f"manifest does not uniquely declare routed file {routed.logical_path}",
        )
        expected = routed.manifest_record()
        _require(
            dict(matches[0]) == expected,
            f"manifest routed-file binding differs from its pinned route: "
            f"{routed.logical_path}",
        )
        path = _rooted_physical_file(
            self.source_roots[routed.source_id],
            routed.source_relative_path,
            "routed flow cache",
        )
        _require(
            path == routed.path
            and path.stat().st_size == routed.bytes,
            f"routed flow cache changed after provenance freeze: {path}",
        )
        if routed.content_sha256 is not None:
            _require(
                sha256_file(path) == routed.content_sha256,
                f"routed flow cache content changed: {path}",
            )
        return path

    def resolve_manifest_pair(
        self,
        episode_record: Mapping[str, object],
        scene: str,
        episode: str,
    ) -> tuple[Path, Path]:
        """Atomically authorize the aggregator/camera pair from one route row."""

        key = _episode_key(f"{scene}/{episode}")
        rows = self.files_by_episode.get(key)
        _require(rows is not None and set(rows) == set(FLOW_FILE_NAMES),
                 f"route artifact lacks an exact cache pair for {key}")
        _require(
            len({row.source_id for row in rows.values()}) == 1,
            f"route cache pair crosses physical sources: {key}",
        )
        before = {
            name: (row.path.stat().st_size, row.path.stat().st_mtime_ns)
            for name, row in rows.items()
        }
        aggregator = self.resolve_manifest_file(
            episode_record, scene, episode, FLOW_FILE_NAMES[0])
        camera = self.resolve_manifest_file(
            episode_record, scene, episode, FLOW_FILE_NAMES[1])
        after = {
            name: (row.path.stat().st_size, row.path.stat().st_mtime_ns)
            for name, row in rows.items()
        }
        _require(before == after, f"routed cache pair changed during resolution: {key}")
        return aggregator, camera


def _load_canonical_artifact(path: Path, expected_sha256: str) -> tuple[dict, str]:
    _require(_valid_sha(expected_sha256), "expected route artifact SHA256 is invalid")
    sidecar = Path(f"{path}.sha256")
    _require(
        path.is_file()
        and sidecar.is_file()
        and not path.is_symlink()
        and not sidecar.is_symlink(),
        "route artifact pair is absent or not physical",
    )
    payload = path.read_bytes()
    actual_sha = sha256_bytes(payload)
    _require(
        actual_sha == expected_sha256,
        f"route artifact SHA256 mismatch: {actual_sha} != {expected_sha256}",
    )
    expected_sidecar = f"{actual_sha}  {path.name}\n".encode("ascii")
    _require(
        sidecar.read_bytes() == expected_sidecar,
        "route artifact SHA sidecar is non-canonical or mismatched",
    )
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as error:
        raise FlowRoutingError("route artifact is invalid JSON") from error
    _require(isinstance(record, dict), "route artifact must be a JSON object")
    _require(
        canonical_json_bytes(record) == payload,
        "route artifact JSON is not canonical",
    )
    return record, actual_sha


def load_route_registry(path: Path, expected_sha256: str) -> FlowRouteRegistry:
    """Load and fully validate a provenance-pinned multi-root route artifact."""

    record, actual_sha = _load_canonical_artifact(path, expected_sha256)
    required_top = {
        "schema_version",
        "status",
        "split_sha256",
        "raw_audit_sha256",
        "route_root",
        "source_roots",
        "official_snapshot_semantics",
        "official_snapshot_sha256",
        "patch_payloads_fully_sha256",
        "counts",
        "pairs",
    }
    _require(set(record) == required_top, "route artifact top-level keys differ")
    _require(record["schema_version"] == ROUTE_SCHEMA_VERSION,
             "route artifact schema version differs")
    _require(record["status"] == ROUTE_STATUS, "route artifact status is not audited")
    split_sha = record["split_sha256"]
    _require(_valid_sha(split_sha), "route artifact split SHA256 is invalid")
    _require(_valid_sha(record["raw_audit_sha256"]),
             "route artifact raw-audit SHA256 is invalid")
    _require(_valid_sha(record["official_snapshot_sha256"]),
             "route official snapshot SHA256 is invalid")
    _require(record["patch_payloads_fully_sha256"] is True,
             "patch payloads are not fully content-pinned")
    route_root = Path(str(record["route_root"])).resolve()
    _require(route_root == path.parent.resolve(),
             "route artifact names a different route root")

    source_root_record = record["source_roots"]
    _require(isinstance(source_root_record, dict) and bool(source_root_record),
             "route source_roots is malformed")
    source_roots: dict[str, Path] = {}
    for source_id, root_value in sorted(source_root_record.items()):
        _require(
            isinstance(source_id, str)
            and bool(source_id)
            and source_id not in source_roots,
            "route source id is malformed or duplicated",
        )
        root = Path(str(root_value))
        _require(root.is_absolute(), f"route source root is not absolute: {root}")
        resolved = root.resolve()
        _require(resolved.is_dir(), f"route source root is unavailable: {root}")
        source_roots[source_id] = resolved

    pairs = record["pairs"]
    counts = record["counts"]
    _require(isinstance(pairs, list) and isinstance(counts, dict),
             "route pairs/counts are malformed")
    files_by_episode: dict[str, dict[str, RoutedFile]] = {}
    source_counts = {source_id: 0 for source_id in source_roots}
    for row in pairs:
        _require(isinstance(row, dict), "route pair must be an object")
        _require(set(row) == {
            "episode", "source_id", "source_relative_chunk", "validation",
        }, "route pair keys differ")
        episode = _episode_key(row["episode"])
        _require(episode not in files_by_episode, f"duplicate route episode: {episode}")
        source_id = row["source_id"]
        _require(source_id in source_roots, f"unknown route source id: {source_id}")
        relative_chunk = _relative_posix(
            row["source_relative_chunk"], "route source chunk")
        expected_chunk = f"{episode}/videos/chunk-000"
        _require(
            relative_chunk == expected_chunk,
            f"route source chunk differs from episode layout: {relative_chunk}",
        )
        validation = row["validation"]
        _require(isinstance(validation, dict), "route validation is malformed")
        file_rows = validation.get("files")
        _require(isinstance(file_rows, list) and len(file_rows) == 2,
                 "route validation must contain exactly one cache pair")
        files: dict[str, RoutedFile] = {}
        for file_record in file_rows:
            _require(isinstance(file_record, dict), "route file record is malformed")
            _require(set(file_record) == {"name", "bytes", "content_sha256"},
                     "route file record keys differ")
            name = file_record["name"]
            _require(name in FLOW_FILE_NAMES and name not in files,
                     f"route cache file is unsupported or duplicated: {name}")
            size = file_record["bytes"]
            _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                     "route cache byte size is malformed")
            content_sha = file_record["content_sha256"]
            _require(content_sha is None or _valid_sha(content_sha),
                     "route cache content SHA256 is malformed")
            if source_id == "flow4096_patch":
                _require(_valid_sha(content_sha),
                         "patch route must content-pin every payload")
            source_relative = f"{relative_chunk}/{name}"
            physical = _rooted_physical_file(
                source_roots[source_id], source_relative, "route source cache")
            _require(physical.stat().st_size == size,
                     f"route cache size differs: {physical}")
            if content_sha is not None:
                _require(sha256_file(physical) == content_sha,
                         f"route cache content SHA256 differs: {physical}")
            logical = f"{episode}/videos/chunk-000/{name}"
            files[name] = RoutedFile(
                episode=episode,
                source_id=source_id,
                logical_path=logical,
                source_relative_path=source_relative,
                bytes=size,
                content_sha256=content_sha,
                path=physical,
            )
        _require(set(files) == set(FLOW_FILE_NAMES),
                 f"route episode lacks an exact cache pair: {episode}")
        files_by_episode[episode] = files
        source_counts[source_id] += 1

    scene_count = len({episode.split("/", 1)[0] for episode in files_by_episode})
    expected_counts = {
        "scenes": scene_count,
        "pairs": len(files_by_episode),
        **source_counts,
    }
    _require(counts == expected_counts,
             f"route counts differ from rows: {counts} != {expected_counts}")
    return FlowRouteRegistry(
        artifact_path=path.resolve(),
        artifact_sha256=actual_sha,
        split_sha256=str(split_sha),
        source_roots=source_roots,
        files_by_episode=files_by_episode,
        raw_record=record,
    )


def registry_from_manifest(manifest: Mapping[str, object]) -> FlowRouteRegistry | None:
    """Return the pinned multi-root registry, or ``None`` for legacy manifests."""

    routing = manifest.get("flow_cache_routing")
    if routing is None:
        return None
    _require(isinstance(routing, dict), "manifest flow_cache_routing is malformed")
    required = {
        "schema_version",
        "mode",
        "route_schema_version",
        "route_status",
        "artifact_path",
        "artifact_sha256",
        "split_sha256",
        "source_roots",
        "episode_count",
    }
    _require(set(routing) == required, "manifest routing keys differ")
    _require(routing["schema_version"] == MANIFEST_ROUTING_SCHEMA_VERSION,
             "manifest routing schema version differs")
    _require(routing["mode"] == MANIFEST_ROUTING_MODE,
             "manifest routing mode differs")
    _require(routing["route_schema_version"] == ROUTE_SCHEMA_VERSION,
             "manifest route schema pin differs")
    _require(routing["route_status"] == ROUTE_STATUS,
             "manifest route status pin differs")
    artifact_path = Path(str(routing["artifact_path"]))
    _require(artifact_path.is_absolute(), "manifest route artifact path is not absolute")
    registry = load_route_registry(
        artifact_path, str(routing["artifact_sha256"]))
    _require(registry.manifest_record() == routing,
             "manifest routing record differs from its pinned artifact")
    split = manifest.get("split")
    _require(
        isinstance(split, Mapping)
        and split.get("sha256") == registry.split_sha256,
        "manifest split differs from the routed-flow split",
    )
    return registry
