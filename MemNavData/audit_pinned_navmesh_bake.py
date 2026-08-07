#!/usr/bin/env python3
"""Cross-audit a receipt-backed NavMesh bake and publish its launcher receipt.

The NavMesh baker validates every scene while producing the bundle.  This
module is the independent, reusable post-audit boundary: it reopens the
published bundle, verifies all content-addressed links between the derived
manifest, bake index, run contract, scene NavMeshes, and scene receipts, then
atomically publishes an idempotent launcher receipt that pins the outer HPC
runtime and clean producer checkout.

The auditor does not invoke Habitat and never recomputes a NavMesh.  It only
accepts exact canonical artifacts and physical files already produced by the
bake stage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

try:
    from MemNavData.build_frozen_geometry_map import (
        GeometryMapBuildError,
        _source_from_record,
        canonical_json_bytes,
        sha256_bytes,
        snapshot_regular_file,
    )
except ImportError:  # direct ``python MemNavData/<script>.py`` execution
    from build_frozen_geometry_map import (  # type: ignore
        GeometryMapBuildError,
        _source_from_record,
        canonical_json_bytes,
        sha256_bytes,
        snapshot_regular_file,
    )


DERIVED_MANIFEST_FILE = "derived_geometry_manifest.json"
BAKE_INDEX_FILE = "navmesh_bake_index.json"
RUN_CONTRACT_FILE = "run_contract.json"
LAUNCHER_RECEIPT_FILE = "navmesh_bake_launcher_receipt.json"
DERIVATION_SCHEMA = "nlsr_derived_geometry_manifest_v1"
INDEX_SCHEMA = "nlsr_navmesh_bake_index_v1"
RUN_CONTRACT_SCHEMA = "nlsr_navmesh_bake_run_contract_v1"
SCENE_RECEIPT_SCHEMA = "nlsr_navmesh_bake_receipt_v1"
LAUNCHER_RECEIPT_SCHEMA = "nlsr_navmesh_bake_launcher_receipt_v1"
VERIFIED_STATUS = "fresh_double_bake_roundtrip_verified"
LAUNCHER_STATUS = "cross_audited"
ALLOWED_SPLIT_ROLES = frozenset(("train", "development"))
SELECTION_BOUNDARY = (
    "scene membership and GLB bytes come only from the pinned expert manifest; "
    "no episode sample, goal, geodesic, success, evaluation, or final data label"
)
DETERMINISM_BOUNDARY = (
    "byte equality is verified for two fresh-simulator bakes in this exact "
    "runtime; no cross-platform or cross-build determinism is claimed"
)
SCENE_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "scene_id",
    "split_role",
    "parent_manifest_sha256",
    "source_glb",
    "superseded_manifest_navmesh_record",
    "requested_settings",
    "requested_settings_sha256",
    "runtime_effective_settings",
    "runtime_effective_settings_sha256",
    "runtime",
    "bake_method",
    "determinism",
    "bake_observation",
    "roundtrip_observation",
    "output_navmesh",
    "selection_boundary",
    "determinism_boundary",
}
BAKE_METHOD = {
    "cpu_only": True,
    "create_renderer": False,
    "fresh_simulator_each_repetition": True,
    "recompute_navmesh": True,
    "save_nav_mesh": True,
    "roundtrip_load_nav_mesh": True,
}


class NavmeshBakeAuditError(RuntimeError):
    """A published bake or its launcher contract failed cross-audit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NavmeshBakeAuditError(message)


def _valid_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def _valid_commit(value: object) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        "producer commit must be a full lowercase Git SHA",
    )
    return value


def _snapshot(path: Path | str, label: str):
    try:
        return snapshot_regular_file(path, label)
    except GeometryMapBuildError as error:
        raise NavmeshBakeAuditError(str(error)) from error


def _physical_directory(path: Path | str, label: str) -> Path:
    source = Path(path)
    _require(
        source.is_dir() and not source.is_symlink(),
        f"{label} is not a physical directory: {source}",
    )
    return source.resolve(strict=True)


def _load_canonical(path: Path | str, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    snapshot = _snapshot(source, label)
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NavmeshBakeAuditError(f"cannot read {label}: {source}") from error
    _require(
        snapshot == _snapshot(source, label)
        and len(raw) == snapshot.byte_count
        and sha256_bytes(raw) == snapshot.content_sha256,
        f"{label} drifted while reading: {source}",
    )
    _require(isinstance(value, dict), f"{label} is not an object")
    try:
        canonical = canonical_json_bytes(value)
    except GeometryMapBuildError as error:
        raise NavmeshBakeAuditError(f"{label} cannot be canonicalized") from error
    _require(raw == canonical, f"noncanonical {label}: {source}")
    return value, raw


def _verify_sidecar(path: Path, raw: bytes) -> str:
    digest = sha256_bytes(raw)
    sidecar = Path(f"{path}.sha256")
    snapshot = _snapshot(sidecar, f"{path.name} sidecar")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    _require(
        snapshot.byte_count == len(expected)
        and snapshot.content_sha256 == sha256_bytes(expected)
        and sidecar.read_bytes() == expected,
        f"exact sidecar mismatch: {path}",
    )
    return digest


def _stable_record(path: Path, logical_path: str) -> dict[str, object]:
    snapshot = _snapshot(path, logical_path)
    return {
        "path": logical_path,
        "path_sha256": sha256_bytes(logical_path.encode("utf-8")),
        "bytes": snapshot.byte_count,
        "content_sha256": snapshot.content_sha256,
    }


def _resolve_record(
    record: object,
    root: Path,
    label: str,
) -> tuple[Path, dict[str, object]]:
    try:
        path, snapshot = _source_from_record(record, root, label)
    except GeometryMapBuildError as error:
        raise NavmeshBakeAuditError(str(error)) from error
    _require(isinstance(record, Mapping), f"{label} record is invalid")
    canonical = {
        "path": record["path"],
        "path_sha256": record["path_sha256"],
        "bytes": snapshot.byte_count,
        "content_sha256": snapshot.content_sha256,
    }
    _require(dict(record) == canonical, f"{label} file record changed")
    return path, canonical


def _file_sha_following_final_symlink(path: Path, label: str) -> tuple[int, str]:
    _require(path.is_file(), f"{label} is missing: {path}")
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise NavmeshBakeAuditError(f"cannot hash {label}: {path}") from error
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(signature_before == signature_after, f"{label} drifted while hashing")
    return int(after.st_size), digest.hexdigest()


def _validate_runtime(
    runtime: object,
    *,
    expected_habitat_version: str,
    expected_bindings_sha256: str,
) -> dict[str, Any]:
    required = {
        "habitat_sim_version",
        "python_version",
        "runtime_files",
        "runtime_fingerprint_sha256",
    }
    _require(
        isinstance(runtime, dict) and set(runtime) == required,
        "runtime identity fields changed",
    )
    _require(
        runtime["habitat_sim_version"] == expected_habitat_version,
        "derived Habitat version changed",
    )
    _require(
        isinstance(runtime["python_version"], str) and runtime["python_version"],
        "runtime Python version is invalid",
    )
    runtime_files = runtime["runtime_files"]
    _require(
        isinstance(runtime_files, dict)
        and set(runtime_files) == {"habitat_sim_init", "habitat_sim_bindings"},
        "runtime file records changed",
    )
    for name, record in runtime_files.items():
        _require(
            isinstance(record, dict)
            and set(record) == {"name", "bytes", "content_sha256"},
            f"runtime {name} record changed",
        )
        _require(
            isinstance(record["name"], str)
            and record["name"]
            and "/" not in record["name"]
            and "\\" not in record["name"],
            f"runtime {name} filename is invalid",
        )
        _require(
            type(record["bytes"]) is int and record["bytes"] > 0,
            f"runtime {name} byte count is invalid",
        )
        _valid_sha256(record["content_sha256"], f"runtime {name} SHA256")
    _require(
        runtime_files["habitat_sim_bindings"]["content_sha256"]
        == expected_bindings_sha256,
        "derived manifest runtime binding changed",
    )
    fingerprint_input = {
        "habitat_sim_version": runtime["habitat_sim_version"],
        "python_version": runtime["python_version"],
        "runtime_files": runtime_files,
    }
    _require(
        runtime["runtime_fingerprint_sha256"]
        == sha256_bytes(canonical_json_bytes(fingerprint_input)),
        "runtime fingerprint changed",
    )
    return runtime


def _validate_observation(value: object, label: str) -> dict[str, Any]:
    fields = {
        "navigable_area_m2",
        "bounds_min_xyz",
        "bounds_max_xyz",
        "vertex_count",
        "index_count",
    }
    _require(isinstance(value, dict) and set(value) == fields, f"{label} changed")
    area = value["navigable_area_m2"]
    _require(
        not isinstance(area, bool)
        and isinstance(area, (int, float))
        and math.isfinite(float(area))
        and float(area) > 0.0,
        f"{label} area is invalid",
    )
    minimum = value["bounds_min_xyz"]
    maximum = value["bounds_max_xyz"]
    for name, bounds in (("minimum", minimum), ("maximum", maximum)):
        _require(
            isinstance(bounds, list)
            and len(bounds) == 3
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in bounds
            ),
            f"{label} {name} bounds are invalid",
        )
    _require(
        all(float(lower) <= float(upper) for lower, upper in zip(minimum, maximum)),
        f"{label} bounds are inverted",
    )
    for name in ("vertex_count", "index_count"):
        _require(
            type(value[name]) is int and value[name] > 0,
            f"{label} {name} is invalid",
        )
    return value


def _observations_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        math.isclose(
            float(first["navigable_area_m2"]),
            float(second["navigable_area_m2"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and all(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(first["bounds_min_xyz"], second["bounds_min_xyz"])
        )
        and all(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(first["bounds_max_xyz"], second["bounds_max_xyz"])
        )
        and first["vertex_count"] == second["vertex_count"]
        and first["index_count"] == second["index_count"]
    )


def _write_once_or_verify(path: Path, payload: bytes, label: str) -> None:
    if os.path.lexists(path):
        snapshot = _snapshot(path, label)
        _require(
            snapshot.byte_count == len(payload)
            and snapshot.content_sha256 == sha256_bytes(payload)
            and path.read_bytes() == payload,
            f"existing {label} differs: {path}",
        )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _require(not os.path.lexists(path), f"{label} appeared concurrently")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class AuditContract:
    output_root: Path
    run_root: Path
    expected_parent_manifest_sha256: str
    expected_settings_sha256: str
    expected_bindings_sha256: str
    expected_producer_sha256: str
    expected_auditor_sha256: str
    expected_habitat_version: str
    expected_producer_commit: str
    expected_launcher_sha256: str
    producer_path: Path
    auditor_path: Path
    launcher_path: Path
    habitat_python_path: Path
    expected_habitat_python_sha256: str
    base_sif_path: Path
    expected_base_sif_sha256: str
    expected_base_sif_bytes: int
    expected_scene_count: int
    expected_fresh_simulator_repetitions: int

    def validate(self) -> "AuditContract":
        for value, label in (
            (self.expected_parent_manifest_sha256, "parent manifest SHA256"),
            (self.expected_settings_sha256, "settings SHA256"),
            (self.expected_bindings_sha256, "Habitat bindings SHA256"),
            (self.expected_producer_sha256, "producer SHA256"),
            (self.expected_auditor_sha256, "auditor SHA256"),
            (self.expected_launcher_sha256, "launcher SHA256"),
            (self.expected_habitat_python_sha256, "Habitat Python SHA256"),
            (self.expected_base_sif_sha256, "base SIF SHA256"),
        ):
            _valid_sha256(value, label)
        _valid_commit(self.expected_producer_commit)
        _require(
            self.expected_habitat_version == "0.3.1",
            "formal audit requires Habitat-Sim 0.3.1",
        )
        _require(
            type(self.expected_base_sif_bytes) is int
            and self.expected_base_sif_bytes > 0,
            "expected base SIF byte count is invalid",
        )
        _require(
            type(self.expected_scene_count) is int and self.expected_scene_count > 0,
            "expected scene count is invalid",
        )
        _require(
            type(self.expected_fresh_simulator_repetitions) is int
            and self.expected_fresh_simulator_repetitions == 2,
            "formal audit requires exactly two fresh simulator bakes",
        )
        return self


def audit_navmesh_bake(contract: AuditContract) -> dict[str, object]:
    """Verify the complete bake graph and publish an idempotent launcher receipt."""

    contract.validate()
    root = _physical_directory(contract.output_root, "bake output root")
    run_root = _physical_directory(contract.run_root, "run root")
    _require(
        root == run_root or run_root in root.parents,
        "bake output root is outside the run root",
    )

    source_specs = (
        (
            contract.producer_path,
            contract.expected_producer_sha256,
            "navmesh baker source",
        ),
        (
            contract.auditor_path,
            contract.expected_auditor_sha256,
            "bake auditor source",
        ),
        (
            contract.launcher_path,
            contract.expected_launcher_sha256,
            "Slurm launcher source",
        ),
    )
    for path, expected_sha, label in source_specs:
        snapshot = _snapshot(path, label)
        _require(snapshot.content_sha256 == expected_sha, f"{label} changed")

    published = _physical_directory(root / "published", "published bake artifacts")
    manifest_path = published / DERIVED_MANIFEST_FILE
    index_path = published / BAKE_INDEX_FILE
    derived, derived_raw = _load_canonical(manifest_path, "derived manifest")
    index, index_raw = _load_canonical(index_path, "bake index")
    derived_sha = _verify_sidecar(manifest_path, derived_raw)
    index_sha = _verify_sidecar(index_path, index_raw)

    derivation = derived.get("geometry_bake_derivation")
    _require(isinstance(derivation, dict), "geometry bake derivation is absent")
    _require(
        derivation.get("schema_version") == DERIVATION_SCHEMA,
        "derived schema changed",
    )
    _require(derivation.get("status") == VERIFIED_STATUS, "derived status changed")
    _require(
        derivation.get("parent_manifest_sha256")
        == contract.expected_parent_manifest_sha256,
        "derived manifest parent binding changed",
    )
    _require(
        derivation.get("settings_file_sha256") == contract.expected_settings_sha256,
        "derived manifest settings binding changed",
    )
    runtime = _validate_runtime(
        derivation.get("runtime"),
        expected_habitat_version=contract.expected_habitat_version,
        expected_bindings_sha256=contract.expected_bindings_sha256,
    )
    _require(
        derivation.get("selection_boundary") == SELECTION_BOUNDARY
        and derivation.get("determinism_boundary") == DETERMINISM_BOUNDARY,
        "derived boundary statement changed",
    )

    _require(index.get("schema_version") == INDEX_SCHEMA, "bake index schema changed")
    _require(index.get("status") == VERIFIED_STATUS, "bake index status changed")
    _require(
        index.get("parent_manifest_sha256") == contract.expected_parent_manifest_sha256,
        "bake index parent changed",
    )
    _require(
        index.get("settings_sha256") == contract.expected_settings_sha256,
        "bake index settings changed",
    )
    _require(index.get("runtime") == runtime, "index runtime differs")
    index_scenes = index.get("scenes")
    _require(
        index.get("scene_count") == contract.expected_scene_count
        and isinstance(index_scenes, dict)
        and len(index_scenes) == contract.expected_scene_count,
        f"bake index does not cover {contract.expected_scene_count} scenes",
    )
    producer_record = index.get("producer")
    _require(
        isinstance(producer_record, dict)
        and producer_record.get("content_sha256") == contract.expected_producer_sha256,
        "bake index producer binding changed",
    )
    _require(
        index.get("fresh_simulator_repetitions")
        == contract.expected_fresh_simulator_repetitions,
        "bake repetition contract changed",
    )
    _require(
        index.get("requested_settings_sha256")
        == derivation.get("requested_settings_sha256"),
        "requested settings signature differs",
    )
    _require(
        index.get("runtime_effective_settings_sha256")
        == derivation.get("runtime_effective_settings_sha256"),
        "effective settings signature differs",
    )
    _require(
        index.get("selection_boundary") == SELECTION_BOUNDARY
        and index.get("determinism_boundary") == DETERMINISM_BOUNDARY,
        "index boundary statement changed",
    )

    roots = derived.get("input_roots")
    _require(isinstance(roots, dict), "derived input roots are invalid")
    geometry_bake_root = roots.get("geometry_bake_root")
    declared_navmesh_root = roots.get("navmesh_root")
    _require(
        isinstance(geometry_bake_root, str) and geometry_bake_root,
        "derived geometry root is invalid",
    )
    _require(
        isinstance(declared_navmesh_root, str) and declared_navmesh_root,
        "derived navmesh root is invalid",
    )
    _require(
        Path(geometry_bake_root).resolve() == root,
        "derived geometry root changed",
    )
    navmesh_root = _physical_directory(root / "scenes", "baked NavMesh root")
    _require(
        Path(declared_navmesh_root).resolve() == navmesh_root,
        "derived navmesh root changed",
    )

    run_contract_path, run_contract_record = _resolve_record(
        derivation.get("run_contract"), root, "run contract"
    )
    run_contract, _run_contract_raw = _load_canonical(run_contract_path, "run contract")
    _require(
        run_contract_record == derivation["run_contract"],
        "run contract file record changed",
    )
    _require(
        run_contract.get("schema_version") == RUN_CONTRACT_SCHEMA,
        "run contract schema changed",
    )
    _require(
        run_contract.get("source_manifest", {}).get("content_sha256")
        == contract.expected_parent_manifest_sha256,
        "run contract parent changed",
    )
    _require(
        run_contract.get("settings", {}).get("content_sha256")
        == contract.expected_settings_sha256,
        "run contract settings changed",
    )
    _require(run_contract.get("runtime") == runtime, "run contract runtime differs")
    _require(
        run_contract.get("producer", {}).get("content_sha256")
        == contract.expected_producer_sha256,
        "run contract producer differs",
    )
    _require(
        run_contract.get("fresh_simulator_repetitions")
        == contract.expected_fresh_simulator_repetitions,
        "run contract repetitions changed",
    )
    _require(
        run_contract.get("selection_boundary") == SELECTION_BOUNDARY
        and run_contract.get("determinism_boundary") == DETERMINISM_BOUNDARY,
        "run contract boundary statement changed",
    )

    index_record_path, index_record = _resolve_record(
        derivation.get("bake_index"), root, "derived bake index"
    )
    _require(
        index_record_path == index_path.resolve(), "derived bake-index path changed"
    )
    _require(
        index_record["content_sha256"] == index_sha, "derived bake-index SHA differs"
    )

    derived_scenes = derived.get("scenes")
    _require(
        isinstance(derived_scenes, list)
        and len(derived_scenes) == contract.expected_scene_count
        and all(isinstance(row, dict) for row in derived_scenes),
        f"derived scene list does not cover {contract.expected_scene_count} scenes",
    )
    scene_ids = [row.get("scene") for row in derived_scenes]
    _require(
        all(isinstance(scene_id, str) and scene_id for scene_id in scene_ids),
        "derived scene id is invalid",
    )
    _require(
        all(isinstance(scene_id, str) and scene_id for scene_id in index_scenes),
        "index scene id is invalid",
    )
    _require(
        len(scene_ids) == len(derived_scenes),
        "derived scene count changed while building the scene index",
    )
    derived_by_scene = {
        scene_id: row for scene_id, row in zip(scene_ids, derived_scenes)
    }
    _require(
        len(derived_by_scene) == contract.expected_scene_count
        and set(derived_by_scene) == set(index_scenes),
        "derived/index scene membership differs",
    )
    settings_record = run_contract.get("settings")
    _require(isinstance(settings_record, dict), "run contract settings are invalid")
    requested = settings_record.get("requested")
    effective = settings_record.get("runtime_effective")
    _require(
        isinstance(requested, dict) and isinstance(effective, dict),
        "run contract settings payload changed",
    )
    requested_signature = sha256_bytes(canonical_json_bytes(requested))
    effective_signature = sha256_bytes(canonical_json_bytes(effective))
    _require(
        requested_signature == derivation.get("requested_settings_sha256"),
        "requested settings contents/signature differ",
    )
    _require(
        effective_signature == derivation.get("runtime_effective_settings_sha256"),
        "effective settings contents/signature differ",
    )
    _require(
        settings_record.get("requested_signature_sha256") == requested_signature
        and settings_record.get("runtime_effective_signature_sha256")
        == effective_signature,
        "run contract settings signature changed",
    )

    for scene_id in sorted(index_scenes):
        index_scene = index_scenes[scene_id]
        derived_scene = derived_by_scene[scene_id]
        _require(
            isinstance(index_scene, dict)
            and set(index_scene) == {"source_glb_sha256", "navmesh", "bake_receipt"},
            f"index scene fields changed: {scene_id}",
        )
        _require(
            derived_scene.get("split_role") in ALLOWED_SPLIT_ROLES,
            f"forbidden derived split role: {scene_id}",
        )
        environment = derived_scene.get("environment")
        _require(
            isinstance(environment, dict), f"environment record absent: {scene_id}"
        )
        _require(
            index_scene["source_glb_sha256"] == environment.get("content_sha256"),
            f"source GLB binding differs: {scene_id}",
        )

        navmesh_path, navmesh_record = _resolve_record(
            index_scene["navmesh"], root, f"{scene_id} index navmesh"
        )
        derived_navmesh_path, derived_navmesh = _resolve_record(
            derived_scene.get("navmesh"),
            navmesh_root,
            f"{scene_id} derived navmesh",
        )
        _require(
            derived_navmesh_path == navmesh_path
            and derived_navmesh["content_sha256"] == navmesh_record["content_sha256"],
            f"derived/index navmesh differs: {scene_id}",
        )

        receipt_path, receipt_record = _resolve_record(
            index_scene["bake_receipt"], root, f"{scene_id} receipt"
        )
        _require(
            derived_scene.get("geometry_bake_receipt") == receipt_record,
            f"derived/index receipt differs: {scene_id}",
        )
        receipt, _receipt_raw = _load_canonical(receipt_path, f"{scene_id} receipt")
        _require(
            set(receipt) == SCENE_RECEIPT_FIELDS,
            f"receipt fields changed: {scene_id}",
        )
        _require(
            receipt.get("schema_version") == SCENE_RECEIPT_SCHEMA
            and receipt.get("status") == VERIFIED_STATUS
            and receipt.get("scene_id") == scene_id
            and receipt.get("split_role") == derived_scene["split_role"],
            f"receipt identity changed: {scene_id}",
        )
        _require(
            receipt.get("parent_manifest_sha256")
            == contract.expected_parent_manifest_sha256,
            f"receipt parent changed: {scene_id}",
        )
        _require(
            receipt.get("source_glb")
            == {
                "manifest_path": environment.get("path"),
                "bytes": environment.get("bytes"),
                "content_sha256": index_scene["source_glb_sha256"],
            },
            f"receipt GLB changed: {scene_id}",
        )
        _require(
            receipt.get("requested_settings") == requested
            and receipt.get("requested_settings_sha256") == requested_signature,
            f"receipt requested settings changed: {scene_id}",
        )
        _require(
            receipt.get("runtime_effective_settings") == effective
            and receipt.get("runtime_effective_settings_sha256") == effective_signature,
            f"receipt effective settings changed: {scene_id}",
        )
        _require(
            receipt.get("runtime") == runtime, f"receipt runtime differs: {scene_id}"
        )
        _require(
            receipt.get("bake_method") == BAKE_METHOD,
            f"receipt bake method changed: {scene_id}",
        )
        _require(
            receipt.get("output_navmesh") == navmesh_record,
            f"receipt output navmesh differs: {scene_id}",
        )
        determinism = receipt.get("determinism")
        _require(
            isinstance(determinism, dict)
            and set(determinism)
            == {
                "fresh_simulator_repetitions",
                "navmesh_sha256_by_repetition",
                "byte_for_byte_equal",
            }
            and determinism["fresh_simulator_repetitions"]
            == contract.expected_fresh_simulator_repetitions
            and determinism["byte_for_byte_equal"] is True
            and determinism["navmesh_sha256_by_repetition"]
            == [navmesh_record["content_sha256"]]
            * contract.expected_fresh_simulator_repetitions,
            f"receipt double-bake proof changed: {scene_id}",
        )
        bake_observation = _validate_observation(
            receipt.get("bake_observation"), f"{scene_id} bake observation"
        )
        roundtrip_observation = _validate_observation(
            receipt.get("roundtrip_observation"),
            f"{scene_id} roundtrip observation",
        )
        _require(
            _observations_equal(bake_observation, roundtrip_observation),
            f"receipt bake/roundtrip observation differs: {scene_id}",
        )
        _require(
            receipt.get("selection_boundary") == SELECTION_BOUNDARY
            and receipt.get("determinism_boundary") == DETERMINISM_BOUNDARY,
            f"receipt boundary statement changed: {scene_id}",
        )

    habitat_python_bytes, habitat_python_sha = _file_sha_following_final_symlink(
        contract.habitat_python_path, "Habitat Python"
    )
    _require(
        habitat_python_sha == contract.expected_habitat_python_sha256,
        "Habitat Python changed during bake",
    )
    base_sif_bytes, base_sif_sha = _file_sha_following_final_symlink(
        contract.base_sif_path, "base SIF"
    )
    _require(
        base_sif_bytes == contract.expected_base_sif_bytes
        and base_sif_sha == contract.expected_base_sif_sha256,
        "base SIF changed during bake",
    )

    launcher_receipt = {
        "schema_version": LAUNCHER_RECEIPT_SCHEMA,
        "status": LAUNCHER_STATUS,
        "producer_commit": contract.expected_producer_commit,
        "launcher": {
            "path": "MemNavData/slurm_nlsr_navmesh_bake.sbatch",
            "content_sha256": contract.expected_launcher_sha256,
        },
        "producer": {
            "path": "MemNavData/bake_pinned_navmeshes.py",
            "content_sha256": contract.expected_producer_sha256,
        },
        "auditor": {
            "path": "MemNavData/audit_pinned_navmesh_bake.py",
            "content_sha256": contract.expected_auditor_sha256,
        },
        "inputs": {
            "parent_manifest_sha256": contract.expected_parent_manifest_sha256,
            "settings_file_sha256": contract.expected_settings_sha256,
        },
        "runtime": {
            "habitat_sim_version": contract.expected_habitat_version,
            "habitat_sim_bindings_sha256": contract.expected_bindings_sha256,
            "habitat_python": {
                "path": str(contract.habitat_python_path),
                "bytes": habitat_python_bytes,
                "content_sha256": contract.expected_habitat_python_sha256,
            },
            "base_sif": {
                "path": str(contract.base_sif_path),
                "bytes": base_sif_bytes,
                "content_sha256": contract.expected_base_sif_sha256,
            },
            "environment_contract": {
                "singularity_cleanenv": True,
                "singularity_no_home": True,
                "python_no_user_site": True,
                "pythonpath_is_exact_producer_checkout": True,
            },
            "habitat_runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        },
        "outputs": {
            "derived_geometry_manifest": _stable_record(
                manifest_path, f"published/{DERIVED_MANIFEST_FILE}"
            ),
            "navmesh_bake_index": _stable_record(
                index_path, f"published/{BAKE_INDEX_FILE}"
            ),
            "scene_count": contract.expected_scene_count,
            "fresh_simulator_repetitions": (
                contract.expected_fresh_simulator_repetitions
            ),
        },
    }
    receipt_bytes = canonical_json_bytes(launcher_receipt)
    receipt_path = run_root / LAUNCHER_RECEIPT_FILE
    receipt_sha = sha256_bytes(receipt_bytes)
    sidecar_bytes = f"{receipt_sha}  {receipt_path.name}\n".encode("ascii")
    _write_once_or_verify(receipt_path, receipt_bytes, "launcher receipt")
    _write_once_or_verify(
        Path(f"{receipt_path}.sha256"), sidecar_bytes, "launcher receipt sidecar"
    )
    _fsync_directory(run_root)
    _require(
        receipt_path.read_bytes() == receipt_bytes
        and Path(f"{receipt_path}.sha256").read_bytes() == sidecar_bytes,
        "launcher receipt publication verification failed",
    )

    return {
        "navmesh_bake_cross_audit": "passed",
        "scene_count": contract.expected_scene_count,
        "derived_manifest_sha256": derived_sha,
        "navmesh_bake_index_sha256": index_sha,
        "launcher_receipt_sha256": receipt_sha,
        "auditor_sha256": contract.expected_auditor_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-audit a pinned NavMesh bake and publish launcher receipt."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--expected-settings-sha256", required=True)
    parser.add_argument("--expected-bindings-sha256", required=True)
    parser.add_argument("--expected-producer-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-habitat-version", required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--producer-path", type=Path, required=True)
    parser.add_argument("--auditor-path", type=Path, required=True)
    parser.add_argument("--launcher-path", type=Path, required=True)
    parser.add_argument("--habitat-python-path", type=Path, required=True)
    parser.add_argument("--expected-habitat-python-sha256", required=True)
    parser.add_argument("--base-sif-path", type=Path, required=True)
    parser.add_argument("--expected-base-sif-sha256", required=True)
    parser.add_argument("--expected-base-sif-bytes", type=int, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument(
        "--expected-fresh-simulator-repetitions", type=int, required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_navmesh_bake(
        AuditContract(
            output_root=args.output_root,
            run_root=args.run_root,
            expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
            expected_settings_sha256=args.expected_settings_sha256,
            expected_bindings_sha256=args.expected_bindings_sha256,
            expected_producer_sha256=args.expected_producer_sha256,
            expected_auditor_sha256=args.expected_auditor_sha256,
            expected_habitat_version=args.expected_habitat_version,
            expected_producer_commit=args.expected_producer_commit,
            expected_launcher_sha256=args.expected_launcher_sha256,
            producer_path=args.producer_path,
            auditor_path=args.auditor_path,
            launcher_path=args.launcher_path,
            habitat_python_path=args.habitat_python_path,
            expected_habitat_python_sha256=(args.expected_habitat_python_sha256),
            base_sif_path=args.base_sif_path,
            expected_base_sif_sha256=args.expected_base_sif_sha256,
            expected_base_sif_bytes=args.expected_base_sif_bytes,
            expected_scene_count=args.expected_scene_count,
            expected_fresh_simulator_repetitions=(
                args.expected_fresh_simulator_repetitions
            ),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
