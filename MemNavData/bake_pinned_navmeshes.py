#!/usr/bin/env python3
"""Bake provenance-backed NavMeshes for the real H24 evaluation geometry.

Why this stage exists
---------------------
``generate_twoleg.make_sim`` calls ``NavMeshSettings.set_defaults()``, changes
the agent radius/height to 0.30 m/1.50 m, and recomputes the NavMesh before it
samples paths.  The formal H24 collector intentionally never recomputes: it
loads the NavMesh named by its manifest.  A hash of an old NavMesh proves its
bytes, but it does *not* prove which settings created those bytes.  Therefore
an old manifest plus a newly asserted settings JSON has a real geometry-parity
gap.

This CPU-only stage closes that gap by creating new physical NavMeshes from the
content-pinned GLBs in a pinned expert/multistage manifest.  It runs only under
an exact Habitat-Sim version and compiled-bindings hash, applies the complete
canonical settings contract, bakes every scene twice in fresh simulators,
requires byte-for-byte identical serializations, round-trip loads the result,
and records a per-scene receipt.  It then publishes an augmented derived expert
manifest whose NavMesh root/records point only at these new files.  That
derived manifest is the input to ``build_frozen_geometry_map.py`` and to later
candidate/H24 stages.

Contract boundary
-----------------
Unit tests prove the orchestration, hashing, monotone publication and resume
contracts with a fake runtime.  They cannot prove determinism of Habitat's C++
Recast implementation or its serialization format.  The *real invocation*
therefore performs two independent fresh-simulator bakes for every GLB and
records their hashes.  This proves repeatability for that exact GLB, settings,
Habitat version and bindings binary in that run; it does not claim universal
cross-platform or cross-build determinism.

No episode sample, goal, success metric, geodesic label, or final-reserved
scene is read to make a bake decision.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

try:
    from MemNavData.build_frozen_geometry_map import (
        FileSnapshot,
        GeometryMapBuildError,
        _source_from_record,
        canonical_json_bytes,
        load_pinned_manifest,
        load_pinned_settings,
        sha256_bytes,
        snapshot_regular_file,
    )
    from MemNavData.habitat_rollout_primitives import (
        NAVMESH_BOOL_FIELDS,
        NAVMESH_FLOAT_FIELDS,
        canonical_navmesh_settings,
        navmesh_settings_signature,
    )
except ImportError:  # direct ``python MemNavData/<script>.py`` execution
    from build_frozen_geometry_map import (  # type: ignore
        FileSnapshot,
        GeometryMapBuildError,
        _source_from_record,
        canonical_json_bytes,
        load_pinned_manifest,
        load_pinned_settings,
        sha256_bytes,
        snapshot_regular_file,
    )
    from habitat_rollout_primitives import (  # type: ignore
        NAVMESH_BOOL_FIELDS,
        NAVMESH_FLOAT_FIELDS,
        canonical_navmesh_settings,
        navmesh_settings_signature,
    )


BAKE_RECEIPT_SCHEMA = "nlsr_navmesh_bake_receipt_v1"
BAKE_INDEX_SCHEMA = "nlsr_navmesh_bake_index_v1"
RUN_CONTRACT_SCHEMA = "nlsr_navmesh_bake_run_contract_v1"
DERIVATION_SCHEMA = "nlsr_derived_geometry_manifest_v1"
BAKE_STATUS = "fresh_double_bake_roundtrip_verified"
EXPECTED_HABITAT_VERSION = "0.3.1"
REQUIRED_AGENT_RADIUS_M = 0.30
REQUIRED_AGENT_HEIGHT_M = 1.50
REPETITIONS = 2
SETTINGS_FILE = "navmesh_settings_habitat_0_3_1_agent030.json"
RUN_CONTRACT_FILE = "run_contract.json"
SCENES_DIRECTORY = "scenes"
PUBLISHED_DIRECTORY = "published"
DERIVED_MANIFEST_FILE = "derived_geometry_manifest.json"
BAKE_INDEX_FILE = "navmesh_bake_index.json"
SCENE_STAGING_PREFIX = ".bake-staging-"
PUBLISH_STAGING_PREFIX = ".publish-staging-"
ALLOWED_SPLIT_ROLES = frozenset(("train", "development"))
SELECTION_BOUNDARY = (
    "scene membership and GLB bytes come only from the pinned expert manifest; "
    "no episode sample, goal, geodesic, success, evaluation, or final data label"
)
DETERMINISM_BOUNDARY = (
    "byte equality is verified for two fresh-simulator bakes in this exact "
    "runtime; no cross-platform or cross-build determinism is claimed"
)


class NavmeshBakeError(RuntimeError):
    """The bake inputs, runtime, output, or resume state violated the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NavmeshBakeError(message)


def _valid_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def _stable_file_record(
    snapshot: FileSnapshot,
    relative_path: str,
) -> dict[str, object]:
    return {
        "path": relative_path,
        "path_sha256": sha256_bytes(relative_path.encode("utf-8")),
        "bytes": snapshot.byte_count,
        "content_sha256": snapshot.content_sha256,
    }


def _byte_file_record(payload: bytes, relative_path: str) -> dict[str, object]:
    return {
        "path": relative_path,
        "path_sha256": sha256_bytes(relative_path.encode("utf-8")),
        "bytes": len(payload),
        "content_sha256": sha256_bytes(payload),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_canonical(path: Path, label: str) -> tuple[dict, bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not physical")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NavmeshBakeError(f"cannot load {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(raw == canonical_json_bytes(value), f"{label} is not canonical")
    return value, raw


def _settings_close(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    _require(set(actual) == set(expected), f"{label} setting fields changed")
    for field in NAVMESH_BOOL_FIELDS:
        _require(
            type(actual[field]) is bool
            and type(expected[field]) is bool
            and actual[field] is expected[field],
            f"{label}.{field} changed",
        )
    for field in NAVMESH_FLOAT_FIELDS:
        _require(
            not isinstance(actual[field], bool)
            and not isinstance(expected[field], bool),
            f"{label}.{field} must be numeric, not boolean",
        )
        _require(
            math.isfinite(float(actual[field]))
            and math.isfinite(float(expected[field]))
            and math.isclose(
                float(actual[field]),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            f"{label}.{field} changed: {actual[field]} != {expected[field]}",
        )


@dataclass(frozen=True)
class NavmeshObservation:
    navigable_area_m2: float
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    vertex_count: int
    index_count: int

    def __post_init__(self) -> None:
        _require(
            not isinstance(self.navigable_area_m2, bool)
            and math.isfinite(self.navigable_area_m2)
            and self.navigable_area_m2 > 0.0,
            "navmesh navigable area must be finite and positive",
        )
        for label, values in (
            ("bounds_min_xyz", self.bounds_min_xyz),
            ("bounds_max_xyz", self.bounds_max_xyz),
        ):
            _require(
                isinstance(values, tuple)
                and len(values) == 3
                and all(
                    not isinstance(value, bool) and math.isfinite(value)
                    for value in values
                ),
                f"{label} must be a finite xyz vector",
            )
        _require(
            type(self.vertex_count) is int
            and type(self.index_count) is int
            and self.vertex_count > 0
            and self.index_count > 0,
            "navmesh vertex/index counts must be positive",
        )
        _require(
            all(
                lower <= upper
                for lower, upper in zip(self.bounds_min_xyz, self.bounds_max_xyz)
            ),
            "navmesh minimum bounds must not exceed maximum bounds",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "navigable_area_m2": self.navigable_area_m2,
            "bounds_min_xyz": list(self.bounds_min_xyz),
            "bounds_max_xyz": list(self.bounds_max_xyz),
            "vertex_count": self.vertex_count,
            "index_count": self.index_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NavmeshObservation":
        required = {
            "navigable_area_m2",
            "bounds_min_xyz",
            "bounds_max_xyz",
            "vertex_count",
            "index_count",
        }
        _require(
            isinstance(value, Mapping) and set(value) == required, "bad observation"
        )
        area = value["navigable_area_m2"]
        minimum = value["bounds_min_xyz"]
        maximum = value["bounds_max_xyz"]
        vertex_count = value["vertex_count"]
        index_count = value["index_count"]
        _require(
            not isinstance(area, bool) and isinstance(area, (int, float)),
            "invalid navmesh area",
        )
        for label, bounds in (("minimum", minimum), ("maximum", maximum)):
            _require(
                isinstance(bounds, list)
                and len(bounds) == 3
                and all(
                    not isinstance(item, bool) and isinstance(item, (int, float))
                    for item in bounds
                ),
                f"invalid navmesh {label} bounds",
            )
        _require(
            type(vertex_count) is int and type(index_count) is int,
            "invalid navmesh topology counts",
        )
        try:
            return cls(
                navigable_area_m2=float(area),
                bounds_min_xyz=tuple(float(item) for item in minimum),
                bounds_max_xyz=tuple(float(item) for item in maximum),
                vertex_count=vertex_count,
                index_count=index_count,
            )
        except (TypeError, ValueError) as error:
            raise NavmeshBakeError("invalid navmesh observation") from error


@dataclass(frozen=True)
class BakeResult:
    effective_settings: Mapping[str, Any]
    bake_observation: NavmeshObservation
    roundtrip_observation: NavmeshObservation


class BakeRuntime(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def assert_settings_contract(
        self, requested: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def bake_once(
        self,
        glb_path: Path,
        navmesh_output: Path,
        requested_settings: Mapping[str, Any],
    ) -> BakeResult: ...

    def validate_output(
        self,
        navmesh_path: Path,
        requested_settings: Mapping[str, Any],
    ) -> BakeResult: ...


def _validated_runtime_identity(
    runtime: BakeRuntime,
    *,
    expected_version: str,
    expected_bindings_sha256: str,
) -> dict[str, Any]:
    identity = copy.deepcopy(runtime.identity)
    required = {
        "habitat_sim_version",
        "python_version",
        "runtime_files",
        "runtime_fingerprint_sha256",
    }
    _require(
        isinstance(identity, dict) and set(identity) == required,
        "runtime identity fields changed",
    )
    _require(
        identity["habitat_sim_version"] == expected_version,
        "runtime identity Habitat version mismatch",
    )
    _require(
        isinstance(identity["python_version"], str)
        and bool(identity["python_version"]),
        "runtime identity Python version is invalid",
    )
    runtime_files = identity["runtime_files"]
    _require(
        isinstance(runtime_files, dict)
        and set(runtime_files) == {"habitat_sim_init", "habitat_sim_bindings"},
        "runtime identity file records changed",
    )
    for label, record in runtime_files.items():
        _require(
            isinstance(record, dict)
            and set(record) == {"name", "bytes", "content_sha256"},
            f"runtime identity {label} record changed",
        )
        _require(
            isinstance(record["name"], str)
            and bool(record["name"])
            and "/" not in record["name"]
            and "\\" not in record["name"],
            f"runtime identity {label} name is invalid",
        )
        _require(
            type(record["bytes"]) is int and record["bytes"] > 0,
            f"runtime identity {label} byte count is invalid",
        )
        _valid_sha256(record["content_sha256"], f"runtime identity {label} SHA256")
    _require(
        runtime_files["habitat_sim_bindings"]["content_sha256"]
        == expected_bindings_sha256,
        "runtime identity bindings SHA256 mismatch",
    )
    fingerprint_input = {
        "habitat_sim_version": identity["habitat_sim_version"],
        "python_version": identity["python_version"],
        "runtime_files": runtime_files,
    }
    _require(
        _valid_sha256(
            identity["runtime_fingerprint_sha256"],
            "runtime identity fingerprint",
        )
        == sha256_bytes(canonical_json_bytes(fingerprint_input)),
        "runtime identity fingerprint mismatch",
    )
    return identity


def _assert_runtime_stable(
    runtime: BakeRuntime,
    expected_identity: Mapping[str, Any],
) -> None:
    _require(runtime.identity == expected_identity, "Habitat runtime identity drifted")


class HabitatBakeRuntime:
    """Thin, fail-closed adapter around the real Habitat-Sim 0.3.1 API."""

    def __init__(
        self,
        habitat_sim: Any,
        *,
        expected_version: str,
        expected_bindings_sha256: str,
    ) -> None:
        version = getattr(habitat_sim, "__version__", None)
        _require(
            version == expected_version, f"Habitat-Sim version mismatch: {version}"
        )
        _require(
            expected_version == EXPECTED_HABITAT_VERSION,
            f"formal bake requires Habitat-Sim {EXPECTED_HABITAT_VERSION}",
        )
        try:
            bindings = habitat_sim._ext.habitat_sim_bindings
            bindings_path = Path(bindings.__file__)
            init_path = Path(habitat_sim.__file__)
        except (AttributeError, TypeError) as error:
            raise NavmeshBakeError(
                "Habitat-Sim runtime file provenance is unavailable"
            ) from error
        bindings_snapshot = snapshot_regular_file(bindings_path, "Habitat bindings")
        init_snapshot = snapshot_regular_file(init_path, "Habitat Python package")
        expected_binding = _valid_sha256(
            expected_bindings_sha256, "expected Habitat bindings SHA256"
        )
        _require(
            bindings_snapshot.content_sha256 == expected_binding,
            "Habitat bindings SHA256 mismatch",
        )
        runtime_files = {
            "habitat_sim_init": {
                "name": init_path.name,
                "bytes": init_snapshot.byte_count,
                "content_sha256": init_snapshot.content_sha256,
            },
            "habitat_sim_bindings": {
                "name": bindings_path.name,
                "bytes": bindings_snapshot.byte_count,
                "content_sha256": bindings_snapshot.content_sha256,
            },
        }
        fingerprint_input = {
            "habitat_sim_version": version,
            "python_version": sys.version.split()[0],
            "runtime_files": runtime_files,
        }
        self._identity = dict(fingerprint_input)
        self._identity["runtime_fingerprint_sha256"] = sha256_bytes(
            canonical_json_bytes(fingerprint_input)
        )
        self.habitat_sim = habitat_sim

    @property
    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def _settings(self, requested: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        settings = self.habitat_sim.NavMeshSettings()
        settings.set_defaults()
        defaults = canonical_navmesh_settings(settings)
        for field in requested:
            if field not in {"agent_radius", "agent_height"}:
                if field in NAVMESH_BOOL_FIELDS:
                    _require(
                        defaults[field] is requested[field],
                        f"Habitat set_defaults parity.{field} changed",
                    )
                else:
                    _require(
                        math.isclose(
                            float(defaults[field]),
                            float(requested[field]),
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        ),
                        f"Habitat set_defaults parity.{field} changed",
                    )
        settings.agent_radius = float(requested["agent_radius"])
        settings.agent_height = float(requested["agent_height"])
        effective = canonical_navmesh_settings(settings)
        _settings_close(effective, requested, "runtime effective settings")
        return settings, effective

    def assert_settings_contract(
        self, requested: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        _settings, effective = self._settings(requested)
        return effective

    @staticmethod
    def _observation(pathfinder: Any) -> NavmeshObservation:
        _require(bool(pathfinder.is_loaded), "Habitat PathFinder is not loaded")
        bounds = pathfinder.get_bounds()
        lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
        upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
        _require(lower.size == 3 and upper.size == 3, "navmesh bounds are invalid")
        return NavmeshObservation(
            navigable_area_m2=float(pathfinder.navigable_area),
            bounds_min_xyz=tuple(float(value) for value in lower),
            bounds_max_xyz=tuple(float(value) for value in upper),
            vertex_count=len(pathfinder.build_navmesh_vertices()),
            index_count=len(pathfinder.build_navmesh_vertex_indices()),
        )

    def _roundtrip(
        self,
        navmesh_path: Path,
        requested_settings: Mapping[str, Any],
    ) -> tuple[dict[str, Any], NavmeshObservation]:
        pathfinder = self.habitat_sim.PathFinder()
        loaded = pathfinder.load_nav_mesh(str(navmesh_path))
        _require(
            loaded is True, f"Habitat failed to load baked navmesh: {navmesh_path}"
        )
        effective = canonical_navmesh_settings(pathfinder.nav_mesh_settings)
        _settings_close(effective, requested_settings, "roundtrip embedded settings")
        return effective, self._observation(pathfinder)

    def bake_once(
        self,
        glb_path: Path,
        navmesh_output: Path,
        requested_settings: Mapping[str, Any],
    ) -> BakeResult:
        settings, effective = self._settings(requested_settings)
        simulator_configuration = self.habitat_sim.SimulatorConfiguration()
        _require(
            hasattr(simulator_configuration, "create_renderer"),
            "Habitat runtime lacks CPU-only create_renderer control",
        )
        simulator_configuration.scene_id = str(glb_path.resolve())
        simulator_configuration.enable_physics = False
        simulator_configuration.create_renderer = False
        if hasattr(simulator_configuration, "load_semantic_mesh"):
            simulator_configuration.load_semantic_mesh = False
        agent_configuration = self.habitat_sim.agent.AgentConfiguration()
        agent_configuration.sensor_specifications = []
        simulator = self.habitat_sim.Simulator(
            self.habitat_sim.Configuration(
                simulator_configuration, [agent_configuration]
            )
        )
        try:
            recomputed = simulator.recompute_navmesh(simulator.pathfinder, settings)
            _require(
                recomputed is True, f"Habitat recompute_navmesh failed: {glb_path}"
            )
            bake_observation = self._observation(simulator.pathfinder)
            saved = simulator.pathfinder.save_nav_mesh(str(navmesh_output))
            _require(saved is True, f"Habitat save_nav_mesh failed: {navmesh_output}")
        finally:
            simulator.close()
        _require(
            navmesh_output.is_file() and not navmesh_output.is_symlink(),
            "Habitat did not create a physical navmesh output",
        )
        roundtrip_settings, roundtrip_observation = self._roundtrip(
            navmesh_output, requested_settings
        )
        _settings_close(roundtrip_settings, effective, "bake/roundtrip settings")
        _observations_close(
            bake_observation, roundtrip_observation, "bake/roundtrip observation"
        )
        return BakeResult(effective, bake_observation, roundtrip_observation)

    def validate_output(
        self,
        navmesh_path: Path,
        requested_settings: Mapping[str, Any],
    ) -> BakeResult:
        effective, observation = self._roundtrip(navmesh_path, requested_settings)
        return BakeResult(effective, observation, observation)


def _observations_close(
    actual: NavmeshObservation,
    expected: NavmeshObservation,
    label: str,
) -> None:
    _require(
        math.isclose(
            actual.navigable_area_m2,
            expected.navigable_area_m2,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        f"{label} navigable area changed",
    )
    _require(
        np.allclose(
            np.asarray(actual.bounds_min_xyz),
            np.asarray(expected.bounds_min_xyz),
            rtol=0.0,
            atol=1e-6,
        )
        and np.allclose(
            np.asarray(actual.bounds_max_xyz),
            np.asarray(expected.bounds_max_xyz),
            rtol=0.0,
            atol=1e-6,
        ),
        f"{label} bounds changed",
    )
    _require(
        actual.vertex_count == expected.vertex_count
        and actual.index_count == expected.index_count,
        f"{label} topology counts changed",
    )


def _files_byte_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_block = left.read(8 * 1024 * 1024)
            right_block = right.read(8 * 1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


@dataclass(frozen=True)
class SceneInput:
    scene_id: str
    split_role: str
    manifest_scene: Mapping[str, Any]
    glb_path: Path
    glb_snapshot: FileSnapshot
    original_navmesh_record: Mapping[str, Any]
    output_key: str


def _resolve_environment_root(
    manifest: Mapping[str, Any],
    override: Path | str | None,
) -> Path:
    roots = manifest.get("input_roots")
    _require(isinstance(roots, Mapping), "manifest.input_roots is invalid")
    raw = override if override is not None else roots.get("environment_root")
    _require(
        isinstance(raw, (str, os.PathLike)) and str(raw), "environment root absent"
    )
    path = Path(raw)
    _require(
        path.is_dir() and not path.is_symlink(), f"environment root invalid: {path}"
    )
    return path.resolve(strict=True)


def _scene_inputs(
    manifest: Mapping[str, Any],
    environment_root: Path,
) -> list[SceneInput]:
    scenes = manifest.get("scenes")
    _require(isinstance(scenes, list) and scenes, "manifest.scenes must be non-empty")
    result = []
    seen_scenes = set()
    seen_glbs = set()
    for index, raw_scene in enumerate(scenes):
        _require(isinstance(raw_scene, Mapping), f"scene {index} is not an object")
        scene_id = raw_scene.get("scene")
        role = raw_scene.get("split_role")
        _require(isinstance(scene_id, str) and scene_id, f"scene {index} id is invalid")
        _require(scene_id not in seen_scenes, f"duplicate scene id: {scene_id}")
        _require(
            role in ALLOWED_SPLIT_ROLES,
            f"forbidden/final scene role for {scene_id}: {role}",
        )
        _require(
            isinstance(raw_scene.get("navmesh"), Mapping),
            f"source navmesh record missing for {scene_id}",
        )
        try:
            glb_path, glb_snapshot = _source_from_record(
                raw_scene.get("environment"),
                environment_root,
                f"{scene_id}.environment",
            )
        except GeometryMapBuildError as error:
            raise NavmeshBakeError(str(error)) from error
        _require(glb_path not in seen_glbs, f"duplicate scene GLB path: {glb_path}")
        seen_scenes.add(scene_id)
        seen_glbs.add(glb_path)
        result.append(
            SceneInput(
                scene_id=scene_id,
                split_role=str(role),
                manifest_scene=raw_scene,
                glb_path=glb_path,
                glb_snapshot=glb_snapshot,
                original_navmesh_record=copy.deepcopy(raw_scene["navmesh"]),
                output_key=sha256_bytes(scene_id.encode("utf-8")),
            )
        )
    summary = manifest.get("summary")
    if isinstance(summary, Mapping) and "scene_count" in summary:
        _require(summary["scene_count"] == len(result), "manifest scene_count changed")
    return result


def _run_contract(
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_manifest_schema: str,
    settings_path: Path,
    settings_sha256: str,
    requested_settings: Mapping[str, Any],
    effective_settings: Mapping[str, Any],
    environment_root: Path,
    runtime_identity: Mapping[str, Any],
    scene_inputs: Sequence[SceneInput],
) -> dict:
    producer = snapshot_regular_file(Path(__file__), "navmesh baker source")
    return {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "content_sha256": source_manifest_sha256,
            "schema_version": source_manifest_schema,
        },
        "settings": {
            "path": str(settings_path.resolve()),
            "content_sha256": settings_sha256,
            "requested": copy.deepcopy(requested_settings),
            "requested_signature_sha256": navmesh_settings_signature(
                requested_settings
            ),
            "runtime_effective": copy.deepcopy(effective_settings),
            "runtime_effective_signature_sha256": navmesh_settings_signature(
                effective_settings
            ),
        },
        "runtime": copy.deepcopy(runtime_identity),
        "environment_root": str(environment_root),
        "scene_glbs": {
            scene.scene_id: {
                "path": scene.manifest_scene["environment"]["path"],
                "bytes": scene.glb_snapshot.byte_count,
                "content_sha256": scene.glb_snapshot.content_sha256,
            }
            for scene in sorted(scene_inputs, key=lambda item: item.scene_id)
        },
        "fresh_simulator_repetitions": REPETITIONS,
        "selection_boundary": SELECTION_BOUNDARY,
        "determinism_boundary": DETERMINISM_BOUNDARY,
        "producer": {
            "path": Path(__file__).name,
            "bytes": producer.byte_count,
            "content_sha256": producer.content_sha256,
        },
    }


def _stage_marker(run_signature: str, scene: SceneInput) -> dict:
    return {
        "run_contract_sha256": run_signature,
        "scene_id": scene.scene_id,
        "output_key": scene.output_key,
    }


def _scene_staging_name(run_signature: str, scene: SceneInput) -> str:
    return f"{SCENE_STAGING_PREFIX}{scene.output_key}-{run_signature}"


def _clean_stale_staging(
    scenes_root: Path,
    run_signature: str,
    scene_by_key: Mapping[str, SceneInput],
) -> None:
    expected = {
        _scene_staging_name(run_signature, scene): scene
        for scene in scene_by_key.values()
    }
    for path in scenes_root.iterdir():
        if not path.name.startswith(SCENE_STAGING_PREFIX):
            continue
        _require(
            path.is_dir() and not path.is_symlink(), f"unsafe staging path: {path}"
        )
        scene = expected.get(path.name)
        _require(scene is not None, f"stale staging ownership mismatch: {path}")
        marker_path = path / "staging_contract.json"
        if os.path.lexists(marker_path):
            marker, _raw = _load_canonical(marker_path, "staging contract")
            _require(
                marker == _stage_marker(run_signature, scene),
                f"stale staging ownership mismatch: {path}",
            )
        shutil.rmtree(path)


def _receipt_expected_fields() -> set[str]:
    return {
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


def _validate_receipt(
    receipt: object,
    *,
    scene: SceneInput,
    parent_manifest_sha256: str,
    requested_settings: Mapping[str, Any],
    effective_settings: Mapping[str, Any],
    runtime: BakeRuntime,
    runtime_identity: Mapping[str, Any],
    navmesh_path: Path,
    output_root: Path,
) -> dict:
    _require(
        isinstance(receipt, dict) and set(receipt) == _receipt_expected_fields(),
        f"bake receipt fields changed for {scene.scene_id}",
    )
    _require(
        receipt["schema_version"] == BAKE_RECEIPT_SCHEMA
        and receipt["status"] == BAKE_STATUS
        and receipt["scene_id"] == scene.scene_id
        and receipt["split_role"] == scene.split_role
        and receipt["parent_manifest_sha256"] == parent_manifest_sha256,
        f"bake receipt identity changed for {scene.scene_id}",
    )
    _require(
        receipt["source_glb"]
        == {
            "manifest_path": scene.manifest_scene["environment"]["path"],
            "bytes": scene.glb_snapshot.byte_count,
            "content_sha256": scene.glb_snapshot.content_sha256,
        },
        f"receipt GLB binding changed for {scene.scene_id}",
    )
    _require(
        receipt["superseded_manifest_navmesh_record"] == scene.original_navmesh_record,
        f"receipt superseded navmesh binding changed for {scene.scene_id}",
    )
    _require(receipt["requested_settings"] == requested_settings, "settings changed")
    _require(
        receipt["requested_settings_sha256"]
        == navmesh_settings_signature(requested_settings),
        "requested settings signature changed",
    )
    _settings_close(
        receipt["runtime_effective_settings"],
        effective_settings,
        "receipt effective settings",
    )
    _require(
        receipt["runtime_effective_settings_sha256"]
        == navmesh_settings_signature(receipt["runtime_effective_settings"]),
        "effective settings signature changed",
    )
    _assert_runtime_stable(runtime, runtime_identity)
    _require(receipt["runtime"] == runtime_identity, "receipt runtime changed")
    _require(
        receipt["bake_method"]
        == {
            "cpu_only": True,
            "create_renderer": False,
            "fresh_simulator_each_repetition": True,
            "recompute_navmesh": True,
            "save_nav_mesh": True,
            "roundtrip_load_nav_mesh": True,
        },
        "bake method changed",
    )
    determinism = receipt["determinism"]
    _require(
        isinstance(determinism, Mapping)
        and set(determinism)
        == {
            "fresh_simulator_repetitions",
            "navmesh_sha256_by_repetition",
            "byte_for_byte_equal",
        }
        and determinism["fresh_simulator_repetitions"] == REPETITIONS
        and determinism["byte_for_byte_equal"] is True
        and isinstance(determinism["navmesh_sha256_by_repetition"], list)
        and len(determinism["navmesh_sha256_by_repetition"]) == REPETITIONS
        and len(set(determinism["navmesh_sha256_by_repetition"])) == 1,
        "bake determinism receipt changed",
    )
    navmesh_snapshot = snapshot_regular_file(navmesh_path, "baked navmesh")
    relative_navmesh = navmesh_path.relative_to(output_root).as_posix()
    _require(
        receipt["output_navmesh"]
        == _stable_file_record(navmesh_snapshot, relative_navmesh),
        f"baked navmesh bytes changed for {scene.scene_id}",
    )
    _require(
        determinism["navmesh_sha256_by_repetition"]
        == [navmesh_snapshot.content_sha256] * REPETITIONS,
        "determinism hashes differ from final navmesh",
    )
    validation = runtime.validate_output(navmesh_path, requested_settings)
    _assert_runtime_stable(runtime, runtime_identity)
    _settings_close(
        validation.effective_settings,
        receipt["runtime_effective_settings"],
        "resume runtime settings",
    )
    recorded_bake = NavmeshObservation.from_dict(receipt["bake_observation"])
    recorded_roundtrip = NavmeshObservation.from_dict(receipt["roundtrip_observation"])
    _observations_close(
        validation.roundtrip_observation,
        recorded_roundtrip,
        "resume roundtrip",
    )
    _observations_close(recorded_bake, recorded_roundtrip, "recorded bake/roundtrip")
    _require(
        receipt["selection_boundary"] == SELECTION_BOUNDARY
        and receipt["determinism_boundary"] == DETERMINISM_BOUNDARY,
        "receipt contract boundary changed",
    )
    return receipt


def _bake_scene(
    *,
    scene: SceneInput,
    scenes_root: Path,
    output_root: Path,
    run_signature: str,
    parent_manifest_sha256: str,
    requested_settings: Mapping[str, Any],
    effective_settings: Mapping[str, Any],
    runtime: BakeRuntime,
    runtime_identity: Mapping[str, Any],
) -> tuple[Path, Path, str]:
    final_directory = scenes_root / scene.output_key
    navmesh_path = final_directory / "scene.navmesh"
    receipt_path = final_directory / "bake_receipt.json"
    if os.path.lexists(final_directory):
        _require(
            final_directory.is_dir() and not final_directory.is_symlink(),
            f"scene output is not a physical directory: {final_directory}",
        )
        _require(
            {path.name for path in final_directory.iterdir()}
            == {"scene.navmesh", "bake_receipt.json"},
            f"scene output file set changed: {final_directory}",
        )
        receipt, raw = _load_canonical(receipt_path, "bake receipt")
        _validate_receipt(
            receipt,
            scene=scene,
            parent_manifest_sha256=parent_manifest_sha256,
            requested_settings=requested_settings,
            effective_settings=effective_settings,
            runtime=runtime,
            runtime_identity=runtime_identity,
            navmesh_path=navmesh_path,
            output_root=output_root,
        )
        return navmesh_path, receipt_path, sha256_bytes(raw)

    staging = scenes_root / _scene_staging_name(run_signature, scene)
    _require(
        not os.path.lexists(staging), f"scene staging path already exists: {staging}"
    )
    staging.mkdir()
    try:
        _atomic_write(
            staging / "staging_contract.json",
            canonical_json_bytes(_stage_marker(run_signature, scene)),
        )
        results = []
        repeat_paths = []
        for repetition in range(REPETITIONS):
            repeat_path = staging / f"repeat-{repetition}.navmesh"
            _assert_runtime_stable(runtime, runtime_identity)
            results.append(
                runtime.bake_once(
                    scene.glb_path,
                    repeat_path,
                    requested_settings,
                )
            )
            _assert_runtime_stable(runtime, runtime_identity)
            repeat_paths.append(repeat_path)
        snapshots = [
            snapshot_regular_file(path, f"{scene.scene_id} repetition {index}")
            for index, path in enumerate(repeat_paths)
        ]
        _require(
            len({snapshot.content_sha256 for snapshot in snapshots}) == 1
            and len({snapshot.byte_count for snapshot in snapshots}) == 1
            and _files_byte_equal(repeat_paths[0], repeat_paths[1]),
            f"Habitat navmesh serialization is not repeatable for {scene.scene_id}",
        )
        for result in results[1:]:
            _settings_close(
                result.effective_settings,
                results[0].effective_settings,
                "repeated bake settings",
            )
            _observations_close(
                result.bake_observation,
                results[0].bake_observation,
                "repeated bake observation",
            )
            _observations_close(
                result.roundtrip_observation,
                results[0].roundtrip_observation,
                "repeated roundtrip observation",
            )
        _require(
            snapshot_regular_file(scene.glb_path, "GLB post-bake")
            == scene.glb_snapshot,
            f"source GLB drifted while baking {scene.scene_id}",
        )
        final_staged_navmesh = staging / "scene.navmesh"
        os.replace(repeat_paths[0], final_staged_navmesh)
        repeat_paths[1].unlink()
        final_snapshot = snapshot_regular_file(
            final_staged_navmesh, f"{scene.scene_id} final staged navmesh"
        )
        final_relative = (
            Path(SCENES_DIRECTORY) / scene.output_key / "scene.navmesh"
        ).as_posix()
        receipt = {
            "schema_version": BAKE_RECEIPT_SCHEMA,
            "status": BAKE_STATUS,
            "scene_id": scene.scene_id,
            "split_role": scene.split_role,
            "parent_manifest_sha256": parent_manifest_sha256,
            "source_glb": {
                "manifest_path": scene.manifest_scene["environment"]["path"],
                "bytes": scene.glb_snapshot.byte_count,
                "content_sha256": scene.glb_snapshot.content_sha256,
            },
            "superseded_manifest_navmesh_record": copy.deepcopy(
                scene.original_navmesh_record
            ),
            "requested_settings": copy.deepcopy(requested_settings),
            "requested_settings_sha256": navmesh_settings_signature(requested_settings),
            "runtime_effective_settings": copy.deepcopy(results[0].effective_settings),
            "runtime_effective_settings_sha256": navmesh_settings_signature(
                results[0].effective_settings
            ),
            "runtime": copy.deepcopy(runtime_identity),
            "bake_method": {
                "cpu_only": True,
                "create_renderer": False,
                "fresh_simulator_each_repetition": True,
                "recompute_navmesh": True,
                "save_nav_mesh": True,
                "roundtrip_load_nav_mesh": True,
            },
            "determinism": {
                "fresh_simulator_repetitions": REPETITIONS,
                "navmesh_sha256_by_repetition": [
                    snapshot.content_sha256 for snapshot in snapshots
                ],
                "byte_for_byte_equal": True,
            },
            "bake_observation": results[0].bake_observation.to_dict(),
            "roundtrip_observation": results[0].roundtrip_observation.to_dict(),
            "output_navmesh": _stable_file_record(final_snapshot, final_relative),
            "selection_boundary": SELECTION_BOUNDARY,
            "determinism_boundary": DETERMINISM_BOUNDARY,
        }
        receipt_bytes = canonical_json_bytes(receipt)
        _atomic_write(staging / "bake_receipt.json", receipt_bytes)
        (staging / "staging_contract.json").unlink()
        _fsync_directory(staging)
        _require(
            not os.path.lexists(final_directory), "scene output appeared during bake"
        )
        os.rename(staging, final_directory)
        staging = Path()
        _fsync_directory(scenes_root)
        _validate_receipt(
            receipt,
            scene=scene,
            parent_manifest_sha256=parent_manifest_sha256,
            requested_settings=requested_settings,
            effective_settings=effective_settings,
            runtime=runtime,
            runtime_identity=runtime_identity,
            navmesh_path=navmesh_path,
            output_root=output_root,
        )
        return navmesh_path, receipt_path, sha256_bytes(receipt_bytes)
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


def _published_files(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    settings_sha256: str,
    requested_settings: Mapping[str, Any],
    effective_settings: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    scene_inputs: Sequence[SceneInput],
    scene_outputs: Mapping[str, tuple[Path, Path, str]],
    output_root: Path,
) -> dict[str, bytes]:
    producer = snapshot_regular_file(Path(__file__), "navmesh baker source")
    index_scenes = {}
    for scene in sorted(scene_inputs, key=lambda item: item.scene_id):
        navmesh_path, receipt_path, receipt_sha = scene_outputs[scene.scene_id]
        navmesh_snapshot = snapshot_regular_file(navmesh_path, "published navmesh")
        receipt_snapshot = snapshot_regular_file(receipt_path, "published receipt")
        _require(
            receipt_snapshot.content_sha256 == receipt_sha,
            f"receipt changed before publication: {scene.scene_id}",
        )
        index_scenes[scene.scene_id] = {
            "source_glb_sha256": scene.glb_snapshot.content_sha256,
            "navmesh": _stable_file_record(
                navmesh_snapshot, navmesh_path.relative_to(output_root).as_posix()
            ),
            "bake_receipt": _stable_file_record(
                receipt_snapshot, receipt_path.relative_to(output_root).as_posix()
            ),
        }
    index = {
        "schema_version": BAKE_INDEX_SCHEMA,
        "status": BAKE_STATUS,
        "parent_manifest_sha256": manifest_sha256,
        "settings_sha256": settings_sha256,
        "requested_settings_sha256": navmesh_settings_signature(requested_settings),
        "runtime_effective_settings_sha256": navmesh_settings_signature(
            effective_settings
        ),
        "runtime": copy.deepcopy(runtime_identity),
        "fresh_simulator_repetitions": REPETITIONS,
        "scene_count": len(scene_inputs),
        "scenes": index_scenes,
        "selection_boundary": SELECTION_BOUNDARY,
        "determinism_boundary": DETERMINISM_BOUNDARY,
        "producer": {
            "path": Path(__file__).name,
            "bytes": producer.byte_count,
            "content_sha256": producer.content_sha256,
        },
    }
    index_bytes = canonical_json_bytes(index)
    index_sha = sha256_bytes(index_bytes)

    derived = copy.deepcopy(manifest)
    _require(
        "geometry_bake_derivation" not in derived,
        "input manifest is already geometry-derived",
    )
    roots = derived.get("input_roots")
    _require(isinstance(roots, dict), "derived manifest input_roots is invalid")
    roots["navmesh_root"] = str((output_root / SCENES_DIRECTORY).resolve())
    roots["geometry_bake_root"] = str(output_root.resolve())
    by_scene = {scene.scene_id: scene for scene in scene_inputs}
    for raw_scene in derived["scenes"]:
        scene = by_scene[raw_scene["scene"]]
        navmesh_path, receipt_path, _receipt_sha = scene_outputs[scene.scene_id]
        navmesh_snapshot = snapshot_regular_file(navmesh_path, "derived navmesh")
        receipt_snapshot = snapshot_regular_file(receipt_path, "derived receipt")
        raw_scene["navmesh"] = _stable_file_record(
            navmesh_snapshot,
            navmesh_path.relative_to(output_root / SCENES_DIRECTORY).as_posix(),
        )
        raw_scene["geometry_bake_receipt"] = _stable_file_record(
            receipt_snapshot, receipt_path.relative_to(output_root).as_posix()
        )
    derived["geometry_bake_derivation"] = {
        "schema_version": DERIVATION_SCHEMA,
        "status": BAKE_STATUS,
        "parent_manifest_sha256": manifest_sha256,
        "settings_file_sha256": settings_sha256,
        "requested_settings_sha256": navmesh_settings_signature(requested_settings),
        "runtime_effective_settings_sha256": navmesh_settings_signature(
            effective_settings
        ),
        "runtime": copy.deepcopy(runtime_identity),
        "run_contract": _stable_file_record(
            snapshot_regular_file(
                output_root / RUN_CONTRACT_FILE, "published run contract"
            ),
            RUN_CONTRACT_FILE,
        ),
        "bake_index": _byte_file_record(
            index_bytes, f"{PUBLISHED_DIRECTORY}/{BAKE_INDEX_FILE}"
        ),
        "selection_boundary": SELECTION_BOUNDARY,
        "determinism_boundary": DETERMINISM_BOUNDARY,
    }
    manifest_bytes = canonical_json_bytes(derived)
    manifest_output_sha = sha256_bytes(manifest_bytes)
    return {
        BAKE_INDEX_FILE: index_bytes,
        f"{BAKE_INDEX_FILE}.sha256": (
            f"{index_sha}  {BAKE_INDEX_FILE}\n".encode("ascii")
        ),
        DERIVED_MANIFEST_FILE: manifest_bytes,
        f"{DERIVED_MANIFEST_FILE}.sha256": (
            f"{manifest_output_sha}  {DERIVED_MANIFEST_FILE}\n".encode("ascii")
        ),
    }


def _verify_file_set(
    directory: Path, expected: Mapping[str, bytes], label: str
) -> None:
    _require(directory.is_dir() and not directory.is_symlink(), f"{label} is invalid")
    entries = list(directory.iterdir())
    _require(
        all(path.is_file() and not path.is_symlink() for path in entries),
        f"{label} contains a non-physical file",
    )
    _require(
        {path.name for path in entries} == set(expected), f"{label} file set changed"
    )
    for name, payload in expected.items():
        snapshot = snapshot_regular_file(directory / name, f"{label}/{name}")
        _require(
            snapshot.byte_count == len(payload)
            and snapshot.content_sha256 == sha256_bytes(payload),
            f"{label}/{name} changed",
        )


def _publish_staging_name(expected: Mapping[str, bytes]) -> str:
    signature_payload = {
        name: {"bytes": len(payload), "content_sha256": sha256_bytes(payload)}
        for name, payload in sorted(expected.items())
    }
    return f"{PUBLISH_STAGING_PREFIX}{sha256_bytes(canonical_json_bytes(signature_payload))}"


def _clean_stale_publish_staging(
    output_root: Path,
    expected_staging_name: str,
) -> None:
    for path in output_root.iterdir():
        if not path.name.startswith(PUBLISH_STAGING_PREFIX):
            continue
        _require(
            path.name == expected_staging_name,
            f"published staging ownership mismatch: {path}",
        )
        _require(
            path.is_dir() and not path.is_symlink(),
            f"unsafe published staging path: {path}",
        )
        shutil.rmtree(path)


def _publish(output_root: Path, expected: Mapping[str, bytes]) -> str:
    final = output_root / PUBLISHED_DIRECTORY
    staging_name = _publish_staging_name(expected)
    _clean_stale_publish_staging(output_root, staging_name)
    if os.path.lexists(final):
        _verify_file_set(final, expected, "published bake artifacts")
        return "resumed"
    staging = output_root / staging_name
    staging.mkdir()
    try:
        for name, payload in expected.items():
            _atomic_write(staging / name, payload)
        _fsync_directory(staging)
        _verify_file_set(staging, expected, "staged published artifacts")
        _require(not os.path.lexists(final), "published output appeared concurrently")
        os.rename(staging, final)
        staging = Path()
        _fsync_directory(output_root)
        _verify_file_set(final, expected, "published bake artifacts")
        return "written"
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


def bake_geometry_bundle(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    settings_path: Path | str,
    expected_settings_sha256: str,
    expected_habitat_version: str,
    expected_habitat_bindings_sha256: str,
    output_root: Path | str,
    runtime: BakeRuntime,
    environment_root_override: Path | str | None = None,
    resume: bool = False,
) -> dict[str, object]:
    _require(
        expected_habitat_version == EXPECTED_HABITAT_VERSION,
        f"formal bake requires Habitat-Sim {EXPECTED_HABITAT_VERSION}",
    )
    _valid_sha256(expected_habitat_bindings_sha256, "expected Habitat bindings SHA256")
    runtime_identity = _validated_runtime_identity(
        runtime,
        expected_version=expected_habitat_version,
        expected_bindings_sha256=expected_habitat_bindings_sha256,
    )
    try:
        manifest, manifest_snapshot = load_pinned_manifest(
            manifest_path, expected_manifest_sha256
        )
        requested_settings, settings_snapshot = load_pinned_settings(
            settings_path, expected_settings_sha256
        )
    except GeometryMapBuildError as error:
        raise NavmeshBakeError(str(error)) from error
    _require(
        "geometry_bake_derivation" not in manifest,
        "input manifest is already geometry-derived",
    )
    _require(
        math.isclose(
            float(requested_settings["agent_radius"]),
            REQUIRED_AGENT_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(requested_settings["agent_height"]),
            REQUIRED_AGENT_HEIGHT_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "formal settings must use agent radius=0.30 m and height=1.50 m",
    )
    effective_settings = canonical_navmesh_settings(
        runtime.assert_settings_contract(requested_settings)
    )
    _assert_runtime_stable(runtime, runtime_identity)
    _settings_close(effective_settings, requested_settings, "runtime settings contract")
    environment_root = _resolve_environment_root(manifest, environment_root_override)
    scene_inputs = _scene_inputs(manifest, environment_root)
    output = Path(output_root).absolute()
    if os.path.lexists(output):
        _require(resume, f"output exists; --resume is required: {output}")
        _require(output.is_dir() and not output.is_symlink(), "output root is invalid")
    else:
        _require(not resume, "--resume requires an existing output root")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()

    contract = _run_contract(
        source_manifest_path=Path(manifest_path),
        source_manifest_sha256=manifest_snapshot.content_sha256,
        source_manifest_schema=str(manifest["schema_version"]),
        settings_path=Path(settings_path),
        settings_sha256=settings_snapshot.content_sha256,
        requested_settings=requested_settings,
        effective_settings=effective_settings,
        environment_root=environment_root,
        runtime_identity=runtime_identity,
        scene_inputs=scene_inputs,
    )
    contract_bytes = canonical_json_bytes(contract)
    contract_sha = sha256_bytes(contract_bytes)
    contract_path = output / RUN_CONTRACT_FILE
    if os.path.lexists(contract_path):
        existing, raw = _load_canonical(contract_path, "run contract")
        _require(
            existing == contract and raw == contract_bytes,
            "resume run contract differs from current pinned inputs/runtime",
        )
    else:
        _require(not resume, "resume output lacks run contract")
        _atomic_write(contract_path, contract_bytes)

    lock_path = output / ".bake.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise NavmeshBakeError(
                "another bake process owns the output lock"
            ) from error
        scenes_root = output / SCENES_DIRECTORY
        scenes_root.mkdir(exist_ok=True)
        _require(
            scenes_root.is_dir() and not scenes_root.is_symlink(),
            "scenes output root is invalid",
        )
        scene_by_key = {scene.output_key: scene for scene in scene_inputs}
        _clean_stale_staging(scenes_root, contract_sha, scene_by_key)
        allowed_scene_entries = set(scene_by_key)
        actual_final_entries = {
            path.name
            for path in scenes_root.iterdir()
            if not path.name.startswith(SCENE_STAGING_PREFIX)
        }
        _require(
            actual_final_entries <= allowed_scene_entries,
            "scenes output root contains an unexpected entry",
        )
        outputs = {}
        resumed_scenes = 0
        baked_scenes = 0
        for scene in sorted(scene_inputs, key=lambda item: item.scene_id):
            existed = os.path.lexists(scenes_root / scene.output_key)
            outputs[scene.scene_id] = _bake_scene(
                scene=scene,
                scenes_root=scenes_root,
                output_root=output,
                run_signature=contract_sha,
                parent_manifest_sha256=manifest_snapshot.content_sha256,
                requested_settings=requested_settings,
                effective_settings=effective_settings,
                runtime=runtime,
                runtime_identity=runtime_identity,
            )
            if existed:
                resumed_scenes += 1
            else:
                baked_scenes += 1
        _require(
            snapshot_regular_file(manifest_path, "manifest post-bake")
            == manifest_snapshot,
            "source manifest drifted during bake",
        )
        _require(
            snapshot_regular_file(settings_path, "settings post-bake")
            == settings_snapshot,
            "settings file drifted during bake",
        )
        for scene in scene_inputs:
            _require(
                snapshot_regular_file(scene.glb_path, "GLB final recheck")
                == scene.glb_snapshot,
                f"source GLB drifted during run: {scene.scene_id}",
            )
        published = _published_files(
            manifest=manifest,
            manifest_sha256=manifest_snapshot.content_sha256,
            settings_sha256=settings_snapshot.content_sha256,
            requested_settings=requested_settings,
            effective_settings=effective_settings,
            runtime_identity=runtime_identity,
            scene_inputs=scene_inputs,
            scene_outputs=outputs,
            output_root=output,
        )
        publish_status = _publish(output, published)
        derived_bytes = published[DERIVED_MANIFEST_FILE]
        return {
            "status": publish_status,
            "output_root": str(output),
            "derived_manifest": str(
                output / PUBLISHED_DIRECTORY / DERIVED_MANIFEST_FILE
            ),
            "derived_manifest_sha256": sha256_bytes(derived_bytes),
            "bake_index": str(output / PUBLISHED_DIRECTORY / BAKE_INDEX_FILE),
            "scene_count": len(scene_inputs),
            "baked_scene_count": baked_scenes,
            "resumed_scene_count": resumed_scenes,
            "source_manifest_sha256": manifest_snapshot.content_sha256,
            "settings_sha256": settings_snapshot.content_sha256,
            "runtime_fingerprint_sha256": runtime_identity[
                "runtime_fingerprint_sha256"
            ],
            "determinism_boundary": DETERMINISM_BOUNDARY,
        }
    finally:
        os.close(lock_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only, receipt-backed Habitat 0.3.1 NavMesh bake stage.",
        epilog=(
            "This proves same-runtime repeatability for two fresh bakes; it does "
            "not assert universal Habitat serialization determinism."
        ),
    )
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--settings-json",
        type=Path,
        default=Path(__file__).with_name(SETTINGS_FILE),
    )
    parser.add_argument("--expected-settings-sha256", required=True)
    parser.add_argument(
        "--expected-habitat-version",
        required=True,
        choices=[EXPECTED_HABITAT_VERSION],
    )
    parser.add_argument("--expected-habitat-bindings-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--environment-root-override",
        type=Path,
        help="explicit relocation of only the manifest environment_root",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        import habitat_sim
    except ImportError as error:
        raise NavmeshBakeError("Habitat-Sim is required for the real bake") from error
    runtime = HabitatBakeRuntime(
        habitat_sim,
        expected_version=args.expected_habitat_version,
        expected_bindings_sha256=args.expected_habitat_bindings_sha256,
    )
    result = bake_geometry_bundle(
        manifest_path=args.expert_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        settings_path=args.settings_json,
        expected_settings_sha256=args.expected_settings_sha256,
        expected_habitat_version=args.expected_habitat_version,
        expected_habitat_bindings_sha256=args.expected_habitat_bindings_sha256,
        output_root=args.output_root,
        runtime=runtime,
        environment_root_override=args.environment_root_override,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
