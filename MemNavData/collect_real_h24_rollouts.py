#!/usr/bin/env python3
"""Strict sharded collector for real same-state NavDP/Habitat H24 labels.

Inputs are all externally content-pinned:

* a canonical ``novel_candidate_set_v2`` JSON or JSONL file;
* the canonical expert-state manifest and its expected SHA256;
* a canonical scene-to-``FrozenGeometryIdentity`` map and expected SHA256;
* a canonical exact NavDP audit-server provenance map and expected SHA256.

The collector never modifies candidate-set rows.  It emits one protocol-v3
rollout artifact (plus SHA sidecar) and one lossless plan-diagnostics artifact
per decision.  A later, explicit merge stage may copy verified rollout labels
into training records.  This separation permits a pre-collection candidate row
whose rollout labels are all strictly neutral/invalid without pretending those
placeholder labels are observations.

``--dry-run`` performs complete static validation and shard selection without
opening Habitat or contacting NavDP.  ``--preflight-only`` additionally opens
the first selected frozen scene, resets NavDP, and replays its exact preceding
FIFO without a diffusion call.  A normal run performs the same checks as part
of collection.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from MemNavData.habitat_rollout_primitives import FrozenGeometryIdentity
    from MemNavData.novel_candidate_set_schema_v2 import (
        CandidateSetValidationError,
        validate_candidate_dataset,
        validate_candidate_set,
    )
    from MemNavData.novel_rollout_protocol_v2 import (
        PairedRolloutArtifact,
        RolloutProtocolError,
        atomic_write_artifact,
        canonical_sha256,
        collect_paired_rollouts,
        load_artifact,
    )
    from MemNavData.real_h24_rollout_backend import (
        FrozenStateAssets,
        BACKEND_PROTOCOL,
        PinnedHabitatRuntime,
        PurePursuitConfig,
        RealH24RolloutBackend,
        RequestsJsonTransport,
        candidate_arms_from_feature_record,
        load_state_assets_from_manifest,
        sha256_bytes,
    )
except ImportError:  # direct script execution with pinned PYTHONPATH
    from habitat_rollout_primitives import FrozenGeometryIdentity  # type: ignore
    from novel_candidate_set_schema_v2 import (  # type: ignore
        CandidateSetValidationError,
        validate_candidate_dataset,
        validate_candidate_set,
    )
    from novel_rollout_protocol_v2 import (  # type: ignore
        PairedRolloutArtifact,
        RolloutProtocolError,
        atomic_write_artifact,
        canonical_sha256,
        collect_paired_rollouts,
        load_artifact,
    )
    from real_h24_rollout_backend import (  # type: ignore
        FrozenStateAssets,
        BACKEND_PROTOCOL,
        PinnedHabitatRuntime,
        PurePursuitConfig,
        RealH24RolloutBackend,
        RequestsJsonTransport,
        candidate_arms_from_feature_record,
        load_state_assets_from_manifest,
        sha256_bytes,
    )


COLLECTOR_SCHEMA = "real_h24_collector_v3"
GEOMETRY_MAP_SCHEMA = "frozen_geometry_map_v1"
PLAN_DIAGNOSTICS_SCHEMA = "real_h24_plan_diagnostics_v3"
SERVER_INSTANCE_SCHEMA = "real_h24_server_instance_v1"


class CollectorError(RuntimeError):
    """A collection input, receipt, or resume artifact failed validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def _valid_sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def canonical_bytes(value: object) -> bytes:
    try:
        return (json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CollectorError(f"value is not canonical JSON: {error}") from error


def pretty_canonical_bytes(value: object) -> bytes:
    try:
        return (json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CollectorError(f"value is not JSON-compatible: {error}") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                CollectorError(f"{label} contains non-finite {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollectorError(f"{label} is invalid JSON: {error}") from error


def load_canonical_json(
    path: Path | str,
    expected_sha256: str,
    *,
    pretty: bool = False,
) -> object:
    source = Path(path)
    expected = _valid_sha(expected_sha256, f"{source.name} expected SHA256")
    _require(source.is_file(), f"JSON input is missing: {source}")
    raw = source.read_bytes()
    _require(sha256_bytes(raw) == expected, f"JSON SHA mismatch: {source}")
    value = _decode_json(raw, str(source))
    expected_raw = pretty_canonical_bytes(value) if pretty else canonical_bytes(value)
    _require(raw == expected_raw, f"JSON input is not canonical: {source}")
    return value


def load_candidate_records(
    path: Path | str,
    expected_sha256: str,
) -> tuple[list[Mapping[str, Any]], bool]:
    """Load canonical JSON/JSONL and validate deployment fields and labels.

    Returns ``(records, has_precollection_neutral_rows)``.
    """
    source = Path(path)
    expected = _valid_sha(expected_sha256, "candidate-set expected SHA256")
    _require(source.is_file(), f"candidate-set input is missing: {source}")
    raw = source.read_bytes()
    _require(sha256_bytes(raw) == expected, "candidate-set SHA256 mismatch")
    records: list[Mapping[str, Any]] = []
    if source.suffix.lower() == ".jsonl":
        _require(raw.endswith(b"\n"), "canonical JSONL must end in one newline")
        lines = raw.splitlines(keepends=True)
        _require(bool(lines), "candidate JSONL is empty")
        for index, line in enumerate(lines):
            _require(line.strip(), f"candidate JSONL line {index + 1} is blank")
            value = _decode_json(line, f"candidate JSONL line {index + 1}")
            _require(
                line == canonical_bytes(value),
                f"candidate JSONL line {index + 1} is not canonical",
            )
            _require(isinstance(value, Mapping), "candidate JSONL rows must be objects")
            records.append(value)
    else:
        value = _decode_json(raw, "candidate JSON")
        _require(raw == canonical_bytes(value), "candidate JSON is not canonical")
        if isinstance(value, Mapping):
            records = [value]
        else:
            _require(
                isinstance(value, list)
                and all(isinstance(record, Mapping) for record in value),
                "candidate JSON must be one object or an array of objects",
            )
            records = list(value)
    _require(bool(records), "candidate-set input contains no records")

    validation_records = []
    precollection = False
    for index, record in enumerate(records):
        try:
            validate_candidate_set(record)
            validation_records.append(record)
            continue
        except CandidateSetValidationError as original_error:
            if not _is_strict_neutral_precollection(record):
                raise CollectorError(
                    f"candidate record {index} is invalid: {original_error}") from original_error
            patched = _validation_copy_for_neutral_precollection(record)
            try:
                validate_candidate_set(patched)
            except CandidateSetValidationError as patched_error:
                raise CollectorError(
                    f"candidate record {index} neutral precollection schema is invalid: "
                    f"{patched_error}") from patched_error
            validation_records.append(patched)
            precollection = True
    try:
        validate_candidate_dataset(validation_records)
    except CandidateSetValidationError as error:
        raise CollectorError(f"candidate dataset is invalid: {error}") from error
    return records, precollection


_ROLLOUT_ZERO_FLOATS = (
    "geodesic_progress_h8_m",
    "geodesic_progress_h24_m",
    "advantage_h24_m",
)
_ROLLOUT_FALSE_BOOLS = (
    "harm",
    "useful",
    "reachable",
    "collision_h8",
    "regression_h24",
    "rollout_label_valid",
)


def _is_strict_neutral_precollection(record: Mapping[str, Any]) -> bool:
    candidates = record.get("candidates")
    labels = record.get("set_labels")
    if not isinstance(candidates, list) or not isinstance(labels, Mapping):
        return False
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return False
        row = candidate.get("labels")
        if not isinstance(row, Mapping):
            return False
        try:
            if any(float(row.get(key)) != 0.0 for key in _ROLLOUT_ZERO_FLOATS):
                return False
        except (TypeError, ValueError):
            return False
        if any(row.get(key) is not False for key in _ROLLOUT_FALSE_BOOLS):
            return False
    return bool(
        labels.get("candidate_set_has_positive") is False
        and labels.get("candidate_universe_has_positive") is False
        and labels.get("candidate_coverage_miss") is False
        and labels.get("coverage_label_valid") is False
        and labels.get("oracle_best_candidate_id") == "dustbin"
    )


def _validation_copy_for_neutral_precollection(
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    patched = copy.deepcopy(record)
    native = patched["candidates"][0]
    native_labels = native["labels"]
    native_labels["rollout_label_valid"] = True
    native_labels["reachable"] = True
    # Zero progress is a schema-only stand-in used solely to exercise all
    # deployment feature/provenance validation.  The original record remains
    # unchanged and is never written as a rollout observation.
    return patched


@dataclass(frozen=True)
class GeometryMapEntry:
    scene_id: str
    identity_path: Path
    identity_sha256: str
    identity: FrozenGeometryIdentity


def load_geometry_map(
    path: Path | str,
    expected_sha256: str,
) -> dict[str, GeometryMapEntry]:
    source = Path(path).resolve()
    value = load_canonical_json(source, expected_sha256)
    _require(isinstance(value, Mapping), "geometry map must be an object")
    _require(
        set(value) == {"schema_version", "scenes"}
        and value["schema_version"] == GEOMETRY_MAP_SCHEMA,
        "geometry map schema/fields changed",
    )
    scenes = value["scenes"]
    _require(isinstance(scenes, Mapping) and scenes, "geometry map scenes are empty")
    result = {}
    for scene_id, raw_entry in scenes.items():
        _require(isinstance(scene_id, str) and scene_id, "geometry map scene id is invalid")
        _require(
            isinstance(raw_entry, Mapping)
            and set(raw_entry) == {"identity_path", "identity_sha256"},
            f"geometry map entry changed for {scene_id}",
        )
        relative = raw_entry["identity_path"]
        _require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute(),
            f"geometry identity path must be relative for {scene_id}",
        )
        identity_path = (source.parent / relative).resolve()
        try:
            identity_path.relative_to(source.parent)
        except ValueError as error:
            raise CollectorError(
                f"geometry identity escapes map directory: {scene_id}") from error
        expected_identity_sha = _valid_sha(
            raw_entry["identity_sha256"],
            f"geometry identity SHA for {scene_id}",
        )
        identity = FrozenGeometryIdentity.load_json(identity_path)
        _require(
            identity.identity_sha256 == expected_identity_sha,
            f"geometry identity content mismatch for {scene_id}",
        )
        result[scene_id] = GeometryMapEntry(
            scene_id, identity_path, expected_identity_sha, identity)
    return result


def load_server_provenance(
    path: Path | str,
    expected_sha256: str,
) -> dict[str, str]:
    value = load_canonical_json(path, expected_sha256)
    _require(isinstance(value, Mapping) and value, "server provenance must be an object")
    result = {}
    for key, digest in value.items():
        _require(isinstance(key, str) and key, "server provenance key is invalid")
        result[key] = _valid_sha(digest, f"server provenance {key}")
    required = {
        "navdp_server_sha256",
        "policy_agent_sha256",
        "deterministic_seed_sha256",
        "checkpoint_sha256",
        "wrapper_sha256",
    }
    _require(set(result) == required, "server provenance fields changed")
    return result


def verify_candidate_source_policy(
    records: Sequence[Mapping[str, Any]],
    server_provenance: Mapping[str, str],
) -> None:
    """Reject a policy/checkpoint mismatch before opening Habitat or NavDP."""
    expected = server_provenance.get("checkpoint_sha256")
    _valid_sha(expected, "server checkpoint SHA")
    for index, record in enumerate(records):
        provenance = record.get("provenance")
        _require(
            isinstance(provenance, Mapping),
            f"candidate record {index} has no provenance",
        )
        actual = provenance.get("source_policy_sha256")
        _valid_sha(actual, f"candidate record {index} source policy SHA")
        _require(
            actual == expected,
            f"candidate record {index} source policy differs from server checkpoint",
        )


def parse_root_overrides(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    allowed = {"episode_root", "environment_root", "navmesh_root"}
    for raw in values:
        _require("=" in raw, f"root override lacks '=': {raw!r}")
        name, path_text = raw.split("=", 1)
        _require(name in allowed, f"unsupported root override {name!r}")
        _require(name not in result, f"duplicate root override {name!r}")
        path = Path(path_text).resolve()
        _require(path.is_dir(), f"root override is not a directory: {path}")
        result[name] = path
    return result


def assert_pythonpath(repo_root: Path | str | None = None) -> Path:
    expected = (
        Path(__file__).resolve().parents[1]
        if repo_root is None else Path(repo_root).resolve()
    )
    resolved_entries = set()
    for entry in sys.path:
        try:
            resolved_entries.add(Path(entry or os.getcwd()).resolve())
        except OSError:
            continue
    _require(
        expected in resolved_entries,
        "repository root is absent from PYTHONPATH/sys.path; refusing an "
        "ambiguous checkout import",
    )
    imported = Path(sys.modules[__name__].__file__).resolve()
    try:
        imported.relative_to(expected)
    except ValueError as error:
        raise CollectorError("collector imported from a different checkout") from error
    return expected


def selected_shard_records(
    records: Sequence[Mapping[str, Any]],
    shard_index: int,
    shard_count: int,
) -> list[Mapping[str, Any]]:
    _require(
        isinstance(shard_count, int) and not isinstance(shard_count, bool)
        and shard_count >= 1,
        "shard_count must be positive",
    )
    _require(
        isinstance(shard_index, int) and not isinstance(shard_index, bool)
        and 0 <= shard_index < shard_count,
        "shard_index is outside [0, shard_count)",
    )
    ordered = sorted(records, key=lambda record: (
        str(record["provenance"]["state_id"]),
        str(record["provenance"]["goal_epoch"]),
    ))
    return [record for index, record in enumerate(ordered)
            if index % shard_count == shard_index]


def decision_seeds(base_seed: int, state_id: str) -> tuple[int, int, int]:
    _require(
        isinstance(base_seed, int) and not isinstance(base_seed, bool)
        and 0 <= base_seed < 2**63,
        "base_seed is invalid",
    )
    seeds = []
    for commitment in range(3):
        digest = hashlib.sha256(canonical_bytes({
            "base_seed": base_seed,
            "state_id": state_id,
            "commitment_index": commitment,
        })).digest()
        seeds.append(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
    _require(len(set(seeds)) == 3, "derived diffusion seeds collided")
    return tuple(seeds)  # type: ignore[return-value]


def reset_seed(base_seed: int, state_id: str) -> int:
    digest = hashlib.sha256(canonical_bytes({
        "base_seed": base_seed,
        "state_id": state_id,
        "purpose": "navdp_reset",
    })).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def safe_state_stem(state_id: str) -> str:
    prefix = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in state_id
    ).strip("_")[:80]
    return f"{prefix or 'state'}-{sha256_bytes(state_id.encode())[:16]}"


def build_run_signature(
    *,
    candidate_sha256: str,
    manifest_sha256: str,
    geometry_map_sha256: str,
    server_provenance_sha256: str,
    base_seed: int,
    stop_threshold: float,
    legacy_camera_height_m: float,
    controller: PurePursuitConfig,
) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    backend_path = repo_root / "MemNavData/real_h24_rollout_backend.py"
    protocol_path = repo_root / "MemNavData/novel_rollout_protocol_v2.py"
    return sha256_bytes(canonical_bytes({
        "collector_schema": COLLECTOR_SCHEMA,
        "collector_sha256": file_sha256(Path(__file__)),
        "backend_sha256": file_sha256(backend_path),
        "protocol_sha256": file_sha256(protocol_path),
        "candidate_sha256": _valid_sha(candidate_sha256, "candidate SHA"),
        "manifest_sha256": _valid_sha(manifest_sha256, "manifest SHA"),
        "geometry_map_sha256": _valid_sha(geometry_map_sha256, "geometry map SHA"),
        "server_provenance_sha256": _valid_sha(
            server_provenance_sha256, "server provenance SHA"),
        "base_seed": base_seed,
        "stop_threshold": stop_threshold,
        "legacy_camera_height_m": legacy_camera_height_m,
        "controller": {
            "v_max_m": controller.v_max_m,
            "lookahead_m": controller.lookahead_m,
            "min_radius_m": controller.min_radius_m,
            "max_turn_deg": controller.max_turn_deg,
            "snap_tolerance_m": controller.snap_tolerance_m,
            "creep_fraction": controller.creep_fraction,
        },
    }))


def build_server_instance_diagnostics(
    server_url: str,
    server_provenance_sha256: str,
) -> dict[str, object]:
    """Describe the physical connection target without semantic binding."""
    _require(
        isinstance(server_url, str)
        and server_url.startswith(("http://", "https://")),
        "server instance URL must be HTTP(S)",
    )
    normalized = server_url.rstrip("/")
    _require(bool(normalized), "server instance URL is empty")
    return {
        "schema_version": SERVER_INSTANCE_SCHEMA,
        "physical_server_url": normalized,
        "server_provenance_sha256": _valid_sha(
            server_provenance_sha256, "server instance provenance SHA"
        ),
        "scope": "one_state_all_arms",
        "same_state_arms_single_transport_enforced": True,
    }


def _validate_server_instance_diagnostics(value: object) -> Mapping[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {
            "schema_version",
            "physical_server_url",
            "server_provenance_sha256",
            "scope",
            "same_state_arms_single_transport_enforced",
        },
        "server instance diagnostics fields changed",
    )
    _require(
        value["schema_version"] == SERVER_INSTANCE_SCHEMA
        and value["scope"] == "one_state_all_arms"
        and value["same_state_arms_single_transport_enforced"] is True,
        "server instance diagnostics contract changed",
    )
    url = value["physical_server_url"]
    _require(
        isinstance(url, str)
        and url.startswith(("http://", "https://"))
        and url == url.rstrip("/"),
        "server instance physical URL is malformed",
    )
    _valid_sha(value["server_provenance_sha256"], "server instance provenance SHA")
    return value


def _atomic_write_pair(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = pretty_canonical_bytes(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    lock = path.with_name(path.name + ".lock")
    _require(not path.exists() and not sidecar.exists(), f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise CollectorError(f"another writer owns output lock: {lock}") from error
    os.close(lock_descriptor)
    temporary_paths = []
    try:
        for destination, content in (
            (path, encoded),
            (sidecar, f"{sha256_bytes(encoded)}  {path.name}\n".encode("ascii")),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=path.parent)
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return "written"


def load_plan_diagnostics(path: Path | str) -> Mapping[str, Any]:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    _require(source.is_file() and sidecar.is_file(), "plan diagnostics pair is incomplete")
    raw = source.read_bytes()
    value = _decode_json(raw, "plan diagnostics")
    _require(raw == pretty_canonical_bytes(value), "plan diagnostics are noncanonical")
    expected_sidecar = f"{sha256_bytes(raw)}  {source.name}\n".encode("ascii")
    _require(sidecar.read_bytes() == expected_sidecar, "plan diagnostics sidecar mismatch")
    _require(isinstance(value, Mapping), "plan diagnostics must be an object")
    _require(
        set(value) == {
            "schema_version", "artifact_sha256", "run_signature_sha256",
            "state_id", "diffusion_seeds", "server_instance", "by_candidate",
        }
        and value["schema_version"] == PLAN_DIAGNOSTICS_SCHEMA,
        "plan diagnostics schema/fields changed",
    )
    _valid_sha(value["artifact_sha256"], "diagnostics artifact SHA")
    _valid_sha(value["run_signature_sha256"], "diagnostics run signature")
    _require(isinstance(value["state_id"], str), "diagnostics state id is invalid")
    _validate_server_instance_diagnostics(value["server_instance"])
    _require(
        isinstance(value["diffusion_seeds"], list)
        and len(value["diffusion_seeds"]) == 3
        and all(isinstance(seed, int) and not isinstance(seed, bool)
                for seed in value["diffusion_seeds"]),
        "diagnostics diffusion seeds are invalid",
    )
    _require(isinstance(value["by_candidate"], Mapping), "diagnostics candidates invalid")
    for candidate_id, plans in value["by_candidate"].items():
        _require(isinstance(candidate_id, str) and candidate_id, "bad diagnostic candidate id")
        _require(isinstance(plans, Mapping) and len(plans) == 3, "candidate needs three plans")
        for plan_sha, row in plans.items():
            _valid_sha(plan_sha, "diagnostic plan SHA")
            _require(
                isinstance(row, Mapping)
                and set(row) == {
                    "plan_sha256", "server_selected_trajectory_index",
                    "raw_selected_trajectory", "executable_trajectory",
                    "all_trajectory", "all_values", "critic_max",
                    "stop_threshold", "low_critic_fallback_applied",
                    "behaviorally_identical_xy",
                    "server_receipt_sha256",
                },
                "diagnostic plan schema/fields changed",
            )
            _require(row.get("plan_sha256") == plan_sha, "diagnostic plan hash mismatch")
            _valid_sha(row["server_receipt_sha256"], "server receipt SHA")
            selected_index = row["server_selected_trajectory_index"]
            _require(
                selected_index is None
                or (isinstance(selected_index, int)
                    and not isinstance(selected_index, bool)
                    and selected_index >= 0),
                "server selected trajectory index is invalid",
            )
            raw_selected = _trajectory_array(
                row["raw_selected_trajectory"], "raw selected trajectory")
            executable = _trajectory_array(
                row["executable_trajectory"], "executable trajectory")
            all_trajectory = np.asarray(
                _finite_numeric_tree(
                    row["all_trajectory"], "all trajectory"),
                dtype=np.float64,
            )
            all_values = np.asarray(
                _finite_numeric_tree(row["all_values"], "all values"),
                dtype=np.float64,
            )
            _require(
                all_trajectory.ndim == 4
                and all_trajectory.shape[0] == 1
                and all_trajectory.shape[1] >= 1
                and all_trajectory.shape[-2:] == (24, 3)
                and all_values.shape == all_trajectory.shape[:2],
                "candidate trajectory/value tensors are malformed",
            )
            critic_max = row["critic_max"]
            stop_threshold = row["stop_threshold"]
            fallback = row["low_critic_fallback_applied"]
            behaviorally_identical = row["behaviorally_identical_xy"]
            _require(
                isinstance(critic_max, (int, float))
                and not isinstance(critic_max, bool)
                and math.isfinite(float(critic_max))
                and float(critic_max) == float(all_values.max())
                and isinstance(stop_threshold, (int, float))
                and not isinstance(stop_threshold, bool)
                and math.isfinite(float(stop_threshold))
                and isinstance(fallback, bool)
                and fallback
                == (float(critic_max) < float(stop_threshold))
                and isinstance(behaviorally_identical, bool),
                "critic/fallback/behavior metadata are invalid",
            )
            if fallback:
                _require(
                    bool(np.all(executable[:, 0] == 0.0))
                    and bool(np.all(executable[:, 1] == executable[0, 1]))
                    and float(executable[0, 1]) in (-1.0, 0.0, 1.0)
                    and bool(np.array_equal(
                        executable[:, 2], raw_selected[:, 2])),
                    "low-critic executable trajectory is malformed",
                )
            else:
                _require(
                    bool(np.array_equal(executable, raw_selected)),
                    "non-fallback executable differs from raw trajectory",
                )
    return value


def _finite_numeric_tree(value: object, label: str) -> object:
    if isinstance(value, bool):
        raise CollectorError(f"{label} contains a boolean")
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{label} contains NaN or infinity")
        return value
    _require(isinstance(value, list) and bool(value), f"{label} must be a non-empty numeric tree")
    return [
        _finite_numeric_tree(child, f"{label}[{index}]")
        for index, child in enumerate(value)
    ]


def _trajectory_array(value: object, label: str) -> np.ndarray:
    result = np.asarray(_finite_numeric_tree(value, label), dtype=np.float64)
    _require(result.shape == (24, 3), f"{label} must have shape (24, 3)")
    return result


def build_plan_diagnostics(
    artifact: PairedRolloutArtifact,
    backends: Mapping[str, RealH24RolloutBackend],
    *,
    server_instance: Mapping[str, Any],
) -> dict[str, Any]:
    instance = dict(_validate_server_instance_diagnostics(server_instance))
    exported_by_candidate = {}
    expected_ids = {outcome.candidate_id for outcome in artifact.outcomes}
    _require(set(backends) == expected_ids, "backend diagnostics candidate set changed")
    transports = [backend.transport for backend in backends.values()]
    _require(
        bool(transports) and all(transport is transports[0] for transport in transports),
        "same-state candidate arms did not share one live server transport",
    )
    transport_base_url = getattr(transports[0], "base_url", None)
    if transport_base_url is not None:
        _require(
            str(transport_base_url).rstrip("/")
            == instance["physical_server_url"],
            "transport target differs from server instance diagnostics",
        )
    for backend in backends.values():
        actual_server_sha = sha256_bytes(
            canonical_bytes(dict(backend.expected_server_provenance))
        )
        _require(
            actual_server_sha == instance["server_provenance_sha256"],
            "backend server provenance differs from instance diagnostics",
        )
    for candidate_id in sorted(backends):
        exported = backends[candidate_id].export_plan_diagnostics()
        outcome = next(row for row in artifact.outcomes
                       if row.candidate_id == candidate_id)
        expected_plans = {plan.plan_sha256 for plan in outcome.plans}
        _require(set(exported) == expected_plans, "plan diagnostics do not cover artifact plans")
        exported_by_candidate[candidate_id] = exported

    outcome_by_candidate = {
        outcome.candidate_id: outcome for outcome in artifact.outcomes}
    _require("native" in outcome_by_candidate, "diagnostics require a native arm")
    native_xy_by_commitment = {}
    for plan in outcome_by_candidate["native"].plans:
        row = exported_by_candidate["native"][plan.plan_sha256]
        executable = _trajectory_array(
            row["executable_trajectory"], "native executable trajectory")
        _require(
            plan.commitment_index not in native_xy_by_commitment,
            "duplicate native diagnostic commitment",
        )
        native_xy_by_commitment[plan.commitment_index] = executable[:, :2]

    by_candidate = {}
    for candidate_id in sorted(exported_by_candidate):
        plans = {}
        outcome = outcome_by_candidate[candidate_id]
        for plan in outcome.plans:
            row = dict(exported_by_candidate[candidate_id][plan.plan_sha256])
            executable = _trajectory_array(
                row["executable_trajectory"],
                f"{candidate_id} executable trajectory",
            )
            native_xy = native_xy_by_commitment.get(plan.commitment_index)
            _require(
                native_xy is not None,
                "candidate commitment lacks a native diagnostic counterpart",
            )
            row["behaviorally_identical_xy"] = bool(
                np.array_equal(executable[:, :2], native_xy))
            plans[plan.plan_sha256] = row
        by_candidate[candidate_id] = plans
    return {
        "schema_version": PLAN_DIAGNOSTICS_SCHEMA,
        "artifact_sha256": artifact.artifact_sha256,
        "run_signature_sha256": artifact.run_signature_sha256,
        "state_id": artifact.state.state_id,
        "diffusion_seeds": list(artifact.diffusion_seeds),
        "server_instance": instance,
        "by_candidate": by_candidate,
    }


def validate_resume_pair(
    artifact_path: Path,
    diagnostics_path: Path,
    *,
    state_id: str,
    run_signature_sha256: str,
    diffusion_seeds: tuple[int, int, int],
    candidate_ids: Sequence[str],
) -> PairedRolloutArtifact:
    _require(
        artifact_path.is_file()
        and artifact_path.with_suffix(artifact_path.suffix + ".sha256").is_file()
        and diagnostics_path.is_file()
        and diagnostics_path.with_suffix(diagnostics_path.suffix + ".sha256").is_file(),
        "resume requires complete artifact and diagnostics pairs",
    )
    try:
        artifact = load_artifact(artifact_path)
    except RolloutProtocolError as error:
        raise CollectorError(f"resume rollout artifact is invalid: {error}") from error
    diagnostics = load_plan_diagnostics(diagnostics_path)
    _require(artifact.state.state_id == state_id, "resume state id mismatch")
    _require(
        artifact.run_signature_sha256 == run_signature_sha256,
        "resume run signature mismatch",
    )
    _require(artifact.diffusion_seeds == diffusion_seeds, "resume seeds mismatch")
    artifact_ids = [row.candidate_id for row in artifact.outcomes]
    expected_ids = sorted(candidate_ids, key=lambda value: (value != "native", value))
    _require(artifact_ids == expected_ids, "resume candidate arms mismatch")
    _require(
        diagnostics["artifact_sha256"] == artifact.artifact_sha256
        and diagnostics["run_signature_sha256"] == run_signature_sha256
        and diagnostics["state_id"] == state_id
        and diagnostics["diffusion_seeds"] == list(diffusion_seeds)
        and set(diagnostics["by_candidate"]) == set(candidate_ids),
        "resume plan diagnostics disagree with rollout artifact",
    )
    artifact_plan_hashes = {
        row.candidate_id: {plan.plan_sha256 for plan in row.plans}
        for row in artifact.outcomes
    }
    native_outcome = next(
        (row for row in artifact.outcomes if row.candidate_id == "native"),
        None,
    )
    _require(native_outcome is not None, "resume artifact lacks native arm")
    native_xy_by_commitment = {}
    for plan in native_outcome.plans:
        row = diagnostics["by_candidate"]["native"][plan.plan_sha256]
        native_xy_by_commitment[plan.commitment_index] = _trajectory_array(
            row["executable_trajectory"],
            "resume native executable trajectory",
        )[:, :2]
    for candidate_id, expected_plans in artifact_plan_hashes.items():
        _require(
            set(diagnostics["by_candidate"][candidate_id]) == expected_plans,
            "resume diagnostics plan coverage mismatch",
        )
        outcome = next(
            row for row in artifact.outcomes
            if row.candidate_id == candidate_id
        )
        plans_by_sha = {plan.plan_sha256: plan for plan in outcome.plans}
        for plan_sha, diagnostic in diagnostics["by_candidate"][candidate_id].items():
            plan = plans_by_sha[plan_sha]
            executable = _trajectory_array(
                diagnostic["executable_trajectory"],
                "resume executable trajectory",
            )
            expected_identical = bool(np.array_equal(
                executable[:, :2],
                native_xy_by_commitment[plan.commitment_index],
            ))
            _require(
                diagnostic["behaviorally_identical_xy"] == expected_identical,
                "resume behavioral XY equivalence is invalid",
            )
            recomputed = canonical_sha256({
                "backend_protocol": BACKEND_PROTOCOL,
                "server_receipt_sha256": diagnostic["server_receipt_sha256"],
                "raw_selected_trajectory": (
                    diagnostic["raw_selected_trajectory"]),
                "executable_trajectory": (
                    diagnostic["executable_trajectory"]),
                "all_trajectory": diagnostic["all_trajectory"],
                "all_values": diagnostic["all_values"],
                "critic_max": diagnostic["critic_max"],
                "stop_threshold": diagnostic["stop_threshold"],
                "low_critic_fallback_applied": (
                    diagnostic["low_critic_fallback_applied"]),
                "raw_current_rgb_sha256": plan.current_rgb_sha256,
                "raw_current_depth_sha256": plan.current_depth_sha256,
                "raw_goal_sha256": plan.goal_sha256,
            })
            _require(
                recomputed == plan_sha,
                "resume plan diagnostics do not reproduce artifact plan hash",
            )
    return artifact


class SingleSceneRuntimePool:
    """Keep at most one Habitat scene resident while processing sorted states."""

    def __init__(self) -> None:
        self._key: tuple[object, ...] | None = None
        self._runtime: PinnedHabitatRuntime | None = None

    def get(self, assets: FrozenStateAssets) -> PinnedHabitatRuntime:
        key = (
            assets.state.environment_sha256,
            assets.state.navmesh_sha256,
            assets.geometry_identity.identity_sha256,
            assets.camera_intrinsic,
            assets.camera_height_m,
        )
        if self._runtime is not None and key == self._key:
            return self._runtime
        self.close()
        self._runtime = PinnedHabitatRuntime.create_frozen(
            identity=assets.geometry_identity,
            glb_path=assets.glb_path,
            navmesh_path=assets.navmesh_path,
            camera_intrinsic=assets.camera_intrinsic,
            camera_height_m=assets.camera_height_m,
        )
        self._key = key
        return self._runtime

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.simulator.close()
        self._runtime = None
        self._key = None


@dataclass(frozen=True)
class CollectorInputs:
    candidate_path: Path
    candidate_sha256: str
    manifest_path: Path
    manifest_sha256: str
    geometry_map_path: Path
    geometry_map_sha256: str
    server_provenance_path: Path
    server_provenance_sha256: str
    server_url: str
    output_root: Path
    root_overrides: Mapping[str, Path]
    shard_index: int
    shard_count: int
    base_seed: int
    stop_threshold: float
    legacy_camera_height_m: float
    resume: bool
    dry_run: bool
    preflight_only: bool


def _manifest_index(manifest_path: Path, expected_sha: str) -> dict[str, Mapping[str, Any]]:
    raw = manifest_path.read_bytes()
    _require(sha256_bytes(raw) == expected_sha, "expert manifest SHA mismatch")
    manifest = _decode_json(raw, "expert manifest")
    _require(isinstance(manifest, Mapping), "expert manifest must be an object")
    # The manifest builder uses compact canonical JSON with one newline.
    _require(raw == canonical_bytes(manifest), "expert manifest is noncanonical")
    samples = manifest.get("samples")
    _require(isinstance(samples, list), "expert manifest has no samples")
    result = {}
    for sample in samples:
        _require(isinstance(sample, Mapping), "expert manifest sample is invalid")
        state_id = sample.get("sample_id")
        _require(isinstance(state_id, str) and state_id, "manifest sample id is invalid")
        _require(state_id not in result, f"duplicate expert sample {state_id}")
        result[state_id] = sample
    return result


def _check_candidate_manifest_binding(
    record: Mapping[str, Any],
    sample: Mapping[str, Any],
    assets: FrozenStateAssets,
) -> None:
    provenance = record["provenance"]
    state = assets.state
    expected = {
        "state_id": state.state_id,
        "session_id": state.session_id,
        "goal_epoch": state.goal_epoch,
        "goal_sha256": state.goal_sha256,
        "navdp_fifo_sha256": state.manifest_fifo_sha256,
        "scene_id": state.environment_id,
        "environment_id": state.environment_id,
        "navmesh_sha256": state.navmesh_sha256,
        "prefix_frames": sample["causal_prefix"]["frame_count"],
        "prefix_sha256": sample["causal_prefix"]["causal_prefix_sha256"],
        "goal_source_episode_id": sample["goal_source_episode_id"],
    }
    for field, value in expected.items():
        _require(
            provenance.get(field) == value,
            f"candidate/manifest binding mismatch for {field}",
        )


def collect_one_decision(
    *,
    record: Mapping[str, Any],
    assets: FrozenStateAssets,
    runtime: PinnedHabitatRuntime,
    server_url: str,
    server_provenance: Mapping[str, str],
    server_provenance_sha256: str,
    run_signature_sha256: str,
    base_seed: int,
    stop_threshold: float,
    controller: PurePursuitConfig,
) -> tuple[PairedRolloutArtifact, dict[str, Any]]:
    arms = candidate_arms_from_feature_record(record, assets.start_pose)
    _require(len(arms) >= 2, "H24 collection requires at least one residual arm")
    seeds = decision_seeds(base_seed, assets.state.state_id)
    transport = RequestsJsonTransport(server_url)
    backends: dict[str, RealH24RolloutBackend] = {}

    def factory(candidate_id: str) -> RealH24RolloutBackend:
        _require(candidate_id not in backends, "backend factory reused a candidate id")
        backend = RealH24RolloutBackend(
            assets,
            transport,
            runtime,
            expected_server_provenance=server_provenance,
            stop_threshold=stop_threshold,
            reset_seed=reset_seed(base_seed, assets.state.state_id),
            controller=controller,
        )
        backends[candidate_id] = backend
        return backend

    try:
        artifact = collect_paired_rollouts(
            factory,
            assets.state,
            arms,
            seeds,
            run_signature_sha256=run_signature_sha256,
        )
    except (RolloutProtocolError, RuntimeError, ValueError) as error:
        raise CollectorError(
            f"H24 collection failed for {assets.state.state_id}: {error}") from error
    server_instance = build_server_instance_diagnostics(
        server_url, server_provenance_sha256
    )
    return artifact, build_plan_diagnostics(
        artifact, backends, server_instance=server_instance
    )


def run_collector(inputs: CollectorInputs) -> dict[str, Any]:
    assert_pythonpath()
    _require(
        inputs.server_url.startswith(("http://", "https://")),
        "server URL must be HTTP(S)",
    )
    _require(
        math.isfinite(inputs.stop_threshold),
        "stop threshold must be finite",
    )
    _require(
        math.isfinite(inputs.legacy_camera_height_m)
        and inputs.legacy_camera_height_m > 0.0,
        "legacy camera height must be finite and positive",
    )
    _require(
        isinstance(inputs.base_seed, int)
        and not isinstance(inputs.base_seed, bool)
        and 0 <= inputs.base_seed < 2**63,
        "base seed is invalid",
    )
    candidate_sha = _valid_sha(inputs.candidate_sha256, "candidate SHA")
    manifest_sha = _valid_sha(inputs.manifest_sha256, "manifest SHA")
    geometry_map_sha = _valid_sha(inputs.geometry_map_sha256, "geometry map SHA")
    provenance_sha = _valid_sha(
        inputs.server_provenance_sha256, "server provenance SHA")
    records, precollection = load_candidate_records(
        inputs.candidate_path, candidate_sha)
    selected = selected_shard_records(
        records, inputs.shard_index, inputs.shard_count)
    _require(bool(selected), "selected shard contains no candidate decisions")
    geometry_map = load_geometry_map(inputs.geometry_map_path, geometry_map_sha)
    server_provenance = load_server_provenance(
        inputs.server_provenance_path, provenance_sha)
    verify_candidate_source_policy(records, server_provenance)
    manifest_index = _manifest_index(inputs.manifest_path, manifest_sha)
    controller = PurePursuitConfig()
    run_signature = build_run_signature(
        candidate_sha256=candidate_sha,
        manifest_sha256=manifest_sha,
        geometry_map_sha256=geometry_map_sha,
        server_provenance_sha256=provenance_sha,
        base_seed=inputs.base_seed,
        stop_threshold=inputs.stop_threshold,
        legacy_camera_height_m=inputs.legacy_camera_height_m,
        controller=controller,
    )

    # Static materialization is deliberately done for every selected row even
    # in dry-run mode.  It verifies all causal prefix bytes, geometry files,
    # candidate/manifest bindings, and candidate world-point conversion.
    work = []
    for record in selected:
        state_id = str(record["provenance"]["state_id"])
        _require(state_id in manifest_index, f"candidate state absent from manifest: {state_id}")
        scene_id = str(record["provenance"]["scene_id"])
        _require(scene_id in geometry_map, f"scene absent from geometry map: {scene_id}")
        entry = geometry_map[scene_id]
        assets = load_state_assets_from_manifest(
            inputs.manifest_path,
            manifest_sha,
            state_id,
            entry.identity_path,
            root_overrides=inputs.root_overrides,
            legacy_camera_height_m=inputs.legacy_camera_height_m,
        )
        _require(
            assets.geometry_identity.identity_sha256 == entry.identity_sha256,
            "loaded state geometry identity differs from geometry map",
        )
        sample = manifest_index[state_id]
        _check_candidate_manifest_binding(record, sample, assets)
        arms = candidate_arms_from_feature_record(record, assets.start_pose)
        _require(len(arms) >= 2, f"state has no residual candidate: {state_id}")
        stem = safe_state_stem(state_id)
        artifact_path = inputs.output_root / f"shard-{inputs.shard_index:04d}" / f"{stem}.json"
        diagnostics_path = artifact_path.with_name(f"{stem}.plans.json")
        work.append((record, assets, arms, artifact_path, diagnostics_path))

    summary = {
        "schema_version": COLLECTOR_SCHEMA,
        "mode": (
            "dry_run" if inputs.dry_run else
            "preflight_only" if inputs.preflight_only else "collect"
        ),
        "run_signature_sha256": run_signature,
        "server_instance": build_server_instance_diagnostics(
            inputs.server_url, provenance_sha
        ),
        "candidate_count_total": len(records),
        "candidate_count_shard": len(work),
        "precollection_neutral_labels_present": precollection,
        "legacy_camera_height_m": inputs.legacy_camera_height_m,
        "shard_index": inputs.shard_index,
        "shard_count": inputs.shard_count,
        "written": 0,
        "resumed": 0,
        "preflight_prepared": 0,
        "state_ids": [assets.state.state_id for _record, assets, *_rest in work],
    }
    if inputs.dry_run:
        return summary

    runtime_pool = SingleSceneRuntimePool()
    try:
        if inputs.preflight_only:
            record, assets, _arms, _artifact_path, _diagnostics_path = work[0]
            runtime = runtime_pool.get(assets)
            transport = RequestsJsonTransport(inputs.server_url)
            backend = RealH24RolloutBackend(
                assets,
                transport,
                runtime,
                expected_server_provenance=server_provenance,
                stop_threshold=inputs.stop_threshold,
                reset_seed=reset_seed(inputs.base_seed, assets.state.state_id),
                controller=controller,
            )
            preparation = backend.prepare_arm(assets.state)
            _require(preparation.diffusion_calls == 0, "preflight sampled diffusion")
            summary["preflight_prepared"] = 1
            return summary

        for record, assets, arms, artifact_path, diagnostics_path in work:
            seeds = decision_seeds(inputs.base_seed, assets.state.state_id)
            candidates = [arm.candidate_id for arm in arms]
            any_output = any(path.exists() for path in (
                artifact_path,
                artifact_path.with_suffix(artifact_path.suffix + ".sha256"),
                diagnostics_path,
                diagnostics_path.with_suffix(diagnostics_path.suffix + ".sha256"),
            ))
            if any_output:
                _require(inputs.resume, f"output exists without --resume: {artifact_path}")
                validate_resume_pair(
                    artifact_path,
                    diagnostics_path,
                    state_id=assets.state.state_id,
                    run_signature_sha256=run_signature,
                    diffusion_seeds=seeds,
                    candidate_ids=candidates,
                )
                summary["resumed"] += 1
                continue
            runtime = runtime_pool.get(assets)
            artifact, diagnostics = collect_one_decision(
                record=record,
                assets=assets,
                runtime=runtime,
                server_url=inputs.server_url,
                server_provenance=server_provenance,
                server_provenance_sha256=provenance_sha,
                run_signature_sha256=run_signature,
                base_seed=inputs.base_seed,
                stop_threshold=inputs.stop_threshold,
                controller=controller,
            )
            try:
                atomic_write_artifact(artifact_path, artifact)
                _atomic_write_pair(diagnostics_path, diagnostics)
            except (RolloutProtocolError, OSError) as error:
                raise CollectorError(
                    f"cannot atomically persist state {assets.state.state_id}: {error}") from error
            # Read both pairs back from disk before declaring the row complete.
            validate_resume_pair(
                artifact_path,
                diagnostics_path,
                state_id=assets.state.state_id,
                run_signature_sha256=run_signature,
                diffusion_seeds=seeds,
                candidate_ids=candidates,
            )
            summary["written"] += 1
    finally:
        runtime_pool.close()
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sets", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--geometry-map", type=Path, required=True)
    parser.add_argument("--expected-geometry-map-sha256", required=True)
    parser.add_argument("--server-provenance", type=Path, required=True)
    parser.add_argument("--expected-server-provenance-sha256", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--root-override", action="append", default=[],
        metavar="NAME=PATH",
        help="relocate episode_root/environment_root/navmesh_root",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument(
        "--stop-threshold",
        type=float,
        required=True,
        help=(
            "NavDP critic threshold bound into every arm; pass the exact "
            "evaluation value explicitly (the current Habitat baseline uses "
            "-0.5)"
        ),
    )
    parser.add_argument(
        "--legacy-camera-height-m",
        type=float,
        required=True,
        help=(
            "explicit historical generator camera height; checked against "
            "metadata when present and required when absent"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require(
        args.server_url.startswith(("http://", "https://")),
        "server URL must be HTTP(S)",
    )
    _require(
        math.isfinite(args.stop_threshold), "stop threshold must be finite")
    _require(
        math.isfinite(args.legacy_camera_height_m)
        and args.legacy_camera_height_m > 0.0,
        "legacy camera height must be finite and positive",
    )
    inputs = CollectorInputs(
        candidate_path=args.candidate_sets.resolve(),
        candidate_sha256=args.expected_candidate_sha256,
        manifest_path=args.expert_manifest.resolve(),
        manifest_sha256=args.expected_manifest_sha256,
        geometry_map_path=args.geometry_map.resolve(),
        geometry_map_sha256=args.expected_geometry_map_sha256,
        server_provenance_path=args.server_provenance.resolve(),
        server_provenance_sha256=args.expected_server_provenance_sha256,
        server_url=args.server_url.rstrip("/"),
        output_root=args.output_root.resolve(),
        root_overrides=parse_root_overrides(args.root_override),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        base_seed=args.base_seed,
        stop_threshold=args.stop_threshold,
        legacy_camera_height_m=args.legacy_camera_height_m,
        resume=args.resume,
        dry_run=args.dry_run,
        preflight_only=args.preflight_only,
    )
    summary = run_collector(inputs)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(json.dumps({
            "status": "failed_closed",
            "error": str(error),
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
