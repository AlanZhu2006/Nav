#!/usr/bin/env python3
"""Build strictly neutral NLSR candidate sets before real H24 collection.

The bridge intentionally has a narrow trust boundary.  It consumes one
content-pinned causal expert manifest and the frontier artifact generated from
that exact manifest.  An optional second manifest may supply freshly baked
evaluation geometry, but only after this module proves that it is an exact
derived copy whose sole semantic changes are the NavMesh records.  Only
``lingbot_deployment_pose`` proposals can become residual candidates; the
parallel ``teacher_pose`` arm remains audit-only.

The output is canonical ``novel_candidate_set_v2`` JSON/JSONL.  Rollout,
match, co-visibility, and pose labels are neutral.  Proposal-proxy labels are
copied only when the pinned proposal artifact already contains a valid,
internally consistent label table.  Missing deployment features are zeroed and
masked, never guessed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

try:
    from MemNavData.novel_candidate_set_schema_v2 import (
        CANDIDATE_TYPES,
        FEATURE_PRESENCE_MASK_SIZE,
        SCHEMA_VERSION as CANDIDATE_SET_SCHEMA,
        SET_FEATURE_PRESENCE_MASK_SIZE,
        validate_candidate_dataset,
    )
except ImportError:  # pragma: no cover - direct script execution
    from novel_candidate_set_schema_v2 import (  # type: ignore
        CANDIDATE_TYPES,
        FEATURE_PRESENCE_MASK_SIZE,
        SCHEMA_VERSION as CANDIDATE_SET_SCHEMA,
        SET_FEATURE_PRESENCE_MASK_SIZE,
        validate_candidate_dataset,
    )


BUILDER_SCHEMA = "nlsr_precollection_bridge_v2"
RELATION_ARTIFACT_SCHEMA = "nlsr_deployment_relation_artifact_v1"
DERIVED_GEOMETRY_SCHEMA = "nlsr_derived_geometry_manifest_v1"
DERIVED_GEOMETRY_STATUS = "fresh_double_bake_roundtrip_verified"
SUPPORTED_MANIFEST_SCHEMAS = frozenset(
    {
        "nlsr_v2_expert_candidate_manifest_v1",
        "nlsr_v2_expert_candidate_manifest_v2",
        "nlsr_v2_multistage_expert_candidate_manifest_v1",
    }
)
SUPPORTED_PROPOSAL_ARTIFACT_SCHEMAS = frozenset(
    {
        "nlsr_v2_frontier_proposal_artifact_v1",
        "nlsr_v2_frontier_proposal_artifact_v2",
    }
)
DEPLOYMENT_ARM = "lingbot_deployment_pose"
TEACHER_ARM = "teacher_pose"
RELATION_FIELDS = (
    "goal_patch_relation",
    "goal_temporal_relation",
    "local_map_relation",
)
RELATION_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "goal_patch_relation",
        "goal_patch_relation_present",
        "goal_temporal_relation",
        "goal_temporal_relation_present",
        "local_map_relation",
        "local_map_relation_present",
        "pose_translation_p90_m",
        "pose_yaw_p90_deg",
        "pose_uncertainty_present",
        "depth_confidence_mean",
        "depth_confidence_present",
    }
)
PROPOSAL_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "map_xy_m",
        "subgoal_forward_m",
        "subgoal_left_m",
        "distance_m",
        "bearing_rad",
        "frontier_normal_bearing_rad",
        "resolution_m",
        "grid_cell",
        "frontier_boundary_m",
        "frontier_novelty_m",
        "clearance_lower_m",
        "topology_score",
        "context_frame_indices",
        "goal_patch_relation_score",
        "goal_patch_relation_present",
        "selection_sources",
        "source_scales_m",
    }
)
PROPOSAL_KEYS = frozenset(
    {
        "schema_version",
        "valid",
        "invalid_reason",
        "pose_frame_index",
        "scan_frame_indices",
        "goal_patch_relation_present",
        "goal_patch_relation_mask",
        "shortlist_policy",
        "scale_summaries",
        "raw_candidate_count",
        "nms_candidate_count",
        "shortlist_count",
        "candidate_universe",
        "shortlist",
        "nms_suppressed",
    }
)
PROXY_KEYS = frozenset(
    {
        "status",
        "label_valid",
        "labeler_provenance",
        "positive_margin_m",
        "labels",
        "universe_has_positive",
        "shortlist_has_positive",
        "coverage_miss",
        "proposal_sha256",
    }
)
PROXY_LABEL_KEYS = frozenset({"candidate_id", "reachable", "progress_m", "positive"})
ARM_KEYS = frozenset(
    {
        "arm",
        "deployment_eligible_pose_source",
        "pose_provenance",
        "proposal",
        "proposal_proxy",
    }
)
PROPOSAL_RECORD_KEYS = frozenset(
    {
        "sample_id",
        "scene",
        "source_episode",
        "goal_episode",
        "goal_variant",
        "state_name",
        "split_role",
        "decision_frame",
        "causal_prefix_sha256",
        "goal_sha256",
        "patch_score_present",
        "arms",
    }
)
PROPOSAL_RECORD_KEYS_WITH_GOAL_ROLE = frozenset(
    set(PROPOSAL_RECORD_KEYS) | {"goal_role"}
)
ZERO_SHA = "0" * 64
DERIVATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "parent_manifest_sha256",
        "settings_file_sha256",
        "requested_settings_sha256",
        "runtime_effective_settings_sha256",
        "runtime",
        "run_contract",
        "bake_index",
        "selection_boundary",
        "determinism_boundary",
    }
)
STABLE_FILE_RECORD_KEYS = frozenset(
    {"path", "path_sha256", "bytes", "content_sha256"}
)


class PrecollectionBuildError(RuntimeError):
    """A pinned input or deployment/label boundary failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecollectionBuildError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


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
        raise PrecollectionBuildError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(records), "JSONL output cannot be empty")
    return b"".join(canonical_json_bytes(record) for record in records)


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_pinned_canonical_json(
    path: Path | str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    source = Path(path)
    expected = _valid_sha(expected_sha256, f"{source.name} expected SHA256")
    _require(source.is_file(), f"pinned input is missing: {source}")
    raw = source.read_bytes()
    _require(_sha256_bytes(raw) == expected, f"pinned input SHA mismatch: {source}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PrecollectionBuildError(f"{source} contains non-finite constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrecollectionBuildError(
            f"pinned input is invalid JSON: {source}"
        ) from error
    _require(isinstance(value, Mapping), f"pinned input must be an object: {source}")
    _require(
        raw == canonical_json_bytes(value), f"pinned input is noncanonical: {source}"
    )
    return value


def rollout_labeler_code_sha256(repo_root: Path | None = None) -> str:
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else repo_root.resolve()
    )
    files = (
        "MemNavData/collect_real_h24_rollouts.py",
        "MemNavData/real_h24_rollout_backend.py",
        "MemNavData/novel_rollout_protocol_v2.py",
    )
    rows = []
    for relative in files:
        path = root / relative
        _require(path.is_file(), f"rollout-labeler source is missing: {path}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return _sha256_bytes(canonical_json_bytes(rows))


def _exact_mapping(
    value: object,
    keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = frozenset(value)
    _require(
        actual == keys,
        f"{label} fields changed: missing={sorted(keys - actual)} "
        f"extra={sorted(actual - keys)}",
    )
    return value


def _finite_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite numeric",
    )
    result = float(value)
    _require(not nonnegative or result >= 0.0, f"{label} must be non-negative")
    return result


def _finite_vector(value: object, length: int, label: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == length,
        f"{label} must be a length-{length} list",
    )
    return [
        _finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _scene_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    scenes = manifest.get("scenes")
    _require(isinstance(scenes, list) and scenes, "manifest scenes are missing")
    result = {}
    for row in scenes:
        _require(isinstance(row, Mapping), "manifest scene row is malformed")
        scene = row.get("scene")
        _require(isinstance(scene, str) and scene, "manifest scene id is malformed")
        _require(scene not in result, f"duplicate manifest scene {scene}")
        environment = row.get("environment")
        navmesh = row.get("navmesh")
        _require(
            isinstance(environment, Mapping) and isinstance(navmesh, Mapping),
            f"scene {scene} lacks environment/navmesh records",
        )
        _valid_sha(environment.get("content_sha256"), f"{scene} environment SHA")
        _valid_sha(navmesh.get("content_sha256"), f"{scene} navmesh SHA")
        result[scene] = row
    return result


def _sample_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    samples = manifest.get("samples")
    _require(isinstance(samples, list) and samples, "manifest samples are missing")
    result = {}
    for row in samples:
        _require(isinstance(row, Mapping), "manifest sample is malformed")
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id, "sample_id is malformed")
        _require(sample_id not in result, f"duplicate manifest sample {sample_id}")
        result[sample_id] = row
    return result


def _group_index(manifest: Mapping[str, Any]) -> dict[str, str]:
    samples = _sample_index(manifest)
    bindings = manifest.get("sample_group_bindings")
    if bindings is None:
        return {
            sample_id: (
                f"{sample['split_role']}/{sample['scene']}/"
                f"{sample['source_episode']}/{sample['state_name']}"
            )
            for sample_id, sample in samples.items()
        }
    _require(isinstance(bindings, list), "sample_group_bindings must be a list")
    result = {}
    for row in bindings:
        _require(isinstance(row, Mapping), "sample group binding is malformed")
        sample_id = row.get("sample_id")
        group = row.get("counterfactual_pair_group_id")
        _require(sample_id in samples, "group binding references unknown sample")
        _require(
            isinstance(group, str) and group,
            "counterfactual pair group id is malformed",
        )
        _require(sample_id not in result, f"duplicate group binding {sample_id}")
        result[str(sample_id)] = group
    _require(set(result) == set(samples), "sample group bindings are incomplete")
    return result


def _validate_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, str]]:
    _require(
        manifest.get("schema_version") in SUPPORTED_MANIFEST_SCHEMAS,
        f"unsupported manifest schema {manifest.get('schema_version')!r}",
    )
    split = manifest.get("split")
    _require(isinstance(split, Mapping), "manifest split is malformed")
    _valid_sha(split.get("sha256"), "manifest split SHA")
    samples = _sample_index(manifest)
    scenes = _scene_index(manifest)
    groups = _group_index(manifest)
    for sample_id, sample in samples.items():
        scene = sample.get("scene")
        _require(scene in scenes, f"sample {sample_id} references absent scene")
        _require(
            sample.get("split_role") in ("train", "development"),
            f"sample {sample_id} has forbidden split role",
        )
        for key in ("source_episode_id", "goal_source_episode_id"):
            _require(
                isinstance(sample.get(key), str) and sample.get(key),
                f"sample {sample_id} {key} is malformed",
            )
        _require(
            isinstance(sample.get("decision_frame"), int)
            and not isinstance(sample.get("decision_frame"), bool)
            and int(sample["decision_frame"]) >= 1,
            f"sample {sample_id} decision frame is malformed",
        )
        prefix = sample.get("causal_prefix")
        fifo = sample.get("navdp_fifo")
        goal = sample.get("goal")
        _require(
            isinstance(prefix, Mapping)
            and isinstance(fifo, Mapping)
            and isinstance(goal, Mapping),
            f"sample {sample_id} causal records are malformed",
        )
        _valid_sha(prefix.get("causal_prefix_sha256"), f"{sample_id} prefix SHA")
        _valid_sha(fifo.get("fifo_sha256"), f"{sample_id} FIFO SHA")
        _valid_sha(goal.get("content_sha256"), f"{sample_id} goal SHA")
    return samples, scenes, groups


def _validate_stable_file_record(value: object, label: str) -> Mapping[str, Any]:
    record = _exact_mapping(value, STABLE_FILE_RECORD_KEYS, label)
    relative = record["path"]
    _require(isinstance(relative, str) and relative, f"{label}.path is malformed")
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute()
        and ".." not in posix.parts
        and str(posix) == relative,
        f"{label}.path must be a normalized relative POSIX path",
    )
    _require(
        record["path_sha256"] == _sha256_bytes(relative.encode("utf-8")),
        f"{label}.path_sha256 disagrees with path",
    )
    _require(
        isinstance(record["bytes"], int)
        and not isinstance(record["bytes"], bool)
        and int(record["bytes"]) > 0,
        f"{label}.bytes must be a positive integer",
    )
    _valid_sha(record["content_sha256"], f"{label}.content_sha256")
    return record


def _validate_evaluation_geometry_manifest(
    *,
    causal_manifest: Mapping[str, Any],
    causal_manifest_sha256: str,
    evaluation_manifest: Mapping[str, Any] | None,
    evaluation_manifest_sha256: str | None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    """Return geometry scenes after proving the exact causal/derived lineage.

    The proposal and relation artifacts remain bound to ``causal_manifest``.
    A supplied evaluation manifest is accepted only if removing its derivation
    receipt, restoring the original root/NavMesh records, and removing each
    scene bake receipt reproduces the causal manifest exactly.
    """
    _require(
        (evaluation_manifest is None) == (evaluation_manifest_sha256 is None),
        "evaluation geometry manifest and SHA must be supplied together",
    )
    causal_scenes = _scene_index(causal_manifest)
    if evaluation_manifest is None:
        return causal_scenes, {
            "mode": "causal_manifest_geometry",
            "causal_manifest_sha256": causal_manifest_sha256,
            "evaluation_geometry_manifest_sha256": causal_manifest_sha256,
        }

    evaluation_sha = _valid_sha(
        evaluation_manifest_sha256, "evaluation geometry manifest SHA"
    )
    _require(
        _sha256_bytes(canonical_json_bytes(evaluation_manifest)) == evaluation_sha,
        "in-memory evaluation geometry manifest differs from its canonical SHA pin",
    )
    _require(
        "geometry_bake_derivation" not in causal_manifest,
        "causal source manifest is already geometry-derived",
    )
    causal_roots = causal_manifest.get("input_roots")
    evaluation_roots = evaluation_manifest.get("input_roots")
    _require(
        isinstance(causal_roots, Mapping)
        and isinstance(evaluation_roots, Mapping),
        "dual-manifest geometry requires input_roots objects",
    )
    _require(
        "geometry_bake_root" not in causal_roots,
        "causal source manifest already declares a geometry bake root",
    )
    _require(
        isinstance(causal_roots.get("navmesh_root"), str)
        and bool(causal_roots["navmesh_root"]),
        "causal source manifest navmesh root is malformed",
    )
    _require(
        set(evaluation_manifest) == set(causal_manifest) | {"geometry_bake_derivation"},
        "derived manifest top-level fields differ beyond geometry derivation",
    )
    _require(
        set(evaluation_roots) == set(causal_roots) | {"geometry_bake_root"},
        "derived input roots differ beyond navmesh/geometry bake roots",
    )
    for key in causal_roots:
        if key != "navmesh_root":
            _require(
                evaluation_roots[key] == causal_roots[key],
                f"derived input root changed forbidden field {key}",
            )
    _require(
        isinstance(evaluation_roots.get("navmesh_root"), str)
        and bool(evaluation_roots["navmesh_root"])
        and isinstance(evaluation_roots.get("geometry_bake_root"), str)
        and bool(evaluation_roots["geometry_bake_root"]),
        "derived geometry roots are malformed",
    )

    derivation = _exact_mapping(
        evaluation_manifest.get("geometry_bake_derivation"),
        DERIVATION_KEYS,
        "geometry_bake_derivation",
    )
    _require(
        derivation["schema_version"] == DERIVED_GEOMETRY_SCHEMA
        and derivation["status"] == DERIVED_GEOMETRY_STATUS,
        "derived geometry schema/status changed",
    )
    _require(
        derivation["parent_manifest_sha256"] == causal_manifest_sha256,
        "derived geometry parent is not the exact causal manifest SHA",
    )
    for key in (
        "parent_manifest_sha256",
        "settings_file_sha256",
        "requested_settings_sha256",
        "runtime_effective_settings_sha256",
    ):
        _valid_sha(derivation[key], f"geometry_bake_derivation.{key}")
    _require(
        isinstance(derivation["runtime"], Mapping) and bool(derivation["runtime"]),
        "geometry bake runtime provenance is absent",
    )
    canonical_json_bytes(derivation["runtime"])
    _validate_stable_file_record(
        derivation["run_contract"], "geometry_bake_derivation.run_contract"
    )
    _validate_stable_file_record(
        derivation["bake_index"], "geometry_bake_derivation.bake_index"
    )
    for key in ("selection_boundary", "determinism_boundary"):
        _require(
            isinstance(derivation[key], str) and bool(derivation[key]),
            f"geometry_bake_derivation.{key} is malformed",
        )

    causal_scene_rows = causal_manifest.get("scenes")
    evaluation_scene_rows = evaluation_manifest.get("scenes")
    _require(
        isinstance(causal_scene_rows, list)
        and isinstance(evaluation_scene_rows, list)
        and len(evaluation_scene_rows) == len(causal_scene_rows),
        "derived scene list length changed",
    )
    rebuilt_scenes = []
    for index, (causal_scene, evaluation_scene) in enumerate(
        zip(causal_scene_rows, evaluation_scene_rows)
    ):
        _require(
            isinstance(causal_scene, Mapping) and isinstance(evaluation_scene, Mapping),
            f"derived scene row {index} is malformed",
        )
        scene_id = causal_scene.get("scene")
        _require(
            evaluation_scene.get("scene") == scene_id,
            f"derived scene order/identity changed at index {index}",
        )
        _require(
            "geometry_bake_receipt" not in causal_scene
            and set(evaluation_scene) == set(causal_scene) | {"geometry_bake_receipt"},
            f"derived scene {scene_id} fields changed beyond bake receipt",
        )
        _validate_stable_file_record(
            evaluation_scene.get("navmesh"), f"derived scene {scene_id}.navmesh"
        )
        _validate_stable_file_record(
            evaluation_scene.get("geometry_bake_receipt"),
            f"derived scene {scene_id}.geometry_bake_receipt",
        )
        rebuilt_scene = copy.deepcopy(dict(evaluation_scene))
        rebuilt_scene.pop("geometry_bake_receipt")
        rebuilt_scene["navmesh"] = copy.deepcopy(causal_scene["navmesh"])
        _require(
            rebuilt_scene == causal_scene,
            f"derived scene {scene_id} changed non-geometry content",
        )
        rebuilt_scenes.append(rebuilt_scene)

    rebuilt = copy.deepcopy(dict(evaluation_manifest))
    rebuilt.pop("geometry_bake_derivation")
    rebuilt_roots = dict(rebuilt["input_roots"])
    rebuilt_roots.pop("geometry_bake_root")
    rebuilt_roots["navmesh_root"] = copy.deepcopy(causal_roots["navmesh_root"])
    rebuilt["input_roots"] = rebuilt_roots
    rebuilt["scenes"] = rebuilt_scenes
    _require(
        rebuilt == causal_manifest,
        "derived geometry manifest cannot be exactly reduced to its causal parent",
    )
    _samples, evaluation_scenes, _groups = _validate_manifest(evaluation_manifest)
    return evaluation_scenes, {
        "mode": "verified_derived_geometry_manifest",
        "causal_manifest_sha256": causal_manifest_sha256,
        "evaluation_geometry_manifest_sha256": evaluation_sha,
        "parent_manifest_sha256": str(derivation["parent_manifest_sha256"]),
    }


def _validate_proxy(
    raw: object,
    proposal: Mapping[str, Any],
    universe_ids: set[str],
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, bool]]:
    proxy = _exact_mapping(raw, PROXY_KEYS, label)
    expected_proposal_sha = _sha256_bytes(canonical_json_bytes(proposal))
    _require(
        proxy["proposal_sha256"] == expected_proposal_sha,
        f"{label} does not bind the exact proposal",
    )
    valid = proxy["label_valid"]
    _require(type(valid) is bool, f"{label}.label_valid must be boolean")
    labels = proxy["labels"]
    _require(isinstance(labels, list), f"{label}.labels must be a list")
    by_id = {}
    for index, raw_label in enumerate(labels):
        row = _exact_mapping(raw_label, PROXY_LABEL_KEYS, f"{label}.labels[{index}]")
        candidate_id = row["candidate_id"]
        _require(
            isinstance(candidate_id, str) and candidate_id in universe_ids,
            f"{label} proxy label references unknown candidate",
        )
        _require(candidate_id not in by_id, f"{label} has duplicate proxy labels")
        reachable = row["reachable"]
        positive = row["positive"]
        _require(
            type(reachable) is bool and type(positive) is bool,
            f"{label} proxy booleans are malformed",
        )
        progress = _finite_float(row["progress_m"], f"{label} progress")
        _require(
            positive == bool(reachable and progress > 0.0),
            f"{label} proxy positive disagrees with progress/reachability",
        )
        _require(
            reachable or progress == 0.0, f"{label} unreachable progress is nonzero"
        )
        by_id[candidate_id] = row
    if valid:
        _require(
            isinstance(proxy["labeler_provenance"], Mapping)
            and bool(proxy["labeler_provenance"]),
            f"{label} valid labels require non-empty labeler provenance",
        )
        # The complete enclosing proposal artifact is externally pinned.  A
        # canonicality check here ensures nested provenance cannot hide NaN or
        # non-JSON values before it becomes the authority for proxy labels.
        canonical_json_bytes(proxy["labeler_provenance"])
        _require(
            proxy["positive_margin_m"] == 0.0,
            f"{label} positive margin must match candidate-set margin 0.0",
        )
        _require(set(by_id) == universe_ids, f"{label} proxy universe is incomplete")
    else:
        _require(not labels, f"{label} invalid proxy must not carry labels")
        _require(
            proxy["universe_has_positive"] is False
            and proxy["shortlist_has_positive"] is False
            and proxy["coverage_miss"] is False,
            f"{label} invalid proxy summary must be neutral",
        )
    for key in ("universe_has_positive", "shortlist_has_positive", "coverage_miss"):
        _require(type(proxy[key]) is bool, f"{label}.{key} must be boolean")
    return by_id, {
        "valid": valid,
        "universe": bool(proxy["universe_has_positive"]),
        "shortlist": bool(proxy["shortlist_has_positive"]),
        "miss": bool(proxy["coverage_miss"]),
    }


def _validate_proposal_candidate(raw: object, label: str) -> Mapping[str, Any]:
    row = _exact_mapping(raw, PROPOSAL_CANDIDATE_KEYS, label)
    candidate_id = row["candidate_id"]
    _require(
        isinstance(candidate_id, str)
        and candidate_id
        and candidate_id not in ("native", "dustbin"),
        f"{label} candidate id is reserved or malformed",
    )
    for key in (
        "subgoal_forward_m",
        "subgoal_left_m",
        "distance_m",
        "bearing_rad",
        "frontier_normal_bearing_rad",
        "resolution_m",
        "frontier_boundary_m",
        "frontier_novelty_m",
        "clearance_lower_m",
        "topology_score",
        "goal_patch_relation_score",
    ):
        _finite_float(
            row[key],
            f"{label}.{key}",
            nonnegative=key
            in {
                "distance_m",
                "resolution_m",
                "frontier_boundary_m",
                "frontier_novelty_m",
                "clearance_lower_m",
            },
        )
    _require(
        type(row["goal_patch_relation_present"]) is bool,
        f"{label}.goal_patch_relation_present must be boolean",
    )
    _require(
        row["goal_patch_relation_present"]
        or float(row["goal_patch_relation_score"]) == 0.0,
        f"{label} absent goal patch relation must be zero",
    )
    _require(
        math.isclose(
            math.hypot(float(row["subgoal_forward_m"]), float(row["subgoal_left_m"])),
            float(row["distance_m"]),
            rel_tol=1e-5,
            abs_tol=1e-5,
        ),
        f"{label} local subgoal distance is inconsistent",
    )
    return row


def _proposal_index(
    artifact: Mapping[str, Any],
    manifest_sha256: str,
    manifest_samples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    _require(
        artifact.get("schema_version") in SUPPORTED_PROPOSAL_ARTIFACT_SCHEMAS,
        f"unsupported proposal artifact schema {artifact.get('schema_version')!r}",
    )
    provenance = artifact.get("provenance")
    _require(isinstance(provenance, Mapping), "proposal provenance is missing")
    _require(
        provenance.get("input_manifest_sha256") == manifest_sha256,
        "proposal artifact is not bound to the pinned manifest",
    )
    records = artifact.get("records")
    _require(isinstance(records, list), "proposal records must be a list")
    result = {}
    for index, row in enumerate(records):
        _require(isinstance(row, Mapping), f"proposal record {index} is malformed")
        keys = frozenset(row)
        _require(
            keys in (PROPOSAL_RECORD_KEYS, PROPOSAL_RECORD_KEYS_WITH_GOAL_ROLE),
            f"proposal record {index} fields changed: {sorted(keys)}",
        )
        sample_id = row.get("sample_id")
        _require(sample_id in manifest_samples, "proposal references unknown sample")
        _require(sample_id not in result, f"duplicate proposal record {sample_id}")
        sample = manifest_samples[str(sample_id)]
        expected = {
            "scene": sample["scene"],
            "source_episode": sample["source_episode"],
            "goal_episode": sample["goal_episode"],
            "goal_variant": sample["goal_variant"],
            "state_name": sample["state_name"],
            "split_role": sample["split_role"],
            "decision_frame": sample["decision_frame"],
            "causal_prefix_sha256": sample["causal_prefix"]["causal_prefix_sha256"],
            "goal_sha256": sample["goal"]["content_sha256"],
        }
        if "goal_role" in row:
            expected["goal_role"] = sample.get("goal_role")
        for key, value in expected.items():
            _require(
                row.get(key) == value, f"proposal/manifest mismatch: {sample_id}.{key}"
            )
        arms = row.get("arms")
        _require(
            isinstance(arms, Mapping) and set(arms) == {TEACHER_ARM, DEPLOYMENT_ARM},
            f"proposal {sample_id} must contain exactly teacher/deployment arms",
        )
        for arm_name in (TEACHER_ARM, DEPLOYMENT_ARM):
            arm = _exact_mapping(arms[arm_name], ARM_KEYS, f"{sample_id}.{arm_name}")
            _require(arm["arm"] == arm_name, f"{sample_id} arm name changed")
            _require(
                arm["deployment_eligible_pose_source"] is (arm_name == DEPLOYMENT_ARM),
                f"{sample_id} arm deployment eligibility changed",
            )
        deployment = arms[DEPLOYMENT_ARM]
        proposal = _exact_mapping(
            deployment["proposal"], PROPOSAL_KEYS, f"{sample_id}.deployment.proposal"
        )
        _require(
            proposal["valid"] is True, f"{sample_id} has no valid deployment proposal"
        )
        scan_frames = proposal["scan_frame_indices"]
        decision_frame = int(sample["decision_frame"])
        _require(
            isinstance(scan_frames, list)
            and bool(scan_frames)
            and all(
                isinstance(frame, int) and not isinstance(frame, bool)
                for frame in scan_frames
            )
            and scan_frames == sorted(set(scan_frames))
            and all(0 <= frame < decision_frame for frame in scan_frames)
            and proposal["pose_frame_index"] == scan_frames[-1],
            f"{sample_id} proposal scans are not an exclusive causal prefix",
        )
        shortlist = proposal["shortlist"]
        universe = proposal["candidate_universe"]
        _require(
            isinstance(shortlist, list) and shortlist,
            f"{sample_id} has no deployment residual candidate",
        )
        _require(isinstance(universe, list), f"{sample_id} universe is malformed")
        universe_rows = [
            _validate_proposal_candidate(candidate, f"{sample_id}.universe[{item}]")
            for item, candidate in enumerate(universe)
        ]
        shortlist_rows = [
            _validate_proposal_candidate(candidate, f"{sample_id}.shortlist[{item}]")
            for item, candidate in enumerate(shortlist)
        ]
        for candidate in universe_rows:
            context = candidate["context_frame_indices"]
            _require(
                isinstance(context, list)
                and bool(context)
                and all(
                    isinstance(frame, int)
                    and not isinstance(frame, bool)
                    and frame in scan_frames
                    for frame in context
                ),
                f"{sample_id} candidate context escapes the causal scan prefix",
            )
        universe_by_id = {
            str(candidate["candidate_id"]): candidate for candidate in universe_rows
        }
        _require(
            len(universe_by_id) == len(universe_rows),
            f"{sample_id} universe candidate ids are duplicated",
        )
        shortlist_ids = [str(candidate["candidate_id"]) for candidate in shortlist_rows]
        _require(
            len(shortlist_ids) == len(set(shortlist_ids)),
            f"{sample_id} shortlist candidate ids are duplicated",
        )
        for candidate in shortlist_rows:
            candidate_id = str(candidate["candidate_id"])
            _require(
                candidate_id in universe_by_id, f"{sample_id} shortlist not in universe"
            )
            _require(
                canonical_json_bytes(candidate)
                == canonical_json_bytes(universe_by_id[candidate_id]),
                f"{sample_id} shortlist candidate differs from frozen universe",
            )
        _require(
            proposal["shortlist_count"] == len(shortlist_rows)
            and proposal["nms_candidate_count"] == len(universe_rows),
            f"{sample_id} proposal counts disagree with candidate tables",
        )
        _validate_proxy(
            deployment["proposal_proxy"],
            proposal,
            set(universe_by_id),
            f"{sample_id}.deployment.proxy",
        )
        result[str(sample_id)] = row
    _require(
        set(result) == set(manifest_samples),
        "proposal records have missing or extra manifest decisions",
    )
    return result


def _relation_index(
    artifact: Mapping[str, Any] | None,
    *,
    manifest_sha256: str,
    proposal_sha256: str,
    samples: Mapping[str, Mapping[str, Any]],
    proposals: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], dict[str, int]]:
    if artifact is None:
        return {}, {field: 1 for field in RELATION_FIELDS}
    _exact_mapping(
        artifact,
        frozenset(
            {
                "schema_version",
                "input_manifest_sha256",
                "input_proposal_sha256",
                "producer_source_sha256",
                "configuration_sha256",
                "feature_shapes",
                "records",
            }
        ),
        "relation artifact",
    )
    _require(
        artifact["schema_version"] == RELATION_ARTIFACT_SCHEMA,
        "unsupported relation artifact schema",
    )
    _require(
        artifact["input_manifest_sha256"] == manifest_sha256
        and artifact["input_proposal_sha256"] == proposal_sha256,
        "relation artifact input binding changed",
    )
    _valid_sha(artifact["producer_source_sha256"], "relation producer SHA")
    _valid_sha(artifact["configuration_sha256"], "relation configuration SHA")
    shapes_raw = _exact_mapping(
        artifact["feature_shapes"],
        frozenset(RELATION_FIELDS),
        "relation feature shapes",
    )
    shapes = {}
    for field in RELATION_FIELDS:
        value = shapes_raw[field]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"relation shape {field} must be a positive integer",
        )
        shapes[field] = value
    records = artifact["records"]
    _require(isinstance(records, list), "relation records must be a list")
    result = {}
    for index, raw in enumerate(records):
        row = _exact_mapping(
            raw,
            frozenset(
                {
                    "sample_id",
                    "prefix_sha256",
                    "goal_sha256",
                    "candidates",
                }
            ),
            f"relation record {index}",
        )
        sample_id = row["sample_id"]
        _require(sample_id in samples, "relation record references unknown sample")
        _require(sample_id not in result, f"duplicate relation record {sample_id}")
        sample = samples[str(sample_id)]
        _require(
            row["prefix_sha256"] == sample["causal_prefix"]["causal_prefix_sha256"]
            and row["goal_sha256"] == sample["goal"]["content_sha256"],
            f"relation record {sample_id} changed prefix/goal",
        )
        candidates = row["candidates"]
        _require(isinstance(candidates, list), "relation candidates must be a list")
        by_id = {}
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _exact_mapping(
                raw_candidate,
                RELATION_CANDIDATE_KEYS,
                f"relation {sample_id}.candidates[{candidate_index}]",
            )
            candidate_id = candidate["candidate_id"]
            _require(
                isinstance(candidate_id, str) and candidate_id not in by_id,
                f"relation {sample_id} candidate id is duplicated/malformed",
            )
            for field in RELATION_FIELDS:
                present = candidate[f"{field}_present"]
                _require(type(present) is bool, f"relation {field} mask is not boolean")
                vector = _finite_vector(
                    candidate[field], shapes[field], f"relation {sample_id}.{field}"
                )
                _require(
                    present or all(value == 0.0 for value in vector),
                    f"relation {sample_id}.{field} must be zero when absent",
                )
            for field in ("pose_uncertainty_present", "depth_confidence_present"):
                _require(
                    type(candidate[field]) is bool, f"relation {field} must be boolean"
                )
            pose_t = _finite_float(
                candidate["pose_translation_p90_m"],
                "pose translation",
                nonnegative=True,
            )
            pose_yaw = _finite_float(
                candidate["pose_yaw_p90_deg"], "pose yaw", nonnegative=True
            )
            depth = _finite_float(
                candidate["depth_confidence_mean"], "depth confidence", nonnegative=True
            )
            _require(depth <= 1.0, "depth confidence must be in [0,1]")
            _require(
                candidate["pose_uncertainty_present"]
                or (pose_t == 0.0 and pose_yaw == 0.0),
                "absent pose uncertainty must be zero",
            )
            _require(
                candidate["depth_confidence_present"] or depth == 0.0,
                "absent depth confidence must be zero",
            )
            by_id[str(candidate_id)] = candidate
        proposal = proposals[str(sample_id)]["arms"][DEPLOYMENT_ARM]["proposal"]
        expected_ids = {
            str(candidate["candidate_id"]) for candidate in proposal["shortlist"]
        }
        _require(
            set(by_id) == expected_ids,
            f"relation candidate coverage differs for {sample_id}",
        )
        result[str(sample_id)] = by_id
    _require(set(result) == set(samples), "relation records are missing or extra")
    return result, shapes


def _neutral_labels() -> dict[str, object]:
    return {
        "geodesic_progress_h8_m": 0.0,
        "geodesic_progress_h24_m": 0.0,
        "advantage_h24_m": 0.0,
        "harm": False,
        "useful": False,
        "reachable": False,
        "collision_h8": False,
        "regression_h24": False,
        "proposal_proxy_progress_m": 0.0,
        "proposal_proxy_reachable": False,
        "proposal_proxy_positive": False,
        "proposal_proxy_label_valid": False,
        "rollout_label_valid": False,
        "teacher_covisibility": 0.0,
        "covisibility_label_valid": False,
        "pose_residual_forward_m": 0.0,
        "pose_residual_left_m": 0.0,
        "pose_residual_yaw_rad": 0.0,
        "pose_label_valid": False,
    }


def _onehot(candidate_type: str) -> list[float]:
    return [float(kind == candidate_type) for kind in CANDIDATE_TYPES]


def _empty_features(
    candidate_type: str,
    relation_shapes: Mapping[str, int],
) -> dict[str, object]:
    return {
        "candidate_type_onehot": _onehot(candidate_type),
        "goal_patch_relation": [0.0] * relation_shapes["goal_patch_relation"],
        "goal_temporal_relation": [0.0] * relation_shapes["goal_temporal_relation"],
        "local_map_relation": [0.0] * relation_shapes["local_map_relation"],
        "native_proposal_relation": [0.0] * 21,
        "feature_presence_mask": [0.0] * FEATURE_PRESENCE_MASK_SIZE,
        "subgoal_forward_m": 0.0,
        "subgoal_left_m": 0.0,
        "graph_path_m": 0.0,
        "graph_hops": 0.0,
        "frontier_boundary_m": 0.0,
        "frontier_novelty_m": 0.0,
        "pose_translation_p90_m": 0.0,
        "pose_yaw_p90_deg": 0.0,
        "depth_confidence_mean": 0.0,
        "clearance_lower_m": 0.0,
    }


def _residual_features(
    candidate: Mapping[str, Any],
    relation: Mapping[str, Any] | None,
    relation_shapes: Mapping[str, int],
) -> dict[str, object]:
    features = _empty_features("frontier", relation_shapes)
    mask = features["feature_presence_mask"]
    assert isinstance(mask, list)
    if relation is None:
        patch_present = bool(candidate["goal_patch_relation_present"])
        features["goal_patch_relation"] = [
            float(candidate["goal_patch_relation_score"]) if patch_present else 0.0
        ]
        mask[0] = float(patch_present)
    else:
        for index, field in enumerate(RELATION_FIELDS):
            features[field] = [float(value) for value in relation[field]]
            mask[index] = float(bool(relation[f"{field}_present"]))
        features["pose_translation_p90_m"] = float(relation["pose_translation_p90_m"])
        features["pose_yaw_p90_deg"] = float(relation["pose_yaw_p90_deg"])
        features["depth_confidence_mean"] = float(relation["depth_confidence_mean"])
        mask[4] = float(bool(relation["pose_uncertainty_present"]))
        mask[5] = float(bool(relation["depth_confidence_present"]))
    features.update(
        {
            "subgoal_forward_m": float(candidate["subgoal_forward_m"]),
            "subgoal_left_m": float(candidate["subgoal_left_m"]),
            "frontier_boundary_m": float(candidate["frontier_boundary_m"]),
            "frontier_novelty_m": float(candidate["frontier_novelty_m"]),
            "clearance_lower_m": float(candidate["clearance_lower_m"]),
        }
    )
    mask[6] = 1.0
    return features


def _goal_epoch(sample: Mapping[str, Any]) -> str:
    role = sample.get("goal_role")
    if role is None:
        role = "B"
    _require(role in ("B", "C"), f"unsupported goal role {role!r}")
    goal_sha = str(sample["goal"]["content_sha256"])
    return f"{role}:{goal_sha[:16]}"


def build_precollection_records(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    evaluation_geometry_manifest: Mapping[str, Any] | None = None,
    evaluation_geometry_manifest_sha256: str | None = None,
    proposal_artifact: Mapping[str, Any],
    proposal_sha256: str,
    source_policy_sha256: str,
    relation_artifact: Mapping[str, Any] | None = None,
    relation_sha256: str | None = None,
    expected_rollout_labeler_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic strictly neutral precollection records."""
    manifest_sha = _valid_sha(manifest_sha256, "manifest SHA")
    proposal_sha = _valid_sha(proposal_sha256, "proposal SHA")
    policy_sha = _valid_sha(source_policy_sha256, "source policy SHA")
    _require(
        _sha256_bytes(canonical_json_bytes(manifest)) == manifest_sha,
        "in-memory manifest differs from its canonical SHA pin",
    )
    _require(
        _sha256_bytes(canonical_json_bytes(proposal_artifact)) == proposal_sha,
        "in-memory proposal artifact differs from its canonical SHA pin",
    )
    if relation_artifact is None:
        _require(relation_sha256 is None, "relation SHA supplied without artifact")
        relation_sha = ZERO_SHA
    else:
        relation_sha = _valid_sha(relation_sha256, "relation artifact SHA")
        _require(
            _sha256_bytes(canonical_json_bytes(relation_artifact)) == relation_sha,
            "in-memory relation artifact differs from its canonical SHA pin",
        )
    labeler_sha = rollout_labeler_code_sha256()
    if expected_rollout_labeler_sha256 is not None:
        _require(
            labeler_sha
            == _valid_sha(
                expected_rollout_labeler_sha256, "expected rollout labeler SHA"
            ),
            "local rollout-labeler code bundle differs from its pin",
        )
    samples, _causal_scenes, groups = _validate_manifest(manifest)
    geometry_scenes, geometry_lineage = _validate_evaluation_geometry_manifest(
        causal_manifest=manifest,
        causal_manifest_sha256=manifest_sha,
        evaluation_manifest=evaluation_geometry_manifest,
        evaluation_manifest_sha256=evaluation_geometry_manifest_sha256,
    )
    proposals = _proposal_index(proposal_artifact, manifest_sha, samples)
    relations, relation_shapes = _relation_index(
        relation_artifact,
        manifest_sha256=manifest_sha,
        proposal_sha256=proposal_sha,
        samples=samples,
        proposals=proposals,
    )
    split = manifest["split"]
    builder_sha = _sha256_file(Path(__file__))
    dataset_digest = _sha256_bytes(
        canonical_json_bytes(
            {
                "schema": BUILDER_SCHEMA,
                "manifest_sha256": manifest_sha,
                "evaluation_geometry_manifest_sha256": geometry_lineage[
                    "evaluation_geometry_manifest_sha256"
                ],
                "geometry_lineage": geometry_lineage,
                "proposal_sha256": proposal_sha,
                "relation_sha256": relation_sha,
                "source_policy_sha256": policy_sha,
                "feature_builder_sha256": builder_sha,
                "rollout_labeler_sha256": labeler_sha,
            }
        )
    )
    records = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        proposal_record = proposals[sample_id]
        deployment = proposal_record["arms"][DEPLOYMENT_ARM]
        proposal = deployment["proposal"]
        proxy_by_id, proxy_summary = _validate_proxy(
            deployment["proposal_proxy"],
            proposal,
            {
                str(candidate["candidate_id"])
                for candidate in proposal["candidate_universe"]
            },
            f"{sample_id}.deployment.proxy",
        )
        relation_by_id = relations.get(sample_id, {})
        residuals = []
        for candidate in proposal["shortlist"]:
            candidate_id = str(candidate["candidate_id"])
            labels = _neutral_labels()
            if proxy_summary["valid"]:
                proxy = proxy_by_id[candidate_id]
                labels.update(
                    {
                        "proposal_proxy_progress_m": float(proxy["progress_m"]),
                        "proposal_proxy_reachable": bool(proxy["reachable"]),
                        "proposal_proxy_positive": bool(proxy["positive"]),
                        "proposal_proxy_label_valid": True,
                    }
                )
            residuals.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_type": "frontier",
                    "features": _residual_features(
                        candidate, relation_by_id.get(candidate_id), relation_shapes
                    ),
                    "labels": labels,
                }
            )
        scene_id = str(sample["scene"])
        scene = geometry_scenes[scene_id]
        set_features = {
            "feature_presence_mask": [0.0] * SET_FEATURE_PRESENCE_MASK_SIZE,
            "native_stagnation_plans": 0,
            "graph_node_count": 0,
            "graph_edge_count": 0,
            "graph_age_frames": 0,
            "memory_candidate_count": 0,
            "frontier_candidate_count": len(residuals),
        }
        set_features["feature_presence_mask"][4] = 1.0
        set_features["feature_presence_mask"][5] = 1.0
        native = {
            "candidate_id": "native",
            "candidate_type": "native",
            "features": _empty_features("native", relation_shapes),
            "labels": _neutral_labels(),
        }
        dustbin = {
            "candidate_id": "dustbin",
            "candidate_type": "dustbin",
            "features": _empty_features("dustbin", relation_shapes),
            "labels": _neutral_labels(),
        }
        records.append(
            {
                "schema_version": CANDIDATE_SET_SCHEMA,
                "provenance": {
                    "dataset_id": f"nlsr-precollection:{dataset_digest}",
                    "scene_id": scene_id,
                    "episode_id": str(sample["source_episode_id"]),
                    "session_id": str(sample["source_episode_id"]),
                    "group_id": groups[sample_id],
                    "goal_epoch": _goal_epoch(sample),
                    "state_id": sample_id,
                    "state_source": str(sample["state_source"]),
                    "goal_source_episode_id": str(sample["goal_source_episode_id"]),
                    "plan_index": int(sample["decision_frame"]),
                    "prefix_frames": int(sample["decision_frame"]),
                    "prefix_sha256": str(
                        sample["causal_prefix"]["causal_prefix_sha256"]
                    ),
                    "goal_sha256": str(sample["goal"]["content_sha256"]),
                    "navdp_fifo_sha256": str(sample["navdp_fifo"]["fifo_sha256"]),
                    "split_role": str(sample["split_role"]),
                    "split_sha256": str(split["sha256"]),
                    "source_policy_sha256": policy_sha,
                    "candidate_generator_sha256": proposal_sha,
                    "feature_builder_sha256": builder_sha,
                    "rollout_labeler_sha256": labeler_sha,
                    "environment_id": scene_id,
                    "navmesh_sha256": str(scene["navmesh"]["content_sha256"]),
                },
                "set_features": set_features,
                "candidates": [native, *residuals, dustbin],
                "set_labels": {
                    "global_match": False,
                    "strict_no_match": False,
                    "ambiguous": True,
                    "candidate_set_has_positive": False,
                    "candidate_universe_has_positive": False,
                    "candidate_coverage_miss": False,
                    "coverage_label_valid": False,
                    "proposal_proxy_set_has_positive": bool(proxy_summary["shortlist"]),
                    "proposal_proxy_universe_has_positive": bool(
                        proxy_summary["universe"]
                    ),
                    "proposal_proxy_coverage_miss": bool(proxy_summary["miss"]),
                    "proposal_proxy_coverage_label_valid": bool(proxy_summary["valid"]),
                    "oracle_best_candidate_id": "dustbin",
                },
            }
        )
    # The public schema requires a valid native rollout.  For the documented
    # precollection exception, validate a deep copy with the same zero native
    # progress; the serialized rows remain strictly neutral.
    validation_rows = copy.deepcopy(records)
    for row in validation_rows:
        row["candidates"][0]["labels"]["rollout_label_valid"] = True
        row["candidates"][0]["labels"]["reachable"] = True
    try:
        validate_candidate_dataset(validation_rows)
    except Exception as error:
        raise PrecollectionBuildError(
            f"generated candidate dataset is invalid: {error}"
        ) from error
    return records


def _output_bytes(records: Sequence[Mapping[str, Any]], output: Path) -> bytes:
    if output.suffix.lower() == ".jsonl":
        return canonical_jsonl_bytes(records)
    return canonical_json_bytes(list(records))


def write_candidate_records(
    records: Sequence[Mapping[str, Any]],
    output: Path,
    sha_output: Path | None = None,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Atomically write canonical JSON/JSONL and an exact SHA sidecar."""
    _require(not (resume and overwrite), "resume and overwrite are mutually exclusive")
    sidecar = (
        output.with_suffix(output.suffix + ".sha256")
        if sha_output is None
        else sha_output
    )
    _require(output.resolve() != sidecar.resolve(), "output and SHA path must differ")
    payload = _output_bytes(records, output)
    digest = _sha256_bytes(payload)
    sha_payload = f"{digest}  {output.name}\n".encode("ascii")
    exists = output.exists(), sidecar.exists()
    if resume:
        _require(exists == (True, True), "resume requires complete output and sidecar")
        _require(
            output.read_bytes() == payload and sidecar.read_bytes() == sha_payload,
            "resume output differs from deterministic rebuilt content",
        )
        return "resumed", digest
    _require(overwrite or not any(exists), "output exists without --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    for destination, content in ((output, payload), (sidecar, sha_payload)):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return "written", digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--evaluation-geometry-manifest", type=Path)
    parser.add_argument("--expected-evaluation-geometry-manifest-sha")
    parser.add_argument("--proposal-artifact", type=Path, required=True)
    parser.add_argument("--expected-proposal-sha", required=True)
    parser.add_argument("--source-policy-sha", required=True)
    parser.add_argument("--relation-artifact", type=Path)
    parser.add_argument("--expected-relation-sha")
    parser.add_argument("--expected-rollout-labeler-sha")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha-out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(
        (args.relation_artifact is None) == (args.expected_relation_sha is None),
        "relation artifact and expected SHA must be supplied together",
    )
    _require(
        (args.evaluation_geometry_manifest is None)
        == (args.expected_evaluation_geometry_manifest_sha is None),
        "evaluation geometry manifest and expected SHA must be supplied together",
    )
    manifest = load_pinned_canonical_json(args.manifest, args.expected_manifest_sha)
    evaluation_geometry_manifest = (
        load_pinned_canonical_json(
            args.evaluation_geometry_manifest,
            args.expected_evaluation_geometry_manifest_sha,
        )
        if args.evaluation_geometry_manifest is not None
        else None
    )
    proposal = load_pinned_canonical_json(
        args.proposal_artifact, args.expected_proposal_sha
    )
    relation = (
        load_pinned_canonical_json(args.relation_artifact, args.expected_relation_sha)
        if args.relation_artifact is not None
        else None
    )
    records = build_precollection_records(
        manifest=manifest,
        manifest_sha256=args.expected_manifest_sha,
        evaluation_geometry_manifest=evaluation_geometry_manifest,
        evaluation_geometry_manifest_sha256=(
            args.expected_evaluation_geometry_manifest_sha
        ),
        proposal_artifact=proposal,
        proposal_sha256=args.expected_proposal_sha,
        source_policy_sha256=args.source_policy_sha,
        relation_artifact=relation,
        relation_sha256=args.expected_relation_sha,
        expected_rollout_labeler_sha256=args.expected_rollout_labeler_sha,
    )
    status, digest = write_candidate_records(
        records, args.out, args.sha_out, resume=args.resume, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "schema_version": BUILDER_SCHEMA,
                "status": status,
                "records": len(records),
                "output_sha256": digest,
                "rollout_labeler_sha256": rollout_labeler_code_sha256(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
