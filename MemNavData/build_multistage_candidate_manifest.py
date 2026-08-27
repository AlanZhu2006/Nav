#!/usr/bin/env python3
"""Extend an audited routed NLSR manifest with a causal Goal-C stage.

The routed v2 manifest is the immutable source of the original Goal-B rows.
Every source row is reconstructed from raw episode files and compared before
being copied without adding, removing, or changing a field.  Goal-C rows are
then added at the second switch using only the exclusive causal prefix, the
corresponding reconstructed NavDP FIFO, and ``goal_2.jpg``.  Goal pose,
geodesic distance, future observations, model output, and rollout labels are
not candidate features in this artifact.

Legacy generated episodes omitted ``camera_height_m``.  That omission is
never repaired with an implicit constant: callers must supply an explicit
``--legacy-camera-height-m`` and the resulting value/source binding is part of
the canonical artifact.  If metadata contains a different value, the build
fails closed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    from MemNavData.build_novel_candidate_manifest import (
        ALLOWED_ROLES,
        METADATA_RELATIVE,
        ROUTED_SCHEMA_VERSION,
        ManifestError as SourceManifestError,
        canonical_json_bytes,
        load_valid_episode,
        navdp_fifo_record,
        prefix_record,
        relative_file_record,
        sha256_bytes,
        sha256_file,
        write_artifact,
    )
except ImportError:  # Direct ``python MemNavData/<script>.py`` execution.
    from build_novel_candidate_manifest import (  # type: ignore
        ALLOWED_ROLES,
        METADATA_RELATIVE,
        ROUTED_SCHEMA_VERSION,
        ManifestError as SourceManifestError,
        canonical_json_bytes,
        load_valid_episode,
        navdp_fifo_record,
        prefix_record,
        relative_file_record,
        sha256_bytes,
        sha256_file,
        write_artifact,
    )


SCHEMA_VERSION = "nlsr_v2_multistage_expert_candidate_manifest_v1"
EXPECTED_AUDIT_SCHEMA = "nlsr_corrected_manifest_audit_v1"
EXPECTED_AUDIT_STATUS = "corrected_manifest_audited"
GOAL_C_STATE_NAME = "goal_c_t0"
SOURCE_SAMPLE_KEYS = frozenset({
    "sample_id",
    "split_role",
    "scene",
    "state_source",
    "source_episode",
    "source_episode_id",
    "goal_episode",
    "goal_source_episode_id",
    "goal_variant",
    "goal_role",
    "state_name",
    "decision_frame",
    "state_frame",
    "causal_prefix",
    "navdp_fifo",
    "goal",
})


class MultistageManifestError(RuntimeError):
    """A source, isolation, causality, or provenance contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultistageManifestError(message)


def _object_without_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise MultistageManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultistageManifestError(f"{label} is not valid UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise MultistageManifestError(
            f"{label} is not finite canonical JSON") from exc
    _require(payload == canonical, f"{label} is not canonical JSON")
    return value


def _validate_sha256(value: str, label: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )


def load_pinned_canonical_artifact(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict, str]:
    """Load one exact canonical JSON input, rejecting stale or loose pins."""
    _validate_sha256(expected_sha256, f"expected {label} SHA256")
    _require(path.is_file(), f"{label} is missing: {path}")
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    _require(
        actual == expected_sha256,
        f"{label} SHA256 mismatch: {actual} != {expected_sha256}",
    )
    return _parse_canonical_json(payload, label), actual


def _require_mapping_matches_pinned_artifact(
    value: Mapping[str, object],
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    disk_value, _ = load_pinned_canonical_artifact(
        path, expected_sha256, label=label)
    _require(
        canonical_json_bytes(value) == canonical_json_bytes(disk_value),
        f"in-memory {label} differs from its pinned canonical file",
    )


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise MultistageManifestError(f"{label} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MultistageManifestError(f"{label} must be numeric") from exc
    _require(
        math.isfinite(result) and result > 0.0,
        f"{label} must be finite and positive",
    )
    return result


def _load_metadata(path: Path) -> dict:
    _require(path.is_file(), f"episode metadata is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultistageManifestError(
            f"episode metadata is invalid: {path}") from exc
    _require(isinstance(value, dict), f"episode metadata is not an object: {path}")
    return value


def _validate_source_and_audit(
    source: Mapping[str, object],
    source_sha256: str,
    audit: Mapping[str, object],
) -> None:
    _require(
        source.get("schema_version") == ROUTED_SCHEMA_VERSION,
        "input manifest must be the provenance-routed v2 schema",
    )
    _require(
        isinstance(source.get("flow_cache_routing"), dict),
        "routed input manifest lacks flow_cache_routing",
    )
    roots = source.get("input_roots")
    _require(isinstance(roots, dict), "input manifest roots are malformed")
    _require(
        "flow_cache_root" not in roots,
        "routed input must not claim a single flow_cache_root",
    )
    _require(
        source.get("missing_flow_caches") == [],
        "audited routed input must have no missing flow caches",
    )
    summary = source.get("summary")
    _require(isinstance(summary, dict), "input manifest summary is malformed")
    _require(
        summary.get("all_flow_caches_complete") is True,
        "audited routed input has incomplete flow caches",
    )

    _require(
        audit.get("schema_version") == EXPECTED_AUDIT_SCHEMA,
        "input audit schema is not the corrected-manifest audit",
    )
    _require(
        audit.get("status") == EXPECTED_AUDIT_STATUS,
        "input audit status is not corrected_manifest_audited",
    )
    _require(
        audit.get("downstream_ready") is True,
        "input audit is not downstream-ready",
    )
    _require(
        audit.get("manifest_sha256") == source_sha256,
        "input audit does not bind the exact routed manifest",
    )
    _require(
        audit.get("summary") == summary,
        "input audit summary differs from the routed manifest",
    )
    split = source.get("split")
    _require(isinstance(split, dict), "input split record is malformed")
    _require(
        audit.get("split_sha256") == split.get("sha256"),
        "input audit split SHA differs from the routed manifest",
    )


def _record_matches(actual: object, expected: object, label: str) -> None:
    _require(
        canonical_json_bytes(actual) == canonical_json_bytes(expected),
        f"{label} differs from recomputed raw episode content",
    )


def _source_sample(
    *,
    role: str,
    scene: str,
    source: Mapping[str, object],
    goal_episode: Mapping[str, object],
    goal_variant: str,
    state_name: str,
    decision_frame: int,
    episode_root: Path,
    file_hashes: dict[Path, dict],
    goal_key: str,
    goal_role: str,
) -> dict:
    prefix = prefix_record(source, episode_root, decision_frame, file_hashes)
    fifo = navdp_fifo_record(source, episode_root, decision_frame, file_hashes)
    rgb_root = source.get("rgb_root")
    _require(isinstance(rgb_root, Path), "internal RGB root is malformed")
    state_frame_path = rgb_root / f"{decision_frame - 1}.jpg"
    state_frame = file_hashes.get(state_frame_path)
    if state_frame is None:
        state_frame = relative_file_record(state_frame_path, episode_root)
        file_hashes[state_frame_path] = state_frame
    goal_path = goal_episode.get(goal_key)
    _require(isinstance(goal_path, Path), f"internal {goal_key} path is malformed")
    goal = relative_file_record(goal_path, episode_root)
    return {
        "sample_id": (
            f"{role}/{scene}/{source['name']}/{state_name}/{goal_variant}"
        ),
        "split_role": role,
        "scene": scene,
        "state_source": "expert",
        "source_episode": str(source["name"]),
        "source_episode_id": f"{scene}/{source['name']}",
        "goal_episode": str(goal_episode["name"]),
        "goal_source_episode_id": f"{scene}/{goal_episode['name']}",
        "goal_variant": goal_variant,
        "goal_role": goal_role,
        "state_name": state_name,
        "decision_frame": decision_frame,
        "state_frame": state_frame,
        "causal_prefix": prefix,
        "navdp_fifo": fifo,
        "goal": goal,
    }


def _group_binding(sample: Mapping[str, object]) -> dict:
    role = str(sample["split_role"])
    scene = str(sample["scene"])
    source_episode = str(sample["source_episode"])
    state_name = str(sample["state_name"])
    return {
        "sample_id": sample["sample_id"],
        "split_role": role,
        "scene": scene,
        "scene_group_id": f"{role}/{scene}",
        "episode_group_id": f"{role}/{scene}/{source_episode}",
        "counterfactual_pair_group_id": (
            f"{role}/{scene}/{source_episode}/{state_name}"
        ),
    }


def _validate_isolation(
    samples: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
) -> None:
    ids = [sample.get("sample_id") for sample in samples]
    _require(None not in ids and len(ids) == len(set(ids)), "sample IDs are not unique")
    _require(len(groups) == len(samples), "sample/group row count differs")
    by_id = {sample["sample_id"]: sample for sample in samples}
    group_ids = set()
    pair_members: dict[str, list[Mapping[str, object]]] = {}
    for binding in groups:
        sample_id = binding.get("sample_id")
        _require(sample_id in by_id, "group binding names an unknown sample")
        _require(sample_id not in group_ids, "sample has duplicate group bindings")
        group_ids.add(sample_id)
        sample = by_id[sample_id]
        role = sample.get("split_role")
        scene = sample.get("scene")
        _require(role in ALLOWED_ROLES, f"forbidden split role: {role}")
        _require(
            binding.get("split_role") == role and binding.get("scene") == scene,
            "group binding crosses split or scene",
        )
        expected_prefix = f"{role}/{scene}/"
        for key in ("episode_group_id", "counterfactual_pair_group_id"):
            value = binding.get(key)
            _require(
                isinstance(value, str) and value.startswith(expected_prefix),
                f"{key} is not split/scene qualified",
            )
        pair_members.setdefault(
            str(binding["counterfactual_pair_group_id"]), []).append(sample)
    _require(group_ids == set(ids), "some samples lack group bindings")
    for pair_id, members in pair_members.items():
        variants = {member.get("goal_variant") for member in members}
        _require(
            len(members) == 2 and variants == {"factual", "counterfactual"},
            f"counterfactual group is incomplete: {pair_id}",
        )
        _require(
            len({member.get("split_role") for member in members}) == 1
            and len({member.get("scene") for member in members}) == 1
            and len({member.get("source_episode_id") for member in members}) == 1
            and len({member.get("state_name") for member in members}) == 1,
            f"counterfactual group crosses an isolation boundary: {pair_id}",
        )


def build_multistage_manifest(
    *,
    routed_manifest: Mapping[str, object],
    routed_manifest_path: Path,
    routed_manifest_sha256: str,
    routed_manifest_audit: Mapping[str, object],
    routed_manifest_audit_path: Path,
    routed_manifest_audit_sha256: str,
    legacy_camera_height_m: float | None,
) -> dict:
    """Validate/copy Goal-B rows and append causal Goal-C t0 row pairs."""
    _validate_sha256(routed_manifest_sha256, "routed manifest SHA256")
    _validate_sha256(routed_manifest_audit_sha256, "routed manifest audit SHA256")
    _require_mapping_matches_pinned_artifact(
        routed_manifest,
        routed_manifest_path,
        routed_manifest_sha256,
        label="routed manifest",
    )
    _require_mapping_matches_pinned_artifact(
        routed_manifest_audit,
        routed_manifest_audit_path,
        routed_manifest_audit_sha256,
        label="routed manifest audit",
    )
    _validate_source_and_audit(
        routed_manifest, routed_manifest_sha256, routed_manifest_audit)
    if legacy_camera_height_m is not None:
        legacy_camera_height_m = _positive_float(
            legacy_camera_height_m, "legacy_camera_height_m")

    roots = routed_manifest.get("input_roots")
    assert isinstance(roots, dict)  # validated above
    episode_root_raw = roots.get("episode_root")
    _require(
        isinstance(episode_root_raw, str) and episode_root_raw,
        "input manifest episode_root is absent",
    )
    episode_root = Path(episode_root_raw)
    _require(episode_root.is_dir(), f"episode_root is not available: {episode_root}")

    scenes_raw = routed_manifest.get("scenes")
    source_samples_raw = routed_manifest.get("samples")
    _require(isinstance(scenes_raw, list), "input scenes are malformed")
    _require(isinstance(source_samples_raw, list), "input samples are malformed")
    _require(
        all(isinstance(row, dict) for row in source_samples_raw),
        "input sample is not an object",
    )

    recomputed_b_samples = []
    c_samples = []
    camera_height_bindings = []
    file_hashes: dict[Path, dict] = {}
    seen_scenes = set()
    for scene_record in scenes_raw:
        _require(isinstance(scene_record, dict), "input scene is not an object")
        scene = scene_record.get("scene")
        role = scene_record.get("split_role")
        _require(isinstance(scene, str) and scene, "input scene name is invalid")
        _require(role in ALLOWED_ROLES, f"forbidden scene split role: {role}")
        _require(scene not in seen_scenes, f"duplicate input scene: {scene}")
        seen_scenes.add(scene)
        selected_raw = scene_record.get("selected_episodes")
        _require(
            isinstance(selected_raw, list)
            and len(selected_raw) == 2
            and all(isinstance(row, dict) for row in selected_raw),
            f"scene {scene} must contain exactly two selected episodes",
        )

        episodes = []
        for episode_record in selected_raw:
            episode_name = episode_record.get("episode")
            _require(
                isinstance(episode_name, str) and episode_name,
                f"scene {scene} has an invalid episode name",
            )
            episode_path = episode_root / scene / episode_name
            try:
                episode = load_valid_episode(episode_path, scene)
            except SourceManifestError as exc:
                raise MultistageManifestError(
                    f"source episode no longer validates: {scene}/{episode_name}: {exc}"
                ) from exc
            _require(
                str(episode["name"]) == episode_name,
                f"episode name changed: {scene}/{episode_name}",
            )
            expected_episode_fields = {
                "n_frames": int(episode["n_frames"]),
                "switches": [
                    int(episode["switch_a"]), int(episode["switch_b"]),
                ],
                "goal_b_midpoint_frame": int(episode["midpoint"]),
                "metadata": relative_file_record(
                    episode["metadata"], episode_root),
                "parquet": relative_file_record(
                    episode["parquet"], episode_root),
                "goal_b": relative_file_record(
                    episode["goal_b"], episode_root),
                "goal_c": relative_file_record(
                    episode["goal_c"], episode_root),
            }
            for key, expected in expected_episode_fields.items():
                _record_matches(
                    episode_record.get(key), expected,
                    f"{scene}/{episode_name}.{key}",
                )

            metadata_path = episode_path / METADATA_RELATIVE
            metadata = _load_metadata(metadata_path)
            goals = metadata.get("goals")
            _require(
                isinstance(goals, list)
                and len(goals) == 2
                and isinstance(goals[1], dict)
                and goals[1].get("kind") == "revisit",
                f"Goal C metadata must be goals[1].kind=revisit: {scene}/{episode_name}",
            )
            if "camera_height_m" not in metadata:
                _require(
                    legacy_camera_height_m is not None,
                    "episode metadata omits camera_height_m; CLI must supply "
                    "--legacy-camera-height-m explicitly (no default)",
                )
                camera_height = legacy_camera_height_m
                height_source = "explicit_cli:--legacy-camera-height-m"
            else:
                camera_height = _positive_float(
                    metadata["camera_height_m"],
                    f"{scene}/{episode_name}.metadata.camera_height_m",
                )
                if legacy_camera_height_m is not None:
                    _require(
                        camera_height == legacy_camera_height_m,
                        "metadata camera_height_m conflicts with the explicit "
                        f"legacy value for {scene}/{episode_name}",
                    )
                height_source = "episode_metadata:meta/gen_meta.json#camera_height_m"
            assert camera_height is not None
            metadata_record = expected_episode_fields["metadata"]
            camera_height_bindings.append({
                "split_role": role,
                "scene": scene,
                "episode": episode_name,
                "episode_id": f"{scene}/{episode_name}",
                "camera_height_m": camera_height,
                "value_source": height_source,
                "metadata_content_sha256": metadata_record["content_sha256"],
            })
            episodes.append(episode)

        for source_index, source_episode in enumerate(episodes):
            partner = episodes[1 - source_index]
            b_specs = (
                ("goal_b_t0", int(source_episode["switch_a"])),
                ("goal_b_midpoint_t1", int(source_episode["midpoint"])),
            )
            for state_name, decision_frame in b_specs:
                for goal_variant, goal_episode in (
                    ("factual", source_episode),
                    ("counterfactual", partner),
                ):
                    recomputed_b_samples.append(_source_sample(
                        role=str(role),
                        scene=scene,
                        source=source_episode,
                        goal_episode=goal_episode,
                        goal_variant=goal_variant,
                        state_name=state_name,
                        decision_frame=decision_frame,
                        episode_root=episode_root,
                        file_hashes=file_hashes,
                        goal_key="goal_b",
                        goal_role="B",
                    ))
            for goal_variant, goal_episode in (
                ("factual", source_episode),
                ("counterfactual", partner),
            ):
                c_samples.append(_source_sample(
                    role=str(role),
                    scene=scene,
                    source=source_episode,
                    goal_episode=goal_episode,
                    goal_variant=goal_variant,
                    state_name=GOAL_C_STATE_NAME,
                    decision_frame=int(source_episode["switch_b"]),
                    episode_root=episode_root,
                    file_hashes=file_hashes,
                    goal_key="goal_c",
                    goal_role="C",
                ))

    source_samples = list(source_samples_raw)
    _require(
        len(source_samples) == len(recomputed_b_samples),
        "routed input has an unexpected number of Goal-B samples",
    )
    for index, (source_row, expected_row) in enumerate(
            zip(source_samples, recomputed_b_samples)):
        _require(
            set(source_row) == SOURCE_SAMPLE_KEYS,
            f"Goal-B sample {index} schema differs from routed v2",
        )
        _record_matches(source_row, expected_row, f"Goal-B sample {index}")

    # Deep-copying after exact reconstruction deliberately preserves each
    # original B dictionary's canonical bytes while preventing alias mutation.
    preserved_b_samples = copy.deepcopy(source_samples)
    samples = [*preserved_b_samples, *c_samples]
    groups = [_group_binding(sample) for sample in samples]
    _validate_isolation(samples, groups)

    camera_height_bindings.sort(
        key=lambda row: (str(row["split_role"]), str(row["scene"]),
                         str(row["episode"])))
    binding_sha = sha256_bytes(canonical_json_bytes(camera_height_bindings))
    source_b_sequence_sha = sha256_bytes(canonical_json_bytes(source_samples))
    preserved_b_sequence_sha = sha256_bytes(
        canonical_json_bytes(preserved_b_samples))
    _require(
        source_b_sequence_sha == preserved_b_sequence_sha,
        "internal error: Goal-B canonical byte semantics changed",
    )
    role_counts = {
        role: sum(sample["split_role"] == role for sample in samples)
        for role in ALLOWED_ROLES
    }
    c_role_counts = {
        role: sum(
            sample["split_role"] == role and sample["goal_role"] == "C"
            for sample in samples)
        for role in ALLOWED_ROLES
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "frozen causal Goal-B/Goal-C expert-state sampling for later "
            "candidate construction and paired rollout labels; contains no "
            "model output, rollout label, or privileged goal-pose feature"
        ),
        "source_routed_manifest": {
            "path": str(routed_manifest_path.resolve()),
            "sha256": routed_manifest_sha256,
            "schema_version": ROUTED_SCHEMA_VERSION,
            "goal_b_sample_count": len(preserved_b_samples),
            "goal_b_sample_sequence_sha256": source_b_sequence_sha,
            "goal_b_semantics": (
                "copied field-for-field after exact raw-prefix reconstruction"
            ),
        },
        "source_routed_manifest_audit": {
            "path": str(routed_manifest_audit_path.resolve()),
            "sha256": routed_manifest_audit_sha256,
            "schema_version": EXPECTED_AUDIT_SCHEMA,
            "status": EXPECTED_AUDIT_STATUS,
        },
        "configuration": {
            "goal_c_state": (
                "decision_frame=switches[1], exclusive causal prefix; "
                "state_frame=decision_frame-1"
            ),
            "goal_c_variants": ["factual", "counterfactual"],
            "goal_c_factual_rule": (
                "goal_2.jpg from the state source episode whose "
                "metadata goals[1].kind is revisit"
            ),
            "goal_c_counterfactual_rule": (
                "goal_2.jpg from the other selected episode in the same scene"
            ),
            "camera_height_policy": {
                "metadata_field": "camera_height_m",
                "legacy_camera_height_m": legacy_camera_height_m,
                "legacy_value_source": (
                    "explicit_cli:--legacy-camera-height-m"
                    if legacy_camera_height_m is not None else None
                ),
                "missing_metadata_policy": (
                    "require explicit CLI value; no default"
                ),
                "metadata_conflict_policy": (
                    "reject unless metadata value exactly equals explicit "
                    "legacy value"
                ),
            },
            "isolation_policy": {
                "split": "train/development from source scene split only",
                "scene": "counterfactual goals never cross a scene",
                "episode_group": "split/scene/source_episode",
                "state_pair_group": (
                    "split/scene/source_episode/state_name; exactly one "
                    "factual and one counterfactual row"
                ),
            },
            "feature_boundary": {
                "future_observation_as_feature": False,
                "future_parquet_as_feature": False,
                "goal_pose_as_feature": False,
                "geodesic_as_feature": False,
                "model_output_present": False,
                "rollout_label_present": False,
                "full_episode_access": (
                    "integrity/schema validation only; every state binds only "
                    "RGB/depth/parquet rows with index < decision_frame"
                ),
            },
        },
        "provenance": {
            "producer_source": {
                "path": Path(__file__).name,
                "content_sha256": sha256_file(Path(__file__)),
            },
            "camera_height_bindings": camera_height_bindings,
            "camera_height_bindings_sha256": binding_sha,
        },
        "split": copy.deepcopy(routed_manifest["split"]),
        "input_roots": copy.deepcopy(routed_manifest["input_roots"]),
        "flow_cache_routing": copy.deepcopy(
            routed_manifest["flow_cache_routing"]),
        "scenes": copy.deepcopy(scenes_raw),
        "samples": samples,
        "sample_group_bindings": groups,
        "summary": {
            "scene_count": len(seen_scenes),
            "episode_count": len(camera_height_bindings),
            "goal_b_sample_count": len(preserved_b_samples),
            "goal_c_sample_count": len(c_samples),
            "sample_count": len(samples),
            "sample_count_by_split": role_counts,
            "goal_c_sample_count_by_split": c_role_counts,
            "counterfactual_pair_group_count": len(samples) // 2,
            "legacy_camera_height_episode_count": sum(
                binding["value_source"].startswith("explicit_cli")
                for binding in camera_height_bindings
            ),
        },
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-manifest", type=Path, required=True)
    parser.add_argument("--expected-routed-manifest-sha", required=True)
    parser.add_argument("--routed-manifest-audit", type=Path, required=True)
    parser.add_argument("--expected-routed-manifest-audit-sha", required=True)
    parser.add_argument(
        "--legacy-camera-height-m",
        type=float,
        help=(
            "explicit historical camera height; required if any selected "
            "metadata omits camera_height_m (there is intentionally no default)"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha-out", type=Path, help="default: <out>.sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routed, routed_sha = load_pinned_canonical_artifact(
        args.routed_manifest,
        args.expected_routed_manifest_sha,
        label="routed manifest",
    )
    audit, audit_sha = load_pinned_canonical_artifact(
        args.routed_manifest_audit,
        args.expected_routed_manifest_audit_sha,
        label="routed manifest audit",
    )
    manifest = build_multistage_manifest(
        routed_manifest=routed,
        routed_manifest_path=args.routed_manifest,
        routed_manifest_sha256=routed_sha,
        routed_manifest_audit=audit,
        routed_manifest_audit_path=args.routed_manifest_audit,
        routed_manifest_audit_sha256=audit_sha,
        legacy_camera_height_m=args.legacy_camera_height_m,
    )
    sha_output = args.sha_out or Path(f"{args.out}.sha256")
    try:
        status, digest = write_artifact(
            manifest,
            args.out,
            sha_output,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except SourceManifestError as exc:
        raise MultistageManifestError(str(exc)) from exc
    print(json.dumps({
        "status": status,
        "output": str(args.out),
        "sha_output": str(sha_output),
        "sha256": digest,
        **manifest["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
