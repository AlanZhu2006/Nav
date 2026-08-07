#!/usr/bin/env python3
"""Strict training-time contract for external causal LingBot metric scale.

The scale estimator intentionally produces one estimate per source episode,
using only the first prefix before that episode's earliest selected decision.
Training, however, operates on individual candidate sets.  This module joins
those two granularities without guessing:

* a caller must name an exact ``manifest_sample_id``;
* the manifest, scale artifact, producer, configuration and model are SHA
  pinned;
* Goal B/C role, decision, causal prefix and goal bytes are rebound to that
  exact sample;
* every candidate/neighbor anchor must precede the decision; and
* invalid external estimates fail closed instead of falling back to a pooled
  or whole-episode scale.

No goal image, future frame, Habitat pose, Pathfinder result or teacher target
is used to estimate scale.  The only image hash performed here is an integrity
check against the already-pinned causal manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from MemNavData.phase_b_feature_schema import (
        EXTERNAL_CAUSAL_SCALE_SOURCE,
        EXTERNAL_SCALE_QUALITY_COLUMNS,
    )
except ModuleNotFoundError:  # direct script invocation
    from phase_b_feature_schema import (  # type: ignore
        EXTERNAL_CAUSAL_SCALE_SOURCE,
        EXTERNAL_SCALE_QUALITY_COLUMNS,
    )


CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION = "nlsr_v2_causal_ground_scale_v1"
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({
    "nlsr_v2_expert_candidate_manifest_v1",
    "nlsr_v2_expert_candidate_manifest_v2",
    "nlsr_v2_multistage_expert_candidate_manifest_v1",
})
CAUSAL_SAMPLE_ID_COLUMN = "causal_manifest_sample_id"
EXTERNAL_CAUSAL_ROW_COLUMNS = (
    CAUSAL_SAMPLE_ID_COLUMN,
    "causal_split_role",
    "causal_source_episode_id",
    "causal_goal_source_episode_id",
    "causal_goal_variant",
    "causal_goal_role",
    "causal_state_name",
    "causal_decision_frame",
    "causal_prefix_sha256",
    "causal_navdp_fifo_sha256",
    "causal_goal_sha256",
    "causal_manifest_sha256",
    "causal_manifest_schema_version",
    "external_scale_artifact_sha256",
    "external_scale_record_sha256",
    "external_scale_prefix_end_frame_exclusive",
    "external_scale_cam_pose_prefix_sha256",
    "external_scale_rgb_prefix_content_sequence_sha256",
    "external_scale_producer_sha256",
    "external_scale_configuration_sha256",
    "external_scale_lingbot_commit",
    "external_scale_weights_sha256",
    "external_scale_stream_source_sha256",
    *EXTERNAL_SCALE_QUALITY_COLUMNS,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RGB_RELATIVE = Path("videos/chunk-000/observation.images.rgb")


class ExternalCausalScaleError(RuntimeError):
    """A pinned causal-scale or sample-binding contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalCausalScaleError(message)


def canonical_json_bytes(value: object) -> bytes:
    # This byte contract is shared with build_causal_ground_scale.py through
    # novel_frontier_candidates_v2.canonical_json_bytes.  The trailing newline
    # is part of every artifact/configuration/prefix digest, not presentation.
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def ndarray_sha256(value: object) -> str:
    array = np.asarray(value)
    _require(np.issubdtype(array.dtype, np.number),
             "camera-pose prefix is not numeric")
    _require(bool(np.isfinite(array).all()),
             "camera-pose prefix is not finite")
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


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{label} is not a lowercase SHA256")
    return value


def _commit(value: object, label: str) -> str:
    _require(isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None,
             f"{label} is not a full Git commit")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    _require(type(value) is int and int(value) >= 1,
             f"{label} must be a positive integer")
    return int(value)


def _finite_positive(value: object, label: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result > 0.0,
             f"{label} must be finite and positive")
    return result


def _rooted_path(root: Path, relative: object, label: str) -> Path:
    _require(isinstance(relative, str) and bool(relative),
             f"{label} path is missing")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise ExternalCausalScaleError(
            f"{label} escapes its declared root") from error
    return result


def _file_record_path(record: object, root: Path, label: str) -> Path:
    row = _mapping(record, label)
    path = _rooted_path(root, row.get("path"), label)
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(type(row.get("bytes")) is int and path.stat().st_size == row["bytes"],
             f"{label} byte count changed")
    expected = _sha(row.get("content_sha256"), f"{label} content hash")
    _require(sha256_file(path) == expected, f"{label} content changed")
    return path


def _rgb_prefix_record(episode_root: Path, dataset_root: Path,
                       frame_count: int) -> dict[str, object]:
    rows = []
    rgb_root = episode_root / _RGB_RELATIVE
    for frame in range(frame_count):
        path = (rgb_root / f"{frame}.jpg").resolve()
        _require(path.is_file(), f"causal RGB frame is missing: {path}")
        try:
            relative = path.relative_to(dataset_root.resolve()).as_posix()
        except ValueError as error:
            raise ExternalCausalScaleError(
                f"causal RGB frame escapes episode root: {path}") from error
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


@dataclass(frozen=True)
class ExternalCausalScalePins:
    manifest_sha256: str
    artifact_sha256: str
    producer_sha256: str
    configuration_sha256: str
    lingbot_commit: str
    weights_sha256: str
    stream_source_sha256: str

    def validate(self) -> None:
        _sha(self.manifest_sha256, "causal manifest pin")
        _sha(self.artifact_sha256, "external scale artifact pin")
        _sha(self.producer_sha256, "external scale producer pin")
        _sha(self.configuration_sha256, "external scale configuration pin")
        _commit(self.lingbot_commit, "external scale LingBot commit pin")
        _sha(self.weights_sha256, "external scale weight pin")
        _sha(self.stream_source_sha256, "external scale stream-source pin")


@dataclass(frozen=True)
class ExternalCausalScaleBinding:
    sample_id: str
    split_role: str
    scene: str
    source_episode: str
    source_episode_id: str
    goal_source_episode_id: str
    goal_variant: str
    goal_role: str
    state_name: str
    decision_frame: int
    causal_prefix_sha256: str
    navdp_fifo_sha256: str
    goal_sha256: str
    metric_scale_m_per_raw: float
    manifest_sha256: str
    manifest_schema_version: str
    artifact_sha256: str
    record_sha256: str
    scale_prefix_end_frame_exclusive: int
    cam_pose_prefix_sha256: str
    rgb_prefix_content_sequence_sha256: str
    producer_sha256: str
    configuration_sha256: str
    lingbot_commit: str
    weights_sha256: str
    stream_source_sha256: str
    valid_frame_ratio: float
    relative_h_iqr: float
    clamped: int

    def row_fields(self) -> dict[str, object]:
        return {
            CAUSAL_SAMPLE_ID_COLUMN: self.sample_id,
            "causal_split_role": self.split_role,
            "causal_source_episode_id": self.source_episode_id,
            "causal_goal_source_episode_id": self.goal_source_episode_id,
            "causal_goal_variant": self.goal_variant,
            "causal_goal_role": self.goal_role,
            "causal_state_name": self.state_name,
            "causal_decision_frame": self.decision_frame,
            "causal_prefix_sha256": self.causal_prefix_sha256,
            "causal_navdp_fifo_sha256": self.navdp_fifo_sha256,
            "causal_goal_sha256": self.goal_sha256,
            "causal_manifest_sha256": self.manifest_sha256,
            "causal_manifest_schema_version": self.manifest_schema_version,
            "external_scale_artifact_sha256": self.artifact_sha256,
            "external_scale_record_sha256": self.record_sha256,
            "external_scale_prefix_end_frame_exclusive": (
                self.scale_prefix_end_frame_exclusive),
            "external_scale_cam_pose_prefix_sha256": (
                self.cam_pose_prefix_sha256),
            "external_scale_rgb_prefix_content_sequence_sha256": (
                self.rgb_prefix_content_sequence_sha256),
            "external_scale_producer_sha256": self.producer_sha256,
            "external_scale_configuration_sha256": self.configuration_sha256,
            "external_scale_lingbot_commit": self.lingbot_commit,
            "external_scale_weights_sha256": self.weights_sha256,
            "external_scale_stream_source_sha256": self.stream_source_sha256,
            "external_scale_valid_frame_ratio": self.valid_frame_ratio,
            "external_scale_relative_h_iqr": self.relative_h_iqr,
            "external_scale_clamped": self.clamped,
        }


class ExternalCausalScaleContract:
    """Immutable manifest/artifact indices plus exact per-seed rebinding."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        artifact_path: Path,
        pins: ExternalCausalScalePins,
    ) -> None:
        pins.validate()
        self.manifest_path = Path(manifest_path).resolve()
        self.artifact_path = Path(artifact_path).resolve()
        self.pins = pins
        _require(self.manifest_path.is_file(),
                 f"causal manifest is missing: {self.manifest_path}")
        _require(self.artifact_path.is_file(),
                 f"external scale artifact is missing: {self.artifact_path}")
        manifest_bytes = self.manifest_path.read_bytes()
        artifact_bytes = self.artifact_path.read_bytes()
        _require(sha256_bytes(manifest_bytes) == pins.manifest_sha256,
                 "causal manifest SHA changed")
        _require(sha256_bytes(artifact_bytes) == pins.artifact_sha256,
                 "external scale artifact SHA changed")
        try:
            manifest = json.loads(manifest_bytes)
            artifact = json.loads(artifact_bytes)
        except json.JSONDecodeError as error:
            raise ExternalCausalScaleError(
                "causal manifest or scale artifact is invalid JSON") from error
        self.manifest = _mapping(manifest, "causal manifest")
        self.artifact = _mapping(artifact, "external scale artifact")
        _require(self.manifest.get("schema_version")
                 in SUPPORTED_MANIFEST_SCHEMA_VERSIONS,
                 "causal manifest schema is unsupported")
        _require(artifact_bytes == canonical_json_bytes(self.artifact),
                 "external scale artifact is not canonical JSON")
        _require(self.artifact.get("schema_version")
                 == CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
                 "external scale artifact schema is unsupported")

        roots = _mapping(self.manifest.get("input_roots"),
                         "causal manifest input roots")
        self.dataset_root = Path(str(roots.get("episode_root", ""))).resolve()
        _require(self.dataset_root.is_dir(),
                 "causal manifest episode root is unavailable")
        self._episodes = self._index_episodes()
        self._samples = self._index_samples()
        self._configuration, self._estimator = self._validate_provenance()
        self._records = self._index_scale_records()

    @property
    def num_scale_frames(self) -> int:
        return int(self._configuration["num_scale_frames"])

    def sample_descriptor(self, sample_id: str) -> dict[str, object]:
        _require(sample_id in self._samples,
                 f"unknown causal manifest sample: {sample_id}")
        sample = self._samples[sample_id]
        goal = _mapping(sample["goal"], "sample goal")
        return {
            "sample_id": sample_id,
            "split_role": sample["split_role"],
            "scene": sample["scene"],
            "source_episode": sample["source_episode"],
            "goal_role": sample["goal_role"],
            "decision_frame": sample["_decision_frame"],
            "goal_sha256": sample["_goal_sha256"],
            "goal_path": str(_rooted_path(
                self.dataset_root, goal["path"], "sample goal")),
        }

    def selected_sample_ids(
        self, *, split_role: str, goal_roles: Sequence[str] = ("B", "C"),
    ) -> tuple[str, ...]:
        roles = frozenset(map(str, goal_roles))
        _require(bool(roles) and roles.issubset({"B", "C"}),
                 "selected goal roles must be B and/or C")
        result = tuple(sorted(
            sample_id for sample_id, sample in self._samples.items()
            if sample["split_role"] == split_role
            and sample["goal_role"] in roles
            and (str(sample["scene"]), str(sample["source_episode"]))
            in self._records
        ))
        _require(bool(result),
                 "causal manifest/scale selection contains no samples")
        return result

    def expected_row_binding(self, sample_id: str) -> dict[str, object]:
        """Return the immutable row witness for audit-time physical rebinding."""

        _require(sample_id in self._samples,
                 f"unknown causal manifest sample: {sample_id}")
        sample = self._samples[sample_id]
        scene = str(sample["scene"])
        episode = str(sample["source_episode"])
        record = self._records[(scene, episode)]
        rgb_prefix = _mapping(record["rgb_prefix"], "external scale RGB prefix")
        binding = ExternalCausalScaleBinding(
            sample_id=sample_id,
            split_role=str(sample["split_role"]),
            scene=scene,
            source_episode=episode,
            source_episode_id=str(sample["source_episode_id"]),
            goal_source_episode_id=str(sample["goal_source_episode_id"]),
            goal_variant=str(sample["goal_variant"]),
            goal_role=str(sample["goal_role"]),
            state_name=str(sample["state_name"]),
            decision_frame=int(sample["_decision_frame"]),
            causal_prefix_sha256=str(sample["_causal_prefix_sha256"]),
            navdp_fifo_sha256=str(sample["_navdp_fifo_sha256"]),
            goal_sha256=str(sample["_goal_sha256"]),
            metric_scale_m_per_raw=float(record["metric_scale_m_per_raw"]),
            manifest_sha256=self.pins.manifest_sha256,
            manifest_schema_version=str(self.manifest["schema_version"]),
            artifact_sha256=self.pins.artifact_sha256,
            record_sha256=str(record["_record_sha256"]),
            scale_prefix_end_frame_exclusive=int(
                record["prefix_end_frame_exclusive"]),
            cam_pose_prefix_sha256=str(record["cam_pose_prefix_sha256"]),
            rgb_prefix_content_sequence_sha256=str(
                rgb_prefix["content_sequence_sha256"]),
            producer_sha256=self.pins.producer_sha256,
            configuration_sha256=self.pins.configuration_sha256,
            lingbot_commit=self.pins.lingbot_commit,
            weights_sha256=self.pins.weights_sha256,
            stream_source_sha256=self.pins.stream_source_sha256,
            valid_frame_ratio=float(record["_valid_frame_ratio"]),
            relative_h_iqr=float(record["_relative_h_iqr"]),
            clamped=int(record["_clamped"]),
        )
        return {
            "session_id": sample_id,
            "scene": scene,
            "episode": episode,
            "metric_scale_source": EXTERNAL_CAUSAL_SCALE_SOURCE,
            "metric_scale_m_per_raw": binding.metric_scale_m_per_raw,
            **binding.row_fields(),
        }

    def _index_episodes(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        scenes = self.manifest.get("scenes")
        _require(isinstance(scenes, list), "causal manifest scenes are missing")
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        seen_scenes: set[str] = set()
        for raw_scene in scenes:
            scene_row = _mapping(raw_scene, "causal manifest scene")
            scene = scene_row.get("scene")
            role = scene_row.get("split_role")
            _require(isinstance(scene, str) and scene not in seen_scenes,
                     "causal manifest scene is empty or duplicated")
            _require(role in ("train", "development"),
                     "causal manifest scene role is invalid")
            seen_scenes.add(scene)
            episodes = scene_row.get("selected_episodes")
            _require(isinstance(episodes, list),
                     f"causal manifest episodes are missing for {scene}")
            for raw_episode in episodes:
                episode_row = _mapping(raw_episode, "causal manifest episode")
                episode = episode_row.get("episode")
                key = scene, episode
                _require(isinstance(episode, str) and key not in result,
                         "causal manifest episode is empty or duplicated")
                frame_count = _positive_int(
                    episode_row.get("n_frames"), "manifest episode frame count")
                metadata_path = _file_record_path(
                    episode_row.get("metadata"), self.dataset_root,
                    "episode metadata")
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ExternalCausalScaleError(
                        f"episode metadata is invalid: {metadata_path}") from error
                _require(isinstance(metadata, Mapping)
                         and metadata.get("n_frames") == frame_count,
                         "episode metadata frame count changed")
                result[(scene, episode)] = {
                    **episode_row,
                    "_split_role": role,
                    "_camera_height_m": _finite_positive(
                        metadata.get("camera_height_m", 0.5),
                        "episode camera height"),
                }
        return result

    def _index_samples(self) -> dict[str, Mapping[str, Any]]:
        samples = self.manifest.get("samples")
        _require(isinstance(samples, list), "causal manifest samples are missing")
        result: dict[str, Mapping[str, Any]] = {}
        join_keys: dict[tuple[str, str, str, int], str] = {}
        for raw_sample in samples:
            sample = _mapping(raw_sample, "causal manifest sample")
            sample_id = sample.get("sample_id")
            scene = sample.get("scene")
            episode = sample.get("source_episode")
            role = sample.get("split_role")
            goal_role = sample.get("goal_role")
            decision = _positive_int(
                sample.get("decision_frame"), "sample decision frame")
            _require(isinstance(sample_id, str) and sample_id
                     and sample_id not in result,
                     "causal manifest sample_id is empty or duplicated")
            _require(isinstance(scene, str) and isinstance(episode, str)
                     and (scene, episode) in self._episodes,
                     "causal manifest sample episode is unknown")
            episode_row = self._episodes[(scene, episode)]
            _require(role == episode_row["_split_role"],
                     "sample split role differs from its scene")
            source_id = sample.get("source_episode_id")
            _require(source_id == f"{scene}/{episode}",
                     "sample source_episode_id is inconsistent")
            _require(goal_role in ("B", "C"), "sample goal role is invalid")
            prefix = _mapping(sample.get("causal_prefix"),
                              "sample causal prefix")
            _require(prefix.get("frame_count") == decision,
                     "sample causal prefix is not decision-exclusive")
            prefix_sha = _sha(prefix.get("causal_prefix_sha256"),
                              "sample causal-prefix hash")
            fifo = _mapping(sample.get("navdp_fifo"), "sample NavDP FIFO")
            _sha(fifo.get("fifo_sha256"), "sample NavDP FIFO hash")
            goal = _mapping(sample.get("goal"), "sample goal")
            goal_sha = _sha(goal.get("content_sha256"), "sample goal hash")
            goal_path = goal.get("path")
            _require(isinstance(goal_path, str)
                     and Path(goal_path).name
                     == ("goal_1.jpg" if goal_role == "B" else "goal_2.jpg"),
                     "sample goal path disagrees with Goal B/C role")
            goal_episode = sample.get("goal_episode")
            _require(isinstance(goal_episode, str)
                     and sample.get("goal_source_episode_id")
                     == f"{scene}/{goal_episode}",
                     "sample goal source episode is inconsistent")
            _require(sample.get("goal_variant") in ("factual", "counterfactual"),
                     "sample goal variant is invalid")
            _require(isinstance(sample.get("state_name"), str)
                     and bool(sample.get("state_name")),
                     "sample state name is missing")
            _require(decision <= int(episode_row["n_frames"]),
                     "sample decision exceeds its source episode")
            join_key = scene, episode, goal_sha, decision
            _require(join_key not in join_keys,
                     "causal sample join key is ambiguous: "
                     f"{join_keys.get(join_key)} and {sample_id}")
            join_keys[join_key] = sample_id
            # Store the validated hashes so downstream code never reparses a
            # loosely typed nested field.
            result[sample_id] = {
                **sample,
                "_decision_frame": decision,
                "_causal_prefix_sha256": prefix_sha,
                "_navdp_fifo_sha256": fifo["fifo_sha256"],
                "_goal_sha256": goal_sha,
            }
        _require(bool(result), "causal manifest has no samples")
        covered_episodes = {
            (str(sample["scene"]), str(sample["source_episode"]))
            for sample in result.values()
        }
        _require(covered_episodes == set(self._episodes),
                 "causal manifest selected episodes are not exactly covered by samples")
        return result

    def _validate_provenance(
        self,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        provenance = _mapping(self.artifact.get("provenance"),
                              "external scale provenance")
        configuration = _mapping(self.artifact.get("configuration"),
                                 "external scale configuration")
        _require(provenance.get("input_manifest_sha256")
                 == self.pins.manifest_sha256,
                 "external scale was built from a different manifest")
        _require(provenance.get("input_manifest_schema_version")
                 == self.manifest.get("schema_version"),
                 "external scale manifest schema provenance changed")
        configuration_sha = sha256_bytes(canonical_json_bytes(configuration))
        _require(configuration_sha == self.pins.configuration_sha256
                 and provenance.get("configuration_sha256") == configuration_sha,
                 "external scale configuration pin changed")
        _require(provenance.get("producer_source_sha256")
                 == self.pins.producer_sha256,
                 "external scale producer pin changed")
        estimator = _mapping(provenance.get("estimator"),
                             "external scale estimator provenance")
        expected = {
            "kind": "frozen_lingbot_compute_metric_scale_prefix",
            "lingbot_commit": self.pins.lingbot_commit,
            "weights_sha256": self.pins.weights_sha256,
            "lingbot_stream_source_sha256": self.pins.stream_source_sha256,
        }
        for field, value in expected.items():
            _require(estimator.get(field) == value,
                     f"external scale estimator {field} pin changed")
        required = {
            "prefix_frame_cap", "num_scale_frames", "bias_correction",
            "scale_min", "scale_max",
        }
        _require(required.issubset(configuration),
                 "external scale configuration is incomplete")
        prefix_cap = _positive_int(configuration["prefix_frame_cap"],
                                   "scale prefix cap")
        num_scale = _positive_int(configuration["num_scale_frames"],
                                  "scale frame count")
        bias = _finite_positive(configuration["bias_correction"], "scale bias")
        scale_min = _finite_positive(configuration["scale_min"], "scale minimum")
        scale_max = _finite_positive(configuration["scale_max"], "scale maximum")
        _require(prefix_cap >= num_scale and scale_min < scale_max,
                 "external scale configuration range is invalid")
        # Normalize for typed access in record validation.
        return {
            **configuration,
            "prefix_frame_cap": prefix_cap,
            "num_scale_frames": num_scale,
            "bias_correction": bias,
            "scale_min": scale_min,
            "scale_max": scale_max,
        }, estimator

    def _index_scale_records(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        raw_records = self.artifact.get("records")
        _require(isinstance(raw_records, list),
                 "external scale records are missing")
        expected_by_episode: dict[tuple[str, str], dict[str, object]] = {}
        for sample_id, sample in self._samples.items():
            key = str(sample["scene"]), str(sample["source_episode"])
            row = expected_by_episode.setdefault(key, {
                "sample_ids": [], "decision_frames": set(),
            })
            row["sample_ids"].append(sample_id)  # type: ignore[union-attr]
            row["decision_frames"].add(  # type: ignore[union-attr]
                int(sample["_decision_frame"]))
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        for raw_record in raw_records:
            record = _mapping(raw_record, "external scale record")
            scene, episode = record.get("scene"), record.get("episode")
            key = scene, episode
            _require(isinstance(scene, str) and isinstance(episode, str)
                     and key in expected_by_episode and key not in result,
                     "external scale record episode is unknown or duplicated")
            expected = expected_by_episode[key]
            decisions = sorted(expected["decision_frames"])  # type: ignore[arg-type]
            sample_ids = sorted(expected["sample_ids"])  # type: ignore[arg-type]
            episode_row = self._episodes[key]
            _require(record.get("split_role") == episode_row["_split_role"],
                     "external scale record split role changed")
            _require(record.get("sample_ids") == sample_ids
                     and record.get("decision_frames") == decisions
                     and record.get("earliest_decision_frame") == decisions[0],
                     "external scale sample/decision binding changed")
            _require(record.get("episode_frame_count") == episode_row["n_frames"],
                     "external scale episode frame count changed")
            prefix_end = min(
                int(self._configuration["prefix_frame_cap"]), decisions[0])
            _require(prefix_end >= int(self._configuration["num_scale_frames"])
                     and record.get("prefix_end_frame_exclusive") == prefix_end,
                     "external scale causal prefix length changed")
            rgb_prefix = _mapping(record.get("rgb_prefix"),
                                  "external scale RGB prefix")
            _require(rgb_prefix.get("frame_count") == prefix_end,
                     "external scale RGB prefix length changed")
            _sha(rgb_prefix.get("path_sequence_sha256"),
                 "external scale RGB path-sequence hash")
            _sha(rgb_prefix.get("content_sequence_sha256"),
                 "external scale RGB content-sequence hash")
            _sha(record.get("cam_pose_prefix_sha256"),
                 "external scale camera-pose prefix hash")
            _require(isinstance(record.get("cam_pose_prefix_dtype"), str)
                     and bool(record.get("cam_pose_prefix_dtype")),
                     "external scale camera-pose dtype is missing")
            _require(type(record.get("cache_schema_version")) is int,
                     "external scale cache schema is malformed")
            _require(isinstance(record.get("precompute_signature"), str)
                     and bool(record.get("precompute_signature")),
                     "external scale precompute signature is missing")
            _require(type(record.get("valid")) is bool,
                     "external scale validity flag is malformed")
            debug = _mapping(record.get("debug"), "external scale quality debug")
            for field in ("n_points", "n_frames", "n_valid"):
                _require(
                    type(debug.get(field)) is int and int(debug[field]) >= 0,
                    f"external scale debug {field} is malformed",
                )
            debug_frames = int(debug["n_frames"])
            debug_valid = int(debug["n_valid"])
            _require(
                1 <= debug_frames <= prefix_end and debug_valid <= debug_frames,
                "external scale debug frame support is inconsistent",
            )
            if record["valid"]:
                ground_h = _finite_positive(
                    record.get("ground_h_est_raw"), "external ground height")
                metric_scale = _finite_positive(
                    record.get("metric_scale_m_per_raw"), "external metric scale")
                camera_height = _finite_positive(
                    record.get("camera_height_m"), "external camera height")
                _require(math.isclose(
                    camera_height, float(episode_row["_camera_height_m"]),
                    rel_tol=0.0, abs_tol=1e-12),
                    "external scale camera height changed")
                expected_scale = min(max(
                    float(self._configuration["bias_correction"])
                    * camera_height / ground_h,
                    float(self._configuration["scale_min"])),
                    float(self._configuration["scale_max"]))
                _require(math.isclose(metric_scale, expected_scale,
                                      rel_tol=1e-6, abs_tol=1e-6),
                         "external metric scale disagrees with pinned formula")
                debug_h = _finite_positive(
                    debug.get("h_est"), "external scale debug ground height")
                _require(
                    math.isclose(debug_h, ground_h, rel_tol=1e-6, abs_tol=1e-6),
                    "external scale debug ground height changed",
                )
                h_iqr = debug.get("h_iqr")
                _require(
                    not isinstance(h_iqr, bool)
                    and isinstance(h_iqr, (int, float))
                    and math.isfinite(float(h_iqr))
                    and float(h_iqr) >= 0.0,
                    "external scale debug h_iqr is malformed",
                )
                _require(
                    debug_valid >= max(3, debug_frames // 8),
                    "external scale valid-frame support is insufficient",
                )
                unclamped_scale = (
                    float(self._configuration["bias_correction"])
                    * camera_height
                    / ground_h
                )
                valid_frame_ratio = debug_valid / debug_frames
                relative_h_iqr = float(h_iqr) / ground_h
                clamped = int(not math.isclose(
                    metric_scale, unclamped_scale, rel_tol=1e-6, abs_tol=1e-6))
            else:
                _require(record.get("ground_h_est_raw") is None
                         and record.get("metric_scale_m_per_raw") is None,
                         "invalid external scale was not neutralized")
                raise ExternalCausalScaleError(
                    f"external scale estimate is invalid for {scene}/{episode}: "
                    f"{record.get('invalid_reason')}"
                )
            result[key] = {
                **record,
                "_record_sha256": sha256_bytes(canonical_json_bytes(record)),
                "_valid_frame_ratio": valid_frame_ratio,
                "_relative_h_iqr": relative_h_iqr,
                "_clamped": clamped,
            }
        _require(bool(result), "external scale artifact has no records")
        missing_records = set(expected_by_episode) - set(result)
        _require(
            not missing_records and set(result) == set(expected_by_episode),
            "external scale artifact does not exactly cover manifest episodes: "
            f"{sorted(missing_records)[:8]}",
        )
        return result

    def validate_runtime_episode(
        self,
        *,
        scene: str,
        episode: str,
        cam_pose_enc: object,
        cache_schema_version: int,
        precompute_signature: str,
    ) -> None:
        """Re-hash the exact raw/cache prefix consumed by the scale artifact."""
        key = scene, episode
        _require(key in self._records,
                 f"external scale lacks episode {scene}/{episode}")
        record = self._records[key]
        prefix_end = int(record["prefix_end_frame_exclusive"])
        pose = np.asarray(cam_pose_enc)
        _require(pose.ndim == 2 and pose.shape[1] == 9
                 and len(pose) >= prefix_end,
                 "runtime camera-pose cache does not cover scale prefix")
        prefix = np.asarray(pose[:prefix_end])
        _require(prefix.dtype.str == record["cam_pose_prefix_dtype"],
                 "runtime camera-pose prefix dtype changed")
        _require(ndarray_sha256(prefix) == record["cam_pose_prefix_sha256"],
                 "runtime camera-pose prefix changed")
        _require(type(cache_schema_version) is int
                 and cache_schema_version == record["cache_schema_version"],
                 "runtime cache schema differs from scale artifact")
        _require(precompute_signature == record["precompute_signature"],
                 "runtime precompute signature differs from scale artifact")
        episode_root = self.dataset_root / scene / episode
        _require(_rgb_prefix_record(
            episode_root, self.dataset_root, prefix_end) == record["rgb_prefix"],
            "runtime causal RGB prefix changed")

    def bind_seed(
        self,
        *,
        manifest_sample_id: str,
        scene: str,
        episode: str,
        query_path: Path,
        candidate_path: Path,
        candidate_frame: int,
        neighbor_offsets: Sequence[int],
        expected_split_role: str,
    ) -> ExternalCausalScaleBinding:
        """Bind one selected candidate to one exact causal manifest sample."""
        _require(isinstance(manifest_sample_id, str)
                 and manifest_sample_id in self._samples,
                 "candidate lacks a known causal manifest_sample_id")
        sample = self._samples[manifest_sample_id]
        _require(sample["scene"] == scene and sample["source_episode"] == episode,
                 "candidate episode differs from its causal manifest sample")
        _require(sample["split_role"] == expected_split_role,
                 "candidate split role differs from its causal manifest sample")
        source_root = (self.dataset_root / scene / episode).resolve()
        candidate = Path(candidate_path).resolve()
        _require(candidate.is_file(), f"candidate frame is missing: {candidate}")
        try:
            candidate.relative_to(source_root)
        except ValueError as error:
            raise ExternalCausalScaleError(
                "candidate path crosses its causal source episode") from error
        _require(candidate.parent == (source_root / _RGB_RELATIVE).resolve(),
                 "candidate path is not the source episode RGB stream")
        _require(candidate.stem.isdigit()
                 and int(candidate.stem) == int(candidate_frame),
                 "candidate frame differs from its filename")
        goal = _mapping(sample["goal"], "sample goal")
        expected_query = _rooted_path(
            self.dataset_root, goal["path"], "sample goal")
        query = Path(query_path).resolve()
        _require(query == expected_query,
                 "candidate query path differs from its causal sample goal")
        _require(sha256_file(query) == sample["_goal_sha256"],
                 "candidate query bytes differ from its causal sample goal")
        offsets = tuple(int(value) for value in neighbor_offsets)
        _require(bool(offsets), "candidate neighbor offsets are empty")
        decision = int(sample["_decision_frame"])
        anchors = [int(candidate_frame) + offset for offset in offsets]
        _require(min(anchors) >= int(self._configuration["num_scale_frames"]),
                 "candidate neighbor precedes the complete scale block")
        _require(max(anchors) < decision,
                 "candidate or neighbor anchor reaches a future decision frame")
        _require(max(anchors) < int(self._episodes[(scene, episode)]["n_frames"]),
                 "candidate neighbor exceeds its source episode")
        record = self._records.get((scene, episode))
        _require(record is not None,
                 f"external scale lacks episode {scene}/{episode}")
        _require(manifest_sample_id in record["sample_ids"]
                 and decision in record["decision_frames"],
                 "external scale does not bind the selected causal sample")
        _require(int(record["prefix_end_frame_exclusive"]) <= decision,
                 "external scale prefix crosses the selected decision")
        _require(record["valid"] is True,
                 "selected external scale is invalid; pooled fallback is forbidden")
        metric_scale = _finite_positive(
            record["metric_scale_m_per_raw"], "selected external metric scale")
        rgb_prefix = _mapping(record["rgb_prefix"], "external scale RGB prefix")
        return ExternalCausalScaleBinding(
            sample_id=manifest_sample_id,
            split_role=str(sample["split_role"]),
            scene=scene,
            source_episode=episode,
            source_episode_id=str(sample["source_episode_id"]),
            goal_source_episode_id=str(sample["goal_source_episode_id"]),
            goal_variant=str(sample["goal_variant"]),
            goal_role=str(sample["goal_role"]),
            state_name=str(sample["state_name"]),
            decision_frame=decision,
            causal_prefix_sha256=str(sample["_causal_prefix_sha256"]),
            navdp_fifo_sha256=str(sample["_navdp_fifo_sha256"]),
            goal_sha256=str(sample["_goal_sha256"]),
            metric_scale_m_per_raw=metric_scale,
            manifest_sha256=self.pins.manifest_sha256,
            manifest_schema_version=str(self.manifest["schema_version"]),
            artifact_sha256=self.pins.artifact_sha256,
            record_sha256=str(record["_record_sha256"]),
            scale_prefix_end_frame_exclusive=int(
                record["prefix_end_frame_exclusive"]),
            cam_pose_prefix_sha256=str(record["cam_pose_prefix_sha256"]),
            rgb_prefix_content_sequence_sha256=str(
                rgb_prefix["content_sequence_sha256"]),
            producer_sha256=self.pins.producer_sha256,
            configuration_sha256=self.pins.configuration_sha256,
            lingbot_commit=self.pins.lingbot_commit,
            weights_sha256=self.pins.weights_sha256,
            stream_source_sha256=self.pins.stream_source_sha256,
            valid_frame_ratio=float(record["_valid_frame_ratio"]),
            relative_h_iqr=float(record["_relative_h_iqr"]),
            clamped=int(record["_clamped"]),
        )

    def summary(self) -> dict[str, object]:
        return {
            "mode": EXTERNAL_CAUSAL_SCALE_SOURCE,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.pins.manifest_sha256,
            "manifest_schema_version": self.manifest["schema_version"],
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.pins.artifact_sha256,
            "producer_source_sha256": self.pins.producer_sha256,
            "configuration_sha256": self.pins.configuration_sha256,
            "lingbot_commit": self.pins.lingbot_commit,
            "weights_sha256": self.pins.weights_sha256,
            "stream_source_sha256": self.pins.stream_source_sha256,
            "manifest_sample_count": len(self._samples),
            "scale_episode_count": len(self._records),
        }


def validate_external_causal_frame(
    frame: Any,
    *,
    expected_sample_ids: Sequence[str] | None = None,
    expected_split_role: str | None = None,
    expected_row_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate per-row external provenance without reopening GPU inputs.

    This is used by both artifact audit and trainer.  Repeated candidate rows
    may share one manifest sample, but every binding/provenance value for that
    sample must remain identical.
    """
    if "metric_scale_source" not in frame.columns:
        raise ExternalCausalScaleError("rows lack metric_scale_source")
    _require(len(frame) > 0, "external causal artifact has no rows")
    _require(not frame["metric_scale_source"].isna().any(),
             "row has a missing metric_scale_source")
    source = frame["metric_scale_source"].astype(str)
    selected = frame.loc[source.eq(EXTERNAL_CAUSAL_SCALE_SOURCE)].copy()
    if selected.empty:
        _require(expected_sample_ids is None,
                 "rows do not cover any expected external manifest sample")
        return {
            "approved": False,
            "external_rows": 0,
            "external_sessions": 0,
            "external_samples": 0,
            "external_fraction": 0.0,
            "reason": "no external causal-scale rows",
        }
    _require(len(selected) == len(frame),
             "external causal artifact mixes external and legacy scale rows")
    missing = set(EXTERNAL_CAUSAL_ROW_COLUMNS) - set(frame.columns)
    _require(not missing,
             f"external causal rows lack provenance columns: {sorted(missing)}")
    for required_column in (
            "session_id", "scene", "episode", "candidate_frame"):
        _require(required_column in selected.columns,
                 f"external causal rows lack {required_column}")

    def strict_strings(column: str) -> list[str]:
        _require(not selected[column].isna().any(),
                 f"external row {column} is missing")
        values = selected[column].tolist()
        _require(all(isinstance(value, str) and bool(value.strip())
                     and value.strip().lower() != "nan" for value in values),
                 f"external row {column} is empty or malformed")
        return [value.strip() for value in values]

    def exact_ints(column: str, *, minimum: int = 0) -> np.ndarray:
        _require(not selected[column].isna().any(),
                 f"external row {column} is missing")
        try:
            numeric = selected[column].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ExternalCausalScaleError(
                f"external row {column} is not numeric") from error
        _require(bool(np.isfinite(numeric).all()
                      and np.equal(numeric, np.floor(numeric)).all()
                      and np.all(numeric >= minimum)),
                 f"external row {column} is not an exact integer")
        return numeric.astype(np.int64)

    sample_ids = strict_strings(CAUSAL_SAMPLE_ID_COLUMN)
    sessions = strict_strings("session_id")
    scenes = strict_strings("scene")
    episodes = strict_strings("episode")
    source_episode_ids = strict_strings("causal_source_episode_id")
    strict_strings("causal_goal_source_episode_id")
    strict_strings("causal_state_name")
    strict_strings("causal_goal_variant")
    strict_strings("causal_goal_role")
    strict_strings("causal_split_role")
    strict_strings("causal_manifest_schema_version")
    _require(sessions == sample_ids,
             "external row session_id is not its exact manifest sample_id")
    _require(all(source_id == f"{scene}/{episode}" for
                 source_id, scene, episode in zip(
                     source_episode_ids, scenes, episodes)),
             "external row source episode ID is inconsistent")
    for column in (
            "causal_manifest_sha256", "causal_prefix_sha256",
            "causal_navdp_fifo_sha256", "causal_goal_sha256",
            "external_scale_artifact_sha256", "external_scale_record_sha256",
            "external_scale_cam_pose_prefix_sha256",
            "external_scale_rgb_prefix_content_sequence_sha256",
            "external_scale_producer_sha256",
            "external_scale_configuration_sha256",
            "external_scale_weights_sha256",
            "external_scale_stream_source_sha256"):
        for value in selected[column].astype(str):
            _sha(value, f"external row {column}")
    for value in selected["external_scale_lingbot_commit"].astype(str):
        _commit(value, "external row LingBot commit")
    _require(selected["causal_goal_role"].isin(("B", "C")).all(),
             "external row has an invalid goal role")
    _require(selected["causal_goal_variant"].isin(
        ("factual", "counterfactual")).all(),
        "external row has an invalid goal variant")
    _require(all(
        (role == "B" and "goal_b" in state)
        or (role == "C" and "goal_c" in state)
        for role, state in zip(
            selected["causal_goal_role"].astype(str),
            selected["causal_state_name"].astype(str))),
        "external row goal role disagrees with its state")
    _require(selected["causal_split_role"].isin(("train", "development")).all(),
             "external row has an invalid split role")
    if expected_split_role is not None:
        _require(expected_split_role in ("train", "development"),
                 "expected external split role is invalid")
        _require(selected["causal_split_role"].eq(expected_split_role).all(),
                 "external row split role differs from the expected role")
    decisions = exact_ints("causal_decision_frame", minimum=1)
    anchors = exact_ints("candidate_frame", minimum=0)
    prefixes = exact_ints(
        "external_scale_prefix_end_frame_exclusive", minimum=1)
    _require(bool(np.all(decisions > anchors)),
             "external row candidate reaches its decision frame")
    _require(bool(np.all(prefixes >= 1) and np.all(prefixes <= decisions)),
             "external row scale prefix crosses its decision")
    scales = selected["metric_scale_m_per_raw"].to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(scales).all() and np.all(scales > 0.0)),
             "external row metric scale is invalid")
    valid_ratios = selected[
        "external_scale_valid_frame_ratio"].to_numpy(dtype=np.float64)
    relative_iqrs = selected[
        "external_scale_relative_h_iqr"].to_numpy(dtype=np.float64)
    clamped = exact_ints("external_scale_clamped", minimum=0)
    _require(bool(np.isfinite(valid_ratios).all()
                  and np.all(valid_ratios > 0.0)
                  and np.all(valid_ratios <= 1.0)),
             "external row valid-frame ratio is invalid")
    _require(bool(np.isfinite(relative_iqrs).all()
                  and np.all(relative_iqrs >= 0.0)),
             "external row relative h_iqr is invalid")
    _require(bool(np.isin(clamped, (0, 1)).all()),
             "external row clamp flag is invalid")
    # The same manifest sample may have multiple candidate anchors, but all
    # immutable causal/provenance columns must be identical across them.
    for sample_id, group in selected.groupby(
            CAUSAL_SAMPLE_ID_COLUMN, sort=False, dropna=False):
        _require(isinstance(sample_id, str) and bool(sample_id.strip()),
                 "external row sample_id is empty")
        for column in EXTERNAL_CAUSAL_ROW_COLUMNS[1:]:
            _require(group[column].astype(str).nunique(dropna=False) == 1,
                     f"external sample {sample_id} changed {column}")
        _require(group["metric_scale_m_per_raw"].nunique(dropna=False) == 1,
                 f"external sample {sample_id} changed metric scale")
    actual_sample_ids = frozenset(sample_ids)
    if expected_sample_ids is not None:
        expected = frozenset(expected_sample_ids)
        _require(bool(expected)
                 and all(isinstance(value, str) and bool(value.strip())
                         for value in expected),
                 "expected manifest sample set is empty or malformed")
        missing_samples = expected - actual_sample_ids
        extra_samples = actual_sample_ids - expected
        _require(not missing_samples and not extra_samples,
                 "external rows do not exactly cover manifest samples; "
                 f"missing={sorted(missing_samples)[:8]} "
                 f"extra={sorted(extra_samples)[:8]}")
    if expected_row_bindings is not None:
        _require(set(expected_row_bindings) == set(actual_sample_ids),
                 "physical row bindings do not cover the exact row sample set")
        for sample_id, group in selected.groupby(
                CAUSAL_SAMPLE_ID_COLUMN, sort=False, dropna=False):
            expected_binding = expected_row_bindings[str(sample_id)]
            for column, expected_value in expected_binding.items():
                _require(column in group.columns,
                         f"external row lacks physical binding field {column}")
                values = group[column].tolist()
                if (isinstance(expected_value, (int, float))
                        and not isinstance(expected_value, bool)):
                    try:
                        numeric = np.asarray(values, dtype=np.float64)
                    except (TypeError, ValueError) as error:
                        raise ExternalCausalScaleError(
                            f"external sample {sample_id} changed {column}") from error
                    _require(bool(np.isfinite(numeric).all()
                                  and np.allclose(
                                      numeric, float(expected_value),
                                      rtol=1e-10, atol=1e-12)),
                             f"external sample {sample_id} changed {column}")
                else:
                    _require(all(value == expected_value for value in values),
                             f"external sample {sample_id} changed {column}")
    artifacts = sorted(selected[
        "external_scale_artifact_sha256"].astype(str).unique().tolist())
    manifests = sorted(selected[
        "causal_manifest_sha256"].astype(str).unique().tolist())
    global_pin_columns = {
        "producer_source_sha256": "external_scale_producer_sha256",
        "configuration_sha256": "external_scale_configuration_sha256",
        "lingbot_commit": "external_scale_lingbot_commit",
        "weights_sha256": "external_scale_weights_sha256",
        "stream_source_sha256": "external_scale_stream_source_sha256",
        "manifest_schema_version": "causal_manifest_schema_version",
    }
    global_pins: dict[str, str] = {}
    for output_name, column in global_pin_columns.items():
        values = selected[column].astype(str).unique().tolist()
        _require(len(values) == 1,
                 f"external rows mix multiple {column} values")
        global_pins[output_name] = values[0]
    _require(len(artifacts) == 1 and len(manifests) == 1,
             "external rows mix multiple manifest or scale artifacts")
    split_roles = sorted(selected[
        "causal_split_role"].astype(str).unique().tolist())
    canonical_sample_ids = sorted(actual_sample_ids)
    return {
        "approved": True,
        "exact_manifest_sample_coverage_approved": (
            expected_sample_ids is not None),
        "external_rows": int(len(selected)),
        "external_sessions": len(set(sessions)),
        "external_samples": len(actual_sample_ids),
        "external_fraction": float(len(selected) / len(frame)),
        "manifest_sample_ids_sha256": sha256_bytes(
            canonical_json_bytes(canonical_sample_ids)),
        "manifest_sha256": manifests,
        "scale_artifact_sha256": artifacts,
        **global_pins,
        "split_roles": split_roles,
        "source": EXTERNAL_CAUSAL_SCALE_SOURCE,
    }
