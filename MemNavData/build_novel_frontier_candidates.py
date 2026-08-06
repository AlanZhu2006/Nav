#!/usr/bin/env python3
"""Build the causal NLSR-V2 J0-P frontier-proposal artifact.

The input is the frozen manifest produced by
``build_novel_candidate_manifest.py``.  Two pose arms are always reported:

* ``teacher_pose`` decodes the generated trajectory's camera/base transforms;
* ``lingbot_deployment_pose`` uses a validated versioned cache pair, dense
  ``cam_pose_enc``, and a causal prefix ground-height sequence.  A legacy
  whole-episode-only scale or an illegal mapping invalidates this arm.  It
  never fits a GT Sim(2) transform.

The resulting JSON contains proposal geometry and a deployment-only shortlist.
Pathfinder/GT proposal-proxy labels are optional and isolated in a sibling
table through the pure ``ProposalProxyLabeler`` interface; this command-line
entry point intentionally does not instantiate Habitat.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np

try:
    from MemNavData.novel_frontier_candidates_v2 import (
        FrontierConfig,
        FrontierProposalError,
        PlanarScan,
        ProposalProxyLabeler,
        SCHEMA_VERSION as PROPOSAL_SCHEMA_VERSION,
        SE2Pose,
        attach_proposal_proxy_labels,
        canonical_json_bytes,
        depth_to_planar_scan,
        generate_frontier_proposals,
        invalid_proposal,
    )
except ModuleNotFoundError:  # Direct execution from MemNavData/.
    from novel_frontier_candidates_v2 import (  # type: ignore
        FrontierConfig,
        FrontierProposalError,
        PlanarScan,
        ProposalProxyLabeler,
        SCHEMA_VERSION as PROPOSAL_SCHEMA_VERSION,
        SE2Pose,
        attach_proposal_proxy_labels,
        canonical_json_bytes,
        depth_to_planar_scan,
        generate_frontier_proposals,
        invalid_proposal,
    )


ARTIFACT_SCHEMA_VERSION = "nlsr_v2_frontier_proposal_artifact_v1"
EXTERNAL_SCALE_ARTIFACT_SCHEMA_VERSION = (
    "nlsr_v2_frontier_proposal_artifact_v2")
INPUT_MANIFEST_SCHEMA_VERSION = "nlsr_v2_expert_candidate_manifest_v1"
PATCH_SCORE_SCHEMA_VERSION = "nlsr_v2_goal_patch_frame_scores_v1"
CAUSAL_GROUND_SCALE_SCHEMA_VERSION = "nlsr_v2_causal_ground_scale_v1"
TEACHER_ARM = "teacher_pose"
DEPLOYMENT_ARM = "lingbot_deployment_pose"
ARMS = (TEACHER_ARM, DEPLOYMENT_ARM)
DEPTH_RELATIVE = Path("videos/chunk-000/observation.images.depth")
RGB_RELATIVE = Path("videos/chunk-000/observation.images.rgb")
PARQUET_RELATIVE = Path("data/chunk-000/episode_000000.parquet")
METADATA_RELATIVE = Path("meta/gen_meta.json")
FLOW_RELATIVE = Path("videos/chunk-000/lingbot_cam_cache.npz")
AGGREGATOR_FLOW_RELATIVE = Path("videos/chunk-000/lingbot_cache.npz")
CAUSAL_GROUND_PREFIX_SEMANTICS = "causal_prefix_floor_hist_v1"
GROUND_BIAS_CORRECTION = 1.15
GROUND_SCALE_RANGE = (0.8, 6.0)
HABITAT_TO_DATA_ROTATION = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


class CandidateBuildError(RuntimeError):
    """A fail-closed manifest, cache, or artifact violation."""


class DeploymentPoseError(CandidateBuildError):
    """A deployment-pose arm is invalid but the teacher arm remains auditable."""


@dataclass(frozen=True)
class ExternalGroundScaleBinding:
    """Immutable, globally-audited scale record for one manifest episode."""

    artifact_path: str
    artifact_sha256: str
    producer_sha256: str
    configuration_sha256: str
    estimator_kind: str
    lingbot_commit: str
    weights_sha256: str
    stream_source_sha256: str
    configuration: Mapping[str, object]
    record: Mapping[str, object]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_hex_digest(value: object, lengths: tuple[int, ...] = (64,)) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _ndarray_prefix_sha256(value: object) -> str:
    """Match build_causal_ground_scale.ndarray_sha256 without importing it."""
    array = np.asarray(value)
    if (not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()):
        raise DeploymentPoseError(
            "external-scale camera-pose prefix must be finite numeric")
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


def _require_file_record(record: object, path: Path, label: str) -> None:
    if not isinstance(record, Mapping):
        raise CandidateBuildError(f"{label} file record is missing")
    expected_size = record.get("bytes")
    expected_hash = record.get("content_sha256")
    if (isinstance(expected_size, bool) or not isinstance(expected_size, int)
            or expected_size < 0 or not isinstance(expected_hash, str)
            or len(expected_hash) != 64):
        raise CandidateBuildError(f"{label} file record is malformed")
    if not path.is_file():
        raise CandidateBuildError(f"{label} file is missing: {path}")
    if path.stat().st_size != expected_size:
        raise CandidateBuildError(f"{label} byte size changed: {path}")
    if sha256_file(path) != expected_hash:
        raise CandidateBuildError(f"{label} content SHA256 changed: {path}")


def _record_for_path(path: Path, root: Path) -> dict:
    if not path.is_file():
        raise CandidateBuildError(f"causal prefix file is missing: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CandidateBuildError(f"causal file escapes episode root: {path}") from exc
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path),
    }


def _sequence_sha(values: Sequence[object]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


def _external_rgb_prefix_record(
    episode_dir: Path,
    episode_root: Path,
    frame_count: int,
) -> dict[str, object]:
    if (isinstance(frame_count, bool) or not isinstance(frame_count, int)
            or frame_count < 1):
        raise DeploymentPoseError("external-scale RGB prefix length is invalid")
    records = [
        _record_for_path(
            episode_dir / RGB_RELATIVE / f"{frame}.jpg", episode_root)
        for frame in range(frame_count)
    ]
    return {
        "frame_count": frame_count,
        "path_sequence_sha256": _sequence_sha(
            [record["path"] for record in records]),
        "content_sequence_sha256": _sequence_sha(records),
    }


def _matrix(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateBuildError(f"{label} is not numeric") from exc
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise CandidateBuildError(f"{label} must be finite with shape {shape}")
    return matrix


def _canonical_numeric(value: object, label: str) -> object:
    if isinstance(value, (bool, np.bool_)):
        raise CandidateBuildError(f"{label} contains a boolean numeric value")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise CandidateBuildError(f"{label} contains NaN or infinity")
        return 0.0 if result == 0.0 else result
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            _canonical_numeric(item, f"{label}[{index}]")
            for index, item in enumerate(list(value))
        ]
    raise CandidateBuildError(
        f"{label} contains unsupported type {type(value).__name__}")


def load_pose_table(path: Path, expected_rows: int | None = None) -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise CandidateBuildError("pyarrow is required to read causal poses") from exc
    columns = (
        "index",
        "observation.camera_intrinsic",
        "observation.camera_extrinsic",
        "action",
    )
    try:
        table = parquet.read_table(path, columns=list(columns))
    except Exception as exc:
        raise CandidateBuildError(f"cannot read pose parquet {path}: {exc}") from exc
    if tuple(table.column_names) != columns:
        raise CandidateBuildError(f"pose parquet columns changed: {table.column_names}")
    if expected_rows is not None and table.num_rows != int(expected_rows):
        raise CandidateBuildError(
            f"pose parquet row count changed: {table.num_rows} != {expected_rows}")
    rows = []
    for frame, raw in enumerate(table.to_pylist()):
        index = raw.get("index")
        if isinstance(index, bool) or int(index) != index or int(index) != frame:
            raise CandidateBuildError(f"pose parquet index is not contiguous at {frame}")
        rows.append({
            "index": frame,
            "observation.camera_intrinsic": _canonical_numeric(
                _matrix(raw.get("observation.camera_intrinsic"), (3, 3),
                        f"intrinsic[{frame}]").tolist(),
                f"intrinsic[{frame}]"),
            "observation.camera_extrinsic": _canonical_numeric(
                _matrix(raw.get("observation.camera_extrinsic"), (4, 4),
                        f"extrinsic[{frame}]").tolist(),
                f"extrinsic[{frame}]"),
            "action": _canonical_numeric(
                _matrix(raw.get("action"), (4, 4),
                        f"action[{frame}]").tolist(),
                f"action[{frame}]"),
        })
    return rows


def audit_causal_prefix(
    *,
    sample: Mapping[str, object],
    episode_root: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict:
    prefix = sample.get("causal_prefix")
    if not isinstance(prefix, Mapping):
        raise CandidateBuildError("sample causal_prefix is missing")
    decision = sample.get("decision_frame")
    if (isinstance(decision, bool) or not isinstance(decision, int)
            or not 0 < decision <= len(rows)):
        raise CandidateBuildError("sample decision_frame is invalid")
    if (prefix.get("exclusive_end_frame") != decision
            or prefix.get("frame_count") != decision
            or prefix.get("parquet_row_count") != decision):
        raise CandidateBuildError("causal prefix length disagrees with decision frame")
    scene, episode = str(sample["scene"]), str(sample["source_episode"])
    episode_dir = episode_root / scene / episode
    modalities = {}
    for modality, relative, suffix in (
            ("rgb", RGB_RELATIVE, ".jpg"),
            ("depth", DEPTH_RELATIVE, ".png")):
        records = [
            _record_for_path(episode_dir / relative / f"{frame}{suffix}", episode_root)
            for frame in range(decision)
        ]
        modalities[modality] = {
            "path_sequence_sha256": _sequence_sha(
                [record["path"] for record in records]),
            "content_sequence_sha256": _sequence_sha([{
                "path": record["path"],
                "bytes": record["bytes"],
                "content_sha256": record["content_sha256"],
            } for record in records]),
        }
        expected = prefix.get("modalities", {}).get(modality)  # type: ignore[union-attr]
        if expected != modalities[modality]:
            raise CandidateBuildError(
                f"{modality} causal prefix content changed for {sample['sample_id']}")
    parquet_hash = _sequence_sha(list(rows[:decision]))
    if parquet_hash != prefix.get("parquet_rows_sha256"):
        raise CandidateBuildError(
            f"parquet causal prefix changed for {sample['sample_id']}")
    causal = sha256_bytes(canonical_json_bytes({
        "frame_count": decision,
        "rgb": modalities["rgb"],
        "depth": modalities["depth"],
        "parquet_rows_sha256": parquet_hash,
    }))
    if causal != prefix.get("causal_prefix_sha256"):
        raise CandidateBuildError(
            f"causal prefix SHA changed for {sample['sample_id']}")
    return {
        "decision_frame": decision,
        "causal_prefix_sha256": causal,
        "parquet_rows_sha256": parquet_hash,
    }


def _resolved_mount(extrinsic: np.ndarray, frame_convention: str) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    if str(frame_convention).startswith("positions+parquet in data(Zup,M_W)"):
        if np.allclose(rotation, HABITAT_TO_DATA_ROTATION, atol=1e-6):
            return extrinsic
        if np.allclose(rotation, np.eye(3), atol=1e-6):
            corrected = extrinsic.copy()
            corrected[:3, :3] = HABITAT_TO_DATA_ROTATION
            return corrected
        raise CandidateBuildError(
            "generated Z-up teacher pose has an unsupported mount rotation")
    if not np.isfinite(rotation).all() or abs(np.linalg.det(rotation)) < 1e-8:
        raise CandidateBuildError("teacher camera mount is singular")
    return extrinsic


def teacher_se2_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    decision_frame: int,
    frame_convention: str,
) -> dict[int, SE2Pose]:
    poses = {}
    for frame in range(decision_frame):
        action = _matrix(rows[frame]["action"], (4, 4), f"action[{frame}]")
        mount = _resolved_mount(_matrix(
            rows[frame]["observation.camera_extrinsic"], (4, 4),
            f"extrinsic[{frame}]"), frame_convention)
        try:
            base_rotation = action[:3, :3] @ np.linalg.inv(mount[:3, :3])
        except np.linalg.LinAlgError as exc:
            raise CandidateBuildError("teacher camera mount is singular") from exc
        forward = base_rotation @ np.asarray([0.0, 1.0, 0.0])
        norm = math.hypot(float(forward[0]), float(forward[1]))
        if not math.isfinite(norm) or norm < 1e-6:
            raise CandidateBuildError("teacher forward axis leaves the data ground plane")
        poses[frame] = SE2Pose(
            float(action[0, 3]),
            float(action[1, 3]),
            math.atan2(float(forward[1]), float(forward[0])),
        )
    return poses


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise DeploymentPoseError("LingBot quaternion is invalid")
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise DeploymentPoseError("LingBot quaternion is degenerate")
    x, y, z, w = value / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _validate_versioned_cache_pair(
    aggregator_cache_path: Path,
    camera_cache_path: Path,
    *,
    expected_num_frames: int,
):
    """Load the repository's header-only cache validator without package side effects."""
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "InternNav/internnav/model/basemodel/memnav/cache_schema.py"
    )
    if not schema_path.is_file():
        raise DeploymentPoseError(f"cache schema validator is missing: {schema_path}")
    spec = importlib.util.spec_from_file_location(
        "_nlsr_v2_cache_schema", schema_path)
    if spec is None or spec.loader is None:
        raise DeploymentPoseError("cannot load cache schema validator")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.validate_cache_files(
            aggregator_cache_path,
            camera_cache_path,
            expected_num_frames=int(expected_num_frames),
            require_versioned=True,
        )
    except Exception as exc:
        raise DeploymentPoseError(
            f"versioned LingBot cache-pair validation failed: {exc}") from exc


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _memoized_content_sha256(
    path: Path,
    memo: dict[Path, tuple[tuple[int, int], str]] | None,
) -> str:
    resolved = path.resolve()
    signature = _file_signature(resolved)
    if memo is not None and resolved in memo:
        previous_signature, digest = memo[resolved]
        if previous_signature != signature:
            raise DeploymentPoseError(
                f"cache file changed during proposal build: {resolved}")
        return digest
    digest = sha256_file(resolved)
    if memo is not None:
        memo[resolved] = signature, digest
    return digest


def lingbot_deployment_se2_rows(
    aggregator_cache_path: Path,
    cache_path: Path,
    *,
    decision_frame: int,
    episode_frame_count: int,
    camera_height_m: float,
    expected_causal_prefix_sha256: str,
    expected_prefix_builder_sha256: str,
    expected_prefix_configuration_sha256: str,
    external_ground_scale: ExternalGroundScaleBinding | None = None,
    episode_dir: Path | None = None,
    episode_root: Path | None = None,
    validation_memo: dict[tuple[Path, Path, int], tuple[
        tuple[int, int], tuple[int, int], object]] | None = None,
    content_sha_memo: dict[Path, tuple[tuple[int, int], str]] | None = None,
) -> tuple[dict[int, SE2Pose], dict]:
    """Load strict causal ground-anchored x-z poses without GT alignment.

    Versioned LingBot caches have two timelines: ``cam_pose_enc`` is dense and
    row-aligned with raw frames, while ``cam_frame_indices`` maps only sparse
    camera-K/V rows.  The repository's paired-cache validator establishes both
    facts.  A whole-episode ``ground_h_est`` is explicitly *not* causal at a
    midpoint.  The preferred path consumes an immutable external first-prefix
    sidecar; the old dense prefix remains available only when no external
    artifact was explicitly supplied.  Neither path falls back to the
    whole-episode scalar.
    """
    if not aggregator_cache_path.is_file():
        raise DeploymentPoseError(
            f"LingBot aggregator cache is missing: {aggregator_cache_path}")
    if not cache_path.is_file():
        raise DeploymentPoseError(f"LingBot camera cache is missing: {cache_path}")
    pair_key = (
        aggregator_cache_path.resolve(), cache_path.resolve(),
        int(episode_frame_count))
    aggregator_signature = _file_signature(aggregator_cache_path)
    camera_signature = _file_signature(cache_path)
    if validation_memo is not None and pair_key in validation_memo:
        old_aggregator, old_camera, layout = validation_memo[pair_key]
        if (old_aggregator != aggregator_signature
                or old_camera != camera_signature):
            raise DeploymentPoseError(
                "LingBot cache pair changed during proposal build")
    else:
        layout = _validate_versioned_cache_pair(
            aggregator_cache_path,
            cache_path,
            expected_num_frames=episode_frame_count,
        )
        if validation_memo is not None:
            validation_memo[pair_key] = (
                aggregator_signature, camera_signature, layout)
    # Legacy artifacts recorded full-cache hashes.  In external causal-scale
    # mode those files contain future rows, so hashing their complete payload
    # would make a causal decision artifact depend on unseen frames.
    if external_ground_scale is None:
        aggregator_sha = _memoized_content_sha256(
            aggregator_cache_path, content_sha_memo)
        cache_sha = _memoized_content_sha256(cache_path, content_sha_memo)
    else:
        aggregator_sha = None
        cache_sha = None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {"cam_pose_enc", "cam_frame_indices"}
            if external_ground_scale is None:
                required |= {
                    "ground_h_est_prefix",
                    "ground_h_est_prefix_frame_indices",
                    "ground_h_est_prefix_causal_prefix_sha256",
                    "ground_h_est_prefix_semantics",
                    "ground_h_est_prefix_builder_sha256",
                    "ground_h_est_prefix_configuration_sha256",
                }
            missing = sorted(required - set(cache.files))
            if missing:
                if ("ground_h_est" in cache.files
                        and any(name.startswith("ground_h_est_prefix")
                                for name in missing)):
                    raise DeploymentPoseError(
                        "cache has only whole-episode ground_h_est; using it at "
                        "a causal decision would leak future frames")
                raise DeploymentPoseError(
                    "LingBot deployment cache lacks strict scale/frame mapping: "
                    f"{missing}")
            pose_raw = np.asarray(cache["cam_pose_enc"])
            frame_raw = np.asarray(cache["cam_frame_indices"])
            if external_ground_scale is None:
                prefix_ground = np.asarray(
                    cache["ground_h_est_prefix"], dtype=np.float64)
                prefix_ground_frames = np.asarray(
                    cache["ground_h_est_prefix_frame_indices"])
                prefix_input_hashes = np.asarray(
                    cache["ground_h_est_prefix_causal_prefix_sha256"])
                prefix_semantics = str(np.asarray(
                    cache["ground_h_est_prefix_semantics"]).reshape(-1)[0])
                prefix_builder_sha = str(np.asarray(
                    cache["ground_h_est_prefix_builder_sha256"]).reshape(-1)[0])
                prefix_configuration_sha = str(np.asarray(
                    cache["ground_h_est_prefix_configuration_sha256"]
                ).reshape(-1)[0])
            else:
                prefix_ground = prefix_ground_frames = prefix_input_hashes = None
                prefix_semantics = None
                prefix_builder_sha = prefix_configuration_sha = None
            whole_episode_ground_present = "ground_h_est" in cache.files
            whole_episode_ground = (
                float(np.asarray(cache["ground_h_est"]).reshape(-1)[0])
                if external_ground_scale is None
                and whole_episode_ground_present
                and np.asarray(cache["ground_h_est"]).size == 1 else None)
            signature = (
                str(np.asarray(cache["precompute_signature"]).reshape(-1)[0])
                if "precompute_signature" in cache.files else None)
            schema_version = (
                int(np.asarray(cache["cache_schema_version"]).reshape(-1)[0])
                if "cache_schema_version" in cache.files else None)
    except DeploymentPoseError:
        raise
    except Exception as exc:
        raise DeploymentPoseError(f"cannot read LingBot cache: {exc}") from exc
    if (pose_raw.shape != (int(episode_frame_count), 9)
            or not np.issubdtype(pose_raw.dtype, np.number)
            or not np.isfinite(pose_raw[:int(decision_frame)]).all()):
        raise DeploymentPoseError(
            "causal cam_pose_enc must be dense, finite, and raw-frame aligned")
    pose = np.asarray(pose_raw, dtype=np.float64)
    # cam_frame_indices maps sparse K/V rows, not cam_pose_enc.  Its complete
    # policy validation was performed by validate_cache_files above.
    if frame_raw.ndim != 1:
        raise DeploymentPoseError("cam_frame_indices must be one-dimensional")
    try:
        frames = frame_raw.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise DeploymentPoseError("cam_frame_indices is not integer-valued") from exc
    if not np.array_equal(frame_raw, frames):
        raise DeploymentPoseError("cam_frame_indices must contain exact integers")
    if not np.array_equal(frames, np.asarray(layout.cam_frame_indices)):
        raise DeploymentPoseError(
            "cam_frame_indices changed after paired-cache validation")
    if external_ground_scale is not None:
        if episode_dir is None or episode_root is None:
            raise DeploymentPoseError(
                "external ground scale requires the episode/root binding")
        record = external_ground_scale.record
        if record.get("valid") is not True:
            raise DeploymentPoseError(
                "external causal ground scale is unavailable for this episode")
        prefix_end = record.get("prefix_end_frame_exclusive")
        if (isinstance(prefix_end, bool) or not isinstance(prefix_end, int)
                or prefix_end < 1 or prefix_end > int(decision_frame)):
            raise DeploymentPoseError(
                "external ground-scale prefix crosses the decision frame")
        actual_rgb = _external_rgb_prefix_record(
            episode_dir, episode_root, prefix_end)
        if actual_rgb != record.get("rgb_prefix"):
            raise DeploymentPoseError(
                "external ground-scale RGB prefix content changed")
        pose_prefix = pose_raw[:prefix_end]
        if pose_prefix.dtype.str != record.get("cam_pose_prefix_dtype"):
            raise DeploymentPoseError(
                "external ground-scale camera-pose prefix dtype changed")
        pose_prefix_sha = _ndarray_prefix_sha256(pose_prefix)
        if pose_prefix_sha != record.get("cam_pose_prefix_sha256"):
            raise DeploymentPoseError(
                "external ground-scale camera-pose prefix changed")
        if (record.get("cache_schema_version") != schema_version
                or record.get("precompute_signature") != signature):
            raise DeploymentPoseError(
                "external ground scale was built from a different cache generation")
        record_camera_height = record.get("camera_height_m")
        if (isinstance(record_camera_height, bool)
                or not isinstance(record_camera_height, (int, float))
                or not math.isclose(float(record_camera_height),
                                    float(camera_height_m),
                                    rel_tol=0.0, abs_tol=1e-9)):
            raise DeploymentPoseError(
                "external ground-scale camera height changed")
        configuration = external_ground_scale.configuration
        try:
            ground_h = float(record["ground_h_est_raw"])
            metric_scale = float(record["metric_scale_m_per_raw"])
            bias = float(configuration["bias_correction"])
            scale_min = float(configuration["scale_min"])
            scale_max = float(configuration["scale_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentPoseError(
                "external ground-scale numerical record is malformed") from exc
        if not math.isfinite(ground_h) or ground_h <= 0.0:
            raise DeploymentPoseError(
                "external ground-scale floor height is invalid")
        raw_scale = bias * float(camera_height_m) / ground_h
        expected_scale = min(max(raw_scale, scale_min), scale_max)
        if (not all(math.isfinite(value) for value in (
                ground_h, metric_scale, raw_scale, expected_scale))
                or ground_h <= 0.0
                or not math.isclose(metric_scale, expected_scale,
                                    rel_tol=1e-6, abs_tol=1e-6)):
            raise DeploymentPoseError(
                "external ground scale disagrees with its pinned formula")
        scale_provenance = {
            "metric_scale_source": "external_causal_first_prefix_v1",
            "external_scale_artifact_path": external_ground_scale.artifact_path,
            "external_scale_artifact_sha256": (
                external_ground_scale.artifact_sha256),
            "external_scale_producer_sha256": (
                external_ground_scale.producer_sha256),
            "external_scale_configuration_sha256": (
                external_ground_scale.configuration_sha256),
            "external_scale_estimator_kind": (
                external_ground_scale.estimator_kind),
            "external_scale_lingbot_commit": (
                external_ground_scale.lingbot_commit),
            "external_scale_weights_sha256": (
                external_ground_scale.weights_sha256),
            "external_scale_stream_source_sha256": (
                external_ground_scale.stream_source_sha256),
            "external_scale_prefix_end_frame_exclusive": prefix_end,
            "external_scale_rgb_prefix": actual_rgb,
            "external_scale_cam_pose_prefix_sha256": pose_prefix_sha,
            "ground_h_est_prefix_raw_at_decision": ground_h,
            "ground_h_est_prefix_semantics": (
                "causal_first_prefix_floor_hist_v1"),
        }
    else:
        assert prefix_ground is not None
        assert prefix_ground_frames is not None
        assert prefix_input_hashes is not None
        if prefix_ground.shape != (int(episode_frame_count),):
            raise DeploymentPoseError(
                "ground_h_est_prefix must have one dense causal value per raw frame")
        if prefix_ground_frames.shape != (int(episode_frame_count),):
            raise DeploymentPoseError(
                "ground_h_est_prefix_frame_indices must be dense [num_frames]")
        if prefix_input_hashes.shape != (int(episode_frame_count),):
            raise DeploymentPoseError(
                "ground_h_est_prefix_causal_prefix_sha256 must be dense [num_frames]")
        try:
            prefix_frames = prefix_ground_frames.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise DeploymentPoseError(
                "ground_h_est_prefix_frame_indices is not integer-valued") from exc
        expected_frames = np.arange(int(episode_frame_count), dtype=np.int64)
        if (not np.array_equal(prefix_ground_frames, prefix_frames)
                or not np.array_equal(prefix_frames, expected_frames)):
            raise DeploymentPoseError(
                "ground_h_est_prefix_frame_indices must equal raw frame indices")
        if prefix_semantics != CAUSAL_GROUND_PREFIX_SEMANTICS:
            raise DeploymentPoseError(
                "ground_h_est_prefix semantics is missing or unsupported")
        for label, digest in (
                ("builder", prefix_builder_sha),
                ("configuration", prefix_configuration_sha)):
            if not _valid_hex_digest(digest):
                raise DeploymentPoseError(
                    f"ground_h_est_prefix {label} SHA256 is invalid")
        expected_pins = {
            "builder": expected_prefix_builder_sha256,
            "configuration": expected_prefix_configuration_sha256,
        }
        actual_pins = {
            "builder": prefix_builder_sha,
            "configuration": prefix_configuration_sha,
        }
        for label, expected in expected_pins.items():
            if not _valid_hex_digest(expected):
                raise DeploymentPoseError(
                    f"expected ground_h_est_prefix {label} SHA256 is not pinned")
            if actual_pins[label] != expected:
                raise DeploymentPoseError(
                    f"ground_h_est_prefix {label} differs from its expected pin")
        causal_hash = str(prefix_input_hashes[int(decision_frame) - 1])
        if causal_hash != expected_causal_prefix_sha256:
            raise DeploymentPoseError(
                "ground_h_est_prefix was not built from this exact causal prefix")
        ground_h = float(prefix_ground[int(decision_frame) - 1])
        if not math.isfinite(ground_h) or ground_h <= 0.0:
            raise DeploymentPoseError(
                "causal ground_h_est_prefix is unavailable at the decision frame")
        if not math.isfinite(float(camera_height_m)) or float(camera_height_m) <= 0.0:
            raise DeploymentPoseError("camera_height_m must be finite and positive")
        raw_scale = GROUND_BIAS_CORRECTION * float(camera_height_m) / ground_h
        metric_scale = min(max(raw_scale, GROUND_SCALE_RANGE[0]),
                           GROUND_SCALE_RANGE[1])
        scale_provenance = {
            "metric_scale_source": (
                "clamp(1.15 * camera_height_m / "
                "causal_ground_h_est_prefix, 0.8, 6.0)"),
            "ground_h_est_prefix_semantics": prefix_semantics,
            "ground_h_est_prefix_builder_sha256": prefix_builder_sha,
            "ground_h_est_prefix_configuration_sha256": (
                prefix_configuration_sha),
            "ground_h_est_prefix_causal_prefix_sha256": causal_hash,
            "ground_h_est_prefix_raw_at_decision": ground_h,
        }
    rows = {}
    for frame in range(int(decision_frame)):
        rotation = quaternion_xyzw_to_matrix(pose[frame, 3:7])
        # LingBot's camera map ground plane is x-z and optical +z is forward.
        forward = rotation[:, 2]
        norm = math.hypot(float(forward[0]), float(forward[2]))
        if norm < 1e-6:
            raise DeploymentPoseError("LingBot forward axis leaves the x-z ground plane")
        rows[frame] = SE2Pose(
            metric_scale * float(pose[frame, 0]),
            metric_scale * float(pose[frame, 2]),
            math.atan2(float(forward[2]), float(forward[0])),
        )
    return rows, {
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": cache_sha,
        "aggregator_cache_path": str(aggregator_cache_path.resolve()),
        "aggregator_cache_sha256": aggregator_sha,
        "future_cache_payload_hashed": external_ground_scale is None,
        "cache_schema_version": schema_version,
        "precompute_signature": signature,
        "pose_frame_mapping": (
            "dense_cam_pose_enc_row_equals_raw_frame; sparse cam_frame_indices "
            "validated only for camera KV"
        ),
        "sparse_camera_kv_frame_count": len(frames),
        "ground_plane": "LingBot_xz",
        "metric_scale_m_per_raw": metric_scale,
        "metric_scale_unclamped_m_per_raw": raw_scale,
        **scale_provenance,
        "whole_episode_ground_h_est_present_but_not_used": (
            whole_episode_ground_present),
        "camera_height_m": float(camera_height_m),
        "gt_sim2_used": False,
        "causal_dense_pose_count": len(rows),
        "latest_pose_frame": max(rows),
        "pose_age_frames": 0,
    }


def _load_depth(path: Path, unit_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CandidateBuildError("Pillow is required to read depth PNGs") from exc
    try:
        with Image.open(path) as image:
            depth = np.asarray(image).copy()
    except Exception as exc:
        raise CandidateBuildError(f"cannot read depth image {path}: {exc}") from exc
    if depth.ndim == 3:
        if depth.shape[-1] != 1:
            raise CandidateBuildError(f"depth image is not single-channel: {path}")
        depth = depth[..., 0]
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise CandidateBuildError(f"depth image is not numeric HxW: {path}")
    valid = depth > 0
    truncated = (
        depth == np.iinfo(depth.dtype).max
        if np.issubdtype(depth.dtype, np.integer)
        else np.zeros(depth.shape, dtype=bool)
    )
    result = depth.astype(np.float64) * float(unit_m)
    if not np.isfinite(result).all():
        raise CandidateBuildError(f"depth image contains non-finite values: {path}")
    return result, valid, truncated


def _scan_frames(
    poses: Mapping[int, SE2Pose],
    *,
    scan_stride: int,
) -> list[int]:
    frames = sorted(poses)
    if not frames:
        raise CandidateBuildError("pose arm has no causal frames")
    selected = [frame for frame in frames if frame % int(scan_stride) == 0]
    if frames[-1] not in selected:
        selected.append(frames[-1])
    return sorted(set(selected))


def scans_from_pose_rows(
    *,
    poses: Mapping[int, SE2Pose],
    rows: Sequence[Mapping[str, object]],
    depth_root: Path,
    depth_unit_m: float,
    depth_column_stride: int,
    scan_stride: int,
    config: FrontierConfig,
) -> list[PlanarScan]:
    scans = []
    for frame in _scan_frames(poses, scan_stride=scan_stride):
        if not 0 <= frame < len(rows):
            raise CandidateBuildError("pose frame lies outside parquet rows")
        intrinsic = _matrix(
            rows[frame]["observation.camera_intrinsic"], (3, 3),
            f"intrinsic[{frame}]")
        depth, valid, truncated = _load_depth(
            depth_root / f"{frame}.png", depth_unit_m)
        scans.append(depth_to_planar_scan(
            depth,
            intrinsic,
            poses[frame],
            frame,
            valid_mask=valid,
            truncated_mask=truncated,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            column_stride=depth_column_stride,
        ))
    return scans


def load_patch_scores(
    path: Path | None,
    manifest_sha256: str,
    *,
    expected_encoder_checkpoint_sha256: str | None = None,
    expected_feature_builder_sha256: str | None = None,
    expected_configuration_sha256: str | None = None,
) -> tuple[dict, dict]:
    if path is None:
        return {}, {
            "status": "absent",
            "content_sha256": None,
            "encoder_checkpoint_sha256": None,
            "feature_builder_sha256": None,
            "configuration_sha256": None,
            "missing_behavior": "mask_zero_and_skip_goal_patch_top2",
        }
    if not path.is_file():
        raise CandidateBuildError(f"patch-score artifact is missing: {path}")
    content = path.read_bytes()
    try:
        artifact = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CandidateBuildError("patch-score artifact is invalid JSON") from exc
    expected_keys = {
        "schema_version",
        "input_manifest_sha256",
        "encoder_checkpoint_sha256",
        "feature_builder_sha256",
        "configuration_sha256",
        "records",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_keys:
        raise CandidateBuildError("patch-score artifact keys differ from protocol")
    if artifact["schema_version"] != PATCH_SCORE_SCHEMA_VERSION:
        raise CandidateBuildError("patch-score schema version changed")
    if artifact["input_manifest_sha256"] != manifest_sha256:
        raise CandidateBuildError("patch-score artifact targets a different manifest")
    sha_fields = (
        "encoder_checkpoint_sha256",
        "feature_builder_sha256",
        "configuration_sha256",
    )
    expected = {
        "encoder_checkpoint_sha256": expected_encoder_checkpoint_sha256,
        "feature_builder_sha256": expected_feature_builder_sha256,
        "configuration_sha256": expected_configuration_sha256,
    }
    for field in sha_fields:
        digest = artifact[field]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise CandidateBuildError(f"patch-score {field} is invalid")
        if expected[field] is None:
            raise CandidateBuildError(
                f"patch-score input requires an explicit expected {field} pin")
        if digest != expected[field]:
            raise CandidateBuildError(
                f"patch-score {field} differs from its expected pin")
    records = artifact["records"]
    if not isinstance(records, list):
        raise CandidateBuildError("patch-score records must be a list")
    by_sample = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
                "sample_id", "causal_prefix_sha256", "goal_sha256", "frame_scores"}:
            raise CandidateBuildError("patch-score record keys differ from protocol")
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in by_sample:
            raise CandidateBuildError("patch-score sample_id is empty or duplicated")
        frame_scores = record["frame_scores"]
        if not isinstance(frame_scores, list):
            raise CandidateBuildError("patch frame_scores must be a list")
        scores = {}
        for row in frame_scores:
            if not isinstance(row, Mapping) or set(row) != {"frame_index", "score"}:
                raise CandidateBuildError("patch frame-score keys differ from protocol")
            frame = row["frame_index"]
            score = row["score"]
            if (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                    or frame in scores or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not -1.0 <= float(score) <= 1.0):
                raise CandidateBuildError("patch frame score is invalid")
            scores[frame] = float(score)
        for field in ("causal_prefix_sha256", "goal_sha256"):
            digest = record[field]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise CandidateBuildError(
                    f"patch-score record {field} is invalid")
        by_sample[sample_id] = {
            "causal_prefix_sha256": record["causal_prefix_sha256"],
            "goal_sha256": record["goal_sha256"],
            "frame_scores": scores,
        }
    return by_sample, {
        "status": "loaded",
        "path": str(path.resolve()),
        "content_sha256": sha256_bytes(content),
        "encoder_checkpoint_sha256": artifact["encoder_checkpoint_sha256"],
        "feature_builder_sha256": artifact["feature_builder_sha256"],
        "configuration_sha256": artifact["configuration_sha256"],
        "record_count": len(by_sample),
    }


def _scene_records(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        raise CandidateBuildError("causal manifest scenes must be a list")
    result = {}
    for scene in scenes:
        if not isinstance(scene, Mapping) or not isinstance(scene.get("scene"), str):
            raise CandidateBuildError("causal manifest scene record is malformed")
        if scene["scene"] in result:
            raise CandidateBuildError("causal manifest scene is duplicated")
        result[str(scene["scene"])] = scene
    return result


def load_external_ground_scale_bindings(
    *,
    path: Path | None,
    expected_artifact_sha256: str | None,
    expected_producer_sha256: str | None,
    expected_configuration_sha256: str | None,
    expected_lingbot_commit: str | None,
    expected_weights_sha256: str | None,
    expected_stream_source_sha256: str | None,
    manifest: Mapping[str, object],
    manifest_sha256: str,
    allowed_scenes: set[str],
    legacy_builder_pin: str | None,
    legacy_configuration_pin: str | None,
) -> tuple[dict[tuple[str, str], ExternalGroundScaleBinding], dict]:
    """Load one pinned external scale artifact without importing its producer.

    The producer imports this module for canonical I/O helpers, so a reverse
    import would be cyclic.  The short hashing contract is intentionally
    duplicated here and locked by an end-to-end unit test.
    """
    supplied = (
        path,
        expected_artifact_sha256,
        expected_producer_sha256,
        expected_configuration_sha256,
        expected_lingbot_commit,
        expected_weights_sha256,
        expected_stream_source_sha256,
    )
    if not any(value is not None for value in supplied):
        return {}, {
            "mode": "legacy_dense_cache_prefix",
            "artifact_path": None,
            "artifact_sha256": None,
        }
    if not all(value is not None for value in supplied):
        raise CandidateBuildError(
            "external causal ground scale requires its path and every exact pin")
    if legacy_builder_pin is not None or legacy_configuration_pin is not None:
        raise CandidateBuildError(
            "external and legacy dense ground-scale pins are mutually exclusive")
    assert path is not None
    assert expected_artifact_sha256 is not None
    assert expected_producer_sha256 is not None
    assert expected_configuration_sha256 is not None
    assert expected_lingbot_commit is not None
    assert expected_weights_sha256 is not None
    assert expected_stream_source_sha256 is not None
    if (not _valid_hex_digest(expected_artifact_sha256)
            or not _valid_hex_digest(expected_producer_sha256)
            or not _valid_hex_digest(expected_configuration_sha256)
            or not _valid_hex_digest(expected_lingbot_commit, (40, 64))
            or not _valid_hex_digest(expected_weights_sha256)
            or not _valid_hex_digest(expected_stream_source_sha256)):
        raise CandidateBuildError("external ground-scale pin format is invalid")
    if not path.is_file():
        raise CandidateBuildError(
            f"external causal ground-scale artifact is missing: {path}")
    raw = path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_artifact_sha256:
        raise CandidateBuildError(
            "external causal ground-scale artifact SHA changed")
    try:
        artifact = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CandidateBuildError(
            "external causal ground-scale artifact is invalid JSON") from exc
    if not isinstance(artifact, Mapping):
        raise CandidateBuildError(
            "external causal ground-scale artifact must be an object")
    if raw != canonical_json_bytes(artifact):
        raise CandidateBuildError(
            "external causal ground-scale artifact is not canonical JSON")
    if artifact.get("schema_version") != CAUSAL_GROUND_SCALE_SCHEMA_VERSION:
        raise CandidateBuildError(
            "external causal ground-scale schema is unsupported")
    provenance = artifact.get("provenance")
    configuration = artifact.get("configuration")
    records = artifact.get("records")
    if (not isinstance(provenance, Mapping)
            or not isinstance(configuration, Mapping)
            or not isinstance(records, list)):
        raise CandidateBuildError(
            "external causal ground-scale artifact structure is malformed")
    if provenance.get("input_manifest_sha256") != manifest_sha256:
        raise CandidateBuildError(
            "external ground scale was built from a different manifest")
    configuration_sha = sha256_bytes(canonical_json_bytes(configuration))
    if (configuration_sha != expected_configuration_sha256
            or provenance.get("configuration_sha256") != configuration_sha):
        raise CandidateBuildError(
            "external ground-scale configuration pin does not match")
    if provenance.get("producer_source_sha256") != expected_producer_sha256:
        raise CandidateBuildError(
            "external ground-scale producer pin does not match")
    estimator = provenance.get("estimator")
    if not isinstance(estimator, Mapping):
        raise CandidateBuildError(
            "external ground-scale estimator provenance is missing")
    expected_estimator = {
        "kind": "frozen_lingbot_compute_metric_scale_prefix",
        "lingbot_commit": expected_lingbot_commit,
        "weights_sha256": expected_weights_sha256,
        "lingbot_stream_source_sha256": expected_stream_source_sha256,
    }
    for field, expected in expected_estimator.items():
        if estimator.get(field) != expected:
            raise CandidateBuildError(
                f"external ground-scale estimator {field} pin does not match")
    required_configuration = {
        "prefix_frame_cap", "num_scale_frames", "bias_correction",
        "scale_min", "scale_max",
    }
    if not required_configuration.issubset(configuration):
        raise CandidateBuildError(
            "external ground-scale configuration is incomplete")
    if (type(configuration["prefix_frame_cap"]) is not int
            or type(configuration["num_scale_frames"]) is not int):
        raise CandidateBuildError(
            "external ground-scale frame configuration must use integers")
    try:
        prefix_cap = int(configuration["prefix_frame_cap"])
        num_scale = int(configuration["num_scale_frames"])
        bias = float(configuration["bias_correction"])
        scale_min = float(configuration["scale_min"])
        scale_max = float(configuration["scale_max"])
    except (TypeError, ValueError) as exc:
        raise CandidateBuildError(
            "external ground-scale configuration is not numeric") from exc
    if (prefix_cap < num_scale or num_scale < 1
            or not all(math.isfinite(value) for value in (
                bias, scale_min, scale_max))
            or bias <= 0.0 or not 0.0 < scale_min < scale_max):
        raise CandidateBuildError(
            "external ground-scale configuration values are invalid")

    scene_table = _scene_records(manifest)
    manifest_samples = manifest.get("samples")
    if not isinstance(manifest_samples, list):
        raise CandidateBuildError("causal manifest samples must be a list")
    expected_by_episode: dict[tuple[str, str], dict[str, object]] = {}
    for sample in manifest_samples:
        if not isinstance(sample, Mapping):
            raise CandidateBuildError("causal manifest sample is malformed")
        scene = sample.get("scene")
        episode = sample.get("source_episode")
        sample_id = sample.get("sample_id")
        decision = sample.get("decision_frame")
        split_role = sample.get("split_role")
        if (not isinstance(scene, str) or scene not in scene_table
                or not isinstance(episode, str)
                or not isinstance(sample_id, str)
                or isinstance(decision, bool) or not isinstance(decision, int)
                or split_role not in ("train", "development")):
            raise CandidateBuildError(
                "causal manifest sample identity is malformed")
        key = scene, episode
        row = expected_by_episode.setdefault(key, {
            "sample_ids": [], "decision_frames": set(),
            "split_roles": set(),
        })
        row["sample_ids"].append(sample_id)  # type: ignore[union-attr]
        row["decision_frames"].add(decision)  # type: ignore[union-attr]
        row["split_roles"].add(split_role)  # type: ignore[union-attr]
    bindings: dict[tuple[str, str], ExternalGroundScaleBinding] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CandidateBuildError(
                "external ground-scale record is malformed")
        scene, episode = record.get("scene"), record.get("episode")
        if not isinstance(scene, str) or not isinstance(episode, str):
            raise CandidateBuildError(
                "external ground-scale episode identity is malformed")
        key = scene, episode
        if key not in expected_by_episode or key in bindings:
            raise CandidateBuildError(
                "external ground-scale episode is unknown or duplicated")
        expected = expected_by_episode[key]
        roles = expected["split_roles"]
        if (not isinstance(roles, set) or len(roles) != 1
                or record.get("split_role") != next(iter(roles))):
            raise CandidateBuildError(
                "external ground-scale split role differs from the manifest")
        expected_ids = sorted(expected["sample_ids"])  # type: ignore[arg-type]
        expected_decisions = sorted(expected["decision_frames"])  # type: ignore[arg-type]
        if (record.get("sample_ids") != expected_ids
                or record.get("decision_frames") != expected_decisions
                or record.get("earliest_decision_frame")
                != min(expected_decisions)):
            raise CandidateBuildError(
                "external ground-scale sample/decision binding changed")
        scene_record = scene_table[scene]
        episode_record = _episode_record(scene_record, episode)
        if record.get("episode_frame_count") != episode_record.get("n_frames"):
            raise CandidateBuildError(
                "external ground-scale frame count differs from the manifest")
        expected_prefix = min(prefix_cap, min(expected_decisions))
        if (record.get("prefix_end_frame_exclusive") != expected_prefix
                or expected_prefix < num_scale):
            raise CandidateBuildError(
                "external ground-scale causal prefix length changed")
        rgb_prefix = record.get("rgb_prefix")
        pose_sha = record.get("cam_pose_prefix_sha256")
        pose_dtype = record.get("cam_pose_prefix_dtype")
        if (not isinstance(rgb_prefix, Mapping)
                or rgb_prefix.get("frame_count") != expected_prefix
                or not _valid_hex_digest(rgb_prefix.get(
                    "path_sequence_sha256"))
                or not _valid_hex_digest(rgb_prefix.get(
                    "content_sequence_sha256"))
                or not _valid_hex_digest(pose_sha)
                or not isinstance(pose_dtype, str) or not pose_dtype):
            raise CandidateBuildError(
                "external ground-scale prefix provenance is malformed")
        valid = record.get("valid")
        ground_h = record.get("ground_h_est_raw")
        metric_scale = record.get("metric_scale_m_per_raw")
        if type(valid) is not bool:
            raise CandidateBuildError(
                "external ground-scale validity flag is malformed")
        if valid:
            if (isinstance(ground_h, bool) or isinstance(metric_scale, bool)
                    or not isinstance(ground_h, (int, float))
                    or not isinstance(metric_scale, (int, float))
                    or not math.isfinite(float(ground_h))
                    or not math.isfinite(float(metric_scale))
                    or float(ground_h) <= 0.0):
                raise CandidateBuildError(
                    "external ground-scale valid estimate is malformed")
        elif ground_h is not None or metric_scale is not None:
            raise CandidateBuildError(
                "external ground-scale invalid estimate is not neutralized")
        bindings[key] = ExternalGroundScaleBinding(
            artifact_path=str(path.resolve()),
            artifact_sha256=actual_sha,
            producer_sha256=expected_producer_sha256,
            configuration_sha256=configuration_sha,
            estimator_kind=str(estimator["kind"]),
            lingbot_commit=expected_lingbot_commit,
            weights_sha256=expected_weights_sha256,
            stream_source_sha256=expected_stream_source_sha256,
            configuration=dict(configuration),
            record=dict(record),
        )
    required_keys = {
        key for key in expected_by_episode if key[0] in allowed_scenes
    }
    missing = required_keys - set(bindings)
    if missing:
        raise CandidateBuildError(
            "external ground-scale artifact lacks selected episodes: "
            f"{sorted(missing)[:3]}")
    return bindings, {
        "mode": "external_causal_first_prefix_v1",
        "artifact_path": str(path.resolve()),
        "artifact_sha256": actual_sha,
        "producer_source_sha256": expected_producer_sha256,
        "configuration_sha256": configuration_sha,
        "estimator": dict(estimator),
        "selected_episode_count": len(required_keys),
    }


def _episode_record(
    scene_record: Mapping[str, object], episode: str,
) -> Mapping[str, object]:
    episodes = scene_record.get("selected_episodes")
    if not isinstance(episodes, list):
        raise CandidateBuildError("scene selected_episodes is malformed")
    matches = [
        row for row in episodes
        if isinstance(row, Mapping) and row.get("episode") == episode
    ]
    if len(matches) != 1:
        raise CandidateBuildError(
            f"expected one selected episode record for {episode}, found {len(matches)}")
    return matches[0]


def _validate_declared_flow_file(
    episode_record: Mapping[str, object],
    flow_root: Path,
    path: Path,
) -> None:
    flow = episode_record.get("flow_cache")
    if not isinstance(flow, Mapping) or not isinstance(flow.get("files"), list):
        raise DeploymentPoseError("causal manifest flow-cache record is malformed")
    try:
        relative = path.resolve().relative_to(flow_root.resolve()).as_posix()
    except ValueError as exc:
        raise DeploymentPoseError("flow cache escapes its declared root") from exc
    matches = [
        row for row in flow["files"]
        if isinstance(row, Mapping) and row.get("path") == relative
    ]
    if len(matches) != 1:
        raise DeploymentPoseError(
            f"flow cache is not uniquely declared in the causal manifest: {relative}")
    expected_bytes = matches[0].get("bytes")
    if (isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int)
            or expected_bytes < 0):
        raise DeploymentPoseError("flow-cache byte-size record is malformed")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise DeploymentPoseError(
            f"flow cache changed after causal manifest freeze: {relative}")


def _not_requested_proxy(proposal: Mapping[str, object]) -> dict:
    return {
        "status": "not_requested",
        "label_valid": False,
        "labeler_provenance": None,
        "positive_margin_m": None,
        "labels": [],
        "universe_has_positive": False,
        "shortlist_has_positive": False,
        "coverage_miss": False,
        "proposal_sha256": sha256_bytes(canonical_json_bytes(proposal)),
    }


def _arm_record(
    *,
    arm: str,
    sample: Mapping[str, object],
    poses: Mapping[int, SE2Pose],
    rows: Sequence[Mapping[str, object]],
    depth_root: Path,
    patch_scores: Mapping[int, float] | None,
    pose_provenance: Mapping[str, object],
    labeler: ProposalProxyLabeler | None,
    depth_unit_m: float,
    depth_column_stride: int,
    scan_stride: int,
    config: FrontierConfig,
) -> dict:
    scans = scans_from_pose_rows(
        poses=poses,
        rows=rows,
        depth_root=depth_root,
        depth_unit_m=depth_unit_m,
        depth_column_stride=depth_column_stride,
        scan_stride=scan_stride,
        config=config,
    )
    proposal = generate_frontier_proposals(
        scans,
        scans[-1].pose,
        patch_scores_by_frame=patch_scores,
        config=config,
    )
    proxy = (
        attach_proposal_proxy_labels(
            sample_id=str(sample["sample_id"]),
            arm=arm,
            proposal=proposal,
            labeler=labeler,
        )
        if labeler is not None else _not_requested_proxy(proposal)
    )
    return {
        "arm": arm,
        "deployment_eligible_pose_source": arm == DEPLOYMENT_ARM,
        "pose_provenance": dict(pose_provenance),
        "proposal": proposal,
        "proposal_proxy": proxy,
    }


def _invalid_arm(
    arm: str,
    reason: str,
    *,
    pose_provenance: Mapping[str, object] | None = None,
) -> dict:
    proposal = invalid_proposal(reason)
    return {
        "arm": arm,
        "deployment_eligible_pose_source": arm == DEPLOYMENT_ARM,
        "pose_provenance": dict(pose_provenance or {}),
        "proposal": proposal,
        "proposal_proxy": _not_requested_proxy(proposal),
    }


def _config_record(
    config: FrontierConfig,
    *,
    depth_unit_m: float,
    depth_column_stride: int,
    scan_stride: int,
) -> dict:
    return {
        "grid_resolutions_m": list(config.resolutions_m),
        "min_range_m": config.min_range_m,
        "max_range_m": config.max_range_m,
        "ray_step_fraction": config.ray_step_fraction,
        "minimum_component_cells": config.minimum_component_cells,
        "max_representatives_per_component": config.max_representatives_per_component,
        "representative_spacing_m": config.representative_spacing_m,
        "minimum_candidate_distance_m": config.minimum_candidate_distance_m,
        "maximum_candidate_distance_m": config.maximum_candidate_distance_m,
        "occupied_clearance_radius_m": config.occupied_clearance_radius_m,
        "spatial_nms_m": config.spatial_nms_m,
        "bearing_nms_rad": config.bearing_nms_rad,
        "radial_nms_m": config.radial_nms_m,
        "context_frames": config.context_frames,
        "context_view_half_angle_rad": config.context_view_half_angle_rad,
        "max_shortlist": config.max_shortlist,
        "shortlist_slots": [
            "goal_patch_top2", "topology_top2", "angular_diverse_top2"],
        "depth_unit_m": float(depth_unit_m),
        "depth_column_stride": int(depth_column_stride),
        "scan_stride": int(scan_stride),
    }


def build_artifact(
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
    manifest_sha256: str,
    patch_score_path: Path | None = None,
    expected_patch_encoder_checkpoint_sha256: str | None = None,
    expected_patch_feature_builder_sha256: str | None = None,
    expected_patch_configuration_sha256: str | None = None,
    expected_ground_prefix_builder_sha256: str | None = None,
    expected_ground_prefix_configuration_sha256: str | None = None,
    causal_ground_scale_path: Path | None = None,
    expected_causal_ground_scale_sha256: str | None = None,
    expected_ground_scale_producer_sha256: str | None = None,
    expected_ground_scale_configuration_sha256: str | None = None,
    expected_ground_scale_lingbot_commit: str | None = None,
    expected_ground_scale_weights_sha256: str | None = None,
    expected_ground_scale_stream_source_sha256: str | None = None,
    selected_scenes: Sequence[str] = (),
    labeler: ProposalProxyLabeler | None = None,
    depth_unit_m: float = 1e-4,
    depth_column_stride: int = 8,
    scan_stride: int = 4,
    config: FrontierConfig = FrontierConfig(),
) -> dict:
    if manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA_VERSION:
        raise CandidateBuildError("input is not the frozen NLSR-V2 causal manifest")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != manifest_sha256:
        raise CandidateBuildError(
            f"causal manifest SHA mismatch: {actual_manifest_sha} != {manifest_sha256}")
    try:
        manifest_from_file = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateBuildError("pinned causal manifest is invalid JSON") from exc
    if manifest_from_file != manifest:
        raise CandidateBuildError(
            "in-memory manifest differs from the pinned manifest file")
    if (not math.isfinite(float(depth_unit_m)) or float(depth_unit_m) <= 0.0
            or isinstance(depth_column_stride, bool)
            or not isinstance(depth_column_stride, int) or depth_column_stride < 1
            or isinstance(scan_stride, bool) or not isinstance(scan_stride, int)
            or scan_stride < 1):
        raise CandidateBuildError("depth/scanning configuration is invalid")
    roots = manifest.get("input_roots")
    if not isinstance(roots, Mapping):
        raise CandidateBuildError("causal manifest input_roots is missing")
    try:
        episode_root = Path(str(roots["episode_root"]))
        flow_root = Path(str(roots["flow_cache_root"]))
    except KeyError as exc:
        raise CandidateBuildError("causal manifest roots are incomplete") from exc
    if not episode_root.is_dir() or not flow_root.is_dir():
        raise CandidateBuildError("causal manifest episode/flow roots are unavailable")
    scene_records = _scene_records(manifest)
    requested = tuple(dict.fromkeys(map(str, selected_scenes)))
    if requested:
        unknown = set(requested) - set(scene_records)
        if unknown:
            raise CandidateBuildError(f"requested scenes are absent: {sorted(unknown)}")
        allowed_scenes = set(requested)
    else:
        allowed_scenes = set(scene_records)
    patch_by_sample, patch_provenance = load_patch_scores(
        patch_score_path,
        manifest_sha256,
        expected_encoder_checkpoint_sha256=(
            expected_patch_encoder_checkpoint_sha256),
        expected_feature_builder_sha256=(
            expected_patch_feature_builder_sha256),
        expected_configuration_sha256=(
            expected_patch_configuration_sha256),
    )
    external_scales, external_scale_provenance = (
        load_external_ground_scale_bindings(
            path=causal_ground_scale_path,
            expected_artifact_sha256=(
                expected_causal_ground_scale_sha256),
            expected_producer_sha256=(
                expected_ground_scale_producer_sha256),
            expected_configuration_sha256=(
                expected_ground_scale_configuration_sha256),
            expected_lingbot_commit=(
                expected_ground_scale_lingbot_commit),
            expected_weights_sha256=(
                expected_ground_scale_weights_sha256),
            expected_stream_source_sha256=(
                expected_ground_scale_stream_source_sha256),
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            allowed_scenes=allowed_scenes,
            legacy_builder_pin=expected_ground_prefix_builder_sha256,
            legacy_configuration_pin=(
                expected_ground_prefix_configuration_sha256),
        )
    )
    external_scale_mode = bool(external_scales)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise CandidateBuildError("causal manifest samples must be a list")
    manifest_sample_ids = {
        sample.get("sample_id")
        for sample in samples
        if isinstance(sample, Mapping)
        and isinstance(sample.get("sample_id"), str)
    }
    unknown_patch_samples = set(patch_by_sample) - manifest_sample_ids
    if unknown_patch_samples:
        raise CandidateBuildError(
            "patch-score artifact contains samples absent from the causal "
            f"manifest: {sorted(unknown_patch_samples)[:3]}")

    configuration = _config_record(
        config,
        depth_unit_m=depth_unit_m,
        depth_column_stride=depth_column_stride,
        scan_stride=scan_stride,
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("novel_frontier_candidates_v2.py"),
        (Path(__file__).resolve().parents[1]
         / "InternNav/internnav/model/basemodel/memnav/cache_schema.py"),
    )
    source_hashes = {path.name: sha256_file(path) for path in source_paths}
    generator_sha = sha256_bytes(canonical_json_bytes(source_hashes))
    episode_cache: dict[tuple[str, str], dict] = {}
    cache_validation_memo: dict[tuple[Path, Path, int], tuple[
        tuple[int, int], tuple[int, int], object]] = {}
    cache_content_sha_memo: dict[Path, tuple[tuple[int, int], str]] = {}
    records = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise CandidateBuildError("causal manifest sample is malformed")
        scene = str(sample.get("scene", ""))
        if scene not in allowed_scenes:
            continue
        sample_id = sample.get("sample_id")
        source_episode = sample.get("source_episode")
        decision = sample.get("decision_frame")
        if (not isinstance(sample_id, str) or not sample_id
                or not isinstance(source_episode, str) or not source_episode
                or isinstance(decision, bool) or not isinstance(decision, int)):
            raise CandidateBuildError("sample identity/decision fields are malformed")
        scene_record = scene_records[scene]
        episode_record = _episode_record(scene_record, source_episode)
        cache_key = scene, source_episode
        if cache_key not in episode_cache:
            episode_dir = episode_root / scene / source_episode
            metadata_path = episode_dir / METADATA_RELATIVE
            parquet_path = episode_dir / PARQUET_RELATIVE
            _require_file_record(
                episode_record.get("metadata"), metadata_path, "episode metadata")
            _require_file_record(
                episode_record.get("parquet"), parquet_path, "episode parquet")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CandidateBuildError("episode metadata is invalid") from exc
            if not isinstance(metadata, Mapping):
                raise CandidateBuildError("episode metadata must be an object")
            n_frames = metadata.get("n_frames")
            if (isinstance(n_frames, bool) or not isinstance(n_frames, int)
                    or n_frames < 1):
                raise CandidateBuildError("episode n_frames is invalid")
            rows = load_pose_table(parquet_path, expected_rows=n_frames)
            episode_cache[cache_key] = {
                "episode_dir": episode_dir,
                "metadata": metadata,
                "rows": rows,
                "parquet_sha256": sha256_file(parquet_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        cached = episode_cache[cache_key]
        metadata = cached["metadata"]
        rows = cached["rows"]
        assert isinstance(metadata, Mapping) and isinstance(rows, list)
        prefix_audit = audit_causal_prefix(
            sample=sample, episode_root=episode_root, rows=rows)
        goal_path = episode_root / str(sample["goal"]["path"])  # type: ignore[index]
        state_path = episode_root / str(sample["state_frame"]["path"])  # type: ignore[index]
        _require_file_record(sample.get("goal"), goal_path, "goal image")
        _require_file_record(sample.get("state_frame"), state_path, "state frame")
        patch_record = patch_by_sample.get(sample_id)
        if patch_record is not None:
            if patch_record["causal_prefix_sha256"] != prefix_audit[
                    "causal_prefix_sha256"]:
                raise CandidateBuildError(
                    f"patch-score causal prefix mismatch for {sample_id}")
            if patch_record["goal_sha256"] != sample["goal"][  # type: ignore[index]
                    "content_sha256"]:
                raise CandidateBuildError(
                    f"patch-score goal content mismatch for {sample_id}")
            scores = patch_record["frame_scores"]
        else:
            scores = None
        if scores is not None and any(frame >= decision for frame in scores):
            raise CandidateBuildError(
                f"patch scores reference future frames for {sample_id}")
        depth_root = cached["episode_dir"] / DEPTH_RELATIVE
        teacher_poses = teacher_se2_rows(
            rows,
            decision_frame=decision,
            frame_convention=str(metadata.get("frame_convention", "")),
        )
        teacher_pose_provenance = {
            "source": "generated_camera_to_world_remove_recorded_mount",
            "ground_plane": "generated_data_xy",
            "parquet_sha256": cached["parquet_sha256"],
            "metadata_sha256": cached["metadata_sha256"],
            "privileged_teacher_pose": True,
            "gt_sim2_used": False,
            "causal_pose_count": len(teacher_poses),
            "latest_pose_frame": max(teacher_poses),
            "pose_age_frames": 0,
        }
        teacher = _arm_record(
            arm=TEACHER_ARM,
            sample=sample,
            poses=teacher_poses,
            rows=rows,
            depth_root=depth_root,
            patch_scores=scores,
            pose_provenance=teacher_pose_provenance,
            labeler=labeler,
            depth_unit_m=depth_unit_m,
            depth_column_stride=depth_column_stride,
            scan_stride=scan_stride,
            config=config,
        )
        cache_path = flow_root / scene / source_episode / FLOW_RELATIVE
        aggregator_cache_path = (
            flow_root / scene / source_episode / AGGREGATOR_FLOW_RELATIVE)
        try:
            _validate_declared_flow_file(
                episode_record, flow_root, aggregator_cache_path)
            _validate_declared_flow_file(
                episode_record, flow_root, cache_path)
            deployment_poses, deployment_provenance = lingbot_deployment_se2_rows(
                aggregator_cache_path,
                cache_path,
                decision_frame=decision,
                episode_frame_count=int(metadata["n_frames"]),
                camera_height_m=float(metadata.get("camera_height_m", 0.5)),
                expected_causal_prefix_sha256=prefix_audit[
                    "causal_prefix_sha256"],
                expected_prefix_builder_sha256=(
                    expected_ground_prefix_builder_sha256 or ""),
                expected_prefix_configuration_sha256=(
                    expected_ground_prefix_configuration_sha256 or ""),
                external_ground_scale=external_scales.get(cache_key),
                episode_dir=cached["episode_dir"],
                episode_root=episode_root,
                validation_memo=cache_validation_memo,
                content_sha_memo=cache_content_sha_memo,
            )
            deployment = _arm_record(
                arm=DEPLOYMENT_ARM,
                sample=sample,
                poses=deployment_poses,
                rows=rows,
                depth_root=depth_root,
                patch_scores=scores,
                pose_provenance=deployment_provenance,
                labeler=labeler,
                depth_unit_m=depth_unit_m,
                depth_column_stride=depth_column_stride,
                scan_stride=scan_stride,
                config=config,
            )
        except (DeploymentPoseError, FrontierProposalError) as exc:
            deployment = _invalid_arm(
                DEPLOYMENT_ARM,
                f"deployment_pose_invalid:{type(exc).__name__}:{exc}",
                pose_provenance={
                    "source": "lingbot_cam_pose_enc_ground_anchored_xz",
                    "cache_path": str(cache_path.resolve()),
                    "aggregator_cache_path": str(
                        aggregator_cache_path.resolve()),
                    "gt_sim2_used": False,
                    "fail_closed": True,
                    "scale_mode": external_scale_provenance["mode"],
                    "external_scale_artifact_sha256": (
                        external_scale_provenance.get("artifact_sha256")),
                },
            )
        records.append({
            "sample_id": sample_id,
            "scene": scene,
            "source_episode": source_episode,
            "goal_episode": str(sample["goal_episode"]),
            "goal_variant": str(sample["goal_variant"]),
            "state_name": str(sample["state_name"]),
            "split_role": str(sample["split_role"]),
            "decision_frame": decision,
            "causal_prefix_sha256": prefix_audit["causal_prefix_sha256"],
            "goal_sha256": str(sample["goal"]["content_sha256"]),  # type: ignore[index]
            "patch_score_present": scores is not None,
            "arms": {
                TEACHER_ARM: teacher,
                DEPLOYMENT_ARM: deployment,
            },
        })
    if not records:
        raise CandidateBuildError("scene selection produced no causal samples")
    if len({record["sample_id"] for record in records}) != len(records):
        raise CandidateBuildError("output sample ids are not unique")

    arm_summary = {}
    for arm in ARMS:
        valid = [
            row["arms"][arm] for row in records
            if row["arms"][arm]["proposal"]["valid"]
        ]
        labeled = [
            row for row in valid
            if row["proposal_proxy"]["label_valid"]
        ]
        arm_summary[arm] = {
            "valid_sample_count": len(valid),
            "invalid_sample_count": len(records) - len(valid),
            "samples_with_candidates": sum(
                int(row["proposal"]["shortlist_count"] > 0) for row in valid),
            "mean_shortlist_count": (
                float(np.mean([
                    row["proposal"]["shortlist_count"] for row in valid
                ])) if valid else None),
            "proxy_labeled_sample_count": len(labeled),
            "proxy_universe_positive_count": sum(
                int(row["proposal_proxy"]["universe_has_positive"])
                for row in labeled),
            "proxy_shortlist_positive_count": sum(
                int(row["proposal_proxy"]["shortlist_has_positive"])
                for row in labeled),
            "proxy_claim_status": (
                "audit_only_labels_present" if labeled
                else "not_measured_do_not_claim_coverage"),
        }
    split = manifest.get("split")
    if not isinstance(split, Mapping) or not isinstance(split.get("sha256"), str):
        raise CandidateBuildError("causal manifest split provenance is malformed")
    artifact = {
        "schema_version": (
            EXTERNAL_SCALE_ARTIFACT_SCHEMA_VERSION
            if external_scale_mode else ARTIFACT_SCHEMA_VERSION),
        "purpose": (
            "NLSR-V2 J0-P causal proposal and deployment shortlist audit; "
            "not a utility-label candidate-set artifact"
        ),
        "provenance": {
            "input_manifest_path": str(manifest_path.resolve()),
            "input_manifest_sha256": manifest_sha256,
            "input_manifest_schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
            "split_sha256": split["sha256"],
            "proposal_schema_version": PROPOSAL_SCHEMA_VERSION,
            "generator_source_sha256": generator_sha,
            "generator_source_files": source_hashes,
            "configuration_sha256": sha256_bytes(canonical_json_bytes(configuration)),
            "patch_scores": patch_provenance,
            "expected_ground_prefix_builder_sha256": (
                expected_ground_prefix_builder_sha256),
            "expected_ground_prefix_configuration_sha256": (
                expected_ground_prefix_configuration_sha256),
            "causal_ground_scale": external_scale_provenance,
            "privileged_feature_policy": (
                "GT/pathfinder outputs only in arms.*.proposal_proxy; never in "
                "candidate_universe, shortlist, or NMS"
            ),
            "deployment_pose_policy": (
                "header-validated versioned cache pair + dense causal "
                "cam_pose_enc + "
                + ("externally pinned first-prefix ground scale"
                   if external_scale_mode
                   else "legacy dense causal ground_h_est_prefix")
                + " in raw LingBot x-z; whole-episode ground_h_est is "
                "rejected; no GT Sim2 or teacher fallback"
            ),
        },
        "configuration": configuration,
        "selection": {
            "scenes": sorted(allowed_scenes),
            "sample_count": len(records),
            "arms": list(ARMS),
        },
        "records": records,
        "summary": {
            "scene_count": len({row["scene"] for row in records}),
            "sample_count": len(records),
            "arms": arm_summary,
            "proposal_proxy_labeler_supplied": labeler is not None,
            "real_pathfinder_claim": False,
            "interpretation": (
                "candidate coverage may be reported only where proposal_proxy "
                "label_valid=true; teacher and deployment pose arms are distinct"
            ),
        },
    }
    canonical_json_bytes(artifact)
    return artifact


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_artifact(
    artifact: Mapping[str, object],
    output: Path,
    sha_output: Path,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[str, str]:
    if resume and overwrite:
        raise CandidateBuildError("--resume and --overwrite are mutually exclusive")
    if output.resolve() == sha_output.resolve():
        raise CandidateBuildError("JSON and SHA outputs must be distinct")
    payload = canonical_json_bytes(artifact)
    digest = sha256_bytes(payload)
    sidecar = f"{digest}  {output.name}\n".encode("ascii")
    existence = output.exists(), sha_output.exists()
    if resume:
        if existence != (True, True):
            raise CandidateBuildError(
                "resume requires an existing JSON and SHA sidecar")
        if output.read_bytes() != payload or sha_output.read_bytes() != sidecar:
            raise CandidateBuildError(
                "resume artifact differs from the deterministic rebuilt result")
        return "resumed", digest
    if any(existence) and not overwrite:
        raise CandidateBuildError(
            "output already exists; use --resume or explicit --overwrite")
    _atomic_write(output, payload)
    _atomic_write(sha_output, sidecar)
    return "written", digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-manifest-sha", required=True,
        help="required pin for the causal input manifest")
    parser.add_argument(
        "--patch-scores", type=Path,
        help="optional deployment-only per-frame goal-patch score artifact")
    parser.add_argument("--expected-patch-encoder-checkpoint-sha")
    parser.add_argument("--expected-patch-feature-builder-sha")
    parser.add_argument("--expected-patch-configuration-sha")
    parser.add_argument("--expected-ground-prefix-builder-sha")
    parser.add_argument("--expected-ground-prefix-configuration-sha")
    parser.add_argument("--causal-ground-scale", type=Path)
    parser.add_argument("--expected-causal-ground-scale-sha")
    parser.add_argument("--expected-ground-scale-producer-sha")
    parser.add_argument("--expected-ground-scale-configuration-sha")
    parser.add_argument("--expected-ground-scale-lingbot-commit")
    parser.add_argument("--expected-ground-scale-weights-sha")
    parser.add_argument("--expected-ground-scale-stream-source-sha")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--depth-unit-m", type=float, default=1e-4)
    parser.add_argument("--depth-column-stride", type=int, default=8)
    parser.add_argument("--scan-stride", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha-out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise CandidateBuildError(f"causal manifest is missing: {args.manifest}")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateBuildError("causal manifest is invalid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise CandidateBuildError("causal manifest must be an object")
    artifact = build_artifact(
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.expected_manifest_sha,
        patch_score_path=args.patch_scores,
        expected_patch_encoder_checkpoint_sha256=(
            args.expected_patch_encoder_checkpoint_sha),
        expected_patch_feature_builder_sha256=(
            args.expected_patch_feature_builder_sha),
        expected_patch_configuration_sha256=(
            args.expected_patch_configuration_sha),
        expected_ground_prefix_builder_sha256=(
            args.expected_ground_prefix_builder_sha),
        expected_ground_prefix_configuration_sha256=(
            args.expected_ground_prefix_configuration_sha),
        causal_ground_scale_path=args.causal_ground_scale,
        expected_causal_ground_scale_sha256=(
            args.expected_causal_ground_scale_sha),
        expected_ground_scale_producer_sha256=(
            args.expected_ground_scale_producer_sha),
        expected_ground_scale_configuration_sha256=(
            args.expected_ground_scale_configuration_sha),
        expected_ground_scale_lingbot_commit=(
            args.expected_ground_scale_lingbot_commit),
        expected_ground_scale_weights_sha256=(
            args.expected_ground_scale_weights_sha),
        expected_ground_scale_stream_source_sha256=(
            args.expected_ground_scale_stream_source_sha),
        selected_scenes=tuple(args.scene),
        depth_unit_m=args.depth_unit_m,
        depth_column_stride=args.depth_column_stride,
        scan_stride=args.scan_stride,
    )
    sha_output = args.sha_out or Path(f"{args.out}.sha256")
    status, digest = write_artifact(
        artifact,
        args.out,
        sha_output,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": status,
        "output": str(args.out),
        "sha_output": str(sha_output),
        "sha256": digest,
        **artifact["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
