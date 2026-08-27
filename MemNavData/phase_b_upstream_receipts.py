#!/usr/bin/env python3
"""Fail-closed binding for the two audited upstream Phase-B artifacts.

The LingBot feature collector consumes a causal teacher CSV and a per-episode
metric-scale artifact.  Hashing those two payloads is necessary but not
sufficient: the formal run also relies on the independent receipts which prove
that the teacher obeys the manifest boundary and that the scale artifact was
rebound to physical camera/RGB prefixes.  This module validates all four files
as one immutable unit and returns a deterministic summary suitable for a
collector checkpoint signature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TEACHER_AUDIT_SCHEMA = "manifest_native_causal_covisibility_teacher_audit_v1"
TEACHER_AUDIT_STATUS = "audited_not_deployment_approved"
SCALE_ACCEPTANCE_SCHEMA = "nlsr_causal_ground_scale_acceptance_v1"
SCALE_ACCEPTANCE_STATUS = "causal_prefixes_physically_rebound"
RECEIPT_BINDING_SCHEMA = "nlsr_phase_b_upstream_receipt_binding_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class PhaseBUpstreamReceiptError(RuntimeError):
    """One upstream payload or receipt is absent, changed, or inconsistent."""


@dataclass(frozen=True)
class PhaseBUpstreamPins:
    teacher_csv_sha256: str
    teacher_audit_sha256: str
    manifest_sha256: str
    scale_artifact_sha256: str
    scale_acceptance_sha256: str
    scale_acceptance_commit: str
    scale_producer_sha256: str
    scale_configuration_sha256: str
    scale_lingbot_commit: str
    scale_weights_sha256: str
    scale_stream_source_sha256: str

    def validate(self) -> None:
        sha_fields = {
            name: value
            for name, value in self.__dict__.items()
            if name.endswith("_sha256")
        }
        for name, value in sha_fields.items():
            _require(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None,
                f"invalid {name}",
            )
        for name in ("scale_acceptance_commit", "scale_lingbot_commit"):
            value = getattr(self, name)
            _require(
                isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None,
                f"invalid {name}",
            )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseBUpstreamReceiptError(message)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(path: Path) -> str:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"not a physical file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()

    def identity(stat):
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    _require(identity(before) == identity(after), f"file changed while hashing: {path}")
    return digest


def _load_pinned_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = Path(path)
    actual = _stable_sha256(path)
    _require(actual == expected_sha256, f"receipt SHA mismatch: {path}")
    sidecar = Path(f"{path}.sha256")
    _require(
        sidecar.is_file() and not sidecar.is_symlink(),
        f"missing physical receipt sidecar: {sidecar}",
    )
    expected_sidecar = f"{actual}  {path.name}\n"
    _require(
        sidecar.read_text(encoding="ascii") == expected_sidecar,
        f"receipt sidecar mismatch: {sidecar}",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBUpstreamReceiptError(f"invalid receipt JSON: {path}") from error
    _require(isinstance(value, dict), f"receipt is not a JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_phase_b_upstream_receipts(
    *,
    teacher_csv_path: Path,
    teacher_audit_path: Path,
    manifest_path: Path,
    scale_artifact_path: Path,
    scale_acceptance_path: Path,
    pins: PhaseBUpstreamPins,
    expected_samples: int = 600,
    expected_candidates: int = 17_845,
    expected_scenes: int = 50,
    expected_episodes: int = 100,
) -> dict[str, Any]:
    """Validate and summarize the formal teacher/scale receipt bundle."""

    pins.validate()
    _require(expected_samples > 0, "expected_samples must be positive")
    _require(expected_candidates > 0, "expected_candidates must be positive")
    _require(expected_scenes > 0, "expected_scenes must be positive")
    _require(expected_episodes > 0, "expected_episodes must be positive")

    teacher_csv_path = Path(teacher_csv_path).resolve()
    teacher_audit_path = Path(teacher_audit_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    scale_artifact_path = Path(scale_artifact_path).resolve()
    scale_acceptance_path = Path(scale_acceptance_path).resolve()

    physical_hashes = {
        "teacher_csv": _stable_sha256(teacher_csv_path),
        "manifest": _stable_sha256(manifest_path),
        "scale_artifact": _stable_sha256(scale_artifact_path),
    }
    _require(
        physical_hashes["teacher_csv"] == pins.teacher_csv_sha256,
        "teacher CSV changed after pinning",
    )
    _require(
        physical_hashes["manifest"] == pins.manifest_sha256,
        "causal manifest changed after pinning",
    )
    _require(
        physical_hashes["scale_artifact"] == pins.scale_artifact_sha256,
        "causal scale artifact changed after pinning",
    )

    teacher = _load_pinned_receipt(teacher_audit_path, pins.teacher_audit_sha256)
    _require(
        teacher.get("schema_version") == TEACHER_AUDIT_SCHEMA,
        "unexpected teacher audit schema",
    )
    _require(
        teacher.get("status") == TEACHER_AUDIT_STATUS,
        "teacher audit status is not the formal audited state",
    )
    _require(
        teacher.get("deployment_approved") is False,
        "teacher label artifact must not claim deployment approval",
    )
    teacher_csv = _mapping(teacher.get("csv"), "teacher audit csv")
    teacher_manifest = _mapping(teacher.get("manifest"), "teacher audit manifest")
    counts = _mapping(teacher.get("counts"), "teacher audit counts")
    invariants = _mapping(teacher.get("invariants"), "teacher audit invariants")
    phase_b = _mapping(teacher_csv.get("phase_b_contract"), "teacher Phase-B contract")
    _require(
        teacher_csv.get("content_sha256") == pins.teacher_csv_sha256,
        "teacher audit is bound to a different CSV",
    )
    _require(
        teacher_manifest.get("content_sha256") == pins.manifest_sha256,
        "teacher audit is bound to a different manifest",
    )
    _require(
        counts.get("samples") == expected_samples
        and teacher_manifest.get("sample_count") == expected_samples,
        "teacher audit sample coverage mismatch",
    )
    _require(
        counts.get("candidates") == expected_candidates
        and teacher_csv.get("candidate_rows") == expected_candidates,
        "teacher audit candidate coverage mismatch",
    )
    required_invariants = (
        "all_candidate_frames_strictly_before_decision",
        "exact_manifest_sample_cover",
        "phase_b_csv_exact_candidate_cover",
        "phase_b_csv_exact_manifest_session_cover",
    )
    _require(
        all(invariants.get(name) is True for name in required_invariants),
        "teacher audit does not prove the causal Phase-B boundary",
    )
    _require(
        phase_b.get("exact_candidate_cover") is True
        and phase_b.get("session_id") == "sample_id"
        and phase_b.get("causal_manifest_sample_id") == "sample_id"
        and phase_b.get("episode") == "source_episode",
        "teacher CSV does not expose the exact Phase-B join contract",
    )
    _require(
        phase_b.get("allowed_kinds")
        == [
            "manifest_causal_goal_localization_train",
            "manifest_causal_goal_localization_development",
        ],
        "teacher Phase-B split kinds changed",
    )

    acceptance = _load_pinned_receipt(
        scale_acceptance_path, pins.scale_acceptance_sha256
    )
    _require(
        acceptance.get("schema_version") == SCALE_ACCEPTANCE_SCHEMA,
        "unexpected scale acceptance schema",
    )
    _require(
        acceptance.get("status") == SCALE_ACCEPTANCE_STATUS,
        "scale artifact has not passed physical prefix acceptance",
    )
    _require(
        acceptance.get("acceptance_commit") == pins.scale_acceptance_commit,
        "scale acceptance commit mismatch",
    )
    acceptance_inputs = _mapping(acceptance.get("inputs"), "scale acceptance inputs")
    coverage = _mapping(acceptance.get("coverage"), "scale acceptance coverage")
    rebinding = _mapping(
        acceptance.get("physical_rebinding"), "scale physical rebinding"
    )
    producer = _mapping(acceptance.get("producer"), "scale acceptance producer")
    _require(
        acceptance_inputs.get("manifest_sha256") == pins.manifest_sha256
        and acceptance_inputs.get("scale_artifact_sha256")
        == pins.scale_artifact_sha256,
        "scale acceptance is bound to different inputs",
    )
    _require(
        acceptance_inputs.get("manifest_path") == str(manifest_path)
        and acceptance_inputs.get("scale_artifact_path") == str(scale_artifact_path),
        "scale acceptance physical paths differ from collector inputs",
    )
    _require(
        coverage.get("scene_count") == expected_scenes
        and coverage.get("episode_count") == expected_episodes
        and coverage.get("sample_count") == expected_samples
        and coverage.get("future_frames_consumed") == 0
        and coverage.get("all_episode_estimates_valid") is True,
        "scale acceptance coverage is incomplete or non-causal",
    )
    _require(
        rebinding.get("independent_prefix_validation_passes") == 2
        and rebinding.get("routed_cache_pairs_reopened") == expected_episodes
        and rebinding.get("camera_pose_prefix_hash_checks") == 2 * expected_episodes
        and rebinding.get("rgb_prefix_hash_checks") == 2 * expected_episodes,
        "scale acceptance did not independently rebind every physical prefix",
    )
    expected_producer = {
        "source_bundle_sha256": pins.scale_producer_sha256,
        "configuration_sha256": pins.scale_configuration_sha256,
        "lingbot_commit": pins.scale_lingbot_commit,
        "weights_sha256": pins.scale_weights_sha256,
        "stream_source_sha256": pins.scale_stream_source_sha256,
    }
    _require(
        all(producer.get(name) == value for name, value in expected_producer.items()),
        "scale acceptance producer identity mismatch",
    )

    return {
        "schema_version": RECEIPT_BINDING_SCHEMA,
        "teacher": {
            "csv_path": str(teacher_csv_path),
            "csv_sha256": pins.teacher_csv_sha256,
            "audit_path": str(teacher_audit_path),
            "audit_sha256": pins.teacher_audit_sha256,
            "audit_schema_version": TEACHER_AUDIT_SCHEMA,
            "audit_status": TEACHER_AUDIT_STATUS,
            "samples": expected_samples,
            "candidates": expected_candidates,
            "producer_content_sha256": teacher.get("producer_content_sha256"),
            "configuration_sha256": teacher.get("configuration_sha256"),
            "runtime_identity_sha256": teacher.get("runtime_identity_sha256"),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": pins.manifest_sha256,
        },
        "scale": {
            "artifact_path": str(scale_artifact_path),
            "artifact_sha256": pins.scale_artifact_sha256,
            "acceptance_path": str(scale_acceptance_path),
            "acceptance_sha256": pins.scale_acceptance_sha256,
            "acceptance_commit": pins.scale_acceptance_commit,
            "acceptance_schema_version": SCALE_ACCEPTANCE_SCHEMA,
            "acceptance_status": SCALE_ACCEPTANCE_STATUS,
            "scene_count": expected_scenes,
            "episode_count": expected_episodes,
            "sample_count": expected_samples,
            "future_frames_consumed": 0,
            **expected_producer,
        },
    }


__all__ = [
    "PhaseBUpstreamPins",
    "PhaseBUpstreamReceiptError",
    "RECEIPT_BINDING_SCHEMA",
    "validate_phase_b_upstream_receipts",
]
