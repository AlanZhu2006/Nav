#!/usr/bin/env python3
"""Build a manifest-native causal co-visibility teacher for 3-leg episodes.

The source of truth is one content-pinned multistage expert manifest.  For
every sample, the candidate universe is exactly the source episode RGB frames
in ``[0, decision_frame)``.  Exact LingBot DINO CLS similarity may rank and
temporally suppress that universe, but it never supplies a label.  Labels are
either the pinned same-episode metadata ``covis_curve`` or occlusion-aware
depth reprojection from a goal view rendered at the pinned goal-source pose.

Counterfactual samples never consume episode-local ``covis_curve`` values.
They render the goal-source episode pose in the same pinned scene GLB and
compare those goal surface points against pinned source-prefix depth/poses.
This makes the teacher valid for Goal B/C and factual/counterfactual samples
without admitting future source observations.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

try:
    from MemNavData.covisibility_teacher import (
        backproject_world,
        covisibility_label,
        projected_covisibility,
    )
    from MemNavData.build_novel_candidate_manifest import (
        PARQUET_PREFIX_COLUMNS,
    )
except ImportError:  # pragma: no cover - direct script execution
    from covisibility_teacher import (  # type: ignore
        backproject_world,
        covisibility_label,
        projected_covisibility,
    )
    from build_novel_candidate_manifest import (  # type: ignore
        PARQUET_PREFIX_COLUMNS,
    )


SCHEMA_VERSION = "manifest_native_causal_covisibility_teacher_v1"
AUDIT_SCHEMA = "manifest_native_causal_covisibility_teacher_audit_v1"
EMBEDDING_BUNDLE_SCHEMA = "manifest_causal_dino_embedding_bundle_v1"
EMBEDDING_RECEIPT_NAME = "embedding_receipt.json"
PHASE_B_TEACHER_KIND = "manifest_causal_goal_localization"
MANIFEST_SCHEMA = "nlsr_v2_multistage_expert_candidate_manifest_v1"
DINO_IDENTITY_SCHEMA = "manifest_causal_dino_provider_v1"
RENDERER_IDENTITY_SCHEMA = "manifest_causal_goal_depth_renderer_v1"
PROGRESS_SCHEMA = "manifest_causal_covisibility_teacher_progress_v1"
EMBEDDING_SHARD_SCHEMA = "manifest_causal_dino_embedding_shard_v1"
SAMPLE_SHARD_SCHEMA = "manifest_causal_teacher_sample_shard_v1"
ARTIFACT_NAME = "teacher.json"
CSV_NAME = "teacher.csv"
AUDIT_NAME = "audit.json"
FORMAL_MANIFEST_SHA256 = (
    "bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
)
M_W = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
FILE_RECORD_KEYS = frozenset({"path", "path_sha256", "bytes", "content_sha256"})
SAMPLE_KEYS = frozenset(
    {
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
    }
)
PREFIX_KEYS = frozenset(
    {
        "exclusive_end_frame",
        "frame_count",
        "modalities",
        "parquet_columns",
        "parquet_row_count",
        "parquet_rows_sha256",
        "causal_prefix_sha256",
    }
)
CSV_FIELDS = (
    "session_id",
    "sample_id",
    "causal_manifest_sample_id",
    "split_role",
    "scene",
    "episode",
    "source_episode",
    "goal_episode",
    "kind",
    "goal_role",
    "goal_variant",
    "decision_frame",
    "query_path",
    "query_relative_path",
    "candidate_rank",
    "candidate_frame",
    "candidate_path",
    "candidate_relative_path",
    "dino_cosine",
    "label_source",
    "teacher_covis",
    "covisibility",
    "label",
    "query_content_sha256",
    "candidate_rgb_content_sha256",
    "candidate_depth_content_sha256",
    "manifest_sha256",
    "runtime_identity_sha256",
    "no_future",
)
BUNDLE_FILES = frozenset(
    {
        ARTIFACT_NAME,
        f"{ARTIFACT_NAME}.sha256",
        CSV_NAME,
        f"{CSV_NAME}.sha256",
        AUDIT_NAME,
        f"{AUDIT_NAME}.sha256",
    }
)
EMBEDDING_BUNDLE_FILES = frozenset(
    {
        EMBEDDING_RECEIPT_NAME,
        f"{EMBEDDING_RECEIPT_NAME}.sha256",
        "shards",
    }
)
EMBEDDING_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "manifest",
        "dino_provider",
        "embedding_invocation",
        "producer",
        "run_signature_sha256",
        "input_records",
        "input_sequence_sha256",
        "embedding_dimension",
        "shards",
        "exact_cover",
        "deployment_approved",
    }
)


class CausalTeacherError(RuntimeError):
    """A manifest, content, causality, geometry, or output check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalTeacherError(message)


def _strict_zip(*sequences: Sequence[Any]) -> zip:
    """Python-3.9-compatible strict zip for the pinned Habitat runtime."""

    lengths = [len(sequence) for sequence in sequences]
    _require(
        bool(lengths) and len(set(lengths)) == 1,
        f"strict sequence lengths differ: {lengths}",
    )
    return zip(*sequences)


def _phase_b_kind(split_role: object) -> str:
    _require(
        split_role in ("train", "development"),
        "Phase-B CSV split role is invalid",
    )
    return f"{PHASE_B_TEACHER_KIND}_{split_role}"


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
        raise CausalTeacherError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_runtime_record(path: Path | str, label: str) -> dict[str, Any]:
    raw = Path(path)
    resolved = raw.resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{label} runtime file is unavailable",
    )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "content_sha256": sha256_file(resolved),
    }


def _verify_static_runtime_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    for name, record in records.items():
        path = Path(str(record.get("path", "")))
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == record.get("bytes")
            and sha256_file(path) == record.get("content_sha256"),
            f"{label} dependency changed: {name}",
        )


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CausalTeacherError(
            f"cannot verify clean LingBot git provenance: {arguments}"
        ) from error
    return completed.stdout.strip()


def _valid_git_oid(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase Git object id",
    )
    return value


@contextmanager
def _exclusive_stage_writer(progress_directory: Path | str):
    progress = Path(progress_directory)
    progress.parent.mkdir(parents=True, exist_ok=True)
    lock_path = progress.parent / f".{progress.name}.writer.lock"
    _require(not lock_path.is_symlink(), "bundle writer lock must not be a symlink")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CausalTeacherError(
                f"another bundle writer owns {lock_path}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _valid_sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def _duplicate_safe_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, label: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CausalTeacherError(f"{label} contains non-finite {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CausalTeacherError(f"{label} is invalid JSON") from error


def load_pinned_manifest(path: Path | str, expected_sha256: str) -> Mapping[str, Any]:
    source = Path(path)
    expected = _valid_sha(expected_sha256, "expected manifest SHA")
    _require(
        source.is_file() and not source.is_symlink(), f"manifest is absent: {source}"
    )
    raw = source.read_bytes()
    _require(sha256_bytes(raw) == expected, "manifest SHA mismatch")
    value = _decode_json(raw, "manifest")
    _require(isinstance(value, Mapping), "manifest must be an object")
    _require(raw == canonical_json_bytes(value), "manifest is not canonical")
    return value


def make_provider_identity(
    schema_version: str,
    kind: str,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(kind, str) and bool(kind), "provider kind is malformed")
    payload = {
        "schema_version": schema_version,
        "kind": kind,
        "components": copy.deepcopy(dict(components)),
    }
    canonical_json_bytes(payload)
    return {**payload, "identity_sha256": sha256_bytes(canonical_json_bytes(payload))}


def _validate_provider_identity(
    value: object,
    schema_version: str,
    label: str,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"schema_version", "kind", "components", "identity_sha256"},
        f"{label} identity fields changed",
    )
    _require(value["schema_version"] == schema_version, f"{label} schema changed")
    _require(
        isinstance(value["kind"], str) and bool(value["kind"]),
        f"{label} kind is invalid",
    )
    _require(
        isinstance(value["components"], Mapping), f"{label} components are invalid"
    )
    payload = {
        "schema_version": value["schema_version"],
        "kind": value["kind"],
        "components": value["components"],
    }
    _require(
        _valid_sha(value["identity_sha256"], f"{label} identity SHA")
        == sha256_bytes(canonical_json_bytes(payload)),
        f"{label} identity fingerprint mismatch",
    )
    return copy.deepcopy(dict(value))


class DINOEmbeddingProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def embed(self, paths: Sequence[Path]) -> np.ndarray: ...


class GoalDepthRenderer(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def render_depth(
        self,
        *,
        scene_id: str,
        glb_path: Path,
        expected_glb_sha256: str,
        camera_position_habitat: np.ndarray,
        yaw_habitat: float,
        height: int,
        width: int,
        intrinsic: np.ndarray,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class ExactLingBotDINOProvider:
    """Production adapter for the exact LingBot DINOv2-L CLS encoder.

    The checkpoint and every repository source file that defines the encoder
    path are content-pinned before the provider identity is constructed.  The
    adapter intentionally has no classification or geometry API: its only
    authority is returning embeddings for deterministic shortlist ranking.
    """

    def __init__(
        self,
        *,
        lingbot_repo: Path | str,
        weights: Path | str,
        entrypoint_path: Path | str,
        expected_entrypoint_sha256: str,
        expected_python_sha256: str,
        expected_weights_sha256: str,
        expected_source_sha256: Mapping[str, str],
        expected_lingbot_commit: str,
        expected_lingbot_tree: str,
        device: str,
        batch_size: int,
    ) -> None:
        _require(isinstance(device, str) and bool(device), "DINO device is invalid")
        _require(
            isinstance(batch_size, int)
            and not isinstance(batch_size, bool)
            and batch_size >= 1,
            "DINO batch size is invalid",
        )
        self.lingbot_repo = Path(lingbot_repo).resolve()
        self.weights = Path(weights).resolve()
        entrypoint = Path(entrypoint_path)
        _require(
            self.lingbot_repo.is_dir() and not self.lingbot_repo.is_symlink(),
            f"LingBot repository is unavailable: {self.lingbot_repo}",
        )
        _require(
            self.weights.is_file() and not self.weights.is_symlink(),
            f"LingBot checkpoint is unavailable: {self.weights}",
        )
        _require(
            entrypoint.is_file() and not entrypoint.is_symlink(),
            f"Stage-A entrypoint is unavailable: {entrypoint}",
        )
        entrypoint = entrypoint.resolve()
        expected_entrypoint_sha = _valid_sha(
            expected_entrypoint_sha256, "Stage-A entrypoint SHA"
        )
        _require(
            sha256_file(entrypoint) == expected_entrypoint_sha,
            "Stage-A entrypoint differs from external pin",
        )
        self._entrypoint_record = _physical_runtime_record(
            entrypoint, "Stage-A entrypoint"
        )
        self._entrypoint_record["externally_expected_sha256"] = expected_entrypoint_sha
        expected = _valid_sha(expected_weights_sha256, "DINO checkpoint SHA")
        _require(
            sha256_file(self.weights) == expected, "DINO checkpoint content changed"
        )
        self.device = device
        self.batch_size = batch_size
        self._expected_weights_sha256 = expected
        weight_stat = self.weights.stat()
        self._weight_stat_snapshot = (
            weight_stat.st_size,
            weight_stat.st_mtime_ns,
            weight_stat.st_ino,
        )
        self._expected_lingbot_commit = _valid_git_oid(
            expected_lingbot_commit, "expected LingBot commit"
        )
        self._expected_lingbot_tree = _valid_git_oid(
            expected_lingbot_tree, "expected LingBot tree"
        )
        self._verify_git_state()

        implementation = Path(__file__).with_name("diag_distill_geometry_router.py")
        sources = {
            "exact_loader": implementation,
            "vision_transformer": self.lingbot_repo
            / "lingbot_map/layers/vision_transformer.py",
            "preprocessor": self.lingbot_repo / "lingbot_map/utils/load_fn.py",
        }
        source_pins: dict[str, dict[str, object]] = {}
        _require(
            isinstance(expected_source_sha256, Mapping)
            and set(expected_source_sha256) == set(sources),
            "exact DINO source SHA pins are incomplete",
        )
        self._source_paths = sources
        for name, source in sorted(sources.items()):
            _require(
                source.is_file() and not source.is_symlink(),
                f"DINO source is unavailable: {source}",
            )
            expected_source_sha = _valid_sha(
                expected_source_sha256[name], f"DINO {name} source SHA"
            )
            _require(
                sha256_file(source) == expected_source_sha,
                f"DINO {name} source content changed",
            )
            source_pins[name] = {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "content_sha256": expected_source_sha,
            }
        try:
            import numpy.core._multiarray_umath as numpy_multiarray

            if __package__:
                from MemNavData.diag_distill_geometry_router import FEATURE_VERSION
            else:  # pragma: no cover - direct production execution
                from diag_distill_geometry_router import FEATURE_VERSION  # type: ignore
        except ImportError as error:  # pragma: no cover - dependency preflight
            raise CausalTeacherError(
                "exact DINO implementation is unavailable"
            ) from error
        try:
            import torch
            import torch._C
            import PIL
            import PIL._imaging
            import PIL.Image
            import PIL.ImageOps
            import torchvision
            import torchvision.transforms
            import torchvision.transforms.functional
        except ImportError as error:  # pragma: no cover - production preflight
            raise CausalTeacherError(
                "torch, torchvision, and Pillow are required for DINO"
            ) from error
        runtime_paths = {
            "python_executable": sys.executable,
            "numpy": np.__file__,
            "numpy_multiarray": numpy_multiarray.__file__,
            "torch": torch.__file__,
            "torch_native": torch._C.__file__,
            "torchvision": torchvision.__file__,
            "torchvision_transforms": torchvision.transforms.__file__,
            "torchvision_transforms_functional": (
                torchvision.transforms.functional.__file__
            ),
            "pillow": PIL.__file__,
            "pillow_image": PIL.Image.__file__,
            "pillow_image_ops": PIL.ImageOps.__file__,
            "pillow_native": PIL._imaging.__file__,
        }
        _require(
            all(
                isinstance(path, str) and bool(path) for path in runtime_paths.values()
            ),
            "DINO runtime dependency provenance is incomplete",
        )
        self._runtime_records = {
            name: _physical_runtime_record(str(path), f"DINO {name}")
            for name, path in sorted(runtime_paths.items())
        }
        expected_python_sha = _valid_sha(
            expected_python_sha256, "DINO Python executable SHA"
        )
        _require(
            self._runtime_records["python_executable"]["content_sha256"]
            == expected_python_sha,
            "DINO Python executable differs from external pin",
        )
        self._runtime_records["python_executable"]["externally_expected_sha256"] = (
            expected_python_sha
        )
        torch_device = torch.device(device)
        accelerator: dict[str, object] = {"type": torch_device.type}
        if torch_device.type == "cuda":
            _require(torch.cuda.is_available(), "requested DINO CUDA is unavailable")
            device_index = (
                torch.cuda.current_device()
                if torch_device.index is None
                else torch_device.index
            )
            properties = torch.cuda.get_device_properties(device_index)
            accelerator.update(
                {
                    "index": device_index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
        self._identity = make_provider_identity(
            DINO_IDENTITY_SCHEMA,
            "exact_lingbot_dinov2l_cls",
            {
                "feature_version": FEATURE_VERSION,
                "checkpoint": {
                    "path": str(self.weights),
                    "bytes": self.weights.stat().st_size,
                    "content_sha256": expected,
                },
                "sources": source_pins,
                "entrypoint": self._entrypoint_record,
                "lingbot_git": {
                    "commit": self._expected_lingbot_commit,
                    "tree": self._expected_lingbot_tree,
                    "worktree": "tracked_and_untracked_clean",
                    "authority": "clean_commit_pins_transitive_lingbot_import_closure",
                },
                "device": device,
                "batch_size": batch_size,
                "runtime": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pillow": PIL.__version__,
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cudnn": (
                        None
                        if torch.backends.cudnn.version() is None
                        else int(torch.backends.cudnn.version())
                    ),
                    "accelerator": accelerator,
                    "physical_files": self._runtime_records,
                },
                "output_dtype": "float32",
                "authority": "candidate_ranking_only",
            },
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def _verify_import_origins(self, *, require_loaded: bool) -> None:
        expected = {
            "lingbot_map.layers.vision_transformer": self.lingbot_repo
            / "lingbot_map/layers/vision_transformer.py",
            "lingbot_map.utils.load_fn": self.lingbot_repo
            / "lingbot_map/utils/load_fn.py",
        }
        for module_name, expected_path in expected.items():
            module = sys.modules.get(module_name)
            if module is None:
                _require(
                    not require_loaded,
                    f"exact DINO did not import pinned module {module_name}",
                )
                continue
            actual = getattr(module, "__file__", None)
            _require(
                isinstance(actual, str)
                and Path(actual).resolve() == expected_path.resolve(),
                f"exact DINO module origin drifted: {module_name}",
            )

    def _verify_git_state(self, *, check_worktree: bool = True) -> None:
        _require(
            _git_output(self.lingbot_repo, "rev-parse", "HEAD")
            == self._expected_lingbot_commit,
            "LingBot commit changed",
        )
        _require(
            _git_output(self.lingbot_repo, "rev-parse", "HEAD^{tree}")
            == self._expected_lingbot_tree,
            "LingBot tree changed",
        )
        if check_worktree:
            _require(
                _git_output(
                    self.lingbot_repo,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                )
                == "",
                "LingBot worktree is not clean",
            )

    def _verify_static_content(self, *, full_hash: bool = True) -> None:
        self._verify_git_state(check_worktree=full_hash)
        weight_stat = self.weights.stat()
        _require(
            (
                weight_stat.st_size,
                weight_stat.st_mtime_ns,
                weight_stat.st_ino,
            )
            == self._weight_stat_snapshot
            and (
                not full_hash
                or sha256_file(self.weights) == self._expected_weights_sha256
            ),
            "DINO checkpoint changed after provider construction",
        )
        for name, path in self._source_paths.items():
            expected = self._identity["components"]["sources"][name]
            _require(
                path.stat().st_size == expected["bytes"]
                and sha256_file(path) == expected["content_sha256"],
                f"DINO {name} source changed after provider construction",
            )
        _verify_static_runtime_records(self._runtime_records, label="DINO runtime")
        _verify_static_runtime_records(
            {"entrypoint": self._entrypoint_record},
            label="DINO launcher",
        )

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        chunks = list(self.embed_chunks([paths]))
        _require(len(chunks) == 1, "exact DINO single-batch execution changed")
        return chunks[0]

    def embed_chunks(
        self, path_chunks: Sequence[Sequence[Path]]
    ) -> Sequence[np.ndarray]:
        """Yield complete chunks while keeping exactly one DINO model alive.

        The caller atomically publishes each yielded chunk before requesting
        the next one, so a wall-time cancellation loses at most one chunk.
        """

        _require(bool(path_chunks), "DINO chunk collection is empty")
        resolved_chunks = [
            [Path(path).resolve() for path in chunk] for chunk in path_chunks
        ]
        _require(
            all(chunk for chunk in resolved_chunks)
            and all(
                path.is_file() and not path.is_symlink()
                for chunk in resolved_chunks
                for path in chunk
            ),
            "DINO input contains an unavailable physical file",
        )
        self._verify_static_content()
        self._verify_import_origins(require_loaded=False)
        repository_entry = str(self.lingbot_repo)
        sys.path[:] = [entry for entry in sys.path if entry != repository_entry]
        sys.path.insert(0, repository_entry)
        try:
            import torch
            from lingbot_map.layers.vision_transformer import vit_large
            from lingbot_map.utils.load_fn import load_and_preprocess_images
        except ImportError as error:  # pragma: no cover - dependency preflight
            raise CausalTeacherError(
                "exact DINO implementation is unavailable"
            ) from error
        self._verify_import_origins(require_loaded=True)
        raw_state = torch.load(self.weights, map_location="cpu", weights_only=False)
        if (
            isinstance(raw_state, dict)
            and "model" in raw_state
            and isinstance(raw_state["model"], dict)
        ):
            state = raw_state["model"]
        else:
            state = raw_state
        prefix = "aggregator.patch_embed."
        patch_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        _require(
            len(patch_state) == 344,
            f"expected 344 DINO tensors in LingBot checkpoint, got {len(patch_state)}",
        )
        del raw_state, state
        gc.collect()
        model = vit_large(
            img_size=518,
            patch_size=14,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
        )
        model.load_state_dict(patch_state, strict=True)
        del patch_state
        gc.collect()
        model = model.to(self.device).eval()
        torch_device = torch.device(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=torch_device).view(1, 3, 1, 1)
        standard_deviation = torch.tensor(
            [0.229, 0.224, 0.225], device=torch_device
        ).view(1, 3, 1, 1)
        use_cuda = torch_device.type == "cuda"
        try:
            for chunk_index, paths in enumerate(resolved_chunks):
                self._verify_static_content(full_hash=False)
                outputs = []
                for start in range(0, len(paths), self.batch_size):
                    batch_paths = [
                        str(path) for path in paths[start : start + self.batch_size]
                    ]
                    images = load_and_preprocess_images(
                        batch_paths,
                        mode="pad",
                        image_size=518,
                        patch_size=14,
                    )
                    images = images.to(torch_device, non_blocking=use_cuda)
                    autocast = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if use_cuda
                        else torch.autocast("cpu", enabled=False)
                    )
                    with torch.inference_mode(), autocast:
                        encoded = model.forward_features(
                            (images - mean) / standard_deviation
                        )
                    outputs.append(encoded["x_norm_clstoken"].float().cpu().numpy())
                    del images, encoded
                embeddings = np.asarray(
                    np.concatenate(outputs, axis=0), dtype=np.float32
                )
                _require(
                    embeddings.shape == (len(paths), 1024)
                    and np.isfinite(embeddings).all()
                    and np.all(np.linalg.norm(embeddings, axis=1) > 0.0),
                    f"unexpected exact DINO chunk shape/value at {chunk_index}",
                )
                self._verify_import_origins(require_loaded=True)
                self._verify_static_content(full_hash=False)
                yield embeddings
            self._verify_static_content()
        finally:
            del model
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()


class PinnedHabitatGoalDepthRenderer:
    """Render goal depth from a content-pinned Habitat-Sim runtime.

    A simulator is opened only for the currently requested scene/resolution and
    no NavMesh is loaded or rebuilt.  Rendering therefore cannot mutate or
    reinterpret the trajectory graph.  The caller remains responsible for the
    per-request GLB content pin, which is recorded in every teacher row.
    """

    def __init__(
        self,
        *,
        expected_habitat_version: str,
        bindings_file: Path | str,
        expected_bindings_sha256: str,
        entrypoint_path: Path | str,
        expected_entrypoint_sha256: str,
        expected_python_sha256: str,
    ) -> None:
        _require(
            isinstance(expected_habitat_version, str)
            and bool(expected_habitat_version),
            "expected Habitat-Sim version is invalid",
        )
        bindings = Path(bindings_file)
        entrypoint = Path(entrypoint_path)
        _require(
            bindings.is_file() and not bindings.is_symlink(),
            f"Habitat bindings file is unavailable: {bindings}",
        )
        _require(
            entrypoint.is_file() and not entrypoint.is_symlink(),
            f"Stage-B entrypoint is unavailable: {entrypoint}",
        )
        bindings = bindings.resolve()
        entrypoint = entrypoint.resolve()
        expected_entrypoint_sha = _valid_sha(
            expected_entrypoint_sha256, "Stage-B entrypoint SHA"
        )
        _require(
            sha256_file(entrypoint) == expected_entrypoint_sha,
            "Stage-B entrypoint differs from external pin",
        )
        self._entrypoint_record = _physical_runtime_record(
            entrypoint, "Stage-B entrypoint"
        )
        self._entrypoint_record["externally_expected_sha256"] = expected_entrypoint_sha
        expected_sha = _valid_sha(expected_bindings_sha256, "Habitat bindings SHA")
        _require(
            sha256_file(bindings) == expected_sha,
            "Habitat bindings content changed",
        )
        try:
            import habitat_sim
            import magnum
            import quaternion
        except ImportError as error:  # pragma: no cover - real env only
            raise CausalTeacherError(
                "habitat-sim and numpy-quaternion are required"
            ) from error
        actual_version = getattr(habitat_sim, "__version__", None)
        _require(
            actual_version == expected_habitat_version,
            "Habitat-Sim runtime version changed",
        )
        try:
            actual_bindings = Path(
                habitat_sim._ext.habitat_sim_bindings.__file__
            ).resolve()
            habitat_init = Path(habitat_sim.__file__).resolve()
        except (AttributeError, TypeError) as error:  # pragma: no cover - real env only
            raise CausalTeacherError(
                "Habitat-Sim runtime file provenance is unavailable"
            ) from error
        _require(
            actual_bindings == bindings,
            "declared Habitat bindings file is not the imported runtime",
        )
        _require(
            habitat_init.is_file() and not habitat_init.is_symlink(),
            "Habitat-Sim package source is unavailable",
        )
        self._habitat_sim = habitat_sim
        try:
            import numpy.core._multiarray_umath as numpy_multiarray
        except ImportError as error:  # pragma: no cover - real env only
            raise CausalTeacherError(
                "NumPy native runtime provenance is unavailable"
            ) from error
        runtime_paths = {
            "python_executable": sys.executable,
            "numpy": np.__file__,
            "numpy_multiarray": numpy_multiarray.__file__,
            "quaternion": quaternion.__file__,
            "magnum": magnum.__file__,
        }
        _require(
            all(
                isinstance(path, str) and bool(path) for path in runtime_paths.values()
            ),
            "Habitat runtime dependency provenance is incomplete",
        )
        self._runtime_records = {
            name: _physical_runtime_record(str(path), f"Habitat {name}")
            for name, path in sorted(runtime_paths.items())
        }
        expected_python_sha = _valid_sha(
            expected_python_sha256, "Habitat Python executable SHA"
        )
        _require(
            self._runtime_records["python_executable"]["content_sha256"]
            == expected_python_sha,
            "Habitat Python executable differs from external pin",
        )
        self._runtime_records["python_executable"]["externally_expected_sha256"] = (
            expected_python_sha
        )
        runtime_modules = {}
        for name, module in (("quaternion", quaternion), ("magnum", magnum)):
            module_path_raw = getattr(module, "__file__", None)
            _require(
                isinstance(module_path_raw, str),
                f"{name} runtime file provenance is unavailable",
            )
            module_path = Path(module_path_raw).resolve()
            _require(
                module_path.is_file() and not module_path.is_symlink(),
                f"{name} runtime file is unavailable",
            )
            runtime_modules[name] = {
                "path": str(module_path),
                "bytes": module_path.stat().st_size,
                "content_sha256": sha256_file(module_path),
                "version": getattr(module, "__version__", None),
            }
        self._identity = make_provider_identity(
            RENDERER_IDENTITY_SCHEMA,
            "pinned_habitat_goal_depth",
            {
                "habitat_sim_version": actual_version,
                "bindings": {
                    "path": str(bindings),
                    "bytes": bindings.stat().st_size,
                    "content_sha256": expected_sha,
                },
                "python_package": {
                    "path": str(habitat_init),
                    "bytes": habitat_init.stat().st_size,
                    "content_sha256": sha256_file(habitat_init),
                },
                "entrypoint": self._entrypoint_record,
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "runtime_modules": runtime_modules,
                "runtime_physical_files": self._runtime_records,
                "sensor_mount": "agent_origin_at_exact_goal_camera_position",
                "orientation": "quaternion_rotation_vector_[0,yaw_habitat,0]",
                "navmesh_usage": "none",
                "glb_content_pin_checked_on_every_simulator_open": True,
                "physics": False,
                "depth_dtype": "float32",
            },
        )
        self._simulator: object | None = None
        self._simulator_key: tuple[object, ...] | None = None
        self._glb_stat_snapshot: tuple[int, int, int] | None = None

    @property
    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def _ensure_simulator(
        self,
        *,
        glb_path: Path,
        expected_glb_sha256: str,
        height: int,
        width: int,
        intrinsic: np.ndarray,
    ) -> object:
        _verify_static_runtime_records(self._runtime_records, label="Habitat runtime")
        _verify_static_runtime_records(
            {"entrypoint": self._entrypoint_record},
            label="Habitat launcher",
        )
        _require(
            glb_path.is_file() and not glb_path.is_symlink(),
            f"scene GLB is unavailable: {glb_path}",
        )
        expected_glb_sha = _valid_sha(expected_glb_sha256, "render scene GLB SHA")
        _require(height >= 1 and width >= 1, "render resolution is invalid")
        intrinsic = _finite_matrix(intrinsic, (3, 3), "render intrinsic")
        _require(
            math.isclose(float(intrinsic[0, 2]), width / 2.0, abs_tol=1e-6)
            and math.isclose(float(intrinsic[1, 2]), height / 2.0, abs_tol=1e-6)
            and float(intrinsic[0, 0]) > 0.0
            and float(intrinsic[1, 1]) > 0.0,
            "renderer requires the pinned centered positive-focal camera model",
        )
        hfov = math.degrees(
            2.0 * math.atan(float(intrinsic[0, 2]) / float(intrinsic[0, 0]))
        )
        key = (
            str(glb_path.resolve()),
            expected_glb_sha,
            height,
            width,
            float(hfov),
        )
        stat = glb_path.stat()
        stat_snapshot = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if key == self._simulator_key:
            _require(
                self._simulator is not None
                and self._glb_stat_snapshot == stat_snapshot,
                "loaded scene GLB changed during rendering",
            )
            return self._simulator
        _require(
            sha256_file(glb_path) == expected_glb_sha,
            "scene GLB content changed before simulator open",
        )
        self.close()
        habitat_sim = self._habitat_sim
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(glb_path.resolve())
        backend.enable_physics = False
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "depth"
        sensor.sensor_type = habitat_sim.SensorType.DEPTH
        sensor.resolution = [height, width]
        sensor.hfov = hfov
        try:
            import magnum as mn
        except ImportError as error:  # pragma: no cover - real env only
            raise CausalTeacherError(
                "magnum is required for Habitat rendering"
            ) from error
        sensor.position = mn.Vector3(0, 0, 0)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        self._simulator = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [agent])
        )
        self._simulator_key = key
        self._glb_stat_snapshot = stat_snapshot
        return self._simulator

    def render_depth(
        self,
        *,
        scene_id: str,
        glb_path: Path,
        expected_glb_sha256: str,
        camera_position_habitat: np.ndarray,
        yaw_habitat: float,
        height: int,
        width: int,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        del scene_id  # scene identity is bound by the caller's GLB content pin
        simulator = self._ensure_simulator(
            glb_path=glb_path,
            expected_glb_sha256=expected_glb_sha256,
            height=height,
            width=width,
            intrinsic=intrinsic,
        )
        position = np.asarray(camera_position_habitat, dtype=np.float64)
        _require(
            position.shape == (3,)
            and np.isfinite(position).all()
            and math.isfinite(float(yaw_habitat)),
            "goal render pose is malformed",
        )
        try:
            import quaternion
        except ImportError as error:  # pragma: no cover - real env only
            raise CausalTeacherError("numpy-quaternion is required") from error
        state = self._habitat_sim.agent.AgentState()
        state.position = position
        state.rotation = quaternion.from_rotation_vector([0.0, float(yaw_habitat), 0.0])
        simulator.get_agent(0).set_state(state)
        observations = simulator.get_sensor_observations()
        _require("depth" in observations, "Habitat depth sensor output is absent")
        depth = np.asarray(observations["depth"], dtype=np.float32)
        _require(
            depth.shape == (height, width) and np.isfinite(depth).all(),
            "Habitat rendered depth is malformed",
        )
        return depth.copy()

    def close(self) -> None:
        if self._simulator is not None:
            close = getattr(self._simulator, "close", None)
            if callable(close):
                close()
        self._simulator = None
        self._simulator_key = None
        self._glb_stat_snapshot = None


@dataclass(frozen=True)
class TeacherConfig:
    top_k: int = 32
    temporal_nms_radius: int = 4
    backprojection_stride: int = 6
    depth_tolerance_m: float = 0.3
    positive_threshold: float = 0.5
    negative_threshold: float = 0.1

    def __post_init__(self) -> None:
        _require(
            isinstance(self.top_k, int)
            and not isinstance(self.top_k, bool)
            and self.top_k >= 1,
            "top_k must be a positive integer",
        )
        _require(
            isinstance(self.temporal_nms_radius, int)
            and not isinstance(self.temporal_nms_radius, bool)
            and self.temporal_nms_radius >= 0,
            "temporal NMS radius must be a non-negative integer",
        )
        _require(
            isinstance(self.backprojection_stride, int)
            and not isinstance(self.backprojection_stride, bool)
            and self.backprojection_stride >= 1,
            "backprojection stride must be positive",
        )
        _require(
            math.isfinite(self.depth_tolerance_m) and self.depth_tolerance_m > 0.0,
            "depth tolerance must be finite and positive",
        )
        _require(
            0.0 <= self.negative_threshold < self.positive_threshold <= 1.0,
            "co-visibility thresholds are invalid",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "temporal_nms_radius": self.temporal_nms_radius,
            "backprojection_stride": self.backprojection_stride,
            "depth_tolerance_m": self.depth_tolerance_m,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "candidate_universe": "source_episode_frames_[0,decision_frame)",
            "selection_authority": "exact_lingbot_dino_cls_rank_plus_temporal_nms_only",
            "label_authority": "pinned_metadata_curve_or_occlusion_aware_depth_reprojection",
        }


@dataclass(frozen=True)
class FrameAsset:
    frame: int
    rgb_path: Path
    depth_path: Path
    rgb_record: Mapping[str, Any]
    depth_record: Mapping[str, Any]
    intrinsic: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class SampleContext:
    sample: Mapping[str, Any]
    scene_record: Mapping[str, Any]
    environment_path: Path
    environment_record: Mapping[str, Any]
    query_path: Path
    query_record: Mapping[str, Any]
    goal_metadata_record: Mapping[str, Any]
    goal_metadata: Mapping[str, Any]
    goal_index: int
    goal_pose: Mapping[str, Any]
    camera_height_m: float
    goal_intrinsic: np.ndarray
    frames: tuple[FrameAsset, ...]


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = frozenset(value)
    _require(
        actual == keys,
        f"{label} fields changed: missing={sorted(keys - actual)} extra={sorted(actual - keys)}",
    )
    return value


def _finite_float(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite numeric",
    )
    return float(value)


def _finite_matrix(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTeacherError(f"{label} is not numeric") from error
    _require(
        result.shape == shape and np.isfinite(result).all(),
        f"{label} shape/value is invalid",
    )
    return result


def _rigid_transform(value: object, label: str) -> np.ndarray:
    """Validate a finite, proper SE(3) camera transform.

    A merely finite 4x4 matrix is not enough for reprojection: scale, shear,
    reflection, or a malformed homogeneous row silently changes metric
    co-visibility.  Both stored camera extrinsics and actions therefore cross
    this boundary before any geometry is allowed to consume them.
    """

    transform = _finite_matrix(value, (4, 4), label)
    _require(
        np.allclose(
            transform[3],
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
        ),
        f"{label} homogeneous row is not rigid",
    )
    rotation = transform[:3, :3]
    _require(
        np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )
        and math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
        ),
        f"{label} rotation is not a proper SO(3) matrix",
    )
    return transform


def _relative_path(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} path is invalid")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value
        and value != ".",
        f"{label} path is not normalized relative POSIX",
    )
    return value


def _verify_file_record(
    value: object,
    root: Path,
    label: str,
    *,
    expected_relative: str | None = None,
    verification_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]]
    | None = None,
) -> tuple[Path, dict[str, Any]]:
    record = _exact_mapping(value, FILE_RECORD_KEYS, label)
    relative = _relative_path(record["path"], label)
    if expected_relative is not None:
        _require(relative == expected_relative, f"{label} path changed")
    _require(
        record["path_sha256"] == sha256_bytes(relative.encode("utf-8")),
        f"{label} path SHA mismatch",
    )
    _require(
        isinstance(record["bytes"], int)
        and not isinstance(record["bytes"], bool)
        and int(record["bytes"]) > 0,
        f"{label} byte count is invalid",
    )
    content_sha = _valid_sha(record["content_sha256"], f"{label} content SHA")
    root = root.resolve()
    cache_key = (str(root), relative)
    if verification_cache is not None and cache_key in verification_cache:
        cached_path, cached_record = verification_cache[cache_key]
        _require(
            cached_record == record, f"{label} conflicts with its earlier content pin"
        )
        return cached_path, copy.deepcopy(cached_record)
    unresolved = root / relative
    _require(not unresolved.is_symlink(), f"{label} must not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CausalTeacherError(f"{label} escapes declared root") from error
    _require(
        path.is_file() and not path.is_symlink(), f"{label} is not a physical file"
    )
    _require(path.stat().st_size == record["bytes"], f"{label} byte count changed")
    _require(sha256_file(path) == content_sha, f"{label} content changed")
    verified_record = copy.deepcopy(dict(record))
    if verification_cache is not None:
        verification_cache[cache_key] = (path, verified_record)
    return path, copy.deepcopy(verified_record)


def _physical_file_record(
    path: Path,
    root: Path,
    verification_cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _require(not path.is_symlink(), f"candidate file must not be a symlink: {path}")
    resolved = path.resolve()
    if verification_cache is not None and resolved in verification_cache:
        return copy.deepcopy(verification_cache[resolved])
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise CausalTeacherError(
            f"candidate file escapes episode root: {path}"
        ) from error
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"candidate file is absent: {path}",
    )
    record = {
        "path": relative,
        "path_sha256": sha256_bytes(relative.encode("utf-8")),
        "bytes": resolved.stat().st_size,
        "content_sha256": sha256_file(resolved),
    }
    if verification_cache is not None:
        verification_cache[resolved] = record
    return copy.deepcopy(record)


def _load_metadata(path: Path, label: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    value = _decode_json(raw, label)
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _canonical_numeric(value: object, label: str) -> object:
    if isinstance(value, bool):
        raise CausalTeacherError(f"{label} contains boolean numeric data")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains non-finite data")
        return 0.0 if value == 0.0 else value
    if isinstance(value, (list, tuple)):
        return [
            _canonical_numeric(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CausalTeacherError(f"{label} contains unsupported numeric type")


def _matrix_list(value: object, shape: tuple[int, int], label: str) -> list:
    canonical = _canonical_numeric(value, label)
    _require(
        isinstance(canonical, list)
        and len(canonical) == shape[0]
        and all(isinstance(row, list) and len(row) == shape[1] for row in canonical),
        f"{label} must have shape {shape}",
    )
    return canonical


def _read_parquet_prefix(path: Path, exclusive_end: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - dependency preflight
        raise CausalTeacherError(
            "pyarrow is required for causal parquet verification"
        ) from error
    try:
        table = parquet.read_table(path, columns=list(PARQUET_PREFIX_COLUMNS))
    except Exception as error:
        raise CausalTeacherError(
            f"cannot read causal parquet columns: {path}"
        ) from error
    _require(
        tuple(table.column_names) == tuple(PARQUET_PREFIX_COLUMNS)
        and table.num_rows >= exclusive_end,
        "causal parquet columns/row count changed",
    )
    rows = []
    for frame, raw in enumerate(table.slice(0, exclusive_end).to_pylist()):
        index = raw.get("index")
        _require(
            isinstance(index, int) and not isinstance(index, bool) and index == frame,
            f"parquet prefix index changed at {frame}",
        )
        intrinsic = _matrix_list(
            raw.get("observation.camera_intrinsic"),
            (3, 3),
            f"parquet[{frame}].intrinsic",
        )
        extrinsic = _matrix_list(
            raw.get("observation.camera_extrinsic"),
            (4, 4),
            f"parquet[{frame}].extrinsic",
        )
        action = _matrix_list(raw.get("action"), (4, 4), f"parquet[{frame}].action")
        _rigid_transform(extrinsic, f"parquet[{frame}].extrinsic")
        _rigid_transform(action, f"parquet[{frame}].action")
        rows.append(
            {
                "index": index,
                "observation.camera_intrinsic": intrinsic,
                "observation.camera_extrinsic": extrinsic,
                "action": action,
            }
        )
    return rows


def _hash_sequence(values: Sequence[object]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


def _resolved_roots(
    manifest: Mapping[str, Any],
    overrides: Mapping[str, Path | str] | None,
) -> dict[str, Path]:
    raw = manifest.get("input_roots")
    _require(isinstance(raw, Mapping), "manifest input_roots are malformed")
    required = {"episode_root", "environment_root"}
    _require(required <= set(raw), "manifest lacks episode/environment roots")
    override_values = {} if overrides is None else dict(overrides)
    _require(
        set(override_values) <= required, "root overrides contain unsupported keys"
    )
    result = {}
    for key in sorted(required):
        value = override_values.get(key, raw[key])
        _require(isinstance(value, (str, Path)), f"{key} is malformed")
        path = Path(value).resolve()
        _require(
            path.is_dir() and not path.is_symlink(), f"{key} is unavailable: {path}"
        )
        result[key] = path
    return result


def _manifest_indexes(
    manifest: Mapping[str, Any],
    *,
    expected_sample_count: int | None,
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    _require(
        manifest.get("schema_version") == MANIFEST_SCHEMA,
        "input is not the formal multistage causal manifest",
    )
    samples = manifest.get("samples")
    scenes = manifest.get("scenes")
    _require(isinstance(samples, list) and bool(samples), "manifest samples are absent")
    _require(isinstance(scenes, list) and bool(scenes), "manifest scenes are absent")
    if expected_sample_count is not None:
        _require(
            isinstance(expected_sample_count, int)
            and not isinstance(expected_sample_count, bool)
            and expected_sample_count > 0,
            "expected sample count must be positive",
        )
        _require(
            len(samples) == expected_sample_count,
            f"manifest sample count differs from exact pin: {len(samples)} != {expected_sample_count}",
        )
    summary = manifest.get("summary")
    if isinstance(summary, Mapping) and "sample_count" in summary:
        _require(
            summary["sample_count"] == len(samples),
            "manifest summary sample count changed",
        )
    scene_index: dict[str, Mapping[str, Any]] = {}
    for raw_scene in scenes:
        _require(isinstance(raw_scene, Mapping), "manifest scene row is malformed")
        scene_id = raw_scene.get("scene")
        _require(
            isinstance(scene_id, str) and bool(scene_id),
            "manifest scene id is malformed",
        )
        _require(scene_id not in scene_index, f"duplicate manifest scene {scene_id}")
        selected = raw_scene.get("selected_episodes")
        _require(
            isinstance(selected, list)
            and len(selected) == 2
            and all(isinstance(row, Mapping) for row in selected),
            f"scene {scene_id} does not contain exactly two pinned episodes",
        )
        scene_index[scene_id] = raw_scene

    ordered: list[Mapping[str, Any]] = []
    sample_ids = set()
    categories = set()
    source_state_cover: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for raw_sample in samples:
        sample = _exact_mapping(raw_sample, SAMPLE_KEYS, "manifest sample")
        sample_id = sample["sample_id"]
        _require(
            isinstance(sample_id, str) and bool(sample_id), "sample_id is malformed"
        )
        _require(sample_id not in sample_ids, f"duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
        _require(
            sample["split_role"] in ("train", "development"),
            f"forbidden split role for {sample_id}",
        )
        _require(
            sample["scene"] in scene_index, f"sample scene is absent for {sample_id}"
        )
        scene = scene_index[str(sample["scene"])]
        episodes = _episode_index(scene)
        _require(
            sample["source_episode"] in episodes and sample["goal_episode"] in episodes,
            f"sample episode is not selected for {sample_id}",
        )
        source_episode = episodes[str(sample["source_episode"])]
        partner_episode = next(
            episode for episode in episodes if episode != sample["source_episode"]
        )
        _require(
            sample["state_source"] == "expert",
            f"sample state source changed for {sample_id}",
        )
        _require(
            sample["goal_role"] in ("B", "C"), f"goal role changed for {sample_id}"
        )
        _require(
            sample["goal_variant"] in ("factual", "counterfactual"),
            f"goal variant changed for {sample_id}",
        )
        decision = sample["decision_frame"]
        _require(
            isinstance(decision, int)
            and not isinstance(decision, bool)
            and decision >= 1,
            f"decision frame is invalid for {sample_id}",
        )
        source_id = f"{sample['scene']}/{sample['source_episode']}"
        goal_id = f"{sample['scene']}/{sample['goal_episode']}"
        _require(
            sample["source_episode_id"] == source_id,
            f"source episode id changed for {sample_id}",
        )
        _require(
            sample["goal_source_episode_id"] == goal_id,
            f"goal episode id changed for {sample_id}",
        )
        factual = sample["source_episode"] == sample["goal_episode"]
        _require(
            factual == (sample["goal_variant"] == "factual"),
            f"goal variant/source identity disagrees for {sample_id}",
        )
        expected_goal_episode = sample["source_episode"] if factual else partner_episode
        _require(
            sample["goal_episode"] == expected_goal_episode,
            f"goal episode does not match the factual pair for {sample_id}",
        )
        switches = source_episode.get("switches")
        midpoint = source_episode.get("goal_b_midpoint_frame")
        _require(
            isinstance(switches, list)
            and len(switches) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in switches
            )
            and switches[0] < switches[1]
            and isinstance(midpoint, int)
            and not isinstance(midpoint, bool)
            and switches[0] < midpoint < switches[1],
            f"source episode state boundaries are malformed for {sample_id}",
        )
        state_contract = {
            "goal_b_t0": ("B", switches[0]),
            "goal_b_midpoint_t1": ("B", midpoint),
            "goal_c_t0": ("C", switches[1]),
        }
        state_name = sample["state_name"]
        _require(
            state_name in state_contract,
            f"sample state name changed for {sample_id}",
        )
        expected_role, expected_decision = state_contract[str(state_name)]
        _require(
            sample["goal_role"] == expected_role and decision == expected_decision,
            f"sample state role/decision changed for {sample_id}",
        )
        _require(
            sample_id
            == (
                f"{sample['split_role']}/{sample['scene']}/"
                f"{sample['source_episode']}/{state_name}/"
                f"{sample['goal_variant']}"
            ),
            f"sample_id fields disagree for {sample_id}",
        )
        source_state_cover.setdefault(
            (str(sample["scene"]), str(sample["source_episode"])), set()
        ).add((str(state_name), str(sample["goal_variant"])))
        categories.add((str(sample["goal_role"]), str(sample["goal_variant"])))
        ordered.append(sample)
    _require(
        categories
        == {
            ("B", "factual"),
            ("B", "counterfactual"),
            ("C", "factual"),
            ("C", "counterfactual"),
        },
        "manifest does not cover Goal-B/C factual/counterfactual categories",
    )
    expected_state_cover = {
        (state_name, variant)
        for state_name in (
            "goal_b_t0",
            "goal_b_midpoint_t1",
            "goal_c_t0",
        )
        for variant in ("factual", "counterfactual")
    }
    _require(
        all(cover == expected_state_cover for cover in source_state_cover.values()),
        "manifest does not provide an exact six-row state/variant cover per source",
    )
    if expected_sample_count == 600:
        selected_sources = {
            (scene_id, str(episode["episode"]))
            for scene_id, scene in scene_index.items()
            for episode in scene["selected_episodes"]
        }
        _require(
            len(scene_index) == 50 and set(source_state_cover) == selected_sources,
            "formal 600-sample manifest does not cover both episodes in 50 scenes",
        )
    return ordered, scene_index


def _episode_index(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = scene["selected_episodes"]
    result = {}
    for row in rows:
        episode = row.get("episode")
        _require(
            isinstance(episode, str) and bool(episode),
            "selected episode id is malformed",
        )
        _require(episode not in result, f"duplicate selected episode {episode}")
        result[episode] = row
    return result


def _camera_height(
    manifest: Mapping[str, Any],
    *,
    scene_id: str,
    episode_id: str,
    metadata: Mapping[str, Any],
    metadata_record: Mapping[str, Any],
) -> float:
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, Mapping), "manifest provenance is malformed")
    bindings = provenance.get("camera_height_bindings")
    _require(isinstance(bindings, list), "camera height bindings are absent")
    if "camera_height_bindings_sha256" in provenance:
        _require(
            _valid_sha(
                provenance["camera_height_bindings_sha256"],
                "camera height bindings SHA",
            )
            == sha256_bytes(canonical_json_bytes(bindings)),
            "camera height binding collection changed",
        )
    matches = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and row.get("scene") == scene_id
        and row.get("episode") == episode_id
    ]
    _require(
        len(matches) == 1,
        f"camera height binding is not unique for {scene_id}/{episode_id}",
    )
    binding = matches[0]
    _require(
        binding.get("metadata_content_sha256") == metadata_record["content_sha256"],
        f"camera height binding metadata pin changed for {scene_id}/{episode_id}",
    )
    value = _finite_float(binding.get("camera_height_m"), "camera height binding")
    _require(value > 0.0, "camera height must be positive")
    if "camera_height_m" in metadata:
        _require(
            _finite_float(metadata["camera_height_m"], "metadata camera height")
            == value,
            f"metadata/binding camera height differs for {scene_id}/{episode_id}",
        )
    return value


def _goal_metadata(
    metadata: Mapping[str, Any],
    goal_role: str,
    label: str,
    *,
    expected_curve_length: int,
) -> tuple[int, dict[str, Any]]:
    goal_index = 0 if goal_role == "B" else 1
    goals = metadata.get("goals")
    _require(
        isinstance(goals, list)
        and len(goals) == 2
        and isinstance(goals[goal_index], Mapping),
        f"{label} goal list is malformed",
    )
    raw = goals[goal_index]
    expected_name = goal_role
    expected_kind = "novel" if goal_role == "B" else "revisit"
    _require(
        raw.get("name", expected_name) == expected_name, f"{label} goal name changed"
    )
    _require(raw.get("kind") == expected_kind, f"{label} goal kind changed")
    position = raw.get("pos")
    _require(
        isinstance(position, list) and len(position) == 3,
        f"{label} goal position is malformed",
    )
    position_values = [
        _finite_float(value, f"{label} goal position") for value in position
    ]
    yaw = _finite_float(raw.get("yaw_habitat"), f"{label} goal yaw")
    curve = raw.get("covis_curve")
    _require(
        isinstance(curve, list) and bool(curve), f"{label} covisibility curve is absent"
    )
    curve_values = [
        _finite_float(value, f"{label} covisibility curve") for value in curve
    ]
    _require(
        all(0.0 <= value <= 1.0 for value in curve_values),
        f"{label} covisibility curve is out of range",
    )
    _require(
        len(curve_values) == expected_curve_length,
        f"{label} covisibility curve length changed",
    )
    goal = {
        "name": expected_name,
        "kind": expected_kind,
        "position_data_zup_m": position_values,
        "yaw_habitat_rad": yaw,
        "covis_curve": curve_values,
        "covis_curve_sha256": sha256_bytes(canonical_json_bytes(curve_values)),
    }
    goal["goal_pose_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "position_data_zup_m": position_values,
                "yaw_habitat_rad": yaw,
            }
        )
    )
    return goal_index, goal


def _verify_prefix(
    *,
    sample: Mapping[str, Any],
    source_episode_record: Mapping[str, Any],
    episode_root: Path,
    manifest_file_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
    physical_file_cache: dict[Path, dict[str, Any]],
) -> tuple[FrameAsset, ...]:
    sample_id = str(sample["sample_id"])
    scene_id = str(sample["scene"])
    episode_id = str(sample["source_episode"])
    decision = int(sample["decision_frame"])
    n_frames = source_episode_record.get("n_frames")
    _require(
        isinstance(n_frames, int)
        and not isinstance(n_frames, bool)
        and decision <= n_frames,
        f"decision exceeds source episode for {sample_id}",
    )
    parquet_relative = f"{scene_id}/{episode_id}/data/chunk-000/episode_000000.parquet"
    parquet_path, _parquet_record = _verify_file_record(
        source_episode_record.get("parquet"),
        episode_root,
        f"{sample_id}.source_parquet",
        expected_relative=parquet_relative,
        verification_cache=manifest_file_cache,
    )
    parquet_rows = _read_parquet_prefix(parquet_path, decision)
    frames = []
    modality_records: dict[str, dict[str, str]] = {}
    by_modality: dict[str, list[dict[str, Any]]] = {}
    for modality, suffix in (("rgb", ".jpg"), ("depth", ".png")):
        records = []
        for frame in range(decision):
            relative = (
                f"{scene_id}/{episode_id}/videos/chunk-000/"
                f"observation.images.{modality}/{frame}{suffix}"
            )
            path = episode_root / relative
            record = _physical_file_record(path, episode_root, physical_file_cache)
            _require(record["path"] == relative, f"{sample_id} {modality} path changed")
            records.append(record)
        by_modality[modality] = records
        modality_records[modality] = {
            "path_sequence_sha256": _hash_sequence(
                [record["path"] for record in records]
            ),
            "content_sequence_sha256": _hash_sequence(
                [
                    {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "content_sha256": record["content_sha256"],
                    }
                    for record in records
                ]
            ),
        }
    parquet_rows_sha = _hash_sequence(parquet_rows)
    prefix_payload = {
        "frame_count": decision,
        "rgb": modality_records["rgb"],
        "depth": modality_records["depth"],
        "parquet_rows_sha256": parquet_rows_sha,
    }
    reconstructed = {
        "exclusive_end_frame": decision,
        "frame_count": decision,
        "modalities": modality_records,
        "parquet_columns": list(PARQUET_PREFIX_COLUMNS),
        "parquet_row_count": decision,
        "parquet_rows_sha256": parquet_rows_sha,
        "causal_prefix_sha256": sha256_bytes(canonical_json_bytes(prefix_payload)),
    }
    declared = _exact_mapping(
        sample.get("causal_prefix"), PREFIX_KEYS, f"{sample_id}.causal_prefix"
    )
    _require(
        reconstructed == declared, f"causal prefix content changed for {sample_id}"
    )
    state_frame = sample.get("state_frame")
    _require(
        state_frame == by_modality["rgb"][-1],
        f"state frame is not decision_frame-1 for {sample_id}",
    )
    for frame, row in enumerate(parquet_rows):
        frames.append(
            FrameAsset(
                frame=frame,
                rgb_path=episode_root / by_modality["rgb"][frame]["path"],
                depth_path=episode_root / by_modality["depth"][frame]["path"],
                rgb_record=by_modality["rgb"][frame],
                depth_record=by_modality["depth"][frame],
                intrinsic=_finite_matrix(
                    row["observation.camera_intrinsic"],
                    (3, 3),
                    f"{sample_id}.intrinsic[{frame}]",
                ),
                action=_finite_matrix(
                    row["action"], (4, 4), f"{sample_id}.action[{frame}]"
                ),
            )
        )
    return tuple(frames)


def _prepare_contexts(
    *,
    manifest: Mapping[str, Any],
    roots: Mapping[str, Path],
    samples: Sequence[Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
) -> list[SampleContext]:
    contexts = []
    episode_root = roots["episode_root"]
    environment_root = roots["environment_root"]
    manifest_file_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    physical_file_cache: dict[Path, dict[str, Any]] = {}
    prefix_cache: dict[
        tuple[str, str, int, str],
        tuple[tuple[FrameAsset, ...], Mapping[str, Any]],
    ] = {}
    metadata_cache: dict[Path, Mapping[str, Any]] = {}
    goal_intrinsic_cache: dict[Path, np.ndarray] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        scene_id = str(sample["scene"])
        scene = scenes[scene_id]
        _require(
            scene.get("split_role") == sample["split_role"],
            f"scene/sample split differs for {sample_id}",
        )
        environment_path, environment_record = _verify_file_record(
            scene.get("environment"),
            environment_root,
            f"{sample_id}.environment",
            verification_cache=manifest_file_cache,
        )
        episodes = _episode_index(scene)
        source_episode_id = str(sample["source_episode"])
        goal_episode_id = str(sample["goal_episode"])
        _require(
            source_episode_id in episodes and goal_episode_id in episodes,
            f"sample episode is not selected for {sample_id}",
        )
        source_episode = episodes[source_episode_id]
        goal_episode = episodes[goal_episode_id]
        prefix_key = (
            scene_id,
            source_episode_id,
            int(sample["decision_frame"]),
            str(sample["causal_prefix"]["causal_prefix_sha256"]),
        )
        if prefix_key not in prefix_cache:
            prefix_cache[prefix_key] = (
                _verify_prefix(
                    sample=sample,
                    source_episode_record=source_episode,
                    episode_root=episode_root,
                    manifest_file_cache=manifest_file_cache,
                    physical_file_cache=physical_file_cache,
                ),
                copy.deepcopy(dict(sample["causal_prefix"])),
            )
        frames, cached_prefix = prefix_cache[prefix_key]
        _require(
            sample["causal_prefix"] == cached_prefix,
            f"causal prefix conflicts within state pair for {sample_id}",
        )
        _require(
            sample["state_frame"] == frames[-1].rgb_record,
            f"state frame conflicts within state pair for {sample_id}",
        )

        goal_role = str(sample["goal_role"])
        goal_key = "goal_b" if goal_role == "B" else "goal_c"
        goal_filename = "goal_1.jpg" if goal_role == "B" else "goal_2.jpg"
        goal_relative = f"{scene_id}/{goal_episode_id}/{goal_filename}"
        query_path, query_record = _verify_file_record(
            goal_episode.get(goal_key),
            episode_root,
            f"{sample_id}.goal_image",
            expected_relative=goal_relative,
            verification_cache=manifest_file_cache,
        )
        _require(
            sample["goal"] == query_record, f"sample goal pin changed for {sample_id}"
        )
        metadata_relative = f"{scene_id}/{goal_episode_id}/meta/gen_meta.json"
        metadata_path, metadata_record = _verify_file_record(
            goal_episode.get("metadata"),
            episode_root,
            f"{sample_id}.goal_metadata",
            expected_relative=metadata_relative,
            verification_cache=manifest_file_cache,
        )
        if metadata_path not in metadata_cache:
            metadata_cache[metadata_path] = _load_metadata(
                metadata_path, f"{sample_id}.goal_metadata"
            )
        metadata = metadata_cache[metadata_path]
        _require(
            metadata.get("scene") in (scene_id, f"{scene_id}.glb"),
            f"goal metadata scene changed for {sample_id}",
        )
        _require(
            metadata.get("n_frames") == goal_episode.get("n_frames"),
            f"goal metadata frame count changed for {sample_id}",
        )
        suffix = goal_episode_id.removeprefix("episode_")
        _require(
            suffix.isdigit() and metadata.get("ep_idx") == int(suffix),
            f"goal metadata episode index changed for {sample_id}",
        )
        _require(
            metadata.get("n_legs") == 3
            and metadata.get("switches") == goal_episode.get("switches"),
            f"goal metadata three-leg switches changed for {sample_id}",
        )
        _require(
            metadata.get("frame_convention")
            == ("positions+parquet in data(Zup,M_W); yaw_habitat in render frame"),
            f"goal metadata frame convention changed for {sample_id}",
        )
        switches = goal_episode["switches"]
        expected_curve_length = int(switches[0 if goal_role == "B" else 1])
        goal_index, goal_pose = _goal_metadata(
            metadata,
            goal_role,
            sample_id,
            expected_curve_length=expected_curve_length,
        )
        camera_height = _camera_height(
            manifest,
            scene_id=scene_id,
            episode_id=goal_episode_id,
            metadata=metadata,
            metadata_record=metadata_record,
        )
        goal_parquet_relative = (
            f"{scene_id}/{goal_episode_id}/data/chunk-000/episode_000000.parquet"
        )
        goal_parquet_path, _goal_parquet_record = _verify_file_record(
            goal_episode.get("parquet"),
            episode_root,
            f"{sample_id}.goal_parquet",
            expected_relative=goal_parquet_relative,
            verification_cache=manifest_file_cache,
        )
        if goal_parquet_path not in goal_intrinsic_cache:
            goal_row = _read_parquet_prefix(goal_parquet_path, 1)[0]
            goal_intrinsic_cache[goal_parquet_path] = _finite_matrix(
                goal_row["observation.camera_intrinsic"],
                (3, 3),
                f"{sample_id}.goal_intrinsic",
            )
        goal_intrinsic = goal_intrinsic_cache[goal_parquet_path]
        contexts.append(
            SampleContext(
                sample=sample,
                scene_record=scene,
                environment_path=environment_path,
                environment_record=environment_record,
                query_path=query_path,
                query_record=query_record,
                goal_metadata_record=metadata_record,
                goal_metadata=metadata,
                goal_index=goal_index,
                goal_pose=goal_pose,
                camera_height_m=camera_height,
                goal_intrinsic=goal_intrinsic,
                frames=frames,
            )
        )
    _require(
        [context.sample["sample_id"] for context in contexts]
        == [sample["sample_id"] for sample in samples],
        "prepared context order differs from manifest sample order",
    )
    return contexts


def _cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    _require(
        query.ndim == 1
        and candidates.ndim == 2
        and candidates.shape[0] >= 1
        and candidates.shape[1] == query.shape[0]
        and query.shape[0] >= 1
        and np.isfinite(query).all()
        and np.isfinite(candidates).all(),
        "DINO query/candidate embeddings are malformed",
    )
    query_norm = float(np.linalg.norm(query))
    candidate_norm = np.linalg.norm(candidates, axis=1)
    _require(
        query_norm > 0.0 and np.all(candidate_norm > 0.0),
        "DINO embeddings have zero norm",
    )
    scores = candidates @ query / (candidate_norm * query_norm)
    return np.clip(scores, -1.0, 1.0)


def temporal_nms_shortlist(
    frames: Sequence[int],
    scores: Sequence[float],
    *,
    top_k: int,
    radius: int,
) -> list[tuple[int, float]]:
    _require(
        len(frames) == len(scores) and bool(frames),
        "DINO shortlist arrays are malformed",
    )
    _require(len(set(frames)) == len(frames), "DINO shortlist frame ids are duplicated")
    _require(
        all(
            isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0
            for frame in frames
        ),
        "DINO shortlist frame id is invalid",
    )
    _require(
        all(math.isfinite(float(score)) for score in scores),
        "DINO shortlist score is non-finite",
    )
    _require(top_k >= 1 and radius >= 0, "DINO shortlist configuration is invalid")
    ranked = sorted(
        ((int(frame), float(score)) for frame, score in _strict_zip(frames, scores)),
        key=lambda row: (-row[1], row[0]),
    )
    selected: list[tuple[int, float]] = []
    for frame, score in ranked:
        if any(abs(frame - accepted_frame) <= radius for accepted_frame, _ in selected):
            continue
        selected.append((frame, score))
        if len(selected) == top_k:
            break
    _require(bool(selected), "temporal NMS produced an empty shortlist")
    return selected


def _ordered_dino_inputs(
    contexts: Sequence[SampleContext],
) -> tuple[list[Path], dict[Path, Mapping[str, Any]]]:
    """Return the exact, relocation-stable DINO input universe.

    Ordering by the manifest-relative path avoids making embeddings depend on
    which scratch mount happens to host the same content-pinned dataset.
    """

    pins_by_path: dict[Path, Mapping[str, Any]] = {}
    path_by_relative: dict[str, Path] = {}
    for context in contexts:
        bindings = [
            (context.query_path, context.query_record),
            *((frame.rgb_path, frame.rgb_record) for frame in context.frames),
        ]
        for raw_path, raw_record in bindings:
            path = raw_path.resolve()
            record = copy.deepcopy(dict(raw_record))
            relative = _relative_path(record.get("path"), "DINO input")
            if path in pins_by_path:
                _require(
                    pins_by_path[path] == record,
                    f"DINO input has conflicting content pins: {path}",
                )
            else:
                pins_by_path[path] = record
            if relative in path_by_relative:
                _require(
                    path_by_relative[relative] == path,
                    f"DINO relative path resolves to multiple files: {relative}",
                )
            else:
                path_by_relative[relative] = path
    ordered = [path_by_relative[key] for key in sorted(path_by_relative)]
    _require(
        set(ordered) == set(pins_by_path) and bool(ordered),
        "DINO input pin cover is incomplete",
    )
    return ordered, pins_by_path


def _prepare_embedding_input_universe(
    *,
    roots: Mapping[str, Path],
    samples: Sequence[Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Path], dict[Path, Mapping[str, Any]]]:
    """Verify only the RGB/query authority needed by GPU Stage A.

    Depth, pose, metadata, GLB, and geometry validation intentionally remain
    in CPU Stage B.  This keeps the allocated GPU busy while still rebuilding
    every causal RGB-prefix hash from physical files before DINO inference.
    """

    episode_root = roots["episode_root"]
    manifest_file_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    physical_file_cache: dict[Path, dict[str, Any]] = {}
    pins_by_path: dict[Path, Mapping[str, Any]] = {}
    relative_to_path: dict[str, Path] = {}

    def bind(path: Path, record: Mapping[str, Any], label: str) -> None:
        resolved = path.resolve()
        relative = _relative_path(record.get("path"), label)
        if resolved in pins_by_path:
            _require(
                pins_by_path[resolved] == record,
                f"Stage-A DINO input has conflicting pins: {resolved}",
            )
        else:
            pins_by_path[resolved] = copy.deepcopy(dict(record))
        if relative in relative_to_path:
            _require(
                relative_to_path[relative] == resolved,
                f"Stage-A DINO relative path is not unique: {relative}",
            )
        else:
            relative_to_path[relative] = resolved

    prefix_cache: dict[
        tuple[str, str, int, str], tuple[tuple[dict[str, Any], ...], Mapping[str, Any]]
    ] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        scene_id = str(sample["scene"])
        scene = scenes[scene_id]
        episodes = _episode_index(scene)
        source_episode_id = str(sample["source_episode"])
        goal_episode_id = str(sample["goal_episode"])
        source_episode = episodes[source_episode_id]
        goal_episode = episodes[goal_episode_id]
        decision = int(sample["decision_frame"])
        _require(
            isinstance(source_episode.get("n_frames"), int)
            and decision <= int(source_episode["n_frames"]),
            f"Stage-A decision exceeds source episode: {sample_id}",
        )
        prefix = _exact_mapping(
            sample.get("causal_prefix"), PREFIX_KEYS, f"{sample_id}.causal_prefix"
        )
        _require(
            prefix["exclusive_end_frame"] == decision
            and prefix["frame_count"] == decision,
            f"Stage-A causal prefix interval changed: {sample_id}",
        )
        modalities = prefix.get("modalities")
        _require(
            isinstance(modalities, Mapping)
            and set(modalities) == {"rgb", "depth"}
            and isinstance(modalities["rgb"], Mapping)
            and set(modalities["rgb"])
            == {"path_sequence_sha256", "content_sequence_sha256"},
            f"Stage-A causal modality contract changed: {sample_id}",
        )
        prefix_key = (
            scene_id,
            source_episode_id,
            decision,
            str(prefix["causal_prefix_sha256"]),
        )
        if prefix_key not in prefix_cache:
            rgb_records = []
            for frame in range(decision):
                relative = (
                    f"{scene_id}/{source_episode_id}/videos/chunk-000/"
                    f"observation.images.rgb/{frame}.jpg"
                )
                record = _physical_file_record(
                    episode_root / relative,
                    episode_root,
                    physical_file_cache,
                )
                _require(
                    record["path"] == relative,
                    f"Stage-A RGB path changed: {sample_id}/{frame}",
                )
                rgb_records.append(record)
            reconstructed_rgb = {
                "path_sequence_sha256": _hash_sequence(
                    [record["path"] for record in rgb_records]
                ),
                "content_sequence_sha256": _hash_sequence(
                    [
                        {
                            "path": record["path"],
                            "bytes": record["bytes"],
                            "content_sha256": record["content_sha256"],
                        }
                        for record in rgb_records
                    ]
                ),
            }
            _require(
                reconstructed_rgb == modalities["rgb"],
                f"Stage-A RGB causal prefix content changed: {sample_id}",
            )
            prefix_cache[prefix_key] = (
                tuple(copy.deepcopy(rgb_records)),
                copy.deepcopy(dict(prefix)),
            )
        rgb_records, cached_prefix = prefix_cache[prefix_key]
        _require(
            cached_prefix == prefix and sample.get("state_frame") == rgb_records[-1],
            f"Stage-A causal state binding changed: {sample_id}",
        )
        for record in rgb_records:
            bind(
                episode_root / str(record["path"]),
                record,
                f"{sample_id}.candidate_rgb",
            )

        goal_role = str(sample["goal_role"])
        goal_key = "goal_b" if goal_role == "B" else "goal_c"
        goal_filename = "goal_1.jpg" if goal_role == "B" else "goal_2.jpg"
        relative = f"{scene_id}/{goal_episode_id}/{goal_filename}"
        query_path, query_record = _verify_file_record(
            goal_episode.get(goal_key),
            episode_root,
            f"{sample_id}.goal_image",
            expected_relative=relative,
            verification_cache=manifest_file_cache,
        )
        _require(
            sample.get("goal") == query_record,
            f"Stage-A goal content pin changed: {sample_id}",
        )
        bind(query_path, query_record, f"{sample_id}.goal_image")
    ordered = [relative_to_path[key] for key in sorted(relative_to_path)]
    _require(
        bool(ordered) and set(ordered) == set(pins_by_path),
        "Stage-A exact DINO input cover is incomplete",
    )
    return ordered, pins_by_path


def _scene_grouped_work_items(
    contexts: Sequence[SampleContext],
) -> list[tuple[int, SampleContext]]:
    work = sorted(
        enumerate(contexts),
        key=lambda row: (str(row[1].sample["scene"]), row[0]),
    )
    _require(
        sorted(index for index, _context in work) == list(range(len(contexts))),
        "scene-grouped work is not an exact context permutation",
    )
    return work


def _verify_dino_input_content(
    path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == record.get("bytes")
        and sha256_file(path) == record.get("content_sha256"),
        f"{label} changed: {path}",
    )


def _write_fsynced_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_atomic_directory(
    destination: Path,
    files: Mapping[str, bytes],
    *,
    child_directories: Sequence[str] = (),
) -> None:
    _require(not destination.exists(), f"progress path already exists: {destination}")
    parent = destination.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
    )
    try:
        for directory in child_directories:
            _relative_path(directory, "progress child directory")
            (stage / directory).mkdir()
        for name, payload in sorted(files.items()):
            _relative_path(name, "progress file")
            _write_fsynced_file(stage / name, payload)
        _fsync_directory(stage)
        _require(
            not destination.exists(),
            f"progress path appeared during publication: {destination}",
        )
        os.rename(stage, destination)
        _fsync_directory(parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class _TeacherProgressStore:
    """Content-addressed DINO and sample shards for interruption recovery."""

    def __init__(self, root: Path | str, run_payload: Mapping[str, Any]) -> None:
        self.root = Path(root)
        payload = copy.deepcopy(dict(run_payload))
        self.run_signature_sha256 = sha256_bytes(canonical_json_bytes(payload))
        self.run_record = {
            "schema_version": PROGRESS_SCHEMA,
            "status": "resumable_until_exact_final_merge",
            "run_signature_payload": payload,
            "run_signature_sha256": self.run_signature_sha256,
        }
        run_bytes = canonical_json_bytes(self.run_record)
        expected_run_files = {
            "run.json": run_bytes,
            "run.json.sha256": _sidecar_bytes("run.json", run_bytes),
        }
        if self.root.exists() or self.root.is_symlink():
            _require(
                self.root.is_dir() and not self.root.is_symlink(),
                "progress root must be a physical directory",
            )
            _require(
                frozenset(path.name for path in self.root.iterdir())
                == {"run.json", "run.json.sha256", "embeddings", "samples"},
                "progress root file set changed",
            )
            for name, expected in expected_run_files.items():
                path = self.root / name
                _require(
                    path.is_file()
                    and not path.is_symlink()
                    and path.read_bytes() == expected,
                    f"progress run signature drifted: {name}",
                )
            for name in ("embeddings", "samples"):
                path = self.root / name
                _require(
                    path.is_dir() and not path.is_symlink(),
                    f"progress {name} directory changed",
                )
                self._remove_owned_orphan_stages(path)
        else:
            _publish_atomic_directory(
                self.root,
                expected_run_files,
                child_directories=("embeddings", "samples"),
            )
        self.embedding_root = self.root / "embeddings"
        self.sample_root = self.root / "samples"

    @staticmethod
    def _remove_owned_orphan_stages(parent: Path) -> None:
        """Remove only mkdtemp directories from an interrupted shard publish."""

        for path in parent.iterdir():
            name = path.name
            if not (name.startswith(".") and name.endswith(".tmp")):
                continue
            stem = name[1:].split(".", 1)[0]
            parts = stem.split("_", 1)
            owned = (
                len(parts) == 2
                and len(parts[0]) == 6
                and parts[0].isdigit()
                and len(parts[1]) == 20
                and all(character in "0123456789abcdef" for character in parts[1])
            )
            _require(
                owned and path.is_dir() and not path.is_symlink(),
                f"unrecognized progress temporary entry: {path.name}",
            )
            shutil.rmtree(path)
        _fsync_directory(parent)

    @staticmethod
    def _load_shard(
        shard: Path,
        *,
        expected_files: frozenset[str],
        metadata_name: str = "metadata.json",
    ) -> tuple[Mapping[str, Any], bytes]:
        _require(
            shard.is_dir() and not shard.is_symlink(),
            f"progress shard is not a physical directory: {shard}",
        )
        _require(
            frozenset(path.name for path in shard.iterdir()) == expected_files,
            f"progress shard file set changed: {shard.name}",
        )
        for name in expected_files:
            path = shard / name
            _require(
                path.is_file() and not path.is_symlink(),
                f"progress shard file is not physical: {shard.name}/{name}",
            )
        metadata_path = shard / metadata_name
        metadata_bytes = metadata_path.read_bytes()
        value = _decode_json(metadata_bytes, f"progress shard {shard.name}")
        _require(
            isinstance(value, Mapping)
            and metadata_bytes == canonical_json_bytes(value),
            f"progress shard metadata is not canonical: {shard.name}",
        )
        _require(
            (shard / f"{metadata_name}.sha256").read_bytes()
            == _sidecar_bytes(metadata_name, metadata_bytes),
            f"progress shard metadata sidecar changed: {shard.name}",
        )
        return value, metadata_bytes

    def embedding_shard_name(
        self,
        chunk_index: int,
        descriptor: Mapping[str, Any],
    ) -> str:
        descriptor_sha = sha256_bytes(canonical_json_bytes(descriptor))
        return f"{chunk_index:06d}_{descriptor_sha[:20]}"

    def load_embedding(
        self,
        *,
        chunk_index: int,
        descriptor: Mapping[str, Any],
    ) -> tuple[str, np.ndarray] | None:
        name = self.embedding_shard_name(chunk_index, descriptor)
        shard = self.embedding_root / name
        if not shard.exists() and not shard.is_symlink():
            return None
        metadata, _ = self._load_shard(
            shard,
            expected_files=frozenset(
                {"metadata.json", "metadata.json.sha256", "embeddings.npy"}
            ),
        )
        _require(
            metadata.get("schema_version") == EMBEDDING_SHARD_SCHEMA
            and metadata.get("run_signature_sha256") == self.run_signature_sha256
            and metadata.get("descriptor") == descriptor,
            f"DINO embedding shard provenance changed: {name}",
        )
        array_record = metadata.get("array")
        _require(isinstance(array_record, Mapping), "embedding array pin is absent")
        array_bytes = (shard / "embeddings.npy").read_bytes()
        _require(
            len(array_bytes) == array_record.get("bytes")
            and sha256_bytes(array_bytes) == array_record.get("content_sha256"),
            f"DINO embedding shard content changed: {name}",
        )
        try:
            array = np.load(io.BytesIO(array_bytes), allow_pickle=False)
        except Exception as error:
            raise CausalTeacherError(
                f"cannot decode DINO embedding shard: {name}"
            ) from error
        array = np.asarray(array)
        _require(
            array.dtype == np.dtype("float32")
            and list(array.shape) == array_record.get("shape")
            and array.ndim == 2
            and array.shape[0] == len(descriptor["inputs"])
            and array.shape[1] >= 1
            and np.isfinite(array).all()
            and np.all(np.linalg.norm(array, axis=1) > 0.0),
            f"DINO embedding shard array is malformed: {name}",
        )
        return name, array.copy()

    def save_embedding(
        self,
        *,
        chunk_index: int,
        descriptor: Mapping[str, Any],
        embeddings: np.ndarray,
    ) -> str:
        name = self.embedding_shard_name(chunk_index, descriptor)
        destination = self.embedding_root / name
        _require(
            not destination.exists() and not destination.is_symlink(),
            f"DINO embedding shard appeared concurrently: {name}",
        )
        array = np.asarray(embeddings, dtype="<f4", order="C")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        array_bytes = buffer.getvalue()
        metadata = {
            "schema_version": EMBEDDING_SHARD_SCHEMA,
            "run_signature_sha256": self.run_signature_sha256,
            "descriptor": copy.deepcopy(dict(descriptor)),
            "array": {
                "dtype": "float32_le",
                "shape": list(array.shape),
                "bytes": len(array_bytes),
                "content_sha256": sha256_bytes(array_bytes),
            },
        }
        metadata_bytes = canonical_json_bytes(metadata)
        _publish_atomic_directory(
            destination,
            {
                "metadata.json": metadata_bytes,
                "metadata.json.sha256": _sidecar_bytes("metadata.json", metadata_bytes),
                "embeddings.npy": array_bytes,
            },
        )
        return name

    def _sample_name(self, sample_index: int, sample_id: str) -> str:
        return f"{sample_index:06d}_{sha256_bytes(sample_id.encode('utf-8'))[:20]}"

    def load_sample(
        self,
        *,
        sample_index: int,
        sample_id: str,
    ) -> Mapping[str, Any] | None:
        name = self._sample_name(sample_index, sample_id)
        shard = self.sample_root / name
        if not shard.exists() and not shard.is_symlink():
            return None
        metadata, _ = self._load_shard(
            shard,
            expected_files=frozenset({"metadata.json", "metadata.json.sha256"}),
        )
        _require(
            metadata.get("schema_version") == SAMPLE_SHARD_SCHEMA
            and metadata.get("run_signature_sha256") == self.run_signature_sha256
            and metadata.get("sample_index") == sample_index
            and metadata.get("sample_id") == sample_id
            and isinstance(metadata.get("record"), Mapping),
            f"teacher sample shard provenance changed: {name}",
        )
        return copy.deepcopy(dict(metadata["record"]))

    def save_sample(
        self,
        *,
        sample_index: int,
        sample_id: str,
        record: Mapping[str, Any],
    ) -> None:
        name = self._sample_name(sample_index, sample_id)
        destination = self.sample_root / name
        _require(
            not destination.exists() and not destination.is_symlink(),
            f"teacher sample shard appeared concurrently: {name}",
        )
        metadata = {
            "schema_version": SAMPLE_SHARD_SCHEMA,
            "run_signature_sha256": self.run_signature_sha256,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "record": copy.deepcopy(dict(record)),
        }
        metadata_bytes = canonical_json_bytes(metadata)
        _publish_atomic_directory(
            destination,
            {
                "metadata.json": metadata_bytes,
                "metadata.json.sha256": _sidecar_bytes("metadata.json", metadata_bytes),
            },
        )

    def verify_complete(
        self,
        *,
        embedding_names: Sequence[str],
        sample_ids: Sequence[str],
    ) -> None:
        _require(
            frozenset(path.name for path in self.embedding_root.iterdir())
            == frozenset(embedding_names),
            "progress DINO shard cover is not exact",
        )
        expected_samples = frozenset(
            self._sample_name(index, sample_id)
            for index, sample_id in enumerate(sample_ids)
        )
        _require(
            frozenset(path.name for path in self.sample_root.iterdir())
            == expected_samples,
            "progress sample shard cover is not exact",
        )


def _embedding_descriptor_chunks(
    *,
    unique_paths: Sequence[Path],
    pins_by_path: Mapping[Path, Mapping[str, Any]],
    provider_identity_sha256: str,
    chunk_size: int,
) -> list[tuple[int, dict[str, Any], list[Path]]]:
    _require(
        isinstance(chunk_size, int)
        and not isinstance(chunk_size, bool)
        and chunk_size >= 1,
        "embedding chunk size must be a positive integer",
    )
    result = []
    for chunk_index, start in enumerate(range(0, len(unique_paths), chunk_size)):
        paths = list(unique_paths[start : start + chunk_size])
        descriptor = {
            "chunk_index": chunk_index,
            "start": start,
            "end_exclusive": start + len(paths),
            "dino_provider_identity_sha256": provider_identity_sha256,
            "inputs": [
                {
                    "path": pins_by_path[path]["path"],
                    "path_sha256": pins_by_path[path]["path_sha256"],
                    "bytes": pins_by_path[path]["bytes"],
                    "content_sha256": pins_by_path[path]["content_sha256"],
                }
                for path in paths
            ],
        }
        result.append((chunk_index, descriptor, paths))
    _require(bool(result), "DINO embedding chunk cover is empty")
    return result


def _embedding_shard_receipt(
    progress: _TeacherProgressStore,
    *,
    name: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    shard = progress.embedding_root / name
    metadata, metadata_bytes = progress._load_shard(
        shard,
        expected_files=frozenset(
            {"metadata.json", "metadata.json.sha256", "embeddings.npy"}
        ),
    )
    _require(
        metadata.get("schema_version") == EMBEDDING_SHARD_SCHEMA
        and metadata.get("run_signature_sha256") == progress.run_signature_sha256
        and metadata.get("descriptor") == descriptor,
        f"DINO embedding shard provenance changed: {name}",
    )
    array = metadata.get("array")
    _require(isinstance(array, Mapping), "DINO embedding shard array pin is absent")
    array_path = shard / "embeddings.npy"
    array_bytes = array_path.read_bytes()
    _require(
        len(array_bytes) == array.get("bytes")
        and sha256_bytes(array_bytes) == array.get("content_sha256"),
        f"DINO embedding shard content changed: {name}",
    )
    return {
        "name": name,
        "descriptor_sha256": sha256_bytes(canonical_json_bytes(descriptor)),
        "metadata": {
            "bytes": len(metadata_bytes),
            "content_sha256": sha256_bytes(metadata_bytes),
        },
        "array": copy.deepcopy(dict(array)),
    }


def _copy_embedding_bundle_atomic(
    *,
    destination: Path,
    receipt: Mapping[str, Any],
    progress: _TeacherProgressStore,
    resume: bool,
) -> dict[str, Any]:
    receipt_bytes = canonical_json_bytes(receipt)
    expected_receipt_sha = sha256_bytes(receipt_bytes)
    if destination.exists() or destination.is_symlink():
        _require(resume, f"embedding bundle already exists: {destination}")
        _require(
            destination.is_dir()
            and not destination.is_symlink()
            and frozenset(path.name for path in destination.iterdir())
            == EMBEDDING_BUNDLE_FILES,
            "embedding bundle file set changed",
        )
        receipt_path = destination / EMBEDDING_RECEIPT_NAME
        sidecar_path = destination / f"{EMBEDDING_RECEIPT_NAME}.sha256"
        _require(
            receipt_path.is_file()
            and not receipt_path.is_symlink()
            and receipt_path.read_bytes() == receipt_bytes
            and sidecar_path.is_file()
            and not sidecar_path.is_symlink()
            and sidecar_path.read_bytes()
            == _sidecar_bytes(EMBEDDING_RECEIPT_NAME, receipt_bytes),
            "embedding bundle receipt content drifted",
        )
        # Constructing the pinned provider performs the complete physical shard
        # audit, including array decoding and exact input coverage.
        PinnedDINOEmbeddingBundleProvider(
            bundle_directory=destination,
            expected_receipt_sha256=expected_receipt_sha,
            expected_manifest_sha256=str(receipt["manifest"]["sha256"]),
            expected_producer_sha256=receipt["producer"].get("expected_content_sha256"),
        )
        return {
            "status": "resumed",
            "directory": str(destination.resolve()),
            "receipt_sha256": expected_receipt_sha,
            "input_count": receipt["exact_cover"]["input_count"],
        }

    parent = destination.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
    )
    try:
        _write_fsynced_file(stage / EMBEDDING_RECEIPT_NAME, receipt_bytes)
        _write_fsynced_file(
            stage / f"{EMBEDDING_RECEIPT_NAME}.sha256",
            _sidecar_bytes(EMBEDDING_RECEIPT_NAME, receipt_bytes),
        )
        shard_root = stage / "shards"
        shard_root.mkdir()
        for shard_record in receipt["shards"]:
            name = str(shard_record["name"])
            source = progress.embedding_root / name
            destination_shard = shard_root / name
            destination_shard.mkdir()
            for filename in (
                "metadata.json",
                "metadata.json.sha256",
                "embeddings.npy",
            ):
                payload = (source / filename).read_bytes()
                _write_fsynced_file(destination_shard / filename, payload)
            _fsync_directory(destination_shard)
        _fsync_directory(shard_root)
        _fsync_directory(stage)
        _require(
            not destination.exists() and not destination.is_symlink(),
            f"embedding bundle appeared during publication: {destination}",
        )
        os.rename(stage, destination)
        _fsync_directory(parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "written",
        "directory": str(destination.resolve()),
        "receipt_sha256": expected_receipt_sha,
        "input_count": receipt["exact_cover"]["input_count"],
    }


def build_dino_embedding_bundle(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    embedding_provider: DINOEmbeddingProvider,
    progress_directory: Path | str,
    output_directory: Path | str,
    root_overrides: Mapping[str, Path | str] | None = None,
    expected_sample_count: int | None = 600,
    embedding_chunk_size: int = 256,
    expected_producer_sha256: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run Stage A under an exclusive single-writer progress lock."""

    with _exclusive_stage_writer(progress_directory):
        return _build_dino_embedding_bundle_locked(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            embedding_provider=embedding_provider,
            progress_directory=progress_directory,
            output_directory=output_directory,
            root_overrides=root_overrides,
            expected_sample_count=expected_sample_count,
            embedding_chunk_size=embedding_chunk_size,
            expected_producer_sha256=expected_producer_sha256,
            resume=resume,
        )


def _build_dino_embedding_bundle_locked(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    embedding_provider: DINOEmbeddingProvider,
    progress_directory: Path | str,
    output_directory: Path | str,
    root_overrides: Mapping[str, Path | str] | None = None,
    expected_sample_count: int | None = 600,
    embedding_chunk_size: int = 256,
    expected_producer_sha256: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Stage A: materialize a complete, signed DINO-only embedding bundle."""

    output_path = Path(output_directory)
    _require(
        resume or (not output_path.exists() and not output_path.is_symlink()),
        f"DINO embedding bundle already exists without --resume: {output_path}",
    )
    manifest_sha = _valid_sha(manifest_sha256, "manifest SHA")
    _require(
        sha256_bytes(canonical_json_bytes(manifest)) == manifest_sha,
        "in-memory manifest differs from canonical SHA pin",
    )
    dino_identity = _validate_provider_identity(
        embedding_provider.identity, DINO_IDENTITY_SCHEMA, "DINO provider"
    )
    producer_sha = sha256_file(Path(__file__))
    if expected_producer_sha256 is not None:
        _require(
            _valid_sha(expected_producer_sha256, "expected producer source SHA")
            == producer_sha,
            "teacher producer source differs from external pin",
        )
    samples, scenes = _manifest_indexes(
        manifest, expected_sample_count=expected_sample_count
    )
    roots = _resolved_roots(manifest, root_overrides)
    unique_paths, pins_by_path = _prepare_embedding_input_universe(
        roots=roots,
        samples=samples,
        scenes=scenes,
    )
    input_records = [copy.deepcopy(dict(pins_by_path[path])) for path in unique_paths]
    invocation = {
        "chunk_size": embedding_chunk_size,
        "input_order": "manifest_relative_posix_path_ascending",
        "provider_output_cast": "float32",
        "provider_model_loads_per_process_run_max": 1,
        "stage_authority": "candidate_embeddings_only_no_geometry_or_labels",
    }
    run_payload = {
        "output_schema_version": EMBEDDING_BUNDLE_SCHEMA,
        "manifest_sha256": manifest_sha,
        "manifest_sample_id_sequence_sha256": sha256_bytes(
            canonical_json_bytes([str(sample["sample_id"]) for sample in samples])
        ),
        "dino_provider": dino_identity,
        "input_sequence_sha256": sha256_bytes(canonical_json_bytes(input_records)),
        "embedding_invocation": invocation,
        "producer_content_sha256": producer_sha,
        "producer_expected_sha256": expected_producer_sha256,
    }
    progress_path = Path(progress_directory)
    _require(
        resume or (not progress_path.exists() and not progress_path.is_symlink()),
        f"DINO embedding progress already exists without --resume: {progress_path}",
    )
    progress = _TeacherProgressStore(progress_path, run_payload)
    _require(
        not any(progress.sample_root.iterdir()),
        "DINO-only progress unexpectedly contains geometry sample shards",
    )
    chunks = _embedding_descriptor_chunks(
        unique_paths=unique_paths,
        pins_by_path=pins_by_path,
        provider_identity_sha256=str(dino_identity["identity_sha256"]),
        chunk_size=embedding_chunk_size,
    )
    arrays: list[np.ndarray | None] = []
    names: list[str | None] = []
    for chunk_index, descriptor, _paths in chunks:
        cached = progress.load_embedding(chunk_index=chunk_index, descriptor=descriptor)
        if cached is None:
            names.append(None)
            arrays.append(None)
        else:
            name, array = cached
            names.append(name)
            arrays.append(array)
    missing_indices = [index for index, value in enumerate(arrays) if value is None]
    if missing_indices:
        missing_path_chunks = [chunks[index][2] for index in missing_indices]
        for paths in missing_path_chunks:
            for path in paths:
                _verify_dino_input_content(
                    path,
                    pins_by_path[path],
                    label="DINO input before embedding",
                )
        streaming = getattr(embedding_provider, "embed_chunks", None)
        if callable(streaming):
            produced = 0
            for produced, raw_array in enumerate(
                streaming(missing_path_chunks), start=1
            ):
                _require(
                    produced <= len(missing_indices),
                    "DINO streaming provider returned excess chunks",
                )
                index = missing_indices[produced - 1]
                array = np.asarray(raw_array, dtype=np.float32)
                paths = chunks[index][2]
                _require(
                    array.ndim == 2
                    and array.shape[0] == len(paths)
                    and array.shape[1] >= 1
                    and np.isfinite(array).all()
                    and np.all(np.linalg.norm(array, axis=1) > 0.0),
                    f"DINO streaming provider returned malformed chunk {index}",
                )
                for path in paths:
                    _verify_dino_input_content(
                        path,
                        pins_by_path[path],
                        label="DINO input after embedding chunk",
                    )
                arrays[index] = array.copy()
                # Publish before asking the generator for its next chunk.
                names[index] = progress.save_embedding(
                    chunk_index=chunks[index][0],
                    descriptor=chunks[index][1],
                    embeddings=array,
                )
            _require(
                produced == len(missing_indices),
                "DINO streaming provider returned an incomplete chunk cover",
            )
        else:
            missing_paths = [path for paths in missing_path_chunks for path in paths]
            embedded = np.asarray(
                embedding_provider.embed(missing_paths), dtype=np.float32
            )
            _require(
                embedded.ndim == 2
                and embedded.shape[0] == len(missing_paths)
                and embedded.shape[1] >= 1
                and np.isfinite(embedded).all()
                and np.all(np.linalg.norm(embedded, axis=1) > 0.0),
                "DINO provider returned malformed cache-miss embeddings",
            )
            offset = 0
            for index in missing_indices:
                count = len(chunks[index][2])
                array = embedded[offset : offset + count].copy()
                offset += count
                arrays[index] = array
                names[index] = progress.save_embedding(
                    chunk_index=chunks[index][0],
                    descriptor=chunks[index][1],
                    embeddings=array,
                )
            _require(offset == len(missing_paths), "DINO cache-miss split changed")
    for path in unique_paths:
        _verify_dino_input_content(
            path, pins_by_path[path], label="DINO input after embedding"
        )
    _require(
        all(isinstance(name, str) for name in names)
        and all(array is not None for array in arrays),
        "DINO embedding shard cover is incomplete",
    )
    expected_names = [str(name) for name in names]
    _require(
        frozenset(path.name for path in progress.embedding_root.iterdir())
        == frozenset(expected_names),
        "DINO-only progress shard cover is not exact",
    )
    shard_receipts = [
        _embedding_shard_receipt(
            progress,
            name=str(names[index]),
            descriptor=chunks[index][1],
        )
        for index in range(len(chunks))
    ]
    dimensions = {int(array.shape[1]) for array in arrays if array is not None}
    _require(len(dimensions) == 1, "DINO embedding dimensions changed across shards")
    receipt = {
        "schema_version": EMBEDDING_BUNDLE_SCHEMA,
        "status": "complete",
        "manifest": {
            "schema_version": MANIFEST_SCHEMA,
            "sha256": manifest_sha,
            "sample_count": len(samples),
            "sample_id_sequence_sha256": run_payload[
                "manifest_sample_id_sequence_sha256"
            ],
        },
        "dino_provider": dino_identity,
        "embedding_invocation": invocation,
        "producer": {
            "path": Path(__file__).name,
            "content_sha256": producer_sha,
            "expected_content_sha256": expected_producer_sha256,
        },
        "run_signature_sha256": progress.run_signature_sha256,
        "input_records": input_records,
        "input_sequence_sha256": run_payload["input_sequence_sha256"],
        "embedding_dimension": next(iter(dimensions)),
        "shards": shard_receipts,
        "exact_cover": {
            "input_count": len(input_records),
            "shard_count": len(shard_receipts),
            "all_manifest_query_and_causal_candidate_rgb_inputs": True,
            "geometry_or_label_authority": False,
        },
        "deployment_approved": False,
    }
    return _copy_embedding_bundle_atomic(
        destination=output_path,
        receipt=receipt,
        progress=progress,
        resume=resume,
    )


class PinnedDINOEmbeddingBundleProvider:
    """Stage-B-only provider backed by an immutable signed Stage-A bundle.

    This class has deliberately no Torch or LingBot import.  It accepts only
    the complete input sequence signed by Stage A, rechecks every shard and
    physical RGB input at each consumption boundary, and has no cache-miss
    fallback.
    """

    def __init__(
        self,
        *,
        bundle_directory: Path | str,
        expected_receipt_sha256: str,
        expected_manifest_sha256: str,
        expected_producer_sha256: str | None = None,
        episode_root: Path | str | None = None,
    ) -> None:
        root = Path(bundle_directory)
        _require(
            root.is_dir()
            and not root.is_symlink()
            and frozenset(path.name for path in root.iterdir())
            == EMBEDDING_BUNDLE_FILES,
            "embedding bundle file set changed",
        )
        self.root = root.resolve()
        receipt_path = self.root / EMBEDDING_RECEIPT_NAME
        _require(
            receipt_path.is_file() and not receipt_path.is_symlink(),
            "embedding receipt is unavailable",
        )
        raw = receipt_path.read_bytes()
        expected_receipt_sha = _valid_sha(
            expected_receipt_sha256, "expected embedding receipt SHA"
        )
        _require(
            sha256_bytes(raw) == expected_receipt_sha
            and (self.root / f"{EMBEDDING_RECEIPT_NAME}.sha256").read_bytes()
            == _sidecar_bytes(EMBEDDING_RECEIPT_NAME, raw),
            "embedding receipt content or sidecar changed",
        )
        receipt = _decode_json(raw, "embedding receipt")
        _require(
            isinstance(receipt, Mapping)
            and frozenset(receipt) == EMBEDDING_RECEIPT_KEYS
            and raw == canonical_json_bytes(receipt)
            and receipt.get("schema_version") == EMBEDDING_BUNDLE_SCHEMA
            and receipt.get("status") == "complete"
            and receipt.get("deployment_approved") is False,
            "embedding receipt schema/status changed",
        )
        manifest = receipt.get("manifest")
        expected_manifest_sha = _valid_sha(
            expected_manifest_sha256, "expected manifest SHA"
        )
        _require(
            isinstance(manifest, Mapping)
            and set(manifest)
            == {
                "schema_version",
                "sha256",
                "sample_count",
                "sample_id_sequence_sha256",
            }
            and manifest.get("schema_version") == MANIFEST_SCHEMA
            and manifest.get("sha256") == expected_manifest_sha,
            "embedding bundle is bound to a different manifest",
        )
        producer = receipt.get("producer")
        _require(
            isinstance(producer, Mapping)
            and producer.get("path") == Path(__file__).name
            and isinstance(producer.get("content_sha256"), str),
            "embedding producer provenance is malformed",
        )
        producer_sha = _valid_sha(producer["content_sha256"], "embedding producer SHA")
        if expected_producer_sha256 is not None:
            expected_producer_sha = _valid_sha(
                expected_producer_sha256, "expected embedding producer SHA"
            )
            _require(
                producer_sha == expected_producer_sha
                and producer.get("expected_content_sha256") == expected_producer_sha,
                "embedding bundle was produced by a different source",
            )
            _require(
                sha256_file(Path(__file__)) == expected_producer_sha,
                "current embedding consumer source differs from the producer pin",
            )
        identity = _validate_provider_identity(
            receipt.get("dino_provider"), DINO_IDENTITY_SCHEMA, "DINO provider"
        )
        records = receipt.get("input_records")
        shards = receipt.get("shards")
        exact_cover = receipt.get("exact_cover")
        dimension = receipt.get("embedding_dimension")
        _require(
            isinstance(records, list)
            and bool(records)
            and isinstance(shards, list)
            and bool(shards)
            and isinstance(exact_cover, Mapping)
            and exact_cover.get("input_count") == len(records)
            and exact_cover.get("shard_count") == len(shards)
            and exact_cover.get("all_manifest_query_and_causal_candidate_rgb_inputs")
            is True
            and exact_cover.get("geometry_or_label_authority") is False
            and isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension >= 1,
            "embedding receipt exact cover is malformed",
        )
        _require(
            receipt.get("input_sequence_sha256")
            == sha256_bytes(canonical_json_bytes(records)),
            "embedding input sequence fingerprint changed",
        )
        invocation = receipt.get("embedding_invocation")
        _require(
            isinstance(invocation, Mapping)
            and set(invocation)
            == {
                "chunk_size",
                "input_order",
                "provider_output_cast",
                "provider_model_loads_per_process_run_max",
                "stage_authority",
            }
            and invocation.get("input_order")
            == "manifest_relative_posix_path_ascending"
            and invocation.get("provider_output_cast") == "float32"
            and invocation.get("provider_model_loads_per_process_run_max") == 1
            and invocation.get("stage_authority")
            == "candidate_embeddings_only_no_geometry_or_labels",
            "embedding invocation contract changed",
        )
        reconstructed_run_payload = {
            "output_schema_version": EMBEDDING_BUNDLE_SCHEMA,
            "manifest_sha256": expected_manifest_sha,
            "manifest_sample_id_sequence_sha256": manifest["sample_id_sequence_sha256"],
            "dino_provider": identity,
            "input_sequence_sha256": receipt["input_sequence_sha256"],
            "embedding_invocation": invocation,
            "producer_content_sha256": producer_sha,
            "producer_expected_sha256": producer.get("expected_content_sha256"),
        }
        _require(
            receipt.get("run_signature_sha256")
            == sha256_bytes(canonical_json_bytes(reconstructed_run_payload)),
            "embedding run signature cannot be reconstructed",
        )
        shard_root = self.root / "shards"
        _require(
            shard_root.is_dir() and not shard_root.is_symlink(),
            "embedding shard root changed",
        )
        expected_names = []
        vectors: list[np.ndarray] = []
        reconstructed_records: list[Mapping[str, Any]] = []
        snapshots: dict[Path, tuple[int, int, int, str]] = {}
        expected_start = 0
        for shard_index, shard_receipt_raw in enumerate(shards):
            _require(
                isinstance(shard_receipt_raw, Mapping),
                "embedding shard receipt is malformed",
            )
            shard_receipt = shard_receipt_raw
            name = shard_receipt.get("name")
            _require(
                isinstance(name, str)
                and name == PurePosixPath(name).name
                and bool(name),
                "embedding shard name is invalid",
            )
            expected_names.append(name)
            shard = shard_root / name
            metadata, metadata_bytes = _TeacherProgressStore._load_shard(
                shard,
                expected_files=frozenset(
                    {"metadata.json", "metadata.json.sha256", "embeddings.npy"}
                ),
            )
            descriptor = metadata.get("descriptor")
            array_record = metadata.get("array")
            _require(
                metadata.get("schema_version") == EMBEDDING_SHARD_SCHEMA
                and metadata.get("run_signature_sha256")
                == receipt.get("run_signature_sha256")
                and isinstance(descriptor, Mapping)
                and descriptor.get("chunk_index") == shard_index
                and descriptor.get("start") == expected_start
                and descriptor.get("dino_provider_identity_sha256")
                == identity["identity_sha256"]
                and shard_receipt.get("descriptor_sha256")
                == sha256_bytes(canonical_json_bytes(descriptor))
                and shard_receipt.get("metadata")
                == {
                    "bytes": len(metadata_bytes),
                    "content_sha256": sha256_bytes(metadata_bytes),
                }
                and isinstance(array_record, Mapping)
                and shard_receipt.get("array") == array_record,
                f"embedding shard receipt/provenance changed: {name}",
            )
            inputs = descriptor.get("inputs")
            end = descriptor.get("end_exclusive")
            _require(
                isinstance(inputs, list)
                and bool(inputs)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and end == expected_start + len(inputs),
                f"embedding shard interval changed: {name}",
            )
            array_path = shard / "embeddings.npy"
            array_bytes = array_path.read_bytes()
            _require(
                len(array_bytes) == array_record.get("bytes")
                and sha256_bytes(array_bytes) == array_record.get("content_sha256"),
                f"embedding shard content changed: {name}",
            )
            try:
                array = np.load(io.BytesIO(array_bytes), allow_pickle=False)
            except Exception as error:
                raise CausalTeacherError(
                    f"cannot decode embedding shard: {name}"
                ) from error
            array = np.asarray(array)
            _require(
                array.dtype == np.dtype("float32")
                and array.shape == (len(inputs), dimension)
                and np.isfinite(array).all()
                and np.all(np.linalg.norm(array, axis=1) > 0.0),
                f"embedding shard array is malformed: {name}",
            )
            vectors.append(array.copy())
            reconstructed_records.extend(copy.deepcopy(inputs))
            expected_start = end
            for filename in (
                "metadata.json",
                "metadata.json.sha256",
                "embeddings.npy",
            ):
                path = shard / filename
                stat = path.stat()
                snapshots[path] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ino,
                    sha256_file(path),
                )
        _require(
            frozenset(path.name for path in shard_root.iterdir())
            == frozenset(expected_names)
            and len(expected_names) == len(set(expected_names))
            and reconstructed_records == records
            and expected_start == len(records),
            "embedding shard/input cover is not exact",
        )
        normalized_records = []
        for index, raw_record in enumerate(records):
            record = _exact_mapping(
                raw_record, FILE_RECORD_KEYS, f"embedding input[{index}]"
            )
            relative = _relative_path(record["path"], f"embedding input[{index}]")
            _require(
                record["path_sha256"] == sha256_bytes(relative.encode("utf-8"))
                and isinstance(record["bytes"], int)
                and not isinstance(record["bytes"], bool)
                and record["bytes"] > 0,
                f"embedding input record changed: {index}",
            )
            _valid_sha(record["content_sha256"], f"embedding input[{index}] SHA")
            normalized_records.append(copy.deepcopy(dict(record)))
        _require(
            [record["path"] for record in normalized_records]
            == sorted(record["path"] for record in normalized_records)
            and len({record["path"] for record in normalized_records})
            == len(normalized_records),
            "embedding input order or uniqueness changed",
        )
        self._identity = identity
        self._receipt = copy.deepcopy(dict(receipt))
        self._receipt_sha256 = expected_receipt_sha
        self._records = normalized_records
        self._embeddings = np.concatenate(vectors, axis=0)
        self._snapshots = snapshots
        self._episode_root = (
            None if episode_root is None else Path(episode_root).resolve()
        )
        if self._episode_root is not None:
            _require(
                self._episode_root.is_dir() and not self._episode_root.is_symlink(),
                "embedding episode root is unavailable",
            )
        self._consumed = False

    @property
    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    @property
    def bundle_binding(self) -> Mapping[str, Any]:
        return {
            "schema_version": EMBEDDING_BUNDLE_SCHEMA,
            "receipt_sha256": self._receipt_sha256,
            "manifest_sha256": self._receipt["manifest"]["sha256"],
            "input_sequence_sha256": self._receipt["input_sequence_sha256"],
            "input_count": len(self._records),
            "authority": "signed_embeddings_only_no_cache_miss_fallback",
        }

    def _verify_static_bundle(self) -> None:
        _require(
            frozenset(path.name for path in self.root.iterdir())
            == EMBEDDING_BUNDLE_FILES
            and frozenset(path.name for path in (self.root / "shards").iterdir())
            == frozenset(record["name"] for record in self._receipt["shards"]),
            "embedding bundle file or shard cover changed after provider construction",
        )
        raw = (self.root / EMBEDDING_RECEIPT_NAME).read_bytes()
        _require(
            sha256_bytes(raw) == self._receipt_sha256
            and raw == canonical_json_bytes(self._receipt)
            and (self.root / f"{EMBEDDING_RECEIPT_NAME}.sha256").read_bytes()
            == _sidecar_bytes(EMBEDDING_RECEIPT_NAME, raw),
            "embedding bundle receipt changed after provider construction",
        )
        for path, snapshot in self._snapshots.items():
            stat = path.stat()
            _require(
                (stat.st_size, stat.st_mtime_ns, stat.st_ino, sha256_file(path))
                == snapshot,
                f"embedding shard changed after provider construction: {path}",
            )

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        _require(
            not self._consumed, "signed embedding bundle may be consumed only once"
        )
        _require(
            self._episode_root is not None,
            "episode root is required before consuming signed embeddings",
        )
        self._verify_static_bundle()
        requested_records = []
        for raw_path in paths:
            path = Path(raw_path)
            _require(
                path.is_file() and not path.is_symlink(),
                f"DINO cached input is unavailable: {path}",
            )
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(self._episode_root).as_posix()
            except ValueError as error:
                raise CausalTeacherError(
                    f"DINO cached input escapes episode root: {path}"
                ) from error
            requested_records.append((resolved, relative))
        _require(
            [relative for _path, relative in requested_records]
            == [record["path"] for record in self._records],
            "signed embedding cache request is not the exact Stage-A input sequence",
        )
        for (path, _relative), record in _strict_zip(requested_records, self._records):
            _verify_dino_input_content(
                path, record, label="signed embedding physical RGB input"
            )
        self._verify_static_bundle()
        for (path, _relative), record in _strict_zip(requested_records, self._records):
            _verify_dino_input_content(
                path, record, label="signed embedding physical RGB input"
            )
        self._consumed = True
        return self._embeddings.copy()


def _load_depth(path: Path, expected_record: Mapping[str, Any]) -> np.ndarray:
    _require(
        path.stat().st_size == expected_record["bytes"]
        and sha256_file(path) == expected_record["content_sha256"],
        f"depth changed after prefix verification: {path}",
    )
    try:
        from PIL import Image

        with Image.open(path) as image:
            encoded = np.asarray(image)
    except Exception as error:
        raise CausalTeacherError(f"cannot decode depth image: {path}") from error
    _require(encoded.ndim == 2, f"depth image is not single-channel: {path}")
    depth = encoded.astype(np.float64) / 10000.0
    _require(np.isfinite(depth).all(), f"depth image is non-finite: {path}")
    return depth


def _yaw_rotation_habitat(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _goal_camera_pose(
    goal_pose: Mapping[str, Any],
    camera_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    position_data = np.asarray(goal_pose["position_data_zup_m"], dtype=np.float64)
    floor_habitat = M_W.T @ position_data
    camera_habitat = floor_habitat + np.asarray([0.0, camera_height_m, 0.0])
    yaw = float(goal_pose["yaw_habitat_rad"])
    rotation_habitat = _yaw_rotation_habitat(yaw)
    camera_to_world_data = np.eye(4, dtype=np.float64)
    camera_to_world_data[:3, :3] = M_W @ rotation_habitat
    camera_to_world_data[:3, 3] = M_W @ camera_habitat
    _rigid_transform(camera_to_world_data, "goal camera-to-world transform")
    return camera_habitat, camera_to_world_data


def _render_goal_points(
    *,
    context: SampleContext,
    reference_depth: np.ndarray,
    renderer: GoalDepthRenderer,
    renderer_identity_sha256: str,
    config: TeacherConfig,
    environment_verification_cache: dict[Path, tuple[str, int, int, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = reference_depth.shape
    environment_path = context.environment_path.resolve()
    expected_environment_sha = str(context.environment_record["content_sha256"])
    environment_stat = environment_path.stat()
    environment_snapshot = (
        expected_environment_sha,
        environment_stat.st_size,
        environment_stat.st_mtime_ns,
        environment_stat.st_ino,
    )
    if environment_path in environment_verification_cache:
        _require(
            environment_verification_cache[environment_path] == environment_snapshot,
            "scene GLB changed between goal renders",
        )
    else:
        _require(
            environment_path.stat().st_size == context.environment_record["bytes"]
            and sha256_file(environment_path) == expected_environment_sha,
            f"scene GLB changed before goal rendering for {context.sample['sample_id']}",
        )
        environment_verification_cache[environment_path] = environment_snapshot
    camera_habitat, camera_to_world_data = _goal_camera_pose(
        context.goal_pose, context.camera_height_m
    )
    depth = np.asarray(
        renderer.render_depth(
            scene_id=str(context.sample["scene"]),
            glb_path=context.environment_path,
            expected_glb_sha256=expected_environment_sha,
            camera_position_habitat=camera_habitat,
            yaw_habitat=float(context.goal_pose["yaw_habitat_rad"]),
            height=height,
            width=width,
            intrinsic=context.goal_intrinsic,
        ),
        dtype=np.float32,
    )
    _require(
        depth.shape == (height, width) and np.isfinite(depth).all(),
        f"rendered goal depth shape/value changed for {context.sample['sample_id']}",
    )
    render_request = {
        "scene_id": str(context.sample["scene"]),
        "environment_content_sha256": context.environment_record["content_sha256"],
        "goal_pose_sha256": context.goal_pose["goal_pose_sha256"],
        "camera_height_m": context.camera_height_m,
        "intrinsic_sha256": sha256_bytes(
            canonical_json_bytes(context.goal_intrinsic.tolist())
        ),
        "height": height,
        "width": width,
        "renderer_identity_sha256": renderer_identity_sha256,
    }
    render_record = {
        "request_sha256": sha256_bytes(canonical_json_bytes(render_request)),
        "depth_shape": [height, width],
        "depth_dtype": "float32_le",
        "depth_content_sha256": sha256_bytes(
            np.asarray(depth, dtype="<f4", order="C").tobytes(order="C")
        ),
        **render_request,
    }
    points = backproject_world(
        depth.astype(np.float64),
        context.goal_intrinsic,
        camera_to_world_data,
        stride=config.backprojection_stride,
    )
    _require(
        points.ndim == 2 and points.shape[1] == 3, "rendered goal points are malformed"
    )
    return points, render_record


def _candidate_label(
    *,
    context: SampleContext,
    candidate: FrameAsset,
    renderer: GoalDepthRenderer,
    renderer_identity_sha256: str,
    config: TeacherConfig,
    render_cache: dict[str, tuple[np.ndarray, dict[str, Any]]],
    environment_verification_cache: dict[Path, tuple[str, int, int, int]],
) -> dict[str, Any]:
    sample = context.sample
    factual = sample["goal_variant"] == "factual"
    curve = context.goal_pose["covis_curve"]
    if factual and candidate.frame < len(curve):
        score = float(curve[candidate.frame])
        label_source = "metadata_covis_curve"
        rendered = None
        label_input = {
            "source": label_source,
            "metadata_content_sha256": context.goal_metadata_record["content_sha256"],
            "goal_index": context.goal_index,
            "covis_curve_sha256": context.goal_pose["covis_curve_sha256"],
            "candidate_frame": candidate.frame,
        }
    else:
        # Counterfactual samples always enter this branch.  Factual samples use
        # it only when their decision prefix extends beyond the stored curve.
        candidate_depth = _load_depth(candidate.depth_path, candidate.depth_record)
        cache_key = sha256_bytes(
            canonical_json_bytes(
                {
                    "scene": sample["scene"],
                    "environment": context.environment_record["content_sha256"],
                    "goal_pose": context.goal_pose["goal_pose_sha256"],
                    "goal_intrinsic": context.goal_intrinsic.tolist(),
                    "camera_height_m": context.camera_height_m,
                    "shape": list(candidate_depth.shape),
                    "renderer": renderer_identity_sha256,
                    "stride": config.backprojection_stride,
                }
            )
        )
        if cache_key not in render_cache:
            render_cache[cache_key] = _render_goal_points(
                context=context,
                reference_depth=candidate_depth,
                renderer=renderer,
                renderer_identity_sha256=renderer_identity_sha256,
                config=config,
                environment_verification_cache=environment_verification_cache,
            )
        points, rendered = render_cache[cache_key]
        score = projected_covisibility(
            points,
            candidate_depth,
            candidate.intrinsic,
            candidate.action,
            tolerance=config.depth_tolerance_m,
        )
        label_source = "rendered_goal_depth_reprojection"
        label_input = {
            "source": label_source,
            "rendered_goal_depth_sha256": rendered["depth_content_sha256"],
            "candidate_depth_content_sha256": candidate.depth_record["content_sha256"],
            "candidate_action_sha256": sha256_bytes(
                canonical_json_bytes(candidate.action.tolist())
            ),
            "candidate_intrinsic_sha256": sha256_bytes(
                canonical_json_bytes(candidate.intrinsic.tolist())
            ),
            "depth_tolerance_m": config.depth_tolerance_m,
            "backprojection_stride": config.backprojection_stride,
        }
    _require(
        0.0 <= score <= 1.0 and math.isfinite(score), "teacher covisibility is invalid"
    )
    return {
        "label_source": label_source,
        "covisibility": score,
        "label": covisibility_label(
            score,
            positive_threshold=config.positive_threshold,
            negative_threshold=config.negative_threshold,
        ),
        "label_input_sha256": sha256_bytes(canonical_json_bytes(label_input)),
        "label_input": label_input,
        "rendered_goal_depth": rendered,
    }


def _accumulate_label_count(
    candidate: Mapping[str, Any],
    *,
    label_source_counts: dict[str, int],
    label_counts: dict[str, int],
) -> None:
    source = candidate.get("label_source")
    _require(
        source in {"metadata_covis_curve", "rendered_goal_depth_reprojection"},
        "candidate label source is invalid",
    )
    label = candidate.get("label")
    _require(label in (-1, 0, 1), "candidate label is invalid")
    label_source_counts[str(source)] = label_source_counts.get(str(source), 0) + 1
    label_name = {1: "positive", 0: "negative", -1: "ambiguous"}[int(label)]
    label_counts[label_name] += 1


def _candidate_oracle_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> dict[str, Any]:
    """Summarize DINO shortlist quality without any additional inference.

    Recall is conditional on a labelled positive existing somewhere in the
    emitted shortlist.  This deliberately does not claim recall against the
    unlabelled full causal prefix.
    """

    recall_ks = (1, 2, 4, 8, 16, 32)

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        positive_first_ranks = []
        strict_shortlist_no_match = 0
        shortlist_ambiguous = 0
        for record in group:
            candidates = record.get("candidates")
            _require(
                isinstance(candidates, list) and bool(candidates),
                "oracle summary encountered an empty shortlist",
            )
            ranks = []
            scores = []
            for expected_rank, candidate in enumerate(candidates):
                _require(
                    isinstance(candidate, Mapping)
                    and candidate.get("candidate_rank") == expected_rank,
                    "oracle summary candidate ranks are not contiguous",
                )
                score = _finite_float(
                    candidate.get("covisibility"),
                    "oracle candidate covisibility",
                )
                _require(0.0 <= score <= 1.0, "oracle covisibility is out of range")
                scores.append(score)
                if score >= positive_threshold:
                    ranks.append(expected_rank)
            if ranks:
                positive_first_ranks.append(min(ranks))
            elif all(score <= negative_threshold for score in scores):
                strict_shortlist_no_match += 1
            else:
                shortlist_ambiguous += 1
        positive = len(positive_first_ranks)
        total = len(group)
        _require(
            positive + strict_shortlist_no_match + shortlist_ambiguous == total,
            "oracle session partition is incomplete",
        )
        return {
            "sessions": total,
            "session_has_positive": positive,
            "strict_shortlist_no_match": strict_shortlist_no_match,
            "shortlist_ambiguous": shortlist_ambiguous,
            "session_positive_coverage": positive / total if total else None,
            "shortlist_conditional_positive_recall_at_k": {
                str(k): (
                    sum(rank < k for rank in positive_first_ranks) / positive
                    if positive
                    else None
                )
                for k in recall_ks
            },
        }

    grouped: dict[str, list[Mapping[str, Any]]] = {"overall": list(records)}
    for record in records:
        split = str(record.get("split_role"))
        role = str(record.get("goal_role"))
        variant = str(record.get("goal_variant"))
        for key in (
            f"split/{split}",
            f"goal/{role}",
            f"variant/{variant}",
            f"joint/{split}/{role}/{variant}",
        ):
            grouped.setdefault(key, []).append(record)
    group_summaries = {key: summarize(group) for key, group in sorted(grouped.items())}
    by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(str(record.get("scene")), []).append(record)
    scene_summaries = {
        scene: summarize(group) for scene, group in sorted(by_scene.items())
    }
    worst_scene = {}
    for k in recall_ks:
        eligible = [
            (scene, summary)
            for scene, summary in scene_summaries.items()
            if summary["session_has_positive"] > 0
        ]
        if not eligible:
            worst_scene[str(k)] = None
            continue
        scene, summary = min(
            eligible,
            key=lambda row: (
                row[1]["shortlist_conditional_positive_recall_at_k"][str(k)],
                row[0],
            ),
        )
        worst_scene[str(k)] = {
            "scene": scene,
            "positive_sessions": summary["session_has_positive"],
            "recall": summary["shortlist_conditional_positive_recall_at_k"][str(k)],
        }
    return {
        "definition": {
            "unit": "manifest_sample_session",
            "positive": f"candidate_covisibility>={positive_threshold}",
            "strict_shortlist_no_match": (
                f"all_emitted_shortlist_covisibility<={negative_threshold}"
            ),
            "shortlist_ambiguous": (
                "neither_session_has_positive_nor_strict_shortlist_no_match"
            ),
            "recall_denominator": "sessions_with_positive_in_emitted_shortlist",
            "rank_source": "exact_dino_shortlist_candidate_rank_zero_based",
            "no_additional_inference": True,
        },
        "groups": group_summaries,
        "worst_scene_shortlist_conditional_positive_recall_at_k": worst_scene,
    }


def _validate_candidate_label_integrity(
    candidate: Mapping[str, Any],
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> None:
    score = _finite_float(candidate.get("covisibility"), "candidate covisibility")
    _require(0.0 <= score <= 1.0, "candidate covisibility is out of range")
    _require(
        candidate.get("label")
        == covisibility_label(
            score,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        ),
        "candidate label disagrees with co-visibility thresholds",
    )
    label_input = candidate.get("label_input")
    _require(
        isinstance(label_input, Mapping)
        and candidate.get("label_input_sha256")
        == sha256_bytes(canonical_json_bytes(label_input)),
        "candidate label-input fingerprint changed",
    )
    source = candidate.get("label_source")
    _require(label_input.get("source") == source, "candidate label authority changed")
    rendered = candidate.get("rendered_goal_depth")
    if source == "metadata_covis_curve":
        _require(
            rendered is None
            and "metadata_content_sha256" in label_input
            and "covis_curve_sha256" in label_input,
            "metadata label provenance is malformed",
        )
    else:
        _require(
            source == "rendered_goal_depth_reprojection"
            and isinstance(rendered, Mapping),
            "rendered label provenance is malformed",
        )
        render_request = {
            key: value
            for key, value in rendered.items()
            if key
            not in {
                "request_sha256",
                "depth_shape",
                "depth_dtype",
                "depth_content_sha256",
            }
        }
        _require(
            rendered.get("request_sha256")
            == sha256_bytes(canonical_json_bytes(render_request))
            and label_input.get("rendered_goal_depth_sha256")
            == rendered.get("depth_content_sha256")
            and isinstance(candidate.get("candidate_depth"), Mapping)
            and label_input.get("candidate_depth_content_sha256")
            == candidate["candidate_depth"].get("content_sha256"),
            "rendered label content/request pin changed",
        )


def _validate_cached_sample_record(
    *,
    record: Mapping[str, Any],
    sample: Mapping[str, Any],
    context: SampleContext,
    manifest_sha256: str,
    runtime_identity_sha256: str,
    dino_identity_sha256: str,
    config: TeacherConfig,
    score_rows: Sequence[Mapping[str, Any]],
    shortlist: Sequence[tuple[int, float]],
) -> None:
    unhashed = dict(record)
    declared_sha = _valid_sha(
        unhashed.pop("record_sha256", None), "progress sample record SHA"
    )
    _require(
        declared_sha == sha256_bytes(canonical_json_bytes(unhashed)),
        "progress sample record fingerprint changed",
    )
    for key in (
        "sample_id",
        "split_role",
        "scene",
        "source_episode",
        "goal_episode",
        "goal_role",
        "goal_variant",
        "state_name",
        "decision_frame",
    ):
        _require(
            record.get(key) == sample.get(key),
            f"progress sample field changed: {sample['sample_id']}.{key}",
        )
    decision = int(sample["decision_frame"])
    _require(
        record.get("manifest_sha256") == manifest_sha256
        and record.get("runtime_identity_sha256") == runtime_identity_sha256
        and record.get("causal_prefix_sha256")
        == sample["causal_prefix"]["causal_prefix_sha256"]
        and record.get("candidate_frame_domain")
        == {"start_inclusive": 0, "end_exclusive": decision}
        and record.get("no_future_source_observation") is True,
        f"progress sample causal provenance changed: {sample['sample_id']}",
    )
    query = record.get("query")
    _require(
        isinstance(query, Mapping)
        and query.get("path") == context.query_record["path"]
        and query.get("path_sha256") == context.query_record["path_sha256"]
        and query.get("bytes") == context.query_record["bytes"]
        and query.get("content_sha256") == context.query_record["content_sha256"]
        and query.get("goal_metadata_content_sha256")
        == context.goal_metadata_record["content_sha256"]
        and query.get("goal_pose_sha256") == context.goal_pose["goal_pose_sha256"],
        f"progress query content pin changed: {sample['sample_id']}",
    )
    selection = record.get("selection")
    expected_frames = [frame for frame, _score in shortlist]
    _require(
        isinstance(selection, Mapping)
        and selection.get("source") == "exact_lingbot_dino_cls"
        and selection.get("dino_provider_identity_sha256") == dino_identity_sha256
        and selection.get("universe_frame_count") == decision
        and selection.get("universe_scores_sha256")
        == sha256_bytes(canonical_json_bytes(list(score_rows)))
        and selection.get("top_k") == config.top_k
        and selection.get("temporal_nms_radius") == config.temporal_nms_radius
        and selection.get("selected_frame_indices") == expected_frames,
        f"progress DINO selection changed: {sample['sample_id']}",
    )
    candidates = record.get("candidates")
    _require(
        isinstance(candidates, list) and len(candidates) == len(shortlist),
        f"progress shortlist length changed: {sample['sample_id']}",
    )
    frame_by_index = {frame.frame: frame for frame in context.frames}
    for rank, (candidate, expected) in enumerate(_strict_zip(candidates, shortlist)):
        _require(isinstance(candidate, Mapping), "progress candidate is malformed")
        frame_index, dino_score = expected
        frame = frame_by_index[frame_index]
        _require(
            candidate.get("candidate_rank") == rank
            and candidate.get("candidate_frame") == frame_index
            and candidate.get("candidate_path") == frame.rgb_record["path"]
            and candidate.get("candidate_rgb") == frame.rgb_record
            and candidate.get("candidate_depth") == frame.depth_record
            and candidate.get("candidate_action_sha256")
            == sha256_bytes(canonical_json_bytes(frame.action.tolist()))
            and candidate.get("candidate_intrinsic_sha256")
            == sha256_bytes(canonical_json_bytes(frame.intrinsic.tolist()))
            and candidate.get("dino_cosine") == dino_score
            and candidate.get("no_future") is True
            and 0 <= frame_index < decision,
            f"progress candidate selection/content changed: {sample['sample_id']}",
        )
        source = candidate.get("label_source")
        if sample["goal_variant"] == "counterfactual":
            _require(
                source == "rendered_goal_depth_reprojection",
                "progress counterfactual used metadata label",
            )
        elif source == "metadata_covis_curve":
            _require(
                frame_index < len(context.goal_pose["covis_curve"])
                and candidate.get("covisibility")
                == context.goal_pose["covis_curve"][frame_index],
                "progress factual metadata label changed",
            )
        _validate_candidate_label_integrity(
            candidate,
            positive_threshold=config.positive_threshold,
            negative_threshold=config.negative_threshold,
        )
        _accumulate_label_count(
            candidate,
            label_source_counts={},
            label_counts={"positive": 0, "negative": 0, "ambiguous": 0},
        )


def build_teacher_artifact(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    embedding_provider: DINOEmbeddingProvider,
    renderer: GoalDepthRenderer,
    config: TeacherConfig = TeacherConfig(),
    root_overrides: Mapping[str, Path | str] | None = None,
    expected_sample_count: int | None = 600,
    progress_directory: Path | str | None = None,
    embedding_chunk_size: int = 256,
    expected_producer_sha256: str | None = None,
    expected_geometry_sha256: str | None = None,
) -> dict[str, Any]:
    _require(
        isinstance(embedding_chunk_size, int)
        and not isinstance(embedding_chunk_size, bool)
        and embedding_chunk_size >= 1,
        "embedding chunk size must be a positive integer",
    )
    manifest_sha = _valid_sha(manifest_sha256, "manifest SHA")
    _require(
        sha256_bytes(canonical_json_bytes(manifest)) == manifest_sha,
        "in-memory manifest differs from canonical SHA pin",
    )
    dino_identity = _validate_provider_identity(
        embedding_provider.identity, DINO_IDENTITY_SCHEMA, "DINO provider"
    )
    renderer_identity = _validate_provider_identity(
        renderer.identity, RENDERER_IDENTITY_SCHEMA, "goal renderer"
    )
    geometry_source = Path(backproject_world.__code__.co_filename).resolve()
    _require(
        geometry_source.is_file() and not geometry_source.is_symlink(),
        "co-visibility geometry implementation is unavailable",
    )
    geometry_authority = {
        "path": str(geometry_source),
        "bytes": geometry_source.stat().st_size,
        "content_sha256": sha256_file(geometry_source),
        "functions": [
            "backproject_world",
            "projected_covisibility",
            "covisibility_label",
        ],
    }
    producer_sha = sha256_file(Path(__file__))
    if expected_geometry_sha256 is not None:
        _require(
            _valid_sha(expected_geometry_sha256, "expected geometry source SHA")
            == geometry_authority["content_sha256"],
            "co-visibility geometry source differs from external pin",
        )
    if expected_producer_sha256 is not None:
        _require(
            _valid_sha(expected_producer_sha256, "expected producer source SHA")
            == producer_sha,
            "teacher producer source differs from external pin",
        )
    source_trust_anchors = {
        "producer_expected_sha256": expected_producer_sha256,
        "geometry_expected_sha256": expected_geometry_sha256,
        "externally_pinned": (
            expected_producer_sha256 is not None
            and expected_geometry_sha256 is not None
        ),
    }
    embedding_source_raw = getattr(embedding_provider, "bundle_binding", None)
    if embedding_source_raw is None:
        embedding_source = {
            "authority": "live_exact_provider",
            "signed_stage_a_bundle": False,
        }
    else:
        _require(
            isinstance(embedding_source_raw, Mapping),
            "signed embedding source binding is malformed",
        )
        embedding_source = copy.deepcopy(dict(embedding_source_raw))
        _require(
            embedding_source.get("schema_version") == EMBEDDING_BUNDLE_SCHEMA
            and embedding_source.get("manifest_sha256") == manifest_sha
            and embedding_source.get("authority")
            == "signed_embeddings_only_no_cache_miss_fallback",
            "signed embedding source is bound to the wrong manifest or authority",
        )
    embedding_invocation = {
        "chunk_size": embedding_chunk_size,
        "input_order": "manifest_relative_posix_path_ascending",
        "provider_output_cast": "float32",
        "cosine_accumulator": "float64",
        "provider_model_loads_per_process_run_max": (
            0 if embedding_source_raw is not None else 1
        ),
        "embedding_source": embedding_source,
    }
    runtime_payload = {
        "dino_provider_identity_sha256": dino_identity["identity_sha256"],
        "renderer_identity_sha256": renderer_identity["identity_sha256"],
        "geometry_authority": geometry_authority,
        "configuration": config.to_dict(),
        "embedding_invocation": embedding_invocation,
        "source_trust_anchors": source_trust_anchors,
    }
    runtime_identity_sha256 = sha256_bytes(canonical_json_bytes(runtime_payload))
    samples, scenes = _manifest_indexes(
        manifest, expected_sample_count=expected_sample_count
    )
    roots = _resolved_roots(manifest, root_overrides)
    contexts = _prepare_contexts(
        manifest=manifest,
        roots=roots,
        samples=samples,
        scenes=scenes,
    )
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    progress = None
    if progress_directory is not None:
        progress = _TeacherProgressStore(
            progress_directory,
            {
                "output_schema_version": SCHEMA_VERSION,
                "manifest_sha256": manifest_sha,
                "manifest_sample_id_sequence_sha256": sha256_bytes(
                    canonical_json_bytes(expected_ids)
                ),
                "runtime_identity_sha256": runtime_identity_sha256,
                "producer_content_sha256": producer_sha,
                "embedding_invocation": embedding_invocation,
                "source_trust_anchors": source_trust_anchors,
            },
        )

    unique_paths, dino_input_pins = _ordered_dino_inputs(contexts)
    embedding_chunks: list[np.ndarray | None] = []
    embedding_shard_names: list[str | None] = []
    embedding_descriptors: list[dict[str, Any]] = []
    embedding_path_chunks: list[list[Path]] = []
    for chunk_index, start in enumerate(
        range(0, len(unique_paths), embedding_chunk_size)
    ):
        chunk_paths = unique_paths[start : start + embedding_chunk_size]
        descriptor = {
            "chunk_index": chunk_index,
            "start": start,
            "end_exclusive": start + len(chunk_paths),
            "dino_provider_identity_sha256": dino_identity["identity_sha256"],
            "inputs": [
                {
                    "path": dino_input_pins[path]["path"],
                    "bytes": dino_input_pins[path]["bytes"],
                    "content_sha256": dino_input_pins[path]["content_sha256"],
                }
                for path in chunk_paths
            ],
        }
        embedding_descriptors.append(descriptor)
        embedding_path_chunks.append(chunk_paths)
        cached = (
            None
            if progress is None or embedding_source_raw is not None
            else progress.load_embedding(chunk_index=chunk_index, descriptor=descriptor)
        )
        if cached is not None:
            shard_name, chunk_embeddings = cached
            embedding_shard_names.append(shard_name)
            embedding_chunks.append(chunk_embeddings)
        else:
            embedding_shard_names.append(None)
            embedding_chunks.append(None)

    # The legacy exact loader constructs and destroys DINO on every provider
    # call.  Aggregate all cache misses into one call, then split the result
    # into content-addressed progress shards.  This keeps model loads <= 1/run.
    missing_indices = [
        index for index, value in enumerate(embedding_chunks) if value is None
    ]
    if missing_indices:
        missing_paths = [
            path for index in missing_indices for path in embedding_path_chunks[index]
        ]
        missing_embeddings = np.asarray(
            embedding_provider.embed(missing_paths), dtype=np.float32
        )
        _require(
            missing_embeddings.ndim == 2
            and missing_embeddings.shape[0] == len(missing_paths)
            and missing_embeddings.shape[1] >= 1
            and np.isfinite(missing_embeddings).all()
            and np.all(np.linalg.norm(missing_embeddings, axis=1) > 0.0),
            "DINO provider returned malformed cache-miss embeddings",
        )
        offset = 0
        for index in missing_indices:
            count = len(embedding_path_chunks[index])
            chunk_embeddings = missing_embeddings[offset : offset + count].copy()
            offset += count
            embedding_chunks[index] = chunk_embeddings
            if progress is not None and embedding_source_raw is None:
                embedding_shard_names[index] = progress.save_embedding(
                    chunk_index=index,
                    descriptor=embedding_descriptors[index],
                    embeddings=chunk_embeddings,
                )
        _require(offset == len(missing_paths), "DINO cache-miss split changed")
    _require(
        all(value is not None for value in embedding_chunks),
        "DINO embedding chunk cover is incomplete",
    )
    embeddings = np.concatenate(
        [value for value in embedding_chunks if value is not None], axis=0
    )
    _require(
        embeddings.ndim == 2
        and embeddings.shape[0] == len(unique_paths)
        and embeddings.shape[1] >= 1
        and np.isfinite(embeddings).all(),
        "DINO provider returned malformed embeddings",
    )
    _require(
        np.all(np.linalg.norm(embeddings, axis=1) > 0.0),
        "DINO provider returned zero embeddings",
    )
    for path, record in dino_input_pins.items():
        _verify_dino_input_content(
            path, record, label="DINO input during teacher assembly"
        )
    embedding_by_path = {
        path: embeddings[index] for index, path in enumerate(unique_paths)
    }

    record_slots: list[Mapping[str, Any] | None] = [None] * len(contexts)
    render_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    environment_verification_cache: dict[Path, tuple[str, int, int, int]] = {}
    label_source_counts: dict[str, int] = {}
    label_counts = {"positive": 0, "negative": 0, "ambiguous": 0}
    work_items = _scene_grouped_work_items(contexts)
    for sample_index, context in work_items:
        sample = context.sample
        query_embedding = embedding_by_path[context.query_path.resolve()]
        candidate_embeddings = np.stack(
            [embedding_by_path[frame.rgb_path.resolve()] for frame in context.frames]
        )
        scores = _cosine_scores(query_embedding, candidate_embeddings)
        score_rows = [
            {"frame": frame.frame, "dino_cosine": float(score)}
            for frame, score in _strict_zip(context.frames, scores)
        ]
        shortlist = temporal_nms_shortlist(
            [frame.frame for frame in context.frames],
            scores.tolist(),
            top_k=config.top_k,
            radius=config.temporal_nms_radius,
        )
        cached_record = (
            None
            if progress is None
            else progress.load_sample(
                sample_index=sample_index,
                sample_id=str(sample["sample_id"]),
            )
        )
        if cached_record is not None:
            _validate_cached_sample_record(
                record=cached_record,
                sample=sample,
                context=context,
                manifest_sha256=manifest_sha,
                runtime_identity_sha256=runtime_identity_sha256,
                dino_identity_sha256=str(dino_identity["identity_sha256"]),
                config=config,
                score_rows=score_rows,
                shortlist=shortlist,
            )
            record_slots[sample_index] = cached_record
            for candidate in cached_record["candidates"]:
                _accumulate_label_count(
                    candidate,
                    label_source_counts=label_source_counts,
                    label_counts=label_counts,
                )
            continue
        frame_by_index = {frame.frame: frame for frame in context.frames}
        candidate_rows = []
        decision = int(sample["decision_frame"])
        for rank, (frame_index, dino_score) in enumerate(shortlist):
            candidate = frame_by_index[frame_index]
            _require(
                candidate.frame < decision,
                f"future candidate leaked for {sample['sample_id']}",
            )
            authority = _candidate_label(
                context=context,
                candidate=candidate,
                renderer=renderer,
                renderer_identity_sha256=renderer_identity["identity_sha256"],
                config=config,
                render_cache=render_cache,
                environment_verification_cache=environment_verification_cache,
            )
            candidate_rows.append(
                {
                    "candidate_rank": rank,
                    "candidate_frame": candidate.frame,
                    "candidate_path": candidate.rgb_record["path"],
                    "candidate_rgb": copy.deepcopy(dict(candidate.rgb_record)),
                    "candidate_depth": copy.deepcopy(dict(candidate.depth_record)),
                    "candidate_action_sha256": sha256_bytes(
                        canonical_json_bytes(candidate.action.tolist())
                    ),
                    "candidate_intrinsic_sha256": sha256_bytes(
                        canonical_json_bytes(candidate.intrinsic.tolist())
                    ),
                    "dino_cosine": float(dino_score),
                    **authority,
                    "no_future": True,
                }
            )
            _accumulate_label_count(
                candidate_rows[-1],
                label_source_counts=label_source_counts,
                label_counts=label_counts,
            )
        _require(
            bool(candidate_rows), f"sample shortlist is empty: {sample['sample_id']}"
        )
        query = {
            "path": context.query_record["path"],
            "path_sha256": context.query_record["path_sha256"],
            "bytes": context.query_record["bytes"],
            "content_sha256": context.query_record["content_sha256"],
            "goal_metadata_content_sha256": context.goal_metadata_record[
                "content_sha256"
            ],
            "goal_index": context.goal_index,
            "goal_pose_sha256": context.goal_pose["goal_pose_sha256"],
            "camera_height_m": context.camera_height_m,
            "goal_intrinsic_sha256": sha256_bytes(
                canonical_json_bytes(context.goal_intrinsic.tolist())
            ),
        }
        record = {
            "sample_id": sample["sample_id"],
            "split_role": sample["split_role"],
            "scene": sample["scene"],
            "source_episode": sample["source_episode"],
            "goal_episode": sample["goal_episode"],
            "goal_role": sample["goal_role"],
            "goal_variant": sample["goal_variant"],
            "state_name": sample["state_name"],
            "decision_frame": decision,
            "manifest_sha256": manifest_sha,
            "runtime_identity_sha256": runtime_identity_sha256,
            "causal_prefix_sha256": sample["causal_prefix"]["causal_prefix_sha256"],
            "candidate_frame_domain": {
                "start_inclusive": 0,
                "end_exclusive": decision,
            },
            "query": query,
            "environment_content_sha256": context.environment_record["content_sha256"],
            "selection": {
                "source": "exact_lingbot_dino_cls",
                "dino_provider_identity_sha256": dino_identity["identity_sha256"],
                "universe_frame_count": decision,
                "universe_scores_sha256": sha256_bytes(
                    canonical_json_bytes(score_rows)
                ),
                "top_k": config.top_k,
                "temporal_nms_radius": config.temporal_nms_radius,
                "selected_frame_indices": [
                    row["candidate_frame"] for row in candidate_rows
                ],
            },
            "candidates": candidate_rows,
            "no_future_source_observation": True,
        }
        record["record_sha256"] = sha256_bytes(canonical_json_bytes(record))
        if progress is not None:
            progress.save_sample(
                sample_index=sample_index,
                sample_id=str(sample["sample_id"]),
                record=record,
            )
        record_slots[sample_index] = record

    _require(
        all(record is not None for record in record_slots),
        "teacher scene-grouped work did not fill every manifest row",
    )
    records = [record for record in record_slots if record is not None]

    actual_ids = [str(record["sample_id"]) for record in records]
    _require(
        actual_ids == expected_ids and len(set(actual_ids)) == len(actual_ids),
        "teacher sample cover is not exact",
    )
    config_payload = config.to_dict()
    rendered_request_ids = {
        str(candidate["rendered_goal_depth"]["request_sha256"])
        for record in records
        for candidate in record["candidates"]
        if candidate["rendered_goal_depth"] is not None
    }
    candidate_oracle = _candidate_oracle_summary(
        records,
        positive_threshold=config.positive_threshold,
        negative_threshold=config.negative_threshold,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "manifest": {
            "schema_version": MANIFEST_SCHEMA,
            "sha256": manifest_sha,
            "sample_count": len(samples),
            "sample_id_sequence_sha256": sha256_bytes(
                canonical_json_bytes(expected_ids)
            ),
        },
        "configuration": config_payload,
        "configuration_sha256": sha256_bytes(canonical_json_bytes(config_payload)),
        "dino_provider": dino_identity,
        "goal_depth_renderer": renderer_identity,
        "geometry_authority": geometry_authority,
        "runtime_identity_sha256": runtime_identity_sha256,
        "embedding_invocation": embedding_invocation,
        "source_trust_anchors": source_trust_anchors,
        "producer": {
            "path": Path(__file__).name,
            "content_sha256": producer_sha,
        },
        "path_resolution": {
            "schema_version": "manifest_teacher_absolute_path_resolution_v1",
            "episode_root": str(roots["episode_root"].resolve()),
            "relative_path_authority": "manifest_content_pinned_episode_relative_path",
            "csv_path_form": "resolved_absolute_physical_path",
        },
        "exact_cover": {
            "manifest_sample_count": len(expected_ids),
            "output_sample_count": len(actual_ids),
            "sample_ids_equal_in_manifest_order": True,
            "sample_id_sequence_sha256": sha256_bytes(canonical_json_bytes(actual_ids)),
        },
        "summary": {
            "samples": len(records),
            "candidates": sum(len(record["candidates"]) for record in records),
            "label_sources": dict(sorted(label_source_counts.items())),
            "labels": label_counts,
            "rendered_goal_cache_entries": len(rendered_request_ids),
            "candidate_oracle": candidate_oracle,
        },
        "records": records,
        "deployment_approved": False,
    }
    if progress is not None:
        if embedding_source_raw is None:
            _require(
                all(isinstance(name, str) for name in embedding_shard_names),
                "progress DINO shard names are incomplete",
            )
        else:
            _require(
                all(name is None for name in embedding_shard_names),
                "signed Stage-A embeddings leaked into Stage-B progress",
            )
        progress.verify_complete(
            embedding_names=[
                name for name in embedding_shard_names if isinstance(name, str)
            ],
            sample_ids=expected_ids,
        )
    canonical_json_bytes(artifact)
    return artifact


def teacher_csv_bytes(artifact: Mapping[str, Any]) -> bytes:
    """Flatten a teacher artifact into a deterministic, canonical CSV view."""

    _require(artifact.get("schema_version") == SCHEMA_VERSION, "teacher schema changed")
    manifest = artifact.get("manifest")
    _require(isinstance(manifest, Mapping), "teacher manifest pin is absent")
    manifest_sha = _valid_sha(manifest.get("sha256"), "teacher manifest SHA")
    runtime_sha = _valid_sha(
        artifact.get("runtime_identity_sha256"), "teacher runtime identity SHA"
    )
    path_resolution = artifact.get("path_resolution")
    _require(
        isinstance(path_resolution, Mapping)
        and path_resolution.get("schema_version")
        == "manifest_teacher_absolute_path_resolution_v1"
        and path_resolution.get("relative_path_authority")
        == "manifest_content_pinned_episode_relative_path"
        and path_resolution.get("csv_path_form") == "resolved_absolute_physical_path",
        "teacher path-resolution contract changed",
    )
    episode_root = Path(str(path_resolution.get("episode_root", "")))
    _require(
        episode_root.is_absolute()
        and episode_root.is_dir()
        and not episode_root.is_symlink(),
        "teacher episode root is unavailable",
    )
    episode_root = episode_root.resolve()
    records = artifact.get("records")
    _require(isinstance(records, list), "teacher records are malformed")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        _require(isinstance(record, Mapping), "teacher record is malformed")
        _require(
            record.get("manifest_sha256") == manifest_sha, "record manifest pin changed"
        )
        _require(
            record.get("runtime_identity_sha256") == runtime_sha,
            "record runtime pin changed",
        )
        query = record.get("query")
        candidates = record.get("candidates")
        _require(
            isinstance(query, Mapping) and isinstance(candidates, list),
            "teacher record query/candidates are malformed",
        )
        decision = record.get("decision_frame")
        _require(
            isinstance(decision, int)
            and not isinstance(decision, bool)
            and decision >= 1,
            "teacher decision frame is malformed",
        )
        for candidate in candidates:
            _require(isinstance(candidate, Mapping), "teacher candidate is malformed")
            frame = candidate.get("candidate_frame")
            _require(
                isinstance(frame, int)
                and not isinstance(frame, bool)
                and 0 <= frame < decision
                and candidate.get("no_future") is True,
                "teacher CSV detected a future candidate",
            )
            candidate_rgb = candidate.get("candidate_rgb")
            candidate_depth = candidate.get("candidate_depth")
            _require(
                isinstance(candidate_rgb, Mapping)
                and isinstance(candidate_depth, Mapping),
                "candidate content pins are malformed",
            )
            dino_score = _finite_float(candidate.get("dino_cosine"), "DINO score")
            covisibility = _finite_float(
                candidate.get("covisibility"), "candidate covisibility"
            )
            sample_id = record.get("sample_id")
            query_relative = _relative_path(query.get("path"), "teacher query")
            candidate_relative = _relative_path(
                candidate.get("candidate_path"), "teacher candidate"
            )
            _require(
                candidate_rgb.get("path") == candidate_relative,
                "candidate relative path differs from its content pin",
            )
            query_unresolved = episode_root / query_relative
            candidate_unresolved = episode_root / candidate_relative
            for label, unresolved in (
                ("query", query_unresolved),
                ("candidate", candidate_unresolved),
            ):
                _require(
                    not unresolved.is_symlink(),
                    f"teacher {label} path became a symlink",
                )
            query_path = query_unresolved.resolve()
            candidate_path = candidate_unresolved.resolve()
            for label, path, relative, content in (
                ("query", query_path, query_relative, query),
                (
                    "candidate",
                    candidate_path,
                    candidate_relative,
                    candidate_rgb,
                ),
            ):
                try:
                    path.relative_to(episode_root)
                except ValueError as error:
                    raise CausalTeacherError(
                        f"teacher {label} path escapes episode root"
                    ) from error
                _require(
                    path.is_file()
                    and not path.is_symlink()
                    and path.stat().st_size == content.get("bytes")
                    and sha256_file(path) == content.get("content_sha256")
                    and content.get("path") == relative,
                    f"teacher {label} physical content changed",
                )
            dino_csv = json.dumps(
                0.0 if dino_score == 0.0 else dino_score,
                allow_nan=False,
                separators=(",", ":"),
            )
            covis_csv = json.dumps(
                0.0 if covisibility == 0.0 else covisibility,
                allow_nan=False,
                separators=(",", ":"),
            )
            row = {
                "session_id": sample_id,
                "sample_id": sample_id,
                "causal_manifest_sample_id": sample_id,
                "split_role": record.get("split_role"),
                "scene": record.get("scene"),
                "episode": record.get("source_episode"),
                "source_episode": record.get("source_episode"),
                "goal_episode": record.get("goal_episode"),
                "kind": _phase_b_kind(record.get("split_role")),
                "goal_role": record.get("goal_role"),
                "goal_variant": record.get("goal_variant"),
                "decision_frame": decision,
                "query_path": str(query_path),
                "query_relative_path": query_relative,
                "candidate_rank": candidate.get("candidate_rank"),
                "candidate_frame": frame,
                "candidate_path": str(candidate_path),
                "candidate_relative_path": candidate_relative,
                "dino_cosine": dino_csv,
                "label_source": candidate.get("label_source"),
                "teacher_covis": covis_csv,
                "covisibility": covis_csv,
                "label": candidate.get("label"),
                "query_content_sha256": query.get("content_sha256"),
                "candidate_rgb_content_sha256": candidate_rgb.get("content_sha256"),
                "candidate_depth_content_sha256": candidate_depth.get("content_sha256"),
                "manifest_sha256": manifest_sha,
                "runtime_identity_sha256": runtime_sha,
                "no_future": "true",
            }
            _require(
                all(value is not None for value in row.values()),
                "teacher CSV row contains a missing field",
            )
            writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def build_audit_receipt(
    artifact: Mapping[str, Any],
    *,
    artifact_bytes: bytes,
    csv_bytes: bytes,
) -> dict[str, Any]:
    """Create a fail-closed receipt for authority, coverage, and causality."""

    _require(
        artifact_bytes == canonical_json_bytes(artifact),
        "audit artifact bytes are not canonical",
    )
    _require(
        csv_bytes == teacher_csv_bytes(artifact),
        "audit CSV bytes are not the canonical artifact-derived view",
    )
    try:
        csv_text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CausalTeacherError("audit CSV is not UTF-8") from error
    csv_reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    _require(
        tuple(csv_reader.fieldnames or ()) == CSV_FIELDS,
        "audit CSV columns or order changed",
    )
    csv_rows = list(csv_reader)
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("status") == "complete"
        and artifact.get("deployment_approved") is False,
        "audit teacher status/schema changed",
    )
    manifest = artifact.get("manifest")
    exact_cover = artifact.get("exact_cover")
    records = artifact.get("records")
    configuration = artifact.get("configuration")
    geometry_authority = artifact.get("geometry_authority")
    embedding_invocation = artifact.get("embedding_invocation")
    source_trust_anchors = artifact.get("source_trust_anchors")
    _require(
        isinstance(manifest, Mapping)
        and isinstance(exact_cover, Mapping)
        and isinstance(records, list),
        "audit teacher structure is malformed",
    )
    _require(
        isinstance(configuration, Mapping)
        and artifact.get("configuration_sha256")
        == sha256_bytes(canonical_json_bytes(configuration)),
        "audit configuration fingerprint changed",
    )
    _require(
        isinstance(embedding_invocation, Mapping),
        "audit embedding invocation pin is absent",
    )
    _require(
        isinstance(source_trust_anchors, Mapping),
        "audit source trust anchors are absent",
    )
    dino_identity = _validate_provider_identity(
        artifact.get("dino_provider"), DINO_IDENTITY_SCHEMA, "audit DINO provider"
    )
    renderer_identity = _validate_provider_identity(
        artifact.get("goal_depth_renderer"),
        RENDERER_IDENTITY_SCHEMA,
        "audit goal renderer",
    )
    _require(isinstance(geometry_authority, Mapping), "audit geometry pin is absent")
    geometry_path = Path(str(geometry_authority.get("path", "")))
    _require(
        geometry_path.is_file()
        and not geometry_path.is_symlink()
        and geometry_path.stat().st_size == geometry_authority.get("bytes")
        and sha256_file(geometry_path) == geometry_authority.get("content_sha256"),
        "audit geometry implementation content changed",
    )
    producer = artifact.get("producer")
    _require(
        isinstance(producer, Mapping)
        and producer.get("path") == Path(__file__).name
        and producer.get("content_sha256") == sha256_file(Path(__file__)),
        "audit producer implementation content changed",
    )
    manifest_sha = _valid_sha(manifest.get("sha256"), "audit manifest SHA")
    runtime_sha = _valid_sha(
        artifact.get("runtime_identity_sha256"), "audit runtime identity SHA"
    )
    _require(
        runtime_sha
        == sha256_bytes(
            canonical_json_bytes(
                {
                    "dino_provider_identity_sha256": dino_identity["identity_sha256"],
                    "renderer_identity_sha256": renderer_identity["identity_sha256"],
                    "geometry_authority": geometry_authority,
                    "configuration": configuration,
                    "embedding_invocation": embedding_invocation,
                    "source_trust_anchors": source_trust_anchors,
                }
            )
        ),
        "audit runtime fingerprint changed",
    )
    sample_ids: list[str] = []
    candidate_count = 0
    label_source_counts: dict[str, int] = {}
    audited_label_counts = {"positive": 0, "negative": 0, "ambiguous": 0}
    counterfactual_count = 0
    counterfactual_rendered_count = 0
    for record in records:
        _require(isinstance(record, Mapping), "audit record is malformed")
        unhashed_record = dict(record)
        declared_record_sha = _valid_sha(
            unhashed_record.pop("record_sha256", None), "teacher record SHA"
        )
        _require(
            declared_record_sha == sha256_bytes(canonical_json_bytes(unhashed_record)),
            "teacher record fingerprint changed",
        )
        sample_id = record.get("sample_id")
        decision = record.get("decision_frame")
        _require(
            isinstance(sample_id, str)
            and bool(sample_id)
            and isinstance(decision, int)
            and not isinstance(decision, bool)
            and decision >= 1,
            "audit record identity is malformed",
        )
        _require(
            record.get("manifest_sha256") == manifest_sha
            and record.get("runtime_identity_sha256") == runtime_sha
            and record.get("no_future_source_observation") is True,
            "audit record provenance/causality pin changed",
        )
        sample_ids.append(sample_id)
        candidates = record.get("candidates")
        _require(
            isinstance(candidates, list) and bool(candidates),
            "audit shortlist is empty",
        )
        is_counterfactual = record.get("goal_variant") == "counterfactual"
        for candidate in candidates:
            _require(isinstance(candidate, Mapping), "audit candidate is malformed")
            frame = candidate.get("candidate_frame")
            source = candidate.get("label_source")
            _require(
                isinstance(frame, int)
                and not isinstance(frame, bool)
                and 0 <= frame < decision
                and candidate.get("no_future") is True,
                f"audit found future candidate in {sample_id}",
            )
            _require(
                source
                in {
                    "metadata_covis_curve",
                    "rendered_goal_depth_reprojection",
                },
                "audit found a non-geometric label authority",
            )
            _require(
                candidate.get("label") in (-1, 0, 1)
                and 0.0
                <= _finite_float(candidate.get("covisibility"), "audit covisibility")
                <= 1.0,
                "audit label is malformed",
            )
            _validate_candidate_label_integrity(
                candidate,
                positive_threshold=_finite_float(
                    configuration.get("positive_threshold"),
                    "audit positive threshold",
                ),
                negative_threshold=_finite_float(
                    configuration.get("negative_threshold"),
                    "audit negative threshold",
                ),
            )
            label_source_counts[str(source)] = (
                label_source_counts.get(str(source), 0) + 1
            )
            audited_label_counts[
                {1: "positive", 0: "negative", -1: "ambiguous"}[int(candidate["label"])]
            ] += 1
            candidate_count += 1
            if is_counterfactual:
                counterfactual_count += 1
                if source == "rendered_goal_depth_reprojection":
                    counterfactual_rendered_count += 1
    _require(len(sample_ids) == len(set(sample_ids)), "audit sample IDs are duplicated")
    _require(
        exact_cover.get("manifest_sample_count") == len(sample_ids)
        and exact_cover.get("output_sample_count") == len(sample_ids)
        and exact_cover.get("sample_ids_equal_in_manifest_order") is True
        and exact_cover.get("sample_id_sequence_sha256")
        == sha256_bytes(canonical_json_bytes(sample_ids)),
        "audit exact sample cover failed",
    )
    _require(
        counterfactual_count == counterfactual_rendered_count,
        "counterfactual candidate consumed episode-local metadata",
    )
    _require(
        len(csv_rows) == candidate_count,
        "Phase-B CSV candidate cover differs from the teacher artifact",
    )
    csv_index = 0
    csv_session_sequence = []
    for record in records:
        sample_id = str(record["sample_id"])
        query = record["query"]
        for candidate in record["candidates"]:
            row = csv_rows[csv_index]
            csv_index += 1
            csv_session_sequence.append(str(row["session_id"]))
            _require(
                row["session_id"]
                == row["sample_id"]
                == row["causal_manifest_sample_id"]
                == sample_id
                and row["scene"] == str(record["scene"])
                and row["episode"]
                == row["source_episode"]
                == str(record["source_episode"])
                and row["kind"] == _phase_b_kind(record.get("split_role"))
                and row["query_relative_path"] == str(query["path"])
                and row["candidate_relative_path"] == str(candidate["candidate_path"])
                and Path(row["query_path"]).is_absolute()
                and Path(row["candidate_path"]).is_absolute()
                and int(row["candidate_frame"]) == int(candidate["candidate_frame"])
                and float(row["dino_cosine"]) == float(candidate["dino_cosine"])
                and float(row["teacher_covis"])
                == float(row["covisibility"])
                == float(candidate["covisibility"])
                and int(row["label"]) == int(candidate["label"])
                and row["query_content_sha256"] == str(query["content_sha256"])
                and row["candidate_rgb_content_sha256"]
                == str(candidate["candidate_rgb"]["content_sha256"])
                and row["manifest_sha256"] == manifest_sha
                and row["runtime_identity_sha256"] == runtime_sha
                and row["no_future"] == "true",
                f"Phase-B CSV row drifted from artifact candidate {sample_id}",
            )
    _require(csv_index == len(csv_rows), "Phase-B CSV row cursor changed")
    _require(
        set(csv_session_sequence) == set(sample_ids),
        "Phase-B CSV session cover differs from manifest sample cover",
    )
    summary = artifact.get("summary")
    audited_candidate_oracle = _candidate_oracle_summary(
        records,
        positive_threshold=_finite_float(
            configuration.get("positive_threshold"),
            "audit positive threshold",
        ),
        negative_threshold=_finite_float(
            configuration.get("negative_threshold"),
            "audit negative threshold",
        ),
    )
    _require(
        isinstance(summary, Mapping)
        and summary.get("samples") == len(sample_ids)
        and summary.get("candidates") == candidate_count
        and summary.get("label_sources") == dict(sorted(label_source_counts.items())),
        "teacher summary does not match audited records",
    )
    _require(
        summary.get("labels") == audited_label_counts
        and summary.get("candidate_oracle") == audited_candidate_oracle,
        "teacher summary does not match audited records",
    )
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "status": "audited_not_deployment_approved",
        "artifact": {
            "name": ARTIFACT_NAME,
            "bytes": len(artifact_bytes),
            "content_sha256": sha256_bytes(artifact_bytes),
            "schema_version": SCHEMA_VERSION,
        },
        "csv": {
            "name": CSV_NAME,
            "bytes": len(csv_bytes),
            "content_sha256": sha256_bytes(csv_bytes),
            "columns": list(CSV_FIELDS),
            "candidate_rows": candidate_count,
            "phase_b_contract": {
                "kind": f"{PHASE_B_TEACHER_KIND}_{{split_role}}",
                "allowed_kinds": [
                    _phase_b_kind("train"),
                    _phase_b_kind("development"),
                ],
                "session_id": "sample_id",
                "causal_manifest_sample_id": "sample_id",
                "episode": "source_episode",
                "teacher_covis": "covisibility",
                "paths": "resolved_absolute_physical_paths_with_relative_and_content_pins",
                "exact_candidate_cover": True,
            },
        },
        "manifest": {
            "schema_version": MANIFEST_SCHEMA,
            "content_sha256": manifest_sha,
            "sample_count": len(sample_ids),
            "sample_id_sequence_sha256": sha256_bytes(canonical_json_bytes(sample_ids)),
        },
        "runtime_identity_sha256": runtime_sha,
        "embedding_source": copy.deepcopy(
            dict(embedding_invocation.get("embedding_source", {}))
        ),
        "configuration_sha256": artifact.get("configuration_sha256"),
        "producer_content_sha256": artifact.get("producer", {}).get("content_sha256"),
        "authority": {
            "candidate_selection": "exact_lingbot_dino_cls_rank_plus_temporal_nms_only",
            "labels": [
                "metadata_covis_curve",
                "rendered_goal_depth_reprojection",
            ],
            "dino_or_ransac_self_report_used_as_label": False,
            "counterfactual_metadata_curve_allowed": False,
        },
        "invariants": {
            "exact_manifest_sample_cover": True,
            "all_candidate_frames_strictly_before_decision": True,
            "counterfactual_all_use_rendered_geometry": True,
            "goal_b_and_goal_c_present": {
                str(record.get("goal_role")) for record in records
            }
            == {"B", "C"},
            "factual_and_counterfactual_present": {
                str(record.get("goal_variant")) for record in records
            }
            == {"factual", "counterfactual"},
            "phase_b_csv_exact_candidate_cover": True,
            "phase_b_csv_exact_manifest_session_cover": True,
        },
        "counts": {
            "samples": len(sample_ids),
            "candidates": candidate_count,
            "counterfactual_candidates": counterfactual_count,
            "label_sources": dict(sorted(label_source_counts.items())),
        },
        "candidate_oracle": audited_candidate_oracle,
        "deployment_approved": False,
    }
    canonical_json_bytes(receipt)
    return receipt


def _sidecar_bytes(name: str, payload: bytes) -> bytes:
    return f"{sha256_bytes(payload)}  {name}\n".encode("ascii")


def teacher_bundle_payloads(artifact: Mapping[str, Any]) -> dict[str, bytes]:
    artifact_bytes = canonical_json_bytes(artifact)
    csv_bytes = teacher_csv_bytes(artifact)
    audit = build_audit_receipt(
        artifact,
        artifact_bytes=artifact_bytes,
        csv_bytes=csv_bytes,
    )
    audit_bytes = canonical_json_bytes(audit)
    payloads = {
        ARTIFACT_NAME: artifact_bytes,
        f"{ARTIFACT_NAME}.sha256": _sidecar_bytes(ARTIFACT_NAME, artifact_bytes),
        CSV_NAME: csv_bytes,
        f"{CSV_NAME}.sha256": _sidecar_bytes(CSV_NAME, csv_bytes),
        AUDIT_NAME: audit_bytes,
        f"{AUDIT_NAME}.sha256": _sidecar_bytes(AUDIT_NAME, audit_bytes),
    }
    _require(frozenset(payloads) == BUNDLE_FILES, "teacher bundle contract changed")
    return payloads


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_teacher_bundle(
    artifact: Mapping[str, Any],
    output_directory: Path | str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Publish all six files atomically or verify byte-identical resume.

    Resume never trusts an existing receipt.  It rebuilds every expected byte
    from the newly verified inputs and accepts the directory only if the file
    set and all contents match exactly.  Partial and drifted directories fail
    closed.
    """

    output = Path(output_directory)
    _require(output.name not in ("", ".", ".."), "output directory is invalid")
    payloads = teacher_bundle_payloads(artifact)
    if resume:
        _require(
            output.is_dir() and not output.is_symlink(),
            "resume requires an existing physical bundle directory",
        )
        actual_names = frozenset(path.name for path in output.iterdir())
        _require(actual_names == BUNDLE_FILES, "resume bundle file set changed")
        for name, expected in payloads.items():
            path = output / name
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == expected,
                f"resume bundle content drifted: {name}",
            )
        mode = "resumed"
    else:
        _require(
            not output.exists() and not output.is_symlink(),
            "output bundle already exists",
        )
        parent = output.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=parent)
        )
        try:
            for name in sorted(payloads):
                path = stage / name
                with path.open("xb") as handle:
                    handle.write(payloads[name])
                    handle.flush()
                    os.fsync(handle.fileno())
            _fsync_directory(stage)
            _require(not output.exists(), "output bundle appeared during publication")
            os.rename(stage, output)
            _fsync_directory(parent)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        mode = "written"
    return {
        "status": mode,
        "output_directory": str(output.resolve()),
        "artifact_sha256": sha256_bytes(payloads[ARTIFACT_NAME]),
        "csv_sha256": sha256_bytes(payloads[CSV_NAME]),
        "audit_sha256": sha256_bytes(payloads[AUDIT_NAME]),
        "sample_count": artifact["summary"]["samples"],
        "candidate_count": artifact["summary"]["candidates"],
        "deployment_approved": False,
    }


def main() -> None:
    raise CausalTeacherError(
        "the combined DINO+Habitat CLI is disabled; run "
        "build_manifest_causal_dino_embeddings.py in the MemNav/Torch "
        "environment, then assemble_manifest_causal_covisibility_teacher.py "
        "in the Habitat environment"
    )


if __name__ == "__main__":
    main()
