#!/usr/bin/env python3
"""Zero-training feasibility test for LingBot-native goal loop closure.

The current geometry router uses DINO for coarse retrieval and SIFT/RANSAC for
candidate verification.  This diagnostic asks whether LingBot's *own* streaming
geometry can provide the verification signal instead:

1. Select scene/session-balanced positive and hard-negative candidate anchors
   from an existing task-aligned co-visibility teacher CSV.
2. Append the same goal image after the candidate and nearby temporal anchors.
3. Measure whether the independently inferred goal poses agree in the common
   LingBot map frame (pose consensus).
4. Predict depth for both the anchor and appended goal, transform the two point
   clouds into that map frame, and measure their symmetric 3-D overlap.

No model weights are changed.  Source data, feature caches, and checkpoints are
read-only; only a CSV and JSON report are written below ``--out-dir``.

This is deliberately a small feasibility diagnostic, not a deployment router.
Thresholds must not be chosen from final-reserved scenes.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from MemNavData.external_causal_scale_contract import (
        CAUSAL_SAMPLE_ID_COLUMN,
        EXTERNAL_CAUSAL_SCALE_SOURCE,
        ExternalCausalScaleBinding,
        ExternalCausalScaleContract,
        ExternalCausalScalePins,
        sha256_file as contract_sha256_file,
    )
    from MemNavData.flow_cache_routing import (
        FlowRoutingError,
        registry_from_manifest,
    )
    from MemNavData.phase_b_upstream_receipts import (
        PhaseBUpstreamPins,
        validate_phase_b_upstream_receipts,
    )
except ModuleNotFoundError:  # direct script invocation
    from external_causal_scale_contract import (  # type: ignore
        CAUSAL_SAMPLE_ID_COLUMN,
        EXTERNAL_CAUSAL_SCALE_SOURCE,
        ExternalCausalScaleBinding,
        ExternalCausalScaleContract,
        ExternalCausalScalePins,
        sha256_file as contract_sha256_file,
    )
    from flow_cache_routing import (  # type: ignore
        FlowRoutingError,
        registry_from_manifest,
    )
    from phase_b_upstream_receipts import (  # type: ignore
        PhaseBUpstreamPins,
        validate_phase_b_upstream_receipts,
    )


REQUIRED_COLUMNS = {
    "session_id",
    "scene",
    "episode",
    "kind",
    "query_path",
    "candidate_path",
    "candidate_frame",
    "dino_cosine",
    "teacher_covis",
}


@dataclass(frozen=True)
class CandidateSeed:
    session_id: str
    scene: str
    episode: str
    kind: str
    query_path: Path
    candidate_path: Path
    candidate_frame: int
    dino_cosine: float
    teacher_covis: float
    label: int
    session_has_positive: bool
    session_is_strict_no_match: bool
    session_max_covis: float
    causal_manifest_sample_id: Optional[str] = None
    selection_origin: str = "deployment_topk"


@dataclass(frozen=True)
class EpisodePoseData:
    """Ground-truth camera trajectory and generator metadata for one episode."""

    actions: np.ndarray
    base_extrinsic: np.ndarray
    metadata: dict


_HABITAT_TO_DATA_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)
_LINGBOT_TO_DATA_ROTATION_BASIS = np.diag([-1.0, -1.0, 1.0])
_DEFAULT_POOLED_METRIC_SCALE = 2.564
_CHECKPOINT_SCHEMA_VERSION = 3
_CHECKPOINT_FILENAME = "lingbot_goal_loop_closure_checkpoint.sqlite3"
_PROGRESS_FILENAME = "lingbot_goal_loop_closure_progress.json"
_ROWS_FILENAME = "lingbot_goal_loop_closure_rows.csv"
_REPORT_FILENAME = "diagnostic_lingbot_goal_loop_closure.json"


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_write_json(path: Path, value: object) -> None:
    """Durably replace one small JSON artifact in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Durably replace a CSV after it has been completely serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class BoundedEpisodeCache:
    """Small explicit LRU whose evictions can release accelerator state."""

    def __init__(self, capacity: int, on_evict=None) -> None:
        if capacity < 1:
            raise ValueError("episode cache capacity must be positive")
        self.capacity = int(capacity)
        self.on_evict = on_evict
        self._values: "OrderedDict[Tuple[str, str], dict]" = OrderedDict()

    def get_or_load(self, key: Tuple[str, str], loader):
        if key in self._values:
            self._values.move_to_end(key)
            return self._values[key]
        while len(self._values) >= self.capacity:
            old_key, old_value = self._values.popitem(last=False)
            if self.on_evict is not None:
                self.on_evict(old_key, old_value)
            del old_value
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        value = loader()
        self._values[key] = value
        return value

    def clear(self) -> None:
        while self._values:
            key, value = self._values.popitem(last=False)
            if self.on_evict is not None:
                self.on_evict(key, value)
            del value
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __len__(self) -> int:
        return len(self._values)


class CollectionCheckpoint:
    """SQLite-backed, session-atomic collector checkpoint.

    Candidate measurements for a session and its completion marker commit in
    one transaction. A crash can therefore lose at most the current session,
    and a resumed run never treats a partially written session as complete.
    """

    def __init__(self, path: Path, signature: dict, *, resume: bool) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        if existed and not resume:
            raise FileExistsError(
                f"collector checkpoint already exists: {self.path}")
        if resume and not existed:
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {self.path}")
        self.signature_json = canonical_json(signature)
        self.connection = sqlite3.connect(str(self.path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rows (
                seed_index INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS completed_sessions (
                session_id TEXT PRIMARY KEY,
                first_seed_index INTEGER NOT NULL,
                last_seed_index INTEGER NOT NULL,
                expected_seed_count INTEGER NOT NULL,
                row_count INTEGER NOT NULL
            );
        """)
        if not existed:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("schema_version", str(_CHECKPOINT_SCHEMA_VERSION)))
                self.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("signature_json", self.signature_json))
        else:
            metadata = dict(self.connection.execute(
                "SELECT key, value FROM metadata").fetchall())
            if metadata.get("schema_version") != str(
                    _CHECKPOINT_SCHEMA_VERSION):
                self.connection.close()
                raise RuntimeError("collector checkpoint schema mismatch")
            if metadata.get("signature_json") != self.signature_json:
                self.connection.close()
                raise RuntimeError("collector checkpoint signature mismatch")

    def close(self) -> None:
        self.connection.close()

    def completed_sessions(self) -> set[str]:
        return {
            str(row[0]) for row in self.connection.execute(
                "SELECT session_id FROM completed_sessions")
        }

    def last_completed_session(self) -> Optional[str]:
        row = self.connection.execute(
            "SELECT session_id FROM completed_sessions "
            "ORDER BY last_seed_index DESC LIMIT 1").fetchone()
        return str(row[0]) if row is not None else None

    def save_session(
        self,
        session_id: str,
        seed_indices: Sequence[int],
        row_records: Sequence[Tuple[int, dict]],
    ) -> None:
        indices = [int(value) for value in seed_indices]
        if not indices:
            raise ValueError("cannot checkpoint an empty seed session")
        if len(indices) != len(set(indices)):
            raise ValueError("session seed indices are not unique")
        expected = set(indices)
        row_indices = [int(index) for index, _row in row_records]
        if len(row_indices) != len(set(row_indices)):
            raise ValueError("session row indices are not unique")
        if not set(row_indices).issubset(expected):
            raise ValueError("session rows contain an unexpected seed index")
        for _index, row in row_records:
            if str(row.get("session_id")) != str(session_id):
                raise ValueError("row session differs from checkpoint session")
        try:
            with self.connection:
                for seed_index, row in row_records:
                    self.connection.execute(
                        "INSERT INTO rows(seed_index, session_id, payload_json) "
                        "VALUES (?, ?, ?)",
                        (int(seed_index), str(session_id),
                         json.dumps(row)))
                self.connection.execute(
                    "INSERT INTO completed_sessions("
                    "session_id, first_seed_index, last_seed_index, "
                    "expected_seed_count, row_count) VALUES (?, ?, ?, ?, ?)",
                    (str(session_id), min(indices), max(indices), len(indices),
                     len(row_records)))
        except sqlite3.IntegrityError as error:
            raise RuntimeError(
                f"collector session was already checkpointed: {session_id}") \
                from error

    def rows(self) -> List[dict]:
        return [
            json.loads(payload) for (payload,) in self.connection.execute(
                "SELECT payload_json FROM rows ORDER BY seed_index")
        ]

    def progress(self, *, total_sessions: int, total_seeds: int,
                 status: str, last_session: Optional[str]) -> dict:
        completed = int(self.connection.execute(
            "SELECT COUNT(*) FROM completed_sessions").fetchone()[0])
        rows = int(self.connection.execute(
            "SELECT COUNT(*) FROM rows").fetchone()[0])
        seeds = int(self.connection.execute(
            "SELECT COALESCE(SUM(expected_seed_count), 0) "
            "FROM completed_sessions").fetchone()[0])
        return {
            "status": status,
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "completed_sessions": completed,
            "total_sessions": int(total_sessions),
            "completed_seeds": seeds,
            "total_seeds": int(total_seeds),
            "saved_rows": rows,
            "last_completed_session": last_session,
            "checkpoint": str(self.path.resolve()),
            "signature_sha256": hashlib.sha256(
                self.signature_json.encode("utf-8")).hexdigest(),
            "updated_unix": time.time(),
        }


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seed_manifest_sha256(seeds: Sequence[CandidateSeed]) -> str:
    """Hash the exact ordered candidate set used by a resumable collection."""
    records = [{
        "session_id": seed.session_id,
        "scene": seed.scene,
        "episode": seed.episode,
        "kind": seed.kind,
        "query_path": str(seed.query_path),
        "candidate_path": str(seed.candidate_path),
        "candidate_frame": seed.candidate_frame,
        "dino_cosine": seed.dino_cosine,
        "teacher_covis": seed.teacher_covis,
        "label": seed.label,
        "session_has_positive": seed.session_has_positive,
        "session_is_strict_no_match": seed.session_is_strict_no_match,
        "session_max_covis": seed.session_max_covis,
        "causal_manifest_sample_id": seed.causal_manifest_sample_id,
        "selection_origin": seed.selection_origin,
    } for seed in seeds]
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def optional_causal_sample_id(row: pd.Series) -> Optional[str]:
    """Read an explicit manifest key without ever inferring a decision state."""
    if CAUSAL_SAMPLE_ID_COLUMN not in row.index:
        return None
    value = row[CAUSAL_SAMPLE_ID_COLUMN]
    if pd.isna(value):
        return None
    result = str(value).strip()
    return result or None


def session_seed_index_map(
        seeds: Sequence[CandidateSeed]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Return stable session order and fail on non-contiguous sessions."""
    order: List[str] = []
    indices: Dict[str, List[int]] = {}
    last = None
    closed: set[str] = set()
    for seed_index, seed in enumerate(seeds, 1):
        session_id = str(seed.session_id)
        if session_id != last:
            if last is not None:
                closed.add(last)
            if session_id in closed:
                raise RuntimeError(
                    f"candidate session is non-contiguous: {session_id}")
            order.append(session_id)
            indices[session_id] = []
            last = session_id
        indices[session_id].append(seed_index)
    return order, indices


def release_lingbot_device_state(lb) -> None:
    """Drop model-owned KV references before releasing an episode cache."""
    model = getattr(lb, "model", None)
    if model is not None and hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    camera_head = getattr(model, "camera_head", None)
    if camera_head is not None and hasattr(camera_head, "clean_kv_cache"):
        camera_head.clean_kv_cache()
    scale_lru = getattr(lb, "_scale_lru", None)
    if scale_lru is not None:
        scale_lru.clear()


def cuda_memory_summary() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    gib = float(1024 ** 3)
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def git_value(root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            # The shared LingBot checkout is owned by another project member.
            # Scope Git's ownership exception to this one read-only invocation;
            # do not mutate the user's global safe.directory configuration.
            ["git", "-c", f"safe.directory={root.resolve()}",
             "-C", str(root), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def temporal_diverse(rows: pd.DataFrame, count: int,
                     minimum_gap: int) -> List[pd.Series]:
    """Greedy high-DINO selection with a minimum raw-frame separation."""
    chosen: List[pd.Series] = []
    for _, row in rows.sort_values(
            ["dino_cosine", "candidate_frame"],
            ascending=[False, True]).iterrows():
        frame = int(row["candidate_frame"])
        if all(abs(frame - int(old["candidate_frame"])) >= minimum_gap
               for old in chosen):
            chosen.append(row)
            if len(chosen) == count:
                break
    return chosen


def select_balanced_seeds(frame: pd.DataFrame, *, kind: str,
                          sessions: Sequence[str], max_sessions: int,
                          per_class: int, minimum_gap: int,
                          positive_threshold: float,
                          negative_threshold: float,
                          minimum_anchor: int) -> List[CandidateSeed]:
    data = frame.loc[frame["kind"].eq(kind)].copy()
    if sessions:
        data = data.loc[data["session_id"].isin(sessions)]
    data = data.loc[
        data["teacher_covis"].notna()
        & data["dino_cosine"].notna()
        & data["candidate_frame"].ge(minimum_anchor)]
    session_ids = sorted(data["session_id"].unique().tolist())
    if max_sessions:
        session_ids = session_ids[:max_sessions]

    result: List[CandidateSeed] = []
    for session_id in session_ids:
        group = data.loc[data["session_id"].eq(session_id)]
        positive = group.loc[
            group["teacher_covis"].ge(positive_threshold)]
        negative = group.loc[
            group["teacher_covis"].le(negative_threshold)]
        selected = [
            (1, row) for row in temporal_diverse(
                positive, per_class, minimum_gap)
        ] + [
            (0, row) for row in temporal_diverse(
                negative, per_class, minimum_gap)
        ]
        # A session without both classes cannot measure verification separation.
        if not any(label == 1 for label, _ in selected) or not any(
                label == 0 for label, _ in selected):
            continue
        for label, row in selected:
            result.append(CandidateSeed(
                session_id=str(row["session_id"]),
                scene=str(row["scene"]),
                episode=str(row["episode"]),
                kind=str(row["kind"]),
                query_path=Path(str(row["query_path"])).resolve(),
                candidate_path=Path(str(row["candidate_path"])).resolve(),
                candidate_frame=int(row["candidate_frame"]),
                dino_cosine=float(row["dino_cosine"]),
                teacher_covis=float(row["teacher_covis"]),
                label=label,
                session_has_positive=True,
                session_is_strict_no_match=False,
                session_max_covis=float(group["teacher_covis"].max()),
                causal_manifest_sample_id=optional_causal_sample_id(row),
                selection_origin=(
                    "teacher_balanced_positive" if label == 1
                    else "teacher_balanced_negative"),
            ))
    return result


def select_deployment_seeds(
        frame: pd.DataFrame, *, kind: str, sessions: Sequence[str],
        max_sessions: int, top_k: int, minimum_gap: int,
        positive_threshold: float, negative_threshold: float,
        minimum_anchor: int) -> List[CandidateSeed]:
    """Select temporal-diverse top-DINO candidates, including no-match sets.

    Unlike the balanced feasibility sampler, this preserves the deployment
    question at set level. Sessions whose maximum co-visibility is below the
    negative threshold are tagged strict no-match; sessions that have neither a
    strict positive nor a strict no-match remain explicitly ambiguous.
    Candidate rows in the co-visibility ignore band receive ``label=-1`` but
    remain available to a future calibrated set model.
    """
    data = frame.loc[frame["kind"].eq(kind)].copy()
    if sessions:
        data = data.loc[data["session_id"].isin(sessions)]
    data = data.loc[
        data["teacher_covis"].notna()
        & data["dino_cosine"].notna()
        & data["candidate_frame"].ge(minimum_anchor)]
    session_ids = sorted(data["session_id"].unique().tolist())
    if max_sessions:
        session_ids = session_ids[:max_sessions]

    result: List[CandidateSeed] = []
    for session_id in session_ids:
        group = data.loc[data["session_id"].eq(session_id)]
        maximum_covisibility = float(group["teacher_covis"].max())
        has_positive = maximum_covisibility >= positive_threshold
        strict_no_match = maximum_covisibility <= negative_threshold
        for row in temporal_diverse(group, top_k, minimum_gap):
            covisibility = float(row["teacher_covis"])
            label = (1 if covisibility >= positive_threshold else
                     0 if covisibility <= negative_threshold else -1)
            result.append(CandidateSeed(
                session_id=str(row["session_id"]),
                scene=str(row["scene"]),
                episode=str(row["episode"]),
                kind=str(row["kind"]),
                query_path=Path(str(row["query_path"])).resolve(),
                candidate_path=Path(str(row["candidate_path"])).resolve(),
                candidate_frame=int(row["candidate_frame"]),
                dino_cosine=float(row["dino_cosine"]),
                teacher_covis=covisibility,
                label=label,
                session_has_positive=has_positive,
                session_is_strict_no_match=strict_no_match,
                session_max_covis=maximum_covisibility,
                causal_manifest_sample_id=optional_causal_sample_id(row),
                selection_origin="deployment_topk",
            ))
    return result


def select_train_augmented_seeds(
        frame: pd.DataFrame, *, kind: str, sessions: Sequence[str],
        max_sessions: int, top_k: int, minimum_gap: int,
        positive_threshold: float, negative_threshold: float,
        minimum_anchor: int) -> List[CandidateSeed]:
    """Keep deployment candidates, then expose missing train-only supervision.

    Raw DINO rank is not the training target.  If the deployment top-K misses a
    geometric positive (or a hard negative) that is nevertheless present in
    the signed teacher shortlist, a localizer trained only on top-K never sees
    the missing class.  On the train split only, add at most one highest-DINO
    positive and one highest-DINO negative per session.  Development remains
    untouched and uses ``select_deployment_seeds``.
    """

    data = frame.loc[frame["kind"].eq(kind)].copy()
    if sessions:
        data = data.loc[data["session_id"].isin(sessions)]
    data = data.loc[
        data["teacher_covis"].notna()
        & data["dino_cosine"].notna()
        & data["candidate_frame"].ge(minimum_anchor)]
    session_ids = sorted(data["session_id"].unique().tolist())
    if max_sessions:
        session_ids = session_ids[:max_sessions]

    result: List[CandidateSeed] = []
    for session_id in session_ids:
        group = data.loc[data["session_id"].eq(session_id)]
        maximum_covisibility = float(group["teacher_covis"].max())
        has_positive = maximum_covisibility >= positive_threshold
        strict_no_match = maximum_covisibility <= negative_threshold
        selected = temporal_diverse(group, top_k, minimum_gap)
        selected_frames = {int(row["candidate_frame"]) for row in selected}
        selection_origin = {
            int(row["candidate_frame"]): "deployment_topk" for row in selected
        }

        # This augmentation is label-authorized only for the train collector.
        # Pick by DINO within each class so it remains a hard, deployment-like
        # example instead of an oracle-best co-visibility frame.
        augmentation_classes = (
            (lambda value: value >= positive_threshold,
             group.loc[group["teacher_covis"].ge(positive_threshold)],
             "teacher_forced_positive"),
            (lambda value: value <= negative_threshold,
             group.loc[group["teacher_covis"].le(negative_threshold)],
             "teacher_forced_hard_negative"),
        )
        for belongs_to_class, subset, origin in augmentation_classes:
            if any(belongs_to_class(float(row["teacher_covis"]))
                   for row in selected):
                continue
            for row in temporal_diverse(subset, 1, minimum_gap):
                candidate_frame = int(row["candidate_frame"])
                if candidate_frame not in selected_frames:
                    selected.append(row)
                    selected_frames.add(candidate_frame)
                    selection_origin[candidate_frame] = origin

        selected.sort(
            key=lambda row: (-float(row["dino_cosine"]),
                             int(row["candidate_frame"])))
        for row in selected:
            covisibility = float(row["teacher_covis"])
            label = (1 if covisibility >= positive_threshold else
                     0 if covisibility <= negative_threshold else -1)
            result.append(CandidateSeed(
                session_id=str(row["session_id"]),
                scene=str(row["scene"]),
                episode=str(row["episode"]),
                kind=str(row["kind"]),
                query_path=Path(str(row["query_path"])).resolve(),
                candidate_path=Path(str(row["candidate_path"])).resolve(),
                candidate_frame=int(row["candidate_frame"]),
                dino_cosine=float(row["dino_cosine"]),
                teacher_covis=covisibility,
                label=label,
                session_has_positive=has_positive,
                session_is_strict_no_match=strict_no_match,
                session_max_covis=maximum_covisibility,
                causal_manifest_sample_id=optional_causal_sample_id(row),
                selection_origin=selection_origin[int(row["candidate_frame"])],
            ))
    return result


def validate_scene_role(seeds: Sequence[CandidateSeed], manifest: dict,
                        allowed_role: str) -> None:
    allowed_scenes = set(manifest.get(allowed_role, []))
    if not allowed_scenes:
        raise ValueError(
            f"split manifest has no scenes for role {allowed_role}")
    selected_scenes = {seed.scene for seed in seeds}
    leaked = selected_scenes - allowed_scenes
    if leaked:
        raise RuntimeError(
            f"selected scenes outside {allowed_role}: {sorted(leaked)}")


def feature_episode_root(feature_root: Path, seed: CandidateSeed) -> Path:
    # Legacy feature roots may point at one task directory or its parent.  The
    # formal routed-cache path below is preferred for multi-root 3-leg data.
    direct = feature_root / seed.scene / seed.episode
    nested_2leg = feature_root / "mp3d_2leg" / seed.scene / seed.episode
    nested_3leg = feature_root / "mp3d_3leg" / seed.scene / seed.episode
    for path in (direct, nested_2leg, nested_3leg):
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        f"feature episode absent for {seed.scene}/{seed.episode} under "
        f"{feature_root}")


def resolve_routed_feature_cache_pairs(
    manifest: Mapping[str, object],
    seeds: Sequence[CandidateSeed],
    *,
    route_registry: object | None = None,
) -> tuple[
    Dict[Tuple[str, str], Tuple[Path, Path]],
    Mapping[str, object],
] | None:
    """Resolve selected cache pairs through the manifest's physical routing.

    The old collector accepted one broad ``feature_root``.  The formal 3-leg
    set deliberately combines an immutable official cache root with audited
    gap-fill episodes, so guessing a root is both incomplete and unsafe.  A
    routed manifest is therefore the sole authority for every selected pair.
    """

    routing = manifest.get("flow_cache_routing")
    if routing is None:
        if route_registry is not None:
            raise ValueError("route registry supplied for an unrouted manifest")
        return None
    if not isinstance(routing, Mapping):
        raise ValueError("causal manifest flow_cache_routing is malformed")
    try:
        registry = (
            route_registry
            if route_registry is not None
            else registry_from_manifest(manifest)
        )
    except FlowRoutingError as error:
        raise RuntimeError(
            f"causal manifest flow-cache routing is invalid: {error}") from error
    if registry is None or not hasattr(registry, "resolve_manifest_pair"):
        raise RuntimeError("routed causal manifest produced no route registry")

    episodes: Dict[Tuple[str, str], Mapping[str, object]] = {}
    raw_scenes = manifest.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("causal manifest scenes are missing")
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping):
            raise ValueError("causal manifest scene is malformed")
        scene = raw_scene.get("scene")
        raw_episodes = raw_scene.get("selected_episodes")
        if not isinstance(scene, str) or not isinstance(raw_episodes, list):
            raise ValueError("causal manifest scene identity is malformed")
        for raw_episode in raw_episodes:
            if not isinstance(raw_episode, Mapping):
                raise ValueError("causal manifest episode is malformed")
            episode = raw_episode.get("episode")
            key = scene, episode
            if not isinstance(episode, str) or key in episodes:
                raise ValueError(
                    "causal manifest episode identity is empty or duplicated")
            episodes[(scene, episode)] = raw_episode

    selected = sorted({(seed.scene, seed.episode) for seed in seeds})
    if not selected:
        raise ValueError("cannot route an empty candidate selection")
    missing = set(selected) - set(episodes)
    if missing:
        raise RuntimeError(
            f"selected episodes are absent from causal manifest: {sorted(missing)}")
    result: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    for scene, episode in selected:
        try:
            aggregator, camera = registry.resolve_manifest_pair(
                episodes[(scene, episode)], scene, episode)
        except FlowRoutingError as error:
            raise RuntimeError(
                f"routed cache resolution failed for {scene}/{episode}: {error}"
            ) from error
        aggregator = Path(aggregator).resolve()
        camera = Path(camera).resolve()
        if (aggregator.name != "lingbot_cache.npz"
                or camera.name != "lingbot_cam_cache.npz"
                or aggregator.parent != camera.parent):
            raise RuntimeError(
                f"routed cache pair has an invalid layout: {scene}/{episode}")
        result[(scene, episode)] = aggregator, camera
    return result, dict(routing)


def raw_rgb_dir(seed: CandidateSeed) -> Path:
    # Query and candidate may come from different episodes.  LingBot replays
    # the candidate episode, so derive its RGB stream from candidate_path.
    path = seed.candidate_path.parent
    if path.is_dir():
        return path.resolve()
    raise FileNotFoundError(path)


def episode_root_from_image(image_path: Path) -> Path:
    """Return the episode root for a rendered goal or raw RGB frame."""
    image_path = Path(image_path)
    if image_path.name == "goal_image.jpg" or (
            image_path.stem.startswith("goal_")
            and image_path.suffix.lower() == ".jpg"):
        return image_path.parent
    for parent in image_path.parents:
        if parent.name == "videos":
            return parent.parent
    raise ValueError(f"cannot locate episode root for {image_path}")


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(
        value.tolist() if hasattr(value, "tolist") else value,
        dtype=np.float64)
    if array.size != 16:
        raise ValueError(f"{name} must contain 16 values, got {array.shape}")
    return array.reshape(4, 4)


def _resolve_generated_mount(extrinsic: np.ndarray,
                             frame_convention: str) -> np.ndarray:
    """Mirror the MemNav loader's legacy identity-mount compatibility fix."""
    result = np.asarray(extrinsic, dtype=np.float64).copy()
    if not str(frame_convention or "").startswith(
            "positions+parquet in data(Zup,M_W)"):
        return result
    rotation = result[:3, :3]
    if np.allclose(rotation, _HABITAT_TO_DATA_ROTATION, atol=1e-6):
        return result
    if not np.allclose(rotation, np.eye(3), atol=1e-6):
        raise ValueError(
            "generated Z-up episode has an unsupported camera mount")
    result[:3, :3] = _HABITAT_TO_DATA_ROTATION
    return result


def load_episode_pose_data(root: Path) -> EpisodePoseData:
    """Load the exact camera-to-world labels consumed by the NavDP loader."""
    root = Path(root)
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    metadata_path = root / "meta" / "gen_meta.json"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    frame = pd.read_parquet(parquet, columns=[
        "action", "observation.camera_extrinsic"])
    if frame.empty:
        raise ValueError(f"empty pose parquet: {parquet}")
    actions = np.stack([
        _matrix(value, "action") for value in frame["action"]
    ])
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    mount = _resolve_generated_mount(
        _matrix(frame.iloc[0]["observation.camera_extrinsic"],
                "camera extrinsic"),
        str(metadata.get("frame_convention", "")))
    return EpisodePoseData(
        actions=actions, base_extrinsic=mount, metadata=metadata)


def _yaw_habitat_to_data_rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    habitat = np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)
    return _HABITAT_TO_DATA_ROTATION @ habitat


def query_camera_to_world(
        query_path: Path,
        pose_cache: Dict[Path, EpisodePoseData]) -> np.ndarray:
    """Resolve a raw trajectory frame or rendered goal to data-frame c2w."""
    query_path = Path(query_path)
    root = episode_root_from_image(query_path).resolve()
    if root not in pose_cache:
        pose_cache[root] = load_episode_pose_data(root)
    episode = pose_cache[root]
    if query_path.name == "goal_image.jpg":
        goal_index = 0
    elif (query_path.stem.startswith("goal_")
          and query_path.suffix.lower() == ".jpg"):
        try:
            goal_index = int(query_path.stem.split("_", 1)[1]) - 1
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid goal image name: {query_path}") from error
    else:
        try:
            frame_index = int(query_path.stem)
        except ValueError as error:
            raise ValueError(f"invalid raw RGB frame name: {query_path}") from error
        if not 0 <= frame_index < len(episode.actions):
            raise IndexError(f"query frame outside trajectory: {query_path}")
        return episode.actions[frame_index].copy()

    goals = episode.metadata.get("goals", [])
    if not 0 <= goal_index < len(goals):
        raise IndexError(f"goal {goal_index + 1} absent from {root}")
    goal = goals[goal_index]
    position = np.asarray(goal.get("pos"), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"invalid goal position in {root}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _yaw_habitat_to_data_rotation(
        float(goal.get("yaw_habitat", 0.0)))
    result[:3, 3] = position
    return result


def navdp_ground_truth_relative(
        candidate_camera_to_world: np.ndarray,
        query_camera_to_world_pose: np.ndarray,
        base_extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return NavDP planar point-goal and relative camera rotation.

    This deliberately mirrors ``NavDP_Base_Dataset.relative_pose``: remove the
    fixed camera mount before reading base-frame forward/lateral translation,
    while the orientation diagnostic compares camera-to-camera rotations.
    """
    candidate = np.asarray(candidate_camera_to_world, dtype=np.float64)
    query = np.asarray(query_camera_to_world_pose, dtype=np.float64)
    mount = np.asarray(base_extrinsic, dtype=np.float64)
    for name, value in (("candidate", candidate), ("query", query),
                        ("base extrinsic", mount)):
        if value.shape != (4, 4):
            raise ValueError(f"{name} pose must be 4x4, got {value.shape}")
    base_rotation = candidate[:3, :3] @ np.linalg.inv(mount[:3, :3])
    local = base_rotation.T @ (query[:3, 3] - candidate[:3, 3])
    planar = np.array([local[1], -local[0]], dtype=np.float64)
    relative_rotation = candidate[:3, :3].T @ query[:3, :3]
    return planar, relative_rotation


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion is non-finite or degenerate")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def lingbot_relative_prediction(
        anchor_pose9: np.ndarray, goal_pose9: np.ndarray,
        metric_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decode LingBot relative translation/rotation in NavDP conventions."""
    anchor = np.asarray(anchor_pose9, dtype=np.float64)
    goal = np.asarray(goal_pose9, dtype=np.float64)
    if anchor.shape != (9,) or goal.shape != (9,):
        raise ValueError("LingBot pose encodings must each have shape (9,)")
    if not np.isfinite(metric_scale) or metric_scale <= 0.0:
        raise ValueError("metric scale must be finite and positive")
    anchor_rotation = quaternion_xyzw_to_matrix(anchor[3:7])
    goal_rotation = quaternion_xyzw_to_matrix(goal[3:7])
    translation = anchor_rotation.T @ (goal[:3] - anchor[:3])
    # LingBot's camera ground plane is x-z. NavDP point-goal is
    # [forward, lateral] = [LingBot z, -LingBot x].
    planar = float(metric_scale) * np.array(
        [translation[2], -translation[0]], dtype=np.float64)
    relative_rotation = anchor_rotation.T @ goal_rotation
    converted_rotation = (
        _LINGBOT_TO_DATA_ROTATION_BASIS
        @ relative_rotation
        @ _LINGBOT_TO_DATA_ROTATION_BASIS.T)
    return planar, converted_rotation


def rotation_error_degrees(predicted: np.ndarray,
                           target: np.ndarray) -> float:
    relative = np.asarray(predicted).T @ np.asarray(target)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_pose_errors(
        predicted_xy: np.ndarray, target_xy: np.ndarray,
        predicted_rotation: np.ndarray,
        target_rotation: np.ndarray) -> dict:
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    position_error = float(np.linalg.norm(predicted_xy - target_xy))
    predicted_norm = float(np.linalg.norm(predicted_xy))
    target_norm = float(np.linalg.norm(target_xy))
    if predicted_norm <= 1e-9 or target_norm <= 1e-9:
        direction_error = float("nan")
    else:
        cosine = np.clip(
            float(predicted_xy @ target_xy) / (predicted_norm * target_norm),
            -1.0, 1.0)
        direction_error = float(np.degrees(np.arccos(cosine)))
    return {
        "relative_position_error_m": position_error,
        "relative_position_direction_error_deg": direction_error,
        "relative_distance_error_m": abs(predicted_norm - target_norm),
        "relative_rotation_error_deg": rotation_error_degrees(
            predicted_rotation, target_rotation),
    }


def load_cache(lb, cache_path: Path, rgb_dir: Path, num_scale: int) -> dict:
    """Small standalone equivalent of MemNavNet._load_cache."""
    from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream

    cam_path = cache_path.with_name("lingbot_cam_cache.npz")
    with np.load(cache_path) as source, np.load(cam_path) as camera:
        cached = {name: source[name] for name in source.files}
        cam = {name: camera[name] for name in camera.files}
    layout = validate_cache_pair(
        cached, cam, expected_num_scale_frames=num_scale,
        require_versioned=False)
    if "scale_k" in cached and "scale_v" in cached:
        sk, sv, ak, av = LingBotStream._cache_to_layered(
            cached["scale_k"], cached["scale_v"],
            cached["anchor_k"], cached["anchor_v"], lb.device)
    else:
        sk, sv = lb.get_scale_kv(str(rgb_dir))
        ak = torch.as_tensor(
            cached["anchor_k"], device=lb.device,
            dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
        av = torch.as_tensor(
            cached["anchor_v"], device=lb.device,
            dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
    ck, cv = LingBotStream._cam_to_device(
        cam["cam_k"], cam["cam_v"], lb.device)
    result = {
        "scale_k": sk,
        "scale_v": sv,
        "anchor_k": ak,
        "anchor_v": av,
        "cam_k": ck,
        "cam_v": cv,
        "cam_pose_enc": torch.as_tensor(
            cam["cam_pose_enc"], device=lb.device, dtype=torch.float32),
        "ground_h_est": (
            float(cam["ground_h_est"])
            if ("ground_h_est" in cam
                and np.isfinite(float(cam["ground_h_est"])))
            else None),
    }
    if not layout.legacy_dense:
        result["anchor_frame_indices"] = torch.as_tensor(
            layout.anchor_frame_indices, dtype=torch.long)
        result["cam_frame_indices"] = torch.as_tensor(
            layout.cam_frame_indices, dtype=torch.long)
    return result


def quaternion_angle(q1: torch.Tensor, q2: torch.Tensor) -> float:
    q1 = torch.nn.functional.normalize(q1.float(), dim=-1)
    q2 = torch.nn.functional.normalize(q2.float(), dim=-1)
    cosine = torch.sum(q1 * q2).abs().clamp(0.0, 1.0)
    return float(2.0 * torch.acos(cosine))


@torch.no_grad()
def world_cloud(depth: torch.Tensor, confidence: torch.Tensor,
                pose9: torch.Tensor, *, pixel_stride: int,
                confidence_quantile: float, max_points: int) -> Tuple[torch.Tensor, float]:
    """Depth in camera coordinates -> confidence-filtered LingBot-map points."""
    from lingbot_map.utils.rotation import quat_to_mat

    depth = depth.float()
    confidence = confidence.float()
    pose9 = pose9.float()
    height, width = depth.shape
    d = depth[::pixel_stride, ::pixel_stride]
    c = confidence[::pixel_stride, ::pixel_stride]
    ys = torch.arange(
        0, height, pixel_stride, device=depth.device, dtype=torch.float32)
    xs = torch.arange(
        0, width, pixel_stride, device=depth.device, dtype=torch.float32)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    fy = (height / 2.0) / torch.tan(pose9[7] / 2.0)
    fx = (width / 2.0) / torch.tan(pose9[8] / 2.0)
    cam_x = (x - width / 2.0) * d / fx
    cam_y = (y - height / 2.0) * d / fy
    points = torch.stack([cam_x, cam_y, d], dim=-1)
    threshold = torch.quantile(c.reshape(-1), confidence_quantile)
    valid = torch.isfinite(d) & (d > 1e-6) & torch.isfinite(c) & (c >= threshold)
    points = points[valid]
    if points.shape[0] > max_points:
        indices = torch.linspace(
            0, points.shape[0] - 1, max_points,
            device=points.device).round().long()
        points = points[indices]
    rotation = quat_to_mat(torch.nn.functional.normalize(
        pose9[3:7], dim=-1))
    points = points @ rotation.transpose(0, 1) + pose9[:3]
    return points, float(c[valid].mean()) if valid.any() else float("nan")


@torch.no_grad()
def symmetric_cloud_overlap(first: torch.Tensor, second: torch.Tensor,
                            threshold: float) -> Tuple[float, float, float]:
    if not len(first) or not len(second):
        return float("nan"), float("nan"), float("nan")
    distance = torch.cdist(first, second)
    forward = float((distance.min(dim=1).values <= threshold).float().mean())
    backward = float((distance.min(dim=0).values <= threshold).float().mean())
    harmonic = 2.0 * forward * backward / max(forward + backward, 1e-12)
    return forward, backward, harmonic


@torch.no_grad()
def append_goal_at_anchor(lb, cache: dict, rgb_dir: Path,
                          goal_image: torch.Tensor, anchor: int, warm: int,
                          *, pixel_stride: int, confidence_quantile: float,
                          max_points: int, overlap_ratio: float) -> dict:
    """Append one goal and return geometry-native loop-closure measurements."""
    scale = lb.num_scale
    start = max(scale, anchor - warm + 1)
    indices = cache.get("anchor_frame_indices")
    if indices is None:
        lb._inject(
            cache["scale_k"], cache["scale_v"],
            cache["anchor_k"], cache["anchor_v"],
            n_hist=max(0, start - scale), total_frames=start)
    else:
        lb._inject(
            cache["scale_k"], cache["scale_v"],
            cache["anchor_k"], cache["anchor_v"],
            anchor_frame_indices=indices, raw_start=start)

    paths = [rgb_dir / f"{index}.jpg" for index in range(start, anchor + 1)]
    if not paths or not all(path.is_file() for path in paths):
        missing = next((path for path in paths if not path.is_file()), rgb_dir)
        raise FileNotFoundError(missing)
    warm_images = lb.load_images([str(path) for path in paths]).to(lb.device)
    candidate_agg = candidate_psi = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(len(warm_images)):
            candidate_agg, candidate_psi = lb.model._aggregate_features(
                warm_images[index:index + 1][None],
                num_frame_for_scale=scale, num_frame_per_block=1)
        candidate_depth = lb.model._predict_depth(
            candidate_agg, warm_images[-1:][None], candidate_psi)
        goal_agg, goal_psi = lb.model._aggregate_features(
            goal_image[None, None].to(lb.device),
            num_frame_for_scale=scale, num_frame_per_block=1)
        goal_depth = lb.model._predict_depth(
            goal_agg, goal_image[None, None].to(lb.device), goal_psi)

    lb._inject_camera(
        cache["cam_k"], cache["cam_v"], anchor + 1,
        cache.get("cam_frame_indices"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refinement = lb.model.camera_head(
            goal_agg, causal_inference=True,
            num_frame_per_block=1, num_frame_for_scale=scale)
    poses = [item[0, -1].float() for item in refinement]
    goal_pose = poses[-1]
    anchor_pose = cache["cam_pose_enc"][anchor]

    candidate_d = candidate_depth["depth"][0, -1, ..., 0].float()
    candidate_c = candidate_depth["depth_conf"][0, -1].float()
    goal_d = goal_depth["depth"][0, -1, ..., 0].float()
    goal_c = goal_depth["depth_conf"][0, -1].float()
    candidate_cloud, candidate_confidence = world_cloud(
        candidate_d, candidate_c, anchor_pose,
        pixel_stride=pixel_stride,
        confidence_quantile=confidence_quantile,
        max_points=max_points)
    goal_cloud, goal_confidence = world_cloud(
        goal_d, goal_c, goal_pose,
        pixel_stride=pixel_stride,
        confidence_quantile=confidence_quantile,
        max_points=max_points)
    depth_scale = float(torch.median(torch.cat([
        candidate_d[candidate_d > 1e-6].reshape(-1),
        goal_d[goal_d > 1e-6].reshape(-1),
    ])))
    overlap_threshold = max(1e-4, overlap_ratio * depth_scale)
    overlap_forward, overlap_backward, overlap_f1 = symmetric_cloud_overlap(
        candidate_cloud, goal_cloud, overlap_threshold)

    if len(poses) >= 2:
        refine_translation = float((poses[-1][:3] - poses[-2][:3]).norm())
        refine_rotation = quaternion_angle(poses[-1][3:7], poses[-2][3:7])
    else:
        refine_translation = float("nan")
        refine_rotation = float("nan")
    return {
        "anchor": int(anchor),
        "goal_pose": goal_pose.detach().cpu().numpy(),
        "anchor_goal_distance_raw": float((goal_pose[:3] - anchor_pose[:3]).norm()),
        "goal_refine_translation_raw": refine_translation,
        "goal_refine_rotation_deg": math.degrees(refine_rotation),
        "candidate_depth_confidence": candidate_confidence,
        "goal_depth_confidence": goal_confidence,
        "cloud_overlap_candidate_to_goal": overlap_forward,
        "cloud_overlap_goal_to_candidate": overlap_backward,
        "cloud_overlap_f1": overlap_f1,
        "overlap_threshold_raw": overlap_threshold,
        "depth_scale_raw": depth_scale,
    }


def pairwise_pose_dispersion(results: Sequence[dict]) -> Tuple[float, float]:
    if len(results) < 2:
        return float("nan"), float("nan")
    pose = [torch.from_numpy(result["goal_pose"]).float() for result in results]
    translation = []
    rotation = []
    for left in range(len(pose)):
        for right in range(left + 1, len(pose)):
            translation.append(float((pose[left][:3] - pose[right][:3]).norm()))
            rotation.append(math.degrees(quaternion_angle(
                pose[left][3:7], pose[right][3:7])))
    return float(np.median(translation)), float(np.median(rotation))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def jsonable_measurement(measurement: dict) -> dict:
    result = {}
    for key, value in measurement.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def auc_summary(rows: pd.DataFrame) -> Dict[str, dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    definitions = {
        "dino_cosine": ("dino_cosine", 1.0),
        "lingbot_cloud_overlap": ("cloud_overlap_f1_median", 1.0),
        "lingbot_pose_consistency": ("goal_pose_translation_dispersion_norm", -1.0),
        "lingbot_pose_refinement": ("goal_refine_translation_norm_median", -1.0),
    }
    labels = rows["label"].to_numpy(dtype=np.int64)
    result: Dict[str, dict] = {}
    for name, (column, direction) in definitions.items():
        values = rows[column].to_numpy(dtype=np.float64)
        # Ignore-band candidates are retained for a calibrated set model but
        # must not be silently coerced into either binary AUC class.
        valid = np.isfinite(values) & (labels >= 0)
        if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
            result[name] = {"n": int(valid.sum()), "roc_auc": None, "ap": None}
            continue
        score = direction * values[valid]
        result[name] = {
            "n": int(valid.sum()),
            "roc_auc": float(roc_auc_score(labels[valid], score)),
            "ap": float(average_precision_score(labels[valid], score)),
            "expected_direction": "higher" if direction > 0 else "lower",
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internnav-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "InternNav")
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path,
        help="optional scene-role manifest; required with --allowed-role")
    parser.add_argument(
        "--allowed-role", choices=("train", "development", "final_reserved"),
        help="fail if any selected session is outside this frozen scene role")
    parser.add_argument(
        "--feature-root", type=Path,
        help=("legacy single-root LingBot cache location; a routed causal "
              "manifest supersedes this path"))
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha", default="")
    parser.add_argument("--expected-lingbot-commit", default="")
    parser.add_argument(
        "--causal-manifest", type=Path,
        help=("exact multistage manifest used to bind each selected seed; "
              "requires every external-scale pin below"))
    parser.add_argument("--expected-causal-manifest-sha256")
    parser.add_argument("--external-causal-scale-artifact", type=Path)
    parser.add_argument("--expected-external-causal-scale-sha256")
    parser.add_argument("--expected-external-scale-producer-sha256")
    parser.add_argument("--expected-external-scale-configuration-sha256")
    parser.add_argument("--expected-external-scale-lingbot-commit")
    parser.add_argument("--expected-external-scale-weights-sha256")
    parser.add_argument("--expected-external-scale-stream-source-sha256")
    parser.add_argument(
        "--causal-teacher-audit", type=Path,
        help="independent audit receipt for the exact causal teacher CSV")
    parser.add_argument("--expected-causal-teacher-audit-sha256")
    parser.add_argument(
        "--causal-scale-acceptance", type=Path,
        help="independent physical-prefix acceptance receipt for causal scale")
    parser.add_argument("--expected-causal-scale-acceptance-sha256")
    parser.add_argument("--expected-causal-scale-acceptance-commit")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kind", default="revisit_b")
    parser.add_argument(
        "--selection-mode", choices=(
            "balanced", "deployment", "train_augmented"),
        default="balanced",
        help=("balanced: positive/negative feasibility pairs; deployment: "
              "temporal-diverse top-DINO sets including true no-match; "
              "train_augmented: deployment set plus at most one missing "
              "teacher-positive and hard-negative candidate"))
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8,
                        help="candidate count per deployment-mode session")
    parser.add_argument("--candidate-min-gap", type=int, default=4)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.2)
    parser.add_argument("--neighbor-offset", type=int, action="append",
                        default=None,
                        help="repeatable; default: -4, 0, +4")
    parser.add_argument("--warm", type=int, default=64)
    parser.add_argument(
        "--full-replay", action="store_true",
        help=("replay every real frame from the scale block through each "
              "candidate, matching the online pose-only controller"))
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=10)
    parser.add_argument("--confidence-quantile", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=768)
    parser.add_argument("--overlap-ratio", type=float, default=0.025)
    parser.add_argument(
        "--pooled-metric-scale", type=float,
        default=_DEFAULT_POOLED_METRIC_SCALE,
        help="fallback LingBot-units-to-meters scale if ground recovery fails")
    parser.add_argument(
        "--max-cached-episodes", type=int, default=1,
        help=("maximum LingBot episode caches retained on the accelerator; "
              "one is sufficient because selected sessions are contiguous"))
    parser.add_argument(
        "--resume", action="store_true",
        help=("resume only sessions atomically committed in the existing "
              "SQLite checkpoint; the exact candidate/config signature must "
              "match"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = tuple(sorted(set(args.neighbor_offset or (-4, 0, 4))))
    external_arguments = (
        args.causal_manifest,
        args.expected_causal_manifest_sha256,
        args.external_causal_scale_artifact,
        args.expected_external_causal_scale_sha256,
        args.expected_external_scale_producer_sha256,
        args.expected_external_scale_configuration_sha256,
        args.expected_external_scale_lingbot_commit,
        args.expected_external_scale_weights_sha256,
        args.expected_external_scale_stream_source_sha256,
        args.causal_teacher_audit,
        args.expected_causal_teacher_audit_sha256,
        args.causal_scale_acceptance,
        args.expected_causal_scale_acceptance_sha256,
        args.expected_causal_scale_acceptance_commit,
    )
    external_scale_mode = any(value is not None for value in external_arguments)
    if external_scale_mode and not all(value is not None
                                       for value in external_arguments):
        raise ValueError(
            "external causal scale requires its manifest, artifact, and every "
            "exact SHA/model pin")
    if 0 not in offsets:
        raise ValueError("neighbor offsets must include 0")
    if (args.per_class < 1 or args.top_k < 1 or args.max_sessions < 0
            or args.candidate_min_gap < 1 or args.warm < 1
            or args.num_scale < 1 or args.pixel_stride < 1
            or args.max_points < 16 or args.overlap_ratio <= 0.0
            or args.max_cached_episodes < 1
            or not np.isfinite(args.pooled_metric_scale)
            or args.pooled_metric_scale <= 0.0):
        raise ValueError("invalid diagnostic configuration")
    if not 0.0 <= args.negative_threshold < args.positive_threshold <= 1.0:
        raise ValueError("invalid co-visibility thresholds")
    for path in (args.internnav_root, args.teacher_csv,
                 args.lingbot_repo, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    if bool(args.split_manifest) != bool(args.allowed_role):
        raise ValueError(
            "--split-manifest and --allowed-role must be provided together")
    if external_scale_mode and args.allowed_role not in ("train", "development"):
        raise ValueError(
            "external causal-scale collection requires an explicit train or "
            "development scene role")
    if args.split_manifest and not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    sys.path.insert(0, str(args.internnav_root.resolve()))

    weight_sha = sha256(args.weights)
    teacher_sha = sha256(args.teacher_csv)
    source_commit = git_value(
        Path(__file__).resolve().parents[1], "rev-parse", "HEAD")
    lingbot_commit = git_value(args.lingbot_repo, "rev-parse", "HEAD")
    if args.expected_weight_sha and weight_sha != args.expected_weight_sha:
        raise RuntimeError(
            f"LingBot weight SHA mismatch: {weight_sha} != "
            f"{args.expected_weight_sha}")
    if (args.expected_lingbot_commit
            and lingbot_commit != args.expected_lingbot_commit):
        raise RuntimeError(
            f"LingBot commit mismatch: {lingbot_commit} != "
            f"{args.expected_lingbot_commit}")

    external_contract: Optional[ExternalCausalScaleContract] = None
    external_contract_summary: Optional[dict] = None
    upstream_receipts_summary: Optional[dict] = None
    if external_scale_mode:
        assert args.causal_manifest is not None
        assert args.external_causal_scale_artifact is not None
        assert args.expected_causal_manifest_sha256 is not None
        assert args.expected_external_causal_scale_sha256 is not None
        assert args.expected_external_scale_producer_sha256 is not None
        assert args.expected_external_scale_configuration_sha256 is not None
        assert args.expected_external_scale_lingbot_commit is not None
        assert args.expected_external_scale_weights_sha256 is not None
        assert args.expected_external_scale_stream_source_sha256 is not None
        assert args.causal_teacher_audit is not None
        assert args.expected_causal_teacher_audit_sha256 is not None
        assert args.causal_scale_acceptance is not None
        assert args.expected_causal_scale_acceptance_sha256 is not None
        assert args.expected_causal_scale_acceptance_commit is not None
        stream_source = (
            args.internnav_root / "internnav" / "model" / "basemodel"
            / "memnav" / "lingbot_stream.py")
        if not stream_source.is_file():
            raise FileNotFoundError(stream_source)
        stream_source_sha = contract_sha256_file(stream_source)
        if args.expected_external_scale_lingbot_commit != lingbot_commit:
            raise RuntimeError(
                "external scale LingBot commit differs from the collector")
        if args.expected_external_scale_weights_sha256 != weight_sha:
            raise RuntimeError(
                "external scale weights differ from the collector")
        if args.expected_external_scale_stream_source_sha256 != stream_source_sha:
            raise RuntimeError(
                "external scale LingBot stream source differs from the collector")
        external_contract = ExternalCausalScaleContract(
            manifest_path=args.causal_manifest,
            artifact_path=args.external_causal_scale_artifact,
            pins=ExternalCausalScalePins(
                manifest_sha256=args.expected_causal_manifest_sha256,
                artifact_sha256=args.expected_external_causal_scale_sha256,
                producer_sha256=args.expected_external_scale_producer_sha256,
                configuration_sha256=(
                    args.expected_external_scale_configuration_sha256),
                lingbot_commit=args.expected_external_scale_lingbot_commit,
                weights_sha256=args.expected_external_scale_weights_sha256,
                stream_source_sha256=(
                    args.expected_external_scale_stream_source_sha256),
            ),
        )
        if external_contract.num_scale_frames != args.num_scale:
            raise RuntimeError(
                "collector num_scale differs from the external scale contract")
        external_contract_summary = external_contract.summary()
        upstream_receipts_summary = validate_phase_b_upstream_receipts(
            teacher_csv_path=args.teacher_csv,
            teacher_audit_path=args.causal_teacher_audit,
            manifest_path=args.causal_manifest,
            scale_artifact_path=args.external_causal_scale_artifact,
            scale_acceptance_path=args.causal_scale_acceptance,
            pins=PhaseBUpstreamPins(
                teacher_csv_sha256=teacher_sha,
                teacher_audit_sha256=(
                    args.expected_causal_teacher_audit_sha256),
                manifest_sha256=args.expected_causal_manifest_sha256,
                scale_artifact_sha256=(
                    args.expected_external_causal_scale_sha256),
                scale_acceptance_sha256=(
                    args.expected_causal_scale_acceptance_sha256),
                scale_acceptance_commit=(
                    args.expected_causal_scale_acceptance_commit),
                scale_producer_sha256=(
                    args.expected_external_scale_producer_sha256),
                scale_configuration_sha256=(
                    args.expected_external_scale_configuration_sha256),
                scale_lingbot_commit=(
                    args.expected_external_scale_lingbot_commit),
                scale_weights_sha256=(
                    args.expected_external_scale_weights_sha256),
                scale_stream_source_sha256=(
                    args.expected_external_scale_stream_source_sha256),
            ),
        )

    teacher = pd.read_csv(args.teacher_csv)
    missing = REQUIRED_COLUMNS - set(teacher.columns)
    if missing:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    if external_scale_mode and CAUSAL_SAMPLE_ID_COLUMN not in teacher.columns:
        raise ValueError(
            "external causal-scale collection requires the teacher CSV to "
            f"carry explicit {CAUSAL_SAMPLE_ID_COLUMN}; decision state cannot "
            "be inferred from goal path")
    selection_arguments = dict(
        kind=args.kind, sessions=args.session,
        max_sessions=args.max_sessions,
        minimum_gap=args.candidate_min_gap,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        minimum_anchor=args.num_scale)
    if args.selection_mode == "balanced":
        seeds = select_balanced_seeds(
            teacher, per_class=args.per_class, **selection_arguments)
    elif args.selection_mode == "deployment":
        seeds = select_deployment_seeds(
            teacher, top_k=args.top_k, **selection_arguments)
    else:
        if args.allowed_role != "train":
            raise ValueError(
                "train_augmented selection is forbidden outside the train split")
        seeds = select_train_augmented_seeds(
            teacher, top_k=args.top_k, **selection_arguments)
    if not seeds:
        raise RuntimeError(
            f"no {args.selection_mode} candidate seeds selected")
    session_order, seed_indices_by_session = session_seed_index_map(seeds)
    split_manifest_sha = None
    if args.split_manifest:
        with args.split_manifest.open(encoding="utf-8") as handle:
            split_manifest = json.load(handle)
        validate_scene_role(seeds, split_manifest, args.allowed_role)
        split_manifest_sha = sha256(args.split_manifest)
    for seed in seeds:
        if not seed.query_path.is_file():
            raise FileNotFoundError(seed.query_path)
        if not seed.candidate_path.is_file():
            raise FileNotFoundError(seed.candidate_path)

    external_bindings: Dict[CandidateSeed, ExternalCausalScaleBinding] = {}
    if external_contract is not None:
        for seed in seeds:
            if seed.causal_manifest_sample_id is None:
                raise RuntimeError(
                    f"selected seed lacks {CAUSAL_SAMPLE_ID_COLUMN}: "
                    f"{seed.session_id}/{seed.candidate_frame}")
            external_bindings[seed] = external_contract.bind_seed(
                manifest_sample_id=seed.causal_manifest_sample_id,
                scene=seed.scene,
                episode=seed.episode,
                query_path=seed.query_path,
                candidate_path=seed.candidate_path,
                candidate_frame=seed.candidate_frame,
                neighbor_offsets=offsets,
                expected_split_role=str(args.allowed_role),
            )
        for session_id in session_order:
            sample_ids = {
                external_bindings[seeds[index - 1]].sample_id
                for index in seed_indices_by_session[session_id]
            }
            if len(sample_ids) != 1:
                raise RuntimeError(
                    "one candidate session crosses causal manifest samples: "
                    f"{session_id} -> {sorted(sample_ids)}")

    routed_cache_pairs: Optional[
        Dict[Tuple[str, str], Tuple[Path, Path]]
    ] = None
    routed_cache_provenance: Optional[Mapping[str, object]] = None
    if external_contract is not None:
        routed = resolve_routed_feature_cache_pairs(
            external_contract.manifest, seeds)
        if routed is None:
            raise RuntimeError(
                "formal external causal-scale collection requires the pinned "
                "manifest flow_cache_routing contract")
        routed_cache_pairs, routed_cache_provenance = routed
    elif args.feature_root is None or not args.feature_root.exists():
        raise FileNotFoundError(
            "legacy collection requires an existing --feature-root")

    def cache_pair_for_seed(seed: CandidateSeed) -> Tuple[Path, Path]:
        if routed_cache_pairs is not None:
            key = seed.scene, seed.episode
            if key not in routed_cache_pairs:
                raise RuntimeError(
                    f"selected episode lacks a routed cache pair: {key}")
            return routed_cache_pairs[key]
        assert args.feature_root is not None
        episode_root = feature_episode_root(args.feature_root, seed)
        aggregator = (
            episode_root / "videos" / "chunk-000" / "lingbot_cache.npz")
        return aggregator, aggregator.with_name("lingbot_cam_cache.npz")

    # Validate every selected raw/cache dependency before allocating model
    # weights. This path is also invoked as a standalone Slurm preflight.
    from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair

    checked_episodes = set()
    pose_cache: Dict[Path, EpisodePoseData] = {}
    for seed in seeds:
        key = (seed.scene, seed.episode)
        if key not in checked_episodes:
            checked_episodes.add(key)
            cache_path, cam_path = cache_pair_for_seed(seed)
            for required in (cache_path, cam_path):
                if not required.exists():
                    raise FileNotFoundError(required)
            with np.load(cache_path) as cached, np.load(cam_path) as camera:
                validate_cache_pair(
                    cached, camera,
                    expected_num_scale_frames=args.num_scale,
                    require_versioned=False)
                if external_contract is not None:
                    cache_schema = int(np.asarray(
                        camera["cache_schema_version"]).reshape(-1)[0])
                    precompute_signature = str(np.asarray(
                        camera["precompute_signature"]).reshape(-1)[0])
                    external_contract.validate_runtime_episode(
                        scene=seed.scene,
                        episode=seed.episode,
                        cam_pose_enc=np.asarray(camera["cam_pose_enc"]),
                        cache_schema_version=cache_schema,
                        precompute_signature=precompute_signature,
                    )
        candidate_root = episode_root_from_image(
            seed.candidate_path).resolve()
        if candidate_root not in pose_cache:
            pose_cache[candidate_root] = load_episode_pose_data(candidate_root)
        candidate_pose_data = pose_cache[candidate_root]
        if int(seed.candidate_path.stem) != seed.candidate_frame:
            raise ValueError(
                "candidate_frame disagrees with candidate filename: "
                f"{seed.candidate_path}")
        if not 0 <= seed.candidate_frame < len(candidate_pose_data.actions):
            raise IndexError(
                f"candidate frame outside trajectory: {seed.candidate_path}")
        query_camera_to_world(seed.query_path, pose_cache)
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_passed",
            "n_seeds": len(seeds),
            "n_sessions": len(session_order),
            "n_episodes": len(checked_episodes),
            "n_pose_episodes": len(pose_cache),
            "selection_mode": args.selection_mode,
            "allowed_role": args.allowed_role,
            "split_manifest_sha256": split_manifest_sha,
            "lingbot_commit": lingbot_commit,
            "lingbot_weight_sha256": weight_sha,
            "teacher_csv_sha256": teacher_sha,
            "external_causal_scale": external_contract_summary,
            "upstream_receipts": upstream_receipts_summary,
            "flow_cache_routing": routed_cache_provenance,
        }, indent=2, sort_keys=True))
        return

    from internnav.model.basemodel.memnav.lingbot_stream import (
        LingBotStream,
        ground_scale_from_h_est,
    )

    checkpoint_signature = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "seed_manifest_sha256": seed_manifest_sha256(seeds),
        "total_seeds": len(seeds),
        "total_sessions": len(session_order),
        "compute_config": {
            "selection_mode": args.selection_mode,
            "kind": args.kind,
            "allowed_role": args.allowed_role,
            "sessions": args.session,
            "max_sessions": args.max_sessions,
            "per_class": args.per_class,
            "top_k": args.top_k,
            "candidate_min_gap": args.candidate_min_gap,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "neighbor_offsets": offsets,
            "warm": args.warm,
            "full_replay": args.full_replay,
            "num_scale": args.num_scale,
            "window": args.window,
            "max_frame_num": args.max_frame_num,
            "camera_num_iterations": args.camera_num_iterations,
            "pixel_stride": args.pixel_stride,
            "confidence_quantile": args.confidence_quantile,
            "max_points": args.max_points,
            "overlap_ratio": args.overlap_ratio,
            "pooled_metric_scale": args.pooled_metric_scale,
            "max_cached_episodes": args.max_cached_episodes,
            "device": args.device,
            "metric_scale_mode": (
                EXTERNAL_CAUSAL_SCALE_SOURCE
                if external_contract is not None else "legacy_runtime_or_cached"),
            "feature_cache_mode": (
                "manifest_provenance_pinned_multi_root"
                if routed_cache_pairs is not None
                else "legacy_single_root"),
        },
        "provenance": {
            "source_commit": source_commit,
            "lingbot_commit": lingbot_commit,
            "lingbot_weight_sha256": weight_sha,
            "teacher_csv_sha256": teacher_sha,
            "split_manifest_sha256": split_manifest_sha,
            "feature_root": (
                str(args.feature_root.resolve())
                if args.feature_root is not None else None),
            "flow_cache_routing": routed_cache_provenance,
            "external_causal_scale": external_contract_summary,
            "upstream_receipts": upstream_receipts_summary,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / _ROWS_FILENAME
    json_path = args.out_dir / _REPORT_FILENAME
    progress_path = args.out_dir / _PROGRESS_FILENAME
    checkpoint = CollectionCheckpoint(
        args.out_dir / _CHECKPOINT_FILENAME,
        checkpoint_signature,
        resume=args.resume,
    )
    completed_sessions = checkpoint.completed_sessions()
    unknown_sessions = completed_sessions - set(session_order)
    if unknown_sessions:
        checkpoint.close()
        raise RuntimeError(
            f"checkpoint contains unknown sessions: {sorted(unknown_sessions)}")
    atomic_write_json(progress_path, checkpoint.progress(
        total_sessions=len(session_order), total_seeds=len(seeds),
        status="collecting",
        last_session=checkpoint.last_completed_session()))

    started = time.time()
    lb = LingBotStream(
        lingbot_repo=str(args.lingbot_repo.resolve()),
        weights=str(args.weights.resolve()),
        num_scale=args.num_scale,
        window=args.window,
        max_frame_num=args.max_frame_num,
        camera_num_iterations=args.camera_num_iterations,
        device=args.device,
        scale_lru_size=args.max_cached_episodes,
    ).eval()
    episode_cache = BoundedEpisodeCache(
        args.max_cached_episodes,
        on_evict=lambda _key, _value: release_lingbot_device_state(lb),
    )
    metric_scale_by_episode: Dict[Tuple[str, str], Tuple[float, str]] = {}
    current_session: Optional[str] = None
    current_rows: List[Tuple[int, dict]] = []

    def save_current_session() -> None:
        nonlocal current_rows
        if current_session is None or current_session in completed_sessions:
            return
        checkpoint.save_session(
            current_session,
            seed_indices_by_session[current_session],
            current_rows,
        )
        completed_sessions.add(current_session)
        progress = checkpoint.progress(
            total_sessions=len(session_order), total_seeds=len(seeds),
            status="collecting", last_session=current_session)
        progress["cuda_memory"] = cuda_memory_summary()
        atomic_write_json(progress_path, progress)
        print(
            f"[checkpoint] sessions={len(completed_sessions)}/"
            f"{len(session_order)} rows={progress['saved_rows']} "
            f"last={current_session} cuda={progress['cuda_memory']}",
            flush=True,
        )
        current_rows = []

    for seed_index, seed in enumerate(seeds, 1):
        if seed.session_id != current_session:
            save_current_session()
            current_session = seed.session_id
            current_rows = []
            if current_session in completed_sessions:
                print(
                    f"[resume] skip completed session {current_session}",
                    flush=True,
                )
        if current_session in completed_sessions:
            continue
        key = (seed.scene, seed.episode)
        cache_path, _cam_path = cache_pair_for_seed(seed)
        rgb_dir = raw_rgb_dir(seed)
        # Drop the loop-local reference before get_or_load can evict the prior
        # episode. Otherwise the caller itself keeps all old CUDA tensors alive
        # until after the next episode has already been allocated.
        cache = None
        cache = episode_cache.get_or_load(
            key,
            lambda: load_cache(
                lb, cache_path, rgb_dir, args.num_scale),
        )
        candidate_root = episode_root_from_image(
            seed.candidate_path).resolve()
        candidate_pose_data = pose_cache[candidate_root]
        query_pose = query_camera_to_world(seed.query_path, pose_cache)
        external_binding = external_bindings.get(seed)
        if external_binding is not None:
            metric_scale = external_binding.metric_scale_m_per_raw
            metric_scale_source = EXTERNAL_CAUSAL_SCALE_SOURCE
            previous = metric_scale_by_episode.get(key)
            if previous is not None and previous != (
                    metric_scale, metric_scale_source):
                raise RuntimeError(
                    f"external scale changed within {seed.scene}/{seed.episode}")
            metric_scale_by_episode[key] = (
                metric_scale, metric_scale_source)
        elif key not in metric_scale_by_episode:
            camera_height = float(candidate_pose_data.metadata.get(
                "camera_height_m", 0.5))
            cached_ground_height = cache.get("ground_h_est")
            if cached_ground_height is not None:
                ground_scale = ground_scale_from_h_est(
                    cached_ground_height, camera_height)
                ground_source = "cached_ground_anchored"
            else:
                ground_scale = lb.get_metric_scale(
                    str(rgb_dir), cache["cam_pose_enc"], camera_height)
                ground_source = "runtime_ground_anchored"
            if (ground_scale is not None
                    and np.isfinite(ground_scale) and ground_scale > 0.0):
                metric_scale_by_episode[key] = (
                    float(ground_scale), ground_source)
            else:
                metric_scale_by_episode[key] = (
                    float(args.pooled_metric_scale), "pooled_fallback")
        if external_binding is None:
            metric_scale, metric_scale_source = metric_scale_by_episode[key]
        goal = lb.load_images([str(seed.query_path)])[0].to(lb.device)
        maximum_anchor = min(
            len(cache["cam_pose_enc"]) - 2,
            len(candidate_pose_data.actions) - 1,
            max(int(path.stem) for path in rgb_dir.glob("*.jpg")
                if path.stem.isdigit()))
        if external_binding is not None:
            maximum_anchor = min(
                maximum_anchor, external_binding.decision_frame - 1)
        hypotheses = []
        print(
            f"[{seed_index}/{len(seeds)}] {seed.session_id} "
            f"frame={seed.candidate_frame} label={seed.label} "
            f"covis={seed.teacher_covis:.3f}", flush=True)
        for offset in offsets:
            anchor = seed.candidate_frame + offset
            if not args.num_scale <= anchor <= maximum_anchor:
                continue
            replay_warm = (
                anchor - args.num_scale + 1
                if args.full_replay else args.warm
            )
            measurement = append_goal_at_anchor(
                lb, cache, rgb_dir, goal, anchor, replay_warm,
                pixel_stride=args.pixel_stride,
                confidence_quantile=args.confidence_quantile,
                max_points=args.max_points,
                overlap_ratio=args.overlap_ratio)
            measurement["offset"] = offset
            measurement["replay_frames"] = replay_warm
            anchor_pose9 = cache["cam_pose_enc"][anchor].detach().cpu().numpy()
            predicted_xy, predicted_rotation = lingbot_relative_prediction(
                anchor_pose9, measurement["goal_pose"], metric_scale)
            target_xy, target_rotation = navdp_ground_truth_relative(
                candidate_pose_data.actions[anchor], query_pose,
                candidate_pose_data.base_extrinsic)
            measurement.update(relative_pose_errors(
                predicted_xy, target_xy,
                predicted_rotation, target_rotation))
            measurement.update({
                "metric_scale_m_per_raw": metric_scale,
                "metric_scale_source": metric_scale_source,
                "predicted_relative_xy_m": predicted_xy,
                "target_relative_xy_m": target_xy,
                "target_relative_distance_m": float(np.linalg.norm(target_xy)),
            })
            hypotheses.append(measurement)
        if not hypotheses:
            continue
        translation_dispersion, rotation_dispersion = pairwise_pose_dispersion(
            hypotheses)
        depth_scale = finite_median(
            result["depth_scale_raw"] for result in hypotheses)
        norm = max(depth_scale, 1e-6)
        center = min(hypotheses, key=lambda result: abs(result["offset"]))
        row = {
            "session_id": seed.session_id,
            "scene": seed.scene,
            "episode": seed.episode,
            "kind": seed.kind,
            "query_path": str(seed.query_path),
            "candidate_path": str(seed.candidate_path),
            "candidate_frame": seed.candidate_frame,
            "label": seed.label,
            "session_has_positive": seed.session_has_positive,
            "session_is_strict_no_match": seed.session_is_strict_no_match,
            "session_max_covis": seed.session_max_covis,
            "candidate_selection_origin": seed.selection_origin,
            "teacher_covis": seed.teacher_covis,
            "dino_cosine": seed.dino_cosine,
            "metric_scale_m_per_raw": metric_scale,
            "metric_scale_source": metric_scale_source,
            "n_hypotheses": len(hypotheses),
            "neighbor_offsets": ";".join(str(item["offset"]) for item in hypotheses),
            "depth_scale_raw": depth_scale,
            "goal_pose_translation_dispersion_raw": translation_dispersion,
            "goal_pose_translation_dispersion_norm": translation_dispersion / norm,
            "goal_pose_rotation_dispersion_deg": rotation_dispersion,
            "cloud_overlap_f1_center": center["cloud_overlap_f1"],
            "cloud_overlap_f1_mean": finite_mean(
                item["cloud_overlap_f1"] for item in hypotheses),
            "cloud_overlap_f1_median": finite_median(
                item["cloud_overlap_f1"] for item in hypotheses),
            "anchor_goal_distance_norm_center": (
                center["anchor_goal_distance_raw"] / norm),
            "goal_refine_translation_norm_median": finite_median(
                item["goal_refine_translation_raw"] / max(
                    item["depth_scale_raw"], 1e-6) for item in hypotheses),
            "goal_refine_rotation_deg_median": finite_median(
                item["goal_refine_rotation_deg"] for item in hypotheses),
            "relative_position_error_m_center": center[
                "relative_position_error_m"],
            "relative_position_error_m_median": finite_median(
                item["relative_position_error_m"] for item in hypotheses),
            "relative_position_direction_error_deg_center": center[
                "relative_position_direction_error_deg"],
            "relative_position_direction_error_deg_median": finite_median(
                item["relative_position_direction_error_deg"]
                for item in hypotheses),
            "relative_distance_error_m_center": center[
                "relative_distance_error_m"],
            "relative_rotation_error_deg_center": center[
                "relative_rotation_error_deg"],
            "relative_rotation_error_deg_median": finite_median(
                item["relative_rotation_error_deg"] for item in hypotheses),
            "predicted_relative_xy_m_center_json": json.dumps(
                center["predicted_relative_xy_m"].tolist()),
            "target_relative_xy_m_center_json": json.dumps(
                center["target_relative_xy_m"].tolist()),
            "goal_pose9_center_json": json.dumps(
                center["goal_pose"].tolist()),
            "goal_depth_confidence_mean": finite_mean(
                item["goal_depth_confidence"] for item in hypotheses),
            "candidate_depth_confidence_mean": finite_mean(
                item["candidate_depth_confidence"] for item in hypotheses),
            "hypotheses_json": json.dumps([
                jsonable_measurement(item) for item in hypotheses
            ], sort_keys=True),
        }
        if external_binding is not None:
            row.update(external_binding.row_fields())
        current_rows.append((seed_index, row))
    save_current_session()
    episode_cache.clear()
    result_frame = pd.DataFrame(checkpoint.rows())
    if result_frame.empty:
        checkpoint.close()
        raise RuntimeError("all selected candidate seeds were skipped")

    atomic_write_frame(csv_path, result_frame)
    by_label = {}
    for label, name in ((0, "negative"), (1, "positive")):
        subset = result_frame.loc[result_frame["label"].eq(label)]
        by_label[name] = {
            "n": int(len(subset)),
            "dino_cosine_median": finite_median(subset["dino_cosine"]),
            "cloud_overlap_f1_median": finite_median(
                subset["cloud_overlap_f1_median"]),
            "pose_translation_dispersion_norm_median": finite_median(
                subset["goal_pose_translation_dispersion_norm"]),
            "pose_rotation_dispersion_deg_median": finite_median(
                subset["goal_pose_rotation_dispersion_deg"]),
            "goal_refine_translation_norm_median": finite_median(
                subset["goal_refine_translation_norm_median"]),
            "relative_position_error_m_median": finite_median(
                subset["relative_position_error_m_center"]),
            "relative_direction_error_deg_median": finite_median(
                subset["relative_position_direction_error_deg_center"]),
            "relative_rotation_error_deg_median": finite_median(
                subset["relative_rotation_error_deg_center"]),
        }
    session_rows = result_frame.sort_values(
        ["session_id", "dino_cosine"], ascending=[True, False])
    session_first = session_rows.drop_duplicates("session_id")
    positive_sessions = set(session_first.loc[
        session_first["session_has_positive"], "session_id"])
    selected_positive_sessions = set(result_frame.loc[
        result_frame["label"].eq(1), "session_id"])
    strict_no_match_sessions = set(session_first.loc[
        session_first["session_is_strict_no_match"], "session_id"])
    ambiguous_sessions = set(session_first["session_id"]) - (
        positive_sessions | strict_no_match_sessions)
    candidate_recall = (
        len(positive_sessions & selected_positive_sessions)
        / len(positive_sessions) if positive_sessions else float("nan"))
    selection_origin_counts = {
        str(name): int(count)
        for name, count in result_frame[
            "candidate_selection_origin"].value_counts().sort_index().items()
    }
    report = {
        "status": "diagnostic_not_for_deployment",
        "objective": (
            "test whether LingBot-native pose consensus, point-cloud overlap, "
            "metric relative pose, and uncertainty can localize an ImageGoal "
            "without always invoking SIFT/RANSAC"),
        "limitations": ({
            "balanced": ["small deliberately balanced feasibility subset"],
            "deployment": [
                "top-DINO deployment-style subset; no learned probability calibration yet",
            ],
            "train_augmented": [
                "train-only top-DINO set augmented by at most one signed-teacher "
                "positive and hard negative; development must remain deployment-only",
            ],
        }[args.selection_mode]) + [
            "candidate labels come from task-aligned co-visibility teacher",
            "ground-truth pose errors are evaluation targets, not deployment inputs",
            "no threshold may be selected from final-reserved scenes",
            "closed-loop navigation is not measured here",
        ],
        "n_rows": int(len(result_frame)),
        "n_sessions": int(result_frame["session_id"].nunique()),
        "set_level": {
            "positive_sessions": len(positive_sessions),
            "strict_no_match_sessions": len(strict_no_match_sessions),
            "ambiguous_sessions": len(ambiguous_sessions),
            "positive_session_candidate_recall_at_selected_k": candidate_recall,
            "candidate_selection_origin_counts": selection_origin_counts,
        },
        "by_label": by_label,
        "feature_separation": auc_summary(result_frame),
        "config": {
            "kind": args.kind,
            "selection_mode": args.selection_mode,
            "allowed_role": args.allowed_role,
            "sessions": args.session,
            "max_sessions": args.max_sessions,
            "per_class": args.per_class,
            "top_k": args.top_k,
            "candidate_min_gap": args.candidate_min_gap,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "neighbor_offsets": offsets,
            "warm": args.warm,
            "full_replay": args.full_replay,
            "num_scale": args.num_scale,
            "window": args.window,
            "camera_num_iterations": args.camera_num_iterations,
            "pixel_stride": args.pixel_stride,
            "confidence_quantile": args.confidence_quantile,
            "max_points": args.max_points,
            "overlap_ratio": args.overlap_ratio,
            "pooled_metric_scale": args.pooled_metric_scale,
            "max_cached_episodes": args.max_cached_episodes,
            "resumed": args.resume,
            "metric_scale_mode": (
                EXTERNAL_CAUSAL_SCALE_SOURCE
                if external_contract is not None else "legacy_runtime_or_cached"),
            "feature_cache_mode": (
                "manifest_provenance_pinned_multi_root"
                if routed_cache_pairs is not None
                else "legacy_single_root"),
        },
        "provenance": {
            "source_commit": source_commit,
            "lingbot_commit": lingbot_commit,
            "lingbot_weight_sha256": weight_sha,
            "teacher_csv": str(args.teacher_csv.resolve()),
            "teacher_csv_sha256": teacher_sha,
            "split_manifest": (
                str(args.split_manifest.resolve())
                if args.split_manifest else None),
            "split_manifest_sha256": split_manifest_sha,
            "feature_root": (
                str(args.feature_root.resolve())
                if args.feature_root is not None else None),
            "flow_cache_routing": routed_cache_provenance,
            "external_causal_scale": external_contract_summary,
            "upstream_receipts": upstream_receipts_summary,
            "elapsed_seconds": time.time() - started,
        },
        "rows_csv": str(csv_path.resolve()),
        "collector_checkpoint": str(checkpoint.path.resolve()),
    }
    atomic_write_json(json_path, report)
    final_progress = checkpoint.progress(
        total_sessions=len(session_order), total_seeds=len(seeds),
        status="complete", last_session=session_order[-1])
    final_progress["cuda_memory"] = cuda_memory_summary()
    atomic_write_json(progress_path, final_progress)
    checkpoint.close()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
