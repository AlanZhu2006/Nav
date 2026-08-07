#!/usr/bin/env python3
"""Fail-closed audit for LingBot-native localizer training artifacts.

The exact LingBot collector is expensive, resumable, and produces several
representations of the same result (SQLite checkpoint, CSV, progress JSON and
summary JSON).  A successful Slurm exit is not enough to authorize training:
this audit proves that those representations agree, rebinds every selected
candidate to the frozen corrected co-visibility teacher, and enforces the
scene-role boundary from the immutable split manifest.

Ground-truth pose/error columns are audited as targets only.  The downstream
trainer has its own explicit allow-list of deployment-time input features so
that these columns can never silently leak into the model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Optional

import numpy as np

try:
    from MemNavData.external_causal_scale_contract import (
        ExternalCausalScaleContract,
        ExternalCausalScaleError,
        ExternalCausalScalePins,
        validate_external_causal_frame,
    )
except ModuleNotFoundError:  # direct script invocation
    from external_causal_scale_contract import (  # type: ignore
        ExternalCausalScaleContract,
        ExternalCausalScaleError,
        ExternalCausalScalePins,
        validate_external_causal_frame,
    )


CHECKPOINT_NAME = "lingbot_goal_loop_closure_checkpoint.sqlite3"
CSV_NAME = "lingbot_goal_loop_closure_rows.csv"
PROGRESS_NAME = "lingbot_goal_loop_closure_progress.json"
REPORT_NAME = "diagnostic_lingbot_goal_loop_closure.json"
AUDIT_NAME = "lingbot_native_training_artifact_audit.json"

ROW_KEY = ("scene", "session_id", "candidate_frame")
REQUIRED_ROW_COLUMNS = {
    *ROW_KEY,
    "episode",
    "kind",
    "query_path",
    "candidate_path",
    "label",
    "session_has_positive",
    "session_is_strict_no_match",
    "session_max_covis",
    "teacher_covis",
    "dino_cosine",
    "metric_scale_m_per_raw",
    "metric_scale_source",
    "depth_scale_raw",
    "cloud_overlap_f1_center",
    "anchor_goal_distance_norm_center",
    "goal_refine_translation_norm_median",
    "goal_refine_rotation_deg_median",
    "predicted_relative_xy_m_center_json",
    "target_relative_xy_m_center_json",
    "goal_depth_confidence_mean",
    "candidate_depth_confidence_mean",
}
REQUIRED_TEACHER_COLUMNS = {
    *ROW_KEY,
    "candidate_path",
    "teacher_covis",
    "dino_cosine",
}
FINITE_SCALAR_COLUMNS = (
    "dino_cosine",
    "metric_scale_m_per_raw",
    "depth_scale_raw",
    "cloud_overlap_f1_center",
    "anchor_goal_distance_norm_center",
    "goal_refine_translation_norm_median",
    "goal_refine_rotation_deg_median",
    "goal_depth_confidence_mean",
    "candidate_depth_confidence_mean",
)
SELECTION_ORIGIN_COLUMN = "candidate_selection_origin"
FORMAL_TRAIN_SELECTION_ORIGINS = frozenset({
    "deployment_topk",
    "teacher_forced_positive",
    "teacher_forced_hard_negative",
})


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    if path.exists() or sidecar.exists():
        if (not path.is_file() or path.is_symlink()
                or not sidecar.is_file() or sidecar.is_symlink()):
            raise RuntimeError("existing audit/sidecar pair is incomplete")
        if path.read_bytes() != payload:
            raise RuntimeError("existing artifact audit differs")
        if sidecar.read_bytes() != sidecar_payload:
            raise RuntimeError("existing artifact audit sidecar differs")
        return
    temporaries: list[tuple[Path, bytes]] = []
    try:
        for destination, content in (
                (path, payload), (sidecar, sidecar_payload)):
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent)
            temporary_path = Path(temporary)
            temporaries.append((temporary_path, content))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporaries[0][0], path)
        os.replace(temporaries[1][0], sidecar)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        for temporary_path, _content in temporaries:
            temporary_path.unlink(missing_ok=True)


def load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def parse_xy(value: Any, name: str) -> np.ndarray:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name} JSON: {value!r}") from error
    array = np.asarray(decoded, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite length-two vector")
    return array


def checkpoint_contents(path: Path) -> tuple[dict, dict, list[dict]]:
    """Return metadata, aggregate stats and ordered row payloads."""
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"metadata", "rows", "completed_sessions"}
        if missing := required - tables:
            raise RuntimeError(
                f"checkpoint is missing tables: {sorted(missing)}")
        metadata = dict(connection.execute(
            "SELECT key, value FROM metadata").fetchall())
        payloads = [
            json.loads(payload) for (payload,) in connection.execute(
                "SELECT payload_json FROM rows ORDER BY seed_index")
        ]
        completed = connection.execute(
            "SELECT session_id, expected_seed_count, row_count "
            "FROM completed_sessions ORDER BY first_seed_index").fetchall()
        actual_rows = dict(connection.execute(
            "SELECT session_id, COUNT(*) FROM rows GROUP BY session_id")
            .fetchall())
        for session_id, _expected, declared_rows in completed:
            if int(actual_rows.get(session_id, 0)) != int(declared_rows):
                raise RuntimeError(
                    f"checkpoint row_count mismatch for {session_id}")
        stats = {
            "completed_sessions": len(completed),
            "completed_seeds": int(sum(int(row[1]) for row in completed)),
            "saved_rows": len(payloads),
            "zero_row_sessions": int(sum(int(row[2]) == 0 for row in completed)),
        }
        return metadata, stats, payloads
    finally:
        connection.close()


def _record(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def corrected_teacher_alignment(
    rows,
    teacher,
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> dict:
    """Verify selected rows against the corrected full teacher table."""
    missing_rows = REQUIRED_ROW_COLUMNS - set(rows.columns)
    missing_teacher = REQUIRED_TEACHER_COLUMNS - set(teacher.columns)
    if missing_rows:
        raise ValueError(f"rows CSV missing columns: {sorted(missing_rows)}")
    if missing_teacher:
        raise ValueError(
            f"teacher CSV missing columns: {sorted(missing_teacher)}")
    if rows.duplicated(list(ROW_KEY)).any():
        raise ValueError("rows CSV contains duplicate candidate keys")
    if teacher.duplicated(list(ROW_KEY)).any():
        duplicate = teacher.loc[
            teacher.duplicated(list(ROW_KEY), keep=False), list(ROW_KEY)]
        raise ValueError(
            "corrected teacher candidate keys are not unique: "
            f"{duplicate.head(3).to_dict('records')}")

    teacher_view = teacher[[
        *ROW_KEY, "teacher_covis", "dino_cosine", "candidate_path"
    ]].rename(columns={
        "teacher_covis": "corrected_teacher_covis",
        "dino_cosine": "corrected_dino_cosine",
        "candidate_path": "corrected_candidate_path",
    })
    joined = rows.merge(
        teacher_view, on=list(ROW_KEY), how="left", validate="one_to_one",
        indicator=True)
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[
            ~joined["_merge"].eq("both"), list(ROW_KEY)]
        raise ValueError(
            "selected candidates absent from corrected teacher: "
            f"{missing.head(5).to_dict('records')}")
    stored_covis = joined["teacher_covis"].to_numpy(dtype=np.float64)
    corrected_covis = joined["corrected_teacher_covis"].to_numpy(
        dtype=np.float64)
    if (not np.isfinite(stored_covis).all()
            or not np.isfinite(corrected_covis).all()):
        raise ValueError("teacher co-visibility contains non-finite values")
    covis_error = float(np.max(np.abs(stored_covis - corrected_covis)))
    if covis_error > 1e-9:
        raise ValueError(
            f"corrected teacher co-visibility changed by {covis_error}")
    dino_error = float(np.max(np.abs(
        joined["dino_cosine"].to_numpy(dtype=np.float64)
        - joined["corrected_dino_cosine"].to_numpy(dtype=np.float64))))
    if dino_error > 5e-5:
        raise ValueError(f"DINO alignment changed by {dino_error}")

    expected_label = np.where(
        corrected_covis >= positive_threshold, 1,
        np.where(corrected_covis <= negative_threshold, 0, -1))
    stored_label = joined["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(stored_label, expected_label):
        raise ValueError("stored row labels differ from corrected teacher")

    session_key = ["scene", "session_id"]
    teacher_maximum = teacher.groupby(
        session_key, sort=False, as_index=False)["teacher_covis"].max()
    teacher_maximum = teacher_maximum.rename(columns={
        "teacher_covis": "corrected_session_max_covis",
    })
    joined = joined.merge(
        teacher_maximum, on=session_key, how="left", validate="many_to_one")
    expected_max = joined["corrected_session_max_covis"].to_numpy(
        dtype=np.float64)
    stored_max = joined["session_max_covis"].to_numpy(dtype=np.float64)
    maximum_error = float(np.max(np.abs(expected_max - stored_max)))
    if maximum_error > 1e-9:
        raise ValueError(
            f"session maximum co-visibility changed by {maximum_error}")
    stored_has_positive = np.asarray([
        parse_bool(value) for value in joined["session_has_positive"]
    ])
    stored_no_match = np.asarray([
        parse_bool(value) for value in joined["session_is_strict_no_match"]
    ])
    if not np.array_equal(
            stored_has_positive, expected_max >= positive_threshold):
        raise ValueError("session_has_positive differs from corrected teacher")
    if not np.array_equal(
            stored_no_match, expected_max <= negative_threshold):
        raise ValueError(
            "session_is_strict_no_match differs from corrected teacher")

    selected_positive = joined["label"].eq(1).groupby([
        joined["scene"], joined["session_id"]]).any()
    strict_no_match = joined.drop_duplicates(session_key)[
        "session_is_strict_no_match"].map(parse_bool)
    return {
        "rows_rebound": int(len(joined)),
        "maximum_covisibility_error": covis_error,
        "maximum_dino_error": dino_error,
        "maximum_session_covisibility_error": maximum_error,
        "positive_rows": int(np.sum(stored_label == 1)),
        "negative_rows": int(np.sum(stored_label == 0)),
        "ambiguous_rows": int(np.sum(stored_label == -1)),
        "selected_positive_sessions": int(selected_positive.sum()),
        "strict_no_match_sessions": int(strict_no_match.sum()),
    }


def validate_formal_selection_policy(rows, teacher, signature,
                                     *, expected_role: str) -> dict:
    """Prove train-only teacher augmentation never leaks into development."""

    config = signature.get("compute_config")
    if not isinstance(config, dict):
        raise ValueError("collector compute_config is malformed")
    mode = config.get("selection_mode")
    top_k = config.get("top_k")
    kind = config.get("kind")
    if (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
            or not isinstance(kind, str) or not kind):
        raise ValueError("collector selection top_k/kind is malformed")
    if SELECTION_ORIGIN_COLUMN not in rows.columns:
        raise ValueError(
            f"formal rows lack {SELECTION_ORIGIN_COLUMN}")
    if not {"kind", "split_role"}.issubset(teacher.columns):
        raise ValueError("formal teacher lacks kind/split_role")
    teacher_kind = teacher.loc[
        teacher["kind"].eq(kind)
        & teacher["split_role"].eq(expected_role)].copy()
    if teacher_kind.empty:
        raise ValueError(f"formal teacher has no rows for kind={kind}")

    row_sessions = set(zip(
        rows["scene"].astype(str), rows["session_id"].astype(str)))
    teacher_sessions = set(zip(
        teacher_kind["scene"].astype(str),
        teacher_kind["session_id"].astype(str)))
    if row_sessions != teacher_sessions:
        raise ValueError(
            "formal selection does not exactly cover teacher sessions")
    counts = rows.groupby(["scene", "session_id"], sort=False).size()
    origins = set(rows[SELECTION_ORIGIN_COLUMN].astype(str))

    positive_threshold = float(config.get("positive_threshold", 0.5))
    if expected_role == "train":
        if mode != "train_augmented":
            raise ValueError(
                "formal train collection must use train_augmented selection")
        if not origins.issubset(FORMAL_TRAIN_SELECTION_ORIGINS):
            raise ValueError(
                f"formal train selection has forbidden origins: "
                f"{sorted(origins - FORMAL_TRAIN_SELECTION_ORIGINS)}")
        if int(counts.max()) > top_k + 2:
            raise ValueError("formal train selection exceeds top_k+2 rows")
        teacher_positive = teacher_kind.groupby(
            ["scene", "session_id"], sort=False)["teacher_covis"].max()
        teacher_positive = set(teacher_positive.loc[
            teacher_positive.ge(positive_threshold)].index.tolist())
        selected_positive = rows.loc[
            rows["teacher_covis"].ge(positive_threshold)].groupby(
                ["scene", "session_id"], sort=False).size()
        selected_positive_sessions = set(selected_positive.index.tolist())
        missing_positive = teacher_positive - selected_positive_sessions
        if missing_positive:
            raise ValueError(
                "train augmentation failed to expose teacher positives: "
                f"{sorted(missing_positive)[:5]}")
    elif expected_role == "development":
        if mode != "deployment":
            raise ValueError(
                "formal development collection must use deployment selection")
        if origins != {"deployment_topk"}:
            raise ValueError(
                "development contains train-only teacher augmentation")
        if int(counts.max()) > top_k:
            raise ValueError("formal development selection exceeds top_k rows")
        teacher_positive = teacher_kind.groupby(
            ["scene", "session_id"], sort=False)["teacher_covis"].max()
        teacher_positive = set(teacher_positive.loc[
            teacher_positive.ge(positive_threshold)].index.tolist())
        selected_positive_sessions = set(rows.loc[
            rows["teacher_covis"].ge(positive_threshold),
            ["scene", "session_id"]].itertuples(index=False, name=None))
    else:
        raise ValueError(
            "formal Phase-B selection is restricted to train/development")

    positive_recall = (
        len(teacher_positive & selected_positive_sessions)
        / len(teacher_positive) if teacher_positive else None)
    return {
        "approved": True,
        "selection_mode": mode,
        "top_k": top_k,
        "session_count": len(row_sessions),
        "maximum_rows_per_session": int(counts.max()),
        "selection_origin_counts": {
            str(name): int(count)
            for name, count in rows[
                SELECTION_ORIGIN_COLUMN].value_counts().sort_index().items()
        },
        "teacher_positive_sessions": len(teacher_positive),
        "selected_positive_sessions": len(
            teacher_positive & selected_positive_sessions),
        "positive_session_recall": positive_recall,
        "development_uses_teacher_augmentation": False,
    }


def audit_artifact(
    run_dir: Path,
    teacher_csv: Path,
    split_manifest: Path,
    *,
    expected_role: str,
    expected_scenes: Optional[int] = None,
    expected_sessions: Optional[int] = None,
    expected_seeds: Optional[int] = None,
    expected_rows: Optional[int] = None,
    expected_source_commit: Optional[str] = None,
    expected_teacher_sha256: Optional[str] = None,
    expected_split_sha256: Optional[str] = None,
    out_report: Optional[Path] = None,
    raise_on_failure: bool = True,
) -> dict:
    import pandas as pd

    run_dir = Path(run_dir).resolve()
    teacher_csv = Path(teacher_csv).resolve()
    split_manifest = Path(split_manifest).resolve()
    out_report = Path(out_report or run_dir / AUDIT_NAME)
    errors: list[str] = []
    required_paths = {
        "checkpoint": run_dir / CHECKPOINT_NAME,
        "rows_csv": run_dir / CSV_NAME,
        "progress": run_dir / PROGRESS_NAME,
        "collector_report": run_dir / REPORT_NAME,
        "teacher_csv": teacher_csv,
        "split_manifest": split_manifest,
    }
    for name, path in required_paths.items():
        _record(errors, path.is_file(), f"missing {name}: {path}")
    if errors:
        result = {
            "training_artifact_approved": False,
            "errors": errors,
            "run_dir": str(run_dir),
        }
        atomic_write_json(out_report, result)
        if raise_on_failure:
            raise RuntimeError("; ".join(errors))
        return result

    teacher_sha = sha256(teacher_csv)
    split_sha = sha256(split_manifest)
    if expected_teacher_sha256:
        _record(errors, teacher_sha == expected_teacher_sha256,
                "corrected teacher SHA mismatch")
    if expected_split_sha256:
        _record(errors, split_sha == expected_split_sha256,
                "split manifest SHA mismatch")

    progress = load_json(required_paths["progress"])
    collector_report = load_json(required_paths["collector_report"])
    split = load_json(split_manifest)
    metadata, checkpoint_stats, payloads = checkpoint_contents(
        required_paths["checkpoint"])
    rows = pd.read_csv(required_paths["rows_csv"])
    checkpoint_rows = pd.DataFrame(payloads)
    try:
        signature = json.loads(metadata["signature_json"])
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("checkpoint signature metadata is invalid") from error
    signature_sha = hashlib.sha256(
        canonical_json(signature).encode("utf-8")).hexdigest()

    _record(errors, progress.get("status") == "complete",
            "collector progress is not complete")
    _record(errors, int(progress.get("completed_sessions", -1))
            == checkpoint_stats["completed_sessions"],
            "progress/checkpoint completed-session mismatch")
    _record(errors, int(progress.get("completed_seeds", -1))
            == checkpoint_stats["completed_seeds"],
            "progress/checkpoint completed-seed mismatch")
    _record(errors, int(progress.get("saved_rows", -1))
            == checkpoint_stats["saved_rows"],
            "progress/checkpoint row mismatch")
    _record(errors, progress.get("signature_sha256") == signature_sha,
            "progress/checkpoint signature mismatch")
    _record(errors, len(rows) == checkpoint_stats["saved_rows"],
            "CSV/checkpoint row-count mismatch")
    _record(errors, len(checkpoint_rows) == len(rows),
            "checkpoint payload/CSV row-count mismatch")
    if len(rows) == len(checkpoint_rows) and len(rows):
        missing_payload_columns = set(rows.columns) - set(
            checkpoint_rows.columns)
        _record(errors, not missing_payload_columns,
                "checkpoint payload missing CSV columns: "
                f"{sorted(missing_payload_columns)}")
        if not missing_payload_columns:
            for column in rows.columns:
                left = rows[column].reset_index(drop=True)
                right = checkpoint_rows[column].reset_index(drop=True)
                left_missing = left.isna().to_numpy()
                right_missing = right.isna().to_numpy()
                equal = np.array_equal(left_missing, right_missing)
                present = ~left_missing
                if equal and present.any():
                    left_number = pd.to_numeric(
                        left[present], errors="coerce").to_numpy()
                    right_number = pd.to_numeric(
                        right[present], errors="coerce").to_numpy()
                    numeric = (
                        np.isfinite(left_number).all()
                        and np.isfinite(right_number).all())
                    if numeric:
                        equal = bool(np.allclose(
                            left_number, right_number,
                            rtol=1e-10, atol=1e-12))
                    else:
                        equal = (
                            left[present].astype(str).tolist()
                            == right[present].astype(str).tolist())
                _record(
                    errors, equal,
                    f"checkpoint payload/CSV content mismatch in {column}")

    report_rows = int(collector_report.get("n_rows", -1))
    report_sessions = int(collector_report.get("n_sessions", -1))
    _record(errors, report_rows == len(rows),
            "collector report/CSV row mismatch")
    _record(errors, report_sessions == rows["session_id"].nunique(),
            "collector report/CSV session mismatch")

    if expected_sessions is not None:
        _record(errors,
                checkpoint_stats["completed_sessions"] == expected_sessions,
                f"expected {expected_sessions} completed sessions")
    if expected_seeds is not None:
        _record(errors, checkpoint_stats["completed_seeds"] == expected_seeds,
                f"expected {expected_seeds} completed seeds")
    if expected_rows is not None:
        _record(errors, len(rows) == expected_rows,
                f"expected {expected_rows} saved rows")
    _record(errors, checkpoint_stats["zero_row_sessions"] == 0,
            "one or more completed sessions produced zero rows")

    provenance = signature.get("provenance", {})
    report_provenance = collector_report.get("provenance", {})
    _record(errors, provenance.get("teacher_csv_sha256") == teacher_sha,
            "checkpoint uses a different teacher SHA")
    _record(errors, report_provenance.get("teacher_csv_sha256") == teacher_sha,
            "collector report uses a different teacher SHA")
    _record(errors, provenance.get("split_manifest_sha256") == split_sha,
            "checkpoint uses a different split SHA")
    _record(errors, report_provenance.get("split_manifest_sha256") == split_sha,
            "collector report uses a different split SHA")
    if expected_source_commit:
        _record(errors, provenance.get("source_commit")
                == expected_source_commit,
                "checkpoint source commit mismatch")
        _record(errors, report_provenance.get("source_commit")
                == expected_source_commit,
                "collector report source commit mismatch")

    allowed_scenes = set(split.get(expected_role, []))
    selected_scenes = set(rows["scene"].astype(str))
    _record(errors, bool(allowed_scenes),
            f"split manifest has no {expected_role} scenes")
    _record(errors, selected_scenes.issubset(allowed_scenes),
            f"rows contain scenes outside {expected_role}")
    if expected_scenes is not None:
        _record(errors, len(selected_scenes) == expected_scenes,
                f"expected {expected_scenes} unique scenes")
        _record(errors, selected_scenes == allowed_scenes,
                f"rows do not cover the complete {expected_role} scene set")

    config = signature.get("compute_config", {})
    positive_threshold = float(config.get("positive_threshold", 0.5))
    negative_threshold = float(config.get("negative_threshold", 0.2))
    teacher = None
    try:
        teacher = pd.read_csv(teacher_csv)
        teacher_alignment = corrected_teacher_alignment(
            rows, teacher,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold)
    except (ValueError, KeyError, TypeError) as error:
        errors.append(f"teacher alignment failed: {error}")
        teacher_alignment = {}

    missing_columns = REQUIRED_ROW_COLUMNS - set(rows.columns)
    _record(errors, not missing_columns,
            f"rows CSV missing columns: {sorted(missing_columns)}")
    finite_rates = {}
    if not missing_columns:
        for column in FINITE_SCALAR_COLUMNS:
            values = rows[column].to_numpy(dtype=np.float64)
            finite_rates[column] = float(np.isfinite(values).mean())
            _record(errors, np.isfinite(values).all(),
                    f"non-finite deployment feature: {column}")
        for column in (
                "predicted_relative_xy_m_center_json",
                "target_relative_xy_m_center_json"):
            try:
                vectors = np.stack([
                    parse_xy(value, column) for value in rows[column]
                ])
                finite_rates[column] = float(np.isfinite(vectors).all(axis=1).mean())
            except ValueError as error:
                errors.append(str(error))

    signature_external = provenance.get("external_causal_scale")
    selection_policy = {
        "approved": False,
        "reason": "legacy artifact without external causal-scale contract",
    }
    if isinstance(signature_external, dict):
        try:
            if teacher is None:
                raise ValueError("formal teacher could not be loaded")
            selection_policy = validate_formal_selection_policy(
                rows, teacher, signature, expected_role=expected_role)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"formal selection policy failed: {error}")
            selection_policy = {
                "approved": False,
                "reason": str(error),
            }
    external_contract = None
    expected_external_samples = None
    expected_external_bindings = None
    try:
        if isinstance(signature_external, dict):
            external_contract = ExternalCausalScaleContract(
                manifest_path=Path(str(signature_external["manifest_path"])),
                artifact_path=Path(str(signature_external["artifact_path"])),
                pins=ExternalCausalScalePins(
                    manifest_sha256=str(signature_external["manifest_sha256"]),
                    artifact_sha256=str(signature_external["artifact_sha256"]),
                    producer_sha256=str(
                        signature_external["producer_source_sha256"]),
                    configuration_sha256=str(
                        signature_external["configuration_sha256"]),
                    lingbot_commit=str(signature_external["lingbot_commit"]),
                    weights_sha256=str(signature_external["weights_sha256"]),
                    stream_source_sha256=str(
                        signature_external["stream_source_sha256"]),
                ),
            )
            _record(errors, external_contract.summary() == signature_external,
                    "physical external-scale contract differs from signature")
            expected_external_samples = external_contract.selected_sample_ids(
                split_role=expected_role, goal_roles=("B", "C"))
            expected_external_bindings = {
                sample_id: external_contract.expected_row_binding(sample_id)
                for sample_id in expected_external_samples
            }
        external_causal_scale = validate_external_causal_frame(
            rows,
            expected_sample_ids=expected_external_samples,
            expected_split_role=expected_role,
            expected_row_bindings=expected_external_bindings,
        )
    except (ExternalCausalScaleError, KeyError, TypeError, ValueError) as error:
        errors.append(f"external causal-scale contract failed: {error}")
        external_causal_scale = {
            "approved": False,
            "external_rows": 0,
            "reason": str(error),
        }
    if external_causal_scale.get("external_rows", 0):
        _record(errors,
                external_causal_scale.get("split_roles") == [expected_role],
                "external causal-scale row role differs from audit role")
        report_external = report_provenance.get("external_causal_scale")
        _record(errors, isinstance(signature_external, dict),
                "checkpoint lacks external causal-scale provenance")
        _record(errors, report_external == signature_external,
                "collector report/checkpoint external-scale provenance differs")
        if isinstance(signature_external, dict):
            _record(errors,
                    config.get("metric_scale_mode")
                    == "external_causal_first_prefix_v1",
                    "checkpoint metric-scale mode is not external causal")
            _record(errors,
                    external_causal_scale.get("manifest_sha256")
                    == [signature_external.get("manifest_sha256")],
                    "row/collector causal manifest SHA differs")
            _record(errors,
                    external_causal_scale.get("scale_artifact_sha256")
                    == [signature_external.get("artifact_sha256")],
                    "row/collector external scale artifact SHA differs")
            _record(errors,
                    external_causal_scale.get(
                        "exact_manifest_sample_coverage_approved") is True,
                    "rows do not exactly cover selected manifest samples")
            for row_field, summary_field in (
                    ("producer_source_sha256", "producer_source_sha256"),
                    ("configuration_sha256", "configuration_sha256"),
                    ("lingbot_commit", "lingbot_commit"),
                    ("weights_sha256", "weights_sha256"),
                    ("stream_source_sha256", "stream_source_sha256"),
                    ("manifest_schema_version", "manifest_schema_version")):
                _record(
                    errors,
                    external_causal_scale.get(row_field)
                    == signature_external.get(summary_field),
                    f"row/collector external scale {row_field} differs",
                )

    _record(errors, teacher_alignment.get("positive_rows", 0) > 0,
            "artifact has no positive candidates")
    _record(errors, teacher_alignment.get("negative_rows", 0) > 0,
            "artifact has no negative candidates")
    _record(errors,
            teacher_alignment.get("selected_positive_sessions", 0) > 0,
            "artifact has no session with a selected positive")
    _record(errors, teacher_alignment.get("strict_no_match_sessions", 0) > 0,
            "artifact has no strict no-match sessions")

    identity_payload = {
        "rows_csv_sha256": sha256(required_paths["rows_csv"]),
        "checkpoint_signature_sha256": signature_sha,
        "teacher_csv_sha256": teacher_sha,
        "split_manifest_sha256": split_sha,
        "external_causal_scale_contract_sha256": hashlib.sha256(
            canonical_json(external_causal_scale).encode("utf-8")).hexdigest(),
    }
    result = {
        "training_artifact_approved": not errors,
        "errors": errors,
        "run_dir": str(run_dir),
        "role": expected_role,
        "counts": {
            **checkpoint_stats,
            "csv_rows": int(len(rows)),
            "csv_sessions": int(rows["session_id"].nunique()),
            "scenes": len(selected_scenes),
        },
        "teacher_alignment": teacher_alignment,
        "selection_policy": selection_policy,
        "finite_rates": finite_rates,
        "external_causal_scale": external_causal_scale,
        "provenance": {
            **identity_payload,
            "source_commit": provenance.get("source_commit"),
            "lingbot_commit": provenance.get("lingbot_commit"),
            "lingbot_weight_sha256": provenance.get(
                "lingbot_weight_sha256"),
        },
        "artifact_identity_sha256": hashlib.sha256(
            canonical_json(identity_payload).encode("utf-8")).hexdigest(),
    }
    atomic_write_json(out_report, result)
    if errors and raise_on_failure:
        raise RuntimeError("artifact audit failed: " + "; ".join(errors))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-role", choices=("train", "development", "final_reserved"),
        required=True)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--expected-sessions", type=int)
    parser.add_argument("--expected-seeds", type=int)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--out-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for value, name in (
            (args.expected_scenes, "scenes"),
            (args.expected_sessions, "sessions"),
            (args.expected_seeds, "seeds"),
            (args.expected_rows, "rows")):
        if value is not None and value < 1:
            raise ValueError(f"expected {name} must be positive")
    result = audit_artifact(
        args.run_dir,
        args.teacher_csv,
        args.split_manifest,
        expected_role=args.expected_role,
        expected_scenes=args.expected_scenes,
        expected_sessions=args.expected_sessions,
        expected_seeds=args.expected_seeds,
        expected_rows=args.expected_rows,
        expected_source_commit=args.expected_source_commit,
        expected_teacher_sha256=args.expected_teacher_sha256,
        expected_split_sha256=args.expected_split_sha256,
        out_report=args.out_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
