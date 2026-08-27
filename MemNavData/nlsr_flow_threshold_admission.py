#!/usr/bin/env python3
"""Plan monotone flow-threshold retries under the pinned cache budget.

This CPU helper is called before every GPU precompute attempt.  It validates
any existing cache pair with the exact merger provenance checks, atomically
records the threshold trajectory, and returns one of three actions:

* ``compute`` the episode's fixed minimum threshold when no cache exists;
* ``overwrite`` the same tier after an interrupted write, or exactly the next
  approved tier after a complete over-budget cache; or
* ``accept`` an exact, byte-hash-matching compliant cache without recomputing.

The decision uses only anchor count, scale-frame count, cache provenance and
the pinned schema budget.  It never reads an evaluation label.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from MemNavData.build_nlsr_merged_flow import (
        FLOW_FILES,
        FLOW_THRESHOLD_ADMISSION_SCHEMA,
        FLOW_THRESHOLD_TIERS,
        FlowAuditError,
        _admission_filename,
        _cache_file_hashes,
        _load_cache_schema,
        _patch_budget_compliant,
        _schema_keyframe_budget,
        _threshold_index,
        _validate_pair,
        canonical_bytes,
        validate_threshold_admission_record,
    )
except ImportError:  # direct ``python MemNavData/<script>.py`` execution
    from build_nlsr_merged_flow import (  # type: ignore
        FLOW_FILES,
        FLOW_THRESHOLD_ADMISSION_SCHEMA,
        FLOW_THRESHOLD_TIERS,
        FlowAuditError,
        _admission_filename,
        _cache_file_hashes,
        _load_cache_schema,
        _patch_budget_compliant,
        _schema_keyframe_budget,
        _threshold_index,
        _validate_pair,
        canonical_bytes,
        validate_threshold_admission_record,
    )


SELECTION_BASIS = (
    "lowest approved flow threshold whose anchor_count + "
    "num_scale_frames fits the pinned cache-schema keyframe budget; "
    "no evaluation label"
)


class ThresholdAdmissionError(RuntimeError):
    """A cache or journal cannot support a safe monotone admission decision."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThresholdAdmissionError(message)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        path.parent.is_dir() and not path.parent.is_symlink(), "journal parent invalid"
    )
    if path.is_symlink():
        raise ThresholdAdmissionError(f"journal is a symlink: {path}")
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
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_journal(
    path: Path,
    *,
    episode: str,
    minimum_threshold: float,
    keyframe_budget: int,
) -> dict | None:
    if not os.path.lexists(path):
        return None
    _require(
        path.is_file() and not path.is_symlink(), f"journal is not physical: {path}"
    )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ThresholdAdmissionError(f"cannot load journal: {path}") from error
    _require(raw == canonical_bytes(value), f"journal is not canonical: {path}")
    try:
        return validate_threshold_admission_record(
            value,
            episode=episode,
            minimum_threshold=minimum_threshold,
            keyframe_budget=keyframe_budget,
        )
    except FlowAuditError as error:
        raise ThresholdAdmissionError(f"invalid threshold journal: {error}") from error


def _new_record(
    episode: str,
    minimum_threshold: float,
    keyframe_budget: int,
) -> dict:
    minimum = FLOW_THRESHOLD_TIERS[
        _threshold_index(minimum_threshold, "minimum threshold")
    ]
    return {
        "schema_version": FLOW_THRESHOLD_ADMISSION_SCHEMA,
        "episode": episode,
        "minimum_threshold": minimum,
        "threshold_tiers": list(FLOW_THRESHOLD_TIERS),
        "keyframe_budget": keyframe_budget,
        "num_scale_frames": 8,
        "selection_basis": SELECTION_BASIS,
        "status": "pending",
        "selected_threshold": None,
        "attempts": [],
    }


def _pending_attempt(threshold: float, action: str) -> dict:
    return {
        "threshold": threshold,
        "action": action,
        "outcome": "pending",
        "anchor_count": None,
        "total_memory_frames": None,
        "precompute_signature": None,
        "cache_files_sha256": None,
    }


def _observed_attempt(threshold: float, action: str, validation: dict) -> dict:
    return {
        "threshold": threshold,
        "action": action,
        "outcome": (
            "accepted"
            if validation["strict_patch_keyframe_budget_compliant"]
            else "over_budget"
        ),
        "anchor_count": validation["anchor_count"],
        "total_memory_frames": validation["total_memory_frames"],
        "precompute_signature": validation["precompute_signature"],
        "cache_files_sha256": _cache_file_hashes(validation),
    }


def _validation_matches_attempt(validation: dict, attempt: Mapping[str, Any]) -> bool:
    try:
        return (
            _threshold_index(validation["flow_threshold"], "cache threshold")
            == _threshold_index(attempt["threshold"], "attempt threshold")
            and validation["anchor_count"] == attempt["anchor_count"]
            and validation["total_memory_frames"] == attempt["total_memory_frames"]
            and validation["precompute_signature"] == attempt["precompute_signature"]
            and _cache_file_hashes(validation) == attempt["cache_files_sha256"]
        )
    except (FlowAuditError, KeyError):
        return False


def _pair_paths(patch_flow_root: Path, episode: str) -> tuple[Path, Path]:
    chunk = patch_flow_root / Path(episode) / "videos/chunk-000"
    return chunk / FLOW_FILES[0], chunk / FLOW_FILES[1]


def _inspect_pair(
    cache_schema: Any,
    patch_flow_root: Path,
    episode: str,
    frames: int,
) -> tuple[str, dict | None, str | None]:
    aggregator, camera = _pair_paths(patch_flow_root, episode)
    states = [os.path.lexists(path) for path in (aggregator, camera)]
    if not any(states):
        return "absent", None, None
    if not all(states):
        return "invalid", None, "partial cache pair"
    try:
        validation = _validate_pair(
            cache_schema,
            aggregator,
            camera,
            frames=frames,
            threshold=None,
            patch=True,
            require_budget=False,
        )
    except FlowAuditError as error:
        return "invalid", None, str(error)
    return "valid", validation, None


def _declared_partial_thresholds(
    patch_flow_root: Path,
    episode: str,
) -> list[float]:
    """Read any independently intact threshold scalar from an invalid pair."""

    result = []
    for path in _pair_paths(patch_flow_root, episode):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as cache:
                value = np.asarray(cache["flow_threshold"])
                if value.size != 1:
                    continue
                threshold = float(value.reshape(-1)[0])
            _threshold_index(threshold, f"partial cache threshold at {path}")
            result.append(threshold)
        except FlowAuditError as error:
            raise ThresholdAdmissionError(
                f"partial cache declares an unapproved threshold: {path}"
            ) from error
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return result


def _finish_observation(
    record: dict,
    validation: dict,
    keyframe_budget: int,
) -> None:
    pending = record["attempts"][-1]
    threshold = FLOW_THRESHOLD_TIERS[
        _threshold_index(pending["threshold"], "pending threshold")
    ]
    observed = _observed_attempt(threshold, pending["action"], validation)
    record["attempts"][-1] = observed
    if observed["outcome"] == "accepted":
        record["status"] = "accepted"
        record["selected_threshold"] = threshold
        return
    current_index = _threshold_index(threshold, "over-budget threshold")
    if current_index == len(FLOW_THRESHOLD_TIERS) - 1:
        record["status"] = "exhausted"
        record["selected_threshold"] = None
        return
    _require(
        not _patch_budget_compliant(observed["anchor_count"], 8, keyframe_budget),
        "attempt marked over-budget despite fitting the budget",
    )
    record["attempts"].append(
        _pending_attempt(FLOW_THRESHOLD_TIERS[current_index + 1], "overwrite")
    )
    record["status"] = "pending"
    record["selected_threshold"] = None


def _decision(record: dict, journal: Path, cache_state: str) -> dict:
    if record["status"] == "accepted":
        return {
            "action": "accept",
            "episode": record["episode"],
            "threshold": record["selected_threshold"],
            "overwrite": False,
            "cache_state": cache_state,
            "journal": str(journal),
            "attempt_count": len(record["attempts"]),
        }
    if record["status"] == "exhausted":
        raise ThresholdAdmissionError(
            f"all approved thresholds exceed the keyframe budget for "
            f"{record['episode']}"
        )
    attempt = record["attempts"][-1]
    action = attempt["action"]
    _require(action in {"compute", "overwrite"}, "pending action cannot be executed")
    return {
        "action": action,
        "episode": record["episode"],
        "threshold": attempt["threshold"],
        "overwrite": action == "overwrite",
        "cache_state": cache_state,
        "journal": str(journal),
        "attempt_count": len(record["attempts"]),
    }


def plan_threshold_admission(
    *,
    episode: str,
    frames: int,
    minimum_threshold: float,
    patch_flow_root: Path | str,
    admission_root: Path | str,
    cache_schema: Any,
) -> dict:
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ThresholdAdmissionError("frames must be a positive integer")
    try:
        minimum_index = _threshold_index(minimum_threshold, "minimum threshold")
        keyframe_budget = _schema_keyframe_budget(cache_schema)
        filename = _admission_filename(episode)
    except FlowAuditError as error:
        raise ThresholdAdmissionError(str(error)) from error
    patch_root = Path(patch_flow_root)
    journal_root = Path(admission_root)
    _require(
        patch_root.is_dir() and not patch_root.is_symlink(),
        f"patch flow root is invalid: {patch_root}",
    )
    journal_root.mkdir(parents=True, exist_ok=True)
    _require(
        journal_root.is_dir() and not journal_root.is_symlink(),
        f"admission root is invalid: {journal_root}",
    )
    journal = journal_root / filename
    record = _load_journal(
        journal,
        episode=episode,
        minimum_threshold=minimum_threshold,
        keyframe_budget=keyframe_budget,
    )
    cache_state, validation, invalid_reason = _inspect_pair(
        cache_schema, patch_root, episode, frames
    )

    if record is None:
        record = _new_record(episode, minimum_threshold, keyframe_budget)
        if cache_state == "absent":
            record["attempts"].append(
                _pending_attempt(FLOW_THRESHOLD_TIERS[minimum_index], "compute")
            )
        elif cache_state == "invalid":
            raise ThresholdAdmissionError(
                "untracked existing cache is invalid; refusing an unproven overwrite: "
                f"{invalid_reason}"
            )
        else:
            assert validation is not None
            actual_index = _threshold_index(
                validation["flow_threshold"], "existing cache threshold"
            )
            _require(
                actual_index == minimum_index,
                "untracked existing cache is not at the episode minimum threshold",
            )
            action = (
                "resume"
                if validation["strict_patch_keyframe_budget_compliant"]
                else "inspect_existing"
            )
            record["attempts"].append(
                _observed_attempt(
                    FLOW_THRESHOLD_TIERS[actual_index], action, validation
                )
            )
            if validation["strict_patch_keyframe_budget_compliant"]:
                record["status"] = "accepted"
                record["selected_threshold"] = FLOW_THRESHOLD_TIERS[actual_index]
            elif actual_index == len(FLOW_THRESHOLD_TIERS) - 1:
                record["status"] = "exhausted"
            else:
                record["attempts"].append(
                    _pending_attempt(
                        FLOW_THRESHOLD_TIERS[actual_index + 1], "overwrite"
                    )
                )
        _atomic_write(journal, record)
        return _decision(record, journal, cache_state)

    if record["status"] == "accepted":
        _require(cache_state == "valid", "accepted cache is absent or invalid")
        assert validation is not None
        try:
            validate_threshold_admission_record(
                record,
                episode=episode,
                minimum_threshold=minimum_threshold,
                keyframe_budget=keyframe_budget,
                final_validation=validation,
            )
        except FlowAuditError as error:
            raise ThresholdAdmissionError(
                f"accepted cache no longer exactly resumes: {error}"
            ) from error
        return _decision(record, journal, cache_state)
    if record["status"] == "exhausted":
        return _decision(record, journal, cache_state)

    pending = record["attempts"][-1]
    expected_index = _threshold_index(pending["threshold"], "pending threshold")
    if cache_state == "valid":
        assert validation is not None
        actual_index = _threshold_index(
            validation["flow_threshold"], "actual cache threshold"
        )
        if actual_index == expected_index:
            _finish_observation(record, validation, keyframe_budget)
            _atomic_write(journal, record)
            return _decision(record, journal, cache_state)
        if actual_index == expected_index - 1 and len(record["attempts"]) >= 2:
            previous = record["attempts"][-2]
            _require(
                previous["outcome"] == "over_budget"
                and _validation_matches_attempt(validation, previous),
                "cache at the previous tier does not match the admission journal",
            )
            return _decision(record, journal, cache_state)
        raise ThresholdAdmissionError(
            "cache threshold is not the pending or immediately previous tier; "
            "refusing downgrade/signature mixing"
        )
    if cache_state == "invalid":
        declared = _declared_partial_thresholds(patch_root, episode)
        if any(
            _threshold_index(value, "partial cache threshold") > expected_index
            for value in declared
        ):
            raise ThresholdAdmissionError(
                "invalid cache declares a threshold above the pending tier; "
                "refusing a downgrade overwrite"
            )
        pending["action"] = "overwrite"
        _atomic_write(journal, record)
        return _decision(record, journal, cache_state)
    return _decision(record, journal, cache_state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--minimum-threshold", type=float, required=True)
    parser.add_argument("--patch-flow-root", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--cache-schema", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        schema = _load_cache_schema(args.cache_schema)
        decision = plan_threshold_admission(
            episode=args.episode,
            frames=args.frames,
            minimum_threshold=args.minimum_threshold,
            patch_flow_root=args.patch_flow_root,
            admission_root=args.admission_root,
            cache_schema=schema,
        )
    except FlowAuditError as error:
        raise ThresholdAdmissionError(str(error)) from error
    print(json.dumps(decision, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
