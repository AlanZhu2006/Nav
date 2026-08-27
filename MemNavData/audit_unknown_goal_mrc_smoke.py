#!/usr/bin/env python3
"""Contract-only audit for the Unknown-Goal MRC-v0 timing smoke.

This auditor intentionally does not compute label-conditioned accuracy, AUC,
or a threshold.  The smoke exists only to establish that the frozen evidence
extractor is causal, fixed-view, numerically usable, and affordable before a
full train-only scene-grouped OOF experiment is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import pandas as pd


EXPECTED_SESSION_SCHEMAS = frozenset({
    "unknown_goal_mrc_v0_smoke_sessions_v1",
    "train40_certificate_challenge_manifest_v1",
})
RECEIPT_SCHEMA = "unknown_goal_mrc_v0_contract_smoke_receipt_v1"
CHALLENGE_RECEIPT_SCHEMA = "train40_certificate_challenge_contract_receipt_v1"
REQUIRED_HYPOTHESIS_FIELDS = frozenset({
    "anchor",
    "cloud_overlap_f1",
    "depth_scale_raw",
    "goal_pose",
    "goal_refine_rotation_deg",
    "goal_refine_translation_raw",
    "offset",
    "predicted_relative_xy_m",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_contract(
        rows: pd.DataFrame, report: Mapping[str, object],
        progress: Mapping[str, object],
        session_manifest: Mapping[str, object], *,
        expected_selection_mode: str = "deployment",
        expected_top_k: int = 1,
        expected_selection_origin: str = "deployment_topk",
        require_lightglue_pnp: bool = False,
        expected_session_count: int = 24) -> dict:
    manifest_schema = session_manifest.get("schema_version")
    _require(
        manifest_schema in EXPECTED_SESSION_SCHEMAS,
        "session-manifest schema changed")
    sessions = session_manifest.get("sessions")
    _require(isinstance(sessions, list), "session list is absent")
    sessions = [str(item) for item in sessions]
    expected_sessions = int(session_manifest["selected_session_count"])
    expected_scenes = int(session_manifest["selected_scene_count"])
    _require(expected_sessions == expected_session_count,
             "session-manifest count differs from the frozen run scope")
    _require(len(sessions) == len(set(sessions)) == expected_sessions,
             "session list is not an exact unique cover")
    _require(len(rows) == expected_sessions, "one-row-per-session cover changed")
    _require(rows["session_id"].nunique() == expected_sessions,
             "row sessions are not unique")
    _require(set(rows["session_id"].astype(str)) == set(sessions),
             "output sessions differ from the frozen hash sample")
    _require(rows["scene"].nunique() == expected_scenes,
             "scene coverage changed")
    _require(set(rows["causal_split_role"].astype(str)) == {"train"},
             "non-train data entered the contract smoke")
    _require(set(rows["candidate_selection_origin"].astype(str))
             == {expected_selection_origin},
             "teacher-augmented candidate selection entered the smoke")
    _require(set(rows["n_hypotheses"].astype(int)) == {3},
             "MRC requires exactly three hypotheses for every session")
    _require(set(rows["metric_scale_source"].astype(str))
             == {"external_causal_first_prefix_v1"},
             "metric scale is not the pinned causal-prefix artifact")

    aggregate_columns = (
        "goal_pose_translation_dispersion_norm",
        "goal_pose_rotation_dispersion_deg",
        "cloud_overlap_f1_mean",
        "goal_refine_translation_norm_median",
        "goal_refine_rotation_deg_median",
    )
    for column in aggregate_columns:
        _require(column in rows, f"missing aggregate feature: {column}")
        _require(all(_finite(value) for value in rows[column]),
                 f"non-finite aggregate feature: {column}")

    observed_patterns = set()
    for row in rows.itertuples(index=False):
        offsets = tuple(int(value) for value in str(
            row.neighbor_offsets).split(";"))
        _require(len(offsets) == len(set(offsets)) == 3,
                 f"invalid offsets for {row.session_id}: {offsets}")
        _require(0 in offsets and all(abs(value) <= 4 for value in offsets),
                 f"offset radius/center contract failed: {offsets}")
        _require(all(
            8 <= int(row.candidate_frame) + offset
            < int(row.causal_decision_frame)
            for offset in offsets),
            f"offset crossed scale/decision boundary: {row.session_id}")
        observed_patterns.add(offsets)
        hypotheses = json.loads(str(row.hypotheses_json))
        _require(isinstance(hypotheses, list) and len(hypotheses) == 3,
                 f"hypothesis payload count changed: {row.session_id}")
        _require(tuple(int(item["offset"]) for item in hypotheses) == offsets,
                 f"row/payload offsets disagree: {row.session_id}")
        for hypothesis in hypotheses:
            _require(REQUIRED_HYPOTHESIS_FIELDS <= set(hypothesis),
                     f"hypothesis ABI changed: {row.session_id}")
            _require(_finite(hypothesis["cloud_overlap_f1"]),
                     f"non-finite overlap: {row.session_id}")
            _require(_finite(hypothesis["depth_scale_raw"])
                     and float(hypothesis["depth_scale_raw"]) > 0.0,
                     f"invalid depth scale: {row.session_id}")
            goal_pose = hypothesis["goal_pose"]
            predicted_xy = hypothesis["predicted_relative_xy_m"]
            _require(isinstance(goal_pose, list) and len(goal_pose) == 9
                     and all(_finite(value) for value in goal_pose),
                     f"invalid goal pose: {row.session_id}")
            _require(isinstance(predicted_xy, list) and len(predicted_xy) == 2
                     and all(_finite(value) for value in predicted_xy),
                     f"invalid relative prediction: {row.session_id}")
            if require_lightglue_pnp:
                pnp = hypothesis.get("pnp_lightglue")
                _require(isinstance(pnp, Mapping),
                         f"LightGlue PnP payload absent: {row.session_id}")
                _require(pnp.get("coordinate_source")
                         == "native_rgb_to_lingbot_pad",
                         f"LightGlue coordinate contract changed: {row.session_id}")

    config = report.get("config")
    _require(isinstance(config, Mapping), "collector report lacks config")
    _require(config.get("selection_mode") == expected_selection_mode,
             "selection mode changed")
    _require(int(config.get("top_k", -1)) == expected_top_k, "top-K changed")
    _require(int(config.get("adaptive_neighbor_radius", -1)) == 4,
             "adaptive radius changed")
    _require(int(config.get("adaptive_neighbor_count", -1)) == 3,
             "adaptive view count changed")
    _require(config.get("adaptive_neighbor_policy") == "maximin_spacing_v1",
             "adaptive policy changed")
    _require(bool(config.get("full_replay")), "full causal replay is disabled")
    provenance = report.get("provenance")
    _require(isinstance(provenance, Mapping), "collector report lacks provenance")
    _require(provenance.get("teacher_csv_sha256")
             == session_manifest.get("source_teacher_sha256"),
             "teacher identity differs from the frozen session sample")
    if require_lightglue_pnp:
        _require(bool(config.get("pnp_lightglue")),
                 "LightGlue PnP is disabled")
        selection = provenance.get("lightglue_candidate_selection")
        _require(isinstance(selection, Mapping),
                 "LightGlue candidate-selection provenance is absent")
        _require(selection.get("ranking")
                 == "lightglue_fundamental_rank_v1",
                 "LightGlue ranking contract changed")
        _require(selection.get("uses_teacher_labels") is False,
                 "candidate selection is not label blind")
        for name in ("csv_sha256", "report_sha256"):
            value = selection.get(name)
            _require(isinstance(value, str) and len(value) == 64,
                     f"invalid LightGlue selection provenance: {name}")
    elapsed = float(provenance.get("elapsed_seconds", -1.0))
    _require(math.isfinite(elapsed) and elapsed > 0.0,
             "elapsed time receipt is invalid")
    _require(progress.get("status") == "complete",
             "collector progress is not complete")
    _require(int(progress.get("completed_sessions", -1)) == expected_sessions,
             "progress session count changed")
    cuda_memory = progress.get("cuda_memory")
    _require(isinstance(cuda_memory, Mapping), "CUDA receipt is absent")
    peak_allocated = float(cuda_memory.get("peak_allocated_gib", -1.0))
    _require(math.isfinite(peak_allocated) and peak_allocated > 0.0,
             "CUDA peak allocation is invalid")

    return {
        "schema_version": (
            CHALLENGE_RECEIPT_SCHEMA
            if manifest_schema == "train40_certificate_challenge_manifest_v1"
            else RECEIPT_SCHEMA),
        "status": (
            "train40_challenge_contract_passed_not_closed_loop"
            if manifest_schema == "train40_certificate_challenge_manifest_v1"
            else "contract_smoke_passed_not_effectiveness_evidence"),
        "label_metrics_intentionally_omitted": True,
        "sessions": expected_sessions,
        "scenes": expected_scenes,
        "rows": len(rows),
        "views_per_session": 3,
        "selection_mode": expected_selection_mode,
        "top_k_candidate_generator": expected_top_k,
        "selection_origin": expected_selection_origin,
        "lightglue_pnp_required": require_lightglue_pnp,
        "observed_offset_patterns": [
            list(pattern) for pattern in sorted(observed_patterns)],
        "elapsed_seconds": elapsed,
        "seconds_per_session": elapsed / expected_sessions,
        "peak_cuda_allocated_gib": peak_allocated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--session-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-mode", default="deployment",
        choices=("deployment", "lightglue_ranked"))
    parser.add_argument("--expected-top-k", type=int, default=1)
    parser.add_argument(
        "--expected-selection-origin", default="deployment_topk")
    parser.add_argument("--require-lightglue-pnp", action="store_true")
    parser.add_argument("--expected-session-count", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.rows, args.report, args.progress,
                 args.session_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    with args.progress.open(encoding="utf-8") as handle:
        progress = json.load(handle)
    with args.session_manifest.open(encoding="utf-8") as handle:
        session_manifest = json.load(handle)
    receipt = validate_contract(
        pd.read_csv(args.rows), report, progress, session_manifest,
        expected_selection_mode=args.expected_selection_mode,
        expected_top_k=args.expected_top_k,
        expected_selection_origin=args.expected_selection_origin,
        require_lightglue_pnp=args.require_lightglue_pnp,
        expected_session_count=args.expected_session_count)
    receipt["artifacts"] = {
        "rows_sha256": sha256_file(args.rows),
        "report_sha256": sha256_file(args.report),
        "progress_sha256": sha256_file(args.progress),
        "session_manifest_sha256": sha256_file(args.session_manifest),
    }
    atomic_write_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
