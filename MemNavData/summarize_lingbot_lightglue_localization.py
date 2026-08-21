#!/usr/bin/env python3
"""Audit LightGlue+LingBot relocalization as an actionability certificate.

Co-visibility remains a descriptive field, not the operational ground truth:
sparse overlap can still yield an accurate metric camera pose.  The primary
audit therefore asks whether a frozen, deployment-visible certificate predicts
that the estimated pose is accurate enough to drive navigation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd


POSITION_TOLERANCE_M = 0.75

# Structural, train-only v1 certificate. These are not optimized against the
# supplied rows: 16 is twice the eight-inlier numerical validity floor, and 5%
# requires non-local image support rather than a tiny repeated-texture patch.
CERTIFICATE_MIN_INLIERS = 16
CERTIFICATE_MIN_QUERY_COVERAGE = 0.05
# Correspondences must occupy non-local support in *both* images.  Checking
# only the query side admits a close-up repeated texture that covers much of
# the query while occupying a tiny patch of the history reference.  The same
# structural 5% floor is intentionally used on both sides; it is not fitted
# from the supplied audit rows.
CERTIFICATE_MIN_REFERENCE_COVERAGE = 0.05
CERTIFICATE_MAX_REPROJECTION_RMSE_PX = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session-manifest", type=Path)
    parser.add_argument("--static-rows", type=Path)
    parser.add_argument("--static-report", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_actionable(pnp: Mapping[str, object]) -> bool:
    """Whether the estimated metric PointGoal is inside the safety margin.

    The downstream controller consumes only the target position; the goal
    image supplies appearance and benchmark success is distance-only.  Camera
    yaw is therefore diagnostic rather than part of actionability.  A bearing
    threshold would also be ill-conditioned when the true target vector is
    short even though the estimated target position is already accurate.
    """
    if pnp.get("status") != "ok":
        return False
    if not finite(pnp.get("relative_position_error_m")):
        return False
    return float(pnp["relative_position_error_m"]) <= POSITION_TOLERANCE_M


def has_certificate(pnp: Mapping[str, object]) -> bool:
    required = (
        "inliers", "query_inlier_coverage", "reference_inlier_coverage",
        "reprojection_rmse_px",
    )
    if pnp.get("status") != "ok" or not all(
            finite(pnp.get(name)) for name in required):
        return False
    return (
        int(pnp["inliers"]) >= CERTIFICATE_MIN_INLIERS
        and float(pnp["query_inlier_coverage"])
        >= CERTIFICATE_MIN_QUERY_COVERAGE
        and float(pnp["reference_inlier_coverage"])
        >= CERTIFICATE_MIN_REFERENCE_COVERAGE
        and float(pnp["reprojection_rmse_px"])
        <= CERTIFICATE_MAX_REPROJECTION_RMSE_PX)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total ** 2)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def safe_median(values) -> float | None:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def safe_quantiles(values) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def optional_float(row: object, name: str) -> float:
    value = getattr(row, name, float("nan"))
    return float(value) if finite(value) else float("nan")


def support_band(value: object) -> str | None:
    if not finite(value):
        return None
    value = float(value)
    if value <= 0.10:
        return "strict_or_low_le_0p10"
    if value <= 0.50:
        return "boundary_0p10_to_0p50"
    return "strong_gt_0p50"


def history_gap_band(value: object) -> str | None:
    if not finite(value):
        return None
    value = float(value)
    if value <= 32:
        return "within_lingbot_window_le_32"
    if value <= 96:
        return "delayed_33_to_96"
    return "long_delay_gt_96"


def confusion_summary(group: pd.DataFrame) -> dict:
    true_positive = int((group["certificate"] & group["actionable"]).sum())
    false_positive = int((group["certificate"] & ~group["actionable"]).sum())
    false_negative = int((~group["certificate"] & group["actionable"]).sum())
    true_negative = int((~group["certificate"] & ~group["actionable"]).sum())
    accepted = true_positive + false_positive
    actionable = true_positive + false_negative
    non_actionable = false_positive + true_negative
    return {
        "sessions": int(len(group)),
        "scenes": int(group["scene"].nunique()),
        "accepted": accepted,
        "actionable": actionable,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "acceptance_rate": accepted / len(group) if len(group) else None,
        "precision": true_positive / accepted if accepted else None,
        "precision_wilson_95": wilson(true_positive, accepted),
        "recall": true_positive / actionable if actionable else None,
        "recall_wilson_95": wilson(true_positive, actionable),
        "false_accept_rate": (
            false_positive / non_actionable if non_actionable else None),
        "false_accept_rate_wilson_95": wilson(
            false_positive, non_actionable),
    }


def grouped_confusions(frame: pd.DataFrame, column: str) -> dict[str, dict]:
    result = {}
    for name, group in frame.loc[frame[column].notna()].groupby(
            column, sort=True):
        result[str(name)] = confusion_summary(group)
    return result


def auc(labels, scores) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"n": int(valid.sum()), "roc_auc": None, "ap": None}
    return {
        "n": int(valid.sum()),
        "roc_auc": float(roc_auc_score(labels[valid], scores[valid])),
        "ap": float(average_precision_score(labels[valid], scores[valid])),
    }


def center_hypothesis(payload: str) -> dict:
    hypotheses = json.loads(payload)
    centers = [item for item in hypotheses if int(item["offset"]) == 0]
    if len(centers) != 1:
        raise RuntimeError("each row must contain exactly one center hypothesis")
    return centers[0]


def summarize(rows: pd.DataFrame, collector_report: Mapping[str, object],
              expected_sessions: set[str] | None = None) -> dict:
    if rows.empty or rows["session_id"].duplicated().any():
        raise RuntimeError("localization rows must be non-empty and session-unique")
    records = []
    for row in rows.itertuples(index=False):
        center = center_hypothesis(str(row.hypotheses_json))
        pnp = center.get("pnp_lightglue")
        if not isinstance(pnp, Mapping):
            raise RuntimeError(f"missing LightGlue PnP payload: {row.session_id}")
        records.append({
            "session_id": str(row.session_id),
            "scene": str(row.scene),
            "covis_label": int(row.label),
            "teacher_covis": float(row.teacher_covis),
            "session_max_covis": optional_float(row, "session_max_covis"),
            "candidate_frame": optional_float(row, "candidate_frame"),
            "causal_decision_frame": optional_float(
                row, "causal_decision_frame"),
            "causal_state_name": (
                str(getattr(row, "causal_state_name"))
                if hasattr(row, "causal_state_name") else None),
            "status": str(pnp.get("status")),
            "certificate": has_certificate(pnp),
            "actionable": is_actionable(pnp),
            "inliers": float(pnp.get("inliers", 0.0)),
            "query_coverage": float(pnp.get("query_inlier_coverage", 0.0)),
            "reprojection_rmse": float(
                pnp.get("reprojection_rmse_px", float("nan"))),
            "base_position_error": float(row.relative_position_error_m_center),
            "base_direction_error": float(
                row.relative_position_direction_error_deg_center),
            "base_rotation_error": float(row.relative_rotation_error_deg_center),
            "pnp_position_error": float(
                pnp.get("relative_position_error_m", float("nan"))),
            "pnp_direction_error": float(
                pnp.get("relative_position_direction_error_deg", float("nan"))),
            "pnp_rotation_error": float(
                pnp.get("relative_rotation_error_deg", float("nan"))),
        })
    frame = pd.DataFrame(records)
    frame["selected_anchor_support_band"] = frame["teacher_covis"].map(
        support_band)
    frame["session_support_band"] = frame["session_max_covis"].map(
        support_band)
    frame["history_gap_frames"] = (
        frame["causal_decision_frame"] - frame["candidate_frame"])
    frame["history_gap_band"] = frame["history_gap_frames"].map(
        history_gap_band)
    if expected_sessions is not None and set(frame["session_id"]) != expected_sessions:
        missing = sorted(expected_sessions - set(frame["session_id"]))
        extra = sorted(set(frame["session_id"]) - expected_sessions)
        raise RuntimeError(
            f"frozen session cover differs: missing={missing} extra={extra}")

    true_positive = int((frame["certificate"] & frame["actionable"]).sum())
    false_positive = int((frame["certificate"] & ~frame["actionable"]).sum())
    false_negative = int((~frame["certificate"] & frame["actionable"]).sum())
    true_negative = int((~frame["certificate"] & ~frame["actionable"]).sum())
    accepted = frame.loc[frame["certificate"]]
    actionable = frame.loc[frame["actionable"]]
    accepted_actionable = frame.loc[frame["certificate"] & frame["actionable"]]
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    paired = {}
    for name in ("position", "direction", "rotation"):
        before = accepted_actionable[f"base_{name}_error"].to_numpy()
        after = accepted_actionable[f"pnp_{name}_error"].to_numpy()
        improvements = int((after < before).sum())
        paired[name] = {
            "n": int(len(after)),
            "base_median": safe_median(before),
            "pnp_median": safe_median(after),
            "improved": improvements,
            "worsened": int((after > before).sum()),
        }
        if len(after):
            from scipy.stats import binomtest
            paired[name]["exact_sign_test_two_sided_p"] = float(
                binomtest(improvements, len(after), 0.5).pvalue)

    actionability_labels = frame["actionable"].astype(int)
    feature_auc = {
        "pnp_inliers": auc(actionability_labels, frame["inliers"]),
        "query_inlier_coverage": auc(
            actionability_labels, frame["query_coverage"]),
        "negative_reprojection_rmse": auc(
            actionability_labels, -frame["reprojection_rmse"]),
    }
    by_covis = {}
    for label, group in frame.groupby("covis_label"):
        by_covis[str(int(label))] = {
            "n": int(len(group)),
            "actionable": int(group["actionable"].sum()),
            "certificate_accepted": int(group["certificate"].sum()),
        }

    config = collector_report.get("config")
    if not isinstance(config, Mapping) or not config.get("pnp_lightglue"):
        raise RuntimeError("collector report is not a LightGlue PnP run")
    position_pair = paired["position"]
    effectiveness_checks = {
        "zero_false_positives": false_positive == 0,
        "at_least_five_true_positives": true_positive >= 5,
        "at_least_five_certified_scenes": int(
            accepted_actionable["scene"].nunique()) >= 5,
        "accepted_median_position_improves_lingbot": (
            position_pair["base_median"] is not None
            and position_pair["pnp_median"] is not None
            and position_pair["pnp_median"] < position_pair["base_median"]),
    }
    return {
        "schema_version": "lingbot_lightglue_actionability_audit_v2",
        "status": "train_only_frozen_certificate_audit_not_closed_loop",
        "rows": int(len(frame)),
        "scenes": int(frame["scene"].nunique()),
        "pnp_status_counts": dict(sorted(Counter(frame["status"]).items())),
        "ground_truth_actionable": int(frame["actionable"].sum()),
        "certificate": {
            "definition": {
                "min_pnp_inliers": CERTIFICATE_MIN_INLIERS,
                "min_query_inlier_hull_coverage": (
                    CERTIFICATE_MIN_QUERY_COVERAGE),
                "min_reference_inlier_hull_coverage": (
                    CERTIFICATE_MIN_REFERENCE_COVERAGE),
                "max_reprojection_rmse_px": (
                    CERTIFICATE_MAX_REPROJECTION_RMSE_PX),
            },
            "accepted": int(frame["certificate"].sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": (
                true_positive / precision_denominator
                if precision_denominator else None),
            "precision_wilson_95": wilson(
                true_positive, precision_denominator),
            "recall": (
                true_positive / recall_denominator
                if recall_denominator else None),
            "recall_wilson_95": wilson(true_positive, recall_denominator),
        },
        "actionability_definition": {
            "max_position_error_m": POSITION_TOLERANCE_M,
            "controller_output": "metric_pointgoal_xy",
            "benchmark_success_is_distance_only": True,
            "direction_and_camera_yaw_are_diagnostic_only": True,
            "uses_ground_truth_only_for_audit": True,
        },
        "feature_auc_for_actionability": feature_auc,
        "paired_pose_errors_on_accepted_actionable": paired,
        "by_teacher_covis_label": by_covis,
        "stratified_actionability": {
            "selected_anchor_support": grouped_confusions(
                frame, "selected_anchor_support_band"),
            "session_max_support": grouped_confusions(
                frame, "session_support_band"),
            "history_gap": grouped_confusions(frame, "history_gap_band"),
            "causal_state": grouped_confusions(frame, "causal_state_name"),
            "history_gap_frames_quantiles": safe_quantiles(
                frame["history_gap_frames"]),
            "audit_only_warning": (
                "co-visibility, causal state, and history age are used only "
                "to stratify frozen outputs; none is a deployment input"),
        },
        "covis_label_warning": (
            "co-visibility is not equivalent to localization actionability; "
            "low-covis rows may still have accurate metric PnP poses"),
        "collector_provenance": collector_report.get("provenance"),
        "hpc_effectiveness_gate": {
            "all_checks_pass": all(effectiveness_checks.values()),
            "checks": effectiveness_checks,
            "certified_actionable_sessions": true_positive,
            "certified_actionable_scenes": int(
                accepted_actionable["scene"].nunique()),
            "scope": (
                "authorizes only a subsequent paired closed-loop test; "
                "it is not an SR result"),
        },
    }


def static_certificate_summary(rows: pd.DataFrame) -> dict:
    required = {
        "session_id", "session_label", "fundamental_inliers",
        "fundamental_query_grid_coverage",
    }
    missing = required - set(rows.columns)
    if missing:
        raise RuntimeError(f"static rows missing columns: {sorted(missing)}")
    passed = (
        rows["fundamental_inliers"].ge(32)
        & rows["fundamental_query_grid_coverage"].ge(0.75))
    sessions = rows.assign(certificate=passed).groupby(
        "session_id", as_index=False).agg({
            "certificate": "max", "session_label": "first",
        })
    known = sessions.loc[sessions["session_label"].ge(0)]
    return {
        "rows": int(len(rows)),
        "sessions": int(len(sessions)),
        "known_sessions": int(len(known)),
        "teacher_positive_accepted": int((
            known["certificate"] & known["session_label"].eq(1)).sum()),
        "teacher_negative_accepted": int((
            known["certificate"] & known["session_label"].eq(0)).sum()),
        "teacher_positive_rejected": int((
            ~known["certificate"] & known["session_label"].eq(1)).sum()),
        "teacher_negative_rejected": int((
            ~known["certificate"] & known["session_label"].eq(0)).sum()),
        "definition": {
            "min_fundamental_inliers": 32,
            "min_query_grid_coverage_4x4": 0.75,
        },
    }


def main() -> None:
    args = parse_args()
    for path in (args.rows, args.report):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = pd.read_csv(args.rows)
    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    expected_sessions = None
    manifest_sha = None
    if args.session_manifest:
        with args.session_manifest.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_sessions = {str(item) for item in manifest["sessions"]}
        manifest_sha = sha256_file(args.session_manifest)
    result = summarize(rows, report, expected_sessions)
    result["inputs"] = {
        "rows": str(args.rows.resolve()),
        "rows_sha256": sha256_file(args.rows),
        "report": str(args.report.resolve()),
        "report_sha256": sha256_file(args.report),
        "session_manifest": (
            str(args.session_manifest.resolve()) if args.session_manifest else None),
        "session_manifest_sha256": manifest_sha,
    }
    if bool(args.static_rows) != bool(args.static_report):
        raise ValueError("static rows/report must be supplied together")
    if args.static_rows:
        static_rows = pd.read_csv(args.static_rows)
        with args.static_report.open(encoding="utf-8") as handle:
            static_report = json.load(handle)
        result["static_image_localization"] = static_certificate_summary(
            static_rows)
        result["inputs"].update({
            "static_rows": str(args.static_rows.resolve()),
            "static_rows_sha256": sha256_file(args.static_rows),
            "static_report": str(args.static_report.resolve()),
            "static_report_sha256": sha256_file(args.static_report),
        })
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
