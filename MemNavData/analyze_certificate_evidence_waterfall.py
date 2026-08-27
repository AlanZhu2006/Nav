#!/usr/bin/env python3
"""Audit how each frozen CEC evidence stage changes authorization quality.

The analysis joins the geometry-selected LightGlue/PnP endpoint to the exact
top-8 Fundamental-MAGSAC row from which that endpoint was selected.  It uses
only the consumed train40 challenge.  Ground-truth actionability and support
labels are audit outputs; they never enter a runtime decision.

The ``precheck_plus_pnp_pose`` row is the precise correspondence-precheck-only
authorization ablation: Fundamental support authorizes a PnP pose as soon as
that pose exists, without consuming any of the four PnP certificate quality
checks.  Later rows add the frozen certificate checks cumulatively.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "cec_certificate_evidence_waterfall_v1_20260826"
MANIFEST_SCHEMA = "train40_certificate_challenge_manifest_v1"
GEOMETRY_ORIGIN = "lightglue_fundamental_rank_v1"

POSITION_TOLERANCE_M = 0.75
MIN_INLIERS = 16
MIN_QUERY_COVERAGE = 0.05
MIN_REFERENCE_COVERAGE = 0.05
MAX_REPROJECTION_RMSE_PX = 2.0

STAGE_ORDER = (
    "geometry_ranked_candidate",
    "fundamental_precheck",
    "precheck_plus_pnp_pose",
    "plus_pnp_inliers",
    "plus_query_coverage",
    "plus_reference_coverage",
    "full_certificate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def parse_bool(value: object, *, name: str) -> bool:
    if value is True or value == "True" or value == "true" or value == "1":
        return True
    if value is False or value == "False" or value == "false" or value == "0":
        return False
    raise RuntimeError(f"{name} is not a strict boolean: {value!r}")


def center_pnp(payload: str) -> dict[str, Any]:
    hypotheses = json.loads(payload)
    centers = [item for item in hypotheses if int(item.get("offset", -999)) == 0]
    if len(centers) != 1 or not isinstance(
            centers[0].get("pnp_lightglue"), dict):
        raise RuntimeError("row lacks exactly one LightGlue-PnP center hypothesis")
    return centers[0]["pnp_lightglue"]


def finite_xy(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 2
        and all(finite(item) for item in value)
    )


def actionable(pnp: Mapping[str, Any]) -> bool:
    return bool(
        pnp.get("status") == "ok"
        and finite(pnp.get("relative_position_error_m"))
        and float(pnp["relative_position_error_m"]) <= POSITION_TOLERANCE_M
    )


def stage_decisions(
    static: Mapping[str, str], pnp: Mapping[str, Any],
) -> dict[str, bool]:
    fundamental = bool(
        finite(static.get("fundamental_inliers"))
        and int(float(static["fundamental_inliers"])) >= MIN_INLIERS
        and finite(static.get("fundamental_query_hull_coverage"))
        and float(static["fundamental_query_hull_coverage"])
        >= MIN_QUERY_COVERAGE
        and finite(static.get("fundamental_reference_hull_coverage"))
        and float(static["fundamental_reference_hull_coverage"])
        >= MIN_REFERENCE_COVERAGE
    )
    pose = bool(
        fundamental
        and pnp.get("status") == "ok"
        and finite_xy(pnp.get("predicted_relative_xy_m"))
    )
    pnp_inliers = bool(
        pose
        and finite(pnp.get("inliers"))
        and int(float(pnp["inliers"])) >= MIN_INLIERS
    )
    query_coverage = bool(
        pnp_inliers
        and finite(pnp.get("query_inlier_coverage"))
        and float(pnp["query_inlier_coverage"]) >= MIN_QUERY_COVERAGE
    )
    reference_coverage = bool(
        query_coverage
        and finite(pnp.get("reference_inlier_coverage"))
        and float(pnp["reference_inlier_coverage"])
        >= MIN_REFERENCE_COVERAGE
    )
    full = bool(
        reference_coverage
        and finite(pnp.get("reprojection_rmse_px"))
        and float(pnp["reprojection_rmse_px"])
        <= MAX_REPROJECTION_RMSE_PX
    )
    return {
        "geometry_ranked_candidate": True,
        "fundamental_precheck": fundamental,
        "precheck_plus_pnp_pose": pose,
        "plus_pnp_inliers": pnp_inliers,
        "plus_query_coverage": query_coverage,
        "plus_reference_coverage": reference_coverage,
        "full_certificate": full,
    }


def wilson(successes: int, total: int,
           z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def confusion(records: Iterable[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    rows = list(records)
    accepted = sum(bool(row["stages"][stage]) for row in rows)
    positives = sum(bool(row["actionable"]) for row in rows)
    true_positive = sum(
        bool(row["stages"][stage]) and bool(row["actionable"])
        for row in rows
    )
    false_positive = accepted - true_positive
    false_negative = positives - true_positive
    true_negative = len(rows) - positives - false_positive
    negatives = len(rows) - positives
    return {
        "sessions": len(rows),
        "accepted": accepted,
        "actionable": positives,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": true_positive / accepted if accepted else None,
        "precision_wilson_95": wilson(true_positive, accepted),
        "recall": true_positive / positives if positives else None,
        "recall_wilson_95": wilson(true_positive, positives),
        "false_accept_rate": false_positive / negatives if negatives else None,
        "false_accept_rate_wilson_95": wilson(false_positive, negatives),
    }


def authorization_counts(
    records: Iterable[Mapping[str, Any]], stage: str,
) -> dict[str, Any]:
    rows = list(records)
    positive = [row for row in rows if row["session_has_positive"]]
    strict = [row for row in rows if row["session_is_strict_no_match"]]
    boundary = [
        row for row in rows
        if not row["session_has_positive"]
        and not row["session_is_strict_no_match"]
    ]

    def count(group: list[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "sessions": len(group),
            "authorized": sum(bool(row["stages"][stage]) for row in group),
        }

    return {
        "positive_session": count(positive),
        "strict_no_match": count(strict),
        "boundary_or_ambiguous": count(boundary),
    }


def threshold_accepts(
    evidence: Mapping[str, Any], *, min_inliers: int = MIN_INLIERS,
    min_coverage: float = MIN_QUERY_COVERAGE,
    max_rmse: float = MAX_REPROJECTION_RMSE_PX,
) -> bool:
    """Apply one audit-only operating point to both monotone proof stages."""

    return bool(
        evidence["fundamental_inliers"] >= min_inliers
        and evidence["fundamental_query_coverage"] >= min_coverage
        and evidence["fundamental_reference_coverage"] >= min_coverage
        and evidence["pnp_pose_available"]
        and evidence["pnp_inliers"] >= min_inliers
        and evidence["pnp_query_coverage"] >= min_coverage
        and evidence["pnp_reference_coverage"] >= min_coverage
        and evidence["pnp_rmse"] <= max_rmse
    )


def flag_confusion(
    records: list[Mapping[str, Any]], flags: list[bool],
) -> dict[str, Any]:
    if len(records) != len(flags):
        raise RuntimeError("sensitivity flags differ from record population")
    positives = sum(bool(row["actionable"]) for row in records)
    accepted = sum(flags)
    true_positive = sum(
        flag and bool(row["actionable"])
        for row, flag in zip(records, flags)
    )
    false_positive = accepted - true_positive
    negatives = len(records) - positives
    return {
        "accepted": accepted,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": positives - true_positive,
        "true_negative": negatives - false_positive,
        "precision": true_positive / accepted if accepted else None,
        "recall": true_positive / positives if positives else None,
    }


def sensitivity_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    sweeps = {
        "min_inliers": [8, 12, 16, 24, 32],
        "symmetric_hull_coverage": [0.02, 0.05, 0.10],
        "max_reprojection_rmse_px": [1.0, 2.0, 3.0, 4.0],
    }
    result = {}
    for name, values in sweeps.items():
        rows = []
        for value in values:
            kwargs = {}
            if name == "min_inliers":
                kwargs["min_inliers"] = int(value)
            elif name == "symmetric_hull_coverage":
                kwargs["min_coverage"] = float(value)
            else:
                kwargs["max_rmse"] = float(value)
            counts = flag_confusion(records, [
                threshold_accepts(row["evidence"], **kwargs)
                for row in records
            ])
            rows.append({"value": value, **counts})
        result[name] = rows
    return {
        "scope": "train_only_one_factor_at_a_time_after_operating_point_frozen",
        "baseline": {
            "min_inliers": MIN_INLIERS,
            "symmetric_hull_coverage": MIN_QUERY_COVERAGE,
            "max_reprojection_rmse_px": MAX_REPROJECTION_RMSE_PX,
        },
        "warning": (
            "These rows characterize robustness and do not authorize selecting "
            "a new operating point from consumed or held-out outcomes."
        ),
        "sweeps": result,
    }


def build_records(
    geometry_rows: list[dict[str, str]],
    static_rows: list[dict[str, str]],
    expected_sessions: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not geometry_rows:
        raise RuntimeError("geometry endpoint rows are empty")
    geometry_sessions = [str(row["session_id"]) for row in geometry_rows]
    if len(set(geometry_sessions)) != len(geometry_sessions):
        raise RuntimeError("geometry endpoint rows contain duplicate sessions")
    if expected_sessions is not None and set(geometry_sessions) != expected_sessions:
        raise RuntimeError("geometry endpoint rows differ from frozen manifest")

    static_index: dict[tuple[str, int], dict[str, str]] = {}
    for row in static_rows:
        key = (str(row["session_id"]), int(row["candidate_frame"]))
        if key in static_index:
            raise RuntimeError(f"duplicate static candidate key: {key}")
        static_index[key] = row
    if set(session for session, _ in static_index) != set(geometry_sessions):
        raise RuntimeError("static and endpoint session universes differ")

    records = []
    for row in geometry_rows:
        if row.get("candidate_selection_origin") != GEOMETRY_ORIGIN:
            raise RuntimeError("endpoint row is not geometry-ranked")
        key = (str(row["session_id"]), int(row["candidate_frame"]))
        static = static_index.get(key)
        if static is None:
            raise RuntimeError(f"selected endpoint has no static candidate: {key}")
        if str(static.get("candidate_path")) != str(row.get("candidate_path")):
            raise RuntimeError(f"selected candidate path changed: {key}")
        pnp = center_pnp(str(row["hypotheses_json"]))
        pnp_pose_available = bool(
            pnp.get("status") == "ok"
            and finite_xy(pnp.get("predicted_relative_xy_m")))
        records.append({
            "session_id": key[0],
            "scene": str(row["scene"]),
            "causal_state_name": str(row.get("causal_state_name", "unknown")),
            "actionable": actionable(pnp),
            "session_has_positive": parse_bool(
                row["session_has_positive"], name="session_has_positive"),
            "session_is_strict_no_match": parse_bool(
                row["session_is_strict_no_match"],
                name="session_is_strict_no_match"),
            "stages": stage_decisions(static, pnp),
            "evidence": {
                "fundamental_inliers": int(float(
                    static["fundamental_inliers"])),
                "fundamental_query_coverage": float(
                    static["fundamental_query_hull_coverage"]),
                "fundamental_reference_coverage": float(
                    static["fundamental_reference_hull_coverage"]),
                "pnp_pose_available": pnp_pose_available,
                "pnp_inliers": (
                    int(float(pnp["inliers"]))
                    if finite(pnp.get("inliers")) else -1),
                "pnp_query_coverage": (
                    float(pnp["query_inlier_coverage"])
                    if finite(pnp.get("query_inlier_coverage")) else -1.0),
                "pnp_reference_coverage": (
                    float(pnp["reference_inlier_coverage"])
                    if finite(pnp.get("reference_inlier_coverage")) else -1.0),
                "pnp_rmse": (
                    float(pnp["reprojection_rmse_px"])
                    if finite(pnp.get("reprojection_rmse_px"))
                    else float("inf")),
            },
        })
    return records


def analyze(
    geometry_rows: list[dict[str, str]],
    static_rows: list[dict[str, str]],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_sessions = None
    if manifest is not None:
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise RuntimeError("train40 challenge manifest schema changed")
        sessions = [str(item) for item in manifest.get("sessions", [])]
        if not sessions or len(sessions) != len(set(sessions)):
            raise RuntimeError("frozen manifest session universe is invalid")
        expected_sessions = set(sessions)
    records = build_records(geometry_rows, static_rows, expected_sessions)
    stages = {}
    for index, stage in enumerate(STAGE_ORDER):
        stage_result = confusion(records, stage)
        stage_result["open_set_support_audit"] = authorization_counts(
            records, stage)
        stage_result["accepted_by_causal_state"] = {
            state: sum(
                row["stages"][stage]
                for row in records
                if row["causal_state_name"] == state
            )
            for state in sorted({row["causal_state_name"] for row in records})
        }
        if index:
            previous = stages[STAGE_ORDER[index - 1]]
            stage_result["rejected_by_new_stage"] = (
                int(previous["accepted"]) - int(stage_result["accepted"])
            )
            stage_result["false_positives_removed_by_new_stage"] = (
                int(previous["false_positive"])
                - int(stage_result["false_positive"])
            )
            stage_result["true_positives_removed_by_new_stage"] = (
                int(previous["true_positive"])
                - int(stage_result["true_positive"])
            )
        stages[stage] = stage_result

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "train_only_offline_authorization_ablation_not_closed_loop",
        "sessions": len(records),
        "scenes": len({row["scene"] for row in records}),
        "stage_order": list(STAGE_ORDER),
        "thresholds": {
            "actionable_max_position_error_m": POSITION_TOLERANCE_M,
            "fundamental_and_pnp_min_inliers": MIN_INLIERS,
            "fundamental_and_pnp_min_query_hull_coverage": MIN_QUERY_COVERAGE,
            "fundamental_and_pnp_min_reference_hull_coverage": (
                MIN_REFERENCE_COVERAGE),
            "pnp_max_reprojection_rmse_px": MAX_REPROJECTION_RMSE_PX,
        },
        "stages": stages,
        "threshold_sensitivity": sensitivity_summary(records),
        "interpretation_contract": {
            "precheck_only_runtime_proxy": "precheck_plus_pnp_pose",
            "ground_truth_used_by_runtime": False,
            "support_labels_used_by_runtime": False,
            "candidate_selection": GEOMETRY_ORIGIN,
            "warning": (
                "This consumed train-only audit measures authorization quality; "
                "it is neither a closed-loop SR result nor threshold selection."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-rows", type=Path, required=True)
    parser.add_argument("--static-top8-rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = analyze(
        read_csv(args.geometry_rows),
        read_csv(args.static_top8_rows),
        manifest,
    )
    result["inputs"] = {
        "geometry_rows": str(args.geometry_rows.resolve()),
        "geometry_rows_sha256": sha256_file(args.geometry_rows),
        "static_top8_rows": str(args.static_top8_rows.resolve()),
        "static_top8_rows_sha256": sha256_file(args.static_top8_rows),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
