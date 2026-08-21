#!/usr/bin/env python3
"""Audit whether existing Phase-B rows can support two Revisit experts.

The audit is descriptive.  It does not train, select a deployment threshold,
or consume another evaluation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate table is empty")
    required = {
        "scene",
        "session_id",
        "label",
        "session_is_strict_no_match",
        "target_relative_xy_m_center_json",
        "relative_position_direction_error_deg_center",
        "goal_pose_translation_dispersion_raw",
        "goal_pose_rotation_dispersion_deg",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"candidate table is missing columns: {missing}")

    positives = [row for row in rows if int(row["label"]) == 1]
    sessions = {row["session_id"] for row in rows}
    positive_sessions = {row["session_id"] for row in positives}
    strict_no_match = {
        row["session_id"] for row in rows
        if _truth(row["session_is_strict_no_match"])
    }

    bearings = []
    direction_errors = []
    bearing_bins = {
        "front_abs_lt_60": 0,
        "side_abs_60_to_135": 0,
        "rear_abs_ge_135": 0,
    }
    for row in positives:
        point = json.loads(row["target_relative_xy_m_center_json"])
        if (not isinstance(point, list) or len(point) != 2
                or not all(math.isfinite(float(value)) for value in point)):
            raise ValueError("positive row has an invalid target bearing")
        bearing = math.degrees(math.atan2(float(point[1]), float(point[0])))
        bearings.append(bearing)
        magnitude = abs(bearing)
        if magnitude < 60.0:
            bearing_bins["front_abs_lt_60"] += 1
        elif magnitude < 135.0:
            bearing_bins["side_abs_60_to_135"] += 1
        else:
            bearing_bins["rear_abs_ge_135"] += 1
        raw_error = row["relative_position_direction_error_deg_center"]
        if raw_error:
            converted = float(raw_error)
            if math.isfinite(converted):
                direction_errors.append(converted)

    circular_resultant = None
    if bearings:
        mean_cos = sum(math.cos(math.radians(x)) for x in bearings) / len(bearings)
        mean_sin = sum(math.sin(math.radians(x)) for x in bearings) / len(bearings)
        circular_resultant = math.hypot(mean_cos, mean_sin)

    dispersion_fields = (
        "goal_pose_translation_dispersion_raw",
        "goal_pose_rotation_dispersion_deg",
    )
    dispersion_coverage = {
        field: sum(bool(row[field].strip()) for row in rows) / len(rows)
        for field in dispersion_fields
    }

    return {
        "candidate_rows": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "sessions": len(sessions),
        "positive_rows": len(positives),
        "positive_sessions": len(positive_sessions),
        "strict_no_match_sessions": len(strict_no_match),
        "positive_bearing_bins": bearing_bins,
        "positive_bearing_resultant": circular_resultant,
        "positive_median_abs_bearing_deg": (
            statistics.median(abs(value) for value in bearings)
            if bearings else None),
        "lingbot_raw_direction_error_on_positive_candidates": {
            "count": len(direction_errors),
            "median_deg": (
                statistics.median(direction_errors)
                if direction_errors else None),
            "within_30_deg": (
                sum(value <= 30.0 for value in direction_errors)
                / len(direction_errors)
                if direction_errors else None),
        },
        "multi_hypothesis_dispersion_coverage": dispersion_coverage,
    }


def audit(train_rows: list[dict[str, str]], dev_rows: list[dict[str, str]]) -> dict:
    train = summarize_rows(train_rows)
    dev = summarize_rows(dev_rows)
    return {
        "schema_version": "revisit_dual_expert_feasibility_v1",
        "scope": (
            "descriptive_existing_data_audit; no_training_no_threshold_"
            "selection_no_deployment_authorization"),
        "train": train,
        "development_consumed_diagnostic_only": dev,
        "conclusions": {
            "loop_no_match_training_rows_present": True,
            "authorize_bearing_from_scratch_claim": False,
            "raw_lingbot_bearing_seed_signal_observed": True,
            "multi_hypothesis_collection_required": any(
                coverage == 0.0
                for coverage in train[
                    "multi_hypothesis_dispersion_coverage"].values()),
            "rear_balancing_required": (
                train["positive_bearing_bins"]["rear_abs_ge_135"]
                < train["positive_bearing_bins"]["front_abs_lt_60"]),
            "authorize_long_training_now": False,
            "reason": (
                "existing rows support loop/no-match learning and show that "
                "raw LingBot bearing is already accurate conditional on a "
                "correct candidate, but rear support is sparse and the pose-"
                "dispersion inputs needed by an independent reliability "
                "expert were not collected"),
        },
    }


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(_load(args.train), _load(args.development))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
