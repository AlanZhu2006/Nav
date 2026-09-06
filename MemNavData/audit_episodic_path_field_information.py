#!/usr/bin/env python3
"""Audit whether an ordered causal history contains useful local route cues.

This is deliberately an *evaluation-only upper bound*.  It reads evaluator
positions saved after the sealed HM3D length experiment; those positions are
never available to CEC at runtime.  A positive result only permits the next
LingBot-coordinate replay gate.  It is not a deployable method result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MemNavData.bearing_diagnostics import (
    bearing_error_deg_from_world_delta,
)
from MemNavData.episodic_path_field import FrozenEpisodicPath


AUDIT_SCHEMA_VERSION = "episodic_path_field_information_audit_v1_20260902"
EXPECTED_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
EXPECTED_HISTORIES = 48
GAUGE_FACTOR = 7.0
ANGLE_TOLERANCE_DEG = 1e-6
INDEX_PATTERN = re.compile(r"^(\d{3})_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_facing(delta_xz: Sequence[float]) -> float:
    delta = np.asarray(delta_xz, dtype=np.float64)
    if (delta.shape != (2,) or not np.isfinite(delta).all()
            or float(np.linalg.norm(delta)) <= 1e-12):
        raise ValueError("world delta must be a finite non-zero 2-vector")
    return float(np.arctan2(-delta[0], -delta[1]))


def world_delta_to_forward_left(
    delta_xz: Sequence[float], robot_yaw: float,
) -> np.ndarray:
    """Convert an evaluator-world direction to ``[forward, left]``."""

    relative_yaw = _wrap_angle(_yaw_facing(delta_xz) - float(robot_yaw))
    return np.asarray(
        [math.cos(relative_yaw), math.sin(relative_yaw)],
        dtype=np.float64,
    )


def world_delta_from_bearing(world_bearing_rad: float) -> np.ndarray:
    """Construct an arbitrary world delta with Habitat's yaw convention."""

    bearing = float(world_bearing_rad)
    if not math.isfinite(bearing):
        raise ValueError("world bearing must be finite")
    return np.asarray(
        [-math.sin(bearing), -math.cos(bearing)], dtype=np.float64)


def _angle_between_deg(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if (a.shape != (2,) or b.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("angle inputs must be finite 2-vectors")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        raise ValueError("angle inputs must be non-zero")
    a = a / a_norm
    b = b / b_norm
    # ``acos(dot)`` loses precision at the gauge-invariance limit where the
    # expected angle is zero.  atan2(cross, dot) measures the same unsigned
    # 2-D angle without magnifying roundoff near a dot product of one.
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(abs(math.atan2(cross, dot))))


def _trace_translations(trace: Sequence[dict[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [[row["x"], 0.0, row["z"]] for row in trace],
        dtype=np.float64,
    )
    if (values.ndim != 2 or values.shape[1] != 3
            or not np.isfinite(values).all()):
        raise ValueError("memory trace has invalid evaluator translations")
    return values


def _frame_position(
    trace: Sequence[dict[str, Any]], frame_idx: int,
) -> int:
    matches = [
        index for index, row in enumerate(trace)
        if int(row["frame_idx"]) == int(frame_idx)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one trace row for frame {frame_idx}, got {len(matches)}")
    return matches[0]


def _initial_plan(plans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in plans if int(row.get("step", -1)) == 0]
    if len(matches) != 1:
        raise ValueError(f"expected one step-0 plan, got {len(matches)}")
    return matches[0]


def _revisit_query(episode: dict[str, Any]) -> dict[str, Any]:
    queries = [
        query
        for pair in episode["pairs"]
        for query in pair["queries"]
        if query["analysis_role"] == "revisit"
    ]
    if len(queries) != 1:
        raise ValueError(f"expected one analysis Revisit query, got {len(queries)}")
    return queries[0]


def _plan_index(path: Path) -> int:
    match = INDEX_PATTERN.match(path.parents[1].name)
    if match is None:
        raise ValueError(f"cannot recover population index from {path}")
    return int(match.group(1))


def _median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot compute a median of an empty collection")
    return float(median(values))


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute the frozen information-gate criteria."""

    bins: dict[str, dict[str, Any]] = {}
    for name in sorted({str(row["bin_name"]) for row in rows}):
        selected = [row for row in rows if row["bin_name"] == name]
        direct_median = _median(row["direct_error_deg"] for row in selected)
        route_median = _median(row["route_error_deg"] for row in selected)
        bins[name] = {
            "n": len(selected),
            "direct_error_median_deg": direct_median,
            "route_error_median_deg": route_median,
            "median_improvement_deg": direct_median - route_median,
            "direct_within_30_count": sum(
                row["direct_error_deg"] <= 30.0 for row in selected),
            "route_within_30_count": sum(
                row["route_error_deg"] <= 30.0 for row in selected),
            "direct_within_30_fraction": sum(
                row["direct_error_deg"] <= 30.0 for row in selected
            ) / len(selected),
            "route_within_30_fraction": sum(
                row["route_error_deg"] <= 30.0 for row in selected
            ) / len(selected),
        }

    direct_coverage = sum(
        row["direct_error_deg"] <= 30.0 for row in rows) / len(rows)
    route_coverage = sum(
        row["route_error_deg"] <= 30.0 for row in rows) / len(rows)
    gauge_max = max(row["gauge_direction_error_deg"] for row in rows)
    progress_failures = sum(not row["progress_finite_monotone"] for row in rows)
    criteria = {
        "two_bins_improve_at_least_10_deg": sum(
            item["median_improvement_deg"] >= 10.0 for item in bins.values()
        ) >= 2,
        "no_bin_worsens_more_than_5_deg": all(
            item["median_improvement_deg"] >= -5.0 for item in bins.values()),
        "within_30_coverage_improves_at_least_20pp": (
            route_coverage - direct_coverage >= 0.20),
        "gauge_error_at_most_1e_6_deg": gauge_max <= ANGLE_TOLERANCE_DEG,
        "all_replay_progress_finite_monotone": progress_failures == 0,
    }
    return {
        "n": len(rows),
        "direct_error_median_deg": _median(
            row["direct_error_deg"] for row in rows),
        "route_error_median_deg": _median(
            row["route_error_deg"] for row in rows),
        "paired_error_improvement_median_deg": _median(
            row["direct_error_deg"] - row["route_error_deg"]
            for row in rows),
        "direct_within_30_fraction": direct_coverage,
        "route_within_30_fraction": route_coverage,
        "within_30_improvement_fraction": route_coverage - direct_coverage,
        "maximum_gauge_direction_error_deg": gauge_max,
        "progress_replay_failure_count": progress_failures,
        "bins": bins,
        "criteria": criteria,
        "information_gate_pass": all(criteria.values()),
        "deployment_gate_status": "not_run",
    }


def audit(
    *, manifest_path: Path, evaluation_root: Path,
) -> dict[str, Any]:
    manifest_sha = _sha256(manifest_path)
    manifest = _load_json(manifest_path)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("manifest episodes must be a list")

    plan_paths = sorted(evaluation_root.glob(
        "*/mono_cec/*_revisit_plans.json"))
    plans_by_index: dict[int, Path] = {}
    for path in plan_paths:
        index = _plan_index(path)
        if index in plans_by_index:
            raise ValueError(f"duplicate plan bundle for population index {index}")
        plans_by_index[index] = path

    validation_errors: list[str] = []
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        validation_errors.append(
            f"manifest sha256 {manifest_sha} != frozen "
            f"{EXPECTED_MANIFEST_SHA256}")
    if len(episodes) != EXPECTED_HISTORIES:
        validation_errors.append(
            f"manifest has {len(episodes)} histories, expected {EXPECTED_HISTORIES}")
    if len(plans_by_index) != EXPECTED_HISTORIES:
        validation_errors.append(
            f"found {len(plans_by_index)} CEC plans, expected {EXPECTED_HISTORIES}")

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for episode in episodes:
        population_index = int(episode["population_index"])
        path = plans_by_index.get(population_index)
        if path is None:
            exclusions.append({
                "population_index": population_index,
                "reason": "missing_plan_bundle",
            })
            continue
        try:
            payload = _load_json(path)
            if payload.get("arm") != "certified":
                raise ValueError("plan bundle is not the certified arm")
            if payload.get("analysis_role_not_forwarded") is not True:
                raise ValueError("analysis role hiding receipt is absent")
            if any(row.get("role_label_visible") is True
                   for row in payload["query_leg"]):
                raise ValueError("runtime plan exposed a role label")

            query = _revisit_query(episode)
            initial = _initial_plan(payload["query_leg"])
            accepted = (
                initial.get("certified_relocalization_accepted") is True
                and initial.get("revisit_adapter_takeover") is True
            )
            if not accepted:
                exclusions.append({
                    "population_index": population_index,
                    "scene": episode["scene"],
                    "reason": "certificate_not_accepted_at_query_start",
                    "certificate_reason": initial.get(
                        "certified_relocalization_reason"),
                })
                continue

            leg_a = payload["memory_traces"]["legA"]
            query_trace = payload["memory_traces"]["query"]
            goal_start = int(initial["goal_start_frame"])
            start_position = _frame_position(leg_a, goal_start - 1)
            if start_position != len(leg_a) - 1:
                raise ValueError(
                    "query start is not the exact causal-history tail")
            anchor_frame = int(initial["certified_graph_target_anchor"])
            anchor_position = _frame_position(leg_a, anchor_frame)
            translations = _trace_translations(leg_a)
            path_field = FrozenEpisodicPath.from_history(
                translations,
                start_index=start_position,
                anchor_index=anchor_position,
                metric_scale_m_per_raw=1.0,
            )
            reference = path_field.sample_raw_xz(min(
                path_field.controller_horizon_m,
                path_field.total_length_m,
            ))
            start_xz = translations[start_position, [0, 2]]
            route_delta = reference - start_xz
            robot_yaw = float(leg_a[start_position]["yaw"])
            route_bearing = world_delta_to_forward_left(
                route_delta, robot_yaw)
            target_delta = world_delta_from_bearing(
                float(query["initial_path_bearing_rad"]))
            direct_bearing = np.asarray(
                initial["memory_bearing_unit"], dtype=np.float64)
            direct_error = bearing_error_deg_from_world_delta(
                direct_bearing, target_delta, robot_yaw)
            route_error = bearing_error_deg_from_world_delta(
                route_bearing, target_delta, robot_yaw)

            transformed = FrozenEpisodicPath.from_history(
                translations * GAUGE_FACTOR,
                start_index=start_position,
                anchor_index=anchor_position,
                metric_scale_m_per_raw=1.0 / GAUGE_FACTOR,
            )
            transformed_reference = transformed.sample_raw_xz(min(
                transformed.controller_horizon_m,
                transformed.total_length_m,
            ))
            transformed_delta = (
                transformed_reference
                - GAUGE_FACTOR * start_xz
            )
            gauge_error = _angle_between_deg(route_delta, transformed_delta)

            progress = 0.0
            max_cross_track = 0.0
            monotone = True
            finite = True
            for trace_row in query_trace:
                projection = path_field.project_monotone(
                    [trace_row["x"], 0.0, trace_row["z"]],
                    minimum_progress_m=progress,
                )
                finite = finite and all(math.isfinite(value) for value in (
                    projection.progress_m, projection.cross_track_m))
                monotone = (
                    monotone
                    and projection.progress_m + 1e-9 >= progress
                )
                progress = projection.progress_m
                max_cross_track = max(
                    max_cross_track, projection.cross_track_m)

            rows.append({
                "population_index": population_index,
                "scene": episode["scene"],
                "episode": episode["episode"],
                "bin_name": episode["bin_name"],
                "query_id": query["query_id"],
                "geodesic_from_a_end_m": float(
                    query["geodesic_from_a_end_m"]),
                "selected_anchor_frame": anchor_frame,
                "goal_start_frame": goal_start,
                "history_route_length_m": path_field.total_length_m,
                "direct_error_deg": direct_error,
                "route_error_deg": route_error,
                "paired_error_improvement_deg": direct_error - route_error,
                "gauge_direction_error_deg": gauge_error,
                "progress_finite_monotone": bool(finite and monotone),
                "replay_final_progress_m": progress,
                "replay_max_cross_track_m": max_cross_track,
                "source_plan_sha256": _sha256(path),
            })
        except Exception as error:  # Keep an auditable census of every row.
            exclusions.append({
                "population_index": population_index,
                "scene": episode.get("scene"),
                "reason": "invalid_audit_row",
                "detail": f"{type(error).__name__}: {error}",
            })

    summary = summarize_rows(rows) if rows else None
    if summary is not None and validation_errors:
        summary["information_gate_pass"] = False
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "claim_boundary": (
            "evaluation-route information upper bound; not deployable CEC"),
        "source": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha,
            "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "evaluation_root": str(evaluation_root.resolve()),
            "cec_plan_count": len(plan_paths),
        },
        "population": {
            "manifest_histories": len(episodes),
            "audited_certificate_accepts": len(rows),
            "excluded_count": len(exclusions),
        },
        "validation_errors": validation_errors,
        "exclusions": exclusions,
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit(
        manifest_path=arguments.manifest,
        evaluation_root=arguments.evaluation_root,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    summary = result["summary"]
    print(json.dumps({
        "population": result["population"],
        "validation_errors": result["validation_errors"],
        "summary": summary,
    }, indent=2, sort_keys=True))
    return 0 if summary is not None and not result["validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
