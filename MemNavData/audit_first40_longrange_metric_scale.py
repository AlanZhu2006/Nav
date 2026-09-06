#!/usr/bin/env python3
"""Audit whether the frozen first-40 LingBot scale can metricize CEC range.

The canonical CEC runtime intentionally exposes only a scale-free bearing.  A
full-mono rollout nevertheless logs, at the same planning instant:

* the raw LingBot current-to-goal vector authorized by CEC;
* the immutable first-40 camera-height scale receipt used by dense depth; and
* Habitat's planar Euclidean current-to-goal distance (evaluation only).

This script combines those fields *post hoc*.  It never changes an action and
does not rerun LingBot, PnP, or the controller.  The primary unit is the first
authorized CEC handoff per query, so repeated planning steps cannot inflate N.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
from statistics import mean, median
from typing import Iterable, Mapping, Sequence


DEFAULT_PLAN_GLOB = "*/mono_cec/*_revisit_plans.json"
DEFAULT_FIXED_RADIUS_M = 2.5
FIRST40_SCALE_SCHEMA = "mdtec_first40_scale_receipt_v1_20260819"
FIRST40_SCALE_CONTRACT = "causal_first_prefix_rgb_only_v1"
RAW_VECTOR_UNITS = "lingbot_raw_direction_only"
_POPULATION_ID = re.compile(
    r"^(?P<rank>[0-9]+)_(?P<scene>.+)_(?P<episode>episode_[0-9]+)$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    _require(not isinstance(value, bool), f"{name} must not be bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    _require(math.isfinite(number), f"{name} must be finite")
    if positive:
        _require(number > 0.0, f"{name} must be positive")
    return number


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "distribution requires at least one value")
    return {
        "mean": mean(values),
        "median": median(values),
        "p10": _quantile(values, 0.10),
        "p90": _quantile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    _require(len(left) == len(right), "correlation vectors differ in length")
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return sum(
        lvalue * rvalue
        for lvalue, rvalue in zip(left_centered, right_centered)
    ) / (left_norm * right_norm)


def _exact_two_sided_sign_p(wins: int, losses: int) -> float:
    non_ties = int(wins) + int(losses)
    if non_ties == 0:
        return 1.0
    tail = min(int(wins), int(losses))
    probability = sum(
        math.comb(non_ties, index) for index in range(tail + 1)
    ) / (2.0 ** non_ties)
    return min(1.0, 2.0 * probability)


@dataclass(frozen=True)
class MetricScaleRow:
    population_id: str
    scene: str
    episode: str
    plan_path: str
    query_step: int
    stream_frame: int
    selected_anchor: int
    history_gap_frames: int
    raw_forward: float
    raw_left: float
    raw_norm: float
    first40_scale_m_per_raw: float
    implied_scale_m_per_raw: float
    predicted_distance_m: float
    gt_planar_distance_m: float
    predicted_over_gt: float
    signed_error_m: float
    absolute_error_m: float
    absolute_relative_error: float
    fixed_radius_m: float
    fixed_absolute_error_m: float
    fixed_absolute_relative_error: float
    metric_closer_than_fixed: bool
    scale_valid_frame_ratio: float
    scale_relative_floor_iqr: float
    scale_clamped: bool
    pnp_inliers: int
    pnp_inlier_ratio: float
    pnp_reprojection_rmse_px: float
    pnp_query_coverage: float
    pnp_reference_coverage: float
    pnp_world_planarity: float
    pnp_world_spread_raw: float


@dataclass(frozen=True)
class MetricTrajectoryPoint:
    population_id: str
    scene: str
    query_step: int
    predicted_distance_m: float
    bounded_distance_m: float
    gt_planar_distance_m: float
    metric_absolute_error_m: float
    bounded_absolute_error_m: float
    fixed_absolute_error_m: float


def _population_parts(plan_path: Path) -> tuple[str, str, str]:
    population_id = plan_path.parents[1].name
    matched = _POPULATION_ID.fullmatch(population_id)
    _require(matched is not None, f"unrecognized population id: {population_id}")
    assert matched is not None
    return population_id, matched.group("scene"), matched.group("episode")


def _first_authorized_plan(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    query_leg = payload.get("query_leg")
    _require(isinstance(query_leg, list), "plan payload lacks query_leg list")
    accepted = [
        plan for plan in query_leg
        if isinstance(plan, dict)
        and plan.get("certified_relocalization_accepted") is True
        and plan.get("revisit_adapter_takeover") is True
    ]
    if not accepted:
        return None
    return min(accepted, key=lambda plan: int(plan.get("step", -1)))


def extract_row(
    plan_path: Path,
    *,
    fixed_radius_m: float = DEFAULT_FIXED_RADIUS_M,
) -> MetricScaleRow | None:
    """Extract one first-handoff row and enforce the frozen logging contract."""

    payload = json.loads(plan_path.read_text())
    _require(isinstance(payload, dict), f"{plan_path}: root must be an object")
    _require(payload.get("arm") == "certified", f"{plan_path}: wrong arm")
    _require(
        payload.get("analysis_role_not_forwarded") is True,
        f"{plan_path}: role-hiding receipt missing",
    )
    plan = _first_authorized_plan(payload)
    if plan is None:
        return None

    _require(
        plan.get("memory_unbounded_pointgoal_units") == RAW_VECTOR_UNITS,
        f"{plan_path}: raw vector units changed",
    )
    _require(
        plan.get("memory_unbounded_pointgoal_distance_m") is None,
        f"{plan_path}: runtime unexpectedly exposed metric range",
    )
    _require(
        plan.get("metric_depth_sensor_consumed") is False,
        f"{plan_path}: simulator metric depth was consumed",
    )
    _require(
        plan.get("certified_relocalization_accepted") is True,
        f"{plan_path}: handoff is not certificate-authorized",
    )

    raw_vector = plan.get("memory_unbounded_pointgoal")
    _require(
        isinstance(raw_vector, list) and len(raw_vector) == 2,
        f"{plan_path}: malformed raw pointgoal",
    )
    raw_forward = _finite(raw_vector[0], "raw_forward")
    raw_left = _finite(raw_vector[1], "raw_left")
    logged_raw_norm = _finite(
        plan.get("memory_unbounded_pointgoal_norm"), "raw_norm", positive=True
    )
    recomputed_raw_norm = math.hypot(raw_forward, raw_left)
    _require(
        math.isclose(logged_raw_norm, recomputed_raw_norm, rel_tol=1e-9, abs_tol=1e-9),
        f"{plan_path}: raw vector norm does not reproduce",
    )

    depth_receipt = plan.get("monocular_depth_receipt")
    _require(isinstance(depth_receipt, dict), f"{plan_path}: depth receipt missing")
    _require(
        depth_receipt.get("metric_depth_sensor_consumed") is False,
        f"{plan_path}: depth receipt consumed simulator depth",
    )
    _require(depth_receipt.get("scale_active") is True, f"{plan_path}: scale inactive")
    scale_receipt = depth_receipt.get("scale_receipt")
    _require(isinstance(scale_receipt, dict), f"{plan_path}: scale receipt missing")
    _require(
        scale_receipt.get("schema") == FIRST40_SCALE_SCHEMA,
        f"{plan_path}: scale schema changed",
    )
    _require(
        scale_receipt.get("scale_evidence_contract") == FIRST40_SCALE_CONTRACT,
        f"{plan_path}: scale evidence contract changed",
    )
    _require(scale_receipt.get("scale_valid") is True, f"{plan_path}: invalid scale")
    _require(
        int(scale_receipt.get("scale_prefix_frames", -1)) == 40,
        f"{plan_path}: scale prefix is not first-40",
    )
    _require(
        scale_receipt.get("whole_episode_ground_cache_consumed") is False,
        f"{plan_path}: whole-episode scale leakage",
    )

    scale = _finite(scale_receipt.get("scale_hat"), "scale_hat", positive=True)
    gt_distance = _finite(
        plan.get("evaluation_gt_goal_distance_m"),
        "evaluation_gt_goal_distance_m",
        positive=True,
    )
    predicted_distance = scale * logged_raw_norm
    signed_error = predicted_distance - gt_distance
    absolute_error = abs(signed_error)
    relative_error = absolute_error / gt_distance
    fixed_error = abs(float(fixed_radius_m) - gt_distance)
    fixed_relative_error = fixed_error / gt_distance

    logged_fixed_radius = _finite(
        plan.get("memory_pointgoal_fixed_radius_m"),
        "memory_pointgoal_fixed_radius_m",
        positive=True,
    )
    _require(
        math.isclose(logged_fixed_radius, fixed_radius_m, rel_tol=0.0, abs_tol=1e-12),
        f"{plan_path}: fixed-radius contract changed",
    )
    _require(
        math.isclose(
            _finite(
                plan.get("memory_controller_pointgoal_distance_m"),
                "memory_controller_pointgoal_distance_m",
                positive=True,
            ),
            fixed_radius_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{plan_path}: controller did not receive the frozen radius",
    )

    pnp = plan.get("certified_relocalization_pnp")
    _require(isinstance(pnp, dict), f"{plan_path}: PnP receipt missing")
    population_id, scene, episode = _population_parts(plan_path)
    stream_frame = int(plan.get("frame_idx", -1))
    selected_anchor = int(plan.get("certified_graph_target_anchor", -1))
    _require(stream_frame >= 0, f"{plan_path}: invalid stream frame")
    _require(selected_anchor >= 0, f"{plan_path}: invalid selected anchor")

    return MetricScaleRow(
        population_id=population_id,
        scene=scene,
        episode=episode,
        plan_path=str(plan_path.resolve()),
        query_step=int(plan.get("step", -1)),
        stream_frame=stream_frame,
        selected_anchor=selected_anchor,
        history_gap_frames=stream_frame - selected_anchor,
        raw_forward=raw_forward,
        raw_left=raw_left,
        raw_norm=logged_raw_norm,
        first40_scale_m_per_raw=scale,
        implied_scale_m_per_raw=gt_distance / logged_raw_norm,
        predicted_distance_m=predicted_distance,
        gt_planar_distance_m=gt_distance,
        predicted_over_gt=predicted_distance / gt_distance,
        signed_error_m=signed_error,
        absolute_error_m=absolute_error,
        absolute_relative_error=relative_error,
        fixed_radius_m=float(fixed_radius_m),
        fixed_absolute_error_m=fixed_error,
        fixed_absolute_relative_error=fixed_relative_error,
        metric_closer_than_fixed=absolute_error < fixed_error,
        scale_valid_frame_ratio=_finite(
            scale_receipt.get("valid_frame_ratio"), "valid_frame_ratio"
        ),
        scale_relative_floor_iqr=_finite(
            scale_receipt.get("relative_floor_iqr"), "relative_floor_iqr"
        ),
        scale_clamped=bool(scale_receipt.get("scale_clamped")),
        pnp_inliers=int(pnp.get("inliers", -1)),
        pnp_inlier_ratio=_finite(pnp.get("inlier_ratio"), "pnp_inlier_ratio"),
        pnp_reprojection_rmse_px=_finite(
            pnp.get("reprojection_rmse_px"), "pnp_reprojection_rmse_px"
        ),
        pnp_query_coverage=_finite(
            pnp.get("query_inlier_coverage"), "pnp_query_coverage"
        ),
        pnp_reference_coverage=_finite(
            pnp.get("reference_inlier_coverage"), "pnp_reference_coverage"
        ),
        pnp_world_planarity=_finite(
            pnp.get("world_inlier_planarity"), "pnp_world_planarity"
        ),
        pnp_world_spread_raw=_finite(
            pnp.get("world_inlier_spread_raw"), "pnp_world_spread_raw"
        ),
    )


def extract_trajectory_points(
    plan_path: Path,
    *,
    fixed_radius_m: float = DEFAULT_FIXED_RADIUS_M,
) -> list[MetricTrajectoryPoint]:
    """Extract every authorized readout for an episode-balanced diagnostic.

    These repeated measurements are never treated as independent samples.  A
    downstream summary first reduces them to one correlation/error statistic
    per query.  The trajectories were generated by the fixed-radius controller,
    so the bounded/metric columns remain counterfactual readouts, not rollouts.
    """

    first = extract_row(plan_path, fixed_radius_m=fixed_radius_m)
    if first is None:
        return []
    payload = json.loads(plan_path.read_text())
    query_leg = payload["query_leg"]
    points: list[MetricTrajectoryPoint] = []
    for plan in sorted(
        (
            item for item in query_leg
            if isinstance(item, dict)
            and item.get("certified_relocalization_accepted") is True
            and item.get("revisit_adapter_takeover") is True
        ),
        key=lambda item: int(item.get("step", -1)),
    ):
        _require(
            plan.get("memory_unbounded_pointgoal_units") == RAW_VECTOR_UNITS,
            f"{plan_path}: trajectory raw vector units changed",
        )
        _require(
            plan.get("metric_depth_sensor_consumed") is False,
            f"{plan_path}: trajectory consumed simulator depth",
        )
        raw_norm = _finite(
            plan.get("memory_unbounded_pointgoal_norm"),
            "trajectory_raw_norm",
            positive=True,
        )
        receipt = plan.get("monocular_depth_receipt")
        _require(
            isinstance(receipt, dict) and receipt.get("scale_active") is True,
            f"{plan_path}: trajectory scale inactive",
        )
        scale_receipt = receipt.get("scale_receipt")
        _require(
            isinstance(scale_receipt, dict)
            and scale_receipt.get("schema") == FIRST40_SCALE_SCHEMA
            and scale_receipt.get("scale_valid") is True,
            f"{plan_path}: trajectory first-40 scale invalid",
        )
        scale = _finite(
            scale_receipt.get("scale_hat"), "trajectory_scale_hat", positive=True
        )
        _require(
            math.isclose(
                scale,
                first.first40_scale_m_per_raw,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"{plan_path}: frozen first-40 scale changed during query",
        )
        gt = _finite(
            plan.get("evaluation_gt_goal_distance_m"),
            "trajectory_gt_distance",
            positive=True,
        )
        predicted = scale * raw_norm
        bounded = min(predicted, float(fixed_radius_m))
        points.append(MetricTrajectoryPoint(
            population_id=first.population_id,
            scene=first.scene,
            query_step=int(plan.get("step", -1)),
            predicted_distance_m=predicted,
            bounded_distance_m=bounded,
            gt_planar_distance_m=gt,
            metric_absolute_error_m=abs(predicted - gt),
            bounded_absolute_error_m=abs(bounded - gt),
            fixed_absolute_error_m=abs(float(fixed_radius_m) - gt),
        ))
    return points


def _trajectory_readout_diagnostic(
    plan_paths: Sequence[Path],
    *,
    fixed_radius_m: float,
) -> dict[str, object]:
    grouped = {
        path.parents[1].name: extract_trajectory_points(
            path, fixed_radius_m=fixed_radius_m
        )
        for path in plan_paths
    }
    grouped = {key: value for key, value in grouped.items() if value}
    _require(bool(grouped), "no trajectory readouts were extracted")

    per_query_correlations: list[float] = []
    metric_mae: list[float] = []
    bounded_mae: list[float] = []
    fixed_mae: list[float] = []
    terminal_metric_error: list[float] = []
    terminal_bounded_error: list[float] = []
    readout_counts: list[float] = []
    for points in grouped.values():
        readout_counts.append(float(len(points)))
        correlation = _pearson(
            [point.predicted_distance_m for point in points],
            [point.gt_planar_distance_m for point in points],
        )
        if correlation is not None:
            per_query_correlations.append(correlation)
        metric_mae.append(mean(point.metric_absolute_error_m for point in points))
        bounded_mae.append(mean(point.bounded_absolute_error_m for point in points))
        fixed_mae.append(mean(point.fixed_absolute_error_m for point in points))
        terminal = points[-1]
        terminal_metric_error.append(terminal.metric_absolute_error_m)
        terminal_bounded_error.append(terminal.bounded_absolute_error_m)

    def paired(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
        wins = sum(lvalue < rvalue for lvalue, rvalue in zip(left, right))
        losses = sum(lvalue > rvalue for lvalue, rvalue in zip(left, right))
        ties = len(left) - wins - losses
        return {
            "left_closer": wins,
            "right_closer": losses,
            "ties": ties,
            "exact_two_sided_sign_p": _exact_two_sided_sign_p(wins, losses),
        }

    return {
        "claim_boundary": (
            "episode-balanced post-hoc readout diagnostic; all trajectories "
            "were generated by the canonical fixed-2.5m controller"
        ),
        "query_count": len(grouped),
        "authorized_readout_count": int(sum(readout_counts)),
        "authorized_readouts_per_query": _distribution(readout_counts),
        "per_query_metric_vs_gt_pearson": (
            _distribution(per_query_correlations)
            if per_query_correlations else None
        ),
        "queries_with_positive_metric_vs_gt_correlation": sum(
            value > 0.0 for value in per_query_correlations
        ),
        "episode_balanced_mean_absolute_error_m": {
            "height_scaled_metric": _distribution(metric_mae),
            "bounded_height_scaled_metric_max_2_5m": _distribution(bounded_mae),
            "fixed_2_5m": _distribution(fixed_mae),
        },
        "paired_query_mae_height_scaled_metric_vs_fixed": paired(
            metric_mae, fixed_mae
        ),
        "paired_query_mae_bounded_metric_vs_fixed": paired(
            bounded_mae, fixed_mae
        ),
        "terminal_readout_absolute_error_m": {
            "height_scaled_metric": _distribution(terminal_metric_error),
            "bounded_height_scaled_metric_max_2_5m": _distribution(
                terminal_bounded_error
            ),
            "metric_within_0_5m": sum(
                value <= 0.5 for value in terminal_metric_error
            ),
            "bounded_metric_within_0_5m": sum(
                value <= 0.5 for value in terminal_bounded_error
            ),
        },
    }


def _cluster_bootstrap_ci(
    rows: Sequence[MetricScaleRow],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Scene-cluster bootstrap for mean(metric error - fixed error)."""

    grouped: dict[str, list[MetricScaleRow]] = {}
    for row in rows:
        grouped.setdefault(row.scene, []).append(row)
    scenes = sorted(grouped)
    rng = random.Random(int(seed))
    absolute_deltas: list[float] = []
    relative_deltas: list[float] = []
    for _ in range(int(samples)):
        sampled_rows: list[MetricScaleRow] = []
        for _ in scenes:
            sampled_rows.extend(grouped[rng.choice(scenes)])
        absolute_deltas.append(mean(
            row.absolute_error_m - row.fixed_absolute_error_m
            for row in sampled_rows
        ))
        relative_deltas.append(mean(
            row.absolute_relative_error - row.fixed_absolute_relative_error
            for row in sampled_rows
        ))
    return {
        "unit": "scene",
        "scene_count": len(scenes),
        "samples": int(samples),
        "seed": int(seed),
        "mean_absolute_error_delta_metric_minus_fixed_m_ci95": [
            _quantile(absolute_deltas, 0.025),
            _quantile(absolute_deltas, 0.975),
        ],
        "mean_absolute_relative_error_delta_metric_minus_fixed_ci95": [
            _quantile(relative_deltas, 0.025),
            _quantile(relative_deltas, 0.975),
        ],
    }


def summarize_rows(
    rows: Sequence[MetricScaleRow],
    *,
    plan_count: int,
    fixed_radius_m: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    _require(bool(rows), "no first-handoff rows were extracted")
    metric_errors = [row.absolute_error_m for row in rows]
    metric_relative = [row.absolute_relative_error for row in rows]
    fixed_errors = [row.fixed_absolute_error_m for row in rows]
    fixed_relative = [row.fixed_absolute_relative_error for row in rows]
    wins = sum(
        row.absolute_error_m < row.fixed_absolute_error_m for row in rows
    )
    losses = sum(
        row.absolute_error_m > row.fixed_absolute_error_m for row in rows
    )
    ties = len(rows) - wins - losses
    outliers = [
        asdict(row) for row in sorted(
            rows, key=lambda item: item.absolute_relative_error, reverse=True
        )
        if row.absolute_relative_error > 0.30
    ]
    return {
        "schema": "first40_longrange_metric_scale_posthoc_audit_v1_20260831",
        "claim_boundary": (
            "post-hoc measurement only; no runtime action changed and no "
            "closed-loop superiority claim is authorized"
        ),
        "unit": "first authorized CEC handoff per Revisit query",
        "plan_files": int(plan_count),
        "authorized_queries": len(rows),
        "queries_without_authorized_handoff": int(plan_count - len(rows)),
        "scene_count": len({row.scene for row in rows}),
        "all_first_handoffs_at_step_zero": all(row.query_step == 0 for row in rows),
        "fixed_radius_m": float(fixed_radius_m),
        "metric_distance_m": _distribution(
            [row.predicted_distance_m for row in rows]
        ),
        "gt_planar_distance_m": _distribution(
            [row.gt_planar_distance_m for row in rows]
        ),
        "predicted_over_gt": _distribution(
            [row.predicted_over_gt for row in rows]
        ),
        "first40_scale_m_per_raw": _distribution(
            [row.first40_scale_m_per_raw for row in rows]
        ),
        "implied_scale_m_per_raw": _distribution(
            [row.implied_scale_m_per_raw for row in rows]
        ),
        "metric_error": {
            "absolute_error_m": _distribution(metric_errors),
            "absolute_relative_error": _distribution(metric_relative),
            "within_20_percent": sum(value <= 0.20 for value in metric_relative),
            "within_30_percent": sum(value <= 0.30 for value in metric_relative),
        },
        "fixed_radius_error": {
            "absolute_error_m": _distribution(fixed_errors),
            "absolute_relative_error": _distribution(fixed_relative),
            "within_20_percent": sum(value <= 0.20 for value in fixed_relative),
            "within_30_percent": sum(value <= 0.30 for value in fixed_relative),
        },
        "paired_metric_vs_fixed": {
            "metric_closer": wins,
            "fixed_closer": losses,
            "ties": ties,
            "exact_two_sided_sign_p": _exact_two_sided_sign_p(wins, losses),
            "mean_absolute_error_delta_metric_minus_fixed_m": mean(
                row.absolute_error_m - row.fixed_absolute_error_m for row in rows
            ),
            "mean_absolute_relative_error_delta_metric_minus_fixed": mean(
                row.absolute_relative_error
                - row.fixed_absolute_relative_error
                for row in rows
            ),
        },
        "correlations": {
            "metric_distance_vs_gt_pearson": _pearson(
                [row.predicted_distance_m for row in rows],
                [row.gt_planar_distance_m for row in rows],
            ),
            "raw_norm_vs_gt_pearson": _pearson(
                [row.raw_norm for row in rows],
                [row.gt_planar_distance_m for row in rows],
            ),
            "first40_scale_vs_implied_scale_pearson": _pearson(
                [row.first40_scale_m_per_raw for row in rows],
                [row.implied_scale_m_per_raw for row in rows],
            ),
            "relative_floor_iqr_vs_metric_relative_error_pearson": _pearson(
                [row.scale_relative_floor_iqr for row in rows],
                [row.absolute_relative_error for row in rows],
            ),
        },
        "scale_receipt_quality": {
            "clamped_count": sum(row.scale_clamped for row in rows),
            "relative_floor_iqr": _distribution(
                [row.scale_relative_floor_iqr for row in rows]
            ),
            "valid_frame_ratio": _distribution(
                [row.scale_valid_frame_ratio for row in rows]
            ),
        },
        "scene_cluster_bootstrap": _cluster_bootstrap_ci(
            rows, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "metric_relative_error_above_30_percent": outliers,
    }


def audit(
    evaluation_root: Path,
    *,
    plan_glob: str = DEFAULT_PLAN_GLOB,
    fixed_radius_m: float = DEFAULT_FIXED_RADIUS_M,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260831,
) -> tuple[list[MetricScaleRow], dict[str, object]]:
    plan_paths = sorted(evaluation_root.glob(plan_glob))
    _require(bool(plan_paths), f"no plan files match {plan_glob!r}")
    rows = [
        row for plan_path in plan_paths
        if (row := extract_row(plan_path, fixed_radius_m=fixed_radius_m)) is not None
    ]
    summary = summarize_rows(
        rows,
        plan_count=len(plan_paths),
        fixed_radius_m=fixed_radius_m,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["evaluation_root"] = str(evaluation_root.resolve())
    summary["plan_glob"] = plan_glob
    summary["trajectory_readout_diagnostic"] = _trajectory_readout_diagnostic(
        plan_paths, fixed_radius_m=fixed_radius_m
    )
    return rows, summary


def _write_rows(path: Path, rows: Iterable[MetricScaleRow]) -> None:
    materialized = [asdict(row) for row in rows]
    _require(bool(materialized), "cannot write an empty row table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--plan-glob", default=DEFAULT_PLAN_GLOB)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-radius-m", type=float, default=DEFAULT_FIXED_RADIUS_M)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()

    _require(args.fixed_radius_m > 0.0, "fixed radius must be positive")
    _require(args.bootstrap_samples > 0, "bootstrap samples must be positive")
    rows, summary = audit(
        args.evaluation_root,
        plan_glob=args.plan_glob,
        fixed_radius_m=args.fixed_radius_m,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "first_handoff_rows.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
