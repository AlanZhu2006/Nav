#!/usr/bin/env python3
"""Audit the *independent* certificate lifecycle in a certified 3-leg run.

The ordinary closed-loop report counts one certificate decision per planning
request.  That is useful for checking that the controller stayed in its
intended mode, but it is not the number of independent image-localization
events: a successful localization is cached for the remainder of a goal leg.
This audit separates the first LightGlue/PnP/certificate computation from the
subsequent cached bearing updates and records the construction population that
the positive-only double-Revisit benchmark represents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "certified_3leg_lifecycle_audit_v1_20260813"
LEGS = ("legB", "legC")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean(value: Any) -> bool:
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False", ""):
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def metric_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"metric file must have one row: {path}")
    return rows[0]


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    require(0.0 <= fraction <= 1.0, "percentile fraction outside [0, 1]")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bearing_key(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    return tuple(round(float(component), 9) for component in value)


def world_to_local_forward_left(
        current: dict[str, Any], target_xz: tuple[float, float]) -> tuple[float, float]:
    """Habitat XZ target in the current ``[forward, left]`` frame."""
    yaw = float(current["yaw"])
    dx = float(target_xz[0]) - float(current["x"])
    dz = float(target_xz[1]) - float(current["z"])
    return (
        -math.sin(yaw) * dx - math.cos(yaw) * dz,
        -math.cos(yaw) * dx + math.sin(yaw) * dz,
    )


def angular_error_deg(first: Iterable[float], second: Iterable[float]) -> float:
    left = tuple(float(value) for value in first)
    right = tuple(float(value) for value in second)
    require(len(left) == len(right) == 2, "bearing must have two components")
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    require(left_norm > 1e-12 and right_norm > 1e-12,
            "bearing is degenerate")
    cosine = sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def trace_path_length(trace: list[dict[str, Any]], first: int, last: int) -> float:
    require(0 <= first < len(trace) and 0 <= last < len(trace),
            "trace endpoint outside history")
    lo, hi = sorted((int(first), int(last)))
    return sum(
        math.hypot(
            float(trace[index + 1]["x"]) - float(trace[index]["x"]),
            float(trace[index + 1]["z"]) - float(trace[index]["z"]),
        )
        for index in range(lo, hi)
    )


def _plan_bearing_errors(
        record: dict[str, Any], leg: str,
) -> list[tuple[int, float]]:
    target = tuple(record["goal_xz"][leg])
    trace_by_step = {
        int(row["step"]): row for row in record["rollout_traces"].get(leg, [])
    }
    errors = []
    for plan in record["plans"].get(leg, []):
        predicted = plan.get("memory_bearing_unit")
        step = int(plan.get("step", -1))
        if predicted is None or step not in trace_by_step:
            continue
        oracle = world_to_local_forward_left(trace_by_step[step], target)
        try:
            error = angular_error_deg(predicted, oracle)
        except RuntimeError:
            continue
        errors.append((step, error))
    return errors


def summarize_leg(records: list[dict[str, Any]], leg: str) -> dict[str, Any]:
    """Summarize one goal leg from per-episode plan/metric records."""
    require(leg in LEGS, f"unsupported leg: {leg}")
    request_rows: list[dict[str, Any]] = []
    uncached_rows: list[dict[str, Any]] = []
    episodes_with_leg = 0
    episodes_with_requests = 0
    exactly_one_uncached = 0
    cache_order_valid = 0
    selected_anchor_stable = 0
    shortlist_stable = 0
    bearing_changed_after_cache = 0
    navigation_success = 0
    failed_after_accepted_certificate = []
    all_bearing_errors = []
    first_bearing_errors = []
    episode_median_bearing_errors = []
    episode_max_bearing_errors = []

    for record in records:
        plans = list(record["plans"].get(leg, []))
        if plans:
            episodes_with_leg += 1
        cert = [
            row for row in plans
            if row.get("certified_relocalization_cached") is not None
        ]
        if not cert:
            continue
        episodes_with_requests += 1
        request_rows.extend(cert)
        fresh = [
            row for row in cert
            if row.get("certified_relocalization_cached") is False
        ]
        uncached_rows.extend(fresh)
        if len(fresh) == 1:
            exactly_one_uncached += 1
        cache_flags = [row.get("certified_relocalization_cached") for row in cert]
        if cache_flags and cache_flags[0] is False and all(
                value is True for value in cache_flags[1:]):
            cache_order_valid += 1
        selected = {row.get("router_selected_anchor") for row in cert}
        if len(selected) == 1:
            selected_anchor_stable += 1
        shortlists = {
            tuple(row.get("router_candidate_order_dino") or []) for row in cert
        }
        if len(shortlists) == 1:
            shortlist_stable += 1
        bearings = {
            key for key in map(
                _bearing_key,
                (row.get("memory_bearing_unit") for row in cert),
            ) if key is not None
        }
        if len(cert) > 1 and len(bearings) > 1:
            bearing_changed_after_cache += 1

        succeeded = bool(record["success"][leg])
        navigation_success += int(succeeded)
        bearing_errors = _plan_bearing_errors(record, leg)
        if bearing_errors:
            values = [value for _step, value in bearing_errors]
            all_bearing_errors.extend(values)
            first_bearing_errors.append(values[0])
            episode_median_bearing_errors.append(statistics.median(values))
            episode_max_bearing_errors.append(max(values))
        if (not succeeded and fresh
                and bool(fresh[0].get("certified_relocalization_accepted"))):
            failed_after_accepted_certificate.append({
                "selection_index": int(record["selection_index"]),
                "scene": str(record["scene"]),
                "episode": str(record["episode"]),
                "termination": record["termination"][leg],
                "steps": int(record["steps"][leg]),
                "final_distance_m": float(record["final_distance_m"][leg]),
                "bearing_error_deg": ({
                    "first": bearing_errors[0][1],
                    "median": statistics.median(
                        value for _step, value in bearing_errors),
                    "maximum": max(value for _step, value in bearing_errors),
                } if bearing_errors else None),
            })

    accepted_requests = sum(
        row.get("certified_relocalization_accepted") is True
        for row in request_rows)
    accepted_uncached = sum(
        row.get("certified_relocalization_accepted") is True
        for row in uncached_rows)
    pnp_rows = [
        row.get("certified_relocalization_pnp") or {} for row in uncached_rows
    ]
    latency_ms = [
        float(row["certified_relocalization_uncached_ms"])
        for row in uncached_rows
        if row.get("certified_relocalization_uncached_ms") is not None
    ]
    ranks = [
        int(row["router_selected_candidate_dino_rank"])
        for row in uncached_rows
        if row.get("router_selected_candidate_dino_rank") is not None
    ]

    return {
        "episodes_with_leg_plans": episodes_with_leg,
        "episodes_with_certificate_requests": episodes_with_requests,
        "planning_requests": len(request_rows),
        "independent_uncached_localizations": len(uncached_rows),
        "cached_reuses": sum(
            row.get("certified_relocalization_cached") is True
            for row in request_rows),
        "accepted_planning_requests": accepted_requests,
        "rejected_planning_requests": len(request_rows) - accepted_requests,
        "accepted_independent_localizations": accepted_uncached,
        "rejected_independent_localizations": (
            len(uncached_rows) - accepted_uncached),
        "episodes_exactly_one_uncached_localization": exactly_one_uncached,
        "episodes_uncached_then_cached_only": cache_order_valid,
        "episodes_selected_anchor_stable": selected_anchor_stable,
        "episodes_shortlist_stable": shortlist_stable,
        "episodes_bearing_changed_after_motion": bearing_changed_after_cache,
        "navigation_success": navigation_success,
        "navigation_failures_after_accepted_certificate": (
            failed_after_accepted_certificate),
        "selected_dino_rank_distribution": {
            str(key): value for key, value in sorted(Counter(ranks).items())
        },
        "selected_raw_dino_top1": sum(rank == 1 for rank in ranks),
        "uncached_latency_ms": {
            "mean": (statistics.fmean(latency_ms) if latency_ms else None),
            "median": percentile(latency_ms, 0.5),
            "p90": percentile(latency_ms, 0.9),
            "minimum": min(latency_ms) if latency_ms else None,
            "maximum": max(latency_ms) if latency_ms else None,
        },
        "pnp": {
            "inliers_median": percentile(
                [row["inliers"] for row in pnp_rows if row.get("inliers") is not None],
                0.5),
            "reprojection_rmse_px_median": percentile(
                [row["reprojection_rmse_px"] for row in pnp_rows
                 if row.get("reprojection_rmse_px") is not None], 0.5),
            "query_inlier_coverage_median": percentile(
                [row["query_inlier_coverage"] for row in pnp_rows
                 if row.get("query_inlier_coverage") is not None], 0.5),
            "reference_inlier_coverage_median": percentile(
                [row["reference_inlier_coverage"] for row in pnp_rows
                 if row.get("reference_inlier_coverage") is not None], 0.5),
        },
        "habitat_gt_bearing_audit": {
            "scope": "diagnostic_only_not_deployment_input",
            "request_count": len(all_bearing_errors),
            "first_request_error_deg_median": percentile(
                first_bearing_errors, 0.5),
            "per_episode_median_error_deg_median": percentile(
                episode_median_bearing_errors, 0.5),
            "per_episode_max_error_deg_median": percentile(
                episode_max_bearing_errors, 0.5),
            "all_request_error_deg_p90": percentile(all_bearing_errors, 0.9),
            "all_request_error_deg_maximum": (
                max(all_bearing_errors) if all_bearing_errors else None),
        },
    }


def multigoal_topology_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Contrast old latest-frame reverse routing with anchor-to-anchor routing.

    This is a trace-geometry diagnostic, not a navigation counterfactual.  It
    uses Habitat rollout positions only after the frozen outcomes exist and
    therefore cannot authorize execution by itself.
    """
    rows = []
    for record in records:
        plans_b = [
            row for row in record["plans"].get("legB", [])
            if row.get("certified_relocalization_cached") is not None
        ]
        plans_c = [
            row for row in record["plans"].get("legC", [])
            if row.get("certified_relocalization_cached") is not None
        ]
        if not plans_b or not plans_c:
            continue
        anchor_b = int(plans_b[0]["router_selected_anchor"])
        anchor_c = int(plans_c[0]["router_selected_anchor"])
        trace_a = record["rollout_traces"]["legA"]
        trace_b = record["rollout_traces"]["legB"]
        require(anchor_b < len(trace_a) and anchor_c < len(trace_a),
                "certified anchor escaped online-A trace")
        anchor_route = trace_path_length(trace_a, anchor_b, anchor_c)
        legacy_reverse = (
            trace_path_length(trace_b, 0, len(trace_b) - 1)
            + trace_path_length(trace_a, len(trace_a) - 1, anchor_c)
        )
        rows.append({
            "selection_index": int(record["selection_index"]),
            "scene": str(record["scene"]),
            "episode": str(record["episode"]),
            "B_anchor": anchor_b,
            "C_anchor": anchor_c,
            "temporal_direction": (
                "forward" if anchor_c > anchor_b
                else "reverse" if anchor_c < anchor_b else "same"),
            "anchor_to_anchor_trace_m": anchor_route,
            "legacy_latest_frame_reverse_trace_m": legacy_reverse,
            "legacy_over_anchor_route_ratio": (
                legacy_reverse / anchor_route if anchor_route > 1e-12 else None),
        })
    ratios = [
        row["legacy_over_anchor_route_ratio"] for row in rows
        if row["legacy_over_anchor_route_ratio"] is not None
    ]
    directions = Counter(row["temporal_direction"] for row in rows)
    return {
        "scope": "post_outcome_trace_geometry_hypothesis_not_navigation_effect",
        "eligible_goal_switches": len(rows),
        "temporal_direction_counts": {
            key: value for key, value in sorted(directions.items())
        },
        "legacy_over_anchor_route_ratio_median": percentile(ratios, 0.5),
        "legacy_over_anchor_route_ratio_mean": (
            statistics.fmean(ratios) if ratios else None),
        "interpretation": (
            "A multi-goal graph must support both temporal directions and "
            "start from the previous certified anchor. The old latest-frame "
            "reverse route is structurally longer, but this audit does not "
            "show that executing the shorter route improves SR."
        ),
        "rows": rows,
    }


def audit(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    manifest_path = run_root / "prepared" / "benchmark" / "manifest.json"
    preparation_path = run_root / "prepared" / "preparation_report.json"
    manifest = read_json(manifest_path)
    preparation = read_json(preparation_path)
    episodes = manifest.get("episodes", [])
    require(isinstance(episodes, list) and episodes, "manifest has no episodes")
    manifest_by_identity = {
        (str(row["scene"]), str(row["episode"])): row for row in episodes
    }

    records = []
    for episode_dir in sorted(
            path for path in (run_root / "scenes").iterdir() if path.is_dir()):
        contract = read_json(episode_dir / "episode_contract.json")
        root = episode_dir / "certified"
        plan_paths = list(root.glob("episode_*_plans.json"))
        require(len(plan_paths) == 1, f"expected one plan receipt: {root}")
        plans = read_json(plan_paths[0])
        metric = metric_row(root / "metric.csv")
        identity = (str(metric["scene"]), str(metric["episode"]))
        require(identity in manifest_by_identity,
                f"certified identity absent from manifest: {identity}")
        benchmark = manifest_by_identity[identity]
        variant = benchmark["variants"][str(metric["variant"])]
        require(metric["hybrid_route"] == "certified_relocalization",
                f"wrong route: {root}")
        require(metric["known_revisit_scope"] == "both",
                f"wrong certified scope: {root}")
        require(boolean(metric["shared_A_hashes_ok"]),
                f"shared A hash failed: {root}")
        records.append({
            "selection_index": int(contract["selection_index"]),
            "scene": str(metric["scene"]),
            "episode": str(metric["episode"]),
            "plans": plans,
            "rollout_traces": plans["rollout_traces"],
            "goal_xz": {
                leg: tuple(float(value) for value in
                           variant["goals"][role]["floor_position"])[::2]
                for leg, role in (("legB", "B"), ("legC", "C"))
            },
            "success": {
                "legB": boolean(metric["reached_B"]),
                "legC": boolean(metric["reached_C"]),
            },
            "termination": {
                "legB": str(metric["termination_B"]),
                "legC": str(metric["termination_C"]),
            },
            "steps": {
                "legB": int(metric["steps_B"]),
                "legC": int(metric["steps_C"]),
            },
            "final_distance_m": {
                "legB": float(metric["final_dist_B"]),
                "legC": float(metric["final_dist_C"]),
            },
        })
    records.sort(key=lambda row: row["selection_index"])
    require(len(records) == len(episodes),
            "completed certified episode count differs from manifest")

    statuses = preparation.get("candidate_status", [])
    failure_reasons = Counter(
        str(row.get("reason") or row.get("construction_failure"))
        for row in statuses if not bool(row.get("constructible"))
    )
    contract = manifest.get("contract", {})
    construction = {
        "candidate_population": str(
            manifest.get("selection", {}).get("candidate_population")),
        "candidate_count": int(preparation["candidate_count"]),
        "constructible_count": int(preparation["constructible_count"]),
        "constructible_fraction": (
            float(preparation["constructible_count"])
            / float(preparation["candidate_count"])),
        "selected_count": len(episodes),
        "selected_scene_count": int(preparation["selected_scene_count"]),
        "construction_failure_reasons": {
            key: value for key, value in sorted(failure_reasons.items())
        },
        "selection_observed_navigation_outcomes": bool(
            preparation["causal_seal"][
                "selection_observed_navigation_outcomes"]),
        "positive_support_contract": {
            key: contract.get(key) for key in (
                "v1_min_source_frame_covis",
                "v1_min_max_online_a_covis",
                "v1_max_max_online_a_covis",
                "v1_max_argmax_gap_frames",
            )
        },
    }

    leg_summary = {leg: summarize_leg(records, leg) for leg in LEGS}
    independent = sum(
        row["independent_uncached_localizations"]
        for row in leg_summary.values())
    requests = sum(row["planning_requests"] for row in leg_summary.values())
    accepted = sum(
        row["accepted_independent_localizations"]
        for row in leg_summary.values())
    require(all(
        summary["episodes_with_certificate_requests"]
        == summary["independent_uncached_localizations"]
        == summary["episodes_exactly_one_uncached_localization"]
        == summary["episodes_uncached_then_cached_only"]
        for summary in leg_summary.values()),
        "certificate cache lifecycle invariant failed")

    limitations = {
        "open_set_selector_evaluable": False,
        "certificate_rejection_evaluable": accepted < independent,
        "reason": (
            "The frozen benchmark contains constructed positive Revisit goals "
            "with explicit online-history co-visibility. It measures repeated "
            "memory execution/compositionality, not Novel-vs-Revisit rejection."
        ),
        "external_validity": (
            "Effect estimates apply to the constructible high-support subset; "
            "only the selection rule, not navigation outcomes, chose episodes."
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "manifest_sha256": sha256(manifest_path),
        "episodes": len(records),
        "scene_clusters": len({row["scene"] for row in records}),
        "construction": construction,
        "certificate_lifecycle": {
            "planning_requests": requests,
            "independent_uncached_localizations": independent,
            "cached_reuses": requests - independent,
            "accepted_independent_localizations": accepted,
            "rejected_independent_localizations": independent - accepted,
            "legs": leg_summary,
        },
        "multigoal_topology_audit": multigoal_topology_audit(records),
        "limitations": limitations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
