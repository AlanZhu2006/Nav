#!/usr/bin/env python3
"""Attribute raw-Novel direction to DINO anchor selection or path prior.

This audit is deliberately rollout-free and model-free.  Completed paired
records already contain the raw-DINO selected anchor, the causal online-A
pose trace, and the current query pose.  For every query, this script compares
the physical direction of the selected anchor against the directions of *all*
eligible historical anchors.  It therefore asks whether DINO selected a more
route-aligned part of the causal trajectory than a uniform anchor would have.

The audit is post-hoc mechanism development.  Ground-truth route direction is
used only for scoring completed proposals and never enters a policy request.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "raw_novel_anchor_attribution_v1_20260816"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angular_error_deg(left: float, right: float) -> float:
    return abs(wrap_degrees(float(left) - float(right)))


def local_angle_to_point_deg(
    current_xz: np.ndarray,
    current_yaw: float,
    target_xz: np.ndarray,
) -> float:
    """Habitat world displacement as local forward/left angle."""

    dx, dz = np.asarray(target_xz, dtype=np.float64) - np.asarray(
        current_xz, dtype=np.float64
    )
    sine, cosine = math.sin(float(current_yaw)), math.cos(float(current_yaw))
    forward = -sine * dx - cosine * dz
    left = -cosine * dx + sine * dz
    require(math.hypot(forward, left) > 1e-9, "anchor coincides with query pose")
    return float(math.degrees(math.atan2(left, forward)))


def eligible_anchor_indices(plan: dict[str, Any]) -> list[int]:
    """Recover the exact contiguous raw-retrieval candidate interval.

    The upper endpoint and candidate count are logged by the production
    runtime.  Inferring the lower endpoint from those two values avoids
    duplicating a hidden ``amargin`` constant in this audit.
    """

    frame_index = int(plan["frame_idx"])
    candidate_ceiling = int(plan["candidate_ceiling"])
    candidate_count = int(plan["candidate_count"])
    require(candidate_count > 0, "raw proposal has no eligible anchors")
    upper = min(frame_index - 32, candidate_ceiling)
    lower = upper - candidate_count + 1
    require(lower >= 0 and lower <= upper, "invalid inferred candidate interval")
    indices = list(range(lower, upper + 1))
    require(len(indices) == candidate_count, "candidate interval size mismatch")
    return indices


def first_takeover(plans: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in plans if row.get("revisit_adapter_takeover") is True]
    require(bool(matches), "raw-fixed query has no takeover")
    return min(matches, key=lambda row: int(row["step"]))


def load_record_index(report_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    require(
        report.get("schema_version") == "raw_novel_cohort_shift_audit_v1_20260816",
        "unexpected cohort-shift report schema",
    )
    records = {
        (str(row["cohort"]), str(row["unit"])): row
        for row in report["records"]
    }
    require(len(records) == len(report["records"]), "duplicate report identity")
    return records


def unit_record(
    cohort: str,
    unit: Path,
    reference: dict[str, Any],
    *,
    target_reference: str,
) -> dict[str, Any]:
    plan_paths = sorted((unit / "raw_fixed_bearing").glob("*novel_plans.json"))
    require(len(plan_paths) == 1, f"{unit}: expected one raw-fixed Novel ledger")
    payload = json.loads(plan_paths[0].read_text())
    plan = first_takeover(payload["query_leg"])
    require(int(plan["step"]) == int(reference["first_takeover_step"]),
            f"{unit}: first takeover step changed")
    require(int(plan["anchor"]) == int(reference["first_anchor"]),
            f"{unit}: selected anchor changed")

    query_poses = {int(row["step"]): row for row in payload["rollout_traces"]["query"]}
    current = query_poses[int(plan["step"])]
    current_xz = np.asarray([current["x"], current["z"]], dtype=np.float64)
    current_yaw = float(current["yaw"])
    history = {
        int(row["frame_idx"]): np.asarray([row["x"], row["z"]], dtype=np.float64)
        for row in payload["memory_traces"]["legA"]
    }
    eligible = eligible_anchor_indices(plan)
    missing = sorted(set(eligible) - set(history))
    require(not missing, f"{unit}: eligible anchors absent from online-A trace: {missing}")
    selected = int(plan["anchor"])
    require(selected in eligible, f"{unit}: DINO anchor is outside eligible interval")

    if target_reference == "shortest_path":
        route_info = reference.get("initial_geodesic_reconstruction")
        require(isinstance(route_info, dict), f"{unit}: missing route reconstruction")
        target_angle = float(route_info["first_segment_angle_deg"])
        raw_error = float(reference["first_geodesic_goal_bearing_error_deg"])
    elif target_reference == "direct_goal":
        target_angle = float(reference["first_direct_goal_angle_deg"])
        raw_error = float(reference["first_direct_goal_bearing_error_deg"])
    else:
        raise ValueError(f"unsupported target reference {target_reference!r}")
    anchor_angles = np.asarray(
        [local_angle_to_point_deg(current_xz, current_yaw, history[index])
         for index in eligible],
        dtype=np.float64,
    )
    anchor_errors = np.asarray(
        [angular_error_deg(angle, target_angle) for angle in anchor_angles],
        dtype=np.float64,
    )
    selected_offset = eligible.index(selected)
    selected_angle = float(anchor_angles[selected_offset])
    selected_error = float(anchor_errors[selected_offset])
    raw_angle = float(reference["first_bearing_angle_deg"])
    require(
        abs(raw_error - angular_error_deg(raw_angle, target_angle)) <= 1e-7,
        f"{unit}: stored raw/route angular error changed",
    )
    return {
        "cohort": cohort,
        "unit": unit.name,
        "scene": str(reference["scene"]),
        "episode": str(reference["episode"]),
        "paired_class": str(reference["paired_class"]),
        "first_takeover_step": int(plan["step"]),
        "frame_idx": int(plan["frame_idx"]),
        "candidate_ceiling": int(plan["candidate_ceiling"]),
        "eligible_anchor_count": len(eligible),
        "eligible_anchor_lower": eligible[0],
        "eligible_anchor_upper": eligible[-1],
        "dino_anchor": selected,
        "target_reference": target_reference,
        "target_angle_deg": target_angle,
        "raw_bearing_angle_deg": raw_angle,
        "raw_bearing_error_deg": raw_error,
        "dino_anchor_physical_angle_deg": selected_angle,
        "dino_anchor_physical_error_deg": selected_error,
        "dino_anchor_to_raw_bearing_error_deg": angular_error_deg(
            selected_angle, raw_angle),
        "constant_uturn_error_deg": angular_error_deg(180.0, target_angle),
        "uniform_anchor_error_mean_deg": float(np.mean(anchor_errors)),
        "uniform_anchor_error_median_deg": float(np.median(anchor_errors)),
        "uniform_anchor_probability_le_30_deg": float(np.mean(anchor_errors <= 30.0)),
        "dino_anchor_error_percentile_among_eligible": float(
            np.mean(anchor_errors <= selected_error)),
        "eligible_anchor_errors_deg": anchor_errors.tolist(),
    }


def scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def cluster_bootstrap_advantage(
    records: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    scenes = sorted({str(row["scene"]) for row in records})
    grouped = {scene: [row for row in records if row["scene"] == scene]
               for scene in scenes}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        rows = [row for scene in sampled for row in grouped[str(scene)]]
        values[index] = np.mean([
            row["uniform_anchor_error_mean_deg"]
            - row["dino_anchor_physical_error_deg"]
            for row in rows
        ])
    return {
        "scene_clusters": len(scenes),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_advantage_deg": float(np.mean([
            row["uniform_anchor_error_mean_deg"]
            - row["dino_anchor_physical_error_deg"]
            for row in records
        ])),
        "ci_95_deg": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def cluster_bootstrap_raw_shift(
    records: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Positive values mean LingBot's raw bearing beats anchor direction."""

    scenes = sorted({str(row["scene"]) for row in records})
    grouped = {scene: [row for row in records if row["scene"] == scene]
               for scene in scenes}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        rows = [row for scene in sampled for row in grouped[str(scene)]]
        values[index] = np.mean([
            row["dino_anchor_physical_error_deg"] - row["raw_bearing_error_deg"]
            for row in rows
        ])
    return {
        "scene_clusters": len(scenes),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_advantage_deg": float(np.mean([
            row["dino_anchor_physical_error_deg"] - row["raw_bearing_error_deg"]
            for row in records
        ])),
        "ci_95_deg": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def random_anchor_null(
    records: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    factual_errors = np.asarray(
        [row["dino_anchor_physical_error_deg"] for row in records],
        dtype=np.float64,
    )
    random_means = np.empty(int(resamples), dtype=np.float64)
    random_hits = np.empty(int(resamples), dtype=np.int64)
    error_arrays = [np.asarray(row["eligible_anchor_errors_deg"], dtype=np.float64)
                    for row in records]
    for index in range(int(resamples)):
        sampled = np.asarray([
            errors[int(rng.integers(0, len(errors)))] for errors in error_arrays
        ], dtype=np.float64)
        random_means[index] = float(np.mean(sampled))
        random_hits[index] = int(np.sum(sampled <= 30.0))
    factual_mean = float(np.mean(factual_errors))
    factual_hits = int(np.sum(factual_errors <= 30.0))
    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "factual_mean_error_deg": factual_mean,
        "factual_count_le_30_deg": factual_hits,
        "uniform_random_mean_error_quantiles_2p5_50_97p5_deg": np.quantile(
            random_means, [0.025, 0.5, 0.975]).tolist(),
        "uniform_random_count_le_30_quantiles_2p5_50_97p5": np.quantile(
            random_hits, [0.025, 0.5, 0.975]).tolist(),
        "empirical_probability_random_mean_no_greater_than_factual": float(
            np.mean(random_means <= factual_mean)),
        "empirical_probability_random_hits_no_fewer_than_factual": float(
            np.mean(random_hits >= factual_hits)),
    }


def cohort_summary(
    records: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    dino_errors = [row["dino_anchor_physical_error_deg"] for row in records]
    raw_errors = [row["raw_bearing_error_deg"] for row in records]
    uniform_means = [row["uniform_anchor_error_mean_deg"] for row in records]
    uturn_errors = [row["constant_uturn_error_deg"] for row in records]
    anchor_raw = [row["dino_anchor_to_raw_bearing_error_deg"] for row in records]
    per_query_advantages = [
        uniform - selected for uniform, selected in zip(uniform_means, dino_errors)
    ]
    raw_shift_advantages = [
        selected - raw for selected, raw in zip(dino_errors, raw_errors)
    ]
    return {
        "n": len(records),
        "scenes": len({row["scene"] for row in records}),
        "dino_anchor_physical_error_deg": scalar_summary(dino_errors),
        "raw_bearing_error_deg": scalar_summary(raw_errors),
        "constant_uturn_error_deg": scalar_summary(uturn_errors),
        "uniform_anchor_expected_error_deg": scalar_summary(uniform_means),
        "dino_advantage_over_uniform_anchor_deg": scalar_summary(
            per_query_advantages),
        "scenes_with_positive_mean_dino_advantage": int(sum(
            np.mean([
                row["uniform_anchor_error_mean_deg"]
                - row["dino_anchor_physical_error_deg"]
                for row in records if row["scene"] == scene
            ]) > 0.0
            for scene in {row["scene"] for row in records}
        )),
        "dino_anchor_count_le_30_deg": int(sum(
            error <= 30.0 for error in dino_errors)),
        "uniform_anchor_expected_count_le_30_deg": float(np.sum([
            row["uniform_anchor_probability_le_30_deg"] for row in records
        ])),
        "constant_uturn_count_le_30_deg": int(sum(
            error <= 30.0 for error in uturn_errors)),
        "raw_bearing_count_le_30_deg": int(sum(
            error <= 30.0 for error in raw_errors)),
        "dino_anchor_to_raw_bearing_error_deg": scalar_summary(anchor_raw),
        "raw_advantage_over_dino_anchor_direction_deg": scalar_summary(
            raw_shift_advantages),
        "raw_shift_cluster_bootstrap": cluster_bootstrap_raw_shift(
            records, seed=seed + 2000, resamples=resamples),
        "cluster_bootstrap": cluster_bootstrap_advantage(
            records, seed=seed + 1000, resamples=resamples),
        "uniform_anchor_null": random_anchor_null(
            records, seed=seed, resamples=resamples),
        "status": "post_hoc_rollout_free_mechanism_audit_not_confirmation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", nargs=2,
                        metavar=("NAME", "NATURAL_EVAL_MIRROR"), required=True)
    parser.add_argument("--direction-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--target-reference",
        choices=("shortest_path", "direct_goal"),
        default="shortest_path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.resamples > 0, "resamples must be positive")
    references = load_record_index(args.direction_report)
    records = []
    roots = {}
    for name, root_text in args.cohort:
        root = Path(root_text).resolve()
        roots[name] = str(root)
        units = sorted(
            path for path in root.iterdir()
            if path.is_dir() and (path / "episode_contract.json").is_file()
        )
        require(bool(units), f"{root}: no completed units")
        for unit in units:
            key = (name, unit.name)
            require(key in references, f"{unit}: missing direction reference")
            records.append(unit_record(
                name,
                unit,
                references[key],
                target_reference=args.target_reference,
            ))

    report = {
        "schema_version": SCHEMA,
        "scope": (
            "completed raw-fixed Novel first proposal; exact physical direction "
            "of DINO-selected anchor versus all eligible online-A anchors"
        ),
        "no_new_model_forward": True,
        "no_new_rollout": True,
        "target_reference": args.target_reference,
        "ground_truth_policy_access": "scoring_only_after_completed_outcome",
        "roots": roots,
        "direction_report": str(args.direction_report.resolve()),
        "cohorts": {},
        "records": records,
    }
    for cohort_index, name in enumerate(roots):
        subset = [row for row in records if row["cohort"] == name]
        report["cohorts"][name] = cohort_summary(
            subset,
            seed=int(args.seed) + cohort_index,
            resamples=int(args.resamples),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["cohorts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
