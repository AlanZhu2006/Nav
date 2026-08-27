#!/usr/bin/env python3
"""Audit Attempt-7 versus Phase-2 raw-bearing Novel outcomes.

This is a read-only post-hoc attribution audit.  It does not alter a policy,
rerun an episode, or use a Novel/Revisit role at runtime.  The script reads
the already completed natural-direction outputs and asks whether raw-fixed
Novel gains coincide with a direction that was actually aligned with the
query goal.

The query goal position is reconstructed from evaluator-logged Euclidean
goal distances at multiple known rollout poses.  The reconstruction is then
checked against every logged distance.  This avoids reading the sealed
benchmark sidecar while retaining an explicit numerical residual audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "raw_novel_cohort_shift_audit_v1_20260816"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def angle_deg(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denom) or denom <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def wrapped_error_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def circular_summary_deg(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean_angle_deg": None, "resultant": None}
    radians = np.radians(np.asarray(values, dtype=np.float64))
    cosine = float(np.mean(np.cos(radians)))
    sine = float(np.mean(np.sin(radians)))
    return {
        "n": len(values),
        "mean_angle_deg": float(math.degrees(math.atan2(sine, cosine))),
        "resultant": float(math.hypot(cosine, sine)),
    }


def exact_mcnemar_two_sided(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(int(gains), int(losses)) + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def fisher_exact_two_sided(table: tuple[tuple[int, int], tuple[int, int]]) -> float:
    (a, b), (c, d) = table
    row_one, row_two = a + b, c + d
    col_one, total = a + c, a + b + c + d

    def probability(value: int) -> float:
        return (math.comb(col_one, value)
                * math.comb(total - col_one, row_one - value)
                / math.comb(total, row_one))

    lower = max(0, row_one - (total - col_one))
    upper = min(row_one, col_one)
    observed = probability(a)
    return float(min(1.0, sum(
        probability(value) for value in range(lower, upper + 1)
        if probability(value) <= observed + 1e-15)))


def local_goal_vector(position_xz: np.ndarray, yaw: float,
                      goal_xz: np.ndarray) -> np.ndarray:
    """Return Habitat world displacement as local [forward, left]."""

    dx, dz = np.asarray(goal_xz) - np.asarray(position_xz)
    sine, cosine = math.sin(float(yaw)), math.cos(float(yaw))
    return np.asarray([
        -sine * dx - cosine * dz,
        -cosine * dx + sine * dz,
    ], dtype=np.float64)


def local_to_world(local: np.ndarray, yaw: float) -> np.ndarray:
    forward, left = np.asarray(local, dtype=np.float64)
    sine, cosine = math.sin(float(yaw)), math.cos(float(yaw))
    return np.asarray([
        -forward * sine - left * cosine,
        -forward * cosine + left * sine,
    ], dtype=np.float64)


def reconstruct_goal(plans: list[dict[str, Any]],
                     rollout: list[dict[str, Any]]) -> dict[str, Any]:
    poses = {int(row["step"]): np.asarray([row["x"], row["z"]],
                                           dtype=np.float64)
             for row in rollout}
    observations: list[tuple[int, np.ndarray, float]] = []
    for row in plans:
        step = int(row["step"])
        distance = row.get("evaluation_gt_goal_distance_m")
        if step in poses and distance is not None and math.isfinite(float(distance)):
            observations.append((step, poses[step], float(distance)))
    require(len(observations) >= 3, "too few goal-distance observations")

    _, p0, d0 = observations[0]
    matrix = []
    vector = []
    for _, point, distance in observations[1:]:
        matrix.append(2.0 * (point - p0))
        vector.append(
            float(np.dot(point, point) - np.dot(p0, p0)
                  - (distance * distance - d0 * d0))
        )
    matrix_array = np.asarray(matrix, dtype=np.float64)
    vector_array = np.asarray(vector, dtype=np.float64)
    goal, _residuals, rank, singular = np.linalg.lstsq(
        matrix_array, vector_array, rcond=None)
    require(int(rank) == 2, "goal reconstruction is rank deficient")
    errors = [
        abs(float(np.linalg.norm(goal - point)) - distance)
        for _, point, distance in observations
    ]
    return {
        "goal_xz": goal.tolist(),
        "observation_count": len(observations),
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "max_distance_residual_m": float(max(errors)),
        "rmse_distance_residual_m": float(
            math.sqrt(sum(error * error for error in errors) / len(errors))),
    }


def read_metric(path: Path, role: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["analysis_role"] == role]
    require(len(matches) == 1, f"{path}: expected one {role} row")
    return matches[0]


def first_after_step(plans: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    later = [row for row in plans if int(row["step"]) > int(step)]
    return min(later, key=lambda row: int(row["step"])) if later else None


def geodesic_direction(
    *,
    pathfinder: Any,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    yaw: float,
) -> dict[str, Any]:
    import habitat_sim

    request = habitat_sim.ShortestPath()
    request.requested_start = np.asarray(start_xyz, dtype=np.float32)
    request.requested_end = np.asarray(goal_xyz, dtype=np.float32)
    require(pathfinder.find_path(request), "PathFinder rejected reconstructed goal")
    points = [np.asarray(point, dtype=np.float64) for point in request.points]
    require(points, "PathFinder returned an empty path")
    start_xz = np.asarray(start_xyz, dtype=np.float64)[[0, 2]]
    selected = points[-1]
    for point in points[1:]:
        if float(np.linalg.norm(point[[0, 2]] - start_xz)) >= 0.30:
            selected = point
            break
    local = local_goal_vector(start_xz, yaw, selected[[0, 2]])
    return {
        "distance_m": float(request.geodesic_distance),
        "first_segment_local": local.tolist(),
        "first_segment_angle_deg": float(math.degrees(
            math.atan2(local[1], local[0]))),
        "path_point_count": len(points),
    }


def unit_record(cohort: str, unit: Path, *, mp3d_root: Path | None,
                pathfinders: dict[str, Any]) -> dict[str, Any]:
    contract = json.loads((unit / "episode_contract.json").read_text())
    scene = str(contract["scene"])
    episode = str(contract["episode"])
    raw_metric = read_metric(unit / "raw_fixed_bearing" / "metric.csv", "novel")
    raw_direct_metric = read_metric(unit / "raw_direct" / "metric.csv", "novel")
    native_metric = read_metric(unit / "native" / "metric.csv", "novel")
    certified_metric = read_metric(unit / "certified" / "metric.csv", "novel")
    plan_paths = sorted((unit / "raw_fixed_bearing").glob("*novel_plans.json"))
    require(len(plan_paths) == 1, f"{unit}: expected one raw Novel plan file")
    payload = json.loads(plan_paths[0].read_text())
    plans = payload["query_leg"]
    rollout = payload["rollout_traces"]["query"]
    goal_fit = reconstruct_goal(plans, rollout)
    goal_xz = np.asarray(goal_fit["goal_xz"], dtype=np.float64)
    poses = {int(row["step"]): row for row in rollout}
    takeovers = [row for row in plans if row.get("revisit_adapter_takeover") is True]
    require(takeovers, f"{unit}: raw fixed has no takeover")
    first = min(takeovers, key=lambda row: int(row["step"]))
    direct_paths = sorted((unit / "raw_direct").glob("*novel_plans.json"))
    require(len(direct_paths) == 1, f"{unit}: expected one raw-direct Novel plan file")
    direct_payload = json.loads(direct_paths[0].read_text())
    direct_takeovers = [
        row for row in direct_payload["query_leg"]
        if row.get("revisit_adapter_takeover") is True
    ]
    require(direct_takeovers, f"{unit}: raw direct has no takeover")
    first_raw_direct = min(direct_takeovers, key=lambda row: int(row["step"]))
    first_step = int(first["step"])
    pose = poses[first_step]
    position = np.asarray([pose["x"], pose["z"]], dtype=np.float64)
    yaw = float(pose["yaw"])
    bearing = np.asarray(first["memory_bearing_unit"], dtype=np.float64)
    direct_local = local_goal_vector(position, yaw, goal_xz)
    direct_error = angle_deg(bearing, direct_local)
    bearing_angle_deg = float(math.degrees(math.atan2(bearing[1], bearing[0])))
    direct_angle_deg = float(math.degrees(
        math.atan2(direct_local[1], direct_local[0])))
    predicted_world = local_to_world(bearing, yaw)
    direct_world = goal_xz - position
    world_error = angle_deg(predicted_world, direct_world)
    require(
        direct_error is not None and world_error is not None
        and abs(direct_error - world_error) <= 1e-7,
        f"{unit}: local/world angle conversion mismatch",
    )
    geodesic_info = None
    geodesic_error = None
    if mp3d_root is not None:
        if scene not in pathfinders:
            import habitat_sim

            navmesh = mp3d_root / scene / f"{scene}.navmesh"
            require(navmesh.is_file(), f"missing MP3D navmesh: {navmesh}")
            pathfinder = habitat_sim.PathFinder()
            require(pathfinder.load_nav_mesh(str(navmesh)),
                    f"failed to load navmesh: {navmesh}")
            pathfinders[scene] = pathfinder
        start_xyz = np.asarray([pose["x"], pose["y"], pose["z"]],
                               dtype=np.float64)
        goal_xyz = np.asarray(pathfinders[scene].snap_point(
            [goal_xz[0], pose["y"], goal_xz[1]]), dtype=np.float64)
        require(np.isfinite(goal_xyz).all(),
                f"{unit}: reconstructed goal did not snap to navmesh")
        geodesic_info = geodesic_direction(
            pathfinder=pathfinders[scene], start_xyz=start_xyz,
            goal_xyz=goal_xyz, yaw=yaw)
        geodesic_error = angle_deg(
            bearing, np.asarray(geodesic_info["first_segment_local"],
                                dtype=np.float64))
        geodesic_info["stored_distance_abs_error_m"] = abs(
            float(raw_metric["geodesic_m"]) - geodesic_info["distance_m"])

    next_plan = first_after_step(plans, first_step)
    first_distance = float(first["evaluation_gt_goal_distance_m"])
    next_distance = (
        None if next_plan is None
        else float(next_plan["evaluation_gt_goal_distance_m"])
    )
    post_position = None
    post_displacement = None
    executed_alignment = None
    if next_plan is not None:
        post_step = int(next_plan["step"])
        post_pose = poses[post_step]
        post_position = np.asarray([post_pose["x"], post_pose["z"]],
                                   dtype=np.float64)
        post_displacement = post_position - position
        executed_alignment = angle_deg(post_displacement, direct_world)

    all_errors = []
    for row in takeovers:
        step = int(row["step"])
        if step not in poses or row.get("memory_bearing_unit") is None:
            continue
        step_pose = poses[step]
        step_position = np.asarray([step_pose["x"], step_pose["z"]],
                                   dtype=np.float64)
        target_local = local_goal_vector(step_position, float(step_pose["yaw"]),
                                         goal_xz)
        error = angle_deg(np.asarray(row["memory_bearing_unit"], dtype=np.float64),
                          target_local)
        if error is not None:
            all_errors.append(float(error))

    raw_success = bool(int(raw_metric["reached"]))
    raw_direct_success = bool(int(raw_direct_metric["reached"]))
    native_success = bool(int(native_metric["reached"]))
    paired_class = (
        "gain" if raw_success and not native_success else
        "loss" if native_success and not raw_success else
        "both_success" if native_success and raw_success else
        "both_failure"
    )
    return {
        "cohort": cohort,
        "unit": unit.name,
        "scene": scene,
        "episode": episode,
        "arm_order": contract["arm_order"],
        "flow_threshold": json.loads(
            (unit / "raw_fixed_bearing" / "summary.json").read_text()
        )["memnav_server_info"]["flow_threshold"],
        "native_success": native_success,
        "raw_fixed_success": raw_success,
        "raw_direct_success": raw_direct_success,
        "certified_success": bool(int(certified_metric["reached"])),
        "paired_class": paired_class,
        "initial_geodesic_m": float(raw_metric["geodesic_m"]),
        "raw_steps": int(raw_metric["steps"]),
        "raw_path_len_m": float(raw_metric["path_len_m"]),
        "raw_final_goal_dist_m": float(raw_metric["final_goal_dist_m"]),
        "takeover_plan_count": len(takeovers),
        "plan_count": len(plans),
        "first_takeover_step": first_step,
        "first_goal_distance_m": first_distance,
        "next_decision_step": (
            None if next_plan is None else int(next_plan["step"])),
        "next_goal_distance_m": next_distance,
        "first_horizon_progress_m": (
            None if next_distance is None else first_distance - next_distance),
        "first_executed_displacement_m": (
            None if post_displacement is None
            else float(np.linalg.norm(post_displacement))),
        "first_executed_direct_alignment_deg": executed_alignment,
        "first_raw_score": first.get("raw_score"),
        "first_retrieval_margin": first.get("retrieval_margin"),
        "first_anchor": first.get("anchor"),
        "first_anchor_gap": first.get("anchor_gap"),
        "first_bearing_unit": bearing.tolist(),
        "first_bearing_angle_deg": bearing_angle_deg,
        "first_direct_goal_angle_deg": direct_angle_deg,
        "first_goal_rel_yaw_deg": (
            None if first.get("goal_rel_yaw") is None
            else float(math.degrees(float(first["goal_rel_yaw"])))),
        "raw_direct_first_proposal_match": {
            "same_step": int(first_raw_direct["step"]) == first_step,
            "same_anchor": first_raw_direct.get("anchor") == first.get("anchor"),
            "raw_score_abs_difference": abs(
                float(first_raw_direct["raw_score"]) - float(first["raw_score"])),
            "bearing_error_deg": angle_deg(
                np.asarray(first_raw_direct["memory_bearing_unit"],
                           dtype=np.float64), bearing),
        },
        "first_direct_goal_bearing_error_deg": direct_error,
        "first_geodesic_goal_bearing_error_deg": geodesic_error,
        "initial_geodesic_reconstruction": geodesic_info,
        "all_takeover_bearing_error_median_deg": float(np.median(all_errors)),
        "all_takeover_bearing_error_mean_deg": float(np.mean(all_errors)),
        "all_takeover_bearing_error_le_30_fraction": float(
            np.mean(np.asarray(all_errors) <= 30.0)),
        "goal_reconstruction": goal_fit,
    }


def scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))],
                        dtype=np.float64)
    if not len(finite):
        return {"n": 0, "mean": None, "median": None, "min": None,
                "max": None}
    return {
        "n": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def cohort_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    classes = {name: sum(row["paired_class"] == name for row in records)
               for name in ("gain", "loss", "both_success", "both_failure")}
    by_class = {}
    for name in classes:
        subset = [row for row in records if row["paired_class"] == name]
        by_class[name] = {
            "n": len(subset),
            "first_direct_goal_bearing_error_deg": scalar_summary([
                row["first_direct_goal_bearing_error_deg"] for row in subset]),
            "first_geodesic_goal_bearing_error_deg": scalar_summary([
                row["first_geodesic_goal_bearing_error_deg"] for row in subset
                if row["first_geodesic_goal_bearing_error_deg"] is not None]),
            "first_horizon_progress_m": scalar_summary([
                row["first_horizon_progress_m"] for row in subset
                if row["first_horizon_progress_m"] is not None]),
            "raw_score": scalar_summary([
                float(row["first_raw_score"]) for row in subset
                if row["first_raw_score"] is not None]),
            "retrieval_margin": scalar_summary([
                float(row["first_retrieval_margin"]) for row in subset
                if row["first_retrieval_margin"] is not None]),
        }
    aligned = [row for row in records
               if row["first_geodesic_goal_bearing_error_deg"] is not None
               and row["first_geodesic_goal_bearing_error_deg"] <= 30.0]
    unaligned = [row for row in records
                 if row["first_geodesic_goal_bearing_error_deg"] is not None
                 and row["first_geodesic_goal_bearing_error_deg"] > 30.0]
    alignment_table = (
        (sum(row["raw_fixed_success"] for row in aligned),
         sum(not row["raw_fixed_success"] for row in aligned)),
        (sum(row["raw_fixed_success"] for row in unaligned),
         sum(not row["raw_fixed_success"] for row in unaligned)),
    )
    direct_fixed_gains = sum(
        row["raw_fixed_success"] and not row["raw_direct_success"]
        for row in records)
    direct_fixed_losses = sum(
        row["raw_direct_success"] and not row["raw_fixed_success"]
        for row in records)
    source_angles = [row["first_bearing_angle_deg"] for row in records]
    target_angles = [
        row["initial_geodesic_reconstruction"]["first_segment_angle_deg"]
        for row in records
        if row["initial_geodesic_reconstruction"] is not None
    ]
    return {
        "n": len(records),
        "scenes": len({row["scene"] for row in records}),
        "native_successes": sum(row["native_success"] for row in records),
        "raw_fixed_successes": sum(row["raw_fixed_success"] for row in records),
        "raw_direct_successes": sum(row["raw_direct_success"] for row in records),
        "certified_successes": sum(row["certified_success"] for row in records),
        "paired": classes,
        "raw_fixed_minus_native_mcnemar_p": exact_mcnemar_two_sided(
            classes["gain"], classes["loss"]),
        "raw_fixed_minus_raw_direct": {
            "gains": direct_fixed_gains,
            "losses": direct_fixed_losses,
            "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(
                direct_fixed_gains, direct_fixed_losses),
        },
        "raw_direct_fixed_first_proposal": {
            "all_exact": all(
                row["raw_direct_first_proposal_match"]["same_step"]
                and row["raw_direct_first_proposal_match"]["same_anchor"]
                and row["raw_direct_first_proposal_match"][
                    "raw_score_abs_difference"] == 0.0
                and row["raw_direct_first_proposal_match"][
                    "bearing_error_deg"] == 0.0
                for row in records),
            "maximum_bearing_error_deg": max(
                row["raw_direct_first_proposal_match"]["bearing_error_deg"]
                for row in records),
        },
        "raw_fixed_arm_order_position_counts": {
            str(position): sum(
                row["arm_order"].index("raw_fixed_bearing") == position
                for row in records)
            for position in range(5)
        },
        "flow_threshold_counts": {
            str(value): sum(float(row["flow_threshold"]) == value for row in records)
            for value in sorted({float(row["flow_threshold"]) for row in records})
        },
        "first_direct_goal_bearing_error_deg": scalar_summary([
            row["first_direct_goal_bearing_error_deg"] for row in records]),
        "first_geodesic_goal_bearing_error_deg": scalar_summary([
            row["first_geodesic_goal_bearing_error_deg"] for row in records
            if row["first_geodesic_goal_bearing_error_deg"] is not None]),
        "first_bearing_circular": circular_summary_deg(source_angles),
        "required_geodesic_bearing_circular": circular_summary_deg(target_angles),
        "required_geodesic_bearing_behind_count": sum(
            abs(value) > 90.0 for value in target_angles),
        "alignment_le_30_deg": {
            "aligned_n": len(aligned),
            "unaligned_n": len(unaligned),
            "aligned_raw_successes": alignment_table[0][0],
            "unaligned_raw_successes": alignment_table[1][0],
            "success_contingency": alignment_table,
            "fisher_exact_two_sided_p": fisher_exact_two_sided(alignment_table),
            "status": "post_hoc_mechanism_diagnostic_not_confirmation",
        },
        "first_horizon_progress_m": scalar_summary([
            row["first_horizon_progress_m"] for row in records
            if row["first_horizon_progress_m"] is not None]),
        "goal_reconstruction_max_residual_m": scalar_summary([
            row["goal_reconstruction"]["max_distance_residual_m"]
            for row in records]),
        "local_navmesh_geodesic_distance_abs_error_m": scalar_summary([
            row["initial_geodesic_reconstruction"][
                "stored_distance_abs_error_m"] for row in records
            if row["initial_geodesic_reconstruction"] is not None]),
        "by_paired_class": by_class,
    }


def derangement_summary(records: list[dict[str, Any]], *, seed: int,
                        resamples: int = 100000) -> dict[str, Any]:
    bearings = np.asarray(
        [row["first_bearing_angle_deg"] for row in records], dtype=np.float64)
    targets = np.asarray([
        row["initial_geodesic_reconstruction"]["first_segment_angle_deg"]
        for row in records
    ], dtype=np.float64)
    factual = np.asarray([
        wrapped_error_deg(bearing, target)
        for bearing, target in zip(bearings, targets)
    ], dtype=np.float64)
    circular = circular_summary_deg(bearings.tolist())
    constant = float(circular["mean_angle_deg"])
    constant_errors = np.asarray(
        [wrapped_error_deg(constant, target) for target in targets],
        dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    identities = np.arange(len(records))
    means = []
    hits = []
    for _ in range(int(resamples)):
        while True:
            permutation = rng.permutation(len(records))
            if np.all(permutation != identities):
                break
        errors = np.asarray([
            wrapped_error_deg(bearings[source], targets[target])
            for target, source in enumerate(permutation)
        ])
        means.append(float(np.mean(errors)))
        hits.append(int(np.sum(errors <= 30.0)))
    mean_array = np.asarray(means, dtype=np.float64)
    hit_array = np.asarray(hits, dtype=np.int64)
    return {
        "scope": (
            "first-step local bearings swapped across completed query identities; "
            "not a closed-loop result and does not isolate goal-image from history input"),
        "seed": int(seed),
        "resamples": int(resamples),
        "factual": {
            "mean_error_deg": float(np.mean(factual)),
            "median_error_deg": float(np.median(factual)),
            "count_le_30_deg": int(np.sum(factual <= 30.0)),
        },
        "constant_circular_mean": {
            "bearing_deg": constant,
            "mean_error_deg": float(np.mean(constant_errors)),
            "median_error_deg": float(np.median(constant_errors)),
            "count_le_30_deg": int(np.sum(constant_errors <= 30.0)),
        },
        "deranged": {
            "mean_error_deg_quantiles_2p5_50_97p5": np.quantile(
                mean_array, [0.025, 0.5, 0.975]).tolist(),
            "count_le_30_deg_quantiles_2p5_50_97p5": np.quantile(
                hit_array, [0.025, 0.5, 0.975]).tolist(),
            "empirical_probability_mean_no_greater_than_factual": float(
                np.mean(mean_array <= float(np.mean(factual)))),
            "empirical_probability_hits_no_fewer_than_factual": float(
                np.mean(hit_array >= int(np.sum(factual <= 30.0)))),
        },
        "status": "post_hoc_direction_only_diagnostic_not_causal_confirmation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", nargs=2,
                        metavar=("NAME", "NATURAL_EVAL_ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mp3d-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    roots = {}
    pathfinders: dict[str, Any] = {}
    for name, root_text in args.cohort:
        root = Path(root_text).resolve()
        roots[name] = str(root)
        units = sorted(path for path in root.iterdir()
                       if path.is_dir() and (path / "episode_contract.json").is_file())
        require(units, f"{root}: no completed units")
        records.extend(unit_record(
            name, unit, mp3d_root=args.mp3d_root,
            pathfinders=pathfinders) for unit in units)
    report = {
        "schema_version": SCHEMA,
        "scope": "read-only completed natural-direction Novel raw-bearing attribution",
        "no_new_rollout": True,
        "goal_direction_reference": (
            "direct Euclidean goal bearing reconstructed from logged evaluator distances; "
            "not Habitat shortest-path first-segment bearing"),
        "roots": roots,
        "cohorts": {},
        "records": records,
    }
    for cohort_index, name in enumerate(roots):
        subset = [row for row in records if row["cohort"] == name]
        report["cohorts"][name] = cohort_summary(subset)
        report["cohorts"][name]["first_bearing_derangement"] = (
            derangement_summary(
                subset, seed=20260816 + cohort_index, resamples=100000))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["cohorts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
