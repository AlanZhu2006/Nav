#!/usr/bin/env python3
"""Fixed post-verification failure audit for fresh long-range route tangents.

This script is diagnostic only.  It may run only after the formal aggregate
summary and its independent verifier exist.  The metric panel and failure
partition are frozen in
``hm3d_longrange_route_tangent_failure_audit_freeze_20260903.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "hm3d_longrange_route_tangent_failure_audit_v1_20260903"
FREEZE_SCHEMA = (
    "hm3d_longrange_route_tangent_failure_audit_freeze_v1_20260903"
)
FORMAL_SUMMARY_SCHEMA = "hm3d_longrange_route_tangent_result_v1_20260903"
FORMAL_VERIFY_SCHEMA = (
    "hm3d_longrange_route_tangent_result_verification_v1_20260903"
)
ARMS = (
    "mono_native", "mono_cec_endpoint", "mono_cec_route_tangent",
)
LOW_CRITIC_THRESHOLD = -0.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing receipt for {path}")
    fields = sidecar.read_text().split()
    require(len(fields) >= 2 and fields[0] == digest,
            f"invalid receipt for {path}")
    return digest


def finite(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed):
            output.append(parsed)
    return output


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray(finite(values), dtype=np.float64)
    if not len(array):
        return {
            "n": 0, "min": None, "mean": None, "median": None,
            "p90": None, "max": None,
        }
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def plans_path(arm_root: Path) -> Path:
    candidates = list(arm_root.glob("*_plans.json"))
    require(len(candidates) == 1,
            f"expected one plan payload below {arm_root}")
    return candidates[0]


def motion_panel(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload["rollout_traces"]["query"]
    require(bool(trace), "query rollout trace is empty")
    xyz = np.asarray([
        [row["x"], row["y"], row["z"]] for row in trace
    ], dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3
            and np.isfinite(xyz).all(), "query rollout trace is malformed")
    end = np.asarray(payload["query_result"]["end_position"], dtype=np.float64)
    require(end.shape == (3,) and np.isfinite(end).all(),
            "query endpoint is malformed")
    transitions = np.linalg.norm(np.diff(xyz[:, [0, 2]], axis=0), axis=1)
    transitions_3d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    tail = float(np.linalg.norm(end[[0, 2]] - xyz[-1, [0, 2]]))
    tail_3d = float(np.linalg.norm(end - xyz[-1]))
    recomputed = float(np.sum(transitions) + tail)
    recomputed_3d = float(np.sum(transitions_3d) + tail_3d)
    reported = float(payload["query_result"]["path_len_m"])
    yaw_increments = finite(
        row.get("executed_yaw_rad_since_previous_frame", 0.0)
        for row in trace)
    return {
        "transition_count": int(len(transitions)),
        "stationary_transition_count": int(np.sum(transitions <= 1e-12)),
        "stationary_transition_fraction": (
            0.0 if not len(transitions)
            else float(np.mean(transitions <= 1e-12))
        ),
        "recomputed_planar_path_m": recomputed,
        "recomputed_3d_path_m": recomputed_3d,
        "reported_planar_path_m": reported,
        "reported_minus_recomputed_path_m": float(reported - recomputed),
        "total_absolute_yaw_rad": float(np.sum(np.abs(yaw_increments))),
        "total_absolute_yaw_deg": float(np.degrees(
            np.sum(np.abs(yaw_increments)))),
        "vertical_span_m": float(np.ptp(xyz[:, 1])),
    }


def revisit_goal_position(item: dict[str, Any]) -> np.ndarray:
    """Read the frozen Revisit target without consulting policy outcomes."""

    pairs = item.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 1,
            "manifest item has no unique role pair")
    queries = pairs[0].get("queries")
    require(isinstance(queries, list),
            "manifest role pair has no query list")
    revisits = [
        row for row in queries if row.get("analysis_role") == "revisit"
    ]
    require(len(revisits) == 1,
            "manifest role pair has no unique Revisit target")
    goal = np.asarray(revisits[0].get("floor_position"), dtype=np.float64)
    require(goal.shape == (3,) and np.isfinite(goal).all(),
            "manifest Revisit target is malformed")
    return goal


def actionwise_goal_panel(
    payload: dict[str, Any], goal_position: np.ndarray,
) -> dict[str, Any]:
    """Recompute the closest approach from every recorded physical pose.

    The formal evaluator checks success after every action.  Plan receipts are
    emitted only once per execution horizon, so their diagnostic goal-distance
    samples can miss a closer intermediate pose.  This read-only recomputation
    includes the reported terminal pose and does not change formal outcomes.
    """

    trace = payload["rollout_traces"]["query"]
    require(bool(trace), "query rollout trace is empty")
    xyz = np.asarray([
        [row["x"], row["y"], row["z"]] for row in trace
    ], dtype=np.float64)
    end = np.asarray(payload["query_result"]["end_position"], dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3
            and np.isfinite(xyz).all(), "query rollout trace is malformed")
    require(end.shape == (3,) and np.isfinite(end).all(),
            "query endpoint is malformed")
    samples = np.concatenate([xyz, end[None, :]], axis=0)
    deltas = samples - goal_position[None, :]
    distance_3d = np.linalg.norm(deltas, axis=1)
    distance_planar = np.linalg.norm(deltas[:, [0, 2]], axis=1)
    closest = int(np.argmin(distance_3d))
    return {
        "pose_sample_count": int(len(samples)),
        "minimum_3d_goal_distance_m": float(distance_3d[closest]),
        "minimum_planar_goal_distance_m": float(distance_planar[closest]),
        "vertical_error_at_minimum_3d_m": float(abs(deltas[closest, 1])),
        "minimum_pose_sample_index": closest,
        "terminal_3d_goal_distance_m": float(distance_3d[-1]),
    }


def failure_partition(
    *, outcome: int, accepted: bool, geometry_stop: bool,
    exact_native: bool, termination: str,
) -> str:
    if outcome:
        return "success"
    if not accepted and exact_native:
        return "initial_certificate_reject_exact_native"
    if geometry_stop:
        return "post_accept_geometry_stream_failure"
    if termination == "stuck":
        return "post_accept_stuck_termination"
    if termination in ("max_steps", "budget_exhausted"):
        return "post_accept_budget_exhaustion"
    return "post_accept_other_termination"


def analyze(
    *, population: Path, run_root: Path, formal_summary_path: Path,
    formal_verification_path: Path, freeze_path: Path,
) -> dict[str, Any]:
    freeze_sha = verify_sidecar(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    require(freeze.get("schema_version") == FREEZE_SCHEMA,
            "failure-audit freeze changed")
    require(str(run_root.resolve()) == freeze["formal_run_root"],
            "failure audit points to another formal run")

    summary_sha = verify_sidecar(formal_summary_path)
    verification_sha = verify_sidecar(formal_verification_path)
    summary = json.loads(formal_summary_path.read_text())
    verification = json.loads(formal_verification_path.read_text())
    require(summary.get("schema_version") == FORMAL_SUMMARY_SCHEMA,
            "formal summary schema changed")
    require(verification.get("schema_version") == FORMAL_VERIFY_SCHEMA
            and verification.get("verified") is True
            and verification.get("summary_sha256") == summary_sha,
            "formal result has not passed independent verification")

    manifest_path = population / "role_pairs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["episodes"]) == 23
            and verification.get("histories") == 23,
            "formal population size changed")
    require(sha256_file(manifest_path)
            == verification["benchmark_manifest_sha256"],
            "formal population digest changed")

    per_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    discordant = []
    scene_pairs: dict[str, dict[str, int]] = defaultdict(
        lambda: {arm: 0 for arm in ARMS})
    for index, item in enumerate(manifest["episodes"]):
        goal_position = revisit_goal_position(item)
        label = f"{index:03d}_{item['scene']}_{item['episode']}"
        episode_root = run_root / "evaluation" / label
        completion_path = episode_root / "completion.json"
        verify_sidecar(completion_path)
        completion = json.loads(completion_path.read_text())
        require(completion.get("prefix_equality") is True
                and completion.get("initial_cec_proof_equal_between_methods")
                is True, f"{index}: paired closure failed")

        outcomes = {arm: int(completion["outcomes"][arm]) for arm in ARMS}
        if outcomes["mono_cec_route_tangent"] != outcomes["mono_cec_endpoint"]:
            discordant.append({
                "history_index": index,
                "scene": item["scene"],
                "episode": item["episode"],
                "endpoint": outcomes["mono_cec_endpoint"],
                "route_tangent": outcomes["mono_cec_route_tangent"],
            })
        for arm in ARMS:
            scene_pairs[str(item["scene"])][arm] += outcomes[arm]
            payload = json.loads(plans_path(episode_root / arm).read_text())
            actionwise_goal = actionwise_goal_panel(payload, goal_position)
            plans = payload["query_leg"]
            accepted = any(
                row.get("certified_relocalization_accepted") is True
                for row in plans)
            accepted_plans = [
                row for row in plans
                if row.get("certified_relocalization_accepted") is True
            ]
            selected_anchors = {
                int(row["router_selected_anchor"])
                for row in accepted_plans
                if row.get("router_selected_anchor") is not None
            }
            require(len(selected_anchors) <= 1,
                    f"{index}/{arm}: accepted anchor changed within query")
            stop_rows = [
                row for row in plans if row.get("geometry_stream_stop") is True
            ]
            exact_native = bool(completion.get(
                "fully_rejected_exact_native", {}).get(arm, False))
            termination = str(completion["termination_reason"][arm])
            critics = finite(row.get("navdp_critic_max") for row in plans)
            total_decision_ms = finite(
                row.get("cec_total_decision_ms") for row in plans)
            relocalization_ms = finite(
                row.get("certified_relocalization_ms") for row in plans)
            controller_ms = finite(
                row.get("cec_controller_ms") for row in plans)
            depth_sidecar_ms = finite(
                row.get("cec_depth_sidecar_ms") for row in plans)
            planned_distances = finite(
                row.get("evaluation_gt_goal_distance_m") for row in plans)
            entry: dict[str, Any] = {
                "history_index": index,
                "scene": item["scene"],
                "episode": item["episode"],
                "outcome": outcomes[arm],
                "initial_geodesic_m": float(
                    completion["initial_geodesic_m"][arm]),
                "final_3d_distance_m": float(
                    completion["final_goal_3d_dist_m"][arm]),
                "minimum_planned_3d_goal_distance_m": (
                    None if not planned_distances else min(planned_distances)),
                "realized_path_m": float(completion["path_len_m"][arm]),
                "steps": int(completion["steps"][arm]),
                "termination_reason": termination,
                "blocked_step_count": int(
                    payload["query_result"].get("blocked_step_count", 0)),
                "certificate_accepted": accepted,
                "selected_anchor": (
                    None if not selected_anchors else next(iter(
                        selected_anchors))),
                "initial_proof_equal": bool(completion[
                    "initial_cec_proof_equal_between_methods"]),
                "fully_rejected_exact_native": exact_native,
                "geometry_stream_stop": bool(stop_rows),
                "geometry_stream_stop_reason": (
                    None if not stop_rows else stop_rows[-1].get(
                        "local_tangent_error")),
                "critic_values": critics,
                "total_decision_ms": total_decision_ms,
                "relocalization_ms": relocalization_ms,
                "initial_relocalization_ms": (
                    None if not relocalization_ms else relocalization_ms[0]),
                "subsequent_relocalization_ms": relocalization_ms[1:],
                "controller_ms": controller_ms,
                "depth_sidecar_ms": depth_sidecar_ms,
                "arm_wall_time_seconds": float(
                    completion["wall_time_seconds"][arm]),
                "motion": motion_panel(payload),
                "actionwise_goal": actionwise_goal,
            }
            if arm == "mono_cec_route_tangent":
                active = [
                    row for row in plans
                    if row.get("certified_relocalization_accepted") is True
                    and row.get("local_tangent_status")
                    in ("active", "terminal_tangent")
                ]
                units = np.asarray([
                    row["local_tangent_unit_bearing"] for row in active
                ], dtype=np.float64)
                headings = (
                    [] if not len(active) else np.abs(np.degrees(np.arctan2(
                        units[:, 1], units[:, 0]))).tolist()
                )
                controller = np.asarray([
                    row["local_tangent_controller_pointgoal"] for row in active
                ], dtype=np.float64)
                if len(controller):
                    processed = np.clip(controller, -10.0, 10.0)
                    processed[:, 0] = np.clip(processed[:, 0], 0.0, 10.0)
                    processed_norms = np.linalg.norm(processed, axis=1).tolist()
                else:
                    processed_norms = []
                update_receipts = [
                    receipt for row in active
                    for receipt in row.get(
                        "local_tangent_update_edge_receipts", [])
                ]
                entry.update({
                    "requested_heading_abs_deg": headings,
                    "post_navdp_clip_pointgoal_norm_m": processed_norms,
                    "route_progress_fractions": finite(
                        row.get("local_tangent_progress_fraction")
                        for row in active),
                    "final_route_progress_fraction": (
                        None if not active else float(active[-1][
                            "local_tangent_progress_fraction"])),
                    "max_route_progress_fraction": (
                        None if not active else max(float(row[
                            "local_tangent_progress_fraction"])
                            for row in active)),
                    "cross_track_errors_m": finite(
                        row.get("local_tangent_cross_track_error_m")
                        for row in active),
                    "history_edge_count": (
                        None if not active else int(active[0][
                            "local_tangent_history_edge_count"])),
                    "history_route_extent_m": (
                        None if not active else float(active[0][
                            "local_tangent_route_extent_m"])),
                    "live_update_pnp_inliers": finite(
                        row.get("pnp_inliers") for row in update_receipts),
                    "live_update_pnp_rmse_px": finite(
                        row.get("pnp_reprojection_rmse_px")
                        for row in update_receipts),
                })
            entry["failure_partition"] = failure_partition(
                outcome=entry["outcome"], accepted=accepted,
                geometry_stop=bool(stop_rows), exact_native=exact_native,
                termination=termination,
            )
            per_arm[arm].append(entry)

    panels: dict[str, Any] = {}
    for arm in ARMS:
        rows = per_arm[arm]
        critics = [value for row in rows for value in row["critic_values"]]
        total_decision_ms = [
            value for row in rows for value in row["total_decision_ms"]
        ]
        relocalization_ms = [
            value for row in rows for value in row["relocalization_ms"]
        ]
        subsequent_relocalization_ms = [
            value for row in rows
            for value in row["subsequent_relocalization_ms"]
        ]
        controller_ms = [
            value for row in rows for value in row["controller_ms"]
        ]
        depth_sidecar_ms = [
            value for row in rows for value in row["depth_sidecar_ms"]
        ]
        motions = [row["motion"] for row in rows]
        actionwise_goals = [row["actionwise_goal"] for row in rows]
        transitions = sum(row["transition_count"] for row in motions)
        stationary = sum(row["stationary_transition_count"] for row in motions)
        panel: dict[str, Any] = {
            "episodes": len(rows),
            "successes": sum(row["outcome"] for row in rows),
            "success_rate": float(np.mean([row["outcome"] for row in rows])),
            "spl": float(np.mean([
                row["outcome"] * row["initial_geodesic_m"]
                / max(row["initial_geodesic_m"], row["realized_path_m"],
                      1e-12)
                for row in rows
            ])),
            "final_3d_distance_m": distribution(
                row["final_3d_distance_m"] for row in rows),
            "minimum_planned_3d_goal_distance_m": distribution(
                row["minimum_planned_3d_goal_distance_m"] for row in rows),
            "minimum_actionwise_3d_goal_distance_m": distribution(
                row["minimum_3d_goal_distance_m"]
                for row in actionwise_goals),
            "minimum_actionwise_planar_goal_distance_m": distribution(
                row["minimum_planar_goal_distance_m"]
                for row in actionwise_goals),
            "vertical_error_at_minimum_actionwise_3d_m": distribution(
                row["vertical_error_at_minimum_3d_m"]
                for row in actionwise_goals),
            "reported_final_minus_actionwise_terminal_3d_m": distribution(
                row["final_3d_distance_m"]
                - row["actionwise_goal"]["terminal_3d_goal_distance_m"]
                for row in rows),
            "realized_path_m": distribution(
                row["realized_path_m"] for row in rows),
            "steps": distribution(row["steps"] for row in rows),
            "termination_reason_counts": dict(Counter(
                row["termination_reason"] for row in rows)),
            "failure_partition_counts": dict(Counter(
                row["failure_partition"] for row in rows)),
            "certificate_accept_episode_count": sum(
                row["certificate_accepted"] for row in rows),
            "initial_certificate_reject_episode_count": sum(
                not row["certificate_accepted"] for row in rows),
            "selected_anchor": distribution(
                row["selected_anchor"] for row in rows),
            "selected_anchor_counts": dict(sorted(Counter(
                str(row["selected_anchor"]) for row in rows
                if row["selected_anchor"] is not None).items())),
            "initial_proof_equality_count": sum(
                row["initial_proof_equal"] for row in rows),
            "fully_rejected_exact_native_count": sum(
                row["fully_rejected_exact_native"] for row in rows),
            "geometry_stream_stop_episode_count": sum(
                row["geometry_stream_stop"] for row in rows),
            "geometry_stream_stop_reasons": dict(Counter(
                row["geometry_stream_stop_reason"] for row in rows
                if row["geometry_stream_stop_reason"] is not None)),
            "blocked_step_fraction": float(
                sum(row["blocked_step_count"] for row in rows)
                / max(1, sum(row["steps"] for row in rows))),
            "navdp_critic_max": distribution(critics),
            "navdp_critic_below_minus_0p5_fraction": (
                None if not critics
                else float(np.mean(np.asarray(critics) < LOW_CRITIC_THRESHOLD))),
            "runtime": {
                "arm_wall_time_seconds": distribution(
                    row["arm_wall_time_seconds"] for row in rows),
                "total_decision_ms": distribution(total_decision_ms),
                "relocalization_ms": distribution(relocalization_ms),
                "initial_relocalization_ms": distribution(
                    row["initial_relocalization_ms"] for row in rows),
                "subsequent_relocalization_ms": distribution(
                    subsequent_relocalization_ms),
                "controller_ms": distribution(controller_ms),
                "monocular_depth_sidecar_ms": distribution(
                    depth_sidecar_ms),
            },
            "stationary_transition_count": stationary,
            "stationary_transition_fraction": (
                0.0 if transitions == 0 else stationary / transitions),
            "reported_minus_recomputed_path_m": distribution(
                row["reported_minus_recomputed_path_m"] for row in motions),
            "recomputed_3d_path_m": distribution(
                row["recomputed_3d_path_m"] for row in motions),
            "total_absolute_yaw_deg": distribution(
                row["total_absolute_yaw_deg"] for row in motions),
            "vertical_span_m": distribution(
                row["vertical_span_m"] for row in motions),
        }
        if arm == "mono_cec_route_tangent":
            headings = [
                value for row in rows
                for value in row["requested_heading_abs_deg"]
            ]
            processed_norms = [
                value for row in rows
                for value in row["post_navdp_clip_pointgoal_norm_m"]
            ]
            progress = [
                value for row in rows for value in row["route_progress_fractions"]
            ]
            cross_track = [
                value for row in rows for value in row["cross_track_errors_m"]
            ]
            panel.update({
                "history_edge_count": distribution(
                    row["history_edge_count"] for row in rows),
                "history_route_extent_m": distribution(
                    row["history_route_extent_m"] for row in rows),
                "requested_heading_abs_deg": distribution(headings),
                "requested_heading_above_90deg_fraction": (
                    None if not headings
                    else float(np.mean(np.asarray(headings) > 90.0))),
                "requested_heading_at_least_165deg_fraction": (
                    None if not headings
                    else float(np.mean(np.asarray(headings) >= 165.0))),
                "post_navdp_clip_pointgoal_norm_m": distribution(
                    processed_norms),
                "route_progress_fraction": distribution(progress),
                "final_route_progress_fraction": distribution(
                    row["final_route_progress_fraction"] for row in rows),
                "max_route_progress_fraction": distribution(
                    row["max_route_progress_fraction"] for row in rows),
                "cross_track_error_m": distribution(cross_track),
                "live_update_pnp_inliers": distribution(
                    value for row in rows
                    for value in row["live_update_pnp_inliers"]),
                "live_update_pnp_rmse_px": distribution(
                    value for row in rows
                    for value in row["live_update_pnp_rmse_px"]),
            })
        panels[arm] = panel

    scene_totals = {
        scene: {**counts, "histories": sum(
            1 for row in per_arm["mono_native"] if row["scene"] == scene)}
        for scene, counts in sorted(scene_pairs.items())
    }
    require({arm: panels[arm]["successes"] for arm in ARMS}
            == summary["successes"],
            "diagnostic audit success totals differ from formal summary")
    return {
        "schema_version": SCHEMA,
        "claim_scope": "diagnostic decomposition; formal decision unchanged",
        "formal_result_independently_verified": True,
        "failure_audit_freeze_sha256": freeze_sha,
        "formal_summary_sha256": summary_sha,
        "formal_verification_sha256": verification_sha,
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "panels": panels,
        "scene_totals": scene_totals,
        "primary_discordant_pairs": discordant,
        "render_every_primary_discordant_pair": True,
        "per_episode": {arm: per_arm[arm] for arm in ARMS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--formal-verification", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    result = analyze(
        population=args.population.resolve(), run_root=args.run_root.resolve(),
        formal_summary_path=args.formal_summary.resolve(),
        formal_verification_path=args.formal_verification.resolve(),
        freeze_path=args.freeze.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n")
    print(json.dumps({
        "status": "complete",
        "successes": {arm: result["panels"][arm]["successes"]
                      for arm in ARMS},
        "discordant_pairs": len(result["primary_discordant_pairs"]),
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
