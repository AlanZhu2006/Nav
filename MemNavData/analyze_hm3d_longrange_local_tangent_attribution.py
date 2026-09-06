#!/usr/bin/env python3
"""Read the consumed v2 local-tangent result only after all arms complete.

This analyzer deliberately reports a fixed diagnostic panel rather than a
post-hoc success criterion.  The underlying population contains one already
consumed failure, so its output can locate a mechanism but cannot estimate an
effect size or support a paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from MemNavData.longrange_local_tangent_attribution import TANGENT_ARMS
from MemNavData.run_hm3d_longrange_local_tangent_attribution import (
    SCHEMA_VERSION as EPISODE_SCHEMA,
)
from MemNavData.run_hm3d_longrange_oracle_attribution import (
    load_single_result,
)


SCHEMA_VERSION = "hm3d_longrange_local_tangent_analysis_v1_20260903"
LOW_CRITIC_THRESHOLD = -0.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sha_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing SHA sidecar: {sidecar}")
    require(sidecar.read_text(encoding="utf-8").split()
            == [digest, path.name], f"invalid SHA sidecar: {sidecar}")
    return digest


def finite_values(values: Iterable[Any]) -> np.ndarray:
    output: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed):
            output.append(parsed)
    return np.asarray(output, dtype=np.float64)


def fixed_plan_panel(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only quantities declared before the outcome was opened."""

    receipts = [
        plan for plan in plans
        if plan.get("local_tangent_oracle_arm") is not None
    ]
    require(receipts, "local-tangent plan receipts are empty")
    critic = finite_values(
        plan.get("navdp_critic_max", plan.get("critic_max"))
        for plan in receipts)
    geodesic = finite_values(
        plan.get("local_tangent_oracle_current_geodesic_m")
        for plan in receipts)
    headings = finite_values(
        abs(float(plan["local_tangent_oracle_signed_heading_deg"]))
        for plan in receipts)
    require(len(geodesic) == len(receipts),
            "a local-tangent receipt omitted current geodesic distance")
    require(len(headings) == len(receipts),
            "a local-tangent receipt omitted heading")
    controllers = [
        str(plan["local_tangent_oracle_controller"]) for plan in receipts
    ]
    requested = np.asarray([
        plan["local_tangent_oracle_pointgoal"] for plan in receipts
    ], dtype=np.float64)
    require(requested.shape == (len(receipts), 2)
            and np.isfinite(requested).all(),
            "a local-tangent receipt omitted its PointGoal")
    # Frozen NavDP's process_pointgoal clips negative forward components to
    # zero.  Report how often an oracle direction enters that representation
    # boundary; do not silently interpret it as a route-estimation failure.
    processed = requested.copy()
    processed[:, 0] = np.clip(processed[:, 0], 0.0, 10.0)
    processed = np.clip(processed, -10.0, 10.0)
    processed_norm = np.linalg.norm(processed, axis=1)
    return {
        "plan_count": len(receipts),
        "initial_geodesic_m": float(geodesic[0]),
        "minimum_planned_geodesic_m": float(np.min(geodesic)),
        "last_planned_geodesic_m": float(geodesic[-1]),
        "maximum_geodesic_reduction_m": float(
            geodesic[0] - np.min(geodesic)),
        "critic_finite_count": int(len(critic)),
        "critic_mean": None if not len(critic) else float(np.mean(critic)),
        "critic_median": None if not len(critic) else float(np.median(critic)),
        "critic_below_minus_0p5_count": int(np.sum(
            critic < LOW_CRITIC_THRESHOLD)),
        "critic_below_minus_0p5_fraction": (
            None if not len(critic)
            else float(np.mean(critic < LOW_CRITIC_THRESHOLD))),
        "heading_abs_median_deg": float(np.median(headings)),
        "heading_abs_p90_deg": float(np.percentile(headings, 90)),
        "behind_pointgoal_count": int(np.sum(headings > 90.0)),
        "behind_pointgoal_fraction": float(np.mean(headings > 90.0)),
        "rear_dead_zone_165deg_count": int(np.sum(headings >= 165.0)),
        "rear_dead_zone_165deg_fraction": float(np.mean(headings >= 165.0)),
        "post_navdp_clip_pointgoal_norm_min_m": float(
            np.min(processed_norm)),
        "post_navdp_clip_pointgoal_norm_median_m": float(
            np.median(processed_norm)),
        "mixed_plan_count": controllers.count("mixed_image_pointgoal"),
        "native_plan_count": controllers.count("native_imagegoal"),
    }


def fixed_motion_panel(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload["rollout_traces"]["query"]
    require(trace, "query rollout trace is empty")
    xyz = np.asarray([
        [frame["x"], frame["y"], frame["z"]] for frame in trace
    ], dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3
            and np.isfinite(xyz).all(), "query trace contains invalid poses")
    planar_steps = np.linalg.norm(np.diff(xyz[:, [0, 2]], axis=0), axis=1)
    end = np.asarray(payload["query_result"]["end_position"], dtype=np.float64)
    require(end.shape == (3,) and np.isfinite(end).all(),
            "query result contains an invalid end pose")
    final_planar_step = float(np.linalg.norm(
        end[[0, 2]] - xyz[-1, [0, 2]]))
    realized_total = float(np.sum(planar_steps) + final_planar_step)
    reported_total = float(payload["query_result"]["path_len_m"])
    require(abs(realized_total - reported_total) <= 1e-6,
            "query result path length differs from realized motion")
    return {
        "trace_frame_count": len(trace),
        "stationary_transition_count": int(np.sum(planar_steps <= 1e-12)),
        "stationary_transition_fraction": (
            0.0 if not len(planar_steps)
            else float(np.mean(planar_steps <= 1e-12))),
        "trace_realized_planar_motion_m": float(np.sum(planar_steps)),
        "post_trace_final_planar_step_m": final_planar_step,
        "recomputed_total_realized_planar_motion_m": realized_total,
        "minimum_y_m": float(np.min(xyz[:, 1])),
        "maximum_y_m": float(np.max(xyz[:, 1])),
        "vertical_span_m": float(np.ptp(xyz[:, 1])),
        "final_y_m": float(end[1]),
        "blocked_step_count": int(
            payload["query_result"].get("blocked_step_count", 0)),
        "termination_reason": payload["query_result"].get(
            "termination_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "analysis output already exists")
    completion_path = arguments.episode_root / "completion.json"
    completion_sha = verify_sha_sidecar(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    require(completion.get("schema_version") == EPISODE_SCHEMA,
            "completion schema changed")
    require(completion.get("arms") == list(TANGENT_ARMS),
            "completion arm set/order changed")
    require(completion.get("paper_result_eligible") is False,
            "consumed result was incorrectly marked paper eligible")
    require(completion.get("prefix_equality") is True
            and completion.get("target_proof_equality") is True,
            "paired prefix/proof audit did not pass")
    require(set(completion.get("outcomes", {})) == set(TANGENT_ARMS),
            "all three outcomes were not completed atomically")

    episode = str(completion["episode"])
    panels: dict[str, Any] = {}
    for arm in TANGENT_ARMS:
        arm_root = arguments.episode_root / arm
        row, payload = load_single_result(arm_root, episode)
        require(int(row["reached"]) == int(completion["outcomes"][arm]),
                f"{arm}: raw outcome differs from completion")
        panels[arm] = {
            "outcome": int(row["reached"]),
            "query_steps": int(row["steps"]),
            "realized_path_len_m": float(row["path_len_m"]),
            "final_distance_3d_m": float(row["final_goal_dist_m"]),
            "final_distance_planar_m": float(
                completion["runtime_audits"][arm]
                ["motion_and_success"]["final_distance_planar_m"]),
            "final_vertical_error_m": float(
                completion["runtime_audits"][arm]
                ["motion_and_success"]["final_vertical_error_m"]),
            "plans": fixed_plan_panel(payload["query_leg"]),
            "motion": fixed_motion_panel(payload),
        }

    chord = panels["oracle_chord_mixed_realized"]
    tangent = panels["oracle_tangent_mixed_realized"]
    selective = panels["oracle_tangent_then_native_realized"]

    def delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        """Return left minus right for lower-is-better distance quantities."""

        return {
            "final_distance_3d_m": float(
                left["final_distance_3d_m"] - right["final_distance_3d_m"]),
            "minimum_planned_geodesic_m": float(
                left["plans"]["minimum_planned_geodesic_m"]
                - right["plans"]["minimum_planned_geodesic_m"]),
            "critic_below_minus_0p5_fraction": float(
                (left["plans"]["critic_below_minus_0p5_fraction"] or 0.0)
                - (right["plans"]["critic_below_minus_0p5_fraction"] or 0.0)),
            "stationary_transition_fraction": float(
                left["motion"]["stationary_transition_fraction"]
                - right["motion"]["stationary_transition_fraction"]),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "consumed one-failure mechanism attribution only",
        "paper_result_eligible": False,
        "post_hoc_pass_threshold_defined": False,
        "completion_sha256": completion_sha,
        "history_index": int(completion["history_index"]),
        "scene": completion["scene"],
        "episode": episode,
        "benchmark_manifest_sha256": completion[
            "benchmark_manifest_sha256"],
        "success_distance_contract": completion[
            "success_distance_contract"],
        "path_length_contract": completion["path_length_contract"],
        "panels": panels,
        "fixed_pairwise_deltas_left_minus_right": {
            "continuous_tangent_minus_chord": delta(tangent, chord),
            "selective_tangent_minus_continuous_tangent": delta(
                selective, tangent),
        },
        "interpretation_order": [
            "local route semantics: continuous tangent versus chord",
            "conditioning persistence: selective versus continuous tangent",
            "floor progress: vertical span and final vertical error",
            "downstream policy: critic and stationary-motion fractions",
        ],
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    arguments.out.with_name(arguments.out.name + ".sha256").write_text(
        f"{sha256(arguments.out)}  {arguments.out.name}\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "out": str(arguments.out),
        "outcomes": completion["outcomes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
