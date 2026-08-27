#!/usr/bin/env python3
"""Run the X-NavDP direction actuator on a frozen RGB-D input pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.audit_navdp_critic_direction_sweep import DEFAULT_DIRECTIONS_DEG
from MemNavData.audit_observed_frontier_bearing_coverage import sha256_file
from MemNavData.audit_xnavdp_direction_execution import (
    BASE_NAVDP_SHA256,
    EMBODIMENTS,
    OFFICIAL_POSTTRAIN_SHA256,
    _cold_history,
    _load_official_policy,
    _predict_request,
    _process_depth,
    _process_image,
    _read_csv,
    _trajectory_heading_deg,
    _write_csv,
    summarize_states,
)
from MemNavData.novel_a_bearing_gate import (
    critic_shadow_diagnostics,
    wrap_deg,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pack_report_path = args.input_pack / "report.json"
    pack_states_path = args.input_pack / "states.json"
    pack_inputs_path = args.input_pack / "inputs.npz"
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    if sha256_file(pack_states_path) != pack_report["provenance"][
            "states_sha256"]:
        raise RuntimeError("input-pack state metadata changed")
    if sha256_file(pack_inputs_path) != pack_report["provenance"][
            "inputs_sha256"]:
        raise RuntimeError("input-pack RGB-D arrays changed")
    if pack_report["provenance"][
            "baseline_checkpoint_sha256"] != BASE_NAVDP_SHA256:
        raise RuntimeError("input pack does not bind the frozen base NavDP")

    records = json.loads(pack_states_path.read_text(encoding="utf-8"))
    with np.load(pack_inputs_path, allow_pickle=False) as arrays:
        images = np.asarray(arrays["images_bgr"])
        depths = np.asarray(arrays["depths_m"])
    if len(records) != len(images) or len(records) != len(depths):
        raise RuntimeError("input-pack array and metadata counts differ")
    if args.max_states is not None:
        records = records[:args.max_states]
        images = images[:args.max_states]
        depths = depths[:args.max_states]

    baseline_directions = _read_csv(args.baseline_directions)
    baseline_by_request = {
        (row["scene"], row["episode"], int(row["plan_index"]),
         round(float(row["request_direction_deg"]), 6)): row
        for row in baseline_directions
    }
    policy, load_report = _load_official_policy(args)
    if args.actor_mode == "base":
        # The checkpoint audit proves this decoder is byte-identical to the
        # frozen NavDP base.  Setting ft_step=0 bypasses every new denoising
        # block while retaining the identical X-NavDP preprocessing/Q readout.
        policy.ft_step = 0

    state_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for record, image, depth in zip(records, images, depths):
        scene = record["scene"]
        episode = record["episode"]
        plan_index = int(record["plan_index"])
        oracle_relative = float(record["oracle_relative_deg"])
        current_image = _process_image(image[None])
        input_images = _cold_history(current_image, memory_size=8)
        input_depth = _process_depth(depth[None])
        probes = []
        all_candidate_oracle_errors = []
        for direction_index, direction_deg in enumerate(args.directions_deg):
            response = _predict_request(
                policy,
                input_images=input_images,
                input_depths=input_depth,
                direction_deg=direction_deg,
                radius_m=args.radius_m,
                seed=int(record["diffusion_seed"]),
                sample_num=args.sample_num,
                embodiment=EMBODIMENTS[args.embodiment],
                device=args.device,
            )
            shadow = critic_shadow_diagnostics(
                response, requested_heading_deg=direction_deg)
            selected_trajectory = np.asarray(
                response["trajectory"], dtype=np.float64)
            selected_extent = float(np.linalg.norm(
                selected_trajectory[-1, :2]))
            selected_path_length = float(np.linalg.norm(np.diff(
                np.concatenate([
                    np.zeros((1, 2), dtype=np.float64),
                    selected_trajectory[:, :2],
                ], axis=0),
                axis=0), axis=1).sum())
            selected_heading = shadow["selected_heading_deg"]
            executed_error = (
                None if selected_heading is None else
                abs(wrap_deg(selected_heading - oracle_relative)))
            request_error = abs(wrap_deg(direction_deg - oracle_relative))
            candidate_headings = [
                _trajectory_heading_deg(candidate)
                for candidate in np.asarray(response["all_trajectory"])]
            candidate_oracle_errors = [
                abs(wrap_deg(heading - oracle_relative))
                for heading in candidate_headings if heading is not None]
            all_candidate_oracle_errors.extend(candidate_oracle_errors)
            baseline = baseline_by_request[
                (scene, episode, plan_index, round(direction_deg, 6))]
            row = {
                "scene": scene,
                "episode": episode,
                "plan_index": plan_index,
                "frame_index": int(record["frame_index"]),
                "direction_index": direction_index,
                "request_direction_deg": direction_deg,
                "oracle_relative_deg": oracle_relative,
                "request_error_deg": request_error,
                "selected_heading_deg": selected_heading,
                "selected_extent_m": selected_extent,
                "selected_path_length_m": selected_path_length,
                "selected_request_error_deg": shadow[
                    "selected_request_error_deg"],
                "executed_error_deg": executed_error,
                "best_candidate_request_error_deg": shadow[
                    "best_direction_error_deg"],
                "best_candidate_q_rank": shadow[
                    "best_direction_critic_rank"],
                "candidate_oracle_ceiling_error_deg": (
                    min(candidate_oracle_errors)
                    if candidate_oracle_errors else None),
                "q_max": shadow["critic_max"],
                "q_min": shadow["critic_min"],
                "q_unique_4dp": shadow["critic_unique_4dp"],
                "heading_resultant_r": shadow["heading_resultant_r"],
                "base_selected_heading_deg": baseline[
                    "selected_heading_deg"],
                "base_selected_request_error_deg": baseline[
                    "selected_request_error_deg"],
                "base_executed_error_deg": baseline["executed_error_deg"],
            }
            probes.append(row)
            direction_rows.append(row)

        nearest = min(probes, key=lambda row: row["request_error_deg"])
        q_order = sorted(
            probes, key=lambda row: float(row["q_max"]), reverse=True)
        q_chosen = q_order[0]
        selected_execution_errors = [
            row["executed_error_deg"] for row in probes
            if row["executed_error_deg"] is not None]
        state = {
            "scene": scene,
            "episode": episode,
            "plan_index": plan_index,
            "frame_index": int(record["frame_index"]),
            "oracle_relative_deg": oracle_relative,
            "oracle_request_direction_deg": nearest[
                "request_direction_deg"],
            "oracle_request_error_deg": nearest["request_error_deg"],
            "xnav_oracle_request_heading_deg": nearest[
                "selected_heading_deg"],
            "xnav_oracle_request_fidelity_error_deg": nearest[
                "selected_request_error_deg"],
            "xnav_oracle_request_executed_error_deg": nearest[
                "executed_error_deg"],
            "base_oracle_request_heading_deg": nearest[
                "base_selected_heading_deg"],
            "base_oracle_request_fidelity_error_deg": nearest[
                "base_selected_request_error_deg"],
            "base_oracle_request_executed_error_deg": nearest[
                "base_executed_error_deg"],
            "xnav_request_execution_ceiling_error_deg": (
                min(selected_execution_errors)
                if selected_execution_errors else None),
            "xnav_candidate_execution_ceiling_error_deg": (
                min(all_candidate_oracle_errors)
                if all_candidate_oracle_errors else None),
            "xnav_q_chosen_direction_deg": q_chosen[
                "request_direction_deg"],
            "xnav_q_chosen_request_error_deg": q_chosen[
                "request_error_deg"],
            "xnav_q_chosen_heading_deg": q_chosen[
                "selected_heading_deg"],
            "xnav_q_chosen_executed_error_deg": q_chosen[
                "executed_error_deg"],
            "xnav_q_margin": float(q_order[0]["q_max"])
            - float(q_order[1]["q_max"]),
        }
        state_rows.append(state)
        xnav_error = state["xnav_oracle_request_executed_error_deg"]
        base_error = float(state["base_oracle_request_executed_error_deg"])
        q_error = state["xnav_q_chosen_executed_error_deg"]
        print(
            f"[{scene}/{episode}] oracle={oracle_relative:+.1f} "
            f"request={nearest['request_direction_deg']:+.0f} "
            f"xnav={xnav_error if xnav_error is not None else 'none'} "
            f"base={base_error:.1f} "
            f"q={q_error if q_error is not None else 'none'}",
            flush=True,
        )

    report = {
        "scope": (
            "privileged, failure-enriched, consumed-development plan-0 "
            "architecture diagnostic; X-NavDP receives pure PointGoal queries; "
            "deployment_approved=false"),
        "definitions": {
            "directions_deg": list(args.directions_deg),
            "radius_m": args.radius_m,
            "sample_num_per_request": args.sample_num,
            "embodiment": args.embodiment,
            "actor_mode": args.actor_mode,
            "identical_torch_rng_per_direction": True,
            "history": "cold seven-zero-frame history plus current frame",
            "rtc_previous_trajectory_guidance": False,
            "cross_request_q_is_goal_relevance_score": False,
            "baseline": "frozen NavDP mixed ImageGoal/PointGoal resample",
        },
        "summary": summarize_states(
            state_rows, direction_rows, args.threshold_deg),
        "model_load": load_report,
        "provenance": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "official_root": str(args.official_root.resolve()),
            "input_pack": str(args.input_pack.resolve()),
            "input_pack_report_sha256": sha256_file(pack_report_path),
            "input_pack_inputs_sha256": sha256_file(pack_inputs_path),
            "input_pack_states_sha256": sha256_file(pack_states_path),
            "baseline_directions": str(args.baseline_directions.resolve()),
            "baseline_directions_sha256": sha256_file(
                args.baseline_directions),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    _write_csv(args.out / "states.csv", state_rows)
    _write_csv(args.out / "directions.csv", direction_rows)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack", type=Path, required=True)
    parser.add_argument(
        "--official-root", type=Path,
        default=Path(
            ".diagnostics/xnavdp_official_878740a2011856d0/NavDP"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            ".diagnostics/xnavdp_official_878740a2011856d0/"
            "x-navdp_posttrain.ckpt"))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=OFFICIAL_POSTTRAIN_SHA256)
    parser.add_argument(
        "--baseline-directions", type=Path,
        default=Path(
            ".diagnostics/navdp_critic_direction_sweep_plan0_20260809/"
            "directions.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embodiment", choices=tuple(EMBODIMENTS),
                        default="wheeled")
    parser.add_argument("--actor-mode", choices=("posttrain", "base"),
                        default="posttrain")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--sample-num", type=int, default=8)
    parser.add_argument("--radius-m", type=float, default=2.0)
    parser.add_argument("--threshold-deg", type=float, default=30.0)
    args = parser.parse_args()
    args.directions_deg = DEFAULT_DIRECTIONS_DEG
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        parser.error("X-NavDP checkpoint differs from the frozen pin")
    if args.max_states is not None and args.max_states < 1:
        parser.error("max-states must be positive")
    if args.sample_num < 2 or args.radius_m <= 0 or args.threshold_deg <= 0:
        parser.error("sample-num >=2 and positive radius/threshold are required")
    if args.out.exists():
        parser.error("output directory already exists")
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
