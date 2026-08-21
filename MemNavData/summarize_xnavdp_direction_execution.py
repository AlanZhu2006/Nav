#!/usr/bin/env python3
"""Compare X-NavDP post-trained and byte-identical base actor probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.audit_observed_frontier_bearing_coverage import sha256_file
from MemNavData.audit_xnavdp_direction_execution import (
    _exact_mcnemar_two_sided,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str], include_direction: bool) -> tuple:
    fields: tuple[Any, ...] = (
        row["scene"], row["episode"], int(row["plan_index"]))
    if include_direction:
        fields += (round(float(row["request_direction_deg"]), 6),)
    return fields


def _finite(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def paired_counts(
    left: list[bool], right: list[bool],
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired vectors differ in length")
    gains = sum(a and not b for a, b in zip(left, right))
    losses = sum(b and not a for a, b in zip(left, right))
    return {
        "left_hits": int(sum(left)),
        "right_hits": int(sum(right)),
        "gains": int(gains),
        "losses": int(losses),
        "exact_mcnemar_p": _exact_mcnemar_two_sided(gains, losses),
    }


def summarize(
    post_states: list[dict[str, str]],
    post_directions: list[dict[str, str]],
    base_states: list[dict[str, str]],
    base_directions: list[dict[str, str]],
    threshold_deg: float,
) -> dict[str, Any]:
    post_state_map = {_key(row, False): row for row in post_states}
    base_state_map = {_key(row, False): row for row in base_states}
    post_direction_map = {_key(row, True): row for row in post_directions}
    base_direction_map = {_key(row, True): row for row in base_directions}
    if post_state_map.keys() != base_state_map.keys():
        raise RuntimeError("post/base state sets differ")
    if post_direction_map.keys() != base_direction_map.keys():
        raise RuntimeError("post/base direction sets differ")

    state_keys = sorted(post_state_map)
    post_oracle = [
        _finite(post_state_map[key],
                "xnav_oracle_request_executed_error_deg") is not None
        and float(post_state_map[key][
            "xnav_oracle_request_executed_error_deg"]) <= threshold_deg
        for key in state_keys
    ]
    base_actor_oracle = [
        _finite(base_state_map[key],
                "xnav_oracle_request_executed_error_deg") is not None
        and float(base_state_map[key][
            "xnav_oracle_request_executed_error_deg"]) <= threshold_deg
        for key in state_keys
    ]
    mixed_oracle = [
        _finite(post_state_map[key],
                "base_oracle_request_executed_error_deg") is not None
        and float(post_state_map[key][
            "base_oracle_request_executed_error_deg"]) <= threshold_deg
        for key in state_keys
    ]

    per_direction = {}
    directions = sorted({key[-1] for key in post_direction_map})
    for direction in directions:
        keys = [key for key in sorted(post_direction_map)
                if key[-1] == direction]
        post_errors = [
            _finite(post_direction_map[key], "selected_request_error_deg")
            for key in keys]
        base_errors = [
            _finite(base_direction_map[key], "selected_request_error_deg")
            for key in keys]
        post_extents = [
            _finite(post_direction_map[key], "selected_extent_m")
            for key in keys]
        base_extents = [
            _finite(base_direction_map[key], "selected_extent_m")
            for key in keys]
        post_hits = [value is not None and value <= threshold_deg
                     for value in post_errors]
        base_hits = [value is not None and value <= threshold_deg
                     for value in base_errors]
        comparison = paired_counts(post_hits, base_hits)
        comparison.update({
            "states": len(keys),
            "post_finite": sum(value is not None for value in post_errors),
            "base_actor_finite": sum(
                value is not None for value in base_errors),
            "post_mean_request_error_deg": (
                float(np.mean([value for value in post_errors
                               if value is not None]))
                if any(value is not None for value in post_errors) else None),
            "base_actor_mean_request_error_deg": (
                float(np.mean([value for value in base_errors
                               if value is not None]))
                if any(value is not None for value in base_errors) else None),
            "post_mean_extent_m": float(np.mean([
                value for value in post_extents if value is not None])),
            "base_actor_mean_extent_m": float(np.mean([
                value for value in base_extents if value is not None])),
        })
        per_direction[str(int(direction))] = comparison

    return {
        "states": len(state_keys),
        "scene_clusters": len({key[0] for key in state_keys}),
        "threshold_deg": threshold_deg,
        "oracle_nearest_request": {
            "posttrain_vs_base_actor": paired_counts(
                post_oracle, base_actor_oracle),
            "posttrain_vs_current_mixed": paired_counts(
                post_oracle, mixed_oracle),
            "base_actor_vs_current_mixed": paired_counts(
                base_actor_oracle, mixed_oracle),
        },
        "selected_request_fidelity_by_direction": per_direction,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    post_report_path = args.post / "report.json"
    base_report_path = args.base / "report.json"
    post_report = json.loads(post_report_path.read_text(encoding="utf-8"))
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if post_report["definitions"].get("actor_mode", "posttrain") != "posttrain":
        raise RuntimeError("post artifact is not a post-trained actor run")
    if base_report["definitions"].get("actor_mode") != "base":
        raise RuntimeError("base artifact is not a base-actor run")
    for field in ("checkpoint_sha256", "input_pack_inputs_sha256",
                  "baseline_directions_sha256"):
        if post_report["provenance"][field] != base_report["provenance"][field]:
            raise RuntimeError(f"post/base provenance differs for {field}")
    threshold = float(post_report["summary"]["threshold_deg"])
    if float(base_report["summary"]["threshold_deg"]) != threshold:
        raise RuntimeError("post/base thresholds differ")

    comparison = summarize(
        _read_csv(args.post / "states.csv"),
        _read_csv(args.post / "directions.csv"),
        _read_csv(args.base / "states.csv"),
        _read_csv(args.base / "directions.csv"),
        threshold,
    )
    rear = comparison["selected_request_fidelity_by_direction"]["-180"]
    oracle = comparison["oracle_nearest_request"]
    report = {
        "scope": (
            "consumed plan-0 attribution diagnostic; directional rows are "
            "state-correlated and do not constitute closed-loop evidence"),
        "comparison": comparison,
        "interpretation": {
            "default_executor_replacement_supported": (
                oracle["posttrain_vs_current_mixed"]["gains"]
                > oracle["posttrain_vs_current_mixed"]["losses"]),
            "posttraining_adds_rear_mode_on_this_probe": (
                rear["left_hits"] > rear["right_hits"]),
            "rear_mode_is_deployable_closed_loop_result": False,
            "xnavdp_supplies_imagegoal_direction": False,
            "recommended_role": (
                "retain current mixed-token executor by default; preserve "
                "X-NavDP only as a rear-recovery hypothesis for a separate "
                "train-scene safety/progress gate"),
        },
        "provenance": {
            "post_report": str(post_report_path.resolve()),
            "post_report_sha256": sha256_file(post_report_path),
            "post_states_sha256": sha256_file(args.post / "states.csv"),
            "post_directions_sha256": sha256_file(
                args.post / "directions.csv"),
            "base_report": str(base_report_path.resolve()),
            "base_report_sha256": sha256_file(base_report_path),
            "base_states_sha256": sha256_file(args.base / "states.csv"),
            "base_directions_sha256": sha256_file(
                args.base / "directions.csv"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
