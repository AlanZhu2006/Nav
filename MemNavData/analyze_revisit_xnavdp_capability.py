#!/usr/bin/env python3
"""Describe where X-NavDP capability did and did not transfer in Revisit R2.

This is an attribution audit over an already consumed evaluation set.  Its
bearing bins are descriptive and must never be imported as a deployment gate.
The output makes that limitation machine-readable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = "revisit_xnavdp_capability_audit_v1"

# Coarse circular-statistics bins for description only.  They are not a tuned
# or authorized controller selector.
FORWARD_MAX_ABS_DEG = 60.0
SIDE_MAX_ABS_DEG = 135.0


def bearing_bucket(bearing_deg: float | None) -> str:
    if bearing_deg is None:
        return "router_inactive"
    bearing = float(bearing_deg)
    if not math.isfinite(bearing):
        raise ValueError("bearing must be finite")
    magnitude = abs(bearing)
    if magnitude < FORWARD_MAX_ABS_DEG:
        return "forward"
    if magnitude < SIDE_MAX_ABS_DEG:
        return "side_or_side_rear"
    return "deep_rear"


def _selection_episodes(selection: Mapping[str, Any]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for scene_group in selection.get("scene_groups", []):
        scene = scene_group.get("scene")
        for episode in scene_group.get("episodes", []):
            key = (scene, episode.get("episode"))
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError(f"invalid selection key {key!r}")
            if key in indexed:
                raise ValueError(f"duplicate selection episode {key!r}")
            indexed[key] = dict(episode)
    return indexed


def _empty_bucket() -> dict[str, Any]:
    return {
        "episodes": 0,
        "mixed_successes": 0,
        "base_point_successes": 0,
        "official_x_successes": 0,
        "official_x_control_count": 0,
        "official_x_negative_velocity_controls": 0,
    }


def analyze_payloads(
    selection: Mapping[str, Any], report: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _selection_episodes(selection)
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("R2 report must contain episodes")

    buckets = {
        "forward": _empty_bucket(),
        "side_or_side_rear": _empty_bucket(),
        "deep_rear": _empty_bucket(),
        "router_inactive": _empty_bucket(),
    }
    patterns: Counter[str] = Counter()
    discordant = []
    rows = []

    for episode in episodes:
        scene = episode.get("scene")
        episode_id = episode.get("episode")
        key = (scene, episode_id)
        if key not in selected:
            raise ValueError(f"R2 episode is absent from selection {key!r}")
        first_state = selected[key].get("r0_first_router_state")
        bearing = (first_state.get("bearing_deg")
                   if isinstance(first_state, Mapping) else None)
        bucket_name = bearing_bucket(bearing)

        mixed = bool(episode.get("r0_reached_b"))
        base = bool(episode.get("base_point", {}).get("reached_b"))
        official = bool(episode.get("official_mpc", {}).get("reached_b"))
        safety = episode.get("official_mpc", {}).get("official_safety", {})
        controls = int(safety.get("control_count", 0) or 0)
        negative = int(safety.get("negative_velocity_controls", 0) or 0)
        if controls < 0 or negative < 0 or negative > controls:
            raise ValueError(f"invalid X control counts for {key!r}")

        bucket = buckets[bucket_name]
        bucket["episodes"] += 1
        bucket["mixed_successes"] += int(mixed)
        bucket["base_point_successes"] += int(base)
        bucket["official_x_successes"] += int(official)
        bucket["official_x_control_count"] += controls
        bucket["official_x_negative_velocity_controls"] += negative

        pattern = f"mixed={int(mixed)},base={int(base)},x={int(official)}"
        patterns[pattern] += 1
        row = {
            "scene": scene,
            "episode": episode_id,
            "first_active_bearing_deg": bearing,
            "descriptive_bearing_bucket": bucket_name,
            "mixed_success": mixed,
            "base_point_success": base,
            "official_x_success": official,
            "official_x_control_count": controls,
            "official_x_negative_velocity_controls": negative,
            "official_x_negative_control_fraction": (
                negative / controls if controls else None),
            "mixed_path_b_m": episode.get("r0_path_b_m"),
            "official_x_path_b_m": episode.get(
                "official_mpc", {}).get("path_b_m"),
        }
        rows.append(row)
        if len({mixed, base, official}) > 1:
            discordant.append(row)

    for bucket in buckets.values():
        controls = bucket["official_x_control_count"]
        negative = bucket["official_x_negative_velocity_controls"]
        bucket["official_x_negative_control_fraction"] = (
            negative / controls if controls else None)

    if len(rows) != int(report.get("conditional_b_denominator", -1)):
        raise ValueError("R2 denominator does not match episode records")

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": (
            "post_hoc_consumed_r2_mechanism_description_only; "
            "not_a_selector_not_unseen_not_paper_confirmation"),
        "deployment_authorization": {
            "authorize_global_x_controller": False,
            "authorize_bearing_threshold_router": False,
            "authorize_blind": False,
            "reason": (
                "bearing/outcome relationship was inspected after R2; any "
                "capability envelope requires a fresh pre-registered set"),
        },
        "descriptive_bins_abs_bearing_deg": {
            "forward": f"[0,{FORWARD_MAX_ABS_DEG})",
            "side_or_side_rear": (
                f"[{FORWARD_MAX_ABS_DEG},{SIDE_MAX_ABS_DEG})"),
            "deep_rear": f"[{SIDE_MAX_ABS_DEG},180]",
            "status": "descriptive_only_not_a_controller_gate",
        },
        "episode_count": len(rows),
        "scene_count": len({row["scene"] for row in rows}),
        "outcome_patterns": dict(sorted(patterns.items())),
        "by_first_active_bearing": buckets,
        "discordant_episode_count": len(discordant),
        "discordant_episodes": sorted(
            discordant, key=lambda row: (row["scene"], row["episode"])),
        "episodes": sorted(
            rows, key=lambda row: (row["scene"], row["episode"])),
        "interpretation_contract": {
            "supported": (
                "official X reverse execution is a specialized capability "
                "whose transfer differs across first-active bearing regions"),
            "not_supported": (
                "X is a globally superior Revisit controller, or an angle "
                "threshold selected from R2 will generalize"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_bytes = args.selection.read_bytes()
    report_bytes = args.report.read_bytes()
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    raw_report = json.loads(report_bytes)
    if raw_report.get("selection_sha256") != selection_sha256:
        raise ValueError("R2 report does not reference the supplied selection")
    result = analyze_payloads(
        json.loads(selection_bytes),
        raw_report,
    )
    result["inputs"] = {
        "selection": str(args.selection),
        "selection_sha256": selection_sha256,
        "report": str(args.report),
        "report_sha256": report_sha256,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "episodes": result["episode_count"],
        "scenes": result["scene_count"],
        "by_first_active_bearing": result["by_first_active_bearing"],
        "deployment_authorization": result["deployment_authorization"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
