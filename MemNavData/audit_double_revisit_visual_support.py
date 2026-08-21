#!/usr/bin/env python3
"""Audit whether strict online-A memory contains certifiable C candidates.

This is an evaluation-only read of a completed double-Revisit rollout.  It
scans every causal leg-A RGB frame with the same frozen SuperPoint+LightGlue
and Fundamental-MAGSAC precheck used by certified relocalization.  It never
changes the navigation policy or candidate shortlist.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from certified_relocalization_runtime import (
    CERTIFIED_EPIPOLAR_THRESHOLD_PX,
    CERTIFIED_MINIMUM_ANCHOR,
    fundamental_can_reach_certificate,
    fundamental_support,
    runtime_contract,
)
from lingbot_pnp_localization import LightGluePointMatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--lightglue_repo", type=Path, required=True)
    parser.add_argument("--dependency_root", type=Path, required=True)
    parser.add_argument("--certified_plans", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = json.loads(args.plans.read_text())
    metadata = json.loads(args.meta.read_text())
    trace = plans["memory_traces"]["legA"]
    frames = [int(item["frame_idx"]) for item in trace]
    if not frames or frames != list(range(frames[0], frames[-1] + 1)):
        raise RuntimeError("leg-A online memory trace must be non-empty and contiguous")
    by_frame = {int(item["frame_idx"]): item for item in trace}
    goal_data = metadata["goals"][1]["pos"]
    goal_xz = (float(goal_data[0]), -float(goal_data[1]))

    dino_shortlist = []
    if args.certified_plans is not None:
        certified = json.loads(args.certified_plans.read_text())
        if certified.get("legC"):
            dino_shortlist = list(
                certified["legC"][0].get("router_candidate_order_dino") or [])

    matcher = LightGluePointMatcher(
        args.lightglue_repo,
        dependency_root=args.dependency_root,
        device="cuda:0",
        max_keypoints=2048,
    )
    records = []
    for frame_idx in frames:
        if frame_idx < CERTIFIED_MINIMUM_ANCHOR:
            continue
        image = args.buffer / f"{frame_idx}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        matched = matcher.match_paths(
            image,
            args.goal,
            target_height=518,
            target_width=518,
            patch_size=14,
        )
        support = fundamental_support(
            matched["reference_raw_points"],
            matched["query_raw_points"],
            matched["scores"],
            tuple(matched["reference_raw_hw"]),
            tuple(matched["query_raw_hw"]),
            threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX,
        )
        possible, reason = fundamental_can_reach_certificate(support)
        item = by_frame[frame_idx]
        distance_m = math.hypot(
            float(item["x"]) - goal_xz[0],
            float(item["z"]) - goal_xz[1],
        )
        records.append({
            "frame_idx": frame_idx,
            "step": int(item["step"]),
            "goal_distance_m": distance_m,
            "in_dino_top8": frame_idx in dino_shortlist,
            "dino_rank": (
                dino_shortlist.index(frame_idx) + 1
                if frame_idx in dino_shortlist else None),
            "precheck_passed": possible,
            "precheck_reason": reason,
            **support,
        })

    nearest = min(records, key=lambda item: (
        item["goal_distance_m"], item["frame_idx"]))
    passed = [item for item in records if item["precheck_passed"]]
    shortlisted = [item for item in records if item["in_dino_top8"]]
    payload = {
        "schema_version": "double_revisit_visual_support_audit_v1",
        "scene": args.scene,
        "goal_role": "revisit_C",
        "history_scope": "online_leg_A_only",
        "candidate_ceiling": frames[-1],
        "goal_xz": list(goal_xz),
        "runtime_contract": runtime_contract(),
        "dino_top8": dino_shortlist,
        "summary": {
            "eligible_frames": len(records),
            "precheck_pass_count": len(passed),
            "dino_top8_observed_count": len(shortlisted),
            "dino_top8_precheck_pass_count": sum(
                item["precheck_passed"] for item in shortlisted),
            "nearest_anchor": nearest,
            "nearest_precheck_passed_anchor": (
                min(passed, key=lambda item: (
                    item["goal_distance_m"], item["frame_idx"]))
                if passed else None),
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
