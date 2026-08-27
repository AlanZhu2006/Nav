#!/usr/bin/env python3
"""Diagnose frozen V1 Revisit construction without reading query outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_paper_role_pair_scene import revisit_contract
from build_shared_online_double_revisit import (
    covis_curve,
    enumerate_perturbations,
    goal_world_points,
    load_online_history,
    sha256_file,
)
from generate_twoleg import make_sim


def audit_episode(episode_root: Path) -> dict:
    receipt_path = episode_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    history = load_online_history(episode_root, receipt)
    contract = revisit_contract()
    diagnostic = receipt["anchor_preselection_diagnostic"]
    source_frames = {
        "B": int(diagnostic["frame_1"]),
        "C": int(diagnostic["frame_0"]),
    }
    minimum_frame = int(contract["minimum_eligible_online_frame"])
    camera_height = float(receipt["camera_height_m"])
    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
    roles = {}
    try:
        for role, source_frame in source_frames.items():
            cheap = enumerate_perturbations(
                simulator,
                history,
                source_frame,
                camera_height=camera_height,
                min_translation_m=contract["v1_min_translation_m"],
                max_translation_m=contract["v1_max_translation_m"],
                min_yaw_delta_deg=contract["v1_min_yaw_delta_deg"],
                max_yaw_delta_deg=contract["v1_max_yaw_delta_deg"],
                min_anchor_covis=contract["v1_min_source_frame_covis"],
                minimum_pixel_mae=contract["v1_min_pixel_mae"],
            )
            candidates = []
            for candidate in cheap[:16]:
                camera_position = candidate.position + np.asarray(
                    [0.0, camera_height, 0.0]
                )
                points = goal_world_points(
                    candidate.depth, camera_position, candidate.yaw
                )
                curve = covis_curve(
                    points, history["transforms"], history["depths"]
                )
                best_frame = minimum_frame + int(
                    np.argmax(curve[minimum_frame:])
                )
                best_covis = float(curve[best_frame])
                argmax_gap = abs(best_frame - source_frame)
                argmax_pass = (
                    argmax_gap <= int(contract["v1_max_argmax_gap_frames"])
                )
                covis_pass = (
                    float(contract["v1_min_max_online_a_covis"])
                    <= best_covis
                    <= float(contract["v1_max_max_online_a_covis"])
                )
                candidates.append({
                    "attempt": int(candidate.attempt),
                    "translation_m": float(candidate.translation_m),
                    "yaw_delta_deg": float(candidate.yaw_delta_deg),
                    "source_frame_covis": float(candidate.anchor_covis),
                    "pixel_mae": float(candidate.pixel_mae),
                    "best_frame": int(best_frame),
                    "best_covis": best_covis,
                    "argmax_gap_frames": int(argmax_gap),
                    "argmax_pass": bool(argmax_pass),
                    "covis_range_pass": bool(covis_pass),
                    "fully_passed": bool(argmax_pass and covis_pass),
                })
            roles[role] = {
                "source_frame": source_frame,
                "cheap_candidate_count": len(cheap),
                "fully_audited_count": len(candidates),
                "argmax_pass_count": sum(
                    row["argmax_pass"] for row in candidates
                ),
                "covis_range_pass_count": sum(
                    row["covis_range_pass"] for row in candidates
                ),
                "fully_passed_count": sum(
                    row["fully_passed"] for row in candidates
                ),
                "candidates": candidates,
            }
    finally:
        simulator.close()
    return {
        "scene": str(receipt["scene"]),
        "episode": str(receipt["episode"]),
        "online_a_steps": int(receipt["online_a_steps"]),
        "online_a_receipt_sha256": sha256_file(receipt_path),
        "anchor_preselection_diagnostic": diagnostic,
        "roles": roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = []
    for receipt_path in sorted(args.online_root.glob("*/*/receipt.json")):
        rows.append(audit_episode(receipt_path.parent))
    if not rows:
        raise RuntimeError("no materialized online histories found")
    payload = {
        "schema_version": "revisit_v1_constructibility_audit_v1_20260814",
        "scope": "construction-only diagnostics; no query policy outcome read",
        "contract": revisit_contract(),
        "episodes": rows,
        "summary": {
            "episode_count": len(rows),
            "role_b_fully_passed": sum(
                row["roles"]["B"]["fully_passed_count"] > 0 for row in rows
            ),
            "role_c_fully_passed": sum(
                row["roles"]["C"]["fully_passed_count"] > 0 for row in rows
            ),
        },
        "query_outcomes_read": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
