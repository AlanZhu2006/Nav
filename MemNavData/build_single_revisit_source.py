#!/usr/bin/env python3
"""Build deployable single-Revisit candidates from one causal online history.

The role-pair task asks one independent query after Goal A.  It therefore must
not inherit the unrelated B-to-C distance constraint of a three-leg/double-
Revisit benchmark.  This builder searches eligible historical frames using
only online-history geometry, then applies the unchanged controlled-pose V1
visual-support contract.  No navigation query outcome is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import build_shared_online_double_revisit as legacy
from generate_twoleg import make_sim


SCHEMA_VERSION = "single_revisit_source_v1_20260814"
V1_NAME = legacy.V1_NAME


def build_episode(episode_root: Path, destination: Path, contract: dict) -> dict:
    receipt_path = episode_root / "receipt.json"
    trace_path = episode_root / "online_a_trace.json"
    receipt = json.loads(receipt_path.read_text())
    history = legacy.load_online_history(episode_root, receipt)
    minimum_frame = int(contract["minimum_eligible_online_frame"])
    history["minimum_eligible_frame"] = minimum_frame
    end_margin = int(contract["source_anchor_end_margin_frames"])
    stride = int(contract["source_anchor_stride_frames"])
    endpoint = history["floor_positions"][-1]
    camera_height = float(receipt["camera_height_m"])

    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
    source_rows = []
    rejected_frames = []
    try:
        frame_distances = []
        for frame in range(minimum_frame, len(history["poses"]) - end_margin, stride):
            try:
                distance = legacy.goal_distance(
                    simulator.pathfinder,
                    endpoint,
                    history["floor_positions"][frame],
                )
            except RuntimeError:
                rejected_frames.append({
                    "source_frame": frame,
                    "reason": "disconnected_from_online_a_endpoint",
                })
                continue
            if not (
                float(contract["minimum_query_geodesic_m"])
                <= distance
                <= float(contract["maximum_query_geodesic_m"])
            ):
                rejected_frames.append({
                    "source_frame": frame,
                    "reason": "source_frame_query_distance_outside_contract",
                    "geodesic_from_a_end_m": distance,
                })
                continue
            frame_distances.append((
                abs(distance - float(contract["target_query_geodesic_m"])),
                frame,
                distance,
            ))

        for _distance_error, frame, source_distance in sorted(frame_distances):
            cheap = legacy.enumerate_perturbations(
                simulator,
                history,
                frame,
                camera_height=camera_height,
                min_translation_m=contract["v1_min_translation_m"],
                max_translation_m=contract["v1_max_translation_m"],
                min_yaw_delta_deg=contract["v1_min_yaw_delta_deg"],
                max_yaw_delta_deg=contract["v1_max_yaw_delta_deg"],
                min_anchor_covis=contract["v1_min_source_frame_covis"],
                minimum_pixel_mae=contract["v1_min_pixel_mae"],
            )
            audited = legacy.fully_audit_candidates(
                cheap,
                history,
                frame,
                minimum_eligible_frame=minimum_frame,
                maximum_argmax_gap=contract["v1_max_argmax_gap_frames"],
                minimum_max_covis=contract["v1_min_max_online_a_covis"],
                maximum_max_covis=contract["v1_max_max_online_a_covis"],
            )
            selected = None
            for candidate in audited:
                distance = legacy.goal_distance(
                    simulator.pathfinder, endpoint, candidate.position
                )
                if (
                    float(contract["minimum_query_geodesic_m"])
                    <= distance
                    <= float(contract["maximum_query_geodesic_m"])
                ):
                    selected = (candidate, distance)
                    break
            if selected is None:
                rejected_frames.append({
                    "source_frame": frame,
                    "reason": "no_controlled_v1_candidate_passed",
                    "source_geodesic_from_a_end_m": source_distance,
                    "cheap_candidate_count": len(cheap),
                    "fully_audited_candidate_count": len(audited),
                })
                continue
            candidate, distance = selected
            role = f"R{len(source_rows):02d}"
            source_rows.append({
                "role": role,
                "source_frame": frame,
                "candidate": candidate,
                "geodesic_from_a_end_m": distance,
            })
            if len(source_rows) == int(contract["maximum_revisit_candidates"]):
                break
    finally:
        simulator.close()

    if not source_rows:
        raise RuntimeError(
            "no single-Revisit candidate passed the frozen V1 contract"
        )

    variant_root = destination / V1_NAME
    assets = {
        row["role"]: legacy.write_goal(
            variant_root,
            row["role"],
            row["candidate"].rgb,
            row["candidate"].depth,
        )
        for row in source_rows
    }
    goals = {
        row["role"]: {
            **legacy.perturbation_record(
                row["candidate"],
                history,
                row["source_frame"],
                row["role"],
                camera_height,
            ),
            "geodesic_from_a_end_m": float(row["geodesic_from_a_end_m"]),
        }
        for row in source_rows
    }
    variants = {
        V1_NAME: {
            "goals": goals,
            "assets": assets,
            "candidate_roles": [row["role"] for row in source_rows],
        }
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": receipt["scene"],
        "episode": receipt["episode"],
        "source_online_episode": str(episode_root.resolve()),
        "source_online_receipt_sha256": legacy.sha256_file(receipt_path),
        "source_online_trace_sha256": legacy.sha256_file(trace_path),
        "goal_a": {
            "path": str((episode_root / "goal_a.jpg").resolve()),
            "sha256": legacy.sha256_file(episode_root / "goal_a.jpg"),
        },
        "online_a_steps": len(history["poses"]),
        "selection": {
            "query_outcomes_read": False,
            "candidate_order": (
                "closest source-frame geodesic to frozen target, then frame"
            ),
            "accepted_source_frames": [row["source_frame"] for row in source_rows],
            "rejected_source_frames": rejected_frames,
        },
        "variants": variants,
    }
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "benchmark.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    payload["benchmark_sha256"] = legacy.sha256_file(metadata_path)
    return payload
