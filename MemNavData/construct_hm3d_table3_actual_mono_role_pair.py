#!/usr/bin/env python3
"""Construct one Table-III role pair from an actual mono Goal-A history."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from build_shared_online_double_revisit import (
    angle_delta_degrees, enumerate_perturbations, goal_world_points,
    jpeg_bytes, load_online_history, write_depth_png,
)
from build_shared_online_role_pairs import query_geometry
from collect_hm3d_table3_actual_mono_a import final_path_yaw
from generate_twoleg import covis_curve, geodesic, make_sim, render
from materialize_online_a_traces import (
    SingleAnchorTraceCandidate, materialize_one, native_control_audit,
)
from hm3d_table3_length_contract import (
    SCHEMA_VERSION, angle_separation_degrees, in_bin,
)


RECEIPT_SCHEMA = "hm3d_table3_actual_mono_construction_fragment_v1_20260830"
ELIGIBLE_FRAME_FLOOR = 39
END_MARGIN_FRAMES = 32


class ConstructionIneligible(RuntimeError):
    """The frozen candidate cannot satisfy query geometry/support gates."""


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(value) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float)]


def write_image_pair(rgb, depth, root: Path) -> tuple[str, str]:
    root.mkdir(parents=True)
    rgb_path = root / "goal.jpg"
    depth_path = root / "goal_depth.png"
    rgb_path.write_bytes(jpeg_bytes(rgb))
    write_depth_png(depth_path, depth)
    return sha256(rgb_path), sha256(depth_path)


def measured_geometry(pathfinder, endpoint, goal) -> tuple[float, float, list]:
    geometry = query_geometry(pathfinder, endpoint, goal)
    require(geometry is not None, "query has no finite geodesic/bearing")
    ok, distance, points = geodesic(pathfinder, endpoint, goal)
    require(ok and math.isfinite(distance)
            and abs(float(distance) - geometry[0]) <= 1e-5,
            "query geodesic changed between measurements")
    return float(geometry[0]), float(geometry[1]), points


def eligible_support(curve: np.ndarray) -> tuple[float, int]:
    require(len(curve) > ELIGIBLE_FRAME_FLOOR + END_MARGIN_FRAMES,
            "actual history has no certificate-eligible interior")
    stop = len(curve) - END_MARGIN_FRAMES
    values = curve[ELIGIBLE_FRAME_FLOOR:stop]
    index = ELIGIBLE_FRAME_FLOOR + int(np.argmax(values))
    return float(curve[index]), index


def construct(
    *, row: dict, factual: dict, trace_path: Path, carrier_root: Path,
    materialized_root: Path, destination: Path, protocol: dict,
) -> tuple[dict | None, dict]:
    trace = json.loads(trace_path.read_text())
    require(trace["reached"] is True and native_control_audit(trace)["ok"],
            "factual Goal-A history is not eligible native control")
    require(len(trace["poses"]) >= 40, "factual history has fewer than 40 frames")
    scene, episode = str(factual["scene"]), str(factual["episode"])
    anchor = min(ELIGIBLE_FRAME_FLOOR, len(trace["poses"]) - 1)
    candidate = SingleAnchorTraceCandidate(
        path=trace_path, payload=trace, score_m=0.0, anchor=anchor,
        distance_to_end_m=0.0,
    )
    receipt = materialize_one(
        candidate, asset_root=Path("/unused"), episode_root=carrier_root,
        destination=materialized_root,
        asset_map={scene: Path(row["asset"]["glb_path"])},
    )
    history = load_online_history(materialized_root, receipt)
    endpoint = np.asarray(trace["end_position"], dtype=float)
    camera_height = float(receipt["camera_height_m"])
    simulator = make_sim(
        row["asset"]["glb_path"], row["asset"]["navmesh_path"],
        agent_radius=0.30, recompute_navmesh=False,
    )
    diagnostics = {"source_frames_ranked": [], "revisit_candidates_scored": 0}
    try:
        pathfinder = simulator.pathfinder
        novel_position = np.asarray(row["capacity_geometry"]["second_goal"],
                                    dtype=float)
        novel_distance, novel_bearing, novel_path = measured_geometry(
            pathfinder, endpoint, novel_position)
        bin_spec = next(spec for spec in protocol["length_definition"]["bins_m"]
                        if spec["name"] == row["bin_name"])
        if not in_bin(novel_distance, bin_spec):
            raise ConstructionIneligible(
                "actual endpoint moved Novel outside frozen bin")
        novel_yaw = final_path_yaw(novel_path, novel_position)
        novel_rgb, novel_depth = render(
            simulator, novel_position + [0.0, camera_height, 0.0], novel_yaw)
        novel_points = goal_world_points(
            novel_depth, novel_position + [0.0, camera_height, 0.0], novel_yaw)
        novel_curve = covis_curve(
            novel_points, history["transforms"], history["depths"])
        novel_support, novel_frame = eligible_support(novel_curve)
        if not float(np.max(novel_curve)) < float(
            protocol["query_construction"]["novel_max_history_covis_exclusive"]
        ):
            raise ConstructionIneligible(
                "capacity Novel overlaps the actual causal history")

        source_rows = []
        stop = len(history["poses"]) - END_MARGIN_FRAMES
        for frame in range(ELIGIBLE_FRAME_FLOOR, stop, 4):
            geometry = query_geometry(
                pathfinder, endpoint, history["floor_positions"][frame])
            if geometry is None:
                continue
            distance, _bearing = geometry
            if in_bin(float(distance), bin_spec):
                source_rows.append((abs(float(distance) - novel_distance), frame,
                                    float(distance)))
        source_rows.sort()
        source_rows = source_rows[:8]
        diagnostics["source_frames_ranked"] = [
            {"frame": frame, "distance_m": distance, "novel_error_m": error}
            for error, frame, distance in source_rows
        ]
        selected = None
        for _error, frame, _source_distance in source_rows:
            proposals = enumerate_perturbations(
                simulator, history, frame, camera_height=camera_height,
                min_translation_m=0.20, max_translation_m=0.80,
                min_yaw_delta_deg=12.0, max_yaw_delta_deg=45.0,
                min_anchor_covis=0.55, minimum_pixel_mae=5.0,
            )
            for proposal in proposals[:24]:
                diagnostics["revisit_candidates_scored"] += 1
                geometry = query_geometry(pathfinder, endpoint, proposal.position)
                if geometry is None:
                    continue
                distance, bearing = geometry
                if not in_bin(float(distance), bin_spec):
                    continue
                distance_error = abs(float(distance) - novel_distance)
                if distance_error > float(
                    protocol["query_construction"]["maximum_role_distance_mismatch_m"]
                ):
                    continue
                separation = angle_separation_degrees(bearing, novel_bearing)
                if separation < float(
                    protocol["query_construction"]["minimum_initial_bearing_separation_deg"]
                ):
                    continue
                points = goal_world_points(
                    proposal.depth,
                    proposal.position + [0.0, camera_height, 0.0],
                    proposal.yaw,
                )
                curve = covis_curve(points, history["transforms"], history["depths"])
                support, support_frame = eligible_support(curve)
                if support < float(
                    protocol["query_construction"]["revisit_min_history_covis_inclusive"]
                ):
                    continue
                ranking = (
                    distance_error, abs(support - 0.72),
                    abs(support_frame - frame), frame, proposal.attempt,
                )
                record = {
                    "ranking": ranking, "proposal": proposal,
                    "curve": curve, "support": support,
                    "support_frame": support_frame, "source_frame": frame,
                    "distance": float(distance), "bearing": float(bearing),
                    "distance_error": distance_error, "separation": separation,
                }
                if selected is None or ranking < selected["ranking"]:
                    selected = record
        if selected is None:
            raise ConstructionIneligible(
                "no controlled Revisit satisfies length/support/direction contract")

        proposal = selected["proposal"]
        novel_rgb_sha, novel_depth_sha = write_image_pair(
            novel_rgb, novel_depth, destination / "pair_00/novel")
        revisit_rgb_sha, revisit_depth_sha = write_image_pair(
            proposal.rgb, proposal.depth, destination / "pair_00/revisit")
        common_novel = {
            "query_id": "pair_00_novel", "analysis_role": "novel",
            "goal_rgb": "pair_00/novel/goal.jpg",
            "goal_rgb_sha256": novel_rgb_sha,
            "goal_depth": "pair_00/novel/goal_depth.png",
            "goal_depth_sha256": novel_depth_sha,
            "floor_position": vector(novel_position), "yaw_rad": novel_yaw,
            "geodesic_from_a_end_m": novel_distance,
            "initial_path_bearing_rad": novel_bearing,
            "max_online_a_covis": novel_support,
            "max_online_a_covis_frame": novel_frame,
            "global_max_online_a_covis": float(np.max(novel_curve)),
            "eligible_online_a_frame_floor": ELIGIBLE_FRAME_FLOOR,
            "eligible_online_a_end_margin_frames": END_MARGIN_FRAMES,
            "covis_curve": [float(value) for value in novel_curve],
            "source": "frozen_capacity_second_goal",
        }
        common_revisit = {
            "query_id": "pair_00_revisit", "analysis_role": "revisit",
            "goal_rgb": "pair_00/revisit/goal.jpg",
            "goal_rgb_sha256": revisit_rgb_sha,
            "goal_depth": "pair_00/revisit/goal_depth.png",
            "goal_depth_sha256": revisit_depth_sha,
            "floor_position": vector(proposal.position),
            "yaw_rad": float(proposal.yaw),
            "geodesic_from_a_end_m": selected["distance"],
            "initial_path_bearing_rad": selected["bearing"],
            "max_online_a_covis": selected["support"],
            "max_online_a_covis_frame": selected["support_frame"],
            "eligible_online_a_frame_floor": ELIGIBLE_FRAME_FLOOR,
            "eligible_online_a_end_margin_frames": END_MARGIN_FRAMES,
            "covis_curve": [float(value) for value in selected["curve"]],
            "source_online_frame": selected["source_frame"],
            "translation_from_source_m": float(proposal.translation_m),
            "yaw_delta_from_source_deg": float(proposal.yaw_delta_deg),
            "source_frame_covis": float(proposal.anchor_covis),
            "pixel_mae_from_source": float(proposal.pixel_mae),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scene": scene, "episode": episode,
            "bin_name": row["bin_name"],
            "candidate_identity_sha256": row["candidate_identity_sha256"],
            "history_index": int(row["history_index"]),
            "runtime_geometry": "content_addressed_pinned_navmesh",
            "runtime_navmesh": row["asset"]["navmesh_path"],
            "runtime_navmesh_sha256": row["asset"]["navmesh_sha256"],
            "online_a_episode": str(materialized_root.resolve()),
            "online_a_receipt_sha256": sha256(materialized_root / "receipt.json"),
            "online_a_trace_sha256": sha256(materialized_root / "online_a_trace.json"),
            "online_a_steps": len(history["poses"]),
            "online_a_endpoint": {
                "floor_position": vector(endpoint), "yaw_rad": float(trace["end_yaw"]),
            },
            "pairs": [{
                "pair_id": "pair_00",
                "role_distance_error_m": selected["distance_error"],
                "role_initial_path_bearing_separation_deg": selected["separation"],
                "queries": [common_novel, common_revisit],
            }],
        }
        sidecar = destination / "role_pairs.json"
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
        payload["role_pairs_sha256"] = sha256(sidecar)
        return payload, diagnostics
    finally:
        simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--history-index", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.candidate_plan.read_text())
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version")
            == "hm3d_table3_actual_mono_protocol_v1_20260830",
            "Table-III construction protocol changed")
    require(sha256(args.protocol) == plan.get("protocol_sha256"),
            "candidate plan/construction protocol binding changed")
    require(0 <= args.history_index < len(plan["episodes"]), "history index invalid")
    row = plan["episodes"][args.history_index]
    factual_label = (
        f"{args.history_index:03d}_{row['scene']}_episode_{row['episode']}"
    )
    factual_path = args.run_root / "factual_a" / factual_label / "completion.json"
    require(factual_path.is_file(), "factual Goal-A completion is missing")
    factual_sidecar = factual_path.with_name("completion.json.sha256")
    require(
        factual_sidecar.is_file()
        and factual_sidecar.read_text().split()
        == [sha256(factual_path), "completion.json"],
        "factual Goal-A completion receipt changed",
    )
    factual = json.loads(factual_path.read_text())
    require(factual["candidate_identity_sha256"] == row["candidate_identity_sha256"],
            "factual Goal-A candidate identity changed")
    require(factual.get("runtime_geometry")
            == "content_addressed_pinned_navmesh"
            and factual.get("runtime_navmesh_sha256")
            == row["asset"]["navmesh_sha256"],
            "factual Goal-A runtime geometry changed")
    fragment_root = args.run_root / "construction_fragments" / f"{args.history_index:03d}"
    require(not fragment_root.exists(), "construction fragment exists")
    fragment_root.mkdir(parents=True)
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA, "history_index": args.history_index,
        "scene": row["scene"], "bin_name": row["bin_name"],
        "candidate_identity_sha256": row["candidate_identity_sha256"],
        "factual_A_completion_sha256": sha256(factual_path),
        "query_policy_outcomes_read": False,
    }
    if not factual["history_eligible"]:
        receipt_payload.update({"status": "factual_A_ineligible", "constructed": False})
    else:
        trace_path = Path(factual["trace_path"])
        require(sha256(trace_path) == factual["trace_sha256"],
                "factual Goal-A trace changed")
        materialized = args.run_root / "materialized_a" / row["scene"] / factual["episode"]
        candidate_destination = (
            args.run_root / "role_pair_candidates" / row["scene"] / factual["episode"]
        )
        require(not candidate_destination.exists(), "role-pair candidate exists")
        candidate_parent = args.run_root / "role_pair_candidates" / row["scene"]
        candidate_parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=factual["episode"] + ".tmp.",
            dir=candidate_parent,
        ))
        try:
            payload, diagnostics = construct(
                row=row, factual=factual, trace_path=trace_path,
                carrier_root=args.run_root / "carriers",
                materialized_root=materialized, destination=temporary,
                protocol=protocol,
            )
            temporary.replace(candidate_destination)
            receipt_payload.update({
                "status": "constructed", "constructed": True,
                "role_pair_candidate": str(candidate_destination.resolve()),
                "role_pairs_sha256": payload["role_pairs_sha256"],
                "online_a_episode": payload["online_a_episode"],
                "online_a_steps": payload["online_a_steps"],
                "construction_diagnostics": diagnostics,
            })
        except ConstructionIneligible as error:
            shutil.rmtree(temporary, ignore_errors=True)
            receipt_payload.update({
                "status": "geometry_ineligible", "constructed": False,
                "reason": str(error),
            })
    receipt = fragment_root / "completion.json"
    receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
    receipt.with_name(receipt.name + ".sha256").write_text(
        f"{sha256(receipt)}  {receipt.name}\n")
    print(json.dumps({"history_index": args.history_index,
                      "status": receipt_payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
