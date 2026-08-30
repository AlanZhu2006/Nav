#!/usr/bin/env python3
"""Build one result-blind Table-III role pair from a causal RGB survey.

The survey is a physically ordered sequence of rendered RGB observations along
the frozen geodesic.  Simulator geometry is used only to construct and score
the benchmark.  At query runtime, CEC and NavDP replay RGB only; no survey pose
or simulator depth enters either model.
"""

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
    enumerate_perturbations,
    goal_world_points,
    jpeg_bytes,
    load_online_history,
    write_depth_png,
)
from build_shared_online_role_pairs import query_geometry
from construct_hm3d_table3_actual_mono_role_pair import (
    ConstructionIneligible,
    eligible_support,
    measured_geometry,
)
from generate_twoleg import (
    K,
    covis_curve,
    geodesic,
    make_sim,
    render,
)
from hm3d_table3_causal_survey_contract import survey_frames
from hm3d_table3_length_contract import (
    SCHEMA_VERSION,
    angle_separation_degrees,
    in_bin,
)


PROTOCOL_SCHEMA = "hm3d_table3_causal_survey_protocol_v1_20260830"
FRAGMENT_SCHEMA = "hm3d_table3_causal_survey_fragment_v1_20260830"
TRACE_SCHEMA = "hm3d_table3_causal_survey_trace_v1_20260830"
RECEIPT_SCHEMA = "hm3d_table3_causal_survey_materialized_v1_20260830"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_survey(
    *, simulator, row: dict, frames: list[tuple[np.ndarray, float]],
    destination: Path, protocol: dict,
) -> dict:
    require(not destination.exists(), "survey history already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp.", dir=destination.parent
    ))
    try:
        rgb_root = temporary / "rgb"
        depth_root = temporary / "depth"
        rgb_root.mkdir()
        depth_root.mkdir()
        camera_height = float(protocol["history"]["camera_height_m"])
        poses = []
        for step, (floor_position, yaw) in enumerate(frames):
            if not simulator.pathfinder.is_navigable(floor_position):
                raise ConstructionIneligible(
                    f"survey frame {step} is not navigable"
                )
            rgb, depth = render(
                simulator,
                floor_position + np.asarray([0.0, camera_height, 0.0]),
                yaw,
            )
            encoded = jpeg_bytes(rgb)
            rgb_path = rgb_root / f"{step:06d}.jpg"
            rgb_path.write_bytes(encoded)
            write_depth_png(depth_root / f"{step:06d}.png", depth)
            poses.append({
                "step": step,
                "x": float(floor_position[0]),
                "y": float(floor_position[1]),
                "z": float(floor_position[2]),
                "yaw": float(yaw),
                "jpg_sha256": hashlib.sha256(encoded).hexdigest(),
            })
        minimum_frames = int(protocol["history"]["minimum_frames"])
        if len(poses) < minimum_frames:
            raise ConstructionIneligible("survey history is too short")
        stride = int(protocol["history"]["navdp_fifo_replay_stride"])
        replay_steps = list(range(0, len(poses), stride))
        if replay_steps[-1] != len(poses) - 1:
            replay_steps.append(len(poses) - 1)
        plans = [{
            "step": step,
            "purpose": "navdp_fifo_replay_keyframe_only",
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed": False,
            "diffusion_sampled": False,
        } for step in replay_steps]
        path_length = float(sum(
            np.linalg.norm(second[0] - first[0])
            for first, second in zip(frames[:-1], frames[1:])
        ))
        episode = f"episode_table3_survey_{int(row['history_index']):03d}"
        episode_seed = int(row["history_index"])
        trace = {
            "schema_version": TRACE_SCHEMA,
            "episode": episode,
            "source_scene": str(row["scene"]),
            "source_backend": "controlled_geodesic_survey",
            "source_hybrid_route": "causal_survey",
            "episode_seed": episode_seed,
            "reached": True,
            "termination_reason": "causal_survey_complete",
            "steps": len(poses),
            "path_len": path_length,
            "path_len_at_reach": path_length,
            "step_at_reach": len(poses) - 1,
            "final_goal_dist_m": 0.0,
            "poses": poses,
            "plans": plans,
            "end_position": [
                float(value) for value in frames[-1][0]
            ],
            "end_yaw": float(frames[-1][1]),
            "metric_depth_sensor_reads": 0,
            "query_policy_outcomes_read": False,
        }
        trace_path = temporary / "online_a_trace.json"
        trace_path.write_text(json.dumps(
            trace, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        shutil.copy2(rgb_root / f"{len(poses) - 1:06d}.jpg",
                     temporary / "goal_a.jpg")
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "scene": str(row["scene"]),
            "episode": episode,
            "online_a_reached": True,
            "online_a_steps": len(poses),
            "online_a_trace_sha256": sha256(trace_path),
            "source_asset": str(Path(row["asset"]["glb_path"]).resolve()),
            "source_asset_sha256": str(row["asset"]["glb_sha256"]),
            "camera_height_m": camera_height,
            "camera_intrinsic": np.asarray(K, dtype=float).tolist(),
            "episode_seed": episode_seed,
            "history_source": "controlled_causal_rgb_geodesic_survey",
            "survey_contract": {
                "translation_step_m": float(protocol["history"]["translation_step_m"]),
                "maximum_yaw_step_deg": float(
                    protocol["history"]["maximum_yaw_step_deg"]),
                "navdp_fifo_replay_stride": stride,
                "runtime_memory_input": "RGB only",
                "construction_only_simulator_depth": True,
                "metric_depth_for_query_control_or_CEC": False,
            },
            "goal_a_sha256": sha256(temporary / "goal_a.jpg"),
            "rgb_frame_hashes": [pose["jpg_sha256"] for pose in poses],
            "query_policy_outcomes_read": False,
        }
        receipt_path = temporary / "receipt.json"
        receipt_path.write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        temporary.replace(destination)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_image_pair(rgb, depth, root: Path) -> tuple[str, str]:
    root.mkdir(parents=True)
    rgb_path = root / "goal.jpg"
    depth_path = root / "goal_depth.png"
    rgb_path.write_bytes(jpeg_bytes(rgb))
    write_depth_png(depth_path, depth)
    return sha256(rgb_path), sha256(depth_path)


def construct_queries(
    *, simulator, row: dict, survey_root: Path, destination: Path,
    protocol: dict,
) -> tuple[dict, dict]:
    receipt = json.loads((survey_root / "receipt.json").read_text())
    trace = json.loads((survey_root / "online_a_trace.json").read_text())
    history = load_online_history(survey_root, receipt)
    endpoint = np.asarray(trace["end_position"], dtype=np.float64)
    camera_height = float(receipt["camera_height_m"])
    pathfinder = simulator.pathfinder
    diagnostics = {"source_frames_ranked": [], "revisit_candidates_scored": 0}

    novel_position = np.asarray(
        row["capacity_geometry"]["second_goal"], dtype=np.float64
    )
    novel_distance, novel_bearing, novel_path = measured_geometry(
        pathfinder, endpoint, novel_position
    )
    bin_spec = next(
        spec for spec in protocol["length_definition"]["bins_m"]
        if spec["name"] == row["bin_name"]
    )
    if not in_bin(novel_distance, bin_spec):
        raise ConstructionIneligible("Novel escaped the frozen length bin")
    from collect_hm3d_table3_actual_mono_a import final_path_yaw
    novel_yaw = final_path_yaw(novel_path, novel_position)
    novel_rgb, novel_depth = render(
        simulator, novel_position + [0.0, camera_height, 0.0], novel_yaw
    )
    novel_points = goal_world_points(
        novel_depth, novel_position + [0.0, camera_height, 0.0], novel_yaw
    )
    novel_curve = covis_curve(
        novel_points, history["transforms"], history["depths"]
    )
    novel_support, novel_frame = eligible_support(novel_curve)
    if not float(np.max(novel_curve)) < float(
        protocol["query_construction"]["novel_max_history_covis_exclusive"]
    ):
        raise ConstructionIneligible("capacity Novel overlaps causal survey")

    frame_floor = int(protocol["query_construction"]["eligible_frame_floor"])
    end_margin = int(
        protocol["query_construction"]["eligible_end_margin_frames"]
    )
    source_rows = []
    stop = len(history["poses"]) - end_margin
    for frame in range(frame_floor, stop, 4):
        geometry = query_geometry(
            pathfinder, endpoint, history["floor_positions"][frame]
        )
        if geometry is None:
            continue
        distance, _bearing = geometry
        if in_bin(float(distance), bin_spec):
            source_rows.append((
                abs(float(distance) - novel_distance), frame, float(distance)
            ))
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
            curve = covis_curve(
                points, history["transforms"], history["depths"]
            )
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
                "ranking": ranking, "proposal": proposal, "curve": curve,
                "support": support, "support_frame": support_frame,
                "source_frame": frame, "distance": float(distance),
                "bearing": float(bearing), "distance_error": distance_error,
                "separation": separation,
            }
            if selected is None or ranking < selected["ranking"]:
                selected = record
    if selected is None:
        raise ConstructionIneligible(
            "no controlled Revisit satisfies length/support/direction contract"
        )

    proposal = selected["proposal"]
    novel_rgb_sha, novel_depth_sha = write_image_pair(
        novel_rgb, novel_depth, destination / "pair_00/novel"
    )
    revisit_rgb_sha, revisit_depth_sha = write_image_pair(
        proposal.rgb, proposal.depth, destination / "pair_00/revisit"
    )
    common = {
        "eligible_online_a_frame_floor": frame_floor,
        "eligible_online_a_end_margin_frames": end_margin,
    }
    novel = {
        **common,
        "query_id": "pair_00_novel", "analysis_role": "novel",
        "goal_rgb": "pair_00/novel/goal.jpg",
        "goal_rgb_sha256": novel_rgb_sha,
        "goal_depth": "pair_00/novel/goal_depth.png",
        "goal_depth_sha256": novel_depth_sha,
        "floor_position": [float(value) for value in novel_position],
        "yaw_rad": float(novel_yaw),
        "geodesic_from_a_end_m": novel_distance,
        "initial_path_bearing_rad": novel_bearing,
        "max_online_a_covis": novel_support,
        "max_online_a_covis_frame": novel_frame,
        "global_max_online_a_covis": float(np.max(novel_curve)),
        "covis_curve": [float(value) for value in novel_curve],
        "source": "frozen_capacity_second_goal",
    }
    revisit = {
        **common,
        "query_id": "pair_00_revisit", "analysis_role": "revisit",
        "goal_rgb": "pair_00/revisit/goal.jpg",
        "goal_rgb_sha256": revisit_rgb_sha,
        "goal_depth": "pair_00/revisit/goal_depth.png",
        "goal_depth_sha256": revisit_depth_sha,
        "floor_position": [float(value) for value in proposal.position],
        "yaw_rad": float(proposal.yaw),
        "geodesic_from_a_end_m": selected["distance"],
        "initial_path_bearing_rad": selected["bearing"],
        "max_online_a_covis": selected["support"],
        "max_online_a_covis_frame": selected["support_frame"],
        "covis_curve": [float(value) for value in selected["curve"]],
        "source_online_frame": selected["source_frame"],
        "translation_from_source_m": float(proposal.translation_m),
        "yaw_delta_from_source_deg": float(proposal.yaw_delta_deg),
        "source_frame_covis": float(proposal.anchor_covis),
        "pixel_mae_from_source": float(proposal.pixel_mae),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(row["scene"]),
        "episode": str(receipt["episode"]),
        "bin_name": str(row["bin_name"]),
        "candidate_identity_sha256": row["candidate_identity_sha256"],
        "history_index": int(row["history_index"]),
        "history_source": "controlled_causal_rgb_geodesic_survey",
        "runtime_geometry": "content_addressed_pinned_navmesh",
        "runtime_navmesh": row["asset"]["navmesh_path"],
        "runtime_navmesh_sha256": row["asset"]["navmesh_sha256"],
        "online_a_episode": str(survey_root.resolve()),
        "online_a_receipt_sha256": sha256(survey_root / "receipt.json"),
        "online_a_trace_sha256": sha256(survey_root / "online_a_trace.json"),
        "online_a_steps": len(history["poses"]),
        "online_a_endpoint": {
            "floor_position": [float(value) for value in endpoint],
            "yaw_rad": float(trace["end_yaw"]),
        },
        "pairs": [{
            "pair_id": "pair_00",
            "role_distance_error_m": selected["distance_error"],
            "role_initial_path_bearing_separation_deg": selected["separation"],
            "queries": [novel, revisit],
        }],
    }
    sidecar = destination / "role_pairs.json"
    sidecar.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    payload["role_pairs_sha256"] = sha256(sidecar)
    return payload, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("--history-index", type=int)
    index_group.add_argument("--plan-index", type=int)
    args = parser.parse_args()
    plan_index = (
        int(args.plan_index) if args.plan_index is not None
        else int(args.history_index)
    )
    require(not args.run_root.joinpath(
        "construction_fragments", f"{plan_index:03d}"
    ).exists(), "construction fragment exists")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "causal-survey protocol changed")
    require(sha256(args.candidate_plan)
            == protocol["source_candidate_plan"]["sha256"],
            "source candidate plan changed")
    plan = json.loads(args.candidate_plan.read_text())
    require(len(plan["episodes"])
            == int(protocol["source_candidate_plan"]["candidate_count"]),
            "source candidate count changed")
    require(0 <= plan_index < len(plan["episodes"]),
            "plan index outside candidate plan")
    row = plan["episodes"][plan_index]
    history_index = int(row["history_index"])
    if args.plan_index is None:
        require(history_index == plan_index,
                "legacy candidate order changed")
    else:
        require(int(row.get("plan_index", -1)) == plan_index,
                "expansion candidate order changed")

    survey_tag = (
        f"survey_{history_index:04d}" if args.plan_index is not None
        else f"survey_{history_index:03d}"
    )
    fragment_root = args.run_root / "construction_fragments" / f"{plan_index:03d}"
    fragment_root.mkdir(parents=True)
    survey_root = (
        args.run_root / "survey_histories" / row["scene"]
        / survey_tag
    )
    candidate_root = (
        args.run_root / "role_pair_candidates" / row["scene"]
        / survey_tag
    )
    fragment = {
        "schema_version": FRAGMENT_SCHEMA,
        "history_index": history_index,
        "scene": row["scene"],
        "bin_name": row["bin_name"],
        "candidate_identity_sha256": row["candidate_identity_sha256"],
        "source_candidate_plan_sha256": sha256(args.candidate_plan),
        "protocol_sha256": sha256(args.protocol),
        "query_policy_outcomes_read": False,
    }
    if args.plan_index is not None:
        fragment["plan_index"] = plan_index
    simulator = make_sim(
        row["asset"]["glb_path"], row["asset"]["navmesh_path"],
        agent_radius=0.30, recompute_navmesh=False,
    )
    try:
        start = np.asarray(row["capacity_geometry"]["first_goal"], dtype=float)
        endpoint = np.asarray(row["capacity_geometry"]["query_start"], dtype=float)
        ok, distance, points = geodesic(simulator.pathfinder, start, endpoint)
        require(ok and math.isfinite(distance), "survey geodesic is invalid")
        try:
            frames = survey_frames(
                points,
                step_m=float(protocol["history"]["translation_step_m"]),
                maximum_yaw_step_deg=float(
                    protocol["history"]["maximum_yaw_step_deg"]),
            )
            receipt = write_survey(
                simulator=simulator, row=row, frames=frames,
                destination=survey_root, protocol=protocol,
            )
            candidate_root.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(
                prefix=candidate_root.name + ".tmp.", dir=candidate_root.parent
            ))
            payload, diagnostics = construct_queries(
                simulator=simulator, row=row, survey_root=survey_root,
                destination=temporary, protocol=protocol,
            )
            temporary.replace(candidate_root)
            fragment.update({
                "status": "constructed", "constructed": True,
                "role_pair_candidate": str(candidate_root.resolve()),
                "role_pairs_sha256": payload["role_pairs_sha256"],
                "online_history": str(survey_root.resolve()),
                "online_history_receipt_sha256": sha256(
                    survey_root / "receipt.json"),
                "online_history_trace_sha256": sha256(
                    survey_root / "online_a_trace.json"),
                "online_history_steps": int(receipt["online_a_steps"]),
                "survey_geodesic_m": float(distance),
                "construction_diagnostics": diagnostics,
            })
        except (ConstructionIneligible, ValueError) as error:
            if "temporary" in locals():
                shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(survey_root, ignore_errors=True)
            fragment.update({
                "status": "geometry_ineligible", "constructed": False,
                "reason": str(error), "survey_geodesic_m": float(distance),
            })
    finally:
        simulator.close()
    receipt_path = fragment_root / "completion.json"
    receipt_path.write_text(json.dumps(
        fragment, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    receipt_path.with_name("completion.json.sha256").write_text(
        f"{sha256(receipt_path)}  completion.json\n"
    )
    print(json.dumps({
        "plan_index": plan_index,
        "history_index": history_index,
        "status": fragment["status"],
        "online_history_steps": fragment.get("online_history_steps"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
