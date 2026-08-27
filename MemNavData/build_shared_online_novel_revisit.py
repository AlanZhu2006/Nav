#!/usr/bin/env python3
"""Build a strict online-A -> Novel-B -> Revisit-C benchmark.

The strict-v4 generator defines the initial and Novel goals, but its Revisit
goal is supported by the *expert* A trajectory.  This builder instead consumes
one frozen native NavDP A->B trace pair, verifies every trace RGB in Habitat,
and constructs C exclusively from the observations that NavDP actually saw on
A.  At the C switch, the factual online-B endpoint must be a visual hard
negative.  The NavDP short FIFO is then reset and long-memory retrieval is
bounded at the end of A, so earlier B frames are audited descriptively but
cannot supply C.

Navigation outcomes on C are never read while constructing the benchmark.
Only A/B success is used, because the intended endpoint is SR_C | A,B.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_shared_online_double_revisit as online_goal
from deterministic_eval_protocol import (
    file_sha256,
    validate_leg1_trace,
    validate_shared_trace_source,
)
from generate_twoleg import M_W, cam_to_world_hab, covis_frac, geodesic, make_sim, render
from multigoal_benchmark_contract import (
    ROLE_SYMMETRIC_PROTOCOL,
    RoleSymmetryObservation,
    validate_role_symmetric_contract,
)


SCHEMA_VERSION = "shared_online_novel_revisit_v1_20260813"
ROLE_SEQUENCE = ("initial_imagegoal", "novel", "revisit")


class ConstructibilityError(RuntimeError):
    """The audited online prefix admits no Goal-C under the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def bytes_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def data_to_hab(position) -> np.ndarray:
    return M_W.T @ np.asarray(position, dtype=np.float64)


def parquet_pose_hab(row_action, camera_height: float) -> tuple[np.ndarray, float]:
    transform = np.stack(
        [np.asarray(row, dtype=np.float64) for row in row_action]
    ).reshape(4, 4)
    camera = M_W.T @ transform[:3, 3]
    rotation = M_W.T @ transform[:3, :3]
    yaw = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
    return camera - np.asarray([0.0, camera_height, 0.0]), yaw


def wrap_angle(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def load_trace(
    path: Path,
    *,
    episode: str,
    scene: str,
    goal: bytes,
) -> dict:
    payload = json.loads(path.read_text())
    validate_leg1_trace(
        payload,
        expected_episode=episode,
        expected_seed=int(payload.get("episode_seed", -1)),
        expected_goal_sha256=bytes_sha256(goal),
        expected_source_scene=scene,
    )
    validate_shared_trace_source(payload)
    require(
        payload["source_backend"] in ("hybrid_pose", "navdp"),
        "trace source backend is unsupported",
    )
    require(payload["source_hybrid_route"] == "phase", "trace is not native phase")
    require(
        int(payload["source_retrieval_candidate_min_gap"]) == 16,
        "trace retrieval gap changed",
    )
    require(
        abs(float(payload["source_graph_subgoal_spacing_m"])) <= 1e-12,
        "trace source used graph subgoals",
    )
    require(
        abs(float(payload["source_graph_subgoal_arrival_m"]) - 0.60) <= 1e-12,
        "trace source graph-arrival contract changed",
    )
    interventions = [
        int(plan["step"])
        for plan in payload["plans"]
        if plan.get("router_active") is True
        or plan.get("revisit_adapter_takeover") is True
        or plan.get("forced_anchor") is not None
    ]
    require(not interventions, "A/B trace was not controlled by frozen native NavDP")
    return payload


def trace_history(
    simulator,
    trace: dict,
    *,
    camera_height: float,
) -> dict:
    floor_positions = []
    camera_positions = []
    transforms = []
    depths = []
    rgbs = []
    for pose in trace["poses"]:
        floor = np.asarray(
            [pose["x"], pose["y"], pose["z"]], dtype=np.float64
        )
        camera = floor + np.asarray([0.0, camera_height, 0.0])
        yaw = float(pose["yaw"])
        rgb, depth = render(simulator, camera, yaw)
        encoded = online_goal.jpeg_bytes(rgb)
        require(
            bytes_sha256(encoded) == pose["jpg_sha256"],
            f"trace RGB hash mismatch at step {pose['step']}",
        )
        floor_positions.append(floor)
        camera_positions.append(camera)
        transforms.append(cam_to_world_hab(camera, yaw))
        depths.append(depth)
        rgbs.append(rgb)
    return {
        "trace": trace,
        "poses": trace["poses"],
        "floor_positions": floor_positions,
        "camera_positions": camera_positions,
        "transforms": transforms,
        "depths": depths,
        "rgbs": rgbs,
    }


def measured_geodesic(pathfinder, first: np.ndarray, second: np.ndarray) -> float:
    ok, distance, _points = geodesic(pathfinder, first, second)
    require(ok and np.isfinite(distance), "benchmark geodesic is invalid")
    return float(distance)


def strict_v4_audit(
    simulator,
    episode_dir: Path,
    metadata: dict,
    rows: pd.DataFrame,
) -> dict:
    require(metadata.get("gen_protocol") == ROLE_SYMMETRIC_PROTOCOL, "not strict-v4")
    require(tuple(metadata.get("role_sequence") or ()) == ROLE_SEQUENCE, "wrong roles")
    switch_a, switch_b = [int(value) for value in metadata["switches"]]
    camera_height = float(metadata.get("camera_height_m", 0.5))
    start, _start_yaw = parquet_pose_hab(rows.iloc[0]["action"], camera_height)
    a_terminal, _a_yaw = parquet_pose_hab(
        rows.iloc[switch_a - 1]["action"], camera_height
    )
    b_terminal, b_yaw = parquet_pose_hab(
        rows.iloc[switch_b - 1]["action"], camera_height
    )
    a = data_to_hab(metadata["A"])
    b = data_to_hab(metadata["goals"][0]["pos"])
    geo_a = measured_geodesic(simulator.pathfinder, start, a)
    geo_b = measured_geodesic(simulator.pathfinder, a, b)
    rgb_root = episode_dir / "videos/chunk-000/observation.images.rgb"
    image_b = (episode_dir / "goal_1.jpg").read_bytes()
    b_terminal_image = (rgb_root / f"{switch_b - 1}.jpg").read_bytes()
    observation = RoleSymmetryObservation(
        geo_a_m=geo_a,
        geo_b_m=geo_b,
        initial_pose_error_m=float(np.linalg.norm(start - data_to_hab(metadata["start"]))),
        a_terminal_pose_error_m=float(np.linalg.norm(a_terminal - a)),
        b_terminal_pose_error_m=float(np.linalg.norm(b_terminal - b)),
        b_terminal_yaw_error_deg=abs(float(np.degrees(wrap_angle(
            b_yaw - float(metadata["goals"][0]["yaw_habitat"])
        )))),
        goal_b_matches_terminal_rgb=(image_b == b_terminal_image),
    )
    report = validate_role_symmetric_contract(metadata, observation)
    require(report["ok"], "strict-v4 source failed: " + "; ".join(report["issues"]))
    return {
        "ok": True,
        "protocol": metadata["gen_protocol"],
        "geo_A_m": geo_a,
        "geo_B_m": geo_b,
        "role_distance_error_m": abs(geo_a - geo_b),
        "source_metadata_sha256": file_sha256(
            episode_dir / "meta/gen_meta.json"
        ),
        "source_parquet_sha256": file_sha256(
            episode_dir / "data/chunk-000/episode_000000.parquet"
        ),
    }


def candidate_b_visibility(
    candidate: online_goal.PerturbationCandidate,
    camera_height: float,
    b_history: dict,
) -> dict:
    points = online_goal.goal_world_points(
        candidate.depth,
        candidate.position + np.asarray([0.0, camera_height, 0.0]),
        candidate.yaw,
    )
    rollout_curve = [
        float(covis_frac(points, transform, depth))
        for transform, depth in zip(b_history["transforms"], b_history["depths"])
    ]
    endpoint_floor = np.asarray(
        b_history["trace"]["end_position"], dtype=np.float64
    )
    endpoint_yaw = float(b_history["trace"]["end_yaw"])
    endpoint_camera = endpoint_floor + np.asarray(
        [0.0, camera_height, 0.0]
    )
    _rgb, endpoint_depth = render(
        b_history["simulator"], endpoint_camera, endpoint_yaw
    )
    endpoint_covis = float(covis_frac(
        points,
        cam_to_world_hab(endpoint_camera, endpoint_yaw),
        endpoint_depth,
    ))
    return {
        "rollout_curve": rollout_curve,
        "rollout_max_covis": max(rollout_curve, default=0.0),
        "rollout_argmax": (
            int(np.argmax(rollout_curve)) if rollout_curve else None
        ),
        "switch_endpoint_covis": endpoint_covis,
    }


def select_goal_c(
    simulator,
    a_history: dict,
    b_history: dict,
    contract: dict,
) -> tuple[online_goal.PerturbationCandidate, dict]:
    minimum = int(contract["minimum_eligible_online_a_frame"])
    end = len(a_history["poses"]) - int(contract["source_end_margin_frames"])
    if end <= minimum:
        raise ConstructibilityError(
            "online A is too short for an eligible C anchor"
        )
    camera_height = float(contract["camera_height_m"])
    a_history["minimum_eligible_frame"] = minimum
    b_endpoint = np.asarray(b_history["trace"]["end_position"], dtype=np.float64)
    accepted = []
    source_frames = list(range(minimum, end, int(contract["source_stride_frames"])))
    diagnostics = {
        "online_a_steps": len(a_history["poses"]),
        "online_b_steps": len(b_history["poses"]),
        "source_frames": source_frames,
        "cheap_candidates": 0,
        "online_a_audited_candidates": 0,
        "distance_band_candidates": 0,
        "online_b_endpoint_hard_negative_candidates": 0,
        "minimum_online_b_max_covis_after_distance_gate": None,
        "minimum_online_b_endpoint_covis_after_distance_gate": None,
        "minimum_b_to_c_geodesic_after_audit_m": None,
        "maximum_b_to_c_geodesic_after_audit_m": None,
    }
    audited_distances = []
    distance_gated_tails = []
    distance_gated_endpoints = []
    b_history["simulator"] = simulator
    for source_frame in source_frames:
        cheap = online_goal.enumerate_perturbations(
            simulator,
            a_history,
            source_frame,
            camera_height=camera_height,
            min_translation_m=float(contract["minimum_translation_m"]),
            max_translation_m=float(contract["maximum_translation_m"]),
            min_yaw_delta_deg=float(contract["minimum_yaw_delta_deg"]),
            max_yaw_delta_deg=float(contract["maximum_yaw_delta_deg"]),
            min_anchor_covis=float(contract["minimum_source_frame_covis"]),
            minimum_pixel_mae=float(contract["minimum_pixel_mae"]),
        )
        diagnostics["cheap_candidates"] += len(cheap)
        audited = online_goal.fully_audit_candidates(
            cheap,
            a_history,
            source_frame,
            minimum_eligible_frame=minimum,
            maximum_argmax_gap=int(contract["maximum_argmax_gap_frames"]),
            minimum_max_covis=float(contract["minimum_max_online_a_covis"]),
            maximum_max_covis=float(contract["maximum_max_online_a_covis"]),
            limit=int(contract["maximum_candidates_per_source"]),
        )
        diagnostics["online_a_audited_candidates"] += len(audited)
        for candidate in audited:
            distance = measured_geodesic(
                simulator.pathfinder, b_endpoint, candidate.position
            )
            audited_distances.append(distance)
            if not (
                float(contract["minimum_b_to_c_geodesic_m"])
                <= distance
                <= float(contract["maximum_b_to_c_geodesic_m"])
            ):
                continue
            diagnostics["distance_band_candidates"] += 1
            visibility = candidate_b_visibility(
                candidate, camera_height, b_history
            )
            maximum_tail = float(visibility["rollout_max_covis"])
            endpoint_covis = float(visibility["switch_endpoint_covis"])
            distance_gated_tails.append(maximum_tail)
            distance_gated_endpoints.append(endpoint_covis)
            if endpoint_covis > float(
                    contract["maximum_online_b_endpoint_covis"]):
                continue
            diagnostics["online_b_endpoint_hard_negative_candidates"] += 1
            accepted.append((
                (
                    int(endpoint_covis > float(
                        contract["preferred_online_b_endpoint_covis"])),
                    abs(distance - float(contract["target_b_to_c_geodesic_m"])),
                    endpoint_covis,
                    maximum_tail,
                    -float(candidate.best_covis),
                    abs(float(candidate.translation_m) - 0.30),
                    int(source_frame),
                    int(candidate.attempt),
                ),
                candidate,
                {
                    "source_online_a_frame": int(source_frame),
                    "geo_B_to_C_m": distance,
                    "online_b_rollout_covis_curve": visibility["rollout_curve"],
                    "online_b_rollout_max_covis": maximum_tail,
                    "online_b_rollout_argmax": visibility["rollout_argmax"],
                    "online_b_switch_endpoint_covis": endpoint_covis,
                    "online_b_effective_input_contract": (
                        "NavDP short FIFO reset before C; long-memory candidate "
                        "ceiling frozen at online-A boundary; only the current "
                        "B endpoint image remains visible at the C switch"
                    ),
                },
            ))
    if audited_distances:
        diagnostics["minimum_b_to_c_geodesic_after_audit_m"] = min(
            audited_distances
        )
        diagnostics["maximum_b_to_c_geodesic_after_audit_m"] = max(
            audited_distances
        )
    if distance_gated_tails:
        diagnostics["minimum_online_b_max_covis_after_distance_gate"] = min(
            distance_gated_tails
        )
    if distance_gated_endpoints:
        diagnostics[
            "minimum_online_b_endpoint_covis_after_distance_gate"
        ] = min(distance_gated_endpoints)
    if not accepted:
        raise ConstructibilityError(
            "no online-A-supported, endpoint-negative C exists: "
            + json.dumps(diagnostics, sort_keys=True)
        )
    accepted.sort(key=lambda item: item[0])
    _score, candidate, audit = accepted[0]
    audit.update({
        "source_frames_considered": source_frames,
        "accepted_candidate_count": len(accepted),
        "candidate_diagnostics": diagnostics,
        "selection_rule": (
            "under a mandatory pre-C NavDP FIFO reset and A-bounded long "
            "memory, prefer switch-endpoint covis below the preferred "
            "threshold; then B->C distance near target, lower endpoint and "
            "descriptive full-rollout visibility, stronger online-A support, "
            "controlled perturbation, stable frame/attempt order"
        ),
    })
    return candidate, audit


def build_episode(
    *,
    episode_dir: Path,
    trace_root: Path,
    scene_asset: Path,
    destination: Path,
    contract: dict,
) -> dict:
    metadata_path = episode_dir / "meta/gen_meta.json"
    metadata = json.loads(metadata_path.read_text())
    rows = pd.read_parquet(
        episode_dir / "data/chunk-000/episode_000000.parquet"
    )
    scene = scene_asset.stem
    episode = episode_dir.name
    switch_a = int(metadata["switches"][0])
    rgb_root = episode_dir / "videos/chunk-000/observation.images.rgb"
    goal_a = (rgb_root / f"{switch_a - 1}.jpg").read_bytes()
    goal_b = (episode_dir / "goal_1.jpg").read_bytes()
    trace_a_path = trace_root / f"{episode}_leg1_trace.json"
    trace_b_path = trace_root / f"{episode}_legB_trace.json"
    trace_a = load_trace(
        trace_a_path, episode=episode, scene=scene, goal=goal_a
    )
    trace_b = load_trace(
        trace_b_path, episode=episode, scene=scene, goal=goal_b
    )
    require(trace_a["episode_seed"] == trace_b["episode_seed"], "trace seeds differ")
    require(trace_a["reached"] and trace_b["reached"], "A/B prefix did not succeed")
    require(bool(trace_a["poses"]) and bool(trace_b["poses"]), "empty A/B trace")
    camera_height = float(metadata.get("camera_height_m", 0.5))
    contract = {**contract, "camera_height_m": camera_height}
    start, _start_yaw = parquet_pose_hab(rows.iloc[0]["action"], camera_height)
    first_a = np.asarray([
        trace_a["poses"][0][axis] for axis in ("x", "y", "z")
    ])
    first_b = np.asarray([
        trace_b["poses"][0][axis] for axis in ("x", "y", "z")
    ])
    require(np.allclose(first_a, start, rtol=0.0, atol=1e-6), "online A start differs")
    require(
        np.allclose(first_b, trace_a["end_position"], rtol=0.0, atol=1e-6),
        "online B does not begin at online A endpoint",
    )
    require(
        abs(wrap_angle(
            float(trace_b["poses"][0]["yaw"]) - float(trace_a["end_yaw"])
        )) <= 1e-6,
        "online B yaw does not begin at online A endpoint",
    )

    simulator = make_sim(str(scene_asset), "", agent_radius=0.30)
    try:
        source_audit = strict_v4_audit(simulator, episode_dir, metadata, rows)
        a_history = trace_history(
            simulator, trace_a, camera_height=camera_height
        )
        b_history = trace_history(
            simulator, trace_b, camera_height=camera_height
        )
        candidate, c_audit = select_goal_c(
            simulator, a_history, b_history, contract
        )
    finally:
        simulator.close()

    assets = online_goal.write_goal(
        destination, "C", candidate.rgb, candidate.depth
    )
    c_record = online_goal.perturbation_record(
        candidate,
        a_history,
        int(c_audit["source_online_a_frame"]),
        "C",
        camera_height,
    )
    c_record.update(c_audit)
    benchmark = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "episode": episode,
        "role_sequence": list(ROLE_SEQUENCE),
        "conditional_endpoint": "SR_C_given_online_A_and_online_B_success",
        "source_episode": str(episode_dir.resolve()),
        "source_scene_asset": str(scene_asset.resolve()),
        "source_scene_asset_sha256": file_sha256(scene_asset),
        "source_strict_v4_audit": source_audit,
        "source_metadata_sha256": file_sha256(metadata_path),
        "trace_root": str(trace_root.resolve()),
        "online_a_trace": trace_a_path.name,
        "online_a_trace_sha256": file_sha256(trace_a_path),
        "online_b_trace": trace_b_path.name,
        "online_b_trace_sha256": file_sha256(trace_b_path),
        "episode_seed": int(trace_a["episode_seed"]),
        "camera_height_m": camera_height,
        "online_a_steps": int(trace_a["steps"]),
        "online_b_steps": int(trace_b["steps"]),
        "online_a_end_position": trace_a["end_position"],
        "online_b_end_position": trace_b["end_position"],
        "goal_a_sha256": bytes_sha256(goal_a),
        "goal_b_sha256": bytes_sha256(goal_b),
        "goal_c": c_record,
        "goal_c_asset": assets,
        "online_a_candidate_ceiling_semantics": (
            "candidate ceiling is frozen immediately after replaying A; "
            "all B memory frames are ineligible for C retrieval"
        ),
        "navdp_short_fifo_before_c": "reset",
        "contract": contract,
        "construction_uses_c_navigation_outcomes": False,
    }
    benchmark_path = destination / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    benchmark["benchmark_sha256"] = file_sha256(benchmark_path)
    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--scene-asset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--minimum-eligible-frame", type=int, default=39)
    parser.add_argument("--source-end-margin", type=int, default=16)
    parser.add_argument("--source-stride", type=int, default=4)
    parser.add_argument("--min-b-to-c", type=float, default=2.0)
    parser.add_argument("--max-b-to-c", type=float, default=9.0)
    parser.add_argument("--target-b-to-c", type=float, default=4.0)
    parser.add_argument("--online-b-endpoint-max-covis", type=float, default=0.10)
    parser.add_argument("--preferred-online-b-endpoint-covis", type=float, default=0.05)
    parser.add_argument("--v1-min-translation", type=float, default=0.20)
    parser.add_argument("--v1-max-translation", type=float, default=0.50)
    parser.add_argument("--v1-min-yaw", type=float, default=10.0)
    parser.add_argument("--v1-max-yaw", type=float, default=25.0)
    parser.add_argument("--v1-min-source-covis", type=float, default=0.45)
    parser.add_argument("--v1-min-max-covis", type=float, default=0.50)
    parser.add_argument("--v1-max-max-covis", type=float, default=0.98)
    parser.add_argument("--v1-max-argmax-gap", type=int, default=20)
    parser.add_argument("--v1-min-pixel-mae", type=float, default=5.0)
    parser.add_argument("--max-candidates-per-source", type=int, default=32)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help=("write a sealed zero-accepted manifest when every selected "
              "episode is structurally ineligible; integrity failures still "
              "abort"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), f"output already exists: {args.out}")
    require(args.scene_asset.is_file(), "scene asset is missing")
    require(args.trace_root.is_dir(), "trace root is missing")
    contract = {
        "minimum_eligible_online_a_frame": int(args.minimum_eligible_frame),
        "source_end_margin_frames": int(args.source_end_margin),
        "source_stride_frames": int(args.source_stride),
        "minimum_b_to_c_geodesic_m": float(args.min_b_to_c),
        "maximum_b_to_c_geodesic_m": float(args.max_b_to_c),
        "target_b_to_c_geodesic_m": float(args.target_b_to_c),
        "maximum_online_b_endpoint_covis": float(
            args.online_b_endpoint_max_covis
        ),
        "preferred_online_b_endpoint_covis": float(
            args.preferred_online_b_endpoint_covis
        ),
        "online_b_full_rollout_covis_role": (
            "audited descriptive leakage before a mandatory FIFO reset; "
            "never exposed through A-bounded long-memory retrieval"
        ),
        "minimum_translation_m": float(args.v1_min_translation),
        "maximum_translation_m": float(args.v1_max_translation),
        "minimum_yaw_delta_deg": float(args.v1_min_yaw),
        "maximum_yaw_delta_deg": float(args.v1_max_yaw),
        "minimum_source_frame_covis": float(args.v1_min_source_covis),
        "minimum_max_online_a_covis": float(args.v1_min_max_covis),
        "maximum_max_online_a_covis": float(args.v1_max_max_covis),
        "maximum_argmax_gap_frames": int(args.v1_max_argmax_gap),
        "minimum_pixel_mae": float(args.v1_min_pixel_mae),
        "maximum_candidates_per_source": int(args.max_candidates_per_source),
    }
    require(
        0.0 <= contract["preferred_online_b_endpoint_covis"]
        <= contract["maximum_online_b_endpoint_covis"] <= 1.0,
        "invalid online-B endpoint hard-negative thresholds",
    )
    require(contract["source_stride_frames"] > 0, "source stride must be positive")
    require(
        0 < contract["minimum_b_to_c_geodesic_m"]
        < contract["maximum_b_to_c_geodesic_m"],
        "invalid B-to-C distance band",
    )

    wanted = {
        item.strip() for item in args.episode_ids.split(",") if item.strip()
    }
    episode_dirs = sorted(
        path for path in args.episode_root.glob("episode_*")
        if (path / "meta/gen_meta.json").is_file()
        and (not wanted or path.name in wanted)
    )
    require(bool(episode_dirs), "no strict-v4 episodes selected")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.out.name + ".tmp.", dir=args.out.parent))
    accepted = []
    rejected = []
    try:
        for episode_dir in episode_dirs:
            destination = temporary / episode_dir.name
            destination.mkdir(parents=True)
            try:
                accepted.append(build_episode(
                    episode_dir=episode_dir,
                    trace_root=args.trace_root,
                    scene_asset=args.scene_asset,
                    destination=destination,
                    contract=contract,
                ))
            except ConstructibilityError as error:
                shutil.rmtree(destination)
                rejected.append({"episode": episode_dir.name, "reason": str(error)})
        if not accepted and not args.allow_empty:
            raise ConstructibilityError(
                "no episode passed shared-online construction: "
                + json.dumps(rejected, sort_keys=True)
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "scene": args.scene_asset.stem,
            "purpose": (
                "strict-v4 A/Novel-B with controlled Revisit-C derived only "
                "from factual online-A; pre-C FIFO reset plus an online-A "
                "candidate ceiling isolates long-term memory, while the "
                "factual B switch endpoint remains a visual hard negative"
            ),
            "contract": contract,
            "selected_before_c_navigation": True,
            "accepted": [
                {"episode": item["episode"], "benchmark_sha256": item["benchmark_sha256"]}
                for item in accepted
            ],
            "rejected": rejected,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(args.out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "output": str(args.out),
        "accepted": len(accepted),
        "rejected": rejected,
        "manifest_sha256": file_sha256(args.out / "manifest.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
