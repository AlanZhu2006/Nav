#!/usr/bin/env python3
"""Build route-negative V0/V1 double-Revisit goals from online-A history.

V1 of the original pilot selected anchors only by temporal/spatial separation.
That is insufficient: the factual B return path can still see Goal C from far
away.  V2 selects B/C only when a deterministic Habitat shortest-path proxy
from online-A's endpoint to B remains a visual hard negative for C.  The
closed-loop evaluator still rechecks the complete factual B rollout and
censors C if the proxy was insufficient.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import build_shared_online_double_revisit as v1
from generate_twoleg import (
    covis_frac,
    densify,
    geodesic,
    make_sim,
    render,
    yaw_facing,
)


SCHEMA_VERSION = "shared_online_double_revisit_v2_route_negative_20260812"
V0_NAME = v1.V0_NAME
V1_NAME = v1.V1_NAME


def reference_path_observations(
    simulator,
    start_position: np.ndarray,
    start_yaw: float,
    target_position: np.ndarray,
    *,
    camera_height: float,
    sample_step_m: float,
    yaw_scan_count: int = 1,
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    """Render a deterministic A-end->B geodesic visibility proxy.

    ``yaw_scan_count=1`` preserves the frozen V2 path-facing contract.  Larger
    values form a conservative heading envelope at each path sample and are
    used to diagnose whether a target can leak under controller-dependent yaw.
    """
    if int(yaw_scan_count) < 1:
        raise ValueError("yaw_scan_count must be positive")
    ok, distance, points = geodesic(
        simulator.pathfinder, start_position, target_position
    )
    if not ok or not np.isfinite(distance) or not points:
        raise RuntimeError("reference A-end->B path is invalid")
    observations = []
    def append_yaw_envelope(camera: np.ndarray, base_yaw: float) -> None:
        for yaw_index in range(int(yaw_scan_count)):
            yaw = float(
                base_yaw + yaw_index * (2.0 * np.pi / int(yaw_scan_count))
            )
            _rgb, depth = render(simulator, camera, yaw)
            observations.append((v1.cam_to_world_hab(camera, yaw), depth))

    start_camera = start_position + np.asarray([0.0, camera_height, 0.0])
    append_yaw_envelope(start_camera, float(start_yaw))
    dense = densify(points, sample_step_m)
    for index, position in enumerate(dense):
        if index + 1 < len(dense):
            delta = (dense[index + 1] - position)[[0, 2]]
        elif index > 0:
            delta = (position - dense[index - 1])[[0, 2]]
        else:
            delta = np.asarray([0.0, -1.0])
        yaw = float(yaw_facing(delta))
        camera = np.asarray(position, dtype=float) + np.asarray(
            [0.0, camera_height, 0.0]
        )
        append_yaw_envelope(camera, yaw)
    return float(distance), observations


def tail_curve(
    goal_points: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    return np.asarray(
        [covis_frac(goal_points, transform, depth)
         for transform, depth in observations],
        dtype=float,
    )


def exact_goal_points(history: dict, frame: int) -> np.ndarray:
    return v1.goal_world_points(
        history["depths"][frame],
        history["camera_positions"][frame],
        float(history["poses"][frame]["yaw"]),
    )


def source_pair_score(row: dict, preferred_tail: float, target_geo: float) -> tuple:
    return (
        int(float(row["reference_tail_max_covis"]) > preferred_tail),
        abs(float(row["geo_A_to_B_m"]) - target_geo)
        + abs(float(row["geo_B_to_C_m"]) - target_geo),
        float(row["reference_tail_max_covis"]),
        -int(row["source_anchor_gap_frames"]),
        int(row["B_source_frame"]),
        int(row["C_source_frame"]),
    )


def select_source_pair(
    simulator,
    history: dict,
    contract: dict,
) -> tuple[list[dict], dict]:
    minimum = int(contract["minimum_eligible_online_frame"])
    end_margin = int(contract["source_anchor_end_margin_frames"])
    stride = int(contract["source_anchor_stride_frames"])
    frames = list(range(minimum, len(history["poses"]) - end_margin, stride))
    if len(frames) < 2:
        raise RuntimeError("online-A trace has too few source-anchor candidates")
    endpoint = history["floor_positions"][-1]
    endpoint_yaw = float(history["poses"][-1]["yaw"])
    camera_height = float(
        history["camera_positions"][0][1]
        - history["floor_positions"][0][1]
    )
    goal_points = {frame: exact_goal_points(history, frame) for frame in frames}
    valid = []
    for frame_b in frames:
        position_b = history["floor_positions"][frame_b]
        distance_a_b, observations = reference_path_observations(
            simulator,
            endpoint,
            endpoint_yaw,
            position_b,
            camera_height=camera_height,
            sample_step_m=contract["reference_path_sample_step_m"],
        )
        if distance_a_b < contract["minimum_leg_geodesic_m"]:
            continue
        for frame_c in frames:
            gap = abs(frame_b - frame_c)
            if gap < contract["minimum_anchor_gap_frames"]:
                continue
            ok, distance_b_c, _points = geodesic(
                simulator.pathfinder,
                position_b,
                history["floor_positions"][frame_c],
            )
            if (
                not ok
                or not np.isfinite(distance_b_c)
                or distance_b_c < contract["minimum_leg_geodesic_m"]
            ):
                continue
            curve = tail_curve(goal_points[frame_c], observations)
            maximum = float(np.max(curve))
            if maximum > contract["reference_path_c_tail_max_covis"]:
                continue
            valid.append(
                {
                    "B_source_frame": frame_b,
                    "C_source_frame": frame_c,
                    "source_anchor_gap_frames": gap,
                    "geo_A_to_B_m": distance_a_b,
                    "geo_B_to_C_m": float(distance_b_c),
                    "reference_tail_max_covis": maximum,
                    "reference_tail_argmax": int(np.argmax(curve)),
                    "reference_tail_frames": len(curve),
                    "reference_tail_curve": [float(value) for value in curve],
                    "reference_observation_cache_key": [frame_b, len(observations)],
                }
            )
    if not valid:
        raise RuntimeError("no route-negative online-A B/C source pair exists")
    valid.sort(
        key=lambda row: source_pair_score(
            row,
            contract["preferred_reference_path_c_tail_max_covis"],
            contract["target_leg_geodesic_m"],
        )
    )
    search_summary = {
        "candidate_frames": frames,
        "candidate_frame_count": len(frames),
        "valid_route_negative_pair_count": len(valid),
        "preferred_tail_pair_count": sum(
            row["reference_tail_max_covis"]
            <= contract["preferred_reference_path_c_tail_max_covis"]
            for row in valid
        ),
        "selection_rule": (
            "prefer tail<=preferred limit; then minimize summed absolute "
            "A->B/B->C deviation from target geodesic; then lower tail, "
            "larger temporal gap, and stable frame order"
        ),
    }
    return valid, search_summary


def candidate_goal_points(
    candidate: v1.PerturbationCandidate,
    camera_height: float,
) -> np.ndarray:
    camera = candidate.position + np.asarray([0.0, camera_height, 0.0])
    return v1.goal_world_points(candidate.depth, camera, candidate.yaw)


def select_v1_pair(
    simulator,
    history: dict,
    candidate_sets: dict[str, list[v1.PerturbationCandidate]],
    contract: dict,
) -> tuple[v1.PerturbationCandidate, v1.PerturbationCandidate, dict]:
    endpoint = history["floor_positions"][-1]
    endpoint_yaw = float(history["poses"][-1]["yaw"])
    camera_height = float(
        history["camera_positions"][0][1]
        - history["floor_positions"][0][1]
    )
    rejected_tail = 0
    rejected_geometry = 0
    for candidate_b in candidate_sets["B"]:
        distance_a_b, observations = reference_path_observations(
            simulator,
            endpoint,
            endpoint_yaw,
            candidate_b.position,
            camera_height=camera_height,
            sample_step_m=contract["reference_path_sample_step_m"],
        )
        if distance_a_b < contract["minimum_leg_geodesic_m"]:
            rejected_geometry += len(candidate_sets["C"])
            continue
        for candidate_c in candidate_sets["C"]:
            argmax_gap = abs(
                int(candidate_b.best_frame) - int(candidate_c.best_frame)
            )
            if argmax_gap < contract["minimum_anchor_gap_frames"]:
                rejected_geometry += 1
                continue
            ok, distance_b_c, _points = geodesic(
                simulator.pathfinder,
                candidate_b.position,
                candidate_c.position,
            )
            if (
                not ok
                or not np.isfinite(distance_b_c)
                or distance_b_c < contract["minimum_leg_geodesic_m"]
            ):
                rejected_geometry += 1
                continue
            curve = tail_curve(
                candidate_goal_points(candidate_c, camera_height), observations
            )
            maximum = float(np.max(curve))
            if maximum > contract["reference_path_c_tail_max_covis"]:
                rejected_tail += 1
                continue
            return candidate_b, candidate_c, {
                "geo_A_to_B_m": distance_a_b,
                "geo_B_to_C_m": float(distance_b_c),
                "anchor_argmax_gap_frames": argmax_gap,
                "reference_tail_max_covis": maximum,
                "reference_tail_argmax": int(np.argmax(curve)),
                "reference_tail_frames": len(curve),
                "reference_tail_curve": [float(value) for value in curve],
                "rejected_geometry_pairs_before_selection": rejected_geometry,
                "rejected_tail_pairs_before_selection": rejected_tail,
            }
    raise RuntimeError("no route-negative V1 B/C candidate pair exists")


def reference_tail_record(selection: dict, contract: dict) -> dict:
    return {
        "proxy": "Habitat shortest path with exact A-end view and path-facing samples",
        "sample_step_m": contract["reference_path_sample_step_m"],
        "maximum_allowed": contract["reference_path_c_tail_max_covis"],
        "maximum_covisibility": selection["reference_tail_max_covis"],
        "argmax_reference_frame": selection["reference_tail_argmax"],
        "frames": selection["reference_tail_frames"],
        "curve": selection["reference_tail_curve"],
        "closed_loop_recheck_required": True,
    }


def build_episode(
    episode_root: Path,
    destination: Path,
    contract: dict,
) -> dict:
    receipt = json.loads((episode_root / "receipt.json").read_text())
    history = v1.load_online_history(episode_root, receipt)
    history["minimum_eligible_frame"] = int(
        contract["minimum_eligible_online_frame"]
    )
    camera_height = float(receipt["camera_height_m"])
    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
    try:
        source_candidates, source_search = select_source_pair(
            simulator, history, contract
        )
        candidate_cache = {}

        def audited_candidates(frame: int):
            if frame not in candidate_cache:
                cheap = v1.enumerate_perturbations(
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
                candidate_cache[frame] = v1.fully_audit_candidates(
                    cheap,
                    history,
                    frame,
                    minimum_eligible_frame=contract[
                        "minimum_eligible_online_frame"
                    ],
                    maximum_argmax_gap=contract["v1_max_argmax_gap_frames"],
                    minimum_max_covis=contract[
                        "v1_min_max_online_a_covis"
                    ],
                    maximum_max_covis=contract[
                        "v1_max_max_online_a_covis"
                    ],
                    limit=32,
                )
            return candidate_cache[frame]

        source_selection = None
        candidate_b = None
        candidate_c = None
        v1_selection = None
        rejected_source_pairs = []
        for rank, candidate in enumerate(source_candidates, start=1):
            frame_b = int(candidate["B_source_frame"])
            frame_c = int(candidate["C_source_frame"])
            candidate_sets = {
                "B": audited_candidates(frame_b),
                "C": audited_candidates(frame_c),
            }
            if not candidate_sets["B"] or not candidate_sets["C"]:
                rejected_source_pairs.append({
                    "rank": rank,
                    "B_source_frame": frame_b,
                    "C_source_frame": frame_c,
                    "reason": "no_fully_audited_v1_goal_candidate",
                })
                continue
            try:
                selected_b, selected_c, selected_v1 = select_v1_pair(
                    simulator, history, candidate_sets, contract
                )
            except RuntimeError:
                rejected_source_pairs.append({
                    "rank": rank,
                    "B_source_frame": frame_b,
                    "C_source_frame": frame_c,
                    "reason": "no_joint_route_negative_v1_pair",
                })
                continue
            source_selection = {
                **source_search,
                **candidate,
                "joint_v0_v1_selection_rank": rank,
                "rejected_better_ranked_source_pairs": rejected_source_pairs,
                "audited_v1_source_frame_count": len(candidate_cache),
            }
            candidate_b, candidate_c, v1_selection = (
                selected_b, selected_c, selected_v1
            )
            break
        if source_selection is None:
            raise RuntimeError(
                "no source pair supports both route-negative V0 and V1"
            )
        source_frames = {
            "B": int(source_selection["B_source_frame"]),
            "C": int(source_selection["C_source_frame"]),
        }
        exact_records = {
            role: v1.exact_goal_record(
                history,
                frame,
                role,
                minimum_self_covis=contract["v0_min_self_covis"],
                minimum_eligible_frame=contract["minimum_eligible_online_frame"],
            )
            for role, frame in source_frames.items()
        }
    finally:
        simulator.close()

    v0_dir = destination / V0_NAME
    v1_dir = destination / V1_NAME
    v0_assets = {
        role: v1.copy_exact_goal(v0_dir, episode_root, frame, role)
        for role, frame in source_frames.items()
    }
    v1_candidates = {"B": candidate_b, "C": candidate_c}
    v1_assets = {
        role: v1.write_goal(
            v1_dir,
            role,
            v1_candidates[role].rgb,
            v1_candidates[role].depth,
        )
        for role in ("B", "C")
    }
    v1_records = {
        role: v1.perturbation_record(
            v1_candidates[role],
            history,
            source_frames[role],
            role,
            camera_height,
        )
        for role in ("B", "C")
    }
    for role in ("B", "C"):
        exact_source = episode_root / "rgb" / f"{source_frames[role]:06d}.jpg"
        if v0_assets[role]["rgb_sha256"] != v1.sha256_file(exact_source):
            raise RuntimeError(f"V0 {role} is not byte-identical to online history")
        if v1_assets[role]["rgb_sha256"] == v0_assets[role]["rgb_sha256"]:
            raise RuntimeError(f"V1 {role} unexpectedly equals exact V0 JPEG")

    variants = {
        V0_NAME: {
            "goals": exact_records,
            "assets": v0_assets,
            "leg_geodesics_m": {
                "A_to_B": source_selection["geo_A_to_B_m"],
                "B_to_C": source_selection["geo_B_to_C_m"],
            },
            "anchor_argmax_gap_frames": abs(
                int(exact_records["B"]["max_online_a_covis_frame"])
                - int(exact_records["C"]["max_online_a_covis_frame"])
            ),
            "reference_path_c_tail": reference_tail_record(
                source_selection, contract
            ),
        },
        V1_NAME: {
            "goals": v1_records,
            "assets": v1_assets,
            "leg_geodesics_m": {
                "A_to_B": v1_selection["geo_A_to_B_m"],
                "B_to_C": v1_selection["geo_B_to_C_m"],
            },
            "anchor_argmax_gap_frames": int(
                v1_selection["anchor_argmax_gap_frames"]
            ),
            "reference_path_c_tail": reference_tail_record(
                v1_selection, contract
            ),
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": receipt["scene"],
        "episode": receipt["episode"],
        "source_online_episode": str(episode_root.resolve()),
        "source_online_receipt_sha256": v1.sha256_file(
            episode_root / "receipt.json"
        ),
        "source_online_trace_sha256": v1.sha256_file(
            episode_root / "online_a_trace.json"
        ),
        "goal_a": {
            "path": str((episode_root / "goal_a.jpg").resolve()),
            "sha256": v1.sha256_file(episode_root / "goal_a.jpg"),
        },
        "online_a_steps": len(history["poses"]),
        "source_anchor_assignment": {
            "B_source_frame": source_frames["B"],
            "C_source_frame": source_frames["C"],
            "temporal_gap_frames": abs(
                source_frames["B"] - source_frames["C"]
            ),
            "selection": source_selection,
        },
        "variants": variants,
        "runtime_audits_still_required": [
            "shared online-A replay hash equality across variants and arms",
            "complete factual online-B rollout remains a Goal-C hard negative",
            "Goal-C candidate ceiling never exceeds the online-A boundary",
            "B and C success uses variant-specific positions",
        ],
    }
    benchmark_path = destination / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    payload["benchmark_sha256"] = v1.sha256_file(benchmark_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-eligible-frame", type=int, default=39)
    parser.add_argument("--source-anchor-end-margin", type=int, default=39)
    parser.add_argument("--source-anchor-stride", type=int, default=8)
    parser.add_argument("--min-anchor-gap", type=int, default=32)
    parser.add_argument("--min-leg-geodesic", type=float, default=2.0)
    parser.add_argument("--target-leg-geodesic", type=float, default=3.0)
    parser.add_argument("--reference-path-step", type=float, default=0.25)
    parser.add_argument("--reference-tail-max-covis", type=float, default=0.08)
    parser.add_argument("--preferred-reference-tail-max-covis", type=float, default=0.05)
    parser.add_argument("--v0-min-self-covis", type=float, default=0.95)
    parser.add_argument("--v1-min-translation", type=float, default=0.20)
    parser.add_argument("--v1-max-translation", type=float, default=0.50)
    parser.add_argument("--v1-min-yaw-deg", type=float, default=10.0)
    parser.add_argument("--v1-max-yaw-deg", type=float, default=25.0)
    parser.add_argument("--v1-min-source-covis", type=float, default=0.45)
    parser.add_argument("--v1-min-max-covis", type=float, default=0.50)
    parser.add_argument("--v1-max-max-covis", type=float, default=0.98)
    parser.add_argument("--v1-max-argmax-gap", type=int, default=20)
    parser.add_argument("--v1-min-pixel-mae", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    source_manifest = json.loads((args.online_root / "manifest.json").read_text())
    if source_manifest.get("schema_version") != "shared_online_a_materialized_v1":
        raise RuntimeError("unexpected online-A source schema")
    contract = {
        "minimum_eligible_online_frame": int(args.minimum_eligible_frame),
        "source_anchor_end_margin_frames": int(args.source_anchor_end_margin),
        "source_anchor_stride_frames": int(args.source_anchor_stride),
        "minimum_anchor_gap_frames": int(args.min_anchor_gap),
        "minimum_leg_geodesic_m": float(args.min_leg_geodesic),
        "target_leg_geodesic_m": float(args.target_leg_geodesic),
        "reference_path_sample_step_m": float(args.reference_path_step),
        "reference_path_c_tail_max_covis": float(args.reference_tail_max_covis),
        "preferred_reference_path_c_tail_max_covis": float(
            args.preferred_reference_tail_max_covis
        ),
        "v0_min_self_covis": float(args.v0_min_self_covis),
        "v1_min_translation_m": float(args.v1_min_translation),
        "v1_max_translation_m": float(args.v1_max_translation),
        "v1_min_yaw_delta_deg": float(args.v1_min_yaw_deg),
        "v1_max_yaw_delta_deg": float(args.v1_max_yaw_deg),
        "v1_min_source_frame_covis": float(args.v1_min_source_covis),
        "v1_min_max_online_a_covis": float(args.v1_min_max_covis),
        "v1_max_max_online_a_covis": float(args.v1_max_max_covis),
        "v1_max_argmax_gap_frames": int(args.v1_max_argmax_gap),
        "v1_min_pixel_mae": float(args.v1_min_pixel_mae),
    }
    if not (
        0.0
        <= contract["preferred_reference_path_c_tail_max_covis"]
        <= contract["reference_path_c_tail_max_covis"]
        <= 1.0
    ):
        raise ValueError("invalid reference-path co-visibility thresholds")
    if contract["source_anchor_stride_frames"] < 1:
        raise ValueError("source-anchor stride must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=args.out.name + ".tmp.", dir=args.out.parent)
    )
    episodes = []
    try:
        for source in source_manifest["episodes"]:
            scene = str(source["scene"])
            episode = str(source["episode"])
            source_episode = args.online_root / scene / episode
            destination = temporary / scene / episode
            destination.mkdir(parents=True)
            episodes.append(build_episode(source_episode, destination, contract))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "paired exact-frame and controlled-pose double-Revisit goals "
                "from genuine online A, with method-independent reference-path "
                "hard-negative filtering for Goal C"
            ),
            "source_online_root": str(args.online_root.resolve()),
            "source_online_manifest_sha256": v1.sha256_file(
                args.online_root / "manifest.json"
            ),
            "contract": contract,
            "episodes": episodes,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (temporary / "manifest.json.sha256").write_text(
            v1.sha256_file(manifest_path) + "  manifest.json\n"
        )
        temporary.replace(args.out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.out),
                "episodes": len(episodes),
                "scenes": [episode["scene"] for episode in episodes],
                "manifest_sha256": v1.sha256_file(args.out / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
