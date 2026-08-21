#!/usr/bin/env python3
"""Build paired V0/V1 double-Revisit goals from audited online-A traces.

V0 uses two exact RGB/depth frames that frozen NavDP actually observed while
navigating Goal A.  V1 re-renders each goal from a small, navigable pose
perturbation around the same historical location.  V1 therefore preserves
strong 3D co-visibility without reducing the benchmark to JPEG identity.

This builder is deliberately evaluation-only.  It freezes goal assets and
audits their intrinsic relationship to the online-A history; it does not run
legs B/C and cannot certify whether the eventual online-B path exposes Goal C.
That contamination check belongs to the closed-loop evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from generate_twoleg import (
    backproject,
    cam_to_world_hab,
    covis_curve,
    covis_frac,
    geodesic,
    make_sim,
    render,
    to_world,
)


SCHEMA_VERSION = "shared_online_double_revisit_v1_20260812"
V0_NAME = "v0_exact_online_frame"
V1_NAME = "v1_controlled_pose_perturbation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=95
    )
    return buffer.getvalue()


def write_depth_png(path: Path, depth: np.ndarray) -> None:
    encoded = np.clip(
        np.asarray(depth, dtype=np.float64) * 10000.0, 0, 65535
    ).astype(np.uint16)
    Image.fromarray(encoded).save(path)


def read_depth_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 10000.0


def wrap_radians(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def angle_delta_degrees(first: float, second: float) -> float:
    return abs(float(np.degrees(wrap_radians(first - second))))


def pixel_mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(first, dtype=np.float32)
                - np.asarray(second, dtype=np.float32)
            )
        )
    )


def json_vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float)]


@dataclass
class PerturbationCandidate:
    position: np.ndarray
    yaw: float
    rgb: np.ndarray
    depth: np.ndarray
    translation_m: float
    yaw_delta_deg: float
    anchor_covis: float
    pixel_mae: float
    attempt: int
    curve: np.ndarray | None = None
    best_covis: float | None = None
    best_frame: int | None = None


def load_online_history(episode_root: Path, receipt: dict) -> dict:
    trace = json.loads((episode_root / "online_a_trace.json").read_text())
    poses = trace["poses"]
    if len(poses) != int(receipt["online_a_steps"]):
        raise RuntimeError("online trace length disagrees with receipt")
    expected_steps = list(range(len(poses)))
    actual_steps = [int(pose["step"]) for pose in poses]
    if actual_steps != expected_steps:
        raise RuntimeError("online-A trace frames must be contiguous from zero")

    camera_height = float(receipt["camera_height_m"])
    camera_positions = []
    floor_positions = []
    transforms = []
    depths = []
    rgbs = []
    for pose in poses:
        step = int(pose["step"])
        floor_position = np.asarray(
            [pose["x"], pose["y"], pose["z"]], dtype=float
        )
        camera_position = floor_position + np.asarray(
            [0.0, camera_height, 0.0], dtype=float
        )
        yaw = float(pose["yaw"])
        rgb_path = episode_root / "rgb" / f"{step:06d}.jpg"
        depth_path = episode_root / "depth" / f"{step:06d}.png"
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"missing online frame {step}")
        if sha256_file(rgb_path) != str(pose["jpg_sha256"]):
            raise RuntimeError(f"online RGB hash mismatch at frame {step}")
        floor_positions.append(floor_position)
        camera_positions.append(camera_position)
        transforms.append(cam_to_world_hab(camera_position, yaw))
        depths.append(read_depth_png(depth_path))
        rgbs.append(np.asarray(Image.open(rgb_path).convert("RGB")))
    return {
        "trace": trace,
        "poses": poses,
        "floor_positions": floor_positions,
        "camera_positions": camera_positions,
        "transforms": transforms,
        "depths": depths,
        "rgbs": rgbs,
    }


def goal_world_points(
    depth: np.ndarray, camera_position: np.ndarray, yaw: float
) -> np.ndarray:
    return to_world(
        backproject(depth, stride=6),
        cam_to_world_hab(camera_position, yaw),
    )


def exact_goal_record(
    history: dict,
    frame: int,
    role: str,
    *,
    minimum_self_covis: float,
    minimum_eligible_frame: int,
) -> dict:
    pose = history["poses"][frame]
    goal_points = goal_world_points(
        history["depths"][frame],
        history["camera_positions"][frame],
        float(pose["yaw"]),
    )
    curve = covis_curve(
        goal_points, history["transforms"], history["depths"]
    )
    if not 0 <= minimum_eligible_frame < len(curve):
        raise ValueError("minimum eligible frame is outside online history")
    global_best_frame = int(np.argmax(curve))
    best_frame = minimum_eligible_frame + int(
        np.argmax(curve[minimum_eligible_frame:])
    )
    if float(curve[frame]) < minimum_self_covis:
        raise RuntimeError(
            f"exact {role} frame self-covisibility {float(curve[frame]):.6f} "
            f"is below {minimum_self_covis:.6f}"
        )
    return {
        "role": role,
        "source_online_frame": int(frame),
        "source_online_step": int(pose["step"]),
        "floor_position": json_vector(history["floor_positions"][frame]),
        "camera_position": json_vector(history["camera_positions"][frame]),
        "yaw_rad": float(pose["yaw"]),
        "translation_from_source_m": 0.0,
        "yaw_delta_from_source_deg": 0.0,
        "pixel_mae_from_source": 0.0,
        "source_frame_covis": float(curve[frame]),
        "max_online_a_covis": float(curve[best_frame]),
        "max_online_a_covis_frame": best_frame,
        "eligible_online_a_frame_floor": int(minimum_eligible_frame),
        "global_max_online_a_covis": float(curve[global_best_frame]),
        "global_max_online_a_covis_frame": global_best_frame,
        "covis_curve": [float(value) for value in curve],
    }


def enumerate_perturbations(
    simulator,
    history: dict,
    frame: int,
    *,
    camera_height: float,
    min_translation_m: float,
    max_translation_m: float,
    min_yaw_delta_deg: float,
    max_yaw_delta_deg: float,
    min_anchor_covis: float,
    minimum_pixel_mae: float,
) -> list[PerturbationCandidate]:
    source_floor = history["floor_positions"][frame]
    source_yaw = float(history["poses"][frame]["yaw"])
    source_rgb = history["rgbs"][frame]
    source_depth = history["depths"][frame]
    source_transform = history["transforms"][frame]

    seed_material = (
        f"{history['trace']['source_scene']}/{history['trace']['episode']}/{frame}"
    ).encode()
    phase = (
        int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        / float(2**32)
        * 2.0
        * np.pi
    )
    radii = (0.22, 0.30, 0.38, 0.46)
    directions = [phase + index * np.pi / 8.0 for index in range(16)]
    yaw_offsets_deg = (12.0, -12.0, 18.0, -18.0, 24.0, -24.0)
    pathfinder = simulator.pathfinder
    candidates = []
    attempt = 0
    seen = set()
    for radius in radii:
        for direction_index, direction in enumerate(directions):
            raw = source_floor + np.asarray(
                [radius * np.cos(direction), 0.0, radius * np.sin(direction)]
            )
            snapped = np.asarray(pathfinder.snap_point(raw), dtype=float)
            if not pathfinder.is_navigable(snapped):
                continue
            if abs(float(snapped[1] - source_floor[1])) > 0.20:
                continue
            if float(np.linalg.norm(snapped[[0, 2]] - raw[[0, 2]])) > 0.20:
                continue
            translation = float(
                np.linalg.norm(snapped[[0, 2]] - source_floor[[0, 2]])
            )
            if not min_translation_m <= translation <= max_translation_m:
                continue
            for offset_index in range(len(yaw_offsets_deg)):
                offset = yaw_offsets_deg[
                    (direction_index + offset_index) % len(yaw_offsets_deg)
                ]
                yaw = wrap_radians(source_yaw + np.deg2rad(offset))
                yaw_delta = angle_delta_degrees(yaw, source_yaw)
                if not min_yaw_delta_deg <= yaw_delta <= max_yaw_delta_deg:
                    continue
                identity = (
                    round(float(snapped[0]), 4),
                    round(float(snapped[1]), 4),
                    round(float(snapped[2]), 4),
                    round(float(yaw), 4),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                attempt += 1
                camera_position = snapped + np.asarray(
                    [0.0, camera_height, 0.0]
                )
                rgb, depth = render(simulator, camera_position, yaw)
                points = goal_world_points(depth, camera_position, yaw)
                anchor_covis = covis_frac(
                    points, source_transform, source_depth
                )
                mae = pixel_mae(rgb, source_rgb)
                if anchor_covis < min_anchor_covis or mae < minimum_pixel_mae:
                    continue
                candidates.append(
                    PerturbationCandidate(
                        position=snapped,
                        yaw=yaw,
                        rgb=rgb,
                        depth=depth,
                        translation_m=translation,
                        yaw_delta_deg=yaw_delta,
                        anchor_covis=anchor_covis,
                        pixel_mae=mae,
                        attempt=attempt,
                    )
                )
    return sorted(
        candidates,
        key=lambda item: (
            abs(item.translation_m - 0.30),
            abs(item.yaw_delta_deg - 18.0),
            abs(item.anchor_covis - 0.70),
            item.attempt,
        ),
    )


def fully_audit_candidates(
    candidates: list[PerturbationCandidate],
    history: dict,
    source_frame: int,
    *,
    minimum_eligible_frame: int,
    maximum_argmax_gap: int,
    minimum_max_covis: float,
    maximum_max_covis: float,
    limit: int = 16,
) -> list[PerturbationCandidate]:
    audited = []
    for candidate in candidates[:limit]:
        camera_position = candidate.position + np.asarray(
            [0.0, history["camera_positions"][0][1]
             - history["floor_positions"][0][1], 0.0]
        )
        points = goal_world_points(
            candidate.depth, camera_position, candidate.yaw
        )
        curve = covis_curve(
            points, history["transforms"], history["depths"]
        )
        if not 0 <= minimum_eligible_frame < len(curve):
            raise ValueError("minimum eligible frame is outside online history")
        best_frame = minimum_eligible_frame + int(
            np.argmax(curve[minimum_eligible_frame:])
        )
        best_covis = float(curve[best_frame])
        if abs(best_frame - source_frame) > maximum_argmax_gap:
            continue
        if not minimum_max_covis <= best_covis <= maximum_max_covis:
            continue
        candidate.curve = curve
        candidate.best_covis = best_covis
        candidate.best_frame = best_frame
        audited.append(candidate)
    return audited


def perturbation_record(
    candidate: PerturbationCandidate,
    history: dict,
    source_frame: int,
    role: str,
    camera_height: float,
) -> dict:
    if (
        candidate.curve is None
        or candidate.best_covis is None
        or candidate.best_frame is None
    ):
        raise ValueError("candidate must be fully audited")
    return {
        "role": role,
        "source_online_frame": int(source_frame),
        "source_online_step": int(history["poses"][source_frame]["step"]),
        "floor_position": json_vector(candidate.position),
        "camera_position": json_vector(
            candidate.position + np.asarray([0.0, camera_height, 0.0])
        ),
        "yaw_rad": float(candidate.yaw),
        "translation_from_source_m": float(candidate.translation_m),
        "yaw_delta_from_source_deg": float(candidate.yaw_delta_deg),
        "pixel_mae_from_source": float(candidate.pixel_mae),
        "source_frame_covis": float(candidate.anchor_covis),
        "max_online_a_covis": float(candidate.best_covis),
        "max_online_a_covis_frame": int(candidate.best_frame),
        "eligible_online_a_frame_floor": int(
            history["minimum_eligible_frame"]
        ),
        "candidate_attempt": int(candidate.attempt),
        "covis_curve": [float(value) for value in candidate.curve],
    }


def goal_distance(pathfinder, first: np.ndarray, second: np.ndarray) -> float:
    ok, distance, _ = geodesic(pathfinder, first, second)
    if not ok or not np.isfinite(distance):
        raise RuntimeError("goal pair is not geodesically connected")
    return float(distance)


def write_goal(
    destination: Path,
    role: str,
    rgb: np.ndarray,
    depth: np.ndarray,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    rgb_path = destination / f"goal_{role.lower()}.jpg"
    depth_path = destination / f"goal_{role.lower()}_depth.png"
    rgb_path.write_bytes(jpeg_bytes(rgb))
    write_depth_png(depth_path, depth)
    return {
        "rgb": rgb_path.name,
        "rgb_sha256": sha256_file(rgb_path),
        "depth": depth_path.name,
        "depth_sha256": sha256_file(depth_path),
    }


def copy_exact_goal(
    destination: Path,
    episode_root: Path,
    frame: int,
    role: str,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    rgb_path = destination / f"goal_{role.lower()}.jpg"
    depth_path = destination / f"goal_{role.lower()}_depth.png"
    shutil.copy2(episode_root / "rgb" / f"{frame:06d}.jpg", rgb_path)
    shutil.copy2(episode_root / "depth" / f"{frame:06d}.png", depth_path)
    return {
        "rgb": rgb_path.name,
        "rgb_sha256": sha256_file(rgb_path),
        "depth": depth_path.name,
        "depth_sha256": sha256_file(depth_path),
    }


def build_episode(
    episode_root: Path,
    destination: Path,
    contract: dict,
) -> dict:
    receipt = json.loads((episode_root / "receipt.json").read_text())
    history = load_online_history(episode_root, receipt)
    history["minimum_eligible_frame"] = int(
        contract["minimum_eligible_online_frame"]
    )
    diagnostic = receipt["anchor_preselection_diagnostic"]
    early_frame = int(diagnostic["frame_0"])
    late_frame = int(diagnostic["frame_1"])
    if late_frame - early_frame < int(contract["minimum_anchor_gap_frames"]):
        raise RuntimeError("preselected anchors violate temporal gap")
    # B is the later memory so A-end -> B does not traverse the earlier C region.
    source_frames = {"B": late_frame, "C": early_frame}
    camera_height = float(receipt["camera_height_m"])

    simulator = make_sim(str(receipt["source_asset"]), "", agent_radius=0.30)
    try:
        exact_records = {
            role: exact_goal_record(
                history,
                frame,
                role,
                minimum_self_covis=contract["v0_min_self_covis"],
                minimum_eligible_frame=contract[
                    "minimum_eligible_online_frame"
                ],
            )
            for role, frame in source_frames.items()
        }
        candidate_sets = {}
        for role, frame in source_frames.items():
            cheap = enumerate_perturbations(
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
            candidate_sets[role] = fully_audit_candidates(
                cheap,
                history,
                frame,
                minimum_eligible_frame=contract[
                    "minimum_eligible_online_frame"
                ],
                maximum_argmax_gap=contract["v1_max_argmax_gap_frames"],
                minimum_max_covis=contract["v1_min_max_online_a_covis"],
                maximum_max_covis=contract["v1_max_max_online_a_covis"],
            )
            if not candidate_sets[role]:
                raise RuntimeError(
                    f"no V1 candidate passed for {receipt['scene']}/"
                    f"{receipt['episode']} role {role}"
                )

        endpoint = history["floor_positions"][-1]
        selected_pair = None
        for candidate_b in candidate_sets["B"]:
            for candidate_c in candidate_sets["C"]:
                argmax_gap = abs(
                    int(candidate_b.best_frame) - int(candidate_c.best_frame)
                )
                if argmax_gap < int(contract["minimum_anchor_gap_frames"]):
                    continue
                distance_a_b = goal_distance(
                    simulator.pathfinder, endpoint, candidate_b.position
                )
                distance_b_c = goal_distance(
                    simulator.pathfinder,
                    candidate_b.position,
                    candidate_c.position,
                )
                if (
                    distance_a_b >= contract["minimum_leg_geodesic_m"]
                    and distance_b_c >= contract["minimum_leg_geodesic_m"]
                ):
                    selected_pair = (
                        candidate_b,
                        candidate_c,
                        distance_a_b,
                        distance_b_c,
                        argmax_gap,
                    )
                    break
            if selected_pair is not None:
                break
        if selected_pair is None:
            raise RuntimeError("no jointly valid V1 B/C candidate pair")
        candidate_b, candidate_c, v1_a_b, v1_b_c, v1_argmax_gap = selected_pair

        v0_a_b = goal_distance(
            simulator.pathfinder,
            endpoint,
            history["floor_positions"][late_frame],
        )
        v0_b_c = goal_distance(
            simulator.pathfinder,
            history["floor_positions"][late_frame],
            history["floor_positions"][early_frame],
        )
        if min(v0_a_b, v0_b_c) < contract["minimum_leg_geodesic_m"]:
            raise RuntimeError("V0 exact goals violate leg-distance contract")
    finally:
        simulator.close()

    v0_dir = destination / V0_NAME
    v1_dir = destination / V1_NAME
    v0_assets = {
        role: copy_exact_goal(v0_dir, episode_root, frame, role)
        for role, frame in source_frames.items()
    }
    v1_candidates = {"B": candidate_b, "C": candidate_c}
    v1_assets = {
        role: write_goal(
            v1_dir,
            role,
            v1_candidates[role].rgb,
            v1_candidates[role].depth,
        )
        for role in ("B", "C")
    }
    v1_records = {
        role: perturbation_record(
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
        if v0_assets[role]["rgb_sha256"] != sha256_file(exact_source):
            raise RuntimeError(f"V0 {role} is not byte-identical to online history")
        if v1_assets[role]["rgb_sha256"] == v0_assets[role]["rgb_sha256"]:
            raise RuntimeError(f"V1 {role} unexpectedly equals exact V0 JPEG")

    variants = {
        V0_NAME: {
            "goals": exact_records,
            "assets": v0_assets,
            "leg_geodesics_m": {"A_to_B": v0_a_b, "B_to_C": v0_b_c},
            "anchor_argmax_gap_frames": abs(
                int(exact_records["B"]["max_online_a_covis_frame"])
                - int(exact_records["C"]["max_online_a_covis_frame"])
            ),
        },
        V1_NAME: {
            "goals": v1_records,
            "assets": v1_assets,
            "leg_geodesics_m": {"A_to_B": v1_a_b, "B_to_C": v1_b_c},
            "anchor_argmax_gap_frames": int(v1_argmax_gap),
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": receipt["scene"],
        "episode": receipt["episode"],
        "source_online_episode": str(episode_root.resolve()),
        "source_online_receipt_sha256": sha256_file(
            episode_root / "receipt.json"
        ),
        "source_online_trace_sha256": sha256_file(
            episode_root / "online_a_trace.json"
        ),
        "goal_a": {
            "path": str((episode_root / "goal_a.jpg").resolve()),
            "sha256": sha256_file(episode_root / "goal_a.jpg"),
        },
        "online_a_steps": len(history["poses"]),
        "source_anchor_assignment": {
            "B_later_frame": late_frame,
            "C_earlier_frame": early_frame,
            "temporal_gap_frames": late_frame - early_frame,
            "reason": (
                "B uses the later online-A memory; C uses the earlier memory "
                "to reduce A-end-to-B traversal through the C region"
            ),
        },
        "variants": variants,
        "runtime_audits_still_required": [
            "shared online-A replay hash equality across V0/V1 and policy arms",
            "online-B trajectory does not expose enough Goal-C visual support",
            "B and C success measured against their variant-specific positions",
        ],
    }
    metadata_path = destination / "benchmark.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    payload["benchmark_sha256"] = sha256_file(metadata_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-anchor-gap", type=int, default=32)
    parser.add_argument("--minimum-eligible-frame", type=int, default=39)
    parser.add_argument("--min-leg-geodesic", type=float, default=2.0)
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
        "minimum_anchor_gap_frames": int(args.min_anchor_gap),
        "minimum_eligible_online_frame": int(args.minimum_eligible_frame),
        "minimum_leg_geodesic_m": float(args.min_leg_geodesic),
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
    if not 0.0 < contract["v1_min_max_online_a_covis"]:
        raise ValueError("minimum co-visibility must be positive")
    if not (
        contract["v1_min_translation_m"]
        < contract["v1_max_translation_m"]
    ):
        raise ValueError("invalid translation range")

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
            episodes.append(
                build_episode(source_episode, destination, contract)
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "paired exact-frame and controlled-pose double-Revisit "
                "goals derived exclusively from audited online NavDP Goal-A history"
            ),
            "source_online_root": str(args.online_root.resolve()),
            "source_online_manifest_sha256": sha256_file(
                args.online_root / "manifest.json"
            ),
            "contract": contract,
            "episodes": episodes,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        (temporary / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        temporary.replace(args.out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    summary = {
        "output": str(args.out),
        "episodes": len(episodes),
        "scenes": [episode["scene"] for episode in episodes],
        "manifest_sha256": sha256_file(args.out / "manifest.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
