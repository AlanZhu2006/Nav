#!/usr/bin/env python3
"""Audit generated 3-leg data without running a navigation policy."""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from generate_twoleg import geodesic, make_sim, render
from multigoal_benchmark_contract import (
    DOUBLE_REVISIT_PROTOCOL,
    DoubleRevisitObservation,
    RoleSymmetryObservation,
    validate_double_revisit_contract,
    validate_role_symmetric_contract,
)


M_W = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def data_to_hab(position) -> np.ndarray:
    return M_W.T @ np.asarray(position, dtype=float)


def parquet_pose_hab(action, camera_height: float) -> tuple[np.ndarray, float]:
    transform = np.stack([
        np.asarray(row, dtype=np.float64) for row in action
    ]).reshape(4, 4)
    camera_position = M_W.T @ transform[:3, 3]
    rotation = M_W.T @ transform[:3, :3]
    yaw = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
    floor_position = camera_position - np.asarray(
        [0.0, float(camera_height), 0.0])
    return floor_position, yaw


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_file_sha(left: str, right: str) -> bool:
    return (os.path.isfile(left) and os.path.isfile(right)
            and sha256(left) == sha256(right))


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def goal_matches_render(
    episode_dir: str,
    filename: str,
    goal: dict,
    sim,
    camera_height: float,
) -> bool:
    try:
        floor = data_to_hab(goal["pos"])
        rgb, _ = render(
            sim,
            floor + np.asarray([0.0, camera_height, 0.0]),
            float(goal["yaw_habitat"]),
        )
        return read_bytes(os.path.join(episode_dir, filename)) == jpeg_bytes(rgb)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def audit_episode(episode_dir: str, sim) -> dict:
    meta_path = os.path.join(episode_dir, "meta", "gen_meta.json")
    with open(meta_path) as handle:
        metadata = json.load(handle)
    rows = pd.read_parquet(os.path.join(
        episode_dir, "data", "chunk-000", "episode_000000.parquet"))
    switch_a, switch_b = [int(value) for value in metadata["switches"]]
    camera_height = float(metadata.get("camera_height_m", 0.5))
    first_stored, _ = parquet_pose_hab(rows.iloc[0]["action"], camera_height)
    a_terminal, _ = parquet_pose_hab(
        rows.iloc[switch_a - 1]["action"], camera_height)
    b_terminal, b_yaw = parquet_pose_hab(
        rows.iloc[switch_b - 1]["action"], camera_height)
    a_target = data_to_hab(metadata["A"])
    initial_target = data_to_hab(metadata["start"])
    goal_b, goal_c = metadata["goals"]
    b_target = data_to_hab(goal_b["pos"])
    c_target = data_to_hab(goal_c["pos"])
    rgb_root = os.path.join(
        episode_dir, "videos", "chunk-000", "observation.images.rgb")
    terminal_rgb = os.path.join(rgb_root, f"{switch_b - 1}.jpg")
    goal_rgb = os.path.join(episode_dir, "goal_1.jpg")
    ok_a, geo_a, _ = geodesic(sim.pathfinder, initial_target, a_target)
    ok_b, geo_b, _ = geodesic(sim.pathfinder, a_target, b_target)
    ok_c, geo_c, _ = geodesic(sim.pathfinder, b_target, c_target)
    goal_b_render = goal_matches_render(
        episode_dir, "goal_1.jpg", goal_b, sim, camera_height)
    goal_c_render = goal_matches_render(
        episode_dir, "goal_2.jpg", goal_c, sim, camera_height)
    if metadata.get("gen_protocol") == DOUBLE_REVISIT_PROTOCOL:
        observation = DoubleRevisitObservation(
            geo_a_m=float(geo_a) if ok_a else float("nan"),
            geo_b_m=float(geo_b) if ok_b else float("nan"),
            geo_c_m=float(geo_c) if ok_c else float("nan"),
            initial_pose_error_m=float(np.linalg.norm(
                first_stored - initial_target)),
            a_terminal_pose_error_m=float(np.linalg.norm(a_terminal - a_target)),
            goal_b_matches_render=goal_b_render,
            goal_c_matches_render=goal_c_render,
        )
        report = validate_double_revisit_contract(metadata, observation)
    else:
        observation = RoleSymmetryObservation(
            geo_a_m=float(geo_a) if ok_a else float("nan"),
            geo_b_m=float(geo_b) if ok_b else float("nan"),
            initial_pose_error_m=float(np.linalg.norm(
                first_stored - initial_target)),
            a_terminal_pose_error_m=float(np.linalg.norm(a_terminal - a_target)),
            b_terminal_pose_error_m=float(np.linalg.norm(b_terminal - b_target)),
            b_terminal_yaw_error_deg=abs(wrap_degrees(np.degrees(
                b_yaw - float(goal_b["yaw_habitat"])))),
            goal_b_matches_terminal_rgb=same_file_sha(goal_rgb, terminal_rgb),
        )
        report = validate_role_symmetric_contract(metadata, observation)
    layout_issues: list[str] = []
    n_frames = int(metadata.get("n_frames", -1))
    expected_names = {str(index) for index in range(max(0, n_frames))}
    rgb_names = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(rgb_root, "*.jpg"))}
    depth_root = os.path.join(
        episode_dir, "videos", "chunk-000", "observation.images.depth")
    depth_names = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(depth_root, "*.png"))}
    if len(rows) != n_frames:
        layout_issues.append("Parquet row count differs from n_frames")
    if rgb_names != expected_names:
        layout_issues.append("RGB frame indices are incomplete or contain stale files")
    if depth_names != expected_names:
        layout_issues.append("depth frame indices are incomplete or contain stale files")
    goal_alias = os.path.join(episode_dir, "goal_image.jpg")
    if not same_file_sha(goal_alias, goal_rgb):
        layout_issues.append("goal_image.jpg is not an exact alias of goal_1.jpg")

    if not goal_c_render:
        layout_issues.append("goal_2.jpg does not match the rendered Goal C pose")
    if (metadata.get("gen_protocol") == DOUBLE_REVISIT_PROTOCOL
            and not goal_b_render):
        layout_issues.append(
            "goal_1.jpg does not match the rendered Revisit-B pose")

    if layout_issues:
        report["issues"].extend(layout_issues)
        report["ok"] = False
    return {
        "episode": os.path.basename(episode_dir),
        "scene": metadata.get("scene"),
        "generation_protocol": metadata.get("gen_protocol"),
        "start_heading_offset_deg": metadata.get("start_heading_offset_deg"),
        "goal_b_matches_render": goal_b_render,
        "goal_c_matches_render": goal_c_render,
        "geodesic_source": "recomputed_habitat_pathfinder",
        **report,
    }
def aggregate(records: list[dict]) -> dict:
    offsets = np.radians(np.asarray([
        float(record["start_heading_offset_deg"])
        for record in records
        if record.get("start_heading_offset_deg") is not None
    ], dtype=float))
    geo_a = [record["observation"]["geo_a_m"] for record in records]
    geo_b = [record["observation"]["geo_b_m"] for record in records]
    geo_c = [
        record["observation"]["geo_c_m"] for record in records
        if "geo_c_m" in record["observation"]
    ]
    result = {
        "episodes": len(records),
        "valid_episodes": sum(bool(record["ok"]) for record in records),
        "all_valid": bool(records) and all(record["ok"] for record in records),
        "invalid": [
            {"episode": record["episode"], "issues": record["issues"]}
            for record in records if not record["ok"]
        ],
        "geo_A_m": {
            "min": float(np.min(geo_a)), "mean": float(np.mean(geo_a)),
            "max": float(np.max(geo_a)),
        },
        "geo_B_m": {
            "min": float(np.min(geo_b)), "mean": float(np.mean(geo_b)),
            "max": float(np.max(geo_b)),
        },
        "protocols": sorted({
            str(record.get("generation_protocol")) for record in records
        }),
        "initial_heading": {
            "count": int(len(offsets)),
            "mean_abs_deg": (
                float(np.mean(np.abs(np.degrees(offsets))))
                if len(offsets) else None),
            "resultant_R": (
                float(abs(np.mean(np.exp(1j * offsets))))
                if len(offsets) else None),
            "offsets_deg": np.degrees(offsets).tolist(),
        },
        "goal_B_exact_rgb_matches": sum(
            bool(record["observation"].get("goal_b_matches_terminal_rgb"))
            for record in records),
        "goal_B_render_matches": sum(
            bool(record.get("goal_b_matches_render"))
            for record in records),
        "goal_C_render_matches": sum(
            bool(record.get("goal_c_matches_render"))
            for record in records),
    }
    if geo_c:
        result["geo_C_m"] = {
            "min": float(np.min(geo_c)), "mean": float(np.mean(geo_c)),
            "max": float(np.max(geo_c)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--navmesh", default="")
    parser.add_argument("--agent_radius", type=float, default=0.30)
    parser.add_argument("--out")
    args = parser.parse_args()
    episode_dirs = sorted(
        path for path in glob.glob(os.path.join(args.episode_root, "episode_*"))
        if os.path.isfile(os.path.join(path, "meta", "gen_meta.json")))
    if not episode_dirs:
        raise SystemExit("no generated episodes found")
    sim = make_sim(
        args.scene, args.navmesh, agent_radius=float(args.agent_radius))
    try:
        records = [audit_episode(path, sim) for path in episode_dirs]
    finally:
        sim.close()
    result = {"summary": aggregate(records), "episodes": records}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    if not result["summary"]["all_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
