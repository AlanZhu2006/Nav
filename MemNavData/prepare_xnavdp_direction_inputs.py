#!/usr/bin/env python3
"""Freeze wire-equivalent RGB-D inputs for the X-NavDP direction probe.

This Habitat-only stage reconstructs the consumed plan-0 states used by the
existing NavDP direction sweep, verifies their oracle bearings, and stores the
JPEG/PNG-round-tripped inputs.  It contains no Torch dependency so inference
can remain isolated in the NavDP environment.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from MemNavData.audit_navdp_critic_direction_sweep import (
    depth_png_bytes,
    jpg_bytes,
)
from MemNavData.audit_observed_frontier_bearing_coverage import (
    data_to_hab,
    matrix_from_nested,
    parquet_floor_pose,
    path_initial_bearing,
    sha256_file,
)
from MemNavData.deterministic_eval_protocol import diffusion_plan_seed
from MemNavData.novel_a_bearing_gate import wrap_deg


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _wire_roundtrip(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(io.BytesIO(jpg_bytes(rgb))).convert("RGB")
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    encoded_depth = Image.open(io.BytesIO(depth_png_bytes(depth))).convert("I")
    decoded_depth = np.asarray(encoded_depth, dtype=np.float32) / 10000.0
    return image_bgr, decoded_depth[..., None]


def prepare(args: argparse.Namespace) -> dict:
    import pandas as pd
    from MemNavData.generate_twoleg import geodesic, make_sim, render

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline_report = json.loads(
        args.baseline_report.read_text(encoding="utf-8"))
    baseline_states = _read_csv(args.baseline_states)
    if args.max_states is not None:
        baseline_states = baseline_states[:args.max_states]
    selected_scenes = manifest["selection"]["selected_scenes"]
    anchor_scenes = set(manifest["selection"]["anchor_scenes"])

    images = []
    depths = []
    records = []
    current_scene = None
    sim = None
    try:
        for state_ref in baseline_states:
            scene = state_ref["scene"]
            episode = state_ref["episode"]
            plan_index = int(state_ref["plan_index"])
            if plan_index != 0:
                raise RuntimeError("input pack is frozen to plan_index=0")
            scene_index = selected_scenes.index(scene)
            if scene != current_scene:
                if sim is not None:
                    sim.close()
                glb = args.asset_root / scene / f"{scene}.glb"
                navmesh = args.asset_root / scene / f"{scene}.navmesh"
                expected_asset = manifest["assets"][scene]
                if (not glb.is_file() or not navmesh.is_file()
                        or glb.stat().st_size != int(expected_asset["bytes"])
                        or sha256_file(glb) != expected_asset["sha256"]):
                    raise RuntimeError(f"frozen scene asset mismatch: {scene}")
                sim = make_sim(str(glb), str(navmesh))
                current_scene = scene

            episode_root = (args.legacy_episode_root if scene in anchor_scenes
                            else args.expanded_episode_root)
            episode_dir = episode_root / scene / episode
            episode_item = next(
                row for row in manifest["episodes"][scene]
                if row["episode"] == episode)
            meta_path = episode_dir / "meta/gen_meta.json"
            parquet_path = episode_dir / "data/chunk-000/episode_000000.parquet"
            for label, path in (("metadata", meta_path),
                                ("parquet", parquet_path)):
                expected = episode_item["files"][label]
                if (not path.is_file()
                        or path.stat().st_size != int(expected["bytes"])
                        or sha256_file(path) != expected["sha256"]):
                    raise RuntimeError(
                        f"frozen {label} mismatch: {scene}/{episode}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            parquet = pd.read_parquet(parquet_path)
            camera_height = float(meta.get("camera_height_m", 0.5))
            intrinsic = matrix_from_nested(
                parquet.iloc[0]["observation.camera_intrinsic"])
            initial_floor, _ = parquet_floor_pose(
                parquet.iloc[0]["action"], camera_height)
            goal_floor = data_to_hab(meta["A"])
            goal_floor[1] = initial_floor[1]
            result_root = args.plans_root / f"{scene_index:02d}_{scene}"
            metrics = {
                row["episode"]: row for row in _read_csv(
                    result_root / "navdp_native/metric.csv")}
            plans_path = (
                result_root / "geometry_router" / f"{episode}_plans.json")
            plans = json.loads(plans_path.read_text(encoding="utf-8"))
            frame_index = int(state_ref["frame_index"])
            trace = next(
                row for row in plans["legA_memory_trace"]
                if int(row["frame_idx"]) == frame_index)
            current = np.asarray([
                float(trace["x"]), initial_floor[1], float(trace["z"])
            ], dtype=np.float64)
            assert sim is not None
            rgb, depth = render(
                sim,
                current + np.asarray([0.0, camera_height, 0.0]),
                float(trace["yaw"]),
            )
            image_bgr, depth_wire = _wire_roundtrip(rgb, depth)
            ok, remaining, oracle_path = geodesic(
                sim.pathfinder, current, goal_floor)
            oracle_world = path_initial_bearing(oracle_path, current) if ok else None
            if oracle_world is None:
                raise RuntimeError(
                    f"invalid oracle bearing: {scene}/{episode}/{plan_index}")
            oracle_relative = wrap_deg(math.degrees(
                oracle_world - float(trace["yaw"])))
            baseline_oracle = float(state_ref["oracle_relative_deg"])
            if abs(wrap_deg(oracle_relative - baseline_oracle)) > 1e-4:
                raise RuntimeError("oracle bearing differs from baseline artifact")
            episode_seed = int(float(metrics[episode]["seed"]))
            images.append(image_bgr)
            depths.append(depth_wire)
            records.append({
                "scene": scene,
                "episode": episode,
                "plan_index": plan_index,
                "frame_index": frame_index,
                "intrinsic": intrinsic.tolist(),
                "oracle_relative_deg": oracle_relative,
                "goal_remaining_geodesic_m": float(remaining),
                "episode_seed": episode_seed,
                "diffusion_seed": diffusion_plan_seed(
                    episode_seed, 0, plan_index),
            })
            print(
                f"[{scene}/{episode}] oracle={oracle_relative:+.1f}",
                flush=True)
    finally:
        if sim is not None:
            sim.close()

    args.out.mkdir(parents=True, exist_ok=False)
    inputs_path = args.out / "inputs.npz"
    np.savez_compressed(
        inputs_path,
        images_bgr=np.stack(images).astype(np.uint8),
        depths_m=np.stack(depths).astype(np.float32),
    )
    states_path = args.out / "states.json"
    states_path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    report = {
        "scope": (
            "wire-equivalent RGB-D input freeze for consumed plan-0 "
            "architecture diagnostic; no inference or deployment claim"),
        "states": len(records),
        "scene_clusters": len({row["scene"] for row in records}),
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "baseline_report": str(args.baseline_report.resolve()),
            "baseline_report_sha256": sha256_file(args.baseline_report),
            "baseline_states": str(args.baseline_states.resolve()),
            "baseline_states_sha256": sha256_file(args.baseline_states),
            "baseline_checkpoint_sha256": baseline_report[
                "provenance"]["checkpoint_sha256"],
            "inputs_sha256": sha256_file(inputs_path),
            "states_sha256": sha256_file(states_path),
        },
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("MemNavData/expanded_navdp_router_eval_20260805.json"))
    parser.add_argument(
        "--plans-root", type=Path,
        default=Path(".diagnostics/twentyscene_local_20260808"))
    parser.add_argument(
        "--asset-root", type=Path,
        default=Path("/home/asus/Research/datasets/mp3d_20scene/assets"))
    parser.add_argument(
        "--legacy-episode-root", type=Path,
        default=Path(
            "/home/asus/Research/Nav-axis-uturn/.diagnostics/"
            "unseen_scene_eval_20260803/episodes"))
    parser.add_argument(
        "--expanded-episode-root", type=Path,
        default=Path("/home/asus/Research/datasets/mp3d_20scene/episodes"))
    parser.add_argument(
        "--baseline-report", type=Path,
        default=Path(
            ".diagnostics/navdp_critic_direction_sweep_plan0_20260809/"
            "report.json"))
    parser.add_argument(
        "--baseline-states", type=Path,
        default=Path(
            ".diagnostics/navdp_critic_direction_sweep_plan0_20260809/"
            "states.csv"))
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_states is not None and args.max_states < 1:
        parser.error("max-states must be positive")
    if args.out.exists():
        parser.error("output directory already exists")
    return args


def main() -> None:
    report = prepare(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
