#!/usr/bin/env python3
"""Probe X-NavDP as an isolated PointGoal direction actuator.

The probe reuses the nine consumed Novel-A plan-0 states from the frozen NavDP
critic sweep.  For every state it issues the same eight 45-degree PointGoal
queries to the released X-NavDP point-goal actor, resetting the Torch RNG so
all directions receive identical diffusion noise.  It compares execution of
the oracle-nearest request with the existing mixed ImageGoal/PointGoal NavDP
artifact.

This is a privileged, failure-enriched architecture diagnostic.  It does not
test a deployable ImageGoal direction source, does not use the blind split, and
does not authorize replacing the frozen ImageGoal policy.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
import torch

from MemNavData.audit_navdp_critic_direction_sweep import (
    DEFAULT_DIRECTIONS_DEG,
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
from MemNavData.novel_a_bearing_gate import (
    critic_shadow_diagnostics,
    wrap_deg,
)
from MemNavData.xnavdp_checkpoint_audit import OFFICIAL_XNAVDP_COMMIT


OFFICIAL_POSTTRAIN_SHA256 = (
    "267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24")
BASE_NAVDP_SHA256 = (
    "3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947")
EMBODIMENTS = {"wheeled": 0, "humanoid": 1, "quadruped": 2}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wire_roundtrip(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the JPEG/PNG and color conversion used by both policy servers."""

    image = Image.open(io.BytesIO(jpg_bytes(rgb))).convert("RGB")
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    encoded_depth = Image.open(io.BytesIO(depth_png_bytes(depth))).convert("I")
    decoded_depth = np.asarray(encoded_depth, dtype=np.float32) / 10000.0
    return image_bgr, decoded_depth[..., None]


def _process_image(images: np.ndarray, image_size: int = 224) -> np.ndarray:
    outputs = []
    for image in np.asarray(images):
        height, width = image.shape[:2]
        scale = image_size / max(height, width)
        resized = cv2.resize(image, (-1, -1), fx=scale, fy=scale)
        pad_width = max((image_size - resized.shape[1]) // 2, 0)
        pad_height = max((image_size - resized.shape[0]) // 2, 0)
        padded = np.pad(
            resized,
            ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
            mode="constant", constant_values=0)
        outputs.append(
            cv2.resize(padded, (image_size, image_size)).astype(np.float32)
            / 255.0)
    return np.asarray(outputs)


def _process_depth(depths: np.ndarray, image_size: int = 224) -> np.ndarray:
    outputs = []
    for raw in np.asarray(depths).copy():
        raw[raw == np.inf] = 0
        height, width = raw.shape[:2]
        scale = image_size / max(height, width)
        resized = cv2.resize(raw, (-1, -1), fx=scale, fy=scale)
        pad_width = max((image_size - resized.shape[1]) // 2, 0)
        pad_height = max((image_size - resized.shape[0]) // 2, 0)
        padded = np.pad(
            resized,
            ((pad_height, pad_height), (pad_width, pad_width)),
            mode="constant", constant_values=0)
        output = cv2.resize(padded, (image_size, image_size))
        output[(output > 5.0) | (output < 0.1)] = 0
        outputs.append(output[..., None])
    return np.asarray(outputs)


def _cold_history(current: np.ndarray, memory_size: int = 8) -> np.ndarray:
    history = np.zeros(
        (current.shape[0], memory_size, *current.shape[1:]),
        dtype=np.float32)
    history[:, -1] = current
    return history


def _trajectory_heading_deg(trajectory: np.ndarray) -> float | None:
    planar = np.asarray(trajectory, dtype=np.float64)[:, :2]
    distances = np.linalg.norm(planar, axis=1)
    eligible = np.flatnonzero(distances >= 0.3)
    endpoint = planar[eligible[-1]] if eligible.size else planar[-1]
    if float(np.linalg.norm(endpoint)) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(endpoint[1], endpoint[0])))


def _exact_mcnemar_two_sided(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    lower = min(int(gains), int(losses))
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def summarize_states(
    states: list[dict[str, Any]],
    directions: list[dict[str, Any]],
    threshold_deg: float,
) -> dict[str, Any]:
    def hit(row: dict[str, Any], key: str) -> bool:
        value = row.get(key)
        return value is not None and float(value) <= threshold_deg

    xnav = [hit(row, "xnav_oracle_request_executed_error_deg")
            for row in states]
    base = [hit(row, "base_oracle_request_executed_error_deg")
            for row in states]
    gains = sum(a and not b for a, b in zip(xnav, base))
    losses = sum(b and not a for a, b in zip(xnav, base))
    fidelity = [hit(row, "selected_request_error_deg") for row in directions]
    candidate_fidelity = [
        hit(row, "best_candidate_request_error_deg") for row in directions]
    return {
        "states": len(states),
        "scene_clusters": len({row["scene"] for row in states}),
        "threshold_deg": float(threshold_deg),
        "xnav_oracle_request_hits": int(sum(xnav)),
        "base_mixed_oracle_request_hits": int(sum(base)),
        "xnav_vs_base_gains": int(gains),
        "xnav_vs_base_losses": int(losses),
        "xnav_vs_base_exact_mcnemar_p": _exact_mcnemar_two_sided(
            gains, losses),
        "xnav_request_execution_ceiling_hits": int(sum(
            hit(row, "xnav_request_execution_ceiling_error_deg")
            for row in states)),
        "xnav_candidate_execution_ceiling_hits": int(sum(
            hit(row, "xnav_candidate_execution_ceiling_error_deg")
            for row in states)),
        "xnav_cross_request_q_hits": int(sum(
            hit(row, "xnav_q_chosen_executed_error_deg") for row in states)),
        "selected_request_fidelity_hits": int(sum(fidelity)),
        "selected_request_fidelity_total": len(fidelity),
        "any_candidate_request_fidelity_hits": int(sum(candidate_fidelity)),
        "any_candidate_request_fidelity_total": len(candidate_fidelity),
    }


def _load_official_policy(args: argparse.Namespace):
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(args.official_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if commit != OFFICIAL_XNAVDP_COMMIT:
        raise RuntimeError(
            f"official source commit {commit} differs from frozen pin")
    eval_root = args.official_root / "baselines/x-navdp/eval"
    sys.path.insert(0, str(eval_root))
    from src.policy_network_embodiment import NavDP_Policy_Embodiment

    policy = NavDP_Policy_Embodiment(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        device=args.device,
    )
    try:
        state = torch.load(
            str(args.checkpoint), map_location="cpu",
            weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover
        state = torch.load(
            str(args.checkpoint), map_location="cpu", weights_only=True)
    incompatible = policy.load_state_dict(state, strict=False)
    required_prefixes = (
        "rgbd_encoder.", "point_encoder.", "decoder.", "decoder_ft.",
        "decoder_q.", "q1_heads.", "q2_heads.")
    required_missing = [
        key for key in incompatible.missing_keys
        if key.startswith(required_prefixes)]
    if required_missing:
        raise RuntimeError(
            f"X-NavDP eval model is missing required weights: {required_missing[:8]}")
    del state
    policy.to(args.device)
    policy.eval()
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    return policy, {
        "official_commit": commit,
        "missing_key_count": len(missing_keys),
        "missing_key_prefixes": sorted({
            key.split(".", 1)[0] for key in missing_keys}),
        "unexpected_key_count": len(unexpected_keys),
        "unexpected_key_prefixes": sorted({
            key.split(".", 1)[0] for key in unexpected_keys}),
    }


def _predict_request(
    policy,
    *,
    input_images: np.ndarray,
    input_depths: np.ndarray,
    direction_deg: float,
    radius_m: float,
    seed: int,
    sample_num: int,
    embodiment: int,
    device: str,
) -> dict[str, Any]:
    theta = math.radians(direction_deg)
    goal = np.asarray([[radius_m * math.cos(theta),
                        radius_m * math.sin(theta), 0.0]], dtype=np.float32)
    prev_action = torch.zeros((1, 24, 3), dtype=torch.float32, device=device)
    valid_segment_len = np.zeros(1, dtype=np.int32)
    guidance_factor = np.zeros((1, sample_num), dtype=np.float32)
    devices: Sequence[int] = []
    if str(device).startswith("cuda"):
        device_index = torch.device(device).index
        devices = [torch.cuda.current_device()
                   if device_index is None else device_index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        trajectories, values, _positive, _ = (
            policy.predict_pointgoal_action_with_guidance(
                goal,
                input_images,
                input_depths,
                sample_num=sample_num,
                valid_segment_len=valid_segment_len,
                prev_action=prev_action,
                start_index=0,
                end_index=23,
                guidance_factor=guidance_factor,
                guidance_step=5,
                prefix_attention_schedule="exp",
                embodiment=embodiment,
            ))
    candidates = np.asarray(trajectories[0], dtype=np.float64)
    scores = np.asarray(values[0], dtype=np.float64)
    selected_index = int(np.argmax(scores))
    return {
        "trajectory": candidates[selected_index].tolist(),
        "all_trajectory": candidates.tolist(),
        "all_values": scores.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd
    from MemNavData.generate_twoleg import geodesic, make_sim, render

    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("X-NavDP checkpoint differs from the frozen pin")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline_report = json.loads(
        args.baseline_report.read_text(encoding="utf-8"))
    if baseline_report["provenance"]["checkpoint_sha256"] != BASE_NAVDP_SHA256:
        raise RuntimeError("baseline artifact does not use the frozen NavDP model")
    baseline_states = _read_csv(args.baseline_states)
    baseline_directions = _read_csv(args.baseline_directions)
    if args.max_states is not None:
        baseline_states = baseline_states[:args.max_states]
    baseline_by_request = {
        (row["scene"], row["episode"], int(row["plan_index"]),
         round(float(row["request_direction_deg"]), 6)): row
        for row in baseline_directions
    }
    selected_scenes = manifest["selection"]["selected_scenes"]
    anchor_scenes = set(manifest["selection"]["anchor_scenes"])

    policy, load_report = _load_official_policy(args)
    state_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    current_scene = None
    sim = None
    try:
        for state_ref in baseline_states:
            scene = state_ref["scene"]
            episode = state_ref["episode"]
            plan_index = int(state_ref["plan_index"])
            if plan_index != 0:
                raise RuntimeError("cold-history probe is frozen to plan_index=0")
            if scene != current_scene:
                if sim is not None:
                    sim.close()
                scene_index = selected_scenes.index(scene)
                glb = args.asset_root / scene / f"{scene}.glb"
                navmesh = args.asset_root / scene / f"{scene}.navmesh"
                expected_asset = manifest["assets"][scene]
                if (not glb.is_file() or not navmesh.is_file()
                        or glb.stat().st_size != int(expected_asset["bytes"])
                        or sha256_file(glb) != expected_asset["sha256"]):
                    raise RuntimeError(f"frozen scene asset mismatch: {scene}")
                sim = make_sim(str(glb), str(navmesh))
                current_scene = scene
            else:
                scene_index = selected_scenes.index(scene)

            episode_root = (args.legacy_episode_root if scene in anchor_scenes
                            else args.expanded_episode_root)
            episode_dir = episode_root / scene / episode
            episode_item = next(
                row for row in manifest["episodes"][scene]
                if row["episode"] == episode)
            meta_path = episode_dir / "meta" / "gen_meta.json"
            parquet_path = (
                episode_dir / "data/chunk-000/episode_000000.parquet")
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
            plans = json.loads((
                result_root / "geometry_router" / f"{episode}_plans.json"
            ).read_text(encoding="utf-8"))
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
            current_image = _process_image(image_bgr[None])
            input_images = _cold_history(current_image, memory_size=8)
            input_depth = _process_depth(depth_wire[None])

            ok, _remaining, oracle_path = geodesic(
                sim.pathfinder, current, goal_floor)
            oracle_world = path_initial_bearing(oracle_path, current) if ok else None
            if oracle_world is None:
                raise RuntimeError(
                    f"invalid oracle bearing: {scene}/{episode}/{plan_index}")
            oracle_relative = wrap_deg(math.degrees(
                oracle_world - float(trace["yaw"])))
            if abs(wrap_deg(
                    oracle_relative - float(state_ref["oracle_relative_deg"]))) > 1e-4:
                raise RuntimeError("oracle bearing differs from baseline artifact")
            episode_seed = int(float(metrics[episode]["seed"]))
            seed = diffusion_plan_seed(episode_seed, 0, plan_index)

            probes = []
            all_candidate_oracle_errors = []
            for direction_index, direction_deg in enumerate(
                    args.directions_deg):
                response = _predict_request(
                    policy,
                    input_images=input_images,
                    input_depths=input_depth,
                    direction_deg=direction_deg,
                    radius_m=args.radius_m,
                    seed=seed,
                    sample_num=args.sample_num,
                    embodiment=EMBODIMENTS[args.embodiment],
                    device=args.device,
                )
                shadow = critic_shadow_diagnostics(
                    response, requested_heading_deg=direction_deg)
                selected_heading = shadow["selected_heading_deg"]
                executed_error = (
                    None if selected_heading is None else
                    abs(wrap_deg(selected_heading - oracle_relative)))
                request_error = abs(wrap_deg(direction_deg - oracle_relative))
                candidate_headings = [
                    _trajectory_heading_deg(candidate)
                    for candidate in np.asarray(response["all_trajectory"])]
                candidate_oracle_errors = [
                    abs(wrap_deg(heading - oracle_relative))
                    for heading in candidate_headings if heading is not None]
                all_candidate_oracle_errors.extend(candidate_oracle_errors)
                baseline = baseline_by_request[
                    (scene, episode, plan_index, round(direction_deg, 6))]
                row = {
                    "scene": scene,
                    "episode": episode,
                    "plan_index": plan_index,
                    "frame_index": frame_index,
                    "direction_index": direction_index,
                    "request_direction_deg": direction_deg,
                    "oracle_relative_deg": oracle_relative,
                    "request_error_deg": request_error,
                    "selected_heading_deg": selected_heading,
                    "selected_request_error_deg": shadow[
                        "selected_request_error_deg"],
                    "executed_error_deg": executed_error,
                    "best_candidate_request_error_deg": shadow[
                        "best_direction_error_deg"],
                    "best_candidate_q_rank": shadow[
                        "best_direction_critic_rank"],
                    "candidate_oracle_ceiling_error_deg": (
                        min(candidate_oracle_errors)
                        if candidate_oracle_errors else None),
                    "q_max": shadow["critic_max"],
                    "q_min": shadow["critic_min"],
                    "q_unique_4dp": shadow["critic_unique_4dp"],
                    "heading_resultant_r": shadow["heading_resultant_r"],
                    "base_selected_heading_deg": baseline[
                        "selected_heading_deg"],
                    "base_selected_request_error_deg": baseline[
                        "selected_request_error_deg"],
                    "base_executed_error_deg": baseline[
                        "executed_error_deg"],
                }
                probes.append(row)
                direction_rows.append(row)

            nearest = min(probes, key=lambda row: row["request_error_deg"])
            q_order = sorted(
                probes, key=lambda row: float(row["q_max"]), reverse=True)
            q_chosen = q_order[0]
            state = {
                "scene": scene,
                "episode": episode,
                "plan_index": plan_index,
                "frame_index": frame_index,
                "oracle_relative_deg": oracle_relative,
                "oracle_request_direction_deg": nearest[
                    "request_direction_deg"],
                "oracle_request_error_deg": nearest["request_error_deg"],
                "xnav_oracle_request_heading_deg": nearest[
                    "selected_heading_deg"],
                "xnav_oracle_request_fidelity_error_deg": nearest[
                    "selected_request_error_deg"],
                "xnav_oracle_request_executed_error_deg": nearest[
                    "executed_error_deg"],
                "base_oracle_request_heading_deg": nearest[
                    "base_selected_heading_deg"],
                "base_oracle_request_fidelity_error_deg": nearest[
                    "base_selected_request_error_deg"],
                "base_oracle_request_executed_error_deg": nearest[
                    "base_executed_error_deg"],
                "xnav_request_execution_ceiling_error_deg": min(
                    row["executed_error_deg"] for row in probes
                    if row["executed_error_deg"] is not None),
                "xnav_candidate_execution_ceiling_error_deg": (
                    min(all_candidate_oracle_errors)
                    if all_candidate_oracle_errors else None),
                "xnav_q_chosen_direction_deg": q_chosen[
                    "request_direction_deg"],
                "xnav_q_chosen_request_error_deg": q_chosen[
                    "request_error_deg"],
                "xnav_q_chosen_heading_deg": q_chosen[
                    "selected_heading_deg"],
                "xnav_q_chosen_executed_error_deg": q_chosen[
                    "executed_error_deg"],
                "xnav_q_margin": float(q_order[0]["q_max"])
                - float(q_order[1]["q_max"]),
            }
            state_rows.append(state)
            print(
                f"[{scene}/{episode}] oracle={oracle_relative:+.1f} "
                f"request={nearest['request_direction_deg']:+.0f} "
                f"xnav={state['xnav_oracle_request_executed_error_deg']:.1f} "
                f"base={float(state['base_oracle_request_executed_error_deg']):.1f} "
                f"q={state['xnav_q_chosen_executed_error_deg']:.1f}",
                flush=True,
            )
    finally:
        if sim is not None:
            sim.close()

    report = {
        "scope": (
            "privileged, failure-enriched, consumed-development plan-0 "
            "architecture diagnostic; X-NavDP receives pure PointGoal queries; "
            "deployment_approved=false"),
        "definitions": {
            "directions_deg": list(args.directions_deg),
            "radius_m": args.radius_m,
            "sample_num_per_request": args.sample_num,
            "embodiment": args.embodiment,
            "identical_torch_rng_per_direction": True,
            "history": "cold seven-zero-frame history plus current frame",
            "rtc_previous_trajectory_guidance": False,
            "cross_request_q_is_goal_relevance_score": False,
            "baseline": "frozen NavDP mixed ImageGoal/PointGoal resample",
        },
        "summary": summarize_states(
            state_rows, direction_rows, args.threshold_deg),
        "model_load": load_report,
        "provenance": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "official_root": str(args.official_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "baseline_report": str(args.baseline_report.resolve()),
            "baseline_report_sha256": sha256_file(args.baseline_report),
            "baseline_states_sha256": sha256_file(args.baseline_states),
            "baseline_directions_sha256": sha256_file(
                args.baseline_directions),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    _write_csv(args.out / "states.csv", state_rows)
    _write_csv(args.out / "directions.csv", direction_rows)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root", type=Path,
        default=Path(
            ".diagnostics/xnavdp_official_878740a2011856d0/NavDP"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            ".diagnostics/xnavdp_official_878740a2011856d0/"
            "x-navdp_posttrain.ckpt"))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=OFFICIAL_POSTTRAIN_SHA256)
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
    parser.add_argument(
        "--baseline-directions", type=Path,
        default=Path(
            ".diagnostics/navdp_critic_direction_sweep_plan0_20260809/"
            "directions.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embodiment", choices=tuple(EMBODIMENTS),
                        default="wheeled")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--sample-num", type=int, default=8)
    parser.add_argument("--radius-m", type=float, default=2.0)
    parser.add_argument("--threshold-deg", type=float, default=30.0)
    args = parser.parse_args()
    args.directions_deg = DEFAULT_DIRECTIONS_DEG
    if args.max_states is not None and args.max_states < 1:
        parser.error("max-states must be positive")
    if args.sample_num < 2 or args.radius_m <= 0 or args.threshold_deg <= 0:
        parser.error("sample-num >=2 and positive radius/threshold are required")
    if args.out.exists():
        parser.error("output directory already exists")
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
