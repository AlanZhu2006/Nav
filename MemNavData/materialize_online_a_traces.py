#!/usr/bin/env python3
"""Materialize audited online NavDP Goal-A rollouts for Revisit benchmarks.

The existing paired closed-loop runs store a canonical pose/hash trace for the
actual NavDP Goal-A rollout.  This tool turns those traces into a standalone
RGB/depth asset.  Every RGB frame is re-rendered in Habitat and must reproduce
the hash recorded during the original online rollout; a mismatch fails closed.

The output deliberately contains no Goal-B/Goal-C definition.  Later V0/V1
builders may select targets only from these verified online-A frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from deterministic_eval_protocol import validate_leg1_trace
from generate_twoleg import H, HFOV_DEG, W, make_sim, render


SCHEMA_VERSION = "shared_online_a_materialized_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def write_depth_png(path: Path, depth: np.ndarray) -> None:
    depth_u16 = np.clip(
        np.asarray(depth, dtype=np.float64) * 10000.0, 0, 65535
    ).astype(np.uint16)
    Image.fromarray(depth_u16).save(path)


@dataclass(frozen=True)
class TraceCandidate:
    path: Path
    payload: dict
    score_m: float
    anchor_0: int
    anchor_1: int
    distance_0_to_end_m: float
    distance_1_to_end_m: float
    anchor_pair_distance_m: float

    @property
    def scene(self) -> str:
        return str(self.payload["source_scene"])

    @property
    def episode(self) -> str:
        return str(self.payload["episode"])


@dataclass(frozen=True)
class SingleAnchorTraceCandidate:
    """A successful history with at least one runtime-eligible anchor frame."""

    path: Path
    payload: dict
    score_m: float
    anchor: int
    distance_to_end_m: float

    @property
    def scene(self) -> str:
        return str(self.payload["source_scene"])

    @property
    def episode(self) -> str:
        return str(self.payload["episode"])


def native_control_audit(payload: dict) -> dict:
    """Prove that no memory/router intervention controlled the online A leg."""
    plans = payload["plans"]
    active_steps = [
        int(plan["step"]) for plan in plans if plan.get("router_active") is True
    ]
    takeover_steps = [
        int(plan["step"])
        for plan in plans
        if plan.get("revisit_adapter_takeover") is True
    ]
    forced_anchor_steps = [
        int(plan["step"])
        for plan in plans
        if plan.get("forced_anchor") is not None
    ]
    controllers = sorted(
        {str(plan.get("pose_controller")) for plan in plans}
    )
    ok = not active_steps and not takeover_steps and not forced_anchor_steps
    return {
        "ok": ok,
        "plan_count": len(plans),
        "router_active_steps": active_steps,
        "adapter_takeover_steps": takeover_steps,
        "forced_anchor_steps": forced_anchor_steps,
        "reported_pose_controllers": controllers,
        "interpretation": (
            "router wrapper observed the rollout, but frozen native NavDP "
            "selected every executed Goal-A trajectory"
        ),
    }


def best_separated_pair(
    poses: list[dict],
    *,
    margin: int,
    min_gap: int,
    end_margin: int | None = None,
) -> tuple[float, int, int, float, float, float] | None:
    """Find two well-separated interior frames using deployment-visible poses.

    The score is the minimum of the two distances to the final A pose and the
    inter-anchor distance.  It is only a preselection diagnostic, not a frozen
    B/C definition; the eventual V0/V1 builder remains responsible for its
    full visual/geodesic contract.
    """
    if end_margin is None:
        end_margin = margin
    if margin < 0 or end_margin < 0:
        raise ValueError("anchor margins must be non-negative")
    points = np.asarray([[pose["x"], pose["z"]] for pose in poses], dtype=float)
    count = len(points)
    if count < margin + end_margin + min_gap + 1:
        return None
    endpoint = points[-1]
    best = None
    for first in range(margin, count - end_margin - min_gap):
        for second in range(first + min_gap, count - end_margin):
            d0 = float(np.linalg.norm(points[first] - endpoint))
            d1 = float(np.linalg.norm(points[second] - endpoint))
            pair = float(np.linalg.norm(points[first] - points[second]))
            score = min(d0, d1, pair)
            row = (score, first, second, d0, d1, pair)
            if best is None or row > best:
                best = row
    return best


def discover_candidates(
    trace_root: Path,
    *,
    margin: int,
    min_gap: int,
    min_score_m: float,
    end_margin: int | None = None,
) -> list[TraceCandidate]:
    candidates = []
    for path in sorted(trace_root.rglob("*_leg1_trace.json")):
        payload = json.loads(path.read_text())
        validate_leg1_trace(payload)
        if not payload["reached"]:
            continue
        if not native_control_audit(payload)["ok"]:
            continue
        pair = best_separated_pair(
            payload["poses"],
            margin=margin,
            min_gap=min_gap,
            end_margin=end_margin,
        )
        if pair is None or pair[0] < min_score_m:
            continue
        candidates.append(
            TraceCandidate(path, payload, *pair)
        )
    return sorted(
        candidates,
        key=lambda row: (-row.score_m, row.scene, row.episode, str(row.path)),
    )


def discover_single_anchor_candidates(
    trace_root: Path,
    *,
    minimum_frame: int,
    end_margin: int,
) -> list[SingleAnchorTraceCandidate]:
    """Find histories usable by a one-query Revisit benchmark.

    This deliberately does not require a second historical target or an
    inter-anchor temporal/geodesic transition.  The returned anchor is only a
    materialization diagnostic; the downstream single-Revisit builder searches
    all runtime-eligible frames under its frozen visual contract.
    """
    if minimum_frame < 0 or end_margin < 0:
        raise ValueError("anchor margins must be non-negative")
    candidates = []
    for path in sorted(trace_root.rglob("*_leg1_trace.json")):
        payload = json.loads(path.read_text())
        validate_leg1_trace(payload)
        if not payload["reached"] or not native_control_audit(payload)["ok"]:
            continue
        poses = payload["poses"]
        stop = len(poses) - end_margin
        if minimum_frame >= stop:
            continue
        points = np.asarray(
            [[pose["x"], pose["z"]] for pose in poses], dtype=float
        )
        endpoint = points[-1]
        ranked = sorted(
            (
                -float(np.linalg.norm(points[frame] - endpoint)),
                frame,
            )
            for frame in range(minimum_frame, stop)
        )
        negative_distance, anchor = ranked[0]
        distance = -negative_distance
        candidates.append(
            SingleAnchorTraceCandidate(
                path=path,
                payload=payload,
                score_m=distance,
                anchor=anchor,
                distance_to_end_m=distance,
            )
        )
    return sorted(
        candidates,
        key=lambda row: (-row.score_m, row.scene, row.episode, str(row.path)),
    )


def select_distinct_scenes(
    candidates: list[TraceCandidate], count: int
) -> list[TraceCandidate]:
    selected = []
    scenes = set()
    for candidate in candidates:
        if candidate.scene in scenes:
            continue
        selected.append(candidate)
        scenes.add(candidate.scene)
        if len(selected) == count:
            return selected
    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == count:
            return selected
    return selected


def materialize_one(
    candidate: TraceCandidate | SingleAnchorTraceCandidate,
    *,
    asset_root: Path,
    episode_root: Path,
    destination: Path,
    asset_map: dict[str, Path] | None = None,
) -> dict:
    scene = candidate.scene
    episode = candidate.episode
    asset = (
        asset_map[scene]
        if asset_map is not None and scene in asset_map
        else asset_root / scene / f"{scene}.glb"
    )
    source_episode = episode_root / scene / episode
    source_metadata = source_episode / "meta" / "gen_meta.json"
    source_parquet = (
        source_episode / "data" / "chunk-000" / "episode_000000.parquet"
    )
    for required in (asset, source_metadata, source_parquet):
        if not required.is_file():
            raise FileNotFoundError(required)

    metadata = json.loads(source_metadata.read_text())
    switch = int(metadata["switch_idx"])
    source_goal_a = (
        source_episode
        / "videos"
        / "chunk-000"
        / "observation.images.rgb"
        / f"{switch - 1}.jpg"
    )
    if not source_goal_a.is_file():
        raise FileNotFoundError(source_goal_a)
    if sha256_file(source_goal_a) != candidate.payload["goal_sha256"]:
        raise RuntimeError(f"Goal-A hash mismatch for {scene}/{episode}")

    camera_height = float(metadata.get("camera_height_m", 0.5))
    rgb_root = destination / "rgb"
    depth_root = destination / "depth"
    rgb_root.mkdir(parents=True)
    depth_root.mkdir(parents=True)

    simulator = make_sim(str(asset), "", agent_radius=0.30)
    rendered_hashes = []
    try:
        for pose in candidate.payload["poses"]:
            step = int(pose["step"])
            floor_position = np.asarray(
                [pose["x"], pose["y"], pose["z"]], dtype=float
            )
            rgb, depth = render(
                simulator,
                floor_position + np.asarray([0.0, camera_height, 0.0]),
                float(pose["yaw"]),
            )
            encoded = jpeg_bytes(rgb)
            rendered_sha = hashlib.sha256(encoded).hexdigest()
            if rendered_sha != pose["jpg_sha256"]:
                raise RuntimeError(
                    f"online RGB mismatch at {scene}/{episode} step {step}: "
                    f"expected {pose['jpg_sha256']}, got {rendered_sha}"
                )
            (rgb_root / f"{step:06d}.jpg").write_bytes(encoded)
            write_depth_png(depth_root / f"{step:06d}.png", depth)
            rendered_hashes.append(rendered_sha)
    finally:
        simulator.close()

    shutil.copy2(candidate.path, destination / "online_a_trace.json")
    shutil.copy2(source_goal_a, destination / "goal_a.jpg")
    if isinstance(candidate, SingleAnchorTraceCandidate):
        anchor_diagnostic = {
            "not_a_frozen_goal_definition": True,
            "preselection_mode": "single_runtime_eligible_anchor",
            "frame": candidate.anchor,
            "distance_to_online_a_end_m": candidate.distance_to_end_m,
            "max_distance_score_m": candidate.score_m,
        }
    else:
        anchor_diagnostic = {
            "not_a_frozen_goal_definition": True,
            "preselection_mode": "two_anchor_coverage",
            "frame_0": candidate.anchor_0,
            "frame_1": candidate.anchor_1,
            "temporal_gap_frames": candidate.anchor_1 - candidate.anchor_0,
            "distance_0_to_online_a_end_m": candidate.distance_0_to_end_m,
            "distance_1_to_online_a_end_m": candidate.distance_1_to_end_m,
            "anchor_pair_distance_m": candidate.anchor_pair_distance_m,
            "min_distance_score_m": candidate.score_m,
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "episode": episode,
        "online_a_reached": True,
        "online_a_steps": len(candidate.payload["poses"]),
        "online_a_trace_sha256": sha256_file(candidate.path),
        "online_a_control_audit": native_control_audit(candidate.payload),
        "source_asset": str(asset.resolve()),
        "source_asset_sha256": sha256_file(asset),
        "source_episode": str(source_episode.resolve()),
        "source_metadata_sha256": sha256_file(source_metadata),
        "source_parquet_sha256": sha256_file(source_parquet),
        "goal_a_sha256": sha256_file(source_goal_a),
        "camera_height_m": camera_height,
        "render_contract": {
            "width": W,
            "height": H,
            "horizontal_fov_deg": HFOV_DEG,
            "jpeg_quality": 95,
            "depth_encoding": "uint16_metres_times_10000",
            "all_rgb_hashes_match_original_online_rollout": True,
        },
        "anchor_preselection_diagnostic": anchor_diagnostic,
        "rgb_frame_hashes": rendered_hashes,
    }
    receipt_path = destination / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument(
        "--asset-map-json",
        type=Path,
        help=("optional JSON object mapping stable scene labels to explicit "
              "Habitat stage/asset paths for cross-dataset evaluation"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--anchor-margin", type=int, default=16)
    parser.add_argument("--min-anchor-gap", type=int, default=32)
    parser.add_argument("--min-anchor-score-m", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.anchor_margin < 0 or args.min_anchor_gap < 1:
        raise ValueError("invalid anchor separation contract")
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    asset_map = None
    if args.asset_map_json is not None:
        raw_map = json.loads(args.asset_map_json.read_text())
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("asset map must be a non-empty JSON object")
        asset_map = {str(scene): Path(path).resolve()
                     for scene, path in raw_map.items()}
        for scene, path in asset_map.items():
            if not scene or not path.is_file():
                raise FileNotFoundError(f"invalid asset mapping {scene}: {path}")

    candidates = discover_candidates(
        args.trace_root,
        margin=args.anchor_margin,
        min_gap=args.min_anchor_gap,
        min_score_m=args.min_anchor_score_m,
    )
    selected = select_distinct_scenes(candidates, args.count)
    if len(selected) != args.count:
        raise RuntimeError(
            f"only {len(selected)} eligible online-A traces for requested "
            f"count {args.count}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=args.out.name + ".tmp.", dir=args.out.parent)
    )
    receipts = []
    try:
        for candidate in selected:
            destination = temporary / candidate.scene / candidate.episode
            destination.mkdir(parents=True)
            receipts.append(
                materialize_one(
                    candidate,
                    asset_root=args.asset_root,
                    episode_root=args.episode_root,
                    destination=destination,
                    asset_map=asset_map,
                )
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": (
                "audited online NavDP Goal-A trajectories for subsequent "
                "V0/V1 shared-online-memory benchmark construction"
            ),
            "trace_root": str(args.trace_root.resolve()),
            "selection": {
                "requested_count": args.count,
                "eligible_count": len(candidates),
                "distinct_scene_first": True,
                "anchor_margin": args.anchor_margin,
                "minimum_anchor_gap_frames": args.min_anchor_gap,
                "minimum_preselection_score_m": args.min_anchor_score_m,
                "goals_frozen": False,
            },
            "episodes": receipts,
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

    print(
        json.dumps(
            {
                "output": str(args.out),
                "episodes": len(receipts),
                "frames": sum(row["online_a_steps"] for row in receipts),
                "scenes": [row["scene"] for row in receipts],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
