#!/usr/bin/env python3
"""Train-only audit of a metric image-goal arrival certificate.

The deployed NavDP contract clips every trajectory shorter than 0.5 m to a
zero trajectory.  A zero trajectory is therefore only a request to consider
stopping.  This collector tests a stricter, label-blind condition on the exact
states used by ``audit_navdp_arrival_consensus.py``:

* SuperPoint + LightGlue matches the current RGB to the image goal;
* the frozen v2 geometry certificate must be reachable;
* causal LingBot depth lifts reference matches for PnP;
* the already-frozen first-prefix ground scale converts the relative pose to m.

Ground-truth distance is copied to the output only after inference.  This file
collects measurements; ``summarize_lingbot_pnp_arrival.py`` owns all threshold
comparisons and cannot affect the collector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


SCHEMA_VERSION = "lingbot_pnp_arrival_train_audit_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".csv",
            mode="w", encoding="utf-8", newline="", delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _bool_column(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    require(bool(normalized.isin({"true", "false", "1", "0"}).all()),
            f"{label} is not boolean")
    return normalized.isin({"true", "1"})


def _episode_key(scene: object, episode: object) -> str:
    return f"{str(scene)}/{str(episode)}"


def load_inputs(states_csv: Path, samples_csv: Path,
                inventory_json: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    states = pd.read_csv(states_csv)
    required = {
        "state_id", "scene", "episode", "goal_index", "frame_index",
        "distance_band", "euclidean_distance_m", "arrival_025",
    }
    require(not (required - set(states.columns)),
            f"states CSV missing {sorted(required - set(states.columns))}")
    require(states["state_id"].is_unique, "state IDs are not unique")
    states["arrival_025"] = _bool_column(states["arrival_025"], "arrival_025")
    strict = states["euclidean_distance_m"].astype(float) < 0.25
    require(bool((strict == states["arrival_025"]).all()),
            "stored <=0.25 labels differ from strict GOAT <0.25 labels")

    samples = pd.read_csv(samples_csv)
    sample_required = {"state_id", "sample_index", "selected_zero"}
    require(not (sample_required - set(samples.columns)),
            "samples CSV lacks the native trigger columns")
    sample0 = samples.loc[samples["sample_index"].astype(int).eq(0)].copy()
    require(sample0["state_id"].is_unique,
            "sample-0 native trigger is not unique per state")
    sample0["native_selected_zero_sample0"] = _bool_column(
        sample0["selected_zero"], "selected_zero")
    trigger = sample0.set_index("state_id")["native_selected_zero_sample0"]
    states["native_selected_zero_sample0"] = states["state_id"].map(trigger)
    require(states["native_selected_zero_sample0"].notna().all(),
            "some states have no sample-0 native trigger")
    states["native_selected_zero_sample0"] = states[
        "native_selected_zero_sample0"].astype(bool)

    inventory = json.loads(inventory_json.read_text(encoding="utf-8"))
    episodes = inventory.get("episodes")
    require(isinstance(episodes, list) and episodes, "inventory has no episodes")
    by_episode = {}
    inventory_states = set()
    for item in episodes:
        key = _episode_key(item.get("scene"), item.get("episode"))
        require(key not in by_episode, f"duplicate inventory episode {key}")
        by_episode[key] = dict(item)
        inventory_states.update(map(str, item.get("states", [])))
    require(inventory_states == set(states["state_id"].astype(str)),
            "inventory/state CSV exact cover changed")
    return states, by_episode


def load_route_index(path: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "flow_routes_audited",
            "flow routing is not audited")
    roots = {
        str(name): Path(value).resolve()
        for name, value in payload.get("source_roots", {}).items()
    }
    routes = {}
    for item in payload.get("pairs", []):
        key = str(item.get("episode"))
        require(key not in routes, f"duplicate flow route {key}")
        source_id = str(item.get("source_id"))
        require(source_id in roots, f"unknown flow source {source_id}")
        routes[key] = dict(item)
    return routes, roots


def load_scale_index(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == "nlsr_v2_causal_ground_scale_v1",
            "causal scale schema changed")
    require(payload.get("summary", {}).get("future_frames_consumed") == 0,
            "scale artifact consumed future frames")
    result = {}
    for item in payload.get("records", []):
        key = _episode_key(item.get("scene"), item.get("episode"))
        require(key not in result, f"duplicate scale record {key}")
        require(item.get("valid") is True, f"invalid causal scale for {key}")
        scale = float(item.get("metric_scale_m_per_raw"))
        require(np.isfinite(scale) and scale > 0.0,
                f"non-positive causal scale for {key}")
        result[key] = dict(item)
    return result


def resolve_cache(route: Mapping[str, Any], roots: Mapping[str, Path]) \
        -> tuple[Path, Path]:
    chunk = roots[str(route["source_id"])] / str(route["source_relative_chunk"])
    cache = chunk / "lingbot_cache.npz"
    camera = chunk / "lingbot_cam_cache.npz"
    require(cache.is_file() and camera.is_file(),
            f"missing routed cache pair under {chunk}")
    return cache, camera


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def prepare_matches(rows: pd.DataFrame, episode_root: Path, matcher: Any,
                    *, target_size: int, patch_size: int) -> dict[str, dict]:
    from MemNavData.certified_relocalization_runtime import (
        CERTIFIED_EPIPOLAR_THRESHOLD_PX,
        fundamental_can_reach_certificate,
        fundamental_support,
    )

    prepared = {}
    for row in rows.sort_values(["goal_index", "frame_index"]).itertuples():
        current = (episode_root / "videos" / "chunk-000"
                   / "observation.images.rgb" / f"{int(row.frame_index)}.jpg")
        goal = episode_root / f"goal_{int(row.goal_index)}.jpg"
        require(current.is_file(), f"missing current RGB {current}")
        require(goal.is_file(), f"missing goal RGB {goal}")
        matched = matcher.match_paths(
            current, goal, target_height=target_size,
            target_width=target_size, patch_size=patch_size)
        support = fundamental_support(
            matched["reference_raw_points"], matched["query_raw_points"],
            matched["scores"], tuple(matched["reference_raw_hw"]),
            tuple(matched["query_raw_hw"]),
            threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX)
        possible, reason = fundamental_can_reach_certificate(support)
        prepared[str(row.state_id)] = {
            "current_path": current,
            "goal_path": goal,
            "matched": matched,
            "support": support,
            "precheck_passed": bool(possible),
            "precheck_reason": str(reason),
        }
    return prepared


def _row_prefix(row: Any, prepared: Mapping[str, Any], scale: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "state_id": str(row.state_id),
        "scene": str(row.scene),
        "episode": str(row.episode),
        "goal_index": int(row.goal_index),
        "frame_index": int(row.frame_index),
        "distance_band": str(row.distance_band),
        "euclidean_distance_m": float(row.euclidean_distance_m),
        "arrival_025_strict": bool(float(row.euclidean_distance_m) < 0.25),
        "native_selected_zero_sample0": bool(
            row.native_selected_zero_sample0),
        "metric_scale_m_per_raw": float(scale),
        "precheck_passed": bool(prepared["precheck_passed"]),
        "precheck_reason": str(prepared["precheck_reason"]),
        **{str(key): _jsonable(value)
           for key, value in prepared["support"].items()},
    }


@torch.inference_mode()
def collect_episode(lb: Any, matcher: Any, rows: pd.DataFrame,
                    episode_root: Path, cache_path: Path, camera_path: Path,
                    scale_record: Mapping[str, Any], *, chunk_size: int) \
        -> list[dict]:
    from MemNavData.certified_relocalization_runtime import (
        CERTIFIED_EPIPOLAR_THRESHOLD_PX,
        certificate_decision,
        scale_free_relative_xy,
    )
    from MemNavData.lingbot_pnp_localization import (
        SiftPnPConfig,
        correspondence_pnp_localize,
    )

    scale = float(scale_record["metric_scale_m_per_raw"])
    prefix_end = int(scale_record["prefix_end_frame_exclusive"])
    require(prefix_end <= int(rows["frame_index"].min()),
            "causal scale extends beyond an audited decision state")

    with np.load(cache_path, allow_pickle=False) as source:
        anchor_indices = np.asarray(
            source["anchor_frame_indices"], dtype=np.int64)
        num_frames = int(np.asarray(source["num_frames"]).reshape(-1)[0])
        num_scale = int(np.asarray(
            source["num_scale_frames"]).reshape(-1)[0])
    with np.load(camera_path, allow_pickle=False) as camera:
        poses = np.asarray(camera["cam_pose_enc"], dtype=np.float32)
    require(num_scale == int(lb.num_scale), "cache/model scale-frame mismatch")
    require(len(poses) == num_frames, "camera/cache frame count mismatch")
    maximum = int(rows["frame_index"].max())
    require(maximum < num_frames, "decision frame exceeds routed cache")
    require(np.all(anchor_indices[:-1] < anchor_indices[1:]),
            "anchor indices are not strictly increasing")
    keep_frames = set(map(int, anchor_indices.tolist()))

    prepared = prepare_matches(
        rows, episode_root, matcher, target_size=int(lb.img_size),
        patch_size=int(lb.patch_size))
    by_frame = {
        int(row.frame_index): row
        for row in rows.itertuples(index=False)
    }
    require(len(by_frame) == len(rows),
            "multiple audited goals selected the same physical frame")

    rgb_dir = (episode_root / "videos" / "chunk-000"
               / "observation.images.rgb")
    first_paths = [rgb_dir / f"{index}.jpg" for index in range(num_scale)]
    require(all(path.is_file() for path in first_paths),
            "scale RGB prefix is incomplete")
    lb.model.clean_kv_cache()
    first = lb.load_images([str(path) for path in first_paths])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lb.model._aggregate_features(
            first[None].to(lb.device), num_frame_for_scale=num_scale,
            num_frame_per_block=num_scale)

    output = []
    for chunk_start in range(num_scale, maximum + 1, chunk_size):
        chunk_end = min(maximum + 1, chunk_start + chunk_size)
        paths = [rgb_dir / f"{index}.jpg"
                 for index in range(chunk_start, chunk_end)]
        require(all(path.is_file() for path in paths),
                f"RGB stream gap in {episode_root}")
        images = lb.load_images([str(path) for path in paths])
        for offset, frame_index in enumerate(range(chunk_start, chunk_end)):
            image = images[offset:offset + 1][None].to(lb.device)
            retained = frame_index in keep_frames
            if not retained:
                saved_cache = dict(lb.agg.kv_cache)
                saved_total = int(lb.agg.total_frames_processed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                aggregate, psi = lb.model._aggregate_features(
                    image, num_frame_for_scale=num_scale,
                    num_frame_per_block=1)
            selected = by_frame.get(frame_index)
            if selected is not None:
                item = prepared[str(selected.state_id)]
                result = _row_prefix(selected, item, scale)
                pnp: dict[str, Any] = {
                    "status": str(item["precheck_reason"]),
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                }
                depth_latency = 0.0
                if item["precheck_passed"]:
                    started = time.monotonic()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = lb.model._predict_depth(
                            aggregate, image, psi)
                    depth = prediction["depth"][
                        0, -1, ..., 0].float().cpu().numpy()
                    confidence = prediction["depth_conf"][
                        0, -1].float().cpu().numpy()
                    depth_latency = time.monotonic() - started
                    matched = item["matched"]
                    pnp = correspondence_pnp_localize(
                        matched["reference_points"],
                        matched["query_points"], depth, confidence,
                        poses[frame_index], config=SiftPnPConfig(),
                        match_scores=matched["scores"],
                        epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX)
                certificate = certificate_decision(pnp)
                predicted_distance = None
                predicted_xy = None
                if "pose9" in pnp:
                    predicted_xy = scale * np.asarray(
                        scale_free_relative_xy(
                            poses[frame_index], np.asarray(pnp["pose9"])),
                        dtype=np.float64)
                    predicted_distance = float(np.linalg.norm(predicted_xy))
                result.update({
                    "depth_inference_s": float(depth_latency),
                    "pnp_status": str(pnp.get("status")),
                    "pnp_matches": int(pnp.get("matches", 0)),
                    "pnp_epipolar_inliers": int(
                        pnp.get("epipolar_inliers", 0)),
                    "pnp_depth_valid_matches": int(
                        pnp.get("depth_valid_matches", 0)),
                    "pnp_inliers": int(pnp.get("inliers", 0)),
                    "pnp_inlier_ratio": float(pnp.get("inlier_ratio", 0.0)),
                    "pnp_reprojection_rmse_px": pnp.get(
                        "reprojection_rmse_px"),
                    "pnp_reference_inlier_coverage": pnp.get(
                        "reference_inlier_coverage"),
                    "pnp_query_inlier_coverage": pnp.get(
                        "query_inlier_coverage"),
                    "certificate_accepted": bool(certificate["accepted"]),
                    "certificate_reason": str(certificate["reason"]),
                    "predicted_relative_xy_m_json": (
                        json.dumps(predicted_xy.tolist())
                        if predicted_xy is not None else None),
                    "predicted_distance_m": predicted_distance,
                    "pnp_pose9_json": (
                        json.dumps(_jsonable(pnp["pose9"]))
                        if "pose9" in pnp else None),
                })
                output.append(result)
            if not retained:
                lb.agg.kv_cache.clear()
                lb.agg.kv_cache.update(saved_cache)
                lb.agg.total_frames_processed = saved_total
    lb.model.clean_kv_cache()
    require(len(output) == len(rows), "not every selected state was evaluated")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-csv", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--flow-route-provenance", type=Path, required=True)
    parser.add_argument("--causal-scale-artifact", type=Path, required=True)
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-scale", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-frame-num", type=int, default=4096)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.states_csv.is_file(), "states CSV is missing")
    require(args.samples_csv.is_file(), "samples CSV is missing")
    require(args.inventory_json.is_file(), "inventory JSON is missing")
    require(args.flow_route_provenance.is_file(), "flow routes are missing")
    require(args.causal_scale_artifact.is_file(), "scale artifact is missing")
    require(0 <= args.shard_index < args.shard_count,
            "invalid shard index/count")
    require(args.chunk_size >= 1 and args.num_scale >= 1,
            "chunk and scale sizes must be positive")
    require(args.max_episodes >= 0, "max episodes must be non-negative")
    require(not args.out_dir.exists(), f"output already exists: {args.out_dir}")

    states, inventory = load_inputs(
        args.states_csv, args.samples_csv, args.inventory_json)
    routes, route_roots = load_route_index(args.flow_route_provenance)
    scales = load_scale_index(args.causal_scale_artifact)
    episode_keys = sorted(inventory)
    require(set(episode_keys) <= set(routes), "some episodes lack flow caches")
    require(set(episode_keys) <= set(scales), "some episodes lack causal scale")
    selected_keys = episode_keys[args.shard_index::args.shard_count]
    if args.max_episodes:
        selected_keys = selected_keys[:args.max_episodes]
    require(selected_keys, "shard selected no episodes")

    selection = {
        "schema_version": SCHEMA_VERSION,
        "selection_only": bool(args.selection_only),
        "states_csv": str(args.states_csv.resolve()),
        "states_sha256": sha256_file(args.states_csv),
        "samples_csv": str(args.samples_csv.resolve()),
        "samples_sha256": sha256_file(args.samples_csv),
        "inventory_json": str(args.inventory_json.resolve()),
        "inventory_sha256": sha256_file(args.inventory_json),
        "flow_route_provenance": str(args.flow_route_provenance.resolve()),
        "flow_route_sha256": sha256_file(args.flow_route_provenance),
        "causal_scale_artifact": str(args.causal_scale_artifact.resolve()),
        "causal_scale_sha256": sha256_file(args.causal_scale_artifact),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "episode_count": len(selected_keys),
        "episodes": selected_keys,
        "state_count": int(states.loc[
            states.apply(
                lambda row: _episode_key(row["scene"], row["episode"])
                in set(selected_keys), axis=1)].shape[0]),
    }
    if args.selection_only:
        args.out_dir.mkdir(parents=True)
        atomic_json(args.out_dir / "selection.json", selection)
        return

    sys.path.insert(0, str(args.internnav_root.resolve()))
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher

    torch.manual_seed(0)
    np.random.seed(0)
    lb = LingBotStream(
        lingbot_repo=str(args.lingbot_repo.resolve()),
        weights=str(args.weights.resolve()), num_scale=args.num_scale,
        window=args.window, max_frame_num=args.max_frame_num,
        camera_num_iterations=args.camera_num_iterations,
        device=args.device,
    )
    matcher = LightGluePointMatcher(
        args.lightglue_repo, dependency_root=args.dependency_root,
        device=args.device, max_keypoints=args.max_keypoints)

    started = time.time()
    all_rows = []
    completed = []
    for offset, key in enumerate(selected_keys, start=1):
        item = inventory[key]
        root = Path(str(item["root"]))
        require(root.is_dir(), f"episode root is missing: {root}")
        route = routes[key]
        cache, camera = resolve_cache(route, route_roots)
        expected_frames = int(route["validation"]["num_frames"])
        episode_rows = states.loc[
            states.apply(
                lambda row: _episode_key(row["scene"], row["episode"]) == key,
                axis=1)].copy()
        require(not episode_rows.empty, f"no selected states for {key}")
        print(
            f"[{offset}/{len(selected_keys)}] {key}: "
            f"frames={expected_frames} states={len(episode_rows)}",
            flush=True,
        )
        episode_started = time.time()
        output = collect_episode(
            lb, matcher, episode_rows, root, cache, camera, scales[key],
            chunk_size=args.chunk_size)
        elapsed = time.time() - episode_started
        for row in output:
            row["episode_runtime_s"] = float(elapsed)
        all_rows.extend(output)
        completed.append(key)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        atomic_csv(args.out_dir / "rows.partial.csv", pd.DataFrame(all_rows))
        atomic_json(args.out_dir / "progress.json", {
            **selection,
            "selection_only": False,
            "status": "collecting",
            "completed_episodes": completed,
            "completed_state_count": len(all_rows),
            "runtime_s": float(time.time() - started),
        })

    frame = pd.DataFrame(all_rows).sort_values("state_id", kind="stable")
    require(frame["state_id"].is_unique, "collector emitted duplicate states")
    atomic_csv(args.out_dir / "rows.csv", frame)
    atomic_json(args.out_dir / "report.json", {
        **selection,
        "selection_only": False,
        "status": "complete",
        "row_count": len(frame),
        "precheck_pass_count": int(frame["precheck_passed"].sum()),
        "certificate_accept_count": int(frame["certificate_accepted"].sum()),
        "runtime_s": float(time.time() - started),
    })
    partial = args.out_dir / "rows.partial.csv"
    if partial.exists():
        partial.unlink()
    atomic_json(args.out_dir / "SHA256SUMS.json", {
        name: sha256_file(args.out_dir / name)
        for name in ("rows.csv", "report.json")
    })


if __name__ == "__main__":
    main()
