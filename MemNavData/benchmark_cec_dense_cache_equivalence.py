#!/usr/bin/env python3
"""Microbenchmark exact CEC dense-prefix reuse on one immutable RGB trace.

This is an implementation benchmark, not a navigation evaluation.  It runs the
retained legacy full replay and the optimized cached replay in one model process
and compares their dense depth/confidence outputs at identical anchors.  The
first optimized call also verifies that lazy cache construction does not alter
the original first-query result; a second call measures bounded-window reuse.
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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NavDP.baselines.memnav.policy_agent import MemNavAgent  # noqa: E402


def timed(callable_):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.monotonic()
    value = callable_()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return value, time.monotonic() - started


@torch.no_grad()
def blocked_depth(agent: MemNavAgent, anchor: int, block_size: int):
    """Experimental causal block replay; never used by the runtime."""
    snap = agent._snapshot()
    try:
        cache = agent._live_cache()
        indices = cache.get("anchor_frame_indices")
        if indices is None:
            agent.lb._inject(
                cache["scale_k"], cache["scale_v"],
                cache["anchor_k"], cache["anchor_v"],
                n_hist=0, total_frames=agent.S)
        else:
            agent.lb._inject(
                cache["scale_k"], cache["scale_v"],
                cache["anchor_k"], cache["anchor_v"],
                anchor_frame_indices=indices, raw_start=agent.S)
        final_agg = final_psi = final_images = None
        for start in range(agent.S, int(anchor) + 1, int(block_size)):
            stop = min(int(anchor) + 1, start + int(block_size))
            paths = [
                os.path.join(agent.rgb_dir, f"{index}.jpg")
                for index in range(start, stop)
            ]
            images = agent.lb.load_images(paths)
            final_images = images[None].to(agent.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                final_agg, final_psi = agent.lb.model._aggregate_features(
                    final_images,
                    num_frame_for_scale=agent.S,
                    num_frame_per_block=len(images),
                )
        if final_agg is None or final_images is None:
            raise RuntimeError("blocked replay produced no frame")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = agent.lb.model._predict_depth(
                final_agg, final_images, final_psi)
        return (
            prediction["depth"][0, -1, ..., 0].float().cpu().numpy(),
            prediction["depth_conf"][0, -1].float().cpu().numpy(),
        )
    finally:
        agent._restore(snap)


@torch.no_grad()
def eager_writer_depths(agent: MemNavAgent, anchor: int):
    """Simulate exact per-frame dense depth materialization at memory write."""
    snap = agent._snapshot()
    outputs = {}
    try:
        cache = agent._live_cache()
        indices = cache.get("anchor_frame_indices")
        if indices is None:
            agent.lb._inject(
                cache["scale_k"], cache["scale_v"],
                cache["anchor_k"], cache["anchor_v"],
                n_hist=0, total_frames=agent.S)
        else:
            agent.lb._inject(
                cache["scale_k"], cache["scale_v"],
                cache["anchor_k"], cache["anchor_v"],
                anchor_frame_indices=indices, raw_start=agent.S)
        for start in range(agent.S, int(anchor) + 1, 16):
            stop = min(int(anchor) + 1, start + 16)
            paths = [
                os.path.join(agent.rgb_dir, f"{index}.jpg")
                for index in range(start, stop)
            ]
            images = agent.lb.load_images(paths)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for offset, image in enumerate(images):
                    raw_index = start + offset
                    frame = image[None, None].to(agent.device)
                    agg, psi = agent.lb.model._aggregate_features(
                        frame,
                        num_frame_for_scale=agent.S,
                        num_frame_per_block=1,
                    )
                    prediction = agent.lb.model._predict_depth(
                        agg, frame, psi)
                    outputs[raw_index] = (
                        prediction["depth"][0, -1, ..., 0]
                        .float().cpu().numpy(),
                        prediction["depth_conf"][0, -1]
                        .float().cpu().numpy(),
                    )
        return outputs
    finally:
        agent._restore(snap)


def compare(first: np.ndarray, second: np.ndarray) -> dict:
    delta = np.abs(
        np.asarray(first, dtype=np.float64)
        - np.asarray(second, dtype=np.float64))
    return {
        "shape_equal": list(first.shape) == list(second.shape),
        "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
        "array_equal": bool(np.array_equal(first, second)),
        "maximum_absolute_error": float(delta.max(initial=0.0)),
        "mean_absolute_error": float(delta.mean()) if delta.size else 0.0,
    }


def online_state_digest(agent: MemNavAgent) -> dict:
    digest = hashlib.sha256()

    def update(name, value):
        digest.update(name.encode("utf-8") + b"\0")
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))

    update("anchor_frame_indices", agent.anchor_frame_indices)
    update("cam_frame_indices", agent.cam_frame_indices)
    for name, values in (
        ("dino_cls", agent.dino_cls),
        ("cam_pose", agent.cam_pose),
        ("anchor_k", agent.anchor_k),
        ("anchor_v", agent.anchor_v),
        ("cam_k", agent.cam_k),
        ("cam_v", agent.cam_v),
        ("last_agg", agent._last_agg or []),
    ):
        update(name + "_count", len(values))
        for index, value in enumerate(values):
            update(f"{name}_{index}", value)
    update("last_tokens", agent._last_tokens)
    return {
        "sha256": digest.hexdigest(),
        "anchor_frame_indices": list(agent.anchor_frame_indices),
        "cam_frame_indices": list(agent.cam_frame_indices),
        "dino_frames": len(agent.dino_cls),
        "cam_pose_frames": len(agent.cam_pose),
    }


def snapshot_storage_bytes(snapshot) -> int:
    if snapshot is None:
        return 0
    storages = {}

    def visit(value):
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            storages[int(storage.data_ptr())] = int(storage.nbytes())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(snapshot)
    return int(sum(storages.values()))


def strip_runtime_diagnostics(value):
    """Remove only timing/cache-source receipts before decision comparison."""
    if isinstance(value, dict):
        return {
            key: strip_runtime_diagnostics(item)
            for key, item in value.items()
            if (not key.endswith("_ms")
                and key not in {
                    "cached", "reference_depth_cache",
                    "uncached_relocalization_ms", "relocalization_ms",
                })
        }
    if isinstance(value, list):
        return [strip_runtime_diagnostics(item) for item in value]
    return value


def require_contiguous_frames(rgb_dir: Path, count: int) -> list[Path]:
    paths = []
    missing = []
    for index in range(count):
        candidates = (
            rgb_dir / f"{index}.jpg",
            rgb_dir / f"{index:06d}.jpg",
        )
        path = next((value for value in candidates if value.is_file()), None)
        if path is None:
            missing.append(str(candidates[0]))
        else:
            paths.append(path)
    if missing:
        raise RuntimeError(f"missing contiguous RGB frames: {missing[:3]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--internnav-root", required=True, type=Path)
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--anchors", default="80,140")
    parser.add_argument(
        "--block-sizes", default="2,4,8,16",
        help="experimental multi-frame replay sizes; empty disables")
    parser.add_argument("--benchmark-eager-writer", action="store_true")
    parser.add_argument(
        "--use-eager-agent", action="store_true",
        help="exercise the production online dual-stream implementation")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--lightglue-repo", type=Path)
    parser.add_argument("--lightglue-dependency-root", type=Path)
    parser.add_argument("--end-to-end-goal-index", type=int)
    parser.add_argument("--episode-len", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    anchors = [int(value) for value in args.anchors.split(",") if value]
    block_sizes = [
        int(value) for value in args.block_sizes.split(",") if value]
    if not anchors or min(anchors) < 8:
        raise ValueError("anchors must contain raw indices >= 8")
    frame_count = max(anchors) + 1
    paths = require_contiguous_frames(args.rgb_dir.resolve(), frame_count)
    episode_len = args.episode_len or frame_count

    with tempfile.TemporaryDirectory(prefix="cec_dense_cache_bench.") as root:
        agent = MemNavAgent(
            checkpoint=str(args.checkpoint.resolve()),
            internnav_root=str(args.internnav_root.resolve()),
            device=args.device,
            buffer_root=root,
            flow_gate="auto",
            certified_eager_depth_cache=args.use_eager_agent,
        )
        cuda_after_model_load = (
            int(torch.cuda.memory_allocated())
            if torch.cuda.is_available() else None)
        agent.reset(episode_len=episode_len, seed=args.seed)
        ingest_started = time.monotonic()
        for path in paths:
            agent.add_frame(path.read_bytes())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ingest_s = time.monotonic() - ingest_started
        live_snapshot = agent._snapshot()
        resource = {
            "cuda_allocated_after_model_load_bytes": cuda_after_model_load,
            "cuda_allocated_after_ingest_bytes": (
                int(torch.cuda.memory_allocated())
                if torch.cuda.is_available() else None),
            "cuda_reserved_after_ingest_bytes": (
                int(torch.cuda.memory_reserved())
                if torch.cuda.is_available() else None),
            "live_snapshot_unique_storage_bytes": snapshot_storage_bytes(
                live_snapshot),
            "eager_dense_snapshot_unique_storage_bytes": (
                snapshot_storage_bytes(
                    agent._certified_dense_stream_snapshot)),
            "depth_cache_bytes": int(sum(
                array.nbytes for pair in
                agent._certified_reference_depth_cache.values()
                for array in pair)),
        }

        records = []
        for anchor in ([] if args.ingest_only else anchors):
            (legacy_depth, legacy_conf), legacy_s = timed(
                lambda anchor=anchor:
                agent._certified_reference_depth_legacy(anchor))
            (first_depth, first_conf), first_s = timed(
                lambda anchor=anchor:
                agent._certified_reference_depth(anchor))
            first_stats = dict(agent._certified_dense_replay_last_stats)
            (repeat_depth, repeat_conf), repeat_s = timed(
                lambda anchor=anchor:
                agent._certified_reference_depth(anchor))
            repeat_stats = dict(agent._certified_dense_replay_last_stats)
            blocked = []
            for block_size in block_sizes:
                try:
                    (block_depth, block_conf), block_s = timed(
                        lambda anchor=anchor, block_size=block_size:
                        blocked_depth(agent, anchor, block_size))
                    blocked.append({
                        "block_size": block_size,
                        "runtime_s": block_s,
                        "speedup": legacy_s / block_s,
                        "depth": compare(legacy_depth, block_depth),
                        "confidence": compare(legacy_conf, block_conf),
                    })
                except Exception as error:
                    blocked.append({
                        "block_size": block_size,
                        "error": f"{type(error).__name__}: {error}",
                    })
            eager_writer = None
            if args.benchmark_eager_writer:
                eager_outputs, eager_s = timed(
                    lambda anchor=anchor: eager_writer_depths(agent, anchor))
                eager_depth, eager_conf = eager_outputs[anchor]
                eager_writer = {
                    "runtime_s": eager_s,
                    "frames_materialized": len(eager_outputs),
                    "mean_runtime_s_per_frame": (
                        eager_s / len(eager_outputs)),
                    "cache_bytes": int(sum(
                        array.nbytes for pair in eager_outputs.values()
                        for array in pair)),
                    "depth": compare(legacy_depth, eager_depth),
                    "confidence": compare(legacy_conf, eager_conf),
                }
            records.append({
                "anchor": anchor,
                "legacy_s": legacy_s,
                "first_cached_path_s": first_s,
                "repeat_cached_path_s": repeat_s,
                "repeat_speedup": legacy_s / repeat_s,
                "legacy_vs_first_depth": compare(
                    legacy_depth, first_depth),
                "legacy_vs_first_confidence": compare(
                    legacy_conf, first_conf),
                "legacy_vs_repeat_depth": compare(
                    legacy_depth, repeat_depth),
                "legacy_vs_repeat_confidence": compare(
                    legacy_conf, repeat_conf),
                "first_stats": first_stats,
                "repeat_stats": repeat_stats,
                "experimental_block_replay": blocked,
                "experimental_eager_writer": eager_writer,
            })

        end_to_end = None
        if args.end_to_end_goal_index is not None:
            if args.lightglue_repo is None:
                raise ValueError(
                    "--end-to-end-goal-index requires --lightglue-repo")
            import cv2
            from MemNavData.lingbot_pnp_localization import (
                LightGluePointMatcher,
            )

            matcher = LightGluePointMatcher(
                args.lightglue_repo,
                dependency_root=args.lightglue_dependency_root,
                device=args.device,
                max_keypoints=2048,
            )
            agent.certified_relocalization_matcher = matcher
            goal_index = int(args.end_to_end_goal_index)
            goal_bytes = paths[goal_index].read_bytes()
            goal_key = hashlib.md5(goal_bytes).hexdigest()
            agent._goal_start_frame[goal_key] = agent.n
            candidate = {"anchor": anchors[0], "score": 1.0}

            optimized_depth = agent._certified_reference_depth
            agent._certified_reference_depth = (
                agent._certified_reference_depth_legacy)
            cv2.setRNGSeed(args.seed)
            legacy_result, legacy_e2e_s = timed(
                lambda: agent.certified_relocalize(
                    goal_bytes, [candidate]))
            agent._certified_relocalization_cache.clear()
            agent._certified_reference_depth = optimized_depth
            cv2.setRNGSeed(args.seed)
            optimized_result, optimized_e2e_s = timed(
                lambda: agent.certified_relocalize(
                    goal_bytes, [candidate]))
            legacy_decision = strip_runtime_diagnostics(legacy_result)
            optimized_decision = strip_runtime_diagnostics(optimized_result)
            end_to_end = {
                "goal_index": goal_index,
                "candidate": candidate,
                "legacy_runtime_s": legacy_e2e_s,
                "optimized_runtime_s": optimized_e2e_s,
                "speedup": legacy_e2e_s / optimized_e2e_s,
                "decision_equal": legacy_decision == optimized_decision,
                "legacy_depth_receipt": legacy_result.get(
                    "reference_depth_cache"),
                "optimized_depth_receipt": optimized_result.get(
                    "reference_depth_cache"),
                "legacy": legacy_decision,
                "optimized": optimized_decision,
            }

        tolerance = 1e-5
        decision_equivalent = (None if args.ingest_only else all(
            record[key]["maximum_absolute_error"] <= tolerance
            for record in records
            for key in (
                "legacy_vs_first_depth",
                "legacy_vs_first_confidence",
                "legacy_vs_repeat_depth",
                "legacy_vs_repeat_confidence",
            )
        ))
        payload = {
            "schema_version": "cec_dense_cache_equivalence_v1_20260818",
            "scientific_evaluation": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "rgb_dir": str(args.rgb_dir.resolve()),
            "frame_count": frame_count,
            "episode_len": episode_len,
            "ingest_s": ingest_s,
            "seed": args.seed,
            "online_navigation_state": online_state_digest(agent),
            "resource": resource,
            "production_eager_agent": bool(args.use_eager_agent),
            "eager_writer_error": agent._certified_eager_depth_error,
            "eager_writer_frame_runtime_ms": (
                list(agent._certified_eager_depth_runtime_ms)),
            "tolerance": tolerance,
            "depth_confidence_equivalent": decision_equivalent,
            "records": records,
            "end_to_end_certificate": end_to_end,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
            print(args.output)


if __name__ == "__main__":
    main()
