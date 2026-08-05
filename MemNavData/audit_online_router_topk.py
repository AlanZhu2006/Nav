#!/usr/bin/env python3
"""Reproduce online DINO ranking and geometry recall for one saved episode.

The closed-loop evaluator saves every image seen by the live MemNav server.
This diagnostic consumes that immutable buffer, ranks the complete legal
history with the exact LingBot DINO tower, and applies the same SIFT/essential
matrix test as ``MemNavAgent.verify_retrieval_overlap``.  It therefore answers
whether an online failure came from candidate ranking or geometric rejection
without using Habitat pose as an inference input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from MemNavData.audit_router_candidate_recall import select_temporal_nms
from MemNavData.diag_distill_geometry_router import (
    geometric_teacher,
    load_exact_dino_embeddings,
    sift_description,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_intrinsic(parquet_path: Path) -> np.ndarray:
    import pandas as pd

    rows = pd.read_parquet(parquet_path, columns=["observation.camera_intrinsic"])
    raw = rows.iloc[0]["observation.camera_intrinsic"]
    intrinsic = np.stack([np.asarray(row, dtype=np.float64) for row in raw])
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"invalid camera intrinsic in {parquet_path}")
    return intrinsic


def load_or_compute_embeddings(
    paths: list[Path],
    cache_path: Path | None,
    lingbot_repo: Path,
    weights: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    path_strings = np.asarray([str(path.resolve()) for path in paths])
    weight_sha = file_sha256(weights)
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(cached["paths"].astype(str), path_strings):
            raise RuntimeError("embedding cache path identity mismatch")
        if str(cached["weight_sha"].item()) != weight_sha:
            raise RuntimeError("embedding cache weight identity mismatch")
        embeddings = cached["embeddings"].astype(np.float32, copy=False)
        if embeddings.shape != (len(paths), 1024):
            raise RuntimeError(f"invalid cached embedding shape {embeddings.shape}")
        return embeddings, {"cache_hit": True, "seconds": 0.0}

    embeddings, seconds = load_exact_dino_embeddings(
        paths, lingbot_repo, weights, device, batch_size)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            paths=path_strings,
            embeddings=embeddings.astype(np.float32),
            weight_sha=np.asarray(weight_sha),
        )
    return embeddings, {"cache_hit": False, "seconds": seconds}


def first_selected_positive(selected_frames: list[int], positives: set[int]):
    for rank, frame in enumerate(selected_frames, start=1):
        if frame in positives:
            return {"selected_rank": rank, "frame": frame}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--candidate-min", type=int, required=True)
    parser.add_argument("--candidate-max", type=int, required=True)
    parser.add_argument("--online-anchor", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    import cv2
    import pandas as pd

    args = parse_args()
    if args.candidate_min < 0 or args.candidate_max < args.candidate_min:
        raise ValueError("candidate bounds are invalid")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.report.exists():
        raise FileExistsError(args.report)
    required = (args.buffer_dir, args.goal, args.parquet,
                args.lingbot_repo, args.weights)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    frames = list(range(args.candidate_min, args.candidate_max + 1))
    candidates = [args.buffer_dir / f"{frame}.jpg" for frame in frames]
    missing_candidates = [str(path) for path in candidates if not path.is_file()]
    if missing_candidates:
        raise FileNotFoundError(
            f"missing {len(missing_candidates)} candidate images; "
            f"first={missing_candidates[0]}")

    paths = [args.goal] + candidates
    embeddings, embedding_info = load_or_compute_embeddings(
        paths, args.embedding_cache, args.lingbot_repo, args.weights,
        args.device, args.batch_size)
    normalized = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    scores = normalized[1:] @ normalized[0]

    intrinsic = load_intrinsic(args.parquet)
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher()
    query_features = sift_description(args.goal, sift)
    geometry_started = time.perf_counter()
    geometry = []
    for frame, path, score in zip(frames, candidates, scores):
        result = geometric_teacher(
            query_features, path, intrinsic, sift, matcher)
        passed = bool(
            result["matches"] >= args.min_matches
            and result["inliers"] >= args.min_inliers
            and result["inlier_ratio"] >= args.min_inlier_ratio)
        geometry.append({
            "candidate_frame": frame,
            "candidate_path": str(path.resolve()),
            "dino_cosine": float(score),
            "teacher_pass": int(passed),
            **result,
        })
    geometry_seconds = time.perf_counter() - geometry_started

    ordered = sorted(
        geometry,
        key=lambda row: (-row["dino_cosine"], row["candidate_frame"]),
    )
    rank_by_frame = {
        row["candidate_frame"]: rank
        for rank, row in enumerate(ordered, start=1)
    }
    positives = {
        row["candidate_frame"] for row in geometry if row["teacher_pass"]
    }
    if not positives:
        raise RuntimeError("no candidate passes the configured geometry teacher")
    positive_ranks = sorted(rank_by_frame[frame] for frame in positives)
    top_ks = (1, 2, 4, 8, 16, 32, 64, 128)

    frame_table = pd.DataFrame(geometry).assign(
        session_id="online_episode",
        scene="online_episode",
        kind="revisit",
        teacher_covis=lambda table: table["teacher_pass"].astype(float),
    )
    nms = {}
    for gap in (4, 8, 16, 32):
        selected = select_temporal_nms(frame_table, 32, gap)
        selected_frames = selected["candidate_frame"].astype(int).tolist()
        nms[str(gap)] = {
            "selected_frames": selected_frames,
            "first_positive": first_selected_positive(
                selected_frames, positives),
        }

    report = {
        "purpose": "offline exact online-router top-k recall audit",
        "deployment_approved": False,
        "inputs": {
            "buffer_dir": str(args.buffer_dir.resolve()),
            "goal": str(args.goal.resolve()),
            "goal_sha256": file_sha256(args.goal),
            "parquet": str(args.parquet.resolve()),
            "lingbot_repo": str(args.lingbot_repo.resolve()),
            "weights": str(args.weights.resolve()),
            "weights_sha256": file_sha256(args.weights),
            "candidate_min": args.candidate_min,
            "candidate_max": args.candidate_max,
        },
        "thresholds": {
            "min_matches": args.min_matches,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
        },
        "candidate_count": len(candidates),
        "geometry_positive_count": len(positives),
        "geometry_positive_frames": sorted(positives),
        "geometry_positive_dino_ranks": positive_ranks,
        "best_geometry_positive_rank": positive_ranks[0],
        "median_geometry_positive_rank": float(np.median(positive_ranks)),
        "raw_recall_at_k": {
            str(k): bool(positive_ranks[0] <= k) for k in top_ks
        },
        "temporal_nms_top32": nms,
        "online_anchor": None,
        "top_32": ordered[:32],
        "timing": {
            "dino_seconds": float(embedding_info["seconds"]),
            "dino_cache_hit": bool(embedding_info["cache_hit"]),
            "geometry_seconds": geometry_seconds,
        },
    }
    if args.online_anchor is not None:
        if args.online_anchor not in rank_by_frame:
            raise ValueError("online anchor lies outside the candidate pool")
        row = next(
            item for item in geometry
            if item["candidate_frame"] == args.online_anchor)
        report["online_anchor"] = {
            **row,
            "dino_rank": rank_by_frame[args.online_anchor],
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "report": str(args.report),
        "candidate_count": report["candidate_count"],
        "geometry_positive_count": report["geometry_positive_count"],
        "best_geometry_positive_rank": report["best_geometry_positive_rank"],
        "raw_recall_at_k": report["raw_recall_at_k"],
        "temporal_nms_first_positive": {
            gap: values["first_positive"] for gap, values in nms.items()
        },
        "online_anchor": report["online_anchor"],
        "timing": report["timing"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
