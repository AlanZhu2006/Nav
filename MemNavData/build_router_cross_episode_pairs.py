#!/usr/bin/env python3
"""Expand router training sessions with independent router queries.

For each explicitly selected training scene, adjacent generated episodes are
paired.  Frames sampled from one episode query the full trajectory of its
partner.  Optionally, post-switch return-leg frames query only the same
episode's pre-switch memory.  Frozen DINO cosine supplies retrieval scores and
the unchanged SIFT/essential-matrix verifier supplies pair labels.  No
navigation success, simulator pose, or held-out-scene query is used for
training or calibration.

The output retains the original teacher CSV and appends explicitly role-tagged
training or evaluation sessions.  A separate patch/temporal router script
subsequently performs scene-disjoint training; evaluation-scene rows are never
used for model selection or confidence calibration.

When ``--teacher-top-k`` is positive, all frozen-DINO scores are retained for
temporal features but expensive geometry is evaluated only on the exact hard
top-K consumed by the downstream experiment.  Other rows receive label -1 and
cannot be selected for training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from MemNavData.diag_distill_geometry_router import (
        geometric_teacher,
        sift_description,
    )
    from MemNavData.diag_patch_temporal_router import (
        load_cls_cache,
        parse_path_maps,
        remap_path,
        sha256,
    )
except ModuleNotFoundError:  # direct script invocation
    from diag_distill_geometry_router import (  # type: ignore
        geometric_teacher,
        sift_description,
    )
    from diag_patch_temporal_router import (  # type: ignore
        load_cls_cache,
        parse_path_maps,
        remap_path,
        sha256,
    )


REQUIRED_COLUMNS = (
    "session_id", "scene", "episode", "kind", "query_path",
    "candidate_path", "candidate_frame", "teacher_pass", "matches",
    "inliers", "inlier_ratio", "error", "dino_cosine",
)


def episode_root_from_rgb(path: Path) -> Path:
    if path.parent.name != "observation.images.rgb":
        raise ValueError(f"unexpected RGB path layout: {path}")
    root = path.parents[3]
    if not (root / "meta" / "gen_meta.json").is_file():
        raise FileNotFoundError(root / "meta" / "gen_meta.json")
    return root


def load_intrinsic(episode_root: Path) -> np.ndarray:
    import pandas as pd

    parquet = (episode_root / "data" / "chunk-000" /
               "episode_000000.parquet")
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    rows = pd.read_parquet(
        parquet, columns=["observation.camera_intrinsic"])
    raw = rows.iloc[0]["observation.camera_intrinsic"]
    intrinsic = np.stack([
        np.asarray(row, dtype=np.float64) for row in raw])
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"invalid camera intrinsic in {parquet}")
    return intrinsic


def load_episode_meta(episode_root: Path) -> Tuple[int, int]:
    meta_path = episode_root / "meta" / "gen_meta.json"
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    switch = int(meta["switch_idx"])
    frame_count = int(meta["n_frames"])
    if frame_count < 1 or not 0 < switch < frame_count:
        raise ValueError(
            f"invalid switch/frame count in {meta_path}: "
            f"switch={switch}, frames={frame_count}")
    return switch, frame_count


def collect_episode_frames(frame, scenes: Sequence[str]) -> Dict[
        Tuple[str, str], Dict[int, str]]:
    selected = frame[frame["scene"].isin(scenes)]
    episodes: Dict[Tuple[str, str], Dict[int, str]] = {}
    for row in selected.itertuples():
        key = (row.scene, row.episode)
        paths = episodes.setdefault(key, {})
        candidate_frame = int(row.candidate_frame)
        existing = paths.get(candidate_frame)
        if existing is not None and existing != row.candidate_path:
            raise ValueError(
                f"conflicting frame path for {key}/{candidate_frame}")
        paths[candidate_frame] = row.candidate_path
    for key, paths in episodes.items():
        ordered = sorted(paths)
        if not ordered or ordered != list(range(ordered[-1] + 1)):
            raise ValueError(
                f"episode {key} does not expose a contiguous full trajectory")
    return episodes


def paired_episode_names(episodes: Dict[Tuple[str, str], Dict[int, str]],
                         scene: str) -> Dict[str, str]:
    names = sorted(name for item_scene, name in episodes if item_scene == scene)
    if len(names) < 2 or len(names) % 2:
        raise ValueError(
            f"scene {scene} requires an even number of episodes, got {len(names)}")
    result = {}
    for index, name in enumerate(names):
        result[name] = names[index + 1 if index % 2 == 0 else index - 1]
    return result


def query_indices(frame_count: int, stride: int, margin: int,
                  maximum: int) -> Sequence[int]:
    lower = min(margin, max(frame_count - 1, 0))
    upper = max(lower + 1, frame_count - margin)
    indices = list(range(lower, upper, stride))
    if not indices:
        indices = [frame_count // 2]
    if maximum > 0 and len(indices) > maximum:
        positions = np.linspace(0, len(indices) - 1, maximum)
        indices = [indices[int(round(position))] for position in positions]
        indices = list(dict.fromkeys(indices))
    return indices


def return_query_indices(frame_count: int, switch: int, stride: int,
                         margin: int, maximum: int) -> Sequence[int]:
    """Sample post-switch queries, expressed in full-trajectory indices."""
    if not 0 < switch < frame_count:
        raise ValueError(
            f"switch must be inside trajectory: {switch}/{frame_count}")
    return [
        switch + offset for offset in query_indices(
            frame_count - switch, stride, margin, maximum)
    ]


def hard_teacher_frames(candidate_frames: Sequence[int],
                        scores: Sequence[float], top_k: int) -> frozenset[int]:
    """Select deterministic high-cosine candidates that require geometry."""
    frames = [int(frame) for frame in candidate_frames]
    score_values = [float(score) for score in scores]
    if len(frames) != len(score_values) or not frames:
        raise ValueError("candidate frames and scores must be non-empty/aligned")
    if len(set(frames)) != len(frames):
        raise ValueError("candidate frames must be unique")
    if top_k < 0 or not np.isfinite(score_values).all():
        raise ValueError("top-K must be non-negative and scores finite")
    count = len(frames) if top_k == 0 else min(top_k, len(frames))
    order = sorted(
        range(len(frames)), key=lambda index: (-score_values[index], frames[index]))
    return frozenset(frames[index] for index in order[:count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-teacher-csv", type=Path, required=True)
    parser.add_argument("--cls-cache", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-scene", action="append", default=[])
    parser.add_argument("--evaluation-scene", action="append", default=[])
    parser.add_argument(
        "--path-map", action="append", default=[],
        help="optional OLD=NEW prefix replacement for moved episode images")
    parser.add_argument("--query-stride", type=int, default=32)
    parser.add_argument("--query-margin", type=int, default=16)
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--max-queries-per-episode", type=int, default=0)
    parser.add_argument("--include-return", action="store_true")
    parser.add_argument("--return-query-stride", type=int, default=32)
    parser.add_argument("--return-query-margin", type=int, default=16)
    parser.add_argument("--return-candidate-stride", type=int, default=1)
    parser.add_argument("--max-return-queries-per-episode", type=int, default=0)
    parser.add_argument(
        "--teacher-top-k", type=int, default=0,
        help="evaluate geometry only for this many DINO candidates; 0 means all")
    parser.add_argument("--expected-base-sha", default="")
    return parser.parse_args()


def main() -> None:
    import cv2
    import pandas as pd

    args = parse_args()
    if (args.query_stride < 1 or args.candidate_stride < 1
            or args.return_query_stride < 1
            or args.return_candidate_stride < 1):
        raise ValueError("query/candidate stride must be positive")
    if (args.query_margin < 0 or args.max_queries_per_episode < 0
            or args.return_query_margin < 0
            or args.max_return_queries_per_episode < 0
            or args.teacher_top_k < 0):
        raise ValueError("query margin/maximum must be non-negative")
    for required in (args.base_teacher_csv, args.cls_cache):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.out_csv.exists() or args.report.exists():
        raise FileExistsError(
            "output already exists; choose a new directory or remove it explicitly")
    base_sha = sha256(args.base_teacher_csv)
    if args.expected_base_sha and base_sha != args.expected_base_sha:
        raise RuntimeError(
            f"base teacher SHA mismatch: expected {args.expected_base_sha}, "
            f"got {base_sha}")
    frame = pd.read_csv(args.base_teacher_csv)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"base teacher CSV missing columns: {sorted(missing)}")
    train_scenes = tuple(sorted(set(args.train_scene)))
    evaluation_scenes = tuple(sorted(set(args.evaluation_scene)))
    overlap = set(train_scenes) & set(evaluation_scenes)
    if overlap:
        raise ValueError(
            f"scenes cannot be both training and evaluation: {sorted(overlap)}")
    selected_scenes = tuple(sorted(set(train_scenes) | set(evaluation_scenes)))
    if not selected_scenes:
        raise ValueError("at least one training or evaluation scene is required")
    absent = set(selected_scenes) - set(frame["scene"].unique())
    if absent:
        raise ValueError(f"selected scenes absent from base CSV: {sorted(absent)}")
    cls_by_path, cls_weight_sha = load_cls_cache(args.cls_cache)
    mappings = parse_path_maps(args.path_map)
    episodes = collect_episode_frames(frame, selected_scenes)

    readable = {}
    for paths in episodes.values():
        for raw in paths.values():
            readable[raw] = remap_path(raw, mappings)
    missing_images = [str(path) for path in readable.values() if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images missing; first={missing_images[0]}")
    absent_cls = set(readable) - set(cls_by_path)
    if absent_cls:
        raise ValueError(
            f"CLS cache lacks {len(absent_cls)} training frames")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out_csv.with_suffix(args.out_csv.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher()
    started = time.perf_counter()
    added_pairs = 0
    added_evaluated = 0
    added_positive = 0
    added_sessions = 0
    per_scene = {}

    fieldnames = list(frame.columns)
    with open(partial, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame.to_dict(orient="records"):
            writer.writerow(row)

        for scene in selected_scenes:
            role = "train" if scene in train_scenes else "evaluation"
            cross_kind = f"cross_episode_{role}"
            return_kind = f"within_episode_return_{role}"
            partner_by_episode = paired_episode_names(episodes, scene)
            scene_by_kind = {
                cross_kind: {
                    "sessions": 0, "pairs": 0, "evaluated": 0,
                    "positive": 0},
                return_kind: {
                    "sessions": 0, "pairs": 0, "evaluated": 0,
                    "positive": 0},
            }

            def write_session(*, episode: str, kind: str, session_id: str,
                              query_raw: str,
                              candidate_paths: Dict[int, str],
                              candidate_stride: int,
                              intrinsic: np.ndarray) -> None:
                nonlocal added_pairs, added_evaluated
                nonlocal added_positive, added_sessions
                query_features = sift_description(readable[query_raw], sift)
                session_positive = 0
                session_pairs = 0
                session_evaluated = 0
                candidate_frames = sorted(candidate_paths)[::candidate_stride]
                score_by_frame = {
                    candidate_frame: float(
                        cls_by_path[query_raw]
                        @ cls_by_path[candidate_paths[candidate_frame]])
                    for candidate_frame in candidate_frames}
                evaluated_frames = hard_teacher_frames(
                    candidate_frames,
                    [score_by_frame[frame] for frame in candidate_frames],
                    args.teacher_top_k)
                for candidate_frame in candidate_frames:
                    candidate_raw = candidate_paths[candidate_frame]
                    if candidate_frame in evaluated_frames:
                        geometry = geometric_teacher(
                            query_features, readable[candidate_raw],
                            intrinsic, sift, matcher)
                        passed = int(
                            geometry["matches"] >= 20
                            and geometry["inliers"] >= 12
                            and geometry["inlier_ratio"] >= 0.50)
                        session_evaluated += 1
                        session_positive += passed
                    else:
                        geometry = dict(
                            matches=0, inliers=0, inlier_ratio=0.0,
                            error="not evaluated outside hard top-K")
                        passed = -1
                    output = dict(
                        session_id=session_id,
                        scene=scene,
                        episode=episode,
                        kind=kind,
                        query_path=query_raw,
                        candidate_path=candidate_raw,
                        candidate_frame=int(candidate_frame),
                        teacher_pass=passed,
                        **geometry,
                        dino_cosine=score_by_frame[candidate_frame],
                    )
                    writer.writerow({name: output.get(name, "")
                                     for name in fieldnames})
                    session_pairs += 1
                counts = scene_by_kind[kind]
                counts["sessions"] += 1
                counts["pairs"] += session_pairs
                counts["evaluated"] += session_evaluated
                counts["positive"] += session_positive
                added_sessions += 1
                added_pairs += session_pairs
                added_evaluated += session_evaluated
                added_positive += session_positive
                print(
                    f"[router-teacher] {session_id} "
                    f"positive={session_positive}/"
                    f"{session_evaluated}/{session_pairs}",
                    flush=True)

            for episode, partner in sorted(partner_by_episode.items()):
                candidate_paths = episodes[(scene, episode)]
                partner_paths = episodes[(scene, partner)]
                first_readable = readable[candidate_paths[min(candidate_paths)]]
                episode_root = episode_root_from_rgb(first_readable)
                intrinsic = load_intrinsic(episode_root)
                switch, frame_count = load_episode_meta(episode_root)
                if frame_count != len(candidate_paths):
                    raise ValueError(
                        f"metadata/frame mismatch for {scene}/{episode}: "
                        f"{frame_count} != {len(candidate_paths)}")
                queries = query_indices(
                    len(partner_paths), args.query_stride,
                    args.query_margin, args.max_queries_per_episode)
                for query_frame in queries:
                    if query_frame not in partner_paths:
                        raise ValueError(
                            f"query frame {query_frame} absent from {scene}/{partner}")
                    query_raw = partner_paths[query_frame]
                    session_id = (
                        f"{scene}/{episode}/cross_{partner}_f{query_frame:05d}")
                    write_session(
                        episode=episode,
                        kind=cross_kind,
                        session_id=session_id,
                        query_raw=query_raw,
                        candidate_paths=candidate_paths,
                        candidate_stride=args.candidate_stride,
                        intrinsic=intrinsic)

                if args.include_return:
                    memory_paths = {
                        index: raw for index, raw in candidate_paths.items()
                        if index < switch}
                    return_queries = return_query_indices(
                        frame_count, switch, args.return_query_stride,
                        args.return_query_margin,
                        args.max_return_queries_per_episode)
                    for query_frame in return_queries:
                        session_id = (
                            f"{scene}/{episode}/return_f{query_frame:05d}")
                        write_session(
                            episode=episode,
                            kind=return_kind,
                            session_id=session_id,
                            query_raw=candidate_paths[query_frame],
                            candidate_paths=memory_paths,
                            candidate_stride=args.return_candidate_stride,
                            intrinsic=intrinsic)
            scene_pairs = sum(
                counts["pairs"] for counts in scene_by_kind.values())
            scene_positive = sum(
                counts["positive"] for counts in scene_by_kind.values())
            scene_evaluated = sum(
                counts["evaluated"] for counts in scene_by_kind.values())
            scene_sessions = sum(
                counts["sessions"] for counts in scene_by_kind.values())
            per_scene[scene] = {
                "sessions": scene_sessions,
                "pairs": scene_pairs,
                "evaluated": scene_evaluated,
                "positive": scene_positive,
                "role": role,
                "by_kind": scene_by_kind,
            }
    partial.replace(args.out_csv)
    elapsed = time.perf_counter() - started
    report = {
        "base_teacher_csv": str(args.base_teacher_csv.resolve()),
        "base_teacher_sha256": base_sha,
        "cls_cache": str(args.cls_cache.resolve()),
        "cls_weight_sha256": cls_weight_sha,
        "train_scenes": list(train_scenes),
        "evaluation_scenes": list(evaluation_scenes),
        "query_stride": args.query_stride,
        "query_margin": args.query_margin,
        "candidate_stride": args.candidate_stride,
        "max_queries_per_episode": args.max_queries_per_episode,
        "include_return": args.include_return,
        "return_query_stride": args.return_query_stride,
        "return_query_margin": args.return_query_margin,
        "return_candidate_stride": args.return_candidate_stride,
        "max_return_queries_per_episode": (
            args.max_return_queries_per_episode),
        "teacher_top_k": args.teacher_top_k,
        "base_pairs": int(len(frame)),
        "added_sessions": added_sessions,
        "added_pairs": added_pairs,
        "added_evaluated": added_evaluated,
        "added_unlabeled": added_pairs - added_evaluated,
        "added_positive": added_positive,
        "added_negative": added_evaluated - added_positive,
        "seconds": elapsed,
        "milliseconds_per_added_pair": (
            1000.0 * elapsed / max(added_pairs, 1)),
        "per_scene": per_scene,
        "output_csv": str(args.out_csv.resolve()),
        "output_csv_sha256": sha256(args.out_csv),
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
