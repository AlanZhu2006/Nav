#!/usr/bin/env python3
"""Replace generic SIFT router labels with task-aligned 3D co-visibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from MemNavData.covisibility_teacher import (
        EpisodeCovisibilityCache,
        covisibility_label,
        parse_path_maps,
    )
except ModuleNotFoundError:  # direct script invocation
    from covisibility_teacher import (  # type: ignore
        EpisodeCovisibilityCache,
        covisibility_label,
        parse_path_maps,
    )


REQUIRED_COLUMNS = {
    "session_id", "scene", "episode", "kind", "query_path",
    "candidate_path", "candidate_frame", "teacher_pass", "dino_cosine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_indices(frame, top_k: int):
    ordered = frame.sort_values(
        ["session_id", "dino_cosine", "candidate_frame"],
        ascending=[True, False, True], kind="mergesort")
    if top_k == 0:
        return ordered.index
    return ordered.groupby("session_id", sort=False).head(top_k).index


def confusion(target: np.ndarray, prediction: np.ndarray) -> dict:
    selected = np.isin(target, [0, 1]) & np.isin(prediction, [0, 1])
    target, prediction = target[selected], prediction[selected]
    true_positive = int(np.sum((target == 1) & (prediction == 1)))
    false_positive = int(np.sum((target == 0) & (prediction == 1)))
    false_negative = int(np.sum((target == 1) & (prediction == 0)))
    true_negative = int(np.sum((target == 0) & (prediction == 0)))
    return {
        "evaluated": int(selected.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (true_positive / (true_positive + false_positive)
                      if true_positive + false_positive else None),
        "recall": (true_positive / (true_positive + false_negative)
                   if true_positive + false_negative else None),
    }


def grouped_counts(frame, columns) -> dict:
    result = {}
    for key, group in frame.groupby(columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        name = "/".join(map(str, key))
        result[name] = {
            "pairs": int(len(group)),
            "positive": int(group["teacher_pass"].eq(1).sum()),
            "negative": int(group["teacher_pass"].eq(0).sum()),
            "ignored": int(group["teacher_pass"].eq(-1).sum()),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument(
        "--top-k", type=int, default=32,
        help="DINO candidates to relabel per session; 0 labels the complete pool")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.1)
    parser.add_argument("--depth-stride", type=int, default=6)
    parser.add_argument("--depth-tolerance", type=float, default=0.3)
    parser.add_argument("--depth-cache-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    # Validate thresholds before doing any expensive I/O.
    covisibility_label(
        0.0, args.positive_threshold, args.negative_threshold)
    frame = pd.read_csv(args.input_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {sorted(missing)}")
    if frame.duplicated(["session_id", "candidate_path"]).any():
        raise ValueError("input CSV contains duplicate session/candidate pairs")

    old_teacher = frame["teacher_pass"].to_numpy(dtype=np.int8).copy()
    frame["sift_teacher_pass"] = old_teacher
    frame["teacher_pass"] = -1
    frame["teacher_covis"] = np.nan
    frame["teacher_source"] = "not_evaluated_outside_top_k"
    selected = selected_indices(frame, args.top_k)
    cache = EpisodeCovisibilityCache(
        parse_path_maps(args.path_map),
        depth_cache_size=args.depth_cache_size,
        stride=args.depth_stride,
        tolerance=args.depth_tolerance)

    started = time.perf_counter()
    for count, index in enumerate(selected, 1):
        row = frame.loc[index]
        score, source = cache.pair_covisibility(
            row["query_path"], row["candidate_path"],
            int(row["candidate_frame"]))
        frame.at[index, "teacher_covis"] = score
        frame.at[index, "teacher_source"] = source
        frame.at[index, "teacher_pass"] = covisibility_label(
            score, args.positive_threshold, args.negative_threshold)
        if count == 1 or count % 1000 == 0:
            print(f"[covis-teacher] {count}/{len(selected)}", flush=True)
    elapsed = time.perf_counter() - started

    selected_frame = frame.loc[selected].copy()
    selected_frame["candidate_rank"] = (
        selected_frame.groupby("session_id", sort=False).cumcount() + 1)
    session_groups = selected_frame.groupby("session_id", sort=False)
    sessions_with_positive = int(sum(
        group["teacher_pass"].eq(1).any()
        for _, group in session_groups))
    top1_positive = int(selected_frame.loc[
        selected_frame["candidate_rank"].eq(1), "teacher_pass"].eq(1).sum())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_csv.with_suffix(args.output_csv.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, args.output_csv)

    report = {
        "created_at_unix": time.time(),
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": sha256(args.input_csv),
        "output_csv": str(args.output_csv.resolve()),
        "output_sha256": sha256(args.output_csv),
        "top_k": args.top_k,
        "candidate_pool_complete": bool(len(selected_frame) == len(frame)),
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
        "depth_stride": args.depth_stride,
        "depth_tolerance": args.depth_tolerance,
        "selected_pairs": int(len(selected_frame)),
        "selected_positive": int(selected_frame["teacher_pass"].eq(1).sum()),
        "selected_negative": int(selected_frame["teacher_pass"].eq(0).sum()),
        "selected_ignored": int(selected_frame["teacher_pass"].eq(-1).sum()),
        "sessions": int(selected_frame["session_id"].nunique()),
        "sessions_with_positive_top_k": sessions_with_positive,
        "top1_positive": top1_positive,
        "seconds": elapsed,
        "milliseconds_per_selected_pair": (
            1000.0 * elapsed / max(len(selected_frame), 1)),
        "sift_vs_covis_extremes": confusion(
            selected_frame["teacher_pass"].to_numpy(dtype=np.int8),
            selected_frame["sift_teacher_pass"].to_numpy(dtype=np.int8)),
        "by_kind": grouped_counts(selected_frame, ["kind"]),
        "by_scene": grouped_counts(selected_frame, ["scene"]),
        "by_source": grouped_counts(selected_frame, ["teacher_source"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
