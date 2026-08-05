#!/usr/bin/env python3
"""Audit retrieval candidate recall before training a memory reranker.

The input must contain task-aligned co-visibility for the complete candidate
pool.  The audit deliberately separates two questions:

1. does any usable memory frame exist for a query; and
2. does a particular candidate selector retain one of those frames?

This prevents true Novel queries from being counted as retrieval failures and
also exposes temporal redundancy, where many adjacent high-cosine frames fill
the top-K list.  The script is diagnostic-only and never exports a deployable
router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np


REQUIRED_COLUMNS = {
    "session_id", "scene", "kind", "candidate_path", "candidate_frame",
    "dino_cosine", "teacher_covis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int_list(values: Iterable[str], *, minimum: int) -> tuple[int, ...]:
    parsed = sorted({int(value) for value in values})
    if not parsed or parsed[0] < minimum:
        raise ValueError(f"values must be unique integers >= {minimum}")
    return tuple(parsed)


def cosine_order(group):
    return group.sort_values(
        ["dino_cosine", "candidate_frame", "candidate_path"],
        ascending=[False, True, True], kind="mergesort")


def select_raw_topk(group, top_k: int):
    return cosine_order(group).head(top_k)


def select_temporal_nms(group, top_k: int, min_frame_gap: int):
    """Greedily keep high-score candidates separated in trajectory time."""
    if top_k < 1 or min_frame_gap < 1:
        raise ValueError("top_k and min_frame_gap must be positive")
    selected_indices = []
    selected_frames: list[int] = []
    for row in cosine_order(group).itertuples():
        frame = int(row.candidate_frame)
        if all(abs(frame - other) >= min_frame_gap
               for other in selected_frames):
            selected_indices.append(row.Index)
            selected_frames.append(frame)
            if len(selected_indices) == top_k:
                break
    return group.loc[selected_indices]


def selector_summary(frame, selector: Callable, positive_threshold: float) -> dict:
    sessions = 0
    oracle_positive_sessions = 0
    selected_positive_sessions = 0
    selected_counts = []
    selected_best_overlap = []
    first_positive_ranks = []

    for _session_id, group in frame.groupby("session_id", sort=False):
        sessions += 1
        ordered = cosine_order(group)
        overlap = ordered["teacher_covis"].to_numpy(dtype=np.float64)
        positive_positions = np.flatnonzero(overlap >= positive_threshold)
        if positive_positions.size:
            oracle_positive_sessions += 1
            first_positive_ranks.append(int(positive_positions[0]) + 1)

        selected = selector(group)
        selected_overlap = selected["teacher_covis"].to_numpy(
            dtype=np.float64)
        selected_counts.append(len(selected))
        selected_best_overlap.append(
            float(selected_overlap.max()) if len(selected_overlap) else 0.0)
        if (positive_positions.size
                and np.any(selected_overlap >= positive_threshold)):
            selected_positive_sessions += 1

    recall = (selected_positive_sessions / oracle_positive_sessions
              if oracle_positive_sessions else None)
    return {
        "sessions": sessions,
        "oracle_positive_sessions": oracle_positive_sessions,
        "no_positive_in_candidate_pool": sessions - oracle_positive_sessions,
        "selected_positive_sessions": selected_positive_sessions,
        "conditional_candidate_recall": recall,
        "selected_count_mean": float(np.mean(selected_counts)),
        "selected_count_min": int(min(selected_counts)),
        "selected_best_covis_mean": float(np.mean(selected_best_overlap)),
        "raw_first_positive_rank_mean": (
            float(np.mean(first_positive_ranks))
            if first_positive_ranks else None),
        "raw_first_positive_rank_median": (
            float(np.median(first_positive_ranks))
            if first_positive_ranks else None),
    }


def strategy_audit(frame, top_ks: tuple[int, ...], nms_top_k: int,
                   nms_gaps: tuple[int, ...],
                   positive_threshold: float) -> dict:
    result = {"raw": {}, "temporal_nms": {}}
    for top_k in top_ks:
        result["raw"][str(top_k)] = selector_summary(
            frame, lambda group, k=top_k: select_raw_topk(group, k),
            positive_threshold)
    for gap in nms_gaps:
        result["temporal_nms"][f"k{nms_top_k}_gap{gap}"] = selector_summary(
            frame,
            lambda group, k=nms_top_k, g=gap: select_temporal_nms(
                group, k, g),
            positive_threshold)
    return result


def grouped_audit(frame, column: str, top_ks: tuple[int, ...],
                  nms_top_k: int,
                  nms_gaps: tuple[int, ...],
                  positive_threshold: float) -> dict:
    return {
        str(value): strategy_audit(
            group, top_ks, nms_top_k, nms_gaps, positive_threshold)
        for value, group in frame.groupby(column, sort=True)
    }


def load_scene_roles(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    roles = {}
    for role in ("train", "development", "final_reserved"):
        for scene in manifest.get(role, []):
            if scene in roles:
                raise ValueError(f"scene occurs in multiple roles: {scene}")
            roles[str(scene)] = role
    return roles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--top-k", action="append", default=[],
        help="raw cosine K to audit; repeatable (default: 1,4,8,16,32,64,128)")
    parser.add_argument(
        "--nms-gap", action="append", default=[],
        help="minimum frame gap for temporal-NMS selection; repeatable")
    parser.add_argument("--nms-top-k", type=int, default=32)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if not args.teacher_csv.is_file():
        raise FileNotFoundError(args.teacher_csv)
    if args.split_manifest is not None and not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    if (not np.isfinite(args.positive_threshold)
            or not 0.0 < args.positive_threshold <= 1.0):
        raise ValueError("positive threshold must lie in (0, 1]")
    top_ks = parse_int_list(
        args.top_k or ("1", "4", "8", "16", "32", "64", "128"),
        minimum=1)
    nms_gaps = parse_int_list(
        args.nms_gap or ("4", "8", "16", "32"), minimum=1)
    if args.nms_top_k < 1:
        raise ValueError("nms top-k must be positive")

    frame = pd.read_csv(args.teacher_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    if frame.duplicated(["session_id", "candidate_path"]).any():
        raise ValueError("teacher CSV contains duplicate session/candidate pairs")
    covis = frame["teacher_covis"].to_numpy(dtype=np.float64)
    if not np.isfinite(covis).all():
        raise ValueError(
            "candidate pool is not fully labeled; relabel every candidate "
            "before auditing oracle recall")
    if np.any((covis < 0.0) | (covis > 1.0)):
        raise ValueError("teacher co-visibility must lie in [0, 1]")

    scene_roles = load_scene_roles(args.split_manifest)
    if scene_roles:
        unknown = set(frame["scene"].astype(str)) - set(scene_roles)
        if unknown:
            raise ValueError(f"scenes absent from split manifest: {sorted(unknown)}")
        leaked = set(frame.loc[
            frame["scene"].map(scene_roles).eq("final_reserved"), "scene"])
        if leaked:
            raise RuntimeError(
                f"final-reserved scenes leaked into development audit: {sorted(leaked)}")
        frame["split_role"] = frame["scene"].map(scene_roles)
    else:
        frame["split_role"] = "unspecified"

    started = time.perf_counter()
    report = {
        "deployment_approved": False,
        "purpose": "offline candidate-recall audit",
        "created_at_unix": time.time(),
        "teacher_csv": str(args.teacher_csv.resolve()),
        "teacher_csv_sha256": sha256(args.teacher_csv),
        "split_manifest": (
            str(args.split_manifest.resolve())
            if args.split_manifest is not None else None),
        "candidate_pool_complete": True,
        "positive_threshold": args.positive_threshold,
        "top_k": list(top_ks),
        "temporal_nms_top_k": args.nms_top_k,
        "temporal_nms_gaps": list(nms_gaps),
        "pairs": int(len(frame)),
        "sessions": int(frame["session_id"].nunique()),
        "overall": strategy_audit(
            frame, top_ks, args.nms_top_k, nms_gaps,
            args.positive_threshold),
        "by_split_role": grouped_audit(
            frame, "split_role", top_ks, args.nms_top_k, nms_gaps,
            args.positive_threshold),
        "by_kind": grouped_audit(
            frame, "kind", top_ks, args.nms_top_k, nms_gaps,
            args.positive_threshold),
        "by_scene": grouped_audit(
            frame, "scene", top_ks, args.nms_top_k, nms_gaps,
            args.positive_threshold),
    }
    report["seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise FileExistsError(args.report)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "report": str(args.report),
        "overall": report["overall"],
        "by_split_role": report["by_split_role"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
