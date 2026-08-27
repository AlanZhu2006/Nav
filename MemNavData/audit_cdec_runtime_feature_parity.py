#!/usr/bin/env python3
"""Audit exact train/runtime CDEC features through the production LingBot API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from MemNavData.build_cdec_patch_cache import safe_relative
from MemNavData.cdec_pairwise_runtime import (
    CDECPairwiseRanker,
    pad_dino_image_batch,
    pool_dino_patch_tokens,
    relation_feature_matrix,
    sha256,
)
from NavDP.baselines.memnav.policy_agent import MemNavAgent


SCHEMA_VERSION = "cdec_runtime_feature_parity_v1_20260813"


def deterministic_sessions(session_ids, count: int) -> list[str]:
    unique = sorted({str(value) for value in session_ids})
    count = int(count)
    if count < 1 or count > len(unique):
        raise ValueError("sample count is outside the session universe")
    return sorted(unique, key=lambda value: hashlib.sha256(
        f"cdec-runtime-parity:{value}".encode("utf-8")).hexdigest())[:count]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--patch-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--internnav-root", type=Path, required=True)
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    receipts = {
        "rows_csv_sha256": sha256(args.rows_csv),
        "patch_cache_sha256": sha256(args.patch_cache),
        "artifact_sha256": sha256(args.artifact),
        "checkpoint_sha256": sha256(args.checkpoint),
    }
    expected = {
        "rows_csv_sha256": args.expected_rows_sha256,
        "patch_cache_sha256": args.expected_cache_sha256,
        "artifact_sha256": args.expected_artifact_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
    }
    if receipts != expected:
        raise RuntimeError(f"input receipt mismatch: {receipts} != {expected}")

    frame = pd.read_csv(args.rows_csv)
    selected = deterministic_sessions(frame["session_id"], args.sessions)
    cache = np.load(args.patch_cache, allow_pickle=False)
    paths = cache["relative_paths"].astype(str)
    if len(set(paths.tolist())) != len(paths):
        raise RuntimeError("patch cache path universe is not unique")
    path_index = {value: index for index, value in enumerate(paths)}
    cached_relation = cache["directional_relation"].astype(np.float64)
    if len(cached_relation) != len(frame):
        raise RuntimeError("cache relation rows do not align with CSV")
    ranker = CDECPairwiseRanker(args.artifact, allow_unapproved=True)
    agent = MemNavAgent(
        checkpoint=str(args.checkpoint),
        internnav_root=str(args.internnav_root),
        device=args.device,
        buffer_root=str(args.buffer_root),
        flow_gate="off",
        cdec_pairwise_ranker=ranker,
    )

    token_equal = 0
    token_total = 0
    relation_max_error = 0.0
    score_max_error = 0.0
    top1_equal = 0
    session_receipts = []
    for session_id in selected:
        rows = frame.loc[frame["session_id"].astype(str).eq(session_id)]
        if len(rows) != 8:
            raise RuntimeError(f"session does not contain top-8: {session_id}")
        query_values = rows["query_relative_path"].astype(str).unique()
        if len(query_values) != 1:
            raise RuntimeError(f"query path changed within session: {session_id}")
        relative = [str(query_values[0])] + rows[
            "candidate_relative_path"].astype(str).tolist()
        physical = [args.image_root / safe_relative(value) for value in relative]
        if not all(path.is_file() for path in physical):
            missing = next(path for path in physical if not path.is_file())
            raise FileNotFoundError(missing)
        images = agent.lb.load_images([str(path) for path in physical])
        padded, real_count = pad_dino_image_batch(images)
        with torch.inference_mode():
            patch = agent.lb.dino(padded)["patch"][:real_count]
        online_tokens = pool_dino_patch_tokens(patch)
        expected_tokens = cache["tokens"][[path_index[value] for value in relative]]
        token_equal += int(np.sum(online_tokens == expected_tokens))
        token_total += int(np.prod(expected_tokens.shape))

        features = relation_feature_matrix(
            online_tokens[0], online_tokens[1:],
            rows["dino_cosine"].to_numpy(dtype=np.float64))
        row_indices = rows.index.to_numpy(dtype=np.int64)
        expected_features = cached_relation[row_indices]
        relation_error = float(np.max(np.abs(features - expected_features)))
        relation_max_error = max(relation_max_error, relation_error)
        online_scores = ranker.score_features(features)
        expected_scores = ranker.score_features(expected_features)
        score_error = float(np.max(np.abs(online_scores - expected_scores)))
        score_max_error = max(score_max_error, score_error)
        same_top1 = int(np.argmax(online_scores)) == int(np.argmax(expected_scores))
        top1_equal += int(same_top1)
        session_receipts.append({
            "session_id": session_id,
            "token_exact": bool(np.array_equal(
                online_tokens, expected_tokens)),
            "relation_max_abs_error": relation_error,
            "score_max_abs_error": score_error,
            "top1_equal": same_top1,
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if (
            token_equal == token_total
            and relation_max_error == 0.0
            and score_max_error == 0.0
            and top1_equal == len(selected)) else "fail",
        "question": (
            "Does the production MemNav LingBot API reproduce the fixed-batch "
            "training cache and CDEC scores exactly?"),
        "sample_selection": "SHA256 order over session_id; label blind",
        "sessions": len(selected),
        "images_with_repetition": 9 * len(selected),
        "token_values": token_total,
        "token_exact_values": token_equal,
        "token_exact_fraction": token_equal / token_total,
        "relation_max_abs_error": relation_max_error,
        "score_max_abs_error": score_max_error,
        "top1_exact_agreement": top1_equal,
        "fixed_dino_batch_size": 16,
        "development_or_blind_read": False,
        "inputs": receipts,
        "session_receipts": session_receipts,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
