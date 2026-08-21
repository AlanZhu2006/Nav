#!/usr/bin/env python3
"""Audit train-only selector headroom in the materialized Revisit-C candidates.

This is a post-hoc architecture diagnostic, not a model-selection gate.  It
compares frozen DINO top-1 with an oracle over the *already materialized*
production candidates.  The oracle is only an upper bound for a reranker; it
does not measure candidate generation outside this set or closed-loop SR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_PRODUCTION_SHA = (
    "193c29da7e2904061691361d5285d2211ff61b997619156f8b74262fde18237b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_summary(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "n": len(array),
        "median_deg": float(np.median(array)),
        "p90_deg": float(np.quantile(array, 0.9)),
        "cdf_le_15": int((array <= 15.0).sum()),
        "cdf_le_30": int((array <= 30.0).sum()),
        "cdf_le_45": int((array <= 45.0).sum()),
    }


def cluster_bootstrap(rows: pd.DataFrame, *, resamples: int,
                      seed: int) -> dict[str, Any]:
    scenes = sorted(rows["scene"].astype(str).unique())
    require(len(scenes) >= 2, "cluster bootstrap needs at least two scenes")
    by_scene = {scene: rows.loc[rows["scene"].eq(scene)] for scene in scenes}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        sample = pd.concat([by_scene[str(scene)] for scene in selected],
                           ignore_index=True)
        top1 = sample["dino_top1_error_deg"].le(30.0)
        oracle = sample["candidate_oracle_error_deg"].le(30.0)
        deltas.append(float(oracle.mean() - top1.mean()))
    return {
        "scenes": len(scenes),
        "resamples": resamples,
        "median": float(np.median(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
    }


def analyze(table: pd.DataFrame, *, bootstrap_resamples: int,
            bootstrap_seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "session_id", "scene", "candidate_frame", "dino_cosine",
        "cloud_overlap_f1_center",
        "relative_position_direction_error_deg_center", "causal_split_role",
        "causal_state_name", "causal_goal_variant",
    }
    require(not (required - set(table.columns)), "production columns missing")
    table = table.loc[
        table["causal_split_role"].astype(str).eq("train")
        & table["causal_state_name"].astype(str).eq("goal_c_t0")
        & table["causal_goal_variant"].astype(str).eq("factual")].copy()
    require(table["session_id"].nunique() == 80
            and table["scene"].nunique() == 40,
            "factual train Revisit-C universe changed")
    require(not table.duplicated(["session_id", "candidate_frame"]).any(),
            "duplicate candidate in a session")

    records = []
    for session_id, frame in table.groupby("session_id", sort=True):
        frame = frame.sort_values(
            ["dino_cosine", "candidate_frame"],
            ascending=[False, True], kind="stable").copy()
        errors = frame[
            "relative_position_direction_error_deg_center"].to_numpy(
                dtype=np.float64)
        require(bool(np.isfinite(errors).all()),
                f"non-finite direction error in {session_id}")
        clouds = frame["cloud_overlap_f1_center"].fillna(0.0).to_numpy(
            dtype=np.float64)
        cloud_index = int(np.argmax(clouds))
        records.append({
            "session_id": str(session_id),
            "scene": str(frame.iloc[0]["scene"]),
            "candidate_count": len(frame),
            "dino_top1_frame": int(frame.iloc[0]["candidate_frame"]),
            "dino_top1_error_deg": float(errors[0]),
            "cloud_top1_error_deg": float(errors[cloud_index]),
            "candidate_oracle_error_deg": float(errors.min()),
        })
    rows = pd.DataFrame(records)
    top1_good = rows["dino_top1_error_deg"].le(30.0)
    oracle_good = rows["candidate_oracle_error_deg"].le(30.0)
    require(bool((~top1_good | oracle_good).all()),
            "candidate oracle is worse than top-1")
    summary = {
        "scope": (
            "posthoc_train_only_materialized_candidate_headroom_not_sr"),
        "sessions": len(rows),
        "scenes": rows["scene"].nunique(),
        "candidate_count_distribution": {
            str(int(count)): int(frequency) for count, frequency in
            rows["candidate_count"].value_counts().sort_index().items()
        },
        "methods": {
            "dino_top1": error_summary(rows["dino_top1_error_deg"]),
            "cloud_overlap_top1": error_summary(
                rows["cloud_top1_error_deg"]),
            "materialized_candidate_oracle": error_summary(
                rows["candidate_oracle_error_deg"]),
        },
        "oracle_headroom_over_dino_at_30": {
            "recoverable_top1_failures": int((~top1_good & oracle_good).sum()),
            "unrecoverable_within_candidate_set": int((~oracle_good).sum()),
            "risk_difference": float(oracle_good.mean() - top1_good.mean()),
            "scene_cluster_bootstrap": cluster_bootstrap(
                rows, resamples=bootstrap_resamples, seed=bootstrap_seed),
        },
        "interpretation_contract": {
            "candidate_generation_ceiling": False,
            "cross_candidate_bearing_fusion": False,
            "closed_loop_or_sr": False,
            "training_authorization": False,
            "development_or_blind_read": False,
        },
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", default=EXPECTED_PRODUCTION_SHA)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    require(args.production_rows.is_file(), "production artifact missing")
    require(sha256_file(args.production_rows) == args.expected_sha,
            "production SHA mismatch")
    require(args.bootstrap_resamples >= 1, "bootstrap count must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary, rows = analyze(
        pd.read_csv(args.production_rows),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed)
    summary["input_sha256"] = sha256_file(args.production_rows)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    rows.to_csv(args.out / "rows.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
