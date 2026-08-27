#!/usr/bin/env python3
"""Merge exact-state PnP arrival shards and apply the frozen train gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SCHEMA_VERSION = "lingbot_pnp_arrival_summary_v1"
DISTANCE_THRESHOLDS_M = (
    0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50,
)
PRIMARY_GATE = {
    "trigger": "native_selected_zero_sample0",
    "required_false_positives": 0,
    "minimum_true_positives": 20,
    "minimum_true_positive_scenes": 10,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        handle.write(data)
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


def as_bool(series: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool).to_numpy()
    normalized = series.astype(str).str.strip().str.lower()
    require(bool(normalized.isin({"true", "false", "1", "0"}).all()),
            f"{label} is not boolean")
    return normalized.isin({"true", "1"}).to_numpy()


def confusion(labels: np.ndarray, prediction: np.ndarray,
              scenes: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp_mask = labels & prediction
    fp_mask = ~labels & prediction
    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int((labels & ~prediction).sum())
    tn = int((~labels & ~prediction).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "true_positive_scenes": int(len(set(scenes[tp_mask]))),
        "false_positive_scenes": int(len(set(scenes[fp_mask]))),
    }


def summarize(rows: pd.DataFrame, expected_states: pd.DataFrame) -> dict:
    required = {
        "state_id", "scene", "episode", "euclidean_distance_m",
        "arrival_025_strict", "native_selected_zero_sample0",
        "certificate_accepted", "predicted_distance_m",
    }
    require(not (required - set(rows.columns)),
            f"rows missing {sorted(required - set(rows.columns))}")
    require(rows["state_id"].is_unique, "duplicate collected state IDs")
    require(set(rows["state_id"].astype(str))
            == set(expected_states["state_id"].astype(str)),
            "collected rows do not exactly cover frozen states")
    expected = expected_states.set_index("state_id")
    ordered = rows.sort_values("state_id", kind="stable").reset_index(drop=True)
    for row in ordered.itertuples():
        source = expected.loc[str(row.state_id)]
        require(str(row.scene) == str(source["scene"]),
                f"scene changed for {row.state_id}")
        require(str(row.episode) == str(source["episode"]),
                f"episode changed for {row.state_id}")
        require(math.isclose(
            float(row.euclidean_distance_m),
            float(source["euclidean_distance_m"]), abs_tol=1e-9),
            f"GT distance changed for {row.state_id}")

    labels = as_bool(ordered["arrival_025_strict"], "arrival_025_strict")
    certificate = as_bool(
        ordered["certificate_accepted"], "certificate_accepted")
    native_trigger = as_bool(
        ordered["native_selected_zero_sample0"],
        "native_selected_zero_sample0")
    distances = pd.to_numeric(
        ordered["predicted_distance_m"], errors="coerce").to_numpy(float)
    finite = np.isfinite(distances)
    scenes = ordered["scene"].astype(str).to_numpy()

    points = []
    for threshold in DISTANCE_THRESHOLDS_M:
        pnp_only = certificate & finite & (distances <= threshold)
        triggered = native_trigger & pnp_only
        points.append({
            "predicted_distance_max_m": threshold,
            "pnp_only": confusion(labels, pnp_only, scenes),
            "native_zero_plus_pnp": confusion(labels, triggered, scenes),
            "near_miss_025_050_false_positives": int((
                triggered
                & (ordered["euclidean_distance_m"].to_numpy(float) >= 0.25)
                & (ordered["euclidean_distance_m"].to_numpy(float) <= 0.50)
            ).sum()),
        })

    passing = []
    for point in points:
        stats = point["native_zero_plus_pnp"]
        if (stats["fp"] == PRIMARY_GATE["required_false_positives"]
                and stats["tp"] >= PRIMARY_GATE["minimum_true_positives"]
                and stats["true_positive_scenes"]
                >= PRIMARY_GATE["minimum_true_positive_scenes"]):
            passing.append(point)
    winner = None
    if passing:
        # Frozen tie-break: maximize recovered arrivals, then scene coverage,
        # then choose the smaller (more conservative) distance threshold.
        winner = sorted(
            passing,
            key=lambda item: (
                -item["native_zero_plus_pnp"]["tp"],
                -item["native_zero_plus_pnp"]["true_positive_scenes"],
                item["predicted_distance_max_m"],
            ),
        )[0]

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only exact-state mechanism audit",
        "method_or_goat_threshold_authorized": False,
        "state_count": int(len(ordered)),
        "scene_count": int(ordered["scene"].nunique()),
        "episode_count": int(ordered[["scene", "episode"]]
                             .drop_duplicates().shape[0]),
        "arrival_state_count": int(labels.sum()),
        "nonarrival_state_count": int((~labels).sum()),
        "native_sample0_zero_count": int(native_trigger.sum()),
        "precheck_pass_count": int(as_bool(
            ordered["precheck_passed"], "precheck_passed").sum()),
        "certificate_accept_count": int(certificate.sum()),
        "certificate_accept_arrival_count": int((certificate & labels).sum()),
        "certificate_accept_nonarrival_count": int((certificate & ~labels).sum()),
        "predeclared_distance_thresholds_m": list(DISTANCE_THRESHOLDS_M),
        "operating_points": points,
        "primary_gate": PRIMARY_GATE,
        "primary_gate_passed": winner is not None,
        "selected_train_operating_point": winner,
        "next_action": (
            "freeze_and_test_on_disjoint_goat_without_retuning"
            if winner is not None else
            "zero_trajectory_remains_abstain_never_stop"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-csv", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.expected_shards >= 1, "expected-shards must be positive")
    require(not args.out_dir.exists(), f"output already exists: {args.out_dir}")
    paths = sorted(args.shard_root.glob("shard_*/rows.csv"))
    require(len(paths) == args.expected_shards,
            f"expected {args.expected_shards} shards, found {len(paths)}")
    reports = []
    frames = []
    for path in paths:
        report_path = path.with_name("report.json")
        require(report_path.is_file(), f"missing shard report {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report.get("status") == "complete", "incomplete shard")
        reports.append({
            "path": str(path.resolve()),
            "rows_sha256": sha256_file(path),
            "report_sha256": sha256_file(report_path),
            "shard_index": int(report["shard_index"]),
            "shard_count": int(report["shard_count"]),
        })
        frames.append(pd.read_csv(path))
    require(sorted(item["shard_index"] for item in reports)
            == list(range(args.expected_shards)), "shard index cover changed")
    require(all(item["shard_count"] == args.expected_shards
                for item in reports), "shard count declaration changed")
    rows = pd.concat(frames, ignore_index=True)
    expected = pd.read_csv(args.states_csv)
    report = summarize(rows, expected)
    args.out_dir.mkdir(parents=True)
    merged = rows.sort_values("state_id", kind="stable")
    atomic_csv(args.out_dir / "rows.csv", merged)
    report.update({
        "states_csv": str(args.states_csv.resolve()),
        "states_sha256": sha256_file(args.states_csv),
        "shards": reports,
    })
    atomic_json(args.out_dir / "report.json", report)
    atomic_json(args.out_dir / "SHA256SUMS.json", {
        name: sha256_file(args.out_dir / name)
        for name in ("rows.csv", "report.json")
    })
    (args.out_dir / "SEALED").touch(exist_ok=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
