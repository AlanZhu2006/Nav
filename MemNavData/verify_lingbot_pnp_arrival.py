#!/usr/bin/env python3
"""Independent verifier for the sealed LingBot/PnP arrival audit."""

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


SCHEMA_VERSION = "lingbot_pnp_arrival_independent_verification_v1"
THRESHOLDS = (
    0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50,
)


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


def bools(series: pd.Series, label: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    require(bool(normalized.isin({"true", "false", "1", "0"}).all()),
            f"{label} is not boolean")
    return normalized.isin({"true", "1"}).to_numpy()


def counts(labels: np.ndarray, prediction: np.ndarray,
           scenes: np.ndarray) -> dict:
    tp = labels & prediction
    fp = ~labels & prediction
    return {
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "fn": int((labels & ~prediction).sum()),
        "tn": int((~labels & ~prediction).sum()),
        "precision": (
            float(tp.sum() / prediction.sum()) if prediction.sum() else None),
        "recall": float(tp.sum() / labels.sum()) if labels.sum() else None,
        "true_positive_scenes": int(len(set(scenes[tp]))),
        "false_positive_scenes": int(len(set(scenes[fp]))),
    }


def close(left: object, right: object, label: str) -> None:
    if left is None or right is None:
        require(left is right, f"{label} None mismatch")
    elif isinstance(left, (float, np.floating)) \
            or isinstance(right, (float, np.floating)):
        require(math.isclose(float(left), float(right), abs_tol=1e-12),
                f"{label} differs: {left} vs {right}")
    else:
        require(left == right, f"{label} differs: {left} vs {right}")


def verify(root: Path, states_path: Path) -> dict:
    require((root / "SEALED").is_file(), "merged result is not sealed")
    rows_path = root / "rows.csv"
    report_path = root / "report.json"
    sums_path = root / "SHA256SUMS.json"
    for path in (rows_path, report_path, sums_path, states_path):
        require(path.is_file(), f"missing verifier input {path}")
    sums = json.loads(sums_path.read_text(encoding="utf-8"))
    require(sums.get("rows.csv") == sha256_file(rows_path),
            "sealed row hash differs")
    require(sums.get("report.json") == sha256_file(report_path),
            "sealed report hash differs")

    rows = pd.read_csv(rows_path)
    states = pd.read_csv(states_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(rows["state_id"].is_unique, "duplicate result state")
    require(states["state_id"].is_unique, "duplicate frozen state")
    require(set(rows["state_id"].astype(str))
            == set(states["state_id"].astype(str)), "state cover differs")
    require(report.get("states_sha256") == sha256_file(states_path),
            "report points to a different frozen state file")

    ordered = rows.sort_values("state_id", kind="stable").reset_index(drop=True)
    labels = ordered["euclidean_distance_m"].to_numpy(float) < 0.25
    require(np.array_equal(
        labels, bools(ordered["arrival_025_strict"], "arrival label")),
        "arrival labels are not strict GT <0.25")
    certificate = bools(ordered["certificate_accepted"], "certificate")
    native = bools(
        ordered["native_selected_zero_sample0"], "native zero trigger")
    distances = pd.to_numeric(
        ordered["predicted_distance_m"], errors="coerce").to_numpy(float)
    finite = np.isfinite(distances)
    scenes = ordered["scene"].astype(str).to_numpy()
    gt = ordered["euclidean_distance_m"].to_numpy(float)

    expected_points = report.get("operating_points")
    require(isinstance(expected_points, list)
            and len(expected_points) == len(THRESHOLDS),
            "operating-point grid length differs")
    passing = []
    recomputed = []
    for threshold, expected in zip(THRESHOLDS, expected_points):
        close(expected.get("predicted_distance_max_m"), threshold,
              "distance threshold")
        pnp = certificate & finite & (distances <= threshold)
        primary = native & pnp
        actual = {
            "predicted_distance_max_m": threshold,
            "pnp_only": counts(labels, pnp, scenes),
            "native_zero_plus_pnp": counts(labels, primary, scenes),
            "near_miss_025_050_false_positives": int((
                primary & (gt >= 0.25) & (gt <= 0.50)).sum()),
        }
        for branch in ("pnp_only", "native_zero_plus_pnp"):
            for key, value in actual[branch].items():
                close(expected[branch].get(key), value,
                      f"{threshold}/{branch}/{key}")
        close(expected.get("near_miss_025_050_false_positives"),
              actual["near_miss_025_050_false_positives"],
              f"{threshold}/near-miss FP")
        stats = actual["native_zero_plus_pnp"]
        if (stats["fp"] == 0 and stats["tp"] >= 20
                and stats["true_positive_scenes"] >= 10):
            passing.append(actual)
        recomputed.append(actual)

    winner = None
    if passing:
        winner = sorted(passing, key=lambda item: (
            -item["native_zero_plus_pnp"]["tp"],
            -item["native_zero_plus_pnp"]["true_positive_scenes"],
            item["predicted_distance_max_m"],
        ))[0]
    require(bool(report.get("primary_gate_passed")) == (winner is not None),
            "primary gate decision differs")
    close(report.get("selected_train_operating_point"), winner,
          "selected train operating point")
    require(int(report.get("state_count")) == len(rows),
            "reported state count differs")
    require(int(report.get("arrival_state_count")) == int(labels.sum()),
            "reported arrival count differs")
    require(int(report.get("nonarrival_state_count")) == int((~labels).sum()),
            "reported nonarrival count differs")
    require(report.get("method_or_goat_threshold_authorized") is False,
            "train report improperly authorized deployment")

    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "row_count": int(len(rows)),
        "scene_count": int(rows["scene"].nunique()),
        "arrival_state_count": int(labels.sum()),
        "nonarrival_state_count": int((~labels).sum()),
        "primary_gate_passed": winner is not None,
        "selected_train_operating_point": winner,
        "rows_sha256": sha256_file(rows_path),
        "report_sha256": sha256_file(report_path),
        "states_sha256": sha256_file(states_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--states-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), f"verification output exists: {args.out}")
    result = verify(args.merged_root, args.states_csv)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
