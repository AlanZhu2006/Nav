#!/usr/bin/env python3
"""Paired, reporting-only comparison of two Pi3X bridge-density shadows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_shadow(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        index = int(row["row_index"])
        if index in rows:
            raise ValueError(f"duplicate row_index {index} in {path}")
        rows[index] = row
    return rows


def _paired_binary(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("paired outcomes have unequal lengths")
    gain = sum((not a) and b for a, b in zip(first, second))
    loss = sum(a and (not b) for a, b in zip(first, second))
    discordant = gain + loss
    p_value = (
        float(binomtest(min(gain, loss), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return {
        "n": len(first),
        "first_success": int(sum(first)),
        "second_success": int(sum(second)),
        "gain": int(gain),
        "loss": int(loss),
        "exact_mcnemar_p": p_value,
        "risk_difference": (
            (sum(second) - sum(first)) / len(first) if first else math.nan
        ),
    }


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(array),
        "median_deg": float(np.median(array)) if len(array) else math.nan,
        "q90_deg": float(np.quantile(array, 0.90)) if len(array) else math.nan,
        "within_15deg": int((array <= 15.0).sum()),
        "within_30deg": int((array <= 30.0).sum()),
        "within_45deg": int((array <= 45.0).sum()),
        "catastrophic_gt90deg": int((array > 90.0).sum()),
    }


def _scene_cluster_ci(
    records: Sequence[tuple[str, bool, bool]], *, draws: int = 20_000
) -> list[float]:
    grouped: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for scene, first, second in records:
        grouped[scene].append((first, second))
    scenes = sorted(grouped)
    if not scenes:
        return [math.nan, math.nan]
    generator = np.random.default_rng(0)
    differences = []
    for _ in range(draws):
        sampled = generator.choice(scenes, size=len(scenes), replace=True)
        values = [pair for scene in sampled for pair in grouped[str(scene)]]
        differences.append(np.mean([b for _, b in values]) - np.mean([a for a, _ in values]))
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_rows_sha256 and _sha256(args.rows_csv) != args.expected_rows_sha256:
        raise ValueError("source rows SHA mismatch")
    with args.rows_csv.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    first, second = _load_shadow(args.first), _load_shadow(args.second)
    expected = set(range(len(source)))
    if set(first) != expected or set(second) != expected:
        raise ValueError("shadow row_index universe differs from source rows")
    if args.expected_rows is not None and len(source) != args.expected_rows:
        raise ValueError(f"found {len(source)} source rows, expected {args.expected_rows}")

    for index, row in enumerate(source):
        for shadow in (first, second):
            if shadow[index]["scene"] != row["scene"]:
                raise ValueError(f"scene mismatch at row {index}")

    positive_candidates = [
        index for index, row in enumerate(source)
        if int(row["candidate_label"]) == 1
    ]
    first_errors = {
        index: float(row["goal_bearing_error_deg_reporting_only"])
        for index, row in first.items()
    }
    second_errors = {
        index: float(row["goal_bearing_error_deg_reporting_only"])
        for index, row in second.items()
    }
    candidate_first = [first_errors[index] <= 30.0 for index in positive_candidates]
    candidate_second = [second_errors[index] <= 30.0 for index in positive_candidates]

    sessions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(source):
        sessions[row["session_id"]].append(index)
    positive_sessions = [
        (session_id, indices) for session_id, indices in sorted(sessions.items())
        if int(source[indices[0]]["session_label"]) == 1
    ]

    def session_outcomes(errors: dict[int, float], mode: str) -> list[bool]:
        outcomes = []
        for _, indices in positive_sessions:
            if mode == "top8_ceiling":
                outcomes.append(any(errors[index] <= 30.0 for index in indices))
            elif mode == "raw_dino_top1":
                selected = max(
                    indices,
                    key=lambda index: (
                        float(source[index]["dino_cosine"]),
                        -int(source[index]["candidate_rank"]),
                    ),
                )
                outcomes.append(errors[selected] <= 30.0)
            else:
                raise ValueError(mode)
        return outcomes

    session_metrics = {}
    for mode in ("top8_ceiling", "raw_dino_top1"):
        a = session_outcomes(first_errors, mode)
        b = session_outcomes(second_errors, mode)
        paired = _paired_binary(a, b)
        paired["scene_cluster_bootstrap_95"] = _scene_cluster_ci([
            (source[indices[0]]["scene"], first_value, second_value)
            for (_, indices), first_value, second_value
            in zip(positive_sessions, a, b)
        ])
        session_metrics[mode] = paired

    gap_bins = {}
    for name, lower, upper in (
        ("0_32", 0, 32),
        ("33_96", 33, 96),
        ("97_200", 97, 200),
        ("201_400", 201, 400),
        ("gt400", 401, math.inf),
    ):
        indices = [
            index for index in positive_candidates
            if lower <= (
                int(source[index]["decision_frame"])
                - int(source[index]["candidate_frame"])
            ) <= upper
        ]
        gap_bins[name] = {
            "first": _error_summary([first_errors[index] for index in indices]),
            "second": _error_summary([second_errors[index] for index in indices]),
            "paired_within_30deg": _paired_binary(
                [first_errors[index] <= 30.0 for index in indices],
                [second_errors[index] <= 30.0 for index in indices],
            ),
        }

    result = {
        "schema_version": 1,
        "status": "paired_offline_mechanism_only_not_closed_loop",
        "rows": len(source),
        "scenes": len({row["scene"] for row in source}),
        "sessions": len(sessions),
        "positive_candidates": {
            "first": _error_summary([first_errors[index] for index in positive_candidates]),
            "second": _error_summary([second_errors[index] for index in positive_candidates]),
            "paired_within_30deg": _paired_binary(candidate_first, candidate_second),
        },
        "positive_sessions": len(positive_sessions),
        "session_metrics": session_metrics,
        "positive_candidate_gap_bins": gap_bins,
        "inputs": {
            "rows_csv": str(args.rows_csv),
            "rows_csv_sha256": _sha256(args.rows_csv),
            "first": str(args.first),
            "first_sha256": _sha256(args.first),
            "second": str(args.second),
            "second_sha256": _sha256(args.second),
        },
    }
    _atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-rows-sha256")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps({
        "status": summary["status"],
        "positive_candidates": summary["positive_candidates"]["paired_within_30deg"],
        "session_metrics": summary["session_metrics"],
    }, sort_keys=True))
