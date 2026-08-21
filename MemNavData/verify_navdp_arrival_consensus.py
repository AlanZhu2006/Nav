#!/usr/bin/env python3
"""Independently verify and apply the frozen NavDP arrival-consensus gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERSISTENCE_GRID = (0.25, 0.50, 0.75, 1.00)
CANDIDATE_FRACTION_GRID = (0.00, 0.125, 0.25, 0.50, 0.75, 1.00)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "tp": int((labels & predictions).sum()),
        "fp": int((~labels & predictions).sum()),
        "fn": int((labels & ~predictions).sum()),
        "tn": int((~labels & ~predictions).sum()),
    }


def report_grid_index(report: dict[str, Any]) -> dict[tuple[float, float], dict]:
    result = {}
    for row in report["predeclared_operating_point_grid"]:
        key = (
            float(row["selected_zero_rate_min"]),
            float(row["candidate_zero_fraction_mean_min"]),
        )
        require(key not in result, f"duplicate report grid point {key}")
        result[key] = row
    return result


def verify(root: Path, expected_mode: str,
           collection_name: str = "collection") -> dict[str, Any]:
    root = root.resolve()
    require(
        collection_name in {"collection", "merged_collection"},
        "unsupported collection name",
    )
    collection = root / collection_name
    required = (
        root / "SEALED",
        root / "submission.json",
        root / "source_bundle.sha256",
        collection / "inventory.json",
        collection / "samples.csv",
        collection / "states.csv",
        collection / "report.json",
        collection / "report.json.sha256",
        collection / "SHA256SUMS.json",
    )
    for path in required:
        require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")

    submission = json.loads((root / "submission.json").read_text())
    report = json.loads((collection / "report.json").read_text())
    inventory = json.loads((collection / "inventory.json").read_text())
    declared_hashes = json.loads((collection / "SHA256SUMS.json").read_text())
    samples = pd.read_csv(collection / "samples.csv")
    states = pd.read_csv(collection / "states.csv")

    submission_mode = "repair" if expected_mode == "repair-merged" \
        else expected_mode
    require(submission["mode"] == submission_mode, "submission mode mismatch")
    require(submission["goat_validation_read"] is False, "GOAT read flag changed")
    require(
        submission["method_or_threshold_authorized"] is False,
        "submission unexpectedly authorizes a rule",
    )
    require(report["complete"] is True, "report is incomplete")
    require(report["goat_validation_read"] is False, "report GOAT flag changed")
    require(
        report["method_or_threshold_authorized"] is False,
        "report unexpectedly authorizes a rule",
    )
    require(inventory["selection_only"] is False, "inventory is selection-only")

    source_receipt_hash = sha256_file(root / "source_bundle.sha256")
    require(
        source_receipt_hash == submission["source_receipt_sha256"],
        "copied source receipt SHA mismatch",
    )
    report_receipt = (collection / "report.json.sha256").read_text().split()[0]
    report_hash = sha256_file(collection / "report.json")
    require(report_receipt == report_hash, "report receipt mismatch")
    for name in ("inventory.json", "samples.csv", "states.csv", "report.json"):
        require(
            declared_hashes[name] == sha256_file(collection / name),
            f"collection hash mismatch: {name}",
        )

    expected_counts = {
        "smoke": (1, 1),
        "formal": (40, 80),
        "repair": (3, 6),
        "repair-merged": (40, 80),
    }
    require(expected_mode in expected_counts, f"unsupported mode {expected_mode}")
    expected_scenes, expected_episodes = expected_counts[expected_mode]
    require(report["scene_count"] == expected_scenes, "scene count mismatch")
    require(report["episode_count"] == expected_episodes, "episode count mismatch")
    if expected_mode == "repair-merged":
        require(collection_name == "merged_collection",
                "merged repair must use merged_collection")
        provenance = report.get("merge_provenance", {})
        require(
            provenance.get("prefix_samples_sha256")
            == submission.get("prefix_samples_sha256"),
            "merged prefix lineage differs from submission",
        )
        require(provenance.get("prefix_episode_count") == 74,
                "merged prefix episode count differs")
    require(len(samples) == report["sample_count"], "sample row count mismatch")
    require(len(states) == report["state_count"], "state row count mismatch")
    require(
        len(samples) == int(report["samples_per_state"]) * len(states),
        "samples/state mismatch",
    )
    require(set(samples["state_id"]) == set(states["state_id"]), "state IDs differ")

    group = samples.groupby("state_id", sort=True)
    samples_per_state = int(report["samples_per_state"])
    require(group.size().eq(samples_per_state).all(), "unequal samples per state")
    require(
        group["diffusion_seed"].nunique().eq(samples_per_state).all(),
        "a state reuses a diffusion seed",
    )
    require(
        set(samples["sample_index"]) == set(range(samples_per_state)),
        "sample-index contract mismatch",
    )
    candidate_counts = sorted(map(int, samples["candidate_count"].unique()))
    require(candidate_counts == [16], f"unexpected candidate counts {candidate_counts}")
    mutation = samples["memory_mutated"].astype(str).str.lower()
    require(
        mutation[samples["sample_index"] > 0].eq("false").all(),
        "a read-only resample mutated memory",
    )

    indexed_states = states.set_index("state_id").sort_index()
    selected_zero_rate = group["selected_zero"].mean().sort_index()
    candidate_fraction = group["candidate_zero_fraction"].mean().sort_index()
    require(
        np.allclose(indexed_states["selected_zero_rate"], selected_zero_rate),
        "selected-zero aggregation mismatch",
    )
    require(
        np.allclose(
            indexed_states["candidate_zero_fraction_mean"], candidate_fraction
        ),
        "candidate-fraction aggregation mismatch",
    )
    distances = states["euclidean_distance_m"].to_numpy(dtype=float)
    labels = states["arrival_025"].astype(bool).to_numpy()
    require(
        np.array_equal(labels, distances <= 0.25 + 1e-12),
        "arrival label/distance mismatch",
    )

    reported_grid = report_grid_index(report)
    require(
        set(reported_grid)
        == set((p, c) for p in PERSISTENCE_GRID for c in CANDIDATE_FRACTION_GRID),
        "reported operating grid differs from frozen grid",
    )
    audited_grid = []
    for persistence in PERSISTENCE_GRID:
        for candidate_min in CANDIDATE_FRACTION_GRID:
            prediction = (
                (states["selected_zero_rate"].to_numpy() >= persistence)
                & (
                    states["candidate_zero_fraction_mean"].to_numpy()
                    >= candidate_min
                )
            )
            counts = confusion(labels, prediction)
            reported = reported_grid[(persistence, candidate_min)]
            for name, value in counts.items():
                require(int(reported[name]) == value, f"grid mismatch at {(persistence, candidate_min)}: {name}")
            tp_scenes = int(states.loc[prediction & labels, "scene"].nunique())
            fp_scenes = int(states.loc[prediction & ~labels, "scene"].nunique())
            near_miss_fp = int(
                (
                    prediction
                    & ~labels
                    & (states["euclidean_distance_m"].to_numpy() <= 0.50)
                ).sum()
            )
            gate_pass = bool(
                counts["fp"] == 0
                and near_miss_fp == 0
                and counts["tp"] >= 20
                and tp_scenes >= 10
            )
            audited_grid.append({
                "selected_zero_rate_min": persistence,
                "candidate_zero_fraction_mean_min": candidate_min,
                **counts,
                "tp_scenes": tp_scenes,
                "fp_scenes": fp_scenes,
                "near_miss_fp": near_miss_fp,
                "frozen_direct_stop_gate_pass": gate_pass,
            })

    passing = [row for row in audited_grid if row["frozen_direct_stop_gate_pass"]]
    passing.sort(
        key=lambda row: (
            row["tp"],
            row["selected_zero_rate_min"],
            row["candidate_zero_fraction_mean_min"],
        ),
        reverse=True,
    )
    chosen = passing[0] if passing else None
    return {
        "verified": True,
        "mode": expected_mode,
        "collection_name": collection_name,
        "run_root": str(root),
        "job_id": int(submission["job_id"]),
        "source_receipt_sha256": source_receipt_hash,
        "report_sha256": report_hash,
        "scene_count": int(report["scene_count"]),
        "episode_count": int(report["episode_count"]),
        "goal_count": int(report["goal_count"]),
        "state_count": int(report["state_count"]),
        "arrival_state_count": int(labels.sum()),
        "nonarrival_state_count": int((~labels).sum()),
        "query_count": int(len(samples)),
        "samples_per_state": samples_per_state,
        "candidates_per_query": candidate_counts[0],
        "trajectories_per_state": samples_per_state * candidate_counts[0],
        "unique_diffusion_seeds": int(samples["diffusion_seed"].nunique()),
        "readonly_resample_count": int((samples["sample_index"] > 0).sum()),
        "reported_auc": report["auc"],
        "frozen_direct_stop_gate_pass": bool(passing),
        "frozen_candidate": chosen,
        "audited_operating_point_grid": audited_grid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--expected-mode",
        choices=("smoke", "formal", "repair", "repair-merged"),
        required=True,
    )
    parser.add_argument("--collection-name", default="collection")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.run_root, args.expected_mode, args.collection_name)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
