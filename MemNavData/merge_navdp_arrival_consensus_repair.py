#!/usr/bin/env python3
"""Merge a frozen NavDP arrival prefix with its deterministic gap repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from MemNavData.audit_navdp_arrival_consensus import (
    SCHEMA_VERSION,
    aggregate_states,
    atomic_csv,
    atomic_json,
    build_report,
    load_episode_sources,
    require,
    sha256_file,
)


MERGE_SCHEMA = "navdp_arrival_consensus_prefix_gap_merge_v1_20260815"


def episode_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (str(row.scene), str(row.episode))
        for row in frame[["scene", "episode"]].drop_duplicates().itertuples()
    ]


def verify_samples(frame: pd.DataFrame, expected_pairs: list[tuple[str, str]],
                   samples_per_state: int) -> None:
    require(not frame.empty, "sample input is empty")
    require(episode_pairs(frame) == expected_pairs, "episode ordering differs")
    require(
        frame.groupby("state_id").size().eq(samples_per_state).all(),
        "samples/state contract differs",
    )
    require(
        frame.groupby("state_id")["diffusion_seed"].nunique()
        .eq(samples_per_state).all(),
        "a state reuses a diffusion seed",
    )
    require(
        sorted(map(int, frame["candidate_count"].unique())) == [16],
        "candidate count differs from the verified checkpoint contract",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-samples", type=Path, required=True)
    parser.add_argument("--expected-prefix-sha256", required=True)
    parser.add_argument("--prefix-episode-count", type=int, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--allowed-role", default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(args.prefix_episode_count > 0, "prefix episode count must be positive")
    require(args.prefix_samples.is_file(), "prefix sample artifact is missing")
    require(
        sha256_file(args.prefix_samples) == args.expected_prefix_sha256,
        "prefix sample SHA-256 changed",
    )
    require(not args.out.exists(), "merge output already exists")

    repair_collection = args.repair_root / "collection"
    repair_report_path = repair_collection / "report.json"
    repair_samples_path = repair_collection / "samples.csv"
    repair_hashes_path = repair_collection / "SHA256SUMS.json"
    for path in (repair_report_path, repair_samples_path, repair_hashes_path):
        require(path.is_file(), f"repair artifact is missing: {path}")
    repair_hashes = json.loads(repair_hashes_path.read_text(encoding="utf-8"))
    require(
        repair_hashes["report.json"] == sha256_file(repair_report_path),
        "repair report hash differs",
    )
    require(
        repair_hashes["samples.csv"] == sha256_file(repair_samples_path),
        "repair samples hash differs",
    )
    repair_report = json.loads(repair_report_path.read_text(encoding="utf-8"))
    require(repair_report["complete"] is True, "repair report is incomplete")
    require(
        repair_report["method_or_threshold_authorized"] is False
        and repair_report["goat_validation_read"] is False,
        "repair report changed its authority flags",
    )

    sources = load_episode_sources(
        args.selection_csv, args.split_manifest, args.allowed_role)
    expected_pairs = [(source.scene, source.episode) for source in sources]
    require(len(expected_pairs) == 80, "frozen source universe is not 80 episodes")
    require(
        0 < args.prefix_episode_count < len(expected_pairs),
        "prefix split is outside the source universe",
    )
    prefix_pairs = expected_pairs[:args.prefix_episode_count]
    repair_pairs = expected_pairs[args.prefix_episode_count:]
    require(
        repair_report["source_episode_count"] == len(expected_pairs)
        and repair_report["episode_start_index"] == args.prefix_episode_count
        and repair_report["episode_count"] == len(repair_pairs),
        "repair slice metadata differs from the frozen split",
    )

    prefix = pd.read_csv(args.prefix_samples)
    repair = pd.read_csv(repair_samples_path)
    samples_per_state = int(repair_report["samples_per_state"])
    verify_samples(prefix, prefix_pairs, samples_per_state)
    verify_samples(repair, repair_pairs, samples_per_state)
    require(
        set(prefix["state_id"]).isdisjoint(set(repair["state_id"])),
        "prefix and repair states overlap",
    )

    combined = pd.concat([prefix, repair], ignore_index=True)
    verify_samples(combined, expected_pairs, samples_per_state)
    states = aggregate_states(combined)
    report = build_report(
        combined,
        states,
        selection_csv=args.selection_csv,
        split_manifest=args.split_manifest,
        samples_per_state=samples_per_state,
        stop_threshold=float(repair_report["navdp_stop_threshold_diagnostic"]),
        started_unix=time.time(),
        source_episode_count=len(expected_pairs),
        episode_start_index=0,
    )
    report.update({
        "scope": (
            "train-only offline mechanism audit; deterministic frozen "
            "prefix+gap repair; not a deployment rule"
        ),
        "runtime_s": None,
        "merge_schema_version": MERGE_SCHEMA,
        "merge_provenance": {
            "prefix_samples": str(args.prefix_samples.resolve()),
            "prefix_samples_sha256": args.expected_prefix_sha256,
            "prefix_episode_count": args.prefix_episode_count,
            "repair_root": str(args.repair_root.resolve()),
            "repair_report_sha256": sha256_file(repair_report_path),
            "repair_samples_sha256": sha256_file(repair_samples_path),
            "repair_episode_count": len(repair_pairs),
        },
    })
    require(report["scene_count"] == 40, "merged scene count is not 40")
    require(report["episode_count"] == 80, "merged episode count is not 80")
    require(report["sample_count"] == samples_per_state * report["state_count"],
            "merged sample count differs")

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "merge_schema_version": MERGE_SCHEMA,
        "selection_only": False,
        "episodes": [],
    }
    by_pair = states.groupby(["scene", "episode"], sort=False)
    source_by_pair = {(source.scene, source.episode): source for source in sources}
    for pair in expected_pairs:
        group = by_pair.get_group(pair)
        source = source_by_pair[pair]
        inventory["episodes"].append({
            "scene": pair[0],
            "episode": pair[1],
            "root": str(source.root),
            "goal_count": int(group["goal_index"].nunique()),
            "state_count": int(len(group)),
            "states": list(map(str, group["state_id"])),
        })

    atomic_csv(args.out / "samples.csv", combined)
    atomic_csv(args.out / "states.csv", states)
    atomic_json(args.out / "inventory.json", inventory)
    atomic_json(args.out / "report.json", report)
    hashes = {
        name: sha256_file(args.out / name)
        for name in ("inventory.json", "samples.csv", "states.csv", "report.json")
    }
    atomic_json(args.out / "SHA256SUMS.json", hashes)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
