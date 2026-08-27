#!/usr/bin/env python3
"""Audit and preserve the frozen Natural-V4 factual-B missing-shard repair.

The repair selection is identity/completeness-only.  Navigation outcomes are
never parsed.  Any incomplete output is moved, with a byte-level receipt, to
``failed_attempts`` before the unchanged collector is re-executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hm3d_fullmono_lifelong_natural_v4_factual_b_missing_repair_v1_20260828"
EXPECTED_CANDIDATES = 99
EXPECTED_COMPLETED = 95
FROZEN_REPAIR_SHARDS = {31: (51, 52), 37: (62, 63)}
FROZEN_MISSING_INDICES = (51, 52, 62, 63)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"missing receipt for {path}")
    fields = sidecar.read_text().split()
    digest = sha256(path)
    require(fields == [digest, path.name], f"invalid receipt for {path}")
    return digest


def load_inputs(manifest_path: Path, schedule_path: Path,
                expected_manifest_sha256: str,
                expected_schedule_sha256: str) -> tuple[list[dict], dict]:
    manifest_sha = verify_sidecar(manifest_path)
    schedule_sha = verify_sidecar(schedule_path)
    require(manifest_sha == expected_manifest_sha256,
            "benchmark manifest digest changed")
    require(schedule_sha == expected_schedule_sha256,
            "factual-B schedule digest changed")
    manifest = json.loads(manifest_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list)
            and len(episodes) == EXPECTED_CANDIDATES,
            "frozen factual-B candidate population changed")
    require(schedule.get("candidate_histories") == EXPECTED_CANDIDATES
            and schedule.get("all_candidates_partitioned_once") is True
            and schedule.get("query_policy_outcomes_read") is False
            and schedule.get("navigation_outcomes_read") is False,
            "factual-B result-blind schedule contract changed")
    require(schedule.get("benchmark_manifest_sha256") == manifest_sha,
            "schedule no longer binds the benchmark manifest")
    for shard_index, indices in FROZEN_REPAIR_SHARDS.items():
        shard = schedule["shards"][shard_index]
        require(int(shard["shard_index"]) == shard_index
                and tuple(int(v) for v in shard["history_indices"]) == indices
                and shard.get("navigation_outcomes_read") is False,
                f"frozen repair shard {shard_index} changed")
    return episodes, schedule


def label_for(index: int, item: dict) -> str:
    return f"{index:03d}_{item['scene']}_{item['episode']}"


def audit(run_root: Path, episodes: list[dict]) -> dict:
    factual_root = run_root / "factual_b"
    require(factual_root.is_dir(), "factual-B root is missing")
    completed: list[int] = []
    missing: list[int] = []
    partial: list[int] = []
    for index, item in enumerate(episodes):
        root = factual_root / label_for(index, item)
        completion = root / "completion.json"
        if completion.is_file():
            verify_sidecar(completion)
            completed.append(index)
        else:
            missing.append(index)
            if root.exists():
                require(root.is_dir() and not root.is_symlink(),
                        f"unsafe partial output {root}")
                partial.append(index)
    require(len(completed) == EXPECTED_COMPLETED,
            f"expected {EXPECTED_COMPLETED} completions, found {len(completed)}")
    require(tuple(missing) == FROZEN_MISSING_INDICES,
            f"missing identities changed: {missing}")
    require(tuple(partial) == (51, 62),
            f"partial identities changed: {partial}")
    return {
        "schema_version": SCHEMA,
        "status": "repair_required",
        "candidate_histories": EXPECTED_CANDIDATES,
        "completed_histories": len(completed),
        "missing_history_indices": missing,
        "partial_history_indices": partial,
        "repair_shards": sorted(FROZEN_REPAIR_SHARDS),
        "selection_reads_navigation_outcomes": False,
        "protocol_or_threshold_changed": False,
    }


def archive_partial(run_root: Path, episodes: list[dict], shard_index: int,
                    repair_tag: str) -> dict:
    require(shard_index in FROZEN_REPAIR_SHARDS,
            f"shard {shard_index} is not in the frozen repair")
    require(repair_tag.replace("_", "").replace("-", "").isalnum(),
            "repair tag contains unsupported characters")
    archive_root = (run_root / "failed_attempts" /
                    f"factual_b_{repair_tag}_shard{shard_index:03d}")
    require(not archive_root.exists(), f"repair archive exists: {archive_root}")
    archive_root.mkdir(parents=True)
    rows = []
    for index in FROZEN_REPAIR_SHARDS[shard_index]:
        label = label_for(index, episodes[index])
        source = run_root / "factual_b" / label
        require(not (source / "completion.json").exists(),
                f"repair would overwrite completed history {index}")
        row = {
            "history_index": index,
            "label": label,
            "partial_output_present": source.exists(),
            "files": {},
        }
        if source.exists():
            require(source.is_dir() and not source.is_symlink(),
                    f"unsafe partial output {source}")
            for path in sorted(source.rglob("*")):
                require(not path.is_symlink(), f"partial output symlink: {path}")
                if path.is_file():
                    row["files"][path.relative_to(source).as_posix()] = sha256(path)
            destination = archive_root / label
            source.rename(destination)
            row["archive_path"] = str(destination.resolve())
        rows.append(row)
        require(not source.exists(), f"failed to clear repair target {source}")
    receipt = {
        "schema_version": SCHEMA,
        "status": "partial_outputs_preserved",
        "repair_tag": repair_tag,
        "shard_index": shard_index,
        "history_indices": list(FROZEN_REPAIR_SHARDS[shard_index]),
        "selection_reads_navigation_outcomes": False,
        "navigation_artifacts_interpreted": False,
        "protocol_or_threshold_changed": False,
        "completed_output_overwrite_allowed": False,
        "rows": rows,
    }
    receipt_path = archive_root / "archive_receipt.json"
    receipt_path.write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    receipt_path.with_name(receipt_path.name + ".sha256").write_text(
        f"{sha256(receipt_path)}  {receipt_path.name}\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "archive"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--repair-tag")
    args = parser.parse_args()
    episodes, _ = load_inputs(
        args.manifest, args.schedule,
        args.expected_manifest_sha256, args.expected_schedule_sha256,
    )
    if args.mode == "audit":
        require(args.shard_index is None and args.repair_tag is None,
                "audit does not accept shard arguments")
        result = audit(args.run_root, episodes)
    else:
        require(args.shard_index is not None and args.repair_tag,
                "archive requires shard index and repair tag")
        result = archive_partial(
            args.run_root, episodes, args.shard_index, args.repair_tag
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
