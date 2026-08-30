#!/usr/bin/env python3
"""Freeze an outcome-blind exact repair for missing Table-III Goal-A cells.

Membership is determined only from the existence and byte integrity of each
completion receipt.  The receipt payload is never deserialized, so success or
failure cannot influence which frozen candidate is retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hm3d_table3_actual_mono_a_transport_repair_v1_20260830"
ARCHIVE_SCHEMA = "hm3d_table3_actual_mono_a_partial_archive_v1_20260830"


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
    require(path.is_file() and sidecar.is_file(), f"missing receipt: {path}")
    digest = sha256(path)
    require(sidecar.read_text().split() == [digest, path.name],
            f"invalid receipt: {path}")
    return digest


def episode_name(row: dict) -> str:
    return f"episode_{row['episode']}"


def factual_label(index: int, row: dict) -> str:
    return f"{index:03d}_{row['scene']}_{episode_name(row)}"


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"partial output contains symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--expected-candidate-plan-sha256", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(not args.out.exists(), f"repair plan exists: {args.out}")
    require(not args.archive_root.exists(),
            f"repair archive exists: {args.archive_root}")
    candidate_sha = sha256(args.candidate_plan)
    require(candidate_sha == args.expected_candidate_plan_sha256,
            "candidate plan changed")
    candidate_plan = json.loads(args.candidate_plan.read_text())
    episodes = candidate_plan.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == 125,
            "frozen Table-III candidate count changed")
    require([int(row["history_index"]) for row in episodes] == list(range(125)),
            "candidate identity order changed")

    factual_root = args.run_root / "factual_a"
    carrier_root = args.run_root / "carriers"
    runtime_root = args.run_root / "runtime"
    factual_root.mkdir(parents=True, exist_ok=True)
    completed: list[int] = []
    missing: list[int] = []
    partial_sources: list[tuple[int, str, Path]] = []
    for index, row in enumerate(episodes):
        factual = factual_root / factual_label(index, row)
        completion = factual / "completion.json"
        if completion.is_file():
            # Byte-level verification only.  Never read the JSON payload.
            verify_sidecar(completion)
            completed.append(index)
            continue
        missing.append(index)
        if factual.exists():
            require(factual.is_dir() and not factual.is_symlink(),
                    f"unsafe factual partial: {factual}")
            partial_sources.append((index, "factual_a", factual))
        carrier = carrier_root / str(row["scene"]) / episode_name(row)
        if carrier.exists():
            require(carrier.is_dir() and not carrier.is_symlink(),
                    f"unsafe carrier partial: {carrier}")
            partial_sources.append((index, "carrier", carrier))
        if runtime_root.exists():
            for runtime in sorted(runtime_root.glob(f"table3_a_{index}_*")):
                require(runtime.is_dir() and not runtime.is_symlink(),
                        f"unsafe runtime partial: {runtime}")
                partial_sources.append((index, "runtime", runtime))

    require(missing, "repair requested but no Table-III identity is missing")
    args.archive_root.mkdir(parents=True)
    archived = []
    for ordinal, (index, source_kind, source) in enumerate(partial_sources):
        destination = (
            args.archive_root
            / f"{index:03d}_{source_kind}_{ordinal:03d}_{source.name}"
        )
        files = inventory(source)
        source.rename(destination)
        require(not source.exists() and destination.is_dir(),
                f"failed to archive {source}")
        archived.append({
            "history_index": index,
            "source_kind": source_kind,
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "files": files,
        })

    archive_receipt = {
        "schema_version": ARCHIVE_SCHEMA,
        "entries": archived,
        "completion_payloads_deserialized": False,
        "navigation_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "scientific_thresholds_changed": False,
    }
    archive_path = args.archive_root / "archive_receipt.json"
    archive_path.write_text(json.dumps(
        archive_receipt, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    archive_path.with_name(archive_path.name + ".sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n"
    )

    identities = []
    for index in missing:
        row = episodes[index]
        identities.append({
            "history_index": index,
            "scene": row["scene"],
            "episode": row["episode"],
            "bin_name": row["bin_name"],
            "candidate_identity_sha256": row["candidate_identity_sha256"],
        })
    payload = {
        "schema_version": SCHEMA,
        "status": "repair_required",
        "candidate_plan_sha256": candidate_sha,
        "candidate_count": len(episodes),
        "completed_history_indices": completed,
        "completed_history_count": len(completed),
        "missing_history_indices": missing,
        "missing_history_count": len(missing),
        "repair_identities": identities,
        "archive_receipt_sha256": sha256(archive_path),
        "completion_membership_signal": "existence_plus_byte_receipt_only",
        "completion_payloads_deserialized": False,
        "navigation_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "candidate_identities_changed": False,
        "model_or_controller_changed": False,
        "scientific_thresholds_changed": False,
        "step_budget_changed": False,
        "fallback_completion_allowed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps({
        "completed": len(completed), "missing": len(missing),
        "archived_sources": len(archived),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
