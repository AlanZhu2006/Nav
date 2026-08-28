#!/usr/bin/env python3
"""Fail-closed filesystem audit for the exact HM3D factual-C repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_lifelong_underpowered_collect_repair_v1_20260828"
EXPECTED_POPULATION_SHA = (
    "ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5")
QUERY_OUTPUTS = (
    "shared_c_population",
    "shared_c_evaluation",
    "shared_c_aggregate",
    "shared_c_independent_verification.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def required_result_paths(root: Path, label: str, episode: str) -> list[Path]:
    item = root / "shared_c_collection" / label
    result = item / "result"
    return [
        result / "metric.csv",
        result / f"{episode}_shared_C_trace.json",
        result / f"{episode}_plans.json",
        result / "summary.json",
        item / "compute_identity.json",
        item / "result_inputs.sha256",
    ]


def audit(
    *, protocol_path: Path, run_root: Path, phase: str,
) -> dict[str, Any]:
    protocol = load_object(protocol_path)
    require(protocol.get("schema_version") == SCHEMA,
            "repair protocol schema changed")
    authority = protocol.get("source_authority", {})
    require(str(run_root.resolve()) == authority.get("run_root"),
            "run root differs from frozen repair")
    require(authority.get("population_sha256") == EXPECTED_POPULATION_SHA,
            "frozen source population SHA changed")
    freeze = protocol.get("freeze_boundary", {})
    require(freeze.get(
        "successful_factual_C_navigation_outcomes_read_before_repair") is False,
        "factual-C outcomes were read before repair freeze")
    require(freeze.get("B2_navigation_outcomes_read_before_repair") is False,
            "B2 outcomes were read before repair freeze")
    require(freeze.get("repair_selection_uses_navigation_outcomes") is False,
            "repair selection is outcome-conditioned")

    population_path = run_root / str(authority["population_relative_path"])
    require(sha256_file(population_path) == EXPECTED_POPULATION_SHA,
            "source population file changed")
    population = load_object(population_path)
    rows = population.get("accepted")
    require(isinstance(rows, list) and len(rows) == 22,
            "source population no longer contains 22 histories")
    require(len({str(row["scene"]) for row in rows}) == 15,
            "source scene-cluster count changed")

    incident = protocol.get("incident", {})
    completed = incident.get("completed_indices")
    failed = incident.get("failed_indices")
    require(completed == [2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18,
                          19, 20, 21],
            "completed-index set changed")
    require(failed == [0, 1, 7, 9, 11, 13], "repair-index set changed")
    require(sorted(completed + failed) == list(range(22)),
            "repair and retained sets do not partition the population")
    items = protocol.get("repair_items")
    require(isinstance(items, list) and len(items) == 6,
            "repair item count changed")
    by_index = {int(item["index"]): item for item in items}
    require(sorted(by_index) == failed, "repair-item indices changed")

    archive = Path(protocol["repair_contract"][
        "archive_failed_partial_outputs"])
    require(archive.parent == run_root / "failed_attempts",
            "repair archive escaped the frozen run root")
    if phase in {"pre_archive", "post_archive"}:
        present = [name for name in QUERY_OUTPUTS if (run_root / name).exists()]
        require(not present, "downstream query output exists before repair: "
                + ", ".join(present))

    fingerprints: dict[str, dict[str, str]] = {}
    retained = completed if phase != "ready_to_seal" else list(range(22))
    for index in retained:
        row = rows[index]
        label = f"{index:03d}_{row['scene']}_{row['episode']}"
        paths = required_result_paths(run_root, label, str(row["episode"]))
        missing = [str(path) for path in paths if not path.is_file()]
        require(not missing, f"{label}: completed output is incomplete")
        fingerprints[label] = {
            str(path.relative_to(run_root)): sha256_file(path) for path in paths
        }

    for index in failed:
        row = rows[index]
        item = by_index[index]
        label = f"{index:03d}_{row['scene']}_{row['episode']}"
        require(item.get("label") == label, f"index {index}: label changed")
        active = run_root / "shared_c_collection" / label
        archived = archive / label
        if phase == "pre_archive":
            require(active.is_dir(), f"{label}: failed partial is missing")
            require(not archived.exists(), f"{label}: archive already exists")
            log = active / "logs/evaluator.log"
        elif phase == "post_archive":
            require(not active.exists(), f"{label}: failed partial still active")
            require(archived.is_dir(), f"{label}: failed partial was not archived")
            log = archived / "logs/evaluator.log"
        else:
            require(active.is_dir(), f"{label}: repaired output is missing")
            require(archived.is_dir(), f"{label}: incident archive is missing")
            continue
        require(log.is_file(), f"{label}: evaluator failure log is missing")
        message = log.read_text(errors="replace")
        expected = ("NavDP replay queue length does not match frozen plan count"
                    if item["failure_class"] == "invalid_empty_fifo_assumption"
                    else "shared Goal-B trace rendered RGB mismatch")
        require(expected in message, f"{label}: failure class changed")

    return {
        "schema_version": (
            "hm3d_lifelong_underpowered_collect_repair_audit_v1_20260828"),
        "verified": True,
        "phase": phase,
        "source_population_sha256": EXPECTED_POPULATION_SHA,
        "completed_indices_verified": retained,
        "repair_indices": failed,
        "completed_required_file_sha256": fingerprints,
        "successful_factual_C_navigation_outcomes_read": False,
        "B2_navigation_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("pre_archive", "post_archive", "ready_to_seal"),
        required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = audit(
        protocol_path=args.protocol, run_root=args.run_root, phase=args.phase)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
