#!/usr/bin/env python3
"""Independent verifier for construction-only MP3D Table-1 population."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from audit_shared_online_role_pairs import audit, sha256_file
    from mp3d_table1_new_query_contract import (
        POPULATION_SCHEMA,
        SOURCE_LEDGER_SCHEMAS,
        VERIFICATION_SCHEMA,
        assert_new_query_identity,
        power,
        require,
    )
except ImportError:
    from MemNavData.audit_shared_online_role_pairs import audit, sha256_file
    from MemNavData.mp3d_table1_new_query_contract import (
        POPULATION_SCHEMA,
        SOURCE_LEDGER_SCHEMAS,
        VERIFICATION_SCHEMA,
        assert_new_query_identity,
        power,
        require,
    )


def verify(root: Path, source_ledger_path: Path) -> dict:
    population_path = root / "population_receipt.json"
    checksums = root / "CONSTRUCTION_FILES.sha256"
    checksum_receipt = root / "CONSTRUCTION_FILES.sha256.sha256"
    require(population_path.is_file(), "population receipt missing")
    require(checksums.is_file() and checksum_receipt.is_file(),
            "construction checksum receipt missing")
    require(checksum_receipt.read_text().split()[0] == sha256_file(checksums),
            "construction checksum ledger changed")
    for line in checksums.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"construction artifact changed: {relative}")

    ledger = json.loads(source_ledger_path.read_text())
    require(ledger.get("schema_version") in SOURCE_LEDGER_SCHEMAS,
            "source ledger schema changed")
    source_goals = {
        (str(scene["scene"]), str(episode["episode"])):
            episode.get("consumed_queries", [episode["consumed_goal_b"]])
        for scene in ledger["scenes"]
        for episode in scene["episodes"]
    }
    population = json.loads(population_path.read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "population schema changed")
    require(population.get("source_ledger_sha256")
            == sha256_file(source_ledger_path),
            "population source-ledger binding changed")
    require(population.get("source_query_outcomes_read_for_selection") is False
            and population.get("previous_goal_b_policy_outcomes_read") is False,
            "population selection read a prior query outcome")
    require(population.get("navigation_outcomes_generated") is False,
            "construction root declares a navigation outcome")
    require(population.get("fresh_scene") is False
            and population.get("fresh_history") is False
            and population.get("new_query") is True,
            "MP3D claim boundary changed")
    parent = json.loads((root / "parent_manifest.json").read_text())
    require(parent.get("schema_version")
            == "mp3d_table1_runtime_parent_v1_20260829",
            "runtime parent schema changed")
    require(parent.get("source_ledger_sha256")
            == sha256_file(source_ledger_path),
            "runtime parent source-ledger binding changed")
    require(parent.get("scenes")
            == [str(row["scene"]) for row in ledger["scenes"]],
            "runtime parent scene order changed")

    benchmark = root / "natural_direction"
    benchmark_audit = audit(benchmark)
    manifest = json.loads((benchmark / "manifest.json").read_text())
    identities = set()
    for row in manifest["episodes"]:
        identity = str(row["scene"]), str(row["episode"])
        require(identity in source_goals,
                f"retained history outside source ledger: {identity}")
        require(identity not in identities, "retained history duplicated")
        identities.add(identity)
        assert_new_query_identity(row, source_goals[identity])
    declared = population["power_gate"]
    observed = power(
        list(manifest["episodes"]),
        target_histories=int(declared["target_histories"]),
        target_scenes=int(declared["target_scene_clusters"]),
        minimum_per_stratum=int(
            declared["minimum_histories_per_direction_stratum"]),
    )
    require(observed == declared, "power gate does not reproduce")
    require(int(population["retained_histories"]) == len(identities),
            "retained-history count changed")
    require(int(population["retained_scene_clusters"])
            == len({scene for scene, _episode in identities}),
            "scene-cluster count changed")
    require(int(population["query_count"]) == 2 * len(identities),
            "query count changed")
    require(bool(population["formal_policy_evaluation_authorized"])
            == bool(observed["target_met"]),
            "formal authorization disagrees with power gate")
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "verified": True,
        "construction_only": True,
        "policy_outcomes_read": False,
        "fresh_scene": False,
        "fresh_history": False,
        "new_query": True,
        "source_ledger_sha256": sha256_file(source_ledger_path),
        "population_receipt_sha256": sha256_file(population_path),
        "benchmark_manifest_sha256": benchmark_audit["manifest_sha256"],
        "histories": len(identities),
        "scene_clusters": len({scene for scene, _episode in identities}),
        "queries": 2 * len(identities),
        "consumed_goal_b_overlap": 0,
        "power_gate": observed,
        "formal_policy_evaluation_authorized": bool(observed["target_met"]),
        "benchmark_audit": benchmark_audit,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.source_ledger.resolve())
    write_exclusive(args.out.resolve(), result)
    print(json.dumps({
        "verified": result["verified"],
        "histories": result["histories"],
        "scene_clusters": result["scene_clusters"],
        "power_gate": result["power_gate"],
        "formal_policy_evaluation_authorized": result[
            "formal_policy_evaluation_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
