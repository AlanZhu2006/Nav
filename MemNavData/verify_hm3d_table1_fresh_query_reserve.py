#!/usr/bin/env python3
"""Independent construction verifier for the HM3D Table-1 reserve."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from audit_shared_online_role_pairs import audit, sha256_file
    from hm3d_table1_fresh_query_contract import (
        POPULATION_SCHEMA,
        identity_set,
        power,
        require,
    )
except ImportError:
    from MemNavData.audit_shared_online_role_pairs import audit, sha256_file
    from MemNavData.hm3d_table1_fresh_query_contract import (
        POPULATION_SCHEMA,
        identity_set,
        power,
        require,
    )


SCHEMA = "hm3d_table1_fresh_query_verification_v1_20260829"


def verify(root: Path, original_manifest_path: Path) -> dict:
    population_path = root / "population_receipt.json"
    checksum_path = root / "CONSTRUCTION_FILES.sha256"
    checksum_receipt = root / "CONSTRUCTION_FILES.sha256.sha256"
    require(population_path.is_file(), "population receipt missing")
    require(checksum_path.is_file() and checksum_receipt.is_file(),
            "construction checksum receipt missing")
    require(checksum_receipt.read_text().split()[0] == sha256_file(checksum_path),
            "construction checksum ledger changed")
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"construction artifact changed: {relative}")

    population = json.loads(population_path.read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "population schema changed")
    require(population.get("source_query_outcomes_read_for_selection") is False,
            "population selection read query outcomes")
    require(population.get("navigation_outcomes_generated") is False,
            "construction root contains a declared navigation outcome")
    benchmark = root / "natural_direction"
    benchmark_audit = audit(benchmark)
    manifest = json.loads((benchmark / "manifest.json").read_text())
    consumed = identity_set(json.loads(original_manifest_path.read_text()))
    retained = {
        (str(row["scene"]), str(row["episode"]))
        for row in manifest["episodes"]
    }
    overlap = sorted(retained & consumed)
    require(not overlap, "consumed formal identity leaked into fresh reserve")
    declared = population["power_gate"]
    observed = power(
        list(manifest["episodes"]),
        target_histories=int(declared["target_histories"]),
        target_scenes=int(declared["target_scene_clusters"]),
        minimum_per_stratum=int(
            declared["minimum_histories_per_direction_stratum"]
        ),
    )
    require(observed == declared, "power gate does not reproduce")
    require(int(population["retained_histories"]) == len(retained),
            "retained-history count changed")
    require(int(population["retained_scene_clusters"])
            == len({scene for scene, _episode in retained}),
            "scene-cluster count changed")
    require(int(population["query_count"]) == 2 * len(retained),
            "query count changed")
    require(bool(population["formal_policy_evaluation_authorized"])
            == bool(observed["target_met"]),
            "formal authorization disagrees with power gate")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "construction_only": True,
        "policy_outcomes_read": False,
        "population_receipt_sha256": sha256_file(population_path),
        "benchmark_manifest_sha256": benchmark_audit["manifest_sha256"],
        "original_manifest_sha256": sha256_file(original_manifest_path),
        "consumed_identity_overlap": overlap,
        "histories": len(retained),
        "scene_clusters": len({scene for scene, _episode in retained}),
        "queries": 2 * len(retained),
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
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.root.resolve(), args.original_manifest.resolve())
    write_exclusive(args.out.resolve(), payload)
    print(json.dumps({
        "verified": payload["verified"],
        "histories": payload["histories"],
        "scene_clusters": payload["scene_clusters"],
        "power_gate": payload["power_gate"],
        "formal_policy_evaluation_authorized": payload[
            "formal_policy_evaluation_authorized"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
