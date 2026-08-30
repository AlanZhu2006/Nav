#!/usr/bin/env python3
"""Independent raw-file audit of the powered factual A/B population union."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_fullmono_lifelong_population_union_verification_v1_20260830"
UNION_SCHEMA = "hm3d_fullmono_lifelong_population_union_v1_20260830"
RECEIPT_SCHEMA = "hm3d_fullmono_lifelong_population_union_receipt_v1_20260830"
TABLE2_SCHEMA = "hm3d_table2_leg3_mixed_role_protocol_v1_20260829"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: JSON root changed")
    return payload


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"{path}: SHA sidecar missing")
    fields = sidecar.read_text().split()
    digest = sha256(path)
    require(len(fields) == 2 and fields[0] == digest
            and fields[1] == path.name, f"{path}: SHA sidecar changed")
    return digest


def contained(path: Path, root: Path) -> Path:
    candidate, boundary = path.resolve(), root.resolve()
    require(candidate != boundary and boundary in candidate.parents,
            f"{path}: path escaped its immutable root")
    return candidate


def verify_ledger(root: Path, name: str, excluded_names: set[str]) -> int:
    ledger = root / name
    require(ledger.is_file(), f"{ledger}: ledger missing")
    seen: set[Path] = set()
    for line in ledger.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"{ledger}: malformed row")
        path = contained(root / fields[1].strip(), root)
        require(path.is_file() and path not in seen,
                f"{ledger}: duplicate or missing entry")
        require(sha256(path) == fields[0], f"{path}: digest changed")
        seen.add(path)
    actual = {
        path.resolve() for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names
    }
    require(seen == actual, f"{ledger}: file coverage changed")
    return len(seen)


def verify(root: Path) -> dict[str, Any]:
    receipt_path = root / "union_receipt.json"
    receipt_sha = verify_sidecar(receipt_path)
    receipt = read(receipt_path)
    require(receipt.get("schema_version") == RECEIPT_SCHEMA,
            "union receipt schema changed")
    require(receipt.get("leg3_query_navigation_outcomes_read") is False,
            "union receipt crossed the Leg-3 outcome boundary")

    population_root = root / "population"
    population_path = population_root / "population.json"
    population_sha = verify_sidecar(population_path)
    population = read(population_path)
    require(population.get("schema_version") == UNION_SCHEMA,
            "union population schema changed")
    require((population_root / "SEALED").is_file(),
            "union population is not sealed")
    require(population.get("selection_reads_C_B2_C2_navigation_outcomes") is False
            and population.get("runtime_role_visibility") == "none",
            "union population read a downstream outcome or role")
    require(population_sha == receipt["population_sha256"]
            and sha256(population_root / "SEALED")
            == receipt["population_seal_sha256"],
            "union receipt population binding changed")

    source_contracts = population["source_populations"]
    require([row["name"] for row in source_contracts]
            == ["original_v4", "natural_b_expansion"],
            "union source ordering changed")
    source_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_totals: dict[str, int] = {}
    for source in source_contracts:
        name = str(source["name"])
        run_root = Path(source["run_root"])
        source_population_root = run_root / "population"
        source_population_path = source_population_root / "population.json"
        require(verify_sidecar(source_population_path)
                == source["population_sha256"],
                f"{name}: source population changed")
        require(sha256(source_population_root / "POPULATION_FILES.sha256")
                == source["population_file_ledger_sha256"],
                f"{name}: source file ledger changed")
        source_population = read(source_population_path)
        rows = source_population["accepted"]
        require(len(rows) == int(source["supported_histories"]),
                f"{name}: supported count changed")
        source_totals[name] = len(rows)
        for row in rows:
            key = (name, str(row["scene"]), str(row["episode"]))
            require(key not in source_rows, f"{name}: duplicate source identity")
            source_rows[key] = row

    accepted = population["accepted"]
    require(len(accepted) == sum(source_totals.values())
            == int(population["supported_population"])
            == int(receipt["union_supported_histories"]),
            "union supported count changed")
    seen: set[tuple[str, str, str]] = set()
    global_identities: set[tuple[str, str]] = set()
    for index, row in enumerate(accepted):
        name = str(row["source_population_name"])
        scene, episode = str(row["scene"]), str(row["episode"])
        key = (name, scene, episode)
        require(key in source_rows and key not in seen,
                "union row is absent or duplicated in its source")
        seen.add(key)
        require((scene, episode) not in global_identities,
                "union contains a duplicate runtime identity")
        global_identities.add((scene, episode))
        source_row = source_rows[key]
        require(int(row["population_index"]) == index,
                "union population index changed")
        require(row["source_population_sha256"]
                == receipt["source_population_hashes"][name],
                "union row source hash changed")
        benchmark = contained(population_root / row["benchmark"], population_root)
        require(benchmark.is_file()
                and sha256(benchmark) == row["benchmark_sha256"]
                == source_row["benchmark_sha256"],
                "union benchmark differs from immutable source")
        role = root / "ab_population/role_pairs" / scene / episode / "role_pairs.json"
        require(role.is_file()
                and sha256(role) == row["source_role_pair_sha256"],
                "union role-pair evidence changed")
    require(seen == set(source_rows), "union omitted a supported source prefix")

    scenes = {str(row["scene"]) for row in accepted}
    target_met = (
        len(accepted) >= int(population["target_histories"])
        and len(scenes) >= int(population["target_scene_clusters"])
    )
    require(int(population["scene_clusters"]) == len(scenes)
            and population["target_met"] is target_met
            and population["underpowered"] is (not target_met)
            and receipt["target_met"] is target_met,
            "union power gate changed")

    protocol_path = root / "hm3d_table2_leg3_power_protocol.json"
    protocol_sha = verify_sidecar(protocol_path)
    protocol = read(protocol_path)
    source = protocol["source_population"]
    require(protocol.get("schema_version") == TABLE2_SCHEMA
            and protocol.get("leg3_query_outcomes_read_before_freeze") is False,
            "derived Table-II protocol changed")
    require(Path(source["run_root"]).resolve() == root.resolve()
            and source["population"] == "population/population.json"
            and source["population_sha256"] == population_sha
            and source["seal_sha256"] == sha256(population_root / "SEALED")
            and int(source["actual_AB_successful_histories"]) == len(accepted)
            and int(source["actual_AB_scene_clusters"]) == len(scenes)
            and source["selection_reads_C_B2_C2_navigation_outcomes"] is False
            and source["union_receipt_sha256"] == receipt_sha,
            "derived Table-II protocol source binding changed")
    query = protocol["leg3_queries"]
    gate = protocol["population_gate"]
    runtime = protocol["runtime"]
    require(float(query["novel_max_combined_AB_covis_exclusive"]) == 0.10
            and float(query["revisit_min_combined_AB_covis_inclusive"]) == 0.55
            and query["runtime_role_visibility"] == "none",
            "derived Table-II query thresholds changed")
    require(int(gate["minimum_histories"]) == 16
            and int(gate["minimum_scene_clusters"]) == 10
            and int(gate["minimum_histories_per_direction_stratum"]) == 3,
            "derived Table-II construction gate changed")
    require(runtime["arms"] == ["mono_native", "mono_cec"]
            and int(runtime["maximum_steps"]) == 600
            and int(runtime["execution_horizon"]) == 8,
            "derived Table-II runtime changed")

    population_ledger_entries = verify_ledger(
        population_root, "POPULATION_FILES.sha256",
        {"POPULATION_FILES.sha256", "SEALED"},
    )
    union_ledger_entries = verify_ledger(
        root, "UNION_FILES.sha256",
        {
            "UNION_FILES.sha256", "POPULATION_FILES.sha256",
            "independent_population_union_verification.json",
            "independent_population_union_verification.json.sha256",
        },
    )
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "result_blind": True,
        "leg3_query_navigation_outcomes_read": False,
        "source_supported_histories": source_totals,
        "union_supported_histories": len(accepted),
        "union_scene_clusters": len(scenes),
        "target_met": target_met,
        "population_sha256": population_sha,
        "union_receipt_sha256": receipt_sha,
        "table2_protocol_sha256": protocol_sha,
        "population_file_ledger_entries_verified": population_ledger_entries,
        "union_file_ledger_entries_verified": union_ledger_entries,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "union verification output already exists")
    result = verify(args.root.resolve())
    write_exclusive(args.out.resolve(), result)
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
