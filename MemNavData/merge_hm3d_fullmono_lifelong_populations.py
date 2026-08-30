#!/usr/bin/env python3
"""Seal the original and Natural-B-expansion factual A/B populations.

The merge is deliberately performed only after the expansion raw-file
verifier has passed.  It copies every supported prefix from both immutable
sources, preserves the source provenance of every row, and creates the
derived Table-II protocol before any new Leg-3 query is constructed or run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hm3d_fullmono_lifelong import require, sha256_file


UNION_SCHEMA = "hm3d_fullmono_lifelong_population_union_v1_20260830"
RECEIPT_SCHEMA = "hm3d_fullmono_lifelong_population_union_receipt_v1_20260830"
TABLE2_PROTOCOL_SCHEMA = "hm3d_table2_leg3_mixed_role_protocol_v1_20260829"


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing SHA sidecar for {path}")
    fields = sidecar.read_text().split()
    digest = sha256_file(path)
    require(len(fields) == 2 and fields[0] == digest
            and fields[1] == path.name, f"bad SHA sidecar for {path}")
    return digest


def verify_ledger(root: Path, name: str) -> None:
    ledger = root / name
    require(ledger.is_file(), f"missing source ledger {ledger}")
    for line in ledger.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"malformed source ledger row: {line}")
        candidate = (root / fields[1].strip()).resolve()
        require(root.resolve() in candidate.parents and candidate.is_file(),
                f"source ledger path escaped or disappeared: {candidate}")
        require(sha256_file(candidate) == fields[0],
                f"source ledger digest changed: {candidate}")


def source_contract(*, name: str, run_root: Path, population_sha: str,
                    population_file_ledger_sha: str | None = None) -> dict[str, Any]:
    population_root = run_root / "population"
    population_path = population_root / "population.json"
    seal_path = population_root / "SEALED"
    require(population_path.is_file() and seal_path.is_file(),
            f"{name}: source population is not sealed")
    require(verify_sidecar(population_path) == population_sha,
            f"{name}: source population hash changed")
    ledger_path = population_root / "POPULATION_FILES.sha256"
    if population_file_ledger_sha is not None:
        require(sha256_file(ledger_path) == population_file_ledger_sha,
                f"{name}: source population ledger changed")
    verify_ledger(population_root, "POPULATION_FILES.sha256")
    payload = json.loads(population_path.read_text())
    require(payload.get("selection_reads_C_B2_C2_navigation_outcomes") is False,
            f"{name}: source selection read downstream outcomes")
    require(payload.get("runtime_role_visibility") == "none",
            f"{name}: runtime role was exposed")
    return {
        "name": name,
        "run_root": run_root.resolve(),
        "population_root": population_root.resolve(),
        "population_path": population_path.resolve(),
        "population_sha256": population_sha,
        "population_file_ledger_sha256": sha256_file(ledger_path),
        "seal_sha256": sha256_file(seal_path),
        "payload": payload,
    }


def copy_supported_row(*, source: dict[str, Any], row: dict[str, Any],
                       population_index: int, temporary: Path) -> dict[str, Any]:
    scene, episode = str(row["scene"]), str(row["episode"])
    source_population_root = Path(source["population_root"])
    source_benchmark = (source_population_root / row["benchmark"]).resolve()
    require(source_population_root in source_benchmark.parents,
            "source benchmark escaped its sealed population")
    require(source_benchmark.is_file()
            and sha256_file(source_benchmark) == row["benchmark_sha256"],
            "source benchmark changed")
    destination_benchmark_root = (
        temporary / "population" / "benchmark" / scene / episode
    )
    require(not destination_benchmark_root.exists(),
            "duplicate union benchmark identity")
    destination_benchmark_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_benchmark.parent, destination_benchmark_root)

    source_role_root = (
        Path(source["run_root"]) / "ab_population" / "role_pairs"
        / scene / episode
    )
    require((source_role_root / "role_pairs.json").is_file(),
            "source role-pair evidence is missing")
    destination_role_root = (
        temporary / "ab_population" / "role_pairs" / scene / episode
    )
    require(not destination_role_root.exists(),
            "duplicate union role-pair identity")
    destination_role_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_role_root, destination_role_root)

    copied_benchmark = destination_benchmark_root / source_benchmark.name
    require(sha256_file(copied_benchmark) == row["benchmark_sha256"],
            "copied benchmark differs from source")
    copied_role = destination_role_root / "role_pairs.json"
    return {
        **deepcopy(row),
        "population_index": population_index,
        "source_population_name": source["name"],
        "source_population_sha256": source["population_sha256"],
        "source_population_index": int(row.get("population_index", population_index)),
        "source_AB_history_index_original": int(row["source_AB_history_index"]),
        "source_role_pair_sha256": sha256_file(copied_role),
        "benchmark": str(copied_benchmark.relative_to(temporary / "population")),
        "benchmark_sha256": sha256_file(copied_benchmark),
    }


def write_ledger(root: Path, name: str, *, excluded: set[str]) -> int:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    (root / name).write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files
    ))
    return len(files)


def derived_table2_protocol(*, base: dict[str, Any], out: Path,
                            population_sha: str, seal_sha: str,
                            source_histories: int, accepted: list[dict[str, Any]],
                            union_receipt_sha: str) -> dict[str, Any]:
    protocol = deepcopy(base)
    require(protocol.get("schema_version") == TABLE2_PROTOCOL_SCHEMA,
            "base Table-II protocol schema changed")
    scenes = {str(row["scene"]) for row in accepted}
    protocol["frozen_at"] = datetime.now(timezone.utc).isoformat()
    protocol["scope"] = (
        "powered controlled shared-prefix HM3D continual Table-2 Leg-3 "
        "Novel/Revisit comparison"
    )
    protocol["source_population"] = {
        "run_root": str(out.resolve()),
        "population": "population/population.json",
        "population_sha256": population_sha,
        "seal": "population/SEALED",
        "seal_sha256": seal_sha,
        "source_histories": source_histories,
        "actual_AB_successful_histories": len(accepted),
        "actual_AB_scene_clusters": len(scenes),
        "selection_rule": (
            "exact union of the frozen original supported population and "
            "every supported result-blind Natural-B expansion prefix"
        ),
        "selection_reads_C_B2_C2_navigation_outcomes": False,
        "union_receipt_sha256": union_receipt_sha,
    }
    protocol["power_expansion"] = {
        "reason": "pre-registered Table-II construction power shortfall",
        "original_query_navigation_outcomes_read": False,
        "expansion_query_navigation_outcomes_read": False,
        "no_supported_prefix_filtered": True,
        "no_query_or_certificate_threshold_changed": True,
        "all_source_dependencies_reported": True,
    }
    protocol["leg3_query_outcomes_read_before_freeze"] = False
    return protocol


def merge(*, expansion_protocol_path: Path, expansion_run: Path,
          base_table2_protocol_path: Path, out: Path) -> dict[str, Any]:
    expansion_protocol = json.loads(expansion_protocol_path.read_text())
    union_contract = expansion_protocol["population_union"]
    original_run = Path(union_contract["original_run_root"])
    original = source_contract(
        name="original_v4",
        run_root=original_run,
        population_sha=union_contract["original_population_sha256"],
        population_file_ledger_sha=union_contract[
            "original_population_file_ledger_sha256"
        ],
    )
    require(len(original["payload"]["accepted"])
            == int(union_contract["original_supported_histories"]),
            "original supported-history count changed")

    verification_path = (
        expansion_run
        / "independent_natural_b_expansion_population_verification.json"
    )
    verification_sha = verify_sidecar(verification_path)
    verification = json.loads(verification_path.read_text())
    require(verification.get("verified") is True
            and verification.get("query_navigation_outcomes_read") is False
            and verification.get("factual_C_B2_C2_executed") is False,
            "expansion population did not pass the result-blind verifier")
    expansion = source_contract(
        name="natural_b_expansion",
        run_root=expansion_run,
        population_sha=str(verification["population_sha256"]),
    )
    require(len(expansion["payload"]["accepted"])
            == int(verification["supported_population"]),
            "expansion supported-history count changed")

    require(not out.exists(), f"population union already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    try:
        accepted: list[dict[str, Any]] = []
        identities: set[tuple[str, str]] = set()
        for source in (original, expansion):
            for row in source["payload"]["accepted"]:
                identity = (str(row["scene"]), str(row["episode"]))
                require(identity not in identities,
                        "source populations contain a duplicate identity")
                identities.add(identity)
                accepted.append(copy_supported_row(
                    source=source, row=row, population_index=len(accepted),
                    temporary=temporary,
                ))

        target_histories = int(union_contract["minimum_target_histories"])
        target_scenes = int(union_contract["minimum_target_scene_clusters"])
        scenes = {str(row["scene"]) for row in accepted}
        target_met = (
            len(accepted) >= target_histories and len(scenes) >= target_scenes
        )
        population = {
            "schema_version": UNION_SCHEMA,
            "scope": "pre-Leg3 exact factual A/B population union",
            "protocol_sha256": sha256_file(expansion_protocol_path),
            "intention_to_collect_B": sum(
                int(source["payload"]["intention_to_collect_B"])
                for source in (original, expansion)
            ),
            "supported_population": len(accepted),
            "scene_clusters": len(scenes),
            "strong_support_histories": sum(
                int(bool(row["B_goal_strong_support"])) for row in accepted
            ),
            "target_histories": target_histories,
            "target_scene_clusters": target_scenes,
            "target_met": target_met,
            "underpowered": not target_met,
            "selection_reads_C_B2_C2_navigation_outcomes": False,
            "runtime_role_visibility": "none",
            "source_populations": [
                {
                    "name": source["name"],
                    "run_root": str(source["run_root"]),
                    "population_sha256": source["population_sha256"],
                    "population_file_ledger_sha256": source[
                        "population_file_ledger_sha256"
                    ],
                    "supported_histories": len(source["payload"]["accepted"]),
                }
                for source in (original, expansion)
            ],
            "accepted": accepted,
            "attrition": [],
        }
        population_root = temporary / "population"
        population_path = population_root / "population.json"
        population_path.write_text(json.dumps(
            population, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (population_root / "population.json.sha256").write_text(
            f"{sha256_file(population_path)}  population.json\n"
        )
        (population_root / "SEALED").write_text(
            "sealed exact population union before Table-II Leg-3 construction\n"
        )
        population_ledger_entries = write_ledger(
            population_root, "POPULATION_FILES.sha256",
            excluded={"POPULATION_FILES.sha256", "SEALED"},
        )

        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "verified_expansion_population_receipt_sha256": verification_sha,
            "expansion_execution_protocol_sha256": sha256_file(
                expansion_protocol_path
            ),
            "source_population_hashes": {
                source["name"]: source["population_sha256"]
                for source in (original, expansion)
            },
            "original_supported_histories": len(original["payload"]["accepted"]),
            "expansion_supported_histories": len(expansion["payload"]["accepted"]),
            "union_supported_histories": len(accepted),
            "union_scene_clusters": len(scenes),
            "population_sha256": sha256_file(population_path),
            "population_seal_sha256": sha256_file(population_root / "SEALED"),
            "population_file_ledger_entries": population_ledger_entries,
            "target_met": target_met,
            "leg3_query_navigation_outcomes_read": False,
        }
        receipt_path = temporary / "union_receipt.json"
        receipt_path.write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "union_receipt.json.sha256").write_text(
            f"{sha256_file(receipt_path)}  union_receipt.json\n"
        )

        base_protocol = json.loads(base_table2_protocol_path.read_text())
        derived = derived_table2_protocol(
            base=base_protocol, out=out,
            population_sha=sha256_file(population_path),
            seal_sha=sha256_file(population_root / "SEALED"),
            source_histories=int(population["intention_to_collect_B"]),
            accepted=accepted, union_receipt_sha=sha256_file(receipt_path),
        )
        derived_path = temporary / "hm3d_table2_leg3_power_protocol.json"
        derived_path.write_text(json.dumps(
            derived, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "hm3d_table2_leg3_power_protocol.json.sha256").write_text(
            f"{sha256_file(derived_path)}  {derived_path.name}\n"
        )
        write_ledger(
            temporary, "UNION_FILES.sha256",
            excluded={"UNION_FILES.sha256", "POPULATION_FILES.sha256"},
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-protocol", type=Path, required=True)
    parser.add_argument("--expansion-run", type=Path, required=True)
    parser.add_argument("--base-table2-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        expansion_protocol_path=args.expansion_protocol.resolve(),
        expansion_run=args.expansion_run.resolve(),
        base_table2_protocol_path=args.base_table2_protocol.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
