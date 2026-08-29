#!/usr/bin/env python3
"""Seal all construction-only Table-2 Leg-3 fragments into one population."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import build_final14_role_pair_scene as role_builder
from hm3d_table2_leg3_mixed_role import (
    FRAGMENT_SCHEMA,
    POPULATION_SCHEMA,
    load_protocol,
    power,
    require,
    sha256_file,
)
from shared_online_role_pair_contract import validate_manifest


def finalize(*, protocol_path: Path, source_root: Path, fragments: Path,
             out: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    source_contract = protocol["source_population"]
    require(source_root.resolve() == Path(source_contract["run_root"]).resolve(),
            "Table-2 source root changed")
    source_population_path = source_root / source_contract["population"]
    require(sha256_file(source_population_path)
            == source_contract["population_sha256"],
            "Table-2 source population changed")
    source_population = json.loads(source_population_path.read_text())
    source_rows = list(source_population["accepted"])
    require(len(source_rows) == int(source_contract[
        "actual_AB_successful_histories"
    ]), "Table-2 source A/B prefix count changed")
    require(not out.exists(), f"Table-2 population exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    accepted: list[dict[str, Any]] = []
    attrition: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    try:
        benchmark_root = temporary / "natural_direction"
        prefix_root = temporary / "causal_prefix"
        for index, source_row in enumerate(source_rows):
            matches = sorted(fragments.glob(f"{index:03d}_*"))
            require(len(matches) == 1,
                    f"Table-2 fragment {index} has {len(matches)} matches")
            fragment = matches[0]
            completion_path = fragment / "completion.json"
            require(completion_path.is_file(),
                    f"Table-2 fragment {index} has no completion")
            completion = json.loads(completion_path.read_text())
            require(completion.get("schema_version") == FRAGMENT_SCHEMA
                    and completion.get("status") == "complete",
                    f"Table-2 fragment {index} is incomplete")
            require(int(completion["population_index"]) == index,
                    "Table-2 fragment index changed")
            require(completion["protocol_sha256"] == sha256_file(protocol_path),
                    "Table-2 fragment protocol changed")
            require(completion["source_population_sha256"]
                    == sha256_file(source_population_path),
                    "Table-2 fragment source population changed")
            require(completion.get("leg3_query_policy_outcomes_read") is False
                    and completion.get("old_goal_C_navigation_outcomes_read")
                    is False, "Table-2 construction read a query outcome")
            inputs.append({
                "population_index": index,
                "scene": completion["scene"],
                "episode": completion["episode"],
                "completion_sha256": sha256_file(completion_path),
                "eligible": bool(completion["eligible"]),
            })
            if not completion["eligible"]:
                attrition.append({
                    "population_index": index,
                    "scene": completion["scene"],
                    "episode": completion["episode"],
                    "reason": completion["attrition_reason"],
                    "completion_sha256": sha256_file(completion_path),
                })
                continue
            scene, episode = completion["scene"], completion["episode"]
            source_prefix = fragment / "causal_prefix" / scene / episode
            source_episode = fragment / "role_pair" / scene / episode
            require(source_prefix.is_dir() and source_episode.is_dir(),
                    f"Table-2 eligible fragment {index} lacks assets")
            destination_prefix = prefix_root / scene / episode
            destination_episode = benchmark_root / scene / episode
            destination_prefix.parent.mkdir(parents=True, exist_ok=True)
            destination_episode.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_prefix, destination_prefix)
            shutil.copytree(source_episode, destination_episode)

            sidecar = destination_episode / "role_pairs.json"
            row = json.loads(sidecar.read_text())
            require(row["scene"] == scene and row["episode"] == episode,
                    "Table-2 copied role-pair identity changed")
            row["online_a_episode"] = str(
                (out / "causal_prefix" / scene / episode).resolve()
            )
            row["online_a_receipt_sha256"] = sha256_file(
                destination_prefix / "receipt.json"
            )
            row["online_a_trace_sha256"] = sha256_file(
                destination_prefix / "online_a_trace.json"
            )
            sidecar.write_text(json.dumps(
                row, indent=2, sort_keys=True, allow_nan=False,
            ) + "\n")
            row["role_pairs_sha256"] = sha256_file(sidecar)
            accepted.append(row)

        accepted.sort(key=lambda row: int(row["table2_source_population_index"]))
        contract = role_builder.role_contract(support="standard")
        contract.update({
            "online_history_semantics": (
                "hash_verified_actual_mono_Novel_A_then_Novel_B_prefix"
            ),
            "query_execution": "independent_reset_and_exact_online_a_replay",
            "old_goal_B_and_C_identities_forbidden": True,
            "leg3_queries_constructed_before_policy_rollout": True,
        })
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "HM3D continual Table-2 new Novel/Revisit Leg-3 pairs after "
                "one exact actual-mono A/B prefix"
            ),
            "source_online_root": str((out / "causal_prefix").resolve()),
            "source_online_manifest_sha256": sha256_file(
                source_population_path
            ),
            "construction_seed": 20260829,
            "contract": contract,
            "episodes": accepted,
        }
        if accepted:
            validate_manifest(manifest)
        benchmark_root.mkdir(parents=True, exist_ok=True)
        manifest_path = benchmark_root / "manifest.json"
        manifest_path.write_text(json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        (benchmark_root / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        gate = protocol["population_gate"]
        observed_power = power(
            accepted,
            target_histories=int(gate["minimum_histories"]),
            target_scenes=int(gate["minimum_scene_clusters"]),
            minimum_per_stratum=int(
                gate["minimum_histories_per_direction_stratum"]
            ),
        )
        population = {
            "schema_version": POPULATION_SCHEMA,
            "scope": protocol["scope"],
            "protocol_sha256": sha256_file(protocol_path),
            "source_population": str(source_population_path.resolve()),
            "source_population_sha256": sha256_file(source_population_path),
            "source_A_attempts": int(source_contract["source_histories"]),
            "factual_AB_successful_prefixes": len(source_rows),
            "factual_AB_scene_clusters": int(
                source_contract["actual_AB_scene_clusters"]
            ),
            "leg3_constructible_histories": len(accepted),
            "leg3_scene_clusters": len({row["scene"] for row in accepted}),
            "leg3_query_count": 2 * len(accepted),
            "power_gate": observed_power,
            "formal_policy_evaluation_authorized": bool(
                observed_power["target_met"]
            ),
            "runtime_role_visibility": "none",
            "navigation_outcomes_generated": False,
            "query_outcomes_read_for_selection": False,
            "old_goal_C_outcomes_read_for_construction": False,
            "accepted": [
                {
                    "population_index": index,
                    "source_population_index": int(
                        row["table2_source_population_index"]
                    ),
                    "scene": row["scene"],
                    "episode": row["episode"],
                    "role_pair_sha256": row["role_pairs_sha256"],
                    "selected_revisit_segment": row[
                        "table2_selected_revisit_segment"
                    ],
                }
                for index, row in enumerate(accepted)
            ],
            "attrition": attrition,
            "construction_inputs": inputs,
        }
        receipt_path = temporary / "population_receipt.json"
        receipt_path.write_text(json.dumps(
            population, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name not in {
                "CONSTRUCTION_FILES.sha256", "SEALED",
            }
        )
        checksum = temporary / "CONSTRUCTION_FILES.sha256"
        checksum.write_text("".join(
            f"{sha256_file(path)}  {path.relative_to(temporary)}\n"
            for path in files
        ))
        (temporary / "SEALED").write_text(
            "sealed construction before any Table-2 Leg-3 policy rollout\n"
        )
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        protocol_path=args.protocol.resolve(),
        source_root=args.source_root.resolve(),
        fragments=args.fragments.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "leg3_constructible_histories": result[
            "leg3_constructible_histories"
        ],
        "leg3_scene_clusters": result["leg3_scene_clusters"],
        "power_gate": result["power_gate"],
        "formal_policy_evaluation_authorized": result[
            "formal_policy_evaluation_authorized"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
