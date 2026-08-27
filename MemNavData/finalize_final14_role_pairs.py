#!/usr/bin/env python3
"""Merge final14 per-scene fragments into sealed-ready query populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from audit_shared_online_role_pairs import audit as audit_role_pairs
from final14_role_pair_contract import (
    POPULATION_SCHEMA,
    PROTOCOLS,
    SCENE_BUILD_SCHEMA,
)
from shared_online_role_pair_contract import validate_manifest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _population_summary(
    identities: set[tuple[str, str]],
    *,
    target_histories: int,
    target_scenes: int,
) -> dict[str, Any]:
    histories = len(identities)
    scenes = len({scene for scene, _episode in identities})
    target_met = histories >= target_histories and scenes >= target_scenes
    return {
        "histories": histories,
        "scenes": scenes,
        "target_histories": int(target_histories),
        "target_scenes": int(target_scenes),
        "target_met": target_met,
        "underpowered_if_target_not_met": not target_met,
        "identities": [
            {"scene": scene, "episode": episode}
            for scene, episode in sorted(identities)
        ],
    }


def finalize(
    run_root: Path,
    out: Path,
    *,
    expected_scene_count: int = 14,
    natural_target_histories: int = 28,
    natural_target_scenes: int = 10,
    hard_target_histories: int = 16,
    hard_target_scenes: int = 8,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    inventory_path = run_root / "online_a_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    require(
        int(inventory["source_scenes"]) == expected_scene_count,
        "online-A inventory scene count differs from final14",
    )
    scene_roots = sorted(
        path for path in (run_root / "traces").iterdir() if path.is_dir()
    )
    require(
        len(scene_roots) == expected_scene_count,
        f"expected all {expected_scene_count} scene fragments",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    rows: dict[str, list[dict]] = {name: [] for name in PROTOCOLS}
    identities: dict[str, set[tuple[str, str]]] = {
        name: set() for name in PROTOCOLS
    }
    contracts: dict[str, dict | None] = {name: None for name in PROTOCOLS}
    seeds: dict[str, int | None] = {name: None for name in PROTOCOLS}
    source_online_hashes = []
    attrition = []
    fragment_receipts = []
    try:
        for expected_rank, scene_root in enumerate(scene_roots):
            require(
                scene_root.name.startswith(f"{expected_rank:02d}_"),
                "scene fragment rank/order changed",
            )
            build_root = scene_root / "role_pairs"
            construction_path = build_root / "construction_receipt.json"
            construction = json.loads(construction_path.read_text())
            require(
                construction.get("schema_version") == SCENE_BUILD_SCHEMA,
                "final14 scene construction schema changed",
            )
            require(
                int(construction["scene_rank"]) == expected_rank,
                "construction scene rank changed",
            )
            require(
                construction["policy_outcomes_read"] is False,
                "query outcome leaked into construction",
            )
            require(
                construction["all_materialized_histories_attempted"] is True,
                "eligible materialized history was not attempted",
            )
            require(
                int(construction["maximum_retained_histories_per_scene"]) == 3,
                "per-scene cap changed",
            )
            require(
                int(construction["retained_standard_natural_histories"]) <= 3,
                "per-scene cap was exceeded",
            )
            attrition.extend(construction["attrition"])
            fragment = {
                "scene_root": scene_root.name,
                "scene_rank": expected_rank,
                "construction_receipt_sha256": sha256_file(construction_path),
                "protocol_manifests": {},
            }
            online_manifest_path = scene_root / "online_a" / "manifest.json"
            online_sha = sha256_file(online_manifest_path)
            source_online_hashes.append(online_sha)
            fragment["source_online_manifest_sha256"] = online_sha
            for protocol in PROTOCOLS:
                source_root = build_root / protocol
                manifest_path = source_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                fragment["protocol_manifests"][protocol] = sha256_file(
                    manifest_path
                )
                if contracts[protocol] is None:
                    contracts[protocol] = manifest["contract"]
                    seeds[protocol] = int(manifest["construction_seed"])
                require(
                    manifest["contract"] == contracts[protocol],
                    f"{protocol} contract drift",
                )
                require(
                    int(manifest["construction_seed"]) == seeds[protocol],
                    f"{protocol} construction seed drift",
                )
                require(
                    len(manifest["episodes"]) <= 3,
                    f"{protocol} per-scene population exceeds cap",
                )
                previous_rank = -1
                for episode in manifest["episodes"]:
                    identity = (str(episode["scene"]), str(episode["episode"]))
                    require(
                        identity not in identities[protocol],
                        f"duplicate {protocol} history identity",
                    )
                    episode_rank = int(episode["final14_source_episode_rank"])
                    require(
                        episode_rank > previous_rank,
                        f"{protocol} histories are not in frozen source order",
                    )
                    previous_rank = episode_rank
                    require(
                        int(episode["final14_scene_rank"]) == expected_rank,
                        f"{protocol} episode scene rank changed",
                    )
                    identities[protocol].add(identity)
                    source_episode = source_root / identity[0] / identity[1]
                    destination = temporary / protocol / identity[0] / identity[1]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_episode, destination)
                    rows[protocol].append(episode)
            fragment_receipts.append(fragment)

        require(
            identities["hard_support"].issubset(
                identities["natural_direction"]
            ),
            "hard-support population is not a subset of retained base histories",
        )
        online_aggregate = aggregate_hash(source_online_hashes)
        audits = {}
        for protocol in PROTOCOLS:
            root = temporary / protocol
            root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": "shared_online_role_pair_v1_20260814",
                "purpose": (
                    "final14 natural unsupported Novel plus standard-support Revisit"
                    if protocol == "natural_direction"
                    else "final14 hard-support Revisit subset"
                ),
                "source_online_root": str((run_root / "traces").resolve()),
                "source_online_manifest_sha256": online_aggregate,
                "source_online_manifest_sha256_semantics": (
                    "sha256 of sorted per-scene manifest SHA256 values"
                ),
                "construction_seed": seeds[protocol],
                "contract": contracts[protocol],
                "episodes": sorted(
                    rows[protocol],
                    key=lambda row: (
                        int(row["final14_scene_rank"]),
                        int(row["final14_source_episode_rank"]),
                    ),
                ),
            }
            require(bool(manifest["episodes"]), f"{protocol} population is empty")
            validate_manifest(manifest)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            (root / "manifest.json.sha256").write_text(
                sha256_file(manifest_path) + "  manifest.json\n"
            )
            audits[protocol] = audit_role_pairs(root)

        populations = {
            "natural_standard": _population_summary(
                identities["natural_direction"],
                target_histories=natural_target_histories,
                target_scenes=natural_target_scenes,
            ),
            "hard_support": _population_summary(
                identities["hard_support"],
                target_histories=hard_target_histories,
                target_scenes=hard_target_scenes,
            ),
        }
        receipt = {
            "schema_version": POPULATION_SCHEMA,
            "protocol": "FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md",
            "scope": "construction only; no memory query policy executed",
            "source_manifest_sha256": inventory["manifest_sha256"],
            "source_episodes": int(inventory["source_episodes"]),
            "goal_a_successes": int(inventory["goal_a_successes"]),
            "materialized_histories": int(inventory["materialized_histories"]),
            "populations": populations,
            # Compatibility aliases always refer to the primary natural +
            # standard population, never to the smaller hard subset.
            "role_pair_constructible_histories": populations[
                "natural_standard"
            ]["histories"],
            "role_pair_scene_count": populations["natural_standard"]["scenes"],
            "target_histories": natural_target_histories,
            "target_scenes": natural_target_scenes,
            "target_met": populations["natural_standard"]["target_met"],
            "underpowered_if_target_not_met": not (
                populations["natural_standard"]["target_met"]
                and populations["hard_support"]["target_met"]
            ),
            "paper_policy_evaluation_authorized": True,
            "attrition_count": len(attrition),
            "attrition": attrition,
            "fragment_receipts": fragment_receipts,
            "protocol_audits": audits,
            "policy_outcomes_read": False,
            "hard_novel_queries_are_instrumentation_only": True,
        }
        (temporary / "population_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=14)
    parser.add_argument("--natural-target-histories", type=int, default=28)
    parser.add_argument("--natural-target-scenes", type=int, default=10)
    parser.add_argument("--hard-target-histories", type=int, default=16)
    parser.add_argument("--hard-target-scenes", type=int, default=8)
    args = parser.parse_args()
    result = finalize(
        args.run_root,
        args.out,
        expected_scene_count=args.expected_scene_count,
        natural_target_histories=args.natural_target_histories,
        natural_target_scenes=args.natural_target_scenes,
        hard_target_histories=args.hard_target_histories,
        hard_target_scenes=args.hard_target_scenes,
    )
    print(json.dumps({
        "populations": result["populations"],
        "underpowered_if_target_not_met": result[
            "underpowered_if_target_not_met"
        ],
        "policy_outcomes_read": result["policy_outcomes_read"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
