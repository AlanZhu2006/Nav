#!/usr/bin/env python3
"""Merge immutable per-scene construction fragments into paper benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from audit_shared_online_role_pairs import audit as audit_role_pairs
from shared_online_role_pair_contract import validate_manifest


PROTOCOLS = ("support_controlled", "natural_direction")


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
    encoded = "\n".join(sorted(values)).encode()
    return hashlib.sha256(encoded).hexdigest()


def finalize(
    run_root: Path,
    out: Path,
    *,
    expected_scene_count: int = 16,
    target_histories: int = 20,
    target_scenes: int = 12,
    benchmark_scope: str = "paper",
) -> dict:
    if out.exists():
        raise FileExistsError(out)
    inventory = json.loads((run_root / "online_a_inventory.json").read_text())
    scene_roots = sorted(path for path in (run_root / "traces").iterdir() if path.is_dir())
    require(
        len(scene_roots) == expected_scene_count,
        f"expected all {expected_scene_count} construction fragments",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    rows = {name: [] for name in PROTOCOLS}
    identities = {name: set() for name in PROTOCOLS}
    contracts = {name: None for name in PROTOCOLS}
    seeds = {name: None for name in PROTOCOLS}
    source_online_hashes = []
    source_revisit_hashes = []
    role_attrition = []
    fragment_receipts = []
    try:
        for scene_root in scene_roots:
            build_root = scene_root / "role_pairs"
            construction_path = build_root / "construction_receipt.json"
            construction = json.loads(construction_path.read_text())
            require(
                construction.get("schema_version")
                == "paper_role_pair_scene_build_v2_20260814",
                "construction amendment schema changed",
            )
            require(
                construction.get("query_contract")
                == "one_independent_revisit_query_after_online_a",
                "single-Revisit query contract changed",
            )
            require(construction["policy_outcomes_read"] is False, "query outcome leak")
            role_attrition.extend(construction["attrition"])
            fragment = {
                "scene_root": scene_root.name,
                "construction_receipt_sha256": sha256_file(construction_path),
                "source_online_manifest_sha256": sha256_file(
                    scene_root / "online_a" / "manifest.json"
                ),
                "protocol_manifests": {},
            }
            source_online_hashes.append(fragment["source_online_manifest_sha256"])
            revisit_manifest_path = build_root / "revisit_source" / "manifest.json"
            revisit_manifest = json.loads(revisit_manifest_path.read_text())
            require(
                revisit_manifest.get("schema_version")
                == "single_revisit_source_v1_20260814",
                "legacy double-Revisit source entered paper population",
            )
            source_revisit_hashes.append(sha256_file(revisit_manifest_path))
            for name in PROTOCOLS:
                source_root = build_root / name
                manifest_path = source_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                fragment["protocol_manifests"][name] = sha256_file(manifest_path)
                if contracts[name] is None:
                    contracts[name] = manifest["contract"]
                    seeds[name] = int(manifest["construction_seed"])
                require(manifest["contract"] == contracts[name], "contract drift")
                require(int(manifest["construction_seed"]) == seeds[name], "seed drift")
                for episode in manifest["episodes"]:
                    identity = (str(episode["scene"]), str(episode["episode"]))
                    require(identity not in identities[name], "duplicate history identity")
                    identities[name].add(identity)
                    source_episode = source_root / identity[0] / identity[1]
                    destination = temporary / name / identity[0] / identity[1]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_episode, destination)
                    rows[name].append(episode)
            fragment_receipts.append(fragment)

        require(
            identities["support_controlled"] == identities["natural_direction"],
            "the two protocols do not share the same constructed population",
        )
        online_aggregate = aggregate_hash(source_online_hashes)
        revisit_aggregate = aggregate_hash(source_revisit_hashes)
        audits = {}
        for name in PROTOCOLS:
            root = temporary / name
            root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": "shared_online_role_pair_v1_20260814",
                "purpose": (
                    "paper support-controlled role-free Novel/Revisit benchmark"
                    if name == "support_controlled"
                    else "paper natural-direction role-free Novel/Revisit benchmark"
                ),
                "source_online_root": str((run_root / "traces").resolve()),
                "source_online_manifest_sha256": online_aggregate,
                "source_online_manifest_sha256_semantics": (
                    "sha256 of sorted per-scene manifest SHA256 values"
                ),
                "source_revisit_root": str((run_root / "traces").resolve()),
                "source_revisit_manifest_sha256": revisit_aggregate,
                "source_revisit_manifest_sha256_semantics": (
                    "sha256 of sorted per-scene manifest SHA256 values"
                ),
                "construction_seed": seeds[name],
                "contract": contracts[name],
                "episodes": sorted(
                    rows[name], key=lambda row: (row["scene"], row["episode"])
                ),
            }
            require(bool(manifest["episodes"]), "no constructible paper histories")
            validate_manifest(manifest)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            (root / "manifest.json.sha256").write_text(
                sha256_file(manifest_path) + "  manifest.json\n"
            )
            audits[name] = audit_role_pairs(root)

        population = sorted(identities["support_controlled"])
        scene_count = len({scene for scene, _episode in population})
        target_met = (
            len(population) >= target_histories
            and scene_count >= target_scenes
        )
        receipt = {
            "schema_version": "paper_role_pair_population_v2_20260814",
            "query_contract": "one_independent_revisit_query_after_online_a",
            "benchmark_scope": benchmark_scope,
            "scope": "construction only; no memory policy query executed",
            "source_manifest_sha256": inventory["manifest_sha256"],
            "source_episodes": inventory["source_episodes"],
            "goal_a_successes": inventory["goal_a_successes"],
            "materialized_histories": inventory["materialized_histories"],
            "role_pair_constructible_histories": len(population),
            "role_pair_scene_count": scene_count,
            "target_histories": target_histories,
            "target_scenes": target_scenes,
            "target_met": target_met,
            "paper_policy_evaluation_authorized": True,
            "underpowered_if_target_not_met": not target_met,
            "role_pair_attrition_count": len(role_attrition),
            "role_pair_attrition": role_attrition,
            "population": [
                {"scene": scene, "episode": episode}
                for scene, episode in population
            ],
            "fragment_receipts": fragment_receipts,
            "protocol_audits": audits,
            "policy_outcomes_read": False,
        }
        (temporary / "population_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
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
    parser.add_argument("--expected-scene-count", type=int, default=16)
    parser.add_argument("--target-histories", type=int, default=20)
    parser.add_argument("--target-scenes", type=int, default=12)
    parser.add_argument("--benchmark-scope", default="paper")
    args = parser.parse_args()
    if args.expected_scene_count < 1:
        parser.error("--expected-scene-count must be positive")
    if args.target_histories < 1 or args.target_scenes < 1:
        parser.error("population targets must be positive")
    result = finalize(
        args.run_root,
        args.out,
        expected_scene_count=args.expected_scene_count,
        target_histories=args.target_histories,
        target_scenes=args.target_scenes,
        benchmark_scope=args.benchmark_scope,
    )
    print(json.dumps({
        key: result[key] for key in (
            "goal_a_successes", "materialized_histories",
            "role_pair_constructible_histories", "role_pair_scene_count",
            "target_met", "underpowered_if_target_not_met",
            "policy_outcomes_read",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
