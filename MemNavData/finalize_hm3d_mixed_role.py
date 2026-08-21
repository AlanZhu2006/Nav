#!/usr/bin/env python3
"""Audit scene fragments and seal the reused-HM3D mixed-role population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from audit_shared_online_role_pairs import audit as audit_role_pairs
from shared_online_role_pair_contract import validate_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    args = parser.parse_args()
    construction = args.run_root / "construction"
    traces_root = construction / "traces"
    scene_roots = sorted(path for path in traces_root.iterdir() if path.is_dir())
    if len(scene_roots) != 9:
        raise RuntimeError(f"expected nine scene fragments, found {len(scene_roots)}")
    source_episodes = 0
    goal_a_successes = 0
    materialized = 0
    natural = 0
    fragments = []
    for rank, scene_root in enumerate(scene_roots):
        if not scene_root.name.startswith(f"{rank:02d}_"):
            raise RuntimeError("scene ranks are not contiguous")
        completion_path = scene_root / "completion.json"
        if sha256_file(completion_path) != (
            scene_root / "completion.json.sha256"
        ).read_text().split()[0]:
            raise RuntimeError(f"completion hash changed: {scene_root}")
        row = json.loads(completion_path.read_text())
        if row["query_policy_outcomes_read"] is not False:
            raise RuntimeError("construction read a query outcome")
        source_episodes += len(row["source_episodes"])
        online = json.loads((scene_root / "online_a/manifest.json").read_text())
        goal_a_successes += sum(
            1 for episode in online["episodes"]
            if episode.get("online_a_reached") is True
        ) + sum(
            1 for item in online.get("attrition", [])
            if item.get("reason") not in {"native_a_failed"}
        )
        materialized += len(online["episodes"])
        natural += int(row["natural_histories"])
        fragments.append(row)
    parent_sha = sha256_file(args.parent_manifest)
    inventory = {
        "schema_version": "hm3d_mixed_role_online_a_inventory_v1_20260818",
        "manifest_sha256": parent_sha,
        "source_scenes": 9,
        "source_episodes": source_episodes,
        "goal_a_successes": goal_a_successes,
        "materialized_histories": materialized,
        "natural_constructible_histories_before_finalize": natural,
        "policy_outcomes_read": False,
        "scene_reuse_disclosed": True,
        "fragments": fragments,
    }
    inventory_path = construction / "online_a_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    (construction / "online_a_inventory.json.sha256").write_text(
        sha256_file(inventory_path) + "  online_a_inventory.json\n"
    )
    benchmark_root = args.run_root / "benchmarks"
    if benchmark_root.exists():
        raise FileExistsError(benchmark_root)
    benchmark_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=benchmark_root.name + ".tmp.", dir=benchmark_root.parent
    ))
    rows = []
    identities = set()
    contract = None
    construction_seed = None
    fragment_receipts = []
    try:
        target = temporary / "natural_direction"
        for scene_root in scene_roots:
            source = scene_root / "role_pairs/natural_direction"
            source_manifest_path = source / "manifest.json"
            source_manifest = json.loads(source_manifest_path.read_text())
            if contract is None:
                contract = source_manifest["contract"]
                construction_seed = int(source_manifest["construction_seed"])
            if source_manifest["contract"] != contract:
                raise RuntimeError("scene fragments changed the query contract")
            if int(source_manifest["construction_seed"]) != construction_seed:
                raise RuntimeError("scene fragments changed the construction seed")
            fragment_receipts.append({
                "scene_root": scene_root.name,
                "manifest_sha256": sha256_file(source_manifest_path),
                "histories": len(source_manifest["episodes"]),
            })
            for row in source_manifest["episodes"]:
                identity = (str(row["scene"]), str(row["episode"]))
                if identity in identities:
                    raise RuntimeError(f"duplicate history {identity}")
                identities.add(identity)
                source_episode = source / identity[0] / identity[1]
                destination = target / identity[0] / identity[1]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_episode, destination)
                rows.append(row)
        if not rows:
            raise RuntimeError("HM3D mixed-role population is empty")
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "reused-HM3D natural unsupported Novel plus "
                "standard-support Revisit safety extension"
            ),
            "source_online_root": str(traces_root.resolve()),
            "source_online_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(
                    sha256_file(scene_root / "online_a/manifest.json")
                    for scene_root in scene_roots
                )).encode()
            ).hexdigest(),
            "source_online_manifest_sha256_semantics": (
                "sha256 of sorted per-scene online manifest SHA256 values"
            ),
            "construction_seed": construction_seed,
            "contract": contract,
            "episodes": sorted(
                rows,
                key=lambda row: (
                    int(row["final14_scene_rank"]),
                    int(row["final14_source_episode_rank"]),
                ),
            ),
        }
        validate_manifest(manifest)
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (target / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        audit = audit_role_pairs(target)
        population = {
            "schema_version": "hm3d_mixed_role_population_v1_20260818",
            "scope": (
                "same-scene HM3D mixed-role safety extension; training-free "
                "but not a new scene-disjoint confirmation"
            ),
            "histories": len(identities),
            "scenes": len({scene for scene, _episode in identities}),
            "target_histories": 18,
            "target_scenes": 8,
            "target_met": (
                len(identities) >= 18
                and len({scene for scene, _episode in identities}) >= 8
            ),
            "runtime_role_visibility": "none",
            "policy_outcomes_read": False,
            "scene_reuse_disclosed": True,
            "identities": [
                {"scene": scene, "episode": episode}
                for scene, episode in sorted(identities)
            ],
            "fragment_receipts": fragment_receipts,
            "benchmark_audit": audit,
        }
        (temporary / "population_receipt.json").write_text(
            json.dumps(population, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(benchmark_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    files = sorted(
        path for path in benchmark_root.rglob("*")
        if path.is_file() and path.name not in {
            "BENCHMARK_FILES.sha256", "BENCHMARK_FILES.sha256.sha256", "SEALED"
        }
    )
    checksum_path = benchmark_root / "BENCHMARK_FILES.sha256"
    checksum_path.write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(benchmark_root)}\n"
        for path in files
    ))
    (benchmark_root / "BENCHMARK_FILES.sha256.sha256").write_text(
        sha256_file(checksum_path) + "  BENCHMARK_FILES.sha256\n"
    )
    (benchmark_root / "SEALED").write_text(
        sha256_file(benchmark_root / "natural_direction/manifest.json") + "\n"
    )
    print(json.dumps({
        "inventory": inventory,
        "population": population,
        "benchmark_files": len(files),
        "natural_manifest_sha256": sha256_file(
            benchmark_root / "natural_direction/manifest.json"
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
