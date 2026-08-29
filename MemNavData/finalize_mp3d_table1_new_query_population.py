#!/usr/bin/env python3
"""Seal all constructible MP3D new-query histories before policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from mp3d_table1_new_query_contract import (
        CONSTRUCTION_SEED,
        MINIMUM_PER_STRATUM,
        POPULATION_SCHEMA,
        SCENE_SCHEMA,
        SOURCE_LEDGER_SCHEMAS,
        TARGET_HISTORIES,
        TARGET_SCENES,
        assert_new_query_identity,
        power,
        require,
    )
    from shared_online_role_pair_contract import validate_manifest
except ImportError:
    from MemNavData.mp3d_table1_new_query_contract import (
        CONSTRUCTION_SEED,
        MINIMUM_PER_STRATUM,
        POPULATION_SCHEMA,
        SCENE_SCHEMA,
        SOURCE_LEDGER_SCHEMAS,
        TARGET_HISTORIES,
        TARGET_SCENES,
        assert_new_query_identity,
        power,
        require,
    )
    from MemNavData.shared_online_role_pair_contract import validate_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(*, construction_root: Path, source_ledger_path: Path,
             out: Path, target_histories: int = TARGET_HISTORIES,
             target_scenes: int = TARGET_SCENES,
             minimum_per_stratum: int = MINIMUM_PER_STRATUM) -> dict[str, Any]:
    require(not out.exists(), f"output already exists: {out}")
    ledger = json.loads(source_ledger_path.read_text())
    require(ledger.get("schema_version") in SOURCE_LEDGER_SCHEMAS,
            "source ledger schema changed")
    scene_count = int(ledger["scene_count"])
    require(scene_count == len(ledger["scenes"]) and scene_count > 0,
            "source scene count changed")
    ledger_sha = sha256_file(source_ledger_path)
    source_goals = {
        (str(scene["scene"]), str(episode["episode"])):
            episode.get("consumed_queries", [episode["consumed_goal_b"]])
        for scene in ledger["scenes"]
        for episode in scene["episodes"]
    }
    candidates = []
    fragments = []
    contract = None
    for scene_index in range(scene_count):
        matches = sorted(construction_root.glob(f"{scene_index:02d}_*/"))
        require(len(matches) == 1,
                f"scene index {scene_index} has {len(matches)} fragments")
        root = matches[0]
        receipt_path = root / "construction_receipt.json"
        manifest_path = root / "natural_direction/manifest.json"
        require(receipt_path.is_file() and manifest_path.is_file(),
                f"scene fragment incomplete: {root}")
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("schema_version") == SCENE_SCHEMA,
                "scene receipt schema changed")
        require(int(receipt["scene_index"]) == scene_index,
                "scene receipt index changed")
        require(receipt.get("source_ledger_sha256") == ledger_sha,
                "scene source ledger binding changed")
        require(receipt.get("previous_goal_b_policy_outcomes_read") is False
                and receipt.get("query_policy_outcomes_read") is False,
                "scene construction read a policy outcome")
        manifest = json.loads(manifest_path.read_text())
        if manifest["episodes"]:
            validate_manifest(manifest)
        if contract is None:
            contract = manifest["contract"]
        require(manifest["contract"] == contract,
                "scene query contract changed")
        for row in manifest["episodes"]:
            identity = str(row["scene"]), str(row["episode"])
            require(identity in source_goals,
                    f"retained history outside source ledger: {identity}")
            assert_new_query_identity(row, source_goals[identity])
            candidates.append({
                "source_root": root / "natural_direction",
                "row": row,
            })
        fragments.append({
            "scene": str(receipt["scene"]),
            "scene_index": scene_index,
            "source_histories": int(receipt["source_history_count"]),
            "goal_a_successes": int(receipt["goal_a_successes"]),
            "materialized_histories": int(receipt["materialized_histories"]),
            "query_construction_attempts": int(
                receipt["query_construction_attempts"]),
            "retained_histories": len(manifest["episodes"]),
            "consumed_goal_b_rejections": int(
                receipt["consumed_goal_b_rejections"]),
            "receipt_sha256": sha256_file(receipt_path),
            "manifest_sha256": sha256_file(manifest_path),
        })

    rows = [item["row"] for item in candidates]
    identities = [(str(row["scene"]), str(row["episode"])) for row in rows]
    require(len(identities) == len(set(identities)),
            "new-query population duplicates a history")
    rows.sort(key=lambda row: (
        int(row["final14_scene_rank"]),
        int(row["final14_source_episode_rank"]),
    ))
    observed_power = power(
        rows,
        target_histories=target_histories,
        target_scenes=target_scenes,
        minimum_per_stratum=minimum_per_stratum,
    )
    source_by_identity = {
        (str(item["row"]["scene"]), str(item["row"]["episode"])): item
        for item in candidates
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    try:
        target = temporary / "natural_direction"
        for row in rows:
            identity = str(row["scene"]), str(row["episode"])
            item = source_by_identity[identity]
            destination = target / identity[0] / identity[1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                item["source_root"] / identity[0] / identity[1], destination,
            )
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "MP3D Table-1 role-hidden new-query cross-controller replication"
            ),
            "source_online_root": (
                "per-fragment immutable actual-mono online-A root"
            ),
            "source_online_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(
                    fragment["manifest_sha256"] for fragment in fragments
                )).encode()
            ).hexdigest(),
            "source_online_manifest_sha256_semantics": (
                "sha256 of sorted per-scene new-query manifest hashes"
            ),
            "construction_seed": CONSTRUCTION_SEED,
            "contract": contract,
            "episodes": rows,
        }
        if rows:
            validate_manifest(manifest)
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        (target / "manifest.json.sha256").write_text(
            sha256_file(manifest_path) + "  manifest.json\n"
        )
        population = {
            "schema_version": POPULATION_SCHEMA,
            "scope": (
                "controlled MP3D reused-scene/history replication with new "
                "outcome-blind Novel/Revisit query images"
            ),
            "fresh_scene": False,
            "fresh_history": False,
            "new_query": True,
            "source_query_outcomes_read_for_selection": False,
            "previous_goal_b_policy_outcomes_read": False,
            "navigation_outcomes_generated": False,
            "source_ledger_sha256": ledger_sha,
            "construction_seed": CONSTRUCTION_SEED,
            "power_gate": observed_power,
            "fragments": fragments,
            "retained_histories": len(rows),
            "retained_scene_clusters": len({scene for scene, _ in identities}),
            "query_count": 2 * len(rows),
            "runtime_role_visibility": "none",
            "formal_policy_evaluation_authorized": bool(
                observed_power["target_met"]),
        }
        receipt_path = temporary / "population_receipt.json"
        receipt_path.write_text(json.dumps(
            population, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        runtime_parent = {
            "schema_version": "mp3d_table1_runtime_parent_v1_20260829",
            "purpose": "scene-rank ledger for the generic Table-1 runtime",
            "scenes": [str(row["scene"]) for row in ledger["scenes"]],
            "source_ledger_sha256": ledger_sha,
        }
        (temporary / "parent_manifest.json").write_text(json.dumps(
            runtime_parent, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        checksums = temporary / "CONSTRUCTION_FILES.sha256"
        checksums.write_text("".join(
            f"{sha256_file(path)}  {path.relative_to(temporary)}\n"
            for path in files
        ))
        (temporary / "CONSTRUCTION_FILES.sha256.sha256").write_text(
            sha256_file(checksums) + "  CONSTRUCTION_FILES.sha256\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-histories", type=int, default=TARGET_HISTORIES)
    parser.add_argument("--target-scenes", type=int, default=TARGET_SCENES)
    parser.add_argument(
        "--minimum-per-stratum", type=int, default=MINIMUM_PER_STRATUM,
    )
    args = parser.parse_args()
    result = finalize(
        construction_root=args.construction_root.resolve(),
        source_ledger_path=args.source_ledger.resolve(),
        out=args.out.resolve(),
        target_histories=args.target_histories,
        target_scenes=args.target_scenes,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    print(json.dumps({
        "retained_histories": result["retained_histories"],
        "retained_scene_clusters": result["retained_scene_clusters"],
        "power_gate": result["power_gate"],
        "formal_policy_evaluation_authorized": result[
            "formal_policy_evaluation_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
