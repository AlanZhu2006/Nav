#!/usr/bin/env python3
"""Seal a construction-only HM3D Table-1 fresh-query population."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from hm3d_table1_fresh_query_contract import (
        CONSTRUCTION_SEED,
        POPULATION_SCHEMA,
        PREFIX_SCHEDULE,
        SCENE_SCHEMA,
        identity_set,
        power,
        require,
    )
    from shared_online_role_pair_contract import validate_manifest
except ImportError:
    from MemNavData.hm3d_table1_fresh_query_contract import (
        CONSTRUCTION_SEED,
        POPULATION_SCHEMA,
        PREFIX_SCHEDULE,
        SCENE_SCHEMA,
        identity_set,
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


def finalize(*, construction_root: Path, original_manifest_path: Path,
             out: Path, target_histories: int = 24,
             target_scenes: int = 15,
             minimum_per_stratum: int = 4) -> dict[str, Any]:
    require(not out.exists(), f"output already exists: {out}")
    require(target_histories > 0 and target_scenes > 0,
            "power targets must be positive")
    require(minimum_per_stratum > 0,
            "direction-stratum target must be positive")
    consumed_manifest = json.loads(original_manifest_path.read_text())
    consumed = identity_set(consumed_manifest)
    fragments = []
    candidates = []
    construction_seed = None
    contract = None
    for scene_index in range(PREFIX_SCHEDULE[-1]):
        matches = sorted(construction_root.glob(f"{scene_index:02d}_*/"))
        require(len(matches) == 1,
                f"scene index {scene_index} has {len(matches)} fragments")
        root = matches[0]
        receipt_path = root / "construction_receipt.json"
        require(receipt_path.is_file(), f"missing receipt {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("schema_version") == SCENE_SCHEMA,
                "scene receipt schema changed")
        require(int(receipt["scene_index"]) == scene_index,
                "scene receipt index changed")
        require(receipt.get("query_policy_outcomes_read") is False,
                "construction read query outcomes")
        require(receipt.get("consumed_manifest_sha256")
                == sha256_file(original_manifest_path),
                "scene exclusion ledger changed")
        require(int(receipt["construction_seed"]) == CONSTRUCTION_SEED,
                "construction seed changed")
        manifest_path = root / "natural_direction/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["episodes"]:
            validate_manifest(manifest)
        if construction_seed is None:
            construction_seed = int(manifest["construction_seed"])
            contract = manifest["contract"]
        require(int(manifest["construction_seed"]) == construction_seed,
                "fragment construction seed changed")
        require(manifest["contract"] == contract,
                "fragment query contract changed")
        rows = list(manifest["episodes"])
        for row in rows:
            identity = (str(row["scene"]), str(row["episode"]))
            require(identity not in consumed,
                    f"consumed identity leaked into reserve: {identity}")
            candidates.append({
                "scene_index": scene_index,
                "source_root": root / "natural_direction",
                "row": row,
            })
        fragments.append({
            "scene": receipt["scene"],
            "scene_index": scene_index,
            "source_materialized_histories": int(
                receipt["source_materialized_histories"]),
            "source_online_manifest_sha256": receipt[
                "source_online_manifest_sha256"
            ],
            "consumed_identities_excluded": int(
                receipt["consumed_identities_excluded"]),
            "reserve_histories_attempted": int(
                receipt["reserve_histories_attempted"]),
            "retained_histories": len(rows),
            "receipt_sha256": sha256_file(receipt_path),
            "fragment_manifest_sha256": sha256_file(manifest_path),
        })

    selected_prefix = PREFIX_SCHEDULE[-1]
    selected_rows: list[dict[str, Any]] = []
    selected_power = None
    for prefix in PREFIX_SCHEDULE:
        rows = [
            item["row"] for item in candidates
            if int(item["scene_index"]) < prefix
        ]
        current = power(
            rows,
            target_histories=target_histories,
            target_scenes=target_scenes,
            minimum_per_stratum=minimum_per_stratum,
        )
        if current["target_met"]:
            selected_prefix = prefix
            selected_rows = rows
            selected_power = current
            break
    if selected_power is None:
        selected_rows = [item["row"] for item in candidates]
        selected_power = power(
            selected_rows,
            target_histories=target_histories,
            target_scenes=target_scenes,
            minimum_per_stratum=minimum_per_stratum,
        )

    identities = {
        (str(row["scene"]), str(row["episode"])) for row in selected_rows
    }
    require(len(identities) == len(selected_rows),
            "reserve population duplicates histories")
    selected_items = {
        (str(item["row"]["scene"]), str(item["row"]["episode"])): item
        for item in candidates if int(item["scene_index"]) < selected_prefix
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    try:
        target = temporary / "natural_direction"
        for row in selected_rows:
            identity = (str(row["scene"]), str(row["episode"]))
            item = selected_items[identity]
            destination = target / identity[0] / identity[1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                item["source_root"] / identity[0] / identity[1], destination,
            )
        selected_rows.sort(key=lambda row: (
            int(row["final14_scene_rank"]),
            int(row["final14_source_episode_rank"]),
        ))
        for fragment in fragments:
            fragment["selected_for_population"] = (
                int(fragment["scene_index"]) < selected_prefix
            )
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "HM3D Table-1 fresh-query controller-portability reserve"
            ),
            "source_online_root": "per-fragment absolute source bound in episode",
            "source_online_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(
                    str(fragment["source_online_manifest_sha256"])
                    for fragment in fragments
                    if fragment["selected_for_population"]
                    and fragment["source_online_manifest_sha256"] is not None
                )).encode()
            ).hexdigest(),
            "source_online_manifest_sha256_semantics": (
                "sha256 of sorted selected per-scene online-A manifest hashes"
            ),
            "construction_seed": construction_seed,
            "contract": contract,
            "episodes": selected_rows,
        }
        if selected_rows:
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
                "construction-only fresh query identities; scene-overlap with "
                "the earlier HM3D evaluation is explicit"
            ),
            "source_population": "HM3D fresh full-mono 20260820",
            "source_query_outcomes_read_for_selection": False,
            "navigation_outcomes_generated": False,
            "original_consumed_identity_count": len(consumed),
            "consumed_identity_overlap": 0,
            "construction_seed": construction_seed,
            "prefix_schedule": list(PREFIX_SCHEDULE),
            "selected_scene_prefix": selected_prefix,
            "power_gate": selected_power,
            "fragments": fragments,
            "retained_histories": len(selected_rows),
            "retained_scene_clusters": len({
                str(row["scene"]) for row in selected_rows
            }),
            "query_count": 2 * len(selected_rows),
            "runtime_role_visibility": "none",
            "formal_policy_evaluation_authorized": bool(
                selected_power["target_met"]),
        }
        receipt_path = temporary / "population_receipt.json"
        receipt_path.write_text(json.dumps(
            population, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        checksum = temporary / "CONSTRUCTION_FILES.sha256"
        checksum.write_text("".join(
            f"{sha256_file(path)}  {path.relative_to(temporary)}\n"
            for path in files
        ))
        (temporary / "CONSTRUCTION_FILES.sha256.sha256").write_text(
            sha256_file(checksum) + "  CONSTRUCTION_FILES.sha256\n"
        )
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-histories", type=int, default=24)
    parser.add_argument("--target-scenes", type=int, default=15)
    parser.add_argument("--minimum-per-stratum", type=int, default=4)
    args = parser.parse_args()
    result = finalize(
        construction_root=args.construction_root.resolve(),
        original_manifest_path=args.original_manifest.resolve(),
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
            "formal_policy_evaluation_authorized"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
