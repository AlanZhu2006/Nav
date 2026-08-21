#!/usr/bin/env python3
"""Seal all actual-mono HM3D scene fragments before query evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from MemNavData.audit_shared_online_role_pairs import audit as audit_role_pairs
from MemNavData.hm3d_fullmono_mixed_role import (
    bind_parent_manifest,
    expected_parent_source_count,
    require,
)
from MemNavData.shared_online_role_pair_contract import validate_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_json(path: Path) -> dict:
    require(path.is_file(), f"missing receipt {path}")
    receipt = path.with_name(path.name + ".sha256")
    require(receipt.is_file(), f"missing receipt hash {receipt}")
    require(sha256(path) == receipt.read_text().split()[0],
            f"receipt hash changed: {path}")
    return json.loads(path.read_text())


def choose_scene_prefix(
    fragments: list[dict], selection: Optional[dict],
) -> dict:
    """Choose a frozen prefix from pre-query constructibility only."""

    if selection is None:
        prefix = len(fragments)
        target_histories, target_scenes = 18, 8
        schedule = [prefix]
    else:
        initial = int(selection["initial_scene_prefix"])
        block = int(selection["extension_block_scenes"])
        maximum = int(selection["maximum_scene_prefix"])
        require(initial > 0 and block > 0 and initial <= maximum,
                "invalid prefix schedule")
        schedule = list(range(initial, maximum + 1, block))
        require(schedule[-1] == maximum, "prefix schedule misses maximum")
        target_histories = int(selection["target_histories"])
        target_scenes = int(selection["target_scene_clusters"])
        prefix = maximum
        for candidate in schedule:
            histories = sum(
                int(row["retained_histories"])
                for row in fragments if int(row["scene_index"]) < candidate
            )
            scenes = sum(
                int(row["retained_histories"]) > 0
                for row in fragments if int(row["scene_index"]) < candidate
            )
            if histories >= target_histories and scenes >= target_scenes:
                prefix = candidate
                break
    histories = sum(
        int(row["retained_histories"])
        for row in fragments if int(row["scene_index"]) < prefix
    )
    scenes = sum(
        int(row["retained_histories"]) > 0
        for row in fragments if int(row["scene_index"]) < prefix
    )
    return {
        "selected_scene_prefix": prefix,
        "schedule": schedule,
        "target_histories": target_histories,
        "target_scene_clusters": target_scenes,
        "retained_histories": histories,
        "retained_scene_clusters": scenes,
        "target_met": histories >= target_histories and scenes >= target_scenes,
        "query_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    parent, parent_sha = bind_parent_manifest(
        protocol, args.protocol, args.parent_manifest)
    scene_specs = protocol["dataset"]["scenes"]
    collection_root = args.run_root / "goal_a" / "scenes"
    construction_root = args.run_root / "construction" / "scenes"
    fragments = []
    source_count = goal_a_successes = materialized = 0
    candidate_rows = []
    contract = None
    construction_seed = None
    online_manifest_hashes: dict[int, str] = {}

    benchmark_root = args.run_root / "benchmarks"
    require(not benchmark_root.exists(), "benchmark output already exists")
    benchmark_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=benchmark_root.name + ".tmp.", dir=benchmark_root.parent
    ))
    try:
        target = temporary / "natural_direction"
        for index, spec in enumerate(scene_specs):
            scene = str(spec["scene_id"])
            label = f"{index:02d}_{scene}"
            collection = checked_json(
                collection_root / label / "completion.json"
            )
            construction = checked_json(
                construction_root / label / "completion.json"
            )
            require(collection["scene"] == construction["scene"] == scene,
                    f"{scene}: scene receipt mismatch")
            require(collection["all_sources_retained"] is True,
                    f"{scene}: frozen source population changed")
            require(int(collection["metric_depth_sensor_reads"]) == 0,
                    f"{scene}: Goal-A consumed metric depth")
            require(construction["query_policy_outcomes_read"] is False,
                    f"{scene}: construction read query outcomes")
            source_count += int(collection["source_episode_count"])
            goal_a_successes += int(collection["goal_a_successes"])
            materialized += int(construction["materialization"]["materialized"])

            source = construction_root / label / "role_pairs/natural_direction"
            source_manifest_path = source / "manifest.json"
            fragment_rows = []
            fragment_sha = None
            online_sha = construction["materialization"]["manifest_sha256"]
            if source_manifest_path.is_file():
                source_manifest = json.loads(source_manifest_path.read_text())
                fragment_sha = sha256(source_manifest_path)
                if online_sha is not None:
                    online_manifest_hashes[index] = str(online_sha)
                if contract is None:
                    contract = source_manifest["contract"]
                    construction_seed = int(source_manifest["construction_seed"])
                require(source_manifest["contract"] == contract,
                        "scene fragments changed query contract")
                require(int(source_manifest["construction_seed"]) ==
                        construction_seed,
                        "scene fragments changed construction seed")
                fragment_rows = list(source_manifest["episodes"])
                for row in fragment_rows:
                    candidate_rows.append({
                        "scene_index": index, "source_root": source,
                        "row": row,
                    })
            else:
                require(int(construction["retained_natural_histories"]) == 0 and
                        int(collection["source_episode_count"]) == 0,
                        f"{scene}: missing non-empty role-pair fragment")
            fragments.append({
                "scene": scene,
                "scene_index": index,
                "source_episodes": int(collection["source_episode_count"]),
                "goal_a_successes": int(collection["goal_a_successes"]),
                "materialized_histories": int(
                    construction["materialization"]["materialized"]
                ),
                "retained_histories": len(fragment_rows),
                "collection_completion_sha256": sha256(
                    collection_root / label / "completion.json"
                ),
                "construction_completion_sha256": sha256(
                    construction_root / label / "completion.json"
                ),
                "fragment_manifest_sha256": fragment_sha,
            })

        require(source_count == expected_parent_source_count(protocol, parent),
                "Goal-A source population changed")
        selection = choose_scene_prefix(
            fragments, protocol.get("population_selection"))
        selected_prefix = int(selection["selected_scene_prefix"])
        rows = []
        identities: set[tuple[str, str]] = set()
        for item in candidate_rows:
            if int(item["scene_index"]) >= selected_prefix:
                continue
            row = item["row"]
            identity = (str(row["scene"]), str(row["episode"]))
            require(identity not in identities,
                    f"duplicate constructed history {identity}")
            identities.add(identity)
            destination = target / identity[0] / identity[1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                item["source_root"] / identity[0] / identity[1], destination)
            rows.append(row)
        require(bool(rows), "full-mono HM3D mixed-role population is empty")
        require(len(rows) == int(selection["retained_histories"]),
                "selected prefix history count changed")
        for fragment in fragments:
            fragment["selected_for_query"] = (
                int(fragment["scene_index"]) < selected_prefix)
        rows.sort(key=lambda row: (
            int(row["final14_scene_rank"]),
            int(row["final14_source_episode_rank"]),
        ))
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "actual-online full-monocular HM3D natural unsupported "
                "Novel plus standard-support Revisit"
            ),
            "source_online_root": str(collection_root.resolve()),
            "source_online_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(
                    value for index, value in online_manifest_hashes.items()
                    if index < selected_prefix
                )).encode()
            ).hexdigest(),
            "source_online_manifest_sha256_semantics": (
                "sha256 of sorted per-scene materialized manifest SHA256 values"
            ),
            "construction_seed": construction_seed,
            "contract": contract,
            "episodes": rows,
        }
        validate_manifest(manifest)
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (target / "manifest.json.sha256").write_text(
            sha256(manifest_path) + "  manifest.json\n"
        )
        audit = audit_role_pairs(target)
        population = {
            "schema_version": (
                "hm3d_fullmono_mixed_role_population_v2_20260820"
                if protocol.get("population_selection") is not None
                else "hm3d_fullmono_mixed_role_population_v1_20260820"
            ),
            "scope": protocol["scope"],
            "fresh_scene_generalization": bool(
                parent.get("fresh_scene_generalization", False)),
            "parent_manifest_sha256": parent_sha,
            "source_goal_a_episodes": source_count,
            "target_source_goal_a_episodes": int(
                protocol["dataset"].get(
                    "target_source_episode_count", source_count)),
            "goal_a_successes": goal_a_successes,
            "materialized_histories": materialized,
            "retained_histories": len(rows),
            "retained_scenes": len({scene for scene, _episode in identities}),
            "target_histories": int(selection["target_histories"]),
            "target_scenes": int(selection["target_scene_clusters"]),
            "target_met": bool(selection["target_met"]),
            "population_selection": selection,
            "analysis_contract": protocol.get("analysis", {}),
            "policy_outcomes_read": False,
            "runtime_role_visibility": "none",
            "metric_depth_sensor_reads_goal_a": 0,
            "fragments": fragments,
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
    checksum = benchmark_root / "BENCHMARK_FILES.sha256"
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(benchmark_root)}\n"
        for path in files
    ))
    (benchmark_root / "BENCHMARK_FILES.sha256.sha256").write_text(
        sha256(checksum) + "  BENCHMARK_FILES.sha256\n"
    )
    (benchmark_root / "SEALED").write_text(
        sha256(benchmark_root / "natural_direction/manifest.json") + "\n"
    )
    print(json.dumps(population, sort_keys=True))


if __name__ == "__main__":
    main()
