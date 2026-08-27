#!/usr/bin/env python3
"""Merge independently constructed actual-online NNR episodes and seal them.

The input population is the immutable native A/B-success trace manifest.  A
builder part may classify one member as structurally ineligible, but integrity
errors abort upstream and cannot silently shrink the denominator here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from deterministic_eval_protocol import file_sha256


FINAL_SCHEMA = "shared_online_novel_revisit_sealed_v1_20260814"
BUILDER_SCHEMA = "shared_online_novel_revisit_v1_20260813"
TRACE_SCHEMA = "native_shared_ab_extraction_v1_20260813"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--expected-trace-manifest-sha", required=True)
    parser.add_argument("--expected-source-population", type=int, default=22)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), f"output already exists: {args.out}")
    trace_manifest_path = args.trace_root / "manifest.json"
    require(trace_manifest_path.is_file(), "trace manifest is missing")
    require(
        file_sha256(trace_manifest_path) == args.expected_trace_manifest_sha,
        "trace manifest hash changed",
    )
    trace_manifest = read_object(trace_manifest_path)
    require(
        trace_manifest.get("schema_version") == TRACE_SCHEMA,
        "wrong trace manifest schema",
    )
    require(
        trace_manifest.get("replay_is_observation_only") is True
        and trace_manifest.get("source_memory_observer_present") is False,
        "native observation-only replay disclosure changed",
    )
    source_rows = list(trace_manifest.get("accepted") or [])
    require(
        len(source_rows) == int(args.expected_source_population),
        "native A/B-success source population changed",
    )
    source_keys = [(str(row["scene"]), str(row["episode"])) for row in source_rows]
    require(len(source_keys) == len(set(source_keys)), "duplicate source episode")

    part_records: dict[tuple[str, str], dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(args.parts_root.glob("*/*/manifest.json")):
        manifest = read_object(manifest_path)
        require(manifest.get("schema_version") == BUILDER_SCHEMA, "wrong part schema")
        require(
            manifest.get("selected_before_c_navigation") is True,
            "part was not selected before C navigation",
        )
        scene = str(manifest["scene"])
        accepted = list(manifest.get("accepted") or [])
        rejected = list(manifest.get("rejected") or [])
        require(len(accepted) + len(rejected) == 1, "part must classify one episode")
        contract = manifest.get("contract")
        require(isinstance(contract, dict), "part contract is missing")
        contracts.setdefault(scene, contract)
        require(contracts[scene] == contract, "within-scene construction contract changed")
        if accepted:
            row = accepted[0]
            episode = str(row["episode"])
            episode_dir = manifest_path.parent / episode
            benchmark_path = episode_dir / "benchmark.json"
            require(benchmark_path.is_file(), "accepted benchmark is missing")
            require(
                file_sha256(benchmark_path) == row["benchmark_sha256"],
                "accepted benchmark hash changed",
            )
            benchmark = read_object(benchmark_path)
            require(benchmark.get("construction_uses_c_navigation_outcomes") is False,
                    "benchmark construction read C outcomes")
            record = {
                "status": "accepted",
                "scene": scene,
                "episode": episode,
                "benchmark_sha256": row["benchmark_sha256"],
                "source_dir": episode_dir,
                "source_scene_asset": benchmark["source_scene_asset"],
                "source_scene_asset_sha256": benchmark[
                    "source_scene_asset_sha256"
                ],
            }
        else:
            row = rejected[0]
            episode = str(row["episode"])
            record = {
                "status": "rejected",
                "scene": scene,
                "episode": episode,
                "reason": str(row["reason"]),
            }
        key = (scene, episode)
        require(key not in part_records, f"duplicate builder part: {key}")
        part_records[key] = record

    require(set(part_records) == set(source_keys), "builder parts/source population differ")
    temporary = Path(tempfile.mkdtemp(prefix=args.out.name + ".tmp.", dir=args.out.parent))
    try:
        accepted_rows = []
        rejected_rows = []
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_index, key in enumerate(source_keys):
            record = part_records[key]
            public = {k: v for k, v in record.items() if k != "source_dir"}
            public["source_population_index"] = source_index
            if record["status"] == "accepted":
                selection_index = len(accepted_rows)
                destination = temporary / record["scene"] / record["episode"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(record["source_dir"], destination)
                public["selection_index"] = selection_index
                accepted_rows.append(public)
            else:
                rejected_rows.append(public)
            by_scene[record["scene"]].append(public)

        require(bool(accepted_rows), "no constructible actual-online episode")
        for scene, records in sorted(by_scene.items()):
            scene_root = temporary / scene
            scene_root.mkdir(exist_ok=True)
            scene_manifest = {
                "schema_version": BUILDER_SCHEMA,
                "scene": scene,
                "purpose": (
                    "strict-v4 A/Novel-B with controlled Revisit-C derived "
                    "only from factual online-A"
                ),
                "contract": contracts[scene],
                "selected_before_c_navigation": True,
                "accepted": [
                    {
                        "episode": row["episode"],
                        "benchmark_sha256": row["benchmark_sha256"],
                    }
                    for row in records if row["status"] == "accepted"
                ],
                "rejected": [
                    {"episode": row["episode"], "reason": row["reason"]}
                    for row in records if row["status"] == "rejected"
                ],
            }
            (scene_root / "manifest.json").write_text(
                json.dumps(scene_manifest, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )

        manifest = {
            "schema_version": FINAL_SCHEMA,
            "source_population": "strict-v4 native A-and-B successes",
            "conditional_endpoint": "SR_C_given_frozen_online_A_and_online_B",
            "source_population_size": len(source_rows),
            "constructible_population_size": len(accepted_rows),
            "rejected_population_size": len(rejected_rows),
            "scenes": len({row["scene"] for row in accepted_rows}),
            "trace_manifest_sha256": args.expected_trace_manifest_sha,
            "construction_uses_c_navigation_outcomes": False,
            "accepted": accepted_rows,
            "rejected": rejected_rows,
        }
        manifest["population_fingerprint"] = canonical_sha({
            "accepted": accepted_rows,
            "rejected": rejected_rows,
            "trace_manifest_sha256": args.expected_trace_manifest_sha,
        })
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json.sha256").write_text(
            f"{file_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        (temporary / "SEALED").write_text(
            f"{FINAL_SCHEMA}\n{file_sha256(manifest_path)}\n", encoding="utf-8"
        )
        temporary.replace(args.out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "output": str(args.out),
        "source_population": len(source_rows),
        "constructible_population": len(accepted_rows),
        "rejected_population": len(rejected_rows),
        "manifest_sha256": file_sha256(args.out / "manifest.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
