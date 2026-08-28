#!/usr/bin/env python3
"""Build a fail-closed HM3D shared-trace replay plan bound to source nodes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NODE_PATTERN = re.compile(r"^(?P<family>gh|ga)[0-9]{3}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def partition_for_node(node: str) -> str:
    match = NODE_PATTERN.fullmatch(node)
    require(match is not None, f"unsupported replay node: {node}")
    return "h100_tandon" if match.group("family") == "gh" else "a100_tandon"


def source_label(index: int, row: dict[str, Any]) -> str:
    return f"{index:03d}_{row['scene']}_{row['episode']}"


def build_node_affinity_plan(
    *,
    source_population: Path,
    shared_population: Path,
    collection_root: Path,
    lanes: int = 2,
) -> list[dict[str, Any]]:
    """Map every sealed-C evaluation index to its collection compute node.

    Replaying a frozen RGB trace on another node can change Habitat-rendered
    JPEG bytes.  The collection node is known to reproduce the factual-B
    prefix and generated the sealed-C prefix, so it is the only eligible B2
    replay node.  Lane assignment is deterministic and caps live GPU jobs.
    """

    require(lanes > 0, "lanes must be positive")
    source_rows = load_object(source_population).get("accepted")
    shared_rows = load_object(shared_population).get("accepted")
    require(isinstance(source_rows, list) and source_rows,
            "source population is empty")
    require(isinstance(shared_rows, list) and shared_rows,
            "sealed shared-C population is empty")

    plan = []
    seen_source_indices: set[int] = set()
    for evaluation_index, row in enumerate(shared_rows):
        require(isinstance(row, dict), "shared-C population row is not an object")
        require(row.get("population_index") == evaluation_index,
                "shared-C population index is not contiguous")
        source_index = row.get("source_population_index")
        require(isinstance(source_index, int) and not isinstance(source_index, bool),
                "shared-C source population index is invalid")
        require(0 <= source_index < len(source_rows),
                "shared-C source population index is out of bounds")
        require(source_index not in seen_source_indices,
                "shared-C source population index is duplicated")
        seen_source_indices.add(source_index)
        source_row = source_rows[source_index]
        require(isinstance(source_row, dict), "source population row is not an object")
        require(row.get("scene") == source_row.get("scene")
                and row.get("episode") == source_row.get("episode"),
                "shared-C/source population identity changed")
        label = source_label(source_index, source_row)
        compute_path = collection_root / label / "compute_identity.json"
        compute = load_object(compute_path)
        require(compute.get("schema_version") == "cec_compute_identity_v1_20260824",
                f"{label}: compute identity schema changed")
        host = compute.get("host")
        require(isinstance(host, str), f"{label}: compute host is missing")
        partition = partition_for_node(host)
        plan.append({
            "evaluation_index": evaluation_index,
            "source_population_index": source_index,
            "scene": row["scene"],
            "episode": row["episode"],
            "collection_label": label,
            "node": host,
            "partition": partition,
            "lane": evaluation_index % lanes,
        })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-population", type=Path, required=True)
    parser.add_argument("--shared-population", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()
    plan = build_node_affinity_plan(
        source_population=args.source_population,
        shared_population=args.shared_population,
        collection_root=args.collection_root,
        lanes=args.lanes,
    )
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    for row in plan:
        print("\t".join(str(row[key]) for key in (
            "evaluation_index", "source_population_index", "node",
            "partition", "collection_label", "lane")))


if __name__ == "__main__":
    main()
