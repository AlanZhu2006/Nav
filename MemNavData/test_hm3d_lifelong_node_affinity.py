import json
from pathlib import Path

import pytest

from MemNavData.hm3d_lifelong_node_affinity import (
    build_node_affinity_plan,
    partition_for_node,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_plan_uses_collection_node_and_two_lanes(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    shared = tmp_path / "shared.json"
    collection = tmp_path / "collection"
    rows = [
        {"scene": "scene0", "episode": "episode_0"},
        {"scene": "scene1", "episode": "episode_1"},
        {"scene": "scene2", "episode": "episode_2"},
    ]
    write_json(source, {"accepted": rows})
    write_json(shared, {"accepted": [
        {"population_index": 0, "source_population_index": 2,
         "scene": "scene2", "episode": "episode_2"},
        {"population_index": 1, "source_population_index": 0,
         "scene": "scene0", "episode": "episode_0"},
    ]})
    for index, node in ((0, "gh005"), (2, "ga003")):
        label = f"{index:03d}_{rows[index]['scene']}_{rows[index]['episode']}"
        write_json(collection / label / "compute_identity.json", {
            "schema_version": "cec_compute_identity_v1_20260824",
            "host": node,
        })
    plan = build_node_affinity_plan(
        source_population=source,
        shared_population=shared,
        collection_root=collection,
    )
    assert [(row["evaluation_index"], row["source_population_index"],
             row["node"], row["partition"], row["lane"]) for row in plan] == [
        (0, 2, "ga003", "a100_tandon", 0),
        (1, 0, "gh005", "h100_tandon", 1),
    ]


def test_unknown_node_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unsupported replay node"):
        partition_for_node("gpu001")


def test_identity_drift_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    shared = tmp_path / "shared.json"
    write_json(source, {"accepted": [{"scene": "s", "episode": "e"}]})
    write_json(shared, {"accepted": [{
        "population_index": 0, "source_population_index": 0,
        "scene": "changed", "episode": "e",
    }]})
    with pytest.raises(RuntimeError, match="identity changed"):
        build_node_affinity_plan(
            source_population=source,
            shared_population=shared,
            collection_root=tmp_path / "collection",
        )
