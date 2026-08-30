#!/usr/bin/env python3
"""Freeze a result-blind factual-history plan from the verified capacity graph."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PROTOCOL_SCHEMA = "hm3d_table3_actual_mono_protocol_v1_20260830"
PLAN_SCHEMA = "hm3d_table3_actual_mono_candidate_plan_v1_20260830"
CAPACITY_SUMMARY_SCHEMA = "hm3d_table3_navmesh_capacity_summary_v1_20260830"
CAPACITY_VERIFY_SCHEMA = "hm3d_table3_navmesh_capacity_verification_v1_20260830"
BIN_ORDER = ("0_to_20_m", "20_to_30_m", "30_to_50_m")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n"
    )


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(payload.get("schema_version") == PROTOCOL_SCHEMA,
            "Table-3 actual-mono protocol schema changed")
    require(payload["guards"]["query_policy_outcomes_read_before_population_seal"] is False,
            "protocol permits query-outcome selection")
    require(payload["runtime"]["arms"] == ["mono_native", "mono_cec"],
            "Table-3 arms changed")
    require([row["name"] for row in payload["length_definition"]["bins_m"]]
            == list(BIN_ORDER), "Table-3 bins changed")
    return payload


def round_robin_prefix(
    rows_by_scene: dict[str, list[dict[str, Any]]], count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    maximum = max((len(rows) for rows in rows_by_scene.values()), default=0)
    for rank in range(maximum):
        for scene in sorted(rows_by_scene):
            rows = rows_by_scene[scene]
            if rank < len(rows):
                selected.append(dict(rows[rank], scene=scene,
                                     capacity_candidate_rank=rank))
                if len(selected) == count:
                    return selected
    require(len(selected) >= count,
            f"capacity graph has only {len(selected)}/{count} candidates")
    return selected[:count]


def freeze(
    *, protocol_path: Path, capacity_root: Path,
    parent_manifest: Path, out: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    source = protocol["capacity_source"]
    require(capacity_root.resolve() == Path(source["run_root"]).resolve(),
            "capacity root changed")
    summary_path = capacity_root / source["summary"]
    verify_path = capacity_root / source["independent_verification"]
    require(sha256_file(summary_path) == source["summary_sha256"],
            "capacity summary changed")
    require(sha256_file(verify_path) == source["independent_verification_sha256"],
            "capacity verification changed")
    summary = json.loads(summary_path.read_text())
    verification = json.loads(verify_path.read_text())
    require(summary.get("schema_version") == CAPACITY_SUMMARY_SCHEMA,
            "capacity summary schema changed")
    require(verification.get("schema_version") == CAPACITY_VERIFY_SCHEMA
            and verification.get("verified") is True
            and verification.get("all_geometry_capacity_gates_passed") is True,
            "capacity gate is not independently verified")
    for payload in (summary, verification):
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False,
                "capacity stage read navigation outcomes")

    parent_spec = protocol["parent"]
    require(sha256_file(parent_manifest) == parent_spec["manifest_sha256"],
            "parent asset manifest changed")
    parent = json.loads(parent_manifest.read_text())
    scenes = list(parent["scenes"])
    require(len(scenes) == int(parent_spec["expected_scenes"]),
            "parent scene count changed")
    scene_index = {scene: index for index, scene in enumerate(scenes)}

    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: defaultdict(list) for name in BIN_ORDER
    }
    fragment_hashes = []
    for ledger in summary["scene_fragments"]:
        path = Path(ledger["path"])
        require(path.is_file() and sha256_file(path) == ledger["sha256"],
                "capacity scene fragment changed")
        fragment = json.loads(path.read_text())
        scene = str(fragment["scene"])
        require(scene in scene_index and scene == ledger["scene"],
                "capacity scene escaped parent assets")
        require(fragment.get("query_policy_outcomes_read") is False
                and fragment.get("navigation_outcomes_read") is False,
                "capacity scene fragment read an outcome")
        fragment_hashes.append({"scene": scene, "sha256": ledger["sha256"]})
        for name in BIN_ORDER:
            candidates[name][scene].extend(fragment["candidate_triads"][name])

    requested = protocol["source_candidate_prefix"]["counts"]
    episodes: list[dict[str, Any]] = []
    selected_diagnostics = {}
    seen_identities: set[str] = set()
    for bin_index, name in enumerate(BIN_ORDER):
        chosen = round_robin_prefix(candidates[name], int(requested[name]))
        selected_diagnostics[name] = {
            "selected_candidates": len(chosen),
            "selected_scene_clusters": len({row["scene"] for row in chosen}),
            "available_candidates": sum(len(rows) for rows in candidates[name].values()),
            "available_scene_clusters": len(candidates[name]),
        }
        for within_bin_index, candidate in enumerate(chosen):
            scene = str(candidate.pop("scene"))
            identity_payload = {
                "scene": scene,
                "bin": name,
                "query_start": candidate["query_start"],
                "first_goal": candidate["first_goal"],
                "second_goal": candidate["second_goal"],
            }
            identity = hashlib.sha256(json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            require(identity not in seen_identities, "duplicate candidate identity")
            seen_identities.add(identity)
            history_index = len(episodes)
            episodes.append({
                "history_index": history_index,
                "bin_index": bin_index,
                "bin_name": name,
                "within_bin_index": within_bin_index,
                "scene": scene,
                "scene_index": scene_index[scene],
                "episode": f"table3_b{bin_index}_{within_bin_index:03d}",
                "candidate_identity_sha256": identity,
                "capacity_candidate_rank": int(candidate.pop(
                    "capacity_candidate_rank")),
                "capacity_geometry": candidate,
                "asset": parent["assets"][scene],
                "factual_A_outcome_read": False,
                "query_policy_outcomes_read": False,
            })

    result = {
        "schema_version": PLAN_SCHEMA,
        "scope": protocol["scope"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "capacity_summary": str(summary_path.resolve()),
        "capacity_summary_sha256": sha256_file(summary_path),
        "capacity_verification": str(verify_path.resolve()),
        "capacity_verification_sha256": sha256_file(verify_path),
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": sha256_file(parent_manifest),
        "candidate_count": len(episodes),
        "scene_clusters": len({row["scene"] for row in episodes}),
        "selection_diagnostics": selected_diagnostics,
        "capacity_fragment_hashes": fragment_hashes,
        "episodes": episodes,
        "factual_A_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read_for_selection": False,
        "query_policy_evaluation_authorized": False,
    }
    atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        protocol_path=args.protocol.resolve(),
        capacity_root=args.capacity_root.resolve(),
        parent_manifest=args.parent_manifest.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "scene_clusters": result["scene_clusters"],
        "selection_diagnostics": result["selection_diagnostics"],
        "query_policy_outcomes_read": result["query_policy_outcomes_read"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
