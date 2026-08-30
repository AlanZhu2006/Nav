#!/usr/bin/env python3
"""Independently reproduce the append-only causal-survey expansion plan."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SELECTION_SCHEMA = "hm3d_table3_causal_survey_expansion_selection_v1_20260831"
PLAN_SCHEMA = "hm3d_table3_causal_survey_expansion_plan_v1_20260831"
CONSTRUCTION_PROTOCOL_SCHEMA = "hm3d_table3_causal_survey_protocol_v1_20260830"
FRAGMENT_SCHEMA = "hm3d_table3_causal_survey_fragment_v1_20260830"
CAPACITY_SUMMARY_SCHEMA = "hm3d_table3_navmesh_capacity_summary_v1_20260830"
CAPACITY_VERIFY_SCHEMA = "hm3d_table3_navmesh_capacity_verification_v1_20260830"
VERIFY_SCHEMA = "hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


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


def verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file() and sidecar.read_text().split() == [digest, path.name],
            f"invalid sidecar for {path}")
    return digest


def identity(scene: str, bin_name: str, row: dict[str, Any]) -> str:
    payload = {
        "scene": scene,
        "bin": bin_name,
        "query_start": row["query_start"],
        "first_goal": row["first_goal"],
        "second_goal": row["second_goal"],
    }
    return canonical_sha256(payload)


def independently_order(
    rows_by_scene: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, int, dict[str, Any]]]:
    result: list[tuple[str, int, dict[str, Any]]] = []
    depth = max((len(rows) for rows in rows_by_scene.values()), default=0)
    for rank in range(depth):
        for scene in sorted(rows_by_scene):
            if rank < len(rows_by_scene[scene]):
                result.append((scene, rank, rows_by_scene[scene][rank]))
    return result


def verify(
    *, selection_protocol_path: Path, plan_path: Path,
    construction_protocol_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    construction = json.loads(construction_protocol_path.read_text())
    require(selection.get("schema_version") == SELECTION_SCHEMA,
            "selection protocol schema changed")
    require(plan.get("schema_version") == PLAN_SCHEMA
            and plan["selection_protocol_sha256"]
            == sha256_file(selection_protocol_path),
            "expansion plan binding changed")
    require(plan.get("base_candidates_deleted_or_replaced") is False
            and plan.get("query_policy_outcomes_read") is False
            and plan.get("navigation_policy_outcomes_read") is False
            and plan.get("query_policy_evaluation_authorized") is False,
            "expansion plan crossed the policy boundary")

    base = selection["base"]
    base_plan_path = Path(base["candidate_plan"])
    base_protocol_path = Path(base["construction_protocol"])
    base_run_root = Path(base["construction_run_root"])
    require(sha256_file(base_plan_path) == base["candidate_plan_sha256"]
            == plan["base_candidate_plan_sha256"],
            "base plan hash changed")
    require(sha256_file(base_protocol_path)
            == base["construction_protocol_sha256"]
            == plan["base_construction_protocol_sha256"],
            "base protocol hash changed")
    base_plan = json.loads(base_plan_path.read_text())
    base_protocol = json.loads(base_protocol_path.read_text())
    require(len(base_plan["episodes"]) == int(base["candidate_count"]),
            "base plan count changed")

    base_plan_sha = sha256_file(base_plan_path)
    base_protocol_sha = sha256_file(base_protocol_path)
    base_fragments: list[dict[str, Any]] = []
    base_ledger: list[dict[str, Any]] = []
    for plan_index, candidate in enumerate(base_plan["episodes"]):
        path = (base_run_root / "construction_fragments"
                / f"{plan_index:03d}" / "completion.json")
        digest = verify_sidecar(path)
        row = json.loads(path.read_text())
        require(row.get("schema_version") == FRAGMENT_SCHEMA
                and int(row["history_index"]) == plan_index
                and row["candidate_identity_sha256"]
                == candidate["candidate_identity_sha256"]
                and row["source_candidate_plan_sha256"] == base_plan_sha
                and row["protocol_sha256"] == base_protocol_sha
                and row.get("query_policy_outcomes_read") is False,
                f"base fragment {plan_index} changed")
        base_fragments.append(row)
        base_ledger.append({
            "plan_index": plan_index, "path": str(path.resolve()),
            "sha256": digest,
        })
    require(base_ledger == plan["base_completion_fragments"]
            and canonical_sha256(base_ledger)
            == plan["base_completion_ledger_sha256"],
            "base completion ledger changed")

    bin_order = list(selection["selection"]["bin_order"])
    gate = base_protocol["population_gate"]
    minimum_histories = int(gate["minimum_histories_per_bin"])
    minimum_scenes = int(gate["minimum_scene_clusters_per_bin"])
    base_diagnostics: dict[str, dict[str, Any]] = {}
    deficient_bins: list[str] = []
    for name in bin_order:
        eligible = [row for row in base_fragments
                    if row["bin_name"] == name and row["constructed"]]
        scene_count = len({row["scene"] for row in eligible})
        deficient = len(eligible) < minimum_histories or scene_count < minimum_scenes
        if deficient:
            deficient_bins.append(name)
        base_diagnostics[name] = {
            "frozen_candidates": sum(row["bin_name"] == name
                                     for row in base_fragments),
            "constructible_histories": len(eligible),
            "constructible_scene_clusters": scene_count,
            "deficient_under_original_gate": deficient,
        }
    require(base_diagnostics == plan["base_diagnostics"]
            and deficient_bins == plan["deficient_bins"],
            "base deficiency decision changed")

    capacity = selection["capacity_source"]
    capacity_root = Path(capacity["run_root"])
    summary_path = capacity_root / capacity["summary"]
    verification_path = capacity_root / capacity["independent_verification"]
    require(sha256_file(summary_path) == capacity["summary_sha256"]
            == plan["capacity_summary_sha256"],
            "capacity summary binding changed")
    require(sha256_file(verification_path)
            == capacity["independent_verification_sha256"]
            == plan["capacity_verification_sha256"],
            "capacity verification binding changed")
    summary = json.loads(summary_path.read_text())
    capacity_verify = json.loads(verification_path.read_text())
    require(summary.get("schema_version") == CAPACITY_SUMMARY_SCHEMA
            and capacity_verify.get("schema_version") == CAPACITY_VERIFY_SCHEMA
            and capacity_verify.get("verified") is True
            and capacity_verify.get("all_geometry_capacity_gates_passed") is True,
            "capacity gate no longer passes")

    parent = selection["parent"]
    parent_path = Path(parent["manifest"])
    require(sha256_file(parent_path) == parent["manifest_sha256"]
            == plan["parent_manifest_sha256"],
            "parent manifest binding changed")
    parent_payload = json.loads(parent_path.read_text())
    scenes = list(parent_payload["scenes"])
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    require(len(scenes) == int(parent["expected_scenes"]),
            "parent scene count changed")

    candidates = {name: defaultdict(list) for name in bin_order}
    capacity_ledger: list[dict[str, Any]] = []
    for ledger in summary["scene_fragments"]:
        path = Path(ledger["path"])
        require(sha256_file(path) == ledger["sha256"],
                "capacity fragment changed")
        fragment = json.loads(path.read_text())
        scene = str(fragment["scene"])
        require(scene == ledger["scene"] and scene in scene_index
                and fragment.get("query_policy_outcomes_read") is False
                and fragment.get("navigation_outcomes_read") is False,
                "capacity fragment provenance changed")
        capacity_ledger.append({
            "scene": scene, "path": str(path.resolve()),
            "sha256": ledger["sha256"],
        })
        for name in bin_order:
            candidates[name][scene].extend(fragment["candidate_triads"][name])
    require(capacity_ledger == plan["capacity_fragments"]
            and canonical_sha256(capacity_ledger)
            == plan["capacity_fragment_ledger_sha256"],
            "capacity fragment ledger changed")

    used = {str(row["candidate_identity_sha256"])
            for row in base_plan["episodes"]}
    expected_rows: list[tuple[str, str, int, dict[str, Any], str]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for name in bin_order:
        ordered = independently_order(candidates[name])
        fresh: list[tuple[str, int, dict[str, Any], str]] = []
        capacity_seen: set[str] = set()
        for scene, rank, geometry in ordered:
            digest = identity(scene, name, geometry)
            require(digest not in capacity_seen,
                    "capacity identity duplicated")
            capacity_seen.add(digest)
            if digest not in used:
                fresh.append((scene, rank, geometry, digest))
        selected = fresh if name in deficient_bins else []
        for row in selected:
            require(row[3] not in used, "expected expansion duplicated base")
            used.add(row[3])
            expected_rows.append((name, *row))
        diagnostics[name] = {
            **base_diagnostics[name],
            "verified_capacity_candidates": len(ordered),
            "unused_capacity_candidates": len(fresh),
            "unused_capacity_scene_clusters": len({row[0] for row in fresh}),
            "expansion_candidates": len(selected),
            "expansion_scene_clusters": len({row[0] for row in selected}),
        }
    require(diagnostics == plan["selection_diagnostics"],
            "expansion selection diagnostics changed")
    require(len(expected_rows) == int(plan["candidate_count"])
            == len(plan["episodes"]),
            "expansion candidate count changed")

    within_bin: dict[str, int] = defaultdict(int)
    offset = int(selection["selection"]["global_history_index_offset"])
    for plan_index, (expected, stored) in enumerate(
        zip(expected_rows, plan["episodes"])
    ):
        name, scene, rank, geometry, digest = expected
        index_in_bin = within_bin[name]
        within_bin[name] += 1
        require(int(stored["plan_index"]) == plan_index
                and int(stored["history_index"]) == offset + plan_index
                and stored["bin_name"] == name
                and int(stored["within_bin_index"]) == index_in_bin
                and stored["scene"] == scene
                and int(stored["scene_index"]) == scene_index[scene]
                and int(stored["capacity_candidate_rank"]) == rank
                and stored["candidate_identity_sha256"] == digest
                and stored["capacity_geometry"] == geometry
                and stored["asset"] == parent_payload["assets"][scene]
                and stored.get("base_candidate_replaced") is False
                and stored.get("query_policy_outcomes_read") is False
                and stored.get("navigation_policy_outcomes_read") is False,
                f"expansion episode {plan_index} changed")

    require(construction.get("schema_version") == CONSTRUCTION_PROTOCOL_SCHEMA
            and construction["source_candidate_plan"]["path"]
            == str(plan_path.resolve())
            and construction["source_candidate_plan"]["sha256"]
            == sha256_file(plan_path)
            and int(construction["source_candidate_plan"]["candidate_count"])
            == len(expected_rows),
            "construction protocol/plan binding changed")
    for field in ("history", "query_construction", "length_definition",
                  "runtime", "population_gate"):
        require(construction[field] == base_protocol[field],
                f"construction protocol changed {field}")
    require(construction["amendment"]["append_only"] is True
            and construction["amendment"]["base_candidates_deleted_or_replaced"] is False
            and construction["amendment"]["query_policy_outcomes_read"] is False
            and construction["guards"]["fallback_completion_allowed"] is False,
            "construction amendment guards changed")

    return {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "selection_protocol_sha256": sha256_file(selection_protocol_path),
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "construction_protocol": str(construction_protocol_path.resolve()),
        "construction_protocol_sha256": sha256_file(construction_protocol_path),
        "base_completion_ledger_sha256": canonical_sha256(base_ledger),
        "capacity_fragment_ledger_sha256": canonical_sha256(capacity_ledger),
        "deficient_bins": deficient_bins,
        "candidate_count": len(expected_rows),
        "scene_clusters": len({row["scene"] for row in plan["episodes"]}),
        "selection_diagnostics": diagnostics,
        "base_candidates_deleted_or_replaced": False,
        "query_policy_outcomes_read": False,
        "navigation_policy_outcomes_read": False,
        "query_policy_evaluation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--construction-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        selection_protocol_path=args.selection_protocol.resolve(),
        plan_path=args.plan.resolve(),
        construction_protocol_path=args.construction_protocol.resolve(),
    )
    atomic_json(args.out.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
