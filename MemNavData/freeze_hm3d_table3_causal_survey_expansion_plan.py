#!/usr/bin/env python3
"""Freeze an append-only, pre-policy expansion for causal-survey Table III."""

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
BASE_FRAGMENT_SCHEMA = "hm3d_table3_causal_survey_fragment_v1_20260830"
CAPACITY_SUMMARY_SCHEMA = "hm3d_table3_navmesh_capacity_summary_v1_20260830"
CAPACITY_VERIFY_SCHEMA = "hm3d_table3_navmesh_capacity_verification_v1_20260830"


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
            f"invalid receipt sidecar for {path}")
    return digest


def candidate_identity(scene: str, bin_name: str, row: dict[str, Any]) -> str:
    return canonical_sha256({
        "scene": scene,
        "bin": bin_name,
        "query_start": row["query_start"],
        "first_goal": row["first_goal"],
        "second_goal": row["second_goal"],
    })


def scene_round_robin(
    rows_by_scene: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, int, dict[str, Any]]]:
    ordered: list[tuple[str, int, dict[str, Any]]] = []
    depth = max((len(rows) for rows in rows_by_scene.values()), default=0)
    for rank in range(depth):
        for scene in sorted(rows_by_scene):
            if rank < len(rows_by_scene[scene]):
                ordered.append((scene, rank, rows_by_scene[scene][rank]))
    return ordered


def freeze(
    *, selection_protocol_path: Path, out_plan: Path,
    out_construction_protocol: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = json.loads(selection_protocol_path.read_text())
    require(selection.get("schema_version") == SELECTION_SCHEMA,
            "expansion selection protocol changed")
    guards = selection["guards"]
    require(guards["all_125_base_completion_receipts_required"] is True
            and guards["query_policy_evaluation_authorized"] is False
            and guards["threshold_relaxation"] is False
            and guards["partial_population_allowed"] is False
            and guards["fallback_completion_allowed"] is False,
            "expansion guards changed")

    base = selection["base"]
    base_plan_path = Path(base["candidate_plan"])
    base_protocol_path = Path(base["construction_protocol"])
    base_run_root = Path(base["construction_run_root"])
    require(sha256_file(base_plan_path) == base["candidate_plan_sha256"],
            "base candidate plan changed")
    require(sha256_file(base_protocol_path) == base["construction_protocol_sha256"],
            "base construction protocol changed")
    base_plan = json.loads(base_plan_path.read_text())
    base_protocol = json.loads(base_protocol_path.read_text())
    require(len(base_plan["episodes"]) == int(base["candidate_count"])
            == int(selection["selection"]["global_history_index_offset"]),
            "base candidate count changed")
    require(base_protocol.get("schema_version") == CONSTRUCTION_PROTOCOL_SCHEMA,
            "base causal-survey protocol schema changed")

    base_fragments: list[dict[str, Any]] = []
    base_fragment_ledger: list[dict[str, Any]] = []
    base_plan_sha = sha256_file(base_plan_path)
    base_protocol_sha = sha256_file(base_protocol_path)
    for plan_index, candidate in enumerate(base_plan["episodes"]):
        receipt = (base_run_root / "construction_fragments"
                   / f"{plan_index:03d}" / "completion.json")
        receipt_sha = verify_sidecar(receipt)
        fragment = json.loads(receipt.read_text())
        require(fragment.get("schema_version") == BASE_FRAGMENT_SCHEMA
                and int(fragment["history_index"]) == plan_index
                and fragment["candidate_identity_sha256"]
                == candidate["candidate_identity_sha256"]
                and fragment["source_candidate_plan_sha256"] == base_plan_sha
                and fragment["protocol_sha256"] == base_protocol_sha
                and fragment.get("query_policy_outcomes_read") is False,
                f"base construction fragment {plan_index} changed")
        base_fragments.append(fragment)
        base_fragment_ledger.append({
            "plan_index": plan_index, "path": str(receipt.resolve()),
            "sha256": receipt_sha,
        })

    bin_order = list(selection["selection"]["bin_order"])
    gate = base_protocol["population_gate"]
    required_histories = int(gate["minimum_histories_per_bin"])
    required_scenes = int(gate["minimum_scene_clusters_per_bin"])
    base_diagnostics: dict[str, dict[str, Any]] = {}
    deficient_bins: list[str] = []
    for name in bin_order:
        constructed = [row for row in base_fragments
                       if row["bin_name"] == name and row["constructed"]]
        scene_count = len({row["scene"] for row in constructed})
        deficient = len(constructed) < required_histories or scene_count < required_scenes
        if deficient:
            deficient_bins.append(name)
        base_diagnostics[name] = {
            "frozen_candidates": sum(row["bin_name"] == name
                                     for row in base_fragments),
            "constructible_histories": len(constructed),
            "constructible_scene_clusters": scene_count,
            "deficient_under_original_gate": deficient,
        }

    capacity = selection["capacity_source"]
    capacity_root = Path(capacity["run_root"])
    summary_path = capacity_root / capacity["summary"]
    verification_path = capacity_root / capacity["independent_verification"]
    require(sha256_file(summary_path) == capacity["summary_sha256"],
            "replenishment capacity summary changed")
    require(sha256_file(verification_path)
            == capacity["independent_verification_sha256"],
            "replenishment capacity verifier changed")
    summary = json.loads(summary_path.read_text())
    verification = json.loads(verification_path.read_text())
    require(summary.get("schema_version") == CAPACITY_SUMMARY_SCHEMA
            and verification.get("schema_version") == CAPACITY_VERIFY_SCHEMA
            and verification.get("verified") is True
            and verification.get("all_geometry_capacity_gates_passed") is True,
            "replenishment capacity is not independently verified")
    for payload in (summary, verification):
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False,
                "capacity source crossed the policy-outcome boundary")

    parent = selection["parent"]
    parent_path = Path(parent["manifest"])
    require(sha256_file(parent_path) == parent["manifest_sha256"],
            "parent asset manifest changed")
    parent_payload = json.loads(parent_path.read_text())
    scenes = list(parent_payload["scenes"])
    require(len(scenes) == int(parent["expected_scenes"]),
            "parent scene count changed")
    scene_index = {scene: index for index, scene in enumerate(scenes)}

    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: defaultdict(list) for name in bin_order
    }
    capacity_fragment_ledger: list[dict[str, Any]] = []
    for ledger in summary["scene_fragments"]:
        fragment_path = Path(ledger["path"])
        require(sha256_file(fragment_path) == ledger["sha256"],
                "capacity scene fragment changed")
        fragment = json.loads(fragment_path.read_text())
        scene = str(fragment["scene"])
        require(scene == ledger["scene"] and scene in scene_index,
                "capacity scene escaped the frozen parent")
        require(fragment.get("query_policy_outcomes_read") is False
                and fragment.get("navigation_outcomes_read") is False,
                "capacity fragment read a policy outcome")
        capacity_fragment_ledger.append({
            "scene": scene, "path": str(fragment_path.resolve()),
            "sha256": ledger["sha256"],
        })
        for name in bin_order:
            candidates[name][scene].extend(fragment["candidate_triads"][name])

    used = {str(row["candidate_identity_sha256"])
            for row in base_plan["episodes"]}
    require(len(used) == len(base_plan["episodes"]),
            "base candidate identities are not unique")
    expansion_rows: list[tuple[str, str, int, dict[str, Any], str]] = []
    selection_diagnostics: dict[str, dict[str, Any]] = {}
    for name in bin_order:
        available = scene_round_robin(candidates[name])
        fresh: list[tuple[str, int, dict[str, Any], str]] = []
        seen_capacity: set[str] = set()
        for scene, rank, geometry in available:
            identity = candidate_identity(scene, name, geometry)
            require(identity not in seen_capacity,
                    "duplicate identity inside replenishment capacity")
            seen_capacity.add(identity)
            if identity not in used:
                fresh.append((scene, rank, geometry, identity))
        selected = fresh if name in deficient_bins else []
        for scene, rank, geometry, identity in selected:
            require(identity not in used, "expansion candidate duplicated base")
            used.add(identity)
            expansion_rows.append((name, scene, rank, geometry, identity))
        selection_diagnostics[name] = {
            **base_diagnostics[name],
            "verified_capacity_candidates": len(available),
            "unused_capacity_candidates": len(fresh),
            "unused_capacity_scene_clusters": len({row[0] for row in fresh}),
            "expansion_candidates": len(selected),
            "expansion_scene_clusters": len({row[0] for row in selected}),
        }

    episodes: list[dict[str, Any]] = []
    within_bin: dict[str, int] = defaultdict(int)
    offset = int(selection["selection"]["global_history_index_offset"])
    for plan_index, (name, scene, rank, geometry, identity) in enumerate(expansion_rows):
        bin_index = bin_order.index(name)
        index_in_bin = within_bin[name]
        within_bin[name] += 1
        episodes.append({
            "plan_index": plan_index,
            "history_index": offset + plan_index,
            "bin_index": bin_index,
            "bin_name": name,
            "within_bin_index": index_in_bin,
            "scene": scene,
            "scene_index": scene_index[scene],
            "episode": f"table3_exp_b{bin_index}_{index_in_bin:03d}",
            "candidate_identity_sha256": identity,
            "capacity_candidate_rank": rank,
            "capacity_geometry": geometry,
            "asset": parent_payload["assets"][scene],
            "base_candidate_replaced": False,
            "query_policy_outcomes_read": False,
            "navigation_policy_outcomes_read": False,
        })

    plan = {
        "schema_version": PLAN_SCHEMA,
        "scope": selection["scope"],
        "selection_protocol": str(selection_protocol_path.resolve()),
        "selection_protocol_sha256": sha256_file(selection_protocol_path),
        "base_candidate_plan_sha256": base_plan_sha,
        "base_construction_protocol_sha256": base_protocol_sha,
        "base_completion_fragments": base_fragment_ledger,
        "base_completion_ledger_sha256": canonical_sha256(base_fragment_ledger),
        "capacity_summary_sha256": sha256_file(summary_path),
        "capacity_verification_sha256": sha256_file(verification_path),
        "capacity_fragments": capacity_fragment_ledger,
        "capacity_fragment_ledger_sha256": canonical_sha256(capacity_fragment_ledger),
        "parent_manifest": str(parent_path.resolve()),
        "parent_manifest_sha256": sha256_file(parent_path),
        "base_diagnostics": base_diagnostics,
        "deficient_bins": deficient_bins,
        "selection_diagnostics": selection_diagnostics,
        "candidate_count": len(episodes),
        "scene_clusters": len({row["scene"] for row in episodes}),
        "episodes": episodes,
        "base_candidates_deleted_or_replaced": False,
        "query_policy_outcomes_read": False,
        "navigation_policy_outcomes_read": False,
        "query_policy_evaluation_authorized": False,
    }
    atomic_json(out_plan, plan)

    construction_protocol = {
        "schema_version": CONSTRUCTION_PROTOCOL_SCHEMA,
        "frozen_at": selection["frozen_at"],
        "scope": selection["scope"],
        "amendment": {
            "selection_protocol_sha256": sha256_file(selection_protocol_path),
            "base_candidate_plan_sha256": base_plan_sha,
            "base_completion_ledger_sha256": plan["base_completion_ledger_sha256"],
            "append_only": True,
            "base_candidates_deleted_or_replaced": False,
            "query_policy_outcomes_read": False,
        },
        "source_candidate_plan": {
            "path": str(out_plan.resolve()),
            "sha256": sha256_file(out_plan),
            "candidate_count": len(episodes),
            "selection": "all unused verified-capacity candidates in every bin deficient after the complete base construction census",
        },
        "history": base_protocol["history"],
        "query_construction": base_protocol["query_construction"],
        "length_definition": base_protocol["length_definition"],
        "runtime": base_protocol["runtime"],
        "population_gate": base_protocol["population_gate"],
        "guards": {
            **base_protocol["guards"],
            "append_only_expansion": True,
            "base_candidates_deleted_or_replaced": False,
            "query_policy_outcomes_read_before_population_seal": False,
            "partial_results_prohibited": True,
            "fallback_completion_allowed": False,
        },
    }
    atomic_json(out_construction_protocol, construction_protocol)
    return plan, construction_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-protocol", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--out-construction-protocol", type=Path, required=True)
    args = parser.parse_args()
    plan, construction = freeze(
        selection_protocol_path=args.selection_protocol.resolve(),
        out_plan=args.out_plan.resolve(),
        out_construction_protocol=args.out_construction_protocol.resolve(),
    )
    print(json.dumps({
        "candidate_count": plan["candidate_count"],
        "deficient_bins": plan["deficient_bins"],
        "scene_clusters": plan["scene_clusters"],
        "construction_protocol_sha256": sha256_file(
            args.out_construction_protocol.resolve()),
        "query_policy_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
