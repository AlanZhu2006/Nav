#!/usr/bin/env python3
"""Seal the powered Table-III population before any query policy executes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil

from hm3d_table3_length_contract import SCHEMA_VERSION, validate_manifest


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file()
            and sidecar.read_text().split() == [digest, path.name],
            f"invalid fragment receipt: {path}")
    return digest


def select_powered(rows: list[dict], *, histories: int, scenes: int,
                   maximum_per_scene: int) -> list[dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    require(len(by_scene) >= scenes, "insufficient constructible scene clusters")
    selected, counts = [], Counter()
    # Scene breadth is established before any scene receives a second history.
    for row in rows:
        if counts[row["scene"]] == 0:
            selected.append(row)
            counts[row["scene"]] += 1
            if len(counts) >= scenes:
                break
    for row in rows:
        if row in selected or counts[row["scene"]] >= maximum_per_scene:
            continue
        selected.append(row)
        counts[row["scene"]] += 1
        if len(selected) == histories:
            break
    require(len(selected) == histories, "insufficient constructible histories")
    require(len({row["scene"] for row in selected}) >= scenes,
            "scene breadth gate failed")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "sealed Table-III population exists")
    plan = json.loads(args.candidate_plan.read_text())
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version")
            == "hm3d_table3_actual_mono_protocol_v1_20260830",
            "Table-III population protocol changed")
    require(sha256(args.protocol) == plan.get("protocol_sha256"),
            "candidate plan/population protocol binding changed")
    fragments = []
    for index, candidate in enumerate(plan["episodes"]):
        path = args.run_root / "construction_fragments" / f"{index:03d}/completion.json"
        verify_sidecar(path)
        row = json.loads(path.read_text())
        require(int(row["history_index"]) == index
                and row["candidate_identity_sha256"]
                == candidate["candidate_identity_sha256"],
                "construction fragment identity changed")
        require(row["query_policy_outcomes_read"] is False,
                "construction fragment read query outcomes")
        fragments.append(row)
    gate = protocol["population_gate"]
    selected = []
    diagnostics = {}
    for bin_spec in protocol["length_definition"]["bins_m"]:
        name = bin_spec["name"]
        eligible = [row for row in fragments
                    if row["bin_name"] == name and row["constructed"]]
        chosen = select_powered(
            eligible, histories=int(gate["minimum_histories_per_bin"]),
            scenes=int(gate["minimum_scene_clusters_per_bin"]),
            maximum_per_scene=int(gate["maximum_selected_histories_per_scene_per_bin"]),
        )
        selected.extend(chosen)
        diagnostics[name] = {
            "frozen_candidates": sum(row["bin_name"] == name for row in fragments),
            "factual_A_eligible_and_constructed": len(eligible),
            "selected_histories": len(chosen),
            "selected_scene_clusters": len({row["scene"] for row in chosen}),
        }
    role_root = args.out / "role_pairs"
    role_root.mkdir(parents=True)
    episodes = []
    for selected_index, fragment in enumerate(selected):
        source = Path(fragment["role_pair_candidate"])
        payload = json.loads((source / "role_pairs.json").read_text())
        require(sha256(source / "role_pairs.json")
                == fragment["role_pairs_sha256"],
                "candidate role-pair sidecar changed")
        destination = role_root / payload["scene"] / payload["episode"]
        require(not destination.exists(), "duplicate selected role-pair identity")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        sidecar = destination / "role_pairs.json"
        payload = json.loads(sidecar.read_text())
        payload["population_index"] = selected_index
        payload["scene_index"] = int(plan["episodes"][fragment["history_index"]]["scene_index"])
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
        payload["role_pairs_sha256"] = sha256(sidecar)
        episodes.append(payload)
    contract = {
        "online_history": "actual_frozen_navdp_goal_a_causal_rgb_trace",
        "query_execution": "independent_reset_and_exact_online_a_replay",
        "runtime_role_visibility": "none",
        "bins_m": protocol["length_definition"]["bins_m"],
        "novel_max_covis_exclusive": protocol["query_construction"][
            "novel_max_history_covis_exclusive"],
        "revisit_min_covis_inclusive": protocol["query_construction"][
            "revisit_min_history_covis_inclusive"],
        "maximum_role_distance_mismatch_m": protocol["query_construction"][
            "maximum_role_distance_mismatch_m"],
        "minimum_initial_bearing_separation_deg": protocol["query_construction"][
            "minimum_initial_bearing_separation_deg"],
        "minimum_histories_per_bin": int(gate["minimum_histories_per_bin"]),
        "minimum_scene_clusters_per_bin": int(gate["minimum_scene_clusters_per_bin"]),
        "maximum_selected_histories_per_scene_per_bin": int(
            gate["maximum_selected_histories_per_scene_per_bin"]),
        "query_policy_outcomes_read": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "HM3D actual-mono length-stratified role-free evaluation",
        "contract": contract, "episodes": episodes,
    }
    validate_manifest(manifest)
    manifest_path = role_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    manifest_path.with_name(manifest_path.name + ".sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n")
    population = {
        "schema_version": "hm3d_table3_actual_mono_population_v1_20260830",
        "scope": manifest["scope"],
        "candidate_plan_sha256": sha256(args.candidate_plan),
        "protocol_sha256": sha256(args.protocol),
        "benchmark_manifest_sha256": sha256(manifest_path),
        "histories": len(episodes), "queries": 2 * len(episodes),
        "selection_diagnostics": diagnostics,
        "query_policy_outcomes_read": False,
        "formal_policy_evaluation_authorized": False,
        "fallback_completion_allowed": False,
    }
    receipt = args.out / "population_receipt.json"
    receipt.write_text(json.dumps(population, indent=2, sort_keys=True) + "\n")
    receipt.with_name(receipt.name + ".sha256").write_text(
        f"{sha256(receipt)}  {receipt.name}\n")
    print(json.dumps({"histories": len(episodes), "diagnostics": diagnostics},
                     sort_keys=True))


if __name__ == "__main__":
    main()
