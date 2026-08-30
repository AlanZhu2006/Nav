#!/usr/bin/env python3
"""Independently audit the failed pre-query Table-III population gate.

This audit reads only frozen factual-history and role-pair construction
receipts.  It never reads or executes a query policy.  Its purpose is to
separate a source-history constructibility failure from a navigation-method
result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROTOCOL_SCHEMA = "hm3d_table3_actual_mono_protocol_v1_20260830"
AUDIT_SCHEMA = "hm3d_table3_actual_mono_constructibility_audit_v1_20260830"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(
        sidecar.is_file() and sidecar.read_text().split() == [digest, path.name],
        f"invalid receipt sidecar: {path}",
    )
    return digest


def audit(run_root: Path, candidate_plan: Path, protocol_path: Path) -> dict:
    plan = json.loads(candidate_plan.read_text())
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "Table-III protocol schema changed")
    require(plan.get("protocol_sha256") == sha256(protocol_path),
            "candidate plan/protocol binding changed")
    candidates = plan.get("episodes")
    require(isinstance(candidates, list) and len(candidates) == 125,
            "frozen Table-III candidate count changed")

    bins = [row["name"] for row in protocol["length_definition"]["bins_m"]]
    diagnostics = {
        name: {
            "frozen_candidates": 0,
            "factual_A_reached": 0,
            "factual_A_history_eligible": 0,
            "constructible_role_pairs": 0,
            "reached_scene_clusters": set(),
            "history_eligible_scene_clusters": set(),
            "constructible_scene_clusters": set(),
            "construction_status": Counter(),
            "geometry_failure_reasons": Counter(),
        }
        for name in bins
    }
    factual_artifacts = []
    fragment_artifacts = []

    minimum_frames = int(protocol["history"]["minimum_frames"])
    for index, candidate in enumerate(candidates):
        require(int(candidate["history_index"]) == index,
                "candidate plan order changed")
        scene = str(candidate["scene"])
        episode = str(candidate["episode"])
        bin_name = str(candidate["bin_name"])
        require(bin_name in diagnostics, "candidate length bin changed")
        stats = diagnostics[bin_name]
        stats["frozen_candidates"] += 1

        factual_path = (
            run_root / "factual_a"
            / f"{index:03d}_{scene}_episode_{episode}" / "completion.json"
        )
        factual_digest = verify_sidecar(factual_path)
        factual = json.loads(factual_path.read_text())
        require(
            int(factual["history_index"]) == index
            and factual["scene"] == scene
            and factual["bin_name"] == bin_name
            and factual["candidate_identity_sha256"]
            == candidate["candidate_identity_sha256"],
            f"factual identity changed at {index}",
        )
        require(factual.get("query_policy_outcomes_read") is False,
                f"factual receipt read query outcomes at {index}")
        trace_path = Path(factual["trace_path"])
        require(trace_path.is_file() and sha256(trace_path) == factual["trace_sha256"],
                f"factual trace changed at {index}")
        trace = json.loads(trace_path.read_text())
        require(
            bool(trace["reached"]) == bool(factual["reached_A"])
            and int(trace["steps"]) == int(factual["steps_A"])
            and len(trace["poses"]) == int(factual["steps_A"]),
            f"factual trace/receipt mismatch at {index}",
        )
        expected_eligible = bool(trace["reached"] and
                                 len(trace["poses"]) >= minimum_frames)
        require(bool(factual["history_eligible"]) == expected_eligible,
                f"history eligibility does not reproduce at {index}")
        if factual["reached_A"]:
            stats["factual_A_reached"] += 1
            stats["reached_scene_clusters"].add(scene)
        if factual["history_eligible"]:
            stats["factual_A_history_eligible"] += 1
            stats["history_eligible_scene_clusters"].add(scene)
        factual_artifacts.append(factual_digest)

        fragment_path = (
            run_root / "construction_fragments" / f"{index:03d}"
            / "completion.json"
        )
        fragment_digest = verify_sidecar(fragment_path)
        fragment = json.loads(fragment_path.read_text())
        require(
            int(fragment["history_index"]) == index
            and fragment["scene"] == scene
            and fragment["bin_name"] == bin_name
            and fragment["candidate_identity_sha256"]
            == candidate["candidate_identity_sha256"]
            and fragment["factual_A_completion_sha256"] == factual_digest,
            f"construction identity changed at {index}",
        )
        require(fragment.get("query_policy_outcomes_read") is False,
                f"construction receipt read query outcomes at {index}")
        status = str(fragment["status"])
        require(status in {"factual_A_ineligible", "geometry_ineligible", "constructed"},
                f"unknown construction status at {index}")
        require(bool(fragment["constructed"]) == (status == "constructed"),
                f"construction flag/status mismatch at {index}")
        if status == "factual_A_ineligible":
            require(not factual["history_eligible"],
                    f"eligible factual history was labelled ineligible at {index}")
        else:
            require(factual["history_eligible"],
                    f"ineligible factual history entered geometry at {index}")
        stats["construction_status"][status] += 1
        if status == "geometry_ineligible":
            reason = str(fragment.get("reason", ""))
            require(bool(reason), f"geometry failure lacks a reason at {index}")
            stats["geometry_failure_reasons"][reason] += 1
        elif status == "constructed":
            role_pair = Path(fragment["role_pair_candidate"]) / "role_pairs.json"
            require(role_pair.is_file()
                    and sha256(role_pair) == fragment["role_pairs_sha256"],
                    f"constructed role-pair changed at {index}")
            stats["constructible_role_pairs"] += 1
            stats["constructible_scene_clusters"].add(scene)
        fragment_artifacts.append(fragment_digest)

    gate = protocol["population_gate"]
    minimum_histories = int(gate["minimum_histories_per_bin"])
    minimum_scenes = int(gate["minimum_scene_clusters_per_bin"])
    serial = {}
    for name, stats in diagnostics.items():
        row = dict(stats)
        for field in (
            "reached_scene_clusters",
            "history_eligible_scene_clusters",
            "constructible_scene_clusters",
        ):
            row[field] = len(stats[field])
        row["construction_status"] = dict(stats["construction_status"])
        row["geometry_failure_reasons"] = dict(stats["geometry_failure_reasons"])
        row["powered_gate_passed"] = bool(
            row["constructible_role_pairs"] >= minimum_histories
            and row["constructible_scene_clusters"] >= minimum_scenes
        )
        serial[name] = row

    evaluation_root = run_root / "evaluation"
    query_metric_rows = (
        sum(1 for _ in evaluation_root.rglob("metric.csv"))
        if evaluation_root.exists() else 0
    )
    query_completions = (
        sum(1 for _ in evaluation_root.rglob("completion.json"))
        if evaluation_root.exists() else 0
    )
    require(query_metric_rows == 0 and query_completions == 0,
            "query-policy artifacts exist in failed constructibility run")

    return {
        "schema_version": AUDIT_SCHEMA,
        "verified": True,
        "candidate_plan_sha256": sha256(candidate_plan),
        "protocol_sha256": sha256(protocol_path),
        "run_root": str(run_root),
        "frozen_candidates": len(candidates),
        "bins": serial,
        "population_gate_passed": all(
            row["powered_gate_passed"] for row in serial.values()),
        "formal_policy_evaluation_authorized": False,
        "query_metric_files": query_metric_rows,
        "query_completion_files": query_completions,
        "query_policy_outcomes_read": False,
        "factual_receipt_set_sha256": hashlib.sha256(
            "\n".join(factual_artifacts).encode()
        ).hexdigest(),
        "construction_receipt_set_sha256": hashlib.sha256(
            "\n".join(fragment_artifacts).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "constructibility audit already exists")
    result = audit(args.run_root, args.candidate_plan, args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
