#!/usr/bin/env python3
"""Aggregate the contract-only Fresh-HM3D controller portability pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA = "cec_proof_locked_portability_pilot_summary_v1_20260827"
CONTROLLERS = ("navdp", "vint", "iplanner")
HISTORY_INDICES = (0, 7, 14, 21)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(run_root: Path, query_manifest: Path) -> dict[str, Any]:
    manifest = json.loads(query_manifest.read_text())
    require(manifest.get("schema_version")
            == "cec_first_decision_accepted_population_v1_20260827",
            "accepted query manifest schema changed")
    rows = []
    for path in sorted((run_root / "evaluation").glob(
            "*/*/authority_pair_audit.json")):
        row = json.loads(path.read_text())
        require(row.get("verified") is True, f"unverified cell: {path}")
        require(row.get("handoff_packet_verified") is True,
                f"handoff packet was not verified: {path}")
        require(row.get("source_accepted_manifest_match") is True,
                f"source accepted proof changed: {path}")
        controller = str(row.get("controller"))
        require(controller in CONTROLLERS, f"unexpected controller: {path}")
        cell = path.parent
        contract = json.loads(
            (cell / "authority_pair_contract.json").read_text())
        require(contract.get("controller") == controller,
                f"controller receipt mismatch: {path}")
        require(contract.get("query_manifest_sha256")
                == sha256_file(query_manifest),
                f"query manifest binding changed: {path}")
        order = contract.get("authority_order")
        require(order in (["grant", "forced_reject_native"],
                          ["forced_reject_native", "grant"]),
                f"invalid authority order: {path}")
        rows.append({**row, "authority_order": order,
                     "audit_path": str(path)})
    require(len(rows) == len(CONTROLLERS) * len(HISTORY_INDICES),
            f"pilot cell count {len(rows)} != 12")

    expected = {
        (int(entry["history_index"]), str(entry["scene"]),
         str(entry["episode"]), str(entry["query_id"]))
        for entry in manifest["queries"]
        if int(entry["history_index"]) in HISTORY_INDICES
    }
    require(len(expected) == len(HISTORY_INDICES),
            "pilot histories are not uniquely selected")
    identities = {
        (row["scene"], row["episode"], row["query_id"])
        for row in rows
    }
    require(identities == {(scene, episode, query)
                           for _, scene, episode, query in expected},
            "pilot query population changed")
    require(len({scene for _, scene, _, _ in expected}) == 4,
            "pilot histories are not four scene clusters")

    by_query: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_controller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = (row["scene"], row["episode"], row["query_id"])
        by_query[identity].append(row)
        by_controller[row["controller"]].append(row)
    for identity, group in by_query.items():
        require({row["controller"] for row in group} == set(CONTROLLERS),
                f"controller triad incomplete for {identity}")
        require(len({row["first_handoff_proof_sha256"] for row in group}) == 1,
                f"cross-controller first proof differs for {identity}")
        require(len({row["first_handoff_anchor"] for row in group}) == 1,
                f"cross-controller anchor differs for {identity}")
        require(len({row["handoff_packet_sha256"] for row in group}) == 1,
                f"cross-controller handoff packet differs for {identity}")

    controller_results = {}
    for controller in CONTROLLERS:
        group = by_controller[controller]
        require(len(group) == 4, f"{controller}: incomplete pilot")
        orders = [tuple(row["authority_order"]) for row in group]
        require(orders.count(("grant", "forced_reject_native")) == 2
                and orders.count(("forced_reject_native", "grant")) == 2,
                f"{controller}: authority order is not balanced")
        controller_results[controller] = {
            "n": 4,
            "grant_success": sum(int(row["grant_success"]) for row in group),
            "forced_reject_success": sum(
                int(row["forced_reject_success"]) for row in group),
            "paired_gain": sum(int(row["paired_gain"]) for row in group),
            "paired_loss": sum(int(row["paired_loss"]) for row in group),
            "mean_grant_progress_m": mean(
                float(row["grant_progress_m"]) for row in group),
            "mean_forced_progress_m": mean(
                float(row["forced_reject_progress_m"]) for row in group),
            "performance_used_as_gate": False,
        }

    return {
        "schema_version": SCHEMA,
        "verified": True,
        "claim_scope": (
            "infrastructure and causal-attribution pilot; not an SR result"),
        "run_root": str(run_root),
        "query_manifest_sha256": sha256_file(query_manifest),
        "cells": len(rows),
        "histories": len(HISTORY_INDICES),
        "scene_clusters": 4,
        "controllers": list(CONTROLLERS),
        "same_packet_across_controller_triads": True,
        "performance_used_as_gate": False,
        "controller_results": controller_results,
        "raw_audit_sha256": {
            row["audit_path"]: sha256_file(Path(row["audit_path"]))
            for row in rows
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.run_root.resolve(), args.query_manifest.resolve())
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": True, "cells": payload["cells"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
