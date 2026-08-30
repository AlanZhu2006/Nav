#!/usr/bin/env python3
"""Independently verify the sealed HM3D Table-3 geometry-capacity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_hm3d_table3_navmesh_capacity import (
    SCHEMA,
    SUMMARY_SCHEMA,
    _select_population,
    bin_specs,
    in_bin,
    load_protocol,
    require,
    sha256_file,
)


VERIFY_SCHEMA = "hm3d_table3_navmesh_capacity_verification_v1_20260830"


def verify(summary_path: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "capacity summary schema changed")
    require(summary["protocol_sha256"] == sha256_file(protocol_path),
            "capacity summary protocol binding changed")
    require(summary["query_policy_outcomes_read"] is False
            and summary["navigation_outcomes_read"] is False,
            "capacity summary read policy outcomes")
    require(summary["rendered_support_verified"] is False
            and summary["policy_evaluation_authorized"] is False,
            "capacity summary exceeded geometry-only authority")
    fragments = []
    seen_scenes = set()
    for ledger in summary["scene_fragments"]:
        path = Path(ledger["path"])
        require(path.is_file() and sha256_file(path) == ledger["sha256"],
                "scene fragment ledger changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == SCHEMA,
                "scene fragment schema changed")
        scene = str(row["scene"])
        require(scene == ledger["scene"] and scene not in seen_scenes,
                "duplicate or changed scene fragment")
        seen_scenes.add(scene)
        require(row["query_policy_outcomes_read"] is False
                and row["navigation_outcomes_read"] is False
                and row["policy_evaluation_authorized"] is False,
                "scene fragment crossed authority boundary")
        for spec in bin_specs(protocol):
            for triad in row["candidate_triads"][spec["name"]]:
                require(in_bin(float(triad["first_goal_geodesic_m"]), spec)
                        and in_bin(float(triad["second_goal_geodesic_m"]), spec),
                        "saved triad escaped its distance bin")
                require(float(triad["goal_distance_mismatch"]) <= float(
                    protocol["paired_geometry"]["maximum_goal_distance_mismatch"]
                ) + 1e-9, "saved triad violates distance matching")
                require(float(triad["initial_bearing_separation_deg"]) >= float(
                    protocol["paired_geometry"][
                        "minimum_initial_bearing_separation_deg"]
                ) - 1e-9, "saved triad violates direction separation")
                require(float(triad["goal_to_goal_geodesic_m"]) >= float(
                    protocol["paired_geometry"]["minimum_goal_to_goal_geodesic_m"]
                ) - 1e-9, "saved triad goals are not distinct")
        fragments.append(row)
    require(len(fragments) == int(protocol["parent"]["expected_scenes"]),
            "scene fragment count changed")
    selections, diagnostics = _select_population(fragments, protocol)
    require(selections == summary["selected_geometry_proposals"],
            "geometry proposal selection does not reproduce")
    require(diagnostics == summary["bin_diagnostics"],
            "bin diagnostics do not reproduce")
    all_passed = all(
        row["geometry_capacity_gate_passed"] for row in diagnostics.values()
    )
    require(all_passed == summary["all_geometry_capacity_gates_passed"],
            "global geometry gate does not reproduce")
    return {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "protocol_sha256": sha256_file(protocol_path),
        "scene_fragments": len(fragments),
        "bin_diagnostics": diagnostics,
        "all_geometry_capacity_gates_passed": all_passed,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "rendered_support_verified": False,
        "policy_evaluation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.summary, args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
