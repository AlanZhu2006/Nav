#!/usr/bin/env python3
"""Aggregate the three-history consumed dense-query mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hm3d_longrange_dense_live_query_gate_summary_v2_20260903"
HISTORY_SCHEMA = "hm3d_longrange_dense_live_query_gate_history_v2_20260903"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_summary(run_root: Path, protocol_path: Path) -> dict:
    protocol_sha = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    selected = [int(value) for value in protocol["selected_history_indices"]]
    require(selected == [0, 1, 6],
            "dense-live-query gate selection changed")
    rows = []
    hashes = {}
    for index in selected:
        matches = sorted((run_root / "evaluation").glob(
            f"{index:03d}_*/completion.json"))
        require(len(matches) == 1,
                f"history {index} completion is missing or ambiguous")
        path = matches[0]
        sidecar = path.with_name(path.name + ".sha256")
        words = sidecar.read_text().strip().split()
        require(len(words) == 2 and words[0] == sha256_file(path)
                and words[1] == path.name,
                f"history {index} completion sidecar changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == HISTORY_SCHEMA
                and int(row["history_index"]) == index
                and row["protocol_sha256"] == protocol_sha,
                f"history {index} completion contract changed")
        require(row.get("initial_cec_discrete_proof_equal") is True,
                f"history {index} changed the initial CEC proof")
        require(row.get("motion_model") == "direct_pnp_dense_query"
                and row.get("historical_route_motion_model")
                == "fundamental_then_pnp"
                and row.get("live_query_motion_model") == "direct_pnp"
                and row.get("live_query_sampling") == "per_action_dense",
                f"history {index} changed the stage-specific estimator")
        dense = row.get("dense_sampling_audit") or {}
        require(dense.get("all_completed_intervals_dense") is True
                and dense.get("runtime_role_or_distance_gate_present")
                is False,
                f"history {index} did not use dense causal query motion")
        require(row.get("navigation_claim_allowed") is False
                and row.get("fresh_confirmation_required") is True,
                f"history {index} escaped consumed-mechanism scope")
        rows.append(row)
        hashes[str(index)] = sha256_file(path)
    stops = sum(int(row["geometry_stream_stop_plans"]) for row in rows)
    passed = all(row["mechanism_gate_passed"] is True for row in rows)
    require(passed == (stops == 0), "dense-query pass logic changed")
    return {
        "schema_version": SCHEMA,
        "claim_scope": protocol["claim_scope"],
        "protocol_sha256": protocol_sha,
        "histories": len(rows),
        "history_indices": selected,
        "scene_clusters": len({row["scene"] for row in rows}),
        "geometry_stream_failures": stops,
        "mechanism_gate_passed": passed,
        "descriptive_successes": sum(
            int(row["outcome_descriptive_only"]) for row in rows),
        "query_motion_edges": sum(
            int(row["dense_sampling_audit"]["query_motion_edges"])
            for row in rows),
        "completion_sha256": hashes,
        "decision": (
            "authorize_consumed_full_population_dense_query_interface_"
            "attribution"
            if passed else
            "do_not_relax_thresholds_inspect_failed_atomic_transition"
        ),
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.run_root, args.protocol)
    result_root = args.run_root / "result"
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "summary.json"
    require(not path.exists(), f"summary already exists: {path}")
    encoded = (json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
