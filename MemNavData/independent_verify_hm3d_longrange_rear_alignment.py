#!/usr/bin/env python3
"""Independently recount the consumed rear-alignment diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("route_tangent_unaligned", "route_tangent_rear_aligned")
INDICES = (3, 4, 5, 8, 10, 11, 13, 15, 21)
CELL_SCHEMA = "hm3d_longrange_rear_alignment_history_v1_20260904"
SUMMARY_SCHEMA = "hm3d_longrange_rear_alignment_result_v1_20260904"
VERIFY_SCHEMA = "hm3d_longrange_rear_alignment_verification_v1_20260904"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_p(gain: int, loss: int) -> float:
    count = gain + loss
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, index)
               for index in range(min(gain, loss) + 1)) / (2 ** count)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    completion_hashes = {}
    for index in INDICES:
        matches = sorted((args.run_root / "evaluation").glob(
            f"{index:03d}_*/completion.json"))
        require(len(matches) == 1,
                f"history {index} completion is missing or ambiguous")
        path = matches[0]
        digest = sha256_file(path)
        sidecar = path.with_name("completion.json.sha256")
        words = sidecar.read_text().strip().split() if sidecar.is_file() else []
        require(words == [digest, "completion.json"],
                f"history {index} completion hash changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == CELL_SCHEMA
                and row.get("history_index") == index
                and row.get("protocol_sha256")
                == args.expected_protocol_sha256
                and row.get("prefix_equality") is True
                and row.get("initial_cec_proof_equal") is True
                and row.get("initial_route_packet_equal") is True,
                f"history {index} contract/pairing changed")
        aligned = row["audits"][ARMS[1]]["rear_alignment"]
        unaligned = row["audits"][ARMS[0]]["rear_alignment"]
        require(aligned["alignment_actions"] > 0
                and aligned["alignment_actions"]
                == aligned["fresh_observation_receipts"]
                == aligned["control_edges_consumed"]
                and aligned["visual_pnp_edges_during_alignment"] == 0
                and aligned["zero_translation"] is True
                and aligned["simulator_pose_receipt_used"] is False
                and unaligned["alignment_actions"] == 0
                and aligned["packet_sha256"] == unaligned["packet_sha256"]
                == row["initial_route_packet_sha256"],
                f"history {index} execution audit changed")
        rows.append(row)
        completion_hashes[str(index)] = digest

    successes = {arm: sum(int(row["outcomes"][arm]) for row in rows)
                 for arm in ARMS}
    gain = sum(row["outcomes"][ARMS[1]] == 1
               and row["outcomes"][ARMS[0]] == 0 for row in rows)
    loss = sum(row["outcomes"][ARMS[1]] == 0
               and row["outcomes"][ARMS[0]] == 1 for row in rows)
    geometry = {arm: sum(row["geometry_stream_stop_plans"][arm] > 0
                         for row in rows) for arm in ARMS}
    audit_totals = {
        "alignment_actions": sum(row["audits"][ARMS[1]][
            "rear_alignment"]["alignment_actions"] for row in rows),
        "fresh_observation_receipts": sum(row["audits"][ARMS[1]][
            "rear_alignment"]["fresh_observation_receipts"] for row in rows),
        "control_edges_consumed": sum(row["audits"][ARMS[1]][
            "rear_alignment"]["control_edges_consumed"] for row in rows),
        "visual_pnp_edges_during_alignment": 0,
        "minimum_initial_abs_heading_deg": min(abs(row["audits"][ARMS[1]][
            "rear_alignment"]["initial_heading_deg"]) for row in rows),
        "maximum_yaw_atom_deg": max(row["audits"][ARMS[1]][
            "rear_alignment"]["max_abs_action_deg"] for row in rows),
    }
    by_parent = {}
    for name, indices in (("parent_stuck", {3, 4, 8, 10, 21}),
                          ("parent_success", {5, 11, 13, 15})):
        part = [row for row in rows if row["history_index"] in indices]
        by_parent[name] = {
            "histories": len(part),
            "successes": {arm: sum(row["outcomes"][arm] for row in part)
                          for arm in ARMS},
            "aligned_vs_unaligned_gain": sum(
                row["outcomes"][ARMS[1]] == 1
                and row["outcomes"][ARMS[0]] == 0 for row in part),
            "aligned_vs_unaligned_loss": sum(
                row["outcomes"][ARMS[1]] == 0
                and row["outcomes"][ARMS[0]] == 1 for row in part),
        }

    summary_sidecar = args.summary.with_name("summary.json.sha256")
    summary_sha = sha256_file(args.summary)
    words = (summary_sidecar.read_text().strip().split()
             if summary_sidecar.is_file() else [])
    require(words == [summary_sha, "summary.json"],
            "rear-alignment summary hash changed")
    summary = json.loads(args.summary.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "rear-alignment summary schema changed")
    checks = {
        "histories": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "successes": successes,
        "aligned_vs_unaligned": {
            "gain": gain, "loss": loss, "net": gain - loss,
            "exact_mcnemar_p_descriptive_only": exact_p(gain, loss)},
        "geometry_stop_episodes": geometry,
        "by_formal_parent_partition": by_parent,
        "execution_audit_totals": audit_totals,
    }
    for key, value in checks.items():
        require(summary.get(key) == value,
                f"summary disagrees with independent {key}")
    expected_decision = (
        "advance_proof_bound_rear_alignment_to_fresh_longrange_confirmation"
        if gain >= 2 and loss <= 1 and gain - loss > 0 else
        "do_not_promote_rear_alignment_into_longrange_method")
    require(summary.get("decision") == expected_decision,
            "summary decision disagrees with frozen rule")
    receipt = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "claim_scope": "consumed controller-support mechanism only",
        "protocol_sha256": args.expected_protocol_sha256,
        "summary_sha256": summary_sha,
        "independent_counts": checks,
        "decision": expected_decision,
        "navigation_claim_allowed": False,
        "significance_claim_allowed": False,
        "fresh_confirmation_required": True,
        "completion_sha256": completion_hashes,
    }
    require(not args.out.exists(), "verification output already exists")
    encoded = (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {args.out.name}\n")
    print(json.dumps({"status": "complete", "verified": True,
                      "decision": expected_decision,
                      "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
