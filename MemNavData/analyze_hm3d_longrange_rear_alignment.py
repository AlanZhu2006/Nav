#!/usr/bin/env python3
"""Aggregate the consumed paired long-range rear-alignment diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from MemNavData.hm3d_longrange_rear_alignment import (
    ARMS,
    PARENT_STUCK_INDICES,
    PARENT_SUCCESS_INDICES,
    SELECTED_HISTORY_INDICES,
    require,
)


CELL_SCHEMA = "hm3d_longrange_rear_alignment_history_v1_20260904"
PROTOCOL_SCHEMA = "hm3d_longrange_rear_alignment_protocol_v1_20260904"
SUMMARY_SCHEMA = "hm3d_longrange_rear_alignment_result_v1_20260904"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar_p(gain: int, loss: int) -> float:
    discordant = int(gain) + int(loss)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gain, loss) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "rear-alignment protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA
            and tuple(protocol.get("selected_history_indices", ()))
            == SELECTED_HISTORY_INDICES
            and tuple(protocol.get("arms", ())) == ARMS,
            "rear-alignment protocol contract changed")

    rows = []
    for index in SELECTED_HISTORY_INDICES:
        matches = sorted((args.run_root / "evaluation").glob(
            f"{index:03d}_*/completion.json"))
        require(len(matches) == 1,
                f"history {index} completion is missing or ambiguous")
        path = matches[0]
        sidecar = path.with_name("completion.json.sha256")
        words = sidecar.read_text().strip().split() if sidecar.is_file() else []
        require(words == [sha256_file(path), "completion.json"],
                f"history {index} completion hash changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == CELL_SCHEMA
                and row.get("history_index") == index
                and row.get("protocol_sha256")
                == args.expected_protocol_sha256,
                f"history {index} completion contract changed")
        require(row.get("prefix_equality") is True
                and row.get("initial_cec_proof_equal") is True
                and row.get("initial_route_packet_equal") is True
                and tuple(row.get("arms", ())) == ARMS,
                f"history {index} pairing audit failed")
        aligned = row["audits"][ARMS[1]]["rear_alignment"]
        unaligned = row["audits"][ARMS[0]]["rear_alignment"]
        require(unaligned.get("alignment_actions") == 0
                and aligned.get("alignment_actions", 0) > 0
                and aligned.get("alignment_actions")
                == aligned.get("fresh_observation_receipts")
                == aligned.get("control_edges_consumed")
                and aligned.get("visual_pnp_edges_during_alignment") == 0
                and aligned.get("zero_translation") is True
                and aligned.get("simulator_pose_receipt_used") is False
                and aligned.get("packet_sha256")
                == unaligned.get("packet_sha256")
                == row.get("initial_route_packet_sha256"),
                f"history {index} alignment execution audit failed")
        rows.append(row)

    successes = {arm: sum(int(row["outcomes"][arm]) for row in rows)
                 for arm in ARMS}
    gain = sum(row["outcomes"][ARMS[1]] == 1
               and row["outcomes"][ARMS[0]] == 0 for row in rows)
    loss = sum(row["outcomes"][ARMS[1]] == 0
               and row["outcomes"][ARMS[0]] == 1 for row in rows)
    geometry_stops = {
        arm: sum(int(row["geometry_stream_stop_plans"][arm] > 0)
                 for row in rows) for arm in ARMS
    }
    by_parent_partition = {}
    for name, indices in (
            ("parent_stuck", PARENT_STUCK_INDICES),
            ("parent_success", PARENT_SUCCESS_INDICES)):
        selected = [row for row in rows if row["history_index"] in indices]
        by_parent_partition[name] = {
            "histories": len(selected),
            "successes": {
                arm: sum(int(row["outcomes"][arm]) for row in selected)
                for arm in ARMS
            },
            "aligned_vs_unaligned_gain": sum(
                row["outcomes"][ARMS[1]] == 1
                and row["outcomes"][ARMS[0]] == 0 for row in selected),
            "aligned_vs_unaligned_loss": sum(
                row["outcomes"][ARMS[1]] == 0
                and row["outcomes"][ARMS[0]] == 1 for row in selected),
        }
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
    advance = bool(gain >= 2 and loss <= 1 and gain - loss > 0)
    decision = (
        "advance_proof_bound_rear_alignment_to_fresh_longrange_confirmation"
        if advance else
        "do_not_promote_rear_alignment_into_longrange_method"
    )
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "claim_scope": "consumed controller-support mechanism only",
        "protocol_sha256": args.expected_protocol_sha256,
        "histories": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "successes": successes,
        "aligned_vs_unaligned": {
            "gain": gain,
            "loss": loss,
            "net": gain - loss,
            "exact_mcnemar_p_descriptive_only": exact_mcnemar_p(gain, loss),
        },
        "geometry_stop_episodes": geometry_stops,
        "by_formal_parent_partition": by_parent_partition,
        "execution_audit_totals": audit_totals,
        "decision": decision,
        "navigation_claim_allowed": False,
        "significance_claim_allowed": False,
        "fresh_confirmation_required": True,
        "rows": [{
            "history_index": row["history_index"],
            "scene": row["scene"],
            "formal_parent_partition": row["formal_parent_partition"],
            "formal_parent_outcome": row["formal_parent_outcome"],
            "outcomes": row["outcomes"],
            "termination_reason": row["termination_reason"],
            "geometry_stream_stop_plans": row[
                "geometry_stream_stop_plans"],
            "initial_heading_deg": row["audits"][ARMS[1]][
                "rear_alignment"]["initial_heading_deg"],
            "alignment_actions": row["audits"][ARMS[1]][
                "rear_alignment"]["alignment_actions"],
        } for row in rows],
    }
    output = args.run_root / "result" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "rear-alignment summary already exists")
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.write_bytes(encoded)
    output.with_name("summary.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  summary.json\n")
    print(json.dumps({"status": "complete", "successes": successes,
                      "decision": decision, "output": str(output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
