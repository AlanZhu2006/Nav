#!/usr/bin/env python3
"""Aggregate the preregistered consumed route-interface attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = (
    "direct_pnp_canonical_bearing",
    "direct_pnp_support_projected_bearing",
)
CELL_SCHEMA = "hm3d_longrange_route_interface_attribution_history_v1_20260903"
PROTOCOL_SCHEMA = (
    "hm3d_longrange_route_interface_attribution_protocol_v1_20260903"
)
SUMMARY_SCHEMA = (
    "hm3d_longrange_route_interface_attribution_result_v1_20260903"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def discordance(rows: list[dict], arm: str, reference: str) -> dict[str, int]:
    gain = sum(row["outcomes"][arm] == 1
               and row["outcomes"][reference] == 0 for row in rows)
    loss = sum(row["outcomes"][arm] == 0
               and row["outcomes"][reference] == 1 for row in rows)
    return {"gain": gain, "loss": loss, "net": gain - loss}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "route-interface attribution protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "route-interface attribution protocol schema changed")

    rows = []
    for index in range(23):
        matches = sorted((args.run_root / "evaluation").glob(
            f"{index:03d}_*/completion.json"))
        require(len(matches) == 1,
                f"history {index} completion is missing or ambiguous")
        path = matches[0]
        sidecar = path.with_name("completion.json.sha256")
        words = sidecar.read_text().strip().split() if sidecar.is_file() else []
        require(len(words) == 2 and words[0] == sha256_file(path)
                and words[1] == "completion.json",
                f"history {index} completion hash changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == CELL_SCHEMA
                and row.get("history_index") == index,
                f"history {index} completion contract changed")
        require(row.get("protocol_sha256") == args.expected_protocol_sha256,
                f"history {index} protocol binding changed")
        require(row.get("prefix_equality") is True
                and row.get("initial_cec_proof_equal") is True,
                f"history {index} pairing audit failed")
        require(tuple(row.get("arms", ())) == ARMS,
                f"history {index} arm set changed")
        rows.append(row)

    successes = {
        arm: sum(int(row["outcomes"][arm]) for row in rows) for arm in ARMS
    }
    formal_success = sum(int(row["formal_parent_route_tangent_outcome"])
                         for row in rows)
    support_vs_canonical = discordance(
        rows, ARMS[1], ARMS[0])
    direct_vs_formal_gain = sum(
        row["outcomes"][ARMS[0]] == 1
        and row["formal_parent_route_tangent_outcome"] == 0 for row in rows)
    direct_vs_formal_loss = sum(
        row["outcomes"][ARMS[0]] == 0
        and row["formal_parent_route_tangent_outcome"] == 1 for row in rows)
    direct_vs_formal = {
        "gain": direct_vs_formal_gain,
        "loss": direct_vs_formal_loss,
        "net": direct_vs_formal_gain - direct_vs_formal_loss,
        "same_process_pair": False,
    }
    geometry_stops = {
        arm: sum(int(row["geometry_stream_stop_plans"][arm] > 0)
                 for row in rows) for arm in ARMS
    }
    projected_plans = sum(
        int(row["audits"][ARMS[1]]["controller_support"]
            ["support_projection_plans"]) for row in rows)
    rear_plans = sum(
        int(row["audits"][ARMS[1]]["controller_support"]
            ["rear_source_bearing_plans"]) for row in rows)
    transform_audits_pass = projected_plans == rear_plans

    support_selected = bool(
        support_vs_canonical["net"] > 0
        and geometry_stops[ARMS[1]] <= geometry_stops[ARMS[0]]
        and transform_audits_pass
    )
    direct_selected = bool(
        not support_selected
        and direct_vs_formal["net"] > 0
        and support_vs_canonical["net"] <= 0
    )
    if support_selected:
        decision = "fresh_confirm_direct_pnp_support_projected_bearing"
    elif direct_selected:
        decision = "fresh_confirm_direct_pnp_canonical_bearing"
    else:
        decision = "no_longrange_route_candidate_selected"
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "claim_scope": "consumed mechanism attribution only",
        "protocol_sha256": args.expected_protocol_sha256,
        "histories": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "successes": successes,
        "sealed_formal_route_tangent_successes": formal_success,
        "support_projected_vs_direct_canonical": support_vs_canonical,
        "direct_canonical_vs_sealed_formal": direct_vs_formal,
        "geometry_stop_episodes": geometry_stops,
        "rear_source_bearing_plans": rear_plans,
        "support_projection_plans": projected_plans,
        "support_transform_audits_pass": transform_audits_pass,
        "selected_for_fresh_confirmation": decision,
        "significance_claim_allowed": False,
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
        "rows": [{
            "history_index": row["history_index"],
            "scene": row["scene"],
            "formal_route_tangent": row[
                "formal_parent_route_tangent_outcome"],
            "outcomes": row["outcomes"],
            "geometry_stream_stop_plans": row[
                "geometry_stream_stop_plans"],
            "support_projection_plans": row["audits"][ARMS[1]]
            ["controller_support"]["support_projection_plans"],
        } for row in rows],
    }
    output = args.run_root / "result" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "attribution summary already exists")
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.write_bytes(encoded)
    output.with_name("summary.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  summary.json\n")
    print(json.dumps({
        "status": "complete",
        "successes": successes,
        "decision": decision,
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
