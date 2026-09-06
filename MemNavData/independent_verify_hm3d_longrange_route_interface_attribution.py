#!/usr/bin/env python3
"""Independently recount the consumed route-interface attribution."""

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
SUMMARY_SCHEMA = "hm3d_longrange_route_interface_attribution_result_v1_20260903"
VERIFY_SCHEMA = (
    "hm3d_longrange_route_interface_attribution_verification_v1_20260903"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    completion_hashes = {}
    for index in range(23):
        matches = sorted((args.run_root / "evaluation").glob(
            f"{index:03d}_*/completion.json"))
        require(len(matches) == 1,
                f"history {index} completion is missing or ambiguous")
        path = matches[0]
        sidecar = path.with_name("completion.json.sha256")
        words = sidecar.read_text().strip().split() if sidecar.is_file() else []
        digest = sha256_file(path)
        require(len(words) == 2 and words == [digest, "completion.json"],
                f"history {index} completion hash changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == CELL_SCHEMA
                and row.get("history_index") == index,
                f"history {index} identity changed")
        require(row.get("protocol_sha256") == args.expected_protocol_sha256,
                f"history {index} protocol binding changed")
        require(row.get("prefix_equality") is True
                and row.get("initial_cec_proof_equal") is True,
                f"history {index} pairing audit failed")
        rows.append(row)
        completion_hashes[str(index)] = digest

    successes = {arm: sum(row["outcomes"][arm] for row in rows)
                 for arm in ARMS}
    gain = sum(row["outcomes"][ARMS[1]] == 1
               and row["outcomes"][ARMS[0]] == 0 for row in rows)
    loss = sum(row["outcomes"][ARMS[1]] == 0
               and row["outcomes"][ARMS[0]] == 1 for row in rows)
    geometry = {arm: sum(row["geometry_stream_stop_plans"][arm] > 0
                         for row in rows) for arm in ARMS}
    projected = sum(row["audits"][ARMS[1]]["controller_support"]
                    ["support_projection_plans"] for row in rows)
    rear = sum(row["audits"][ARMS[1]]["controller_support"]
               ["rear_source_bearing_plans"] for row in rows)

    sidecar = args.summary.with_name("summary.json.sha256")
    words = sidecar.read_text().strip().split() if sidecar.is_file() else []
    summary_sha = sha256_file(args.summary)
    require(len(words) == 2 and words == [summary_sha, "summary.json"],
            "summary hash changed")
    summary = json.loads(args.summary.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "summary schema changed")
    checks = {
        "histories": 23,
        "scene_clusters": len({row["scene"] for row in rows}),
        "successes": successes,
        "support_projected_vs_direct_canonical": {
            "gain": gain, "loss": loss, "net": gain - loss},
        "geometry_stop_episodes": geometry,
        "rear_source_bearing_plans": rear,
        "support_projection_plans": projected,
        "support_transform_audits_pass": projected == rear,
    }
    for key, value in checks.items():
        require(summary.get(key) == value,
                f"summary disagrees with independent {key}")
    receipt = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "claim_scope": "consumed mechanism attribution only",
        "protocol_sha256": args.expected_protocol_sha256,
        "summary_sha256": summary_sha,
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
        "independent_counts": checks,
        "completion_sha256": completion_hashes,
    }
    require(not args.out.exists(), "verification output already exists")
    encoded = (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {args.out.name}\n")
    print(json.dumps({"status": "complete", "verified": True,
                      "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
