#!/usr/bin/env python3
"""Audit trajectory-length-bin support from a sealed role-pair manifest.

This is a construction audit, not a navigation-result analysis.  It reads no
arm outcome and reports whether a frozen population can support the requested
shortest-path bins before any additional rollout is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


BINS = (
    ("0_to_20_m", 0.0, 20.0, False),
    ("20_to_30_m", 20.0, 30.0, False),
    ("30_to_50_m", 30.0, 50.0, True),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bin_name(distance: float) -> str | None:
    for name, lower, upper, include_upper in BINS:
        if distance >= lower and (distance <= upper if include_upper else distance < upper):
            return name
    return None


def audit_manifest(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    digest = sha256_file(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "role-pair manifest SHA-256 changed")
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), "manifest must be a JSON object")
    require(payload.get("schema_version") == "shared_online_role_pair_v1_20260814",
            "role-pair manifest schema changed")
    episodes = payload.get("episodes")
    require(isinstance(episodes, list) and episodes, "manifest has no episodes")
    contract = payload.get("contract")
    require(isinstance(contract, dict), "manifest contract is missing")
    require(contract.get("runtime_role_visibility") == "none",
            "runtime role visibility changed")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for episode in episodes:
        require(isinstance(episode, dict), "episode row is not an object")
        scene = str(episode.get("scene", ""))
        episode_id = str(episode.get("episode", ""))
        require(scene and episode_id, "episode identity is incomplete")
        pairs = episode.get("pairs")
        require(isinstance(pairs, list) and pairs, "episode has no role pair")
        for pair in pairs:
            require(isinstance(pair, dict), "pair row is not an object")
            queries = pair.get("queries")
            require(isinstance(queries, list) and queries,
                    "role pair has no queries")
            for query in queries:
                require(isinstance(query, dict), "query row is not an object")
                role = str(query.get("analysis_role", ""))
                require(role in {"novel", "revisit"}, "unexpected analysis role")
                identity = (scene, episode_id, str(query.get("query_id", "")))
                require(identity[2] and identity not in identities,
                        "duplicate or missing query identity")
                identities.add(identity)
                try:
                    distance = float(query["geodesic_from_a_end_m"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError("query geodesic is missing or invalid") from exc
                require(math.isfinite(distance) and distance >= 0.0,
                        "query geodesic must be finite and non-negative")
                rows.append({
                    "scene": scene,
                    "episode": episode_id,
                    "query_id": identity[2],
                    "role": role,
                    "geodesic_m": distance,
                    "bin": _bin_name(distance),
                })

    require({row["role"] for row in rows} == {"novel", "revisit"},
            "both Novel and Revisit must be present")
    counts: dict[str, dict[str, int]] = {}
    for name, _lower, _upper, _include_upper in BINS:
        role_counts = {
            role: sum(row["bin"] == name and row["role"] == role for row in rows)
            for role in ("novel", "revisit")
        }
        counts[name] = {**role_counts, "all": sum(role_counts.values())}

    ranges: dict[str, dict[str, float | int]] = {}
    for role in ("novel", "revisit", "all"):
        values = [row["geodesic_m"] for row in rows
                  if role == "all" or row["role"] == role]
        ranges[role] = {
            "n": len(values),
            "minimum_m": min(values),
            "median_m": statistics.median(values),
            "maximum_m": max(values),
        }

    outside = [row for row in rows if row["bin"] is None]
    long_bins_populated = (
        counts["20_to_30_m"]["all"] > 0
        and counts["30_to_50_m"]["all"] > 0
    )
    return {
        "schema_version": "role_pair_length_constructibility_audit_v1_20260829",
        "verified": True,
        "scope": "sealed role-pair construction only; no navigation outcomes",
        "manifest": str(path.resolve()),
        "manifest_sha256": digest,
        "manifest_schema": payload["schema_version"],
        "histories": len(episodes),
        "scene_clusters": len({row["scene"] for row in rows}),
        "queries": len(rows),
        "contract_query_geodesic_band_m": [
            float(contract["minimum_query_geodesic_m"]),
            float(contract["maximum_query_geodesic_m"]),
        ],
        "requested_bins": counts,
        "distance_ranges": ranges,
        "outside_requested_bins": len(outside),
        "long_bins_populated": long_bins_populated,
        "table3_constructible_from_this_population": long_bins_populated,
        "navigation_outcomes_read": False,
        "recommended_action": (
            "post-hoc Table 3 is unsupported; construct and freeze a new "
            "20-50 m benchmark before any length-stratified rollout"
            if not long_bins_populated else
            "the frozen population contains all requested bins"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_manifest(args.manifest, args.expected_sha256)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
