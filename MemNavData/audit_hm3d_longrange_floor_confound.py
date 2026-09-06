#!/usr/bin/env python3
"""Audit vertical-displacement confounding in the sealed HM3D length pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "hm3d_longrange_floor_confound_audit_v1_20260903"
EXPECTED_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
BIN_ORDER = ("0_to_20_m", "20_to_30_m", "30_to_50_m")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_xyz(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite xyz position")
    return result


def revisit_query(item: dict[str, Any]) -> dict[str, Any]:
    matches = [
        query
        for pair in item.get("pairs", [])
        for query in pair.get("queries", [])
        if query.get("analysis_role") == "revisit"
    ]
    require(len(matches) == 1,
            "each history must contain exactly one Revisit query")
    return matches[0]


def audit_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == 48,
            "sealed length manifest must contain 48 histories")
    rows: list[dict[str, Any]] = []
    for item in episodes:
        query = revisit_query(item)
        start = finite_xyz(
            item["online_a_endpoint"]["floor_position"], "history endpoint")
        goal = finite_xyz(query["floor_position"], "Revisit goal")
        geodesic = float(query["geodesic_from_a_end_m"])
        require(math.isfinite(geodesic) and geodesic > 0.0,
                "Revisit geodesic must be positive and finite")
        rows.append({
            "population_index": int(item["population_index"]),
            "scene": str(item["scene"]),
            "episode": str(item["episode"]),
            "bin_name": str(item["bin_name"]),
            "geodesic_m": geodesic,
            "start_y_m": float(start[1]),
            "goal_y_m": float(goal[1]),
            "vertical_displacement_m": float(abs(start[1] - goal[1])),
        })

    require({row["bin_name"] for row in rows} == set(BIN_ORDER),
            "sealed length bin names changed")
    by_bin: dict[str, Any] = {}
    for name in BIN_ORDER:
        selected = [row for row in rows if row["bin_name"] == name]
        require(len(selected) == 16, f"{name} does not contain 16 histories")
        vertical = np.asarray([
            row["vertical_displacement_m"] for row in selected
        ], dtype=np.float64)
        geodesic = np.asarray([
            row["geodesic_m"] for row in selected
        ], dtype=np.float64)
        by_bin[name] = {
            "n": len(selected),
            "geodesic_min_m": float(np.min(geodesic)),
            "geodesic_max_m": float(np.max(geodesic)),
            "vertical_min_m": float(np.min(vertical)),
            "vertical_median_m": float(np.median(vertical)),
            "vertical_max_m": float(np.max(vertical)),
            "vertical_at_least_0p5m_count": int(np.sum(vertical >= 0.5)),
            "vertical_at_least_1m_count": int(np.sum(vertical >= 1.0)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": (
            "construction-only audit; no navigation outcome is consumed"),
        "history_count": len(rows),
        "bin_order": list(BIN_ORDER),
        "by_bin": by_bin,
        "rows": rows,
        "interpretation": (
            "geodesic range is confounded with vertical displacement; "
            "future evaluation must stratify same-floor and multi-floor routes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.out.exists(), "audit output already exists")
    digest = sha256(arguments.manifest)
    require(digest == EXPECTED_MANIFEST_SHA256,
            "sealed length manifest SHA-256 changed")
    payload = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    result = audit_manifest(payload)
    result["manifest_sha256"] = digest
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    arguments.out.write_bytes(encoded)
    arguments.out.with_name(arguments.out.name + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + f"  {arguments.out.name}\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "out": str(arguments.out),
        "by_bin": result["by_bin"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
