#!/usr/bin/env python3
"""Fail-closed pre-query audit for the HM3D underpowered amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_fullmono_lifelong_underpowered_amendment_v1_20260828"
QUERY_OUTPUTS = (
    "shared_c_collection",
    "shared_c_population",
    "shared_c_evaluation",
    "shared_c_aggregate",
    "shared_c_independent_verification.json",
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def audit(
    *, protocol_path: Path, run_root: Path, require_pristine: bool = False,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    require(protocol.get("schema_version") == SCHEMA,
            "underpowered amendment schema changed")
    frozen = protocol.get("freeze_boundary", {})
    require(frozen.get("factual_C_outcomes_read") is False,
            "factual-C outcome was read before amendment freeze")
    require(frozen.get("B2_outcomes_read") is False,
            "B2 outcome was read before amendment freeze")
    require(protocol.get("amends", {}).get(
        "original_powered_confirmation_claim_permanently_withheld") is True,
        "underpowered interpretation guard is missing")

    source = protocol.get("source_population", {})
    require(str(run_root.resolve()) == source.get("run_root"),
            "run root differs from frozen amendment")
    population_path = run_root / str(source.get("relative_path"))
    population = load_json(population_path)
    require(sha256_file(population_path) == source.get("sha256"),
            "source population SHA-256 changed")
    seal_root = population_path.parent
    require((seal_root / "SEALED").is_file(), "source population is not sealed")
    receipt = seal_root / "population.json.sha256"
    require(receipt.is_file(), "population SHA receipt is missing")
    fields = receipt.read_text().strip().split()
    require(len(fields) == 2 and fields[0] == source.get("sha256")
            and fields[1] == "population.json",
            "population SHA receipt changed")
    rows = population.get("accepted")
    require(isinstance(rows, list), "source population accepted rows missing")
    require(len(rows) == int(source.get("histories", -1)),
            "source population count changed")
    scenes = {str(row["scene"]) for row in rows}
    require(len(scenes) == int(source.get("scene_clusters", -1)),
            "source population scene count changed")

    verification_path = run_root / str(
        source.get("independent_population_verification"))
    verification = load_json(verification_path)
    require(verification.get("verified") is True,
            "independent source-population verifier did not pass")
    require(verification.get("population_sha256") == source.get("sha256"),
            "independent verifier is bound to another population")
    require(int(verification.get("supported_population", -1)) == len(rows),
            "independent verifier population count changed")
    require(int(verification.get("scene_clusters", -1)) == len(scenes),
            "independent verifier scene count changed")
    require(verification.get("query_navigation_outcomes_read") is False,
            "query outcomes were read before the amendment")
    require(verification.get("factual_C_B2_C2_executed") is False,
            "a post-prefix query was already executed")
    require(verification.get("target_met") is False,
            "amendment no longer describes an underpowered population")

    if require_pristine:
        present = [name for name in QUERY_OUTPUTS if (run_root / name).exists()]
        require(not present,
                "post-freeze query output already exists: " + ", ".join(present))

    return {
        "schema_version": "hm3d_lifelong_underpowered_prequery_audit_v1_20260828",
        "verified": True,
        "population_sha256": source["sha256"],
        "histories": len(rows),
        "scene_clusters": len(scenes),
        "strong_support_histories": int(
            verification.get("strong_support_histories", -1)),
        "query_outcomes_read": False,
        "original_powered_confirmation_claim_withheld": True,
        "pristine_query_root_required": bool(require_pristine),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-pristine", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = audit(
        protocol_path=args.protocol,
        run_root=args.run_root,
        require_pristine=args.require_pristine,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
