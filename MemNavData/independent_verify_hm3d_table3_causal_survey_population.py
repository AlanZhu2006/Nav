#!/usr/bin/env python3
"""Independently authorize the sealed causal-survey Table-III population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_hm3d_table3_length_role_pairs import audit


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "causal-survey verification exists")
    receipt_path = args.population_root / "population_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(
        receipt.get("schema_version")
        == "hm3d_table3_causal_survey_population_v1_20260830"
        and receipt.get("history_source")
        == "controlled_causal_rgb_geodesic_survey"
        and receipt["query_policy_outcomes_read"] is False
        and receipt["formal_policy_evaluation_authorized"] is False
        and receipt["fallback_completion_allowed"] is False,
        "causal-survey population bypassed independent authorization",
    )
    benchmark = audit(args.population_root / "role_pairs")
    require(
        benchmark["ok"] is True
        and benchmark["query_policy_outcomes_read"] is False
        and benchmark["online_history"]
        == "controlled_causal_rgb_geodesic_survey",
        "independent causal-survey benchmark audit failed",
    )
    require(benchmark["manifest_sha256"]
            == receipt["benchmark_manifest_sha256"],
            "population/benchmark binding changed")
    result = {
        "schema_version": "hm3d_table3_causal_survey_population_verification_v1_20260830",
        "verified": True,
        "population_receipt_sha256": sha256(receipt_path),
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "histories_by_bin": benchmark["histories_by_bin"],
        "scene_clusters_by_bin": benchmark["scene_clusters_by_bin"],
        "history_source": benchmark["online_history"],
        "query_policy_outcomes_read": False,
        "formal_policy_evaluation_authorized": True,
        "fallback_completion_allowed": False,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )


if __name__ == "__main__":
    main()
