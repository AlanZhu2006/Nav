#!/usr/bin/env python3
"""Independent receipt-level verifier for the HM3D ViNT controller pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "vint_controller_native_hm3d_verification_v1_20260828"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value)
               for value in range(min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def verify(
    run_root: Path,
    summary_path: Path,
    benchmark_manifest: Path,
) -> dict:
    summary = json.loads(summary_path.read_text())
    require(summary.get("verified") is True,
            "aggregate summary is not verified")
    require(summary.get("benchmark_manifest_sha256")
            == digest(benchmark_manifest),
            "aggregate is bound to a different benchmark")
    audit_paths = sorted((run_root / "evaluation").glob(
        "*/vint/controller_native_pair_audit.json"))
    require(len(audit_paths) == int(summary["histories"]),
            "raw audit count differs from summary")
    rows = []
    observed_hashes = {}
    identities = set()
    for path in audit_paths:
        cell = json.loads(path.read_text())
        require(cell.get("verified") is True
                and cell.get("controller") == "vint"
                and cell.get("reject_policy") == "controller_native_exact"
                and cell.get("query_count") == 2,
                f"invalid raw cell: {path}")
        identity = str(cell["scene"]), str(cell["episode"])
        require(identity not in identities, f"duplicate raw cell: {path}")
        identities.add(identity)
        rows.extend(cell["query_results"])
        observed_hashes[str(path)] = digest(path)
    require(len(rows) == int(summary["queries"]),
            "raw query count differs from summary")
    require({row["analysis_role"] for row in rows} == {"novel", "revisit"},
            "raw role set changed")
    require(len({row["scene"] for row in rows})
            == int(summary["scene_clusters"]),
            "raw scene-cluster count differs")

    def counts(group):
        gains = sum(int(row["paired_gain"]) for row in group)
        losses = sum(int(row["paired_loss"]) for row in group)
        return {
            "n": len(group),
            "native_success": sum(int(row["native_success"]) for row in group),
            "cec_success": sum(int(row["grant_success"]) for row in group),
            "paired_gain": gains,
            "paired_loss": losses,
            "mcnemar_exact_p": exact_mcnemar(gains, losses),
        }

    groups = {
        "all": rows,
        "novel": [row for row in rows if row["analysis_role"] == "novel"],
        "revisit": [row for row in rows if row["analysis_role"] == "revisit"],
    }
    recomputed = {name: counts(group) for name, group in groups.items()}
    for name, result in recomputed.items():
        reported = summary["results"][name]
        for field in (
            "n", "native_success", "cec_success", "paired_gain", "paired_loss",
        ):
            require(int(reported[field]) == int(result[field]),
                    f"{name}: aggregate {field} differs from raw receipts")
        require(math.isclose(
            float(reported["mcnemar_exact_p"]),
            float(result["mcnemar_exact_p"]), abs_tol=1e-15, rel_tol=0.0),
            f"{name}: McNemar p differs from raw receipts")
    novel_takeovers = sum(
        int(row["grant_takeover_plans"]) > 0 for row in groups["novel"])
    require(novel_takeovers
            == int(summary["safety"]["novel_takeover_queries"]),
            "Novel takeover count differs from raw receipts")
    all_reject = [row for row in rows
                  if int(row["grant_takeover_plans"]) == 0]
    require(all(row.get("exact_fallback_trace_match") is True
                for row in all_reject),
            "a raw all-reject query is not exact fallback")

    cells = summary.get("cells")
    require(isinstance(cells, list) and len(cells) == len(audit_paths),
            "summary raw-cell inventory changed")
    reported_hashes = {
        str(cell["audit_path"]): str(cell["audit_sha256"])
        for cell in cells
    }
    require(reported_hashes == observed_hashes,
            "a raw pair audit changed after aggregation")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "summary_sha256": digest(summary_path),
        "benchmark_manifest_sha256": digest(benchmark_manifest),
        "raw_histories": len(audit_paths),
        "raw_queries": len(rows),
        "scene_clusters": len({row["scene"] for row in rows}),
        "recomputed": recomputed,
        "novel_takeover_queries": novel_takeovers,
        "all_reject_exact_fallback": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_root.resolve(), args.summary.resolve(),
        args.benchmark_manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "verified": True,
        "histories": result["raw_histories"],
        "queries": result["raw_queries"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
