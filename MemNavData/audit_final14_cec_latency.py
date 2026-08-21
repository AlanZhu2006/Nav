#!/usr/bin/env python3
"""Recompute CEC first-query and cache-hit latency from raw plan files.

The runtime deliberately carries ``uncached_relocalization_ms`` forward in a
cached result so the original localization receipt remains inspectable.  A
latency audit must consequently filter by ``certified_relocalization_cached``;
collecting every non-null value repeats the first-query latency once per
navigation replan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOLS = ("natural_direction", "hard_support")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(plan: dict[str, Any], key: str, source: Path) -> float:
    value = plan.get(key)
    require(value is not None, f"{source}: missing {key}")
    number = float(value)
    require(math.isfinite(number) and number >= 0.0,
            f"{source}: invalid {key}={value!r}")
    return number


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def query_role(path: Path) -> str:
    if path.name.endswith("_novel_plans.json"):
        return "novel"
    if path.name.endswith("_revisit_plans.json"):
        return "revisit"
    raise RuntimeError(f"cannot infer query role from {path}")


def audit_protocol(run_root: Path, protocol: str) -> dict[str, Any]:
    plan_files = sorted(
        (run_root / "evaluation" / protocol).glob(
            "*/certified/*_plans.json"
        )
    )
    require(plan_files, f"{protocol}: no certified plan files")

    first_query_ms: list[float] = []
    cached_update_ms: list[float] = []
    legacy_carried_ms: list[float] = []
    records: list[dict[str, Any]] = []
    for path in plan_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        requests = [
            plan for plan in payload["query_leg"]
            if plan.get("certified_relocalization_cached") in (True, False)
        ]
        require(requests, f"{path}: no certificate requests")
        first = [
            plan for plan in requests
            if plan["certified_relocalization_cached"] is False
        ]
        cached = [
            plan for plan in requests
            if plan["certified_relocalization_cached"] is True
        ]
        require(len(first) == 1, f"{path}: expected one first query")
        require(requests[0] is first[0], f"{path}: first request was cached")
        require(all(
            plan["certified_relocalization_cached"] is True
            for plan in requests[1:]
        ), f"{path}: invalid cache lifecycle")

        first_ms = finite(
            first[0], "certified_relocalization_uncached_ms", path
        )
        first_query_ms.append(first_ms)
        cached_update_ms.extend(
            finite(plan, "certified_relocalization_ms", path)
            for plan in cached
        )
        legacy_carried_ms.extend(
            finite(plan, "certified_relocalization_uncached_ms", path)
            for plan in requests
        )
        records.append({
            "path": str(path),
            "role": query_role(path),
            "accepted": first[0].get(
                "certified_relocalization_accepted"
            ) is True,
            "selected_anchor": first[0].get("router_selected_anchor"),
            "first_query_ms": first_ms,
            "request_count": len(requests),
            "cached_update_count": len(cached),
        })

    anchor_pairs = [
        (float(record["selected_anchor"]), record["first_query_ms"])
        for record in records if record["selected_anchor"] is not None
    ]
    anchor_latency_pearson = None
    if len(anchor_pairs) > 1:
        anchor_latency_pearson = float(
            np.corrcoef(np.asarray(anchor_pairs, dtype=np.float64).T)[0, 1]
        )
    return {
        "plan_files": len(plan_files),
        "cache_lifecycle_valid": True,
        "first_query_uncached_ms": distribution(first_query_ms),
        "cached_update_ms": distribution(cached_update_ms),
        "legacy_carried_uncached_ms": distribution(legacy_carried_ms),
        "legacy_to_true_sample_ratio": (
            len(legacy_carried_ms) / len(first_query_ms)
        ),
        "first_query_by_role": {
            role: distribution([
                record["first_query_ms"] for record in records
                if record["role"] == role
            ])
            for role in ("novel", "revisit")
        },
        "first_query_by_accept": {
            str(accepted).lower(): distribution([
                record["first_query_ms"] for record in records
                if record["accepted"] is accepted
            ])
            for accepted in (True, False)
        },
        "selected_anchor_latency_pearson": anchor_latency_pearson,
        "slowest_first_queries": sorted(
            records, key=lambda record: record["first_query_ms"],
            reverse=True,
        )[:8],
    }


def audit(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    require((run_root / "evaluation").is_dir(),
            f"missing evaluation directory under {run_root}")
    return {
        "schema_version": "final14_cec_latency_audit_v1_20260818",
        "source_run_root": str(run_root),
        "scientific_outcomes_modified": False,
        "protocols": {
            protocol: audit_protocol(run_root, protocol)
            for protocol in PROTOCOLS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
