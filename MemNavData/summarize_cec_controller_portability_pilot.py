#!/usr/bin/env python3
"""Aggregate the frozen all-CEC controller latency/failure pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from audit_cec_controller_portability_pilot import LATENCY_FIELDS, percentile
from audit_cec_controller_portability_smoke import CONTROLLERS, require


SCHEMA = "cec_controller_portability_pilot_summary_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_indices(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    require(values and len(values) == len(set(values)),
            "expected indices must be unique")
    return values


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def summarize(root: Path, expected_indices: list[int]) -> dict[str, Any]:
    controller_rows = {
        controller: {
            "query_rows": [],
            "latencies": {field: [] for field in LATENCY_FIELDS},
            "takeover": 0,
            "fallback": 0,
        }
        for controller in CONTROLLERS
    }
    histories = []
    for index in expected_indices:
        matches = sorted((root / "evaluation").glob(f"{index:03d}_*"))
        require(len(matches) == 1, f"history {index}: output missing/ambiguous")
        history = matches[0]
        completion_path = history / "completion.json"
        audit_path = history / "independent_audit.json"
        require(completion_path.is_file() and audit_path.is_file(),
                f"history {index}: completion seal missing")
        completion = json.loads(completion_path.read_text())
        audit = json.loads(audit_path.read_text())
        require(completion.get("complete") is True
                and completion.get("history_index") == index,
                f"history {index}: completion invalid")
        require(audit.get("verified") is True,
                f"history {index}: independent audit failed")
        require(completion.get("independent_audit_sha256")
                == sha256_file(audit_path),
                f"history {index}: audit digest changed")
        require({run["controller"] for run in audit["runs"]}
                == set(CONTROLLERS),
                f"history {index}: controller matrix incomplete")
        histories.append({
            "index": index,
            "scene": completion["scene"],
            "episode": completion["episode"],
            "completion_sha256": sha256_file(completion_path),
            "audit_sha256": sha256_file(audit_path),
        })
        for controller in CONTROLLERS:
            result = history / controller / "result"
            with (result / "metric.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2
                    and {row["analysis_role"] for row in rows}
                    == {"novel", "revisit"},
                    f"history {index}/{controller}: metric rows changed")
            controller_rows[controller]["query_rows"].extend(rows)
            for plan_path in sorted(result.glob("*_plans.json")):
                payload = json.loads(plan_path.read_text())
                for plan in payload["query_leg"]:
                    if plan["cec_takeover"]:
                        controller_rows[controller]["takeover"] += 1
                    else:
                        controller_rows[controller]["fallback"] += 1
                    for field in LATENCY_FIELDS:
                        value = plan.get(field)
                        if value is None:
                            continue
                        require(isinstance(value, (int, float))
                                and math.isfinite(float(value))
                                and float(value) >= 0.0,
                                f"history {index}/{controller}: bad latency")
                        controller_rows[controller]["latencies"][field].append(
                            float(value))

    controllers = {}
    for controller, payload in controller_rows.items():
        rows = payload["query_rows"]
        roles = {
            role: [row for row in rows if row["analysis_role"] == role]
            for role in ("novel", "revisit")
        }
        controllers[controller] = {
            "queries": len(rows),
            "descriptive_successes_by_role": {
                role: sum(int(row["reached"]) for row in role_rows)
                for role, role_rows in roles.items()
            },
            "queries_by_role": {
                role: len(role_rows) for role, role_rows in roles.items()
            },
            "takeover_decisions": payload["takeover"],
            "fallback_decisions": payload["fallback"],
            "latency_ms": {
                field: describe(values)
                for field, values in payload["latencies"].items()
            },
        }
    return {
        "schema": SCHEMA,
        "scope": "latency_failure_pilot_not_sr_confirmation",
        "complete": True,
        "expected_history_indices": expected_indices,
        "histories": histories,
        "history_count": len(histories),
        "scene_clusters": len({row["scene"] for row in histories}),
        "controllers": controllers,
        "interpretation": (
            "Descriptive success counts are pilot diagnostics only. This "
            "summary does not provide paired statistical performance evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-indices", required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    payload = summarize(cli.root.resolve(), parse_indices(cli.expected_indices))
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "complete": True,
        "histories": payload["history_count"],
        "scene_clusters": payload["scene_clusters"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
