#!/usr/bin/env python3
"""Independent raw-receipt verifier for the held-out evidence ladder audit."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def finite_pose(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 9:
        return False
    try:
        return all(
            not isinstance(item, bool) and math.isfinite(float(item))
            for item in value)
    except (TypeError, ValueError, OverflowError):
        return False


def role(path: Path) -> str:
    if path.name.endswith("_novel_plans.json"):
        return "novel"
    if path.name.endswith("_revisit_plans.json"):
        return "revisit"
    raise RuntimeError(f"unknown role receipt: {path}")


def recount(run_root: Path, protocol: str, arm: str) -> dict[str, Any]:
    pattern = str(run_root / "evaluation" / protocol / "*" / arm / "*_plans.json")
    files = [Path(value) for value in sorted(glob.glob(pattern))]
    rows = []
    for path in files:
        current_role = role(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        plans = payload.get("query_leg")
        if not isinstance(plans, list):
            raise RuntimeError(f"{path}: missing plans")
        uncached = [
            item for item in plans
            if isinstance(item, dict)
            and item.get("certified_relocalization_cached") is False
            and isinstance(item.get("certified_relocalization_certificate"), dict)
        ]
        if len(uncached) != 1:
            raise RuntimeError(f"{path}: uncached decision count changed")
        item = uncached[0]
        attempts = item.get("certified_relocalization_proposal_attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError(f"{path}: proposal attempts missing")
        precheck = any(
            isinstance(attempt, dict) and attempt.get("precheck_passed") is True
            for attempt in attempts)
        pnp = item.get("certified_relocalization_pnp")
        if not isinstance(pnp, dict):
            raise RuntimeError(f"{path}: PnP receipt missing")
        pose = bool(
            precheck and pnp.get("status") == "ok"
            and finite_pose(pnp.get("pose9")))
        full = item.get("certified_relocalization_accepted") is True
        atomic = item["certified_relocalization_certificate"].get("accepted") is True
        if full != atomic or (full and not pose):
            raise RuntimeError(f"{path}: non-monotone certificate receipt")
        rows.append({
            "query_id": f"{path.parents[1].name}/{path.name}",
            "role": current_role,
            "fundamental_precheck": precheck,
            "precheck_plus_pnp_pose": pose,
            "full_certificate": full,
            "sha256": digest(path),
        })
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("query population is duplicated")
    by_role = {}
    for name in ("novel", "revisit"):
        selected = [row for row in rows if row["role"] == name]
        by_role[name] = {
            "queries": len(selected),
            "fundamental_precheck": sum(
                row["fundamental_precheck"] for row in selected),
            "precheck_plus_pnp_pose": sum(
                row["precheck_plus_pnp_pose"] for row in selected),
            "full_certificate": sum(
                row["full_certificate"] for row in selected),
        }
    return {
        "queries": len(rows),
        "by_role": by_role,
        "query_receipt_sha256_sequence": hashlib.sha256(
            "\n".join(row["sha256"] for row in rows).encode()).hexdigest(),
    }


def verify(run_root: Path, protocol: str, arm: str,
           report: dict[str, Any]) -> dict[str, Any]:
    observed = recount(run_root, protocol, arm)
    if observed["queries"] != int(report.get("queries", -1)):
        raise RuntimeError("query count differs")
    if observed["query_receipt_sha256_sequence"] != report.get(
            "query_receipt_sha256_sequence"):
        raise RuntimeError("ordered raw receipt hashes differ")
    for role_name, values in observed["by_role"].items():
        published = report["by_role"][role_name]
        for field, value in values.items():
            if int(published.get(field, -1)) != value:
                raise RuntimeError(f"{role_name}.{field} differs")
    contract = report.get("interpretation_contract", {})
    if (contract.get("navigation_outcomes_read") is not False
            or contract.get("threshold_or_method_selection_authorized") is not False):
        raise RuntimeError("post-hoc interpretation contract changed")
    return {
        "schema_version": (
            "independent_closed_loop_certificate_evidence_ladder_v1_20260826"),
        "verified": True,
        "status": "independent_raw_receipt_recount_passed_not_sr_arm",
        **observed,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", default="natural_direction")
    parser.add_argument("--arm", default="mono_cec")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify(args.run_root.resolve(), args.protocol, args.arm, report)
    result["inputs"] = {
        "run_root": str(args.run_root.resolve()),
        "report_sha256": digest(args.report),
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
