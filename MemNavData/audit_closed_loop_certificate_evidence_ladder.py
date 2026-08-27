#!/usr/bin/env python3
"""Recount precheck/PnP/certificate authorization from frozen plan receipts.

The audit never reads navigation outcomes.  It asks, on a completed mixed-role
run, how many query episodes would be authorized by (1) the correspondence
precheck, (2) the precheck plus an available PnP bearing, and (3) the complete
frozen certificate.  Role labels are used only after execution for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROLES = ("novel", "revisit")
SCHEMA_VERSION = "closed_loop_certificate_evidence_ladder_audit_v1_20260826"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def finite_pose9(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 9
        and all(finite(item) for item in value)
    )


def role_from_path(path: Path) -> str:
    for role in ROLES:
        if path.name.endswith(f"_{role}_plans.json"):
            return role
    raise RuntimeError(f"cannot infer analysis role from {path}")


def decision(plan: Mapping[str, Any]) -> dict[str, Any]:
    attempts = plan.get("certified_relocalization_proposal_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("uncached certificate decision lacks proposal attempts")
    precheck = any(
        isinstance(attempt, dict) and attempt.get("precheck_passed") is True
        for attempt in attempts
    )
    pnp = plan.get("certified_relocalization_pnp")
    if not isinstance(pnp, dict):
        raise RuntimeError("uncached certificate decision lacks PnP receipt")
    pose = bool(
        precheck
        and pnp.get("status") == "ok"
        and finite_pose9(pnp.get("pose9"))
    )
    accepted = plan.get("certified_relocalization_accepted") is True
    certificate = plan.get("certified_relocalization_certificate")
    if not isinstance(certificate, dict):
        raise RuntimeError("uncached decision lacks atomic certificate")
    if accepted != (certificate.get("accepted") is True):
        raise RuntimeError("endpoint and atomic certificate acceptance differ")
    if accepted and not pose:
        raise RuntimeError("full certificate accepted without a PnP bearing")
    return {
        "fundamental_precheck": precheck,
        "precheck_plus_pnp_pose": pose,
        "full_certificate": accepted,
        "reason": str(plan.get("certified_relocalization_reason")),
        "pnp_status": str(pnp.get("status")),
        "selected_anchor": plan.get("router_selected_anchor"),
        "proposal_attempts": len(attempts),
    }


def query_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plans = payload.get("query_leg")
    if not isinstance(plans, list) or not plans:
        raise RuntimeError(f"{path}: query_leg plans missing")
    uncached = [
        plan for plan in plans
        if isinstance(plan, dict)
        and plan.get("certified_relocalization_cached") is False
        and isinstance(plan.get("certified_relocalization_certificate"), dict)
    ]
    if len(uncached) != 1:
        raise RuntimeError(
            f"{path}: expected one uncached certificate decision, got {len(uncached)}")
    first = decision(uncached[0])
    evaluated = [
        plan for plan in plans
        if isinstance(plan, dict)
        and isinstance(plan.get("certified_relocalization_certificate"), dict)
    ]
    for repeated in evaluated:
        value = decision(repeated)
        for field in (
            "fundamental_precheck", "precheck_plus_pnp_pose",
            "full_certificate", "reason", "pnp_status", "selected_anchor",
        ):
            if value[field] != first[field]:
                raise RuntimeError(f"{path}: cached decision drifted at {field}")
    directory = path.parents[1].name
    return {
        "query_id": f"{directory}/{path.name}",
        "role": role_from_path(path),
        "plans": len(plans),
        "certificate_evaluated_plans": len(evaluated),
        "plan_receipt_sha256": sha256_file(path),
        **first,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records or len({row["query_id"] for row in records}) != len(records):
        raise RuntimeError("query receipt population is empty or duplicated")
    stages = (
        "fundamental_precheck",
        "precheck_plus_pnp_pose",
        "full_certificate",
    )
    by_role = {}
    for role in ROLES:
        rows = [row for row in records if row["role"] == role]
        if not rows:
            raise RuntimeError(f"mixed-role population lacks {role}")
        by_role[role] = {
            "queries": len(rows),
            **{stage: sum(row[stage] for row in rows) for stage in stages},
            "precheck_pose_rejected_by_full_certificate": sum(
                row["precheck_plus_pnp_pose"] and not row["full_certificate"]
                for row in rows),
            "first_decision_reason_counts": {
                reason: sum(row["reason"] == reason for row in rows)
                for reason in sorted({row["reason"] for row in rows})
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "posthoc_authorization_audit_not_closed_loop_arm",
        "queries": len(records),
        "query_receipt_sha256_sequence": hashlib.sha256(
            "\n".join(row["plan_receipt_sha256"] for row in records).encode()
        ).hexdigest(),
        "stages": list(stages),
        "by_role": by_role,
        "records": records,
        "interpretation_contract": {
            "navigation_outcomes_read": False,
            "role_labels_consumed_by_runtime": False,
            "threshold_or_method_selection_authorized": False,
            "warning": (
                "This consumed held-out receipt audit establishes authorization "
                "separation only; it is not a fourth closed-loop controller arm."
            ),
        },
    }


def collect(run_root: Path, protocol: str, arm: str) -> list[dict[str, Any]]:
    root = run_root / "evaluation" / protocol
    files = sorted(
        path
        for role in ROLES
        for path in root.glob(f"*/{arm}/*_{role}_plans.json")
    )
    return [query_record(path) for path in files]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(collect(args.run_root.resolve(), args.protocol, args.arm))
    if args.expected_queries is not None and result["queries"] != args.expected_queries:
        raise RuntimeError(
            f"query population differs: {result['queries']} != {args.expected_queries}")
    result["inputs"] = {
        "run_root": str(args.run_root.resolve()),
        "protocol": args.protocol,
        "arm": args.arm,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
