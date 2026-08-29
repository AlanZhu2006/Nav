#!/usr/bin/env python3
"""Independently verify the compact Final14 CEC mechanism audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


AUDIT_SCHEMA = "final14_cec_mechanism_audit_v1_20260830"
LEDGER_SCHEMA = "final14_cec_mechanism_ledger_v1_20260830"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--authority-summary", type=Path, required=True)
    parser.add_argument("--mono-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit, ledger = load(args.audit), load(args.ledger)
    authority, mono = load(args.authority_summary), load(args.mono_summary)
    require(audit.get("schema_version") == AUDIT_SCHEMA, "audit schema changed")
    require(ledger.get("schema_version") == LEDGER_SCHEMA, "ledger schema changed")
    require(audit["source_sha256"]["ledger"] == digest(args.ledger),
            "ledger digest mismatch")
    require(audit["source_sha256"]["authority_summary"]
            == digest(args.authority_summary), "authority digest mismatch")
    require(audit["source_sha256"]["mono_summary"]
            == digest(args.mono_summary), "mono digest mismatch")

    records = ledger["records"]
    roles = {
        role: [row for row in records if row["analysis_role"] == role]
        for role in ("novel", "revisit")
    }
    recount: dict[str, Any] = {}
    for role, rows in roles.items():
        require(len(rows) == 21, f"role count changed: {role}")
        recount[role] = {
            "dino_top1_supported": sum(row["dino_top1_supported"] for row in rows),
            "dino_top8_contains_supported_anchor": sum(
                row["dino_top8_contains_supported_anchor"] for row in rows
            ),
            "geometry_selected_supported": sum(
                row["geometry_selected_supported"] for row in rows
            ),
            "finite_pnp_authorized": sum(
                row["finite_pnp_witness_available"] for row in rows
            ),
            "strict_cec_authorized": sum(
                row["strict_certificate_accept"] for row in rows
            ),
            "finite_pnp_successes": sum(
                row["closed_loop"]["mono_unthresholded_witness"]["success"]
                for row in rows
            ),
            "strict_cec_successes": sum(
                row["closed_loop"]["mono_cec"]["success"] for row in rows
            ),
            "raw_dino_successes": int(
                mono["results"][role]["arms"]["mono_raw_fixed"]["successes"]
            ),
        }
        proposal = audit["proposal_diagnostics"][role]
        ladder = audit["operational_ladder"]
        require(recount[role]["dino_top1_supported"]
                == proposal["dino_top1_supported"], "top1 recount mismatch")
        require(recount[role]["dino_top8_contains_supported_anchor"]
                == proposal["dino_top8_contains_supported_anchor"],
                "top8 recount mismatch")
        require(recount[role]["geometry_selected_supported"]
                == proposal["geometry_selected_supported"],
                "selected-support recount mismatch")
        require(recount[role]["finite_pnp_authorized"]
                == ladder["proposal_matched_finite_pnp"]["by_role"][role]["authorized"],
                "finite-PnP authorization mismatch")
        require(recount[role]["strict_cec_authorized"]
                == ladder["strict_cec"]["by_role"][role]["authorized"],
                "strict authorization mismatch")
        require(recount[role]["raw_dino_successes"]
                == ladder["raw_dino_always_on"]["by_role"][role]["successes"],
                "raw success mismatch")
        require(recount[role]["finite_pnp_successes"]
                == authority["results"][role]["arms"]
                ["mono_unthresholded_witness"]["successes"],
                "finite-PnP success mismatch")

    ranks = Counter(
        str(row["geometry_selected_dino_rank"])
        for row in roles["revisit"]
    )
    require(dict(sorted(ranks.items())) == {
        key: value for key, value in
        audit["revisit_selected_dino_rank_histogram"].items() if value
    }, "selected-rank histogram mismatch")
    require(recount["novel"]["finite_pnp_authorized"] == 18
            and recount["novel"]["strict_cec_authorized"] == 2,
            "Novel authority result changed")
    require(recount["revisit"]["finite_pnp_authorized"] == 21
            and recount["revisit"]["strict_cec_authorized"] == 21,
            "Revisit authority result changed")

    payload = {
        "schema_version": "final14_cec_mechanism_independent_verification_v1_20260830",
        "verified": True,
        "audit_sha256": digest(args.audit),
        "ledger_sha256": digest(args.ledger),
        "authority_summary_sha256": digest(args.authority_summary),
        "mono_summary_sha256": digest(args.mono_summary),
        "queries_recomputed": len(records),
        "runtime_role_visibility": "none",
        "recount": recount,
        "navigation_threshold_selected_or_changed": False,
        "fresh_confirmation": False,
    }
    atomic_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
