#!/usr/bin/env python3
"""Compile the Final14 retrieval/witness/authority mechanism audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA = "final14_cec_mechanism_audit_v1_20260830"
LEDGER_SCHEMA = "final14_cec_mechanism_ledger_v1_20260830"
ROLES = ("novel", "revisit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
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


def count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(row[field]) for row in rows)


def compile_audit(
    ledger: dict[str, Any],
    authority: dict[str, Any],
    authority_verification: dict[str, Any],
    mono: dict[str, Any],
    mono_verification: dict[str, Any],
    *,
    source_hashes: Mapping[str, str],
    source_uris: Mapping[str, str],
) -> dict[str, Any]:
    require(ledger.get("schema_version") == LEDGER_SCHEMA,
            "mechanism ledger schema changed")
    require(authority_verification.get("verified") is True,
            "authority source is not independently verified")
    require(mono_verification.get("verified") is True
            and mono_verification.get("authorized") is True,
            "mono factorial source is not independently verified")
    manifest_sha = ledger["benchmark_manifest_sha256"]
    require(authority["benchmark_manifest_sha256"] == manifest_sha
            and mono["benchmark_manifest_sha256"] == manifest_sha,
            "source populations differ")
    require(authority.get("runtime_role_visibility") == "none"
            and mono.get("runtime_role_visibility") == "none",
            "runtime role label was visible")

    records = ledger["records"]
    require(len(records) == 42, "expected 42 mechanism queries")
    by_role = {
        role: [row for row in records if row["analysis_role"] == role]
        for role in ROLES
    }
    require(all(len(rows) == 21 for rows in by_role.values()),
            "role balance changed")

    proposal = {}
    for role, rows in by_role.items():
        proposal[role] = {
            "n": len(rows),
            "dino_top1_supported": count(rows, "dino_top1_supported"),
            "dino_top8_contains_supported_anchor": count(
                rows, "dino_top8_contains_supported_anchor"
            ),
            "geometry_selected_supported": count(
                rows, "geometry_selected_supported"
            ),
            "maximum_constructed_history_covis": max(
                float(row["max_history_covis"]) for row in rows
            ),
        }

    authority_rows = {}
    for role in ROLES:
        source = authority["results"][role]
        authority_rows[role] = {
            "n": 21,
            "finite_pnp_authorized": int(
                source["arms"]["mono_unthresholded_witness"]["accepted_queries"]
            ),
            "strict_cec_authorized": int(
                source["arms"]["mono_cec"]["accepted_queries"]
            ),
            "finite_pnp_successes": int(
                source["arms"]["mono_unthresholded_witness"]["successes"]
            ),
            "strict_cec_successes": int(
                source["arms"]["mono_cec"]["successes"]
            ),
        }
        require(authority_rows[role]["finite_pnp_authorized"] == count(
            by_role[role], "finite_pnp_witness_available"
        ), f"finite-PnP ledger/summary mismatch: {role}")
        require(authority_rows[role]["strict_cec_authorized"] == count(
            by_role[role], "strict_certificate_accept"
        ), f"strict CEC ledger/summary mismatch: {role}")

    raw_rows = {}
    for role in ROLES:
        raw = mono["results"][role]["arms"]["mono_raw_fixed"]
        raw_rows[role] = {
            "n": int(raw["n"]),
            "authorized": int(raw["n"]),
            "successes": int(raw["successes"]),
        }
        require(raw_rows[role]["n"] == 21, f"raw role count changed: {role}")

    strict_vs_witness = authority["results"]["all"][
        "strict_cec_minus_unthresholded_witness"
    ]
    strict_vs_raw = mono["results"]["all"]["contrasts"][
        "mono_cec_minus_mono_raw_fixed"
    ]
    rank_histogram = Counter(
        int(row["geometry_selected_dino_rank"])
        for row in by_role["revisit"]
    )

    return {
        "schema_version": SCHEMA,
        "status": "complete_posthoc_mechanism_audit",
        "scope": "consumed_final14_attribution_not_fresh_confirmation",
        "population": {
            "histories": 21,
            "scene_clusters": 10,
            "queries": 42,
            "novel_queries": 21,
            "revisit_queries": 21,
            "runtime_role_visibility": "none",
            "query_controller_depth": "monocular_sidecar",
            "goal_a_history_boundary": (
                "original metric-NavDP Goal-A RGB replay; this is not a "
                "full-mono Goal-A confirmation"
            ),
            "benchmark_manifest_sha256": manifest_sha,
        },
        "diagnostic_contract": {
            "supported_anchor_threshold_covis": 0.5,
            "threshold_is_runtime_gate": False,
            "threshold_selected_after_navigation_outcomes": False,
            "novel_construction_max_covis_exclusive": 0.1,
            "revisit_construction_max_covis_minimum": 0.55,
            "support_metric_role": (
                "address-coverage diagnostic only; not localization accuracy"
            ),
        },
        "proposal_diagnostics": proposal,
        "revisit_selected_dino_rank_histogram": {
            str(rank): rank_histogram.get(rank, 0) for rank in range(1, 9)
        },
        "operational_ladder": {
            "raw_dino_always_on": {
                "proposal_matched_to_strict_cec": False,
                "interpretation": (
                    "same population and monocular controller depth, but a "
                    "different direct-top1 candidate contract"
                ),
                "by_role": raw_rows,
            },
            "proposal_matched_finite_pnp": {
                "proposal_matched_to_strict_cec": True,
                "authority_policy": "finite PnP pose available",
                "by_role": {
                    role: {
                        "n": authority_rows[role]["n"],
                        "authorized": authority_rows[role]["finite_pnp_authorized"],
                        "successes": authority_rows[role]["finite_pnp_successes"],
                    } for role in ROLES
                },
            },
            "strict_cec": {
                "proposal_matched_to_finite_pnp": True,
                "authority_policy": "frozen operational certificate",
                "by_role": {
                    role: {
                        "n": authority_rows[role]["n"],
                        "authorized": authority_rows[role]["strict_cec_authorized"],
                        "successes": authority_rows[role]["strict_cec_successes"],
                    } for role in ROLES
                },
            },
        },
        "paired_closed_loop_contrasts": {
            "strict_cec_minus_proposal_matched_finite_pnp": {
                "gains": int(strict_vs_witness["gains"]),
                "losses": int(strict_vs_witness["losses"]),
                "risk_difference_pp": float(
                    strict_vs_witness["risk_difference_pp"]
                ),
                "exact_mcnemar_two_sided_p": float(
                    strict_vs_witness["exact_mcnemar_two_sided_p"]
                ),
            },
            "strict_cec_minus_raw_dino_always_on": {
                "gains": int(strict_vs_raw["gains"]),
                "losses": int(strict_vs_raw["losses"]),
                "risk_difference_pp": float(strict_vs_raw["risk_difference_pp"]),
                "exact_mcnemar_two_sided_p": float(
                    strict_vs_raw["exact_mcnemar_two_sided_p"]
                ),
            },
        },
        "conclusions": {
            "revisit_addressability": (
                "DINO top-8 contains a >=0.5-covis historical address in "
                "21/21 supported Revisit queries; top-1 does so in 19/21."
            ),
            "finite_pose_is_not_authority": (
                "A finite PnP pose authorizes 18/21 constructed unsupported "
                "Novel queries, while strict CEC authorizes 2/21."
            ),
            "revisit_utility_preserved": (
                "Finite-PnP and strict CEC both authorize 21/21 and succeed "
                "on 20/21 Revisit queries."
            ),
            "closed_loop_strength_boundary": (
                "Strict CEC is 28/42 versus 25/42 for proposal-matched finite "
                "PnP (p=0.375) and 23/42 for raw DINO (p=0.0625); these are "
                "mechanism trends, not confirmed superiority claims."
            ),
        },
        "source_uris": dict(source_uris),
        "source_sha256": dict(source_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--authority-summary", type=Path, required=True)
    parser.add_argument("--authority-verification", type=Path, required=True)
    parser.add_argument("--mono-summary", type=Path, required=True)
    parser.add_argument("--mono-verification", type=Path, required=True)
    parser.add_argument("--ledger-uri", required=True)
    parser.add_argument("--authority-summary-uri", required=True)
    parser.add_argument("--mono-summary-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "ledger": args.ledger,
        "authority_summary": args.authority_summary,
        "authority_verification": args.authority_verification,
        "mono_summary": args.mono_summary,
        "mono_verification": args.mono_verification,
    }
    objects = {name: read_object(path) for name, path in inputs.items()}
    payload = compile_audit(
        objects["ledger"], objects["authority_summary"],
        objects["authority_verification"], objects["mono_summary"],
        objects["mono_verification"],
        source_hashes={name: sha256_file(path) for name, path in inputs.items()},
        source_uris={
            "ledger": args.ledger_uri,
            "authority_summary": args.authority_summary_uri,
            "mono_summary": args.mono_summary_uri,
        },
    )
    atomic_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
