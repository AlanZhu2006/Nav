#!/usr/bin/env python3
"""Freeze one construction-selected route-tangent confirmation history.

Selection uses only the already frozen population manifest.  The longest
30--50 m history not consumed by route-tangent development is selected before
the old CEC plan is opened.  The plan contributes only the previously frozen
strict-certificate anchor; navigation outcome fields are never inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_MANIFEST_SHA256 = (
    "cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451"
)
SCHEMA_VERSION = "route_tangent_confirmation_freeze_v1_20260903"
CONSUMED_ROUTE_TANGENT_INDICES = (16, 40)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_strict_authorization(plan: dict[str, Any]) -> dict[str, Any]:
    require(plan.get("analysis_role_not_forwarded") is True,
            "old paired evaluation forwarded the analysis role")
    rows = plan.get("query_leg")
    require(isinstance(rows, list) and rows,
            "old paired evaluation has no query decisions")
    first = rows[0]
    require(first.get("certified_relocalization_accepted") is True,
            "construction-selected history lacks an initial CEC authorization")
    require(first.get("certified_relocalization_authority_policy")
            == "strict_certificate",
            "initial authorization did not use the strict certificate")
    require(first.get("certified_relocalization_reason")
            == "certificate_accepted",
            "initial authorization reason changed")
    anchor = first.get("anchor")
    require(isinstance(anchor, int) and anchor >= 0,
            "initial authorization has no valid anchor")
    return {
        "authorized_anchor": int(anchor),
        "authority_policy": "strict_certificate",
        "authority_reason": "certificate_accepted",
        "analysis_role_not_forwarded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    require(not arguments.out.exists(), "freeze output already exists")
    require(sha256(arguments.manifest) == FROZEN_MANIFEST_SHA256,
            "frozen population manifest changed")
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list), "population manifest has no episodes")

    # This ordering is fixed before any plan/result file is opened.
    eligible = [
        row for row in episodes
        if row.get("bin_name") == "30_to_50_m"
        and int(row.get("population_index", -1))
        not in CONSUMED_ROUTE_TANGENT_INDICES
    ]
    require(eligible, "no unconsumed 30--50 m history remains")
    selected = sorted(
        eligible,
        key=lambda row: (
            -int(row["online_a_steps"]),
            int(row["population_index"]),
        ),
    )[0]
    population_index = int(selected["population_index"])
    require(episodes[population_index]["candidate_identity_sha256"]
            == selected["candidate_identity_sha256"],
            "population index no longer addresses the selected history")

    plan_path = (
        arguments.evaluation_root
        / f"{population_index:03d}_{selected['scene']}_{selected['episode']}"
        / "mono_cec"
        / f"{selected['episode']}_pair_00_revisit_plans.json"
    )
    require(plan_path.is_file(), "frozen CEC plan is absent")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authorization = first_strict_authorization(plan)

    revisit = next(
        query
        for query in selected["pairs"][0]["queries"]
        if query.get("analysis_role") == "revisit"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "construction-only selection and previously frozen CEC authority; "
            "no route-tangent output, controller action, navigation success, "
            "or SR was read"
        ),
        "selection_rule": (
            "maximum online_a_steps among frozen 30_to_50_m histories after "
            "excluding route-tangent development indices 16 and 40; ties use "
            "the smallest population_index"
        ),
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "consumed_route_tangent_indices": list(
            CONSUMED_ROUTE_TANGENT_INDICES),
        "eligible_population_count": len(eligible),
        "history_index": population_index,
        "history_id": int(selected["history_index"]),
        "scene": str(selected["scene"]),
        "episode": str(selected["episode"]),
        "bin_name": str(selected["bin_name"]),
        "online_a_steps": int(selected["online_a_steps"]),
        "online_a_trace_sha256": str(selected["online_a_trace_sha256"]),
        "online_a_receipt_sha256": str(selected["online_a_receipt_sha256"]),
        "candidate_identity_sha256": str(
            selected["candidate_identity_sha256"]),
        "revisit_source_frame_analysis_only": int(
            revisit["source_online_frame"]),
        "revisit_geodesic_from_history_end_m_analysis_only": float(
            revisit["geodesic_from_a_end_m"]),
        "source_cec_plan": str(plan_path.resolve()),
        "source_cec_plan_sha256": sha256(plan_path),
        **authorization,
        "route_tangent_output_read": False,
        "navigation_outcome_read": False,
        "sr_read": False,
    }
    arguments.out.mkdir(parents=True)
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = arguments.out / "confirmation_freeze.json"
    output.write_bytes(encoded)
    (arguments.out / "confirmation_freeze.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest()
        + "  confirmation_freeze.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "frozen",
        "history_index": population_index,
        "online_a_steps": int(selected["online_a_steps"]),
        "authorized_anchor": int(authorization["authorized_anchor"]),
        "out": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
