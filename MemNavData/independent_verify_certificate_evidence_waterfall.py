#!/usr/bin/env python3
"""Independently recount the train40 CEC evidence waterfall from raw CSVs.

This verifier deliberately does not import the production waterfall analysis.
It reconstructs the selected-candidate join and every cumulative decision, then
checks the published report and the earlier frozen full-certificate audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


STAGES = (
    "geometry_ranked_candidate",
    "fundamental_precheck",
    "precheck_plus_pnp_pose",
    "plus_pnp_inliers",
    "plus_query_coverage",
    "plus_reference_coverage",
    "full_certificate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"invalid serialized boolean: {value!r}")


def center(row: dict[str, str]) -> dict[str, Any]:
    hypotheses = json.loads(row["hypotheses_json"])
    values = [item for item in hypotheses if int(item["offset"]) == 0]
    if len(values) != 1 or not isinstance(values[0].get("pnp_lightglue"), dict):
        raise RuntimeError("center PnP endpoint is malformed")
    return values[0]["pnp_lightglue"]


def decisions(static: dict[str, str], pnp: dict[str, Any]) -> tuple[bool, ...]:
    f_pass = bool(
        int(float(static["fundamental_inliers"])) >= 16
        and float(static["fundamental_query_hull_coverage"]) >= 0.05
        and float(static["fundamental_reference_hull_coverage"]) >= 0.05
    )
    xy = pnp.get("predicted_relative_xy_m")
    pose = bool(
        f_pass and pnp.get("status") == "ok"
        and isinstance(xy, list) and len(xy) == 2
        and all(finite(value) for value in xy)
    )
    inliers = bool(
        pose and finite(pnp.get("inliers")) and int(pnp["inliers"]) >= 16)
    query = bool(
        inliers and finite(pnp.get("query_inlier_coverage"))
        and float(pnp["query_inlier_coverage"]) >= 0.05)
    reference = bool(
        query and finite(pnp.get("reference_inlier_coverage"))
        and float(pnp["reference_inlier_coverage"]) >= 0.05)
    full = bool(
        reference and finite(pnp.get("reprojection_rmse_px"))
        and float(pnp["reprojection_rmse_px"]) <= 2.0)
    return True, f_pass, pose, inliers, query, reference, full


def verify(
    geometry: list[dict[str, str]],
    static: list[dict[str, str]],
    manifest: dict[str, Any],
    report: dict[str, Any],
    prior_audit: dict[str, Any],
) -> dict[str, Any]:
    sessions = [str(value) for value in manifest["sessions"]]
    if (manifest.get("schema_version") !=
            "train40_certificate_challenge_manifest_v1"):
        raise RuntimeError("manifest schema changed")
    if len(sessions) != 480 or len(set(sessions)) != 480:
        raise RuntimeError("manifest is not the frozen 480-session universe")
    if len(geometry) != 480 or len({row["session_id"] for row in geometry}) != 480:
        raise RuntimeError("geometry endpoints are not 480 unique sessions")
    if {row["session_id"] for row in geometry} != set(sessions):
        raise RuntimeError("geometry endpoints differ from manifest")
    if len(static) != 3840:
        raise RuntimeError("static table is not frozen top-8 x 480")
    index: dict[tuple[str, int], dict[str, str]] = {}
    for row in static:
        key = (row["session_id"], int(row["candidate_frame"]))
        if key in index:
            raise RuntimeError("static selected-candidate key is duplicated")
        index[key] = row
    if {key[0] for key in index} != set(sessions):
        raise RuntimeError("static table differs from manifest")

    records = []
    for row in geometry:
        if row["candidate_selection_origin"] != "lightglue_fundamental_rank_v1":
            raise RuntimeError("geometry proposal origin changed")
        key = (row["session_id"], int(row["candidate_frame"]))
        selected = index.get(key)
        if selected is None or selected["candidate_path"] != row["candidate_path"]:
            raise RuntimeError("selected candidate join failed")
        pnp = center(row)
        records.append({
            "actionable": bool(
                pnp.get("status") == "ok"
                and finite(pnp.get("relative_position_error_m"))
                and float(pnp["relative_position_error_m"]) <= 0.75),
            "positive": boolean(row["session_has_positive"]),
            "strict": boolean(row["session_is_strict_no_match"]),
            "decisions": decisions(selected, pnp),
        })

    observed = {}
    for stage_index, stage in enumerate(STAGES):
        accepted = sum(row["decisions"][stage_index] for row in records)
        actionable_count = sum(row["actionable"] for row in records)
        true_positive = sum(
            row["decisions"][stage_index] and row["actionable"]
            for row in records)
        false_positive = accepted - true_positive
        values = {
            "accepted": accepted,
            "actionable": actionable_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": actionable_count - true_positive,
            "true_negative": len(records) - actionable_count - false_positive,
        }
        published = report["stages"][stage]
        for field, value in values.items():
            if int(published.get(field, -1)) != value:
                raise RuntimeError(
                    f"{stage}.{field}: observed {value}, "
                    f"published {published.get(field)}")
        support = published["open_set_support_audit"]
        expected_support = {
            "positive_session": (
                sum(row["positive"] for row in records),
                sum(row["positive"] and row["decisions"][stage_index]
                    for row in records)),
            "strict_no_match": (
                sum(row["strict"] for row in records),
                sum(row["strict"] and row["decisions"][stage_index]
                    for row in records)),
            "boundary_or_ambiguous": (
                sum(not row["positive"] and not row["strict"] for row in records),
                sum((not row["positive"] and not row["strict"])
                    and row["decisions"][stage_index] for row in records)),
        }
        for group, (total, authorized) in expected_support.items():
            if support[group] != {"sessions": total, "authorized": authorized}:
                raise RuntimeError(f"{stage}.{group} support recount differs")
        observed[stage] = values

    full = observed["full_certificate"]
    prior = prior_audit["certificate"]
    for field in ("accepted", "true_positive", "false_positive",
                  "false_negative", "true_negative"):
        if int(prior[field]) != int(full[field]):
            raise RuntimeError(f"full waterfall differs from prior audit: {field}")
    if report.get("stage_order") != list(STAGES):
        raise RuntimeError("published stage order changed")
    return {
        "schema_version": (
            "independent_cec_certificate_evidence_waterfall_v1_20260826"),
        "verified": True,
        "status": "independent_raw_recount_passed_not_closed_loop",
        "sessions": len(records),
        "stage_counts": observed,
        "prior_full_certificate_counts_equal": True,
    }


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
    parser.add_argument("--geometry-rows", type=Path, required=True)
    parser.add_argument("--static-top8-rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(
        read_rows(args.geometry_rows),
        read_rows(args.static_top8_rows),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.report.read_text(encoding="utf-8")),
        json.loads(args.prior_audit.read_text(encoding="utf-8")),
    )
    result["inputs"] = {
        name: sha256_file(path)
        for name, path in (
            ("geometry_rows_sha256", args.geometry_rows),
            ("static_top8_rows_sha256", args.static_top8_rows),
            ("manifest_sha256", args.manifest),
            ("report_sha256", args.report),
            ("prior_audit_sha256", args.prior_audit),
        )
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
