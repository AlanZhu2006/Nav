#!/usr/bin/env python3
"""Independently reconstruct the CDEC dual-proposal certificate result.

This verifier deliberately does not import either production summarizer.  It
uses only the frozen CSV/collector report, the train-scene role list, and the
published structural PnP thresholds.  It is intended to run after the official
report exists and to fail closed on any disagreement.
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
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "independent_cdec_dual_certificate_verify_v1_20260813"
GEOMETRY_ORIGIN = "lightglue_fundamental_rank_v1"
CDEC_ORIGIN = "cdec_scene_oof_pairwise_rank_v1"
EXPECTED_SESSIONS = 480
EXPECTED_ROWS = 2 * EXPECTED_SESSIONS

POSITION_TOLERANCE_M = 0.75
CERTIFICATE_MIN_INLIERS = 16
CERTIFICATE_MIN_QUERY_COVERAGE = 0.05
CERTIFICATE_MIN_REFERENCE_COVERAGE = 0.05
CERTIFICATE_MAX_REPROJECTION_RMSE_PX = 2.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_bool(value: object, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise RuntimeError(f"invalid boolean in {field}: {value!r}")


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, losses) + 1)
    )
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def center_pnp(payload: str) -> Mapping[str, Any]:
    hypotheses = json.loads(payload)
    require(isinstance(hypotheses, list), "hypotheses_json is not a list")
    centers = [
        item
        for item in hypotheses
        if isinstance(item, Mapping) and int(item.get("offset", 999999)) == 0
    ]
    require(len(centers) == 1, "row lacks exactly one center hypothesis")
    pnp = centers[0].get("pnp_lightglue")
    require(isinstance(pnp, Mapping), "center lacks pnp_lightglue")
    return pnp


def has_certificate(pnp: Mapping[str, Any]) -> bool:
    required = (
        "inliers",
        "query_inlier_coverage",
        "reference_inlier_coverage",
        "reprojection_rmse_px",
    )
    if pnp.get("status") != "ok" or not all(finite(pnp.get(k)) for k in required):
        return False
    return (
        int(pnp["inliers"]) >= CERTIFICATE_MIN_INLIERS
        and float(pnp["query_inlier_coverage"])
        >= CERTIFICATE_MIN_QUERY_COVERAGE
        and float(pnp["reference_inlier_coverage"])
        >= CERTIFICATE_MIN_REFERENCE_COVERAGE
        and float(pnp["reprojection_rmse_px"])
        <= CERTIFICATE_MAX_REPROJECTION_RMSE_PX
    )


def is_actionable(pnp: Mapping[str, Any]) -> bool:
    return (
        pnp.get("status") == "ok"
        and finite(pnp.get("relative_position_error_m"))
        and float(pnp["relative_position_error_m"]) <= POSITION_TOLERANCE_M
    )


def record_from_csv(row: Mapping[str, str]) -> dict[str, Any]:
    pnp = center_pnp(row["hypotheses_json"])
    certificate = has_certificate(pnp)
    actionable = is_actionable(pnp)
    return {
        "session_id": str(row["session_id"]),
        "scene": str(row["scene"]),
        "candidate_frame": int(float(row["candidate_frame"])),
        "candidate_path": str(row["candidate_path"]),
        "teacher_candidate_label": int(float(row["label"])),
        "session_has_positive": parse_bool(
            row["session_has_positive"], "session_has_positive"
        ),
        "session_is_strict_no_match": parse_bool(
            row["session_is_strict_no_match"], "session_is_strict_no_match"
        ),
        "certificate": certificate,
        "actionable": actionable,
        "certified_actionable": certificate and actionable,
        "certificate_false_positive": certificate and not actionable,
        "pnp_status": str(pnp.get("status")),
        "pnp_inliers": int(pnp.get("inliers", 0)),
    }


def load_dual_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "session_id",
            "scene",
            "candidate_frame",
            "candidate_path",
            "label",
            "session_has_positive",
            "session_is_strict_no_match",
            "candidate_selection_origin",
            "hypotheses_json",
        }
        require(reader.fieldnames is not None, "CSV has no header")
        require(not (required - set(reader.fieldnames)), "CSV columns changed")
        raw_rows = list(reader)
    require(len(raw_rows) == EXPECTED_ROWS, f"row count changed: {len(raw_rows)}")

    policies: dict[str, dict[str, dict[str, Any]]] = {
        GEOMETRY_ORIGIN: {},
        CDEC_ORIGIN: {},
    }
    for row in raw_rows:
        origin = row["candidate_selection_origin"]
        require(origin in policies, f"unexpected proposal origin: {origin}")
        record = record_from_csv(row)
        session = record["session_id"]
        require(session not in policies[origin], f"duplicate {origin}/{session}")
        policies[origin][session] = record

    geometry = policies[GEOMETRY_ORIGIN]
    cdec = policies[CDEC_ORIGIN]
    require(len(geometry) == EXPECTED_SESSIONS, "geometry session count changed")
    require(len(cdec) == EXPECTED_SESSIONS, "CDEC session count changed")
    require(set(geometry) == set(cdec), "paired session universes differ")
    for session in geometry:
        for field in (
            "scene",
            "session_has_positive",
            "session_is_strict_no_match",
        ):
            require(
                geometry[session][field] == cdec[session][field],
                f"paired metadata differs ({field}): {session}",
            )
    return geometry, cdec


def summarize_policy(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records.values())
    return {
        "sessions": len(rows),
        "teacher_positive_top1": sum(
            row["teacher_candidate_label"] == 1 for row in rows
        ),
        "certificate_accepted": sum(bool(row["certificate"]) for row in rows),
        "ground_truth_actionable": sum(bool(row["actionable"]) for row in rows),
        "certified_actionable": sum(
            bool(row["certified_actionable"]) for row in rows
        ),
        "certificate_false_positive": sum(
            bool(row["certificate_false_positive"]) for row in rows
        ),
        "accepted_scenes": len(
            {str(row["scene"]) for row in rows if row["certificate"]}
        ),
    }


def cascade(
    primary: Mapping[str, Mapping[str, Any]],
    fallback: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(set(primary) == set(fallback), "cascade universes differ")
    selected: dict[str, dict[str, Any]] = {}
    for session in sorted(primary):
        first = primary[session]
        second = fallback[session]
        chosen = first if first["certificate"] else second
        selected[session] = {
            **chosen,
            "selected_source": "primary" if first["certificate"] else "fallback",
            "second_certificate_invoked": not first["certificate"],
        }
    summary = summarize_policy(selected)
    summary.update(
        {
            "second_certificate_invocations": sum(
                bool(row["second_certificate_invoked"])
                for row in selected.values()
            ),
            "primary_accepts": sum(
                bool(row["certificate"]) for row in primary.values()
            ),
            "fallback_rescues": sum(
                (not primary[key]["certificate"])
                and fallback[key]["certificate"]
                for key in primary
            ),
        }
    )
    return summary, selected


def paired(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    require(set(first) == set(second), "paired universes differ")
    gains = sum(first[key][field] and not second[key][field] for key in first)
    losses = sum(second[key][field] and not first[key][field] for key in first)
    return {
        "gains": int(gains),
        "losses": int(losses),
        "exact_mcnemar_p": exact_mcnemar(int(gains), int(losses)),
    }


def same_anchor_repeatability(
    geometry: Mapping[str, Mapping[str, Any]],
    cdec: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decision_fields = (
        "certificate",
        "actionable",
        "certified_actionable",
        "certificate_false_positive",
        "pnp_status",
        "pnp_inliers",
    )
    same_anchor = [
        key
        for key in geometry
        if geometry[key]["candidate_frame"] == cdec[key]["candidate_frame"]
    ]
    mismatches = [
        key
        for key in same_anchor
        if geometry[key]["candidate_path"] != cdec[key]["candidate_path"]
        or any(
            geometry[key][field] != cdec[key][field] for field in decision_fields
        )
    ]
    return {
        "same_anchor_sessions": len(same_anchor),
        "decision_equal": len(same_anchor) - len(mismatches),
        "decision_mismatches": len(mismatches),
        "mismatch_session_ids": sorted(mismatches),
    }


def reconstruct(
    geometry: Mapping[str, Mapping[str, Any]],
    cdec: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cdec_first_summary, cdec_first = cascade(cdec, geometry)
    geometry_first_summary, geometry_first = cascade(geometry, cdec)
    geometry_summary = summarize_policy(geometry)
    cdec_summary = summarize_policy(cdec)
    repeatability = same_anchor_repeatability(geometry, cdec)

    cdec_vs_geometry = paired(cdec, geometry, "certified_actionable")
    cdec_first_vs_geometry = paired(
        cdec_first, geometry, "certified_actionable"
    )
    geometry_first_vs_geometry = paired(
        geometry_first, geometry, "certified_actionable"
    )
    no_extra_false_positive = (
        geometry_first_summary["certificate_false_positive"]
        <= geometry_summary["certificate_false_positive"]
    )
    requirements = {
        "at_least_one_certified_actionable_rescue": (
            geometry_first_vs_geometry["gains"] > 0
        ),
        "cannot_lose_geometry_certified_actionable": (
            geometry_first_vs_geometry["losses"] == 0
        ),
        "no_extra_certificate_false_positive": no_extra_false_positive,
        "same_anchor_certificate_decisions_repeatable": (
            repeatability["decision_mismatches"] == 0
        ),
    }
    return {
        "policies": {
            "geometry": geometry_summary,
            "cdec_oof": cdec_summary,
            "cdec_first_then_geometry_on_reject": cdec_first_summary,
            "geometry_first_then_cdec_on_reject": geometry_first_summary,
        },
        "paired": {
            "cdec_oof_minus_geometry_certified_actionable": cdec_vs_geometry,
            "cdec_first_cascade_minus_geometry_certified_actionable": (
                cdec_first_vs_geometry
            ),
            "geometry_first_cascade_minus_geometry_certified_actionable": (
                geometry_first_vs_geometry
            ),
        },
        "proposal_identity": {
            "same_selected_anchor": sum(
                geometry[key]["candidate_frame"] == cdec[key]["candidate_frame"]
                for key in geometry
            ),
            "both_certificates_accept": sum(
                geometry[key]["certificate"] and cdec[key]["certificate"]
                for key in geometry
            ),
            "geometry_only_accepts": sum(
                geometry[key]["certificate"] and not cdec[key]["certificate"]
                for key in geometry
            ),
            "cdec_only_accepts": sum(
                cdec[key]["certificate"] and not geometry[key]["certificate"]
                for key in geometry
            ),
            "same_anchor_repeatability": repeatability,
        },
        "method_gate": {
            "pass": all(requirements.values()),
            "deployment_order": "geometry_first_then_cdec_on_reject",
            "requirements": requirements,
        },
    }


def verify_subset(
    expected: Mapping[str, Any], actual: Mapping[str, Any], prefix: str = ""
) -> None:
    for key, value in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        require(key in actual, f"official report lacks {path}")
        observed = actual[key]
        if isinstance(value, Mapping):
            require(isinstance(observed, Mapping), f"official {path} is not an object")
            verify_subset(value, observed, path)
        else:
            require(observed == value, f"official mismatch at {path}")


def verify(
    *,
    dual_rows: Path,
    collector_report_path: Path,
    official_report_path: Path,
    role_split_path: Path,
) -> dict[str, Any]:
    collector = json.loads(collector_report_path.read_text())
    official = json.loads(official_report_path.read_text())
    config = collector.get("config")
    require(isinstance(config, Mapping), "collector config is missing")
    require(
        config.get("selection_mode") == "cdec_geometry_dual_ranked",
        "collector selection mode changed",
    )
    require(config.get("neighbor_offsets") == [0], "neighbor offsets changed")
    require(config.get("full_replay") is True, "full replay is not enabled")

    geometry, cdec = load_dual_rows(dual_rows)
    split = json.loads(role_split_path.read_text())
    train = {str(scene) for scene in split.get("train", [])}
    require(len(train) == 40, "frozen train role is not train40")
    row_scenes = {str(row["scene"]) for row in geometry.values()}
    require(row_scenes == train, "collector scene universe is not exactly train40")

    reconstructed = reconstruct(geometry, cdec)
    verify_subset(reconstructed, official)
    scope = official.get("scope")
    require(isinstance(scope, Mapping), "official scope is missing")
    for key, expected in {
        "train_scenes_only": True,
        "row_scene_held_out_for_cdec": True,
        "development_or_blind_read": False,
        "one_view_full_replay": True,
        "activation_learned": False,
        "same_gpu_same_lingbot_process": True,
    }.items():
        require(scope.get(key) is expected, f"official scope mismatch: {key}")
    inputs = official.get("inputs")
    require(isinstance(inputs, Mapping), "official inputs are missing")
    require(
        inputs.get("dual_rows_sha256") == sha256_file(dual_rows),
        "official dual-row SHA mismatch",
    )
    require(
        inputs.get("dual_report_sha256") == sha256_file(collector_report_path),
        "official collector-report SHA mismatch",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "scope": {
            "independent_of_primary_summarizer": True,
            "train40_only": True,
            "development_or_blind_read": False,
            "closed_loop": False,
        },
        "inputs": {
            "dual_rows": str(dual_rows.resolve()),
            "dual_rows_sha256": sha256_file(dual_rows),
            "collector_report": str(collector_report_path.resolve()),
            "collector_report_sha256": sha256_file(collector_report_path),
            "official_report": str(official_report_path.resolve()),
            "official_report_sha256": sha256_file(official_report_path),
            "role_split": str(role_split_path.resolve()),
            "role_split_sha256": sha256_file(role_split_path),
        },
        "reconstructed": reconstructed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-rows", type=Path, required=True)
    parser.add_argument("--collector-report", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--role-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), f"output already exists: {args.out}")
    result = verify(
        dual_rows=args.dual_rows,
        collector_report_path=args.collector_report,
        official_report_path=args.official_report,
        role_split_path=args.role_split,
    )
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
